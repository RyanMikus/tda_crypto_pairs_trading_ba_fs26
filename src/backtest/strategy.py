"""SignalStrategy-Interface und die gemeinsame Z-Score-Basisklasse.

Der Trick der Austauschbarkeit (Pairwise vs. Cluster-Spread, vom Nutzer als
pluggable Strategy-Interface gefordert): Entry-/Exit-Logik über den Z-Score
ist für BEIDE identisch (BaseSpreadZScoreStrategy). Der einzige Unterschied
ist, WIE aus einer Clique SpreadDefinitions entstehen -- genau das ist die
eine abstrakte Methode `build_definitions`. Für die Engine ist "eine Position
pro Paar" vs. "eine Position pro Clique" dadurch kein Sonderfall, sondern nur
eine andere Länge der von build_definitions zurückgegebenen Liste.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from Src.backtest.cliques import CliqueSnapshot
from Src.backtest.config import BasketParams
from Src.backtest.types import BarContext, EntrySignal, ExitDecision, NO_EXIT, Position, SpreadDefinition


@dataclass
class StrategyState:
    """Ergebnis von on_snapshot: die handelbaren Spreads dieses Snapshots."""

    membership_key: str
    fitted_at: pd.Timestamp
    definitions: list[SpreadDefinition]
    rejected: list[dict] = field(default_factory=list)


class SignalStrategy(Protocol):
    name: str

    def required_history_candles(self, params: BasketParams) -> int: ...

    def on_snapshot(
        self, *, snapshot: CliqueSnapshot, ctx: BarContext, params: BasketParams,
    ) -> StrategyState: ...

    def generate_entries(
        self, *, ctx: BarContext, state: StrategyState, params: BasketParams,
    ) -> list[EntrySignal]: ...

    def check_exit(
        self, *, position: Position, ctx: BarContext, params: BasketParams,
    ) -> ExitDecision: ...


class BaseSpreadZScoreStrategy(ABC):
    """Z-Score-Entry/Exit, geteilt von Pairwise- und Cluster-Strategie."""

    name: str = "base_spread_zscore"

    def required_history_candles(self, params: BasketParams) -> int:
        return max(params.formation_window_candles, params.bollinger_window_candles)

    @abstractmethod
    def build_definitions(
        self, *, snapshot: CliqueSnapshot, window: pd.DataFrame, params: BasketParams,
    ) -> tuple[list[SpreadDefinition], list[dict]]:
        """window: Preis-DataFrame (formation_window_candles x symbols),
        Ende <= aktuelle Bar (siehe on_snapshot)."""

    def on_snapshot(
        self, *, snapshot: CliqueSnapshot, ctx: BarContext, params: BasketParams,
    ) -> StrategyState:
        window = ctx.prices.history(snapshot.symbols, params.formation_window_candles, field="close")
        definitions, rejected = self.build_definitions(snapshot=snapshot, window=window, params=params)
        for definition in definitions:
            # Clique-Diagnostik aus dem Snapshot in die SpreadDefinition durchreichen,
            # damit sie später im Trade-Log landet (trades.parquet: persistence,
            # clique_size) -- der direkte Brückenschlag zurück zur TDA-Fragestellung.
            definition.diagnostics.setdefault("clique_persistence", snapshot.persistence)
            definition.diagnostics.setdefault("clique_size", snapshot.size)
            definition.diagnostics.setdefault("lookback_days", snapshot.lookback_days)
        return StrategyState(
            membership_key=snapshot.membership_key,
            fitted_at=ctx.timestamp,
            definitions=definitions,
            rejected=rejected,
        )

    def generate_entries(
        self, *, ctx: BarContext, state: StrategyState, params: BasketParams,
    ) -> list[EntrySignal]:
        asset_position_counts: dict[int, int] = {}
        for pos in ctx.open_positions:
            for asset_id in pos.asset_ids:
                asset_position_counts[asset_id] = asset_position_counts.get(asset_id, 0) + 1

        signals = []
        for definition in state.definitions:
            if any(
                asset_position_counts.get(a, 0) >= params.max_positions_per_asset
                for a in definition.asset_ids
            ):
                continue

            log_prices = ctx.prices.log_last(definition.symbols)
            if np.isnan(log_prices).any():
                continue

            price_window = ctx.prices.history(definition.symbols, params.bollinger_window_candles, field="close")
            if price_window.isna().any().any():
                continue
            z = definition.rolling_zscore(price_window)
            if z is None:
                continue
            if abs(z) < params.entry_z:
                continue
            if params.stop_z is not None and abs(z) >= params.stop_z:
                # Signal ist per Definition schon ungueltig (stop_z ist die
                # Invalidierungsschwelle) -- ein Entry HIER wuerde sofort im
                # naechsten Bar auf den Stop laufen. Kein sinnvolles Setup.
                continue

            # Spread über seinem Mittelwert (z > 0) -> Spread SHORT (direction=-1),
            # Spread darunter (z < 0) -> Spread LONG (direction=+1).
            direction = -1 if z > 0 else 1
            signals.append(EntrySignal(
                spread=definition,
                direction=direction,
                signal_value=z,
                spread_value=definition.spread_value(log_prices),
                priority=abs(z),
            ))
        return signals

    def check_exit(
        self, *, position: Position, ctx: BarContext, params: BasketParams,
    ) -> ExitDecision:
        if position.bars_held < params.min_holding_candles:
            return NO_EXIT
        price_window = ctx.prices.history(position.spread.symbols, params.bollinger_window_candles, field="close")
        if price_window.isna().any().any():
            return NO_EXIT
        z = position.spread.rolling_zscore(price_window)
        if z is None:
            return NO_EXIT
        if abs(z) <= params.exit_z:
            return ExitDecision(True, "target", signal_value=z)
        return NO_EXIT


def _quality_filter(diag: dict, params: BasketParams) -> tuple[bool, str | None]:
    """ADF/Half-Life-Filter, geteilt von Pairwise- und Cluster-Strategie."""
    if params.require_adf and not diag.get("adf_stationary", False):
        return False, "adf_not_stationary"
    half_life = diag.get("half_life_periods", np.inf)
    if not (params.min_half_life_candles <= half_life <= params.max_half_life_candles):
        return False, "half_life_out_of_range"
    return True, None


class PairwiseZScoreStrategy(BaseSpreadZScoreStrategy):
    """Bildet aus einer Clique alle Ticker-Paare, filtert per Korrelation vor
    (Cliquen mit bis zu 19 Assets ergäben sonst bis zu 171 Paare -- die volle
    ADF/KPSS/Hurst-Diagnostik auf allen wäre der Performance-Killer), prüft
    die Top-Kandidaten per analyze_spread_stationarity und behält die
    schnellsten (niedrigste Half-Life) bis max_candidate_pairs."""

    name = "pairwise_zscore"

    def build_definitions(
        self, *, snapshot: CliqueSnapshot, window: pd.DataFrame, params: BasketParams,
    ) -> tuple[list[SpreadDefinition], list[dict]]:
        from Src.signals import spread as sp

        symbols = list(snapshot.symbols)
        sym_to_asset_id = dict(zip(snapshot.symbols, snapshot.asset_ids))
        rejected: list[dict] = []

        if len(symbols) < 2:
            return [], rejected

        returns = sp.compute_log_returns(window)
        if returns.shape[0] < 2:
            return [], [{"reason": "insufficient_returns_for_prefilter"}]

        corr = returns.corr().abs()
        pair_scores = []
        for i, t1 in enumerate(symbols):
            for t2 in symbols[i + 1:]:
                score = corr.loc[t1, t2]
                pair_scores.append((t1, t2, score if np.isfinite(score) else -1.0))
        pair_scores.sort(key=lambda p: -p[2])
        top_pairs = pair_scores[: params.pair_prefilter_top_n]

        candidates = []
        for t1, t2, _ in top_pairs:
            try:
                diag = sp.analyze_spread_stationarity(window, t1, t2, alpha=params.adf_alpha)
            except Exception as exc:  # numerisch instabile Paare (z.B. konstante Preise) überspringen
                rejected.append({"pair": (t1, t2), "reason": f"diagnostics_failed: {exc}"})
                continue

            ok, why = _quality_filter(diag, params)
            if not ok:
                rejected.append({"pair": (t1, t2), "reason": why, **diag})
                continue
            candidates.append((t1, t2, diag))

        candidates.sort(key=lambda c: c[2]["half_life_periods"])
        candidates = candidates[: params.max_candidate_pairs]

        definitions = []
        for t1, t2, diag in candidates:
            hedge_ratio = diag["hedge_ratio"]
            spread_series = sp.compute_price_spread(window, t1, t2, hedge_ratio)
            mu, sigma = sp.spread_zscore_params(spread_series)
            definitions.append(SpreadDefinition(
                kind="pairwise",
                asset_ids=(sym_to_asset_id[t1], sym_to_asset_id[t2]),
                symbols=(t1, t2),
                weights=(1.0, -hedge_ratio),
                mu=mu,
                sigma=sigma,
                fitted_at=window.index[-1],
                formation_start=window.index[0],
                diagnostics=diag,
            ))
        return definitions, rejected


class ClusterSpreadStrategy(BaseSpreadZScoreStrategy):
    """Bildet aus einer Clique GENAU EINE Position: den gewichteten N-Asset-
    Spread aus der stärksten Johansen-Kointegrationsbeziehung. Läuft über
    dieselbe BacktestEngine wie PairwiseZScoreStrategy, ohne dass diese
    angefasst werden muss -- das ist der Interface-Test."""

    name = "cluster_johansen"

    def build_definitions(
        self, *, snapshot: CliqueSnapshot, window: pd.DataFrame, params: BasketParams,
    ) -> tuple[list[SpreadDefinition], list[dict]]:
        from Src.signals import spread as sp

        symbols = list(snapshot.symbols)
        sym_to_asset_id = dict(zip(snapshot.symbols, snapshot.asset_ids))
        rejected: list[dict] = []

        if len(symbols) < 2:
            return [], rejected

        try:
            johansen = sp.compute_johansen_test(window[symbols])
        except Exception as exc:  # z.B. singuläre Kovarianzmatrix bei (fast) konstanten Preisen
            return [], [{"reason": f"johansen_failed: {exc}"}]

        if johansen["n_coint_relations"] < params.min_johansen_relations:
            return [], [{
                "reason": "insufficient_coint_relations",
                "n_coint_relations": johansen["n_coint_relations"],
            }]

        weights = sp.normalize_cointegration_vector(johansen["eigenvectors"]["relation_1"])
        cluster_spread_series = sp.build_cluster_spread(window, weights)
        diag = sp.evaluate_spread_quality(cluster_spread_series, alpha=params.adf_alpha)
        diag["n_coint_relations"] = johansen["n_coint_relations"]

        ok, why = _quality_filter(diag, params)
        if not ok:
            return [], [{"reason": why, **diag}]

        mu, sigma = sp.spread_zscore_params(cluster_spread_series)
        definition = SpreadDefinition(
            kind="cluster",
            asset_ids=tuple(sym_to_asset_id[s] for s in weights.index),
            symbols=tuple(weights.index),
            weights=tuple(float(w) for w in weights.to_numpy()),
            mu=mu,
            sigma=sigma,
            fitted_at=window.index[-1],
            formation_start=window.index[0],
            diagnostics=diag,
        )
        return [definition], rejected


#!#################### NEU: GarchWilksToleranceStrategy -- portiert die
#!#################### GARCH(1,1)/Wilks-Toleranzintervall-Signalkette aus
#!#################### notebooks/03_Tradeing/garch_pairtrading_engine.py
#!#################### (Lin, Chen & Syu 2021) auf die aktuelle Engine.
#!####################
#!#################### Bewusst NICHT von BaseSpreadZScoreStrategy abgeleitet:
#!#################### dort ist Entry/Exit ein fester entry_z/exit_z-
#!#################### Schwellenwert auf EINEM bei Formation eingefrorenen
#!#################### (mu, sigma). Hier sind Entry/Exit rollierend neu
#!#################### geschaetzte Wilks-Toleranzgrenzen (X_(l), X_(u)) auf
#!#################### Xmax = max(|Xt|) ueber alle Paare der Clique, mit
#!#################### Xt = Rt/sigma_t (Rt = Return-Differenz, sigma_t =
#!#################### GARCH(1,1)-Volatilitaet) -- siehe
#!#################### src/signals/{garch,tolerance,garch_tolerance}.py.
class GarchWilksToleranceStrategy:
    """GARCH(1,1)-Vola-Normierung + nichtparametrisches Wilks-Toleranz-
    intervall (Lin, Chen & Syu 2021) als SignalStrategy.

    Ablauf pro Clique-Snapshot (on_snapshot/build_definitions):
        GARCH(1,1) wird EINMAL pro Paar auf dem Formationsfenster gefittet
        (teuer, Nelder-Mead-ML) -- die gefitteten omega/alpha/beta landen in
        SpreadDefinition.diagnostics und werden bis zum naechsten Snapshot
        wiederverwendet (siehe engine._get_or_fit_state-Cache), statt bei
        jeder Bar neu zu fitten.

    Ablauf pro Bar (generate_entries/check_exit):
        Aus der laufenden Preishistorie wird Rt = r_Ai,t - r_Aj,t gebildet,
        mit dem GARCH-Sigma-Pfad (garch.garch_sigma_path, feste Parameter
        aus der Formation) zu Xt = Rt/sigma_t normiert -- guenstig, kein
        Refit. Xmax = max(|Xt|) ueber alle Paare der Clique, Wilks-Grenzen
        (X_(l), X_(u)) aus der Xmax-Historie der letzten
        wilks_lookback_candles Bars. Entry, wenn |Xt| des extremsten Paares
        die obere Grenze durchbricht (Lin et al., S. 3: "we find (X_(l),
        X_(u)) as the lower and upper trading signals"). Exit, sobald |Xt|
        DES GEHALTENEN PAARES die Mitte (X_(l)+X_(u))/2 erreicht/
        unterschreitet (siehe garch_tolerance.py-Modul-Docstring,
        "Exit-Mitte" -- kein zusaetzliches Lookback-Fenster noetig).

    SpreadDefinition.weights=(1.0, -1.0) (KEIN Hedge-Ratio -- Lin et al.
    verwenden die rohe Return-Differenz, nicht den kointegrierten Preis-
    Spread): definiert nur, wie Portfolio die beiden Legs dollar-neutral
    positioniert; die eigentliche Entry/Exit-Signal-Berechnung laeuft
    separat auf dem Return-Spread (siehe oben), nicht ueber
    SpreadDefinition.rolling_zscore.
    """

    name = "garch_wilks"

    def required_history_candles(self, params: BasketParams) -> int:
        return params.formation_window_candles

    def on_snapshot(
        self, *, snapshot: CliqueSnapshot, ctx: BarContext, params: BasketParams,
    ) -> StrategyState:
        from Src.signals import spread as sp
        from Src.signals.garch import compute_garch_volatility

        symbols = list(snapshot.symbols)
        sym_to_asset_id = dict(zip(snapshot.symbols, snapshot.asset_ids))
        rejected: list[dict] = []
        definitions: list[SpreadDefinition] = []

        #!#################### FIX: len(symbols)-Check MUSS vor history() laufen,
        #!#################### sonst wirft history() bei <2 Symbolen unnoetig den
        #!#################### Warm-up-Guard G7 (formation_window_candles Kerzen
        #!#################### angefragt, obwohl das Ergebnis wegen <2 Symbolen
        #!#################### ohnehin verworfen wuerde).
        if len(symbols) < 2:
            return StrategyState(membership_key=snapshot.membership_key, fitted_at=ctx.timestamp, definitions=[], rejected=rejected)
        #!####################

        window = ctx.prices.history(snapshot.symbols, params.formation_window_candles, field="close")
        returns = sp.compute_log_returns(window)

        for i, t1 in enumerate(symbols):
            for t2 in symbols[i + 1:]:
                if t1 not in returns.columns or t2 not in returns.columns:
                    rejected.append({"pair": (t1, t2), "reason": "missing_returns"})
                    continue
                return_spread = sp.compute_return_spread(returns, t1, t2)
                try:
                    _, fit_info = compute_garch_volatility(
                        return_spread, dist=params.garch_dist, max_iter=params.garch_max_iter,
                    )
                except ValueError as exc:  # zu wenig gueltige Beobachtungen im Formationsfenster
                    rejected.append({"pair": (t1, t2), "reason": f"garch_fit_failed: {exc}"})
                    continue
                if not fit_info["converged"]:
                    rejected.append({"pair": (t1, t2), "reason": "garch_not_converged", **fit_info})
                    continue

                definitions.append(SpreadDefinition(
                    kind="pairwise",
                    asset_ids=(sym_to_asset_id[t1], sym_to_asset_id[t2]),
                    symbols=(t1, t2),
                    weights=(1.0, -1.0),
                    mu=0.0,  # ungenutzt -- Entry/Exit laeuft ueber Wilks-Grenzen, nicht rolling_zscore
                    sigma=1.0,  # dito
                    fitted_at=window.index[-1],
                    formation_start=window.index[0],
                    diagnostics={
                        "garch_omega": fit_info["omega"], "garch_alpha": fit_info["alpha"],
                        "garch_beta": fit_info["beta"], "garch_nu": fit_info["nu"],
                        "clique_persistence": snapshot.persistence, "clique_size": snapshot.size,
                        "lookback_days": snapshot.lookback_days,
                    },
                ))

        # Jede SpreadDefinition traegt einen Verweis auf ALLE Clique-Geschwister
        # dieses Snapshots (inkl. sich selbst) -- Xmax braucht bei Entry UND
        # bei Exit dieselbe Paar-Menge; ohne diesen Verweis muesste check_exit
        # (das kein StrategyState bekommt, siehe SignalStrategy-Protokoll)
        # GARCH sonst pro Bar neu fitten, um Xmax der Clique zu rekonstruieren.
        for definition in definitions:
            definition.diagnostics["clique_siblings"] = tuple(definitions)

        return StrategyState(
            membership_key=snapshot.membership_key, fitted_at=ctx.timestamp,
            definitions=definitions, rejected=rejected,
        )

    def _pair_xt(self, definition: SpreadDefinition, ctx: BarContext, params: BasketParams) -> pd.Series | None:
        """Xt = Rt/sigma_t fuer EIN Paar, ueber die letzten
        wilks_lookback_candles+1 Bars (braucht +1, weil compute_log_returns
        eine Beobachtung durch die Differenzierung verliert). None, falls
        nicht genug Historie ohne Datenluecken vorliegt."""
        from Src.signals import spread as sp
        from Src.signals.garch import garch_sigma_path

        n_needed = params.wilks_lookback_candles + 1
        if ctx.bar_index + 1 < n_needed:
            return None
        price_window = ctx.prices.history(definition.symbols, n_needed, field="close")
        if price_window.isna().any().any():
            return None

        returns = sp.compute_log_returns(price_window)
        return_spread = sp.compute_return_spread(returns, *definition.symbols)
        diag = definition.diagnostics
        sigma = garch_sigma_path(
            return_spread, omega=diag["garch_omega"], alpha=diag["garch_alpha"], beta=diag["garch_beta"],
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return return_spread / sigma.replace(0.0, np.nan)

    def _wilks_limits(self, xmax_history: pd.Series, params: BasketParams) -> tuple[float, float] | None:
        from Src.signals.tolerance import wilks_tolerance_limits

        clean = xmax_history.dropna().to_numpy()
        if len(clean) < 3:
            return None
        try:
            return wilks_tolerance_limits(clean, p_content=params.wilks_p_content, confidence=params.wilks_confidence)
        except ValueError:
            return None

    def _clique_xmax(
        self, definitions: list[SpreadDefinition], ctx: BarContext, params: BasketParams,
    ) -> tuple[list[tuple[SpreadDefinition, pd.Series]], pd.Series | None]:
        """Xt je Paar + Xmax_t = max(|Xt|) ueber die gesamte Paarmenge (Lin
        et al. Step 3), gemeinsam fuer generate_entries UND check_exit -- die
        Wilks-Grenzen (Entry-Threshold wie Exit-Mitte) muessen auf DERSELBEN
        Xmax-Historie sitzen, sonst waeren sie fuer Entry und Exit
        unterschiedlich streng, obwohl es dieselbe Clique ist."""
        xt_by_definition: list[tuple[SpreadDefinition, pd.Series]] = []
        for definition in definitions:
            xt = self._pair_xt(definition, ctx, params)
            if xt is None or xt.empty or pd.isna(xt.iloc[-1]):
                continue
            xt_by_definition.append((definition, xt))
        if not xt_by_definition:
            return xt_by_definition, None
        xmax_history = pd.concat([xt.abs() for _, xt in xt_by_definition], axis=1).max(axis=1)
        return xt_by_definition, xmax_history

    def generate_entries(
        self, *, ctx: BarContext, state: StrategyState, params: BasketParams,
    ) -> list[EntrySignal]:
        asset_position_counts: dict[int, int] = {}
        for pos in ctx.open_positions:
            for asset_id in pos.asset_ids:
                asset_position_counts[asset_id] = asset_position_counts.get(asset_id, 0) + 1

        eligible = [
            d for d in state.definitions
            if not any(asset_position_counts.get(a, 0) >= params.max_positions_per_asset for a in d.asset_ids)
        ]
        xt_by_definition, xmax_history = self._clique_xmax(eligible, ctx, params)
        if xmax_history is None:
            return []

        limits = self._wilks_limits(xmax_history, params)
        if limits is None:
            return []
        _, upper_limit = limits

        # Step 4: NUR das Paar mit dem aktuell groessten |Xt| ist Handelskandidat
        # (Lin et al.: "choose the maximum X^{Ai-Aj}_{n+1}... Xmax_{n+1}").
        definition, xt = max(xt_by_definition, key=lambda item: abs(item[1].iloc[-1]))
        current_xt = float(xt.iloc[-1])
        if abs(current_xt) <= upper_limit:
            return []
        if params.stop_z is not None and abs(current_xt) >= params.stop_z:
            return []  # Signal waere sofort ungueltig, kein sinnvolles Setup (analog BaseSpreadZScoreStrategy)

        log_prices = ctx.prices.log_last(definition.symbols)
        if np.isnan(log_prices).any():
            return []

        # current_xt > 0: r_Ai ueberdurchschnittlich POSITIV relativ zu r_Aj ->
        # Reversion-Erwartung "Ai faellt relativ, Aj steigt relativ" -> Short Ai/Long Aj.
        # SpreadDefinition.weights=(1.0,-1.0) bildet spread=log(P_Ai)-log(P_Aj);
        # direction=-1 kehrt das Vorzeichen fuer Portfolio.dollar_neutral_fractions um.
        direction = -1 if current_xt > 0 else 1
        return [EntrySignal(
            spread=definition, direction=direction, signal_value=current_xt,
            spread_value=definition.spread_value(log_prices), priority=abs(current_xt),
        )]

    def check_exit(
        self, *, position: Position, ctx: BarContext, params: BasketParams,
    ) -> ExitDecision:
        if position.bars_held < params.min_holding_candles:
            return NO_EXIT

        # Xmax auf DERSELBEN Paarmenge wie beim Entry (clique_siblings, siehe
        # on_snapshot) -- nicht nur auf position.spread selbst, sonst waere
        # exit_mid enger als die beim Entry ueberschrittene upper_limit und
        # der Trade koennte theoretisch nie schliessen (siehe Modul-Docstring
        # "Exit-Mitte").
        siblings = position.spread.diagnostics.get("clique_siblings", (position.spread,))
        xt_by_definition, xmax_history = self._clique_xmax(list(siblings), ctx, params)
        if xmax_history is None:
            return NO_EXIT

        own_xt = next((xt for d, xt in xt_by_definition if d.symbols == position.spread.symbols), None)
        if own_xt is None:
            return NO_EXIT
        current_xt = float(own_xt.iloc[-1])

        limits = self._wilks_limits(xmax_history, params)
        if limits is None:
            return NO_EXIT
        lower_limit, upper_limit = limits
        exit_mid = (lower_limit + upper_limit) / 2.0

        if abs(current_xt) <= exit_mid:
            return ExitDecision(True, "target", signal_value=current_xt)
        return NO_EXIT
#!####################


STRATEGY_REGISTRY: dict[str, type] = {
    PairwiseZScoreStrategy.name: PairwiseZScoreStrategy,
    ClusterSpreadStrategy.name: ClusterSpreadStrategy,
    GarchWilksToleranceStrategy.name: GarchWilksToleranceStrategy,  #!#################### NEU
}


def build_strategy(name: str, **kwargs) -> SignalStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unbekannte Strategie {name!r}. Verfügbar: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**kwargs)
