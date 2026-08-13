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


STRATEGY_REGISTRY: dict[str, type] = {
    PairwiseZScoreStrategy.name: PairwiseZScoreStrategy,
    ClusterSpreadStrategy.name: ClusterSpreadStrategy,
}


def build_strategy(name: str, **kwargs) -> SignalStrategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unbekannte Strategie {name!r}. Verfügbar: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**kwargs)
