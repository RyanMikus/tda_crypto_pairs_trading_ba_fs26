"""Engine-weite Break-/Exit-Bedingungen, unabhängig von der Strategie.

Geordnete Liste (`ExitRule`-Objekte) statt einzelner Booleans, damit neue
Break-Conditions ohne Änderungen an engine.py ergänzt werden können.

Die strategie-spezifische "target"-Exit-Bedingung (Z-Score kehrt in den
exit_z-Korridor zurück) läuft NICHT hier, sondern in
BaseSpreadZScoreStrategy.check_exit -- sie läuft in der Engine NACH dieser
Kette (siehe engine.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.backtest.types import BarContext, ExitDecision, NO_EXIT, Position
from src.clustering.helpers import HelperFunctions
from src.config import candles_per_day


class ExitRule(Protocol):
    reason: str

    def check(self, position: Position, ctx: BarContext) -> ExitDecision: ...


@dataclass
class DataGapRule:
    """Zwangs-Exit, wenn für eine der Legs zu viele aufeinanderfolgende Kerzen
    kein gültiger Preis vorlag (Portfolio.mark_to_market pflegt
    position.missing_price_bars). Deckt sowohl temporäre Datenlücken als auch
    echtes Delisting ab -- beides ist zur Laufzeit (kausal, ohne Blick in die
    Zukunft) nicht unterscheidbar. Ein eigenständiges DelistingRule wäre nur
    mit vorab bekannten Delisting-Daten sinnvoll (aktuell nicht in
    assets.parquet vorhanden)."""

    reason: str = "data_gap"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if position.missing_price_bars >= ctx.params.max_data_gap_candles:
            return ExitDecision(True, self.reason)
        return NO_EXIT


@dataclass
class StopLossRule:
    """|Z-Score| >= stop_z -> sofortiger Exit. Generisch für jede Strategie,
    da SpreadDefinition.rolling_zscore() unabhängig davon funktioniert, ob
    der Spread aus 2 (Pairwise) oder N (Cluster) Assets besteht. Nutzt
    denselben rollierenden (Bollinger-)Z-Score wie Entry/Target-Exit
    (BaseSpreadZScoreStrategy) -- "N Sigma" bedeutet damit für alle drei
    Regeln konsistent dasselbe, aktuelle Band."""

    reason: str = "stop"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if ctx.params.stop_z is None:
            return NO_EXIT
        price_window = ctx.prices.history(position.spread.symbols, ctx.params.bollinger_window_candles, field="close")
        if price_window.isna().any().any():
            return NO_EXIT  # DataGapRule kümmert sich um fehlende Preise
        z = position.spread.rolling_zscore(price_window)
        if z is None:
            return NO_EXIT
        if abs(z) >= ctx.params.stop_z:
            return ExitDecision(True, self.reason, signal_value=z)
        return NO_EXIT


@dataclass
class MaxHoldingDaysRule:
    """Zwangs-Exit nach BasketParams.max_holding_days (absoluter Kalendertage-
    Hard-Cap) -- ersetzt die fruehere, half-life-relative TimeoutRule
    (max_holding_half_life_multiplier * Half-Life pro Position), die entfernt
    wurde, weil eine variable Pro-Position-Haltedauer sich nicht sinnvoll als
    feste Tageszahl vorgeben liess. None deaktiviert die Regel."""

    reason: str = "max_holding_days"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if ctx.params.max_holding_days is None:
            return NO_EXIT
        max_holding_candles = ctx.params.max_holding_days * candles_per_day(ctx.candle_minutes)
        if position.bars_held >= max_holding_candles:
            return ExitDecision(True, self.reason)
        return NO_EXIT


@dataclass
class CliqueBreakRule:
    """Exit, sobald die beim Entry gehandelte Clique beim aktuellen Snapshot
    nicht mehr die aktive Clique dieses Baskets ist (Mitgliedschaftswechsel
    oder Basket komplett ohne Clique)."""

    reason: str = "clique_break"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if not ctx.params.exit_on_clique_break:
            return NO_EXIT
        if ctx.snapshot is None or ctx.snapshot.membership_key != position.membership_key:
            return ExitDecision(True, self.reason)
        return NO_EXIT


@dataclass
class TopologicalBreakRule:
    """Topologischer Struktur-Break-Detektor (docs/project/woo_2026a_additional-information.pdf,
    Abschnitt 2): Single-Linkage + fcluster bei epsilon* - delta, NUR auf den
    Assets der offenen Clique (kein Basket-weiter Re-Discovery-Schritt --
    genau das macht das Verfahren günstig, siehe PDF und Prototyp in
    notebooks/02_tda_clustering/05_topological_break.ipynb).

    epsilon* = position.entry_death (bei Entry aus CliqueSnapshot.death
    eingefroren -- NICHT persistence, siehe cliques.CliqueSnapshot.persistence-
    Kommentar). Rolling-Window per Default = position.entry_lookback_days in
    Kerzen (überschreibbar via topo_break_window_candles).

    Zensierter Fall (entry_death == epsilon_cap der Discovery, siehe
    05_topological_break.ipynb Abschnitt 4/7): bewusst KEINE Sonderbehandlung
    -- eine Clique, die bei der Discovery nie extern verschmolzen ist, hat
    schlicht keine beobachtete Aussengrenze; die Regel greift dort seltener,
    das ist die korrekte Konsequenz, keine Notlösung. Für die Auswertung
    trägt jede ausgelöste ExitDecision `signal_value=tau`, damit sich
    zensierte Fälle (tau nahe/gleich epsilon_cap - delta) im Nachhinein aus
    den Trades herausfiltern lassen.
    """

    reason: str = "topological_break"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if not ctx.params.topo_break_enabled:
            return NO_EXIT


        window_candles = ctx.params.topo_break_window_candles or round(
            position.entry_lookback_days * candles_per_day(ctx.candle_minutes)
        )

        if ctx.bar_index < window_candles - 1:
            return NO_EXIT  # zu wenig Historie -- kein Fehler, einfach (noch) nicht prüfbar
        if len(position.symbols) < 2:
            return NO_EXIT  # Single-Linkage braucht mindestens 2 Assets

        prices = ctx.prices.history(position.symbols, window_candles, field="close")
        returns = prices.pct_change().dropna(how="any")
        if returns.shape[0] < 2:
            return NO_EXIT

        tau = position.entry_death - ctx.params.topo_break_delta
        if tau <= 0:
            # entry_death lag bereits bei/unter delta (degenerierte Clique,
            # in der Praxis nicht beobachtet -- death-Werte liegen deutlich
            # darueber). Kein sinnvoller Schnitt moeglich: fcluster mit
            # t<=0 wuerde JEDEN Punkt zum Singleton machen (alle Merge-Hoehen
            # sind >= 0), also bei jedem Check sofort "gebrochen" melden,
            # unabhaengig vom tatsaechlichen Zustand. Lieber gar nicht pruefen.
            return NO_EXIT

        dist = HelperFunctions.correlation_distance(returns)
        z = linkage(squareform(dist, checks=False), method="single")
        labels = fcluster(z, t=tau, criterion="distance")
        if len(set(labels)) > 1:
            return ExitDecision(True, self.reason, signal_value=tau)
        return NO_EXIT


DEFAULT_EXIT_CHAIN: list[ExitRule] = [
    DataGapRule(),
    StopLossRule(),
    TopologicalBreakRule(),
    MaxHoldingDaysRule(),
    CliqueBreakRule(),
]
"""Reihenfolge ist bewusst: Risiko-/Datenqualitäts-Regeln zuerst, dann die
strukturelle Integritätsprüfung (topologisch, dann Mitgliedschaft), dann
Zeit-Limits. TopologicalBreakRule ist per BasketParams.topo_break_enabled
standardmässig AUS (siehe config.py) -- ohne Wirkung, bis ein Run das
bewusst aktiviert."""


def evaluate_exit_chain(
    position: Position, ctx: BarContext, chain: list[ExitRule] = DEFAULT_EXIT_CHAIN,
) -> ExitDecision:
    for rule in chain:
        decision = rule.check(position, ctx)
        if decision.should_exit:
            return decision
    return NO_EXIT
