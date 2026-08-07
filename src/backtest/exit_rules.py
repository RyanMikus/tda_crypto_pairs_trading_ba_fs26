"""Engine-weite Break-/Exit-Bedingungen, unabhängig von der Strategie.

Geordnete Liste (`ExitRule`-Objekte) statt einzelner Booleans, damit neue
Break-Conditions ohne Änderungen an engine.py ergänzt werden können -- allen
voran der vom Nutzer später selbst gebaute topologische Struktur-Break-
Detektor (docs/project/BA_aufgabenstellung_zusatz.pdf, Single-Linkage +
fcluster bei epsilon* - delta): er reiht sich als weiteres ExitRule-Objekt in
DEFAULT_EXIT_CHAIN ein.

Die strategie-spezifische "target"-Exit-Bedingung (Z-Score kehrt in den
exit_z-Korridor zurück) läuft NICHT hier, sondern in
BaseSpreadZScoreStrategy.check_exit -- sie läuft in der Engine NACH dieser
Kette (siehe engine.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.backtest.types import BarContext, ExitDecision, NO_EXIT, Position


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
    da SpreadDefinition.zscore() unabhängig davon funktioniert, ob der Spread
    aus 2 (Pairwise) oder N (Cluster) Assets besteht."""

    reason: str = "stop"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if ctx.params.stop_z is None:
            return NO_EXIT
        log_prices = ctx.prices.log_last(position.spread.symbols)
        if np.isnan(log_prices).any():
            return NO_EXIT  # DataGapRule kümmert sich um fehlende Preise
        z = position.spread.zscore(log_prices)
        if abs(z) >= ctx.params.stop_z:
            return ExitDecision(True, self.reason, signal_value=z)
        return NO_EXIT


@dataclass
class TimeoutRule:
    reason: str = "timeout"

    def check(self, position: Position, ctx: BarContext) -> ExitDecision:
        if position.bars_held >= ctx.params.max_holding_candles:
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


DEFAULT_EXIT_CHAIN: list[ExitRule] = [
    DataGapRule(),
    StopLossRule(),
    TimeoutRule(),
    CliqueBreakRule(),
]
"""Reihenfolge ist bewusst: Risiko-/Datenqualitäts-Regeln zuerst. Ein
künftiger TopologicalBreakRule reiht sich hier einfach ein (z.B. vor
StopLossRule), ohne dass engine.py, BasketParams oder die bestehenden Regeln
angefasst werden müssen."""


def evaluate_exit_chain(
    position: Position, ctx: BarContext, chain: list[ExitRule] = DEFAULT_EXIT_CHAIN,
) -> ExitDecision:
    for rule in chain:
        decision = rule.check(position, ctx)
        if decision.should_exit:
            return decision
    return NO_EXIT
