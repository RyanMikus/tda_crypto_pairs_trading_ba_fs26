"""Kern-Datentypen der Backtest-Engine: keine Abhängigkeit auf config.py/
cliques.py/strategy.py, um Import-Zyklen zu vermeiden (siehe Modul-Layout im
Plan: backtest -> signals -> clustering -> config, einbahnig).

`from __future__ import annotations` macht alle Typannotationen zu Strings --
dadurch dürfen BarContext/BacktestResult auf BasketParams/CliqueSnapshot/
BacktestConfig aus anderen backtest-Modulen referenzieren, ohne sie zur
Laufzeit zu importieren (nur unter TYPE_CHECKING, siehe unten).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.backtest.cliques import CliqueSnapshot
    from src.backtest.config import BacktestConfig, BasketParams


EXIT_REASONS = (
    "target",
    "stop",
    "timeout",
    "clique_break",
    "topological_break",
    "data_gap",
    "delisted",
    "risk_limit",
    "end_of_backtest",
)
"""Bekannte Exit-Gründe, rein dokumentarisch -- exit_reason bleibt ein offener
str, damit neue ExitRules kein Schema-Update brauchen."""


@dataclass(frozen=True)
class SpreadDefinition:
    """Das bei Entry eingefrorene Spread-Rezept -- alles Nötige, um auf jeder
    späteren Kerze denselben Spread/Z-Score zu rekonstruieren, ohne neu zu
    schätzen. Analogie zu features.compute_frozen_betas.

    spread_t = sum_i(weights[i] * log(price_i,t)); pairwise: weights=(1.0, -hedge_ratio).
    """

    kind: Literal["pairwise", "cluster"]
    asset_ids: tuple[int, ...]
    symbols: tuple[str, ...]
    weights: tuple[float, ...]
    mu: float  # Formations-Referenzwert für Diagnostik -- NICHT mehr für den Live-Z-Score verwendet, siehe rolling_zscore()
    sigma: float  # dito
    fitted_at: pd.Timestamp
    formation_start: pd.Timestamp
    diagnostics: dict = field(default_factory=dict)

    def spread_value(self, log_prices: np.ndarray) -> float:
        return float(np.dot(self.weights, log_prices))

    def zscore(self, log_prices: np.ndarray) -> float:
        if self.sigma == 0 or not np.isfinite(self.sigma):
            return 0.0
        return (self.spread_value(log_prices) - self.mu) / self.sigma

    def rolling_zscore(self, price_window: pd.DataFrame) -> float | None:
        """Rollierender Z-Score (klassisches Bollinger-Band-Prinzip): Mittelwert/
        Std werden NEU aus price_window berechnet (die letzten N Kerzen bis
        einschliesslich jetzt, kausal via PriceWindow.history -- Guard G2
        verhindert strukturell jeden Blick über die aktuelle Bar hinaus),
        statt aus dem bei Formation eingefrorenen mu/sigma. Nur die Hedge-
        Ratio (self.weights) bleibt fix -- das ist die Kointegrationsbeziehung
        aus OLS/Johansen, ein anderes Konzept als das Band selbst.

        Args:
            price_window: (n_candles, n_symbols)-DataFrame, Spalten in
                DERSELBEN Reihenfolge wie self.symbols (siehe
                PriceWindow.history(self.symbols, ...)).

        Returns:
            Aktueller Z-Score, oder None falls die Fenster-Std 0/nicht endlich
            ist (kein sinnvolles Signal möglich -- z.B. konstante Preise).
        """
        log_prices = np.log(price_window[list(self.symbols)].to_numpy())
        spread_series = log_prices @ np.asarray(self.weights)
        sigma = float(spread_series.std(ddof=0))
        if not np.isfinite(sigma) or sigma == 0:
            return None
        return float((spread_series[-1] - spread_series.mean()) / sigma)


@dataclass(frozen=True)
class EntrySignal:
    spread: SpreadDefinition
    direction: int  # +1 = Spread long, -1 = Spread short
    signal_value: float  # Z-Score bei Signalauslösung
    spread_value: float
    priority: float  # Ranking bei knapper Kapazität, i.d.R. |signal_value|
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: str | None = None
    signal_value: float | None = None


NO_EXIT = ExitDecision(should_exit=False)


@dataclass
class Leg:
    asset_id: int
    symbol: str
    weight: float  # signiertes Log-Preis-Gewicht aus SpreadDefinition
    quantity: float = 0.0  # signierte Stückzahl, bei Fill gesetzt
    entry_price: float = float("nan")
    exit_price: float | None = None
    entry_notional: float = 0.0
    last_price: float = float("nan")  # letzter gültiger Mark-Preis, für NaN-Bars (siehe Portfolio.mark_to_market)


@dataclass
class Position:
    position_id: str
    strategy: str
    basket: str
    spread: SpreadDefinition
    params: "BasketParams"  # eingefroren bei Entry -- Kosten/Limits am Exit bleiben konsistent
    membership_key: str
    discovery_timestamp: pd.Timestamp
    entry_death: float  # = epsilon* der Clique bei Entry, eingefroren (siehe CliqueSnapshot.death)
    entry_lookback_days: int  # = CliqueSnapshot.lookback_days bei Entry, Default-Fenster für TopologicalBreakRule
    direction: int
    legs: list[Leg]
    entry_timestamp: pd.Timestamp
    entry_signal_value: float
    entry_spread_value: float
    gross_notional: float
    entry_equity: float
    entry_cost: float = 0.0
    bars_held: int = 0
    unrealized_pnl: float = 0.0
    mfe: float = 0.0  # max favorable excursion (PnL-Einheiten)
    mae: float = 0.0  # max adverse excursion
    missing_price_bars: int = 0  # aufeinanderfolgende NaN-Bars -> data_gap-Exit

    @property
    def symbols(self) -> list[str]:
        return [leg.symbol for leg in self.legs]

    @property
    def asset_ids(self) -> list[int]:
        return [leg.asset_id for leg in self.legs]


class PriceWindow:
    """Read-only-Sicht auf die Preismatrizen, hart bei bar_index abgeschnitten.

    Strukturelle Look-Ahead-Guard (Plan-Guard G2): eine Strategie bekommt nur
    dieses Objekt, nie die vollen Arrays -- .history()/.last() können nie über
    bar_index hinaus lesen. Felder mit führendem Unterstrich sind bewusst
    "privat"; Strategien sollen ausschliesslich die public Methoden nutzen.
    """

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        symbol_to_col: dict[str, int],
        timestamps: pd.DatetimeIndex,
        bar_index: int,
    ) -> None:
        self._arrays = arrays
        self._symbol_to_col = symbol_to_col
        self._timestamps = timestamps
        self.bar_index = bar_index

    def _cols(self, symbols: Sequence[str]) -> list[int]:
        missing = [s for s in symbols if s not in self._symbol_to_col]
        if missing:
            raise KeyError(f"Symbole nicht im Preis-Panel: {missing}")
        return [self._symbol_to_col[s] for s in symbols]

    def history(self, symbols: Sequence[str], n_candles: int, field: str = "close") -> pd.DataFrame:
        """Zeilen [bar_index - n_candles + 1 .. bar_index], EINSCHLIESSLICH bar_index."""
        start = self.bar_index - n_candles + 1
        if start < 0:
            raise ValueError(
                f"history() braucht {n_candles} Kerzen vor/inkl. bar_index={self.bar_index}, "
                f"aber es sind nur {self.bar_index + 1} vorhanden (Warm-up-Guard G7)."
            )
        cols = self._cols(symbols)
        arr = self._arrays[field][start:self.bar_index + 1][:, cols]
        idx = self._timestamps[start:self.bar_index + 1]
        return pd.DataFrame(arr, index=idx, columns=list(symbols))

    def last(self, symbols: Sequence[str], field: str = "close") -> pd.Series:
        cols = self._cols(symbols)
        vals = self._arrays[field][self.bar_index, cols]
        return pd.Series(vals, index=list(symbols))

    def log_last(self, symbols: Sequence[str]) -> np.ndarray:
        return np.log(self._arrays["close"][self.bar_index, self._cols(symbols)])

    @property
    def timestamp(self) -> pd.Timestamp:
        return self._timestamps[self.bar_index]


@dataclass(frozen=True)
class BarContext:
    """Was eine Strategie auf einer Kerze sehen darf."""

    timestamp: pd.Timestamp
    bar_index: int
    basket: str
    params: "BasketParams"
    snapshot: "CliqueSnapshot | None"
    prices: PriceWindow
    open_positions: tuple[Position, ...]
    equity: float
    free_slots: int


@dataclass
class BacktestResult:
    run_id: str
    config: "BacktestConfig"
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    metrics_by_basket: pd.DataFrame
    positions_curve: pd.DataFrame | None
    meta: dict

    def summary(self) -> pd.Series:
        return pd.Series(self.metrics)
