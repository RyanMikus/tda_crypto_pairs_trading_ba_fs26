"""Konfiguration der Backtest-Engine: BasketParams (pro Basket überschreibbar),
BacktestConfig (ein kompletter Run) sowie der Calibration/Backtest-Split-Helper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

import pandas as pd

from src.config import CANDLE_MINUTES

_TIMEDELTA_AUTO = "auto"


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp


def split_calibration_backtest(
    full_start, full_end, fraction: float = 0.2,
) -> tuple[DateRange, DateRange]:
    """Teilt [full_start, full_end] in einen Calibration- und einen Backtest-
    Zeitraum, nach demselben Muster wie BACKTEST_FRACTION in
    notebooks/03_signal_generation/01_forward_return_screening.ipynb: die
    letzten `fraction` des verfügbaren Zeitraums sind der Backtest/Test-
    Zeitraum (nie für Kalibrierung angefasst), der Rest ist Calibration.

    Returns:
        (calibration_range, backtest_range), beide als DateRange(start, end).
        calibration_range.end < backtest_range.start ist garantiert.
    """
    full_start = pd.Timestamp(full_start)
    full_end = pd.Timestamp(full_end)
    if full_end <= full_start:
        raise ValueError(f"full_end ({full_end}) muss nach full_start ({full_start}) liegen.")

    full_range_days = (full_end - full_start).days + 1
    backtest_days = max(1, int(full_range_days * fraction))
    backtest_start = full_end - pd.Timedelta(days=backtest_days - 1)
    calibration_end = backtest_start - pd.Timedelta(days=1)

    if calibration_end < full_start:
        raise ValueError(
            f"fraction={fraction} lässt keinen Calibration-Zeitraum übrig "
            f"(full_range_days={full_range_days})."
        )

    return DateRange(full_start, calibration_end), DateRange(backtest_start, full_end)


@dataclass(frozen=True)
class BasketParams:
    """Alle Strategie-/Ausführungsparameter, die PRO BASKET unterschiedlich
    sein dürfen. frozen=True, damit Overrides per dataclasses.replace
    entstehen und ein einmal gebautes Objekt nicht versehentlich mutiert wird.
    """

    # --- Signal ---
    entry_z: float = 2.0
    exit_z: float = 0.5

    # --- Break-Conditions (siehe exit_rules.py) ---
    stop_z: float | None = 4.0
    # Zeitlicher Exit skaliert PRO POSITION mit der bei Entry geschätzten
    # Half-Life (position.spread.diagnostics["half_life_periods"]) statt einer
    # festen Kerzenzahl -- Daumenregel nach Chan: Haltedauer als Vielfaches der
    # Half-Life, sonst brechen v.a. Paare mit langsamer Reversion regelmässig
    # am Timeout ab, bevor sie konvergieren (siehe Notebook-Test mit fixem
    # 10h-Timeout: 37% Zwangsschliessungen vor Reversion). Fehlt half_life_periods
    # (z.B. isoliert konstruierte Positionen in Tests), greift TimeoutRule nie.
    max_holding_half_life_multiplier: float = 4.0
    exit_on_clique_break: bool = True  # Mitgliedschaftswechsel beim nächsten Snapshot
    max_data_gap_candles: int = 6
    # Topologischer Struktur-Break-Detektor (Single-Linkage/fcluster bei ε*-δ, siehe
    # docs/project/BA_aufgabenstellung_zusatz.pdf und exit_rules.TopologicalBreakRule).
    # epsilon* = position.entry_death (eingefroren bei Entry aus CliqueSnapshot.death,
    # NICHT persistence). topo_break_window_candles=None -> Default ist
    # position.entry_lookback_days in Kerzen umgerechnet (CANDLES_PER_DAY).
    topo_break_enabled: bool = False
    topo_break_delta: float = 0.001
    topo_break_window_candles: int | None = None

    # --- Formation / Qualitätsfilter ---
    formation_window_candles: int = 288 * 7
    # Rollierendes Bollinger-Fenster für Entry/Exit/Stop-Z-Score (siehe
    # SpreadDefinition.rolling_zscore), bewusst deutlich kürzer als
    # formation_window_candles (das bleibt für Hedge-Ratio/ADF/Half-Life),
    # damit das Band tatsächlich zeitnah auf Volatilitätsänderungen reagiert.
    bollinger_window_candles: int = 288
    max_candidate_pairs: int = 5
    pair_prefilter_top_n: int = 15
    require_adf: bool = True
    adf_alpha: float = 0.05
    min_half_life_candles: float = 6.0
    max_half_life_candles: float = 288 * 2
    min_johansen_relations: int = 1

    # --- Positionslimits ---
    min_holding_candles: int = 1
    max_concurrent_positions: int = 3
    max_positions_per_asset: int = 1

    # --- Kapital / Ausführung ---
    capital_per_position: float = 0.10  # Anteil des AKTUELLEN Equity (Compounding)
    dollar_neutralize: bool = True
    fee_bps: float = 10.0  # Binance Spot Taker-Standardsatz (0.10 %)
    slippage_model: Literal["bps", "high_low"] = "bps"
    slippage_bps: float = 2.0
    slippage_high_low_factor: float = 0.25


_BASKET_PARAM_FIELDS = frozenset(f.name for f in fields(BasketParams))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass
class BacktestConfig:
    label: str
    description: str = ""

    calibration_range: DateRange | None = None
    backtest_range: DateRange | None = None
    baskets: tuple[str, ...] = ("Basket 1", "Basket 2", "Basket 3", "Basket 4")
    clique_variant: str = "raw_24h"  # z.B. "raw_6h"/"raw_12h"/"raw_24h"/"residual_6h"/"residual_12h"/"residual_24h"

    strategy_name: str = "pairwise_zscore"
    strategy_kwargs: dict = field(default_factory=dict)

    default_params: BasketParams = field(default_factory=BasketParams)
    basket_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    initial_capital: float = 100_000.0
    max_gross_exposure: float = 1.0
    max_total_positions: int = 12

    clique_availability_lag: pd.Timedelta | Literal["auto"] = _TIMEDELTA_AUTO
    max_snapshot_age: pd.Timedelta | Literal["auto"] = _TIMEDELTA_AUTO
    execution_lag_candles: int = 1
    fill_price_field: Literal["open", "close"] = "open"

    benchmark_symbol: str = "BTCUSDT"
    risk_free_rate: float = 0.0
    candle_minutes: int = CANDLE_MINUTES
    record_positions_curve: bool = False

    @property
    def start(self) -> pd.Timestamp:
        if self.backtest_range is None:
            raise ValueError("backtest_range ist nicht gesetzt.")
        return self.backtest_range.start

    @property
    def end(self) -> pd.Timestamp:
        if self.backtest_range is None:
            raise ValueError("backtest_range ist nicht gesetzt.")
        return self.backtest_range.end

    def params_for(self, basket: str) -> BasketParams:
        """default_params + Overrides dieses Baskets. Unbekannte Override-Keys
        werfen (ein Tippfehler in einem Notebook soll NICHT still ignoriert
        werden, sonst läuft der Backtest scheinbar erfolgreich mit falschen
        Parametern)."""
        override = self.basket_overrides.get(basket, {})
        unknown = set(override) - _BASKET_PARAM_FIELDS
        if unknown:
            raise ValueError(f"Unbekannte BasketParams-Felder für {basket!r}: {sorted(unknown)}")
        return replace(self.default_params, **override)

    def resolved_params(self) -> dict[str, BasketParams]:
        return {b: self.params_for(b) for b in self.baskets}

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "description": self.description,
            "calibration_range": asdict(self.calibration_range) if self.calibration_range else None,
            "backtest_range": asdict(self.backtest_range) if self.backtest_range else None,
            "baskets": list(self.baskets),
            "clique_variant": self.clique_variant,
            "strategy_name": self.strategy_name,
            "strategy_kwargs": self.strategy_kwargs,
            "default_params": asdict(self.default_params),
            "basket_overrides": self.basket_overrides,
            "resolved_params": {b: asdict(p) for b, p in self.resolved_params().items()},
            "initial_capital": self.initial_capital,
            "max_gross_exposure": self.max_gross_exposure,
            "max_total_positions": self.max_total_positions,
            "clique_availability_lag": self.clique_availability_lag,
            "max_snapshot_age": self.max_snapshot_age,
            "execution_lag_candles": self.execution_lag_candles,
            "fill_price_field": self.fill_price_field,
            "benchmark_symbol": self.benchmark_symbol,
            "risk_free_rate": self.risk_free_rate,
            "candle_minutes": self.candle_minutes,
            "record_positions_curve": self.record_positions_curve,
        }
        return _json_safe(d)

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestConfig":
        d = dict(d)
        d.pop("resolved_params", None)

        def _range(r):
            if r is None:
                return None
            return DateRange(pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))

        return cls(
            label=d["label"],
            description=d.get("description", ""),
            calibration_range=_range(d.get("calibration_range")),
            backtest_range=_range(d.get("backtest_range")),
            baskets=tuple(d.get("baskets", ("Basket 1", "Basket 2", "Basket 3", "Basket 4"))),
            clique_variant=d.get("clique_variant", "raw_24h"),
            strategy_name=d.get("strategy_name", "pairwise_zscore"),
            strategy_kwargs=d.get("strategy_kwargs", {}),
            default_params=BasketParams(**d.get("default_params", {})),
            basket_overrides=d.get("basket_overrides", {}),
            initial_capital=d.get("initial_capital", 100_000.0),
            max_gross_exposure=d.get("max_gross_exposure", 1.0),
            max_total_positions=d.get("max_total_positions", 12),
            clique_availability_lag=d.get("clique_availability_lag", _TIMEDELTA_AUTO),
            max_snapshot_age=d.get("max_snapshot_age", _TIMEDELTA_AUTO),
            execution_lag_candles=d.get("execution_lag_candles", 1),
            fill_price_field=d.get("fill_price_field", "open"),
            benchmark_symbol=d.get("benchmark_symbol", "BTCUSDT"),
            risk_free_rate=d.get("risk_free_rate", 0.0),
            candle_minutes=d.get("candle_minutes", CANDLE_MINUTES),
            record_positions_curve=d.get("record_positions_curve", False),
        )

    def fingerprint(self) -> str:
        """sha1 über to_dict() ohne label/description, 8 Zeichen -- macht
        sichtbar, ob zwei gespeicherte Runs dieselbe Konfiguration hatten."""
        payload = self.to_dict()
        payload.pop("label", None)
        payload.pop("description", None)
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:8]
