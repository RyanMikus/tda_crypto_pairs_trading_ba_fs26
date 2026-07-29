"""Mutable dashboard state + the combined-lookback date bound logic.

Two independent lookbacks eat into the available date range from the same
end: compute_rolling_volume_table needs `basket_lookback_days` days of
warmup, build_point_cloud needs `corr_lookback_days` days of warmup. The
earliest valid date is therefore data_min + max(both) - 1 (see the identical
off-by-one comment on the date_slider in 01_baskets.ipynb)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def min_valid_date(
    data_min: pd.Timestamp, basket_lookback_days: int, corr_lookback_days: int
) -> pd.Timestamp:
    warmup_days = max(basket_lookback_days, corr_lookback_days) - 1
    return data_min + pd.Timedelta(days=warmup_days)


def clamp_date(
    date: pd.Timestamp,
    data_min: pd.Timestamp,
    data_max: pd.Timestamp,
    basket_lookback_days: int,
    corr_lookback_days: int,
) -> pd.Timestamp:
    lower = min_valid_date(data_min, basket_lookback_days, corr_lookback_days)
    if date < lower:
        return lower
    if date > data_max:
        return data_max
    return date


@dataclass
class DashboardState:
    date: pd.Timestamp
    basket_lookback_days: int = 7
    corr_lookback_days: int = 30
    basket_upper_bounds: tuple[float, ...] = (100_000_000.0, 250_000_000.0, 1_000_000_000.0)
    regression_target: str = "BTC+ETH"  # "BTC" | "ETH" | "BTC+ETH" | "PCA"
    epsilon: float = 1.0
    heatmap_kind: str = "distance"  # "correlation" | "distance" -- distance shown by default
    history_hidden_labels: set = field(default_factory=set)  # basket/"Universe" labels hidden via legend click

    @property
    def basket_labels(self) -> list[str]:
        return [f"Basket {n}" for n in range(1, len(self.basket_upper_bounds) + 2)]

    @property
    def history_labels(self) -> list[str]:
        return self.basket_labels + ["Universe"]
