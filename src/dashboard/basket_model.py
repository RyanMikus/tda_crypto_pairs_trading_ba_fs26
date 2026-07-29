"""Unified basket configuration shared by every basket-aware view in the
dashboard (bar chart, universe-over-time, per-basket heatmap/dendrogram) --
replaces the two independent basket configs the notebooks used to define
(BASKET_UPPER_BOUNDS in 01_baskets.ipynb vs POINTCLOUD_BASKET_UPPER_BOUNDS in
03_pointclouds.ipynb)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import _bootstrap  # noqa: F401

from config import DATE_COL, ASSET_ID_COL, VOLUME_COL
from helperfunctions import HelperFunctions

helper = HelperFunctions()


@dataclass(frozen=True)
class BasketConfig:
    upper_bounds: tuple[float, ...]

    @property
    def labels(self) -> list[str]:
        return [f"Basket {n}" for n in range(1, len(self.upper_bounds) + 2)]


def compute_rolling_volume(universe_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    return helper.compute_rolling_volume_table(
        universe_df,
        id_column=ASSET_ID_COL,
        date_column=DATE_COL,
        volume_column=VOLUME_COL,
        lookback_days=lookback_days,
    )


def build_baskets_for_date(
    rolling_volume_df: pd.DataFrame, date, basket_config: BasketConfig
) -> pd.DataFrame:
    return helper.build_volume_baskets(
        rolling_volume_df,
        date,
        id_column=ASSET_ID_COL,
        date_column=DATE_COL,
        basket_upper_bounds=list(basket_config.upper_bounds),
    )


def basket_asset_ids(basket_df: pd.DataFrame, label: str) -> set:
    return set(basket_df.loc[basket_df["basket"] == label, ASSET_ID_COL])
