"""One-time data loading for the dashboard, mirroring the loading cells
duplicated at the top of 03_pointclouds.ipynb / 04_residual_pointclouds.ipynb."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import _bootstrap  # noqa: F401  (adds notebooks/ dirs to sys.path)

from config import (
    ASSETS_PATH, ASSET_DAILY_PATH, OHLCV_DIR,
    DATE_COL, ASSET_ID_COL, TIMESTAMP_COL, CLOSE_COL, MARKET_SYMBOL_COL,
)
from helperfunctions import STABLECOIN_SYMBOLS


@dataclass
class AppData:
    assets_df: pd.DataFrame
    id_to_symbol: dict
    sym_to_id: dict
    excluded_ids: set
    ohlcv_df: pd.DataFrame
    universe_df: pd.DataFrame
    data_min: pd.Timestamp
    data_max: pd.Timestamp


def load_app_data(progress_callback: Callable[[str], None] | None = None) -> AppData:
    def _report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    _report("Loading assets...")
    assets_df = pd.read_parquet(ASSETS_PATH)[["asset_id", "symbol"]].drop_duplicates()
    id_to_symbol = dict(zip(assets_df["asset_id"], assets_df["symbol"]))
    sym_to_id = dict(zip(assets_df["symbol"], assets_df["asset_id"]))
    excluded_ids = set(
        assets_df.loc[assets_df["symbol"].isin(STABLECOIN_SYMBOLS), "asset_id"].tolist()
    )

    _report("Loading OHLCV data...")
    ohlcv_files = sorted(OHLCV_DIR.glob("*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in ohlcv_files], ignore_index=True)
    raw["base_symbol"] = raw[MARKET_SYMBOL_COL].str.replace("USDT$", "", regex=True)
    raw = raw.merge(
        assets_df.rename(columns={"symbol": "base_symbol"}), on="base_symbol", how="left"
    )
    raw = raw.dropna(subset=["asset_id"])
    raw["asset_id"] = raw["asset_id"].astype(int)
    raw[TIMESTAMP_COL] = pd.to_datetime(raw[TIMESTAMP_COL], utc=True)
    raw[DATE_COL] = raw[TIMESTAMP_COL].dt.normalize()
    ohlcv_df = raw[[TIMESTAMP_COL, DATE_COL, ASSET_ID_COL, CLOSE_COL]].copy()

    _report("Loading universe data...")
    universe_df = pd.read_parquet(ASSET_DAILY_PATH)
    universe_df[DATE_COL] = pd.to_datetime(universe_df[DATE_COL], utc=True)

    data_min = ohlcv_df[DATE_COL].min()
    data_max = ohlcv_df[DATE_COL].max()

    _report("Done.")
    return AppData(
        assets_df=assets_df,
        id_to_symbol=id_to_symbol,
        sym_to_id=sym_to_id,
        excluded_ids=excluded_ids,
        ohlcv_df=ohlcv_df,
        universe_df=universe_df,
        data_min=data_min,
        data_max=data_max,
    )
