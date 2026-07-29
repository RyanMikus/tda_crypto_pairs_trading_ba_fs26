"""Point-cloud / correlation compute pipeline, cached per parameter combination
so dragging a slider doesn't re-trigger the full pipeline for values already
seen. Mirrors the per-scope loop in 03_pointclouds.ipynb chapter 2 and the
residualization in 04_residual_pointclouds.ipynb, but generalized to a single
(scope, regression_target) request instead of looping over every basket/factor
at once.

Every build() call returns a status dict (mirrors the status pattern the
notebooks already use for basket/factor comparisons: "ok" | "empty" |
"single" | "no_data" | "missing_factor") so the UI can render an inline
message instead of a blank/crashed plot."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from . import _bootstrap  # noqa: F401

from config import DATE_COL, ASSET_ID_COL
from helperfunctions import HelperFunctions
from .basket_model import BasketConfig, compute_rolling_volume, build_baskets_for_date, basket_asset_ids
from .data_loader import AppData
from .state import min_valid_date

MIN_CLUSTER_SIZE = 2  # matches helperfunctions.HelperFunctions.MIN_CLUSTER_SIZE default

helper = HelperFunctions()

UNIVERSE_SCOPE = "Universe"

REGRESSION_FACTOR_COLUMNS = {
    "BTC": ["BTC"],
    "ETH": ["ETH"],
    "BTC+ETH": ["BTC", "ETH"],
}
REGRESSION_TARGETS = ["BTC", "ETH", "BTC+ETH", "PCA"]


@dataclass(frozen=True)
class CloudRequest:
    date: pd.Timestamp
    basket_lookback_days: int
    corr_lookback_days: int
    basket_upper_bounds: tuple
    scope: str
    regression_target: str | None  # None = raw, else one of REGRESSION_TARGETS

    def cache_key(self):
        return (
            self.date.value,
            self.basket_lookback_days,
            self.corr_lookback_days,
            self.basket_upper_bounds,
            self.scope,
            self.regression_target,
        )


class _LRUCache:
    def __init__(self, maxsize: int = 64):
        self._maxsize = maxsize
        self._store: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)


class CloudComputeService:
    """Owns the rolling-volume cache and the per-request result cache for one
    loaded AppData instance."""

    def __init__(self, data: AppData):
        self._data = data
        self._rolling_volume_cache: dict = {}
        self._result_cache = _LRUCache(maxsize=64)

    def rolling_volume(self, basket_lookback_days: int) -> pd.DataFrame:
        cached = self._rolling_volume_cache.get(basket_lookback_days)
        if cached is not None:
            return cached
        table = compute_rolling_volume(self._data.universe_df, basket_lookback_days)
        self._rolling_volume_cache[basket_lookback_days] = table
        return table

    def build(self, req: CloudRequest) -> dict:
        key = req.cache_key()
        cached = self._result_cache.get(key)
        if cached is not None:
            return cached

        result = self._build_uncached(req)
        self._result_cache.put(key, result)
        return result

    def _build_uncached(self, req: CloudRequest) -> dict:
        if req.scope == UNIVERSE_SCOPE:
            exclude_ids = set(self._data.excluded_ids)
        else:
            rolling_volume_df = self.rolling_volume(req.basket_lookback_days)
            basket_config = BasketConfig(upper_bounds=req.basket_upper_bounds)
            basket_df = build_baskets_for_date(rolling_volume_df, req.date, basket_config)
            scope_ids = basket_asset_ids(basket_df, req.scope)

            if len(scope_ids) == 0:
                return {"status": "empty"}
            if len(scope_ids) == 1:
                only_id = next(iter(scope_ids))
                return {"status": "single", "symbol": self._data.id_to_symbol.get(only_id, f"ID:{only_id}")}

            all_ids = set(self._data.universe_df[ASSET_ID_COL].unique())
            exclude_ids = set(self._data.excluded_ids) | (all_ids - scope_ids)

        raw = helper.build_point_cloud(
            ohlcv_df=self._data.ohlcv_df,
            universe_df=self._data.universe_df,
            end_date=req.date,
            window_size_days=req.corr_lookback_days,
            exclude_asset_ids=exclude_ids,
        )
        if raw is None:
            return {"status": "no_data"}

        syms = [self._data.id_to_symbol.get(a, f"ID:{a}") for a in raw["asset_ids"]]
        returns_df = pd.DataFrame(raw["point_cloud"], columns=syms, index=raw["timestamps"])

        if req.regression_target:
            if req.regression_target == "PCA":
                returns_df = helper.residualize_against_pca(returns_df, n_components=1)
            else:
                factor_columns = REGRESSION_FACTOR_COLUMNS[req.regression_target]
                missing = [f for f in factor_columns if f not in returns_df.columns]
                if missing:
                    return {"status": "missing_factor", "missing": missing}
                returns_df = helper.residualize_returns(returns_df, factor_columns=factor_columns)

            if returns_df.shape[1] < 2:
                return {"status": "no_data"}

        corr_matrix = np.corrcoef(returns_df.to_numpy().T)
        dist_matrix = helper.correlation_distance(returns_df)

        return {
            "status": "ok",
            "symbols": list(returns_df.columns),
            "n_points": returns_df.shape[0],
            "corr_matrix": corr_matrix,
            "dist_matrix": dist_matrix,
            "start_date": raw["start_date"],
            "end_date": raw["end_date"],
        }


def compute_basket_history(
    service: CloudComputeService,
    basket_lookback_days: int,
    basket_upper_bounds: tuple,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Scans every date with a rolling-median volume entry and records basket
    membership counts -- adapted from the "Full Change Log Across All Dates"
    style scan in 01_baskets.ipynb, but only counting sizes (not the full
    per-asset change log)."""
    rolling_volume_df = service.rolling_volume(basket_lookback_days)
    basket_config = BasketConfig(upper_bounds=basket_upper_bounds)

    available_dates = sorted(rolling_volume_df[DATE_COL].dropna().unique())
    total = len(available_dates)

    rows = []
    for i, date in enumerate(available_dates):
        if cancel_check and cancel_check():
            break

        basket_df = build_baskets_for_date(rolling_volume_df, date, basket_config)
        counts = basket_df["basket"].value_counts()
        row = {"date": pd.Timestamp(date)}
        for label in basket_config.labels:
            row[label] = int(counts.get(label, 0))
        row["Universe"] = int(counts.sum())
        rows.append(row)

        if progress_callback and (i + 1) % 20 == 0:
            progress_callback(i + 1, total)

    if progress_callback:
        progress_callback(len(rows), total)

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def compute_cluster_history(
    service: CloudComputeService,
    data: AppData,
    basket_lookback_days: int,
    corr_lookback_days: int,
    basket_upper_bounds: tuple,
    regression_target: str,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """One-time (expensive) scan: for every historical date and every basket,
    builds the raw and regressed point cloud/correlation-distance once and
    caches the resulting linkage matrix + asset count. Cutting at a specific
    epsilon afterward (see cut_cluster_history) is cheap and doesn't require
    rebuilding anything -- only this scan needs re-running when lookback/
    bounds/regression_target change.

    Returns a dict keyed by (basket_label, "raw"|"regressed") -> DataFrame
    indexed by date with columns "linkage" (object dtype, ndarray or None)
    and "n_assets" (int)."""
    lower_date = min_valid_date(data.data_min, basket_lookback_days, corr_lookback_days)
    all_dates = pd.date_range(lower_date, data.data_max, freq="D", tz="UTC")
    basket_config = BasketConfig(upper_bounds=basket_upper_bounds)
    labels = basket_config.labels

    records = {(label, mode): [] for label in labels for mode in ("raw", "regressed")}
    total = len(all_dates)

    for i, date in enumerate(all_dates):
        if cancel_check and cancel_check():
            break

        for label in labels:
            for mode in ("raw", "regressed"):
                req = CloudRequest(
                    date=date, basket_lookback_days=basket_lookback_days,
                    corr_lookback_days=corr_lookback_days, basket_upper_bounds=basket_upper_bounds,
                    scope=label, regression_target=(None if mode == "raw" else regression_target),
                )
                result = service.build(req)
                if result.get("status") == "ok":
                    z = linkage(squareform(result["dist_matrix"], checks=False), method="single")
                    n_assets = len(result["symbols"])
                else:
                    z = None
                    n_assets = 0
                records[(label, mode)].append({"date": date, "linkage": z, "n_assets": n_assets})

        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(i + 1, total)

    # Actual dates processed, not the hardcoded full `total` -- matters if
    # cancel_check broke the loop early (a superseded/cancelled scan), so this
    # doesn't misreport 100% completion for a partial run.
    scanned = len(next(iter(records.values()))) if records else 0
    if progress_callback:
        progress_callback(scanned, total)

    return {key: pd.DataFrame(recs).set_index("date") for key, recs in records.items()}


def cut_cluster_history(
    cluster_history: dict, *, epsilon: float, min_cluster_size: int = MIN_CLUSTER_SIZE
) -> dict:
    """Cheap: re-cuts already-cached linkage matrices at the given epsilon --
    no point clouds are rebuilt. Returns a dict keyed the same way as
    compute_cluster_history's output, with DataFrame columns "n_clusters" and
    "n_assets"."""
    result = {}
    for key, df in cluster_history.items():
        n_clusters = []
        for z in df["linkage"]:
            if z is None or len(z) == 0:
                n_clusters.append(0)
                continue
            cluster_labels = fcluster(z, t=epsilon, criterion="distance")
            counts = pd.Series(cluster_labels).value_counts()
            n_clusters.append(int((counts >= min_cluster_size).sum()))
        result[key] = pd.DataFrame(
            {"n_clusters": n_clusters, "n_assets": df["n_assets"].to_numpy()}, index=df.index
        )
    return result


def build_wide_metric_df(cut_result: dict, *, labels: list, mode: str, metric: str) -> pd.DataFrame:
    """Reshapes cut_cluster_history's per-(label, mode) output into a single
    wide DataFrame for one mode/metric: one column per basket label plus a
    "Sum" column (sum across baskets) -- the same shape
    draw_universe_over_time_line_chart already expects."""
    columns = {}
    for label in labels:
        df = cut_result.get((label, mode))
        if df is not None:
            columns[label] = df[metric]

    wide = pd.DataFrame(columns)
    # Guarded separately from the "not empty" case: assigning a 0-column
    # DataFrame's own .sum(axis=1) result works fine, but wide.sum(axis=1) on
    # a DataFrame with zero columns raises ValueError ("Cannot set a
    # DataFrame without columns to the column Sum") rather than producing an
    # empty Series -- can happen if none of `labels` are present in cut_result.
    wide["Sum"] = wide.sum(axis=1) if wide.columns.size else 0
    return wide
