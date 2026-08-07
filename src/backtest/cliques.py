"""Liest den Clique-Cache (data/processed/cliques/) ein und macht ihn für die
Engine nutzbar -- bewusst AGNOSTISCH gegenüber dem Discovery-Intervall: der
Cache ist aktuell täglich (siehe HelperFunctions.discover_cliques_over_dates),
soll aber später auch mit kürzeren Intervallen (2h/3h/4h/6h/12h) erzeugt
werden können, ohne dass dieses Modul angefasst werden muss. Deshalb:
`discovery_timestamp` bleibt immer ein voller Timestamp, nie `.normalize()`
oder `.date()` -- und das Intervall wird aus den Daten selbst inferiert.

Look-Ahead-Guard G1: HelperFunctions.build_point_cloud (src/clustering/
helpers.py:139-153) normiert end_date auf Mitternacht UTC und filtert
DATE_COL <= end_date -- eine für Tag D discoverte Clique kennt damit ALLE
288 Kerzen von Tag D. `availability_lag="auto"` kompensiert das, indem eine
Clique erst 1 Discovery-Intervall nach ihrem discovery_timestamp handelbar
wird.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import CLIQUES_DIR


def _membership_key(asset_ids) -> str:
    blob = ",".join(str(a) for a in sorted(asset_ids))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _median_timedelta(timedeltas: list[pd.Timedelta]) -> pd.Timedelta:
    if not timedeltas:
        return pd.Timedelta(days=1)
    seconds = sorted(td.total_seconds() for td in timedeltas)
    n = len(seconds)
    mid = n // 2
    med = seconds[mid] if n % 2 == 1 else (seconds[mid - 1] + seconds[mid]) / 2
    return pd.Timedelta(seconds=med)


@dataclass(frozen=True)
class CliqueSnapshot:
    discovery_timestamp: pd.Timestamp  # NIE normalize()/date() -- bleibt voller Timestamp
    available_from: pd.Timestamp  # discovery_timestamp + availability_lag
    valid_until: pd.Timestamp  # discovery_timestamp + max_snapshot_age
    basket: str
    asset_ids: tuple[int, ...]  # NUR aufgelöste Assets, 1:1 parallel zu symbols
    symbols: tuple[str, ...]  # 1:1 parallel zu asset_ids (asset_ids[i] <-> symbols[i])
    dropped_asset_ids: tuple[int, ...]  # nicht auflösbare Assets, informativ, NICHT in asset_ids
    persistence: float  # = epsilon_death der TDA-Discovery (H0: birth immer 0)
    size: int  # Grösse der ORIGINAL discoverten Clique (kann > len(asset_ids) sein, falls Assets dropped wurden)
    lookback_days: int
    membership_key: str  # sha1 über sorted(asset_ids), 12 Zeichen -- NACH dem Dropping berechnet


class CliqueTimeline:
    def __init__(
        self,
        snapshots_by_basket: dict[str, list[CliqueSnapshot]],
        meta: dict,
        interval: pd.Timedelta,
    ) -> None:
        self._snapshots_by_basket = snapshots_by_basket
        self._meta = meta
        self._interval = interval

    @classmethod
    def from_cache(
        cls,
        *,
        variant: Literal["raw", "residual"],
        id_to_symbol: dict[int, str],
        availability_lag: pd.Timedelta | Literal["auto"] = "auto",
        max_snapshot_age: pd.Timedelta | Literal["auto"] = "auto",
        baskets: list[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        cliques_dir: Path = CLIQUES_DIR,
    ) -> "CliqueTimeline":
        """Liest cliques_<variant>.parquet + .meta.json.

        Intervall-Agnostik (drei Regeln):
          1. discovery_date wird zu discovery_timestamp -- voller Timestamp,
             kein Code hier nimmt "1 Tag" an.
          2. infer_interval() = Median der Diffs je Basket. Bei
             availability_lag="auto": Lag = inferiertes Intervall, falls
             < 1 Tag, sonst 1 Tag (siehe Modul-Docstring, Guard G1 -- die
             tägliche Discovery sieht den ganzen Tag D).
          3. max_snapshot_age="auto" -> 2 * inferiertes Intervall. Ohne
             Ablauf würde ein Snapshot unbegrenzt aktiv bleiben, obwohl die
             Abwesenheit eines Folgeeintrags "keine Clique gefunden" bedeutet.
        """
        path = cliques_dir / f"cliques_{variant}.parquet"
        meta_path = cliques_dir / f"cliques_{variant}.meta.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} nicht gefunden -- zuerst "
                "notebooks/02_tda_clustering/04_clique_cache.ipynb laufen lassen."
            )

        df = pd.read_parquet(path)
        df["discovery_date"] = pd.to_datetime(df["discovery_date"], utc=True)
        df["asset_ids"] = df["asset_ids"].apply(list)
        meta = json.load(open(meta_path))

        if baskets is not None:
            df = df[df["basket"].isin(baskets)]
        if start is not None:
            df = df[df["discovery_date"] >= _ensure_utc(start)]
        if end is not None:
            df = df[df["discovery_date"] <= _ensure_utc(end)]
        df = df.sort_values(["basket", "discovery_date"]).reset_index(drop=True)

        interval_by_basket: dict[str, pd.Timedelta] = {}
        for basket, sub in df.groupby("basket"):
            interval_by_basket[basket] = cls._infer_interval_from_series(sub["discovery_date"])
        overall_interval = _median_timedelta(list(interval_by_basket.values()))

        snapshots_by_basket: dict[str, list[CliqueSnapshot]] = {}
        for basket, sub in df.groupby("basket"):
            interval = interval_by_basket[basket]
            lag = availability_lag
            if lag == "auto":
                lag = interval if interval < pd.Timedelta(days=1) else pd.Timedelta(days=1)
            age = max_snapshot_age
            if age == "auto":
                age = 2 * interval

            rows = []
            for row in sub.itertuples():
                original_ids = row.asset_ids
                resolved_ids, symbols, dropped = [], [], []
                for a in original_ids:
                    s = id_to_symbol.get(a)
                    if s is None:
                        dropped.append(a)
                    else:
                        resolved_ids.append(a)
                        symbols.append(s)
                ts = row.discovery_date
                rows.append(CliqueSnapshot(
                    discovery_timestamp=ts,
                    available_from=ts + lag,
                    valid_until=ts + age,
                    basket=basket,
                    asset_ids=tuple(resolved_ids),
                    symbols=tuple(symbols),
                    dropped_asset_ids=tuple(dropped),
                    persistence=float(row.persistence),
                    size=int(row.size),
                    lookback_days=int(row.lookback_days),
                    membership_key=_membership_key(resolved_ids),
                ))
            snapshots_by_basket[basket] = sorted(rows, key=lambda s: s.available_from)

        if baskets is not None:
            for b in baskets:
                snapshots_by_basket.setdefault(b, [])

        return cls(snapshots_by_basket, meta, overall_interval)

    @staticmethod
    def _infer_interval_from_series(ts_series: pd.Series) -> pd.Timedelta:
        diffs = ts_series.sort_values().diff().dropna()
        if len(diffs) == 0:
            return pd.Timedelta(days=1)
        return diffs.median()

    def infer_interval(self, basket: str | None = None) -> pd.Timedelta:
        if basket is None:
            return self._interval
        snaps = self._snapshots_by_basket.get(basket, [])
        ts_series = pd.Series([s.discovery_timestamp for s in snaps])
        return self._infer_interval_from_series(ts_series)

    def snapshots(self, basket: str) -> list[CliqueSnapshot]:
        """Sortiert nach available_from."""
        return self._snapshots_by_basket.get(basket, [])

    def active_at(self, basket: str, ts: pd.Timestamp) -> CliqueSnapshot | None:
        snaps = self.snapshots(basket)
        avail = [s.available_from for s in snaps]
        idx = bisect.bisect_right(avail, ts) - 1
        if idx < 0:
            return None
        snap = snaps[idx]
        return snap if snap.valid_until >= ts else None

    def coverage(self) -> pd.DataFrame:
        rows = []
        for basket, snaps in self._snapshots_by_basket.items():
            if not snaps:
                rows.append({"basket": basket, "n_snapshots": 0, "first": None, "last": None, "interval": None})
                continue
            rows.append({
                "basket": basket,
                "n_snapshots": len(snaps),
                "first": snaps[0].discovery_timestamp,
                "last": snaps[-1].discovery_timestamp,
                "interval": self.infer_interval(basket),
            })
        return pd.DataFrame(rows)

    def meta(self) -> dict:
        return self._meta

    def describe(self) -> str:
        lines = [f"Discovery-Intervall (Median über Baskets): {self._interval}"]
        for _, row in self.coverage().iterrows():
            lines.append(
                f"  {row['basket']}: {row['n_snapshots']} Snapshots, "
                f"{row['first']} → {row['last']}, Intervall {row['interval']}"
            )
        return "\n".join(lines)


class TimelineCursor:
    """Pro Basket ein monoton vorwärtslaufender Zeiger -- O(1) amortisiert pro
    Bar statt bisect bei jedem Aufruf über den ganzen Backtest."""

    def __init__(self, timeline: CliqueTimeline, basket: str) -> None:
        self._snaps = timeline.snapshots(basket)
        self._pos = -1
        self._current: CliqueSnapshot | None = None

    def advance(self, ts: pd.Timestamp) -> tuple[CliqueSnapshot | None, bool]:
        """(aktiver Snapshot, membership_changed) für Zeitpunkt ts.

        ts muss über aufeinanderfolgende Aufrufe monoton steigen (Bar-Loop)."""
        while self._pos + 1 < len(self._snaps) and self._snaps[self._pos + 1].available_from <= ts:
            self._pos += 1

        candidate = self._snaps[self._pos] if self._pos >= 0 else None
        if candidate is not None and candidate.valid_until < ts:
            candidate = None

        prev_key = self._current.membership_key if self._current is not None else None
        new_key = candidate.membership_key if candidate is not None else None
        changed = prev_key != new_key

        self._current = candidate
        return candidate, changed
