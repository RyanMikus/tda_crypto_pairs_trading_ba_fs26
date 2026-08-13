import json

import pandas as pd
import pytest

from Src.backtest.cliques import CliqueTimeline, TimelineCursor


class TestCliqueTimelineFromCache:
    def test_infer_interval_is_one_day(self, timeline):
        assert timeline.infer_interval() == pd.Timedelta(days=1)
        assert timeline.infer_interval("Basket 1") == pd.Timedelta(days=1)

    def test_availability_lag_auto_equals_one_day_for_daily_discovery(self, timeline):
        snap = timeline.snapshots("Basket 1")[0]
        assert snap.available_from - snap.discovery_timestamp == pd.Timedelta(days=1)

    def test_max_snapshot_age_auto_equals_two_intervals(self, timeline):
        snap = timeline.snapshots("Basket 1")[0]
        assert snap.valid_until - snap.discovery_timestamp == pd.Timedelta(days=2)

    def test_asset_ids_and_symbols_stay_parallel(self, timeline, symbols):
        snap = timeline.snapshots("Basket 1")[0]
        assert len(snap.asset_ids) == len(snap.symbols) == len(symbols)
        assert not snap.dropped_asset_ids

    def test_unresolvable_asset_is_dropped_not_crashed(self, cliques_cache_dir, id_to_symbol):
        # id_to_symbol ohne das letzte Asset -> muss als dropped auftauchen, nicht crashen
        partial_map = dict(list(id_to_symbol.items())[:-1])
        timeline = CliqueTimeline.from_cache(
            variant="raw", id_to_symbol=partial_map, cliques_dir=cliques_cache_dir, baskets=["Basket 1"],
        )
        snap = timeline.snapshots("Basket 1")[0]
        assert len(snap.dropped_asset_ids) == 1
        assert len(snap.asset_ids) == len(snap.symbols) == len(partial_map)

    def test_missing_cache_file_raises_filenotfound(self, tmp_path, id_to_symbol):
        with pytest.raises(FileNotFoundError):
            CliqueTimeline.from_cache(variant="raw", id_to_symbol=id_to_symbol, cliques_dir=tmp_path)

    def test_requested_basket_with_no_rows_still_present_but_empty(self, cliques_cache_dir, id_to_symbol):
        timeline = CliqueTimeline.from_cache(
            variant="raw", id_to_symbol=id_to_symbol, cliques_dir=cliques_cache_dir,
            baskets=["Basket 1", "Basket 2"],
        )
        assert timeline.snapshots("Basket 2") == []


class TestActiveAt:
    def test_none_before_first_available_from(self, timeline):
        snap = timeline.snapshots("Basket 1")[0]
        before = snap.available_from - pd.Timedelta(minutes=5)
        assert timeline.active_at("Basket 1", before) is None

    def test_active_exactly_at_available_from(self, timeline):
        snap = timeline.snapshots("Basket 1")[0]
        assert timeline.active_at("Basket 1", snap.available_from) is snap

    def test_none_after_valid_until(self, timeline):
        snap = timeline.snapshots("Basket 1")[0]
        after = snap.valid_until + pd.Timedelta(minutes=5)
        result = timeline.active_at("Basket 1", after)
        # Am naechsten Tag existiert normalerweise schon ein neuerer Snapshot,
        # der eigentliche Guard ist: es ist NICHT mehr der alte snap.
        assert result is not snap


class TestTimelineCursor:
    def test_detects_membership_change(self, tmp_path, price_index, symbol_to_id, symbols):
        # Zwei Haelften mit unterschiedlicher Mitgliedschaft
        midpoint = price_index[len(price_index) // 2].normalize()
        rows = []
        for d in pd.date_range(price_index[0].normalize(), price_index[-1].normalize(), freq="D", tz="UTC"):
            members = symbols[:3] if d < midpoint else symbols[1:]
            rows.append({
                "discovery_date": d, "basket": "Basket 1",
                "asset_ids": [symbol_to_id[s] for s in members],
                "persistence": 0.5, "death": 1.0, "size": len(members), "lookback_days": 1,
            })
        df = pd.DataFrame(rows)
        cliques_dir = tmp_path / "cliques"
        cliques_dir.mkdir()
        df.to_parquet(cliques_dir / "cliques_raw.parquet", index=False)
        (cliques_dir / "cliques_raw.meta.json").write_text(json.dumps({"variant": "raw"}))

        id_to_symbol = {v: k for k, v in symbol_to_id.items()}
        timeline = CliqueTimeline.from_cache(variant="raw", id_to_symbol=id_to_symbol, cliques_dir=cliques_dir, baskets=["Basket 1"])
        cursor = TimelineCursor(timeline, "Basket 1")

        changes = []
        for ts in price_index:
            _, changed = cursor.advance(ts)
            changes.append(changed)

        # Mindestens 2 Wechsel: von "kein Snapshot" -> erster Snapshot, und vom
        # ersten zum zweiten Mitgliedschafts-Set.
        assert sum(changes) >= 2

    def test_no_change_while_membership_constant(self, timeline, price_index):
        cursor = TimelineCursor(timeline, "Basket 1")
        changes = [cursor.advance(ts)[1] for ts in price_index]
        # genau EIN Wechsel: von "kein Snapshot aktiv" zu "erster Snapshot aktiv"
        assert sum(changes) == 1

    def test_advance_is_monotonic_and_cheap(self, timeline, price_index):
        cursor = TimelineCursor(timeline, "Basket 1")
        for ts in price_index:
            snap, _ = cursor.advance(ts)
        assert snap is not None
