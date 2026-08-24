import numpy as np
import pandas as pd
import pytest

from src.config import OHLCV_DIR
from src.data import full_candle_index, load_ohlcv_panel, resample_ohlcv_panel


def _write_month(tmp_path, year, month, start_day, n_candles):
    idx = pd.date_range(f"{year:04d}-{month:02d}-{start_day:02d}", periods=n_candles, freq="5min", tz="UTC")
    df = pd.DataFrame({"timestamp": idx, "market_symbol": "AAAUSDT", "close": 1.0})
    df.to_parquet(tmp_path / f"ohlcv_{year:04d}-{month:02d}.parquet", index=False)


def _real_data_start() -> pd.Timestamp:
    """Frühester Zeitstempel der chronologisch ersten vorhandenen ohlcv_*.parquet-Datei
    -- ersetzt das inzwischen entfernte src.data.available_data_range nur für die
    Real-Data-Fixtures unten, die lediglich einen gültigen Startpunkt brauchen."""
    files = sorted(OHLCV_DIR.glob("ohlcv_*.parquet"))
    first = pd.read_parquet(files[0], columns=["timestamp"])
    return pd.to_datetime(first["timestamp"], utc=True).min()


def _five_min_panel(n=13):
    """13 5-Min-Kerzen ab 2024-01-01 00:00, close=0..12 (linear ansteigend) --
    macht jede Aggregationsregel von Hand nachrechenbar (siehe Testfaelle)."""
    idx = full_candle_index(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=5 * (n - 1)), candle_minutes=5)
    close = pd.DataFrame({"BTCUSDT": np.arange(n, dtype=float)}, index=idx)
    return {
        "open": close.copy(),
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume_base": pd.DataFrame({"BTCUSDT": np.ones(n)}, index=idx),
    }


class TestResampleOhlcvPanel:
    """Ueberprueft, dass 10/15/30-Min-Kerzen aus den 5-Min-Rohdaten korrekt
    aggregiert werden: open=first, high=max, low=min, close=last,
    volume=sum je Zielfenster (siehe src.data._RESAMPLE_AGG_BY_FIELD)."""

    def test_15min_aggregation_matches_manual_calculation(self):
        panel = _five_min_panel(n=13)  # close 0..12, letztes 15-Min-Fenster unvollstaendig (nur 1 Kerze: 12)
        out = resample_ohlcv_panel(panel, source_minutes=5, target_minutes=15)

        expected_index = pd.DatetimeIndex([
            "2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30",
            "2024-01-01 00:45", "2024-01-01 01:00",
        ], tz="UTC")
        assert list(out["close"].index) == list(expected_index)

        # Fenster [00:00,00:15) enthaelt close-Werte 0,1,2 (high=1,2,3 / low=-1,0,1)
        assert out["open"]["BTCUSDT"].iloc[0] == 0.0     # erste Kerze im Fenster
        assert out["close"]["BTCUSDT"].iloc[0] == 2.0    # letzte Kerze im Fenster
        assert out["high"]["BTCUSDT"].iloc[0] == 3.0     # max(1,2,3)
        assert out["low"]["BTCUSDT"].iloc[0] == -1.0     # min(-1,0,1)
        assert out["volume_base"]["BTCUSDT"].iloc[0] == 3.0  # sum(1,1,1)

        # letztes Fenster [01:00,01:15) enthaelt nur EINE Kerze (close=12) -- Rand darf nicht crashen
        assert out["close"]["BTCUSDT"].iloc[-1] == 12.0
        assert out["open"]["BTCUSDT"].iloc[-1] == 12.0
        assert out["volume_base"]["BTCUSDT"].iloc[-1] == 1.0

    def test_10min_aggregation(self):
        panel = _five_min_panel(n=6)  # close 0..5
        out = resample_ohlcv_panel(panel, source_minutes=5, target_minutes=10)

        # Fenster [00:00,00:10) enthaelt close 0,1 -> open=0, close=1, high=2, low=-1, vol=2
        assert out["open"]["BTCUSDT"].iloc[0] == 0.0
        assert out["close"]["BTCUSDT"].iloc[0] == 1.0
        assert out["high"]["BTCUSDT"].iloc[0] == 2.0
        assert out["low"]["BTCUSDT"].iloc[0] == -1.0
        assert out["volume_base"]["BTCUSDT"].iloc[0] == 2.0

    def test_30min_aggregation(self):
        panel = _five_min_panel(n=13)  # close 0..12, genau 2 volle 30-Min-Fenster + Rest
        out = resample_ohlcv_panel(panel, source_minutes=5, target_minutes=30)

        # Fenster [00:00,00:30) enthaelt close 0..5 -> open=0, close=5, high=6, low=-1, vol=6
        assert out["open"]["BTCUSDT"].iloc[0] == 0.0
        assert out["close"]["BTCUSDT"].iloc[0] == 5.0
        assert out["high"]["BTCUSDT"].iloc[0] == 6.0
        assert out["low"]["BTCUSDT"].iloc[0] == -1.0
        assert out["volume_base"]["BTCUSDT"].iloc[0] == 6.0

    def test_identity_when_target_equals_source(self):
        panel = _five_min_panel(n=5)
        out = resample_ohlcv_panel(panel, source_minutes=5, target_minutes=5)
        assert out is panel  # unveraendert zurueckgegeben, kein Resampling-Overhead

    def test_raises_when_target_not_multiple_of_source(self):
        panel = _five_min_panel(n=5)
        with pytest.raises(ValueError, match="Vielfaches"):
            resample_ohlcv_panel(panel, source_minutes=5, target_minutes=7)

    def test_raises_on_unknown_field(self):
        panel = {"unknown_field": _five_min_panel(n=5)["close"]}
        with pytest.raises(ValueError, match="Keine Resample-Regel"):
            resample_ohlcv_panel(panel, source_minutes=5, target_minutes=15)

    def test_raises_on_non_positive_minutes(self):
        panel = _five_min_panel(n=5)
        with pytest.raises(ValueError, match="positiv"):
            resample_ohlcv_panel(panel, source_minutes=5, target_minutes=0)


@pytest.mark.skipif(not OHLCV_DIR.exists(), reason="OHLCV-Rohdaten (data/raw/fetched/asset_ohlcv_5m/) nicht vorhanden")
class TestResampleOhlcvPanelOnRealData:
    """Laedt ein kleines echtes 5-Min-Zeitfenster (load_ohlcv_panel) und
    prueft, dass resample_ohlcv_panel darauf strukturell UND inhaltlich
    korrekt arbeitet -- Ergaenzung zu den synthetischen Unit-Tests oben."""

    @pytest.fixture(scope="class")
    def real_panel_5m(self):
        start = _real_data_start()
        symbols = None  # erste paar Symbole im Zeitraum reichen, siehe load_symbol_map-Reihenfolge
        panel = load_ohlcv_panel(
            start=start, end=start + pd.Timedelta(hours=6),
            fields=("open", "high", "low", "close", "volume_base"),
        )
        # auf 2 Symbole mit vollstaendigen Daten in diesem Fenster einschraenken,
        # damit die Vergleichsrechnung unten nicht an NaN-Luecken scheitert.
        complete_symbols = panel["close"].dropna(axis=1, how="any").columns[:2].tolist()
        assert len(complete_symbols) >= 1, "kein Symbol mit vollstaendigen Daten im Testfenster gefunden"
        return {field: df[complete_symbols] for field, df in panel.items()}

    def test_15min_panel_has_correct_shape_and_grid(self, real_panel_5m):
        out = resample_ohlcv_panel(real_panel_5m, source_minutes=5, target_minutes=15)

        # real_panel_5m deckt [start, start+6h] INKLUSIV ab (load_ohlcv_panel-
        # Semantik) -> 73 5-Min-Kerzen (00:00..06:00), 5min zu 15min = jeweils
        # 3 Rohkerzen pro Zielkerze -> 73/3 aufgerundet = 25 Zielkerzen
        # (letztes Fenster [06:00,06:15) enthaelt nur die eine Randkerze 06:00).
        n_source = len(real_panel_5m["close"])
        expected_n_target = -(-n_source // 3)  # ceil division
        assert len(out["close"]) == expected_n_target
        # Index-Raster tatsaechlich aequidistant auf 15min, keine krummen Zeitstempel
        diffs = out["close"].index.to_series().diff().dropna().unique()
        assert list(diffs) == [pd.Timedelta(minutes=15)]

    def test_15min_ohlc_consistency_with_source_window(self, real_panel_5m):
        """OHLC-Grundregel muss fuer JEDES aggregierte Fenster gelten:
        low <= open,close <= high -- unabhaengig vom genauen Wert eine
        harte strukturelle Kontrolle, dass die Aggregation nicht z.B.
        high/low vertauscht oder auf dem falschen Feld rechnet."""
        out = resample_ohlcv_panel(real_panel_5m, source_minutes=5, target_minutes=15)
        for field in ("open", "close"):
            assert (out[field] <= out["high"]).all().all()
            assert (out[field] >= out["low"]).all().all()
        assert (out["low"] <= out["high"]).all().all()

    def test_15min_close_matches_manual_groupby_on_source(self, real_panel_5m):
        """Vergleicht resample_ohlcv_panel NICHT gegen sich selbst, sondern
        gegen eine unabhaengig (pandas.resample statt manuellem groupby, aber
        mit einer ANDEREN Formulierung: .last() auf demselben Bin) berechnete
        Referenz -- deckt ab, dass close tatsaechlich der LETZTE Wert jedes
        15-Min-Fensters ist, nicht z.B. still der erste oder ein Mittelwert."""
        out = resample_ohlcv_panel(real_panel_5m, source_minutes=5, target_minutes=15)
        reference_close = real_panel_5m["close"].resample("15min").last()
        pd.testing.assert_frame_equal(out["close"], reference_close)

    def test_30min_volume_sums_to_source_total(self, real_panel_5m):
        """volume_base darf beim Aggregieren nichts verlieren/verdoppeln --
        Summe ueber alle 30-Min-Fenster muss der Summe der Rohdaten
        entsprechen (bis auf einen unvollstaendigen Rand-Bin am Ende, der
        hier durch das exakt 6h/30min=12-teilbare Fenster nicht auftritt)."""
        out = resample_ohlcv_panel(real_panel_5m, source_minutes=5, target_minutes=30)
        pd.testing.assert_series_equal(
            out["volume_base"].sum(), real_panel_5m["volume_base"].sum(), check_names=False,
        )

@pytest.mark.skipif(not OHLCV_DIR.exists(), reason="OHLCV-Rohdaten (data/raw/fetched/asset_ohlcv_5m/) nicht vorhanden")
class TestRawPanelCandleContinuity:
    """Prueft, dass aufeinanderfolgende 5-Min-Rohkerzen (load_ohlcv_panel)
    lueckenlos ineinander uebergehen: close einer Kerze sollte nahe am open
    der naechsten liegen. Das ist eine Eigenschaft der ROHDATEN (Datenqualitaet
    des Anbieters), nicht des Resamplings -- deshalb hier separat statt in
    TestResampleOhlcvPanelOnRealData."""

    @classmethod
    @pytest.fixture(scope="class")
    def real_panel_5m(cls):
        start = _real_data_start()
        panel = load_ohlcv_panel(
            start=start, end=start + pd.Timedelta(hours=6),
            fields=("open", "close"),
        )
        complete_symbols = panel["close"].dropna(axis=1, how="any").columns[:5].tolist()
        assert len(complete_symbols) >= 1, "kein Symbol mit vollstaendigen Daten im Testfenster gefunden"
        return {field: df[complete_symbols] for field, df in panel.items()}

    def test_close_is_close_to_next_open(self, real_panel_5m):
        """close(t) vs. open(t+1) darf relativ zum Preisniveau nur eine kleine
        Abweichung haben (hier: max. 5%) -- ein Bruch waere z.B. ein Zeichen
        fuer fehlende/ausgelassene Kerzen im Rohdatensatz. Bewusst keine
        Gleichheit gefordert (echte Maerkte koennen zwischen Kerzen minimal
        springen), nur eine Plausibilitaetsgrenze."""
        close = real_panel_5m["close"]
        open_ = real_panel_5m["open"]

        close_t = close.iloc[:-1].reset_index(drop=True)
        open_t_plus_1 = open_.iloc[1:].reset_index(drop=True)

        rel_gap = ((open_t_plus_1 - close_t) / close_t).abs()
        max_gap = rel_gap.max().max()
        assert max_gap < 0.05, f"groesste relative Luecke close(t)->open(t+1): {max_gap:.4%} (>5%, moegliche Datenluecke)"

