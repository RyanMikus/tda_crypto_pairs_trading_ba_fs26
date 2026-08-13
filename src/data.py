"""OHLCV-Panel-Loading, generalisiert aus notebooks/03_signal_generation/TradeData.ipynb.

`load_close_prices` ist der rückwärtskompatible Wrapper mit identischer
Semantik wie im Notebook (inkl. dropna(axis=0, how="any")) -- TradeData.ipynb
importiert ab jetzt von hier statt die Funktion lokal zu definieren.

`load_ohlcv_panel` ist die generalisierte Variante für die Backtest-Engine:
mehrere Felder (nicht nur close), kein Wegwerfen ganzer Zeitstempel wegen
eines einzelnen neu gelisteten Symbols (ein Asset, das erst ab Monat X
existiert, darf nicht rückwirkend den ganzen Zeitraum vorher aus dem Panel
löschen -- die Engine behandelt NaN pro Symbol selbst, siehe backtest/engine.py).
"""

from pathlib import Path
from typing import Sequence

import pandas as pd

from Src.config import ASSETS_PATH, OHLCV_DIR

OHLCV_FIELDS = ("open", "high", "low", "close", "volume_base", "volume_quote", "trade_count")


def _month_files(start_ts: pd.Timestamp, end_ts: pd.Timestamp, ohlcv_dir: Path) -> list[Path]:
    # Auf den 1. des jeweiligen Monats normieren, BEVOR date_range(freq="MS")
    # aufgerufen wird -- sonst liefert date_range bei start_ts/end_ts im
    # selben Kalendermonat (aber nicht beide am 1.) eine leere Liste, weil
    # "MS" das erste Monatsende-Datum >= start_ts sucht und das kann bereits
    # nach end_ts liegen.
    month_start = start_ts.replace(day=1).normalize()
    month_end = end_ts.replace(day=1).normalize()
    months = pd.date_range(month_start, month_end, freq="MS")
    files = sorted({ohlcv_dir / f"ohlcv_{m.year:04d}-{m.month:02d}.parquet" for m in months})
    return [f for f in files if f.exists()]


def load_close_prices(tickers: list[str], start_date: str, n_candles: int) -> pd.DataFrame:
    """Lädt close-Preise für die gegebenen Ticker ab start_date, begrenzt auf
    n_candles Kerzen. Liest nur die dafür nötigen Monatsdateien.

    Falls ab start_date weniger als n_candles Kerzen tatsächlich vorhanden sind,
    wird NICHT auf die volle Fenstergrösse bestanden -- es werden einfach so
    viele Kerzen wie vorhanden zurückgegeben.

    Args:
        tickers: Liste von market_symbol-Strings, z.B. ["BTCUSDT", "ETHUSDT"].
        start_date: Startdatum (inklusive), z.B. "2025-01-01".
        n_candles: Maximale Anzahl 5min-Kerzen ab start_date pro Ticker.

    Returns:
        DataFrame (timestamp x ticker) mit close-Preisen, NaN-Zeilen entfernt
        (nur Zeitstempel, an denen alle Ticker einen Wert haben).
    """
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = start_ts + pd.Timedelta(minutes=5 * n_candles) + pd.Timedelta(days=2)

    files = _month_files(start_ts, end_ts, OHLCV_DIR)
    if not files:
        raise FileNotFoundError(f"Keine OHLCV-Dateien für Zeitraum ab {start_date} gefunden.")

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "market_symbol", "close"])
        df = df[df["market_symbol"].isin(tickers)]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw[raw["timestamp"] >= start_ts]

    price_matrix = (
        raw.pivot_table(index="timestamp", columns="market_symbol", values="close", aggfunc="last")
        .sort_index()
    )

    missing = [t for t in tickers if t not in price_matrix.columns]
    if missing:
        print(f"⚠️  Keine Daten gefunden für: {missing}")

    price_matrix = price_matrix.dropna(axis=0, how="any")
    price_matrix = price_matrix.iloc[:n_candles]
    return price_matrix


def full_candle_index(start: pd.Timestamp, end: pd.Timestamp, candle_minutes: int = 5) -> pd.DatetimeIndex:
    """Lückenloser 5-Min-UTC-DatetimeIndex von start bis end (inklusive)."""
    return pd.date_range(start, end, freq=f"{candle_minutes}min", tz="UTC")


def load_ohlcv_panel(
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    n_candles: int | None = None,
    symbols: Sequence[str] | None = None,
    fields: Sequence[str] = ("close",),
    ohlcv_dir: Path = OHLCV_DIR,
    reindex_full_grid: bool = True,
) -> dict[str, pd.DataFrame]:
    """Lädt ein OHLCV-Panel {field: DataFrame(index=timestamp UTC, columns=market_symbol)}.

    Anders als load_close_prices wird KEIN dropna(axis=0, how="any") gemacht --
    ein einzelnes neu gelistetes Symbol würde sonst den kompletten Zeitraum
    davor aus dem Panel löschen. NaN pro Symbol muss von der aufrufenden Stelle
    (Engine) selbst behandelt werden.

    Args:
        start: Startzeitpunkt (inklusive).
        end: Endzeitpunkt (inklusive). Alternativ n_candles angeben.
        n_candles: Anzahl 5-Min-Kerzen ab start, falls end nicht angegeben ist.
        symbols: Einzuschliessende market_symbol-Strings. None = alle im Zeitraum
            vorkommenden Symbole.
        fields: Teilmenge von OHLCV_FIELDS, die als eigene DataFrames zurückgegeben
            werden sollen (Standard: nur "close").
        reindex_full_grid: falls True, wird jedes Feld-DataFrame auf einen
            lückenlosen 5-Min-Index reindiziert (fehlende Kerzen -> NaN) statt
            nur die tatsächlich vorhandenen Zeitstempel zu behalten.

    Returns:
        dict field -> DataFrame(index=timestamp, columns=market_symbol).
    """
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
    elif n_candles is not None:
        end_ts = start_ts + pd.Timedelta(minutes=5 * (n_candles - 1))
    else:
        raise ValueError("Entweder end oder n_candles muss angegeben werden.")

    unknown_fields = set(fields) - set(OHLCV_FIELDS)
    if unknown_fields:
        raise ValueError(f"Unbekannte OHLCV-Felder: {sorted(unknown_fields)}")

    files = _month_files(start_ts, end_ts + pd.Timedelta(days=1), ohlcv_dir)
    if not files:
        raise FileNotFoundError(f"Keine OHLCV-Dateien für Zeitraum {start_ts} bis {end_ts} gefunden.")

    columns_needed = ["timestamp", "market_symbol", *fields]
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=columns_needed)
        if symbols is not None:
            df = df[df["market_symbol"].isin(symbols)]
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw[(raw["timestamp"] >= start_ts) & (raw["timestamp"] <= end_ts)]

    grid = full_candle_index(start_ts, end_ts) if reindex_full_grid else None

    panel: dict[str, pd.DataFrame] = {}
    for field in fields:
        matrix = raw.pivot_table(
            index="timestamp", columns="market_symbol", values=field, aggfunc="last",
        ).sort_index()
        if grid is not None:
            matrix = matrix.reindex(grid)
        panel[field] = matrix

    return panel


def load_symbol_map(assets_path: Path = ASSETS_PATH) -> tuple[dict[int, str], dict[str, int]]:
    """(asset_id -> binance_symbol, binance_symbol -> asset_id).

    Assets ohne aufgelöstes Binance-Symbol (binance_symbol ist NaN) fehlen in
    beiden Richtungen.
    """
    assets_df = pd.read_parquet(assets_path)[["asset_id", "binance_symbol"]].dropna()
    id_to_symbol = dict(zip(assets_df["asset_id"].astype(int), assets_df["binance_symbol"]))
    symbol_to_id = {v: k for k, v in id_to_symbol.items()}
    return id_to_symbol, symbol_to_id
