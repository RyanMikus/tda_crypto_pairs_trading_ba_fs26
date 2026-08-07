from pathlib import Path

# Project root — one level above src/
PROJECT_ROOT = Path(__file__).parent.parent

# Root paths — works from any notebook two levels deep (notebooks/*/)
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DIR          = DATA_DIR / "raw"
FETCHED_DIR      = RAW_DIR / "fetched"
PROCESSED_DIR    = DATA_DIR / "processed"
ASSETS_PATH      = FETCHED_DIR / "assets.parquet"
ASSET_DAILY_DIR  = FETCHED_DIR / "asset_daily"
ASSET_DAILY_PATH = ASSET_DAILY_DIR / "asset_daily.parquet"
OHLCV_DIR        = FETCHED_DIR / "asset_ohlcv_5m"
CLIQUES_DIR      = PROCESSED_DIR / "cliques"
BACKTESTS_DIR    = PROCESSED_DIR / "backtests"

# Shared column names
DATE_COL          = "date"
ASSET_ID_COL      = "asset_id"
TIMESTAMP_COL     = "timestamp"
CLOSE_COL         = "close"
MARKET_SYMBOL_COL = "market_symbol"
BASE_SYMBOL_COL   = "base_symbol"
VOLUME_COL        = "global_volume_24h_usd"

# Shared universe thresholds
MIN_VOLUME_24H_USD = 50_000_000
TOP_N               = 100
VS_CURRENCY         = "usd"

# Backtest-Engine: Kerzen-/Zeitkonstanten (Krypto handelt 24/7/365, kein Börsenkalender)
CANDLE_MINUTES     = 5
CANDLES_PER_DAY    = 24 * 60 // CANDLE_MINUTES  # 288
CANDLES_PER_YEAR   = 365 * 24 * 60 // CANDLE_MINUTES  # 105_120

# Binance Spot-Gebührensätze in bps (Stand Implementierung -- bei Bedarf gegen
# die aktuelle Binance-Gebührentabelle verifizieren, VIP-Stufen ändern sich).
# "standard" (kein BNB-Rabatt, kein VIP-Status) ist der Default in BasketParams.
BINANCE_FEE_PRESETS_BPS = {
    "standard": 10.0,
    "bnb_discount": 7.5,
    "vip1": 9.0,
}
