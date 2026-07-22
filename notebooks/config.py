from pathlib import Path

# Project root — one level above notebooks/
PROJECT_ROOT = Path(__file__).parent.parent

# Root paths — works from any notebook two levels deep (notebooks/*/)
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DIR          = DATA_DIR / "raw"
FETCHED_DIR      = RAW_DIR / "fetched"
PROCESSED_DIR    = DATA_DIR / "processed"
SRC_PATH         = PROJECT_ROOT / "Src"
SECRET_PATH      = SRC_PATH / "secret.cfg"
ASSETS_PATH      = FETCHED_DIR / "assets.parquet"
ASSET_DAILY_DIR  = FETCHED_DIR / "asset_daily"
ASSET_DAILY_PATH = ASSET_DAILY_DIR / "asset_daily.parquet"
OHLCV_DIR        = FETCHED_DIR / "asset_ohlcv_5m"

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
TOP_N              = 100
VS_CURRENCY        = "usd"
