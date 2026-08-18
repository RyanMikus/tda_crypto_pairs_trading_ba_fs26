# data/raw/fetched/

Rohdaten von zwei externen Quellen: **CoinGecko** (Asset-Universum + Marktkennzahlen) und
**Binance** (Kerzendaten). Erzeugt von den Notebooks in `notebooks/01_data_aquisition/`,
Pfade zentral definiert in `src/config.py`.

## `assets.parquet`

Das für die gesamte Analyse verwendete Coin-Universum (welche Coins gelten über den
betrachteten Zeitraum als "relevant", und mit welchem Binance-Symbol sie gehandelt werden).
Erzeugt von `notebooks/01_data_aquisition/01_assets.ipynb` über die
[CoinGecko-API](https://docs.coingecko.com/reference/coins-markets) (Coins-Markets,
Coin-Tickers, Exchanges-List) — inkl. Filterung auf Mindest-Handelsvolumen, Top-Market-Cap
und Listung an mindestens `MIN_TOP_EXCHANGE_COUNT` grossen Exchanges (siehe
`src/config.py`), sowie Ausschluss von Stablecoins. Das Universum wird pro vollständigem
Lauf einmal komplett aus historisch gecachten CoinGecko-Antworten neu rekonstruiert, nicht
Tag für Tag live berechnet.

Spalten: `asset_id`, `coingecko_id`, `symbol`, `name`, `circulating_supply`,
`total_supply`, `max_supply`, `top_exchange_count`, `binance_symbol` (Binance-Spot-Ticker,
z. B. `AAVEUSDT` — die Brücke zu den OHLCV-Daten unten).

## `asset_daily/asset_daily.parquet`

Tägliche Preis-/Marktkennzahlen pro Asset (`asset_id`), von CoinGecko
(`/coins/{id}/market_chart/range`). Spalten: `date`, `asset_id`, `price_usd`,
`market_cap_usd`, `global_volume_24h_usd`. Dient u. a. der Volumen-/Market-Cap-Filterung
bei der Basket-Bildung (`notebooks/02_tda_clustering/01_baskets.ipynb`).

## `asset_ohlcv_5m/ohlcv_YYYY-MM.parquet`

5-Minuten-Kerzendaten pro Kalendermonat (ein Parquet-File je Monat), von Binance's
öffentlicher Klines-API (`https://data-api.binance.vision/api/v3/klines`). Erzeugt von
`notebooks/01_data_aquisition/02_ohlcv.ipynb` für alle `binance_symbol`-Werte aus
`assets.parquet`.

Spalten: `timestamp` (UTC, Kerzen-Öffnungszeit), `market_symbol`, `open`, `high`, `low`,
`close`, `volume_base`, `volume_quote`, `trade_count`. Das ist die Basisauflösung für
sämtliche Backtests — andere Kerzenlängen (10/15/30/60/240 min) werden zur Laufzeit per
`src.data.resample_ohlcv_panel` daraus abgeleitet, nicht separat gespeichert.
