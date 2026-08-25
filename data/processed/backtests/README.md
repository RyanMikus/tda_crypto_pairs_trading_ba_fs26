# data/processed/backtests/

Gespeicherte Backtest-Runs (`src/backtest/runs.py::save_run`), aufgerufen am Ende jedes
Notebooks in `notebooks/03_backtest/`. Pro Run ein Unterordner
`<run_id>/` (Zeitstempel + Label-Slug + Config-Fingerprint), Ordnerinhalt selbst steht in
`.gitignore` (`data/processed/backtests/*/`) — nur der Gesamt-Index unten wird versioniert.

## `_index.parquet`

Eine Zeile pro jemals gespeichertem Run, automatisch bei jedem `save_run`-Aufruf neu
aufgebaut (`list_runs(refresh_index=True)`). Spalten: `run_id`, `created_at`, `label`,
`description`, `strategy_name`, `clique_variant`, `baskets`, `backtest_start`,
`backtest_end`, `fingerprint`, sowie die wichtigsten Kennzahlen (`n_trades`,
`sharpe_daily`, `cagr_daily`, `max_drawdown_pct`, `runtime_seconds`) als Schnellübersicht.
Erlaubt `notebooks/03_backtest/02_run_comparison.ipynb`, mehrere Runs zu vergleichen, ohne
jeden einzelnen Run-Ordner öffnen zu müssen.

## Inhalt eines `<run_id>/`-Ordners (nicht versioniert)

`config.json` (vollständige `BacktestConfig` inkl. `BasketParams`), `metrics.json` (alle
berechneten Kennzahlen), `metrics_by_basket.parquet`, `trades.parquet` (jeder einzelne
Trade), `equity_curve.parquet` (inkl. Drawdown-Spalten).
