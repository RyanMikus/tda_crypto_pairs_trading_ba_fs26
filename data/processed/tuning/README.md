# data/processed/tuning/

Ergebnisse der Optuna/TPE-Walk-Forward-Hyperparametersuche
(`src/backtest/tuning.py::tune_basket_params`/`save_tuning_results`), ausgeführt über
`notebooks/04_backtest/08_tune_dynamic_vol.ipynb` (automatische Alle-Baskets-Suche oder das
interaktive Widget-Panel). Ordner steht in `.gitignore` — lokale Zwischenergebnisse,
einzelne Läufe werden nur bei Bedarf explizit mit `git add -f` versioniert.

## Aktuelles Format (pro Lauf ein Unterordner)

`<run_id>/` (z. B. `20260815-165601_quick-tuning-pairwise-zscore-basket-1/`) mit:

- `best_params.json` — pro Basket die besten gefundenen Parameter, `best_objective`
  (`mean(fold_returns) - std(fold_returns)` über alle Walk-Forward-Folds), `n_trials`,
  `n_folds`, sowie `candle_minutes`/`fee_preset` (welche Kerzenlänge/welches Kostenmodell
  für diesen Lauf verwendet wurde — macht spätere Vergleiche über
  `src.backtest.tuning.load_all_tuning_results` möglich).
- `trials_<basket>.parquet` — vollständige Trial-Historie dieses Basket-Laufs (ein Zeile
  pro Optuna-Trial, alle gesampelten Parameter + Zielgrösse).

## `optuna_*.db`

SQLite-Storage der Optuna-Studies (persistenter TPE-Sampler-Zustand) — ermöglicht das
Fortsetzen einer Suche über mehrere Notebook-Läufe hinweg und die Anzeige im
`optuna-dashboard` (siehe Abschnitt "Optuna-Dashboard manuell starten" in
`08_tune_dynamic_vol.ipynb`). `optuna_walkforward.db` ist die gemeinsame Default-Datei,
`optuna_<RUN_LABEL>.db` sind an einen bestimmten Lauf gekoppelte Studies.

## Alte Dateien (`best_params_bollinger.json`, `best_params_ewma.json`,
## `best_params_garch.json`, `tuning_pairtrading_ewma_*.parquet`, `optuna_trials_Basket_1.parquet`)

Legacy-Artefakte aus früheren, mittlerweile gelöschten Tuner-Skripten
(`notebooks/03_Tradeing/quick_tune_*.py`, altes `pairtrading_engine.py`) — nicht vom
aktuellen `src/backtest/tuning.py` erzeugt oder gelesen. Können gefahrlos gelöscht werden,
sobald sie nicht mehr als Referenz für alte Ergebnisse gebraucht werden.
