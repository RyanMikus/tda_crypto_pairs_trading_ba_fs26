# data/processed/

Alle Ergebnisse, die aus den Rohdaten (`data/raw/`) abgeleitet sind — nichts hier wird von
Hand editiert, alles ist über die jeweiligen Notebooks/Module reproduzierbar.

- `clusters/` — TDA-Cluster-Discovery-Ergebnisse (`notebooks/02_tda_clustering/04_cluster_cache.ipynb`).
- `tuning/` — Optuna-Hyperparameter-Suchläufe (`notebooks/03_backtest/03_tune_dynamic_vol.ipynb`,
  `src/backtest/tuning.py`). In `.gitignore` (lokale Zwischenergebnisse), einzelne Dateien
  können aber bewusst mit `git add -f` versioniert werden.
- `backtests/` — gespeicherte Backtest-Runs (`src/backtest/runs.py::save_run`), aufgerufen
  von jedem Notebook in `notebooks/03_backtest/`.
