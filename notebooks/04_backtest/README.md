# Backtest-Notebooks

Alle Notebooks in diesem Ordner laufen auf derselben Engine (`src/backtest/`).
Konvention, damit Runs untereinander vergleichbar bleiben:

1. **Neues Backtest-Notebook = `_template.ipynb` kopieren.**
   `cp _template.ipynb NN_kurzname.ipynb`, `NN` zweistellig fortlaufend
   (analog zu `01_data_aquisition/`, `02_tda_clustering/`, `03_signal_generation/`).

2. **Nur zwei Zellen werden angepasst:**
   - die Konfigurationszelle (Basket-Auswahl, Strategie, `BasketParams`,
     `basket_overrides`)
   - die Interpretations-Zelle ganz am Ende (Freitext für die Auswertung)

   Weicht ein Notebook strukturell davon ab (z.B. andere Reihenfolge, andere
   Metriken), gehört die Änderung nach `src/backtest/`, nicht ins Notebook —
   sonst sind die Ergebnisse nicht mehr mit anderen Runs vergleichbar.

3. `_template.ipynb` selbst wird **nicht ausgeführt** (`RUN_LABEL =
   "TEMPLATE_DO_NOT_RUN"` löst einen `SystemExit` vor dem eigentlichen
   Backtest-Lauf aus). Der Guard verschwindet automatisch, sobald `RUN_LABEL`
   im kopierten Notebook geändert wird.

4. Jedes Notebook speichert am Ende genau einen Run
   (`src.backtest.runs.save_run`). Für mehrere Parametervarianten:
   mehrere Notebooks, oder eine Schleife über Configs in einem eigenen
   `NN_sweep_*.ipynb` — aber immer mit `save_run` pro Variante, damit jede
   Konfiguration einzeln im Vergleich auftaucht.

5. `04_run_comparison.ipynb` führt **nie** einen Backtest aus, sondern liest
   ausschließlich gespeicherte Runs aus `data/processed/backtests/` — bleibt
   dadurch in Sekunden reproduzierbar.
6. `06_strategy_comparison.ipynb` führt mehrere Strategien (Default:
   `pairwise_zscore` vs. `garch_wilks`) auf denselben Baskets/Zeitraum aus,
   speichert jeden Run einzeln (Punkt 4) und vergleicht sie danach mit
   denselben Bausteinen wie `04_run_comparison.ipynb`. Funktionales Pendant
   zum alten `notebooks/03_Tradeing/strategy_comparison.ipynb`, hier auf der
   aktuellen Engine statt der alten `pairtrading_engine.py`-Familie.
7. `08_tune_dynamic_vol.ipynb` bedient `src.backtest.tuning.tune_basket_params`
   für `pairwise_zscore` mit `symmetric_exit`/`dynamic_threshold_enabled`/
   `dynamic_sizing_enabled` fest an (siehe docs/project/"Suggested methods für
   improving trading performance.pdf" A.1/A.2/E) sowohl über eine automatische
   Alle-Baskets-Suche (Abschnitt 4) als auch über ein zusätzliches
   ipywidgets-Panel für einzelne Baskets (Abschnitt 4b, Checkbox "tunen" +
   Fix-/Bereichsfeld pro Parameter, siehe `src.backtest.tuning.SearchSpec`) --
   inkl. Checkbox, um automatisch alle Kerzenlängen (5/10/15/30/60/240 min)
   nacheinander durchzutunen.
8. `09_tuning_overview.ipynb` liest **alle** gespeicherten Tuning-Läufe aus
   `08` (`src.backtest.tuning.load_all_tuning_results`) und führt pro
   Basket×Kerzenlänge-Kombination einen vollständigen Backtest über den
   **gesamten** Zeitraum aus, geteilt in Test (`CALIBRATION_RANGE`,
   in-sample bzgl. Parameterwahl) und Validierung (`BACKTEST_RANGE`, echtes
   Out-of-Sample) -- inkl. Overfitting-Check (Sharpe Test vs. Validierung),
   Basket-/Kerzenlängen-Vergleich und mehreren Performance-Grafiken. Jede
   Kombination wird als zwei separate Runs gespeichert (Punkt 4,
   Label-Suffix `-test`/`-validation`).

## Look-Ahead-Guards

Jedes Notebook druckt vor dem Lauf `engine.describe_guards()` aus (G1–G8, siehe
`src/backtest/engine.py`). Wichtigster Punkt: der Calibration-Zeitraum
(`split_calibration_backtest`) darf sich nicht mit dem Backtest-Zeitraum
überschneiden — das wird beim Bauen der `BacktestConfig` hart geprüft.

## Bekannt fehlend (bewusst, siehe Plan)

Ein topologischer Struktur-Break-Detektor (Single-Linkage + `fcluster` bei
`ε* − δ`, siehe `docs/project/BA_aufgabenstellung_zusatz.pdf`) ist **nicht**
Teil der Engine — er wird bewusst separat und sorgfältig ergänzt. Die
Exit-Rule-Architektur (`src/backtest/exit_rules.py`) ist so gebaut, dass eine
solche Regel später als weiteres `ExitRule`-Objekt in `DEFAULT_EXIT_CHAIN`
eingehängt werden kann, ohne `engine.py` zu ändern.
