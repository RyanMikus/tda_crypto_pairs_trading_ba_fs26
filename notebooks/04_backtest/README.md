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
