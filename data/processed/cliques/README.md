# data/processed/cliques/

TDA-Clique-Discovery-Ergebnisse: für jeden Basket und jeden Zeitschritt die aktuell
persistenteste Clique (Gruppe kointegrierter/korrelierter Assets), gefunden per
`HelperFunctions.get_persistent_clusters` (`src/clustering/helpers.py`) über
Vietoris-Rips-Persistenzhomologie auf einer Korrelationsdistanz-Punktwolke. Erzeugt von
`notebooks/02_tda_clustering/04_clique_cache.ipynb` — einmal für den gesamten verfügbaren
Zeitraum, damit nachgelagerte Notebooks (Backtest-Engine, Signal-Screening) diese Dateien
nur noch einlesen statt die teure Discovery jedes Mal zu wiederholen.

Sechs gleichwertige Datensätze: {`raw`, `residual`} × {`6h`, `12h`, `24h`}
Discovery-Kadenz — `cliques_{variant}_{cadence}.parquet` + zugehörige
`cliques_{variant}_{cadence}.meta.json`.

- `raw` — Clustering direkt auf den Preis-Returns.
- `residual` — Clustering auf Markt-bereinigten Residual-Returns (siehe
  `notebooks/02_tda_clustering/03_residual_pointclouds.ipynb`).
- Kadenz (`6h`/`12h`/`24h`) — wie oft neu nach Cliquen gesucht wird.

## `.parquet`-Spalten

`discovery_date`, `basket` (z. B. "Basket 1" — nach Marktkapitalisierungs-Bändern
segmentiert, siehe `src/config.py::BASKET_UPPER_BOUNDS`-Äquivalent), `asset_ids`
(Mitglieder der gefundenen Clique), `persistence` (TDA-Persistenz der Clique),
`death` (Todeswert im Persistenzdiagramm), `size` (Anzahl Assets), `lookback_days`
(Kalibrierungsfenster für dieses Discovery-Ereignis).

## `.meta.json`-Felder

Lauf-Parameter des Discovery-Prozesses: `variant`, `cadence_hours`, `epsilon_cap`,
`basket_upper_bounds` (Markt-Cap-Grenzen der Baskets), `date_range`, `date_stride`,
sowie `best_lookback`/`best_threshold` pro Basket (aus einer vorgelagerten
Sensitivitätsanalyse übernommene, theoriegeleitet fixierte Werte — siehe
`notebooks/02_tda_clustering/01_baskets.ipynb`).
