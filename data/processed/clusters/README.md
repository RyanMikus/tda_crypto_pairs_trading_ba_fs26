# data/processed/clusters/

TDA-Cluster-Discovery-Ergebnisse: für jeden Basket und jeden Zeitschritt die aktuell
persistenteste Cluster (Gruppe kointegrierter/korrelierter Assets), gefunden per
`HelperFunctions.get_persistent_clusters` (`src/clustering/helpers.py`) über
Vietoris-Rips-Persistenzhomologie auf einer Korrelationsdistanz-Punktwolke. Erzeugt von
`notebooks/02_tda_clustering/04_cluster_cache.ipynb` — einmal für den gesamten verfügbaren
Zeitraum, damit nachgelagerte Notebooks (Backtest-Engine, Signal-Screening) diese Dateien
nur noch einlesen statt die teure Discovery jedes Mal zu wiederholen.

Sechs gleichwertige Datensätze: {`raw`, `residual`} × {`6h`, `12h`, `24h`}
Discovery-Kadenz — `clusters_{variant}_{cadence}.parquet` + zugehörige
`clusters_{variant}_{cadence}.meta.json`.

- `raw` — Clustering direkt auf den Preis-Returns.
- `residual` — Clustering auf Markt-bereinigten Residual-Returns (siehe
  `notebooks/02_tda_clustering/03_residual_pointclouds.ipynb`).
- Kadenz (`6h`/`12h`/`24h`) — wie oft neu nach Clustern gesucht wird.

## `.parquet`-Spalten

`discovery_date`, `basket` (z. B. "Basket 1" — nach Marktkapitalisierungs-Bändern
segmentiert, siehe `src/config.py::BASKET_UPPER_BOUNDS`-Äquivalent), `asset_ids`
(Mitglieder der gefundenen Cluster), `persistence` (TDA-Persistenz der Cluster),
`death` (Todeswert im Persistenzdiagramm), `size` (Anzahl Assets), `lookback_days`
(Kalibrierungsfenster für dieses Discovery-Ereignis).

## `.meta.json`-Felder

Lauf-Parameter des Discovery-Prozesses: `variant`, `cadence_hours`, `epsilon_cap`,
`basket_upper_bounds` (Markt-Cap-Grenzen der Baskets), `date_range`, `date_stride`,
sowie `best_lookback`/`best_threshold` pro Basket (fest hinterlegte Werte, siehe
`notebooks/02_tda_clustering/02_pointclouds.ipynb`/`03_residual_pointclouds.ipynb`,
Abschnitt 4 — die dortige Jaccard-Stabilitätsanalyse liefert ergänzende Einblicke,
ist aber kein alleiniges Auswahlkriterium).
