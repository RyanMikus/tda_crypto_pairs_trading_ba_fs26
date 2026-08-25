# Topological Data Analysis for Multi-Asset Crypto Mean-Reversion Trading

Bachelorarbeit FS 2026, Ostschweizer Fachhochschule (OST) — Ryan Mikus und Ilija Amato.

Untersucht, ob sich mittels topologischer Datenanalyse (Vietoris-Rips-Filtration,
persistente Homologie) handelbare Cliquen von Kryptowährungen mit Mean-Reversion-
Verhalten identifizieren lassen, und backtestet eine darauf aufbauende Pairs-Trading-
Strategie (Bollinger-Band/Z-Score) auf Binance-Daten.

Die vollständige Arbeit liegt unter [`docs/bachelorarbeit.pdf`](docs/bachelorarbeit.pdf),
die Verteidigungs-Präsentation unter [`docs/presentation.pdf`](docs/presentation.pdf).

## Setup

Siehe [`SETUP.md`](SETUP.md) für die Windows- (Conda) und Mac-Installation (venv).
Tests laufen über [`notebooks/run_tests.ipynb`](notebooks/run_tests.ipynb) oder
`python -m pytest`.

## Struktur

```
notebooks/
├── 01_data_aquisition/       Asset-Universum & OHLCV-Daten abrufen
├── 02_tda_clustering/        Liquiditäts-Baskets, Punktwolken, Clique-Discovery,
│                             Struktur-Break-Detektor
├── 03_backtest/              Pairwise-Z-Score-Backtest, Run-Vergleich, Tuning
├── BA_visualizations.ipynb   Abbildungen für die Bachelorarbeit (-> images/)
└── run_tests.ipynb           Testsuite direkt im Notebook ausführen

src/                          Wiederverwendbare Logik hinter den Notebooks
├── clustering/                TDA-Helpers (Punktwolken, Persistenz, Cliquen)
├── signals/                    Spread-/Feature-Berechnung
└── backtest/                   Engine, Strategie, Exit-Regeln, Tuning

data/
├── raw/                        Unveränderte Rohdaten (CoinGecko, Binance)
└── processed/                   Abgeleitete Ergebnisse (Cliquen, Tuning, Backtest-Runs)

docs/
├── bachelorarbeit.pdf           Die Arbeit selbst
├── presentation.pdf             Verteidigungs-Präsentation
├── project/                     Aufgabenstellungen & Zusatzmaterial des Referenten
├── references/                  Zitierte wissenschaftliche Literatur
└── uml/                         Architektur-Diagramme

images/                          Abbildungen der Arbeit (generiert von BA_visualizations.ipynb)
tests/                           pytest-Suite
```

Jeder Unterordner unter `data/` sowie `notebooks/03_backtest/` hat eine eigene
`README.md` mit Details zu Herkunft, Format und erzeugendem Notebook/Modul.

## Notebook-Reihenfolge

Für einen kompletten Durchlauf von Rohdaten bis Backtest-Ergebnis:

1. [`01_data_aquisition/01_assets.ipynb`](notebooks/01_data_aquisition/01_assets.ipynb) — Asset-Universum bestimmen
2. [`01_data_aquisition/02_ohlcv.ipynb`](notebooks/01_data_aquisition/02_ohlcv.ipynb) — OHLCV-Daten abrufen
3. [`02_tda_clustering/01_baskets.ipynb`](notebooks/02_tda_clustering/01_baskets.ipynb) — Liquiditäts-Baskets bilden
4. [`02_tda_clustering/02_pointclouds.ipynb`](notebooks/02_tda_clustering/02_pointclouds.ipynb) — Punktwolken & Parameter-Kalibrierung
5. [`02_tda_clustering/03_residual_pointclouds.ipynb`](notebooks/02_tda_clustering/03_residual_pointclouds.ipynb) — residualisierte Variante
6. [`02_tda_clustering/04_clique_cache.ipynb`](notebooks/02_tda_clustering/04_clique_cache.ipynb) — Clique-Cache berechnen
7. [`02_tda_clustering/05_topological_break.ipynb`](notebooks/02_tda_clustering/05_topological_break.ipynb) — Struktur-Break-Detektor
8. [`03_backtest/`](notebooks/03_backtest/) — Backtest, Run-Vergleich, Tuning (siehe eigene [README](notebooks/03_backtest/README.md))
