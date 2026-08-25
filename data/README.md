# data/

Alle Marktdaten und Pipeline-Ergebnisse dieses Projekts. Zwei Hauptbereiche:

- `raw/` — unveränderte, direkt von externen APIs abgerufene Rohdaten (CoinGecko, Binance).
- `processed/` — daraus abgeleitete Zwischen- und Endergebnisse (TDA-Cliquen, Hyperparameter-Tuning, Backtest-Runs).

Jeder Unterordner enthält eine eigene `README.md` mit Details zu Herkunft, Format und
erzeugendem Notebook/Modul. Rohdaten werden von `notebooks/01_data_aquisition/` erzeugt,
alles unter `processed/` von den Notebooks in `notebooks/02_tda_clustering/` und
`notebooks/03_backtest/` sowie den zugehörigen `src/`-Modulen.
