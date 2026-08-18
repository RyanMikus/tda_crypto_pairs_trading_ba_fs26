# data/raw/

Unveränderte Rohdaten, direkt von externen APIs abgerufen. Nichts hier wird von Hand
bearbeitet — bei Bedarf (neuer Zeitraum, neue Assets) werden die Notebooks in
`notebooks/01_data_aquisition/` erneut ausgeführt.

- `fetched/` — die eigentlichen Rohdaten-Parquets (Assets-Universum, Tagesdaten, OHLCV).
- `cache/` (lokal, nicht versioniert — siehe `.gitignore`) — zwischengespeicherte rohe
  API-Antworten (CoinGecko), damit `01_assets.ipynb` bei einem erneuten Lauf nicht alles
  neu abfragen muss.
