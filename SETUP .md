# Setup-Anleitung

## Projektstruktur

```
Projektordner/
├── requirements.txt      ← Alle Pakete (Single Source of Truth)
├── environment.yml       ← Windows-Setup (Conda, liest requirements.txt)
├── setup_mac.sh          ← Mac-Setup (venv + pip, liest requirements.txt)
└── SETUP.md              ← Diese Anleitung
```

Beide Plattformen verwenden dieselbe `requirements.txt`.
Neue Pakete **nur dort** hinzufügen.

---

## Windows-Setup (Conda)

### 1. Voraussetzungen installieren

**Miniconda** → [Download](https://docs.anaconda.com/miniconda/install/)

**Visual Studio Build Tools** → [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Installer starten
- Workload **"Desktop development with C++"** auswählen
- Installieren (ca. 2-3 GB, dauert 10-15 Min)
- PC neu starten

> **Hinweis:** VS Code ist NICHT dasselbe wie die Build Tools.
> VS Code = Editor. Build Tools = C++ Compiler. Du brauchst beides.

### 2. Environment erstellen

Anaconda Prompt öffnen, in den Projektordner navigieren:

```bash
cd C:\Users\DEIN_NAME\Pfad\zum\Projektordner
conda env create -f environment.yml
```

Dauert ca. 5-10 Minuten.

### 3. Aktivieren

```bash
conda activate tda-trading
```

Links im Terminal sollte jetzt `(tda-trading)` stehen.

### 4. In VS Code verwenden

1. `Ctrl+Shift+P` → **"Python: Select Interpreter"**
2. **tda-trading (Python 3.11)** auswählen
3. Für Jupyter Notebooks: gleichen Kernel wählen

### 5. Testen

```bash
python -c "import gtda; import matplotlib; import ccxt; print('Alles OK')"
```

### Environment aktualisieren / löschen

```bash
# Nach Änderungen an requirements.txt:
conda env update -f environment.yml --prune

# Komplett löschen und neu erstellen:
conda deactivate
conda env remove -n tda-trading
conda env create -f environment.yml
```

---

## Mac-Setup (venv + pip, ohne Conda)

### 1. Voraussetzungen installieren

**Homebrew** (falls noch nicht vorhanden):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Python 3.11:**
```bash
brew install python@3.11
```

**Xcode Command Line Tools:**
```bash
xcode-select --install
```

### 2. Setup-Script ausführen

Terminal öffnen, in den Projektordner navigieren:

```bash
cd /Pfad/zum/Projektordner
chmod +x setup_mac.sh
./setup_mac.sh
```

Das Script macht automatisch alles:
- Prüft ob Python 3.11 und Xcode vorhanden sind
- Erstellt ein `.venv` im Projektordner
- Installiert alle Pakete aus `requirements.txt`
- Registriert den Jupyter-Kernel
- Testet ob alles funktioniert

### 3. Aktivieren (jedes Mal vor dem Arbeiten)

```bash
source .venv/bin/activate
```

Links im Terminal sollte jetzt `(.venv)` stehen.

### 4. In VS Code verwenden

1. `Cmd+Shift+P` → **"Python: Select Interpreter"**
2. **.venv (Python 3.11)** auswählen
3. Für Jupyter Notebooks: gleichen Kernel wählen

### 5. Testen

```bash
python -c "import gtda; import matplotlib; import ccxt; print('Alles OK')"
```

### Environment aktualisieren / löschen

```bash
# Aktivieren und neue Pakete installieren:
source .venv/bin/activate
pip install -r requirements.txt

# Komplett löschen und neu erstellen:
deactivate
rm -rf .venv
./setup_mac.sh
```

---

## Warum Python 3.11?

| Paket | Einschränkung | Ergebnis |
|-------|---------------|----------|
| pandas ≥3.0 | braucht Python ≥3.11 | 3.10 fällt weg |
| giotto-tda 0.6.2 | Windows-Wheels nur bis 3.11 | 3.12 fällt weg |
| **→ Python 3.11** | **einziger Sweet Spot** | **funktioniert überall** |

## Warum nicht alle 72 Pakete pinnen?

Von den 72 Paketen in der alten `requirements.txt` sind nur ~15 **direkte Abhängigkeiten** (die du selbst importierst). Der Rest sind transitive Pakete die automatisch mitkommen. Alle exakt zu pinnen verursacht Versionskonflikte. Industriestandard: nur direkte Abhängigkeiten auflisten.

---

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError` in Jupyter | Falscher Kernel gewählt → Kernel auf `tda-trading` / `.venv` wechseln |
| giotto-tda kompiliert nicht | **Windows:** Build Tools installiert? **Mac:** `xcode-select --install` |
| `conda env create` ist sehr langsam | `conda install -n base conda-libmamba-solver` dann nochmal |
| `CondaError: Run 'conda init'` | `conda init` ausführen, Terminal neu starten |
| `brew: command not found` (Mac) | Homebrew installieren (siehe oben) |
| `python3.11: command not found` (Mac) | `brew install python@3.11` |
| Jupyter findet Kernel nicht | `python -m ipykernel install --user --name tda-trading` |
| Paketkonflikt | Environment löschen und neu erstellen (siehe oben) |
