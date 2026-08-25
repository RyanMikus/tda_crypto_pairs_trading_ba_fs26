#!/bin/bash
# ============================================================
# TDA Crypto Trading – Mac Setup (ohne Conda)
# ============================================================
# Dieses Script erstellt ein Python venv und installiert alles.
#
# VORAUSSETZUNG:
#   - Python 3.11 muss installiert sein
#   - Falls nicht: brew install python@3.11
#   - Xcode CLI Tools: xcode-select --install
#
# VERWENDUNG:
#   chmod +x setup_mac.sh
#   ./setup_mac.sh
# ============================================================

set -e  # Bei Fehler abbrechen

# Farben für Output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "=========================================="
echo "  TDA Crypto Trading – Mac Setup"
echo "=========================================="
echo ""

# --- 1. Python 3.11 finden ---
PYTHON=""
for cmd in python3.11 python3; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oE '3\.[0-9]+')
        if [ "$version" = "3.11" ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}Python 3.11 nicht gefunden!${NC}"
    echo ""
    echo "Installiere es mit Homebrew:"
    echo "  brew install python@3.11"
    echo ""
    echo "Falls Homebrew nicht installiert ist:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo -e "${GREEN}✓ Python gefunden: $($PYTHON --version)${NC}"

# --- 2. Xcode CLI Tools prüfen ---
if ! xcode-select -p &> /dev/null; then
    echo -e "${YELLOW}Xcode Command Line Tools werden installiert...${NC}"
    xcode-select --install
    echo "Bitte warte bis die Installation fertig ist und starte dieses Script erneut."
    exit 1
fi
echo -e "${GREEN}✓ Xcode CLI Tools vorhanden${NC}"

# --- 3. Altes venv löschen falls vorhanden ---
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Altes venv gefunden, wird gelöscht...${NC}"
    rm -rf "$VENV_DIR"
fi

# --- 4. Neues venv erstellen ---
echo ""
echo "Erstelle virtuelles Environment..."
$PYTHON -m venv "$VENV_DIR"
echo -e "${GREEN}✓ venv erstellt in $VENV_DIR${NC}"

# --- 5. venv aktivieren ---
source "$VENV_DIR/bin/activate"

# --- 6. pip upgraden ---
echo ""
echo "Aktualisiere pip..."
pip install --upgrade pip setuptools wheel --quiet

# --- 7. Pakete installieren ---
echo ""
echo "Installiere Pakete (das dauert ca. 3-5 Minuten)..."
pip install -r requirements.txt

# --- 8. Jupyter Kernel registrieren ---
echo ""
echo "Registriere Jupyter Kernel..."
python -m ipykernel install --user --name tda-trading --display-name "TDA Trading (Python 3.11)"

# --- 9. Test ---
echo ""
echo "Teste Installation..."
python -c "
import numpy; print(f'  numpy       {numpy.__version__}')
import pandas; print(f'  pandas      {pandas.__version__}')
import scipy; print(f'  scipy       {scipy.__version__}')
import sklearn; print(f'  scikit-learn {sklearn.__version__}')
import matplotlib; print(f'  matplotlib  {matplotlib.__version__}')
import plotly; print(f'  plotly      {plotly.__version__}')
import statsmodels; print(f'  statsmodels {statsmodels.__version__}')
import optuna; print(f'  optuna      {optuna.__version__}')
print()
print('  ✓ Alle Pakete erfolgreich installiert!')
"

echo ""
echo -e "${GREEN}=========================================="
echo "  Setup abgeschlossen!"
echo "==========================================${NC}"
echo ""
echo "Environment aktivieren:"
echo "  source .venv/bin/activate"
echo ""
echo "In VS Code:"
echo "  Cmd+Shift+P → 'Python: Select Interpreter'"
echo "  → .venv (Python 3.11) auswählen"
echo ""
