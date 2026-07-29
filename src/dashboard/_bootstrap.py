"""Adds the notebook source directories to sys.path so dashboard modules can
import the shared HelperFunctions/cloudvisualisation/config code without
duplicating it. Import this before importing config/helperfunctions/
cloudvisualisation from anywhere in the dashboard package."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOKS_DIR = _PROJECT_ROOT / "notebooks"
_TDA_DIR = _NOTEBOOKS_DIR / "02_tda_clustering"

for _path in (_NOTEBOOKS_DIR, _TDA_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
