"""Gemeinsame Bausteine fuer die drei schnellen Walk-Forward-Tuner
(quick_tune_ewma.py, quick_tune_bollinger.py, quick_tune_garch.py).

Kernidee ggue. den alten Tunern (tune_backtest.py/tune_bollinger.py/
tune_garch.py/fast_tune.py, siehe deren Git-Historie): der Nutzer gibt nur
noch an, WIE VIELE Walk-Forward-Fenster (n_windows) er ueber den
verfuegbaren Zeitraum sehen will -- train_days/val_days/step_days/
embargo_days werden daraus automatisch UND sinnvoll abgeleitet
(auto_walkforward_layout), bleiben aber ueberschreibbar (jeder Tuner nimmt
weiterhin optionale train_days/val_days/step_days/embargo_days-Argumente
entgegen, die auto_walkforward_layout nur als Fallback fuellt).

Fixe Kosten (Modul-Docstring-Historie: die alten Tuner hatten FIXED_FEE_BPS
= 1.5, ein unrealistisch guenstiger Satz). Hier stattdessen
Src.config.BINANCE_FEE_PRESETS_BPS["standard"] (10.0 bps, Binance-Spot-
Taker-Standard) als Default -- konsistent mit Src/backtest/config.py
(die andere, parallele Engine fuer 04_backtest), per fee_preset umschaltbar.

Beste Parameter je (Basket, Strategie) werden zentral abgelegt:
    data/processed/tuning/best_params_<strategie>.json
    { basket: {..params.., "objective": float, "n_trials": int, "tuned_at": iso-str} }
und zusaetzlich als BEST_PARAMS-Dict direkt oben in das jeweilige
*_pairtrading_engine.py-Modul geschrieben (siehe write_best_params_to_engine_module)
-- rein informativ (Docstring-Kommentar + Dict-Literal), die Engines lesen
diesen Wert NICHT automatisch, um bestehende Aufrufe nicht stillschweigend
zu veraendern.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from Src.config import BINANCE_FEE_PRESETS_BPS, PROCESSED_DIR

TUNING_DIR = PROCESSED_DIR / "tuning"
FIXED_INITIAL_CAPITAL = 10_000.0
FIXED_CONFIDENCE = 0.95  # Wilks-Toleranzintervall-Konfidenz, siehe tolerance.wilks_tolerance_limits

# Marker-Kommentare, zwischen denen write_best_params_to_engine_module den
# BEST_PARAMS-Block im Ziel-Modul ersetzt -- macht wiederholte Schreibvorgaenge
# idempotent (kein Anhaengen von Duplikaten bei jedem Tuning-Lauf).
_BEST_PARAMS_BEGIN = "# === BEST_PARAMS (auto-generiert von tuning_common.write_best_params_to_engine_module) ==="
_BEST_PARAMS_END = "# === ENDE BEST_PARAMS ==="


def resolve_fee_bps(fee_preset: str | float = "standard") -> float:
    """Loest einen Fee-Preset-Namen (siehe Src.config.BINANCE_FEE_PRESETS_BPS)
    oder direkt einen bps-Float auf.

    Args:
        fee_preset: entweder ein Schluessel in BINANCE_FEE_PRESETS_BPS
            ("standard"=10.0, "bnb_discount"=7.5, "vip1"=9.0) oder direkt ein
            bps-Wert als float/int.
    """
    if isinstance(fee_preset, (int, float)):
        return float(fee_preset)
    if fee_preset not in BINANCE_FEE_PRESETS_BPS:
        raise ValueError(
            f"Unbekannter fee_preset {fee_preset!r}, erwartet einen von "
            f"{sorted(BINANCE_FEE_PRESETS_BPS)} oder einen numerischen bps-Wert."
        )
    return BINANCE_FEE_PRESETS_BPS[fee_preset]


# ============================================================================
# Automatisches Walk-Forward-Layout
# ============================================================================

@dataclass
class WalkForwardLayout:
    train_days: int
    val_days: int
    step_days: int
    embargo_days: int
    n_windows: int


def auto_walkforward_layout(
    train_val_start,
    train_val_end,
    *,
    n_windows: int,
    val_fraction: float = 0.25,
    embargo_fraction: float = 0.05,
) -> WalkForwardLayout:
    """Leitet train_days/val_days/step_days/embargo_days automatisch aus dem
    verfuegbaren Trainings-/Validierungszeitraum und der gewuenschten Anzahl
    Walk-Forward-Fenster (n_windows) ab.

    Layout-Logik: der Gesamtzeitraum wird so aufgeteilt, dass genau
    n_windows Fenster mit konstanter Schrittweite hineinpassen, jedes Fenster
    besteht aus einem Trainings- und einem (kleineren) Validierungsblock plus
    einem kurzen Embargo-Puffer dazwischen (Leakage-Schutz durch die
    ueberlappende Lookback-Historie am Fensteruebergang):

        block_days   = total_days / n_windows           (Schrittweite)
        val_days     = round(block_days * val_fraction)  (min. 3 Tage)
        embargo_days = round(block_days * embargo_fraction) (min. 1 Tag)
        train_days   = block_days - val_days - embargo_days (Rest, min. 5 Tage)

    Bei zu kurzem Gesamtzeitraum fuer die gewuenschte n_windows-Anzahl werden
    die Mindestgroessen (train_days>=5, val_days>=3, embargo_days>=1)
    erzwungen -- generate_walkforward_windows (pairtrading_engine.py) liefert
    dann ggf. weniger als n_windows tatsaechliche Fenster (kein Fehler, siehe
    dortige Dokumentation), was der Aufrufer ueber die Laenge der
    zurueckgegebenen windows-Liste bemerkt.

    Args:
        train_val_start, train_val_end: verfuegbarer Trainings-/Validierungs-
            zeitraum (NICHT der finale Testzeitraum).
        n_windows: gewuenschte Anzahl Walk-Forward-Fenster (die einzige
            Pflichtangabe des Nutzers, siehe Modul-Docstring).
        val_fraction, embargo_fraction: Anteil des Fensters (block_days), der
            auf Validierung bzw. Embargo entfaellt -- restliches Fenster ist
            Training. Defaults so gewaehlt, dass Training deutlich groesser
            als Validierung bleibt (klassisches Walk-Forward-Verhaeltnis).

    Returns:
        WalkForwardLayout mit den abgeleiteten Werten.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows muss >= 1 sein, erhalten: {n_windows}")

    total_days = (pd.Timestamp(train_val_end) - pd.Timestamp(train_val_start)).days
    if total_days < 10:
        raise ValueError(
            f"Zeitraum [{train_val_start}, {train_val_end}) ist mit {total_days} Tagen "
            f"zu kurz fuer ein automatisches Walk-Forward-Layout (mindestens 10 Tage noetig)."
        )

    block_days = max(1, total_days // n_windows)
    val_days = max(3, round(block_days * val_fraction))
    embargo_days = max(1, round(block_days * embargo_fraction))
    train_days = max(5, block_days - val_days - embargo_days)
    step_days = max(1, block_days)

    return WalkForwardLayout(
        train_days=train_days, val_days=val_days, step_days=step_days,
        embargo_days=embargo_days, n_windows=n_windows,
    )


# ============================================================================
# Optuna-Study-Storage
# ============================================================================

def create_or_load_study(
    study_name: str, *, storage_path: Path | str, seed: int = 42,
) -> "optuna.Study":
    """Erstellt eine neue oder laedt eine bestehende Optuna-Study aus einem
    persistenten SQLite-Storage. Gemeinsame Basis fuer alle drei Quick-Tuner
    -- storage_path ist Pflicht (jeder Tuner hat seine eigene .db-Datei,
    siehe dortiges DEFAULT_STORAGE_PATH), damit die drei Strategien nicht in
    derselben Study landen (unterschiedliche Suchraeume/Objectives waeren
    sonst nicht vergleichbar).
    """
    import optuna

    path = Path(storage_path)
    storage_url = f"sqlite:///{path.resolve()}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(
        study_name=study_name, storage=storage_url, sampler=sampler,
        direction="maximize", load_if_exists=True,
    )


# ============================================================================
# Beste Parameter: Datei-Persistenz
# ============================================================================

def best_params_path(strategy: str) -> Path:
    """Pfad zur zentralen Best-Params-Datei einer Strategie, siehe
    Modul-Docstring. strategy z.B. 'ewma', 'bollinger', 'garch'."""
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    return TUNING_DIR / f"best_params_{strategy}.json"


def save_best_params(
    strategy: str, basket: str, params: dict, *, objective: float, n_trials: int,
) -> Path:
    """Schreibt/aktualisiert den Eintrag fuer (strategy, basket) in der
    zentralen Best-Params-JSON-Datei (siehe best_params_path) -- bestehende
    Eintraege anderer Baskets bleiben erhalten.

    Returns:
        Pfad der geschriebenen Datei.
    """
    path = best_params_path(strategy)
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

    data[basket] = {
        **params,
        "objective": objective,
        "n_trials": n_trials,
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_best_params(strategy: str, basket: str | None = None) -> dict:
    """Laedt die zentrale Best-Params-Datei einer Strategie.

    Args:
        strategy: 'ewma', 'bollinger' oder 'garch'.
        basket: falls gesetzt, wird nur der Eintrag dieses Baskets
            zurueckgegeben (leeres dict falls (noch) nicht getunt).

    Returns:
        dict {basket: {..params..}} (basket=None) oder direkt {..params..}
        (basket gesetzt).
    """
    path = best_params_path(strategy)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(basket, {}) if basket is not None else data


# ============================================================================
# Beste Parameter: Sichtbar im Engine-Modul hinterlegen
# ============================================================================

def write_best_params_to_engine_module(strategy: str, module_path: Path | str) -> None:
    """Schreibt/ersetzt einen BEST_PARAMS-Block direkt im Ziel-Engine-Modul
    (z.B. pairtrading_engine.py), rein informativ -- siehe Modul-Docstring.

    Idempotent: ein bereits vorhandener Block (zwischen den _BEST_PARAMS_*-
    Markern) wird ersetzt statt dupliziert. Wird kein vorhandener Block
    gefunden, haengt die Funktion einen neuen ans Dateiende an.

    Args:
        strategy: siehe best_params_path -- bestimmt, welche JSON-Datei
            gelesen wird.
        module_path: Pfad zum *_pairtrading_engine.py, in das der Block
            geschrieben wird.
    """
    all_params = load_best_params(strategy)
    module_path = Path(module_path)
    source = module_path.read_text(encoding="utf-8")

    lines = [
        _BEST_PARAMS_BEGIN,
        "# Von quick_tune_*.py gefundene, je Basket beste Parameter (Walk-Forward-",
        "# Validierungs-Objective) -- NUR zur Referenz, wird von den run_*_backtest-",
        "# Funktionen NICHT automatisch gelesen. Quelle: " + str(best_params_path(strategy)),
        "BEST_PARAMS = {",
    ]
    for basket, params in sorted(all_params.items()):
        clean = {k: v for k, v in params.items() if k not in ("tuned_at",)}
        lines.append(f"    {basket!r}: {clean!r},")
    lines.append("}")
    lines.append(_BEST_PARAMS_END)
    block = "\n".join(lines)

    pattern = re.compile(
        re.escape(_BEST_PARAMS_BEGIN) + r".*?" + re.escape(_BEST_PARAMS_END),
        re.DOTALL,
    )
    if pattern.search(source):
        # sub() interpretiert Backslashes im Replacement-String als Escapes
        # (z.B. \U in Windows-Pfaden) -- eine Funktion als Replacement
        # umgeht das, weil ihr Rueckgabewert woertlich eingesetzt wird.
        new_source = pattern.sub(lambda _m: block, source)
    else:
        new_source = source.rstrip("\n") + "\n\n\n" + block + "\n"

    module_path.write_text(new_source, encoding="utf-8")


# ============================================================================
# Zeitraum-Aufteilung: Walk-Forward-Suchraum vs. finaler Testzeitraum
# ============================================================================

def split_walkforward_test(
    start, end, *, test_fraction: float = 0.2,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Teilt [start, end) in (train_val_start, train_val_end, test_start,
    test_end): die letzten test_fraction des Gesamtzeitraums bilden den
    finalen, waehrend des Tunings nie beruehrten Testzeitraum.

    Returns:
        (train_val_start, train_val_end, test_start, test_end)
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction muss in (0, 1) liegen, erhalten: {test_fraction}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    total_days = (end_ts - start_ts).days
    test_days = max(1, int(round(total_days * test_fraction)))
    test_start = end_ts - pd.Timedelta(days=test_days)

    return start_ts, test_start, test_start, end_ts
