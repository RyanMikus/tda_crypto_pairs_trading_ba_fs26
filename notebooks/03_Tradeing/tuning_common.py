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

import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
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


def suggest_param(
    trial: "optuna.Trial",
    name: str,
    *,
    kind: str,
    default_range,
    fixed_params: dict,
    search_ranges: dict,
):
    """Einheitliche Optuna-suggest_*-Umleitung fuer alle drei Quick-Tuner:
    entscheidet pro Parameter zwischen fix, gesuchter Custom-Range oder
    gesuchter Default-Range -- ersetzt die vorher in jedem Tuner separat
    wiederholten `fixed_params.get(name, trial.suggest_*(name, lo, hi))`-
    Zeilen und macht Suchbereiche von aussen (search_ranges) ueberschreibbar,
    ohne die Tuner-Objective-Funktionen selbst anzufassen.

    Prioritaet: fixed_params (Parameter wird NICHT gesucht) > search_ranges
    (Parameter wird gesucht, aber mit eigenem Bereich statt Default) >
    default_range (Parameter wird mit dem im Tuner hart codierten
    Standardbereich gesucht).

    Args:
        trial: aktueller Optuna-Trial.
        name: Parametername (Schluessel in fixed_params/search_ranges).
        kind: "int", "float" oder "categorical".
        default_range: (lo, hi) fuer int/float, Liste der Optionen fuer
            categorical -- wird verwendet, falls weder fixed_params noch
            search_ranges einen Eintrag fuer name haben.
        fixed_params: siehe *_pairtrading_engine-Tuner-Docstrings.
        search_ranges: dict {param_name: (lo, hi)} bzw. {param_name: [options]}
            fuer categorical -- ueberschreibt NUR den Suchbereich, der
            Parameter bleibt gesucht (im Gegensatz zu fixed_params).

    Returns:
        Der vorgeschlagene (oder fixierte) Wert.
    """
    if name in fixed_params:
        return fixed_params[name]

    lo_hi_or_options = search_ranges.get(name, default_range)
    if kind == "int":
        lo, hi = lo_hi_or_options
        return trial.suggest_int(name, int(lo), int(hi))
    if kind == "float":
        lo, hi = lo_hi_or_options
        return trial.suggest_float(name, float(lo), float(hi))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(lo_hi_or_options))
    raise ValueError(f"Unbekannte kind={kind!r}, erwartet 'int'/'float'/'categorical'.")


def search_space_signature(fixed_params: dict, search_ranges: dict) -> str:
    """Stabiler kurzer Hash ueber (fixed_params, search_ranges) -- 8 Hex-
    Zeichen aus SHA-256 ueber die kanonische (sortierte) JSON-Repraesentation.

    Grund: Optuna speichert die Parameter-VERTEILUNG (Typ + Bereich/Choices)
    beim ERSTEN Trial fest in der Study. Wird derselbe Parametername in
    einem spaeteren Trial mit einem anderen Bereich/anderen Choices gesucht
    (z.B. weil im Notebook zwischen zwei Laeufen der Suchbereich verstellt
    oder ein Parameter von "tunen" auf "fix" umgeschaltet wurde), lehnt
    Optuna das mit "CategoricalDistribution does not support dynamic value
    space" (analog fuer Int-/FloatDistribution) hart ab.

    Wird als Suffix an den Study-Namen gehaengt (siehe die drei
    tune_*_strategy-Funktionen): identischer Suchraum -> identische
    Signatur -> dieselbe Study wird weiterverwendet (Trials sammeln sich
    an). Veraenderter Suchraum -> andere Signatur -> automatisch eine neue,
    saubere Study -- kein manuelles Loeschen von .db-Dateien noetig, keine
    verlorenen Trials aus frueheren Laeufen mit anderem Suchraum (die
    bleiben einfach unter ihrer eigenen Signatur erhalten).
    """
    payload = json.dumps(
        {"fixed_params": fixed_params, "search_ranges": search_ranges},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


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
# Optuna-Dashboard: automatischer Hintergrundstart + Browser-Oeffnen
# ============================================================================

_DASHBOARD_PROCESSES: dict[int, subprocess.Popen] = {}
DASHBOARD_PORTS = {"ewma": 8080, "bollinger": 8081, "garch": 8082}


def _port_is_open(host: str, port: int, *, timeout: float = 0.3) -> bool:
    """True, falls unter host:port bereits etwas auf Verbindungen lauscht
    (egal ob unser eigenes Dashboard oder ein fremder Prozess)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def ensure_dashboard_running(
    storage_path: Path | str, *, port: int, host: str = "localhost",
    open_browser: bool = True, startup_timeout_seconds: float = 8.0,
) -> str:
    """Startet optuna-dashboard fuer storage_path als Hintergrundprozess,
    falls unter host:port noch nichts laeuft, und oeffnet den Browser-Tab.

    Gedacht, um vor jedem Quick-Tune-Lauf aufgerufen zu werden (siehe
    quick_tuning.ipynb-Button-Callbacks) -- idempotent: ein bereits
    laufendes Dashboard (egal ob von uns oder manuell gestartet) wird nicht
    doppelt gestartet, der Browser-Tab wird trotzdem geoeffnet, damit der
    Nutzer bei jedem Klick wieder direkt zum Dashboard gelangt.

    Der Hintergrundprozess ist ein eigener `optuna-dashboard`-Subprozess
    (nicht optuna_dashboard.run_server() direkt, das blockiert den
    aufrufenden Thread) -- ueberlebt dadurch auch einen Kernel-Neustart,
    muss also bei Bedarf manuell beendet werden (Task-Manager/`kill`).

    Args:
        storage_path: Pfad zur SQLite-Study-Datei (z.B. DEFAULT_STORAGE_PATH
            eines der drei quick_tune_*-Module).
        port: Port fuer diese Study -- siehe DASHBOARD_PORTS (ein eigener
            Port je Strategie, damit alle drei gleichzeitig laufen koennen).
        host: Bind-Adresse, Default "localhost".
        open_browser: True (Default) = Browser-Tab automatisch oeffnen.
        startup_timeout_seconds: maximale Wartezeit, bis der neu gestartete
            Server tatsaechlich Verbindungen annimmt (per Port-Polling alle
            0.3s geprueft, ~2s Kaltstart sind ueblich) -- nur relevant, wenn
            tatsaechlich neu gestartet wurde.

    Returns:
        Die URL des Dashboards (http://host:port/).
    """
    url = f"http://{host}:{port}/"

    if not _port_is_open(host, port):
        if shutil.which("optuna-dashboard") is None:
            print(
                f"WARNUNG: optuna-dashboard-Kommando nicht gefunden (pip install optuna-dashboard) "
                f"-- Dashboard wird nicht gestartet, Tuning laeuft trotzdem normal weiter."
            )
            return url

        path = Path(storage_path)
        storage_url = f"sqlite:///{path.resolve()}"
        proc = subprocess.Popen(
            ["optuna-dashboard", storage_url, "--host", host, "--port", str(port), "--server", "wsgiref"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _DASHBOARD_PROCESSES[port] = proc
        print(f"optuna-dashboard wird gestartet: {url} (Storage: {storage_url}, PID {proc.pid}) ...")

        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            if _port_is_open(host, port):
                print(f"optuna-dashboard bereit: {url}")
                break
            time.sleep(0.3)
        else:
            print(f"WARNUNG: optuna-dashboard antwortet nach {startup_timeout_seconds:.0f}s noch nicht "
                  f"unter {url} -- Browser wird trotzdem geoeffnet, ggf. manuell neu laden.")
    else:
        print(f"optuna-dashboard laeuft bereits unter {url}")

    if open_browser:
        webbrowser.open(url)

    return url


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
    start, end, *, test_fraction: float = 0.2, embargo_days: int = 1,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Teilt [start, end) in (train_val_start, train_val_end, test_start,
    test_end): die letzten test_fraction des Gesamtzeitraums bilden den
    finalen, waehrend des Tunings nie beruehrten Testzeitraum.

    embargo_days trennt train_val_end von test_start um einen kleinen
    Puffer (Default 1 Tag): auto_walkforward_layout/generate_walkforward_
    windows koennen ein Walk-Forward-Fenster erzeugen, dessen val_end EXAKT
    auf train_val_end faellt -- ohne Puffer wuerde assert_no_test_period_leakage
    das dann als Leakage werten (val_end >= test_start, Grenze inklusive),
    obwohl train_val_end nur die Suchraum-Obergrenze ist. Der Puffer
    verkleinert den Walk-Forward-Suchraum minimal, verhindert aber diesen
    Grenzfall zuverlaessig.

    Returns:
        (train_val_start, train_val_end, test_start, test_end)
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction muss in (0, 1) liegen, erhalten: {test_fraction}")
    if embargo_days < 0:
        raise ValueError(f"embargo_days darf nicht negativ sein, erhalten: {embargo_days}")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    total_days = (end_ts - start_ts).days
    test_days = max(1, int(round(total_days * test_fraction)))
    test_start = end_ts - pd.Timedelta(days=test_days)
    train_val_end = test_start - pd.Timedelta(days=embargo_days)

    return start_ts, train_val_end, test_start, end_ts
