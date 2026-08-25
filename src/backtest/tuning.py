"""Walk-Forward-Hyperparameteroptimierung PRO BASKET für die Backtest-Engine
(src/backtest/engine.py), per Optuna/TPE (walk-forward Fenster, pro-Basket-
Study, persistenter SQLite-Storage, Robustheits-Zielgrösse
mean(fold_returns) - std(fold_returns)) auf BacktestConfig/BasketParams/
BacktestEngine.

Theoriegeleitete Filterparameter vs. tunbare Signalparameter ("Data-Snooping-
Bias" bei Ernie Chan): NUR entry_z und exit_z sind standardmässig Teil der
Optuna-Suche (siehe DEFAULT_SEARCH_SPACE). stop_z sowie alle Filter-/Fensterparameter
(bollinger_window_candles, formation_window_candles, min/max_half_life_candles,
require_adf, adf_alpha, max_candidate_pairs, pair_prefilter_top_n) bleiben FIX
aus dem übergebenen base_params -- das sind statistische Konventionen bzw.
Risiko-Caps, keine gegen die Sharpe/Return optimierbaren Stellschrauben.

Isolation pro Basket: jeder Trial läuft mit baskets=(basket,) -- 100% Kapital
nur für diesen einen Basket in der Fold-Simulation, unabhängig von den
anderen drei. Vermeidet das Henne-Ei-Problem einer gemeinsamen Optimierung
aller vier Baskets gleichzeitig (wessen Parameter sind für die jeweils
anderen drei fix, während Basket X gesucht wird?).

Zeitraum: Folds liegen ausschliesslich innerhalb calibration_range (nie im
finalen backtest_range, siehe BacktestConfig/split_calibration_backtest) --
das ist strukturell erzwungen, da generate_walkforward_folds ausschliesslich
Teile von calibration_range zurückgibt, nie darüber hinaus.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.backtest.config import BacktestConfig, BasketParams, DateRange
from src.backtest.config import _BASKET_PARAM_FIELDS
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import build_strategy
from src.config import CANDLE_MINUTES, TUNING_DIR, candles_per_day


FloatRange = tuple[float, float]
IntRange = tuple[int, int]
CategoricalChoices = list  # list[str | float | int]
SearchSpec = Union[FloatRange, IntRange, CategoricalChoices]
"""Suchraum-Spezifikation fuer EINEN BasketParams-Feldnamen -- die
KONTAINER-ART entscheidet, NICHT der Inhalt:
    - tuple (low, high), BEIDE Werte int   -> trial.suggest_int
    - tuple (low, high), mind. ein float   -> trial.suggest_float
    - list [opt1, opt2, ...] (JEDE Laenge) -> trial.suggest_categorical
Bewusst tuple=Bereich vs. list=kategorial statt am Inhalt zu raten: eine
rein inhaltsbasierte Unterscheidung waere fuer numerische 2-Options-
Kategorien (z.B. [0.90, 0.95]) falsch (sieht aus wie ein Bereich, ist aber
eine Auswahl) -- siehe _is_range_spec."""

DEFAULT_SEARCH_SPACE: dict[str, SearchSpec] = {
    "entry_z": (1.0, 4.0),
    "exit_z": (0.0, 4.0),  # oberes Ende wird pro Trial auf entry_z-0.1 gekappt, siehe _suggest_param
}
"""Default-Suchraum fuer PairwiseZScoreStrategy: entry_z/exit_z."""


def _is_range_spec(spec: SearchSpec) -> bool:
    """tuple -> Bereich, list -> kategorial (siehe SearchSpec-Docstring).
    len(spec)==2-Pruefung bleibt als Absicherung gegen Tippfehler (ein
    3-Tupel waere weder als Bereich noch als int/float sinnvoll)."""
    if not isinstance(spec, tuple):
        return False
    if len(spec) != 2:
        raise ValueError(f"Bereichs-Spec (tuple) braucht genau 2 Werte (low, high), erhalten: {spec!r}")
    return True


def _suggest_param(trial: "optuna.Trial", name: str, spec: SearchSpec, *, resolved: dict):  # noqa: F821
    """Ein Optuna-suggest_*-Aufruf für EINEN Parameter, dispatcht nach Spec-Form.

    entry_z/exit_z-Kopplung (exit_z < entry_z, siehe ursprüngliche
    fest-verdrahtete Logik) bleibt NUR erhalten, wenn beide tatsächlich Teil
    des Suchraums sind -- resolved enthält die in DIESEM Trial bereits
    gezogenen Werte (Suchraum-Dict-Reihenfolge = Zugriffsreihenfolge, Python
    3.7+ Dicts sind geordnet).
    """
    if name == "exit_z" and "entry_z" in resolved and _is_range_spec(spec):
        low, high = spec
        high = min(high, max(resolved["entry_z"] - 0.1, low))
        return trial.suggest_float(name, low, high)

    if _is_range_spec(spec):
        low, high = spec
        if isinstance(low, int) and isinstance(high, int):
            return trial.suggest_int(name, low, high)
        return trial.suggest_float(name, float(low), float(high))

    return trial.suggest_categorical(name, list(spec))


def _required_history_candles_worst_case(
    *, strategy, base_params: BasketParams, search_space: dict[str, SearchSpec] | None,
) -> int:
    """required_history_candles(base_params), aber mit jedem im search_space
    genannten Feld auf sein Maximum gesetzt (Bereichs-Obergrenze bei tuple,
    max() der Optionen bei list) -- deckt jeden Wert ab, den _suggest_param
    fuer dieses Feld tatsaechlich sampeln kann. Felder ausserhalb des
    search_space bleiben unveraendert aus base_params (dort FIX per
    Definition)."""
    if not search_space:
        return strategy.required_history_candles(base_params)

    overrides: dict[str, float | int] = {}
    for name, spec in search_space.items():
        if name not in _BASKET_PARAM_FIELDS:
            continue
        if _is_range_spec(spec):
            _low, high = spec
        else:
            high = max(spec)
        overrides[name] = high

    worst_case_params = replace(base_params, **overrides)
    return strategy.required_history_candles(worst_case_params)


def generate_walkforward_folds(
    calibration_range: DateRange, *, fold_days: int, warmup_days: int,
) -> list[DateRange]:
    """Zerlegt calibration_range in nicht-überlappende fold_days-Blöcke.

    Der erste Fold beginnt bei calibration_range.start + warmup_days (Tage,
    die als reine Preis-Historie für formation_window_candles/
    bollinger_window_candles gebraucht werden, siehe Warm-up-Guard G7 in
    engine.py -- calibration_range.start ist der Anfang des GESAMTEN
    geladenen Panels, davor existiert keine Preis-Historie). Ein
    unvollständiger Rest am Ende von calibration_range wird verworfen
    (analog zu split_calibration_backtest in config.py: lieber ein Fold
    weniger als ein krummes, schwer vergleichbares letztes Fold).

    Fold-Endpunkte sind wie überall in diesem Projekt TAGESGRANULAR und
    INKLUSIV (siehe split_calibration_backtest) -- ein Fold [t, t+fold_days-1]
    deckt beim Engine-Lauf effektiv alle 5-Min-Kerzen von t 00:00 bis
    t+fold_days-1 00:00 ab (letzter Tag nur bis Mitternacht, nicht bis
    23:55 -- derselbe, bereits etablierte Kompromiss wie bei backtest_range).

    Args:
        calibration_range: NIE der finale backtest_range -- Folds dürfen
            diesen Zeitraum strukturell nicht verlassen (siehe Modul-Docstring).
        fold_days: Länge JEDES Folds in Tagen -- entspricht gleichzeitig der
            Schrittweite (Stride), da die Folds nicht überlappen.
        warmup_days: Tage reiner Preis-Historie vor dem ersten Fold.

    Returns:
        Liste von DateRange(start, end), aufsteigend sortiert. Leer, falls
        selbst ein einziger vollständiger Fold nicht in [calibration_range.start
        + warmup_days, calibration_range.end] passt.
    """
    if fold_days <= 0:
        raise ValueError("fold_days muss > 0 sein.")
    if warmup_days < 0:
        raise ValueError("warmup_days darf nicht negativ sein.")

    cursor = calibration_range.start + pd.Timedelta(days=warmup_days)
    folds: list[DateRange] = []
    while True:
        fold_end = cursor + pd.Timedelta(days=fold_days - 1)
        if fold_end > calibration_range.end:
            break
        folds.append(DateRange(cursor, fold_end))
        cursor = cursor + pd.Timedelta(days=fold_days)
    return folds


def fold_days_for_n_windows(
    calibration_range: DateRange, *, n_windows: int, base_params: BasketParams,
    strategy_name: str = "pairwise_zscore", candle_minutes: int = CANDLE_MINUTES,
    search_space: dict[str, SearchSpec] | None = None,
) -> int:
    """Rechnet die gewuenschte Fenster-ANZAHL (n_windows) in die von
    generate_walkforward_folds/tune_basket_params erwartete Fenster-GROESSE
    (fold_days) um: fold_days = floor((range_days - warmup_days) / n_windows).

    warmup_days wird -- wie in tune_basket_params selbst -- aus
    strategy.required_history_candles(base_params) abgeleitet, damit die
    Rueckrechnung exakt zu den Folds passt, die ein tune_basket_params-Aufruf
    mit dem zurueckgegebenen fold_days tatsaechlich erzeugen wuerde.

    Args:
        calibration_range: derselbe Zeitraum, der auch an tune_basket_params
            uebergeben wird.
        n_windows: gewuenschte Anzahl Walk-Forward-Fenster (>= 1).
        base_params, strategy_name, candle_minutes: siehe tune_basket_params
            -- bestimmen warmup_days.
        search_space: falls uebergeben (derselbe search_space wie beim
            zugehoerigen tune_basket_params-Aufruf), wird warmup_days aus dem
            WORST CASE ueber alle darin moeglichen Kandidaten abgeleitet
            (siehe _required_history_candles_worst_case) -- noetig, falls
            formation_window_candles/vol_regime_percentile_window_candles/
            vol_regime_lookback_candles selbst gesucht werden, sonst kann ein
            Trial mehr Vorlauf brauchen, als hier eingeplant wurde.

    Returns:
        fold_days (>= 1), das bei generate_walkforward_folds moeglichst nah
        an n_windows tatsaechliche Fenster ergibt (kann wegen Rundung und
        Restlaenge leicht abweichen -- Ergebnis mit generate_walkforward_folds
        gegenpruefen, siehe notebooks/03_backtest/03_tune_dynamic_vol.ipynb).

    Raises:
        ValueError: falls n_windows < 1, oder falls selbst n_windows=1 kein
            positives fold_days ergibt (Zeitraum nach Abzug von warmup_days
            zu kurz).
    """
    if n_windows < 1:
        raise ValueError(f"n_windows muss >= 1 sein, erhalten: {n_windows}")

    strategy = build_strategy(strategy_name)
    required_candles = _required_history_candles_worst_case(
        strategy=strategy, base_params=base_params, search_space=search_space,
    )
    warmup_days = math.ceil(required_candles / candles_per_day(candle_minutes))

    range_days = (calibration_range.end - calibration_range.start).days + 1
    available_days = range_days - warmup_days
    fold_days = available_days // n_windows

    if fold_days < 1:
        raise ValueError(
            f"calibration_range ({range_days} Tage) minus warmup_days ({warmup_days}, aus "
            f"required_history_candles={required_candles}) reicht nicht für n_windows={n_windows} "
            f"Fenster -- calibration_range vergrössern oder n_windows verkleinern."
        )
    return fold_days


def _slice_panel(panel: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Schneidet jedes Feld des Panels auf [start, end] zu, BEVOR ein Fold-
    Engine gebaut wird -- BacktestEngine.__init__ konvertiert das gesamte
    übergebene Panel per .to_numpy(), bei hunderten Fold-Läufen wäre das
    volle DATA_START..DATA_END-Panel jedes Mal neu zu konvertieren reine
    Verschwendung."""
    return {field: df.loc[start:end] for field, df in panel.items()}


def _run_fold(
    *,
    panel: dict[str, pd.DataFrame],
    timeline,
    basket: str,
    params: BasketParams,
    fold: DateRange,
    strategy_name: str,
    initial_capital: float,
    warmup_days: int,
    candle_minutes: int = CANDLE_MINUTES,
) -> tuple[float, int]:
    """Läuft EINEN Fold für EINEN Basket mit einem Kandidaten-Parametersatz,
    gibt (total_return, n_trades) zurück (0.0/0 falls keine Trades/leere
    Equity-Kurve). n_trades wird gebraucht, um Kandidaten mit zu wenigen
    Trades zu erkennen (siehe min_trades_per_fold in tune_basket_params).
    Bewusst OHNE bt_metrics.compute_all_metrics -- für die reine
    Zielgrösse reicht der direkte Equity-Endstand, der volle Metrik-Katalog
    (Sharpe/Sortino/Beta/...) wäre unnötiger Overhead bei hunderten Läufen."""
    sliced = _slice_panel(panel, fold.start - pd.Timedelta(days=warmup_days), fold.end)
    config = BacktestConfig(
        label="tuning-fold",
        baskets=(basket,),
        backtest_range=fold,
        calibration_range=None,
        strategy_name=strategy_name,
        default_params=params,
        initial_capital=initial_capital,
        candle_minutes=candle_minutes,
    )
    engine = BacktestEngine(config=config, panel=sliced, timeline=timeline)
    result = engine.run()
    n_trades = len(result.trades)
    equity = result.equity_curve["equity"] if not result.equity_curve.empty else None
    if equity is None or len(equity) == 0:
        return 0.0, n_trades
    return float(equity.iloc[-1] / initial_capital - 1.0), n_trades


def _objective_metric(fold_returns: list[float]) -> float:
    """mean(fold_returns) - std(fold_returns) -- Robustheit statt Peak-
    Performance auf einem einzelnen Glücks-Fold."""
    if not fold_returns:
        return float("-inf")
    arr = np.array(fold_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("-inf")
    mean_r = float(arr.mean())
    std_r = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
    return mean_r - std_r


@dataclass
class TuningResult:
    basket: str
    study: "optuna.Study"  # noqa: F821 -- nur unter TYPE_CHECKING/zur Laufzeit via optuna importiert
    trials_df: pd.DataFrame
    best_params: dict
    folds: list[DateRange]


def tune_basket_params(
    *,
    basket: str,
    panel: dict[str, pd.DataFrame],
    timeline,
    base_params: BasketParams,
    calibration_range: DateRange,
    strategy_name: str = "pairwise_zscore",
    initial_capital: float = 100_000.0,
    fold_days: int = 30,
    n_trials: int = 10,
    seed: int = 42,
    min_trades_per_fold: float = 1.0,
    study_name: str | None = None,
    storage_path: Path | str | None = None,
    use_persistent_storage: bool = True,
    candle_minutes: int = CANDLE_MINUTES,
    search_space: dict[str, SearchSpec] | None = None,
    progress: bool = False,
) -> TuningResult:
    """Optuna/TPE-Suche über search_space (Default: entry_z/exit_z, siehe
    DEFAULT_SEARCH_SPACE) für EINEN
    Basket, Zielgrösse = _objective_metric über alle Walk-Forward-Folds
    innerhalb calibration_range (siehe Modul-Docstring für die
    Isolations-Begründung).


    min_trades_per_fold (Analog zu SCREEN_MIN_SIGNALS im alten Tuner):
    mean(fold_returns) - std(fold_returns) allein belohnt PERVERS
    Kandidaten, die kaum handeln -- viele Folds mit ~0 Trades ziehen die
    Rendite jedes Folds Richtung 0 UND senken damit die Streuung zwischen
    den Folds künstlich, was den mean-std-Score anhebt, ohne dass die
    Strategie tatsächlich besser wäre. Kandidaten mit
    sum(n_trades über alle Folds) < min_trades_per_fold * len(folds) werden
    deshalb per optuna.TrialPruned() aus der Study ausgeschlossen (zählen
    nicht zu study.best_params/best_value).

    Persistenter SQLite-Storage (Default, wie im alten Tuner): eine Study mit
    demselben study_name (Default: basket) kann über mehrere Aufrufe hinweg
    mit weiteren Trials fortgesetzt werden, statt bei jedem Aufruf von vorn
    zu beginnen -- wichtig, da n_trials hier die Anzahl NEUER Trials in
    diesem Aufruf ist, nicht die Gesamtzahl der Study.


    Raises:
        ValueError: falls calibration_range nach Abzug der Warm-up-Tage
            keinen einzigen vollständigen Fold enthält.
        RuntimeError: falls KEIN einziger Trial (weder in diesem Aufruf noch
            in einer evtl. geladenen früheren Study) den min_trades_per_fold-
            Filter besteht -- min_trades_per_fold senken oder n_trials erhöhen.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)


    resolved_search_space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE
    unknown = set(resolved_search_space) - _BASKET_PARAM_FIELDS
    if unknown:
        raise ValueError(f"search_space enthält keine BasketParams-Felder: {sorted(unknown)}")


    strategy = build_strategy(strategy_name)

    required_candles = _required_history_candles_worst_case(
        strategy=strategy, base_params=base_params, search_space=resolved_search_space,
    )

    warmup_days = math.ceil(required_candles / candles_per_day(candle_minutes))


    folds = generate_walkforward_folds(calibration_range, fold_days=fold_days, warmup_days=warmup_days)
    if not folds:
        raise ValueError(
            f"Keine Walk-Forward-Folds für calibration_range={calibration_range} mit "
            f"fold_days={fold_days}, warmup_days={warmup_days} (aus required_history_candles="
            f"{required_candles}) -- Kalibrierungszeitraum vergrössern oder fold_days verkleinern."
        )

    trial_log: list[dict] = []
    _n_trials_seen = 0
    _t_start = time.perf_counter()

    def objective(trial: "optuna.Trial") -> float:
        nonlocal _n_trials_seen
        suggested: dict = {}
        for name, spec in resolved_search_space.items():
            suggested[name] = _suggest_param(trial, name, spec, resolved=suggested)


        _n_trials_seen += 1
        if progress:
            elapsed = time.perf_counter() - _t_start
            eta_str = ""
            if _n_trials_seen > 1:
                eta = elapsed / (_n_trials_seen - 1) * (n_trials - (_n_trials_seen - 1))
                eta_str = f" | ~{eta:.0f}s verbleibend"
            print(f"[Trial {_n_trials_seen}/{n_trials}] (Study-Trial-Nr. {trial.number}) {suggested}{eta_str}")


        candidate = replace(base_params, **suggested)
        fold_results = []
        for fi, fold in enumerate(folds, start=1):
            fold_return, fold_trades = _run_fold(
                panel=panel, timeline=timeline, basket=basket, params=candidate, fold=fold,
                strategy_name=strategy_name, initial_capital=initial_capital, warmup_days=warmup_days,
                candle_minutes=candle_minutes,
            )
            fold_results.append((fold_return, fold_trades))

            if progress:
                print(f"    Fold {fi}/{len(folds)} [{fold.start.date()}, {fold.end.date()}]: "
                      f"{fold_trades} Trades, return={fold_return:.4f}")

        fold_returns = [r for r, _ in fold_results]
        total_trades = sum(n for _, n in fold_results)
        min_required_trades = min_trades_per_fold * len(folds)


        if total_trades < min_required_trades:
            if progress:
                print(f"  -> PRUNED (nur {total_trades} Trades, {min_required_trades:.0f} noetig)\n")
            trial_log.append({
                "trial_number": trial.number, **suggested,
                "status": "pruned_min_trades", "total_trades": total_trades,
                "objective": np.nan, "mean_fold_return": np.nan, "std_fold_return": np.nan,
                "n_folds": len(folds),
            })
            raise optuna.TrialPruned()

        objective_value = _objective_metric(fold_returns)
        arr = np.array(fold_returns, dtype=float)
        if progress:
            print(f"  -> ABGESCHLOSSEN: objective={objective_value:.4f}\n")
        trial_log.append({
            "trial_number": trial.number, **suggested,
            "status": "completed", "total_trades": total_trades,
            "objective": objective_value,
            "mean_fold_return": float(arr.mean()), "std_fold_return": float(arr.std(ddof=0)),
            "n_folds": len(folds),
        })

        return objective_value

    if use_persistent_storage:
        resolved_storage_path = Path(storage_path) if storage_path is not None else TUNING_DIR / "optuna_walkforward.db"
        resolved_storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{resolved_storage_path.resolve()}"
        resolved_study_name = study_name or basket


        n_existing_trials = 0
        if resolved_storage_path.exists() and resolved_study_name in optuna.get_all_study_names(storage=storage_url):
            n_existing_trials = len(optuna.load_study(study_name=resolved_study_name, storage=storage_url).trials)
        sampler = optuna.samplers.TPESampler(seed=seed + n_existing_trials)


        study = optuna.create_study(
            study_name=resolved_study_name, storage=storage_url,
            sampler=sampler, direction="maximize", load_if_exists=True,
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_completed == 0:
        raise RuntimeError(
            f"Kein einziger Trial für Basket {basket!r} (weder in diesem Aufruf noch in einer "
            f"evtl. geladenen frueheren Study) hat den min_trades_per_fold-Filter "
            f"({min_trades_per_fold} * {len(folds)} Folds = {min_trades_per_fold * len(folds):.0f} Trades) "
            f"bestanden -- min_trades_per_fold senken oder n_trials erhoehen."
        )

    return TuningResult(
        basket=basket, study=study, trials_df=pd.DataFrame(trial_log),
        best_params=dict(study.best_params), folds=folds,
    )


def _tuning_slug(label: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return s or "tuning"


def save_tuning_results(
    results: dict[str, TuningResult], *, label: str, tuning_dir: Path = TUNING_DIR,
    candle_minutes: int = CANDLE_MINUTES,
    fee_preset: str | None = None,
    fixed_overrides: dict[str, dict] | dict | None = None,
) -> Path:
    """Speichert best_params.json (alle Baskets) + trials_<basket>.parquet
    (je Basket) unter tuning_dir/<run_id>/. run_id = Zeitstempel + Slug(label),
    analog zu runs.save_run.

    fixed_overrides: die BasketParams-Felder, die WAEHREND DIESES LAUFS fix
    waren (also NICHT gesucht wurden -- z.B. "tunen"=AUS im Widget-Panel in
    03_tune_dynamic_vol.ipynb). result.best_params (aus Optuna study.best_params)
    enthaelt NUR die tatsaechlich gesuchten Felder -- ohne fixed_overrides gehen
    die fixen Werte beim Speichern verloren und ein spaeterer Konsument (z.B.
    04_tuning_overview.ipynb) muesste eigene, potenziell abweichende Defaults
    fuer diese Felder raten. Entweder EIN dict (gleiche Fix-Werte fuer alle
    Baskets in `results`, z.B. das Widget-Panel-Sequenzmuster) oder ein
    {basket: dict}-Mapping (unterschiedliche Fix-Werte je Basket). None (Default)
    speichert einen leeren Fix-Werte-Block -- unveraendertes Verhalten fuer
    Aufrufer, die das Feature nicht nutzen.

    Returns:
        Pfad zum Lauf-Ordner.
    """
    import json

    created_at = pd.Timestamp.now(tz="UTC")
    run_id = f"{created_at:%Y%m%d-%H%M%S}_{_tuning_slug(label)}"
    run_dir = tuning_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def _fixed_overrides_for(basket: str) -> dict:
        if fixed_overrides is None:
            return {}
        if basket in results and set(results.keys()) & set(fixed_overrides.keys()):
            # {basket: dict}-Form erkannt (mind. ein Basket-Key trifft) -- pro Basket holen.
            return dict(fixed_overrides.get(basket, {}))
        return dict(fixed_overrides)

    best_params_payload = {
        basket: {
            "best_params": result.best_params,
            "fixed_overrides": _fixed_overrides_for(basket),
            "best_objective": float(result.study.best_value),
            "n_trials": len(result.trials_df),
            "n_folds": len(result.folds),
            "candle_minutes": candle_minutes,
            "fee_preset": fee_preset,
        }
        for basket, result in results.items()
    }
    (run_dir / "best_params.json").write_text(json.dumps(best_params_payload, indent=2))

    for basket, result in results.items():
        if not result.trials_df.empty:
            slug = _tuning_slug(basket)
            result.trials_df.to_parquet(run_dir / f"trials_{slug}.parquet", index=False)

    return run_dir


def load_all_tuning_results(tuning_dir: Path = TUNING_DIR) -> pd.DataFrame:
    """Liest jede best_params.json unter tuning_dir/<run_id>/ und baut eine
    lange Tabelle (eine Zeile pro Basket-Eintrag pro Run) mit den Spalten
    run_id, basket, candle_minutes, fee_preset, best_objective, n_trials,
    n_folds, best_params (dict), fixed_overrides (dict).

    Läufe aus save_tuning_results VOR der candle_minutes/fee_preset-Ergänzung
    haben candle_minutes=NaN bzw. fee_preset=None (Feld fehlte im
    gespeicherten JSON) -- werden nicht übersprungen, aber beim Vergleich
    sind sie nicht eindeutig zuordenbar. Ebenso haben Läufe VOR der
    fixed_overrides-Ergänzung fixed_overrides={} (Feld fehlte) -- ein
    Konsument wie 04_tuning_overview.ipynb faellt dann fuer diese Felder auf
    seine eigenen Defaults zurueck, statt die (nicht mehr rekonstruierbaren)
    tatsaechlichen Fix-Werte des damaligen Laufs zu verwenden.

    Returns:
        DataFrame, leer (mit den o.g. Spalten) falls tuning_dir keine
        gespeicherten Läufe enthält.
    """
    import json

    columns = ["run_id", "basket", "candle_minutes", "fee_preset", "best_objective", "n_trials", "n_folds", "best_params", "fixed_overrides"]
    if not tuning_dir.exists():
        return pd.DataFrame(columns=columns)

    rows = []
    for run_dir in sorted(p for p in tuning_dir.iterdir() if p.is_dir()):
        best_params_path = run_dir / "best_params.json"
        if not best_params_path.exists():
            continue
        payload = json.loads(best_params_path.read_text())
        for basket, entry in payload.items():
            rows.append({
                "run_id": run_dir.name,
                "basket": basket,
                "candle_minutes": entry.get("candle_minutes", float("nan")),
                "fee_preset": entry.get("fee_preset"),
                "best_objective": entry.get("best_objective"),
                "n_trials": entry.get("n_trials"),
                "n_folds": entry.get("n_folds"),
                "best_params": entry.get("best_params"),
                "fixed_overrides": entry.get("fixed_overrides") or {},
            })
    return pd.DataFrame(rows, columns=columns)


DASHBOARD_PORTS: dict[str, int] = {"pairwise_zscore": 8080}
"""Ein eigener Port je Strategie, damit mehrere Dashboards gleichzeitig
laufen koennen (siehe ensure_dashboard_running)."""

_DASHBOARD_PROCESSES: dict[int, "subprocess.Popen"] = {}  # noqa: F821 -- nur zur Laufzeit importiert


def _port_is_open(host: str, port: int, *, timeout: float = 0.3) -> bool:
    """True, falls unter host:port bereits etwas auf Verbindungen lauscht
    (egal ob unser eigenes Dashboard oder ein fremder Prozess)."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def ensure_dashboard_running(
    storage_path: Path | str, *, port: int, host: str = "localhost",
    open_browser: bool = True, startup_timeout_seconds: float = 8.0,
) -> str:
    """Startet optuna-dashboard fuer storage_path als Hintergrundprozess,
    falls unter host:port noch nichts laeuft, und oeffnet den Browser-Tab.

    Gedacht, um vor jedem Tuning-Lauf aufgerufen zu werden (siehe
    notebooks/03_backtest/03_tune_dynamic_vol.ipynb-Button-Callbacks) --
    idempotent: ein bereits laufendes Dashboard (egal ob von uns oder
    manuell gestartet) wird nicht doppelt gestartet, der Browser-Tab wird
    trotzdem geoeffnet, damit der Nutzer bei jedem Klick wieder direkt zum
    Dashboard gelangt.

    Der Hintergrundprozess ist ein eigener `optuna-dashboard`-Subprozess
    (nicht optuna_dashboard.run_server() direkt, das blockiert den
    aufrufenden Thread) -- ueberlebt dadurch auch einen Kernel-Neustart,
    muss also bei Bedarf manuell beendet werden (Task-Manager/`kill`).

    Args:
        storage_path: Pfad zur SQLite-Study-Datei (z.B. TUNING_DIR /
            "optuna_walkforward.db", der Default-Storage von
            tune_basket_params).
        port: Port fuer diese Study -- siehe DASHBOARD_PORTS (ein eigener
            Port je Strategie, damit alle gleichzeitig laufen koennen).
        host: Bind-Adresse, Default "localhost".
        open_browser: True (Default) = Browser-Tab automatisch oeffnen.
        startup_timeout_seconds: maximale Wartezeit, bis der neu gestartete
            Server tatsaechlich Verbindungen annimmt (per Port-Polling alle
            0.3s geprueft, ~2s Kaltstart sind ueblich) -- nur relevant, wenn
            tatsaechlich neu gestartet wurde.

    Returns:
        Die URL des Dashboards (http://host:port/).
    """
    import shutil
    import subprocess
    import time
    import webbrowser

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

