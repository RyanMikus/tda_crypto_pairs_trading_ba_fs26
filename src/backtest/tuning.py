"""Walk-Forward-Hyperparameteroptimierung PRO BASKET für die aktuelle
Backtest-Engine (src/backtest/engine.py), per Optuna/TPE.

Überträgt dasselbe Muster wie notebooks/03_Tradeing/tune_backtest.py (Optuna/
TPE, walk-forward Fenster, pro-Basket-Study, persistenter SQLite-Storage,
Robustheits-Zielgrösse mean(fold_returns) - std(fold_returns)) auf den
aktuellen Engine (BacktestConfig/BasketParams/BacktestEngine) -- der alte
Tuner läuft gegen eine separate, ältere pairtrading_engine.py und wird hier
nicht verändert.

Theoriegeleitete Filterparameter vs. tunbare Signalparameter (dieselbe
Trennung wie im alten Tuner, siehe dortiger Modul-Docstring, "Data-Snooping-
Bias" bei Ernie Chan): NUR entry_z, exit_z und max_holding_half_life_multiplier
sind Teil der Optuna-Suche. stop_z sowie alle Filter-/Fensterparameter
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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.config import BacktestConfig, BasketParams, DateRange
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import build_strategy
from src.config import CANDLES_PER_DAY, TUNING_DIR


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
    Performance auf einem einzelnen Glücks-Fold, identisch zur Methodik in
    notebooks/03_Tradeing/tune_backtest.py::_objective_metric."""
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
) -> TuningResult:
    """Optuna/TPE-Suche über (entry_z, exit_z, max_holding_half_life_multiplier)
    für EINEN Basket, Zielgrösse = _objective_metric über alle Walk-Forward-
    Folds innerhalb calibration_range (siehe Modul-Docstring für die
    Suchraum-/Isolations-Begründung).

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

    strategy = build_strategy(strategy_name)
    required_candles = strategy.required_history_candles(base_params)
    warmup_days = math.ceil(required_candles / CANDLES_PER_DAY)

    folds = generate_walkforward_folds(calibration_range, fold_days=fold_days, warmup_days=warmup_days)
    if not folds:
        raise ValueError(
            f"Keine Walk-Forward-Folds für calibration_range={calibration_range} mit "
            f"fold_days={fold_days}, warmup_days={warmup_days} (aus required_history_candles="
            f"{required_candles}) -- Kalibrierungszeitraum vergrössern oder fold_days verkleinern."
        )

    trial_log: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        entry_z = trial.suggest_float("entry_z", 1.0, 4.0)
        exit_z = trial.suggest_float("exit_z", 0.0, max(entry_z - 0.1, 0.0))
        max_holding_half_life_multiplier = trial.suggest_float("max_holding_half_life_multiplier", 1.0, 8.0)

        candidate = replace(
            base_params, entry_z=entry_z, exit_z=exit_z,
            max_holding_half_life_multiplier=max_holding_half_life_multiplier,
        )
        fold_results = [
            _run_fold(
                panel=panel, timeline=timeline, basket=basket, params=candidate, fold=fold,
                strategy_name=strategy_name, initial_capital=initial_capital, warmup_days=warmup_days,
            )
            for fold in folds
        ]
        fold_returns = [r for r, _ in fold_results]
        total_trades = sum(n for _, n in fold_results)
        min_required_trades = min_trades_per_fold * len(folds)

        if total_trades < min_required_trades:
            trial_log.append({
                "trial_number": trial.number, "entry_z": entry_z, "exit_z": exit_z,
                "max_holding_half_life_multiplier": max_holding_half_life_multiplier,
                "status": "pruned_min_trades", "total_trades": total_trades,
                "objective": np.nan, "mean_fold_return": np.nan, "std_fold_return": np.nan,
                "n_folds": len(folds),
            })
            raise optuna.TrialPruned()

        objective_value = _objective_metric(fold_returns)
        arr = np.array(fold_returns, dtype=float)
        trial_log.append({
            "trial_number": trial.number, "entry_z": entry_z, "exit_z": exit_z,
            "max_holding_half_life_multiplier": max_holding_half_life_multiplier,
            "status": "completed", "total_trades": total_trades,
            "objective": objective_value,
            "mean_fold_return": float(arr.mean()), "std_fold_return": float(arr.std(ddof=0)),
            "n_folds": len(folds),
        })
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=seed)
    if use_persistent_storage:
        resolved_storage_path = Path(storage_path) if storage_path is not None else TUNING_DIR / "optuna_walkforward.db"
        resolved_storage_path.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=study_name or basket, storage=f"sqlite:///{resolved_storage_path.resolve()}",
            sampler=sampler, direction="maximize", load_if_exists=True,
        )
    else:
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


def tune_all_baskets(
    *, baskets: tuple[str, ...], panel: dict[str, pd.DataFrame], timeline, base_params: BasketParams,
    calibration_range: DateRange, seed: int = 42, **kwargs,
) -> dict[str, TuningResult]:
    """tune_basket_params für jeden Basket in Folge, mit kurzer
    Fortschrittsausgabe -- **kwargs wird an jeden Aufruf durchgereicht
    (strategy_name, fold_days, n_trials, min_trades_per_fold, storage_path, ...).

    seed wird PRO BASKET versetzt (seed, seed+1, seed+2, ...) statt für alle
    Baskets identisch übergeben: Optunas TPESampler sampelt seine ersten
    n_startup_trials (Default 10) rein zufällig aus dem geseedeten RNG --
    bei identischem Seed UND identischer Suchraum-Definition testet jeder
    Basket sonst exakt dieselbe Sequenz an Kandidaten (nur an
    unterschiedlichen Daten gemessen), was wie eine "unabhängige" Suche
    aussieht, aber keine ist."""
    results: dict[str, TuningResult] = {}
    for i, basket in enumerate(baskets):
        print(f"Tuning {basket} ...")
        result = tune_basket_params(
            basket=basket, panel=panel, timeline=timeline, base_params=base_params,
            calibration_range=calibration_range, seed=seed + i, **kwargs,
        )
        print(f"  {len(result.folds)} Folds, bestes Objective={result.study.best_value:.4f}, "
              f"best_params={result.best_params}")
        results[basket] = result
    return results


def build_basket_overrides(results: dict[str, TuningResult]) -> dict[str, dict]:
    """Extrahiert {basket: best_params} -- direkt als BacktestConfig.basket_overrides einsetzbar."""
    return {basket: dict(result.best_params) for basket, result in results.items()}
