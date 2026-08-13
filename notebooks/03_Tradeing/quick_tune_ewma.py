"""Schneller Walk-Forward-Optuna-Tuner fuer die EWMA-Toleranzintervall-
Pairtrading-Strategie (pairtrading_engine.py, Lin, Chen & Syu 2021).

Ersetzt die frueheren fast_tune.py (kein Walk-Forward, nur EIN Backtest pro
Trial) und tune_backtest.py (Walk-Forward, aber alle Fenstergroessen manuell)
durch EINEN Tuner: weiterhin Walk-Forward (robuster als ein Einzel-Backtest),
aber der Nutzer gibt nur noch n_windows an -- train_days/val_days/step_days/
embargo_days werden automatisch aus dem verfuegbaren Zeitraum abgeleitet
(siehe tuning_common.auto_walkforward_layout), bleiben aber ueberschreibbar.

Zeitliche Trennung: train_val_start/end (Walk-Forward-Suchraum) vs.
    test_start/end (finaler, waehrend des Tunings nie beruehrter
    Testzeitraum) -- vor jedem Trial prueft
    pairtrading_engine.assert_no_test_period_leakage.

Theoriegeleitet vs. tunbar (Data-Snooping-Vermeidung, siehe Ernie Chan):
    NICHT Teil der Optuna-Suche: min_persistence (Clique-Discovery,
        vorgelagert), ADF-autolag/KPSS-regression/FDR-alpha (separate
        Stationaritaets-Diagnostik, literaturbasierte Standardwerte).
    TEIL der Optuna-Suche: lookback_candles, p_content, trade_mode,
        recalib_every, max_holding_hours, max_concurrent_positions,
        ewma_decay, exit_lookback_candles.

Kosten: Src.config.BINANCE_FEE_PRESETS_BPS["standard"] (10.0 bps) als
Default, siehe tuning_common.resolve_fee_bps -- per fee_preset umschaltbar.

Beste Parameter je Basket werden in
data/processed/tuning/best_params_ewma.json abgelegt UND per
write_best_params_to_engine_module() als BEST_PARAMS-Referenzblock in
pairtrading_engine.py geschrieben.

Persistenter Optuna-Storage (SQLite): quick_tune_ewma_study.db, neben diesem
Modul. optuna-dashboard sqlite:///quick_tune_ewma_study.db --port 8080.

Windows: KMP_DUPLICATE_LIB_OK=TRUE wird vorsorglich gesetzt (OpenMP/DLL-
Konflikt bei parallelen Trials).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import pairtrading_engine as pte
import tuning_common as tc

DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent / "quick_tune_ewma_study.db"
STRATEGY_NAME = "ewma"


@dataclass
class QuickTuneResult:
    """Ergebnis eines vollstaendigen tune-Laufs fuer EINEN Basket.

    n_trials_total/trials_df beziehen sich nur auf die in DIESEM Aufruf neu
    gelaufenen Trials. best_params bezieht sich auf die GESAMTE Study.
    """

    basket: str
    study: "optuna.Study"
    trials_df: pd.DataFrame
    best_params: dict
    best_objective: float
    n_trials_total: int
    layout: tc.WalkForwardLayout


def _objective_metric(fold_returns: list[float]) -> float:
    if not fold_returns:
        return -np.inf
    arr = np.array(fold_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return -np.inf
    return float(arr.mean())


def tune_ewma_strategy(
    *,
    basket: str,
    cliques_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    id_to_symbol: dict,
    timestamp_column: str,
    asset_id_column: str,
    close_column: str,
    train_val_start,
    train_val_end,
    test_start,
    test_end,
    n_windows: int = 4,
    train_days: int | None = None,
    val_days: int | None = None,
    step_days: int | None = None,
    embargo_days: int | None = None,
    n_trials: int = 40,
    seed: int = 42,
    fee_preset: str | float = "standard",
    output_path=None,
    study_name: str | None = None,
    storage_path: Path | str | None = None,
    use_persistent_storage: bool = True,
    fixed_params: dict | None = None,
    search_ranges: dict | None = None,
    write_to_engine_module: bool = True,
    open_dashboard: bool = True,
    dashboard_port: int = tc.DASHBOARD_PORTS["ewma"],
) -> QuickTuneResult:
    """Fuehrt das schnelle Walk-Forward-Tuning der EWMA-Strategie fuer EINEN
    Basket durch.

    Default-Suchraum, sofern ein Parameter weder in fixed_params noch in
    search_ranges auftaucht:
        lookback_candles: int in [20, 200]
        p_content: float in [0.80, 0.99]
        trade_mode: kategorial {"max_only", "all_significant"}
        recalib_every: int in [1, 96]
        max_holding_hours: float in [1, 96]
        max_concurrent_positions: int in [1, 25]
        ewma_decay: float in [0.80, 0.995]
        exit_lookback_candles: int in [5, 150]

    Args:
        basket: Name des Baskets (Logging/Rueckgabe; cliques_df sollte
            bereits gefiltert sein).
        cliques_df, ohlcv_df, id_to_symbol, timestamp_column, asset_id_column,
            close_column: wie pairtrading_engine.run_pairtrading_backtest.
        train_val_start, train_val_end: Zeitraum der Walk-Forward-Fenster.
        test_start, test_end: finaler Testzeitraum, waehrend des Tunings nie
            angefasst, nur zur Leakage-Pruefung.
        n_windows: gewuenschte Anzahl Walk-Forward-Fenster -- die einzige
            Pflichtangabe zur Fenstergroesse (siehe Modul-Docstring).
            train_days/val_days/step_days/embargo_days werden daraus
            automatisch abgeleitet (tuning_common.auto_walkforward_layout),
            koennen aber einzeln ueberschrieben werden.
        train_days, val_days, step_days, embargo_days: optionale manuelle
            Overrides -- falls gesetzt, ersetzen sie den jeweiligen
            automatisch abgeleiteten Wert.
        n_trials: Anzahl NEUER Optuna-Trials in diesem Aufruf.
        seed: Seed fuer den TPE-Sampler (nur bei neuer Study wirksam).
        fee_preset: siehe tuning_common.resolve_fee_bps.
        output_path: falls gesetzt, trials_df als Parquet schreiben.
        study_name: Name der Optuna-Study. Default: f"{basket}_ewma".
        storage_path: Pfad zur SQLite-Datei. Default: DEFAULT_STORAGE_PATH.
        use_persistent_storage: True (Default) = SQLite-Storage.
        fixed_params: optionales dict, Teilmenge der acht Suchparameter --
            fixierte Parameter werden NICHT von Optuna gesucht.
        search_ranges: optionales dict {param_name: (lo, hi)} (bzw.
            {param_name: [optionen]} fuer trade_mode) -- ueberschreibt NUR
            den Suchbereich eines weiterhin gesuchten Parameters (siehe
            tuning_common.suggest_param). Ein Parameter, der sowohl in
            fixed_params als auch in search_ranges auftaucht, wird fixiert
            (fixed_params hat Vorrang).
        write_to_engine_module: True (Default) = beste Parameter zusaetzlich
            als BEST_PARAMS-Referenzblock in pairtrading_engine.py schreiben.
        open_dashboard: True (Default) = optuna-dashboard fuer diese Study
            automatisch starten (falls noch nicht gestartet) und den
            Browser-Tab oeffnen, siehe tuning_common.ensure_dashboard_running.
            Ohne Effekt, falls use_persistent_storage=False (In-Memory-Study
            hat kein Dashboard).
        dashboard_port: Port fuer das EWMA-Dashboard, siehe
            tuning_common.DASHBOARD_PORTS.

    Returns:
        QuickTuneResult mit Study, Trials-DataFrame, besten Parametern,
        Objective-Wert und dem verwendeten WalkForwardLayout.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fee_bps = tc.resolve_fee_bps(fee_preset)

    fixed_params = fixed_params or {}
    search_ranges = search_ranges or {}
    known_keys = {
        "lookback_candles", "p_content", "trade_mode", "recalib_every",
        "max_holding_hours", "max_concurrent_positions", "ewma_decay",
        "exit_lookback_candles",
    }
    unknown_fixed = set(fixed_params) - known_keys
    if unknown_fixed:
        raise ValueError(f"fixed_params enthaelt unbekannte Keys: {sorted(unknown_fixed)}")
    unknown_ranges = set(search_ranges) - known_keys
    if unknown_ranges:
        raise ValueError(f"search_ranges enthaelt unbekannte Keys: {sorted(unknown_ranges)}")

    layout = tc.auto_walkforward_layout(train_val_start, train_val_end, n_windows=n_windows)
    effective_train_days = train_days if train_days is not None else layout.train_days
    effective_val_days = val_days if val_days is not None else layout.val_days
    effective_step_days = step_days if step_days is not None else layout.step_days
    effective_embargo_days = embargo_days if embargo_days is not None else layout.embargo_days

    windows = pte.generate_walkforward_windows(
        train_val_start, train_val_end,
        train_days=effective_train_days, val_days=effective_val_days,
        step_days=effective_step_days, embargo_days=effective_embargo_days,
    )
    if not windows:
        raise ValueError(
            f"Keine Walk-Forward-Fenster erzeugt fuer [{train_val_start}, {train_val_end}) "
            f"mit train_days={effective_train_days}, val_days={effective_val_days}, "
            f"step_days={effective_step_days}, embargo_days={effective_embargo_days} -- "
            f"Zeitraum vergroessern oder n_windows verkleinern."
        )
    pte.assert_no_test_period_leakage(windows, test_start, test_end)

    print(f"Walk-Forward-Layout fuer '{basket}' (n_windows={n_windows} angefragt, "
          f"{len(windows)} tatsaechlich erzeugt): train_days={effective_train_days}, "
          f"val_days={effective_val_days}, step_days={effective_step_days}, "
          f"embargo_days={effective_embargo_days}")

    trial_log: list[dict] = []
    n_new_trials_seen = 0
    t_start = time.perf_counter()

    def objective(trial: "optuna.Trial") -> float:
        nonlocal n_new_trials_seen

        sp = lambda name, kind, default_range: tc.suggest_param(  # noqa: E731
            trial, name, kind=kind, default_range=default_range,
            fixed_params=fixed_params, search_ranges=search_ranges,
        )
        lookback_candles = sp("lookback_candles", "int", (20, 200))
        p_content = sp("p_content", "float", (0.80, 0.99))
        trade_mode = sp("trade_mode", "categorical", ["max_only", "all_significant"])
        recalib_every = sp("recalib_every", "int", (1, 96))
        max_holding_hours = sp("max_holding_hours", "float", (1.0, 96.0))
        max_concurrent_positions = sp("max_concurrent_positions", "int", (1, 25))
        ewma_decay = sp("ewma_decay", "float", (0.80, 0.995))
        exit_lookback_candles = sp("exit_lookback_candles", "int", (5, 150))

        n_new_trials_seen += 1
        elapsed = time.perf_counter() - t_start
        eta_str = ""
        if n_new_trials_seen > 1:
            eta = elapsed / (n_new_trials_seen - 1) * (n_trials - (n_new_trials_seen - 1))
            eta_str = f" | ~{eta:.0f}s verbleibend"
        print(f"[Trial {n_new_trials_seen}/{n_trials}] (Study-Trial-Nr. {trial.number}) "
              f"lookback={lookback_candles} p_content={p_content:.3f} trade_mode={trade_mode} "
              f"recalib_every={recalib_every} max_holding_hours={max_holding_hours:.1f} "
              f"max_concurrent={max_concurrent_positions} ewma_decay={ewma_decay:.3f} "
              f"exit_lookback={exit_lookback_candles}{eta_str}")

        pte.assert_no_test_period_leakage(windows, test_start, test_end)
        max_holding_candles = max(1, int(round(max_holding_hours * 60 / 5)))
        fold_returns = []

        for wi, w in enumerate(windows, start=1):
            trades_df = pte.run_pairtrading_backtest(
                cliques_df=cliques_df, ohlcv_df=ohlcv_df, id_to_symbol=id_to_symbol,
                timestamp_column=timestamp_column, asset_id_column=asset_id_column, close_column=close_column,
                start_date=w["val_start"], end_date=w["val_end"] - pd.Timedelta(days=1),
                decay=ewma_decay, lookback_candles=lookback_candles,
                p_content=p_content, confidence=tc.FIXED_CONFIDENCE,
                exit_lookback_candles=exit_lookback_candles,
                max_holding_candles=max_holding_candles, fee_bps=fee_bps,
                trade_mode=trade_mode, recalib_every=recalib_every,
            )
            equity_curve = pte.compute_equity_curve(
                trades_df, initial_capital=tc.FIXED_INITIAL_CAPITAL,
                max_concurrent_positions=max_concurrent_positions,
            )
            stats = pte.compute_backtest_stats(trades_df, equity_curve)
            fold_returns.append(stats["total_return"])
            print(f"    Fenster {wi}/{len(windows)} [{w['val_start'].date()}, {w['val_end'].date()}): "
                  f"{stats['n_trades']} Trades, total_return={stats['total_return']:.4f}")

        objective_value = _objective_metric(fold_returns)
        arr = np.array(fold_returns, dtype=float)
        arr_finite = arr[np.isfinite(arr)]

        print(f"  -> ABGESCHLOSSEN: objective={objective_value:.4f}\n")

        trial_log.append({
            "trial_number": trial.number, "lookback_candles": lookback_candles,
            "p_content": p_content, "trade_mode": trade_mode, "recalib_every": recalib_every,
            "max_holding_hours": max_holding_hours, "max_concurrent_positions": max_concurrent_positions,
            "ewma_decay": ewma_decay, "exit_lookback_candles": exit_lookback_candles,
            "status": "completed", "objective": objective_value,
            "mean_fold_return": float(arr_finite.mean()) if len(arr_finite) else np.nan,
            "std_fold_return": float(arr_finite.std(ddof=0)) if len(arr_finite) else np.nan,
        })
        return objective_value

    if use_persistent_storage:
        # Suchraum-Signatur als Suffix -- siehe tuning_common.search_space_signature
        # Docstring: verhindert Optuna's "dynamic value space"-Fehler, falls
        # sich fixed_params/search_ranges seit einem frueheren Lauf auf
        # demselben (Basket, Strategie) geaendert haben.
        base_study_name = study_name or f"{basket}_ewma"
        signature = tc.search_space_signature(fixed_params, search_ranges)
        effective_study_name = f"{base_study_name}_{signature}"
        resolved_storage_path = Path(storage_path) if storage_path is not None else DEFAULT_STORAGE_PATH
        study = tc.create_or_load_study(effective_study_name, storage_path=resolved_storage_path, seed=seed)
        n_before = len(study.trials)
        print(f"Quick-Tune-EWMA-Study '{effective_study_name}' geladen/erstellt "
              f"({n_before} Trials vorhanden, sqlite:///{resolved_storage_path.resolve()}).")
        if open_dashboard:
            tc.ensure_dashboard_running(resolved_storage_path, port=dashboard_port)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"Quick-Tuning startet: {n_trials} neue Trials, {len(windows)} Walk-Forward-Fenster pro Trial.\n")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Quick-Tuning fertig in {time.perf_counter() - t_start:.1f}s.")

    trials_df = pd.DataFrame(trial_log)
    if output_path is not None:
        trials_df.to_parquet(output_path, index=False)

    n_completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_completed == 0:
        raise RuntimeError(
            f"Kein einziger Trial ist abgeschlossen -- pruefen, ob cliques_df/ohlcv_df "
            f"fuer Basket '{basket}' im Zeitraum [{train_val_start}, {train_val_end}) Daten enthalten."
        )

    # study.best_params enthaelt NUR die von Optuna gesuchten Keys -- Parameter,
    # die ueber fixed_params fixiert waren, tauchen dort nicht auf. Ohne sie
    # hier wieder einzumischen wuerde der gespeicherte/persistierte best_params-
    # Satz unvollstaendig sein (run_pairtrading_backtest braucht ALLE Keys).
    best_params = {**fixed_params, **dict(study.best_params)}
    best_objective = float(study.best_value)
    tc.save_best_params(
        STRATEGY_NAME, basket, best_params, objective=best_objective, n_trials=len(study.trials),
    )
    if write_to_engine_module:
        tc.write_best_params_to_engine_module(STRATEGY_NAME, Path(__file__).resolve().parent / "pairtrading_engine.py")

    return QuickTuneResult(
        basket=basket, study=study, trials_df=trials_df,
        best_params=best_params, best_objective=best_objective,
        n_trials_total=n_trials, layout=layout,
    )


def summarize_quick_tune_results(results: list[QuickTuneResult]) -> None:
    """Druckt eine kurze Zusammenfassung ueber mehrere QuickTuneResult-
    Objekte (i.d.R. eines pro Basket)."""
    print("=" * 78)
    print("ZUSAMMENFASSUNG: Quick-Tuning EWMA-Toleranzintervall-Strategie")
    print("=" * 78)
    for r in results:
        print(f"\nBasket: {r.basket}")
        print(f"  Fenster (angefragt/verwendet): n_windows={r.layout.n_windows}")
        print(f"  Trials gesamt:   {r.n_trials_total}")
        print(f"  Bestes Objective: {r.best_objective:.4f}")
        print(f"  Beste Parameter: {r.best_params}")
    print("\n" + "=" * 78)
