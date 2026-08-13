"""Schneller Walk-Forward-Optuna-Tuner fuer die GARCH-Toleranzintervall-
Pairtrading-Strategie (garch_pairtrading_engine.py).

Ersetzt tune_garch.py: gleiches Walk-Forward-Prinzip, aber der Nutzer gibt
nur noch n_windows an -- train_days/val_days/step_days/embargo_days werden
automatisch aus dem verfuegbaren Zeitraum abgeleitet (siehe
tuning_common.auto_walkforward_layout), bleiben aber ueberschreibbar.

WICHTIG -- Rechenkosten: GARCH(1,1) wird PRO PAAR, PRO CLIQUE-TAG per
Maximum-Likelihood neu gefittet (siehe src.signals.garch.
compute_garch_volatility) -- deutlich teurer als EWMA. Kleinere n_trials/
n_windows fuer die erste Exploration empfohlen (Default n_trials=15).

Zeitliche Trennung: train_val_start/end (Walk-Forward-Suchraum) vs.
    test_start/end (finaler, waehrend des Tunings nie beruehrter
    Testzeitraum) -- vor jedem Trial prueft
    pairtrading_engine.assert_no_test_period_leakage.

Theoriegeleitet vs. tunbar: NICHT Teil der Optuna-Suche: min_persistence,
    ADF/KPSS/FDR, sowie die GARCH-Parameter omega/alpha/beta/nu selbst (die
    werden pro Paar/Tag per Maximum-Likelihood datengetrieben gefittet).
    TEIL der Optuna-Suche: lookback_candles, p_content, trade_mode,
    recalib_every, max_holding_hours, max_concurrent_positions. Kein
    exit_lookback_candles/ewma_decay hier -- der Exit ergibt sich direkt aus
    der Mitte der Wilks-Toleranzgrenzen (siehe garch_pairtrading_engine.py).

Kosten: Src.config.BINANCE_FEE_PRESETS_BPS["standard"] (10.0 bps) als
Default, siehe tuning_common.resolve_fee_bps -- per fee_preset umschaltbar.

Beste Parameter je Basket werden in
data/processed/tuning/best_params_garch.json abgelegt UND per
write_best_params_to_engine_module() als BEST_PARAMS-Referenzblock in
garch_pairtrading_engine.py geschrieben.

Persistenter Optuna-Storage (SQLite): quick_tune_garch_study.db, neben
diesem Modul. optuna-dashboard sqlite:///quick_tune_garch_study.db --port 8082.

Windows: KMP_DUPLICATE_LIB_OK=TRUE wird vorsorglich gesetzt.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import garch_pairtrading_engine as ge
import pairtrading_engine as pte
import tuning_common as tc

FIXED_GARCH_DIST = "t"  # Student-t-Innovationen, siehe Aufgabenstellung-Zusatz-PDF (Crypto-Fat-Tails)
FIXED_GARCH_MAX_ITER = 500

DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent / "quick_tune_garch_study.db"
STRATEGY_NAME = "garch"


@dataclass
class QuickTuneResult:
    """Ergebnis eines vollstaendigen tune-Laufs fuer EINEN Basket."""

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


def tune_garch_strategy(
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
    n_windows: int = 3,
    train_days: int | None = None,
    val_days: int | None = None,
    step_days: int | None = None,
    embargo_days: int | None = None,
    n_trials: int = 15,
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
    dashboard_port: int = tc.DASHBOARD_PORTS["garch"],
) -> QuickTuneResult:
    """Fuehrt das schnelle Walk-Forward-Tuning der GARCH-Toleranzintervall-
    Strategie fuer EINEN Basket durch -- siehe
    quick_tune_ewma.tune_ewma_strategy fuer das gemeinsame Grundprinzip.

    Default-Suchraum, sofern ein Parameter weder in fixed_params noch in
    search_ranges auftaucht (lookback_candles ist ein int-Range, keine
    festen Stuetzstellen mehr -- konsistent mit quick_tune_ewma/
    quick_tune_bollinger. p_content bleibt kategorial: {0.90, 0.95} sind die
    einzigen ueblichen Wilks-Konfidenz-Stuetzstellen):
        lookback_candles: int in [60, 160]
        p_content: kategorial {0.90, 0.95}
        trade_mode: kategorial {"max_only", "all_significant"}
        recalib_every: int in [6, 48]
        max_holding_hours: float in [4, 48]
        max_concurrent_positions: int in [3, 15]

    Args:
        n_windows, train_days, val_days, step_days, embargo_days: siehe
            quick_tune_ewma.tune_ewma_strategy -- automatisches Layout,
            manuell ueberschreibbar. Default n_windows=3 (statt 4 bei
            EWMA/Bollinger) wegen der hoeheren GARCH-Fit-Kosten pro Trial.
        n_trials: Default 15 (statt 40) -- siehe Modul-Docstring.
        fee_preset: siehe tuning_common.resolve_fee_bps.
        fixed_params: optionales dict, Teilmenge der sechs Suchparameter --
            fixierte Parameter werden NICHT von Optuna gesucht.
        search_ranges: optionales dict {param_name: (lo, hi)} (bzw.
            {param_name: [optionen]} fuer trade_mode/p_content) --
            ueberschreibt NUR den Suchbereich eines weiterhin gesuchten
            Parameters (siehe tuning_common.suggest_param).
        write_to_engine_module: True (Default) = beste Parameter zusaetzlich
            als BEST_PARAMS-Referenzblock in garch_pairtrading_engine.py
            schreiben.
        open_dashboard: True (Default) = optuna-dashboard fuer diese Study
            automatisch starten (falls noch nicht gestartet) und den
            Browser-Tab oeffnen, siehe tuning_common.ensure_dashboard_running.
        dashboard_port: Port fuer das GARCH-Dashboard, siehe
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
        "lookback_candles", "p_content", "trade_mode",
        "recalib_every", "max_holding_hours", "max_concurrent_positions",
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
        lookback_candles = sp("lookback_candles", "int", (60, 160))
        p_content = sp("p_content", "categorical", [0.90, 0.95])
        trade_mode = sp("trade_mode", "categorical", ["max_only", "all_significant"])
        recalib_every = sp("recalib_every", "int", (6, 48))
        max_holding_hours = sp("max_holding_hours", "float", (4.0, 48.0))
        max_concurrent_positions = sp("max_concurrent_positions", "int", (3, 15))

        n_new_trials_seen += 1
        elapsed = time.perf_counter() - t_start
        eta_str = ""
        if n_new_trials_seen > 1:
            eta = elapsed / (n_new_trials_seen - 1) * (n_trials - (n_new_trials_seen - 1))
            eta_str = f" | ~{eta:.0f}s verbleibend"
        print(f"[Trial {n_new_trials_seen}/{n_trials}] (Study-Trial-Nr. {trial.number}) "
              f"lookback={lookback_candles} p_content={p_content} trade_mode={trade_mode} "
              f"recalib_every={recalib_every} max_holding_hours={max_holding_hours:.1f} "
              f"max_concurrent={max_concurrent_positions}{eta_str}")

        pte.assert_no_test_period_leakage(windows, test_start, test_end)
        max_holding_candles = max(1, int(round(max_holding_hours * 60 / 5)))
        fold_returns = []

        for wi, w in enumerate(windows, start=1):
            trades_df = ge.run_garch_backtest(
                cliques_df=cliques_df, ohlcv_df=ohlcv_df, id_to_symbol=id_to_symbol,
                timestamp_column=timestamp_column, asset_id_column=asset_id_column, close_column=close_column,
                start_date=w["val_start"], end_date=w["val_end"] - pd.Timedelta(days=1),
                lookback_candles=lookback_candles, p_content=p_content, confidence=tc.FIXED_CONFIDENCE,
                max_holding_candles=max_holding_candles, fee_bps=fee_bps,
                trade_mode=trade_mode, recalib_every=recalib_every,
                garch_dist=FIXED_GARCH_DIST, garch_max_iter=FIXED_GARCH_MAX_ITER,
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
        base_study_name = study_name or f"{basket}_garch"
        signature = tc.search_space_signature(fixed_params, search_ranges)
        effective_study_name = f"{base_study_name}_{signature}"
        resolved_storage_path = Path(storage_path) if storage_path is not None else DEFAULT_STORAGE_PATH
        study = tc.create_or_load_study(effective_study_name, storage_path=resolved_storage_path, seed=seed)
        n_before = len(study.trials)
        print(f"Quick-Tune-GARCH-Study '{effective_study_name}' geladen/erstellt "
              f"({n_before} Trials vorhanden, sqlite:///{resolved_storage_path.resolve()}).")
        if open_dashboard:
            tc.ensure_dashboard_running(resolved_storage_path, port=dashboard_port)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"Quick-Tuning startet: {n_trials} neue Trials, {len(windows)} Walk-Forward-Fenster pro Trial "
          f"(GARCH-Fit pro Paar/Tag -- kann dauern).\n")
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
    # Satz unvollstaendig sein (run_garch_backtest braucht ALLE Keys).
    best_params = {**fixed_params, **dict(study.best_params)}
    best_objective = float(study.best_value)
    tc.save_best_params(
        STRATEGY_NAME, basket, best_params, objective=best_objective, n_trials=len(study.trials),
    )
    if write_to_engine_module:
        tc.write_best_params_to_engine_module(
            STRATEGY_NAME, Path(__file__).resolve().parent / "garch_pairtrading_engine.py",
        )

    return QuickTuneResult(
        basket=basket, study=study, trials_df=trials_df,
        best_params=best_params, best_objective=best_objective,
        n_trials_total=n_trials, layout=layout,
    )


def summarize_quick_tune_results(results: list[QuickTuneResult]) -> None:
    """Druckt eine kurze Zusammenfassung ueber mehrere QuickTuneResult-
    Objekte (i.d.R. eines pro Basket)."""
    print("=" * 78)
    print("ZUSAMMENFASSUNG: Quick-Tuning GARCH-Toleranzintervall-Strategie")
    print("=" * 78)
    for r in results:
        print(f"\nBasket: {r.basket}")
        print(f"  Fenster (angefragt/verwendet): n_windows={r.layout.n_windows}")
        print(f"  Trials gesamt:   {r.n_trials_total}")
        print(f"  Bestes Objective: {r.best_objective:.4f}")
        print(f"  Beste Parameter: {r.best_params}")
    print("\n" + "=" * 78)
