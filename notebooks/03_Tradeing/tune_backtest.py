"""Zweistufiges Hyperparameter-Tuning fuer die Pairtrading-Backtest-Pipeline.

Stufe 1 (guenstiger Signal-Screen, siehe pairtrading_engine.screen_signal_quality):
    Fuer jeden Optuna-Trial wird ZUERST, ohne vollen Backtest, geprueft, ob
    die gezogene (lookback_candles, p_content)-Kombination ueberhaupt einen
    handelbaren Information Coefficient (IC) liefert. Reicht der IC nicht,
    wird der Trial ueber optuna.TrialPruned() abgebrochen -- der teure volle
    Backtest (Stufe 2) laeuft NUR fuer vielversprechende Kandidaten.

Stufe 2 (Optuna/TPE ueber Walk-Forward-Fenster):
    Nur wenn Stufe 1 gruenes Licht gibt, wird pairtrading_engine.
    run_pairtrading_backtest ueber ALLE Walk-Forward-Fenster (siehe
    pairtrading_engine.generate_walkforward_windows) ausgefuehrt -- je Fenster
    einmal Training (fuer die Rekalibrierung innerhalb des Backtests selbst,
    implizit ueber recalib_every) und einmal Validierung. Zielgroesse ist die
    MITTLERE Validierungs-Rendite ueber alle Fenster MINUS ein Strafterm fuer
    hohe Streuung zwischen den Fenstern (Robustheit statt Glueckstreffer).

Zeitliche Trennung (kritisch fuer Validitaet):
    tune_pairtrading_strategy nimmt train_val_start/train_val_end (fuer die
    Walk-Forward-Fenster, IN DENEN Optuna sucht) UND test_start/test_end
    (finaler, waehrend des GESAMTEN Tunings unberuehrter Testzeitraum)
    entgegen. Vor jedem einzelnen Trial wird
    pairtrading_engine.assert_no_test_period_leakage aufgerufen -- ein falsch
    konfigurierter Suchraum bricht das Tuning sofort mit einer klaren
    Fehlermeldung ab, statt den Testzeitraum still zu kontaminieren.

Theoriegeleitete Filterparameter vs. tunbare Signalparameter (wichtig fuer die
methodische Nachvollziehbarkeit in der Thesis -- siehe Ernie Chan, Kapitel
Backtesting, zu Data-Snooping-Bias):
    NICHT Teil der Optuna-Suche (theoriegeleitet, an anderer Stelle in der
    Pipeline fest verankert, keine freien Stellschrauben fuer diesen Tuner):
        - min_persistence (H0-Persistenzschwelle der Vietoris-Rips-Clique-
          Discovery, siehe 02_tda_clustering/helperfunctions.py) -- bestimmt,
          WELCHE Cliquen ueberhaupt existieren, bevor irgendein Backtest
          beginnt. Wird hier nicht anders behandelt als die Cliquen selbst:
          cliques_df kommt als fertiges Ergebnis dieser vorgelagerten Stufe.
        - ADF-autolag, KPSS-regression, FDR-alpha (Stationaritaets-/Mean-
          Reversion-Diagnostik auf Preis-Level-Spreads, siehe
          03_signal_generation/TradeData.ipynb) -- eine SEPARATE, vorgelagerte
          Diagnose-Pipeline auf einem ANDEREN Prozess (Preis-Level-Spread statt
          Renditedifferenz, siehe pairtrading_engine.py-Modul-Docstring
          "WICHTIGE ABGRENZUNG"). Diese Tests entscheiden, ob ein Paar/eine
          Clique ueberhaupt als handelbar gilt -- sie sind statistische
          Vortests mit klar definierten, literaturbasierten Konventionen
          (ADF Standard-Autolag-Auswahl nach Akaike, KPSS Regression="c" fuer
          Niveau-Stationaritaet, FDR-Korrektur alpha=0.05 als Standard-
          Signifikanzniveau bei multiplem Testen), KEINE Hyperparameter einer
          Handelsstrategie, deren "optimaler" Wert vom Backtest-Ergebnis
          abhaengen wuerde. Sie ex-post gegen die Sharpe Ratio zu optimieren
          waere klassisches Data-Snooping (die Auswahlkriterien wuerden an die
          Testperiode angepasst, nicht an eine theoretische Vorgabe).
    TEIL der Optuna-Suche (siehe SEARCH_SPACE_DOC unten): lookback_candles,
        p_content, trade_mode, recalib_every, max_holding_hours,
        max_concurrent_positions -- alles Parameter, die NUR die Handels-
        SIGNALE/das Trade-Management innerhalb von run_pairtrading_backtest
        steuern, nachdem die Cliquen/Paare bereits (durch die obigen,
        theoriegeleiteten Filter) feststehen.

Persistenter Optuna-Storage (SQLite) + optuna-dashboard:
    create_or_load_study() baut eine Study gegen einen SQLite-Storage
    (sqlite:///optuna_study.db im gleichen Verzeichnis wie dieses Modul,
    ueberschreibbar) statt einer In-Memory-Study -- die Study bleibt ueber
    Python-Prozess-Neustarts hinweg erhalten (load_if_exists=True: erneutes
    Aufrufen mit demselben study_name haengt weitere Trials an, statt die
    Study zu verlieren), UND optuna-dashboard kann denselben Storage parallel
    (read-only) oeffnen, um den Fortschritt live im Browser zu verfolgen
    (Parameter-Importance, Parallel-Coordinate-Plot, Optimization-History) --
    siehe Docstring von create_or_load_study fuer den Start-Befehl.

Windows-Kompatibilitaet (siehe Aufgabenstellung, Punkt 6):
    VietorisRipsPersistence (02_tda_clustering) ist NICHT Teil dieses Moduls
    und wird hier nicht aufgerufen -- das dort bekannte OpenMP/DLL-Problem bei
    n_jobs=-1 betrifft ausschliesslich die Clique-Discovery-Stufe, nicht das
    Tuning hier. Trotzdem wird KMP_DUPLICATE_LIB_OK=TRUE beim Modul-Import
    vorsorglich gesetzt (siehe unten), falls parallele Optuna-Trials
    (n_jobs>1 in study.optimize, hier nicht standardmaessig aktiviert, aber
    vom Aufrufer waehlbar) zusammen mit numpy/scipy in mehreren Prozessen
    dieselbe libiomp5md.dll laden -- ein bekanntes Windows-Symptom bei
    mehreren OpenMP-Runtimes im selben Prozessbaum, nicht spezifisch fuer
    dieses Modul, aber prophylaktisch billig zu setzen.

Verwendet ausschliesslich bereits vorhandene Bausteine aus
pairtrading_engine.py (run_pairtrading_backtest, screen_signal_quality,
generate_walkforward_windows, assert_no_test_period_leakage,
compute_equity_curve, compute_backtest_stats) -- die Backtest-Kernfunktion
selbst wird NICHT veraendert, nur aufgerufen.
"""

import os

# Siehe Modul-Docstring, Abschnitt "Windows-Kompatibilitaet" -- MUSS vor dem
# ersten numpy/scipy-Import stehen, um wirksam zu sein.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import pairtrading_engine as pte


# ============================================================================
# Fixe Parameter (nicht Teil der Optuna-Suche, siehe Aufgabenstellung)
# ============================================================================

FIXED_EWMA_DECAY = 0.94
FIXED_CONFIDENCE = 0.95
FIXED_FEE_BPS = 1.5
FIXED_INITIAL_CAPITAL = 10_000.0

# Stufe-1-Screen: Mindest-IC und Mindestanzahl Signale, damit ein Trial ueberhaupt
# den vollen Backtest durchlaufen darf. Kein Paper-Wert -- Heuristik, siehe
# Thesis-Diskussion zur Wahl dieser Schwellen.
SCREEN_MIN_IC = 0.05
SCREEN_MIN_SIGNALS = 10

# Default-Speicherort der persistenten SQLite-Study -- neben diesem Modul,
# damit ein Notebook in 03_Tradeing/ ohne weitere Pfadangaben denselben
# Storage findet wie ein separat gestartetes optuna-dashboard (siehe
# create_or_load_study-Docstring fuer den Start-Befehl).
DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent / "optuna_study.db"


def create_or_load_study(
    study_name: str, *, storage_path: Path | str | None = None, seed: int = 42,
) -> "optuna.Study":
    """Erstellt eine neue oder laedt eine bestehende Optuna-Study aus einem
    persistenten SQLite-Storage.

    Getrennt von tune_pairtrading_strategy aufrufbar, damit derselbe Storage
    -- unabhaengig vom Python-Prozess, der gerade tunt -- auch von
    optuna-dashboard geoeffnet werden kann. load_if_exists=True sorgt dafuer,
    dass ein erneuter Aufruf mit demselben study_name (z.B. "Basket 1" nach
    einem Notebook-Neustart) an die bestehenden Trials ANHAENGT, statt sie zu
    verlieren -- wichtig, weil ein einzelner Tuning-Lauf je nach n_trials und
    Anzahl Walk-Forward-Fenster mehrere Stunden dauern kann und nicht in
    einer einzigen Sitzung abgeschlossen werden muss.

    Args:
        study_name: eindeutiger Name der Study (z.B. der Basket-Name) --
            optuna-dashboard listet Studies nach diesem Namen.
        storage_path: Pfad zur SQLite-Datei. Default: DEFAULT_STORAGE_PATH
            (optuna_study.db neben diesem Modul).
        seed: Seed fuer den TPE-Sampler (nur bei NEU erstellter Study
            wirksam -- eine geladene Study behaelt ihren urspruenglichen
            Sampler-Zustand).

    Returns:
        optuna.Study, verbunden mit dem SQLite-Storage.

    Dashboard starten (siehe auch Modul-Docstring):
        optuna-dashboard sqlite:///optuna_study.db --port 8080 --server wsgiref
        Dann im Browser: http://127.0.0.1:8080
        (--server wsgiref ist auf Windows noetig -- der Dashboard-Default
        gunicorn ist Unix-only und schlaegt mit "ModuleNotFoundError: No
        module named 'fcntl'" fehl; getestet und bestaetigt.)
    """
    import optuna

    path = Path(storage_path) if storage_path is not None else DEFAULT_STORAGE_PATH
    storage_url = f"sqlite:///{path.resolve()}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(
        study_name=study_name, storage=storage_url, sampler=sampler,
        direction="maximize", load_if_exists=True,
    )


@dataclass
class TuningResult:
    """Ergebnis eines vollstaendigen tune_pairtrading_strategy-Laufs fuer EINEN Basket.

    n_trials_total/n_trials_pruned_stage1/trials_df beziehen sich NUR auf die
    in DIESEM Aufruf neu gelaufenen Trials -- bei use_persistent_storage=True
    UND einer bereits bestehenden Study kann `study.trials` mehr (bereits
    frueher gelaufene) Trials enthalten als trials_df Zeilen hat.
    best_params bezieht sich dagegen auf die GESAMTE Study (alle Trials,
    auch aus frueheren Aufrufen), da study.best_params ueber alle Trials
    optimiert.
    """

    basket: str
    study: "optuna.Study"
    trials_df: pd.DataFrame
    best_params: dict
    n_trials_total: int
    n_trials_pruned_stage1: int


def _objective_metric(fold_returns: list[float]) -> float:
    """Kombinierte Zielgroesse: mittlere Fold-Rendite MINUS Strafterm fuer
    Streuung zwischen den Walk-Forward-Faltungen.

    Bewusst NICHT nur die mittlere Rendite (das wuerde den Tuner auf einen
    einzelnen Ausreisser-Fold optimieren lassen) -- der Strafterm
    (Standardabweichung ueber die Folds) sorgt dafuer, dass eine
    Parameterkombination, die in JEDEM Fold aehnlich abschneidet, einer
    Kombination mit hoher mittlerer, aber stark schwankender Rendite
    vorgezogen wird (Robustheit vor Peak-Performance, siehe Aufgabenstellung).

    Args:
        fold_returns: total_return je Walk-Forward-Validierungsfenster.

    Returns:
        mean(fold_returns) - std(fold_returns); -np.inf falls fold_returns leer
        (kein einziges gueltiges Fenster -- Trial soll nicht gewinnen koennen).
    """
    if not fold_returns:
        return -np.inf
    arr = np.array(fold_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return -np.inf
    mean_return = float(arr.mean())
    std_return = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
    return mean_return - std_return


def tune_pairtrading_strategy(
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
    train_days: int = 30,
    val_days: int = 10,
    step_days: int = 15,
    embargo_days: int = 5,
    n_trials: int = 50,
    seed: int = 42,
    output_path=None,
    study_name: str | None = None,
    storage_path: Path | str | None = None,
    use_persistent_storage: bool = True,
) -> TuningResult:
    """Fuehrt das zweistufige Optuna-Tuning fuer EINEN Basket durch.

    Suchraum (siehe Aufgabenstellung; ausfuehrliche Begruendung, welche
    Parameter WARUM tunbar sind bzw. bewusst NICHT getunt werden, siehe
    Modul-Docstring, Abschnitt "Theoriegeleitete Filterparameter vs.
    tunbare Signalparameter"):
        lookback_candles: kategorial {60, 80, 100}
        p_content: kategorial {0.90, 0.95}
        trade_mode: kategorial {"max_only", "all_significant"}
        recalib_every: int in [6, 48]
        max_holding_hours: float in [4, 48] (intern zu max_holding_candles
            umgerechnet, 5-Min-Kerzen -- siehe backtest_pairtrading.ipynb-
            Konvention max_holding_candles = round(h * 60 / 5))
        max_concurrent_positions: int in [3, 15]

    Args:
        basket: Name des Baskets (nur fuer Logging/Rueckgabe, cliques_df sollte
            bereits auf diesen Basket gefiltert sein).
        cliques_df, ohlcv_df, id_to_symbol, timestamp_column, asset_id_column,
            close_column: wie run_pairtrading_backtest/screen_signal_quality.
        train_val_start, train_val_end: Gesamtzeitraum, INNERHALB dessen die
            Walk-Forward-Fenster liegen (siehe generate_walkforward_windows).
        test_start, test_end: finaler Testzeitraum -- wird waehrend des
            gesamten Tunings NIE angefasst, nur zur Leakage-Pruefung verwendet
            (siehe assert_no_test_period_leakage). Der Aufrufer ist dafuer
            verantwortlich, diesen Zeitraum SPAETER, einmalig, mit den
            best_params final zu evaluieren (siehe Notebook-Abschnitt).
        train_days, val_days, step_days, embargo_days: siehe
            generate_walkforward_windows.
        n_trials: Anzahl NEU zu startender Optuna-Trials in diesem Aufruf
            (inkl. Stufe-1-geprunter Trials). Bei use_persistent_storage=True
            UND einer bereits bestehenden Study mit demselben study_name
            kommen diese Trials zu den vorhandenen HINZU (siehe
            create_or_load_study) -- n_trials ist also nicht die Gesamtzahl
            der Trials in der Study, sondern die Anzahl NEUER Trials.
        seed: Seed fuer den TPE-Sampler (nur bei NEUER Study wirksam, siehe
            create_or_load_study).
        output_path: falls gesetzt, wird trials_df dorthin als Parquet
            geschrieben (fuer spaetere Auswertung/Visualisierung in der Thesis).
        study_name: Name der Optuna-Study fuer den persistenten Storage.
            Default: basket (z.B. "Basket 1") -- so kann optuna-dashboard
            Studies eindeutig einem Basket zuordnen.
        storage_path: Pfad zur SQLite-Datei, siehe create_or_load_study.
            Nur relevant, wenn use_persistent_storage=True.
        use_persistent_storage: True (Default) nutzt einen SQLite-Storage
            (siehe create_or_load_study) -- die Study bleibt ueber
            Prozess-Neustarts erhalten UND optuna-dashboard kann sie live
            mitlesen (siehe Modul-Docstring). False nutzt eine reine
            In-Memory-Study (kein Dashboard, kein Neustart-Fortschritt) --
            nur fuer schnelle Tests/CI ohne Dashboard-Bedarf gedacht.

    Returns:
        TuningResult mit dem optuna.Study-Objekt, einem DataFrame ueber alle
        Trials (Parameter + Zielgroesse + Status), den besten gefundenen
        Parametern (als reines dict, direkt an run_pairtrading_backtest
        durchreichbar), und Trial-Zaehlern zur Dokumentation der
        Rechenzeit-Ersparnis durch Stufe-1-Pruning.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    windows = pte.generate_walkforward_windows(
        train_val_start, train_val_end,
        train_days=train_days, val_days=val_days, step_days=step_days, embargo_days=embargo_days,
    )
    if not windows:
        raise ValueError(
            f"Keine Walk-Forward-Fenster erzeugt fuer [{train_val_start}, {train_val_end}) "
            f"mit train_days={train_days}, val_days={val_days}, step_days={step_days}, "
            f"embargo_days={embargo_days} -- Zeitraum vergroessern oder Fenstergroessen verkleinern."
        )
    # Sicherung: VOR dem ersten Trial pruefen, dass kein Fenster den finalen
    # Testzeitraum beruehrt (siehe Modul-Docstring).
    pte.assert_no_test_period_leakage(windows, test_start, test_end)

    trial_log: list[dict] = []
    n_pruned_stage1 = 0
    n_new_trials_seen = 0  # Zaehler NUR fuer diesen Aufruf, unabhaengig von
    # trial.number (= globale Trial-Nummer der Study -- bei einer geladenen
    # Study mit bereits vorhandenen Trials NICHT bei 0 beginnend, wuerde also
    # eine "[Trial 47/2]"-Anzeige erzeugen, wenn die Study schon 45 Trials hat).
    t_start = time.perf_counter()

    def objective(trial: "optuna.Trial") -> float:
        nonlocal n_pruned_stage1, n_new_trials_seen

        lookback_candles = trial.suggest_categorical("lookback_candles", [60, 80, 100])
        p_content = trial.suggest_categorical("p_content", [0.90, 0.95])
        trade_mode = trial.suggest_categorical("trade_mode", ["max_only", "all_significant"])
        recalib_every = trial.suggest_int("recalib_every", 6, 48)
        max_holding_hours = trial.suggest_float("max_holding_hours", 4.0, 48.0)
        max_concurrent_positions = trial.suggest_int("max_concurrent_positions", 3, 15)

        n_new_trials_seen += 1
        elapsed = time.perf_counter() - t_start
        eta_str = ""
        if n_new_trials_seen > 1:
            eta = elapsed / (n_new_trials_seen - 1) * (n_trials - (n_new_trials_seen - 1))
            eta_str = f" | ~{eta:.0f}s verbleibend"
        print(f"[Trial {n_new_trials_seen}/{n_trials}] (Study-Trial-Nr. {trial.number}) "
              f"lookback={lookback_candles} p_content={p_content} "
              f"trade_mode={trade_mode} recalib_every={recalib_every} "
              f"max_holding_hours={max_holding_hours:.1f} max_concurrent={max_concurrent_positions}{eta_str}")

        # Erneute Leakage-Pruefung JE Trial (siehe Modul-Docstring) -- billig
        # (reine Timestamp-Vergleiche), aber die zusaetzliche Sicherheit ist
        # ausdruecklich Teil der Aufgabenstellung.
        pte.assert_no_test_period_leakage(windows, test_start, test_end)

        # --- Stufe 1: guenstiger Signal-Screen ---
        print(f"  Stufe 1: Signal-Screen (lookback={lookback_candles}, p_content={p_content}) laeuft ...")
        screen_df = pte.screen_signal_quality(
            cliques_df, ohlcv_df,
            timestamp_column=timestamp_column, asset_id_column=asset_id_column, close_column=close_column,
            start_date=train_val_start, end_date=train_val_end,
            param_combinations=[{"lookback_candles": lookback_candles, "p_content": p_content}],
            ewma_decay=FIXED_EWMA_DECAY, confidence=FIXED_CONFIDENCE,
        )
        ic = float(screen_df["ic"].iloc[0])
        n_signals = int(screen_df["n_signals"].iloc[0])
        print(f"  Stufe 1 fertig: IC={ic:.4f} (Schwelle={SCREEN_MIN_IC}), n_signals={n_signals} (Schwelle={SCREEN_MIN_SIGNALS})")

        if not np.isfinite(ic) or ic < SCREEN_MIN_IC or n_signals < SCREEN_MIN_SIGNALS:
            n_pruned_stage1 += 1
            print(f"  -> GEPRUNED (Stufe 1 nicht bestanden), kein voller Backtest fuer diesen Trial.\n")
            trial_log.append({
                "trial_number": trial.number, "lookback_candles": lookback_candles,
                "p_content": p_content, "trade_mode": trade_mode, "recalib_every": recalib_every,
                "max_holding_hours": max_holding_hours, "max_concurrent_positions": max_concurrent_positions,
                "stage1_ic": ic, "stage1_n_signals": n_signals, "status": "pruned_stage1",
                "objective": np.nan, "mean_fold_return": np.nan, "std_fold_return": np.nan,
            })
            raise optuna.TrialPruned()

        # --- Stufe 2: voller Backtest ueber alle Walk-Forward-Fenster ---
        print(f"  Stufe 1 bestanden -> Stufe 2: voller Backtest ueber {len(windows)} Walk-Forward-Fenster ...")
        max_holding_candles = max(1, int(round(max_holding_hours * 60 / 5)))
        fold_returns = []

        for wi, w in enumerate(windows, start=1):
            trades_df = pte.run_pairtrading_backtest(
                cliques_df=cliques_df, ohlcv_df=ohlcv_df, id_to_symbol=id_to_symbol,
                timestamp_column=timestamp_column, asset_id_column=asset_id_column, close_column=close_column,
                start_date=w["val_start"], end_date=w["val_end"] - pd.Timedelta(days=1),
                decay=FIXED_EWMA_DECAY, lookback_candles=lookback_candles,
                p_content=p_content, confidence=FIXED_CONFIDENCE,
                max_holding_candles=max_holding_candles, fee_bps=FIXED_FEE_BPS,
                trade_mode=trade_mode, recalib_every=recalib_every,
            )
            equity_curve = pte.compute_equity_curve(
                trades_df, initial_capital=FIXED_INITIAL_CAPITAL,
                max_concurrent_positions=max_concurrent_positions,
            )
            stats = pte.compute_backtest_stats(trades_df, equity_curve)
            fold_returns.append(stats["total_return"])
            print(f"    Fenster {wi}/{len(windows)} [{w['val_start'].date()}, {w['val_end'].date()}): "
                  f"{stats['n_trades']} Trades, total_return={stats['total_return']:.4f}")

        objective_value = _objective_metric(fold_returns)
        arr = np.array(fold_returns, dtype=float)
        arr_finite = arr[np.isfinite(arr)]

        print(f"  -> ABGESCHLOSSEN: objective={objective_value:.4f} "
              f"(mean_fold_return={float(arr_finite.mean()) if len(arr_finite) else float('nan'):.4f}, "
              f"std_fold_return={float(arr_finite.std(ddof=0)) if len(arr_finite) else float('nan'):.4f})\n")

        trial_log.append({
            "trial_number": trial.number, "lookback_candles": lookback_candles,
            "p_content": p_content, "trade_mode": trade_mode, "recalib_every": recalib_every,
            "max_holding_hours": max_holding_hours, "max_concurrent_positions": max_concurrent_positions,
            "stage1_ic": ic, "stage1_n_signals": n_signals, "status": "completed",
            "objective": objective_value,
            "mean_fold_return": float(arr_finite.mean()) if len(arr_finite) else np.nan,
            "std_fold_return": float(arr_finite.std(ddof=0)) if len(arr_finite) else np.nan,
        })
        return objective_value

    if use_persistent_storage:
        effective_study_name = study_name or basket
        resolved_storage_path = Path(storage_path) if storage_path is not None else DEFAULT_STORAGE_PATH
        study = create_or_load_study(effective_study_name, storage_path=storage_path, seed=seed)
        n_trials_before = len(study.trials)
        print(f"Optuna-Study '{effective_study_name}' geladen/erstellt "
              f"(bereits {n_trials_before} Trials vorhanden, SQLite-Storage: "
              f"sqlite:///{resolved_storage_path.resolve()}).")
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"Optuna-Tuning startet: {n_trials} neue Trials, {len(windows)} Walk-Forward-Fenster "
          f"pro vollstaendigem Trial.\n")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Optuna-Tuning fertig in {time.perf_counter() - t_start:.1f}s "
          f"({n_pruned_stage1}/{n_trials} neue Trials in Stufe 1 gepruned).")

    trials_df = pd.DataFrame(trial_log)
    if output_path is not None:
        trials_df.to_parquet(output_path, index=False)

    # study.best_params wirft eine schwer verstaendliche ValueError, wenn
    # KEIN einziger Trial Stufe 1 ueberstanden hat (z.B. SCREEN_MIN_IC zu
    # hoch fuer diesen Basket/Zeitraum gewaehlt) -- hier explizit und mit
    # einer klaren, umsetzbaren Fehlermeldung abfangen statt den kryptischen
    # Optuna-Fehler durchzureichen.
    # Wichtig: hier auf der GESAMTEN Study pruefen (study.trials), nicht nur
    # auf trials_df (nur dieser Aufruf) -- bei use_persistent_storage=True
    # kann ein Aufruf zufaellig 0 neue completed Trials liefern, obwohl die
    # Study aus frueheren Aufrufen bereits gueltige best_params hat.
    n_completed_in_study = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    if n_completed_in_study == 0:
        raise RuntimeError(
            f"Kein einziger Trial (weder in diesem Aufruf noch in einer evtl. "
            f"geladenen frueheren Study) hat den Stufe-1-Signal-Screen ueberstanden "
            f"({n_pruned_stage1}/{n_trials} Trials in diesem Aufruf gepruned) -- "
            f"SCREEN_MIN_IC={SCREEN_MIN_IC} bzw. SCREEN_MIN_SIGNALS={SCREEN_MIN_SIGNALS} "
            f"sind fuer Basket '{basket}' im Zeitraum [{train_val_start}, {train_val_end}) "
            f"zu streng, oder das Signal ist in diesem Basket/Zeitraum tatsaechlich "
            f"schwach. Screening-Schwellen lockern oder Zeitraum/Basket pruefen."
        )

    return TuningResult(
        basket=basket, study=study, trials_df=trials_df,
        best_params=dict(study.best_params), n_trials_total=n_trials,
        n_trials_pruned_stage1=n_pruned_stage1,
    )


def summarize_tuning_results(results: list[TuningResult]) -> None:
    """Druckt eine kurze Zusammenfassung ueber mehrere TuningResult-Objekte
    (i.d.R. eines pro Basket) -- Trial-Zahlen, beste Parameter, Pruning-Quote.

    Fuer die Thesis: die Pruning-Quote (n_trials_pruned_stage1 / n_trials_total)
    dokumentiert direkt die Rechenzeit-Ersparnis durch den zweistufigen Ansatz
    (jeder geprunte Trial hat KEINEN vollen Backtest ausgefuehrt).
    """
    print("=" * 78)
    print("ZUSAMMENFASSUNG: Zweistufiges Hyperparameter-Tuning")
    print("=" * 78)
    for r in results:
        pruned_pct = r.n_trials_pruned_stage1 / r.n_trials_total * 100 if r.n_trials_total else 0.0
        print(f"\nBasket: {r.basket}")
        print(f"  Trials gesamt:            {r.n_trials_total}")
        print(f"  Davon Stufe-1-gepruned:   {r.n_trials_pruned_stage1} ({pruned_pct:.1f}% -- "
              f"kein voller Backtest fuer diese Trials noetig)")
        print(f"  Beste Parameter:          {r.best_params}")
        completed = r.trials_df.loc[r.trials_df['status'] == 'completed']
        if not completed.empty:
            best_row = completed.loc[completed['objective'].idxmax()]
            print(f"  Bestes Objective:         {best_row['objective']:.4f} "
                  f"(mean_fold_return={best_row['mean_fold_return']:.4f}, "
                  f"std_fold_return={best_row['std_fold_return']:.4f})")
    print("\n" + "=" * 78)
