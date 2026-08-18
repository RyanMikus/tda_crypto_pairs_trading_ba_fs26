import numpy as np
import pandas as pd
import pytest

from src.backtest.config import BasketParams, DateRange
from src.backtest.tuning import (
    TuningResult, build_basket_overrides, generate_walkforward_folds,
    tune_all_baskets, tune_basket_params,
)
from src.backtest.tuning import _is_range_spec, _suggest_param, save_tuning_results  #!#################### NEU
from src.backtest.tuning import load_all_tuning_results  #!#################### NEU
from src.backtest.tuning import DASHBOARD_PORTS, _port_is_open, ensure_dashboard_running  #!#################### NEU
from src.backtest.tuning import fold_days_for_n_windows  #!#################### NEU
from src.backtest.tuning import _required_history_candles_worst_case  #!#################### NEU
from src.backtest.strategy import build_strategy  #!#################### NEU


class TestGenerateWalkforwardFolds:
    def test_correct_count_and_no_overlap(self):
        cal = DateRange(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-31", tz="UTC"))
        folds = generate_walkforward_folds(cal, fold_days=10, warmup_days=1)

        assert len(folds) == 3
        assert folds[0] == DateRange(pd.Timestamp("2025-01-02", tz="UTC"), pd.Timestamp("2025-01-11", tz="UTC"))
        assert folds[-1].end == cal.end
        for prev, nxt in zip(folds, folds[1:]):
            assert nxt.start == prev.end + pd.Timedelta(days=1)  # keine Ueberlappung

    def test_empty_when_too_short(self):
        cal = DateRange(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-05", tz="UTC"))
        assert generate_walkforward_folds(cal, fold_days=30, warmup_days=1) == []

    def test_raises_on_invalid_fold_days(self):
        cal = DateRange(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-31", tz="UTC"))
        with pytest.raises(ValueError):
            generate_walkforward_folds(cal, fold_days=0, warmup_days=1)

    def test_raises_on_negative_warmup(self):
        cal = DateRange(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-01-31", tz="UTC"))
        with pytest.raises(ValueError):
            generate_walkforward_folds(cal, fold_days=10, warmup_days=-1)


#!#################### NEU
class TestFoldDaysForNWindows:
    """fold_days_for_n_windows -- Ruecktransformation von n_windows (wie im
    alten auto_walkforward_layout) auf das von generate_walkforward_folds/
    tune_basket_params erwartete fold_days."""

    def _params(self):
        return BasketParams(formation_window_candles=288 * 7, bollinger_window_candles=288)

    def test_roundtrips_to_requested_window_count(self):
        cal = DateRange(pd.Timestamp("2024-07-23", tz="UTC"), pd.Timestamp("2026-02-26", tz="UTC"))
        params = self._params()
        for n_windows in (3, 6, 10, 20):
            fold_days = fold_days_for_n_windows(cal, n_windows=n_windows, base_params=params, strategy_name="pairwise_zscore")
            import math
            from src.config import candles_per_day
            warmup_days = math.ceil(params.formation_window_candles / candles_per_day(5))
            folds = generate_walkforward_folds(cal, fold_days=fold_days, warmup_days=warmup_days)
            assert len(folds) == n_windows

    def test_raises_when_range_too_short_for_n_windows(self):
        cal = DateRange(pd.Timestamp("2024-07-23", tz="UTC"), pd.Timestamp("2024-07-25", tz="UTC"))
        with pytest.raises(ValueError, match="reicht nicht"):
            fold_days_for_n_windows(cal, n_windows=1, base_params=self._params(), strategy_name="pairwise_zscore")

    def test_raises_on_n_windows_below_one(self):
        cal = DateRange(pd.Timestamp("2024-07-23", tz="UTC"), pd.Timestamp("2026-02-26", tz="UTC"))
        with pytest.raises(ValueError, match="n_windows muss >= 1"):
            fold_days_for_n_windows(cal, n_windows=0, base_params=self._params(), strategy_name="pairwise_zscore")

    def test_search_space_widens_warmup_below_default(self):
        """Steht formation_window_candles selbst im search_space, muss die
        Warm-up-Schaetzung dessen Bereichs-OBERGRENZE beruecksichtigen -- ein
        groesserer search_space-Bereich darf fold_days nie GROESSER machen als
        ohne search_space (mehr Warm-up -> weniger Resttage fuer Folds)."""
        cal = DateRange(pd.Timestamp("2024-07-23", tz="UTC"), pd.Timestamp("2026-02-26", tz="UTC"))
        params = self._params()
        fold_days_without = fold_days_for_n_windows(cal, n_windows=3, base_params=params, strategy_name="pairwise_zscore")
        fold_days_with = fold_days_for_n_windows(
            cal, n_windows=3, base_params=params, strategy_name="pairwise_zscore",
            search_space={"formation_window_candles": (288, 288 * 60)},  # Obergrenze weit ueber params-Fixwert
        )
        assert fold_days_with <= fold_days_without
#!####################


#!#################### NEU
class TestRequiredHistoryCandlesWorstCase:
    """_required_history_candles_worst_case -- Regressionstest fuer den Bug,
    bei dem ein Optuna-Trial formation_window_candles/vol_regime_* groesser
    sampeln konnte als base_params vorsah, weil warmup_days/die Panel-
    Slice-Grenze pro Fold vorher NUR aus base_params (nicht aus dem
    tatsaechlichen search_space-Wertebereich) berechnet wurde -- fuehrte zum
    Warm-up-Guard-Crash (ValueError, G7) mitten in study.optimize()."""

    def test_no_search_space_matches_base_params_only(self):
        strategy = build_strategy("pairwise_zscore")
        params = BasketParams(formation_window_candles=2016)
        assert _required_history_candles_worst_case(
            strategy=strategy, base_params=params, search_space=None,
        ) == strategy.required_history_candles(params)

    def test_range_spec_uses_upper_bound(self):
        strategy = build_strategy("pairwise_zscore")
        params = BasketParams(formation_window_candles=2016)
        worst = _required_history_candles_worst_case(
            strategy=strategy, base_params=params,
            search_space={"formation_window_candles": (500, 9000)},
        )
        assert worst == 9000  # Obergrenze, NICHT der base_params-Fixwert 2016

    def test_dynamic_vol_fields_are_summed_at_worst_case(self):
        """Reproduziert exakt den gemeldeten Fall: dynamic_threshold_enabled/
        dynamic_sizing_enabled ziehen vol_regime_percentile_window_candles +
        vol_regime_lookback_candles heran (siehe
        BaseSpreadZScoreStrategy.required_history_candles) -- beide im
        search_space -> Summe der jeweiligen Obergrenzen, nicht der
        base_params-Fixwerte."""
        strategy = build_strategy("pairwise_zscore")
        params = BasketParams(
            dynamic_threshold_enabled=True, dynamic_sizing_enabled=True,
            vol_regime_percentile_window_candles=17280, vol_regime_lookback_candles=288,
        )
        worst = _required_history_candles_worst_case(
            strategy=strategy, base_params=params,
            search_space={
                "vol_regime_percentile_window_candles": (2880, 51840),
                "vol_regime_lookback_candles": (12, 1440),
            },
        )
        assert worst == 51840 + 1440  # NICHT 17280 + 288 (die base_params-Fixwerte)

    def test_categorical_spec_uses_max_of_options(self):
        strategy = build_strategy("pairwise_zscore")
        params = BasketParams(formation_window_candles=2016)
        worst = _required_history_candles_worst_case(
            strategy=strategy, base_params=params,
            search_space={"formation_window_candles": [1000, 5000, 3000]},
        )
        assert worst == 5000

    def test_fields_outside_search_space_stay_at_base_params_value(self):
        strategy = build_strategy("pairwise_zscore")
        params = BasketParams(formation_window_candles=9999)
        worst = _required_history_candles_worst_case(
            strategy=strategy, base_params=params,
            search_space={"entry_z": (1.0, 4.0)},  # formation_window_candles NICHT im search_space
        )
        assert worst == 9999
#!####################


def _tiny_params():
    """Analog zu TestEndToEndConsistency._params in test_engine.py -- klein
    genug, um innerhalb des ~5.5-Tage-synthetic_panel mehrere Folds zu erlauben."""
    return BasketParams(
        formation_window_candles=100, bollinger_window_candles=50,
        entry_z=1.5, exit_z=0.5, stop_z=4.0,
        max_candidate_pairs=2, pair_prefilter_top_n=6,
        min_half_life_candles=0.1, max_half_life_candles=10_000,
        max_concurrent_positions=2, fee_bps=5.0, slippage_bps=1.0,
    )


class TestTuneBasketParamsSmoke:
    """Kein echter Optuna-Lauf mit vielen Trials (zu langsam fuer die Suite) --
    nur end-to-end Plumbing-Check auf synthetischen Daten (kein Storage-I/O)."""

    def test_runs_end_to_end_on_synthetic_data(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))

        result = tune_basket_params(
            basket="Basket 1", panel=synthetic_panel, timeline=timeline, base_params=_tiny_params(),
            calibration_range=calibration_range, fold_days=1, n_trials=2, use_persistent_storage=False,
            min_trades_per_fold=0.0,  # reiner Plumbing-Check, unabhaengig von tatsaechlicher Handelsaktivitaet
        )

        assert len(result.folds) >= 2
        assert len(result.trials_df) == 2
        assert set(result.best_params) == {"entry_z", "exit_z"}
        assert np.isfinite(result.study.best_value)

    def test_raises_when_calibration_range_too_short(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(hours=6))  # << 1 Fold + Warm-up
        with pytest.raises(ValueError, match="Keine Walk-Forward-Folds"):
            tune_basket_params(
                basket="Basket 1", panel=synthetic_panel, timeline=timeline, base_params=_tiny_params(),
                calibration_range=calibration_range, fold_days=1, n_trials=1, use_persistent_storage=False,
            )

    def test_raises_when_no_trial_passes_min_trades_filter(self, synthetic_panel, timeline):
        """Ein unerreichbar hoher min_trades_per_fold-Filter muss ALLE Trials
        pruned -- soll dann klar scheitern statt stillschweigend "leere"
        best_params zurueckzugeben."""
        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))
        with pytest.raises(RuntimeError, match="min_trades_per_fold-Filter"):
            tune_basket_params(
                basket="Basket 1", panel=synthetic_panel, timeline=timeline, base_params=_tiny_params(),
                calibration_range=calibration_range, fold_days=1, n_trials=2, use_persistent_storage=False,
                min_trades_per_fold=1_000_000.0,
            )


#!#################### NEU
class TestGenericSearchSpace:
    """search_space (statt der frueher fest verdrahteten entry_z/exit_z) --
    src/backtest/strategy.py: GarchWilksToleranceStrategy braucht z.B.
    wilks_p_content/wilks_confidence statt entry_z/exit_z, siehe Modul-Docstring."""

    def test_is_range_spec_tuple_vs_list(self):
        # tuple -> IMMER Bereich, auch wenn die Werte wie eine 2er-Auswahl aussehen
        assert _is_range_spec((1.0, 4.0))
        assert _is_range_spec((0.90, 0.95))
        assert _is_range_spec((1, 8))
        # list -> IMMER kategorial, auch bei rein numerischem Inhalt
        assert not _is_range_spec([0.90, 0.95])
        assert not _is_range_spec(["bps", "high_low"])

    def test_is_range_spec_rejects_wrong_length_tuple(self):
        with pytest.raises(ValueError, match="genau 2 Werte"):
            _is_range_spec((1.0, 2.0, 3.0))

    def test_suggest_param_int_range_uses_suggest_int(self):
        calls = []

        class _FakeTrial:
            def suggest_int(self, name, low, high):
                calls.append(("int", name, low, high))
                return low

            def suggest_float(self, name, low, high):
                calls.append(("float", name, low, high))
                return low

            def suggest_categorical(self, name, choices):
                calls.append(("categorical", name, choices))
                return choices[0]

        trial = _FakeTrial()
        value = _suggest_param(trial, "wilks_lookback_candles", (60, 300), resolved={})
        assert value == 60
        assert calls == [("int", "wilks_lookback_candles", 60, 300)]

    def test_suggest_param_categorical_list_uses_suggest_categorical(self):
        class _FakeTrial:
            def suggest_categorical(self, name, choices):
                return choices[-1]

        trial = _FakeTrial()
        value = _suggest_param(trial, "wilks_p_content", [0.90, 0.95], resolved={})
        assert value == 0.95

    def test_suggest_param_exit_z_capped_by_entry_z(self):
        """Regressionstest fuer die urspruengliche exit_z<entry_z-Kopplung --
        muss beim generischen Umbau erhalten bleiben, WENN beide im
        Suchraum stehen (siehe _suggest_param-Docstring)."""
        captured = {}

        class _FakeTrial:
            def suggest_float(self, name, low, high):
                captured[name] = (low, high)
                return high

        trial = _FakeTrial()
        _suggest_param(trial, "exit_z", (0.0, 4.0), resolved={"entry_z": 2.0})
        assert captured["exit_z"] == (0.0, 1.9)  # min(4.0, entry_z-0.1)

    def test_runs_end_to_end_with_custom_search_space(self, synthetic_panel, timeline):
        """Ein Suchraum ausserhalb der drei Default-Parameter (hier:
        stop_z/max_concurrent_positions als Platzhalter fuer z.B.
        wilks_p_content/wilks_confidence bei garch_wilks) muss genauso
        durchlaufen wie der Default -- reiner Plumbing-Check."""
        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))

        result = tune_basket_params(
            basket="Basket 1", panel=synthetic_panel, timeline=timeline, base_params=_tiny_params(),
            calibration_range=calibration_range, fold_days=1, n_trials=2, use_persistent_storage=False,
            min_trades_per_fold=0.0,
            search_space={"stop_z": (2.0, 5.0), "max_concurrent_positions": (1, 3)},
        )

        assert set(result.best_params) == {"stop_z", "max_concurrent_positions"}
        assert set(result.trials_df.columns) >= {"stop_z", "max_concurrent_positions"}

    def test_unknown_search_space_field_raises(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))

        with pytest.raises(ValueError, match="keine BasketParams-Felder"):
            tune_basket_params(
                basket="Basket 1", panel=synthetic_panel, timeline=timeline, base_params=_tiny_params(),
                calibration_range=calibration_range, fold_days=1, n_trials=1, use_persistent_storage=False,
                search_space={"does_not_exist": (1.0, 2.0)},
            )
#!####################


class TestTuneAllBasketsAndOverrides:
    def test_build_basket_overrides_from_tune_all_baskets(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))

        results = tune_all_baskets(
            baskets=("Basket 1",), panel=synthetic_panel, timeline=timeline, base_params=_tiny_params(),
            calibration_range=calibration_range, fold_days=1, n_trials=2, use_persistent_storage=False,
            min_trades_per_fold=0.0,
        )
        overrides = build_basket_overrides(results)

        assert set(overrides) == {"Basket 1"}
        assert set(overrides["Basket 1"]) == {"entry_z", "exit_z"}

    def test_seed_is_offset_per_basket(self, synthetic_panel, timeline, monkeypatch):
        """Regressionstest fuer den Seed-Kollisions-Bug: identischer Seed +
        identischer Suchraum liess alle Baskets dieselbe Trial-Sequenz ziehen
        (siehe tune_all_baskets-Docstring). tune_basket_params selbst wird
        hier NICHT ausgefuehrt (kein echter Optuna-Lauf) -- nur geprueft,
        dass tune_all_baskets pro Basket einen ansteigenden seed uebergibt."""
        calls = []

        class _FakeStudy:
            best_value = 0.0

        def _fake_tune_basket_params(*, basket, seed, **kwargs):
            calls.append((basket, seed))
            return TuningResult(basket=basket, study=_FakeStudy(), trials_df=pd.DataFrame(), best_params={}, folds=[])

        monkeypatch.setattr("src.backtest.tuning.tune_basket_params", _fake_tune_basket_params)

        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))
        tune_all_baskets(
            baskets=("Basket 1", "Basket 2", "Basket 3"), panel=synthetic_panel, timeline=timeline,
            base_params=_tiny_params(), calibration_range=calibration_range, seed=100,
        )

        assert calls == [("Basket 1", 100), ("Basket 2", 101), ("Basket 3", 102)]


#!#################### NEU
class TestSaveTuningResults:
    """save_tuning_results fehlte komplett -- 05_tuning.ipynb rief
    bt_tuning.save_tuning_results(...) auf und brach mit AttributeError ab."""

    def _fake_result(self, basket="Basket 1"):
        class _FakeStudy:
            best_value = 1.23

        return TuningResult(
            basket=basket, study=_FakeStudy(),
            trials_df=pd.DataFrame([{"trial_number": 0, "entry_z": 2.0, "objective": 1.23}]),
            best_params={"entry_z": 2.0}, folds=[],
        )

    def test_writes_best_params_json_and_trials_parquet(self, tmp_path):
        results = {"Basket 1": self._fake_result("Basket 1"), "Basket 2": self._fake_result("Basket 2")}
        run_dir = save_tuning_results(results, label="my run!", tuning_dir=tmp_path)

        assert run_dir.exists()
        assert (run_dir / "best_params.json").exists()
        assert (run_dir / "trials_basket-1.parquet").exists()
        assert (run_dir / "trials_basket-2.parquet").exists()

        import json
        payload = json.loads((run_dir / "best_params.json").read_text())
        assert payload["Basket 1"]["best_params"] == {"entry_z": 2.0}
        assert payload["Basket 1"]["best_objective"] == pytest.approx(1.23)

    def test_skips_empty_trials_df(self, tmp_path):
        empty_result = TuningResult(
            basket="Basket 1", study=self._fake_result().study,
            trials_df=pd.DataFrame(), best_params={}, folds=[],
        )
        run_dir = save_tuning_results({"Basket 1": empty_result}, label="empty", tuning_dir=tmp_path)
        assert not (run_dir / "trials_basket-1.parquet").exists()

    #!#################### NEU
    def test_stores_candle_minutes_per_basket(self, tmp_path):
        run_dir = save_tuning_results(
            {"Basket 1": self._fake_result("Basket 1")}, label="15min run", tuning_dir=tmp_path, candle_minutes=15,
        )
        import json
        payload = json.loads((run_dir / "best_params.json").read_text())
        assert payload["Basket 1"]["candle_minutes"] == 15

    def test_candle_minutes_defaults_to_5(self, tmp_path):
        run_dir = save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="default", tuning_dir=tmp_path)
        import json
        payload = json.loads((run_dir / "best_params.json").read_text())
        assert payload["Basket 1"]["candle_minutes"] == 5

    def test_stores_fee_preset_per_basket(self, tmp_path):
        run_dir = save_tuning_results(
            {"Basket 1": self._fake_result("Basket 1")}, label="zero cost run", tuning_dir=tmp_path, fee_preset="zero_cost",
        )
        import json
        payload = json.loads((run_dir / "best_params.json").read_text())
        assert payload["Basket 1"]["fee_preset"] == "zero_cost"

    def test_fee_preset_defaults_to_none(self, tmp_path):
        run_dir = save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="default", tuning_dir=tmp_path)
        import json
        payload = json.loads((run_dir / "best_params.json").read_text())
        assert payload["Basket 1"]["fee_preset"] is None
    #!####################
#!####################


#!#################### NEU
class TestLoadAllTuningResults:
    def _fake_result(self, basket="Basket 1"):
        class _FakeStudy:
            best_value = 1.23

        return TuningResult(
            basket=basket, study=_FakeStudy(),
            trials_df=pd.DataFrame([{"trial_number": 0, "entry_z": 2.0, "objective": 1.23}]),
            best_params={"entry_z": 2.0}, folds=[],
        )

    def test_empty_dir_returns_empty_dataframe_with_columns(self, tmp_path):
        df = load_all_tuning_results(tuning_dir=tmp_path / "does_not_exist")
        assert df.empty
        assert list(df.columns) == ["run_id", "basket", "candle_minutes", "fee_preset", "best_objective", "n_trials", "n_folds", "best_params"]

    def test_aggregates_across_multiple_runs_and_candle_minutes(self, tmp_path):
        save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="5min", tuning_dir=tmp_path, candle_minutes=5)
        save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="15min", tuning_dir=tmp_path, candle_minutes=15)
        save_tuning_results({"Basket 2": self._fake_result("Basket 2")}, label="30min", tuning_dir=tmp_path, candle_minutes=30)

        df = load_all_tuning_results(tuning_dir=tmp_path)
        assert len(df) == 3
        assert set(df["candle_minutes"]) == {5, 15, 30}
        assert set(df.loc[df["candle_minutes"] == 30, "basket"]) == {"Basket 2"}

    def test_ignores_run_dirs_without_best_params_json(self, tmp_path):
        stray_dir = tmp_path / "not_a_tuning_run"
        stray_dir.mkdir()
        save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="run", tuning_dir=tmp_path, candle_minutes=5)

        df = load_all_tuning_results(tuning_dir=tmp_path)
        assert len(df) == 1

    #!#################### NEU
    def test_aggregates_fee_preset(self, tmp_path):
        save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="standard", tuning_dir=tmp_path, fee_preset="standard")
        save_tuning_results({"Basket 1": self._fake_result("Basket 1")}, label="zero cost", tuning_dir=tmp_path, fee_preset="zero_cost")

        df = load_all_tuning_results(tuning_dir=tmp_path)
        assert set(df["fee_preset"]) == {"standard", "zero_cost"}
    #!####################
#!####################


#!#################### NEU
class TestOptunaDashboard:
    """ensure_dashboard_running, portiert aus
    notebooks/03_Tradeing/tuning_common.py (Modul jetzt geloescht)."""

    def test_dashboard_ports_cover_all_strategies(self):
        from src.backtest.strategy import STRATEGY_REGISTRY

        assert set(DASHBOARD_PORTS) == set(STRATEGY_REGISTRY)

    def test_port_is_open_false_for_unused_port(self):
        assert _port_is_open("localhost", 8199, timeout=0.2) is False

    def test_ensure_dashboard_running_without_binary_returns_url_and_warns(self, monkeypatch, capsys, tmp_path):
        """Ohne installiertes optuna-dashboard-Kommando darf die Funktion
        NICHT crashen -- sie gibt die URL trotzdem zurueck und warnt nur,
        damit ein Tuning-Lauf ohne das optionale Paket weiterlaeuft."""
        monkeypatch.setattr("shutil.which", lambda _name: None)
        url = ensure_dashboard_running(tmp_path / "doesnotexist.db", port=8199, open_browser=False)
        assert url == "http://localhost:8199/"
        assert "nicht gefunden" in capsys.readouterr().out
#!####################
