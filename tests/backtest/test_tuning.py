import numpy as np
import pandas as pd
import pytest

from Src.backtest.config import BasketParams, DateRange
from Src.backtest.tuning import (
    TuningResult, build_basket_overrides, generate_walkforward_folds, tune_all_baskets, tune_basket_params,
)


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
        assert set(result.best_params) == {"entry_z", "exit_z", "max_holding_half_life_multiplier"}
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
        assert set(overrides["Basket 1"]) == {"entry_z", "exit_z", "max_holding_half_life_multiplier"}

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

        monkeypatch.setattr("Src.backtest.tuning.tune_basket_params", _fake_tune_basket_params)

        idx = synthetic_panel["close"].index
        calibration_range = DateRange(idx[0], idx[0] + pd.Timedelta(days=4))
        tune_all_baskets(
            baskets=("Basket 1", "Basket 2", "Basket 3"), panel=synthetic_panel, timeline=timeline,
            base_params=_tiny_params(), calibration_range=calibration_range, seed=100,
        )

        assert calls == [("Basket 1", 100), ("Basket 2", 101), ("Basket 3", 102)]
