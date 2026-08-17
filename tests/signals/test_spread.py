"""Tests für src/signals/spread.py -- die Bausteine, auf denen
PairwiseZScoreStrategy und ClusterSpreadStrategy aufbauen. Ein Regressionsfehler
hier würde sich still in jeden Backtest fortpflanzen, deshalb eigene, von der
Engine unabhängige Tests."""

import numpy as np
import pandas as pd
import pytest

from src.signals import spread as sp


class TestHedgeRatioAndSpread:
    def test_estimate_hedge_ratio_recovers_known_relationship(self):
        idx = pd.date_range("2025-01-01", periods=500, freq="5min", tz="UTC")
        rng = np.random.default_rng(1)
        common = rng.normal(0, 1, len(idx)).cumsum()
        log_p1 = 0.001 * common + 4.0
        log_p2 = 0.002 * common + 3.0  # hedge_ratio(p1 ~ p2) sollte ~0.5 sein
        df = pd.DataFrame({"A": np.exp(log_p1), "B": np.exp(log_p2)}, index=idx)

        hedge_ratio = sp.estimate_hedge_ratio(df, "A", "B")
        assert hedge_ratio == pytest.approx(0.5, abs=0.05)

    def test_compute_price_spread_matches_manual_formula(self):
        df = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 55.0]})
        spread = sp.compute_price_spread(df, "A", "B", hedge_ratio=0.5)
        expected = np.log(df["A"]) - 0.5 * np.log(df["B"])
        np.testing.assert_allclose(spread.to_numpy(), expected.to_numpy())

    def test_spread_zscore_params_matches_manual_mean_std(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, sigma = sp.spread_zscore_params(series)
        assert mu == pytest.approx(series.mean())
        assert sigma == pytest.approx(series.std(ddof=0))


class TestHalfLife:
    def test_ar1_process_recovers_theoretical_half_life(self):
        """y_t = phi*y_{t-1} + eps, bekanntes phi -- die Regression
        Δy = λ*y_lag + μ sollte λ≈phi-1 wiederfinden, also
        half_life = -ln(2)/λ ≈ ln(2)/(1-phi)."""
        rng = np.random.default_rng(2)
        phi = 0.9
        n = 3000
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.normal(0, 0.1)

        half_life = sp.compute_half_life(pd.Series(y))
        theoretical = np.log(2) / (1 - phi)
        assert half_life == pytest.approx(theoretical, rel=0.2)

    def test_random_walk_has_no_finite_meaningful_half_life(self):
        """Ein driftloser Random Walk hat theoretisch phi=1 (keine Mean-
        Reversion) -- die geschaetzte Half-Life muss um Groessenordnungen
        laenger sein als die eines echten AR(1) mit phi=0.9 (~7 Kerzen,
        siehe test_ar1_process_recovers_theoretical_half_life)."""
        rng = np.random.default_rng(3)
        y = rng.normal(0, 1, 2000).cumsum()
        half_life = sp.compute_half_life(pd.Series(y))
        assert half_life == float("inf") or half_life > 50


class TestEvaluateSpreadQuality:
    def test_detects_stationary_series(self):
        rng = np.random.default_rng(4)
        stationary = pd.Series(rng.normal(0, 1, 500))
        diag = sp.evaluate_spread_quality(stationary)
        assert diag["adf_stationary"]
        assert diag["hurst"] < 0.5

    def test_detects_trending_nonstationary_series(self):
        rng = np.random.default_rng(4)
        trending = pd.Series(np.arange(500, dtype=float) + rng.normal(0, 1, 500))
        diag = sp.evaluate_spread_quality(trending)
        assert not diag["adf_stationary"]


class TestNormalizeCointegrationVector:
    def test_largest_weight_becomes_positive(self):
        weights = pd.Series([-0.5, 0.1, -0.05], index=["A", "B", "C"])
        normalized = sp.normalize_cointegration_vector(weights)
        assert normalized["A"] > 0

    def test_flipping_all_signs_gives_identical_result(self):
        """Kernproblem, das die Funktion loesen soll: Johansen-Eigenvektoren
        haben beliebiges Vorzeichen -- (w) und (-w) muessen auf denselben
        normierten Spread abgebildet werden."""
        weights = pd.Series([-0.5, 0.1, -0.05], index=["A", "B", "C"])
        normalized_a = sp.normalize_cointegration_vector(weights)
        normalized_b = sp.normalize_cointegration_vector(-weights)
        pd.testing.assert_series_equal(normalized_a, normalized_b)

    def test_result_sums_abs_to_one(self):
        weights = pd.Series([2.0, -1.0, 0.5], index=["A", "B", "C"])
        normalized = sp.normalize_cointegration_vector(weights)
        assert normalized.abs().sum() == pytest.approx(1.0)

    def test_scaling_input_does_not_change_result(self):
        weights = pd.Series([2.0, -1.0, 0.5], index=["A", "B", "C"])
        pd.testing.assert_series_equal(
            sp.normalize_cointegration_vector(weights),
            sp.normalize_cointegration_vector(weights * 100),
        )


class TestJohansenAndClusterSpread:
    def test_compute_johansen_test_detects_relation_in_cointegrated_system(self):
        idx = pd.date_range("2025-01-01", periods=800, freq="5min", tz="UTC")
        rng = np.random.default_rng(5)
        common = rng.normal(0, 1, len(idx)).cumsum()
        df = pd.DataFrame({
            "A": np.exp(0.01 * common + 4.0 + rng.normal(0, 0.01, len(idx))),
            "B": np.exp(0.02 * common + 3.0 + rng.normal(0, 0.01, len(idx))),
            "C": np.exp(rng.normal(0, 1, len(idx)).cumsum() * 0.01 + 2.0),
        }, index=idx)

        result = sp.compute_johansen_test(df)
        assert result["n_coint_relations"] >= 1
        assert set(result["eigenvectors"].index) == {"A", "B", "C"}

    def test_build_cluster_spread_matches_manual_weighted_sum(self):
        df = pd.DataFrame({"A": [100.0, 110.0], "B": [50.0, 55.0]})
        weights = pd.Series({"A": 0.6, "B": -0.4})
        spread = sp.build_cluster_spread(df, weights)
        expected = 0.6 * np.log(df["A"]) - 0.4 * np.log(df["B"])
        np.testing.assert_allclose(spread.to_numpy(), expected.to_numpy())


class TestRealizedVolOfSpreadReturns:
    """sigma_S,t^RV -- Bausteine fuer Dynamic Volatility-Adjusted Thresholds
    (docs/project/"Suggested methods...pdf" A.2) und Dynamic Position Sizing (E)."""

    def test_compute_spread_return_rv_matches_manual_std(self):
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        log_a = np.concatenate([[0.0], np.cumsum(returns)])
        df = pd.DataFrame({"A": np.exp(log_a), "B": np.ones(len(log_a))})
        rv = sp.compute_spread_return_rv(df, (1.0, 0.0))
        assert rv == pytest.approx(np.std(returns, ddof=0))

    def test_compute_spread_return_rv_none_for_constant_spread(self):
        df = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]})
        assert sp.compute_spread_return_rv(df, (1.0, -1.0)) is None

    def test_compute_spread_return_rv_none_with_fewer_than_two_returns(self):
        df = pd.DataFrame({"A": [1.0], "B": [1.0]})
        assert sp.compute_spread_return_rv(df, (1.0, -1.0)) is None

    def test_compute_realized_vol_regime_matches_manual_rolling_calc(self):
        rng = np.random.default_rng(3)
        m = 5
        returns = rng.normal(0, 0.01, 40)
        log_a = np.concatenate([[0.0], np.cumsum(returns)])
        df = pd.DataFrame({"A": np.exp(log_a), "B": np.ones(len(log_a))})

        result = sp.compute_realized_vol_regime(
            df, (1.0, 0.0), rv_lookback_candles=m, low_percentile=20.0, high_percentile=80.0,
        )

        rv_series = pd.Series(returns).rolling(m, min_periods=m).std(ddof=0).dropna()
        assert result["rv_base"] == pytest.approx(rv_series.median())
        assert result["rv_low"] == pytest.approx(rv_series.quantile(0.20))
        assert result["rv_high"] == pytest.approx(rv_series.quantile(0.80))
        assert result["rv_low"] <= result["rv_base"] <= result["rv_high"]

    def test_compute_realized_vol_regime_none_when_too_few_points(self):
        """Braucht >= 2*m gueltige RV-Punkte -- weniger ist zu duenn fuer eine
        sinnvolle Perzentil-Schaetzung (z.B. waehrend des Warm-ups)."""
        m = 5
        returns = np.random.default_rng(0).normal(0, 0.01, 2 * m - 1)
        log_a = np.concatenate([[0.0], np.cumsum(returns)])
        df = pd.DataFrame({"A": np.exp(log_a), "B": np.ones(len(log_a))})
        assert sp.compute_realized_vol_regime(df, (1.0, 0.0), rv_lookback_candles=m) is None
