import empyrical as ep
import numpy as np
import pandas as pd
import pytest

from Src.backtest import metrics as m
from Src.backtest.config import BacktestConfig, BasketParams, DateRange
from Src.backtest.engine import BacktestEngine


class TestMaxDrawdownDetail:
    def test_matches_known_curve(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")
        equity = pd.Series([100, 110, 105, 90, 95, 120, 115, 80, 85, 130], index=idx, dtype=float)
        dd = m.max_drawdown_detail(equity)

        assert dd["max_drawdown_pct"] == pytest.approx(80 / 120 - 1)
        assert dd["max_drawdown_dollar"] == pytest.approx(80 - 120)
        assert dd["peak_ts"] == idx[5]
        assert dd["trough_ts"] == idx[7]
        assert dd["recovery_ts"] == idx[9]
        assert dd["duration_to_trough"] == pd.Timedelta(days=2)
        assert dd["duration_to_recovery"] == pd.Timedelta(days=4)

    def test_no_recovery_returns_none(self):
        idx = pd.date_range("2025-01-01", periods=4, freq="1D", tz="UTC")
        equity = pd.Series([100.0, 120.0, 80.0, 90.0], index=idx)  # nie zurueck auf 120
        dd = m.max_drawdown_detail(equity)
        assert dd["recovery_ts"] is None
        assert dd["duration_to_recovery"] is None

    def test_empty_equity_returns_nan_not_crash(self):
        dd = m.max_drawdown_detail(pd.Series(dtype=float))
        assert np.isnan(dd["max_drawdown_pct"])


class TestSharpeRatio:
    def test_matches_manual_calculation(self):
        idx = pd.date_range("2025-01-01", periods=252, freq="1D", tz="UTC")
        rng = np.random.default_rng(42)
        returns = 0.001 + rng.normal(0, 0.0005, len(idx) - 1)
        equity = pd.Series(100 * np.cumprod(1 + np.insert(returns, 0, 0)), index=idx)

        computed = m.to_returns(equity)
        sharpe = ep.sharpe_ratio(computed.to_numpy(), annualization=365.0)
        manual = (computed.mean() / computed.std(ddof=1)) * np.sqrt(365.0)
        assert sharpe == pytest.approx(manual, rel=1e-6)


class TestBetaAlpha:
    def test_buy_and_hold_beta_against_itself_is_one(self):
        idx = pd.date_range("2025-01-01", periods=252, freq="1D", tz="UTC")
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0005, 0.01, len(idx) - 1)
        equity = pd.Series(100 * np.cumprod(1 + np.insert(returns, 0, 0)), index=idx)

        strategy_returns = m.to_returns(equity)
        bench_returns = m.benchmark_returns(equity, equity.index)
        result = m.beta_alpha(strategy_returns, bench_returns, ppy=365.0)

        assert result["beta"] == pytest.approx(1.0, abs=1e-9)
        assert result["alpha_annualized"] == pytest.approx(0.0, abs=1e-9)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_too_few_observations_returns_nan_not_crash(self):
        idx = pd.date_range("2025-01-01", periods=2, freq="1D", tz="UTC")
        returns = pd.Series([0.01], index=idx[1:])
        result = m.beta_alpha(returns, returns, ppy=365.0)
        assert np.isnan(result["beta"])


class TestPeriodsPerYear:
    def test_5min_candles(self):
        assert m.periods_per_year(5) == pytest.approx(365 * 24 * 60 / 5)

    def test_daily_candles(self):
        assert m.periods_per_year(24 * 60) == pytest.approx(365.0)


class TestTradeStatsEdgeCases:
    def test_empty_trades_does_not_crash(self):
        stats = m.trade_stats(pd.DataFrame())
        assert stats["n_trades"] == 0
        assert np.isnan(stats["win_rate"])

    def test_all_losing_trades_cost_drag_is_nan_not_crash(self):
        trades = pd.DataFrame({
            "net_pnl": [-10.0, -20.0], "gross_pnl": [-10.0, -20.0], "total_cost": [1.0, 2.0],
        })
        drag = m.cost_drag(trades)
        assert np.isnan(drag["cost_pct_of_gross_profit"])
        assert drag["total_cost"] == pytest.approx(3.0)


class TestComputeAllMetricsIntegration:
    def test_runs_on_real_engine_output_without_crashing(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        params = BasketParams(
            formation_window_candles=300, entry_z=1.5, exit_z=0.5, stop_z=4.0,
            max_candidate_pairs=2, pair_prefilter_top_n=6,
            min_half_life_candles=0.1, max_half_life_candles=10_000,
            max_concurrent_positions=2, fee_bps=5.0, slippage_bps=1.0,
        )
        config = BacktestConfig(
            label="metrics-integration-test",
            backtest_range=DateRange(idx[350], idx[-1]),
            calibration_range=DateRange(idx[0], idx[349]),
            baskets=("Basket 1",), strategy_name="pairwise_zscore", default_params=params,
        )
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        result = engine.run()

        metrics = m.compute_all_metrics(
            equity_curve=result.equity_curve, trades=result.trades,
            benchmark_close=synthetic_panel["close"]["CCCUSDT"], config=config,
        )

        assert "sharpe_daily" in metrics
        assert "beta_vs_benchmark" in metrics
        assert metrics["n_trades"] == len(result.trades)
        # keine NaN in absolut zentralen Kennzahlen, wenn Trades vorhanden sind
        assert not np.isnan(metrics["total_return"])
