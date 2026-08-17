import pandas as pd
import pytest

from src.backtest.config import BacktestConfig, BasketParams, DateRange
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import StrategyState
from src.backtest.types import NO_EXIT


class _NoOpStrategy:
    """Handelt nie -- Referenz-Strategie fuer den Nullreferenz-Test (Equity
    muss exakt flach bleiben, siehe Plan-Verifikationsschritt 3)."""

    name = "test_no_op"

    def required_history_candles(self, params):
        return 10

    def on_snapshot(self, *, snapshot, ctx, params):
        return StrategyState(membership_key=snapshot.membership_key, fitted_at=ctx.timestamp, definitions=[])

    def generate_entries(self, *, ctx, state, params):
        return []

    def check_exit(self, *, position, ctx, params):
        return NO_EXIT


class TestNoOpStrategyKeepsEquityFlat:
    def test_equity_never_moves(self, synthetic_panel, timeline, register_strategy):
        register_strategy("test_no_op", _NoOpStrategy)
        idx = synthetic_panel["close"].index
        config = BacktestConfig(
            label="noop-test",
            backtest_range=DateRange(idx[100], idx[-1]),
            calibration_range=DateRange(idx[0], idx[99]),
            baskets=("Basket 1",), strategy_name="test_no_op",
        )
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        result = engine.run()

        assert (result.equity_curve["equity"] == config.initial_capital).all()
        assert result.trades.empty


class TestWarmupGuard:
    def test_raises_when_insufficient_history_before_start(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        params = BasketParams(formation_window_candles=10_000)  # mehr als vor idx[100] verfuegbar
        config = BacktestConfig(
            label="warmup-test",
            backtest_range=DateRange(idx[100], idx[-1]),
            calibration_range=DateRange(idx[0], idx[99]),
            baskets=("Basket 1",), strategy_name="pairwise_zscore", default_params=params,
        )
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        with pytest.raises(ValueError, match="Warm-up-Guard"):
            engine.run()


class TestTimestampGridGuard:
    def test_raises_when_start_not_on_grid(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        config = BacktestConfig(
            label="grid-test",
            backtest_range=DateRange(idx[100] + pd.Timedelta(minutes=1), idx[-1]),
            calibration_range=DateRange(idx[0], idx[99]),
            baskets=("Basket 1",), strategy_name="pairwise_zscore",
        )
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        with pytest.raises(ValueError, match="Kerzenraster"):
            engine.run()


def _consistency_config(idx, params, label):
    return BacktestConfig(
        label=label,
        backtest_range=DateRange(idx[350], idx[-1]),
        calibration_range=DateRange(idx[0], idx[349]),
        baskets=("Basket 1",), strategy_name="pairwise_zscore", default_params=params,
    )


class TestEndToEndConsistency:
    def _params(self, stop_z=4.0):
        return BasketParams(
            formation_window_candles=300, entry_z=1.5, exit_z=0.5, stop_z=stop_z,
            max_candidate_pairs=2, pair_prefilter_top_n=6,
            min_half_life_candles=0.1, max_half_life_candles=10_000,
            max_concurrent_positions=2, fee_bps=5.0, slippage_bps=1.0,
        )

    def test_final_equity_matches_initial_capital_plus_realized_pnl(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        config = _consistency_config(idx, self._params(), "pairwise-consistency-test")
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        result = engine.run()

        assert not result.trades.empty, "Test-Setup sollte Trades erzeugen -- sonst ist die Kernaussage nicht pruefbar"
        assert len(engine.warnings) == 0

        final_equity = result.equity_curve["equity"].iloc[-1]
        expected_final_equity = config.initial_capital + result.trades["net_pnl"].sum()
        assert final_equity == pytest.approx(expected_final_equity, rel=1e-9)

        # Force-Close am Ende -- keine offene Position darf uebrig bleiben
        assert result.equity_curve["n_open_positions"].iloc[-1] == 0

    def test_no_trade_ever_opens_beyond_stop_threshold(self, synthetic_panel, timeline):
        """End-to-End-Absicherung des Bugfixes aus strategy.py: kein einziger
        Trade im echten Lauf darf mit |entry_z| >= stop_z eroeffnet worden sein."""
        idx = synthetic_panel["close"].index
        params = self._params(stop_z=3.0)
        config = _consistency_config(idx, params, "stop-guard-test")
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        result = engine.run()

        assert not result.trades.empty
        assert (result.trades["entry_z"].abs() < params.stop_z).all()

    def test_every_position_id_appears_exactly_once_in_trade_log(self, synthetic_panel, timeline):
        idx = synthetic_panel["close"].index
        config = _consistency_config(idx, self._params(), "no-duplicate-trades-test")
        engine = BacktestEngine(config=config, panel=synthetic_panel, timeline=timeline)
        result = engine.run()

        assert not result.trades.empty
        assert result.trades["position_id"].is_unique
