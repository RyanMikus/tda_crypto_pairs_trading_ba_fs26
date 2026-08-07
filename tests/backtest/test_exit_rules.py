import numpy as np
import pytest

from src.backtest.config import BasketParams
from src.backtest.exit_rules import (
    CliqueBreakRule,
    DataGapRule,
    DEFAULT_EXIT_CHAIN,
    StopLossRule,
    TimeoutRule,
    evaluate_exit_chain,
)


class TestDataGapRule:
    def test_triggers_at_threshold(self, make_position, make_bar_context):
        params = BasketParams(max_data_gap_candles=3)
        position = make_position(params=params, missing_price_bars=3)
        ctx = make_bar_context(params=params)
        decision = DataGapRule().check(position, ctx)
        assert decision.should_exit
        assert decision.reason == "data_gap"

    def test_does_not_trigger_below_threshold(self, make_position, make_bar_context):
        params = BasketParams(max_data_gap_candles=3)
        position = make_position(params=params, missing_price_bars=2)
        ctx = make_bar_context(params=params)
        assert not DataGapRule().check(position, ctx).should_exit


class TestStopLossRule:
    def test_triggers_beyond_threshold(self, make_position, make_bar_context, make_price_window):
        params = BasketParams(stop_z=4.0)
        position = make_position(params=params)  # weights=(1.0, -0.5), mu=0, sigma=1
        # spread = log(p1) - 0.5*log(p2); p2=1.0 -> spread = log(p1); log(p1)=4.5 -> z=4.5
        prices = make_price_window({"AAAUSDT": np.exp(4.5), "BBBUSDT": 1.0})
        ctx = make_bar_context(params=params, prices=prices)
        decision = StopLossRule().check(position, ctx)
        assert decision.should_exit
        assert decision.reason == "stop"
        assert decision.signal_value == pytest.approx(4.5)

    def test_does_not_trigger_within_threshold(self, make_position, make_bar_context, make_price_window):
        params = BasketParams(stop_z=4.0)
        position = make_position(params=params)
        prices = make_price_window({"AAAUSDT": np.exp(1.0), "BBBUSDT": 1.0})  # z=1.0
        ctx = make_bar_context(params=params, prices=prices)
        assert not StopLossRule().check(position, ctx).should_exit

    def test_disabled_when_stop_z_is_none(self, make_position, make_bar_context, make_price_window):
        params = BasketParams(stop_z=None)
        position = make_position(params=params)
        prices = make_price_window({"AAAUSDT": np.exp(50.0), "BBBUSDT": 1.0})  # absurd hoher z
        ctx = make_bar_context(params=params, prices=prices)
        assert not StopLossRule().check(position, ctx).should_exit

    def test_missing_price_does_not_trigger(self, make_position, make_bar_context, make_price_window):
        """DataGapRule ist fuer fehlende Preise zustaendig, nicht StopLossRule."""
        params = BasketParams(stop_z=4.0)
        position = make_position(params=params)
        prices = make_price_window({"AAAUSDT": np.nan, "BBBUSDT": 1.0})
        ctx = make_bar_context(params=params, prices=prices)
        assert not StopLossRule().check(position, ctx).should_exit


class TestTimeoutRule:
    def test_triggers_at_threshold(self, make_position, make_bar_context):
        params = BasketParams(max_holding_candles=10)
        position = make_position(params=params, bars_held=10)
        ctx = make_bar_context(params=params)
        assert TimeoutRule().check(position, ctx).reason == "timeout"

    def test_does_not_trigger_below_threshold(self, make_position, make_bar_context):
        params = BasketParams(max_holding_candles=10)
        position = make_position(params=params, bars_held=9)
        ctx = make_bar_context(params=params)
        assert not TimeoutRule().check(position, ctx).should_exit


class TestCliqueBreakRule:
    def test_triggers_when_snapshot_membership_differs(self, make_position, make_bar_context):
        params = BasketParams(exit_on_clique_break=True)
        position = make_position(params=params, membership_key="old-key")

        class FakeSnapshot:
            membership_key = "new-key"

        ctx = make_bar_context(params=params, snapshot=FakeSnapshot())
        assert CliqueBreakRule().check(position, ctx).reason == "clique_break"

    def test_triggers_when_no_snapshot_active(self, make_position, make_bar_context):
        params = BasketParams(exit_on_clique_break=True)
        position = make_position(params=params, membership_key="old-key")
        ctx = make_bar_context(params=params, snapshot=None)
        assert CliqueBreakRule().check(position, ctx).should_exit

    def test_does_not_trigger_when_membership_matches(self, make_position, make_bar_context):
        params = BasketParams(exit_on_clique_break=True)
        position = make_position(params=params, membership_key="same-key")

        class FakeSnapshot:
            membership_key = "same-key"

        ctx = make_bar_context(params=params, snapshot=FakeSnapshot())
        assert not CliqueBreakRule().check(position, ctx).should_exit

    def test_disabled_by_param(self, make_position, make_bar_context):
        params = BasketParams(exit_on_clique_break=False)
        position = make_position(params=params, membership_key="old-key")
        ctx = make_bar_context(params=params, snapshot=None)
        assert not CliqueBreakRule().check(position, ctx).should_exit


class TestEvaluateExitChain:
    def test_returns_first_matching_reason(self, make_position, make_bar_context):
        params = BasketParams(max_data_gap_candles=3, max_holding_candles=5)
        # Sowohl DataGap als auch Timeout wuerden greifen -- DataGap steht zuerst in DEFAULT_EXIT_CHAIN
        position = make_position(params=params, missing_price_bars=3, bars_held=5)
        ctx = make_bar_context(params=params)
        decision = evaluate_exit_chain(position, ctx, DEFAULT_EXIT_CHAIN)
        assert decision.reason == "data_gap"

    def test_no_exit_when_nothing_matches(self, make_position, make_bar_context, make_price_window):
        params = BasketParams(max_data_gap_candles=99, max_holding_candles=99, stop_z=4.0, exit_on_clique_break=False)
        position = make_position(params=params, bars_held=0, missing_price_bars=0)
        prices = make_price_window({"AAAUSDT": np.exp(0.1), "BBBUSDT": 1.0})
        ctx = make_bar_context(params=params, prices=prices)
        decision = evaluate_exit_chain(position, ctx, DEFAULT_EXIT_CHAIN)
        assert not decision.should_exit
