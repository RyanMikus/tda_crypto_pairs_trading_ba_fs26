import numpy as np
import pytest

from src.backtest.config import BasketParams
from src.backtest.exit_rules import (
    CliqueBreakRule,
    DataGapRule,
    DEFAULT_EXIT_CHAIN,
    StopLossRule,
    TimeoutRule,
    TopologicalBreakRule,
    evaluate_exit_chain,
)


def _three_asset_position(make_position, make_spread_definition, params, entry_death, entry_lookback_days=1):
    spread = make_spread_definition(
        symbols=("AAAUSDT", "BBBUSDT", "CCCUSDT"), asset_ids=(1, 2, 3), weights=(1.0, -0.5, -0.5),
    )
    return make_position(
        params=params, spread=spread, entry_death=entry_death, entry_lookback_days=entry_lookback_days,
    )


def _zeros_and_spike_window(make_multi_bar_price_window, n, spike=1.0, bbb_price=1.0):
    """AAAUSDT: n-1 Kerzen bei Preis=1.0 (log=0), letzte Kerze bei exp(spike),
    BBBUSDT konstant -> rollierender Z-Score der letzten Kerze ist exakt
    sign(spike) * sqrt(n-1) (siehe test_strategy.py fuer die Herleitung)."""
    aaa = [1.0] * (n - 1) + [float(np.exp(spike))]
    bbb = [bbb_price] * n
    return make_multi_bar_price_window({"AAAUSDT": aaa, "BBBUSDT": bbb})


def _cohesive_and_broken_prices(make_multi_bar_price_window, n_bars=13):
    """Zwei Preis-Historien fuer AAA/BBB/CCC ueber n_bars Kerzen:
    'cohesive' -- alle drei Assets folgen exakt demselben Pfad (rho=1, Distanz=0).
    'broken' -- CCC folgt dem GEGENTEIL von AAA/BBB (rho=-1, Distanz=2, maximal)."""
    rng_returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015, -0.02, 0.01, -0.01, 0.005, -0.015, 0.02, -0.01])
    assert len(rng_returns) == n_bars - 1
    prices_common = np.insert(100 * np.cumprod(1 + rng_returns), 0, 100.0)
    prices_anti = np.insert(100 * np.cumprod(1 - rng_returns), 0, 100.0)

    cohesive = make_multi_bar_price_window({
        "AAAUSDT": prices_common, "BBBUSDT": prices_common, "CCCUSDT": prices_common,
    })
    broken = make_multi_bar_price_window({
        "AAAUSDT": prices_common, "BBBUSDT": prices_common, "CCCUSDT": prices_anti,
    })
    return cohesive, broken


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
    def test_triggers_beyond_threshold(self, make_position, make_bar_context, make_multi_bar_price_window):
        params = BasketParams(stop_z=4.0, bollinger_window_candles=26)
        position = make_position(params=params)  # weights=(1.0, -0.5)
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=26)  # z=5.0 > stop_z
        ctx = make_bar_context(params=params, prices=prices)
        decision = StopLossRule().check(position, ctx)
        assert decision.should_exit
        assert decision.reason == "stop"
        assert decision.signal_value == pytest.approx(5.0)

    def test_does_not_trigger_within_threshold(self, make_position, make_bar_context, make_multi_bar_price_window):
        params = BasketParams(stop_z=4.0, bollinger_window_candles=2)
        position = make_position(params=params)
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=2)  # z=1.0
        ctx = make_bar_context(params=params, prices=prices)
        assert not StopLossRule().check(position, ctx).should_exit

    def test_disabled_when_stop_z_is_none(self, make_position, make_bar_context, make_price_window):
        params = BasketParams(stop_z=None)
        position = make_position(params=params)
        prices = make_price_window({"AAAUSDT": np.exp(50.0), "BBBUSDT": 1.0})  # absurd hoher z -- irrelevant, stop_z=None kehrt vor jedem Preiszugriff zurueck
        ctx = make_bar_context(params=params, prices=prices)
        assert not StopLossRule().check(position, ctx).should_exit

    def test_missing_price_does_not_trigger(self, make_position, make_bar_context, make_multi_bar_price_window):
        """DataGapRule ist fuer fehlende Preise zustaendig, nicht StopLossRule."""
        params = BasketParams(stop_z=4.0, bollinger_window_candles=2)
        position = make_position(params=params)
        prices = make_multi_bar_price_window({"AAAUSDT": [1.0, np.nan], "BBBUSDT": [1.0, 1.0]})
        ctx = make_bar_context(params=params, prices=prices)
        assert not StopLossRule().check(position, ctx).should_exit


class TestTimeoutRule:
    def test_triggers_at_threshold(self, make_position, make_bar_context, make_spread_definition):
        # max_holding = multiplier * half_life = 2.0 * 5.0 = 10
        params = BasketParams(max_holding_half_life_multiplier=2.0)
        spread = make_spread_definition(diagnostics={"half_life_periods": 5.0})
        position = make_position(params=params, spread=spread, bars_held=10)
        ctx = make_bar_context(params=params)
        assert TimeoutRule().check(position, ctx).reason == "timeout"

    def test_does_not_trigger_below_threshold(self, make_position, make_bar_context, make_spread_definition):
        params = BasketParams(max_holding_half_life_multiplier=2.0)
        spread = make_spread_definition(diagnostics={"half_life_periods": 5.0})
        position = make_position(params=params, spread=spread, bars_held=9)
        ctx = make_bar_context(params=params)
        assert not TimeoutRule().check(position, ctx).should_exit

    def test_does_not_trigger_when_half_life_missing(self, make_position, make_bar_context):
        """Ohne half_life_periods in der Diagnostik (z.B. isoliert konstruierte
        Position ohne echten Strategie-Durchlauf) ist die Regel wirkungslos,
        statt faelschlich sofort zu triggern."""
        params = BasketParams(max_holding_half_life_multiplier=2.0)
        position = make_position(params=params, bars_held=10_000)
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


class TestTopologicalBreakRule:
    def test_disabled_by_default(self, make_position, make_bar_context, make_spread_definition, make_multi_bar_price_window):
        params = BasketParams(topo_break_window_candles=10)  # topo_break_enabled default False
        position = _three_asset_position(make_position, make_spread_definition, params, entry_death=0.5)
        _, broken = _cohesive_and_broken_prices(make_multi_bar_price_window)
        ctx = make_bar_context(params=params, prices=broken, bar_index=broken.bar_index)
        assert not TopologicalBreakRule().check(position, ctx).should_exit

    def test_triggers_when_structure_breaks(self, make_position, make_bar_context, make_spread_definition, make_multi_bar_price_window):
        params = BasketParams(topo_break_enabled=True, topo_break_delta=0.01, topo_break_window_candles=10)
        position = _three_asset_position(make_position, make_spread_definition, params, entry_death=0.5)
        _, broken = _cohesive_and_broken_prices(make_multi_bar_price_window)
        ctx = make_bar_context(params=params, prices=broken, bar_index=broken.bar_index)
        decision = TopologicalBreakRule().check(position, ctx)
        assert decision.should_exit
        assert decision.reason == "topological_break"
        assert decision.signal_value == pytest.approx(0.49)  # entry_death - delta

    def test_does_not_trigger_when_still_cohesive(self, make_position, make_bar_context, make_spread_definition, make_multi_bar_price_window):
        params = BasketParams(topo_break_enabled=True, topo_break_delta=0.01, topo_break_window_candles=10)
        position = _three_asset_position(make_position, make_spread_definition, params, entry_death=0.5)
        cohesive, _ = _cohesive_and_broken_prices(make_multi_bar_price_window)
        ctx = make_bar_context(params=params, prices=cohesive, bar_index=cohesive.bar_index)
        assert not TopologicalBreakRule().check(position, ctx).should_exit

    def test_no_exit_before_enough_history(self, make_position, make_bar_context, make_spread_definition, make_multi_bar_price_window):
        """Warm-up-Guard: bar_index < window_candles - 1 -> (noch) nicht pruefbar, kein Fehler."""
        params = BasketParams(topo_break_enabled=True, topo_break_delta=0.01, topo_break_window_candles=10)
        position = _three_asset_position(make_position, make_spread_definition, params, entry_death=0.5)
        _, broken = _cohesive_and_broken_prices(make_multi_bar_price_window)
        ctx = make_bar_context(params=params, prices=broken, bar_index=5)  # < window_candles - 1 = 9
        assert not TopologicalBreakRule().check(position, ctx).should_exit

    def test_no_exit_when_entry_death_at_or_below_delta(self, make_position, make_bar_context, make_spread_definition, make_multi_bar_price_window):
        """Regressionstest: entry_death <= delta wuerde tau<=0 ergeben --
        fcluster mit t<=0 macht JEDEN Punkt zum Singleton (alle Merge-Hoehen
        sind >= 0), also faelschlich 'immer gebrochen' melden. Muss
        stattdessen gar nicht erst pruefen, auch bei eindeutig zerfallenden
        Preisen nicht."""
        params = BasketParams(topo_break_enabled=True, topo_break_delta=0.01, topo_break_window_candles=10)
        position = _three_asset_position(make_position, make_spread_definition, params, entry_death=0.005)  # < delta
        _, broken = _cohesive_and_broken_prices(make_multi_bar_price_window)
        ctx = make_bar_context(params=params, prices=broken, bar_index=broken.bar_index)
        assert not TopologicalBreakRule().check(position, ctx).should_exit


class TestEvaluateExitChain:
    def test_returns_first_matching_reason(self, make_position, make_bar_context, make_spread_definition):
        params = BasketParams(max_data_gap_candles=3, max_holding_half_life_multiplier=1.0)
        spread = make_spread_definition(diagnostics={"half_life_periods": 5.0})
        # Sowohl DataGap als auch Timeout wuerden greifen -- DataGap steht zuerst in DEFAULT_EXIT_CHAIN
        position = make_position(params=params, spread=spread, missing_price_bars=3, bars_held=5)
        ctx = make_bar_context(params=params)
        decision = evaluate_exit_chain(position, ctx, DEFAULT_EXIT_CHAIN)
        assert decision.reason == "data_gap"

    def test_no_exit_when_nothing_matches(self, make_position, make_bar_context, make_multi_bar_price_window):
        params = BasketParams(max_data_gap_candles=99, stop_z=4.0, exit_on_clique_break=False, bollinger_window_candles=2)
        # keine half_life_periods in der Diagnostik -> TimeoutRule wirkungslos
        position = make_position(params=params, bars_held=0, missing_price_bars=0)
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=2)  # z=1.0 < stop_z
        ctx = make_bar_context(params=params, prices=prices)
        decision = evaluate_exit_chain(position, ctx, DEFAULT_EXIT_CHAIN)
        assert not decision.should_exit
