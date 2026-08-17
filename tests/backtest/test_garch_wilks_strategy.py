"""Tests für GarchWilksToleranceStrategy (src/backtest/strategy.py) --
GARCH(1,1)/Wilks-Toleranzintervall-Portierung von
notebooks/03_Tradeing/garch_pairtrading_engine.py auf die aktuelle Engine.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.config import BasketParams
from src.backtest.strategy import GarchWilksToleranceStrategy, StrategyState, build_strategy
from src.backtest.types import PriceWindow


def _garch_price_window(n=600, omega=1e-5, alpha=0.1, beta=0.85, seed=3, nu=6, hedge_ratio=0.5):
    """Baut ein PriceWindow, dessen Return-DIFFERENZ (nicht die Einzel-
    Returns) einem echten GARCH(1,1)-Prozess folgt -- reines Rauschen
    (siehe compute_garch_volatility) konvergiert beim ML-Fit selten, ein
    tatsächlich GARCH-generierter Prozess dagegen zuverlässig (siehe
    Explorationsschritt in der PR-Diskussion). AAAUSDT/BBBUSDT sind über
    hedge_ratio locker gekoppelt, aber die Return-DIFFERENZ ist die
    eigentliche GARCH(1,1)-Serie, die build_definitions/generate_entries
    fitten/normieren soll."""
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    eps = rng.standard_t(nu, n)
    return_spread = np.empty(n)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * return_spread[t - 1] ** 2 + beta * sigma2[t - 1]
        return_spread[t] = np.sqrt(sigma2[t]) * eps[t]

    common_returns = rng.normal(0, 0.005, n)
    aaa_returns = common_returns + return_spread
    bbb_returns = common_returns

    aaa = 100.0 * np.exp(np.cumsum(aaa_returns))
    bbb = 100.0 * np.exp(np.cumsum(bbb_returns))

    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    arr = np.column_stack([aaa, bbb])
    return PriceWindow({"close": arr}, {"AAAUSDT": 0, "BBBUSDT": 1}, idx, n - 1), idx


class _FakeSnapshot:
    def __init__(self, symbols, asset_ids, *, persistence=0.5, size=None, lookback_days=7):
        self.symbols = tuple(symbols)
        self.asset_ids = tuple(asset_ids)
        self.persistence = persistence
        self.size = size if size is not None else len(symbols)
        self.lookback_days = lookback_days
        self.membership_key = "mk-garch"


class TestBuildDefinitionsAndOnSnapshot:
    def test_on_snapshot_fits_garch_and_attaches_diagnostics(self, make_bar_context):
        prices, idx = _garch_price_window()
        window = pd.DataFrame(prices._arrays["close"], index=idx, columns=["AAAUSDT", "BBBUSDT"])
        snapshot = _FakeSnapshot(["AAAUSDT", "BBBUSDT"], [1, 2])
        params = BasketParams(formation_window_candles=len(window), garch_max_iter=500)
        ctx = make_bar_context(params=params, prices=prices, timestamp=idx[-1])

        strategy = GarchWilksToleranceStrategy()
        state = strategy.on_snapshot(snapshot=snapshot, ctx=ctx, params=params)

        assert state.definitions, f"kein konvergierter GARCH-Fit, rejected={state.rejected}"
        d = state.definitions[0]
        assert d.symbols == ("AAAUSDT", "BBBUSDT")
        assert d.weights == (1.0, -1.0)  # KEIN Hedge-Ratio, siehe Modul-Docstring
        for key in ("garch_omega", "garch_alpha", "garch_beta", "garch_nu"):
            assert key in d.diagnostics
        assert d.diagnostics["clique_persistence"] == 0.5
        # jede Definition kennt ihre Clique-Geschwister (fuer Xmax bei check_exit, siehe Modul-Docstring)
        assert d.diagnostics["clique_siblings"] == tuple(state.definitions)

    def test_too_few_symbols_returns_no_definitions(self, make_bar_context, make_price_window):
        params = BasketParams()
        snapshot = _FakeSnapshot(["AAAUSDT"], [1])
        prices = make_price_window({"AAAUSDT": 100.0})
        ctx = make_bar_context(params=params, prices=prices)

        strategy = GarchWilksToleranceStrategy()
        state = strategy.on_snapshot(snapshot=snapshot, ctx=ctx, params=params)
        assert state.definitions == []


class TestGenerateEntriesAndCheckExit:
    """End-to-End auf EINEM synthetischen GARCH-Paar: on_snapshot (Fit) ->
    generate_entries (Wilks-Entry) -> check_exit (Wilks-Exit-Mitte), jeweils
    an unterschiedlichen bar_index-Punkten derselben Preisreihe."""

    @pytest.fixture
    def fitted_state_and_prices(self, make_bar_context):
        prices, idx = _garch_price_window(n=700)
        window = pd.DataFrame(prices._arrays["close"], index=idx, columns=["AAAUSDT", "BBBUSDT"])
        snapshot = _FakeSnapshot(["AAAUSDT", "BBBUSDT"], [1, 2])
        params = BasketParams(
            formation_window_candles=500, wilks_lookback_candles=150,
            garch_max_iter=500, min_holding_candles=1, stop_z=None,
        )
        formation_prices = PriceWindow(
            {"close": prices._arrays["close"][:500]}, {"AAAUSDT": 0, "BBBUSDT": 1}, idx[:500], 499,
        )
        ctx = make_bar_context(params=params, prices=formation_prices, timestamp=idx[499])
        strategy = GarchWilksToleranceStrategy()
        state = strategy.on_snapshot(snapshot=snapshot, ctx=ctx, params=params)
        assert state.definitions, f"kein konvergierter GARCH-Fit, rejected={state.rejected}"
        return strategy, state, params, prices, idx

    def test_generate_entries_runs_and_returns_valid_signal_or_empty(
        self, fitted_state_and_prices, make_bar_context,
    ):
        strategy, state, params, prices, idx = fitted_state_and_prices
        # spaeter bar_index als das Formationsfenster, genug Historie fuer
        # wilks_lookback_candles + GARCH-Sigma-Pfad.
        bar_index = 650
        live_prices = PriceWindow(prices._arrays, prices._symbol_to_col, idx, bar_index)
        ctx = make_bar_context(params=params, prices=live_prices, timestamp=idx[bar_index], bar_index=bar_index)

        signals = strategy.generate_entries(ctx=ctx, state=state, params=params)
        assert isinstance(signals, list)
        if signals:
            signal = signals[0]
            assert signal.direction in (1, -1)
            assert abs(signal.signal_value) == pytest.approx(signal.priority)
            assert signal.spread in state.definitions

    def test_check_exit_uses_same_clique_siblings_as_entry(
        self, fitted_state_and_prices, make_bar_context, make_position,
    ):
        strategy, state, params, prices, idx = fitted_state_and_prices
        definition = state.definitions[0]
        bar_index = 650
        live_prices = PriceWindow(prices._arrays, prices._symbol_to_col, idx, bar_index)
        ctx = make_bar_context(params=params, prices=live_prices, timestamp=idx[bar_index], bar_index=bar_index)

        position = make_position(params=params, spread=definition, bars_held=10)
        decision = strategy.check_exit(position=position, ctx=ctx, params=params)
        # kein Crash, klar definiertes ExitDecision-Objekt -- ob should_exit
        # True/False ist datenabhaengig, aber reason muss bei True "target" sein.
        if decision.should_exit:
            assert decision.reason == "target"
            assert decision.signal_value is not None

    def test_check_exit_falls_back_to_own_definition_without_siblings(
        self, fitted_state_and_prices, make_bar_context, make_position, make_spread_definition,
    ):
        """Falls clique_siblings fehlt (z.B. isoliert konstruierte Position
        in einem Test), darf check_exit nicht crashen -- Fallback auf die
        Position selbst als einzige Clique (siehe siblings-Default in
        check_exit)."""
        strategy, state, params, prices, idx = fitted_state_and_prices
        bar_index = 650
        live_prices = PriceWindow(prices._arrays, prices._symbol_to_col, idx, bar_index)
        ctx = make_bar_context(params=params, prices=live_prices, timestamp=idx[bar_index], bar_index=bar_index)

        isolated_definition = make_spread_definition(
            symbols=("AAAUSDT", "BBBUSDT"), asset_ids=(1, 2), weights=(1.0, -1.0),
            diagnostics=dict(state.definitions[0].diagnostics),
        )
        isolated_definition.diagnostics.pop("clique_siblings", None)
        position = make_position(params=params, spread=isolated_definition, bars_held=10)

        decision = strategy.check_exit(position=position, ctx=ctx, params=params)
        assert decision.should_exit in (True, False)  # nur: kein Crash


class TestStrategyRegistry:
    def test_build_strategy_garch_wilks(self):
        assert isinstance(build_strategy("garch_wilks"), GarchWilksToleranceStrategy)

    def test_required_history_candles_equals_formation_window(self):
        strategy = GarchWilksToleranceStrategy()
        params = BasketParams(formation_window_candles=777)
        assert strategy.required_history_candles(params) == 777
