import numpy as np
import pandas as pd
import pytest

from src.backtest.config import BasketParams
from src.backtest.strategy import (
    ClusterSpreadStrategy,
    PairwiseZScoreStrategy,
    StrategyState,
    build_strategy,
)


def _state(definitions):
    return StrategyState(membership_key="mk", fitted_at=pd.Timestamp("2025-01-01", tz="UTC"), definitions=definitions)


def _zeros_and_spike_window(make_multi_bar_price_window, n, spike=1.0, bbb_price=1.0):
    """AAAUSDT: n-1 Kerzen bei Preis=1.0 (log=0), letzte Kerze bei exp(spike).
    BBBUSDT konstant -> Spread = log(AAA) (der -0.5*log(BBB)-Term faellt weg,
    log(1)=0). Der rollierende Z-Score der letzten Kerze ist dann -- aus der
    Populations-Std-Formel herleitbar, unabhaengig vom genauen Wert von
    spike -- exakt sign(spike) * sqrt(n-1). Macht die Ziel-Z-Werte der alten
    (eingefrorenen) Tests exakt reproduzierbar: n=2->z=1.0, n=5->z=2.0,
    n=10->z=3.0, n=26->z=5.0."""
    aaa = [1.0] * (n - 1) + [float(np.exp(spike))]
    bbb = [bbb_price] * n
    return make_multi_bar_price_window({"AAAUSDT": aaa, "BBBUSDT": bbb})


def _near_zero_z_window(make_multi_bar_price_window, last=0.05):
    """Alternierende, verrauschte AAAUSDT-Historie (Std != 0) plus eine
    letzte Kerze nahe am rollierenden Mittelwert (~0) -> kleiner, aber nicht
    exakt-0 Z-Score (fuer Target-Exit-Tests, die |z| <= exit_z brauchen).
    9 Kerzen, bollinger_window_candles muss dazu passend auf 9 gesetzt werden."""
    aaa_log = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, last]
    aaa = [float(np.exp(lp)) for lp in aaa_log]
    bbb = [1.0] * len(aaa_log)
    return make_multi_bar_price_window({"AAAUSDT": aaa, "BBBUSDT": bbb})


class TestGenerateEntries:
    def test_no_signal_below_entry_threshold(self, make_spread_definition, make_bar_context, make_multi_bar_price_window):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=4.0, bollinger_window_candles=2)
        definition = make_spread_definition()
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=2)  # z=1.0 < entry_z
        ctx = make_bar_context(params=params, prices=prices)
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert signals == []

    def test_signal_between_entry_and_stop(self, make_spread_definition, make_bar_context, make_multi_bar_price_window):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=4.0, bollinger_window_candles=10)
        definition = make_spread_definition()
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=10)  # z=3.0
        ctx = make_bar_context(params=params, prices=prices)
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert len(signals) == 1
        assert signals[0].direction == -1  # z>0 -> Spread SHORT
        assert signals[0].signal_value == pytest.approx(3.0)

    def test_no_signal_when_already_beyond_stop_threshold(self, make_spread_definition, make_bar_context, make_multi_bar_price_window):
        """Regressionstest fuer einen echten Bug: ein Signal, dessen Z-Score
        schon jenseits von stop_z liegt, ist per Definition ungueltig (stop_z
        ist die Invalidierungsschwelle) -- ein Entry HIER wuerde im
        naechsten Bar sofort wieder auf den Stop laufen. generate_entries
        MUSS solche Kandidaten verwerfen."""
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=4.0, bollinger_window_candles=26)
        definition = make_spread_definition()
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=26)  # z=5.0 > stop_z
        ctx = make_bar_context(params=params, prices=prices)
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert signals == []

    def test_extreme_zscore_allowed_when_stop_z_is_none(self, make_spread_definition, make_bar_context, make_multi_bar_price_window):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=None, bollinger_window_candles=26)
        definition = make_spread_definition()
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=26)  # z=5.0
        ctx = make_bar_context(params=params, prices=prices)
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert len(signals) == 1

    def test_skips_definition_when_asset_at_position_limit(
        self, make_spread_definition, make_bar_context, make_multi_bar_price_window, make_position,
    ):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=4.0, max_positions_per_asset=1, bollinger_window_candles=10)
        definition = make_spread_definition(asset_ids=(1, 2))
        existing = make_position(
            params=params,
            spread=make_spread_definition(asset_ids=(1, 99), symbols=("AAAUSDT", "ZZZUSDT")),
        )
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=10)  # z=3.0, waere sonst ein gueltiges Signal
        ctx = make_bar_context(params=params, prices=prices, open_positions=[existing])
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert signals == []

    def test_direction_sign_convention_for_negative_zscore(self, make_spread_definition, make_bar_context, make_multi_bar_price_window):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=4.0, bollinger_window_candles=10)
        definition = make_spread_definition()
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=10, spike=-1.0)  # z=-3.0 -> LONG
        ctx = make_bar_context(params=params, prices=prices)
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert signals[0].direction == 1

    def test_skips_definition_with_missing_price(self, make_spread_definition, make_bar_context, make_multi_bar_price_window):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(entry_z=2.0, stop_z=4.0, bollinger_window_candles=2)
        definition = make_spread_definition()
        prices = make_multi_bar_price_window({"AAAUSDT": [1.0, np.nan], "BBBUSDT": [1.0, 1.0]})
        ctx = make_bar_context(params=params, prices=prices)
        signals = strategy.generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert signals == []


class TestCheckExit:
    def test_no_exit_before_min_holding(self, make_bar_context, make_multi_bar_price_window, make_position):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(exit_z=0.5, min_holding_candles=5, bollinger_window_candles=9)
        position = make_position(params=params, bars_held=2)
        prices = _near_zero_z_window(make_multi_bar_price_window)  # waere sonst ein Target-Exit
        ctx = make_bar_context(params=params, prices=prices)
        assert not strategy.check_exit(position=position, ctx=ctx, params=params).should_exit

    def test_target_exit_after_min_holding(self, make_bar_context, make_multi_bar_price_window, make_position):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(exit_z=0.5, min_holding_candles=1, bollinger_window_candles=9)
        position = make_position(params=params, bars_held=1)
        prices = _near_zero_z_window(make_multi_bar_price_window)  # |z| ~ 0.047 <= exit_z
        ctx = make_bar_context(params=params, prices=prices)
        decision = strategy.check_exit(position=position, ctx=ctx, params=params)
        assert decision.should_exit
        assert decision.reason == "target"

    def test_no_exit_when_still_far_from_mean(self, make_bar_context, make_multi_bar_price_window, make_position):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(exit_z=0.5, min_holding_candles=1, bollinger_window_candles=5)
        position = make_position(params=params, bars_held=5)
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=5)  # z=2.0 > exit_z
        ctx = make_bar_context(params=params, prices=prices)
        assert not strategy.check_exit(position=position, ctx=ctx, params=params).should_exit


class TestSymmetricExit:
    """Opposite-Band-Exit (BasketParams.symmetric_exit=True): haelt bis zum
    GEGENUEBERLIEGENDEN Band statt bis zur Rueckkehr zur Mitte."""

    def test_short_exits_at_opposite_band(self, make_bar_context, make_multi_bar_price_window, make_position):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(symmetric_exit=True, entry_z=2.0, min_holding_candles=1, bollinger_window_candles=5)
        position = make_position(params=params, bars_held=1, direction=-1)  # Short, Entry bei z~+2.0
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=5, spike=-1.0)  # z=-2.0
        ctx = make_bar_context(params=params, prices=prices)
        decision = strategy.check_exit(position=position, ctx=ctx, params=params)
        assert decision.should_exit
        assert decision.reason == "target"

    def test_long_exits_at_opposite_band(self, make_bar_context, make_multi_bar_price_window, make_position):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(symmetric_exit=True, entry_z=2.0, min_holding_candles=1, bollinger_window_candles=5)
        position = make_position(params=params, bars_held=1, direction=1)  # Long, Entry bei z~-2.0
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=5, spike=1.0)  # z=+2.0
        ctx = make_bar_context(params=params, prices=prices)
        decision = strategy.check_exit(position=position, ctx=ctx, params=params)
        assert decision.should_exit
        assert decision.reason == "target"

    def test_no_exit_before_reaching_opposite_band(self, make_bar_context, make_multi_bar_price_window, make_position):
        """Kleiner Z-Score nahe der Mitte (waere unter dem alten exit_z=0.5-
        Regime ein Target-Exit) loest im symmetrischen Modus NICHTS aus --
        der Spread muss erst bis zum gegenueberliegenden Band durchlaufen."""
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(symmetric_exit=True, entry_z=0.5, min_holding_candles=1, bollinger_window_candles=9)
        position = make_position(params=params, bars_held=1, direction=-1)
        prices = _near_zero_z_window(make_multi_bar_price_window)  # z ~ +0.047, noch nicht durchgekreuzt
        ctx = make_bar_context(params=params, prices=prices)
        assert not strategy.check_exit(position=position, ctx=ctx, params=params).should_exit

    def test_uses_entry_threshold_frozen_at_entry_not_current_params(
        self, make_bar_context, make_multi_bar_price_window, make_position,
    ):
        """b (das Zielband) kommt aus position.entry_threshold_z (bei Entry
        eingefroren -- ueber EntrySignal.meta gesetzt, siehe
        Portfolio._execute_open), NICHT aus dem aktuell an check_exit
        uebergebenen params.entry_z -- wichtig, sobald entry_z sich waehrend
        der Haltedauer aendert (regimeabhaengige dynamische Thresholds,
        siehe BasketParams.dynamic_threshold_enabled), damit eine offene
        Position konsistent zu dem Band bleibt, fuer das sie eroeffnet wurde."""
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(symmetric_exit=True, entry_z=3.0, min_holding_candles=1, bollinger_window_candles=5)
        position = make_position(params=params, bars_held=1, direction=-1, entry_threshold_z=2.0)
        prices = _zeros_and_spike_window(make_multi_bar_price_window, n=5, spike=-1.0)  # z=-2.0
        ctx = make_bar_context(params=params, prices=prices)
        decision = strategy.check_exit(position=position, ctx=ctx, params=params)
        assert decision.should_exit  # b=2.0 (eingefroren) erreicht, obwohl aktuelles params.entry_z=3.0 waere


def _regime_probe_window(make_multi_bar_price_window, spike=1.0, m=4):
    """bollinger_window_candles = m+1, damit Z-Score-Fenster UND das fuer
    sigma_S,t^RV genutzte Fenster (m+1 Kerzen) EXAKT dieselben Daten sehen --
    macht beide Groessen von Hand nachrechenbar. AAAUSDT: m Kerzen bei
    Preis=1.0 (log=0), letzte Kerze bei exp(spike); BBBUSDT konstant (faellt
    aus dem Spread mit weights=(1.0,-0.5) heraus, siehe Default in
    make_spread_definition). z = sign(spike)*sqrt(m) (siehe
    _zeros_and_spike_window-Herleitung mit n=m+1). Die Spread-Return-Reihe
    ist [0]*(m-1) + [spike] -> rv_t = population-std davon."""
    n = m + 1
    aaa = [1.0] * (n - 1) + [float(np.exp(spike))]
    bbb = [1.0] * n
    prices = make_multi_bar_price_window({"AAAUSDT": aaa, "BBBUSDT": bbb})
    rv_t = float(np.std([0.0] * (m - 1) + [spike], ddof=0))
    z = float(np.sign(spike) * np.sqrt(m))
    return prices, rv_t, z


class TestDynamicThreshold:
    """BasketParams.dynamic_threshold_enabled (PDF A.2): entry_z verschiebt
    sich um entry_z_delta, je nachdem ob sigma_S,t^RV unter rv_low oder
    ueber rv_high liegt."""

    def test_high_vol_regime_widens_threshold_and_suppresses_entry(
        self, make_spread_definition, make_bar_context, make_multi_bar_price_window,
    ):
        m = 4
        prices, rv_t, z = _regime_probe_window(make_multi_bar_price_window, spike=1.0, m=m)
        # rv_high knapp UNTER rv_t -> HIGH-Vol-Regime -> Threshold = entry_z + delta = 2.5 > z=2.0
        definition = make_spread_definition(diagnostics={"rv_base": rv_t, "rv_low": 0.0, "rv_high": rv_t * 0.999})
        params = BasketParams(
            entry_z=2.0, entry_z_delta=0.5, stop_z=None, bollinger_window_candles=m + 1,
            dynamic_threshold_enabled=True, vol_regime_lookback_candles=m,
        )
        ctx = make_bar_context(params=params, prices=prices)
        signals = PairwiseZScoreStrategy().generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert signals == []  # waere OHNE Regime-Anpassung ein gueltiges Signal (|z|=2.0 >= entry_z=2.0)

    def test_low_vol_regime_tightens_threshold_and_enables_entry(
        self, make_spread_definition, make_bar_context, make_multi_bar_price_window,
    ):
        m = 4
        prices, rv_t, z = _regime_probe_window(make_multi_bar_price_window, spike=1.0, m=m)
        # rv_low knapp UEBER rv_t -> LOW-Vol-Regime -> Threshold = entry_z - delta = 2.0 == z
        definition = make_spread_definition(diagnostics={"rv_base": rv_t, "rv_low": rv_t * 1.001, "rv_high": 999.0})
        params = BasketParams(
            entry_z=2.5, entry_z_delta=0.5, stop_z=None, bollinger_window_candles=m + 1,
            dynamic_threshold_enabled=True, vol_regime_lookback_candles=m,
        )
        ctx = make_bar_context(params=params, prices=prices)
        signals = PairwiseZScoreStrategy().generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert len(signals) == 1  # waere mit dem statischen entry_z=2.5 kein Signal (|z|=2.0 < 2.5)
        assert signals[0].meta["entry_threshold_z"] == pytest.approx(2.0)

    def test_missing_regime_diagnostics_falls_back_to_static_entry_z(
        self, make_spread_definition, make_bar_context, make_multi_bar_price_window,
    ):
        """Kein rv_base/rv_low/rv_high in diagnostics (z.B. Warm-up, regime_window
        nicht gesetzt) -> unveraendertes statisches Verhalten, kein Crash."""
        m = 4
        prices, _, _ = _regime_probe_window(make_multi_bar_price_window, spike=1.0, m=m)
        definition = make_spread_definition()  # diagnostics={}
        params = BasketParams(
            entry_z=2.0, stop_z=None, bollinger_window_candles=m + 1,
            dynamic_threshold_enabled=True, vol_regime_lookback_candles=m,
        )
        ctx = make_bar_context(params=params, prices=prices)
        signals = PairwiseZScoreStrategy().generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert len(signals) == 1
        assert signals[0].meta["entry_threshold_z"] == pytest.approx(2.0)


class TestDynamicSizing:
    """BasketParams.dynamic_sizing_enabled (PDF E): sizing_multiplier =
    min(rv_base/rv_t, sizing_multiplier_cap), landet in EntrySignal.meta."""

    def test_sizing_multiplier_matches_rv_base_over_rv_t(
        self, make_spread_definition, make_bar_context, make_multi_bar_price_window,
    ):
        m = 4
        prices, rv_t, _ = _regime_probe_window(make_multi_bar_price_window, spike=1.0, m=m)
        rv_base = rv_t * 1.5
        definition = make_spread_definition(diagnostics={"rv_base": rv_base, "rv_low": 0.0, "rv_high": 999.0})
        params = BasketParams(
            entry_z=2.0, stop_z=None, bollinger_window_candles=m + 1,
            dynamic_sizing_enabled=True, vol_regime_lookback_candles=m, sizing_multiplier_cap=10.0,
        )
        ctx = make_bar_context(params=params, prices=prices)
        signals = PairwiseZScoreStrategy().generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert len(signals) == 1
        assert signals[0].meta["sizing_multiplier"] == pytest.approx(rv_base / rv_t)

    def test_sizing_multiplier_is_capped(
        self, make_spread_definition, make_bar_context, make_multi_bar_price_window,
    ):
        m = 4
        prices, rv_t, _ = _regime_probe_window(make_multi_bar_price_window, spike=1.0, m=m)
        rv_base = rv_t * 100  # ohne Cap ein absurd hoher Multiplikator
        definition = make_spread_definition(diagnostics={"rv_base": rv_base, "rv_low": 0.0, "rv_high": 999.0})
        params = BasketParams(
            entry_z=2.0, stop_z=None, bollinger_window_candles=m + 1,
            dynamic_sizing_enabled=True, vol_regime_lookback_candles=m, sizing_multiplier_cap=3.0,
        )
        ctx = make_bar_context(params=params, prices=prices)
        signals = PairwiseZScoreStrategy().generate_entries(ctx=ctx, state=_state([definition]), params=params)
        assert len(signals) == 1
        assert signals[0].meta["sizing_multiplier"] == pytest.approx(3.0)


class TestStrategyRegistry:
    def test_build_strategy_pairwise(self):
        assert isinstance(build_strategy("pairwise_zscore"), PairwiseZScoreStrategy)

    def test_build_strategy_cluster(self):
        assert isinstance(build_strategy("cluster_johansen"), ClusterSpreadStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            build_strategy("does_not_exist")


class _FakeSnapshot:
    def __init__(self, symbols, asset_ids):
        self.symbols = tuple(symbols)
        self.asset_ids = tuple(asset_ids)


class TestPairwiseBuildDefinitions:
    def test_finds_the_cointegrated_pair(self, synthetic_panel):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(
            pair_prefilter_top_n=6, max_candidate_pairs=2,
            # sehr niedrig, da das synthetische AAA/BBB-Paar wegen des dominanten
            # gemeinsamen Faktors extrem schnell mean-revertiert (< 1 Kerze)
            min_half_life_candles=0.1, max_half_life_candles=10_000,
            require_adf=True, adf_alpha=0.1,
        )
        window = synthetic_panel["close"].iloc[:800]
        snapshot = _FakeSnapshot(window.columns, [1, 2, 3, 4])

        definitions, rejected = strategy.build_definitions(snapshot=snapshot, window=window, params=params)

        assert definitions, f"keine Definitionen gefunden, rejected={rejected}"
        found = {frozenset(d.symbols) for d in definitions}
        assert frozenset({"AAAUSDT", "BBBUSDT"}) in found

    def test_regime_window_adds_rv_diagnostics(self, synthetic_panel):
        """Wird ein regime_window uebergeben (dynamic_threshold_enabled/
        dynamic_sizing_enabled), muessen rv_base/rv_low/rv_high in den
        Diagnostiken jeder Definition landen -- siehe
        src.signals.spread.compute_realized_vol_regime."""
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(
            pair_prefilter_top_n=6, max_candidate_pairs=2,
            min_half_life_candles=0.1, max_half_life_candles=10_000,
            require_adf=True, adf_alpha=0.1, vol_regime_lookback_candles=20,
        )
        window = synthetic_panel["close"].iloc[:800]
        regime_window = synthetic_panel["close"].iloc[:1600]  # deckt N+m komfortabel
        snapshot = _FakeSnapshot(window.columns, [1, 2, 3, 4])

        definitions, rejected = strategy.build_definitions(
            snapshot=snapshot, window=window, params=params, regime_window=regime_window,
        )

        assert definitions, f"keine Definitionen gefunden, rejected={rejected}"
        for d in definitions:
            assert d.diagnostics["rv_base"] > 0
            assert d.diagnostics["rv_low"] <= d.diagnostics["rv_base"] <= d.diagnostics["rv_high"]

    def test_definition_carries_clique_diagnostics_after_on_snapshot(self, synthetic_panel, make_bar_context, make_price_window):
        strategy = PairwiseZScoreStrategy()
        params = BasketParams(
            formation_window_candles=800, pair_prefilter_top_n=6, max_candidate_pairs=2,
            min_half_life_candles=0.1, max_half_life_candles=10_000, require_adf=True, adf_alpha=0.1,
        )
        window = synthetic_panel["close"].iloc[:800]
        snapshot = _FakeSnapshot(window.columns, [1, 2, 3, 4])
        snapshot.membership_key = "mk"
        snapshot.persistence = 0.77
        snapshot.size = 4
        snapshot.lookback_days = 7

        from src.backtest.types import PriceWindow
        arrays = {"close": window.to_numpy()}
        symbol_to_col = {s: i for i, s in enumerate(window.columns)}
        prices = PriceWindow(arrays, symbol_to_col, window.index, len(window) - 1)
        ctx = make_bar_context(params=params, prices=prices, timestamp=window.index[-1])

        state = strategy.on_snapshot(snapshot=snapshot, ctx=ctx, params=params)
        assert state.definitions, "Formation sollte mindestens eine Definition liefern"
        assert state.definitions[0].diagnostics["clique_persistence"] == 0.77
        assert state.definitions[0].diagnostics["clique_size"] == 4


class TestClusterBuildDefinitions:
    def test_runs_without_crashing(self, synthetic_panel):
        strategy = ClusterSpreadStrategy()
        params = BasketParams(
            min_johansen_relations=1, min_half_life_candles=1, max_half_life_candles=10_000,
            require_adf=True, adf_alpha=0.1,
        )
        window = synthetic_panel["close"].iloc[:800]
        snapshot = _FakeSnapshot(window.columns, [1, 2, 3, 4])

        definitions, rejected = strategy.build_definitions(snapshot=snapshot, window=window, params=params)

        if definitions:
            d = definitions[0]
            assert d.kind == "cluster"
            assert len(d.weights) == len(d.symbols) == len(d.asset_ids)
            assert sum(abs(w) for w in d.weights) == pytest.approx(1.0)
        else:
            assert rejected  # muss einen Ablehnungsgrund dokumentieren, nicht stillschweigend leer sein
