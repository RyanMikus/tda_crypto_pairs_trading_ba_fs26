import numpy as np
import pandas as pd
import pytest

from src.backtest.types import PriceWindow, SpreadDefinition


def _make_price_window(bar_index=10, n_bars=20, symbols=("AAAUSDT", "BBBUSDT")):
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="5min", tz="UTC")
    close = np.arange(n_bars * len(symbols), dtype=float).reshape(n_bars, len(symbols)) + 100.0
    arrays = {"close": close}
    symbol_to_col = {s: i for i, s in enumerate(symbols)}
    return PriceWindow(arrays, symbol_to_col, idx, bar_index), idx, close


class TestPriceWindow:
    def test_last_returns_current_bar_values(self):
        window, idx, close = _make_price_window(bar_index=5)
        last = window.last(["AAAUSDT", "BBBUSDT"])
        assert last["AAAUSDT"] == close[5, 0]
        assert last["BBBUSDT"] == close[5, 1]

    def test_history_includes_bar_index_inclusive(self):
        window, idx, close = _make_price_window(bar_index=10)
        hist = window.history(["AAAUSDT"], n_candles=3)
        assert len(hist) == 3
        assert hist.index[-1] == idx[10]
        np.testing.assert_array_equal(hist["AAAUSDT"].to_numpy(), close[8:11, 0])

    def test_history_cannot_reach_before_start_of_data(self):
        window, _, _ = _make_price_window(bar_index=1)
        with pytest.raises(ValueError):
            window.history(["AAAUSDT"], n_candles=5)  # bräuchte Bars vor Index 0

    def test_history_cannot_see_future_bars(self):
        """Struktureller Look-Ahead-Schutz (Guard G2): history() darf niemals
        Zeilen NACH bar_index zurückgeben, egal wie n_candles gewählt wird."""
        window, idx, close = _make_price_window(bar_index=5, n_bars=20)
        hist = window.history(["AAAUSDT"], n_candles=3)
        assert hist.index.max() <= idx[5]

    def test_unknown_symbol_raises_keyerror(self):
        window, _, _ = _make_price_window(bar_index=5)
        with pytest.raises(KeyError):
            window.last(["NOPEUSDT"])

    def test_log_last_matches_manual_log(self):
        window, _, close = _make_price_window(bar_index=7)
        log_prices = window.log_last(["AAAUSDT", "BBBUSDT"])
        np.testing.assert_allclose(log_prices, np.log(close[7, :2]))


class TestSpreadDefinition:
    def _make(self, mu=0.0, sigma=1.0):
        return SpreadDefinition(
            kind="pairwise", asset_ids=(1, 2), symbols=("AAAUSDT", "BBBUSDT"),
            weights=(1.0, -0.5), mu=mu, sigma=sigma,
            fitted_at=pd.Timestamp("2025-01-01", tz="UTC"),
            formation_start=pd.Timestamp("2024-12-25", tz="UTC"),
        )

    def test_spread_value_is_weighted_sum_of_log_prices(self):
        definition = self._make()
        log_prices = np.array([np.log(100.0), np.log(50.0)])
        expected = 1.0 * log_prices[0] + (-0.5) * log_prices[1]
        assert definition.spread_value(log_prices) == pytest.approx(expected)

    def test_zscore_matches_manual_formula(self):
        definition = self._make(mu=1.0, sigma=2.0)
        log_prices = np.array([np.log(100.0), np.log(50.0)])
        expected = (definition.spread_value(log_prices) - 1.0) / 2.0
        assert definition.zscore(log_prices) == pytest.approx(expected)

    def test_zscore_zero_sigma_returns_zero_not_inf(self):
        definition = self._make(mu=0.0, sigma=0.0)
        log_prices = np.array([1.0, 2.0])
        assert definition.zscore(log_prices) == 0.0

    def test_diagnostics_dict_is_mutable_despite_frozen_dataclass(self):
        """SpreadDefinition ist frozen, aber `diagnostics` (ein dict-Feld) darf
        nach der Konstruktion noch ergaenzt werden -- BaseSpreadZScoreStrategy.
        on_snapshot() nutzt genau das, um Snapshot-Diagnostik nachzutragen."""
        definition = self._make()
        definition.diagnostics["cluster_persistence"] = 0.42
        assert definition.diagnostics["cluster_persistence"] == 0.42
