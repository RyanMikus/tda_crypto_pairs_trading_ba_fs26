import numpy as np
import pandas as pd
import pytest

from src.backtest.config import BasketParams
from src.backtest.portfolio import Portfolio, dollar_neutral_fractions, transaction_cost
from src.backtest.types import EntrySignal, SpreadDefinition


class TestDollarNeutralFractions:
    def test_sums_abs_to_one(self):
        fractions = dollar_neutral_fractions(np.array([1.0, -0.5]))
        assert np.abs(fractions).sum() == pytest.approx(1.0)

    def test_splits_long_and_short_side_evenly(self):
        fractions = dollar_neutral_fractions(np.array([1.0, -1.0]))
        assert fractions[0] == pytest.approx(0.5)
        assert fractions[1] == pytest.approx(0.5)

    def test_uneven_weights_still_split_50_50_gross(self):
        # long-Seite (nur "3.0") bekommt 50% brutto, short-Seite (nur "-1.0") auch 50%
        fractions = dollar_neutral_fractions(np.array([3.0, -1.0]))
        assert fractions[0] == pytest.approx(0.5)
        assert fractions[1] == pytest.approx(0.5)

    def test_falls_back_to_simple_normalization_when_one_side_empty(self):
        fractions = dollar_neutral_fractions(np.array([2.0, 1.0]))  # keine Short-Seite
        assert fractions.sum() == pytest.approx(1.0)
        assert fractions[0] == pytest.approx(2 / 3)
        assert fractions[1] == pytest.approx(1 / 3)


class TestTransactionCost:
    def test_bps_model(self):
        params = BasketParams(fee_bps=10.0, slippage_model="bps", slippage_bps=2.0)
        cost = transaction_cost(10_000.0, params=params, bar={})
        assert cost == pytest.approx(10_000 * (10 + 2) / 10_000)

    def test_high_low_model_scales_with_bar_range(self):
        params = BasketParams(fee_bps=10.0, slippage_model="high_low", slippage_high_low_factor=0.25)
        bar = {"open": 100, "high": 102, "low": 98, "close": 100}
        cost = transaction_cost(10_000.0, params=params, bar=bar)
        fee = 10_000 * 10 / 10_000
        rel_range = (102 - 98) / 100
        slip = 10_000 * 0.25 * rel_range
        assert cost == pytest.approx(fee + slip)


def _spread_def(weights=(1.0, -0.5)):
    return SpreadDefinition(
        kind="pairwise", asset_ids=(1, 2), symbols=("AAAUSDT", "BBBUSDT"), weights=weights,
        mu=0.0, sigma=1.0,
        fitted_at=pd.Timestamp("2025-01-01", tz="UTC"), formation_start=pd.Timestamp("2024-12-25", tz="UTC"),
    )


class _FakeSnapshot:
    membership_key = "mk"
    discovery_timestamp = pd.Timestamp("2025-01-01", tz="UTC")
    death = 1.0
    lookback_days = 1


def _make_portfolio(prices_by_bar, *, fill_price_field="open"):
    """prices_by_bar: list of {"AAAUSDT": p1, "BBBUSDT": p2}, ein dict pro Kerze."""
    symbols = ["AAAUSDT", "BBBUSDT"]
    arr = np.array([[row[s] for s in symbols] for row in prices_by_bar], dtype=float)
    arrays = {"open": arr, "high": arr + 1, "low": arr - 1, "close": arr}
    return Portfolio(
        initial_capital=100_000.0, arrays=arrays,
        symbol_to_col={s: i for i, s in enumerate(symbols)},
        fill_price_field=fill_price_field, candle_minutes=5,
    )


class TestPortfolioOpen:
    def test_open_reduces_cash_by_notional_plus_cost(self):
        portfolio = _make_portfolio([{"AAAUSDT": 100.0, "BBBUSDT": 50.0}] * 2)
        params = BasketParams(capital_per_position=0.1, dollar_neutralize=True, fee_bps=10.0, slippage_bps=0.0)
        signal = EntrySignal(spread=_spread_def(), direction=1, signal_value=3.0, spread_value=0.0, priority=3.0)

        cash_before = portfolio.cash
        portfolio.queue_open(
            signal=signal, basket="Basket 1", snapshot=_FakeSnapshot(), params=params,
            strategy_name="pairwise_zscore", fill_at_bar_index=1,
        )
        portfolio.execute_pending(bar_index=1, timestamp=pd.Timestamp("2025-01-01 00:05", tz="UTC"))

        assert len(portfolio.open_positions) == 1
        position = next(iter(portfolio.open_positions.values()))

        gross_notional = params.capital_per_position * cash_before
        expected_cost = gross_notional * params.fee_bps / 10_000
        assert position.entry_cost == pytest.approx(expected_cost)

        cash_change = portfolio.cash - cash_before
        expected_cash_change = -sum(leg.quantity * leg.entry_price for leg in position.legs) - position.entry_cost
        assert cash_change == pytest.approx(expected_cash_change)

    def test_equity_only_drops_by_cost_not_by_notional(self):
        """Kein PnL beim Öffnen selbst (siehe Modul-Docstring in portfolio.py)
        -- die Equity sinkt nur um die Kosten, nicht um das Notional."""
        portfolio = _make_portfolio([{"AAAUSDT": 100.0, "BBBUSDT": 50.0}] * 3)
        params = BasketParams(capital_per_position=0.1, fee_bps=10.0, slippage_bps=0.0)
        signal = EntrySignal(spread=_spread_def(), direction=1, signal_value=3.0, spread_value=0.0, priority=3.0)

        equity_before = portfolio.equity(0)
        portfolio.queue_open(
            signal=signal, basket="Basket 1", snapshot=_FakeSnapshot(), params=params,
            strategy_name="pairwise_zscore", fill_at_bar_index=1,
        )
        portfolio.execute_pending(bar_index=1, timestamp=pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        equity_after = portfolio.mark_to_market(1)

        position = next(iter(portfolio.open_positions.values()))
        assert equity_after == pytest.approx(equity_before - position.entry_cost)

    def test_no_fill_when_price_missing(self):
        prices = [{"AAAUSDT": 100.0, "BBBUSDT": 50.0}, {"AAAUSDT": np.nan, "BBBUSDT": 50.0}]
        portfolio = _make_portfolio(prices)
        params = BasketParams(capital_per_position=0.1)
        signal = EntrySignal(spread=_spread_def(), direction=1, signal_value=3.0, spread_value=0.0, priority=3.0)

        portfolio.queue_open(
            signal=signal, basket="Basket 1", snapshot=_FakeSnapshot(), params=params,
            strategy_name="pairwise_zscore", fill_at_bar_index=1,
        )
        portfolio.execute_pending(bar_index=1, timestamp=pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        assert len(portfolio.open_positions) == 0  # Signal verfaellt, keine Order-Warteschlange ueber die Bar hinaus


class TestPortfolioClose:
    def test_realizes_pnl_from_price_move(self):
        prices = [
            {"AAAUSDT": 100.0, "BBBUSDT": 50.0},
            {"AAAUSDT": 100.0, "BBBUSDT": 50.0},
            {"AAAUSDT": 110.0, "BBBUSDT": 50.0},  # AAA steigt, BBB unveraendert
        ]
        portfolio = _make_portfolio(prices)
        params = BasketParams(capital_per_position=0.1, fee_bps=0.0, slippage_bps=0.0)
        signal = EntrySignal(spread=_spread_def(), direction=1, signal_value=3.0, spread_value=0.0, priority=3.0)

        portfolio.queue_open(
            signal=signal, basket="Basket 1", snapshot=_FakeSnapshot(), params=params,
            strategy_name="pairwise_zscore", fill_at_bar_index=1,
        )
        portfolio.execute_pending(bar_index=1, timestamp=pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        portfolio.mark_to_market(1)
        position = next(iter(portfolio.open_positions.values()))
        aaa_leg = next(leg for leg in position.legs if leg.symbol == "AAAUSDT")

        portfolio.queue_close(position=position, reason="target", fill_at_bar_index=2)
        portfolio.execute_pending(bar_index=2, timestamp=pd.Timestamp("2025-01-01 00:10", tz="UTC"))

        assert len(portfolio.open_positions) == 0
        assert len(portfolio.trade_log) == 1
        trade = portfolio.trade_log[0]
        expected_gross_pnl = aaa_leg.quantity * (110.0 - 100.0)
        assert trade["gross_pnl"] == pytest.approx(expected_gross_pnl)
        assert trade["net_pnl"] == pytest.approx(expected_gross_pnl)  # kostenfrei in diesem Test
        assert trade["exit_reason"] == "target"
        assert trade["is_forced"] is False

    def test_close_falls_back_to_last_valid_price_when_fill_price_missing(self):
        prices = [
            {"AAAUSDT": 100.0, "BBBUSDT": 50.0},
            {"AAAUSDT": 100.0, "BBBUSDT": 50.0},
            {"AAAUSDT": np.nan, "BBBUSDT": 50.0},  # Fill-Preis fehlt beim Exit
        ]
        portfolio = _make_portfolio(prices)
        params = BasketParams(capital_per_position=0.1, fee_bps=0.0, slippage_bps=0.0)
        signal = EntrySignal(spread=_spread_def(), direction=1, signal_value=3.0, spread_value=0.0, priority=3.0)

        portfolio.queue_open(
            signal=signal, basket="Basket 1", snapshot=_FakeSnapshot(), params=params,
            strategy_name="pairwise_zscore", fill_at_bar_index=1,
        )
        portfolio.execute_pending(bar_index=1, timestamp=pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        portfolio.mark_to_market(1)  # aktualisiert last_price=100.0 fuer AAA
        position = next(iter(portfolio.open_positions.values()))

        portfolio.queue_close(position=position, reason="data_gap", fill_at_bar_index=2)
        portfolio.execute_pending(bar_index=2, timestamp=pd.Timestamp("2025-01-01 00:10", tz="UTC"))

        trade = portfolio.trade_log[0]
        assert trade["net_pnl"] == pytest.approx(0.0)  # kein Preis-Update -> kein PnL


class TestForceCloseAll:
    def test_closes_every_open_position(self):
        portfolio = _make_portfolio([{"AAAUSDT": 100.0, "BBBUSDT": 50.0}] * 3)
        params = BasketParams(capital_per_position=0.1, fee_bps=0.0, slippage_bps=0.0)
        signal = EntrySignal(spread=_spread_def(), direction=1, signal_value=3.0, spread_value=0.0, priority=3.0)

        portfolio.queue_open(
            signal=signal, basket="Basket 1", snapshot=_FakeSnapshot(), params=params,
            strategy_name="pairwise_zscore", fill_at_bar_index=1,
        )
        portfolio.execute_pending(bar_index=1, timestamp=pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        assert len(portfolio.open_positions) == 1

        portfolio.force_close_all(bar_index=1, ts=pd.Timestamp("2025-01-01 00:05", tz="UTC"))
        assert len(portfolio.open_positions) == 0
        assert portfolio.trade_log[-1]["exit_reason"] == "end_of_backtest"
        assert portfolio.trade_log[-1]["is_forced"] is True


class TestCanOpen:
    def _empty_portfolio(self):
        arrays = {f: np.ones((2, 2)) for f in ("open", "high", "low", "close")}
        return Portfolio(initial_capital=100_000.0, arrays=arrays, symbol_to_col={"A": 0, "B": 1}, fill_price_field="open")

    def test_respects_max_total_positions(self):
        portfolio = self._empty_portfolio()
        portfolio.open_positions["fake"] = object()  # simuliert 1 offene Position
        assert not portfolio.can_open(estimated_gross_notional=1.0, equity=100_000.0, max_gross_exposure=1.0, max_total_positions=1)

    def test_respects_max_gross_exposure(self):
        portfolio = self._empty_portfolio()
        assert not portfolio.can_open(estimated_gross_notional=200_000.0, equity=100_000.0, max_gross_exposure=1.0, max_total_positions=10)
        assert portfolio.can_open(estimated_gross_notional=50_000.0, equity=100_000.0, max_gross_exposure=1.0, max_total_positions=10)
