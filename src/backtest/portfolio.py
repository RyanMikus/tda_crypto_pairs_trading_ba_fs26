"""Positionsverwaltung: Order-Warteschlange (Look-Ahead-Guard G3: Fill immer
auf der Bar NACH der Entscheidung, nie auf derselben), Kosten-/Slippage-
Modell, Mark-to-Market, Trade-Log.

Equity-Buchhaltung (siehe Plan, Abschnitt "Kapitalallokation"):
    equity_t = cash + sum_ueber_offene_Positionen(sum_legs(quantity_k * price_k,t))
Bei Entry wird NICHT der PnL realisiert, sondern nur das Notional "gekauft"
(cash -= sum(quantity_k * entry_price_k) + entry_cost) -- der Gewinn/Verlust
entsteht rein aus der Neubewertung (Mark-to-Market) der offenen Legs. Bei
Exit wird das Notional wieder "verkauft" (cash += sum(quantity_k * exit_price_k)
- exit_cost). Damit ist equity_t automatisch konsistent mit der obigen Formel,
ohne PnL doppelt zu zaehlen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtest.types import EntrySignal, Leg, Position

if TYPE_CHECKING:
    from src.backtest.cliques import CliqueSnapshot
    from src.backtest.config import BasketParams

FORCED_EXIT_REASONS = {"data_gap", "end_of_backtest"}


def dollar_neutral_fractions(weights: np.ndarray) -> np.ndarray:
    """Skaliert |weights| so, dass Long-Seite und Short-Seite je 50% des
    Bruttonotionals bekommen (dollar-neutral). Fällt auf einfache
    |w|/sum(|w|)-Gewichtung zurück, falls eine Seite leer ist (z.B. alle
    Gewichte gleiches Vorzeichen -- kann bei entarteten Spreads passieren)."""
    pos_sum = weights[weights > 0].sum()
    neg_sum = -weights[weights < 0].sum()
    if pos_sum <= 0 or neg_sum <= 0:
        return np.abs(weights) / np.abs(weights).sum()
    scale_pos = 0.5 / pos_sum
    scale_neg = 0.5 / neg_sum
    return np.where(weights > 0, weights * scale_pos, -weights * scale_neg)


def transaction_cost(notional: float, *, params: "BasketParams", bar: dict) -> float:
    """Fee (bps) + Slippage. slippage_model="high_low": Proxy für den
    effektiven Spread aus der Fill-Kerze -- skaliert automatisch mit der
    tatsächlichen Illiquidität (relevant für Basket 4), statt eine geratene
    feste bps-Zahl zu nehmen."""
    fee = notional * params.fee_bps / 10_000
    if params.slippage_model == "bps":
        slip = notional * params.slippage_bps / 10_000
    else:
        rel_range = (bar["high"] - bar["low"]) / bar["close"] if bar.get("close") else 0.0
        slip = notional * params.slippage_high_low_factor * rel_range
    return fee + slip


class Portfolio:
    def __init__(
        self,
        *,
        initial_capital: float,
        arrays: dict[str, np.ndarray],
        symbol_to_col: dict[str, int],
        fill_price_field: str = "open",
        candle_minutes: int = 5,
    ) -> None:
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.open_positions: dict[str, Position] = {}
        self.positions_by_basket: dict[str, list[str]] = {}
        self.trade_log: list[dict] = []
        self.realized_pnl_cum = 0.0
        self.cost_cum = 0.0

        self._arrays = arrays
        self._symbol_to_col = symbol_to_col
        self._fill_price_field = fill_price_field
        self._candle_minutes = candle_minutes
        self._seq = 0
        self._pending_opens: list[dict] = []
        self._pending_closes: list[dict] = []

    # ------------------------------------------------------------------
    # Kapazitäts-/Exposure-Abfragen
    # ------------------------------------------------------------------

    def free_slots(self, basket: str, params: "BasketParams", max_total_positions: int) -> int:
        basket_open = len(self.positions_by_basket.get(basket, []))
        total_open = len(self.open_positions)
        return max(0, min(params.max_concurrent_positions - basket_open, max_total_positions - total_open))

    @property
    def current_gross_notional(self) -> float:
        return sum(pos.gross_notional for pos in self.open_positions.values())

    def asset_in_use_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for pos in self.open_positions.values():
            for asset_id in pos.asset_ids:
                counts[asset_id] = counts.get(asset_id, 0) + 1
        return counts

    def can_open(self, *, estimated_gross_notional: float, equity: float, max_gross_exposure: float, max_total_positions: int) -> bool:
        if len(self.open_positions) >= max_total_positions:
            return False
        if self.current_gross_notional + estimated_gross_notional > max_gross_exposure * equity:
            return False
        return True

    # ------------------------------------------------------------------
    # Preis-Zugriff
    # ------------------------------------------------------------------

    def _price(self, field: str, symbol: str, bar_index: int) -> float:
        col = self._symbol_to_col[symbol]
        return float(self._arrays[field][bar_index, col])

    def _fill_price(self, symbol: str, bar_index: int) -> float:
        return self._price(self._fill_price_field, symbol, bar_index)

    def _mark_price(self, symbol: str, bar_index: int) -> float:
        return self._price("close", symbol, bar_index)

    def _bar_ohlc(self, symbol: str, bar_index: int) -> dict:
        return {f: self._price(f, symbol, bar_index) for f in ("open", "high", "low", "close") if f in self._arrays}

    # ------------------------------------------------------------------
    # Equity / Mark-to-Market
    # ------------------------------------------------------------------

    def equity(self, bar_index: int) -> float:
        """Reine Lesefunktion, keine Nebenwirkungen -- für die Kapital-
        allokation bei Entry (siehe _execute_open) und für Guard-Checks."""
        total = self.cash
        for position in self.open_positions.values():
            for leg in position.legs:
                price = self._mark_price(leg.symbol, bar_index)
                if not np.isfinite(price):
                    price = leg.last_price
                total += leg.quantity * price
        return total

    def mark_to_market(self, bar_index: int) -> float:
        """Wird GENAU EINMAL pro Bar aufgerufen (Engine-Schritt 2): erhöht
        bars_held, aktualisiert missing_price_bars/last_price/unrealized_pnl."""
        equity = self.cash
        for position in self.open_positions.values():
            position.bars_held += 1
            any_missing = False
            leg_value = 0.0
            entry_value = 0.0
            for leg in position.legs:
                price = self._mark_price(leg.symbol, bar_index)
                if not np.isfinite(price):
                    any_missing = True
                    price = leg.last_price
                else:
                    leg.last_price = price
                leg_value += leg.quantity * price
                entry_value += leg.quantity * leg.entry_price
            position.missing_price_bars = position.missing_price_bars + 1 if any_missing else 0
            position.unrealized_pnl = leg_value - entry_value
            position.mfe = max(position.mfe, position.unrealized_pnl)
            position.mae = min(position.mae, position.unrealized_pnl)
            equity += leg_value
        return equity

    # ------------------------------------------------------------------
    # Order-Warteschlange
    # ------------------------------------------------------------------

    def queue_open(
        self, *, signal: EntrySignal, basket: str, snapshot: "CliqueSnapshot", params: "BasketParams",
        strategy_name: str, fill_at_bar_index: int,
    ) -> None:
        self._pending_opens.append({
            "signal": signal, "basket": basket, "snapshot": snapshot,
            "params": params, "strategy_name": strategy_name, "fill_at_bar_index": fill_at_bar_index,
        })

    def queue_close(self, *, position: Position, reason: str, fill_at_bar_index: int) -> None:
        self._pending_closes.append({
            "position_id": position.position_id, "reason": reason, "fill_at_bar_index": fill_at_bar_index,
        })

    def execute_pending(self, *, bar_index: int, timestamp: pd.Timestamp) -> None:
        """Führt alle Orders aus, deren fill_at_bar_index <= bar_index ist
        (Look-Ahead-Guard G3: Fill immer execution_lag_candles NACH der
        Entscheidung, nie auf derselben Bar). Closes zuerst, damit Kapital/
        Asset-Slots noch in derselben Bar für neue Opens frei werden."""
        ready_closes = [c for c in self._pending_closes if c["fill_at_bar_index"] <= bar_index]
        self._pending_closes = [c for c in self._pending_closes if c["fill_at_bar_index"] > bar_index]
        for item in ready_closes:
            self._execute_close(item["position_id"], item["reason"], bar_index, timestamp)

        ready_opens = [o for o in self._pending_opens if o["fill_at_bar_index"] <= bar_index]
        self._pending_opens = [o for o in self._pending_opens if o["fill_at_bar_index"] > bar_index]
        for item in ready_opens:
            self._execute_open(
                item["signal"], item["basket"], item["snapshot"], item["params"],
                item["strategy_name"], bar_index, timestamp,
            )

    def _execute_open(
        self, signal: EntrySignal, basket: str, snapshot: "CliqueSnapshot", params: "BasketParams",
        strategy_name: str, bar_index: int, ts: pd.Timestamp,
    ) -> None:
        definition = signal.spread
        fill_prices = {}
        for sym in definition.symbols:
            price = self._fill_price(sym, bar_index)
            if not np.isfinite(price):
                return  # nicht fuellbar -- Signal verfaellt, keine Warteschlange ueber die Bar hinaus
            fill_prices[sym] = price

        equity = self.equity(bar_index)
        gross_notional = params.capital_per_position * equity
        weights = np.array(definition.weights, dtype=float)
        fractions = dollar_neutral_fractions(weights) if params.dollar_neutralize else np.abs(weights) / np.abs(weights).sum()

        self._seq += 1
        position_id = f"{basket.replace(' ', '')}-{ts:%Y%m%dT%H%M%S}-{self._seq}"

        legs: list[Leg] = []
        entry_cost_total = 0.0
        for k, (asset_id, sym) in enumerate(zip(definition.asset_ids, definition.symbols)):
            leg_notional = float(fractions[k]) * gross_notional
            signed_fraction = signal.direction * np.sign(weights[k])
            quantity = signed_fraction * leg_notional / fill_prices[sym]
            legs.append(Leg(
                asset_id=asset_id, symbol=sym, weight=float(weights[k]), quantity=float(quantity),
                entry_price=fill_prices[sym], entry_notional=leg_notional, last_price=fill_prices[sym],
            ))
            entry_cost_total += transaction_cost(leg_notional, params=params, bar=self._bar_ohlc(sym, bar_index))

        position = Position(
            position_id=position_id, strategy=strategy_name, basket=basket, spread=definition, params=params,
            membership_key=snapshot.membership_key, discovery_timestamp=snapshot.discovery_timestamp,
            direction=signal.direction, legs=legs, entry_timestamp=ts,
            entry_signal_value=signal.signal_value, entry_spread_value=signal.spread_value,
            gross_notional=gross_notional, entry_equity=equity, entry_cost=entry_cost_total,
        )

        self.cash -= sum(leg.quantity * leg.entry_price for leg in legs) + entry_cost_total
        self.cost_cum += entry_cost_total

        self.open_positions[position_id] = position
        self.positions_by_basket.setdefault(basket, []).append(position_id)

    def _execute_close(self, position_id: str, reason: str, bar_index: int, ts: pd.Timestamp) -> None:
        position = self.open_positions.pop(position_id, None)
        if position is None:
            return
        self.positions_by_basket[position.basket].remove(position_id)

        exit_cost_total = 0.0
        gross_pnl = 0.0
        exit_log_prices = np.empty(len(position.legs))
        for i, leg in enumerate(position.legs):
            price = self._fill_price(leg.symbol, bar_index)
            if not np.isfinite(price):
                price = leg.last_price  # letzter bekannter Preis (z.B. Delisting/Datenluecke)
            leg.exit_price = price
            gross_pnl += leg.quantity * (price - leg.entry_price)
            exit_cost_total += transaction_cost(abs(leg.quantity * price), params=position.params, bar=self._bar_ohlc(leg.symbol, bar_index))
            exit_log_prices[i] = np.log(price)

        self.cash += sum(leg.quantity * leg.exit_price for leg in position.legs) - exit_cost_total
        self.cost_cum += exit_cost_total

        total_cost = position.entry_cost + exit_cost_total
        net_pnl = gross_pnl - total_cost
        self.realized_pnl_cum += net_pnl

        exit_spread_value = position.spread.spread_value(exit_log_prices)
        exit_z = position.spread.zscore(exit_log_prices)
        diag = position.spread.diagnostics

        self.trade_log.append({
            "position_id": position.position_id,
            "strategy": position.strategy,
            "basket": position.basket,
            "spread_kind": position.spread.kind,
            "membership_key": position.membership_key,
            "discovery_timestamp": position.discovery_timestamp,
            "asset_ids": list(position.asset_ids),
            "symbols": list(position.symbols),
            "weights": list(position.spread.weights),
            "n_legs": len(position.legs),
            "direction": position.direction,
            "entry_timestamp": position.entry_timestamp,
            "exit_timestamp": ts,
            "bars_held": position.bars_held,
            "holding_hours": position.bars_held * self._candle_minutes / 60.0,
            "entry_spread": position.entry_spread_value,
            "exit_spread": exit_spread_value,
            "entry_z": position.entry_signal_value,
            "exit_z": exit_z,
            "gross_notional": position.gross_notional,
            "entry_equity": position.entry_equity,
            "gross_pnl": gross_pnl,
            "entry_cost": position.entry_cost,
            "exit_cost": exit_cost_total,
            "total_cost": total_cost,
            "net_pnl": net_pnl,
            "return_on_notional": net_pnl / position.gross_notional if position.gross_notional else np.nan,
            "return_on_equity": net_pnl / position.entry_equity if position.entry_equity else np.nan,
            "mfe": position.mfe,
            "mae": position.mae,
            "exit_reason": reason,
            "is_forced": reason in FORCED_EXIT_REASONS,
            "adf_pvalue": diag.get("adf_pvalue", np.nan),
            "kpss_pvalue": diag.get("kpss_pvalue", np.nan),
            "hurst": diag.get("hurst", np.nan),
            "half_life_periods": diag.get("half_life_periods", np.nan),
            "n_coint_relations": diag.get("n_coint_relations", np.nan),
            "persistence": diag.get("clique_persistence", np.nan),
            "clique_size": diag.get("clique_size", np.nan),
        })

    def force_close_all(self, *, bar_index: int, ts: pd.Timestamp, reason: str = "end_of_backtest") -> None:
        """Schliesst alle offenen Positionen SOFORT auf dieser Bar (kein
        Ausfuehrungs-Lag) -- gedacht fuer das Ende des Backtests."""
        for position_id in list(self.open_positions):
            self._execute_close(position_id, reason, bar_index, ts)
