"""BacktestEngine: bar-für-bar Simulation (5-Min-Kerzen-Loop).

Reihenfolge innerhalb einer Bar (siehe Plan, Abschnitt "Engine-Loop"):
  1. Fills aus vorherigen Bars ausführen (Look-Ahead-Guard G3)
  2. Mark-to-Market
  3. Clique-Snapshots pro Basket vorrücken
  4. Exits prüfen (Engine-weite Regeln zuerst, dann strategiespezifisch)
  5. Entries generieren (nur bei freier Kapazität)
  6. Equity-Zeile schreiben
"""

from __future__ import annotations

import pandas as pd

from src.backtest.cliques import CliqueTimeline, TimelineCursor
from src.backtest.config import BacktestConfig
from src.backtest.exit_rules import DEFAULT_EXIT_CHAIN, ExitRule, evaluate_exit_chain
from src.backtest.portfolio import Portfolio
from src.backtest.strategy import StrategyState, build_strategy
from src.backtest.types import BacktestResult, BarContext, PriceWindow

_DEFAULT_MAX_STATE_AGE = pd.Timedelta(days=7)


class BacktestEngine:
    def __init__(
        self,
        *,
        config: BacktestConfig,
        panel: dict[str, pd.DataFrame],
        timeline: CliqueTimeline,
        exit_chain: list[ExitRule] | None = None,
    ) -> None:
        self.config = config
        self.timeline = timeline
        self.exit_chain = exit_chain if exit_chain is not None else DEFAULT_EXIT_CHAIN
        self.strategy = build_strategy(config.strategy_name, **config.strategy_kwargs)

        self._timestamps = panel["close"].index
        self._symbols = list(panel["close"].columns)
        self._symbol_to_col = {s: i for i, s in enumerate(self._symbols)}
        self._arrays = {field: df.to_numpy(dtype=float) for field, df in panel.items()}

        self.portfolio = Portfolio(
            initial_capital=config.initial_capital,
            arrays=self._arrays,
            symbol_to_col=self._symbol_to_col,
            fill_price_field=config.fill_price_field,
            candle_minutes=config.candle_minutes,
        )

        self._state_cache: dict[tuple[str, str], StrategyState] = {}
        self.warnings: list[str] = []

    # ------------------------------------------------------------------
    # Zeit-/Warm-up-Hilfsfunktionen
    # ------------------------------------------------------------------

    def _timestamp_index(self, ts) -> int:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        pos = self._timestamps.searchsorted(ts)
        if pos >= len(self._timestamps) or self._timestamps[pos] != ts:
            raise ValueError(
                f"Timestamp {ts} liegt nicht exakt auf dem geladenen 5-Min-Kerzenraster "
                f"({self._timestamps[0]} … {self._timestamps[-1]})."
            )
        return int(pos)

    def _resolved_availability_lag(self) -> pd.Timedelta:
        lag = self.config.clique_availability_lag
        if lag != "auto":
            return lag
        interval = self.timeline.infer_interval()
        return interval if interval < pd.Timedelta(days=1) else pd.Timedelta(days=1)

    def _check_warmup(self, start_idx: int) -> None:
        for basket in self.config.baskets:
            params = self.config.params_for(basket)
            required = self.strategy.required_history_candles(params)
            if start_idx - required + 1 < 0:
                raise ValueError(
                    f"Warm-up-Guard (G7): {basket} braucht {required} Kerzen Historie vor "
                    f"config.start, aber nur {start_idx + 1} sind im geladenen Panel vorhanden. "
                    f"Panel früher starten lassen oder formation_window_candles verkleinern."
                )

    def describe_guards(self) -> str:
        lag = self._resolved_availability_lag()
        interval = self.timeline.infer_interval()
        required = {b: self.strategy.required_history_candles(self.config.params_for(b)) for b in self.config.baskets}
        lines = [
            f"G1 Clique-Verfügbarkeitslag: {lag} (Discovery-Intervall: {interval})",
            "G2 PriceWindow-Cutoff: aktiv (Strategien können strukturell nicht über bar_index hinaus lesen)",
            f"G3 Ausführungs-Lag: {self.config.execution_lag_candles} Kerze(n), Fill-Preis-Feld: {self.config.fill_price_field}",
            "G4 Eingefrorene Schätzer: SpreadDefinition wird bei Formation fixiert, kein Re-Fit während offener Position",
            "G5 Basket-Zuordnung: aus dem Clique-Cache übernommen, nicht zur Bar-Zeit neu abgeleitet",
            f"G6 Calibration-Zeitraum: {self.config.calibration_range}  |  Backtest-Zeitraum: {self.config.backtest_range}",
            f"G7 Warm-up (Kerzen vor config.start, pro Basket): {required}",
            f"G8 Snapshot-Ablauf (max_snapshot_age): {self.config.max_snapshot_age}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formations-Caching
    # ------------------------------------------------------------------

    def _get_or_fit_state(self, basket, snapshot, ctx, params) -> StrategyState | None:
        key = (basket, snapshot.membership_key)
        cached = self._state_cache.get(key)
        if cached is not None and (ctx.timestamp - cached.fitted_at) <= _DEFAULT_MAX_STATE_AGE:
            return cached
        try:
            state = self.strategy.on_snapshot(snapshot=snapshot, ctx=ctx, params=params)
        except Exception as exc:  # Formation ist datengetrieben (OLS/Johansen) -- kann an Randfällen scheitern
            self.warnings.append(f"{ctx.timestamp} {basket}: on_snapshot fehlgeschlagen ({exc})")
            return None
        self._state_cache[key] = state
        return state

    def _equity_row(self, ts, equity: float) -> dict:
        row = {
            "timestamp": ts,
            "equity": equity,
            "equity_normalized": equity / self.config.initial_capital,
            "cash": self.portfolio.cash,
            "realized_pnl_cum": self.portfolio.realized_pnl_cum,
            "cost_cum": self.portfolio.cost_cum,
            "gross_exposure": self.portfolio.current_gross_notional,
            "n_open_positions": len(self.portfolio.open_positions),
        }
        for basket in self.config.baskets:
            row[f"n_open_{basket.replace(' ', '_')}"] = len(self.portfolio.positions_by_basket.get(basket, []))
        return row

    # ------------------------------------------------------------------
    # Haupt-Loop
    # ------------------------------------------------------------------

    def run(self, *, progress: bool = False) -> BacktestResult:
        start_idx = self._timestamp_index(self.config.start)
        end_idx = self._timestamp_index(self.config.end)
        self._check_warmup(start_idx)

        cursors = {b: TimelineCursor(self.timeline, b) for b in self.config.baskets}
        equity_rows: list[dict] = []

        iterator = range(start_idx, end_idx + 1)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=end_idx - start_idx + 1)

        for i in iterator:
            ts = self._timestamps[i]

            # 1) Fills aus vorherigen Bars
            self.portfolio.execute_pending(bar_index=i, timestamp=ts)

            # 2) Mark-to-Market
            equity = self.portfolio.mark_to_market(i)

            # 3) Snapshots vorrücken
            active_snapshot = {b: cursors[b].advance(ts)[0] for b in self.config.baskets}

            prices = PriceWindow(self._arrays, self._symbol_to_col, self._timestamps, i)
            open_positions = tuple(self.portfolio.open_positions.values())
            fill_at = i + self.config.execution_lag_candles

            # 4) Exits
            for position in open_positions:
                params = self.config.params_for(position.basket)
                ctx = BarContext(
                    timestamp=ts, bar_index=i, basket=position.basket, params=params,
                    snapshot=active_snapshot[position.basket], prices=prices,
                    open_positions=open_positions, equity=equity, free_slots=0,
                )
                decision = evaluate_exit_chain(position, ctx, self.exit_chain)
                if not decision.should_exit:
                    decision = self.strategy.check_exit(position=position, ctx=ctx, params=params)
                if decision.should_exit:
                    self.portfolio.queue_close(position=position, reason=decision.reason, fill_at_bar_index=fill_at)

            # 5) Entries
            for basket in self.config.baskets:
                params = self.config.params_for(basket)
                free = self.portfolio.free_slots(basket, params, self.config.max_total_positions)
                snapshot = active_snapshot[basket]
                if free == 0 or snapshot is None:
                    continue

                ctx = BarContext(
                    timestamp=ts, bar_index=i, basket=basket, params=params, snapshot=snapshot,
                    prices=prices, open_positions=open_positions, equity=equity, free_slots=free,
                )
                state = self._get_or_fit_state(basket, snapshot, ctx, params)
                if state is None or not state.definitions:
                    continue

                signals = self.strategy.generate_entries(ctx=ctx, state=state, params=params)
                signals.sort(key=lambda s: -s.priority)
                for signal in signals[:free]:
                    estimated_gross = params.capital_per_position * equity
                    if not self.portfolio.can_open(
                        estimated_gross_notional=estimated_gross, equity=equity,
                        max_gross_exposure=self.config.max_gross_exposure,
                        max_total_positions=self.config.max_total_positions,
                    ):
                        continue
                    self.portfolio.queue_open(
                        signal=signal, basket=basket, snapshot=snapshot, params=params,
                        strategy_name=self.strategy.name, fill_at_bar_index=fill_at,
                    )

            # 6) Equity-Zeile
            equity_rows.append(self._equity_row(ts, equity))

        # Nachlauf: alle offenen Positionen zum letzten close schliessen.
        last_ts = self._timestamps[end_idx]
        self.portfolio.force_close_all(bar_index=end_idx, ts=last_ts, reason="end_of_backtest")
        if equity_rows:
            equity_rows[-1] = self._equity_row(last_ts, self.portfolio.equity(end_idx))

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp") if equity_rows else pd.DataFrame()
        trades = pd.DataFrame(self.portfolio.trade_log)

        meta = {
            "n_bars": end_idx - start_idx + 1,
            "warnings": self.warnings,
            "engine_version": "0.1",
        }

        return BacktestResult(
            run_id="",
            config=self.config,
            equity_curve=equity_curve,
            trades=trades,
            metrics={},
            metrics_by_basket=pd.DataFrame(),
            positions_curve=None,
            meta=meta,
        )
