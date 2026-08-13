"""Equity-/Drawdown-/Trade-Visualisierung für Backtest-Notebooks. Statisches
matplotlib, konsistent mit dem bestehenden Diagnostik-Stil in
notebooks/03_signal_generation/TradeData.ipynb."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Src.backtest import metrics as bt_metrics
from Src.backtest.types import BacktestResult

if TYPE_CHECKING:
    from src.backtest.config import BasketParams


def plot_equity_and_drawdown(
    result: BacktestResult, *, benchmark_close: pd.Series | None = None, show: bool = True,
):
    """Equity-Kurve (normiert auf 1.0) oben, Drawdown-% unten. Benchmark
    (z.B. BTC-Buy&Hold) optional zum direkten visuellen Vergleich."""
    equity = result.equity_curve["equity"]
    equity_norm = equity / equity.iloc[0]
    dd = bt_metrics.attach_drawdown_columns(result.equity_curve)["drawdown_pct"]

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_eq.plot(equity_norm.index, equity_norm.values, color="#2a78d6", linewidth=1.2, label=result.config.label)
    if benchmark_close is not None:
        bench = benchmark_close.reindex(equity.index).ffill()
        bench_norm = bench / bench.iloc[0]
        ax_eq.plot(bench_norm.index, bench_norm.values, color="#999999", linewidth=1.0, linestyle="--", label="Benchmark")
    ax_eq.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax_eq.set_ylabel("Equity (normiert)")
    ax_eq.set_title(f"Equity-Kurve — {result.config.label}")
    ax_eq.legend(frameon=False)
    ax_eq.grid(True, alpha=0.25)

    ax_dd.fill_between(dd.index, dd.values * 100, 0, color="#e34948", alpha=0.6)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    if show:
        plt.show()
    return fig, (ax_eq, ax_dd)


def plot_exit_reason_breakdown(trades: pd.DataFrame, *, show: bool = True):
    """Balken: Anzahl Trades je Exit-Grund, gefärbt nach Summe des net_pnl
    (rot = Verlustquelle, grün = Gewinnquelle) -- zeigt z.B. ob zu viele
    Trades im Timeout statt am Target enden (Half-Life-Filter falsch kalibriert)."""
    if trades.empty:
        print("Keine Trades -- nichts zu plotten.")
        return None, None

    grouped = trades.groupby("exit_reason")["net_pnl"].agg(["count", "sum"]).sort_values("count", ascending=False)
    colors = ["#2a9d4f" if v >= 0 else "#e34948" for v in grouped["sum"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index, grouped["count"], color=colors)
    for i, (reason, row) in enumerate(grouped.iterrows()):
        ax.text(i, row["count"], f"{row['sum']:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Anzahl Trades")
    ax.set_title("Exit-Gründe (Zahl = Summe net_pnl)")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_holding_period_histogram(trades: pd.DataFrame, *, bins: int = 30, show: bool = True):
    if trades.empty:
        print("Keine Trades -- nichts zu plotten.")
        return None, None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades["holding_hours"], bins=bins, color="#2a78d6", alpha=0.8)
    ax.set_xlabel("Haltedauer (Stunden)")
    ax.set_ylabel("Anzahl Trades")
    ax.set_title("Haltedauer-Verteilung")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_pnl_distribution(trades: pd.DataFrame, *, bins: int = 30, show: bool = True):
    if trades.empty:
        print("Keine Trades -- nichts zu plotten.")
        return None, None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trades["net_pnl"], bins=bins, color="#2a78d6", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Netto-PnL pro Trade ($)")
    ax.set_ylabel("Anzahl Trades")
    ax.set_title("PnL-Verteilung")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_basket_metric_comparison(
    metrics_by_basket: pd.DataFrame, *, metric: str = "expectancy", baseline: float = 0.0, show: bool = True,
):
    """Balken: <metric> je Basket (Index von bt_metrics.metrics_by_basket) --
    zeigt auf einen Blick, wie unterschiedlich einzelne Baskets performen
    (z.B. um sichtbar zu machen, dass ein globaler Parametersatz einen
    einzelnen Basket abwürgt, siehe Diagnose vor der Walk-Forward-
    Optimierung pro Basket, Notebook Abschnitt 11). baseline trennt
    gut/schlecht eingefärbte Balken -- 0.0 für PnL-artige Metriken
    (expectancy, total_return), 1.0 für profit_factor."""
    if metrics_by_basket.empty or metric not in metrics_by_basket.columns:
        print(f"Keine Basket-Metriken oder Spalte {metric!r} nicht vorhanden -- nichts zu plotten.")
        return None, None

    values = metrics_by_basket[metric].replace([np.inf, -np.inf], np.nan).dropna()
    colors = ["#2a9d4f" if v >= baseline else "#e34948" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(values.index.astype(str), values, color=colors)
    ax.axhline(baseline, color="black", linewidth=0.6, linestyle=":")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom" if v >= baseline else "top", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} nach Basket")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_pnl_vs_diagnostic(trades: pd.DataFrame, column: str, *, show: bool = True):
    """Streudiagramm net_pnl vs. einer Formations-Diagnostik-Spalte (z.B.
    half_life_periods, persistence, clique_size) -- der direkte Brückenschlag
    zurück zur TDA-Fragestellung: waren Trades aus persistenteren/schnelleren
    Cliquen tatsächlich profitabler?"""
    if trades.empty or column not in trades.columns:
        print(f"Keine Trades oder Spalte {column!r} nicht vorhanden -- nichts zu plotten.")
        return None, None

    valid = trades[[column, "net_pnl"]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2a9d4f" if v >= 0 else "#e34948" for v in valid["net_pnl"]]
    ax.scatter(valid[column], valid["net_pnl"], c=colors, alpha=0.7, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel(column)
    ax.set_ylabel("Netto-PnL pro Trade ($)")
    ax.set_title(f"Netto-PnL vs. {column}")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _reconstruct_spread_band(
    trade: pd.Series, panel: dict[str, pd.DataFrame], *, bollinger_window_candles: int, buffer_candles: int,
) -> pd.DataFrame:
    """Rekonstruiert Spread + rollierenden Mittelwert/Std für ein Zeitfenster
    um einen Trade -- exakt dasselbe Rezept wie SpreadDefinition.rolling_zscore
    (types.py), nur als Serie statt als einzelner Punkt, damit sich daraus
    Bollinger-Bänder für einen Plot ableiten lassen."""
    symbols = list(trade["symbols"])
    weights = np.asarray(trade["weights"], dtype=float)
    close = panel["close"]
    idx = close.index

    entry_loc, exit_loc = idx.get_indexer([trade["entry_timestamp"], trade["exit_timestamp"]])
    if entry_loc == -1 or exit_loc == -1:
        raise ValueError(f"entry_/exit_timestamp von {trade['position_id']} nicht im Panel-Index gefunden.")

    # zusaetzlich bollinger_window_candles VOR dem Plot-Beginn laden, damit
    # das rollierende Fenster ab dem ersten geplotteten Punkt vollstaendig ist
    data_start = max(0, entry_loc - buffer_candles - bollinger_window_candles)
    data_end = min(len(idx) - 1, exit_loc + buffer_candles)
    window = close.iloc[data_start:data_end + 1][symbols]

    log_prices = np.log(window.to_numpy())
    spread = pd.Series(log_prices @ weights, index=window.index, name="spread")
    roll = spread.rolling(bollinger_window_candles, min_periods=bollinger_window_candles)

    plot_start = idx[max(0, entry_loc - buffer_candles)]
    return pd.DataFrame({"spread": spread, "mean": roll.mean(), "std": roll.std(ddof=0)}).loc[plot_start:]


def plot_trade_bollinger(
    trade: pd.Series,
    panel: dict[str, pd.DataFrame],
    params: "BasketParams",
    *,
    buffer_candles: int = 144,
    ax=None,
    show: bool = True,
):
    """Detailplot eines einzelnen Trades: rekonstruierter Spread mit
    rollierenden Bollinger-Bändern (Mittelwert ± entry_z/exit_z/stop_z * Std --
    exakt das Band, das SpreadDefinition.rolling_zscore live für Entry/Exit
    ausgewertet hat) sowie Entry-/Exit-Markern sowie farblich markierter
    Haltedauer. Macht sichtbar, an welcher Bandposition eine Position
    eröffnet/geschlossen wurde."""
    band = _reconstruct_spread_band(
        trade, panel, bollinger_window_candles=params.bollinger_window_candles, buffer_candles=buffer_candles,
    )

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    ax.axvspan(trade["entry_timestamp"], trade["exit_timestamp"], color="#f2c94c", alpha=0.15, zorder=0)
    ax.plot(band.index, band["spread"], color="#2a78d6", linewidth=1.1, label="Spread", zorder=3)
    ax.plot(band.index, band["mean"], color="#666666", linewidth=0.9, linestyle="--", label="Rollierender Mittelwert", zorder=2)

    for z, alpha in ((params.entry_z, 0.10), (params.exit_z, 0.18)):
        upper, lower = band["mean"] + z * band["std"], band["mean"] - z * band["std"]
        ax.fill_between(band.index, lower, upper, color="#2a78d6", alpha=alpha, linewidth=0, zorder=1)
        ax.plot(band.index, upper, color="#2a78d6", linewidth=0.5, alpha=0.5, zorder=1)
        ax.plot(band.index, lower, color="#2a78d6", linewidth=0.5, alpha=0.5, zorder=1)

    if params.stop_z is not None:
        ax.plot(band.index, band["mean"] + params.stop_z * band["std"], color="#e34948", linewidth=0.7, linestyle=":", label=f"±{params.stop_z:g}σ (stop)", zorder=1)
        ax.plot(band.index, band["mean"] - params.stop_z * band["std"], color="#e34948", linewidth=0.7, linestyle=":", zorder=1)

    marker_color = "#2a9d4f" if trade["net_pnl"] >= 0 else "#e34948"
    entry_marker = "^" if trade["direction"] > 0 else "v"
    ax.scatter([trade["entry_timestamp"]], [trade["entry_spread"]], marker=entry_marker, s=110, color="#1a1a1a", zorder=5, label="Entry")
    ax.scatter([trade["exit_timestamp"]], [trade["exit_spread"]], marker="o", s=90, color=marker_color, edgecolor="#1a1a1a", zorder=5, label="Exit")

    symbols = " / ".join(trade["symbols"])
    direction = "long" if trade["direction"] > 0 else "short"
    ax.set_title(
        f"{symbols} — {trade['exit_reason']} — net_pnl={trade['net_pnl']:.1f}$ "
        f"({trade['holding_hours']:.1f}h, {direction} spread, entry_z={trade['entry_z']:.2f})",
        fontsize=9.5,
    )
    ax.set_ylabel("Spread (log-Preis-Kombination)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="best")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=25, labelsize=8)

    if own_fig:
        plt.tight_layout()
        if show:
            plt.show()
    return fig, ax


def _pick_sample_trades(trades: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wählt bis zu n Trades im Round-Robin über exit_reason (jeweils
    absteigend nach |net_pnl|) -- so zeigt die Stichprobe sowohl profitable
    Mean-Reversion (target) als auch erzwungene Exits (timeout/stop/
    clique_break), statt nur von den häufigsten/profitabelsten dominiert zu
    werden."""
    pools = {
        reason: g.reindex(g["net_pnl"].abs().sort_values(ascending=False).index).index.tolist()
        for reason, g in trades.groupby("exit_reason", sort=False)
    }
    reasons = list(pools)
    selected: list = []
    i = 0
    while len(selected) < n and any(pools.values()):
        reason = reasons[i % len(reasons)]
        if pools[reason]:
            selected.append(pools[reason].pop(0))
        i += 1
    return trades.loc[selected]


def plot_sample_trades_bollinger(
    result: BacktestResult,
    panel: dict[str, pd.DataFrame],
    *,
    n: int = 6,
    buffer_candles: int = 144,
    ncols: int = 2,
    show: bool = True,
):
    """Raster aus n Beispiel-Trades mit rollierenden Bollinger-Bändern (siehe
    plot_trade_bollinger), Auswahl über _pick_sample_trades gemischt nach
    exit_reason."""
    trades = result.trades
    if trades.empty:
        print("Keine Trades -- nichts zu plotten.")
        return None, None

    sample = _pick_sample_trades(trades, n)
    nrows = int(np.ceil(len(sample) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)

    for ax, (_, trade) in zip(axes.flat, sample.iterrows()):
        params = result.config.params_for(trade["basket"])
        plot_trade_bollinger(trade, panel, params, buffer_candles=buffer_candles, ax=ax, show=False)
    for ax in axes.flat[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"Beispiel-Trades mit rollierenden Bollinger-Bändern — {result.config.label}", fontsize=11)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes
