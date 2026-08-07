"""Equity-/Drawdown-/Trade-Visualisierung für Backtest-Notebooks. Statisches
matplotlib, konsistent mit dem bestehenden Diagnostik-Stil in
notebooks/03_signal_generation/TradeData.ipynb."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import metrics as bt_metrics
from src.backtest.types import BacktestResult


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
