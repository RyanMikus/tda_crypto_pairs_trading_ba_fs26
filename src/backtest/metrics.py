"""Performance-Kennzahlen: dünner Wrapper um empyrical-reloaded für die
Standard-Statistikformeln (Sharpe/Sortino/Calmar/CAGR/Volatilität), plus
eigener Code für alles, was die Trade-/Positionsstruktur braucht (Win-Rate,
Profit-Factor, Cost-Drag, Exit-Reason-Verteilung, Drawdown in % UND $ mit
Dauer -- das deckt keine Library ab, weil es Positionsobjekte statt einer
reinen Return-Serie braucht).

Zwei Frequenzen: *_bar (native 5-Min-Aufloesung) und *_daily (Tagesbasis).
Die Tages-Werte sind die Headline -- eine 5-Min-Sharpe ist wegen Mikro-
struktur-Rauschen und Autokorrelation der Bar-Returns nicht mit ueblichen
Tages-Sharpes vergleichbar (siehe Plan, Abschnitt Metriken).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import empyrical as ep
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import CANDLE_MINUTES

if TYPE_CHECKING:
    from src.backtest.config import BacktestConfig

DAYS_PER_YEAR = 365.0


def periods_per_year(candle_minutes: int = CANDLE_MINUTES) -> float:
    """Krypto handelt 24/7/365 -- kein Boersenkalender, keine 252 Handelstage."""
    return DAYS_PER_YEAR * 24.0 * 60.0 / candle_minutes


def to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def resample_equity(equity: pd.Series, freq: str = "1D") -> pd.Series:
    return equity.resample(freq).last().dropna()


def max_drawdown_detail(equity: pd.Series) -> dict:
    """Ergaenzung zu empyrical.max_drawdown: zusaetzlich absoluter $-Verlust,
    Peak-/Trough-/Recovery-Zeitpunkt und Dauer (Zeit UND Geld, wie gefordert)."""
    if equity.empty:
        return {
            "max_drawdown_pct": np.nan, "max_drawdown_dollar": np.nan,
            "peak_ts": None, "trough_ts": None, "recovery_ts": None,
            "duration_to_trough": None, "duration_to_recovery": None,
        }

    running_max = equity.cummax()
    drawdown_pct = equity / running_max - 1.0
    trough_ts = drawdown_pct.idxmin()
    peak_val = running_max.loc[trough_ts]

    pre_trough = equity.loc[:trough_ts]
    peak_ts = pre_trough[pre_trough == peak_val].index[-1]

    post_trough = equity.loc[trough_ts:]
    recovered = post_trough[post_trough >= peak_val]
    recovery_ts = recovered.index[0] if len(recovered) > 0 else None

    return {
        "max_drawdown_pct": float(drawdown_pct.loc[trough_ts]),
        "max_drawdown_dollar": float(equity.loc[trough_ts] - peak_val),
        "peak_ts": peak_ts,
        "trough_ts": trough_ts,
        "recovery_ts": recovery_ts,
        "duration_to_trough": trough_ts - peak_ts,
        "duration_to_recovery": (recovery_ts - peak_ts) if recovery_ts is not None else None,
    }


def attach_drawdown_columns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Pro-Bar drawdown_pct/drawdown_dollar-Spalten, fürs Persistieren/Plotten."""
    equity = equity_curve["equity"]
    running_max = equity.cummax()
    out = equity_curve.copy()
    out["drawdown_pct"] = equity / running_max - 1.0
    out["drawdown_dollar"] = equity - running_max
    return out


def benchmark_returns(benchmark_close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = benchmark_close.reindex(index).ffill()
    return aligned.pct_change().dropna()


def beta_alpha(returns: pd.Series, benchmark_ret: pd.Series, *, ppy: float, risk_free_rate: float = 0.0) -> dict:
    """OLS r_p = alpha + beta*r_benchmark + eps. Eine markt-neutrale Pairs-/
    Cluster-Spread-Strategie SOLL beta ~ 0 haben -- ein signifikant von 0
    verschiedenes Beta ist ein Diagnosebefund (Legs nicht dollar-neutral,
    Hedge-Ratio veraltet), keine zu maximierende Kennzahl."""
    aligned = pd.concat([returns - risk_free_rate, benchmark_ret], axis=1, join="inner").dropna()
    aligned.columns = ["strategy", "benchmark"]
    if len(aligned) < 3:
        return {
            "beta": np.nan, "beta_stderr": np.nan, "beta_tstat": np.nan,
            "alpha_per_period": np.nan, "alpha_annualized": np.nan,
            "r_squared": np.nan, "correlation": np.nan, "n_obs": len(aligned),
        }

    x = sm.add_constant(aligned["benchmark"].to_numpy())
    model = sm.OLS(aligned["strategy"].to_numpy(), x).fit()
    alpha_per_period, beta = model.params

    return {
        "beta": float(beta),
        "beta_stderr": float(model.bse[1]),
        "beta_tstat": float(model.tvalues[1]),
        "alpha_per_period": float(alpha_per_period),
        "alpha_annualized": float((1 + alpha_per_period) ** ppy - 1),
        "r_squared": float(model.rsquared),
        "correlation": float(aligned["strategy"].corr(aligned["benchmark"])),
        "n_obs": int(len(aligned)),
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    empty = {
        "n_trades": 0, "n_winners": 0, "n_losers": 0, "win_rate": np.nan,
        "profit_factor": np.nan, "expectancy": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
        "payoff_ratio": np.nan, "largest_win": np.nan, "largest_loss": np.nan,
        "avg_holding_candles": np.nan, "avg_holding_hours": np.nan, "median_holding_hours": np.nan,
        "max_holding_hours": np.nan, "avg_return_on_notional": np.nan,
        "exit_reason_counts": {}, "pct_forced_exits": np.nan,
    }
    if trades.empty:
        return empty

    pnl = trades["net_pnl"]
    winners, losers = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = winners.sum(), -losers.sum()

    return {
        "n_trades": len(trades),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "win_rate": len(winners) / len(trades),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
        "expectancy": float(pnl.mean()),
        "avg_win": float(winners.mean()) if len(winners) else np.nan,
        "avg_loss": float(losers.mean()) if len(losers) else np.nan,
        "payoff_ratio": (winners.mean() / abs(losers.mean())) if len(winners) and len(losers) and losers.mean() != 0 else np.nan,
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "avg_holding_candles": float(trades["bars_held"].mean()),
        "avg_holding_hours": float(trades["holding_hours"].mean()),
        "median_holding_hours": float(trades["holding_hours"].median()),
        "max_holding_hours": float(trades["holding_hours"].max()),
        "avg_return_on_notional": float(trades["return_on_notional"].mean()),
        "exit_reason_counts": trades["exit_reason"].value_counts().to_dict(),
        "pct_forced_exits": float(trades["is_forced"].mean()),
    }


def turnover(trades: pd.DataFrame, equity_curve: pd.DataFrame, *, ppy: float) -> float:
    """Annualisiertes gehandeltes Notional / Ø-Equity (Entry + Exit je einmal
    gehandelt -> Faktor 2 auf das Entry-Bruttonotional)."""
    if trades.empty or equity_curve.empty:
        return 0.0
    traded_notional = (trades["gross_notional"] * 2).sum()
    avg_equity = equity_curve["equity"].mean()
    years = len(equity_curve) / ppy
    if years <= 0 or avg_equity <= 0:
        return np.nan
    return float(traded_notional / avg_equity / years)


def cost_drag(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_cost": 0.0, "cost_pct_of_gross_profit": np.nan}
    total_cost = float(trades["total_cost"].sum())
    gross_profit = trades.loc[trades["gross_pnl"] > 0, "gross_pnl"].sum()
    return {
        "total_cost": total_cost,
        "cost_pct_of_gross_profit": float(total_cost / gross_profit) if gross_profit > 0 else np.nan,
    }


def exposure_stats(equity_curve: pd.DataFrame) -> dict:
    if equity_curve.empty:
        return {
            "time_in_market_pct": np.nan, "avg_gross_exposure": np.nan, "max_gross_exposure": np.nan,
            "avg_n_open_positions": np.nan, "max_n_open_positions": np.nan,
        }
    return {
        "time_in_market_pct": float((equity_curve["n_open_positions"] > 0).mean()),
        "avg_gross_exposure": float(equity_curve["gross_exposure"].mean()),
        "max_gross_exposure": float(equity_curve["gross_exposure"].max()),
        "avg_n_open_positions": float(equity_curve["n_open_positions"].mean()),
        "max_n_open_positions": float(equity_curve["n_open_positions"].max()),
    }


def metrics_by_basket(trades: pd.DataFrame, config: "BacktestConfig") -> pd.DataFrame:
    rows = []
    for basket in config.baskets:
        sub = trades[trades["basket"] == basket] if not trades.empty else trades
        stats = trade_stats(sub)
        stats["exit_reason_counts"] = json.dumps(stats["exit_reason_counts"])
        stats["basket"] = basket
        rows.append(stats)
    return pd.DataFrame(rows).set_index("basket")


#! WIRD NICHT MEHR GEBRAUCHT
def metrics_by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return trades.groupby("exit_reason").agg(
        n_trades=("net_pnl", "count"),
        total_net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
        avg_holding_hours=("holding_hours", "mean"),
    )


def compute_all_metrics(
    *, equity_curve: pd.DataFrame, trades: pd.DataFrame,
    benchmark_close: pd.Series | None, config: "BacktestConfig",
) -> dict:
    """Berechnet ALLE Kennzahlen auf zwei Frequenzen (*_bar/*_daily, Tages-
    Werte sind die Headline) plus die Trade-Log-Statistiken. Annualisierung
    wird empyrical explizit übergeben (annualization=ppy), NICHT über
    period="daily" geraten -- 5-Min-Krypto-Bars haben kein Pendant in
    empyricals eingebauten Perioden-Presets."""
    ppy_bar = periods_per_year(config.candle_minutes)
    ppy_daily = DAYS_PER_YEAR

    equity_bar = equity_curve["equity"] if "equity" in equity_curve else pd.Series(dtype=float)
    returns_bar = to_returns(equity_bar)
    equity_daily = resample_equity(equity_bar, "1D") if len(equity_bar) else equity_bar
    returns_daily = to_returns(equity_daily)

    metrics: dict = {
        "total_return": float(equity_bar.iloc[-1] / config.initial_capital - 1) if len(equity_bar) else np.nan,
    }

    for suffix, returns, ppy in (("bar", returns_bar, ppy_bar), ("daily", returns_daily, ppy_daily)):
        if len(returns) < 2:
            metrics[f"cagr_{suffix}"] = np.nan
            metrics[f"ann_volatility_{suffix}"] = np.nan
            metrics[f"sharpe_{suffix}"] = np.nan
            metrics[f"sortino_{suffix}"] = np.nan
            metrics[f"calmar_{suffix}"] = np.nan
            continue
        arr = returns.to_numpy()
        metrics[f"cagr_{suffix}"] = float(ep.annual_return(arr, annualization=ppy))
        metrics[f"ann_volatility_{suffix}"] = float(ep.annual_volatility(arr, annualization=ppy))
        metrics[f"sharpe_{suffix}"] = float(ep.sharpe_ratio(arr, risk_free=config.risk_free_rate, annualization=ppy))
        metrics[f"sortino_{suffix}"] = float(ep.sortino_ratio(arr, required_return=config.risk_free_rate, annualization=ppy))
        metrics[f"calmar_{suffix}"] = float(ep.calmar_ratio(arr, annualization=ppy))

    dd = max_drawdown_detail(equity_bar)
    metrics["max_drawdown_pct"] = dd["max_drawdown_pct"]
    metrics["max_drawdown_dollar"] = dd["max_drawdown_dollar"]
    metrics["max_drawdown_peak_ts"] = dd["peak_ts"]
    metrics["max_drawdown_trough_ts"] = dd["trough_ts"]
    metrics["max_drawdown_recovery_ts"] = dd["recovery_ts"]
    metrics["max_drawdown_duration_to_trough"] = dd["duration_to_trough"]
    metrics["max_drawdown_duration_to_recovery"] = dd["duration_to_recovery"]

    if benchmark_close is not None and len(returns_daily) >= 3:
        bench_ret_daily = benchmark_returns(benchmark_close, equity_daily.index)
        ba = beta_alpha(returns_daily, bench_ret_daily, ppy=ppy_daily, risk_free_rate=config.risk_free_rate)
        metrics.update({f"{k}_vs_benchmark": v for k, v in ba.items()})

    metrics.update(trade_stats(trades))
    metrics["turnover_annualized"] = turnover(trades, equity_curve, ppy=ppy_daily)
    metrics.update(cost_drag(trades))
    metrics.update(exposure_stats(equity_curve))

    return metrics
