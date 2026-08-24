"""Run-Speicherung: legt einen Backtest-Run als Ordner unter
data/processed/backtests/ ab (analog zum etablierten data/processed/cliques/-
Präzedenzfall: teuer zu berechnen, deshalb versioniert statt gitignored) und
stellt Vergleichs-Helper über mehrere Runs bereit.

Bewusst NICHT "persistence.py" genannt: "Persistenz" ist im Rest des Projekts
bereits ein fest belegter TDA-Begriff (epsilon_death - epsilon_birth einer
Clique, siehe src/backtest/cliques.py::CliqueSnapshot.persistence) -- ein
gleichnamiges Modul für "Backtest-Runs auf Platte speichern" wäre verwirrend.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import metrics as bt_metrics
from src.backtest.config import BacktestConfig
from src.backtest.types import BacktestResult
from src.config import BACKTESTS_DIR

_INDEX_COLUMNS = [
    "run_id", "created_at", "label", "description", "strategy_name", "clique_variant",
    "baskets", "backtest_start", "backtest_end", "fingerprint",
    "n_trades", "sharpe_daily", "cagr_daily", "max_drawdown_pct", "runtime_seconds",
]

HIGHER_IS_BETTER = {
    "cagr_daily", "cagr_bar", "sharpe_daily", "sharpe_bar", "sortino_daily", "sortino_bar",
    "calmar_daily", "calmar_bar", "win_rate", "profit_factor", "avg_return_on_notional",
    "total_return", "time_in_market_pct",
}
LOWER_IS_BETTER = {
    "max_drawdown_pct", "max_drawdown_dollar", "total_cost", "cost_pct_of_gross_profit",
    "ann_volatility_daily", "ann_volatility_bar", "pct_forced_exits",
}

DEFAULT_COMPARE_METRICS = [
    "cagr_daily", "ann_volatility_daily", "sharpe_daily", "sortino_daily", "calmar_daily",
    "max_drawdown_pct", "max_drawdown_dollar", "beta_vs_benchmark", "alpha_annualized_vs_benchmark",
    "n_trades", "win_rate", "profit_factor", "avg_holding_hours", "turnover_annualized",
    "total_cost", "cost_pct_of_gross_profit", "time_in_market_pct",
]


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return s or "run"


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def save_run(result: BacktestResult, *, backtests_dir: Path = BACKTESTS_DIR, overwrite: bool = False) -> Path:
    """Speichert config.json/metrics.json/trades.parquet/equity_curve.parquet
    unter backtests_dir/<run_id>/ und setzt result.run_id. Aktualisiert
    _index.parquet danach vollständig neu (siehe list_runs)."""
    backtests_dir.mkdir(parents=True, exist_ok=True)

    created_at = pd.Timestamp.now(tz="UTC")
    fingerprint = result.config.fingerprint()
    run_id = f"{created_at:%Y%m%d-%H%M%S}_{_slug(result.config.label)}_{fingerprint}"
    run_dir = backtests_dir / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"{run_dir} existiert bereits (overwrite=True zum Überschreiben).")
    run_dir.mkdir(parents=True, exist_ok=True)

    result.run_id = run_id

    config_payload = {
        **result.config.to_dict(),
        "created_at": created_at.isoformat(),
        "fingerprint": fingerprint,
        "engine_version": result.meta.get("engine_version"),
        "runtime_seconds": result.meta.get("runtime_seconds"),
        "git_sha": result.meta.get("git_sha"),
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2, default=_json_default))
    (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=_json_default))

    if not result.metrics_by_basket.empty:
        result.metrics_by_basket.to_parquet(run_dir / "metrics_by_basket.parquet")

    result.trades.to_parquet(run_dir / "trades.parquet", index=False)

    if not result.equity_curve.empty:
        equity_to_save = bt_metrics.attach_drawdown_columns(result.equity_curve)
        equity_to_save.reset_index().to_parquet(run_dir / "equity_curve.parquet", index=False)

    _rebuild_index(backtests_dir)
    return run_dir


def load_run(run_id: str, *, load_equity: bool = True, backtests_dir: Path = BACKTESTS_DIR) -> BacktestResult:
    run_dir = backtests_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Kein Run {run_id!r} unter {backtests_dir}.")

    config_payload = json.loads((run_dir / "config.json").read_text())
    config = BacktestConfig.from_dict(config_payload)
    metrics = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}

    trades_path = run_dir / "trades.parquet"
    trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    mbb_path = run_dir / "metrics_by_basket.parquet"
    metrics_by_basket = pd.read_parquet(mbb_path) if mbb_path.exists() else pd.DataFrame()

    equity_curve = pd.DataFrame()
    equity_path = run_dir / "equity_curve.parquet"
    if load_equity and equity_path.exists():
        equity_curve = pd.read_parquet(equity_path).set_index("timestamp")

    meta = {
        "created_at": config_payload.get("created_at"),
        "engine_version": config_payload.get("engine_version"),
        "fingerprint": config_payload.get("fingerprint"),
        "runtime_seconds": config_payload.get("runtime_seconds"),
    }

    return BacktestResult(
        run_id=run_id, config=config, equity_curve=equity_curve, trades=trades,
        metrics=metrics, metrics_by_basket=metrics_by_basket, positions_curve=None, meta=meta,
    )


def _rebuild_index(backtests_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in backtests_dir.iterdir() if p.is_dir()):
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        backtest_range = cfg.get("backtest_range") or {}
        rows.append({
            "run_id": run_dir.name,
            "created_at": cfg.get("created_at"),
            "label": cfg.get("label"),
            "description": cfg.get("description"),
            "strategy_name": cfg.get("strategy_name"),
            "clique_variant": cfg.get("clique_variant"),
            "baskets": cfg.get("baskets"),
            "backtest_start": backtest_range.get("start"),
            "backtest_end": backtest_range.get("end"),
            "fingerprint": cfg.get("fingerprint"),
            "n_trades": metrics.get("n_trades"),
            "sharpe_daily": metrics.get("sharpe_daily"),
            "cagr_daily": metrics.get("cagr_daily"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "runtime_seconds": cfg.get("runtime_seconds"),
        })

    df = pd.DataFrame(rows, columns=_INDEX_COLUMNS)
    if not df.empty:
        df = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    df.to_parquet(backtests_dir / "_index.parquet", index=False)
    return df


def list_runs(*, backtests_dir: Path = BACKTESTS_DIR, refresh_index: bool = False) -> pd.DataFrame:
    index_path = backtests_dir / "_index.parquet"
    if refresh_index or not index_path.exists():
        if not backtests_dir.exists():
            return pd.DataFrame(columns=_INDEX_COLUMNS)
        return _rebuild_index(backtests_dir)
    return pd.read_parquet(index_path)

def _style_compare(df: pd.DataFrame):
    """Hebt pro Metrik-Zeile den besten Run hervor (nur für Metriken in
    HIGHER_IS_BETTER/LOWER_IS_BETTER -- alle anderen bleiben ungefärbt, da
    "besser" für sie nicht eindeutig ist, z.B. n_trades)."""
    def _highlight_row(row):
        name = row.name
        numeric = pd.to_numeric(row, errors="coerce")
        if name in HIGHER_IS_BETTER:
            best = numeric.max()
        elif name in LOWER_IS_BETTER:
            best = numeric.min()
        else:
            return ["" for _ in row]
        return ["font-weight: bold; background-color: #e6f4ea" if v == best else "" for v in numeric]

    return df.style.apply(_highlight_row, axis=1)


def compare_runs(
    run_ids: list[str] | None = None,
    *,
    metrics: list[str] | None = None,
    orient: str = "metrics_as_rows",
    by_basket: bool = False,
    highlight_best: bool = True,
    backtests_dir: Path = BACKTESTS_DIR,
):
    """Vergleichstabelle über mehrere Runs. Default-Orientierung: Index =
    Metrikname, Spalten = run_id (man vergleicht typischerweise wenige Runs
    über viele Metriken). by_basket=True nutzt metrics_by_basket (nur Trade-
    Log-Statistiken, KEINE Equity-basierten Metriken -- die Engine trackt
    keine Basket-eigene Equity-Kurve)."""
    if run_ids is None:
        run_ids = list_runs(backtests_dir=backtests_dir)["run_id"].tolist()
    results = {rid: load_run(rid, load_equity=False, backtests_dir=backtests_dir) for rid in run_ids}

    if by_basket:
        metric_keys = metrics if metrics is not None else None
        pieces = []
        for rid, res in results.items():
            mbb = res.metrics_by_basket
            if metric_keys is not None:
                mbb = mbb[[c for c in metric_keys if c in mbb.columns]]
            pieces.append(mbb.stack().rename(rid))
        out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame()
        if not out.empty:
            out.index.names = ["basket", "metric"]
        return out

    metric_keys = metrics if metrics is not None else DEFAULT_COMPARE_METRICS
    table = pd.DataFrame({rid: {k: res.metrics.get(k, np.nan) for k in metric_keys} for rid, res in results.items()})

    if orient == "runs_as_rows":
        table = table.T
        return table  # Styler-Highlighting ist nur für orient="metrics_as_rows" implementiert

    if highlight_best:
        return _style_compare(table)
    return table


def diff_configs(run_ids: list[str], *, backtests_dir: Path = BACKTESTS_DIR) -> pd.DataFrame:
    """Zeigt NUR die Config-Felder, in denen sich die angegebenen Runs
    unterscheiden -- inkl. der aufgelösten Pro-Basket-Parameter. Ohne diese
    Tabelle ist compare_runs wenig aussagekräftig: man sieht, dass ein Run
    besser ist, aber nicht warum."""
    flat = {}
    for rid in run_ids:
        cfg = load_run(rid, load_equity=False, backtests_dir=backtests_dir).config.to_dict()
        flat[rid] = pd.json_normalize(cfg, sep=".").iloc[0].to_dict()

    df = pd.DataFrame(flat)
    mask = df.apply(lambda row: len({str(v) for v in row}) > 1, axis=1)
    return df[mask]


def compare_equity_curves(
    run_ids: list[str], *, normalize: bool = True, resample: str | None = "1D", backtests_dir: Path = BACKTESTS_DIR,
) -> pd.DataFrame:
    series = {}
    for rid in run_ids:
        res = load_run(rid, load_equity=True, backtests_dir=backtests_dir)
        if res.equity_curve.empty:
            continue
        eq = res.equity_curve["equity"]
        if resample:
            eq = eq.resample(resample).last()
        if normalize:
            eq = eq / eq.iloc[0]
        series[rid] = eq
    return pd.DataFrame(series)


def compare_trade_distributions(
    run_ids: list[str], *, by: str = "exit_reason", backtests_dir: Path = BACKTESTS_DIR,
) -> pd.DataFrame:
    frames = []
    for rid in run_ids:
        res = load_run(rid, load_equity=False, backtests_dir=backtests_dir)
        if res.trades.empty:
            continue
        grouped = res.trades.groupby(by)["net_pnl"].agg(["count", "mean", "sum"])
        grouped["run_id"] = rid
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).reset_index()
