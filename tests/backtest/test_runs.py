import pandas as pd
import pytest

from Src.backtest import metrics as bt_metrics
from Src.backtest import runs as bt_runs
from Src.backtest.config import BacktestConfig, BasketParams, DateRange
from Src.backtest.types import BacktestResult


def _make_result(label, *, entry_z=2.0):
    idx = pd.date_range("2025-01-01", periods=20, freq="1D", tz="UTC")
    equity = 100_000 + pd.Series(range(20), dtype=float).to_numpy() * 10.0
    equity_curve = pd.DataFrame({"equity": equity, "n_open_positions": 0, "gross_exposure": 0.0}, index=idx)
    equity_curve.index.name = "timestamp"

    trades = pd.DataFrame({
        "position_id": ["p1", "p2"], "basket": ["Basket 1", "Basket 1"],
        "net_pnl": [10.0, -5.0], "gross_pnl": [12.0, -4.0], "total_cost": [2.0, 1.0],
        "exit_reason": ["target", "stop"], "bars_held": [3, 1], "holding_hours": [0.25, 0.08],
        "return_on_notional": [0.01, -0.005], "is_forced": [False, False],
    })

    config = BacktestConfig(
        label=label, description="test run",
        calibration_range=DateRange(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
        backtest_range=DateRange(idx[0], idx[-1]),
        baskets=("Basket 1",), default_params=BasketParams(entry_z=entry_z),
    )
    metrics = {"n_trades": 2, "sharpe_daily": 1.23, "cagr_daily": 0.1, "max_drawdown_pct": -0.02}
    metrics_by_basket = bt_metrics.metrics_by_basket(trades, config)

    return BacktestResult(
        run_id="", config=config, equity_curve=equity_curve, trades=trades,
        metrics=metrics, metrics_by_basket=metrics_by_basket, positions_curve=None,
        meta={"engine_version": "test", "runtime_seconds": 1.0},
    )


@pytest.fixture
def frozen_now(monkeypatch):
    """Pinnt pd.Timestamp.now() auf einen festen Wert -- run_id enthaelt den
    Zeitstempel sekundengenau, ohne das waere der overwrite-Test flaky."""
    fixed = {"value": pd.Timestamp("2025-06-01T12:00:00", tz="UTC")}
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: fixed["value"]))
    return fixed


class TestSaveLoadRoundtrip:
    def test_save_creates_expected_files(self, tmp_path):
        result = _make_result("roundtrip-a")
        run_dir = bt_runs.save_run(result, backtests_dir=tmp_path)
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "trades.parquet").exists()
        assert (run_dir / "equity_curve.parquet").exists()
        assert result.run_id

    def test_load_restores_metrics_and_fingerprint(self, tmp_path):
        result = _make_result("roundtrip-b", entry_z=2.0)
        bt_runs.save_run(result, backtests_dir=tmp_path)
        loaded = bt_runs.load_run(result.run_id, backtests_dir=tmp_path)

        assert loaded.metrics["sharpe_daily"] == pytest.approx(1.23)
        assert loaded.config.fingerprint() == result.config.fingerprint()
        assert loaded.config.params_for("Basket 1").entry_z == 2.0

    def test_load_restores_trade_values(self, tmp_path):
        result = _make_result("roundtrip-c")
        bt_runs.save_run(result, backtests_dir=tmp_path)
        loaded = bt_runs.load_run(result.run_id, backtests_dir=tmp_path)
        assert loaded.trades["net_pnl"].tolist() == pytest.approx([10.0, -5.0])

    def test_load_missing_run_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            bt_runs.load_run("does-not-exist", backtests_dir=tmp_path)

    def test_existing_run_not_overwritten_without_flag(self, tmp_path, frozen_now):
        result_a = _make_result("roundtrip-d")
        bt_runs.save_run(result_a, backtests_dir=tmp_path)

        result_b = _make_result("roundtrip-d")  # identische Config -> identischer fingerprint+run_id
        with pytest.raises(FileExistsError):
            bt_runs.save_run(result_b, backtests_dir=tmp_path, overwrite=False)

    def test_overwrite_flag_allows_replacing(self, tmp_path, frozen_now):
        result_a = _make_result("roundtrip-e")
        bt_runs.save_run(result_a, backtests_dir=tmp_path)
        result_b = _make_result("roundtrip-e")
        bt_runs.save_run(result_b, backtests_dir=tmp_path, overwrite=True)  # darf nicht raisen


class TestListAndDeleteRuns:
    def test_list_runs_finds_saved_run(self, tmp_path):
        result = _make_result("list-test")
        bt_runs.save_run(result, backtests_dir=tmp_path)
        runs = bt_runs.list_runs(backtests_dir=tmp_path, refresh_index=True)
        assert result.run_id in runs["run_id"].tolist()

    def test_list_runs_on_empty_dir_returns_empty_frame(self, tmp_path):
        runs = bt_runs.list_runs(backtests_dir=tmp_path)
        assert runs.empty

    def test_delete_run_removes_directory_and_updates_index(self, tmp_path):
        result = _make_result("delete-test")
        run_dir = bt_runs.save_run(result, backtests_dir=tmp_path)
        bt_runs.delete_run(result.run_id, confirm=True, backtests_dir=tmp_path)
        assert not run_dir.exists()
        assert result.run_id not in bt_runs.list_runs(backtests_dir=tmp_path, refresh_index=True)["run_id"].tolist()

    def test_delete_run_requires_explicit_confirm(self, tmp_path):
        result = _make_result("delete-test-2")
        bt_runs.save_run(result, backtests_dir=tmp_path)
        with pytest.raises(ValueError):
            bt_runs.delete_run(result.run_id, confirm=False, backtests_dir=tmp_path)


class TestCompareRunsAndDiffConfigs:
    def test_compare_runs_returns_metric_per_run(self, tmp_path):
        a = _make_result("cmp-a", entry_z=1.5)
        b = _make_result("cmp-b", entry_z=2.5)
        bt_runs.save_run(a, backtests_dir=tmp_path)
        bt_runs.save_run(b, backtests_dir=tmp_path)

        table = bt_runs.compare_runs([a.run_id, b.run_id], backtests_dir=tmp_path, highlight_best=False)
        assert set(table.columns) == {a.run_id, b.run_id}
        assert table.loc["n_trades", a.run_id] == 2

    def test_diff_configs_shows_only_differing_fields(self, tmp_path):
        a = _make_result("diff-a", entry_z=1.5)
        b = _make_result("diff-b", entry_z=2.5)
        bt_runs.save_run(a, backtests_dir=tmp_path)
        bt_runs.save_run(b, backtests_dir=tmp_path)

        diff = bt_runs.diff_configs([a.run_id, b.run_id], backtests_dir=tmp_path)
        assert "default_params.entry_z" in diff.index
        assert "clique_variant" not in diff.index  # identisch in beiden Configs -> nicht in der Diff

    def test_compare_equity_curves_normalizes_to_start_at_one(self, tmp_path):
        a = _make_result("curve-a")
        bt_runs.save_run(a, backtests_dir=tmp_path)
        curves = bt_runs.compare_equity_curves([a.run_id], backtests_dir=tmp_path, resample=None)
        assert curves[a.run_id].iloc[0] == pytest.approx(1.0)

    def test_compare_trade_distributions_groups_by_exit_reason(self, tmp_path):
        a = _make_result("dist-a")
        bt_runs.save_run(a, backtests_dir=tmp_path)
        dist = bt_runs.compare_trade_distributions([a.run_id], backtests_dir=tmp_path)
        assert set(dist["exit_reason"]) == {"target", "stop"}
