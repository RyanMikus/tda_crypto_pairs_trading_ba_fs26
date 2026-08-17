import pandas as pd
import pytest

from src.backtest.config import BacktestConfig, BasketParams, DateRange, split_calibration_backtest


class TestSplitCalibrationBacktest:
    def test_no_overlap(self):
        cal, bt = split_calibration_backtest("2024-01-01", "2024-12-31", fraction=0.2)
        assert cal.end < bt.start

    def test_fraction_approximately_matches_requested_share(self):
        cal, bt = split_calibration_backtest("2024-01-01", "2024-12-31", fraction=0.2)
        total_days = (bt.end - cal.start).days + 1
        backtest_days = (bt.end - bt.start).days + 1
        assert abs(backtest_days / total_days - 0.2) < 0.02

    def test_full_range_is_covered(self):
        start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")
        cal, bt = split_calibration_backtest(start, end, fraction=0.2)
        assert cal.start == start
        assert bt.end == end

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError):
            split_calibration_backtest("2024-12-31", "2024-01-01")

    def test_fraction_too_large_raises(self):
        with pytest.raises(ValueError):
            split_calibration_backtest("2024-01-01", "2024-01-05", fraction=1.5)


class TestBasketParamsOverride:
    def test_params_for_applies_override(self):
        config = BacktestConfig(
            label="t",
            backtest_range=DateRange(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")),
            basket_overrides={"Basket 1": {"entry_z": 1.8}},
        )
        assert config.params_for("Basket 1").entry_z == 1.8
        assert config.params_for("Basket 2").entry_z == BasketParams().entry_z  # unveraendert

    def test_unknown_override_field_raises(self):
        config = BacktestConfig(
            label="t",
            backtest_range=DateRange(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")),
            basket_overrides={"Basket 1": {"not_a_real_field": 1}},
        )
        with pytest.raises(ValueError, match="not_a_real_field"):
            config.params_for("Basket 1")

    def test_resolved_params_covers_all_baskets(self):
        config = BacktestConfig(
            label="t",
            backtest_range=DateRange(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")),
            baskets=("Basket 1", "Basket 2"),
        )
        resolved = config.resolved_params()
        assert set(resolved) == {"Basket 1", "Basket 2"}


class TestBacktestConfigSerialization:
    def _make_config(self):
        cal = DateRange(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-06-01", tz="UTC"))
        bt = DateRange(pd.Timestamp("2024-06-02", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC"))
        return BacktestConfig(
            label="roundtrip-test", description="x",
            calibration_range=cal, backtest_range=bt,
            baskets=("Basket 1",), basket_overrides={"Basket 1": {"entry_z": 1.7}},
        )

    def test_roundtrip_preserves_fingerprint(self):
        config = self._make_config()
        restored = BacktestConfig.from_dict(config.to_dict())
        assert restored.fingerprint() == config.fingerprint()

    def test_roundtrip_preserves_resolved_params(self):
        config = self._make_config()
        restored = BacktestConfig.from_dict(config.to_dict())
        assert restored.params_for("Basket 1").entry_z == 1.7

    def test_fingerprint_ignores_label_and_description(self):
        a = self._make_config()
        b = BacktestConfig(**{**vars(a), "label": "different-label", "description": "different"})
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_with_params(self):
        a = self._make_config()
        b = self._make_config()
        b.default_params = BasketParams(entry_z=99.0)
        assert a.fingerprint() != b.fingerprint()

    def test_start_end_properties(self):
        config = self._make_config()
        assert config.start == config.backtest_range.start
        assert config.end == config.backtest_range.end
