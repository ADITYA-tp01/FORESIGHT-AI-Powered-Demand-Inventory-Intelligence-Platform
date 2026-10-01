"""Phase 2: metrics, seasonal-naive baseline, segmentation, backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from foresight.evaluation.metrics import bias, evaluate_forecast, mae, mape, rmse, wape
from foresight.evaluation.segmentation import abc_segment, pareto_summary, xyz_segment
from foresight.forecasting.baseline import (
    generate_baseline_forecasts,
    seasonal_naive_forecast,
    seasonal_naive_with_fallback,
    trailing_mean_fallback,
)


def test_wape_basic_and_zero_safe():
    y_true = np.array([10.0, 0.0, 20.0])
    y_pred = np.array([8.0, 5.0, 25.0])
    # |10-8| + |0-5| + |20-25| = 12; sum(actual) = 30
    assert wape(y_true, y_pred) == pytest.approx(12 / 30)
    # All-zero actuals: WAPE undefined (not division by zero crash)
    assert np.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))


def test_bias_direction():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([6.0, 6.0])
    # positive bias = under-forecasting
    assert bias(y_true, y_pred) == pytest.approx(0.4)
    # negative bias = over-forecasting
    over = np.array([14.0, 14.0])
    assert bias(y_true, over) == pytest.approx(-0.4)


def test_mae_rmse_mape():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 16.0])
    assert mae(y_true, y_pred) == pytest.approx(3.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 16) / 2))
    assert mape(y_true, y_pred) == pytest.approx(np.mean([0.2, 0.2]))
    # MAPE skips zero actuals instead of dividing by zero
    assert not np.isnan(mape(np.array([0.0, 10.0]), np.array([1.0, 10.0])))


def test_evaluate_forecast_keys():
    m = evaluate_forecast(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert set(m) == {"wape", "mape", "bias", "mae", "rmse"}


def test_seasonal_naive_uses_t_minus_52():
    # 60 weeks of history: week i has value i
    history = pd.Series(np.arange(1.0, 61.0))
    fc = seasonal_naive_forecast(history, horizon=4, seasonality=52)
    # forecast for weeks 61..64 comes from weeks 9..12
    assert fc.tolist() == [9.0, 10.0, 11.0, 12.0]


def test_seasonal_naive_cold_start_nan():
    history = pd.Series([1.0] * 10)
    fc = seasonal_naive_forecast(history, horizon=8, seasonality=52)
    assert np.all(np.isnan(fc))


def test_tier1_used_when_seasonal_positive():
    # Constant positive history: tier 1 should fire
    history = pd.Series([5.0] * 60)
    forecast, tiers = seasonal_naive_with_fallback(history, horizon=8)
    assert np.all(forecast == 5.0)
    assert set(tiers.tolist()) == {1}


def test_tier2_fallback_when_seasonal_zero():
    # First 52 weeks zero, last 4 weeks positive -> seasonal source is 0 -> tier 2
    history = pd.Series([0.0] * 52 + [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    forecast, tiers = seasonal_naive_with_fallback(history, horizon=4)
    assert set(tiers.tolist()) == {2}
    assert np.all(forecast == 4.0)


def test_tier3_fallback_when_no_signal():
    # All history zero: tier 1 (0) and tier 2 (mean 0 -> NaN) fail -> tier 3
    history = pd.Series([0.0] * 60)
    forecast, tiers = seasonal_naive_with_fallback(
        history, horizon=4, category_avg=10.0, sku_ratio=0.5
    )
    assert set(tiers.tolist()) == {3}
    assert np.all(forecast == 5.0)


def test_tier3_zero_when_no_category():
    history = pd.Series([0.0] * 60)
    forecast, tiers = seasonal_naive_with_fallback(history, horizon=4)
    assert set(tiers.tolist()) == {3}
    assert np.all(forecast == 0.0)


def test_trailing_mean_fallback():
    assert trailing_mean_fallback(pd.Series([1.0, 2.0, 3.0, 4.0]), 8, 4) == pytest.approx(2.5)
    assert np.isnan(trailing_mean_fallback(pd.Series([0.0] * 4), 8, 4))
    assert np.isnan(trailing_mean_fallback(pd.Series([1.0]), 8, 4))


def _mini_panel(weeks: int = 60, skus: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s in range(skus):
        for w in range(weeks):
            year, week = 2022 + w // 52, w % 52 + 1
            rows.append(
                {
                    "year_week": f"{year:04d}-W{week:02d}",
                    "sku_id": f"SKU{s:05d}",
                    "units_sold": 10 + s,
                    "revenue": float((10 + s) * 10),
                }
            )
    panel = pd.DataFrame(rows)
    master = pd.DataFrame(
        {
            "sku_id": [f"SKU{s:05d}" for s in range(skus)],
            "category": ["Dairy & Bakery"] * skus,
            "unit_price": [10.0] * skus,
        }
    )
    return panel, master


def test_generate_forecasts_shape_and_alignment():
    panel, master = _mini_panel(weeks=60, skus=3)
    future = [f"2023-W0{i}" for i in range(5, 9)]
    fc = generate_baseline_forecasts(
        panel, master, horizon=4, cutoff_week="2023-W04", future_weeks=future
    )
    assert len(fc) == 3 * 4
    assert set(fc["year_week"]) == set(future)
    # 56 weeks of history, constant positive sales -> tier 1 (seasonal) fires
    assert set(fc["tier_used"].tolist()) == {1}
    # SKU1 constant 11 units -> seasonal source equals the constant
    sku1 = fc[fc["sku_id"] == "SKU00001"]
    assert np.allclose(sku1["forecast"], 11.0)


def test_generate_forecasts_cold_start_uses_tier2():
    # < 52 weeks of history: seasonal unavailable, trailing mean positive -> tier 2
    panel, master = _mini_panel(weeks=60, skus=1)
    fc = generate_baseline_forecasts(
        panel, master, horizon=2, cutoff_week="2022-W10"
    )
    assert set(fc["tier_used"].tolist()) == {2}
    assert np.allclose(fc["forecast"], 10.0)


def test_abc_segment_classes():
    # 10 SKUs: one dominates revenue
    rows = []
    revenues = [1000] + [10] * 9
    for i, rev in enumerate(revenues):
        rows.append(
            {
                "year_week": "2022-W01",
                "sku_id": f"SKU{i:05d}",
                "units_sold": int(rev),
                "revenue": float(rev),
            }
        )
    panel = pd.DataFrame(rows)
    abc = abc_segment(panel)
    assert abc.iloc[0]["abc"] == "A"
    assert abc.iloc[0]["sku_id"] == "SKU00000"
    # Tail SKUs fall to C once cumulative exceeds 90%
    assert "C" in set(abc["abc"])


def test_xyz_segment_boundaries():
    # Stable demand -> X
    stable = pd.DataFrame(
        {
            "year_week": [f"2022-W{w:02d}" for w in range(1, 21)],
            "sku_id": ["S"] * 20,
            "units_sold": [10, 10, 11, 9, 10, 10, 11, 9, 10, 10, 11, 9, 10, 10, 11, 9, 10, 10, 11, 9],
        }
    )
    xyz = xyz_segment(stable)
    assert xyz.iloc[0]["xyz"] == "X"

    # Lumpy demand -> Z
    lumpy = pd.DataFrame(
        {
            "year_week": [f"2022-W{w:02d}" for w in range(1, 21)],
            "sku_id": ["S"] * 20,
            "units_sold": [0, 0, 50, 0, 0, 0, 80, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 90, 0, 0],
        }
    )
    assert xyz_segment(lumpy).iloc[0]["xyz"] == "Z"


def test_pareto_summary():
    rows = []
    for i, rev in enumerate([1000, 500, 300, 200]):
        rows.append(
            {"year_week": "2022-W01", "sku_id": f"S{i}", "units_sold": 1, "revenue": float(rev)}
        )
    summary = pareto_summary(pd.DataFrame(rows), top_share=0.5)
    assert summary["n_skus"] == 4
    assert summary["top_sku_count"] == 2
    # top 2 of 4 = 1500/2000
    assert summary["top_revenue_share"] == pytest.approx(0.75)


def test_backtest_summary_folds_and_overall():
    from foresight.forecasting.validation import portfolio_backtest_summary

    # 60 weeks of constant history: two folds
    panel, master = _mini_panel(weeks=60, skus=2)
    folds = [
        {"train_end_week": 50, "valid_weeks": 5},
        {"train_end_week": 55, "valid_weeks": 5},
    ]
    summary = portfolio_backtest_summary(panel, master, folds)
    assert len(summary["folds"]) == 2
    assert summary["model"] == "seasonal_naive_3tier"
    # Constant history, seasonal source positive -> perfect forecast
    assert summary["overall_wape"] == pytest.approx(0.0, abs=1e-12)
    for f in summary["folds"]:
        assert f["n_rows"] == 2 * f["valid_weeks"]  # 2 SKUs x valid weeks


def test_backtest_rejects_out_of_range_fold():
    from foresight.forecasting.validation import portfolio_backtest_summary

    panel, master = _mini_panel(weeks=60, skus=1)
    with pytest.raises(ValueError, match="out of range"):
        portfolio_backtest_summary(
            panel, master, [{"train_end_week": 58, "valid_weeks": 8}]
        )


def test_no_future_leakage_in_cutoff():
    """Perturbing post-cutoff actuals must not change baseline forecasts."""
    panel, master = _mini_panel(weeks=60, skus=2)
    cutoff = "2022-W40"
    future = ["2023-W01", "2023-W02"]
    fc_before = generate_baseline_forecasts(
        panel, master, horizon=2, cutoff_week=cutoff, future_weeks=future
    )
    perturbed = panel.copy()
    mask = perturbed["year_week"] > cutoff
    perturbed.loc[mask, "units_sold"] = perturbed.loc[mask, "units_sold"] * 100
    fc_after = generate_baseline_forecasts(
        perturbed, master, horizon=2, cutoff_week=cutoff, future_weeks=future
    )
    pd.testing.assert_frame_equal(fc_before, fc_after)
