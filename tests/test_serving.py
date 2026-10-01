"""Serving-layer compilation for the dashboard (Plan 7.1)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

WEEKS_8 = [f"2026-W{w:02d}" for w in range(2, 10)]


def mini_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002", "SKU00003", "SKU00004"],
            "sku_name": ["Milk 1L", "Notebook", "Cola 500ml", "USB Cable"],
            "category": ["Dairy & Bakery", "Grocery", "Grocery", "Dairy & Bakery"],
            "subcategory": ["Milk", "Paper", "Soda", "Accessories"],
            "brand": ["Sunrise", "SoftTouch", "FizzCo", "TechLine"],
            "unit_price": [10.0, 5.0, 3.0, 8.0],
            "cost_price": [7.0, 3.0, 2.0, 5.0],
        }
    )


def mini_weekly(n_weeks: int = 55) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")  # Monday of ISO 2024-W01
    weeks = []
    for i in range(n_weeks):
        iso = (start + pd.Timedelta(weeks=i)).isocalendar()
        weeks.append(f"{iso.year:04d}-W{iso.week:02d}")
    rows = [
        {"sku_id": sku, "year_week": week, "units_sold": 10.0}
        for sku in ("SKU00001", "SKU00002", "SKU00003", "SKU00004")
        for week in weeks
    ]
    return pd.DataFrame(rows)


def mini_forecast() -> pd.DataFrame:
    rows = [
        {
            "sku_id": sku,
            "year_week": week,
            "forecast": 12.0,
            "forecast_q10": 8.0,
            "forecast_q90": 16.0,
        }
        for sku in ("SKU00001", "SKU00002", "SKU00003", "SKU00004")
        for week in WEEKS_8
    ]
    return pd.DataFrame(rows)


def mini_risk_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002"],
            "category": ["Dairy & Bakery", "Grocery"],
            "n_stores": [2, 1],
            "stock_on_hand": [5, 500],
            "on_order_units": [0, 0],
            "reorder_point": [8, 50],
            "safety_stock": [4, 20],
            "lead_time_days": [7, 7],
            "lead_time_weeks": [1, 1],
            "last_restock_date": ["2025-12-01", "2025-06-01"],
            "restock_age_days": [30, 213],
            "snapshot_date": pd.to_datetime(["2025-12-31", "2025-12-31"]),
            "forward_weekly_mean": [12.0, 12.0],
            "forward_8w_demand": [96.0, 96.0],
            "ltd": [12.0, 12.0],
            "pab": [-7.0, 488.0],
            "stockout_risk": [0.91, 0.01],
            "stockout_high": [True, False],
            "wos": [0.42, 41.7],
            "overstock_high": [False, True],
            "stale_restock": [False, True],
            "slow_mover_high": [False, True],
            "quadrant": ["reorder_now", "markdown_clear"],
            "unit_price": [10.0, 5.0],
            "cost_price": [7.0, 3.0],
            "rupees_at_risk": [60.0, 0.0],
            "locked_capital": [0.0, 1200.0],
        }
    )


def mini_cv_summary() -> dict:
    return {"ml": {"overall_wape": 0.3}, "comparison": {"baseline_overall_wape": 0.4}}


def mini_promo_weekly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year_week": ["2026-W02", "2026-W03", "2026-W04"],
            "sku_id": ["SKU00001", "SKU00001", "SKU00002"],
            "promo_days": [7, 7, 5],
            "promo_discount_pct": [20.0, 20.0, 10.0],
            "promo_type": ["Percentage Discount"] * 3,
            "promo_target_level": ["SKU"] * 3,
            "is_promo_active": [1, 0, 1],
        }
    )


def test_build_decision_grid_includes_unscored_skus():
    from foresight.serving.build import build_decision_grid

    grid = build_decision_grid(mini_risk_scores(), mini_master(), mini_forecast())
    assert len(grid) == 4  # every master SKU, not just scored ones
    assert set(grid["scored"]) == {True, False}
    unscored = grid.loc[grid["sku_id"] == "SKU00003"].iloc[0]
    assert unscored["quadrant"] == "not_scored"
    assert pd.isna(unscored["stockout_risk"])
    assert unscored["rupees_at_risk"] == 0.0
    assert unscored["forward_8w_demand"] == 96.0  # forecast demand still visible

    scored = grid.loc[grid["sku_id"] == "SKU00001"].iloc[0]
    assert scored["quadrant"] == "reorder_now"
    assert scored["rupee_impact"] == scored["rupees_at_risk"] + scored["locked_capital"]
    assert bool(scored["stockout_high"]) is True


def test_build_dashboard_kpis_sections_and_values():
    from foresight.serving.build import build_dashboard_kpis, build_decision_grid

    grid = build_decision_grid(mini_risk_scores(), mini_master(), mini_forecast())
    kpis = build_dashboard_kpis(grid, mini_forecast(), mini_cv_summary())

    section = kpis[kpis["section"] == "kpi"].set_index("metric")["value"]
    assert section["active_skus"] == 4
    assert section["scored_skus"] == 2
    assert section["forecast_8w_units"] == 4 * 8 * 12.0
    assert section["stockout_rupee_exposure"] == 60.0
    assert section["locked_capital"] == 1200.0
    assert section["reorder_now_skus"] == 1
    assert section["ml_wape"] == 0.3
    assert section["baseline_wape"] == 0.4
    assert section["wape_reduction_pct"] == pytest.approx(25.0)

    category = kpis[kpis["section"] == "category"]
    grocery = category[(category["category"] == "Grocery")].set_index("metric")["value"]
    assert grocery["n_skus"] == 2
    assert grocery["markdown_clear_count"] == 1
    assert set(category["category"]) == {"Dairy & Bakery", "Grocery"}


def test_build_forecast_summary_merges_baseline():
    from foresight.serving.build import build_forecast_summary

    baseline = pd.DataFrame(
        {
            "sku_id": ["SKU00001"] * 8,
            "year_week": WEEKS_8,
            "forecast": [10.0] * 8,
            "tier_used": [1] * 8,
        }
    )
    out = build_forecast_summary(mini_forecast(), baseline, mini_master())
    assert list(out.columns) == [
        "sku_id", "category", "year_week", "forecast", "forecast_q10",
        "forecast_q90", "baseline", "tier_used",
    ]
    row = out[out["sku_id"] == "SKU00001"].iloc[0]
    assert row["baseline"] == 10.0
    assert row["category"] == "Dairy & Bakery"
    # SKUs missing from the baseline frame fall back to 0.0, never NaN
    assert (out["baseline"].notna()).all()


def test_promo_windows_active_only():
    from foresight.serving.build import build_promo_windows

    windows = build_promo_windows(mini_promo_weekly())
    assert len(windows) == 2  # is_promo_active == 1 rows only
    assert set(windows["sku_id"]) == {"SKU00001", "SKU00002"}


def test_write_history_partitions_slug_collision(tmp_path):
    from foresight.serving.build import write_history_partitions

    master = pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002"],
            "category": ["Home-Care", "Home Care"],  # both slug to Home_Care
        }
    )
    mapping = write_history_partitions(mini_weekly(4), master, tmp_path)
    assert set(mapping) == {"Home-Care", "Home Care"}
    assert len(set(mapping.values())) == 2
    for relative in mapping.values():
        assert (tmp_path / relative).exists()


def test_build_serving_layer_writes_all_stores(tmp_path):
    from foresight.serving.build import build_serving_layer

    baseline = pd.DataFrame(
        [
            {"sku_id": sku, "year_week": week, "forecast": 10.0, "tier_used": 1}
            for sku in ("SKU00001", "SKU00002", "SKU00003", "SKU00004")
            for week in WEEKS_8
        ]
    )
    settings = SimpleNamespace(serving_dir=tmp_path)
    result = build_serving_layer(
        settings=settings,
        weekly=mini_weekly(),
        sku_master=mini_master(),
        promo_weekly=mini_promo_weekly(),
        risk_scores=mini_risk_scores(),
        forecast=mini_forecast(),
        cv_summary=mini_cv_summary(),
        baseline=baseline,
    )

    for name in (
        "dashboard_kpis.parquet",
        "decision_grid.parquet",
        "forecast_summary.parquet",
        "history_weekly.parquet",
        "promo_windows.parquet",
        "serving_manifest.json",
    ):
        assert (tmp_path / name).exists(), name

    import json

    manifest = json.loads((tmp_path / "serving_manifest.json").read_text(encoding="utf-8"))
    assert manifest["active_skus"] == 4
    assert manifest["scored_skus"] == 2
    assert set(manifest["history_partitions"]) == {"Dairy & Bakery", "Grocery"}
    assert manifest["forecast_weeks"] == ["2026-W02", "2026-W09"]
    for relative in manifest["history_partitions"].values():
        part = pd.read_parquet(tmp_path / relative)
        assert {"sku_id", "year_week", "units_sold"} <= set(part.columns)

    grid = pd.read_parquet(tmp_path / "decision_grid.parquet")
    assert len(grid) == 4
    assert result["active_skus"] == 4
    assert result["history_partitions"] == 2
