"""Phase 3 feature pipeline unit tests on synthetic data (no dataset files)."""

from __future__ import annotations

import pandas as pd
import pytest
from tests.helpers import empty_promo_weekly, mini_sku_master, mini_weekly_panel

from foresight.features.feature_pipeline import (
    FORBIDDEN_FEATURES,
    build_feature_matrix,
    feature_matrix_columns,
)
from foresight.preprocessing.calendar import build_calendar


@pytest.fixture(scope="module")
def inputs(settings):
    return {
        "panel": mini_weekly_panel(),
        "calendar": build_calendar("2023-01-01", "2024-12-31"),
        "promo": empty_promo_weekly(),
        "master": mini_sku_master(),
        "cfg": settings.features,
    }


@pytest.fixture(scope="module")
def matrix(inputs):
    return build_feature_matrix(
        inputs["panel"], inputs["calendar"], inputs["promo"], inputs["master"], inputs["cfg"]
    )


def test_feature_matrix_schema(matrix, inputs):
    assert len(matrix) == len(inputs["panel"])
    cols = set(matrix.columns)
    assert {"sku_id", "year_week", "units_sold"} <= cols
    assert FORBIDDEN_FEATURES.isdisjoint(cols)
    features = feature_matrix_columns(matrix)
    assert "units_sold" not in features
    assert "sku_id" not in features and "year_week" not in features
    expected = {"lag_1", "lag_52", "roll_4_mean", "zero_freq_8", "week_sin", "month_cos",
                "is_holiday_week", "holiday_proximity", "is_promo_active",
                "promo_discount_pct", "category_code", "brand_code", "gross_margin_pct"}
    assert expected <= set(features)


def test_lags_read_only_prior_weeks(matrix, inputs):
    panel = inputs["panel"]
    sku = "SKU00001"
    series = panel[panel["sku_id"] == sku].reset_index(drop=True)
    pos = 60
    row = matrix[(matrix["sku_id"] == sku) & (matrix["year_week"] == series.loc[pos, "year_week"])]
    assert len(row) == 1
    assert row["lag_1"].iloc[0] == float(series.loc[pos - 1, "units_sold"])
    assert row["lag_52"].iloc[0] == float(series.loc[pos - 52, "units_sold"])


def test_rolling_reads_only_prior_weeks(matrix, inputs):
    panel = inputs["panel"]
    sku = "SKU00002"
    series = panel[panel["sku_id"] == sku].reset_index(drop=True)
    pos = 60
    row = matrix[(matrix["sku_id"] == sku) & (matrix["year_week"] == series.loc[pos, "year_week"])]
    window = series.loc[pos - 4 : pos - 1, "units_sold"]
    assert row["roll_4_mean"].iloc[0] == pytest.approx(float(window.mean()))


def test_warmup_rows_have_nan_lags(matrix):
    first = matrix.sort_values(["sku_id", "year_week"]).groupby("sku_id", observed=True).head(1)
    assert first["lag_1"].isna().all()
    full = matrix[matrix["lag_52"].notna()]
    per_sku = matrix["sku_id"].nunique()
    assert len(matrix) - len(full) == per_sku * 52


def test_promo_features_merge(settings):
    panel = mini_weekly_panel()
    calendar = build_calendar("2023-01-01", "2024-12-31")
    week = sorted(panel["year_week"].unique())[20]
    promo = pd.DataFrame(
        {
            "year_week": [week, week],
            "sku_id": ["SKU00001", "SKU00002"],
            "promo_days": [5, 5],
            "promo_discount_pct": [25.0, 25.0],
            "promo_type": ["Percentage Discount", "Clearance"],
            "promo_target_level": ["Brand", "All"],
            "is_promo_active": [1, 1],
        }
    )
    m = build_feature_matrix(panel, calendar, promo, mini_sku_master(), settings.features)
    hot = m[m["year_week"] == week].set_index("sku_id")
    assert hot.loc["SKU00001", "is_promo_active"] == 1
    assert hot.loc["SKU00001", "promo_type_percentage_discount"] == 1
    assert hot.loc["SKU00001", "promo_target_brand"] == 1
    assert hot.loc["SKU00001", "promo_discount_pct"] == pytest.approx(25.0)
    assert hot.loc["SKU00002", "promo_type_clearance"] == 1
    other_week = sorted(panel["year_week"].unique())[40]
    cold = m[m["year_week"] == other_week]
    assert (cold["is_promo_active"] == 0).all()
    assert (cold["promo_discount_pct"] == 0).all()


def test_calendar_missing_week_raises(settings):
    panel = mini_weekly_panel()
    calendar = build_calendar("2023-01-01", "2023-06-30")  # ends mid-panel
    with pytest.raises(ValueError, match="Calendar has no rows"):
        build_feature_matrix(
            panel, calendar, empty_promo_weekly(), mini_sku_master(), settings.features
        )


def test_future_weeks_allowed_with_extended_calendar(settings):
    panel = mini_weekly_panel()
    calendar = build_calendar("2023-01-01", "2024-12-31")
    weeks = sorted(panel["year_week"].unique())
    last = weeks[-1]
    year, week = map(int, last.split("-W"))
    future = f"{year:04d}-W{week + 1:02d}"
    placeholder = pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002"],
            "year_week": [future, future],
            "units_sold": [0.0, 0.0],
            "unit_price": [10.0, 11.0],
        }
    )
    frame = pd.concat([panel, placeholder], ignore_index=True)
    m = build_feature_matrix(frame, calendar, empty_promo_weekly(), mini_sku_master(),
                             settings.features)
    fut = m[m["year_week"] == future]
    assert len(fut) == 2
    assert fut["week_sin"].notna().all()
