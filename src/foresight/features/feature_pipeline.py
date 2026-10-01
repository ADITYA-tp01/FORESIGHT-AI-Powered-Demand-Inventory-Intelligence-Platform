"""Feature matrix assembly for the global demand forecaster.

`build_feature_matrix` whitelists panel inputs (keys, target, unit_price) and
attaches lag, rolling, calendar, promo and SKU metadata features. Concurrent
revenue quantities are deliberately excluded: they would leak same-week actuals
into the forecast.
"""

from __future__ import annotations

import pandas as pd

from foresight.features.calendar_features import build_calendar_features
from foresight.features.lag_features import build_lag_features
from foresight.features.promo_features import build_promo_features
from foresight.features.rolling_features import build_rolling_features

KEY_COLS = ("sku_id", "year_week")
TARGET_COL = "units_sold"
PANEL_INPUT_COLS = ("sku_id", "year_week", "units_sold", "unit_price")
FORBIDDEN_FEATURES = frozenset({"revenue", "gross_revenue", "promo_trans_count"})


def feature_matrix_columns(matrix: pd.DataFrame) -> list[str]:
    """Feature columns = everything except keys and the target."""
    return [c for c in matrix.columns if c not in KEY_COLS and c != TARGET_COL]


def build_feature_matrix(
    weekly_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    feature_cfg: dict,
) -> pd.DataFrame:
    """Assemble the supervised learning frame for the global forecaster.

    Returns one row per sku-week with columns:
    keys (sku_id, year_week) + features + target (units_sold).
    Sorted by (sku_id, year_week); lag/rolling features use only prior weeks.
    """
    missing = set(PANEL_INPUT_COLS) - set(weekly_panel.columns)
    if missing:
        raise ValueError(f"weekly_panel missing required columns: {sorted(missing)}")

    out = weekly_panel.loc[:, list(PANEL_INPUT_COLS)].copy()
    out = out.sort_values(["sku_id", "year_week"], kind="mergesort").reset_index(drop=True)

    promo_cfg = feature_cfg["promo"]
    out = build_lag_features(out, feature_cfg["lags"]["units_sold"])
    rolling_cfg = feature_cfg["rolling"]
    out = build_rolling_features(
        out,
        rolling_cfg["windows_weeks"],
        rolling_cfg["stats"],
        rolling_cfg["zero_sales_window"],
    )
    cal_cfg = feature_cfg["calendar"]
    out = build_calendar_features(
        out,
        calendar,
        cal_cfg["fourier_week_period"],
        cal_cfg["fourier_month_period"],
        cal_cfg["holiday_proximity_days"],
    )
    out = build_promo_features(
        out, promo_weekly, promo_cfg["promo_types"], promo_cfg["target_levels"]
    )
    out = build_meta_features(out, sku_master)

    feature_cols = feature_matrix_columns(out)
    forbidden = FORBIDDEN_FEATURES & set(feature_cols)
    if forbidden:
        raise ValueError(f"Leakage-prone columns in feature matrix: {sorted(forbidden)}")
    for col in feature_cols:
        if out[col].dtype == "float64":
            out[col] = out[col].astype("float32")
    return out


def build_meta_features(df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Attach SKU metadata: category/subcategory/brand codes, prices, margin."""
    master = sku_master.drop_duplicates(subset="sku_id")[
        ["sku_id", "category", "subcategory", "brand", "unit_price", "cost_price"]
    ].rename(columns={"unit_price": "master_unit_price"})
    out = df.merge(master, on="sku_id", how="left")
    if out["cost_price"].isna().any():
        unknown = sorted(set(out.loc[out["cost_price"].isna(), "sku_id"]))
        raise ValueError(f"sku_master missing rows for sku_ids: {unknown[:5]} (n={len(unknown)})")

    for src, dst in (
        ("category", "category_code"),
        ("subcategory", "subcategory_code"),
        ("brand", "brand_code"),
    ):
        categories = sorted(pd.unique(master[src]))
        codes = pd.Categorical(out[src], categories=categories).codes
        if (codes < 0).any():
            raise ValueError(f"Unmapped {src} values present in panel")
        out[dst] = codes.astype("int16")
        out = out.drop(columns=[src])

    out["gross_margin_pct"] = (
        (out["master_unit_price"] - out["cost_price"]) / out["master_unit_price"]
    ).astype("float32")
    return out.drop(columns=["master_unit_price"])
