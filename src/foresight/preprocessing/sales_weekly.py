"""Daily → weekly rollup and dense zero-demand panel."""

from __future__ import annotations

import pandas as pd


def _to_weekly(grain: pd.DataFrame) -> pd.DataFrame:
    grain = grain.copy()
    grain["date"] = pd.to_datetime(grain["date"])
    grain["year_week"] = grain["date"].apply(lambda d: f"{d.isocalendar()[0]:04d}-W{d.isocalendar()[1]:02d}")
    weekly = (
        grain.groupby(["year_week", "sku_id"], observed=True, dropna=False)
        .agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            unit_price=("unit_price", "max") if "unit_price" in grain.columns else ("revenue", "max"),
            promo_trans_count=("promo_trans_count", "sum")
            if "promo_trans_count" in grain.columns
            else ("units_sold", "count"),
        )
        .reset_index()
    )
    return weekly


def rollup_daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return _to_weekly(daily)


def build_dense_weekly_panel(
    weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    date_range = pd.date_range(start_date, end_date, freq="D")
    iso_weeks = sorted({f"{d.isocalendar()[0]:04d}-W{d.isocalendar()[1]:02d}" for d in date_range})
    catalog = sku_master[["sku_id"]].drop_duplicates().assign(_tmp_key=1)
    week_index = pd.DataFrame({"year_week": iso_weeks, "_tmp_key": 1})
    panel = catalog.merge(week_index, on="_tmp_key", how="inner").drop(columns="_tmp_key")
    merged = panel.merge(weekly, on=["year_week", "sku_id"], how="left")
    merged["units_sold"] = merged["units_sold"].fillna(0).astype("int32")
    merged["revenue"] = merged["revenue"].fillna(0.0).astype("float32")
    merged["gross_revenue"] = merged["gross_revenue"].fillna(0.0).astype("float32")
    price_lookup = sku_master.set_index("sku_id")["unit_price"]
    merged["unit_price"] = merged["unit_price"].fillna(merged["sku_id"].map(price_lookup)).astype("float32")
    merged["promo_trans_count"] = merged["promo_trans_count"].fillna(0).astype("int32")
    return merged
