"""Inventory normalization: category lead times, censored-demand stub."""

from __future__ import annotations

import pandas as pd


def normalize_inventory(
    inventory: pd.DataFrame,
    sku_master: pd.DataFrame,
    lead_time_days: dict[str, int],
    on_order_units_default: int = 0,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    merged = inventory.merge(sku_master[["sku_id", "category"]], on="sku_id", how="left")
    if merged["category"].isnull().any():
        unknown = sorted(set(merged.loc[merged["category"].isnull(), "sku_id"]))
        raise KeyError(f"Unknown categories for SKUs: {unknown[:5]}")
    merged["lead_time_days"] = merged["category"].map(lead_time_days).astype("Int32")
    if merged["lead_time_days"].isnull().any():
        missing = sorted(set(merged.loc[merged["lead_time_days"].isnull(), "category"]))
        raise KeyError(f"Missing lead times for categories: {missing}")
    merged["on_order_units"] = on_order_units_default
    merged["snapshot_date"] = pd.to_datetime(snapshot_date) if snapshot_date else pd.NaT
    return merged


def demand_is_censored(observed_units_sold: int, stock_on_hand_before: int) -> bool:
    return observed_units_sold > 0 and stock_on_hand_before <= 0
