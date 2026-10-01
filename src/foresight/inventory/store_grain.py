"""Store vs chain grain reconciliation (Plan 6.1) — dual-path execution.

PATH 1 (chain): `score_chain_risk` compares chain stock against chain demand
for central purchasing.

PATH 2 (store): trailing sales shares
    w(store, sku) = store_units / chain_units   (trailing `store_share_window_weeks`)
disaggregate the chain view:
    store_demand = w * chain_forecast
SKUs with zero chain units in the window get a uniform share across the
stores stocking them (no demand signal exists to weight by).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from foresight.inventory.risk_scoring import sigmoid


def window_date_range(week_label: str, window_weeks: int) -> tuple[date, date]:
    """Trailing `window_weeks` weeks ending on the Sunday of `week_label`."""
    year, week = map(int, week_label.split("-W"))
    end = date.fromisocalendar(year, week, 7)
    start = date.fromisocalendar(year, week, 1) - timedelta(weeks=window_weeks - 1)
    return start, end


def compute_store_shares(
    store_daily: pd.DataFrame,
    inventory: pd.DataFrame,
    cutoff_week: str,
    window_weeks: int,
) -> pd.DataFrame:
    """Trailing-window sales shares per (sku, store) over inventory pairs.

    Parameters
    ----------
    store_daily : date, store_id, sku_id, units_sold (data/processed/sales_store_daily.parquet)
    inventory : store x sku normalized snapshot (defines stocking pairs)
    cutoff_week : ISO week label the window ends on (inclusive)
    window_weeks : trailing window length from configs/risk.yaml
    """
    start, end = window_date_range(cutoff_week, window_weeks)
    dates = pd.to_datetime(store_daily["date"])
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    units = (
        store_daily.loc[mask]
        .groupby(["sku_id", "store_id"], observed=True)["units_sold"]
        .sum()
        .reset_index(name="store_units")
    )

    base = inventory.loc[:, ["sku_id", "store_id"]].drop_duplicates()
    base = base.merge(units, on=["sku_id", "store_id"], how="left")
    base["store_units"] = base["store_units"].fillna(0.0)
    base["chain_units"] = base.groupby("sku_id", observed=True)["store_units"].transform("sum")
    stocking_stores = base.groupby("sku_id", observed=True)["store_id"].transform("size")

    has_sales = base["chain_units"] > 0
    base["share"] = np.where(
        has_sales,
        base["store_units"] / base["chain_units"].clip(lower=1.0),
        1.0 / stocking_stores,
    )
    return base.loc[:, ["sku_id", "store_id", "store_units", "chain_units", "share"]]


def score_store_risk(
    chain_risk: pd.DataFrame,
    store_shares: pd.DataFrame,
    inventory: pd.DataFrame,
    risk_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Disaggregate a scored chain view to stores (Plan 6.1 Path 2).

    store_LTD = share * chain_LTD; each store's own SOH / safety stock come
    straight from the snapshot rows, so an outage alert localises depleted
    stores even when the chain still holds cover.
    """
    eps = float(risk_cfg["inventory_defaults"]["epsilon"])
    store_stock = (
        inventory.groupby(["sku_id", "store_id"], observed=True)
        .agg(
            store_stock_on_hand=("stock_on_hand", "sum"),
            store_safety_stock=("safety_stock", "sum"),
        )
        .reset_index()
    )
    chain_cols = chain_risk.loc[:, ["sku_id", "ltd", "on_order_units"]]
    out = store_shares.merge(store_stock, on=["sku_id", "store_id"], how="left")
    out = out.merge(chain_cols, on="sku_id", how="left", validate="many_to_one")
    if out["ltd"].isna().any():
        missing = sorted(out.loc[out["ltd"].isna(), "sku_id"].unique())
        raise KeyError(f"store shares reference SKUs absent from chain risk: {missing[:5]}")

    out["store_ltd"] = out["share"] * out["ltd"]
    out["store_on_order"] = out["share"] * out["on_order_units"]
    out["store_pab"] = (
        out["store_stock_on_hand"] + out["store_on_order"] - out["store_ltd"]
    )
    ss = out["store_safety_stock"].to_numpy(dtype="float64")
    pab = out["store_pab"].to_numpy(dtype="float64")
    out["store_stockout_risk"] = sigmoid((ss - pab) / (ss + eps))
    out["store_outage"] = out["store_pab"] <= 0
    return out
