"""Chain-level inventory risk scoring (Plan 6.2) and 4-quadrant decisioning (6.3).

Formulas, all thresholds from configs/risk.yaml:

    lead_time_weeks = ceil(lead_time_days / 7)
    LTD             = sum of forecast over the first lead_time_weeks
    PAB             = stock_on_hand + on_order_units - LTD
    stockout_risk   = sigmoid((safety_stock - PAB) / (safety_stock + epsilon))
    stockout HIGH   = PAB <= 0
    WOS             = stock_on_hand / (mean(forecast over horizon) + epsilon)
    overstock HIGH  = WOS > overstock_wos_weeks
                      or stock_on_hand > overstock_reorder_multiple * reorder_point
    slow-mover HIGH = overstock HIGH and restock_age_days >= restock_stale_days
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

QUADRANT_LABELS = {
    (True, False): "reorder_now",
    (True, True): "watch_volatile",
    (False, True): "markdown_clear",
    (False, False): "healthy",
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Logistic function; scipy.special.expit is overflow-safe at extremes."""
    return expit(np.asarray(values, dtype="float64"))


def lead_time_weeks(lead_time_days: pd.Series | np.ndarray) -> np.ndarray:
    """Weeks of cover needed for a lead time: ceil(days / 7)."""
    return np.ceil(np.asarray(lead_time_days, dtype="float64") / 7.0).astype(int)


def _forecast_blocks(
    chain: pd.DataFrame, forecasts: pd.DataFrame, horizon_weeks: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-SKU (LTD, forward-horizon total, span) from an ordered forecast frame.

    LTD sums the first ceil(lead_time_days / 7) forecast weeks; the forward
    block spans `min(horizon_weeks, available)` weeks (the plan's 8-week
    forward demand and WOS divisor).
    """
    weeks = sorted(forecasts["year_week"].unique())
    if not weeks:
        raise ValueError("forecasts is empty")
    wide = (
        forecasts.pivot(index="sku_id", columns="year_week", values="forecast")
        .reindex(index=chain["sku_id"], columns=weeks)
        .fillna(0.0)
    )
    cumulative = wide.to_numpy(dtype="float64").cumsum(axis=1)
    n_sku = len(chain)
    n_weeks = len(weeks)

    lt_idx = np.minimum(lead_time_weeks(chain["lead_time_days"]), n_weeks) - 1
    if (lt_idx < 0).any():
        raise ValueError("lead_time_days must be positive to compute LTD")
    ltd = cumulative[np.arange(n_sku), lt_idx]

    forward_span = min(int(horizon_weeks), n_weeks)
    forward_total = cumulative[:, forward_span - 1]
    return ltd, forward_total, forward_span


def score_chain_risk(
    inventory: pd.DataFrame,
    forecasts: pd.DataFrame,
    risk_cfg: dict[str, Any],
    horizon_weeks: int,
) -> pd.DataFrame:
    """Score every SKU with inventory rows at chain grain (Plan 6.1 Path 1).

    Parameters
    ----------
    inventory : store x sku normalized snapshot (data/interim/inventory_normalized.parquet)
    forecasts : long frame sku_id, year_week, forecast (future weeks only)
    risk_cfg : full configs/risk.yaml dict
    horizon_weeks : forward window used for WOS / forward demand (config horizon)

    Returns
    -------
    DataFrame, one row per inventory SKU, with LTD, PAB, stockout risk, WOS,
    predicted flags (stockout_high / slow_mover_high) and the quadrant label.
    """
    defaults = risk_cfg["inventory_defaults"]
    eps = float(defaults["epsilon"])

    inv = inventory.assign(
        last_restock_date=pd.to_datetime(inventory["last_restock_date"], errors="coerce")
    )
    chain = (
        inv.groupby("sku_id", observed=True)
        .agg(
            category=("category", "first"),
            n_stores=("store_id", "nunique"),
            stock_on_hand=("stock_on_hand", "sum"),
            on_order_units=("on_order_units", "sum"),
            reorder_point=("reorder_point", "sum"),
            safety_stock=("safety_stock", "sum"),
            lead_time_days=("lead_time_days", "first"),
            last_restock_date=("last_restock_date", "min"),
            snapshot_date=("snapshot_date", "max"),
        )
        .reset_index()
    )

    ltd, forward_total, forward_span = _forecast_blocks(chain, forecasts, horizon_weeks)
    chain["lead_time_weeks"] = lead_time_weeks(chain["lead_time_days"])
    chain["ltd"] = ltd
    chain["forward_weekly_mean"] = forward_total / forward_span
    chain["forward_8w_demand"] = forward_total

    soh = chain["stock_on_hand"].to_numpy(dtype="float64")
    on_order = chain["on_order_units"].to_numpy(dtype="float64")
    ss = chain["safety_stock"].to_numpy(dtype="float64")

    chain["pab"] = soh + on_order - ltd
    chain["stockout_risk"] = sigmoid((ss - chain["pab"].to_numpy()) / (ss + eps))
    chain["stockout_high"] = chain["pab"] <= 0

    chain["wos"] = soh / (chain["forward_weekly_mean"].to_numpy() + eps)
    over_wos = chain["wos"].to_numpy() > float(defaults["overstock_wos_weeks"])
    rp = chain["reorder_point"].to_numpy(dtype="float64")
    over_rp = soh > float(defaults["overstock_reorder_multiple"]) * rp
    chain["overstock_high"] = over_wos | over_rp

    chain["restock_age_days"] = (chain["snapshot_date"] - chain["last_restock_date"]).dt.days
    chain["stale_restock"] = chain["restock_age_days"].fillna(-1) >= int(
        defaults["restock_stale_days"]
    )
    chain["slow_mover_high"] = chain["overstock_high"] & chain["stale_restock"]

    pair_keys = list(zip(chain["stockout_high"], chain["overstock_high"], strict=True))
    chain["quadrant"] = pd.Series(pair_keys, index=chain.index).map(QUADRANT_LABELS)

    ordered = [
        "sku_id",
        "category",
        "n_stores",
        "stock_on_hand",
        "on_order_units",
        "reorder_point",
        "safety_stock",
        "lead_time_days",
        "lead_time_weeks",
        "last_restock_date",
        "restock_age_days",
        "snapshot_date",
        "forward_weekly_mean",
        "forward_8w_demand",
        "ltd",
        "pab",
        "stockout_risk",
        "stockout_high",
        "wos",
        "overstock_high",
        "stale_restock",
        "slow_mover_high",
        "quadrant",
    ]
    return chain.loc[:, ordered]
