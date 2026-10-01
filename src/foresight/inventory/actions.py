"""Rupee-impact quantification behind the decision grid (Plan 6.4).

    Rupees at Risk = max(0, LTD - (SOH + on_order)) * unit_price
    Locked Capital = max(0, SOH - (forward_8w_demand + SS)) * cost_price

All prices come from sku_master; nothing is imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.inventory.risk_scoring import QUADRANT_LABELS


def add_rupee_impact(risk: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Attach unit/cost prices and the two rupee metrics to a risk frame."""
    master = sku_master.loc[:, ["sku_id", "unit_price", "cost_price"]].drop_duplicates(
        subset="sku_id"
    )
    out = risk.merge(master, on="sku_id", how="left", validate="one_to_one")
    if out["unit_price"].isna().any() or out["cost_price"].isna().any():
        missing = sorted(out.loc[out["unit_price"].isna(), "sku_id"])
        raise KeyError(f"sku_master missing prices for SKUs: {missing[:5]}")

    soh = out["stock_on_hand"].to_numpy(dtype="float64")
    on_order = out["on_order_units"].to_numpy(dtype="float64")
    out["rupees_at_risk"] = (
        np.maximum(0.0, out["ltd"].to_numpy() - (soh + on_order))
        * out["unit_price"].to_numpy(dtype="float64")
    )
    out["locked_capital"] = (
        np.maximum(0.0, soh - (out["forward_8w_demand"].to_numpy() + out["safety_stock"].to_numpy()))
        * out["cost_price"].to_numpy(dtype="float64")
    )
    return out


def impact_summary(risk: pd.DataFrame) -> dict[str, float]:
    """Chain totals for dashboards and the D4 memo."""
    counts = risk["quadrant"].value_counts()
    # Always all four quadrants so downstream schemas stay stable (zero-fill).
    quadrant_counts = {
        label: int(counts.get(label, 0)) for label in sorted(QUADRANT_LABELS.values())
    }
    return {
        "n_skus_scored": int(len(risk)),
        "rupees_at_risk": float(risk["rupees_at_risk"].sum()),
        "locked_capital": float(risk["locked_capital"].sum()),
        "quadrant_counts": quadrant_counts,
    }
