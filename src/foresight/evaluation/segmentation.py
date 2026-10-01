"""ABC (revenue contribution) and XYZ (demand predictability) segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def abc_segment(
    weekly_panel: pd.DataFrame,
    a_threshold: float = 0.70,
    b_threshold: float = 0.90,
    revenue_col: str = "revenue",
    sku_col: str = "sku_id",
) -> pd.DataFrame:
    """
    ABC segmentation by cumulative revenue contribution.
    Class A: top 70% cumulative revenue.
    Class B: next 20% cumulative revenue (70-90%).
    Class C: final 10% cumulative revenue (90-100%).
    """
    rev = (
        weekly_panel.groupby(sku_col, observed=True)[revenue_col]
        .sum()
        .reset_index()
        .rename(columns={revenue_col: "total_revenue"})
    )
    rev = rev.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    total = rev["total_revenue"].sum()
    if total <= 0:
        rev["abc"] = "C"
        rev["cum_revenue_share"] = 0.0
        rev["revenue_share"] = 0.0
        return rev
    rev["revenue_share"] = rev["total_revenue"] / total
    rev["cum_revenue_share"] = rev["revenue_share"].cumsum()

    # Class by the cumulative share BEFORE this SKU: the item that pushes the
    # running total past a threshold stays in the upper class (standard ABC).
    prev_cum = rev["cum_revenue_share"] - rev["revenue_share"]
    rev["abc"] = [
        "A" if p < a_threshold else ("B" if p < b_threshold else "C")
        for p in prev_cum
    ]
    return rev


def xyz_segment(
    weekly_panel: pd.DataFrame,
    units_col: str = "units_sold",
    sku_col: str = "sku_id",
    x_max: float = 0.5,
    y_max: float = 1.0,
) -> pd.DataFrame:
    """
    XYZ segmentation by coefficient of variation (CV = std / mean).
    Class X: CV <= 0.5 (highly predictable).
    Class Y: 0.5 < CV <= 1.0 (moderate volatility).
    Class Z: CV > 1.0 (intermittent / lumpy).
    SKUs with zero mean demand are classified Z.
    """
    stats = weekly_panel.groupby(sku_col, observed=True)[units_col].agg(["mean", "std"]).reset_index()
    stats.columns = [sku_col, "demand_mean", "demand_std"]
    stats["demand_std"] = stats["demand_std"].fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stats["cv"] = np.where(stats["demand_mean"] > 0, stats["demand_std"] / stats["demand_mean"], np.inf)
    stats["xyz"] = np.where(
        stats["cv"] <= x_max,
        "X",
        np.where(stats["cv"] <= y_max, "Y", "Z"),
    )
    return stats


def abc_xyz_matrix(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame | None = None,
    a_threshold: float = 0.70,
    b_threshold: float = 0.90,
    x_max: float = 0.5,
    y_max: float = 1.0,
) -> pd.DataFrame:
    """
    Combined ABC × XYZ segmentation per SKU.
    Returns DataFrame: sku_id, total_revenue, cum_revenue_share, abc,
                       demand_mean, demand_std, cv, xyz, segment (e.g. 'AX').
    """
    abc = abc_segment(weekly_panel, a_threshold=a_threshold, b_threshold=b_threshold)
    xyz = xyz_segment(weekly_panel, x_max=x_max, y_max=y_max)
    merged = abc.merge(xyz, on="sku_id", how="outer")
    merged["segment"] = merged["abc"].fillna("C") + merged["xyz"].fillna("Z")
    if sku_master is not None:
        keep = [c for c in ("sku_id", "category", "brand") if c in sku_master.columns]
        merged = merged.merge(sku_master[keep], on="sku_id", how="left")
    return merged.sort_values("total_revenue", ascending=False).reset_index(drop=True)


def pareto_summary(
    weekly_panel: pd.DataFrame,
    revenue_col: str = "revenue",
    sku_col: str = "sku_id",
    top_share: float = 0.20,
) -> dict[str, float]:
    """
    Compute Pareto skew: revenue share of the top `top_share` fraction of SKUs.
    Returns dict with n_skus, top_sku_count, top_revenue_share.
    """
    rev = weekly_panel.groupby(sku_col, observed=True)[revenue_col].sum().sort_values(ascending=False)
    total = rev.sum()
    n = len(rev)
    k = max(1, int(round(n * top_share)))
    return {
        "n_skus": int(n),
        "top_sku_count": k,
        "top_revenue_share": float(rev.iloc[:k].sum() / total) if total > 0 else 0.0,
    }
