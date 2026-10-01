"""Promotion precedence: highest discount wins, per-SKU-day resolution."""

from __future__ import annotations

import pandas as pd


def resolve_sku_day_promos(promotions: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_skus = sku_master["sku_id"].tolist()
    for _, promo in promotions.sort_values("discount_pct", ascending=False).iterrows():
        target = promo["target_type"]
        value = promo["target_value"]
        if target == "All":
            eligible = all_skus
        elif target == "Category":
            eligible = sku_master.loc[sku_master["category"] == value, "sku_id"].tolist()
        elif target == "Brand":
            eligible = sku_master.loc[sku_master["brand"] == value, "sku_id"].tolist()
        elif target == "SKU":
            eligible = [value] if value in set(all_skus) else []
        else:
            continue
        date_range = pd.date_range(promo["start_date"], promo["end_date"], freq="D")
        for d in date_range:
            for sku in eligible:
                rows.append(
                    {
                        "date": d,
                        "sku_id": sku,
                        "promo_id": promo["promo_id"],
                        "promo_discount_pct": float(promo["discount_pct"]),
                        "promo_type": promo["promo_type"],
                        "promo_target_level": promo["target_type"],
                    }
                )
    resolved = pd.DataFrame(rows)
    if resolved.empty:
        return pd.DataFrame(
            columns=[
                "date", "sku_id", "promo_id", "promo_discount_pct", "promo_type", "promo_target_level"
            ]
        )
    return resolved.sort_values("promo_discount_pct", ascending=False).drop_duplicates(
        subset=["date", "sku_id"], keep="first"
    )


def sku_week_promo_features(
    daily_resolved: pd.DataFrame, active_min_days: int = 3
) -> pd.DataFrame:
    daily = daily_resolved.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year_week"] = daily["date"].apply(lambda d: f"{d.isocalendar()[0]:04d}-W{d.isocalendar()[1]:02d}")
    weekly = daily.groupby(["year_week", "sku_id"], observed=True, dropna=False).agg(
        promo_days=("promo_id", "count"),
        promo_discount_pct=("promo_discount_pct", "mean"),
        promo_type=("promo_type", lambda vals: ";".join(sorted(set(vals)))),
        promo_target_level=("promo_target_level", lambda vals: ";".join(sorted(set(vals)))),
    ).reset_index()
    weekly["is_promo_active"] = (weekly["promo_days"] >= active_min_days).astype("int32")
    return weekly
