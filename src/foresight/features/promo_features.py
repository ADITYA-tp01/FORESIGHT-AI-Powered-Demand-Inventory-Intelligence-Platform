"""Promotion features: ex-ante promo calendar signals per SKU-week.

Promotions are scheduled in advance, so week-`t` promo state is known when the
forecast for week `t` is made. Discount pct and flags are join keys only —
no concurrent sales quantities are used.
"""

from __future__ import annotations

import pandas as pd


def _slug(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def promo_feature_columns(
    promo_types: list[str],
    target_levels: list[str],
) -> tuple[list[str], list[str]]:
    """Return (type one-hot column names, target one-hot column names)."""
    type_cols = [f"promo_type_{_slug(t)}" for t in promo_types]
    target_cols = [f"promo_target_{_slug(t)}" for t in target_levels]
    return type_cols, target_cols


def _observed_vocab(column: pd.Series) -> set[str]:
    observed: set[str] = set()
    for cell in column.dropna().unique():
        observed.update(part for part in str(cell).split(";") if part)
    return observed


def build_promo_features(
    df: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    promo_types: list[str],
    target_levels: list[str],
) -> pd.DataFrame:
    """Merge is_promo_active, discount pct, type/target one-hots onto the panel."""
    unknown_types = _observed_vocab(promo_weekly["promo_type"]) - set(promo_types)
    unknown_targets = _observed_vocab(promo_weekly["promo_target_level"]) - set(target_levels)
    if unknown_types:
        raise ValueError(f"Unlisted promo_type values: {sorted(unknown_types)}")
    if unknown_targets:
        raise ValueError(f"Unlisted promo_target_level values: {sorted(unknown_targets)}")

    type_cols, target_cols = promo_feature_columns(promo_types, target_levels)
    pw = promo_weekly.copy()
    type_sets = pw["promo_type"].fillna("").str.split(";")
    target_sets = pw["promo_target_level"].fillna("").str.split(";")
    for promo_type, col in zip(promo_types, type_cols, strict=True):
        pw[col] = [int(promo_type in parts) for parts in type_sets]
    for level, col in zip(target_levels, target_cols, strict=True):
        pw[col] = [int(level in parts) for parts in target_sets]

    keep = ["year_week", "sku_id", "is_promo_active", "promo_discount_pct", *type_cols, *target_cols]
    out = df.merge(pw[keep], on=["year_week", "sku_id"], how="left")
    active = pd.to_numeric(out["is_promo_active"], errors="coerce").fillna(0)
    out["is_promo_active"] = active.astype("int8")
    discount = pd.to_numeric(out["promo_discount_pct"], errors="coerce").fillna(0.0)
    out["promo_discount_pct"] = discount.astype("float32")
    for col in [*type_cols, *target_cols]:
        numeric = pd.to_numeric(out[col], errors="coerce").fillna(0)
        out[col] = numeric.astype("int8")
    return out
