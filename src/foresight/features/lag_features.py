"""Autoregressive lag features on the dense weekly panel.

Every lag column is a pure shift within a SKU's own history, so features for
week `t` only ever read actuals from weeks `t-1, ..., t-lag` (never `t`).
"""

from __future__ import annotations

import pandas as pd


def build_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add `lag_<k>` columns for each k in `lags` (grouped by sku_id).

    `df` must already be sorted by (sku_id, year_week); shift is per-SKU.
    """
    if not lags:
        return df
    grouped = df.groupby("sku_id", observed=True, sort=False)["units_sold"]
    for k in lags:
        if k < 1:
            raise ValueError(f"Lag must be >= 1, got {k}")
        df[f"lag_{k}"] = grouped.shift(int(k))
    return df
