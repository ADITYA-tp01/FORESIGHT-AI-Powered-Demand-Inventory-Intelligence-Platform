"""Trailing rolling statistics over strictly past demand.

All windows are computed on `shift(1)` demand (the week before the row's own
week), so rolling features for week `t` read only weeks `t-1, ..., t-window`.
"""

from __future__ import annotations

import pandas as pd


def build_rolling_features(
    df: pd.DataFrame,
    windows: list[int],
    stats: list[str],
    zero_window: int,
) -> pd.DataFrame:
    """Add `roll_<w>_<stat>` columns and `zero_freq_<zero_window>`.

    The zero-frequency feature is the share of zero-demand weeks in the
    trailing `zero_window` (again on shifted demand).
    """
    allowed = {"mean", "std", "min", "max", "median", "sum"}
    unknown = set(stats) - allowed
    if unknown:
        raise ValueError(f"Unsupported rolling stats: {sorted(unknown)}")

    sku = df["sku_id"]
    lag1 = df.groupby("sku_id", observed=True, sort=False)["units_sold"].shift(1)
    by = lag1.groupby(sku, sort=False)

    for w in windows:
        if w < 1:
            raise ValueError(f"Rolling window must be >= 1, got {w}")
        rolling = by.rolling(int(w), min_periods=int(w))
        for stat in stats:
            df[f"roll_{w}_{stat}"] = getattr(rolling, stat)().droplevel(0)

    zero_ind = (lag1 == 0).astype("float64").where(lag1.notna())
    df[f"zero_freq_{zero_window}"] = (
        zero_ind.groupby(sku, sort=False)
        .rolling(int(zero_window), min_periods=int(zero_window))
        .mean()
        .droplevel(0)
    )
    return df
