"""Seasonal-Naive baseline with 3-tier fallback hierarchy.

Tier 1: seasonal-naive  forecast(t+h) = actual(t+h-52), only if observed & > 0
Tier 2: trailing mean    forecast(t+h) = mean(actual t-1..t-4), only if > 0
Tier 3: category run-rate forecast     = category_mean * sku_ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foresight.utils.dates import advance_week


def seasonal_naive_forecast(
    history: pd.Series | np.ndarray,
    horizon: int = 8,
    seasonality: int = 52,
) -> np.ndarray:
    """
    Seasonal-Naive: forecast week T+h from actual week T+h-seasonality.

    Returns array of length `horizon`; NaN where history is insufficient.
    """
    values = np.asarray(history, dtype=float)
    n = len(values)
    out = np.full(horizon, np.nan)
    start = n - seasonality
    if start < 0:
        return out
    src = values[start : start + horizon]
    out[: len(src)] = src
    return out


def trailing_mean_fallback(
    history: pd.Series | np.ndarray,
    horizon: int = 8,
    window: int = 4,
) -> float:
    """Tier 2: mean of the trailing `window` observations. NaN if unavailable."""
    values = np.asarray(history, dtype=float)
    if len(values) < window:
        return np.nan
    recent = values[-window:]
    mean = float(np.nanmean(recent))
    if np.isnan(mean) or mean <= 0:
        return np.nan
    return mean


def category_runrate_fallback(category_avg: float, sku_ratio: float) -> float:
    """Tier 3: category_mean * sku_ratio. NaN if not usable."""
    if category_avg is None or sku_ratio is None:
        return np.nan
    if np.isnan(category_avg) or np.isnan(sku_ratio):
        return np.nan
    if category_avg <= 0 or sku_ratio <= 0:
        return np.nan
    return float(category_avg * sku_ratio)


def seasonal_naive_with_fallback(
    history: pd.Series | np.ndarray,
    horizon: int = 8,
    seasonality: int = 52,
    trailing_window: int = 4,
    category_avg: float | None = None,
    sku_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the 3-tier fallback per forecast week.

    Returns
    -------
    (forecast, tiers)
        forecast: float array of length `horizon`
        tiers: int array (1, 2 or 3) recording which tier produced each week
    """
    values = np.asarray(history, dtype=float)
    seasonal = seasonal_naive_forecast(values, horizon, seasonality)
    tier2 = trailing_mean_fallback(values, horizon, trailing_window)
    tier3 = category_runrate_fallback(
        category_avg if category_avg is not None else np.nan,
        sku_ratio if sku_ratio is not None else np.nan,
    )

    forecast = np.zeros(horizon, dtype=float)
    tiers = np.ones(horizon, dtype=int)
    for h in range(horizon):
        s = seasonal[h]
        if not np.isnan(s) and s > 0:
            forecast[h] = s
            tiers[h] = 1
        elif not np.isnan(tier2):
            forecast[h] = tier2
            tiers[h] = 2
        elif not np.isnan(tier3):
            forecast[h] = tier3
            tiers[h] = 3
        else:
            forecast[h] = 0.0
            tiers[h] = 3
    return forecast, tiers


def compute_sku_category_ratios(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    category_col: str = "category",
    value_col: str = "units_sold",
) -> pd.DataFrame:
    """
    Precompute SKU-to-category mean ratios for the Tier 3 fallback.

    Returns DataFrame: sku_id, category, sku_mean, cat_mean, sku_ratio.
    """
    sku_means = weekly_panel.groupby("sku_id", observed=True)[value_col].mean().reset_index()
    sku_means.columns = ["sku_id", "sku_mean"]

    sku_cat = sku_master[["sku_id", category_col]].drop_duplicates()
    sku_cat_means = sku_means.merge(sku_cat, on="sku_id", how="left")
    sku_cat_means[category_col] = sku_cat_means[category_col].fillna("Unknown")

    cat_means = sku_cat_means.groupby(category_col, observed=True)["sku_mean"].mean().reset_index()
    cat_means.columns = [category_col, "cat_mean"]

    ratios = sku_cat_means.merge(cat_means, on=category_col, how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios["sku_ratio"] = ratios["sku_mean"] / ratios["cat_mean"]
    ratios["sku_ratio"] = (
        ratios["sku_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    return ratios[["sku_id", category_col, "sku_mean", "cat_mean", "sku_ratio"]]


def generate_baseline_forecasts(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    horizon: int = 8,
    seasonality: int = 52,
    trailing_window: int = 4,
    cutoff_week: str | None = None,
    future_weeks: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate seasonal-naive baseline forecasts for all SKUs.

    Parameters
    ----------
    weekly_panel : dense weekly panel (year_week, sku_id, units_sold)
    horizon : forecast horizon in weeks
    seasonality : seasonal period (52 weeks)
    trailing_window : Tier 2 window
    cutoff_week : if set, only history <= this week is visible (backtesting)
    future_weeks : explicit forecast week labels (used by backtester for exact
        alignment with validation weeks); derived from cutoff_week if omitted

    Returns
    -------
    DataFrame: sku_id, year_week, forecast, tier_used
    """
    panel = weekly_panel if cutoff_week is None else weekly_panel[weekly_panel["year_week"] <= cutoff_week]

    if future_weeks is None:
        last_week = panel["year_week"].max()
        future_weeks = advance_week(last_week, horizon)

    ratios = compute_sku_category_ratios(panel, sku_master)
    ratio_lookup = ratios.set_index("sku_id")[["sku_ratio", "cat_mean"]].to_dict("index")

    results = []
    ordered = panel.sort_values(["sku_id", "year_week"])
    for sku, series in ordered.groupby("sku_id", observed=True)["units_sold"]:
        meta = ratio_lookup.get(sku, {"sku_ratio": np.nan, "cat_mean": np.nan})
        forecast, tiers = seasonal_naive_with_fallback(
            history=series.to_numpy(dtype=float),
            horizon=horizon,
            seasonality=seasonality,
            trailing_window=trailing_window,
            category_avg=meta["cat_mean"],
            sku_ratio=meta["sku_ratio"],
        )
        sku_tier = int(tiers.max())
        for week, pred in zip(future_weeks, forecast, strict=True):
            results.append(
                {
                    "sku_id": sku,
                    "year_week": week,
                    "forecast": max(0.0, float(pred)),
                    "tier_used": sku_tier,
                }
            )

    return pd.DataFrame(results)
