"""Robust forecast evaluation metrics: WAPE, MAPE, Bias, MAE, RMSE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    
    WAPE = sum(|y_true - y_pred|) / sum(y_true)
    
    Unlike MAPE, WAPE does not divide by individual actuals,
    eliminating division-by-zero errors on intermittent zero-demand weeks.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        WAPE value (0 = perfect, 1 = 100% error)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    denominator = np.sum(y_true)
    if denominator == 0:
        return np.nan
    
    return np.sum(np.abs(y_true - y_pred)) / denominator


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    Only computed on non-zero actuals to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    if not mask.any():
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Forecast Bias.
    
    Bias = sum(y_true - y_pred) / sum(y_true)
    
    Positive bias = under-forecasting (actual > predicted)
    Negative bias = over-forecasting (actual < predicted)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    denominator = np.sum(y_true)
    if denominator == 0:
        return np.nan
    
    return np.sum(y_true - y_pred) / denominator


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sku_id: str | None = None,
) -> dict[str, float]:
    """
    Compute all metrics for a single SKU or aggregated.
    
    Returns dict with: WAPE, MAPE, Bias, MAE, RMSE
    """
    return {
        "wape": wape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def evaluate_by_sku(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    actual_col: str = "units_sold",
    forecast_col: str = "forecast",
    sku_col: str = "sku_id",
    week_col: str = "year_week",
) -> pd.DataFrame:
    """
    Evaluate forecasts per SKU.
    
    Parameters
    ----------
    actuals : pd.DataFrame
        DataFrame with sku_id, year_week, units_sold
    forecasts : pd.DataFrame
        DataFrame with sku_id, year_week, forecast
    
    Returns
    -------
    pd.DataFrame
        Per-SKU metrics with columns: sku_id, WAPE, MAPE, Bias, MAE, RMSE
    """
    merged = actuals.merge(
        forecasts[[sku_col, week_col, forecast_col]],
        on=[sku_col, week_col],
        how="inner",
    )
    
    if merged.empty:
        return pd.DataFrame(columns=[sku_col, "wape", "mape", "bias", "mae", "rmse"])
    
    results = []
    for sku, group in merged.groupby(sku_col):
        metrics = evaluate_forecast(
            group[actual_col].values,
            group[forecast_col].values,
        )
        metrics[sku_col] = sku
        results.append(metrics)
    
    return pd.DataFrame(results)


def evaluate_by_category(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    sku_master: pd.DataFrame,
    actual_col: str = "units_sold",
    forecast_col: str = "forecast",
    sku_col: str = "sku_id",
    week_col: str = "year_week",
    category_col: str = "category",
) -> pd.DataFrame:
    """
    Evaluate forecasts aggregated by category.
    """
    merged = actuals.merge(
        forecasts[[sku_col, week_col, forecast_col]],
        on=[sku_col, week_col],
        how="inner",
    )
    merged = merged.merge(sku_master[[sku_col, category_col]], on=sku_col, how="left")
    
    if merged.empty:
        return pd.DataFrame(columns=[category_col, "wape", "mape", "bias", "mae", "rmse"])
    
    results = []
    for cat, group in merged.groupby(category_col):
        metrics = evaluate_forecast(
            group[actual_col].values,
            group[forecast_col].values,
        )
        metrics[category_col] = cat
        results.append(metrics)
    
    return pd.DataFrame(results)


def aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Aggregate metrics across all SKUs/weeks (portfolio-level).
    """
    return evaluate_forecast(y_true, y_pred)