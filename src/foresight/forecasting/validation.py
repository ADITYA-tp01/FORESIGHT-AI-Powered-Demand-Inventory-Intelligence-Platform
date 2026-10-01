"""Rolling-origin temporal validation for the seasonal-naive baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.evaluation.metrics import bias, evaluate_forecast, mae, rmse, wape
from foresight.forecasting.baseline import generate_baseline_forecasts


def backtest_baseline(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    folds: list[dict],
    seasonality: int = 52,
    trailing_window: int = 4,
) -> pd.DataFrame:
    """
    Run rolling-origin backtest of the seasonal-naive baseline.

    folds: [{"train_end_week": int (1-based ordinal), "valid_weeks": int}, ...]
    Ordinals index the sorted unique week labels of `weekly_panel`.

    Returns one row per fold per SKU: fold, sku_id, n_weeks, wape, bias, mae, rmse
    """
    all_weeks = sorted(weekly_panel["year_week"].unique())
    n_weeks = len(all_weeks)

    rows = []
    for fold_i, fold in enumerate(folds, start=1):
        train_end_ordinal = int(fold["train_end_week"])
        valid_weeks = int(fold["valid_weeks"])
        _check_fold_range(fold_i, train_end_ordinal, valid_weeks, n_weeks)
        train_end_label = all_weeks[train_end_ordinal - 1]
        valid_labels = all_weeks[train_end_ordinal : train_end_ordinal + valid_weeks]

        merged = _fold_predictions(
            weekly_panel, sku_master, train_end_label, valid_labels,
            seasonality=seasonality, trailing_window=trailing_window,
        )
        for sku, grp in merged.groupby("sku_id", observed=True):
            m = evaluate_forecast(grp["units_sold"].to_numpy(), grp["forecast"].to_numpy())
            m["fold"] = fold_i
            m["sku_id"] = sku
            m["n_weeks"] = len(grp)
            rows.append(m)

    return pd.DataFrame(rows)


def _check_fold_range(fold_i: int, train_end: int, valid_weeks: int, n_weeks: int) -> None:
    if train_end < 1 or train_end + valid_weeks > n_weeks:
        raise ValueError(
            f"Fold {fold_i} out of range: train_end={train_end}, "
            f"valid={valid_weeks}, available={n_weeks}"
        )


def _fold_predictions(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    train_end_label: str,
    valid_labels: list[str],
    seasonality: int,
    trailing_window: int,
) -> pd.DataFrame:
    forecasts = generate_baseline_forecasts(
        weekly_panel,
        sku_master,
        horizon=len(valid_labels),
        seasonality=seasonality,
        trailing_window=trailing_window,
        cutoff_week=train_end_label,
        future_weeks=valid_labels,
    )
    actuals = weekly_panel[weekly_panel["year_week"].isin(valid_labels)]
    return actuals.merge(
        forecasts[["sku_id", "year_week", "forecast"]],
        on=["sku_id", "year_week"],
        how="inner",
    )


def portfolio_backtest_summary(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    folds: list[dict],
    seasonality: int = 52,
    trailing_window: int = 4,
) -> dict[str, object]:
    """
    Portfolio-level (pooled) metrics per fold plus overall pooled metrics.

    Pooled WAPE = sum(|err|) / sum(actual) across all SKUs and weeks in a fold.
    """
    all_weeks = sorted(weekly_panel["year_week"].unique())
    n_weeks = len(all_weeks)

    fold_summaries = []
    pooled_true: list = []
    pooled_pred: list = []
    for fold_i, fold in enumerate(folds, start=1):
        train_end_ordinal = int(fold["train_end_week"])
        valid_weeks = int(fold["valid_weeks"])
        _check_fold_range(fold_i, train_end_ordinal, valid_weeks, n_weeks)
        train_end_label = all_weeks[train_end_ordinal - 1]
        valid_labels = all_weeks[train_end_ordinal : train_end_ordinal + valid_weeks]

        merged = _fold_predictions(
            weekly_panel, sku_master, train_end_label, valid_labels,
            seasonality=seasonality, trailing_window=trailing_window,
        )
        y_true = merged["units_sold"].to_numpy(dtype=float)
        y_pred = merged["forecast"].to_numpy(dtype=float)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)

        fold_summaries.append(
            {
                "fold": fold_i,
                "train_end_week": train_end_label,
                "valid_weeks": valid_weeks,
                "n_rows": int(len(merged)),
                "wape": wape(y_true, y_pred),
                "bias": bias(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
            }
        )

    all_true = np.concatenate(pooled_true) if pooled_true else np.array([])
    all_pred = np.concatenate(pooled_pred) if pooled_pred else np.array([])

    return {
        "model": "seasonal_naive_3tier",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "folds": fold_summaries,
        "overall_wape": wape(all_true, all_pred),
        "overall_bias": bias(all_true, all_pred),
        "overall_mae": mae(all_true, all_pred),
        "overall_rmse": rmse(all_true, all_pred),
    }


def run_backtest_and_log(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    folds: list[dict],
    metrics_dir: Path,
    seasonality: int = 52,
    trailing_window: int = 4,
    filename: str = "cv_summary.json",
) -> dict[str, object]:
    """Run portfolio backtest and write the JSON summary to metrics_dir."""
    summary = portfolio_backtest_summary(
        weekly_panel,
        sku_master,
        folds,
        seasonality=seasonality,
        trailing_window=trailing_window,
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / filename).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
