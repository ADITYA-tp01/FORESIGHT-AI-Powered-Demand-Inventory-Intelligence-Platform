"""Training frame preparation and model fitting for the global forecaster."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from foresight.features.feature_pipeline import TARGET_COL, feature_matrix_columns
from foresight.forecasting.models import build_regressor


def longest_lag(feature_names: list[str]) -> int:
    """Largest lag_<k> among feature names; the warm-up filter for training."""
    lags = [int(name.split("_", 1)[1]) for name in feature_names if name.startswith("lag_")]
    if not lags:
        raise ValueError("No lag_* features found; cannot filter warm-up rows")
    return max(lags)


def full_history_mask(matrix: pd.DataFrame) -> pd.Series:
    """True for rows whose longest lag is observable (complete history)."""
    longest = longest_lag(feature_matrix_columns(matrix))
    return matrix[f"lag_{longest}"].notna()


def training_frame(
    matrix: pd.DataFrame,
    train_end_week: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Training rows = full-history rows up to (and including) `train_end_week`."""
    mask = full_history_mask(matrix)
    if train_end_week is not None:
        mask = mask & (matrix["year_week"] <= train_end_week)
    frame = matrix.loc[mask]
    if frame.empty:
        raise ValueError(f"No training rows (train_end_week={train_end_week})")
    feature_names = feature_matrix_columns(matrix)
    X = frame.loc[:, feature_names]
    y = frame[TARGET_COL].astype("float64")
    return X, y, feature_names


def train_model(
    matrix: pd.DataFrame,
    model_cfg: dict[str, Any],
    train_end_week: str | None = None,
    quantile: float | None = None,
) -> tuple[HistGradientBoostingRegressor, list[str], dict[str, Any]]:
    """Fit one global model; returns (model, feature_names, fit_stats)."""
    X, y, feature_names = training_frame(matrix, train_end_week)
    model = build_regressor(model_cfg, feature_names, quantile=quantile)
    started = time.perf_counter()
    model.fit(X, y)
    fit_stats = {
        "train_rows": int(len(X)),
        "train_time_s": round(time.perf_counter() - started, 2),
        "n_iter": int(model.n_iter_),
        "quantile": quantile,
    }
    return model, feature_names, fit_stats
