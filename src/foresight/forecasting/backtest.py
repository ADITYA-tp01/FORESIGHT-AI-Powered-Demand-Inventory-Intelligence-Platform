"""ML backtest orchestration: per-fold train/rollout/score with resumable cache.

Long-running fits (minutes each) are checkpointed individually to
`artifacts/models/*.joblib` plus a JSON progress file, so an interrupted
phase-3 run resumes instead of retraining from scratch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from foresight.evaluation.metrics import bias, mae, rmse, wape
from foresight.forecasting.prediction import recursive_forecast
from foresight.forecasting.training import train_model
from foresight.utils.dates import advance_week

ModelBundle = tuple[Any, list[str], dict[str, Any]]


def fold_windows(weekly_panel: pd.DataFrame, folds: list[dict]) -> list[dict]:
    """Resolve 1-based week ordinals to (train_end_label, valid_labels)."""
    all_weeks = sorted(weekly_panel["year_week"].unique())
    n_weeks = len(all_weeks)
    windows = []
    for fold_i, fold in enumerate(folds, start=1):
        train_end_ordinal = int(fold["train_end_week"])
        valid_weeks = int(fold["valid_weeks"])
        if train_end_ordinal < 1 or train_end_ordinal + valid_weeks > n_weeks:
            raise ValueError(
                f"Fold {fold_i} out of range: train_end={train_end_ordinal}, "
                f"valid={valid_weeks}, available={n_weeks}"
            )
        windows.append(
            {
                "fold": fold_i,
                "train_end_label": all_weeks[train_end_ordinal - 1],
                "valid_labels": all_weeks[
                    train_end_ordinal : train_end_ordinal + valid_weeks
                ],
            }
        )
    return windows


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def load_progress(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"models": {}, "folds": {}}


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def get_or_train_model(
    key: str,
    train_fn: Callable[[], ModelBundle],
    progress: dict[str, Any],
    progress_path: Path,
    models_dir: Path,
) -> ModelBundle:
    """Load a cached fit or train + persist it, checkpointing progress."""
    entry = progress["models"].get(key)
    model_file = models_dir / f"{key}.joblib"
    if entry is not None and model_file.exists():
        model = joblib.load(model_file)
        return model, entry["feature_names"], entry

    model, feature_names, stats = train_fn()
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    entry = {
        **stats,
        "feature_names": feature_names,
        "model_file": model_file.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    progress["models"][key] = entry
    save_progress(progress_path, progress)
    return model, feature_names, entry


def _quantile_col(alpha: float) -> str:
    return f"forecast_q{int(round(alpha * 100)):02d}"


def run_ml_backtest(
    matrix: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    feature_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    folds: list[dict],
    progress_path: Path,
    models_dir: Path,
    forecasts_dir: Path,
    quantile_fold: int | None = None,
    quantiles: list[float] | None = None,
) -> dict[str, Any]:
    """Train + backtest the global forecaster across rolling-origin folds.

    The `quantile_fold` fold additionally trains quantile models and reports
    empirical interval coverage from the same point-path rollout.
    """
    progress = load_progress(progress_path)
    windows = fold_windows(weekly_panel, folds)
    quantiles = quantiles or []

    fold_summaries: list[dict[str, Any]] = []
    pooled_true: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    coverage_summary: dict[str, Any] | None = None
    feature_names: list[str] = []

    for window in windows:
        fold_i = window["fold"]
        train_end_label = window["train_end_label"]
        valid_labels = window["valid_labels"]

        point_model, feature_names, point_stats = get_or_train_model(
            key=f"fold_{fold_i}_point",
            train_fn=lambda te=train_end_label: train_model(matrix, model_cfg, te),
            progress=progress,
            progress_path=progress_path,
            models_dir=models_dir,
        )

        quantile_models: dict[float, Any] = {}
        if fold_i == quantile_fold:
            for alpha in quantiles:
                qmodel, _, _ = get_or_train_model(
                    key=f"fold_{fold_i}_{_quantile_col(alpha)}",
                    train_fn=lambda te=train_end_label, a=alpha: train_model(
                        matrix, model_cfg, te, quantile=a
                    ),
                    progress=progress,
                    progress_path=progress_path,
                    models_dir=models_dir,
                )
                quantile_models[alpha] = qmodel

        forecasts = _fold_forecasts(
            weekly_panel,
            calendar,
            promo_weekly,
            sku_master,
            feature_cfg,
            point_model,
            feature_names,
            valid_labels,
            train_end_label,
            quantile_models,
            forecasts_dir / f"fold_{fold_i}_forecasts.parquet",
        )

        actuals = weekly_panel.loc[weekly_panel["year_week"].isin(valid_labels)]
        merged = actuals.merge(
            forecasts, on=["sku_id", "year_week"], how="inner", validate="one_to_one"
        )
        if merged.empty:
            raise ValueError(f"Fold {fold_i}: no rows joined between actuals and forecasts")
        y_true = merged["units_sold"].to_numpy(dtype=float)
        y_pred = merged["forecast"].to_numpy(dtype=float)

        summary = {
            "fold": fold_i,
            "train_end_week": train_end_label,
            "valid_weeks": len(valid_labels),
            "n_rows": int(len(merged)),
            "train_rows": point_stats["train_rows"],
            "train_time_s": point_stats["train_time_s"],
            "n_iter": point_stats["n_iter"],
            **pooled_metrics(y_true, y_pred),
        }

        if fold_i == quantile_fold and quantiles:
            nominal = float(max(quantiles) - min(quantiles))
            lo = merged[_quantile_col(min(quantiles))].to_numpy(dtype=float)
            hi = merged[_quantile_col(max(quantiles))].to_numpy(dtype=float)
            inside = (y_true >= lo) & (y_true <= hi)
            coverage_summary = {
                "fold": fold_i,
                "nominal": nominal,
                "coverage": float(inside.mean()),
                "mean_width": float(np.mean(hi - lo)),
                "n_rows": int(len(merged)),
            }
            summary["coverage"] = coverage_summary["coverage"]

        fold_summaries.append(summary)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        progress["folds"][str(fold_i)] = summary
        save_progress(progress_path, progress)

    all_true = np.concatenate(pooled_true)
    all_pred = np.concatenate(pooled_pred)
    overall = pooled_metrics(all_true, all_pred)
    return {
        "model": f"{model_cfg['backend']}_{model_cfg['loss']}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(feature_names),
        "folds": fold_summaries,
        **{f"overall_{name}": value for name, value in overall.items()},
        "interval_coverage": coverage_summary,
    }


def _fold_forecasts(
    weekly_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    feature_cfg: dict[str, Any],
    point_model: Any,
    feature_names: list[str],
    valid_labels: list[str],
    train_end_label: str,
    quantile_models: dict[float, Any],
    cache_path: Path,
) -> pd.DataFrame:
    """Rollout for one fold, cached as parquet to make reruns cheap."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    forecasts = recursive_forecast(
        weekly_panel=weekly_panel,
        calendar=calendar,
        promo_weekly=promo_weekly,
        sku_master=sku_master,
        feature_cfg=feature_cfg,
        point_model=point_model,
        feature_names=feature_names,
        future_weeks=valid_labels,
        cutoff_week=train_end_label,
        quantile_models=quantile_models,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(cache_path, index=False, compression="snappy")
    return forecasts


def run_production(
    matrix: pd.DataFrame,
    weekly_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    feature_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    quantiles: list[float],
    horizon_weeks: int,
    progress_path: Path,
    models_dir: Path,
    forecasts_dir: Path,
) -> dict[str, Any]:
    """Fit the production bundle (point + quantiles) on full history and
    forecast `horizon_weeks` beyond the panel. Writes:
    - models_dir/global_forecaster.joblib
    - forecasts_dir/forecast_<h>w.parquet
    """
    progress = load_progress(progress_path)

    point_model, feature_names, _ = get_or_train_model(
        key="production_point",
        train_fn=lambda: train_model(matrix, model_cfg),
        progress=progress,
        progress_path=progress_path,
        models_dir=models_dir,
    )
    quantile_models: dict[float, Any] = {}
    for alpha in quantiles:
        qmodel, _, _ = get_or_train_model(
            key=f"production_{_quantile_col(alpha)}",
            train_fn=lambda a=alpha: train_model(matrix, model_cfg, quantile=a),
            progress=progress,
            progress_path=progress_path,
            models_dir=models_dir,
        )
        quantile_models[alpha] = qmodel

    last_week = str(weekly_panel["year_week"].max())
    future_weeks = advance_week(last_week, horizon_weeks)
    forecasts = recursive_forecast(
        weekly_panel=weekly_panel,
        calendar=calendar,
        promo_weekly=promo_weekly,
        sku_master=sku_master,
        feature_cfg=feature_cfg,
        point_model=point_model,
        feature_names=feature_names,
        future_weeks=future_weeks,
        cutoff_week=None,
        quantile_models=quantile_models,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "global_forecaster.joblib"
    joblib.dump(
        {
            "point": point_model,
            "quantiles": quantile_models,
            "feature_names": feature_names,
            "model_cfg": model_cfg,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        bundle_path,
    )
    forecast_path = forecasts_dir / f"forecast_{horizon_weeks}w.parquet"
    forecasts.to_parquet(forecast_path, index=False, compression="snappy")

    result = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": progress["models"]["production_point"]["train_rows"],
        "first_week": future_weeks[0],
        "last_week": future_weeks[-1],
        "horizon_weeks": horizon_weeks,
        "model_file": str(bundle_path),
        "forecast_file": str(forecast_path),
    }
    progress["production"] = result
    save_progress(progress_path, progress)
    return result
