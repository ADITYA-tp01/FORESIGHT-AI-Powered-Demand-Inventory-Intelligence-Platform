"""Recursive multi-step forecast rollout for the global forecaster.

Each future week is appended to history with the point prediction (clipped at
zero), so lag/rolling features for week t+h see the model's own path rather
than unavailable actuals. Quantile models score the same point-path features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from foresight.features.feature_pipeline import (
    PANEL_INPUT_COLS,
    build_feature_matrix,
)


def recursive_forecast(
    weekly_panel: pd.DataFrame,
    calendar: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    feature_cfg: dict[str, Any],
    point_model: Any,
    feature_names: list[str],
    future_weeks: list[str],
    cutoff_week: str | None = None,
    quantile_models: dict[float, Any] | None = None,
) -> pd.DataFrame:
    """Forecast `future_weeks` for every SKU in the panel.

    Parameters
    ----------
    weekly_panel : dense weekly panel (all history up to cutoff)
    point_model : fitted point-forecast model
    feature_names : training-time feature column order
    future_weeks : explicit week labels to predict, in order
    cutoff_week : history visible to the model (inclusive); None = all
    quantile_models : {alpha: fitted model} for interval columns forecast_qXX

    Returns
    -------
    DataFrame: sku_id, year_week, forecast[, forecast_q10, forecast_q90]
    """
    hist = weekly_panel.loc[:, list(PANEL_INPUT_COLS)].copy()
    if cutoff_week is not None:
        hist = hist.loc[hist["year_week"] <= cutoff_week]
    hist = hist.sort_values(["sku_id", "year_week"], kind="mergesort")
    if hist.empty:
        raise ValueError(f"No history at/before cutoff_week={cutoff_week}")

    last_prices = hist.groupby("sku_id", observed=True, sort=False)["unit_price"].last()
    skus = sorted(hist["sku_id"].unique())
    prices = last_prices.reindex(skus)
    if prices.isna().any():
        master_prices = sku_master.drop_duplicates(subset="sku_id").set_index("sku_id")["unit_price"]
        prices = prices.fillna(pd.Series(skus).map(master_prices).set_axis(skus))
    price_values = prices.to_numpy(dtype="float64")

    results: list[pd.DataFrame] = []
    for week in future_weeks:
        placeholders = pd.DataFrame(
            {
                "sku_id": skus,
                "year_week": week,
                "units_sold": 0.0,
                "unit_price": price_values,
            }
        )
        frame = pd.concat([hist, placeholders], ignore_index=True)
        feats = build_feature_matrix(frame, calendar, promo_weekly, sku_master, feature_cfg)
        rows = feats.loc[feats["year_week"] == week]
        X = rows.loc[:, feature_names]

        out = rows.loc[:, ["sku_id", "year_week"]].copy()
        out["forecast"] = np.maximum(point_model.predict(X), 0.0)
        if quantile_models:
            for alpha, model in quantile_models.items():
                col = f"forecast_q{int(round(alpha * 100)):02d}"
                out[col] = np.maximum(model.predict(X), 0.0)
        results.append(out)

        placeholders["units_sold"] = out["forecast"].to_numpy()
        hist = pd.concat([hist, placeholders], ignore_index=True)
        del frame, feats, rows

    if not results:
        raise ValueError("future_weeks is empty")
    return pd.concat(results, ignore_index=True)
