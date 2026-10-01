"""Model construction for the global forecaster.

Phase 3 uses sklearn's HistGradientBoostingRegressor (lightgbm is optional and
not installed). Loss defaults to absolute_error to align with the WAPE/MAE
scoring rule; quantile losses power the 80% prediction intervals.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor

SUPPORTED_BACKENDS = ("hist_gradient_boosting",)


def resolve_categorical_indices(
    feature_names: list[str],
    categorical_names: list[str] | None,
) -> list[int]:
    """Map configured categorical feature names to column indices."""
    if not categorical_names:
        return []
    missing = [name for name in categorical_names if name not in feature_names]
    if missing:
        raise ValueError(f"categorical_features not present in feature matrix: {missing}")
    return [feature_names.index(name) for name in categorical_names]


def build_regressor(
    model_cfg: dict[str, Any],
    feature_names: list[str],
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    """Construct an unfitted HGB regressor from config.

    `quantile` switches the loss to pinball loss for interval models.
    `early_stopping` stays off (Rule 5): sklearn's internal split shuffles.
    """
    backend = model_cfg.get("backend", "hist_gradient_boosting")
    if backend not in SUPPORTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{backend}' is not available; use 'hist_gradient_boosting'."
        )

    categorical = resolve_categorical_indices(feature_names, model_cfg.get("categorical_features"))
    loss = "quantile" if quantile is not None else model_cfg["loss"]
    params: dict[str, Any] = {
        "loss": loss,
        "learning_rate": float(model_cfg["learning_rate"]),
        "max_iter": int(model_cfg["max_iter"]),
        "max_depth": int(model_cfg["max_depth"]),
        "random_state": int(model_cfg["random_state"]),
        "early_stopping": bool(model_cfg.get("early_stopping", False)),
        "categorical_features": categorical or None,
    }
    if quantile is not None:
        params["quantile"] = float(quantile)
    return HistGradientBoostingRegressor(**params)
