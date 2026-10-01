"""Phase 3 model smoke tests: HGB config, training frame, recursive rollout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.helpers import empty_promo_weekly, mini_sku_master, mini_weekly_panel

from foresight.features.feature_pipeline import build_feature_matrix, feature_matrix_columns
from foresight.forecasting.models import build_regressor, resolve_categorical_indices
from foresight.forecasting.prediction import recursive_forecast
from foresight.forecasting.training import full_history_mask, train_model, training_frame
from foresight.preprocessing.calendar import build_calendar
from foresight.utils.dates import advance_week


@pytest.fixture(scope="module")
def feature_inputs(settings):
    return {
        "panel": mini_weekly_panel(),
        "calendar": build_calendar("2023-01-01", "2024-12-31"),
        "promo": empty_promo_weekly(),
        "master": mini_sku_master(),
        "cfg": settings.features,
    }


@pytest.fixture(scope="module")
def matrix(feature_inputs):
    return build_feature_matrix(
        feature_inputs["panel"],
        feature_inputs["calendar"],
        feature_inputs["promo"],
        feature_inputs["master"],
        feature_inputs["cfg"],
    )


@pytest.fixture(scope="module")
def fast_model_cfg(settings):
    cfg = dict(settings.forecast["model"])
    cfg["max_iter"] = 5
    cfg["max_depth"] = 4
    return cfg


def test_resolve_categorical_indices():
    names = ["lag_1", "category_code", "brand_code"]
    assert resolve_categorical_indices(names, ["category_code", "brand_code"]) == [1, 2]
    with pytest.raises(ValueError, match="not present"):
        resolve_categorical_indices(names, ["missing_code"])


def test_build_regressor_reads_config(settings, matrix):
    names = feature_matrix_columns(matrix)
    cfg = settings.forecast["model"]
    model = build_regressor(cfg, names)
    assert model.early_stopping is False  # Rule 5: no random internal splits
    assert model.loss == "absolute_error"
    assert model.categorical_features == [names.index("category_code"),
                                          names.index("subcategory_code"),
                                          names.index("brand_code")]
    qmodel = build_regressor(cfg, names, quantile=0.10)
    assert qmodel.loss == "quantile"
    assert qmodel.quantile == pytest.approx(0.10)
    with pytest.raises(NotImplementedError):
        build_regressor({**cfg, "backend": "lightgbm"}, names)


def test_training_frame_excludes_warmup_and_respects_cutoff(matrix, settings):
    mask = full_history_mask(matrix)
    assert mask.sum() == matrix["sku_id"].nunique() * (80 - 52)
    X, y, names = training_frame(matrix)
    assert len(X) == len(y) == mask.sum()
    assert "units_sold" not in names

    cutoff = sorted(matrix["year_week"].unique())[60]
    X_cut, y_cut, _ = training_frame(matrix, train_end_week=cutoff)
    assert (matrix.loc[X_cut.index, "year_week"] <= cutoff).all()
    assert len(X_cut) < len(X)
    assert len(X_cut) == len(y_cut)


def test_train_model_smoke(matrix, fast_model_cfg):
    model, names, stats = train_model(matrix, fast_model_cfg, train_end_week="2024-W30")
    assert stats["train_rows"] > 0
    assert stats["n_iter"] == 5
    preds = model.predict(pd.DataFrame(matrix.loc[full_history_mask(matrix), names]).head(20))
    assert len(preds) == 20
    assert np.isfinite(preds).all()


def test_recursive_forecast_shape_and_floor(feature_inputs, fast_model_cfg, matrix):
    model, names, _ = train_model(matrix, fast_model_cfg)
    panel = feature_inputs["panel"]
    last_week = str(panel["year_week"].max())
    future = advance_week(last_week, 3)

    fc = recursive_forecast(
        weekly_panel=panel,
        calendar=feature_inputs["calendar"],
        promo_weekly=feature_inputs["promo"],
        sku_master=feature_inputs["master"],
        feature_cfg=feature_inputs["cfg"],
        point_model=model,
        feature_names=names,
        future_weeks=future,
        cutoff_week=last_week,
        quantile_models={0.10: model, 0.90: model},
    )
    n_skus = panel["sku_id"].nunique()
    assert len(fc) == n_skus * 3
    assert sorted(fc["year_week"].unique()) == sorted(future)
    assert (fc["forecast"] >= 0).all()
    assert (fc["forecast_q10"] >= 0).all()
    assert np.isfinite(fc["forecast"]).all()
    assert (fc["forecast_q10"] <= fc["forecast_q90"]).all()
    # cutoff respected: only future weeks come back
    assert (fc["year_week"] > last_week).all()


def test_recursive_forecast_deterministic(feature_inputs, fast_model_cfg, matrix):
    model, names, _ = train_model(matrix, fast_model_cfg)
    panel = feature_inputs["panel"]
    kwargs = dict(
        weekly_panel=panel,
        calendar=feature_inputs["calendar"],
        promo_weekly=feature_inputs["promo"],
        sku_master=feature_inputs["master"],
        feature_cfg=feature_inputs["cfg"],
        point_model=model,
        feature_names=names,
        future_weeks=advance_week(str(panel["year_week"].max()), 2),
        cutoff_week=str(panel["year_week"].max()),
    )
    a = recursive_forecast(**kwargs)
    b = recursive_forecast(**kwargs)
    pd.testing.assert_frame_equal(a, b)


def test_advance_week_crosses_year_boundary():
    assert advance_week("2024-W52", 2) == ["2025-W01", "2025-W02"]
    assert advance_week("2025-W51", 3) == ["2025-W52", "2026-W01", "2026-W02"]
