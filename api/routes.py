"""HTTP endpoints (docs/Plan.md 8.2): health, forecast, risk, batch risk."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from foresight import __version__
from foresight.config import get_settings

from .schemas import (
    BatchRiskRequest,
    BatchRiskResponse,
    ForecastResponse,
    HealthResponse,
    RiskResponse,
    WeeklyForecastItem,
)
from .store import ServingStore, UnknownSkuError, get_store, iso_week_monday

router = APIRouter()

StoreDep = Annotated[ServingStore, Depends(get_store)]

_UNSCORED_ACTION = "No inventory data — excluded from risk scoring."


def _quadrant_action(quadrant: str) -> str:
    quadrants = get_settings().risk.get("quadrants", {})
    if quadrant in quadrants:
        return str(quadrants[quadrant])
    return _UNSCORED_ACTION


def _risk_response(row: pd.Series) -> RiskResponse:
    quadrant = str(row["quadrant"])
    return RiskResponse(
        sku_id=str(row["sku_id"]),
        quadrant=quadrant,
        recommended_action=_quadrant_action(quadrant),
        stockout_risk_score=(
            None if pd.isna(row["stockout_risk"]) else float(row["stockout_risk"])
        ),
        weeks_of_supply=None if pd.isna(row["wos"]) else float(row["wos"]),
        sales_at_risk_rupees=float(row["rupees_at_risk"]),
        locked_capital_rupees=float(row["locked_capital"]),
    )


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health(store: StoreDep) -> HealthResponse:
    artifact = get_settings().artifacts_dir / "forecasts" / "forecast_8w.parquet"
    if artifact.exists():
        stamp = datetime.fromtimestamp(artifact.stat().st_mtime, tz=timezone.utc).isoformat()
    else:
        stamp = str(store.manifest["generated_at"])
    return HealthResponse(
        status="ok",
        version=__version__,
        phase="6",
        model_artifact_timestamp=stamp,
        serving_generated_at=str(store.manifest["generated_at"]),
    )


@router.get("/forecast/{sku_id}", response_model=ForecastResponse, tags=["forecast"])
def forecast(
    sku_id: str,
    store: StoreDep,
    horizon_weeks: Annotated[int, Query(ge=1, le=12)] = 8,
) -> ForecastResponse:
    rows = store.forecast_frame(sku_id)
    if rows.empty:
        raise UnknownSkuError([sku_id])
    items = [
        WeeklyForecastItem(
            week_start=iso_week_monday(row.year_week).isoformat(),
            forecast_units=float(row.forecast),
            lower_bound=float(row.forecast_q10),
            upper_bound=float(row.forecast_q90),
        )
        for row in rows.head(horizon_weeks).itertuples(index=False)
    ]
    return ForecastResponse(
        sku_id=sku_id,
        model_version=__version__,
        generated_at=str(store.manifest["generated_at"]),
        forecast=items,
    )


@router.get("/risk/{sku_id}", response_model=RiskResponse, tags=["risk"])
def risk(sku_id: str, store: StoreDep) -> RiskResponse:
    row = store.risk_row(sku_id)
    if row is None:
        raise UnknownSkuError([sku_id])
    return _risk_response(row)


@router.post("/batch/risk", response_model=BatchRiskResponse, tags=["risk"])
def batch_risk(
    payload: BatchRiskRequest,
    store: StoreDep,
) -> BatchRiskResponse:
    requested = list(dict.fromkeys(payload.sku_ids))
    max_batch = get_settings().api.max_batch_skus
    if len(requested) > max_batch:
        raise HTTPException(
            status_code=422,
            detail={"error": "batch_too_large", "max_batch_skus": max_batch},
        )
    items: list[RiskResponse] = []
    missing: list[str] = []
    for sku_id in requested:
        row = store.risk_row(sku_id)
        if row is None:
            missing.append(sku_id)
        else:
            items.append(_risk_response(row))
    if missing:
        raise UnknownSkuError(missing)
    return BatchRiskResponse(items=items)
