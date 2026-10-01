"""Pydantic v2 request/response contracts (docs/Plan.md 8.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    version: str
    phase: str = Field(examples=["6"])
    model_artifact_timestamp: str
    serving_generated_at: str


class ForecastRequest(BaseModel):
    sku_id: str = Field(..., examples=["SKU04321"])
    horizon_weeks: int = Field(8, ge=1, le=12)


class WeeklyForecastItem(BaseModel):
    week_start: str = Field(
        description="Monday date of the forecast week (ISO 8601)",
        examples=["2026-01-05"],
    )
    forecast_units: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    sku_id: str
    model_version: str
    generated_at: str
    forecast: list[WeeklyForecastItem]


class RiskResponse(BaseModel):
    sku_id: str
    quadrant: str = Field(examples=["reorder_now"])
    recommended_action: str
    stockout_risk_score: float | None = Field(
        default=None,
        description="null when the SKU has no inventory snapshot (quadrant not_scored)",
    )
    weeks_of_supply: float | None = None
    sales_at_risk_rupees: float
    locked_capital_rupees: float


class BatchRiskRequest(BaseModel):
    sku_ids: list[str] = Field(..., min_length=1, examples=[["SKU04321", "SKU00007"]])


class BatchRiskResponse(BaseModel):
    items: list[RiskResponse]
