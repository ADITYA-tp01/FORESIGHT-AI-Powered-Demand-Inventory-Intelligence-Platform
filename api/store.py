"""Read-only access to the Phase 5 serving layer for the scoring API.

The service never touches raw CSVs or retrains (non-negotiable rule 8);
it only serves pre-aggregated parquet from ``artifacts/serving/``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from foresight.config import get_settings


class ServingUnavailableError(RuntimeError):
    """Serving parquet missing; rebuild with ``python -m foresight.pipeline --phase 5``."""


class UnknownSkuError(LookupError):
    """One or more requested SKU ids are absent from the decision grid."""

    def __init__(self, sku_ids: list[str]) -> None:
        self.sku_ids = list(sku_ids)
        super().__init__(f"Unknown SKU id(s): {', '.join(self.sku_ids)}")


def default_serving_dir() -> Path:
    env = os.environ.get("FORESIGHT_SERVING_DIR")
    return Path(env).resolve() if env else get_settings().serving_dir


def iso_week_monday(week_label: str) -> date:
    year, week = map(int, str(week_label).split("-W"))
    return date.fromisocalendar(year, week, 1)


@dataclass(frozen=True, eq=False)
class ServingStore:
    serving_dir: Path
    grid: pd.DataFrame
    forecast: pd.DataFrame
    manifest: dict[str, Any]

    @classmethod
    def load(cls, serving_dir: Path | str) -> ServingStore:
        directory = Path(serving_dir)
        manifest_path = directory / "serving_manifest.json"
        if not manifest_path.exists():
            raise ServingUnavailableError(
                f"Serving layer not found at {directory}; "
                "build it with `python -m foresight.pipeline --phase 5`"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return cls(
            serving_dir=directory,
            grid=pd.read_parquet(directory / "decision_grid.parquet"),
            forecast=pd.read_parquet(directory / "forecast_summary.parquet"),
            manifest=manifest,
        )

    def forecast_frame(self, sku_id: str) -> pd.DataFrame:
        rows = self.forecast.loc[self.forecast["sku_id"] == sku_id]
        return rows.sort_values("year_week").reset_index(drop=True)

    def risk_row(self, sku_id: str) -> pd.Series | None:
        rows = self.grid.loc[self.grid["sku_id"] == sku_id]
        if rows.empty:
            return None
        return rows.iloc[0]


_store: ServingStore | None = None


def get_store() -> ServingStore:
    """Process-cached serving store; tests override via FastAPI dependency_overrides."""
    global _store
    if _store is None:
        _store = ServingStore.load(default_serving_dir())
    return _store
