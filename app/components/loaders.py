"""Cached data loaders for the Streamlit app (docs/Plan.md 7.1).

Every loader reads a pre-aggregated parquet store — never the 1.05M-row weekly
panel — and is wrapped in `@st.cache_data(ttl=...)` from `configs/config.yaml`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from foresight.config import load_settings  # noqa: E402

_SETTINGS = load_settings()
_TTL = int(_SETTINGS.dashboard["cache_ttl_seconds"])


def settings():
    """Runtime settings (config-driven dashboard knobs)."""
    return _SETTINGS


def _read(name: str) -> Path:
    path = _SETTINGS.serving_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Serving store missing: {path}. Run `python -m foresight.pipeline --phase 5`."
        )
    return path


@st.cache_data(ttl=_TTL)
def manifest() -> dict:
    path = _read("serving_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=_TTL)
def kpis() -> pd.DataFrame:
    return pd.read_parquet(_read("dashboard_kpis.parquet"))


@st.cache_data(ttl=_TTL)
def decision_grid() -> pd.DataFrame:
    return pd.read_parquet(_read("decision_grid.parquet"))


@st.cache_data(ttl=_TTL)
def forecast_summary() -> pd.DataFrame:
    return pd.read_parquet(_read("forecast_summary.parquet"))


@st.cache_data(ttl=_TTL)
def history_weekly() -> pd.DataFrame:
    return pd.read_parquet(_read("history_weekly.parquet"))


@st.cache_data(ttl=_TTL)
def promo_windows() -> pd.DataFrame:
    return pd.read_parquet(_read("promo_windows.parquet"))


@st.cache_data(ttl=_TTL)
def store_risk() -> pd.DataFrame:
    path = _SETTINGS.artifacts_dir / "risk" / "store_risk.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Store risk missing: {path}. Run `python -m foresight.pipeline --phase 4`."
        )
    return pd.read_parquet(path)


@st.cache_data(ttl=_TTL)
def category_history(category: str) -> pd.DataFrame:
    """Per-SKU weekly history partition (only the drilled category loads)."""
    relative = manifest()["history_partitions"].get(category)
    if relative is None:
        raise KeyError(f"No history partition for category {category!r}")
    return pd.read_parquet(_SETTINGS.serving_dir / relative)


def kpi_values() -> pd.Series:
    """Portfolio KPI section indexed by metric name."""
    frame = kpis()
    return frame.loc[frame["section"] == "kpi"].set_index("metric")["value"]


def category_metrics(metric: str) -> pd.Series:
    frame = kpis()
    rows = frame.loc[(frame["section"] == "category") & (frame["metric"] == metric)]
    return rows.set_index("category")["value"].sort_values(ascending=False)
