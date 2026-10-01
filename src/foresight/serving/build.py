"""Pre-aggregated serving layer (docs/Plan.md 7.1).

The dashboard never scans the 1.05M-row weekly panel at request time. This
module compiles lightweight parquet stores under `artifacts/serving/`:

    dashboard_kpis.parquet     portfolio KPIs + category breakdown (long format)
    decision_grid.parquet      5,000 SKUs x 1 row: quadrant, rupee impact, stock
    forecast_summary.parquet   8-week horizon point/interval + seasonal-naive
    history_weekly.parquet     category x week units for portfolio trajectories
    promo_windows.parquet      active promo sku-weeks for chart shading
    history/<category>.parquet per-SKU weekly history partitioned by category
    serving_manifest.json      file inventory for health checks / navigation
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from foresight.forecasting.baseline import generate_baseline_forecasts
from foresight.utils.io import ensure_dir

QUADRANTS = ("healthy", "markdown_clear", "reorder_now", "watch_volatile")

GRID_COLUMNS = [
    "sku_id",
    "sku_name",
    "category",
    "subcategory",
    "brand",
    "unit_price",
    "cost_price",
    "scored",
    "quadrant",
    "stockout_risk",
    "wos",
    "stockout_high",
    "overstock_high",
    "slow_mover_high",
    "n_stores",
    "stock_on_hand",
    "on_order_units",
    "reorder_point",
    "safety_stock",
    "lead_time_days",
    "lead_time_weeks",
    "last_restock_date",
    "restock_age_days",
    "forward_weekly_mean",
    "forward_8w_demand",
    "ltd",
    "pab",
    "rupees_at_risk",
    "locked_capital",
    "rupee_impact",
]


def slugify(name: str) -> str:
    """Filesystem-safe token for a category name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "unknown"


def build_decision_grid(
    risk_scores: pd.DataFrame,
    sku_master: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """One row per master SKU (5,000); SKUs without stock rows stay visible
    with `scored=False` and quadrant `not_scored` (never fabricated risk)."""
    master = sku_master.loc[
        :, ["sku_id", "sku_name", "category", "subcategory", "brand", "unit_price", "cost_price"]
    ].drop_duplicates(subset="sku_id")
    forward = forecast.groupby("sku_id", observed=True)["forecast"].sum().rename(
        "forward_8w_demand"
    )
    grid = master.merge(forward, on="sku_id", how="left")
    grid["forward_8w_demand"] = grid["forward_8w_demand"].fillna(0.0)

    risk_cols = [c for c in risk_scores.columns if c not in {
        "sku_id", "category", "snapshot_date", "forward_8w_demand",
        "unit_price", "cost_price",
    }]
    grid = grid.merge(risk_scores.loc[:, ["sku_id", *risk_cols]], on="sku_id", how="left")

    grid["scored"] = grid["stockout_risk"].notna()
    grid["quadrant"] = grid["quadrant"].fillna("not_scored")
    for col in ("stockout_high", "overstock_high", "slow_mover_high"):
        # NaN (unscored SKUs) reads as not flagged; `scored` carries coverage.
        grid[col] = grid[col].eq(True)
    for col in ("rupees_at_risk", "locked_capital"):
        grid[col] = grid[col].fillna(0.0)
    grid["rupee_impact"] = grid["rupees_at_risk"] + grid["locked_capital"]
    grid["forward_weekly_mean"] = grid["forward_weekly_mean"].fillna(0.0)
    grid["forward_8w_demand"] = grid["forward_8w_demand"].fillna(0.0)
    return grid.loc[:, GRID_COLUMNS].sort_values("sku_id", ignore_index=True)


def build_dashboard_kpis(
    grid: pd.DataFrame, forecast: pd.DataFrame, cv_summary: dict[str, Any]
) -> pd.DataFrame:
    """Long-format KPI store: section, category, metric, value."""
    ml = cv_summary["ml"]["overall_wape"]
    baseline = cv_summary["comparison"]["baseline_overall_wape"]
    rows: list[dict[str, Any]] = []

    def add_kpi(metric: str, value: float) -> None:
        rows.append({"section": "kpi", "category": "", "metric": metric, "value": float(value)})

    add_kpi("active_skus", len(grid))
    add_kpi("scored_skus", int(grid["scored"].sum()))
    add_kpi("forecast_8w_units", forecast["forecast"].sum())
    add_kpi("stockout_rupee_exposure", grid["rupees_at_risk"].sum())
    add_kpi("locked_capital", grid["locked_capital"].sum())
    add_kpi("reorder_now_skus", int((grid["quadrant"] == "reorder_now").sum()))
    add_kpi("markdown_clear_skus", int((grid["quadrant"] == "markdown_clear").sum()))
    add_kpi("ml_wape", ml)
    add_kpi("baseline_wape", baseline)
    add_kpi("wape_reduction_pct", (baseline - ml) / baseline * 100.0)

    category_metrics = {
        "n_skus": grid.groupby("category", observed=True)["sku_id"].size(),
        "scored_skus": grid[grid["scored"]].groupby("category", observed=True)["sku_id"].size(),
        "forecast_8w_units": grid.groupby("category", observed=True)["forward_8w_demand"].sum(),
        "rupees_at_risk": grid.groupby("category", observed=True)["rupees_at_risk"].sum(),
        "locked_capital": grid.groupby("category", observed=True)["locked_capital"].sum(),
        "stockout_high_count": grid[grid["stockout_high"]].groupby(
            "category", observed=True
        )["sku_id"].size(),
    }
    for quadrant in QUADRANTS:
        category_metrics[f"{quadrant}_count"] = grid[grid["quadrant"] == quadrant].groupby(
            "category", observed=True
        )["sku_id"].size()

    for metric, series in category_metrics.items():
        for category, value in series.items():
            rows.append(
                {"section": "category", "category": category, "metric": metric, "value": float(value)}
            )
    return pd.DataFrame(rows, columns=["section", "category", "metric", "value"])


def build_forecast_summary(
    forecast: pd.DataFrame, baseline: pd.DataFrame, sku_master: pd.DataFrame
) -> pd.DataFrame:
    """8-week horizon with intervals + seasonal-naive comparison line."""
    cats = sku_master.loc[:, ["sku_id", "category"]].drop_duplicates(subset="sku_id")
    out = forecast.merge(cats, on="sku_id", how="left")
    base = baseline.rename(columns={"forecast": "baseline"})
    out = out.merge(base, on=["sku_id", "year_week"], how="left", validate="one_to_one")
    out["baseline"] = out["baseline"].fillna(0.0)
    out["tier_used"] = out["tier_used"].fillna(3).astype(int)
    ordered = ["sku_id", "category", "year_week", "forecast", "forecast_q10", "forecast_q90", "baseline", "tier_used"]
    return out.loc[:, ordered].sort_values(["sku_id", "year_week"], ignore_index=True)


def build_history_weekly(weekly: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Category x week units for portfolio trajectories (tiny, ~2.5K rows)."""
    cats = sku_master.loc[:, ["sku_id", "category"]].drop_duplicates(subset="sku_id")
    joined = weekly.loc[:, ["sku_id", "year_week", "units_sold"]].merge(
        cats, on="sku_id", how="left"
    )
    return (
        joined.groupby(["category", "year_week"], observed=True)["units_sold"]
        .sum()
        .reset_index()
        .sort_values(["category", "year_week"], ignore_index=True)
    )


def write_history_partitions(
    weekly: pd.DataFrame, sku_master: pd.DataFrame, serving_dir: Path
) -> dict[str, str]:
    """Per-category weekly history files; returns category -> relative path."""
    cats = sku_master.loc[:, ["sku_id", "category"]].drop_duplicates(subset="sku_id")
    joined = weekly.loc[:, ["sku_id", "year_week", "units_sold"]].merge(
        cats, on="sku_id", how="inner"
    )
    ensure_dir(serving_dir / "history")
    mapping: dict[str, str] = {}
    used_slugs: set[str] = set()
    for category, frame in joined.groupby("category", observed=True, sort=True):
        slug = slugify(str(category))
        while slug in used_slugs:
            slug = f"{slug}_{len(used_slugs)}"
        used_slugs.add(slug)
        relative = f"history/{slug}.parquet"
        frame = frame.loc[:, ["sku_id", "year_week", "units_sold"]].sort_values(
            ["sku_id", "year_week"], ignore_index=True
        )
        frame.to_parquet(serving_dir / relative, index=False, compression="snappy")
        mapping[str(category)] = relative
    return mapping


def build_promo_windows(promo_weekly: pd.DataFrame) -> pd.DataFrame:
    """Active promo sku-weeks only, for chart shading."""
    active = promo_weekly.loc[promo_weekly["is_promo_active"] == 1]
    keep = ["sku_id", "year_week", "promo_discount_pct", "promo_type"]
    return active.loc[:, keep].sort_values(["sku_id", "year_week"], ignore_index=True)


def build_serving_layer(
    settings: Any,
    weekly: pd.DataFrame,
    sku_master: pd.DataFrame,
    promo_weekly: pd.DataFrame,
    risk_scores: pd.DataFrame,
    forecast: pd.DataFrame,
    cv_summary: dict[str, Any],
    baseline: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compile every serving store; returns a summary with row counts."""
    serving_dir = ensure_dir(settings.serving_dir)

    if baseline is None:
        future_weeks = sorted(forecast["year_week"].unique())
        baseline = generate_baseline_forecasts(
            weekly, sku_master, future_weeks=future_weeks
        )

    grid = build_decision_grid(risk_scores, sku_master, forecast)
    kpis = build_dashboard_kpis(grid, forecast, cv_summary)
    summary = build_forecast_summary(forecast, baseline, sku_master)
    history = build_history_weekly(weekly, sku_master)
    promos = build_promo_windows(promo_weekly)
    partitions = write_history_partitions(weekly, sku_master, serving_dir)

    stores = {
        "dashboard_kpis.parquet": kpis,
        "decision_grid.parquet": grid,
        "forecast_summary.parquet": summary,
        "history_weekly.parquet": history,
        "promo_windows.parquet": promos,
    }
    files: dict[str, dict[str, Any]] = {}
    for name, frame in stores.items():
        path = serving_dir / name
        frame.to_parquet(path, index=False, compression="snappy")
        files[name] = {"rows": int(len(frame)), "bytes": path.stat().st_size}

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "history_partitions": partitions,
        "forecast_weeks": [
            str(forecast["year_week"].min()),
            str(forecast["year_week"].max()),
        ],
        "active_skus": int(len(grid)),
        "scored_skus": int(grid["scored"].sum()),
    }
    manifest_path = serving_dir / "serving_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "serving_dir": str(serving_dir),
        "files": {name: meta["rows"] for name, meta in files.items()},
        "history_partitions": len(partitions),
        "active_skus": manifest["active_skus"],
        "scored_skus": manifest["scored_skus"],
        "manifest": str(manifest_path),
    }
