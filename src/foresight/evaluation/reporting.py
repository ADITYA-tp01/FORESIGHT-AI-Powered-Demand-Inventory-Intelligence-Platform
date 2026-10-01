"""Deliverable D2: EDA insight memo + baseline WAPE scorecard generation."""

from __future__ import annotations

import pandas as pd

from foresight.evaluation.segmentation import abc_xyz_matrix, pareto_summary


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _monthly_units(weekly_panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    """Aggregate weekly units to ISO months via the calendar week->month map."""
    week_month = calendar[["year_week", "month"]].drop_duplicates("year_week")
    merged = weekly_panel.merge(week_month, on="year_week", how="left")
    return merged.groupby("month", observed=True)["units_sold"].sum().sort_values(ascending=False)


def build_eda_report(
    weekly_panel: pd.DataFrame,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory: pd.DataFrame | None,
    backtest_summary: dict | None,
    settings,
) -> str:
    """Assemble reports/eda_report.md (Deliverable D2)."""
    n_skus = int(weekly_panel["sku_id"].nunique())
    n_weeks = int(weekly_panel["year_week"].nunique())
    total_units = int(weekly_panel["units_sold"].sum())
    total_revenue = float(weekly_panel["revenue"].sum())

    pareto = pareto_summary(weekly_panel)
    seg = abc_xyz_matrix(weekly_panel, sku_master)

    abc_counts = seg["abc"].value_counts().to_dict()
    xyz_counts = seg["xyz"].value_counts().to_dict()
    segment_counts = seg["segment"].value_counts().to_dict()

    zero_share = float((weekly_panel["units_sold"] == 0).mean())

    monthly = _monthly_units(weekly_panel, calendar)
    top_month = int(monthly.index[0]) if len(monthly) else 0
    bottom_month = int(monthly.index[-1]) if len(monthly) else 0
    top_month_units = int(monthly.iloc[0]) if len(monthly) else 0
    bottom_month_units = int(monthly.iloc[-1]) if len(monthly) else 0

    promo_txn_share = None
    promo_week_share = None
    if "promo_trans_count" in weekly_panel.columns:
        n_promo_txn = int(weekly_panel["promo_trans_count"].sum())
        n_total_txn = int(settings.audit_expectations.n_sales_contaminated)
        promo_txn_share = n_promo_txn / n_total_txn if n_total_txn else None
        with_promo = weekly_panel.loc[
            (weekly_panel["promo_trans_count"] > 0) & (weekly_panel["units_sold"] > 0),
            "units_sold",
        ].sum()
        promo_week_share = float(with_promo / total_units) if total_units else None

    promo_lines = []
    if promo_txn_share is not None:
        promo_lines.append(f"- Promo share of all transactions: **{_pct(promo_txn_share)}**")
    if promo_week_share is not None:
        promo_lines.append(
            f"- Units sold in SKU-weeks with >=1 promo transaction: **{_pct(promo_week_share)}**"
        )
    promo_lines.append(
        "- Overlap precedence rule: highest discount wins (41 overlapping campaign pairs resolved)."
    )
    promo_section = "\n".join(promo_lines)

    inventory_section = "Inventory snapshot unavailable."
    if inventory is not None and not inventory.empty and "stock_on_hand" in inventory.columns:
        mean_stock = float(inventory["stock_on_hand"].mean())
        median_stock = float(inventory["stock_on_hand"].median())
        outages = int((inventory["stock_on_hand"] <= 0).sum())
        inventory_section = (
            f"- Snapshot records: {len(inventory):,}\n"
            f"- Mean stock: {mean_stock:.2f} units (median {median_stock:.1f})\n"
            f"- Store-SKU outages (stock <= 0): {outages:,}\n"
        )

    scorecard = _scorecard(backtest_summary)

    abc_table = "\n".join(
        f"| {k} | {abc_counts.get(k, 0):,} | {_pct(abc_counts.get(k, 0) / n_skus)} |"
        for k in ("A", "B", "C")
    )
    xyz_table = "\n".join(
        f"| {k} | {xyz_counts.get(k, 0):,} | {_pct(xyz_counts.get(k, 0) / n_skus)} |"
        for k in ("X", "Y", "Z")
    )
    top_segments = sorted(segment_counts.items(), key=lambda kv: -kv[1])[:9]
    seg_table = "\n".join(
        f"| {k} | {v:,} | {_pct(v / n_skus)} |" for k, v in top_segments
    )

    return f"""# Deliverable D2 — EDA Insight Memo & Baseline Benchmark

- Project: {settings.system.project_name}
- Grain: weekly × SKU ({n_skus:,} SKUs × {n_weeks} weeks)
- Total units: {total_units:,}
- Total revenue: {total_revenue:,.0f} PKR

## 1. Demand Dynamics

### Pareto Skew
- Top {pareto['top_sku_count']:,} SKUs ({_pct(pareto['top_sku_count'] / n_skus)} of catalog) generate **{_pct(pareto['top_revenue_share'])}** of revenue.

### Sparsity
- Zero-demand SKU-weeks: **{_pct(zero_share)}** of the dense panel.

### Seasonality (units by month)
- Peak month: **month {top_month}** ({top_month_units:,} units)
- Trough month: **month {bottom_month}** ({bottom_month_units:,} units)

## 2. ABC / XYZ Segmentation

### ABC (revenue contribution)
| Class | SKUs | Share |
|---|---|---|
{abc_table}

### XYZ (coefficient of variation)
| Class | SKUs | Share |
|---|---|---|
{xyz_table}

### Combined Segments (top 9)
| Segment | SKUs | Share |
|---|---|---|
{seg_table}

Class meanings: X: CV <= 0.5 (predictable), Y: 0.5 < CV <= 1.0 (moderate), Z: CV > 1.0 (intermittent/lumpy).

## 3. Promotional Activity
{promo_section}

## 4. Inventory Health
{inventory_section}
## 5. Seasonal-Naive Baseline Scorecard (Mandatory Benchmark)

3-tier fallback hierarchy: (1) seasonal-naive t-52w if observed & > 0,
(2) trailing 4-week mean if > 0, (3) category run-rate.

{scorecard}

## Governance
- `sku_inventory_flags.csv` used strictly as out-of-sample evaluation resource; never a feature.
- All validation is rolling-origin temporal (no random shuffling).
- Primary metric: WAPE (division-by-zero safe on zero-demand weeks).
"""


def _scorecard(backtest_summary: dict | None) -> str:
    if not backtest_summary or not backtest_summary.get("folds"):
        return "_Backtest not yet run._"
    lines = [
        "| Fold | Train End | Valid Weeks | Rows | WAPE | Bias | MAE | RMSE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in backtest_summary["folds"]:
        w = f["wape"]
        wape_str = _pct(w) if w == w else "n/a"  # NaN-safe
        lines.append(
            f"| {f['fold']} | {f['train_end_week']} | {f['valid_weeks']} | {f['n_rows']:,} | "
            f"{wape_str} | {f['bias']:+.4f} | {f['mae']:.3f} | {f['rmse']:.3f} |"
        )
    overall = backtest_summary.get("overall_wape")
    if overall is not None and overall == overall:
        lines += ["", f"**Overall pooled WAPE: {_pct(overall)}**"]
    return "\n".join(lines)


def build_baseline_scorecard_md(backtest_summary: dict) -> str:
    """Compact baseline scorecard markdown for reports/."""
    lines = [
        "# Baseline Benchmark Scorecard",
        "",
        f"- Model: `{backtest_summary.get('model', 'seasonal_naive_3tier')}`",
        f"- Generated: {backtest_summary.get('generated_at', 'n/a')}",
        "",
        _scorecard(backtest_summary),
        "",
        "ML models must beat these numbers on identical folds to earn deployment.",
    ]
    return "\n".join(lines) + "\n"


def build_ml_comparison(baseline_summary: dict, ml_summary: dict) -> dict:
    """Fold-by-fold and overall WAPE comparison of ML vs the Phase 2 baseline."""
    baseline_folds = {int(f["fold"]): f for f in baseline_summary["folds"]}
    fold_rows = []
    for fold in ml_summary["folds"]:
        base_wape = float(baseline_folds[int(fold["fold"])]["wape"])
        ml_wape = float(fold["wape"])
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "baseline_wape": base_wape,
                "ml_wape": ml_wape,
                "delta": ml_wape - base_wape,
                "improved": bool(ml_wape < base_wape),
            }
        )
    base_overall = float(baseline_summary["overall_wape"])
    ml_overall = float(ml_summary["overall_wape"])
    reduction_pct = (base_overall - ml_overall) / base_overall * 100.0 if base_overall else float("nan")
    return {
        "baseline_overall_wape": base_overall,
        "ml_overall_wape": ml_overall,
        "overall_wape_delta": ml_overall - base_overall,
        "overall_wape_reduction_pct": reduction_pct,
        "beats_baseline": bool(ml_overall < base_overall),
        "folds": fold_rows,
    }


def build_model_evaluation_md(
    baseline_summary: dict,
    ml_summary: dict,
    comparison: dict,
    settings,
) -> str:
    """Deliverable D3: backtest evaluation memo for reports/model_evaluation.md."""
    model_cfg = settings.forecast["model"]
    feature_cfg = settings.features
    lags = feature_cfg["lags"]["units_sold"]
    windows = feature_cfg["rolling"]["windows_weeks"]
    stats = feature_cfg["rolling"]["stats"]

    fold_lines = [
        "| Fold | Train End | Valid Weeks | Rows | Baseline WAPE | ML WAPE | delta WAPE | ML Bias |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row, fold in zip(comparison["folds"], ml_summary["folds"], strict=True):
        delta = row["delta"]
        fold_lines.append(
            f"| {row['fold']} | {fold['train_end_week']} | {fold['valid_weeks']} | {fold['n_rows']:,} | "
            f"{_pct(row['baseline_wape'])} | {_pct(row['ml_wape'])} | "
            f"{delta * 100:+.2f} pp | {fold['bias']:+.4f} |"
        )

    coverage = ml_summary.get("interval_coverage")
    if coverage:
        interval_section = (
            f"- Nominal level: **{_pct(coverage['nominal'])}** "
            f"(quantiles {model_cfg['quantiles'][0]:.2f} / {model_cfg['quantiles'][1]:.2f})\n"
            f"- Empirical coverage on fold {coverage['fold']}: "
            f"**{_pct(coverage['coverage'])}** over {coverage['n_rows']:,} SKU-weeks\n"
            f"- Mean interval width: {coverage['mean_width']:.1f} units\n"
            f"- Interval models scored the same point-path features as the point model "
            "(no independent recursive paths)."
        )
    else:
        interval_section = "_Interval coverage not evaluated._"

    beats = comparison["beats_baseline"]
    n_improved = sum(1 for f in comparison["folds"] if f["improved"])
    n_folds = len(comparison["folds"])
    reduction = comparison["overall_wape_reduction_pct"]
    relative = (
        f"{reduction:.2f}% relative reduction"
        if reduction >= 0
        else f"{-reduction:.2f}% relative increase"
    )
    verdict = (
        f"The global forecaster **{'beats' if beats else 'does NOT beat'}** the "
        f"seasonal-naive baseline: pooled WAPE moved from "
        f"{_pct(comparison['baseline_overall_wape'])} to "
        f"{_pct(comparison['ml_overall_wape'])} ({relative}), "
        f"improving {n_improved}/{n_folds} folds."
    )

    return f"""# Deliverable D3 — Demand Forecasting Engine Evaluation

- Project: {settings.system.project_name}
- Generated: {ml_summary.get('generated_at', 'n/a')}

## 1. Methodology

- **Model**: global `HistGradientBoostingRegressor` (single model across all SKUs)
  - loss `{model_cfg['loss']}`, learning_rate {model_cfg['learning_rate']},
    max_depth {model_cfg['max_depth']}, max_iter {model_cfg['max_iter']},
    random_state {model_cfg['random_state']}
  - `early_stopping: false` — sklearn's internal split shuffles (Rule 5: rolling-origin only)
  - categorical features: {', '.join(model_cfg.get('categorical_features', []))}
- **Features**: {ml_summary['n_features']} total
  - lags {lags}; trailing rollings {windows} × {stats} on shifted demand + zero-frequency
  - calendar Fourier (week period {feature_cfg['calendar']['fourier_week_period']},
    month period {feature_cfg['calendar']['fourier_month_period']}), holiday flags/proximity
  - promo one-hots, is_promo_active, discount pct; SKU category/subcategory/brand codes,
    unit price, cost price, margin
  - excluded by design: revenue, gross_revenue, promo_trans_count (concurrent actuals)
- **Validation**: 3 rolling-origin folds from `configs/forecast.yaml` (identical to baseline);
  recursive multi-step rollout feeding point predictions into future lags.

## 2. Scorecard — ML vs Seasonal-Naive Baseline

{chr(10).join(fold_lines)}

**Overall pooled WAPE: baseline {_pct(comparison['baseline_overall_wape'])} →
ML {_pct(comparison['ml_overall_wape'])} ({relative})**

## 3. Prediction Intervals (80%)

{interval_section}

## 4. Honest Verdict

{verdict}

## Governance
- No random splits anywhere: rolling-origin temporal validation only (Rule 5).
- `sku_inventory_flags.csv` never enters training or features.
- All numbers computed from `artifacts/metrics/cv_summary.json` (`ml` + `comparison` keys).
"""
