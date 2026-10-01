# Deliverable D3 — Demand Forecasting Engine Evaluation

- Project: Project FORESIGHT
- Generated: 2026-09-30T18:04:05.783306+00:00

## 1. Methodology

- **Model**: global `HistGradientBoostingRegressor` (single model across all SKUs)
  - loss `absolute_error`, learning_rate 0.08,
    max_depth 8, max_iter 400,
    random_state 42
  - `early_stopping: false` — sklearn's internal split shuffles (Rule 5: rolling-origin only)
  - categorical features: category_code, subcategory_code, brand_code
- **Features**: 45 total
  - lags [1, 2, 4, 8, 13, 26, 52]; trailing rollings [4, 8, 12] × ['mean', 'std', 'min', 'max'] on shifted demand + zero-frequency
  - calendar Fourier (week period 52,
    month period 12), holiday flags/proximity
  - promo one-hots, is_promo_active, discount pct; SKU category/subcategory/brand codes,
    unit price, cost price, margin
  - excluded by design: revenue, gross_revenue, promo_trans_count (concurrent actuals)
- **Validation**: 3 rolling-origin folds from `configs/forecast.yaml` (identical to baseline);
  recursive multi-step rollout feeding point predictions into future lags.

## 2. Scorecard — ML vs Seasonal-Naive Baseline

| Fold | Train End | Valid Weeks | Rows | Baseline WAPE | ML WAPE | delta WAPE | ML Bias |
|---|---|---|---|---|---|---|---|
| 1 | 2024-W52 | 8 | 40,000 | 38.01% | 30.33% | -7.67 pp | +0.0547 |
| 2 | 2025-W16 | 8 | 40,000 | 35.29% | 26.96% | -8.33 pp | +0.0476 |
| 3 | 2025-W40 | 12 | 60,000 | 39.05% | 33.28% | -5.77 pp | -0.0274 |

**Overall pooled WAPE: baseline 37.76% →
ML 30.81% (18.39% relative reduction)**

## 3. Prediction Intervals (80%)

- Nominal level: **80.00%** (quantiles 0.10 / 0.90)
- Empirical coverage on fold 3: **77.63%** over 60,000 SKU-weeks
- Mean interval width: 19.5 units
- Interval models scored the same point-path features as the point model (no independent recursive paths).

## 4. Honest Verdict

The global forecaster **beats** the seasonal-naive baseline: pooled WAPE moved from 37.76% to 30.81% (18.39% relative reduction), improving 3/3 folds.

## Governance
- No random splits anywhere: rolling-origin temporal validation only (Rule 5).
- `sku_inventory_flags.csv` never enters training or features.
- All numbers computed from `artifacts/metrics/cv_summary.json` (`ml` + `comparison` keys).
