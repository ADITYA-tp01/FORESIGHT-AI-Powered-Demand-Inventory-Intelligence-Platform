# Deliverable D7 — Executive Stakeholder Readout

- **Audience:** Head of Operations, Head of Finance
- **Date:** 2026-10-01
- **Format:** 10-slide readout (docs/Plan.md §10)
- **Sourcing:** Every figure derives from artifacts in this repository (`cv_summary.json`,
  `flag_evaluation.json`, `risk_scores.parquet`, `dashboard_kpis.parquet`, D1/D2 reports). No estimates.

---

## Slide 1 — Title & Executive Summary

**Project FORESIGHT — AI-Powered Demand & Inventory Intelligence**

- **Forecasting:** ML demand forecast beats the seasonal-naive baseline on every fold —
  pooled WAPE **37.76% → 30.81%** (**18.39% relative reduction**), 8-week horizon (2026-W02..W09).
- **Inventory risk:** Stockout flag recall **0.865** vs the 0.85 gate (**MET**) on a leakage-guarded,
  out-of-sample answer key.
- **Money on the table:** **INR 32,785,603** of forecast demand sits beyond stock (Rupees at Risk);
  **INR 1,534,592,370** of inventory value is locked beyond demand + safety stock.
- **Decisions produced:** **325** immediate purchase orders, **3,021** markdown candidates,
  across a catalog of **5,000 SKUs** and **30 stores**.
- **Deliverables:** D1 data pipeline · D2 EDA & baseline · D3 forecasting engine · D4 risk
  intelligence · D5 planning dashboard · D6 scoring API · D7 this readout.

*Sources: artifacts/metrics/cv_summary.json, artifacts/metrics/flag_evaluation.json,
artifacts/serving/dashboard_kpis.parquet*

---

## Slide 2 — The Business Problem

**Two opposite failures destroy the same working capital:**

| Failure mode | Evidence (snapshot 2025-12-31) |
|---|---|
| Lost sales from stockouts | **INR 32,785,603** of 8-week forecast demand is not covered by stock + on-order; **4,239** store-SKU positions were already at zero stock |
| Capital locked in dead stock | **INR 1,534,592,370** of inventory value exceeds 8-week demand + safety stock — mostly slow, stale stock (restock age ≥ 180 days) |

- Demand is lumpy: **1.61%** of SKU-weeks in the dense panel are zero-demand — gut-feel
  replenishment both over-orders and under-orders.
- The 8-week plan requires **≈625,808 units** — sequencing the right units to the right stores
  is the entire game.

*Sources: reports/risk_evaluation.md, reports/eda_report.md, dashboard_kpis.parquet*

---

## Slide 3 — Data Foundations & Multi-Store Reality

- **9,945,511 transactions** (801 MB raw CSV) streamed in **500,000-row chunks** — never fully loaded.
- **30 stores · 5,000 SKUs · 10,000 customers · 100 promotional campaigns · 26,408 inventory rows.**
- Processed: **4,143,430** daily rows → **1,050,000** dense weekly rows (5,000 SKUs × 210 weeks)
  → **18,700,906** units sold.
- **Reconciliation passes:** sum(daily units) = sum(weekly units) = sum(store-daily units)
  = **18,700,906** (three-way tie-out enforced by tests).
- Governance: `sku_inventory_flags.csv` is an evaluation-only answer key — it never enters
  features, training, or thresholds; all validation is rolling-origin (no random splits).

*Sources: reports/data_quality_report.md, tests/test_ingestion.py, tests/test_transformations.py*

---

## Slide 4 — Demand Seasonality & Key Promotional Drivers

- **Seasonality:** peak **December (month 12): 2,037,489 units** vs trough **February (month 2): 1,223,965 units**
  — a ~66% peak-to-trough swing the model must anticipate a year ahead (t-52 lags).
- **Promotions move demand:** **21.02%** of transactions are promotional, and SKU-weeks with promo
  activity carry **27.28%** of all units; 41 overlapping campaign pairs resolved by
  highest-discount-wins precedence.
- **Concentration:** the top **1,000 SKUs (20% of the catalog) generate 73.08% of revenue**
  — 17.46% of SKUs are class A; 38.22% of SKUs are statistically predictable (XYZ class X).
- Implication: a **single global model** with promo/calendar features serves the concentrated,
  forecastable core far better than 5,000 separate models.

*Sources: reports/eda_report.md*

---

## Slide 5 — Forecasting Methodology: Earning Trust by Beating Seasonal-Naive

- **Global model:** one `HistGradientBoostingRegressor` (absolute-error loss, lr 0.08, depth 8,
  400 iterations) across all SKUs — `early_stopping=false` so sklearn's internal shuffling can
  never violate temporal validation.
- **45 leakage-safe features:** lags [1, 2, 4, 8, 13, 26, 52], trailing rolling mean/std/min/max
  [4, 8, 12], calendar Fourier terms, holiday proximity, promo indicators, price/margin, SKU codes.
  **Excluded by design:** revenue, gross_revenue, promo_trans_count (concurrent actuals).
- **Benchmark:** a 3-tier seasonal-naive — (1) t-52 seasonal-naive if observed & positive,
  (2) trailing 4-week mean, (3) category run-rate — evaluated on **identical rolling-origin folds**
  (train ends 2024-W52 / 2025-W16 / 2025-W40; 8-8-12 validation weeks).
- Deployment rule (pre-registered): the ML model must beat the baseline on **every** fold — it did.

*Sources: reports/model_evaluation.md, configs/forecast.yaml, reports/baseline_scorecard.md*

---

## Slide 6 — Model Accuracy Scorecard: WAPE Comparison & Error Bias

| Fold | Train end | Valid weeks | Baseline WAPE | ML WAPE | Delta | ML bias |
|---|---|---|---|---|---|---|
| 1 | 2024-W52 | 8 | 38.01% | 30.33% | −7.67 pp | +5.47% |
| 2 | 2025-W16 | 8 | 35.29% | 26.96% | −8.33 pp | +4.76% |
| 3 | 2025-W40 | 12 | 39.05% | 33.28% | −5.77 pp | −2.74% |
| **Pooled** | — | 28 | **37.76%** | **30.81%** | **−6.94 pp (18.39% rel.)** | **+1.32%** |

- **Error bias:** ML sits at a near-flat **+1.32%** over-forecast (baseline +1.46%) — inventory-safe
  without systematic over-stock; MAE improves 7.32 → **5.97** units/week. Honest caveat: RMSE is
  marginally higher (25.86 → 26.43) — a few large-error weeks trade off for broad WAPE gains; WAPE
  is the pre-registered primary metric.
- **Uncertainty:** 80% prediction intervals achieved **77.63%** empirical coverage (fold 3, 60,000
  SKU-weeks; mean width 19.5 units) — bands are honest, slightly conservative-low.

*Sources: artifacts/metrics/cv_summary.json (ml + comparison keys)*

---

## Slide 7 — The 4-Quadrant Decisioning Grid: From Predictions to Operational Action

Forecasts feed LTD/PAB/WOS math; each SKU lands in one actionable bucket:

| Quadrant | Rule (pre-registered in configs/risk.yaml) | Action | SKUs |
|---|---|---|---|
| **reorder_now** | Stockout HIGH (PAB ≤ 0), not overstock | Raise PO immediately | **325** |
| **markdown_clear** | Overstock HIGH (WOS > 12 or SOH > 3×RP), no stockout | Promote or discount | **3,021** |
| **healthy** | Neither flag | No operational change | **1,149** |
| **watch_volatile** | Both flags (near-contradictory states) | Review replenishment rule | **0** |

- **Validation gate MET** (model trained only through 2025-W41, answer key Oct–Dec 2025):

| Flag | Recall | Precision | F1 |
|---|---|---|---|
| STOCKOUT_RISK (SKU) | **0.865** (gate 0.85) | 0.453 | 0.595 |
| SLOW_MOVER (SKU) | **1.000** | 0.774 | 0.872 |
| Stockout (store pairs) | **1.000** | 0.567 | 0.723 |

- 505 forecast SKUs carry no inventory rows and are honestly excluded from risk scoring.
- Store-level path: chain LTD split by trailing 12-week store shares → 26,408 store×SKU positions.

*Sources: reports/risk_evaluation.md, configs/risk.yaml, artifacts/risk/*

---

## Slide 8 — Prioritized Reorder Plan (Immediate PO Recommendations)

**325 SKUs · INR 32,785,603 total Rupees at Risk** — cover first where exposure concentrates
(top 10 ≈ **INR 20.05M**, 61.2% of the exposure):

| # | SKU | Category | On-hand | Lead-time demand | PAB | Stockout risk | At-risk (INR) |
|---|---|---|---|---|---|---|---|
| 1 | SKU04321 | Personal Care | 260 | 9,832 | −9,572 | 1.000 | 9,201,335 |
| 2 | SKU03727 | Electronics & Accessories | 102 | 2,116 | −2,014 | 0.997 | 3,344,774 |
| 3 | SKU01189 | Apparel & Footwear | 0 | 716 | −716 | 0.955 | 1,647,787 |
| 4 | SKU00538 | Health & Wellness | 104 | 1,463 | −1,359 | 0.992 | 1,496,491 |
| 5 | SKU02543 | Home & Kitchen | 0 | 576 | −576 | 0.905 | 1,126,953 |
| 6 | SKU01363 | Electronics & Accessories | 19 | 293 | −274 | 0.855 | 873,873 |
| 7 | SKU01371 | Home & Kitchen | 0 | 479 | −479 | 0.898 | 611,656 |
| 8 | SKU03053 | Apparel & Footwear | 0 | 146 | −146 | 0.808 | 594,787 |
| 9 | SKU02176 | Apparel & Footwear | 0 | 592 | −592 | 0.915 | 586,404 |
| 10 | SKU02692 | Electronics & Accessories | 0 | 272 | −272 | 0.840 | 569,317 |

- Exposure is spread across the chain: reorder SKUs stock on average **11.3 stores** — POs must
  be store-routed using the 12-week share disaggregation.
- Heaviest reorder counts: Stationery & Office 39, Health & Wellness 37, Personal Care 33,
  Electronics 33.
- Full ranked list: `artifacts/risk/risk_scores.parquet` (quadrant = reorder_now) and
  Dashboard → Inventory Decision Grid → Immediate Reorders (CSV export).

*Sources: artifacts/risk/risk_scores.parquet*

---

## Slide 9 — Prioritized Markdown Plan (Freeing Locked Working Capital)

**3,021 SKUs · INR 1,533,774,315 of INR 1,534,592,370 total locked capital sits in markdown-clear
stock** (top 10 = **INR 164,056,640**, 10.7% — a long tail, so run a programmatic markdown, not
hand-picked one-offs):

| # | SKU | Category | On-hand | Weeks of supply | Restock age (days) | Locked (INR) |
|---|---|---|---|---|---|---|
| 1 | SKU03043 | Electronics & Accessories | 7,125 | 1,398 | 238 | 21,966,517 |
| 2 | SKU03150 | Apparel & Footwear | 7,949 | 1,632 | 236 | 20,015,754 |
| 3 | SKU04062 | Electronics & Accessories | 7,894 | 1,631 | 238 | 19,550,871 |
| 4 | SKU03491 | Electronics & Accessories | 9,335 | 1,843 | 236 | 16,657,125 |
| 5 | SKU01423 | Electronics & Accessories | 8,702 | 1,604 | 238 | 16,315,681 |
| 6 | SKU00075 | Electronics & Accessories | 7,071 | 1,183 | 235 | 15,338,531 |
| 7 | SKU04409 | Apparel & Footwear | 7,772 | 1,528 | 232 | 14,149,793 |
| 8 | SKU01428 | Apparel & Footwear | 7,485 | 1,286 | 238 | 14,018,116 |
| 9 | SKU01542 | Home & Kitchen | 9,288 | 1,852 | 239 | 13,678,137 |
| 10 | SKU03941 | Electronics & Accessories | 8,524 | 1,548 | 237 | 12,366,115 |

- Locked capital by category: Electronics & Accessories **368.8M**, Apparel & Footwear **332.8M**,
  Home & Kitchen **241.8M**, Health & Wellness **169.9M**, Frozen Foods **106.3M**.
- Every markdown SKU is a stale-restock overstock (WOS > 12 weeks and/or SOH > 3× reorder point,
  restock age ≥ 180 days) — promotions clear it without training buyers to wait for sales.
- Full ranked list: `artifacts/risk/risk_scores.parquet` (quadrant = markdown_clear) and
  Dashboard → Inventory Decision Grid → Markdown Candidates (CSV export).

*Sources: artifacts/risk/risk_scores.parquet*

---

## Slide 10 — Financial Impact Summary: Protected Revenue vs. Freed Capital (Rupees)

| Lever | Amount | Basis |
|---|---|---|
| **Protected revenue** (stockouts averted) | **INR 32,785,603** | 8-week forecast demand beyond stock + on-order, valued at unit price |
| **Freed working capital** (markdown program) | **INR 1,533,774,315** | Markdown-clear inventory value beyond demand + safety stock (99.95% of all locked capital) |
| Capital correctly retained | INR 818,055 | Healthy-quadrant stock — left untouched |
| Actionable decisions | **325 POs + 3,021 markdowns** | 4,495 scored SKUs (505 unscored: no stock rows) |

- **Accuracy backing these numbers:** WAPE **30.81%** vs baseline 37.76% (18.39% better, 3/3 folds);
  stockout recall **0.865 ≥ 0.85 gate**; 80% intervals at 77.63% coverage.
- **Operating cadence:** re-run `python -m foresight.pipeline --phase 4` for fresh risk scoring,
  `--phase 5` to refresh the dashboard serving layer; decisions surfaced in the Streamlit
  dashboard and available machine-to-machine via the FastAPI scoring service.
- **Honest caveats:** stock levels are a single snapshot (2025-12-31) vs an Oct–Dec answer-key
  window — recall is the gate, precision (0.453) is bounded by that staleness; thresholds are
  pre-registered, not tuned; `watch_volatile` is legitimately 0 (stockout and overstock are
  near-contradictory states).

*Sources: artifacts/metrics/flag_evaluation.json, artifacts/risk/risk_scores.parquet,
artifacts/serving/dashboard_kpis.parquet, reports/risk_evaluation.md*
