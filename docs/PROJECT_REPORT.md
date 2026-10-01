# PROJECT FORESIGHT — COMPREHENSIVE TECHNICAL & BUSINESS REPORT
## AI-Powered Demand Forecasting & Working Capital Risk Intelligence Platform

---

**Client / Engagement:** NorthBay Living  
**Program:** Zidio Development — Senior Data Science & Supply Chain Intelligence Project  
**Author:** Aditya Raj (Lead Data Scientist & ML Engineer)  
**Version:** 1.0.0 (Production Certified)  
**Date:** October 2026  
**Repository:** [GitHub — ADITYA-tp01/FORESIGHT-AI-Powered-Demand-Inventory-Intelligence-Platform](https://github.com/ADITYA-tp01/FORESIGHT-AI-Powered-Demand-Inventory-Intelligence-Platform)  
**Live Production Application:** [Streamlit Cloud Dashboard](https://foresight-ai-powered-demand-inventory-intelligence-platform.streamlit.app)  

---

## TABLE OF CONTENTS
1. [Executive Summary & High-Impact Outcomes](#1-executive-summary--high-impact-outcomes)
2. [Business Problem & Strategic Context](#2-business-problem--strategic-context)
3. [Architectural Governance & Non-Negotiable Rules](#3-architectural-governance--non-negotiable-rules)
4. [Data Ingestion, Auditing & Unit Reconciliation (Phase 0 & 1)](#4-data-ingestion-auditing--unit-reconciliation-phase-0--1)
5. [Exploratory Data Analysis & Catalog Segmentation (Phase 2)](#5-exploratory-data-analysis--catalog-segmentation-phase-2)
6. [Benchmarking Methodology: The 3-Tier Seasonal-Naive Baseline](#6-benchmarking-methodology-the-3-tier-seasonal-naive-baseline)
7. [Machine Learning Forecasting Engine (Phase 3)](#7-machine-learning-forecasting-engine-phase-3)
8. [Inventory Decisioning & Rupee Exposure Matrix (Phase 4)](#8-inventory-decisioning--rupee-exposure-matrix-phase-4)
9. [Enterprise Microservices & User Interfaces (Phase 5 & 6)](#9-enterprise-microservices--user-interfaces-phase-5--6)
10. [Automated Quality Assurance & Verification Audit](#10-automated-quality-assurance--verification-audit)
11. [Operational Deployment & Buyer Handoff Strategy](#11-operational-deployment--buyer-handoff-strategy)
12. [Limitations, Assumptions & Long-Term Roadmap](#12-limitations-assumptions--long-term-roadmap)

---

## 1. Executive Summary & High-Impact Outcomes

### 1.1 Project Objective
**Project FORESIGHT** is an enterprise-grade demand forecasting and inventory-risk intelligence platform engineered for **NorthBay Living**, a large multi-channel retail network in Pakistan operating **5,000 SKUs** across **12 merchandise categories**, **30 stores**, and **9.95 million transactions** (~801 MB raw data). 

The primary business directive was to eliminate gut-feel replenishment, curb devastating stockouts, release locked working capital trapped in dead stock, and deliver an audit-defensible ML platform that outperforms standard retail baselines while providing real-time software decision tools for both operations and executive finance.

### 1.2 Executive Scorecard & Validated Results
Every metric and financial figure in this report is programmatically derived from reproducible pipeline runs and verified against ground-truth evaluation keys without synthetic data fabrication:

| Strategic Dimension | Baseline / Heuristic | Project FORESIGHT ML | Business & Financial Impact |
|---|---|---|---|
| **Demand Forecast Accuracy (WAPE)** | 37.76% (Seasonal-Naive $t-52$) | **30.81%** (Global HGB) | **-18.39% Relative Error Reduction** across catalog |
| **Backtest Cross-Validation Consistency** | N/A | **3 out of 3 Folds Won** | Consistent outperformance across all temporal backtests |
| **80% Prediction Interval Coverage** | N/A | **77.63%** empirical coverage | Highly reliable bounds for dynamic safety stock buffers |
| **Stockout Risk Identification Recall** | Rule-of-thumb | **86.5%** | **Target $\ge 85.0\%$ Met** against ground-truth anomalies |
| **Store Outage Detection Coverage** | Fragmented audit | **100% Coverage** (2,703 / 2,703) | All historical store-SKU stockout pairs captured |
| **Dead Stock / Slow Mover Recall** | Arbitrary aging | **100% Recall (F1: 0.872)** | Complete capture of non-moving capital investments |
| **Protected Revenue (Sales at Risk)** | Unmonitored | **INR 32,785,603** | **325 Immediate Purchase Orders** prioritized |
| **Locked Working Capital Surfaced** | Unmonitored | **INR 1,534,592,370** | **3,021 Markdown Candidates** surfaced for cash release |
| **Actionable Cash Recovery Target** | INR 0 | **INR 1,533,774,315** | **99.95%** of excess capital targeted for liquidation |

---

## 2. Business Problem & Strategic Context

### 2.1 The Retail Inventory Dilemma
Multi-channel retail operations face a perennial dilemma: **under-stocking** causes lost sales, angry customers, and brand erosion, while **over-stocking** locks up valuable working capital, accumulates warehousing holding costs, and leads to inventory obsolescence.

At NorthBay Living, this dilemma was exacerbated by:
1. **Catalog Scale:** 5,000 active SKUs across fast-moving consumables (Grocery, Beverages, Dairy) and long-lead durable goods (Home & Kitchen, Apparel).
2. **Geographical Dispersion:** 30 multi-channel retail stores spread across distinct urban regions in Pakistan (Karachi, Lahore, Islamabad, Faisalabad, Rawalpindi, etc.).
3. **Severe Demand Volatility:** Peak sales in December reached **2,037,489 units** compared to February trough volumes of **1,223,965 units**—a ~66% peak-to-trough swing driven by seasonal festivities and winter holidays.
4. **Promotional Distortion:** Over **21.02% of sales transactions** and **27.28% of total unit volume** occurred under active promotional campaigns, creating sudden spikes that broke traditional time-series moving averages.

### 2.2 Quantifying the Financial Damage
Prior to FORESIGHT, operations relied on static reorder heuristics. An empirical audit of the year-end inventory position revealed:
* **INR 32.79M in Demand at Risk:** 325 SKUs were on the verge of stockouts where projected demand over the replenishment lead time vastly exceeded available stock.
* **4,239 Store-Level Zero-Stock Outages:** Store shelves were frequently empty even when regional central depots carried inventory.
* **INR 1.53B in Frozen Liquidity:** 3,021 overstocked SKUs had forward cover exceeding 12 to 52+ weeks, severely straining operational cash flows.

---

## 3. Architectural Governance & Non-Negotiable Rules

To ensure scientific defensibility, reproducibility, and enterprise readiness, Project FORESIGHT was executed under **seven strict architectural rules**:

1. **Rule 1 — Zero Data Fabrication:** Absolutely no synthetic transactions or invented rows. All models and tables reflect actual audited retail datasets.
2. **Rule 2 — Memory-Capped Chunked Ingestion (< 500 MB RAM):** Raw sales transactions (801 MB, 9.95M rows) are streamed in 500,000-row chunks. Unchunked in-memory loads are strictly forbidden.
3. **Rule 3 — Strict Target Leakage Prevention:** `sku_inventory_flags.csv` is an out-of-sample ground-truth evaluation key; it never enters feature sets or model training. Concurrent sales, revenues, and transaction counts are excluded from feature matrices.
4. **Rule 4 — Mandatory Seasonal-Naive Benchmark:** A formal 3-tier fallback seasonal-naive baseline is enforced. Advanced ML must prove superior accuracy on identical evaluation windows to earn production deployment.
5. **Rule 5 — Strictly Temporal Validation (Zero Data Shuffling):** Never use randomized train/test splits. All cross-validation enforces chronological rolling origins (`shuffle=False`).
6. **Rule 6 — Explicit Domain Heuristics:** Category lead times, service level $Z$-scores, and weeks-of-supply thresholds are documented and configurable in PyYAML files (`configs/*.yaml`).
7. **Rule 7 — Single-Command End-to-End Reproducibility:** The entire pipeline executes from raw extraction to dashboard serving via `python -m foresight.pipeline`.

---

## 4. Data Ingestion, Auditing & Unit Reconciliation (Phase 0 & 1)

### 4.1 Data Asset Inventory
The data lake integrates 7 relational and operational assets:

```
Dataset/
├── retail_contaminated_dataset/ (Operational Extract)
│   ├── sales_transactions.csv   [801.1 MB, 9,945,511 rows]
│   ├── sku_master.csv           [464.0 KB, 5,000 SKUs]
│   ├── store_master.csv         [2.0 KB, 30 Stores]
│   ├── customer_master.csv      [554.5 KB, 10,000 Customers]
│   ├── inventory_snapshot.csv   [932.8 KB, 26,408 Inventory Positions]
│   ├── promotions.csv           [8.5 KB, 100 Campaigns]
│   └── sku_inventory_flags.csv  [123.7 KB, 600 Anomaly Flags (Eval Only)]
└── retail_clean_dataset/        (Benchmark Control Extract)
    └── sales_transactions.csv   [803.2 MB, 9,972,038 rows]
```

### 4.2 Data Cleansing & Deduplication Policy
* **Contamination Profiling:** The operational extract contained 26,527 invalid transactions, including negative quantities (-1 to -5), zero/negative unit prices, and out-of-bounds promotional discounts.
* **Streaming Validation:** A custom chunk-iterator streams 500,000 rows per batch, validating each record against schema constraints:
  $$\text{Quantity} \in [1, 5], \quad \text{Unit Price} > 0.0, \quad \text{Discount} \in [0.0, 1.0]$$
* **Receipt Deduplication (`retain_multiscan`):** Legitimate multi-scan occurrences (same customer purchasing multiple units of the same SKU on the same receipt timestamp) are retained and summed rather than discarded as duplicate keys.

### 4.3 Three-Way Unit Sales Reconciliation
To prove 100% mathematical integrity across aggregation grains, an automated assertion verifies that no units are lost or double-counted:

$$\sum \text{Daily Units} = \sum \text{Weekly Units} = \sum \text{Store-Daily Units} = \mathbf{18,700,906 \text{ units}}$$

* **Reconciliation Error:** **0.000%** across all 5,000 SKUs and 210 weeks.

### 4.4 Pakistan Calendar & Holiday Engineering
A dedicated calendar module (`data/external/pakistan_holidays.py`) models 1,461 calendar days (2022 to 2026), capturing:
* **Fixed National Holidays:** Pakistan Day (March 23), Independence Day (August 14), Quaid-e-Azam Day (December 25), Labour Day (May 1).
* **Lunar Islamic Holidays:** Eid-ul-Fitr, Eid-ul-Adha, Ashura, Eid Milad-un-Nabi (dynamically mapped across shifting Gregorian dates).
* **Proximity Flags:** 7-day pre-holiday and post-holiday ramp windows to capture commercial grocery and apparel buying surges.

---

## 5. Exploratory Data Analysis & Catalog Segmentation (Phase 2)

### 5.1 Pareto Revenue Skew (ABC Analysis)
Catalog revenue exhibits extreme concentration adhering to the Pareto principle:
* **Class A (Top 80% Revenue):** Generated by **873 SKUs (17.46%)** of the catalog.
* **Class B (Next 15% Revenue):** Generated by **1,332 SKUs (26.64%)** of the catalog.
* **Class C (Bottom 5% Revenue):** Generated by **2,795 SKUs (55.90%)** of the catalog.
* The top **1,000 SKUs (20.00%) account for 73.08% of total sales revenue**.

```
Pareto Revenue Distribution:
[Class A: 17.5% SKUs | 80% Rev] ====> Core margin engine; zero-stockout mandate
[Class B: 26.6% SKUs | 15% Rev] ====> Stable recurring lines; standard replenishment
[Class C: 55.9% SKUs |  5% Rev] ====> Long-tail; high dead-stock hazard
```

### 5.2 Demand Volatility (XYZ Analysis)
Demand predictability was segmented using the weekly coefficient of variation ($CV = \sigma / \mu$):
* **Class X ($CV \le 0.50$):** **1,911 SKUs (38.22%)** — highly stable, continuous demand; prime candidates for automated statistical replenishment.
* **Class Y ($0.50 < CV \le 1.00$):** **1,858 SKUs (37.16%)** — moderate seasonality and promotional responsiveness.
* **Class Z ($CV > 1.00$):** **1,231 SKUs (24.62%)** — lumpy, intermittent demand requiring wide safety stock intervals.

### 5.3 Promotional Mechanics
* **Volume Lift:** Active promotional weeks yield an average **+29.8% unit volume lift** over non-promo weeks.
* **Conflict Resolution:** 41 overlapping campaign pairs were resolved via a deterministic rule: **Highest Discount Wins**.

---

## 6. Benchmarking Methodology: The 3-Tier Seasonal-Naive Baseline

In accordance with Rule 4, advanced machine learning models cannot be deployed on raw faith; they must beat an industry-standard **Seasonal-Naive baseline** evaluated on identical cross-validation folds.

### 6.1 Baseline Hierarchy Definition
Given the 52-week annual retail cycle, the baseline forecast $\hat{y}_{i, t+h}$ for SKU $i$ at week $t+h$ uses a 3-tier fallback cascade:

$$\hat{y}_{i, t+h} = \begin{cases} 
y_{i, t+h-52} & \text{Tier 1: if } y_{i, t+h-52} \text{ exists and } > 0 \\
\frac{1}{4} \sum_{k=1}^4 y_{i, t-k} & \text{Tier 2: if Tier 1 missing, use trailing 4-week mean} \\
\bar{Y}_{\text{category}, t-1} & \text{Tier 3: if SKU history missing, use category run-rate}
\end{cases}$$

### 6.2 Error Metric: Weighted Absolute Percentage Error (WAPE)
Mean Absolute Percentage Error (MAPE) is notoriously flawed in retail because zero-demand weeks cause division-by-zero errors. Project FORESIGHT enforces **WAPE**:

$$\text{WAPE} = \frac{\sum_{i,t} |y_{i,t} - \hat{y}_{i,t}|}{\sum_{i,t} y_{i,t}}$$

* **Baseline Pooled Performance:** **37.76% WAPE** across all validation folds. This formed the hurdle rate for ML deployment.

---

## 7. Machine Learning Forecasting Engine (Phase 3)

### 7.1 Global Model Architecture
Instead of training 5,000 fragile individual models, FORESIGHT implements a **Unified Global Forecaster** powered by `HistGradientBoostingRegressor` (Histogram-based Gradient Boosting).

**Architectural Rationale:**
1. **Cross-Series Learning:** Fast movers share price elasticity and holiday response patterns with slow movers in the same category.
2. **Cold-Start Resilience:** New or low-volume SKUs inherit category-level parameters seamlessly.
3. **Execution Velocity:** Trains on 1.05M weekly records in under 90 seconds.
4. **Strict Temporal Integrity:** Configured with `early_stopping=False` to prevent Scikit-Learn from executing randomized internal validation splits that violate Rule 5.

### 7.2 Feature Pipeline (45 Strictly Causal Features)
All features are engineered with strict causal boundaries ($t-k$ where $k \ge 1$):

```
Feature Union Architecture:
├── Causal Lags (7):            t-1, t-2, t-4, t-8, t-13, t-26, t-52
├── Rolling Windows (12):       Trailing 4w, 8w, 12w (Mean, Std, Min, Max)
├── Calendar & Cyclical (8):    Fourier sin/cos terms (annual & monthly cycles), week of year
├── Holiday Telemetry (4):      Is_holiday, holiday_type, pre_holiday_7d, post_holiday_7d
├── Promotional Signals (5):    Is_promo_active, discount_depth, promo_duration, promo_type
├── Catalog & Pricing (5):      Category_code, unit_price, price_vs_cat_mean, target_margin
└── Out-of-Fold Baselines (4):  Seasonal-naive lag-52, trailing 4w moving average
```

* **Target Leakage Assertions:** An automated suite (`tests/test_leakage.py`) asserts that `revenue`, `gross_revenue`, `promo_trans_count`, and future actuals never enter the feature matrix.

### 7.3 Rolling-Origin Cross-Validation Results
Validation used 3 non-shuffled expanding temporal folds simulating real operational quarterly deployments:

| Fold Index | Training Window | Validation Window | Baseline WAPE | FORESIGHT ML WAPE | Relative Error Reduction | ML Bias |
|---|---|---|:---:|:---:|:---:|:---:|
| **Fold 1** | 2022-W01 – 2024-W52 (156w) | 2025-W01 – 2025-W08 (8w) | 38.01% | **30.33%** | **-20.21%** | +0.42 units |
| **Fold 2** | 2022-W01 – 2025-W16 (172w) | 2025-W17 – 2025-W24 (8w) | 35.29% | **26.96%** | **-23.60%** | -0.18 units |
| **Fold 3** | 2022-W01 – 2025-W40 (196w) | 2025-W41 – 2025-W52 (12w) | 39.05% | **33.28%** | **-14.78%** | +0.81 units |
| **POOLED** | **Full Backtest Horizon** | **28 Validation Weeks** | **37.76%** | **30.81%** | **-18.39%** | **+0.35 units** |

* **Gate M3 Requirement:** ML must beat baseline across all folds. **Verdict: PASSED (3/3 Folds Won).**

### 7.4 Quantile Regression & 80% Prediction Intervals
To support stochastic inventory safety stock calculations, FORESIGHT trains auxiliary quantile estimators at the **10th percentile ($q=0.10$)** and **90th percentile ($q=0.90$)** using pinball loss.
* **Empirical Holdout Coverage:** **77.63%** on out-of-sample holdout (closely tracking the nominal 80.0% target).
* **Operational Value:** Provides planners with upper and lower demand boundaries to absorb supply chain shocks.

---

## 8. Inventory Decisioning & Rupee Exposure Matrix (Phase 4)

### 8.1 Supply Chain Physics & Formulas
Inventory intelligence connects forward ML forecasts with warehouse telemetry:

1. **Lead Time Demand (LTD):**
   $$\text{LTD}_i = \sum_{t=1}^{\lceil L_i / 7 \rceil} \hat{y}_{i, t}$$
   *(where $L_i$ is category lead time in days, ranging from 7 days for Dairy to 21 days for Home & Kitchen).*

2. **Projected Available Balance (PAB):**
   $$\text{PAB}_i = \text{Current On-Hand}_i + \text{On-Order Units}_i - \text{LTD}_i$$

3. **Forward Weeks of Supply (WOS):**
   $$\text{WOS}_i = \frac{\text{Current On-Hand}_i}{\frac{1}{8} \sum_{t=1}^8 \hat{y}_{i, t}}$$

4. **Dynamic Safety Stock (SS):**
   $$\text{SS}_i = Z_{\text{service}} \times \sigma_{i, \text{lead time}} = 1.645 \times \sqrt{\frac{L_i}{7}} \times \sigma_{\text{forecast error}}$$

5. **Financial Exposure Formulas:**
   $$\text{Rupees at Risk}_i = \max(0, -\text{PAB}_i) \times \text{Selling Price}_i$$
   $$\text{Locked Capital}_i = \max(0, \text{On-Hand}_i - (\hat{Y}_{i, 8w} + \text{SS}_i)) \times \text{Unit Cost}_i$$

### 8.2 The 4-Quadrant Strategic Grid
Every SKU is triaged into an operational action quadrant:

```
                      FORWARD WEEKS OF SUPPLY (WOS)
                        Low (< 4w)           High (> 12w)
                 ┌──────────────────────┬──────────────────────┐
                 │    REORDER NOW       │   WATCH VOLATILE     │
   PAB <= 0      │     (Quadrant 1)     │     (Quadrant 4)     │
                 │   325 SKUs (7.2%)    │     0 SKUs (0.0%)    │
PROJECTED        │  INR 32.79M At Risk  │   INR 0.00 At Risk   │
AVAILABLE        ├──────────────────────┼──────────────────────┤
BALANCE          │      HEALTHY         │   MARKDOWN / CLEAR   │
                 │     (Quadrant 2)     │     (Quadrant 3)     │
   PAB > 0       │  1,149 SKUs (25.6%)  │  3,021 SKUs (67.2%)  │
                 │   Optimal Pipeline   │  INR 1.53B Capital   │
                 └──────────────────────┴──────────────────────┘
```

### 8.3 Ground-Truth Anomaly Validation (`sku_inventory_flags.csv`)
To prove risk classification accuracy, FORESIGHT was audited against 600 blind anomaly flags:

| Target Risk Class | Answer Key Total | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Recall Rate | Precision | F1-Score | Target Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stockout Risk** | 200 SKUs | 173 | 209 | 27 | **86.5%** | 0.453 | 0.595 | **$\ge 85.0\%$ MET** |
| **Slow Mover** | 400 SKUs | 400 | 117 | 0 | **100.0%** | 0.774 | **0.872** | **Exceeded** |
| **Store Outage Pairs** | 2,703 Pairs | 2,703 | 2,066 | 0 | **100.0%** | 0.567 | 0.723 | **Exceeded** |

*Note on Precision:* Precision is naturally bounded by the fact that the answer key reflects a historical point-in-time snapshot, whereas FORESIGHT evaluates forward multi-week cumulative risk. High recall was the primary mandate to protect NorthBay Living from stockouts.

---

## 9. Enterprise Microservices & User Interfaces (Phase 5 & 6)

### 9.1 Sub-25ms Serving Cache Architecture
To achieve instantaneous sub-second response times without querying 10M rows in production, Phase 5 compiles a pre-aggregated Parquet serving layer (`artifacts/serving/`, ~3.3 MB):
* `dashboard_kpis.parquet`: High-level portfolio financial metrics.
* `decision_grid.parquet`: 5,000-row quadrant triage matrix.
* `forecast_summary.parquet`: 8-week forward point forecasts with $q10/q90$ bands and baseline comparison.
* `history_weekly.parquet` & partitioned category stores: Historical actuals.

### 9.2 Interactive Operations Dashboard (Streamlit)
* **Hosting:** Deployed live on Streamlit Cloud.
* **4 Operational Pages:**
  1. **Executive Overview:** High-level portfolio KPIs, Rupee exposure, 52-week demand trajectory, and category risk breakdown.
  2. **Forecast Explorer:** Interactive SKU-level time-series visualization showing historical actuals, point forecast, 80% confidence ribbon, seasonal-naive baseline, and active promotional overlays.
  3. **Inventory Decision Grid:** 4-quadrant interactive Plotly scatter matrix with one-click exportable tables for Immediate Purchase Orders and Markdown Liquidation.
  4. **SKU Drilldown:** Deep SKU telemetry, lead times, safety stocks, reorder points, and store-by-store inventory allocation charts.

### 9.3 Real-Time REST Scoring Microservice (FastAPI)
* **Framework:** FastAPI + Pydantic v2 + Uvicorn (`api/main.py`).
* **Endpoints:**
  * `GET /health`: Cluster health, platform version, and artifact freshness timestamps.
  * `GET /forecast/{sku_id}`: Returns 8-week forward predictions with lower/upper 80% intervals.
  * `GET /risk/{sku_id}`: Real-time risk quadrant, weeks of supply, and rupee impact.
  * `POST /batch/risk`: Bulk risk evaluation for enterprise ERP batches (up to 100 SKUs).
  * Auto-generated interactive Swagger UI at `/docs`.

---

## 10. Automated Quality Assurance & Verification Audit

The platform is fortified by a comprehensive test suite (`tests/`) containing **121 automated tests** achieving 100% pass rates:

```text
================================== test session starts ===================================
platform win32 -- Python 3.10+, pytest-8.3.3, pluggy-1.5.0
rootdir: C:\Users\adity\Documents\Zidio\FORESIGHT – AI-Powered Demand & Inventory Intelligence Platform
configfile: pyproject.toml
collected 121 items

tests/test_api.py .............                                                    [ 10%]
tests/test_app.py .....                                                            [ 14%]
tests/test_baseline.py ...............                                             [ 27%]
tests/test_calendar.py ...                                                         [ 29%]
tests/test_features.py ...........                                                 [ 38%]
tests/test_flag_evaluation.py ........                                             [ 45%]
tests/test_forecasting.py ..........                                               [ 53%]
tests/test_ingestion.py .........                                                  [ 61%]
tests/test_inventory.py ....                                                       [ 64%]
tests/test_leakage.py ...........                                                  [ 73%]
tests/test_phase0.py ......                                                        [ 78%]
tests/test_promotions.py ..                                                        [ 80%]
tests/test_risk.py ............                                                    [ 90%]
tests/test_risk_scoring.py ............                                            [100%]

================================== 121 passed in 37.42s ==================================
```

### Static Analysis & Linter Audit
Executed via Astral `ruff`:
```powershell
python -m ruff check src tests api app
# Output: All checks passed! (0 errors, 0 warnings)
```

---

## 11. Operational Deployment & Buyer Handoff Strategy

### 11.1 Workflow for Procurement Buyers (Immediate Reorders)
1. Navigate to Dashboard Page 3 (**Inventory Decision Grid**) ➔ Tab: **Immediate Reorders**.
2. Filter by Category (e.g. `Grocery`, `Dairy & Bakery`).
3. Click **Export CSV** to extract the **325 prioritized purchase orders**.
4. Hand off directly to suppliers with recommended reorder quantities:
   $$\text{PO Quantity}_i = \text{LTD}_i + \text{SS}_i - (\text{On-Hand}_i + \text{On-Order}_i)$$
5. Protects **INR 32.79M** in customer sales across the upcoming 8 weeks.

### 11.2 Workflow for Merchandisers (Markdown Liquidation)
1. Navigate to Dashboard Page 3 ➔ Tab: **Markdown Candidates**.
2. Identify the **3,021 overstocked SKUs** ($WOS > 12$ weeks).
3. Implement tiered liquidation discounts:
   * **12–26 Weeks of Supply:** 15% promotional discount.
   * **26–52 Weeks of Supply:** 30% promotional discount.
   * **> 52 Weeks of Supply (Dead Stock):** 50% clearance bundle to release trapped cash immediately.
4. Recovers up to **INR 1.53B** in working capital for reinvestment into high-margin Class A inventory.

---

## 12. Limitations, Assumptions & Long-Term Roadmap

### 12.1 Analytical Assumptions & Boundary Conditions
1. **Point-in-Time Inventory Snapshot:** The operational extract contained a single inventory snapshot (2025-12-31). Historical inventory burn-down was estimated from daily sales velocities.
2. **Category Lead Times:** Where vendor contracts were absent, lead times were standardized by category heuristics (7–21 days) in `configs/risk.yaml`.
3. **Lunar Holiday Projection:** Islamic holidays are astronomical; projections beyond 2026 should be updated annually via national moon-sighting calendars.

### 12.2 Production Roadmap (Phase 7+)
1. **Automated ERP Webhook Integration:** Connect FastAPI `/batch/risk` directly to SAP / Microsoft Dynamics to generate purchase orders automatically.
2. **Store-to-Store Inventory Balancing:** Implement a linear programming transportation solver to transfer excess stock from low-velocity stores to stockout-prone stores prior to placing external POs.
3. **Dynamic Price Elasticity Engine:** Incorporate real-time competitor scrapers to optimize discount depth dynamically during clearance markdowns.

---

### Certification & Approvals
**Lead Data Scientist:** Aditya Raj  
**Status:** Certified Production Complete (v1.0.0)  
**Deliverables Completed:** D1 (Data Pipeline Audit), D2 (EDA & Baseline), D3 (ML Forecaster), D4 (Risk Decisioning), D5 (Streamlit Dashboard), D6 (FastAPI Service), D7 (Executive Readout).  
