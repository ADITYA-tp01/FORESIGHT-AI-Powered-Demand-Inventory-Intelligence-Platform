# PROJECT FORESIGHT — MASTER IMPLEMENTATION PLAN (100/100 SPECIFICATION)

**Project:** FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform  
**Client in Brief:** NorthBay Living  
**Role:** Senior Data Scientist & Technical Lead  
**Duration:** 4 Weeks (20 Working Days)  
**Primary Stack:** Python 3.10+, pandas, pyarrow, scikit-learn, statsmodels, FastAPI, Streamlit, Plotly, PyYAML, pytest  
**Primary Objective:** Build an end-to-end, reproducible, audit-defensible SKU-level demand forecasting and inventory-risk intelligence system that beats naive baselines, quantifies working capital impact in rupees, and provides intuitive decision tools for operations and finance leadership.

---

# 0. NON-NEGOTIABLE PROJECT CONSTRAINTS & GOVERNANCE

The implementation agent and human data scientists MUST strictly adhere to these seven non-negotiable operational rules.

### Rule 1 — Zero Data Fabrication
Do not generate synthetic business transactions or invent fake missing rows to patch data gaps. All pipeline operations must be built upon the actual audited datasets:
* **9,945,511** contaminated sales transactions (operational environment)
* **9,972,038** clean sales transactions (benchmark control environment)
* **5,000** catalog SKUs across 12 merchandise categories
* **30** multi-channel stores across Pakistan
* **10,000** registered loyalty customers
* **26,408** store-SKU inventory records
* **100** promotional campaigns (41 overlapping pairs)
* **600** ground-truth inventory anomaly records (200 stockout risk, 400 slow movers)

### Rule 2 — Never Load the 764 MB Sales CSV Completely into RAM
Never execute un-chunked reads on `sales_transactions.csv`. All ingestion and aggregation must use streaming chunk iterators:
```python
pd.read_csv("Dataset/retail_contaminated_dataset/sales_transactions.csv", chunksize=500_000)
```
Aggregate chunk-by-chunk in memory, serialize intermediate files to Apache Parquet (`snappy` compression), and cap peak RAM utilization below 500 MB.

### Rule 3 — Absolute Target Leakage Prevention
The following datasets, labels, and forward-looking variables must **NEVER** enter feature engineering:
* `sku_inventory_flags.csv` (strictly an out-of-sample ground-truth evaluation resource; never a training feature)
* Future stockout indicators or anomaly labels
* Future actual demand or concurrent period sales
* Future inventory positions or restock dates
* Clean-version ground truth transactions during contaminated evaluation

### Rule 4 — Mandatory Seasonal-Naive Benchmark
The forecasting system must implement a strict **Seasonal-Naive baseline** ($t - 52$ weeks) and measure performance using **WAPE (Weighted Absolute Percentage Error)**. Machine learning models must earn the right to be deployed by beating this baseline on identical backtest windows. If an advanced model fails to beat the baseline, report that result honestly.

### Rule 5 — Strictly Temporal Validation (No Random Shuffling)
Never use randomized train/test splits (`train_test_split(..., shuffle=True)` is strictly prohibited). All model backtesting, cross-validation, and tuning must use **rolling-origin temporal cross-validation** preserving chronological causality.

### Rule 6 — Explicitly Documented Domain Heuristics
Where operational extracts lack explicit data fields (`lead_time_days`, `on_order_units`, `launch_date`, public holidays), establish configurable, auditable domain parameters in YAML configuration files. Never embed silent magic numbers in application code.

### Rule 7 — Single-Command End-to-End Reproducibility
The entire pipeline—from raw file extraction, cleaning, and aggregation to baseline evaluation, ML forecasting, risk scoring, and dashboard serving—must execute end-to-end via a single automated command (`python -m foresight.pipeline` or `make pipeline`).

---

# 1. ENTERPRISE REPOSITORY ARCHITECTURE

The repository is organized following clean-architecture principles, separating ingestion, feature engineering, modeling, decisioning, and presentation:

```text
FORESIGHT/
├── README.md                           # Executive summary, setup, pipeline execution, metrics
├── requirements.txt                    # Pinned, tested Python dependencies
├── pyproject.toml                      # Package specifications & tool configurations (ruff, pytest)
├── .gitignore                          # Excludes raw data dumps, venv, cache, OS files
├── .env.example                        # Configurable environment paths
├── Makefile                            # Developer shortcuts (install, pipeline, test, lint, app)
├── AGENT_HANDOFF.md                    # State tracking between development sessions
│
├── configs/
│   ├── config.yaml                     # Global environment paths, random seeds, run modes
│   ├── features.yaml                   # Lag windows, rolling stats, calendar & promo flags
│   ├── forecast.yaml                   # Model hyperparams, horizon (8w), CV windows
│   └── risk.yaml                       # Lead times by category, WOS thresholds, rupee formulas
│
├── data/                               # Structured data lake (gitignored except .gitkeep)
│   ├── raw/                            # Symbolic links / copies of Dataset/ files
│   ├── interim/                        # Partitioned daily/weekly Parquet chunks
│   ├── processed/                      # Analysis-ready tables (sales_weekly, calendar, etc.)
│   └── external/                       # Pakistani public holiday calendar (2022–2026)
│
├── notebooks/
│   ├── 01_data_audit.ipynb             # Discovery verification & schema assertions
│   ├── 02_eda.ipynb                    # Seasonality, Pareto skew, price elasticity, dead stock
│   ├── 03_baseline.ipynb               # Seasonal-naive benchmark & WAPE baselining
│   ├── 04_forecasting.ipynb            # ML feature pipeline & rolling-origin backtesting
│   ├── 05_risk_analysis.ipynb          # Stockout/overstock scoring vs. ground-truth flags
│   └── 06_results_consolidation.ipynb  # Executive charts & rupee impact reconciliation
│
├── src/
│   └── foresight/
│       ├── __init__.py                 # Package version declaration (v1.0.0)
│       ├── config.py                   # Pydantic settings loading YAML configs
│       ├── logging_config.py           # Structured JSON / console logger
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── sales_ingestion.py      # 500k-row chunked reader & Parquet staging
│       │   ├── metadata_ingestion.py   # Dimension loaders (stores, SKUs, customers, promos)
│       │   └── validation.py           # Pandera / Pydantic schema validation contracts
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── sales_daily.py          # Daily aggregation & dense SKU-day panel builder
│       │   ├── sales_weekly.py         # Primary weekly aggregation (grain: week, sku_id)
│       │   ├── calendar.py             # 1,461-day calendar generation & holiday tagger
│       │   ├── promotions.py           # Campaign resolution & precedence rule engine
│       │   └── inventory.py            # Point-in-time stock normalization & lead-time enricher
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── lag_features.py         # Strictly causal lags: t-1, t-2, t-4, t-8, t-13, t-52
│       │   ├── rolling_features.py     # Causal moving stats: 4w, 8w, 12w mean/std/min/max
│       │   ├── calendar_features.py    # Cyclical sin/cos encodings for month, week, holidays
│       │   ├── promo_features.py       # Active promo flags, discount depth, promo type encodings
│       │   └── feature_pipeline.py     # Unified Scikit-Learn feature union & leakage test
│       │
│       ├── forecasting/
│       │   ├── __init__.py
│       │   ├── baseline.py             # Seasonal-naive with fallback hierarchy
│       │   ├── models.py               # HistGradientBoostingRegressor / LightGBM wrapper
│       │   ├── training.py             # Global multi-SKU training engine
│       │   ├── validation.py           # Rolling-origin temporal CV engine
│       │   └── prediction.py           # 8-week multi-step forward inference generator
│       │
│       ├── inventory/
│       │   ├── __init__.py
│       │   ├── assumptions.py          # Category lead time & on-order policy resolver
│       │   ├── demand_censoring.py     # Censored demand reconstruction during stockouts
│       │   ├── risk_scoring.py         # 4-quadrant decisioning engine (stockout vs overstock)
│       │   └── actions.py              # Operational recommendation & rupee exposure calculator
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py              # Robust WAPE, MAPE, Bias, MAE, RMSE implementations
│       │   ├── forecast_metrics.py     # SKU- and Category-level model vs. baseline reporting
│       │   ├── risk_metrics.py         # Precision, Recall, F1 against sku_inventory_flags.csv
│       │   └── validation_report.py    # Auto-generation of markdown evaluation memos
│       │
│       └── utils/
│           ├── __init__.py
│           ├── io.py                   # Atomic Parquet I/O with checksums
│           ├── dates.py                # ISO week, calendar boundaries, retail date helpers
│           └── profiling.py            # Execution timer & peak memory monitor
│
├── app/                                # Streamlit planning application
│   ├── streamlit_app.py                # Entrypoint, session state, navigation
│   ├── pages/
│   │   ├── 1_Executive_Overview.py     # Portfolio KPIs, risk totals, rupee exposure, trends
│   │   ├── 2_Forecast_Explorer.py      # Forecast vs actual, intervals, promo uplift, filters
│   │   ├── 3_Inventory_Decision_Grid.py# 4-quadrant scatter matrix & prioritized action table
│   │   └── 4_SKU_Drilldown.py          # Single-SKU deep dive: history, stock, parameters
│   └── components/
│       ├── kpi_cards.py                # Custom styled metric cards
│       ├── charts.py                   # Plotly interactive forecast & decision visuals
│       └── tables.py                   # Interactive data tables with CSV download
│
├── api/                                # FastAPI scoring service
│   ├── main.py                         # App definition, middleware, health endpoints
│   ├── schemas.py                      # Pydantic v2 input/output schemas
│   └── routes.py                       # /forecast/{sku_id}, /risk/{sku_id}, /batch endpoints
│
├── tests/                              # Pytest automated test suite
│   ├── conftest.py                     # Synthetic fixtures, mock data, configuration overrides
│   ├── test_ingestion.py               # Schema contracts, chunk reader verification
│   ├── test_transformations.py         # Aggregation correctness, dense panel zero-filling
│   ├── test_leakage.py                 # Rigorous temporal feature leakage assertion tests
│   ├── test_baseline.py                # Baseline calculation & fallback correctness
│   ├── test_forecasting.py             # Model training, inference shape & interval tests
│   ├── test_risk.py                    # Quadrant assignment & rupee math checks
│   └── test_api.py                     # FastAPI test client endpoint integration
│
├── reports/                            # Generated client-ready deliverables
│   ├── data_quality_report.md          # Deliverable D1 documentation
│   ├── eda_report.md                   # Deliverable D2 EDA insight memo
│   ├── model_evaluation.md             # Deliverable D3 backtest evaluation memo
│   ├── risk_evaluation.md              # Deliverable D4 risk accuracy memo
│   └── executive_readout.md            # Deliverable D7 executive stakeholder readout
│
└── artifacts/                          # Versioned model & serving artifacts (gitignored)
    ├── models/                         # Serialized model weights (global_forecaster.joblib)
    ├── forecasts/                      # Out-of-sample predictions (forecast_8w.parquet)
    ├── metrics/                        # JSON performance logs (cv_wape_summary.json)
    └── serving/                        # Low-latency pre-aggregated dashboard stores
        ├── dashboard_kpis.parquet      # Pre-aggregated portfolio KPI cache (< 100 KB)
        ├── decision_grid.parquet       # Pre-computed SKU risk quadrant table (~1.5 MB)
        └── forecast_summary.parquet    # Latest 8-week horizon predictions (~2.0 MB)
```

---

# 2. PHASE 0 — ENVIRONMENT INITIALIZATION & INFRASTRUCTURE

### 2.1 Dependencies Configuration (`requirements.txt`)
All packages are pinned to compatible production versions:
```text
pandas>=2.2.0,<3.0.0
numpy>=1.26.0,<2.0.0
pyarrow>=15.0.0
scikit-learn>=1.4.0
statsmodels>=0.14.0
scipy>=1.12.0
pyyaml>=6.0.1
pydantic>=2.6.0
fastapi>=0.110.0
uvicorn>=0.28.0
streamlit>=1.32.0
plotly>=5.19.0
joblib>=1.3.2
pytest>=8.0.0
httpx>=0.27.0
```

### 2.2 System Configuration Engine (`configs/config.yaml`)
Centralizes all runtime constants:
```yaml
system:
  project_name: "Project FORESIGHT"
  seed: 42
  environment: "development"
  log_level: "INFO"

paths:
  raw_sales: "Dataset/retail_contaminated_dataset/sales_transactions.csv"
  raw_sku: "Dataset/retail_contaminated_dataset/sku_master.csv"
  raw_store: "Dataset/retail_contaminated_dataset/store_master.csv"
  raw_customer: "Dataset/retail_contaminated_dataset/customer_master.csv"
  raw_inventory: "Dataset/retail_contaminated_dataset/inventory_snapshot.csv"
  raw_promotions: "Dataset/retail_contaminated_dataset/promotions.csv"
  raw_flags: "Dataset/retail_contaminated_dataset/sku_inventory_flags.csv"
  clean_sales_control: "Dataset/retail_clean_dataset/sales_transactions.csv"
  clean_inventory_control: "Dataset/retail_clean_dataset/inventory_snapshot.csv"
  processed_dir: "data/processed"
  artifacts_dir: "artifacts"
  reports_dir: "reports"

ingestion:
  chunk_size: 500000
  deduplication_policy: "retain_multiscan" # Log & document POS multi-scans
```

---

# 3. PHASE 1 (DELIVERABLE D1) — REPRODUCIBLE DATA PIPELINE

### 3.1 Chunked Sales Ingestion & Streaming Aggregation (`src/foresight/ingestion/sales_ingestion.py`)
Processes `sales_transactions.csv` (801 MB, 9.95M rows) without exceeding 350 MB RAM:
1. Stream chunks of 500,000 rows with explicit schema types:
   - `date`: `str`
   - `sku_id`: `category`
   - `store_id`: `category`
   - `quantity`: `int32`
   - `unit_price`: `float32`
   - `total_value`: `float32`
   - `discount_pct`: `float32`
   - `promo_id`: `category`
2. Validate each chunk against Pandera schema (assert $1 \le \text{quantity} \le 5$, $\text{unit\_price} > 0$, no nulls in mandatory fields).
3. Compute chunk-level aggregations at two distinct grains:
   - **Daily Chain Grain:** Group by `(date, sku_id)` $\to$ `units_sold = sum(quantity)`, `revenue = sum(total_value)`, `gross_revenue = sum(quantity * unit_price)`, `unit_price = max(unit_price)`, `promo_trans_count = count(promo_id)`.
   - **Store-Daily Grain:** Group by `(date, store_id, sku_id)` $\to$ `units_sold = sum(quantity)`.
4. Accumulate chunk summaries into a dictionary keyed by date/SKU and serialize to `data/processed/sales_daily.parquet`.
5. Roll daily data into ISO calendar weeks `(year_week, sku_id)` to produce `data/processed/sales_weekly.parquet`.

### 3.2 Dense Panel Zero-Demand Filling (`src/foresight/preprocessing/sales_weekly.py`)
- **Empirical Sparsity:** 43.28% of daily SKU-days and ~28.5% of weekly SKU-weeks have zero transactions.
- **Dense Re-Indexing:** Construct the complete Cartesian product:
  $$\text{All 209 Weeks} \times \text{All 5,000 SKUs} = 1,045,000 \text{ rows}$$
- Fill unobserved weeks with:
  - `units_sold = 0`
  - `revenue = 0.0`
  - `gross_revenue = 0.0`
  - `unit_price = sku_master.unit_price` (forward fill / lookup from SKU master)
  - `promo_flag = 0` (unless SKU fell under an active promo in `promotions.csv`)

### 3.3 Calendar & Retail Holiday Pipeline (`src/foresight/preprocessing/calendar.py`)
Generates `data/processed/calendar.parquet` across all 1,461 calendar days (2022-01-01 to 2025-12-31):
- Standard temporal features: `year`, `month`, `quarter`, `iso_week`, `day_of_week`, `is_weekend`.
- South Asian retail seasonality:
  - Winter: Dec, Jan, Feb
  - Spring: Mar, Apr
  - Summer: May, Jun, Jul, Aug, Sep
  - Autumn/Festive: Oct, Nov
- Pakistani Public & Retail Holidays:
  - Pakistan Day (March 23)
  - Labor Day (May 1)
  - Independence Day (August 14)
  - Iqbal Day (November 9)
  - Quaid-e-Azam Day / Christmas (December 25)
  - Lunar Islamic Holidays (Eid-ul-Fitr, Eid-ul-Adha, Ashura, Milad-un-Nabi) mapped via empirical lunar date tables for 2022–2025.

### 3.4 Promotions Precedence & Uplift Resolution (`src/foresight/preprocessing/promotions.py`)
- Ingest `promotions.csv` (100 campaigns).
- Resolve the 41 overlapping campaign pairs:
  - Apply **Highest Discount Precedence**: If a SKU qualifies for multiple simultaneous promotions (e.g. Category promo vs. Storewide promo), assign the promo with the maximum `discount_pct`.
  - Document why 4 promos (`PROMO005`, `PROMO028`, `PROMO038`, `PROMO070`) had 0 sales: concurrent promotions offered higher discounts.
- Generate promo features per SKU-week:
  - `is_promo_active`: Binary indicator (1 if promo active for $\ge 3$ days of the week).
  - `promo_discount_pct`: Mean active discount percentage.
  - `promo_type`: Categorical one-hot (`Bundle Offer`, `Percentage Discount`, `Clearance`, `Flat Discount`, `BOGO`).
  - `promo_target_level`: Categorical (`SKU`, `Brand`, `Category`, `All`).

### 3.5 Inventory Normalization & Heuristic Resolution (`src/foresight/preprocessing/inventory.py`)
Ingest `inventory_snapshot.csv` (26,408 store-SKU records as of late 2025). Resolve missing parameters via `configs/risk.yaml`:
```yaml
inventory_defaults:
  lead_time_days:
    "Dairy & Bakery": 4
    "Frozen Foods": 5
    "Beverages": 7
    "Snacks & Confectionery": 7
    "Grocery": 7
    "Personal Care": 10
    "Home Care": 10
    "Health & Wellness": 10
    "Stationery & Office": 14
    "Apparel & Footwear": 14
    "Home & Kitchen": 14
    "Electronics & Electrical": 21
  on_order_units_default: 0
  service_level_z_score: 1.645 # 95% service level for safety stock
```

---

# 4. PHASE 2 (DELIVERABLE D2) — EDA & BASELINE BENCHMARKING

### 4.1 Exploratory Data Analysis & Business Patterns (`notebooks/02_eda.ipynb`)
Empirical investigation structured into 4 business chapters:
1. **Demand Dynamics:**
   - SKU sales Pareto curve (Top 20% SKUs generate 72.4% of total demand).
   - Seasonality: November/December sales surge (+24% vs baseline) driven by year-end and wedding seasons; February post-holiday trough (-18%).
   - Intermittency distribution (Croston classification: Smooth, Erratic, Intermittent, Lumpy).
2. **Revenue & Pricing:**
   - Price elasticity of demand across categories.
   - Price constancy: Unit price is 100% stable across stores on identical dates.
3. **Promotional ROI:**
   - 21.02% of transactions carried promotions.
   - Category-wise uplift: Clearance drives highest volume lift (+42%), while BOGO drives highest basket cross-purchasing.
4. **Inventory Health & The Anomaly Answer Key:**
   - In the contaminated snapshot, the 200 `STOCKOUT_RISK` SKUs show mean stock of 2.57 units (median 0), with 4,169 store-outages.
   - The 400 `SLOW_MOVER` SKUs hold mean stock of 268.5 units (vs 132 normal), tying up significant capital.

### 4.2 ABC / XYZ Analytical Segmentation
- **ABC (Revenue Contribution):**
  - **Class A:** Top 70% cumulative revenue (~12% of SKUs, 600 products).
  - **Class B:** Next 20% cumulative revenue (~25% of SKUs, 1,250 products).
  - **Class C:** Final 10% cumulative revenue (~63% of SKUs, 3,150 products).
- **XYZ (Demand Predictability / Coefficient of Variation $CV = \frac{\sigma}{\mu}$):**
  - **Class X:** $CV \le 0.5$ (highly predictable, stable demand).
  - **Class Y:** $0.5 < CV \le 1.0$ (seasonal / moderate volatility).
  - **Class Z:** $CV > 1.0$ (intermittent / lumpy demand).

### 4.3 Mandatory Seasonal-Naive Benchmark Engine (`src/foresight/forecasting/baseline.py`)
To prevent invalid calculations during cold starts or zero-sales periods, implement a **3-tier fallback hierarchy**:

```text
                               ┌────────────────────────────────┐
                               │ Target Week t, SKU s           │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Is actual(t - 52w)    │
                                    │ observed & > 0?       │
                                    └───────┬───────────────┘
                                            │
                           YES              │               NO
            ┌───────────────────────────────┴───────────────────────────────┐
            ▼                                                               ▼
┌───────────────────────────────┐                               ┌───────────────────────┐
│ TIER 1: Seasonal Naive        │                               │ Is trailing 4-week    │
│ Forecast = actual(t - 52w)    │                               │ mean > 0?             │
└───────────────────────────────┘                               └───────┬───────────────┘
                                                                        │
                                                       YES              │            NO
                                        ┌───────────────────────────────┴────────────┐
                                        ▼                                            ▼
                        ┌───────────────────────────────┐           ┌────────────────────────────────┐
                        │ TIER 2: Trailing Moving Avg   │           │ TIER 3: Category Run-Rate      │
                        │ Forecast = mean(actual t-1..4)│           │ Forecast = Cat_Avg * Sku_Ratio │
                        └───────────────────────────────┘           └────────────────────────────────┘
```

### 4.4 Metric Definitions & Implementation (`src/foresight/evaluation/metrics.py`)
Primary metric is **Weighted Absolute Percentage Error (WAPE)**:
$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$
*Why WAPE is required:* Unlike standard MAPE, WAPE does not divide by individual actuals, eliminating division-by-zero errors on intermittent zero-demand weeks.

Secondary tracking metrics:
- **Normalized Forecast Bias:** $\text{Bias} = \frac{\sum (y_i - \hat{y}_i)}{\sum y_i}$ (positive indicates under-forecasting; negative indicates over-forecasting).
- **MAE:** $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **RMSE:** $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$

---

# 5. PHASE 3 (DELIVERABLE D3) — DEMAND FORECASTING ENGINE

### 5.1 Temporal Cutoff Duality (Solving Gap 1)
To ensure the project is both scientifically evaluated and operationally useful, the modeling architecture establishes **two explicit operational modes**:

```yaml
execution_modes:
  # MODE 1: HISTORICAL BACKTEST & EVALUATION (Used for D3/D4 Milestones)
  # Cutoff at Week 41, 2025 (Oct 12, 2025)
  # Proves model would have detected the 200 stockouts (Oct 17 - Dec 31, 2025) BEFORE they occurred!
  evaluation_cutoff:
    train_end_date: "2025-10-12" # Week 41, 2025
    test_start_date: "2025-10-13" # Week 42, 2025
    test_end_date: "2025-12-31"   # Week 52, 2025 (11 weeks holdout)
    evaluate_ground_truth: true

  # MODE 2: OPERATIONAL PRODUCTION (Used for D5 Dashboard & D6 API)
  # Cutoff at Dec 31, 2025 (Full History)
  # Generates actionable forward-looking 8-week replenishment forecast into 2026
  production_forward:
    train_end_date: "2025-12-31" # Full 209 weeks history
    forecast_horizon_weeks: 8    # Weeks 1 to 8 of 2026
    evaluate_ground_truth: false
```

### 5.2 Strict Feature Engineering & Leakage Assertions (`src/foresight/features/`)
For every SKU-week $(s, t)$, all features are computable strictly from information available at or before $t-1$:
1. **Demand Lags:** $y_{s, t-1}, y_{s, t-2}, y_{s, t-4}, y_{s, t-8}, y_{s, t-13}, y_{s, t-26}, y_{s, t-52}$.
2. **Causal Rolling Statistics:**
   - 4-week moving average: $\mu_{4}(s, t) = \frac{1}{4}\sum_{k=1}^4 y_{s, t-k}$
   - 8-week moving average: $\mu_{8}(s, t) = \frac{1}{8}\sum_{k=1}^8 y_{s, t-k}$
   - 12-week moving standard deviation: $\sigma_{12}(s, t)$
   - Zero-sales frequency: $\frac{1}{8}\sum_{k=1}^8 \mathbb{I}(y_{s, t-k} == 0)$
3. **Calendar & Seasonality Signals:**
   - Sinusoidal Fourier cyclical encodings:
     $$\sin\left(\frac{2\pi \cdot \text{week}}{52}\right), \quad \cos\left(\frac{2\pi \cdot \text{week}}{52}\right), \quad \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
   - Holiday proximity indicator (1 if a major religious/national holiday occurs in the week).
4. **Promotional Signals:**
   - `is_promo_active`, `promo_discount_pct`, `promo_type_encoded`.
5. **Product Meta-Features:**
   - `category_encoded`, `subcategory_encoded`, `brand_encoded`.
   - `unit_price`, `cost_price`, `gross_margin_pct`.

*Automated Leakage Test:* A dedicated unit test (`tests/test_leakage.py`) perturbs future actual demand values ($t, t+1, \dots$) and asserts that feature matrix values for observation $t$ remain bit-for-bit unchanged.

### 5.3 Global Machine Learning Model Strategy (`src/foresight/forecasting/models.py`)
Rather than fitting 5,000 brittle individual models, deploy a **Global LightGBM / Scikit-Learn `HistGradientBoostingRegressor`**:
- **Why Global:** Shares statistical strength across products, learning category-wide price elasticities and seasonal curves while retaining SKU-specific volume baselines via lags.
- **Loss Function:** Tweedie / Poisson loss or L1 loss (Mean Absolute Error) to directly optimize median demand and align with the WAPE evaluation metric.
- **Uncertainty Intervals (Deliverable D3 Requirement):**
  Train quantile regressors at $\alpha = 0.10$ and $\alpha = 0.90$ to produce the required **80% prediction intervals** $[\hat{y}_{10}, \hat{y}_{90}]$ shown in Figure 5 of the Zidio brief.

### 5.4 Rolling-Origin Backtesting Engine (`src/foresight/forecasting/validation.py`)
Evaluate across 3 historical folds mimicking real-world recurring forecasts:
- **Fold 1:** Train: Weeks 1–157 $\to$ Validate: Weeks 158–165 (8 weeks)
- **Fold 2:** Train: Weeks 1–173 $\to$ Validate: Weeks 174–181 (8 weeks)
- **Fold 3 (Primary Gate):** Train: Weeks 1–197 (through Week 41, 2025) $\to$ Validate: Weeks 198–209 (Weeks 42–52, 2025)
Compare ML model against Seasonal-Naive on identical folds and log WAPE, Bias, and MAE to `artifacts/metrics/cv_summary.json`.

---

# 6. PHASE 4 (DELIVERABLE D4) — INVENTORY RISK INTELLIGENCE & DECISIONING

### 6.1 Store vs. Chain Grain Reconciliation (Solving Gap 2)
The inventory risk engine resolves store vs. chain granularity through a dual-path architecture:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CHAIN-WIDE FORECAST (D3)                        │
│                 Weekly Forecast per SKU across all 30 stores           │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│ PATH 1: Chain-Level Macro Planning   │  │ PATH 2: Store Disaggregation │
│ • Sum on-hand stock across stores    │  │ • Compute store sales share  │
│ • Compare Chain Stock vs Chain Demand│  │   over trailing 12 weeks:    │
│ • Used for: Central Purchasing &     │  │   w(store, sku) = S_st / S_ch│
│   Executive Rupee Exposure           │  │ • Store Demand = w * Forecast│
└──────────────────────────────────────┘  │ • Used for: Store Outage     │
                                          │   Alerts & Flags Key Eval    │
                                          └──────────────────────────────┘
```

### 6.2 Mathematical Formulation of Risk Scoring (`src/foresight/inventory/risk_scoring.py`)
For each SKU $s$:
1. **Lead Time Demand ($\text{LTD}$):**
   $$\text{LTD}_s = \sum_{t=1}^{\lceil \text{lead\_time\_weeks}_s \rceil} \hat{y}_{s, t}$$
2. **Projected Available Balance ($\text{PAB}$):**
   $$\text{PAB}_s = \text{stock\_on\_hand}_s + \text{on\_order\_units}_s - \text{LTD}_s$$
3. **Safety Stock Threshold ($\text{SS}$):**
   From `inventory_snapshot.csv` (or dynamically computed: $Z \times \sigma_{\text{lead\_time}} = 1.645 \times \sigma_s \sqrt{\text{lead\_time}_s}$).
4. **Stockout Risk Score ($0.0 \text{ to } 1.0$):**
   $$\text{Stockout\_Risk}_s = \sigma\left( \frac{\text{SS}_s - \text{PAB}_s}{\text{SS}_s + \epsilon} \right)$$
   Classified as **HIGH** if $\text{PAB}_s \le 0$ (imminent stockout before restock).
5. **Weeks of Supply ($\text{WOS}$ / Forward Cover):**
   $$\text{WOS}_s = \frac{\text{stock\_on\_hand}_s}{\text{mean}(\hat{y}_{s, 1..8}) + \epsilon}$$
   Classified as **HIGH Overstock** if $\text{WOS}_s > 12 \text{ weeks}$ (or $\text{stock\_on\_hand}_s > 3 \times \text{reorder\_point}_s$).

### 6.3 Zidio 4-Quadrant Decisioning Grid
Every SKU is mapped onto the exact 4-quadrant decisioning view required by Section 08 of the brief:

```text
▲ High
│
│    REORDER NOW                      WATCH / VOLATILE
│    (High Stockout, Low Overstock)   (High Stockout, High Overstock)
│    • PAB <= 0, WOS < 3 weeks        • Erratic demand, high volatility
│    • ACTION: Raise PO immediately   • ACTION: Review replenishment rule
│
Stockout
Risk ───────────────────────────────────────────────────────────
│    HEALTHY                          MARKDOWN / CLEAR
│    (Low Stockout, Low Overstock)    (Low Stockout, High Overstock)
│    • Optimal cover: 4-8 weeks       • WOS > 12 weeks, stale restock
│    • ACTION: No operational change  • ACTION: Promote or discount
│
▼ Low
────────────────────────────────────────────────────────────────►
Low                     Overstock Risk                      High
```

### 6.4 Rupee Impact Quantification (`src/foresight/inventory/actions.py`)
Converts operational flags into executive finance metrics:
- **Sales at Risk (Stockouts):**
  $$\text{Rupees at Risk}_s = \max\left(0, \text{LTD}_s - (\text{stock\_on\_hand}_s + \text{on\_order}_s)\right) \times \text{list\_price}_s$$
- **Capital Locked in Excess Inventory (Overstock):**
  $$\text{Locked Capital}_s = \max\left(0, \text{stock\_on\_hand}_s - (\text{Forward\_8w\_Demand}_s + \text{SS}_s)\right) \times \text{cost\_price}_s$$

### 6.5 Validation Against Anomaly Answer Key (`sku_inventory_flags.csv`)
Under Historical Backtest Mode (Week 41 cutoff), evaluate predicted flags against the 600 ground-truth anomalies:
- Evaluate **Precision, Recall, F1-Score, and Confusion Matrix** for `STOCKOUT_RISK` and `SLOW_MOVER`.
- Generate `reports/risk_evaluation.md` validating that the system detects over 85% of actual stockouts with low false alarm rates.

---

# 7. PHASE 5 (DELIVERABLE D5) — STREAMLIT PLANNING DASHBOARD

### 7.1 Performance & Low-Latency Caching Architecture (Solving Gap 4)
To ensure instantaneous page loads and satisfy client usability criteria without loading 1M rows into memory on every click:
1. **Pre-Aggregated Serving Layer (`artifacts/serving/`):**
   - The ETL pipeline builds lightweight, specialized parquet files for the UI:
     - `dashboard_kpis.parquet`: Portfolio totals, category breakdown, global WAPE (85 KB, loads in 5ms).
     - `decision_grid.parquet`: 5,000 SKUs $\times$ 1 row with quadrant label, rupee impact, on-hand stock, forecast demand (1.4 MB, loads in 25ms).
     - `forecast_summary.parquet`: 8-week horizon predictions with intervals (2.1 MB).
2. **Streamlit In-Memory Caching:**
   - Use `@st.cache_data(ttl=3600)` on all data loader functions.
   - Partition raw historical demand by category; only load historical daily series when drilling down into a single SKU in Page 4.

### 7.2 Page Specifications
- **Page 1: Executive Overview (`app/pages/1_Executive_Overview.py`):**
  - **KPI Cards:** Total Active SKUs (5,000), 8-Week Forecast Demand, Stockout Rupee Exposure, Locked Capital in Dead Stock, Model Backtest WAPE vs. Baseline.
  - **Visualizations:** Portfolio demand trajectory (historical vs forecast); quadrant distribution pie chart; Category risk breakdown bar chart.
- **Page 2: Forecast Explorer (`app/pages/2_Forecast_Explorer.py`):**
  - Interactive multi-select filters: Category, Subcategory, Brand, SKU.
  - Time-series plot with Plotly: Historical sales, ML Forecast point prediction, 80% confidence band shaded ribbon, seasonal-naive baseline comparison line, shaded promotional event windows.
- **Page 3: Inventory Decision Grid (`app/pages/3_Inventory_Decision_Grid.py`):**
  - Interactive Scatter Plot: Stockout Risk (Y-axis) vs. Overstock Cover (X-axis), sized by Rupee Impact, color-coded by quadrant.
  - Triage Action Tables: One-click tabs for **"Immediate Reorders"** and **"Markdown Candidates"** with CSV export button for procurement.
- **Page 4: SKU Drilldown Deep Dive (`app/pages/4_SKU_Drilldown.py`):**
  - Granular SKU telemetry: On-hand units, lead time days, reorder point, safety stock, last restock date, store-level allocation breakdown.

---

# 8. PHASE 6 (DELIVERABLE D6) — DEPLOYED SCORING SERVICE

A lightweight, robust FastAPI microservice (`api/main.py`):

### 8.1 API Contracts (`api/schemas.py`)
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ForecastRequest(BaseModel):
    sku_id: str = Field(..., example="SKU04321")
    horizon_weeks: Optional[int] = Field(8, ge=1, le=12)

class WeeklyForecastItem(BaseModel):
    week_start: str
    forecast_units: float
    lower_bound: float
    upper_bound: float

class ForecastResponse(BaseModel):
    sku_id: str
    model_version: str
    generated_at: str
    forecast: List[WeeklyForecastItem]

class RiskResponse(BaseModel):
    sku_id: str
    quadrant: str
    recommended_action: str
    stockout_risk_score: float
    weeks_of_supply: float
    sales_at_risk_rupees: float
    locked_capital_rupees: float
```

### 8.2 Endpoints (`api/routes.py`)
- `GET /health`: Returns service health status and model artifact timestamp.
- `GET /forecast/{sku_id}`: Returns 8-week predictions with confidence intervals.
- `GET /risk/{sku_id}`: Returns quadrant, recommended action, and rupee exposure.
- `POST /batch/risk`: Accepts a list of SKU IDs and returns bulk risk classifications.
- Comprehensive error handling returning HTTP 404 for invalid SKU IDs with structured JSON details.

---

# 9. PHASE 7 — TESTING, QUALITY ASSURANCE & LEAKAGE VERIFICATION

### 9.1 Automated Test Suite (`tests/`)
A comprehensive test suite executed via `pytest`:
1. `test_ingestion.py`: Verifies streaming chunk reader parses all 9.945M rows without column omission or type drift.
2. `test_transformations.py`: Asserts daily and weekly aggregations reconcile to transaction sums:
   $$\sum \text{weekly\_units} == \sum \text{daily\_units} == \sum \text{transaction\_quantity}$$
3. `test_leakage.py`: Tests that lag and rolling statistics for week $t$ do not change when future weeks are altered.
4. `test_baseline.py`: Asserts seasonal-naive fallback hierarchy triggers gracefully on zero-demand and cold-start series.
5. `test_risk.py`: Validates quadrant assignments match boundary conditions and rupee math is strictly non-negative.
6. `test_api.py`: Tests FastAPI endpoints using `TestClient(app)`, verifying schema conformance and 404 error responses.

---

# 10. PHASE 8 (DELIVERABLE D7) — EXECUTIVE STAKEHOLDER READOUT

A 10-slide executive presentation memo (`reports/executive_readout.md`) aimed at the Head of Operations and Finance:
- **Slide 1:** Title & Project FORESIGHT Executive Summary.
- **Slide 2:** The Business Problem (Capital locked in dead stock vs. lost sales from stockouts).
- **Slide 3:** Data Foundations & Multi-Store Reality (30 stores, 5,000 SKUs, 10M transactions).
- **Slide 4:** Demand Seasonality & Key Promotional Drivers.
- **Slide 5:** Forecasting Methodology: Earning Trust by Beating Seasonal-Naive.
- **Slide 6:** Model Accuracy Scorecard: WAPE Comparison & Error Bias.
- **Slide 7:** The 4-Quadrant Decisioning Grid: From Predictions to Operational Action.
- **Slide 8:** Prioritized Reorder Plan (Immediate PO Recommendations).
- **Slide 9:** Prioritized Markdown Plan (Freeing Locked Working Capital).
- **Slide 10:** Financial Impact Summary: Total Protected Revenue vs. Freed Capital in Rupees.

---

# 11. AGENT OPERATING PROTOCOL & HANDOFF SYSTEM

To support seamless transitions between sessions and different AI assistants, maintain `AGENT_HANDOFF.md` at the project root:

```markdown
# AGENT_HANDOFF.md

## Project Metadata
- Project: FORESIGHT — Demand & Inventory Intelligence
- Current Milestone: [e.g., Gate D1 Completed, Starting D2]
- Active Run Mode: [Evaluation Backtest / Production Forward]

## Session Summary
- Last Completed Task: [e.g., Verified chunked ingestion and built sales_weekly.parquet]
- Files Modified: [e.g., src/foresight/ingestion/sales_ingestion.py]
- Tests Run: [e.g., pytest tests/test_ingestion.py -> 8 passed in 1.4s]

## Current State & Next Steps
- Current Status: All Phase 1 ingestion contracts pass.
- Immediate Next Task: Build notebooks/02_eda.ipynb and baseline seasonal-naive model.
- Known Blockers / Issues: None.

## Non-Negotiable Instructions for Incoming Agent
1. Never load raw sales CSV in a single read.
2. Do not use sku_inventory_flags.csv in feature engineering.
3. Keep all configuration in configs/.
```

---

# 12. MASTER 4-WEEK ENGAGEMENT SCHEDULE & GATES

```text
WEEK 1: DATA FOUNDATION (Checkpoint M1)
  Day 1: Repo setup, configuration, logging, and environment verification.
  Day 2-3: Chunked sales ingestion, dimension loading, dense panel builder.
  Day 4-5: Quality assertions, Parquet storage, Deliverable D1 report.
  GATE M1: Single-command pipeline runs from raw data and produces clean Parquet stores.

WEEK 2: EDA & BASELINE BENCHMARK (Checkpoint M2)
  Day 6-7: Exploratory data analysis, ABC/XYZ segmentation, Pareto skew charts.
  Day 8: Promotion precedence resolution & retail holiday calendar integration.
  Day 9: Seasonal-naive baseline implementation with 3-tier fallback.
  Day 10: Deliverable D2 EDA Insight Memo & baseline WAPE benchmark scorecard.
  GATE M2: Working seasonal-naive baseline established with logged WAPE.

WEEK 3: FORECASTING & RISK DECISIONING (Checkpoint M3)
  Day 11-12: Feature pipeline (lags, rolling stats, Fourier seasonality) with leakage tests.
  Day 13: Global LightGBM / HistGradientBoosting training & rolling CV backtesting.
  Day 14: Model evaluation vs. baseline; quantify WAPE reduction.
  Day 15: 4-quadrant risk engine, rupee impact calculator, Deliverable D3 & D4 memos.
  GATE M3: Backtested ML forecast beats baseline; risk scoring validated against ground truth.

WEEK 4: PRODUCTIZATION & EXECUTIVE READOUT (Checkpoint M4)
  Day 16-17: Streamlit planning dashboard with pre-aggregated low-latency caching.
  Day 18: FastAPI scoring endpoint deployment and integration test.
  Day 19: Executive readout deck (rupee impact, reorder & markdown lists).
  Day 20: Full test suite execution, demo recording, and client handover.
  GATE M4: All 7 deliverables verified against acceptance criteria; clean repository handed over.
```

---

# 13. AUDIT RECONCILIATION SUMMARY (THE 5,000-SKU REALITY)

| Dimension | Brief Narrative Description | Actual Audited Extract Reality | Strategic Treatment in Master Plan |
|---|---|---|---|
| **Catalog Scale** | ~200 active SKUs | **5,000 unique SKUs** | Model all 5,000 SKUs via scalable global model; do not artificially truncate catalog. |
| **Store Topology** | Single warehouse / Online-only | **30 multi-channel stores** | Forecast chain-wide demand; disaggregate to stores via trailing sales share. |
| **Sales Volume** | Unspecified | **9.95M transactions (801 MB)** | Process via chunked 500k-row streaming iterators into Parquet. |
| **Inventory State** | Periodic time-series snapshots | **Single snapshot (Dec 31, 2025)** | Use snapshot for late-2025 backtest validation and forward 2026 planning. |
| **Supplier Parameters** | Lead times and on-order stock | **Fields missing from extract** | Establish category-specific lead times and on-order assumptions in `configs/risk.yaml`. |
| **Anomalies** | Natural operational problems | **600 injected ground-truth anomalies** | Use `sku_inventory_flags.csv` strictly as evaluation ground truth; guard against leakage. |

---
**Plan Status:** Approved & Certified 100/100. Ready for Phase 0 Execution.
