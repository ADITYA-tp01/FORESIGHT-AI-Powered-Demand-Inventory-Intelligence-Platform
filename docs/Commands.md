# PROJECT FORESIGHT — LIVE PRESENTATION & COMMAND CHEATSHEET

> **Platform:** AI-Powered Demand & Inventory Intelligence Platform  
> **Client:** NorthBay Living (5,000 SKUs, 30 Stores, 9.95M Transactions)  
> **Target Audience:** Head of Operations, Supply Chain Directors & Chief Financial Officer (CFO)

---

## 0. Quick Terminal Initialization

Always open **PowerShell** and navigate to the project directory:

```powershell
cd "c:\Users\adity\Documents\Zidio\FORESIGHT – AI-Powered Demand & Inventory Intelligence Platform"
.venv\Scripts\activate
```

---

## 1. Live Demo: Interactive Operations Dashboard (Streamlit)

**Best for:** Presenting to business leadership, inventory managers, and procurement leads.

### Command to Run:
```powershell
streamlit run app/streamlit_app.py
```
> Opens automatically in your browser at: **`http://localhost:8501`**

### Live Presentation Walkthrough:
1. **Page 1 — Executive Overview:**
   - Highlight portfolio KPIs: **5,000 SKUs**, **INR 32.79M Sales at Risk**, and **INR 1.53B Working Capital Locked**.
   - Show the portfolio trajectory (historical demand vs. 8-week ML forward forecast).
   - Point out the category risk mix breakdown.
2. **Page 2 — Forecast Explorer:**
   - Filter by Category (e.g. `Beverages` or `Grocery`) and select a SKU (e.g. `SKU00001`).
   - Show the **ML point forecast** vs. the **Seasonal-Naive baseline**.
   - Show the **80% prediction interval (shaded ribbon)** and explain uncertainty bounds for safety stock.
   - Point out promotional campaign overlays that drive demand spikes.
3. **Page 3 — Inventory Decision Grid:**
   - Show the **4-Quadrant Scatter Matrix** (Stockout Risk vs. Forward Weeks of Supply).
   - Click the **"Immediate Reorders"** tab: show the 325 SKUs needing immediate PO placement. Click **Export CSV** to demonstrate operational handover to buyers.
   - Click the **"Markdown Candidates"** tab: show the 3,021 overstocked SKUs ready for promotional discount to recover INR 1.53B in cash.
4. **Page 4 — SKU Drilldown:**
   - Pick an individual SKU to show deep telemetry: lead time days, reorder point, safety stock, and store-by-store inventory distribution.

---

## 2. Live Demo: Real-Time Scoring Microservice (FastAPI)

**Best for:** Presenting to enterprise architects, technical directors, and software engineering teams.

### Command to Run:
```powershell
python -m uvicorn api.main:app --port 8000 --reload
```
> Open Swagger UI in your browser at: **`http://localhost:8000/docs`**

### Live Endpoints to Demonstrate:
* **System Health:**
  ```text
  GET http://localhost:8000/health
  ```
  *Shows service status, version 1.0.0, and serving artifact freshness timestamp.*
* **Forward 8-Week Forecast for a SKU:**
  ```text
  GET http://localhost:8000/forecast/SKU00001
  ```
  *Returns weekly predicted units with exact 10th and 90th percentile bounds.*
* **Real-Time Inventory Risk Classification:**
  ```text
  GET http://localhost:8000/risk/SKU00001
  ```
  *Returns quadrant label (`reorder_now`, `healthy`, `markdown_clear`), weeks of supply, and rupee impact.*
* **Batch Enterprise Risk Scoring:**
  ```text
  POST http://localhost:8000/batch/risk
  Payload: {"sku_ids": ["SKU00001", "SKU00002", "SKU00100"]}
  ```

---

## 3. Live Demo: Single-Command Pipeline Reproducibility (Rule 7)

**Best for:** Proving scientific rigor, data governance, and complete end-to-end automation.

### Command to Run (End-to-End):
```powershell
python -m foresight.pipeline
```
*Executes all phases (0 through 5) sequentially from raw data to serving cache.*

### Granular Phase Demonstration Commands:
| Phase | Command | What It Proves to the Client |
|---|---|---|
| **Phase 0** | `python -m foresight.pipeline --phase 0` | Validates environment, verifies 801 MB sales file, and checks YAML configs. |
| **Phase 1** | `python -m foresight.pipeline --phase 1` | Demonstrates chunked 500k-row streaming ingestion (<500 MB RAM) & dense panels. |
| **Phase 2** | `python -m foresight.pipeline --phase 2` | Generates ABC/XYZ segmentation & computes Seasonal-Naive benchmark (37.76% WAPE). |
| **Phase 3** | `python -m foresight.pipeline --phase 3` | Trains Global HGB ML model, beats baseline across 3/3 folds (30.81% WAPE). |
| **Phase 4** | `python -m foresight.pipeline --phase 4` | Runs 4-quadrant decisioning and verifies 86.5% stockout recall against ground truth. |
| **Phase 5** | `python -m foresight.pipeline --phase 5` | Builds pre-aggregated serving cache for sub-25ms dashboard load times. |

---

## 4. Live Demo: Test Suite & Code Quality Audit

**Best for:** Defending the technical implementation against engineering audits.

### Run Automated Tests (121 Tests, 100% Passing):
```powershell
python -m pytest tests -q
```
*Proves:*
- Ingestion chunking & exact 18,700,906 unit reconciliation
- Zero feature leakage (`test_leakage.py` strictly tests that future sales cannot leak into features)
- Baseline fallback correctness
- 4-quadrant boundary consistency and non-negative rupee formulas
- FastAPI endpoint schema conformance
- Streamlit headless AppTest smoke tests

### Run Code Style & Linter:
```powershell
python -m ruff check src tests api app
```
*Proves 100% clean production code without lint errors.*

---

## 5. Presentation Talking Points & Golden Numbers

Keep these exact verified figures handy during Q&A:

```text
================================================================================
                    PROJECT FORESIGHT — EXECUTIVE SCORECARD
================================================================================
Catalog Scale:               5,000 SKUs across 12 categories, 30 stores
Sales Volume:                9,945,511 contaminated transactions (801 MB CSV)
Total Units Sold:            18,700,906 units (exact 3-way reconciliation)

FORECASTING ACCURACY:
  • Seasonal-Naive Baseline: 37.76% WAPE (3-tier fallback hierarchy)
  • FORESIGHT Global ML:     30.81% WAPE (-18.39% relative error reduction)
  • Fold Consistency:        Beat baseline on 3 out of 3 backtest folds
  • 80% Prediction Band:     77.63% empirical coverage on out-of-sample holdout

INVENTORY RISK ACCURACY (vs. Answer Key):
  • Stockout Risk Recall:    86.5% (Exceeds >= 85.0% target)
  • Store Outage Coverage:   100% (All 2,703 key store-outage pairs captured)
  • Slow Mover F1-Score:     0.872 (100% recall on dead stock)

FINANCIAL IMPACT (RUPEES):
  • Protected Revenue:       INR 32,785,603 (Sales at imminent risk of stockout)
  • Total Locked Capital:    INR 1,534,592,370 in excess/stale inventory
  • Actionable Cash Recovery:INR 1,533,774,315 (99.9% of locked capital recovered)
  • Operational Priorities:  325 Immediate Purchase Orders | 3,021 Markdowns
================================================================================
```

---

## 6. Deliverables Reference Index

| Deliverable | File Path | Description |
|---|---|---|
| **Executive Readout (10 Slides)** | [`reports/executive_readout.md`](reports/executive_readout.md) | Primary slide memo for Head of Operations & Finance |
| **Data Audit Report (D1)** | [`reports/data_quality_report.md`](reports/data_quality_report.md) | Ingestion and unit reconciliation audit |
| **EDA & Baseline Memo (D2)** | [`reports/eda_report.md`](reports/eda_report.md) | Seasonality, Pareto curve, and ABC/XYZ analysis |
| **Model Evaluation (D3)** | [`reports/model_evaluation.md`](reports/model_evaluation.md) | ML vs Baseline scorecard and prediction intervals |
| **Risk Intelligence (D4)** | [`reports/risk_evaluation.md`](reports/risk_evaluation.md) | Validation against `sku_inventory_flags.csv` |
| **Master Implementation Plan** | [`docs/Plan.md`](docs/Plan.md) | 100/100 Master Engagement Specification |
