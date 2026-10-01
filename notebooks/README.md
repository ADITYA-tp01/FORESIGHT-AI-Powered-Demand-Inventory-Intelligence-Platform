# Project FORESIGHT — Analytical & Research Notebooks

This directory contains the 6 master discovery, benchmarking, and evaluation notebooks specified in Section 1 of `Plan.md`:

1. **`01_data_audit.ipynb`** — Discovery verification, dimension validation, Pandera schema assertions, and streaming chunked sales ingestion demonstration.
2. **`02_eda.ipynb`** — Comprehensive EDA (seasonality curves, Pareto revenue skew, price elasticity, dead stock analysis, ABC/XYZ segmentation).
3. **`03_baseline.ipynb`** — Mandatory Seasonal-Naive baseline ($t-52$ weeks) implementation, 3-tier fallback hierarchy, and rolling-origin WAPE baselining.
4. **`04_forecasting.ipynb`** — Global machine learning feature pipeline, HistGradientBoosting training, WAPE reduction backtesting, and 80% prediction intervals.
5. **`05_risk_analysis.ipynb`** — 4-quadrant inventory risk decisioning (LTD, PAB, WOS) and ground-truth validation against `sku_inventory_flags.csv`.
6. **`06_results_consolidation.ipynb`** — Executive charts, financial rupee impact reconciliation, and prioritized PO / markdown action tables.
