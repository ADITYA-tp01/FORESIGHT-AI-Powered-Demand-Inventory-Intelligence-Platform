# AGENT_HANDOFF.md

## Project Metadata
- Project: FORESIGHT — Demand & Inventory Intelligence
- Current Milestone: **100.0% PROJECT COMPLETION CERTIFIED.** All phases (0–8), deliverables D1–D7,
  all 6 analytical notebooks (01–06), single-command pipeline runner (Rule 7), Streamlit dashboard,
  FastAPI scoring microservice, and master documentation complete. Full suite 121 passed, ruff clean.
- Active Run Mode: Historical backtest (evaluation_cutoff in configs/forecast.yaml); production_forward bundle trained for 2026-W02..W09.

## Session Summary
- Last Completed Task: Phase 8 — wrote `reports/executive_readout.md` (10-slide executive readout, audience Head of
  Operations & Finance). Every figure verified programmatically against artifacts before delivery (cv_summary,
  flag_evaluation, risk_scores, dashboard_kpis, D1/D2 reports); two precision fixes applied (Slide 9 row values
  re-rounded from exact parquet sums; Slide 8 top-10 stated as ≈20.05M to avoid 1-rupee rounding drift).
- Readout headlines: WAPE 37.76% → 30.81% (18.39% rel., 3/3 folds); stockout recall 0.865 ≥ 0.85 MET;
  protected revenue INR 32,785,603; freed capital INR 1,533,774,315 of INR 1,534,592,370 total locked;
  325 immediate POs + 3,021 markdowns; honest caveats included (single snapshot, precision 0.453 bounded by
  staleness, RMSE slightly higher for ML, watch_volatile legitimately 0).
- Final gate: `python -m pytest tests -q` -> **121 passed**; `ruff check src tests api app` -> clean.

## Current State & Next Steps
- Current Status: **Gate M3 forecast criterion met** (Phase 3): ML overall WAPE **30.81%** vs baseline **37.76%**
  (18.39% relative reduction; folds 38.01→30.33, 35.29→26.96, 39.05→33.28); 80% interval coverage fold 3 = 77.63%.
- **Gate M3 risk criterion met** (Phase 4, `artifacts/metrics/flag_evaluation.json` + `reports/risk_evaluation.md`):
  - Dedicated model trained ONLY through **2025-W41** (never the production bundle — leakage guard);
    eval forecast cached at `artifacts/forecasts/eval_cutoff_2025-W41_8w.parquet`, model `artifacts/models/eval_cutoff_2025-W41_point.joblib`.
  - STOCKOUT_RISK: recall **0.865 ≥ 0.85 target MET**, precision 0.453, F1 0.595 (tp 173 / fp 209 / fn 27).
  - SLOW_MOVER: recall **1.000**, precision 0.774, F1 **0.872** (overstock + restock age ≥ 180 d).
  - Store-pair stockout: recall **1.000**, precision 0.567, F1 0.723 (2703 key pairs all covered).
  - Honest caveats in report: single snapshot (2025-12-31) vs Oct–Dec key window limits precision; thresholds
    pre-registered in `configs/risk.yaml`, not tuned to labels; watch_volatile quadrant is legitimately 0
    (PAB ≤ 0 and overstock are near-contradictory by construction).
- Phase 4 outputs: `artifacts/risk/risk_scores.parquet` (4,495 SKUs × 27 cols; 505 forecast SKUs excluded — no stock rows),
  `artifacts/risk/store_risk.parquet` (26,408 store×sku, shares sum to 1), production grid = healthy 1,149 /
  markdown_clear 3,021 / reorder_now 325 / watch_volatile 0; Rupees at Risk INR 32.79M, Locked Capital INR 1,534.59M.
- Caching: `artifacts/metrics/phase4_progress.json` holds the eval-cutoff model checkpoint; delete eval model +
  forecast parquet to force retrain after config changes. `artifacts/serving/serving_manifest.json` gates all
  dashboard tests (delete it to confirm skipif behavior; rebuild with `--phase 5`).
- Phase 5 serving artifacts (`artifacts/serving/`): `dashboard_kpis.parquet` (long-format section/category/metric/value),
  `decision_grid.parquet` (5,000 rows incl. 505 `not_scored` with NaN risk, 0.0 rupees — never fabricated),
  `forecast_summary.parquet` (8w × q10/q90 + `baseline` from `generate_baseline_forecasts`),
  `history_weekly.parquet` + `history/<category-slug>.parquet`, `promo_windows.parquet`, `serving_manifest.json`.
  Dashboard knobs in `configs/config.yaml → dashboard:` (cache ttl 3600 s, max 10 SKUs/page, 52-week trajectory).
  Rebuild serving: `python -m foresight.pipeline --phase 5` (~3 s, idempotent); launch dashboard:
  `streamlit run app/streamlit_app.py` (headless boot verified); AppTest page smokes in `tests/test_app.py`.
- Completion status: all plan sections executed — §1–§6 (pipeline D1–D4), §7 (dashboard D5), §8 (API D6),
  §9 (QA: all six named tests present incl. real-file ingestion stream, three-way reconciliation, `test_risk.py`
  boundaries + non-negative sweep; QA fix: `sigmoid` → `scipy.special.expit`, Phase 4 results unchanged),
  §10 (readout D7). Verification commands: `python -m pytest tests -q` (121), `ruff check src tests api app`,
  `python -m foresight.pipeline --phase N`, `streamlit run app/streamlit_app.py`, `python -m uvicorn api.main:app`.
- Phase 6 build notes: `api/{__init__,schemas,store,routes,main}.py` + `configs/config.yaml → api.max_batch_skus:100` +
  `ApiSettings` in `config.py` + `tests/test_api.py` (14); pyproject `pythonpath = ["src", "."]`. Live-verified with
  `python -m uvicorn api.main:app` (bootstrap in `api/__init__.py` makes PYTHONPATH unnecessary).
- Phase 6 API notes: risk numbers are `null` for `not_scored` SKUs (505 forecast-only SKUs) — JSON cannot carry NaN;
  quadrant action text sourced from `configs/risk.yaml → quadrants` (fallback for not_scored); `model_version` =
  package `__version__`; `generated_at` = serving manifest; health `model_artifact_timestamp` = mtime of
  `artifacts/forecasts/forecast_8w.parquet` (fallback manifest timestamp); batch dedupes input, 404 lists missing ids,
  cap from config (`api.max_batch_skus`); `FORESIGHT_SERVING_DIR` env overrides the store path; tests override the
  `get_store` dependency (mini serving built in tmp_path).
- Note later-phases test still expects `run_pipeline(phase=6)` to raise (API is not a pipeline phase).
- Known Blockers / Issues: `make` not installed on this Windows host; use `python -m foresight.pipeline --phase N`
  (or set PYTHONPATH=src when running python -c). SKU category in extract is `Electronics & Accessories`.
  lunar holidays only known 2022–2025 (2026 calendar has fixed holidays only); promo rows end 2025-11-21 (2026 promos = 0).

## Non-Negotiable Instructions for Incoming Agent
1. Never load raw sales CSV in a single read. Use `chunksize=500_000`.
2. Do not use `sku_inventory_flags.csv` in feature engineering (evaluation resource only; Phase 4's
   `flag_evaluation.py` is the single sanctioned reader, invoked from the evaluation step only).
3. Keep all configuration in `configs/`.
4. Work only inside this FORESIGHT folder. Do not write project code into `Docs/` or `Dataset/`.
5. ML model must beat baseline WAPE on identical folds (Fold 1/2/3 above) before deployment claims.
6. All validation must be rolling-origin temporal; `shuffle=False` enforced by config. sklearn HGB `early_stopping`
   must stay `false` (its internal split shuffles — Rule 5).
7. Never add `revenue`, `gross_revenue`, `promo_trans_count` to the feature matrix (concurrent-actual leakage;
   enforced by test + `FORBIDDEN_FEATURES` guard in `feature_pipeline.py`).
8. Dashboard/serving code must read pre-aggregated Parquet (`artifacts/risk/*`, `artifacts/forecasts/*`) — never
   retrain or touch raw CSVs at request time.
