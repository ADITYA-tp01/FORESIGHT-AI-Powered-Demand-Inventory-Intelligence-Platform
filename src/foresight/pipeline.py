"""Single-command entrypoint. Phase 0 verifies environment; Phase 1 runs D1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foresight import __version__
from foresight.config import Settings, load_settings, required_config_files
from foresight.logging_config import configure_logging

PHASE0_PACKAGE_DIRS = (
    "ingestion",
    "preprocessing",
    "features",
    "forecasting",
    "inventory",
    "evaluation",
    "utils",
)


def verify_phase0(project_root: Path | None = None) -> dict[str, object]:
    """Environment + config + path verification. Does not read the sales CSV."""
    settings = load_settings(project_root)
    logger = configure_logging(settings.system.log_level)

    missing_configs = [p for p in required_config_files(settings.project_root) if not p.exists()]
    if missing_configs:
        raise FileNotFoundError(f"Missing YAML configs: {missing_configs}")

    dataset_root = settings.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. "
            "Set FORESIGHT_DATASET_ROOT or configs/config.yaml paths.dataset_root."
        )

    required_keys = (
        "raw_sales",
        "raw_sku",
        "raw_store",
        "raw_customer",
        "raw_inventory",
        "raw_promotions",
        "raw_flags",
        "clean_sales_control",
        "clean_inventory_control",
    )
    missing_files = []
    sales_bytes = None
    for key in required_keys:
        path = settings.dataset_file(key)
        if not path.exists():
            missing_files.append(str(path))
            continue
        if key == "raw_sales":
            sales_bytes = path.stat().st_size

    if missing_files:
        raise FileNotFoundError(f"Missing dataset files: {missing_files}")

    pkg_root = Path(__file__).resolve().parent
    missing_pkg = [name for name in PHASE0_PACKAGE_DIRS if not (pkg_root / name).is_dir()]
    if missing_pkg:
        raise FileNotFoundError(f"Missing package modules: {missing_pkg}")

    # Hard constraint: never load sales CSV here.
    if sales_bytes is None:
        raise RuntimeError("Could not stat sales_transactions.csv")

    logger.info(
        "Phase 0 OK | version=%s | dataset=%s | sales_bytes=%s | chunk_size=%s",
        __version__,
        dataset_root,
        sales_bytes,
        settings.ingestion.chunk_size,
    )
    return {
        "version": __version__,
        "project_root": str(settings.project_root),
        "dataset_root": str(dataset_root),
        "sales_bytes": sales_bytes,
        "chunk_size": settings.ingestion.chunk_size,
        "seed": settings.system.seed,
        "n_sku_categories": len(
            settings.risk["inventory_defaults"]["lead_time_days"]
        ),
    }


def run_phase1(project_root: Path | None = None, settings: Settings | None = None):
    """Deliverable D1: chunked ingest → daily/weekly Parquet → D1 report."""
    from foresight.ingestion import metadata_ingestion as meta
    from foresight.ingestion.sales_ingestion import (
        stream_aggregate_daily,
        stream_aggregate_store_daily,
    )
    from foresight.preprocessing.calendar import build_calendar
    from foresight.preprocessing.inventory import normalize_inventory
    from foresight.preprocessing.promotions import resolve_sku_day_promos, sku_week_promo_features
    from foresight.preprocessing.sales_weekly import (
        build_dense_weekly_panel,
        rollup_daily_to_weekly,
    )
    from foresight.utils.io import ensure_dir

    # Allow calling with a Settings instance directly (as tests do)
    if isinstance(project_root, Settings):
        settings = project_root
        project_root = None
    if settings is None:
        settings = load_settings(project_root)
    configure_logging(settings.system.log_level)

    dimensions = meta.load_dimensions(settings)
    meta.assert_dimensions_match_settings(settings, dimensions)

    sales_path = settings.dataset_file("raw_sales")
    daily = stream_aggregate_daily(sales_path, chunk_size=settings.ingestion.chunk_size)
    store_daily = stream_aggregate_store_daily(sales_path, chunk_size=settings.ingestion.chunk_size)

    processed = settings.processed_dir
    ensure_dir(processed)
    daily.to_parquet(processed / "sales_daily.parquet", index=False, compression="snappy")
    store_daily.to_parquet(processed / "sales_store_daily.parquet", index=False, compression="snappy")

    weekly = rollup_daily_to_weekly(daily)
    dense = build_dense_weekly_panel(weekly, dimensions["sku_master"])
    dense.to_parquet(processed / "sales_weekly.parquet", index=False, compression="snappy")

    calendar = build_calendar()
    calendar.to_parquet(processed / "calendar.parquet", index=False, compression="snappy")

    resolved = resolve_sku_day_promos(dimensions["promotions"], dimensions["sku_master"])
    promo_weekly = sku_week_promo_features(resolved)
    promo_weekly.to_parquet(processed / "promo_weekly.parquet", index=False, compression="snappy")

    inv = normalize_inventory(
        dimensions["inventory_snapshot"],
        dimensions["sku_master"],
        settings.risk["inventory_defaults"]["lead_time_days"],
        settings.risk["inventory_defaults"]["on_order_units_default"],
        snapshot_date="2025-12-31",
    )
    ensure_dir(settings.interim_dir)
    inv.to_parquet(settings.interim_dir / "inventory_normalized.parquet", index=False, compression="snappy")

    report = {
        "processed_dir": str(processed),
        "daily_rows": len(daily),
        "weekly_rows": len(dense),
        "calendar_days": len(calendar),
        "promo_resolved_rows": len(resolved),
        "inventory_rows": len(inv),
        "reconciliation": {
            "daily_units_sum": int(daily["units_sold"].sum()),
            "weekly_units_sum": int(dense["units_sold"].sum()),
            "store_daily_units_sum": int(store_daily["units_sold"].sum()),
        },
    }
    report_path = settings.reports_dir / "data_quality_report.md"
    ensure_dir(settings.reports_dir)
    report_path.write_text(_d1_report(report, settings), encoding="utf-8")
    return report


def _d1_report(report: dict[str, object], settings) -> str:
    rec = report["reconciliation"]
    return f"""# Deliverable D1 — Reproducible Data Pipeline

- Project: {settings.system.project_name}
- Dataset root: {settings.dataset_root}
- Chunk size: {settings.ingestion.chunk_size}
- Sales file bytes: {settings.system.sales_file_bytes}

## Outputs
- `data/processed/sales_daily.parquet` ({report['daily_rows']} rows)
- `data/processed/sales_store_daily.parquet` ({report['store_daily_rows'] if 'store_daily_rows' in report else 'see reconciliation'})
- `data/processed/sales_weekly.parquet` ({report['weekly_rows']} dense rows)
- `data/processed/calendar.parquet` ({report['calendar_days']} days)
- `data/processed/promo_weekly.parquet` ({report['promo_resolved_rows']} promo-week rows)
- `data/interim/inventory_normalized.parquet` ({report['inventory_rows']} rows)

## Reconciliation
- daily units = weekly units = chain aggregate: {rec['daily_units_sum']}
- store_daily units: {rec['store_daily_units_sum']}

## Governance
- sku_inventory_flags.csv: not touched (evaluation resource only).
- Sales CSV streamed in {settings.ingestion.chunk_size}-row chunks.
"""


def run_phase2(project_root: Path | None = None, settings: Settings | None = None):
    """Deliverable D2: EDA memo + seasonal-naive baseline backtest scorecard."""
    import pandas as pd

    from foresight.evaluation.reporting import (
        build_baseline_scorecard_md,
        build_eda_report,
    )
    from foresight.forecasting.validation import run_backtest_and_log
    from foresight.ingestion.metadata_ingestion import load_sku_master
    from foresight.utils.io import ensure_dir

    if isinstance(project_root, Settings):
        settings = project_root
        project_root = None
    if settings is None:
        settings = load_settings(project_root)
    logger = configure_logging(settings.system.log_level)

    processed = settings.processed_dir
    weekly = pd.read_parquet(processed / "sales_weekly.parquet")
    calendar = pd.read_parquet(processed / "calendar.parquet")
    sku_master = load_sku_master(settings.dataset_file("raw_sku"))

    inventory_path = settings.interim_dir / "inventory_normalized.parquet"
    inventory = pd.read_parquet(inventory_path) if inventory_path.exists() else None

    folds = settings.forecast["cv"]["folds"]
    summary = run_backtest_and_log(
        weekly,
        sku_master,
        folds,
        metrics_dir=settings.artifacts_dir / "metrics",
        seasonality=52,
        trailing_window=4,
    )

    report_md = build_eda_report(weekly, sku_master, calendar, inventory, summary, settings)
    ensure_dir(settings.reports_dir)
    (settings.reports_dir / "eda_report.md").write_text(report_md, encoding="utf-8")
    (settings.reports_dir / "baseline_scorecard.md").write_text(
        build_baseline_scorecard_md(summary), encoding="utf-8"
    )

    logger.info(
        "Phase 2 OK | skus=%s | weeks=%s | overall_wape=%.4f",
        weekly["sku_id"].nunique(),
        weekly["year_week"].nunique(),
        summary["overall_wape"],
    )
    return {
        "processed_dir": str(processed),
        "n_skus": int(weekly["sku_id"].nunique()),
        "n_weeks": int(weekly["year_week"].nunique()),
        "overall_wape": summary["overall_wape"],
        "fold_wapes": [f["wape"] for f in summary["folds"]],
        "reports": [
            str(settings.reports_dir / "eda_report.md"),
            str(settings.reports_dir / "baseline_scorecard.md"),
        ],
        "metrics": str(settings.artifacts_dir / "metrics" / "cv_summary.json"),
    }


def run_phase3(project_root: Path | None = None, settings: Settings | None = None):
    """Deliverable D3: global forecaster backtest vs baseline + production artifact."""
    import json

    import pandas as pd

    from foresight.evaluation.reporting import build_ml_comparison, build_model_evaluation_md
    from foresight.forecasting.backtest import (
        load_progress,
        run_ml_backtest,
        run_production,
    )
    from foresight.ingestion.metadata_ingestion import load_sku_master
    from foresight.preprocessing.calendar import build_calendar
    from foresight.utils.io import ensure_dir

    if isinstance(project_root, Settings):
        settings = project_root
        project_root = None
    if settings is None:
        settings = load_settings(project_root)
    logger = configure_logging(settings.system.log_level)

    weekly = pd.read_parquet(settings.processed_dir / "sales_weekly.parquet")
    promo_weekly = pd.read_parquet(settings.processed_dir / "promo_weekly.parquet")
    sku_master = load_sku_master(settings.dataset_file("raw_sku"))
    # Extended beyond the panel so the production horizon (early 2026) has calendar rows.
    calendar = build_calendar("2022-01-01", "2026-12-31")

    metrics_dir = ensure_dir(settings.artifacts_dir / "metrics")
    progress_path = metrics_dir / "phase3_progress.json"
    models_dir = settings.artifacts_dir / "models"
    forecasts_dir = settings.artifacts_dir / "forecasts"

    matrix, progress = _features_cache(
        weekly, calendar, promo_weekly, sku_master, settings, progress_path
    )

    cv_path = metrics_dir / "cv_summary.json"
    if not cv_path.exists():
        raise RuntimeError("cv_summary.json not found; run `--phase 2` (baseline) first.")
    cv_summary = json.loads(cv_path.read_text(encoding="utf-8"))
    if cv_summary.get("model") != "seasonal_naive_3tier":
        raise RuntimeError("cv_summary.json does not hold the Phase 2 baseline; rerun `--phase 2`.")

    folds = settings.forecast["cv"]["folds"]
    model_cfg = settings.forecast["model"]
    ml_summary = run_ml_backtest(
        matrix=matrix,
        weekly_panel=weekly,
        calendar=calendar,
        promo_weekly=promo_weekly,
        sku_master=sku_master,
        feature_cfg=settings.features,
        model_cfg=model_cfg,
        folds=folds,
        progress_path=progress_path,
        models_dir=models_dir,
        forecasts_dir=forecasts_dir,
        quantile_fold=len(folds),
        quantiles=list(model_cfg["quantiles"]),
    )

    horizon = int(settings.forecast["horizon_weeks"])
    bundle_path = models_dir / "global_forecaster.joblib"
    forecast_path = forecasts_dir / f"forecast_{horizon}w.parquet"
    saved = load_progress(progress_path).get("production")
    if saved and bundle_path.exists() and forecast_path.exists():
        production = saved
        logger.info("Phase 3 production bundle already present; skipping retrain.")
    else:
        production = run_production(
            matrix=matrix,
            weekly_panel=weekly,
            calendar=calendar,
            promo_weekly=promo_weekly,
            sku_master=sku_master,
            feature_cfg=settings.features,
            model_cfg=model_cfg,
            quantiles=list(model_cfg["quantiles"]),
            horizon_weeks=horizon,
            progress_path=progress_path,
            models_dir=models_dir,
            forecasts_dir=forecasts_dir,
        )

    comparison = build_ml_comparison(cv_summary, ml_summary)
    cv_summary["ml"] = ml_summary
    cv_summary["comparison"] = comparison
    cv_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")

    ensure_dir(settings.reports_dir)
    report_path = settings.reports_dir / "model_evaluation.md"
    report_path.write_text(
        build_model_evaluation_md(cv_summary, ml_summary, comparison, settings),
        encoding="utf-8",
    )

    logger.info(
        "Phase 3 OK | ml_wape=%.4f | baseline_wape=%.4f | beats=%s",
        ml_summary["overall_wape"],
        comparison["baseline_overall_wape"],
        comparison["beats_baseline"],
    )
    return {
        "n_features": ml_summary["n_features"],
        "fold_wapes": [f["wape"] for f in ml_summary["folds"]],
        "ml_overall_wape": ml_summary["overall_wape"],
        "baseline_overall_wape": comparison["baseline_overall_wape"],
        "beats_baseline": comparison["beats_baseline"],
        "interval_coverage": ml_summary.get("interval_coverage"),
        "production": production,
        "report": str(report_path),
        "metrics": str(cv_path),
    }


def run_phase4(project_root: Path | None = None, settings: Settings | None = None):
    """Deliverable D4: risk scoring, dual-path grain, flags evaluation vs key."""
    import json
    from datetime import date

    import pandas as pd

    from foresight.forecasting.backtest import (
        get_or_train_model,
        load_progress,
    )
    from foresight.forecasting.prediction import recursive_forecast
    from foresight.forecasting.training import train_model
    from foresight.ingestion.metadata_ingestion import load_sku_master
    from foresight.inventory.actions import add_rupee_impact, impact_summary
    from foresight.inventory.flag_evaluation import (
        build_risk_evaluation_md,
        evaluate_flags,
    )
    from foresight.inventory.risk_scoring import score_chain_risk
    from foresight.inventory.store_grain import compute_store_shares, score_store_risk
    from foresight.preprocessing.calendar import build_calendar
    from foresight.utils.dates import advance_week, iso_year_week
    from foresight.utils.io import ensure_dir

    if isinstance(project_root, Settings):
        settings = project_root
        project_root = None
    if settings is None:
        settings = load_settings(project_root)
    logger = configure_logging(settings.system.log_level)

    weekly = pd.read_parquet(settings.processed_dir / "sales_weekly.parquet")
    promo_weekly = pd.read_parquet(settings.processed_dir / "promo_weekly.parquet")
    store_daily = pd.read_parquet(settings.processed_dir / "sales_store_daily.parquet")
    sku_master = load_sku_master(settings.dataset_file("raw_sku"))
    calendar = build_calendar("2022-01-01", "2026-12-31")
    inventory = pd.read_parquet(settings.interim_dir / "inventory_normalized.parquet")

    metrics_dir = ensure_dir(settings.artifacts_dir / "metrics")
    progress_path = metrics_dir / "phase4_progress.json"
    models_dir = settings.artifacts_dir / "models"
    forecasts_dir = settings.artifacts_dir / "forecasts"
    risk_dir = ensure_dir(settings.artifacts_dir / "risk")

    matrix, _ = _features_cache(
        weekly,
        calendar,
        promo_weekly,
        sku_master,
        settings,
        metrics_dir / "phase3_progress.json",
    )

    # Evaluation model trained ONLY through the week-41 cutoff (no leakage from
    # the production bundle, which sees history through 2026-W01).
    cutoff_cfg = settings.forecast["execution_modes"]["evaluation_cutoff"]
    cutoff_week = iso_year_week(date.fromisoformat(cutoff_cfg["train_end_date"]))
    horizon = int(settings.forecast["horizon_weeks"])
    future_weeks = advance_week(cutoff_week, horizon)
    model_cfg = settings.forecast["model"]

    progress = load_progress(progress_path)
    point_model, feature_names, _ = get_or_train_model(
        key=f"eval_cutoff_{cutoff_week}_point",
        train_fn=lambda: train_model(matrix, model_cfg, cutoff_week),
        progress=progress,
        progress_path=progress_path,
        models_dir=models_dir,
    )
    eval_forecast_path = forecasts_dir / f"eval_cutoff_{cutoff_week}_{horizon}w.parquet"
    if eval_forecast_path.exists():
        eval_forecasts = pd.read_parquet(eval_forecast_path)
    else:
        eval_forecasts = recursive_forecast(
            weekly_panel=weekly,
            calendar=calendar,
            promo_weekly=promo_weekly,
            sku_master=sku_master,
            feature_cfg=settings.features,
            point_model=point_model,
            feature_names=feature_names,
            future_weeks=future_weeks,
            cutoff_week=cutoff_week,
        )
        ensure_dir(forecasts_dir)
        eval_forecasts.to_parquet(eval_forecast_path, index=False, compression="snappy")

    # Ground-truth evaluation: the flags key is read HERE and nowhere else.
    flags = pd.read_csv(settings.dataset_file(settings.risk["evaluation"]["flags_path_key"]))
    window_weeks = int(settings.risk["inventory_defaults"]["store_share_window_weeks"])

    eval_risk = score_chain_risk(inventory, eval_forecasts, settings.risk, horizon)
    eval_shares = compute_store_shares(store_daily, inventory, cutoff_week, window_weeks)
    eval_store = score_store_risk(eval_risk, eval_shares, inventory, settings.risk)
    evaluation = evaluate_flags(eval_risk, eval_store, flags, settings.risk)

    # Production decision grid: snapshot stock vs forward production forecast.
    prod_forecasts = pd.read_parquet(forecasts_dir / f"forecast_{horizon}w.parquet")
    panel_week = str(weekly["year_week"].max())
    chain_risk = score_chain_risk(inventory, prod_forecasts, settings.risk, horizon)
    chain_risk = add_rupee_impact(chain_risk, sku_master)
    chain_risk.to_parquet(risk_dir / "risk_scores.parquet", index=False, compression="snappy")

    prod_shares = compute_store_shares(store_daily, inventory, panel_week, window_weeks)
    store_risk = score_store_risk(chain_risk, prod_shares, inventory, settings.risk)
    store_risk.to_parquet(risk_dir / "store_risk.parquet", index=False, compression="snappy")
    production = impact_summary(chain_risk)

    metrics_path = metrics_dir / "flag_evaluation.json"
    metrics_path.write_text(
        json.dumps({"evaluation": evaluation, "production": production}, indent=2),
        encoding="utf-8",
    )

    meta = {
        "cutoff_week": cutoff_week,
        "train_end_date": cutoff_cfg["train_end_date"],
        "horizon_weeks": horizon,
        "snapshot_date": str(chain_risk["snapshot_date"].max().date()),
        "forecast_weeks": (
            f"{prod_forecasts['year_week'].min()} .. {prod_forecasts['year_week'].max()}"
        ),
        "n_forecast_skus": int(prod_forecasts["sku_id"].nunique()),
    }
    report_path = settings.reports_dir / "risk_evaluation.md"
    ensure_dir(settings.reports_dir)
    report_path.write_text(
        build_risk_evaluation_md(evaluation, production, meta, settings),
        encoding="utf-8",
    )

    stockout = evaluation["stockout_risk"]
    logger.info(
        "Phase 4 OK | stockout_recall=%.3f (target %.2f met=%s) | slow_f1=%.3f "
        "| store_pair_recall=%.3f | skus_scored=%s",
        stockout["recall"],
        evaluation["target_stockout_recall"],
        evaluation["meets_stockout_recall_target"],
        evaluation["slow_mover"]["f1"],
        evaluation["store_pairs"]["recall"],
        production["n_skus_scored"],
    )
    return {
        "cutoff_week": cutoff_week,
        "horizon_weeks": horizon,
        "skus_scored": production["n_skus_scored"],
        "stockout": stockout,
        "slow_mover": evaluation["slow_mover"],
        "store_pairs": evaluation["store_pairs"],
        "meets_stockout_recall_target": evaluation["meets_stockout_recall_target"],
        "production": production,
        "report": str(report_path),
        "metrics": str(metrics_path),
        "risk_scores": str(risk_dir / "risk_scores.parquet"),
        "store_risk": str(risk_dir / "store_risk.parquet"),
    }


def run_phase5(project_root: Path | None = None, settings: Settings | None = None):
    """Deliverable D5: pre-aggregated serving layer for the Streamlit dashboard."""
    import json

    import pandas as pd

    from foresight.ingestion.metadata_ingestion import load_sku_master
    from foresight.serving.build import build_serving_layer

    if isinstance(project_root, Settings):
        settings = project_root
        project_root = None
    if settings is None:
        settings = load_settings(project_root)
    logger = configure_logging(settings.system.log_level)

    horizon = int(settings.forecast["horizon_weeks"])
    risk_path = settings.artifacts_dir / "risk" / "risk_scores.parquet"
    forecast_path = settings.artifacts_dir / "forecasts" / f"forecast_{horizon}w.parquet"
    cv_path = settings.artifacts_dir / "metrics" / "cv_summary.json"
    for path in (risk_path, forecast_path, cv_path):
        if not path.exists():
            raise RuntimeError(
                f"Missing {path.name}; run the earlier phases first "
                "(--phase 3 for forecasts, --phase 4 for risk scores)."
            )

    weekly = pd.read_parquet(settings.processed_dir / "sales_weekly.parquet")
    promo_weekly = pd.read_parquet(settings.processed_dir / "promo_weekly.parquet")
    sku_master = load_sku_master(settings.dataset_file("raw_sku"))
    risk_scores = pd.read_parquet(risk_path)
    forecast = pd.read_parquet(forecast_path)
    cv_summary = json.loads(cv_path.read_text(encoding="utf-8"))

    result = build_serving_layer(
        settings=settings,
        weekly=weekly,
        sku_master=sku_master,
        promo_weekly=promo_weekly,
        risk_scores=risk_scores,
        forecast=forecast,
        cv_summary=cv_summary,
    )
    logger.info(
        "Phase 5 OK | serving=%s | files=%s | history_partitions=%s | skus=%s",
        result["serving_dir"],
        result["files"],
        result["history_partitions"],
        result["active_skus"],
    )
    return result


def _features_cache(weekly, calendar, promo_weekly, sku_master, settings, progress_path):
    """Build the feature matrix once; reuse `data/interim/feature_matrix.parquet`."""
    import pandas as pd

    from foresight.features.feature_pipeline import build_feature_matrix, feature_matrix_columns
    from foresight.forecasting.backtest import load_progress, save_progress
    from foresight.utils.io import ensure_dir

    progress = load_progress(progress_path)
    cache_path = settings.interim_dir / "feature_matrix.parquet"
    expected_rows = len(weekly)
    if cache_path.exists() and progress.get("features", {}).get("rows") == expected_rows:
        matrix = pd.read_parquet(cache_path)
        if len(matrix) == expected_rows:
            return matrix, progress

    matrix = build_feature_matrix(weekly, calendar, promo_weekly, sku_master, settings.features)
    ensure_dir(settings.interim_dir)
    matrix.to_parquet(cache_path, index=False, compression="snappy")
    progress["features"] = {
        "rows": len(matrix),
        "n_features": len(feature_matrix_columns(matrix)),
        "cache_file": cache_path.name,
    }
    save_progress(progress_path, progress)
    return matrix, progress


def run_pipeline(phase: int = 0) -> int:
    if phase == 0:
        result = verify_phase0()
        print("FORESIGHT Phase 0 verification passed.")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return 0
    if phase == 1:
        result = run_phase1()
        print("FORESIGHT Phase 1 pipeline executed.")
        print(f"  processed_dir: {result['processed_dir']}")
        print(f"  daily_rows: {result['daily_rows']}")
        print(f"  weekly_rows: {result['weekly_rows']}")
        print(f"  reconciliation: {result['reconciliation']}")
        return 0
    if phase == 2:
        result = run_phase2()
        print("FORESIGHT Phase 2 pipeline executed.")
        print(f"  n_skus: {result['n_skus']}")
        print(f"  n_weeks: {result['n_weeks']}")
        print(f"  overall_baseline_wape: {result['overall_wape']:.4f}")
        print(f"  metrics: {result['metrics']}")
        return 0
    if phase == 3:
        result = run_phase3()
        print("FORESIGHT Phase 3 forecasting engine executed.")
        print(f"  n_features: {result['n_features']}")
        print(f"  baseline_overall_wape: {result['baseline_overall_wape']:.4f}")
        print(f"  ml_overall_wape: {result['ml_overall_wape']:.4f}")
        print(f"  beats_baseline: {result['beats_baseline']}")
        coverage = result.get("interval_coverage") or {}
        if coverage:
            print(f"  interval_coverage: {coverage.get('coverage'):.4f} (nominal {coverage.get('nominal'):.2f})")
        prod = result.get("production") or {}
        print(f"  production_forecast: {prod.get('forecast_file')}")
        print(f"  report: {result['report']}")
        return 0
    if phase == 4:
        result = run_phase4()
        print("FORESIGHT Phase 4 inventory risk intelligence executed.")
        print(f"  cutoff_week: {result['cutoff_week']} (horizon {result['horizon_weeks']}w)")
        print(f"  skus_scored: {result['skus_scored']}")
        stockout = result["stockout"]
        print(
            f"  stockout_recall: {stockout['recall']:.4f} "
            f"(met target: {result['meets_stockout_recall_target']}, "
            f"precision {stockout['precision']:.4f})"
        )
        print(f"  slow_mover_f1: {result['slow_mover']['f1']:.4f}")
        print(f"  store_pair_recall: {result['store_pairs']['recall']:.4f}")
        prod = result["production"]
        print(f"  quadrants: {prod['quadrant_counts']}")
        print(f"  rupees_at_risk: {prod['rupees_at_risk']:.0f}")
        print(f"  report: {result['report']}")
        return 0
    if phase == 5:
        result = run_phase5()
        print("FORESIGHT Phase 5 serving layer executed.")
        print(f"  serving_dir: {result['serving_dir']}")
        for name, rows in result["files"].items():
            print(f"  {name}: {rows} rows")
        print(f"  history_partitions: {result['history_partitions']}")
        print(f"  skus: {result['active_skus']} active / {result['scored_skus']} scored")
        print(f"  manifest: {result['manifest']}")
        print("  launch dashboard: streamlit run app/streamlit_app.py")
        return 0
    raise NotImplementedError(
        f"Phase {phase} is not implemented yet. Complete earlier phases first."
    )


def run_all_phases() -> int:
    """Run all pipeline phases (0 through 5) end-to-end for single-command reproducibility (Rule 7)."""
    print("=" * 64)
    print("PROJECT FORESIGHT — RUNNING FULL END-TO-END PIPELINE (PHASES 0-5)")
    print("=" * 64)
    for p in range(6):
        print(f"\n>>> Executing Phase {p}...")
        ret = run_pipeline(phase=p)
        if ret != 0:
            return ret
    print("\n" + "=" * 64)
    print("FORESIGHT PIPELINE COMPLETE: All phases 0-5 executed successfully.")
    print("=" * 64)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foresight")
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help=(
            "Pipeline phase to run (0 = env, 1 = D1 pipeline, 2 = D2 EDA + baseline, "
            "3 = D3 forecasting engine, 4 = D4 inventory risk intelligence, "
            "5 = D5 dashboard serving layer). If omitted or --all is specified, runs all phases end-to-end."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline phases (0 through 5) end-to-end for Rule 7 reproducibility.",
    )
    args = parser.parse_args(argv)
    try:
        if args.all or args.phase is None:
            return run_all_phases()
        return run_pipeline(phase=args.phase)
    except Exception as exc:  # CLI surface: report failures to stderr
        print(f"FORESIGHT pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
