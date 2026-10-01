"""Phase 1 leakage guards: flags file never enters processed training tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.helpers import (
    default_mini_sales_rows,
    mini_inventory,
    mini_promotions,
    mini_sku_master,
    mini_store_master,
    write_sales_csv,
)


def test_phase1_processed_tables_do_not_contain_flag_columns(tmp_path: Path):
    from foresight.pipeline import run_phase1

    data_root = tmp_path / "mini"
    (data_root / "retail_contaminated_dataset").mkdir(parents=True)
    (data_root / "retail_clean_dataset").mkdir(parents=True)
    write_sales_csv(
        data_root / "retail_contaminated_dataset" / "sales_transactions.csv",
        default_mini_sales_rows(),
    )
    write_sales_csv(
        data_root / "retail_clean_dataset" / "sales_transactions.csv",
        default_mini_sales_rows(),
    )
    mini_sku_master().to_csv(data_root / "retail_contaminated_dataset" / "sku_master.csv", index=False)
    mini_store_master().to_csv(
        data_root / "retail_contaminated_dataset" / "store_master.csv", index=False
    )
    pd.DataFrame(
        {
            "cust_id": ["CUST00001"],
            "age": [30],
            "gender": ["Female"],
            "city": ["Karachi"],
            "loyalty_segment": ["Gold"],
            "preferred_channel": ["In-Store"],
            "registration_date": ["2021-01-01"],
        }
    ).to_csv(data_root / "retail_contaminated_dataset" / "customer_master.csv", index=False)
    mini_promotions().to_csv(
        data_root / "retail_contaminated_dataset" / "promotions.csv", index=False
    )
    mini_inventory().to_csv(
        data_root / "retail_contaminated_dataset" / "inventory_snapshot.csv", index=False
    )
    pd.DataFrame(
        {
            "sku_id": ["SKU00001"],
            "flag": ["STOCKOUT_RISK"],
            "affected_stores": ["ST01"],
            "window_start": ["2025-11-01"],
            "window_end": ["2025-11-15"],
            "notes": ["synthetic fixture — evaluation only"],
        }
    ).to_csv(data_root / "retail_contaminated_dataset" / "sku_inventory_flags.csv", index=False)
    mini_inventory().to_csv(
        data_root / "retail_clean_dataset" / "inventory_snapshot.csv", index=False
    )

    project_root = tmp_path / "project"
    (project_root / "configs").mkdir(parents=True)
    # Create a custom config.yaml pointing to the mini dataset
    # Use forward slashes for YAML compatibility
    data_root_str = str(data_root).replace("\\", "/")
    config_content = f"""system:
  project_name: "Project FORESIGHT"
  version: "1.0.0"
  seed: 42
  environment: "development"
  log_level: "INFO"
  sales_file_bytes: 801086326
  sales_file_mib: 763.98

paths:
  dataset_root: "{data_root_str}"
  raw_sales: "retail_contaminated_dataset/sales_transactions.csv"
  raw_sku: "retail_contaminated_dataset/sku_master.csv"
  raw_store: "retail_contaminated_dataset/store_master.csv"
  raw_customer: "retail_contaminated_dataset/customer_master.csv"
  raw_inventory: "retail_contaminated_dataset/inventory_snapshot.csv"
  raw_promotions: "retail_contaminated_dataset/promotions.csv"
  raw_flags: "retail_contaminated_dataset/sku_inventory_flags.csv"
  clean_sales_control: "retail_clean_dataset/sales_transactions.csv"
  clean_inventory_control: "retail_clean_dataset/inventory_snapshot.csv"
  processed_dir: "data/processed"
  interim_dir: "data/interim"
  artifacts_dir: "artifacts"
  reports_dir: "reports"
  configs_dir: "configs"

ingestion:
  chunk_size: 500000
  parquet_compression: "snappy"
  max_peak_ram_mb: 500
  deduplication_policy: "retain_multiscan"

audit_expectations:
  n_skus: 2
  n_stores: 2
  n_customers: 1
  n_promotions: 3
  n_flags: 1
  n_sales_contaminated: 4
  n_sales_clean: 4
  n_inventory_contaminated: 2
"""
    (project_root / "configs" / "config.yaml").write_text(config_content)
    
    # Copy other config files
    import shutil
    src_configs = Path(__file__).resolve().parents[1] / "configs"
    for fname in ("features.yaml", "forecast.yaml", "risk.yaml"):
        shutil.copy2(src_configs / fname, project_root / "configs" / fname)
    
    (project_root / "data" / "processed").mkdir(parents=True)
    (project_root / "data" / "interim").mkdir(parents=True)
    (project_root / "reports").mkdir(parents=True)

    result = run_phase1(project_root)
    processed = Path(result["processed_dir"])
    weekly = pd.read_parquet(processed / "sales_weekly.parquet")
    forbidden = {"flag", "affected_stores", "window_start", "window_end", "notes"}
    assert forbidden.isdisjoint(set(weekly.columns))
    assert "STOCKOUT_RISK" not in weekly.astype(str).to_numpy().ravel()


def test_feature_matrix_ignores_future_and_concurrent_demand():
    """Perturb week t and later: features at/before week t must be bit-identical."""
    from tests.helpers import empty_promo_weekly, mini_sku_master, mini_weekly_panel

    from foresight.config import load_settings
    from foresight.features.feature_pipeline import (
        FORBIDDEN_FEATURES,
        build_feature_matrix,
        feature_matrix_columns,
    )
    from foresight.preprocessing.calendar import build_calendar

    settings = load_settings()
    panel = mini_weekly_panel()
    calendar = build_calendar("2023-01-01", "2024-12-31")
    promo = empty_promo_weekly()
    master = mini_sku_master()

    original = build_feature_matrix(panel, calendar, promo, master, settings.features)
    feature_cols = feature_matrix_columns(original)

    weeks = sorted(panel["year_week"].unique())
    cut_week = weeks[60]
    perturbed = panel.copy()
    mask = perturbed["year_week"] >= cut_week
    perturbed.loc[mask, "units_sold"] = perturbed.loc[mask, "units_sold"] * 7 + 3
    perturbed.loc[mask, "revenue"] = perturbed.loc[mask, "revenue"] * 11.0 + 1.0
    perturbed.loc[mask, "gross_revenue"] = perturbed.loc[mask, "gross_revenue"] * 13.0 + 2.0
    assert not panel["units_sold"].equals(perturbed["units_sold"])

    rebuilt = build_feature_matrix(perturbed, calendar, promo, master, settings.features)

    # 1) Every feature for weeks strictly before the perturbation is unchanged.
    before = original["year_week"] < cut_week
    pd.testing.assert_frame_equal(
        original.loc[before, ["sku_id", "year_week", *feature_cols]],
        rebuilt.loc[before, ["sku_id", "year_week", *feature_cols]],
        check_exact=True,
    )
    # 2) Features AT week t are unchanged too (concurrent actuals are unused).
    at_cut = original["year_week"] == cut_week
    pd.testing.assert_frame_equal(
        original.loc[at_cut, feature_cols].reset_index(drop=True),
        rebuilt.loc[at_cut, feature_cols].reset_index(drop=True),
        check_exact=True,
    )
    # 3) The target itself did change — proving the perturbation was live.
    assert not original.loc[at_cut, "units_sold"].reset_index(drop=True).equals(
        rebuilt.loc[at_cut, "units_sold"].reset_index(drop=True)
    )
    # 4) Leakage-prone columns never appear as features.
    assert FORBIDDEN_FEATURES.isdisjoint(set(original.columns))
