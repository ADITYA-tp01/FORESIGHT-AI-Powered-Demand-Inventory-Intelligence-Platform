"""Phase 0 environment, config, and leakage-governance tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from foresight import __version__
from foresight.config import CONFIG_FILENAMES, required_config_files
from foresight.pipeline import verify_phase0
from foresight.utils.dates import iso_year_week


def test_package_version():
    assert __version__ == "1.0.0"


def test_all_yaml_configs_exist(project_root: Path):
    files = required_config_files(project_root)
    assert [p.name for p in files] == list(CONFIG_FILENAMES)
    for path in files:
        assert path.exists(), path


def test_settings_load_and_seed(settings):
    assert settings.system.seed == 42
    assert settings.ingestion.chunk_size == 500_000
    assert settings.ingestion.max_peak_ram_mb == 500
    assert settings.system.sales_file_bytes == 801_086_326


def test_dataset_files_exist_without_loading_sales(settings):
    sales = settings.dataset_file("raw_sales")
    assert sales.exists()
    size = sales.stat().st_size
    assert size == settings.system.sales_file_bytes
    # Never pd.read_csv the sales file in Phase 0.
    assert sales.name == "sales_transactions.csv"


def test_dimension_files_exist(settings):
    for key in ("raw_sku", "raw_store", "raw_customer", "raw_inventory", "raw_promotions", "raw_flags"):
        assert settings.dataset_file(key).exists(), key
    assert settings.dataset_file("clean_sales_control").exists()
    assert settings.dataset_file("clean_inventory_control").exists()


def test_risk_yaml_covers_all_audited_categories(settings):
    expected = {
        "Apparel & Footwear",
        "Beverages",
        "Dairy & Bakery",
        "Electronics & Accessories",
        "Frozen Foods",
        "Grocery",
        "Health & Wellness",
        "Home & Kitchen",
        "Home Care",
        "Personal Care",
        "Snacks & Confectionery",
        "Stationery & Office",
    }
    configured = set(settings.risk["inventory_defaults"]["lead_time_days"].keys())
    assert configured == expected


def test_flags_file_is_evaluation_only(settings):
    forbidden = settings.features["leakage"]["forbidden_sources"]
    assert "sku_inventory_flags.csv" in forbidden
    flags = settings.dataset_file("raw_flags")
    assert flags.name == "sku_inventory_flags.csv"


def test_forecast_forbids_shuffle(settings):
    assert settings.forecast["cv"]["shuffle"] is False
    assert settings.forecast["horizon_weeks"] == 8


def test_phase0_entrypoint(project_root: Path):
    result = verify_phase0(project_root)
    assert result["version"] == "1.0.0"
    assert result["chunk_size"] == 500_000
    assert result["n_sku_categories"] == 12


def test_iso_year_week_helper():
    assert iso_year_week(date(2025, 10, 12)) == "2025-W41"


def test_later_phases_not_silently_implemented():
    import pytest

    from foresight.pipeline import run_pipeline

    with pytest.raises(NotImplementedError):
        run_pipeline(phase=6)
