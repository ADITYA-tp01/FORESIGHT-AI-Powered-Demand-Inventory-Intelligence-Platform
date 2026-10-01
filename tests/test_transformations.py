"""Daily → weekly dense panel and metadata loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.helpers import (
    default_mini_sales_rows,
    mini_promotions,
    mini_sku_master,
    write_sales_csv,
)


def _daily_from_mini(tmp_path: Path) -> pd.DataFrame:
    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    from foresight.ingestion.sales_ingestion import stream_aggregate_daily

    return stream_aggregate_daily(path, chunk_size=2)


def test_weekly_roll_up_reconciles_to_daily_units(tmp_path: Path):
    daily = _daily_from_mini(tmp_path)
    from foresight.preprocessing.sales_weekly import rollup_daily_to_weekly

    weekly = rollup_daily_to_weekly(daily)
    assert weekly["units_sold"].sum() == daily["units_sold"].sum()
    w01 = weekly.set_index(["year_week", "sku_id"]).loc[("2022-W01", "SKU00001")]
    assert int(w01["units_sold"]) == 4


def test_transaction_quantity_reconciles_through_daily_to_weekly(tmp_path: Path):
    """Plan 9.1 item 2: sum(weekly_units) == sum(daily_units) == sum(quantity)."""
    from foresight.ingestion.sales_ingestion import stream_aggregate_daily
    from foresight.preprocessing.sales_weekly import rollup_daily_to_weekly

    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    transactions = pd.read_csv(path)
    daily = stream_aggregate_daily(path, chunk_size=2)
    weekly = rollup_daily_to_weekly(daily)

    transaction_units = int(transactions["quantity"].sum())
    assert transaction_units > 0
    assert int(daily["units_sold"].sum()) == transaction_units
    assert int(weekly["units_sold"].sum()) == transaction_units


def test_dense_panel_zero_fills_missing_sku_weeks(tmp_path: Path):
    daily = _daily_from_mini(tmp_path)
    sku = mini_sku_master()
    from foresight.preprocessing.sales_weekly import (
        build_dense_weekly_panel,
        rollup_daily_to_weekly,
    )

    sparse = rollup_daily_to_weekly(daily)
    # Pass explicit date range matching the test data (2022-W01 to 2022-W02)
    dense = build_dense_weekly_panel(sparse, sku, start_date="2022-01-03", end_date="2022-01-10")
    # 2 ISO weeks in the mini history × 2 SKUs
    assert len(dense) == 4
    missing = dense.set_index(["year_week", "sku_id"]).loc[("2022-W01", "SKU00002")]
    assert int(missing["units_sold"]) == 0
    assert float(missing["revenue"]) == 0.0
    assert float(missing["unit_price"]) == pytest.approx(5.0)


def test_metadata_loader_reads_sku_and_store(tmp_path: Path):
    sku_path = tmp_path / "sku_master.csv"
    mini_sku_master().to_csv(sku_path, index=False)
    from foresight.ingestion.metadata_ingestion import load_sku_master

    sku = load_sku_master(sku_path)
    assert len(sku) == 2
    assert set(sku["category"]) == {"Dairy & Bakery", "Stationery & Office"}


def test_promo_loader_parses_dates(tmp_path: Path):
    path = tmp_path / "promotions.csv"
    mini_promotions().to_csv(path, index=False)
    from foresight.ingestion.metadata_ingestion import load_promotions

    promos = load_promotions(path)
    assert len(promos) == 3
    assert pd.api.types.is_datetime64_any_dtype(promos["start_date"])
