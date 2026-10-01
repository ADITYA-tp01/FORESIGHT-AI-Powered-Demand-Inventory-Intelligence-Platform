"""Chunked sales ingestion contracts. Fixtures only — never the 801 MB file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.helpers import default_mini_sales_rows, write_sales_csv


def test_iter_sales_chunks_respects_chunk_size(tmp_path: Path):
    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    from foresight.ingestion.sales_ingestion import iter_sales_chunks

    chunks = list(iter_sales_chunks(path, chunk_size=2))
    assert len(chunks) == 2
    assert sum(len(c) for c in chunks) == 4
    assert list(chunks[0].columns)[:4] == ["date", "receipt_id", "store_id", "sku_id"]


def test_validate_sales_chunk_rejects_quantity_outside_1_to_5(tmp_path: Path):
    path = write_sales_csv(
        tmp_path / "bad.csv",
        "2022-01-03,R1,ST01,SKU00001,CUST00001,6,10.0,60.0,In-Store,0.0,\n",
    )
    from foresight.ingestion.sales_ingestion import iter_sales_chunks, validate_sales_chunk

    chunk = next(iter_sales_chunks(path, chunk_size=10))
    with pytest.raises(ValueError, match="quantity"):
        validate_sales_chunk(chunk)


def test_validate_sales_chunk_rejects_non_positive_price(tmp_path: Path):
    path = write_sales_csv(
        tmp_path / "bad.csv",
        "2022-01-03,R1,ST01,SKU00001,CUST00001,1,0.0,0.0,In-Store,0.0,\n",
    )
    from foresight.ingestion.sales_ingestion import iter_sales_chunks, validate_sales_chunk

    chunk = next(iter_sales_chunks(path, chunk_size=10))
    with pytest.raises(ValueError, match="unit_price"):
        validate_sales_chunk(chunk)


def test_daily_aggregation_retains_multiscan_and_sums_quantity(tmp_path: Path):
    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    from foresight.ingestion.sales_ingestion import (
        aggregate_chunk_daily,
        iter_sales_chunks,
        validate_sales_chunk,
    )

    chunk = validate_sales_chunk(next(iter_sales_chunks(path, chunk_size=50)))
    daily = aggregate_chunk_daily(chunk)
    row = daily.set_index(["date", "sku_id"]).loc[("2022-01-03", "SKU00001")]
    assert int(row["units_sold"]) == 4
    assert float(row["revenue"]) == pytest.approx(40.0)
    assert float(row["gross_revenue"]) == pytest.approx(40.0)


def test_daily_aggregation_computes_unit_price_and_promo_count(tmp_path: Path):
    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    from foresight.ingestion.sales_ingestion import (
        aggregate_chunk_daily,
        iter_sales_chunks,
        validate_sales_chunk,
    )

    chunk = validate_sales_chunk(next(iter_sales_chunks(path, chunk_size=50)))
    daily = aggregate_chunk_daily(chunk)
    assert {"unit_price", "promo_trans_count"}.issubset(daily.columns)
    # 2022-01-03 SKU00001: two scans, no promo -> unit_price 10.0, promo count 0
    row = daily.set_index(["date", "sku_id"]).loc[("2022-01-03", "SKU00001")]
    assert float(row["unit_price"]) == pytest.approx(10.0)
    assert int(row["promo_trans_count"]) == 0
    # 2022-01-10 SKU00001: promo txn present
    row2 = daily.set_index(["date", "sku_id"]).loc[("2022-01-10", "SKU00001")]
    assert int(row2["promo_trans_count"]) == 1


def test_store_daily_aggregation_keeps_store_grain(tmp_path: Path):
    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    from foresight.ingestion.sales_ingestion import aggregate_chunk_store_daily, iter_sales_chunks

    chunk = next(iter_sales_chunks(path, chunk_size=50))
    store_daily = aggregate_chunk_store_daily(chunk)
    assert set(store_daily.columns) >= {"date", "store_id", "sku_id", "units_sold"}
    assert store_daily["units_sold"].sum() == 8


def test_stream_sales_merges_chunks_without_full_file_read(tmp_path: Path, monkeypatch):
    path = write_sales_csv(tmp_path / "sales.csv", default_mini_sales_rows())
    from foresight.ingestion import sales_ingestion as mod

    calls: list[int | None] = []
    real_read = pd.read_csv

    def guarded_read_csv(*args, **kwargs):
        calls.append(kwargs.get("chunksize"))
        if kwargs.get("chunksize") is None:
            raise AssertionError("un-chunked pd.read_csv of sales is forbidden")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(mod.pd, "read_csv", guarded_read_csv)
    daily = mod.stream_aggregate_daily(path, chunk_size=2)
    assert daily["units_sold"].sum() == 8
    assert all(c == 2 for c in calls)
    assert len(daily) == 3  # two SKUs on 2022-01-10 plus one on 2022-01-03


def test_full_sales_file_streams_all_rows_without_type_drift(settings):
    """Plan 9.1 item 1: all ~9.945M real rows parse with stable columns/dtypes."""
    from foresight.ingestion.sales_ingestion import (
        SALES_SCHEMA,
        iter_sales_chunks,
        validate_sales_chunk,
    )

    path = settings.dataset_file("raw_sales")
    if not path.exists():
        pytest.skip("raw sales file not present on this host")

    expected_dtypes = None
    total_rows = 0
    for chunk in iter_sales_chunks(path, chunk_size=settings.ingestion.chunk_size):
        assert list(chunk.columns) == list(SALES_SCHEMA)  # no column omission
        if expected_dtypes is None:
            expected_dtypes = chunk.dtypes
        pd.testing.assert_series_equal(chunk.dtypes, expected_dtypes)  # no type drift
        validate_sales_chunk(chunk)
        total_rows += len(chunk)
    assert total_rows == settings.audit_expectations.n_sales_contaminated
