"""Chunked, streaming sales ingestion with validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foresight.config import Settings
from foresight.logging_config import configure_logging

SALES_SCHEMA = {
    "date": str,
    "receipt_id": str,
    "store_id": str,
    "sku_id": str,
    "customer_id": str,
    "quantity": "int32",
    "unit_price": "float32",
    "total_value": "float32",
    "channel": str,
    "discount_pct": "float32",
    "promo_id": str,
}


def logger(settings: Settings | None = None):
    return configure_logging(settings.system.log_level if settings else "INFO")


def iter_sales_chunks(path: Path, chunk_size: int = 500_000):
    return pd.read_csv(
        path,
        dtype=SALES_SCHEMA,
        chunksize=chunk_size,
        na_filter=False,
    )


def validate_sales_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if chunk["quantity"].isnull().any() or not ((chunk["quantity"] >= 1) & (chunk["quantity"] <= 5)).all():
        raise ValueError(f"quantity violation in chunk shape={chunk.shape}")
    if (chunk["unit_price"] <= 0).any():
        raise ValueError("unit_price must be strictly positive")
    if chunk[["date", "sku_id", "store_id", "quantity", "unit_price"]].isnull().any(axis=None):
        raise ValueError("mandatory sales fields contain nulls")
    return chunk


def _daily_from_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    marked = chunk.assign(
        _gross=chunk["quantity"] * chunk["unit_price"],
        _promo=(chunk["promo_id"].astype("string").fillna("").str.len() > 0).astype("int32"),
    )
    daily = (
        marked.groupby(["date", "sku_id"], observed=True, dropna=False)
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum"),
            gross_revenue=("_gross", "sum"),
            unit_price=("unit_price", "max"),
            promo_trans_count=("_promo", "sum"),
        )
        .reset_index()
    )
    return daily


def aggregate_chunk_daily(chunk: pd.DataFrame) -> pd.DataFrame:
    return _daily_from_chunk(chunk)


def aggregate_chunk_store_daily(chunk: pd.DataFrame) -> pd.DataFrame:
    return (
        chunk.groupby(["date", "store_id", "sku_id"], observed=True, dropna=False)
        .agg(units_sold=("quantity", "sum"))
        .reset_index()
    )


def stream_aggregate_daily(path: Path, chunk_size: int = 500_000) -> pd.DataFrame:
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for chunk in iter_sales_chunks(path, chunk_size):
        chunk = validate_sales_chunk(chunk)
        daily = _daily_from_chunk(chunk)
        for _, row in daily.iterrows():
            key = (str(row["date"]), str(row["sku_id"]))
            if key not in acc:
                acc[key] = {
                    "units_sold": 0.0,
                    "revenue": 0.0,
                    "gross_revenue": 0.0,
                    "unit_price": 0.0,
                    "promo_trans_count": 0.0,
                }
            acc[key]["units_sold"] += row["units_sold"]
            acc[key]["revenue"] += row["revenue"]
            acc[key]["gross_revenue"] += row["gross_revenue"]
            acc[key]["unit_price"] = max(acc[key]["unit_price"], row["unit_price"])
            acc[key]["promo_trans_count"] += row["promo_trans_count"]
    if not acc:
        return pd.DataFrame(
            columns=[
                "date",
                "sku_id",
                "units_sold",
                "revenue",
                "gross_revenue",
                "unit_price",
                "promo_trans_count",
            ]
        )
    rows = [
        {
            "date": date,
            "sku_id": sku,
            "units_sold": vals["units_sold"],
            "revenue": vals["revenue"],
            "gross_revenue": vals["gross_revenue"],
            "unit_price": vals["unit_price"],
            "promo_trans_count": vals["promo_trans_count"],
        }
        for (date, sku), vals in acc.items()
    ]
    daily = pd.DataFrame(rows)
    daily["units_sold"] = daily["units_sold"].astype("int32")
    daily["revenue"] = daily["revenue"].astype("float32")
    daily["gross_revenue"] = daily["gross_revenue"].astype("float32")
    daily["unit_price"] = daily["unit_price"].astype("float32")
    daily["promo_trans_count"] = daily["promo_trans_count"].astype("int32")
    return daily


def stream_aggregate_store_daily(path: Path, chunk_size: int = 500_000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in iter_sales_chunks(path, chunk_size):
        frames.append(aggregate_chunk_store_daily(chunk))
    if not frames:
        return pd.DataFrame(columns=["date", "store_id", "sku_id", "units_sold"])
    out = (
        pd.concat(frames, ignore_index=True)
        .groupby(["date", "store_id", "sku_id"], observed=True, dropna=False)["units_sold"]
        .sum()
        .reset_index()
    )
    out["units_sold"] = out["units_sold"].astype("int32")
    return out
