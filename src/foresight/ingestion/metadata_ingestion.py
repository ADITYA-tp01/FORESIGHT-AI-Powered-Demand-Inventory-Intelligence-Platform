"""Dimension loaders and validation contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from foresight.config import Settings


def load_sku_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sku_id": str})
    if df["sku_id"].isnull().any():
        raise ValueError("sku_master contains null sku_id")
    return df


def load_store_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"store_id": str})
    if df["store_id"].isnull().any():
        raise ValueError("store_master contains null store_id")
    return df


def load_customer_master(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"cust_id": str})


def load_promotions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"promo_id": str})
    for col in ("start_date", "end_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if df["promo_id"].isnull().any():
        raise ValueError("promotions contains null promo_id")
    return df


def load_inventory_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"store_id": str, "sku_id": str})


def load_dimensions(settings: Settings) -> dict[str, pd.DataFrame]:
    return {
        "sku_master": load_sku_master(settings.dataset_file("raw_sku")),
        "store_master": load_store_master(settings.dataset_file("raw_store")),
        "customer_master": load_customer_master(settings.dataset_file("raw_customer")),
        "promotions": load_promotions(settings.dataset_file("raw_promotions")),
        "inventory_snapshot": load_inventory_snapshot(settings.dataset_file("raw_inventory")),
    }


def assert_dimensions_match_settings(settings: Settings, dimensions: dict[str, pd.DataFrame]) -> None:
    audit = settings.audit_expectations
    checks = {
        "sku_master": audit.n_skus,
        "store_master": audit.n_stores,
        "customer_master": audit.n_customers,
        "promotions": audit.n_promotions,
        "inventory_snapshot": audit.n_inventory_contaminated,
    }
    for name, expected in checks.items():
        actual = len(dimensions.get(name, pd.DataFrame()))
        if actual != expected:
            raise ValueError(f"{name} expected {expected} rows, got {actual}")
