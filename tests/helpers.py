"""Tiny retail fixtures for Phase 1 unit tests. Never the 801 MB sales dump."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SALES_HEADER = (
    "date,receipt_id,store_id,sku_id,customer_id,quantity,unit_price,"
    "total_value,channel,discount_pct,promo_id\n"
)


def write_sales_csv(path: Path, rows: str) -> Path:
    path.write_text(SALES_HEADER + rows, encoding="utf-8")
    return path


def mini_sku_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002"],
            "sku_name": ["Milk 1L", "Notebook"],
            "category": ["Dairy & Bakery", "Stationery & Office"],
            "subcategory": ["Milk", "Notebooks"],
            "unit_price": [10.0, 5.0],
            "cost_price": [7.0, 3.0],
            "brand": ["SunriseFoods", "SoftTouch"],
        }
    )


def mini_store_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": ["ST01", "ST02"],
            "store_name": ["Store 1", "Store 2"],
            "city": ["Karachi", "Lahore"],
            "store_type": ["Supermarket", "Express Store"],
            "opening_date": ["2017-01-01", "2018-01-01"],
        }
    )


def mini_promotions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "promo_id": ["PROMO001", "PROMO002", "PROMO003"],
            "promo_name": ["Brand sale", "All-store weaker", "Later week"],
            "start_date": ["2022-01-10", "2022-01-10", "2022-01-17"],
            "end_date": ["2022-01-16", "2022-01-16", "2022-01-23"],
            "discount_pct": [20.0, 5.0, 10.0],
            "promo_type": ["Percentage Discount", "BOGO", "Clearance"],
            "target_type": ["Brand", "All", "SKU"],
            "target_value": ["SunriseFoods", "All", "SKU00002"],
        }
    )


def mini_inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": ["ST01", "ST02"],
            "sku_id": ["SKU00001", "SKU00002"],
            "stock_on_hand": [12, 80],
            "reorder_point": [10, 20],
            "safety_stock": [4, 8],
            "last_restock_date": ["2025-12-01", "2025-06-01"],
        }
    )


def default_mini_sales_rows() -> str:
    # 2022-01-03 = ISO 2022-W01 Monday; 2022-01-10 = 2022-W02 Monday.
    # Duplicate line is a POS multi-scan and must be retained.
    return (
        "2022-01-03,R1,ST01,SKU00001,CUST00001,2,10.0,20.0,In-Store,0.0,\n"
        "2022-01-03,R1,ST01,SKU00001,CUST00001,2,10.0,20.0,In-Store,0.0,\n"
        "2022-01-10,R2,ST01,SKU00001,CUST00001,1,10.0,8.0,In-Store,20.0,PROMO001\n"
        "2022-01-10,R3,ST02,SKU00002,CUST00001,3,5.0,15.0,Online,0.0,\n"
    )


PROMO_WEEKLY_COLUMNS = [
    "year_week",
    "sku_id",
    "promo_days",
    "promo_discount_pct",
    "promo_type",
    "promo_target_level",
    "is_promo_active",
]


def mini_weekly_panel(n_weeks: int = 80, n_skus: int = 2) -> pd.DataFrame:
    """Dense synthetic weekly panel mirroring the Phase 1 schema.

    Deterministic demand (includes zero weeks), ISO week labels starting
    2023-W01, and revenue columns that must never reach the feature matrix.
    """
    start = pd.Timestamp("2023-01-02")  # Monday of ISO 2023-W01
    weeks = []
    for i in range(n_weeks):
        iso = (start + pd.Timedelta(weeks=i)).isocalendar()
        weeks.append(f"{iso.year:04d}-W{iso.week:02d}")
    rows = []
    for s in range(n_skus):
        sku = f"SKU0000{s + 1}"
        price = 10.0 + s
        for i, week in enumerate(weeks):
            units = (i * 7 + s * 5) % 19
            rows.append(
                {
                    "sku_id": sku,
                    "year_week": week,
                    "units_sold": units,
                    "revenue": units * price,
                    "gross_revenue": units * price,
                    "unit_price": price,
                    "promo_trans_count": 0,
                }
            )
    return pd.DataFrame(rows)


def empty_promo_weekly() -> pd.DataFrame:
    """Promo table with schema but no rows (no promotions in the fixture)."""
    return pd.DataFrame(columns=PROMO_WEEKLY_COLUMNS)
