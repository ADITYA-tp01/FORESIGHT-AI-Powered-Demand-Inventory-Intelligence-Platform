"""Inventory snapshot normalization with YAML lead-time heuristics."""

from __future__ import annotations

import pytest
from tests.helpers import mini_inventory, mini_sku_master


def test_inventory_adds_category_lead_times_and_on_order_default():
    from foresight.preprocessing.inventory import normalize_inventory

    lead_times = {
        "Dairy & Bakery": 4,
        "Stationery & Office": 14,
    }
    inv = normalize_inventory(
        mini_inventory(),
        mini_sku_master(),
        lead_time_days=lead_times,
        on_order_units_default=0,
        snapshot_date="2025-12-31",
    )
    milk = inv.loc[inv["sku_id"] == "SKU00001"].iloc[0]
    assert int(milk["lead_time_days"]) == 4
    assert int(milk["on_order_units"]) == 0
    assert str(milk["snapshot_date"])[:10] == "2025-12-31"
    note = inv.loc[inv["sku_id"] == "SKU00002"].iloc[0]
    assert int(note["lead_time_days"]) == 14


def test_unknown_category_raises():
    from foresight.preprocessing.inventory import normalize_inventory

    sku = mini_sku_master()
    sku.loc[0, "category"] = "Unknown Cat"
    with pytest.raises(KeyError, match="Missing lead times for categories"):
        normalize_inventory(
            mini_inventory(),
            sku,
            lead_time_days={"Dairy & Bakery": 4},
            on_order_units_default=0,
            snapshot_date="2025-12-31",
        )
