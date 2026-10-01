"""Risk scoring math, rupee impact, and store-grain disaggregation (Plan 6.1-6.4)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

SNAPSHOT = pd.Timestamp("2025-12-31")
FORECAST_WEEKS = [f"2026-W{w:02d}" for w in range(2, 10)]


def risk_inventory() -> pd.DataFrame:
    """Four archetypes: reorder-now, slow-mover, healthy, watch-volatile."""
    rows = [
        # SKU00001: chain SOH 5, LTD(1w)=10 -> PAB -5 (reorder_now)
        ("ST01", "SKU00001", 2, 4, 2, "2025-12-01", "Dairy & Bakery", 7),
        ("ST02", "SKU00001", 3, 4, 2, "2025-12-01", "Dairy & Bakery", 7),
        # SKU00002: WOS 150 + restock 365 days old -> markdown_clear / slow-mover
        ("ST01", "SKU00002", 1000, 100, 10, "2025-01-01", "Electronics & Accessories", 21),
        ("ST03", "SKU00002", 500, 100, 10, "2025-01-01", "Electronics & Accessories", 21),
        # SKU00003: WOS 6, PAB 40 -> healthy
        ("ST02", "SKU00003", 60, 50, 8, "2025-12-15", "Stationery & Office", 14),
        # SKU00004: PAB -9 but SOH > 3 x RP(0) -> watch_volatile
        ("ST02", "SKU00004", 1, 0, 0, "2025-12-01", "Dairy & Bakery", 7),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "last_restock_date",
            "category",
            "lead_time_days",
        ],
    ).assign(on_order_units=0, snapshot_date=SNAPSHOT)


def flat_forecasts(skus: list[str] | None = None, weekly: float = 10.0) -> pd.DataFrame:
    skus = skus or ["SKU00001", "SKU00002", "SKU00003", "SKU00004"]
    rows = [
        {"sku_id": sku, "year_week": week, "forecast": weekly}
        for sku in skus
        for week in FORECAST_WEEKS
    ]
    return pd.DataFrame(rows)


def test_lead_time_weeks_ceils_days():
    from foresight.inventory.risk_scoring import lead_time_weeks

    weeks = lead_time_weeks([4, 7, 8, 21])
    assert list(weeks) == [1, 1, 2, 3]


def test_score_chain_risk_math_and_quadrants(settings):
    from foresight.inventory.risk_scoring import score_chain_risk

    risk = score_chain_risk(risk_inventory(), flat_forecasts(), settings.risk, horizon_weeks=8)
    risk = risk.set_index("sku_id")

    reorder = risk.loc["SKU00001"]
    assert reorder["lead_time_weeks"] == 1
    assert reorder["ltd"] == 10.0
    assert reorder["pab"] == -5.0
    assert bool(reorder["stockout_high"]) is True
    # sigmoid((SS - PAB) / (SS + eps)) with SS=4, PAB=-5 -> sigmoid(9 / 4)
    assert reorder["stockout_risk"] == pytest.approx(1 / (1 + math.exp(-2.25)), rel=1e-6)
    assert reorder["wos"] == pytest.approx(0.5, rel=1e-6)
    assert reorder["forward_8w_demand"] == 80.0
    assert bool(reorder["overstock_high"]) is False
    assert reorder["quadrant"] == "reorder_now"

    slow = risk.loc["SKU00002"]
    assert slow["lead_time_weeks"] == 3
    assert slow["ltd"] == 30.0
    assert slow["pab"] == 1470.0
    assert bool(slow["stockout_high"]) is False
    assert slow["wos"] == pytest.approx(150.0, rel=1e-6)
    assert bool(slow["overstock_high"]) is True
    assert bool(slow["stale_restock"]) is True
    assert bool(slow["slow_mover_high"]) is True
    assert slow["quadrant"] == "markdown_clear"

    healthy = risk.loc["SKU00003"]
    assert healthy["lead_time_weeks"] == 2
    assert healthy["ltd"] == 20.0
    assert healthy["pab"] == 40.0
    assert healthy["wos"] == pytest.approx(6.0, rel=1e-6)
    assert bool(healthy["stale_restock"]) is False
    assert healthy["quadrant"] == "healthy"

    watch = risk.loc["SKU00004"]
    assert watch["pab"] == -9.0
    assert bool(watch["stockout_high"]) is True
    assert bool(watch["overstock_high"]) is True
    assert bool(watch["slow_mover_high"]) is False
    assert watch["quadrant"] == "watch_volatile"

    assert set(risk["quadrant"]) == {
        "reorder_now",
        "watch_volatile",
        "healthy",
        "markdown_clear",
    }


def test_score_chain_risk_missing_forecast_fills_zero(settings):
    from foresight.inventory.risk_scoring import score_chain_risk

    forecasts = flat_forecasts(skus=["SKU00001", "SKU00002", "SKU00004"])
    risk = score_chain_risk(risk_inventory(), forecasts, settings.risk, horizon_weeks=8)
    missing = risk.loc[risk["sku_id"] == "SKU00003"].iloc[0]
    assert missing["ltd"] == 0.0
    assert missing["forward_8w_demand"] == 0.0
    assert bool(missing["stockout_high"]) is False
    assert bool(missing["overstock_high"]) is True  # zero demand + stock on hand


def test_add_rupee_impact_formulas():
    from foresight.inventory.actions import add_rupee_impact

    risk = pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002"],
            "stock_on_hand": [5, 1500],
            "on_order_units": [0, 0],
            "safety_stock": [4, 20],
            "ltd": [10.0, 30.0],
            "forward_8w_demand": [80.0, 80.0],
        }
    )
    master = pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002"],
            "unit_price": [10.0, 5.0],
            "cost_price": [7.0, 3.0],
        }
    )
    out = add_rupee_impact(risk, master)
    # Rupees at Risk: max(0, LTD - (SOH + on_order)) * unit_price
    assert out.loc[0, "rupees_at_risk"] == 50.0  # (10 - 5) * 10
    assert out.loc[1, "rupees_at_risk"] == 0.0  # LTD 30 far below SOH 1500
    # Locked Capital: max(0, SOH - (forward_8w + SS)) * cost_price
    assert out.loc[0, "locked_capital"] == 0.0  # 5 - (80 + 4) < 0
    assert out.loc[1, "locked_capital"] == 4200.0  # (1500 - 100) * 3


def test_compute_store_shares_window_and_uniform_fallback():
    from foresight.inventory.store_grain import compute_store_shares

    store_daily = pd.DataFrame(
        {
            "date": [
                "2025-02-17",  # 2025-W08: outside window
                "2025-02-24",  # 2025-W09: inside
                "2025-03-03",  # 2025-W10: inside (cutoff week)
                "2025-03-10",  # 2025-W11: after cutoff
                "2025-02-24",
                "2025-03-03",
            ],
            "store_id": ["ST01", "ST01", "ST01", "ST01", "ST02", "ST02"],
            "sku_id": ["SKU00001"] * 4 + ["SKU00001"] * 2,
            "units_sold": [100, 6, 2, 100, 2, 2],
        }
    )
    inventory = pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00001", "SKU00002", "SKU00002"],
            "store_id": ["ST01", "ST02", "ST01", "ST03"],
        }
    )
    shares = compute_store_shares(store_daily, inventory, "2025-W10", window_weeks=2)
    shares = shares.set_index(["sku_id", "store_id"])

    # Window units: ST01 = 6+2 = 8, ST02 = 2+2 = 4 (W08/W11 excluded)
    assert shares.loc[("SKU00001", "ST01"), "store_units"] == 8.0
    assert shares.loc[("SKU00001", "ST02"), "store_units"] == 4.0
    assert shares.loc[("SKU00001", "ST01"), "share"] == 8.0 / 12.0
    assert shares.loc[("SKU00001", "ST02"), "share"] == 4.0 / 12.0
    # No sales in window -> uniform across stocking stores
    sku2 = shares.loc["SKU00002"]
    assert list(sku2["share"]) == [0.5, 0.5]
    for sku in ("SKU00001", "SKU00002"):
        assert shares.loc[sku]["share"].sum() == 1.0


def test_score_store_risk_disaggregation(settings):
    from foresight.inventory.risk_scoring import score_chain_risk
    from foresight.inventory.store_grain import score_store_risk

    chain = score_chain_risk(risk_inventory(), flat_forecasts(), settings.risk, horizon_weeks=8)
    shares = pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00001"],
            "store_id": ["ST01", "ST02"],
            "store_units": [3.0, 7.0],
            "chain_units": [10.0, 10.0],
            "share": [0.3, 0.7],
        }
    )
    store = score_store_risk(chain, shares, risk_inventory(), settings.risk)
    store = store.set_index("store_id")

    assert store.loc["ST01", "store_ltd"] == 3.0
    assert store.loc["ST02", "store_ltd"] == 7.0
    assert store["store_ltd"].sum() == 10.0  # disaggregation preserves chain LTD
    # ST01 SOH 2 vs LTD 3 -> outage; ST02 SOH 3 vs LTD 7 -> outage
    assert bool(store.loc["ST01", "store_outage"]) is True
    assert bool(store.loc["ST02", "store_outage"]) is True
    assert store.loc["ST02", "store_stockout_risk"] > store.loc["ST01", "store_stockout_risk"]


def test_impact_summary_zero_fills_missing_quadrants():
    from foresight.inventory.actions import impact_summary

    risk = pd.DataFrame(
        {
            "quadrant": ["healthy", "healthy"],
            "rupees_at_risk": [0.0, 5.0],
            "locked_capital": [1.0, 2.0],
        }
    )
    out = impact_summary(risk)
    assert out["n_skus_scored"] == 2
    assert out["rupees_at_risk"] == 5.0
    assert out["quadrant_counts"] == {
        "healthy": 2,
        "markdown_clear": 0,
        "reorder_now": 0,
        "watch_volatile": 0,
    }


def test_store_outage_localises_single_depleted_store(settings):
    from foresight.inventory.risk_scoring import score_chain_risk
    from foresight.inventory.store_grain import score_store_risk

    chain = score_chain_risk(risk_inventory(), flat_forecasts(), settings.risk, horizon_weeks=8)
    shares = pd.DataFrame(
        {
            "sku_id": ["SKU00003", "SKU00003"],
            "store_id": ["ST02", "ST05"],
            "store_units": [4.0, 6.0],
            "chain_units": [10.0, 10.0],
            "share": [0.4, 0.6],
        }
    )
    inventory = pd.DataFrame(
        {
            "sku_id": ["SKU00003", "SKU00003"],
            "store_id": ["ST02", "ST05"],
            "stock_on_hand": [50, 10],
            "safety_stock": [4, 4],
        }
    )
    store = score_store_risk(chain, shares, inventory, settings.risk)
    store = store.set_index("store_id")
    # Chain is healthy (LTD 20), but ST05 holds only 10 against 12 share-LTD
    assert bool(store.loc["ST02", "store_outage"]) is False
    assert bool(store.loc["ST05", "store_outage"]) is True
