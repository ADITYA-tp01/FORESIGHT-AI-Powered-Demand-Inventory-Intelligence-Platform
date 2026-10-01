"""Plan 9.1 item 5: quadrant boundary conditions and non-negative rupee math.

All thresholds are read from configs/risk.yaml via the shared `settings`
fixture — no hard-coded copies of production thresholds.
"""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

SNAPSHOT = pd.Timestamp("2025-12-31")
FORECAST_WEEKS = [f"2026-W{w:02d}" for w in range(2, 10)]


def _inventory_row(
    sku: str,
    stock_on_hand: float,
    reorder_point: float,
    safety_stock: float,
    lead_time_days: int,
    last_restock_date: str,
    on_order_units: float = 0.0,
    category: str = "Test",
) -> dict:
    return {
        "store_id": "ST01",
        "sku_id": sku,
        "category": category,
        "stock_on_hand": stock_on_hand,
        "on_order_units": on_order_units,
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "lead_time_days": lead_time_days,
        "last_restock_date": last_restock_date,
        "snapshot_date": SNAPSHOT,
    }


def _forecasts(per_sku_weekly: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sku_id": sku, "year_week": week, "forecast": weekly}
            for sku, weekly in per_sku_weekly.items()
            for week in FORECAST_WEEKS
        ]
    )


def _score(inventory_rows: list[dict], per_sku_weekly: dict[str, float], settings):
    from foresight.inventory.risk_scoring import score_chain_risk

    inventory = pd.DataFrame(inventory_rows)
    forecasts = _forecasts(per_sku_weekly)
    scored = score_chain_risk(inventory, forecasts, settings.risk, horizon_weeks=8)
    return scored.set_index("sku_id")


def test_stockout_boundary_pab_exactly_zero_is_high(settings):
    """stockout_high = PAB <= 0: inclusive at zero, excluded one unit above."""
    scored = _score(
        [
            _inventory_row("SKU_A", 1, 5, 4, 7, "2025-12-01", on_order_units=9),
            _inventory_row("SKU_B", 2, 5, 4, 7, "2025-12-01", on_order_units=9),
        ],
        {"SKU_A": 10.0, "SKU_B": 10.0},
        settings,
    )
    assert scored.loc["SKU_A", "pab"] == 0.0
    assert bool(scored.loc["SKU_A", "stockout_high"]) is True
    assert scored.loc["SKU_A", "quadrant"] == "reorder_now"

    assert scored.loc["SKU_B", "pab"] == 1.0
    assert bool(scored.loc["SKU_B", "stockout_high"]) is False
    assert scored.loc["SKU_B", "quadrant"] == "healthy"


def test_overstock_wos_boundary_strict_inequality(settings):
    """overstock_high = WOS > threshold: strictly-below stays out, above flips."""
    threshold = float(settings.risk["inventory_defaults"]["overstock_wos_weeks"])
    weekly = 10.0
    below = _inventory_row("SKU_W1", weekly * (threshold - 0.1), 10_000, 0, 7, "2025-12-01")
    above = _inventory_row("SKU_W2", weekly * (threshold + 0.1), 10_000, 0, 7, "2025-12-01")
    scored = _score([below, above], {"SKU_W1": weekly, "SKU_W2": weekly}, settings)

    assert scored.loc["SKU_W1", "wos"] < threshold
    assert bool(scored.loc["SKU_W1", "overstock_high"]) is False
    assert scored.loc["SKU_W1", "quadrant"] == "healthy"

    assert scored.loc["SKU_W2", "wos"] > threshold
    assert bool(scored.loc["SKU_W2", "overstock_high"]) is True
    assert scored.loc["SKU_W2", "quadrant"] == "markdown_clear"


def test_overstock_reorder_point_boundary_strict_inequality(settings):
    """SOH > multiple x RP is exclusive at exact equality (wos kept below cap)."""
    multiple = float(settings.risk["inventory_defaults"]["overstock_reorder_multiple"])
    weekly = 20.0
    rp = 40.0
    exact = _inventory_row("SKU_R1", multiple * rp, rp, 0, 7, "2025-12-01")
    above = _inventory_row("SKU_R2", multiple * rp + 1, rp, 0, 7, "2025-12-01")
    scored = _score([exact, above], {"SKU_R1": weekly, "SKU_R2": weekly}, settings)

    assert scored.loc["SKU_R1", "wos"] < float(
        settings.risk["inventory_defaults"]["overstock_wos_weeks"]
    )
    assert bool(scored.loc["SKU_R1", "overstock_high"]) is False
    assert scored.loc["SKU_R1", "quadrant"] == "healthy"

    assert bool(scored.loc["SKU_R2", "overstock_high"]) is True
    assert scored.loc["SKU_R2", "quadrant"] == "markdown_clear"


def test_stale_restock_boundary_inclusive_at_threshold(settings):
    """slow_mover = overstock AND restock_age >= threshold (inclusive)."""
    stale_days = int(settings.risk["inventory_defaults"]["restock_stale_days"])
    at_threshold = SNAPSHOT - pd.Timedelta(days=stale_days)
    just_below = SNAPSHOT - pd.Timedelta(days=stale_days - 1)
    scored = _score(
        [
            _inventory_row("SKU_S1", 500, 10_000, 0, 7, at_threshold.date().isoformat()),
            _inventory_row("SKU_S2", 500, 10_000, 0, 7, just_below.date().isoformat()),
        ],
        {"SKU_S1": 10.0, "SKU_S2": 10.0},
        settings,
    )
    assert scored.loc["SKU_S1", "restock_age_days"] == stale_days
    assert bool(scored.loc["SKU_S1", "stale_restock"]) is True
    assert bool(scored.loc["SKU_S1", "slow_mover_high"]) is True

    assert scored.loc["SKU_S2", "restock_age_days"] == stale_days - 1
    assert bool(scored.loc["SKU_S2", "stale_restock"]) is False
    assert bool(scored.loc["SKU_S2", "slow_mover_high"]) is False


def test_stockout_risk_is_half_when_pab_equals_safety_stock(settings):
    """sigmoid((SS - PAB) / (SS + eps)) == 0.5 exactly at PAB == SS."""
    scored = _score(
        [_inventory_row("SKU_H", 100, 5, 90, 7, "2025-12-01")],
        {"SKU_H": 10.0},
        settings,
    )
    assert scored.loc["SKU_H", "pab"] == 90.0
    assert scored.loc["SKU_H", "stockout_risk"] == pytest.approx(0.5, abs=1e-12)


def test_stockout_risk_stays_within_unit_interval(settings):
    scored = _score(
        [
            _inventory_row("SKU_L", 0, 5, 4, 7, "2025-12-01"),  # deep stockout
            _inventory_row("SKU_M", 500, 5, 4, 7, "2025-12-01"),  # deep surplus
        ],
        {"SKU_L": 10.0, "SKU_M": 10.0},
        settings,
    )
    assert scored["stockout_risk"].between(0.0, 1.0, inclusive="neither").all()


def test_quadrant_labels_cover_all_flag_pairs():
    from foresight.inventory.risk_scoring import QUADRANT_LABELS

    assert set(QUADRANT_LABELS) == {(True, False), (True, True), (False, True), (False, False)}
    assert len(set(QUADRANT_LABELS.values())) == 4


def test_rupee_math_strictly_non_negative_across_sweep(settings):
    """Rupees at Risk / Locked Capital never go negative over a wide grid."""
    from foresight.inventory.actions import add_rupee_impact, impact_summary
    from foresight.inventory.risk_scoring import QUADRANT_LABELS, score_chain_risk

    grid = itertools.product(
        [0.0, 5.0, 50.0, 500.0],  # stock on hand
        [0.0, 7.0],  # on order
        [1.0, 10.0, 100.0],  # reorder point
        [7, 21],  # lead time days
        [10.0, 15.0],  # safety stock
    )
    rows, weekly_by_sku = [], {}
    for index, (soh, on_order, rp, lead, safety) in enumerate(grid):
        sku = f"SWEEP{index:04d}"
        rows.append(
            _inventory_row(sku, soh, rp, safety, lead, "2025-06-15", on_order_units=on_order)
        )
        weekly_by_sku[sku] = 10.0 + index % 3

    scored = score_chain_risk(pd.DataFrame(rows), _forecasts(weekly_by_sku), settings.risk, 8)
    master = pd.DataFrame(
        {"sku_id": [row["sku_id"] for row in rows], "unit_price": 10.0, "cost_price": 7.0}
    )
    with_impact = add_rupee_impact(scored, master)

    assert with_impact["rupees_at_risk"].notna().all()
    assert with_impact["locked_capital"].notna().all()
    assert (with_impact["rupees_at_risk"] >= 0.0).all()
    assert (with_impact["locked_capital"] >= 0.0).all()
    assert (with_impact["rupees_at_risk"] == 0.0).any()  # clamps engaged
    assert (with_impact["locked_capital"] == 0.0).any()
    assert with_impact["quadrant"].isin(set(QUADRANT_LABELS.values())).all()

    summary = impact_summary(with_impact)
    assert summary["rupees_at_risk"] >= 0.0
    assert summary["locked_capital"] >= 0.0
    assert sum(summary["quadrant_counts"].values()) == summary["n_skus_scored"] == len(rows)
