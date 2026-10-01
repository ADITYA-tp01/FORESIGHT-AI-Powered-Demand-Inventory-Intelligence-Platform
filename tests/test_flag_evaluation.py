"""Flag evaluation vs sku_inventory_flags.csv answer key (Plan 6.5)."""

from __future__ import annotations

import pandas as pd

EVALUATION_CFG = {"flags_path_key": "raw_flags", "target_stockout_recall": 0.85}


def toy_chain_risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002", "SKU00003", "SKU00004"],
            "stockout_high": [True, False, False, True],
            "slow_mover_high": [False, True, False, False],
        }
    )


def toy_store_risk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00001", "SKU00002", "SKU00003", "SKU00004"],
            "store_id": ["ST01", "ST02", "ST01", "ST03", "ST02"],
            "store_outage": [True, False, True, False, True],
        }
    )


def toy_flags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku_id": ["SKU00001", "SKU00002", "SKU00003"],
            "flag": ["STOCKOUT_RISK", "SLOW_MOVER", "STOCKOUT_RISK"],
            "affected_stores": ["ST01;ST02", "ST01", "ST03"],
        }
    )


def test_binary_metrics_confusion_and_ratios():
    from foresight.inventory.flag_evaluation import binary_metrics

    result = binary_metrics(
        predicted={"a", "b", "c"},
        actual={"b", "c", "d"},
        universe={"a", "b", "c", "d", "e", "f"},
    )
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 2)
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert result["f1"] == 2 / 3


def test_binary_metrics_empty_sets_do_not_divide_by_zero():
    from foresight.inventory.flag_evaluation import binary_metrics

    no_pred = binary_metrics(predicted=set(), actual={"a"}, universe={"a", "b"})
    assert no_pred["recall"] == 0.0
    assert no_pred["precision"] == 0.0
    assert no_pred["f1"] == 0.0

    no_actual = binary_metrics(predicted={"a"}, actual=set(), universe={"a", "b"})
    assert no_actual["recall"] == 0.0
    assert no_actual["precision"] == 0.0  # "a" is a false positive
    assert no_actual["fp"] == 1
    assert no_actual["tn"] == 1
    assert no_actual["f1"] == 0.0


def test_evaluate_flags_metrics_and_target():
    from foresight.inventory.flag_evaluation import evaluate_flags

    result = evaluate_flags(toy_chain_risk(), toy_store_risk(), toy_flags(), {"evaluation": EVALUATION_CFG})

    stockout = result["stockout_risk"]
    # predicted {1,4}, actual {1,3}
    assert (stockout["tp"], stockout["fp"], stockout["fn"], stockout["tn"]) == (1, 1, 1, 1)
    assert stockout["precision"] == 0.5
    assert stockout["recall"] == 0.5
    assert stockout["f1"] == 0.5

    slow = result["slow_mover"]
    assert (slow["tp"], slow["fp"], slow["fn"]) == (1, 0, 0)
    assert slow["f1"] == 1.0

    pairs = result["store_pairs"]
    # key pairs {(1,ST01),(1,ST02),(3,ST03)}; predicted {(1,ST01),(2,ST01),(4,ST02)}
    assert (pairs["tp"], pairs["fp"], pairs["fn"]) == (1, 2, 2)
    assert pairs["n_universe"] == 5

    assert result["meets_stockout_recall_target"] is False  # 0.50 < 0.85
    assert result["key_rows_outside_universe"] == 0


def test_evaluate_flags_counts_key_rows_outside_universe():
    from foresight.inventory.flag_evaluation import evaluate_flags

    flags = pd.concat(
        [toy_flags(), pd.DataFrame({"sku_id": ["SKU99999"], "flag": ["SLOW_MOVER"],
                                    "affected_stores": ["ST01"]})],
        ignore_index=True,
    )
    result = evaluate_flags(toy_chain_risk(), toy_store_risk(), flags, {"evaluation": EVALUATION_CFG})
    assert result["key_rows_outside_universe"] == 1


def test_report_contains_verdict_and_quadrants(settings):
    from foresight.inventory.flag_evaluation import (
        build_risk_evaluation_md,
        evaluate_flags,
    )

    evaluation = evaluate_flags(
        toy_chain_risk(), toy_store_risk(), toy_flags(), {"evaluation": EVALUATION_CFG}
    )
    production = {
        "n_skus_scored": 4,
        "rupees_at_risk": 1234.0,
        "locked_capital": 6789.0,
        "quadrant_counts": {"healthy": 1, "markdown_clear": 1, "reorder_now": 2},
    }
    meta = {
        "cutoff_week": "2025-W41",
        "train_end_date": "2025-10-12",
        "horizon_weeks": 8,
        "snapshot_date": "2025-12-31",
        "forecast_weeks": "2026-W02 .. 2026-W09",
        "n_forecast_skus": 4,
    }
    md = build_risk_evaluation_md(evaluation, production, meta, settings)

    assert "NOT MET" in md  # stockout recall 0.50 < 0.85 target
    assert "2025-W41" in md
    assert "STOCKOUT_RISK (SKU)" in md
    assert "markdown_clear" in md
    assert "INR 1,234" in md
    assert "sku_inventory_flags.csv" in md
