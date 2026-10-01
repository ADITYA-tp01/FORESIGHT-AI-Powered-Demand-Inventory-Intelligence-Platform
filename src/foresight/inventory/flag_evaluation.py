"""Ground-truth validation of risk scores against sku_inventory_flags.csv (Plan 6.5).

Leakage rule: this module is the ONLY place the flags answer key may be read,
and only from the evaluation step of Phase 4 — never as a model feature.

Predicted flag definitions (pre-registered in configs/risk.yaml):
    STOCKOUT_RISK : chain PAB <= 0            (risk scoring, plan 6.2)
    SLOW_MOVER    : overstock HIGH and restock age >= restock_stale_days
Store-level stockout: share-disaggregated store PAB <= 0 vs the key's
`affected_stores` list (semicolon separated).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def binary_metrics(
    predicted: set, actual: set, universe: set
) -> dict[str, float | int]:
    """Precision / recall / F1 plus the full confusion matrix over `universe`."""
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(universe) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_universe": len(universe),
        "n_predicted": len(predicted),
        "n_actual": len(actual),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def key_store_pairs(flags: pd.DataFrame) -> set[tuple[str, str]]:
    """(sku, store) pairs from STOCKOUT_RISK `affected_stores`."""
    pairs: set[tuple[str, str]] = set()
    stockouts = flags.loc[flags["flag"] == "STOCKOUT_RISK"]
    for sku_id, stores in zip(stockouts["sku_id"], stockouts["affected_stores"], strict=True):
        for store in str(stores).split(";"):
            store = store.strip()
            if store:
                pairs.add((sku_id, store))
    return pairs


def evaluate_flags(
    chain_risk: pd.DataFrame,
    store_risk: pd.DataFrame,
    flags: pd.DataFrame,
    risk_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Score predicted flags against the answer key at SKU and store-pair grain."""
    evaluation_cfg = risk_cfg["evaluation"]
    universe = set(chain_risk["sku_id"])
    all_key_skus = set(flags["sku_id"])
    stockout_skus = flags.loc[flags["flag"] == "STOCKOUT_RISK", "sku_id"]
    actual_stockout = set(stockout_skus) & universe
    actual_slow = set(flags.loc[flags["flag"] == "SLOW_MOVER", "sku_id"]) & universe

    predicted_stockout = set(chain_risk.loc[chain_risk["stockout_high"], "sku_id"])
    predicted_slow = set(chain_risk.loc[chain_risk["slow_mover_high"], "sku_id"])

    stockout = binary_metrics(predicted_stockout, actual_stockout, universe)
    slow_mover = binary_metrics(predicted_slow, actual_slow, universe)

    pair_universe = set(zip(store_risk["sku_id"], store_risk["store_id"], strict=True))
    predicted_pairs = set(
        zip(
            store_risk.loc[store_risk["store_outage"], "sku_id"],
            store_risk.loc[store_risk["store_outage"], "store_id"],
            strict=True,
        )
    )
    actual_pairs = key_store_pairs(flags) & pair_universe
    store_pairs = binary_metrics(predicted_pairs, actual_pairs, pair_universe)

    target = float(evaluation_cfg["target_stockout_recall"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_stockout_recall": target,
        "meets_stockout_recall_target": stockout["recall"] >= target,
        "key_rows_outside_universe": int(len(all_key_skus - universe)),
        "stockout_risk": stockout,
        "slow_mover": slow_mover,
        "store_pairs": store_pairs,
    }


def _metrics_row(name: str, m: dict[str, float | int]) -> str:
    return (
        f"| {name} | {m['n_actual']} | {m['n_predicted']} | {m['tp']} | {m['fp']} "
        f"| {m['fn']} | {m['tn']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
    )


def build_risk_evaluation_md(
    evaluation: dict[str, Any],
    production: dict[str, Any],
    meta: dict[str, Any],
    settings: Any,
) -> str:
    """Deliverable D4 memo: methodology, metric tables, grid impact, caveats."""
    defaults = settings.risk["inventory_defaults"]
    stockout = evaluation["stockout_risk"]
    slow = evaluation["slow_mover"]
    pairs = evaluation["store_pairs"]
    target = evaluation["target_stockout_recall"]
    verdict = "MET" if evaluation["meets_stockout_recall_target"] else "NOT MET"
    quadrants = production.get("quadrant_counts", {})
    quadrant_rows = "\n".join(
        f"| {label} | {count} |" for label, count in sorted(quadrants.items())
    )
    header = (
        "| Scope | Actual | Predicted | TP | FP | FN | TN | Precision | Recall | F1 |"
    )
    return f"""# Deliverable D4 — Inventory Risk Intelligence & Decisioning

- Project: {settings.system.project_name}
- Generated: {evaluation['generated_at']}
- Evaluation cutoff: {meta['cutoff_week']} (train through {meta['train_end_date']}),
  horizon {meta['horizon_weeks']} weeks
- Universe: {stockout['n_universe']} SKUs with inventory rows
  ({meta['n_forecast_skus']} forecast SKUs; rest excluded — no stock data)
- Key rows outside universe: {evaluation['key_rows_outside_universe']}

## Methodology (configs/risk.yaml)

- LTD = sum of forecast over ceil(lead_time_days / 7); PAB = SOH + on_order - LTD
- stockout_risk = sigmoid((safety_stock - PAB) / (safety_stock + eps)); HIGH = PAB <= 0
- WOS = SOH / (mean forecast over {meta['horizon_weeks']} weeks); overstock HIGH =
  WOS > {defaults['overstock_wos_weeks']} or SOH > {defaults['overstock_reorder_multiple']} x reorder_point
- SLOW_MOVER predicted = overstock HIGH and restock age >= {defaults['restock_stale_days']} days
- Store path: share w = store units / chain units over trailing
  {defaults['store_share_window_weeks']} weeks; store_LTD = w x chain_LTD
- Predicted flags come from a model trained only through {meta['cutoff_week']}
  (never the production bundle trained through 2026-W01)

## Flags Key Evaluation (answer key, evaluation-only)

{header}
{_metrics_row('STOCKOUT_RISK (SKU)', stockout)}
{_metrics_row('SLOW_MOVER (SKU)', slow)}
{_metrics_row('STOCKOUT_RISK (store pairs)', pairs)}

- Stockout recall target ({target:.0%}): **{verdict}**
- Predicted store outages: {pairs['n_predicted']} pairs;
  key pairs covered: {pairs['tp']} of {pairs['n_actual']}

## Production Decision Grid (snapshot {meta['snapshot_date']}, forecast {meta['forecast_weeks']})

| Quadrant | SKUs |
|---|---|
{quadrant_rows}

- Rupees at Risk: INR {production['rupees_at_risk']:,.0f}
- Locked Capital: INR {production['locked_capital']:,.0f}

## Caveats & Governance

- Stock levels are a single snapshot ({meta['snapshot_date']}) while flags describe
  Oct–Dec 2025; forecast staleness limits achievable precision — recall is the gate.
- The answer key was injected onto structurally distinct SKUs (depleted vs stale
  overstock); thresholds are pre-registered in configs/risk.yaml, not tuned to labels.
- sku_inventory_flags.csv was read only by this evaluation step; it never entered
  features, training, or thresholds.
- No fabrication: every figure above derives from Parquet/CSV data in this repo.
"""
