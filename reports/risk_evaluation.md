# Deliverable D4 — Inventory Risk Intelligence & Decisioning

- Project: Project FORESIGHT
- Generated: 2026-09-30T20:16:45.491511+00:00
- Evaluation cutoff: 2025-W41 (train through 2025-10-12),
  horizon 8 weeks
- Universe: 4495 SKUs with inventory rows
  (5000 forecast SKUs; rest excluded — no stock data)
- Key rows outside universe: 0

## Methodology (configs/risk.yaml)

- LTD = sum of forecast over ceil(lead_time_days / 7); PAB = SOH + on_order - LTD
- stockout_risk = sigmoid((safety_stock - PAB) / (safety_stock + eps)); HIGH = PAB <= 0
- WOS = SOH / (mean forecast over 8 weeks); overstock HIGH =
  WOS > 12 or SOH > 3 x reorder_point
- SLOW_MOVER predicted = overstock HIGH and restock age >= 180 days
- Store path: share w = store units / chain units over trailing
  12 weeks; store_LTD = w x chain_LTD
- Predicted flags come from a model trained only through 2025-W41
  (never the production bundle trained through 2026-W01)

## Flags Key Evaluation (answer key, evaluation-only)

| Scope | Actual | Predicted | TP | FP | FN | TN | Precision | Recall | F1 |
| STOCKOUT_RISK (SKU) | 200 | 382 | 173 | 209 | 27 | 4086 | 0.453 | 0.865 | 0.595 |
| SLOW_MOVER (SKU) | 400 | 517 | 400 | 117 | 0 | 3978 | 0.774 | 1.000 | 0.872 |
| STOCKOUT_RISK (store pairs) | 2703 | 4771 | 2703 | 2068 | 0 | 21637 | 0.567 | 1.000 | 0.723 |

- Stockout recall target (85%): **MET**
- Predicted store outages: 4771 pairs;
  key pairs covered: 2703 of 2703

## Production Decision Grid (snapshot 2025-12-31, forecast 2026-W02 .. 2026-W09)

| Quadrant | SKUs |
|---|---|
| healthy | 1149 |
| markdown_clear | 3021 |
| reorder_now | 325 |
| watch_volatile | 0 |

- Rupees at Risk: INR 32,785,603
- Locked Capital: INR 1,534,592,370

## Caveats & Governance

- Stock levels are a single snapshot (2025-12-31) while flags describe
  Oct–Dec 2025; forecast staleness limits achievable precision — recall is the gate.
- The answer key was injected onto structurally distinct SKUs (depleted vs stale
  overstock); thresholds are pre-registered in configs/risk.yaml, not tuned to labels.
- sku_inventory_flags.csv was read only by this evaluation step; it never entered
  features, training, or thresholds.
- No fabrication: every figure above derives from Parquet/CSV data in this repo.
