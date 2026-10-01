# Deliverable D2 — EDA Insight Memo & Baseline Benchmark

- Project: Project FORESIGHT
- Grain: weekly × SKU (5,000 SKUs × 210 weeks)
- Total units: 18,700,906
- Total revenue: 10,885,476,352 PKR

## 1. Demand Dynamics

### Pareto Skew
- Top 1,000 SKUs (20.00% of catalog) generate **73.08%** of revenue.

### Sparsity
- Zero-demand SKU-weeks: **1.61%** of the dense panel.

### Seasonality (units by month)
- Peak month: **month 12** (2,037,489 units)
- Trough month: **month 2** (1,223,965 units)

## 2. ABC / XYZ Segmentation

### ABC (revenue contribution)
| Class | SKUs | Share |
|---|---|---|
| A | 873 | 17.46% |
| B | 1,348 | 26.96% |
| C | 2,779 | 55.58% |

### XYZ (coefficient of variation)
| Class | SKUs | Share |
|---|---|---|
| X | 1,911 | 38.22% |
| Y | 3,089 | 61.78% |
| Z | 0 | 0.00% |

### Combined Segments (top 9)
| Segment | SKUs | Share |
|---|---|---|
| CY | 2,069 | 41.38% |
| BY | 744 | 14.88% |
| CX | 710 | 14.20% |
| BX | 604 | 12.08% |
| AX | 597 | 11.94% |
| AY | 276 | 5.52% |

Class meanings: X: CV <= 0.5 (predictable), Y: 0.5 < CV <= 1.0 (moderate), Z: CV > 1.0 (intermittent/lumpy).

## 3. Promotional Activity
- Promo share of all transactions: **21.02%**
- Units sold in SKU-weeks with >=1 promo transaction: **27.28%**
- Overlap precedence rule: highest discount wins (41 overlapping campaign pairs resolved).

## 4. Inventory Health
- Snapshot records: 26,408
- Mean stock: 166.20 units (median 140.0)
- Store-SKU outages (stock <= 0): 4,239

## 5. Seasonal-Naive Baseline Scorecard (Mandatory Benchmark)

3-tier fallback hierarchy: (1) seasonal-naive t-52w if observed & > 0,
(2) trailing 4-week mean if > 0, (3) category run-rate.

| Fold | Train End | Valid Weeks | Rows | WAPE | Bias | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| 1 | 2024-W52 | 8 | 40,000 | 38.01% | +0.0622 | 6.245 | 11.693 |
| 2 | 2025-W16 | 8 | 40,000 | 35.29% | +0.0431 | 6.631 | 10.685 |
| 3 | 2025-W40 | 12 | 60,000 | 39.05% | -0.0259 | 8.491 | 37.328 |

**Overall pooled WAPE: 37.76%**

## Governance
- `sku_inventory_flags.csv` used strictly as out-of-sample evaluation resource; never a feature.
- All validation is rolling-origin temporal (no random shuffling).
- Primary metric: WAPE (division-by-zero safe on zero-demand weeks).
