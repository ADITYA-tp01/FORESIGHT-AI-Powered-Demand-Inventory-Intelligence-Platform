# Baseline Benchmark Scorecard

- Model: `seasonal_naive_3tier`
- Generated: 2026-09-30T16:54:21.107392+00:00

| Fold | Train End | Valid Weeks | Rows | WAPE | Bias | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| 1 | 2024-W52 | 8 | 40,000 | 38.01% | +0.0622 | 6.245 | 11.693 |
| 2 | 2025-W16 | 8 | 40,000 | 35.29% | +0.0431 | 6.631 | 10.685 |
| 3 | 2025-W40 | 12 | 60,000 | 39.05% | -0.0259 | 8.491 | 37.328 |

**Overall pooled WAPE: 37.76%**

ML models must beat these numbers on identical folds to earn deployment.
