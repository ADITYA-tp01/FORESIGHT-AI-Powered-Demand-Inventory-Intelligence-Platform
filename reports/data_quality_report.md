# Deliverable D1 — Reproducible Data Pipeline

- Project: Project FORESIGHT
- Dataset root: C:\Users\adity\Documents\Zidio\Dataset
- Chunk size: 500000
- Sales file bytes: 801086326

## Outputs
- `data/processed/sales_daily.parquet` (4143430 rows)
- `data/processed/sales_store_daily.parquet` (see reconciliation)
- `data/processed/sales_weekly.parquet` (1050000 dense rows)
- `data/processed/calendar.parquet` (1461 days)
- `data/processed/promo_weekly.parquet` (1585393 promo-week rows)
- `data/interim/inventory_normalized.parquet` (26408 rows)

## Reconciliation
- daily units = weekly units = chain aggregate: 18700906
- store_daily units: 18700906

## Governance
- sku_inventory_flags.csv: not touched (evaluation resource only).
- Sales CSV streamed in 500000-row chunks.
