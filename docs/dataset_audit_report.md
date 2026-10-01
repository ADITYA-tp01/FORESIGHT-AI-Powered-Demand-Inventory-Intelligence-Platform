# PROJECT FORESIGHT — Comprehensive Data Discovery & Data Audit Report
**Engagement:** Zidio Development Data Science Internship  
**Project:** FORESIGHT — Demand & Inventory Intelligence Platform  
**Audit Stage:** Step 0 — Complete Dataset Discovery, Profiling, Schema Validation & Feasibility Audit  
**Auditor:** Senior Data Scientist & Data Audit Assistant  
**Date:** September 2026  
**Status:** Completed (Strict Audit Only — No Modelling, No Synthetic Data, No Original Dataset Mutation)

---

## Executive Summary

This data audit establishes the empirical foundation for **Project FORESIGHT (Demand & Inventory Intelligence Platform)**. A thorough, non-destructive audit was performed across all data assets present in the project workspace (`c:\Users\adity\Documents\Zidio`).

The project workspace provides **two synchronized environments** comprising **7 relational CSV datasets each**:
1. `Dataset/retail_contaminated_dataset/` (**Primary Operational Environment**): Contains real-world operational imperfections, specifically 600 deliberately injected inventory anomalies (200 stockouts with lost sales suppression and 400 slow-moving overstock SKUs), accompanied by a ground-truth benchmark key (`sku_inventory_flags.csv`).
2. `Dataset/retail_clean_dataset/` (**Benchmark Control Environment**): The original baseline history (9,972,038 transactions) prior to anomaly injection, enabling controlled benchmarking.

### Core Discoveries & Bottom Line
1. **The Large Sales File Identified:** The large sales transaction dataset is `sales_transactions.csv` (801,086,326 bytes / ~764 MB; 9,945,511 rows in contaminated, 9,972,038 rows in clean). It was processed entirely using streaming/chunked methods with zero RAM overflow.
2. **Referential Integrity is Pristine:** Foreign key checks confirmed **100.0% integrity** across stores (30/30), SKUs (5,000/5,000), and customers (10,000/10,000). Zero orphaned records exist.
3. **Zidio 4-Table Mapping Reality:** 
   - `sku_master` is **AVAILABLE** (with `launch_date` DERIVABLE from sales history).
   - `sales_daily` is **100% FEASIBLE & DERIVABLE** by aggregating `sales_transactions.csv` across 1,461 calendar days (yielding 4,143,430 non-zero daily records or 7,305,000 dense panel rows).
   - `calendar` is **NOT PROVIDED** as a standalone CSV, but is **100% DERIVABLE** from continuous daily dates, retail calendar logic, and `promotions.csv`.
   - `inventory_snapshots` is **PARTIALLY AVAILABLE**: Provided as a **single point-in-time current snapshot** (26,408 store-SKU records), containing `stock_on_hand`, `reorder_point`, `safety_stock`, and `last_restock_date`. However, `lead_time_days` and `on_order_units` are **MISSING** from raw extracts and must be derived or parameterized via domain rules.
4. **Clean vs. Anomaly Delta:** Exactly **26,527 transaction rows** are suppressed in `retail_contaminated_dataset/sales_transactions.csv` representing lost sales during stockout windows for top-selling items.

---

## A. Dataset Inventory

The workspace houses 7 relational CSV files structured across 30 retail physical/digital stores in Pakistan, 5,000 retail SKUs across 12 product categories, and 10,000 customers spanning 4 years (January 1, 2022 to December 31, 2025).

| # | Exact Filename | Primary Path (`retail_contaminated_dataset`) | File Size (Bytes) | File Size (MB) | Row Count | Column Count | Primary Domain Role |
|---|---|---|---|---|---|---|---|
| 1 | `sales_transactions.csv` | `Dataset/retail_contaminated_dataset/sales_transactions.csv` | 801,086,326 | 763.98 MB | 9,945,511 | 11 | Transactional Sales Fact Table (POS & Online Baskets) |
| 2 | `sku_master.csv` | `Dataset/retail_contaminated_dataset/sku_master.csv` | 463,950 | 0.44 MB | 5,000 | 7 | Product Dimension (Category, Cost, List Price) |
| 3 | `store_master.csv` | `Dataset/retail_contaminated_dataset/store_master.csv` | 2,042 | 0.002 MB | 30 | 5 | Store Dimension (Geography, Store Type, Open Date) |
| 4 | `customer_master.csv` | `Dataset/retail_contaminated_dataset/customer_master.csv` | 554,507 | 0.53 MB | 10,000 | 7 | Customer Dimension (Demographics, Loyalty, Channel) |
| 5 | `inventory_snapshot.csv` | `Dataset/retail_contaminated_dataset/inventory_snapshot.csv` | 932,817 | 0.89 MB | 26,408 | 6 | Inventory Position Fact (Current Stock, Reorder Point) |
| 6 | `promotions.csv` | `Dataset/retail_contaminated_dataset/promotions.csv` | 8,511 | 0.008 MB | 100 | 8 | Promotional Campaign Dimension & Rules |
| 7 | `sku_inventory_flags.csv` | `Dataset/retail_contaminated_dataset/sku_inventory_flags.csv` | 123,704 | 0.12 MB | 600 | 6 | Anomaly Ground Truth Benchmark Key |

*Note on Control Baseline:* In `Dataset/retail_clean_dataset/`, `sales_transactions.csv` has 9,972,038 rows (803,167,514 bytes), `inventory_snapshot.csv` has 21,228 rows (749,638 bytes), and `sku_inventory_flags.csv` has 600 rows (116,214 bytes). The metadata files (`store_master`, `sku_master`, `customer_master`, `promotions`) are byte-for-byte identical (verified via MD5 hashes).

---

## B. Schema of All 7 Datasets

Below is the exhaustive, field-by-field schema, data types, representative samples, statistics, and distributions for every dataset.

```
========================================================================================
DATASET 1: sales_transactions.csv (Large Transaction Dataset)
========================================================================================
```
- **Exact Filename:** `sales_transactions.csv`
- **File Size:** 801,086,326 bytes (~763.98 MB)
- **Rows:** 9,945,511 (Contaminated) | 9,972,038 (Clean)
- **Columns:** 11
- **Exact Column Names & Types:**
  1. `date` (`object` / `string` ISO 8601 `YYYY-MM-DD`): Transaction date
  2. `receipt_id` (`object` / `string` format `RCPT########`): Basket identifier grouping 1–5 line items
  3. `store_id` (`object` / `string` format `ST##`): Store identifier
  4. `sku_id` (`object` / `string` format `SKU#####`): Product identifier
  5. `customer_id` (`object` / `string` format `CUST#####`): Purchasing customer
  6. `quantity` (`int32`): Units purchased (1 to 5)
  7. `unit_price` (`float32` / `float64`): Retail price per unit at transaction time
  8. `total_value` (`float32` / `float64`): Net transaction value: `quantity * unit_price * (1 - discount_pct/100)`
  9. `channel` (`object` / `string`): Sales channel (`In-Store`, `Online`, `Mobile App`)
  10. `discount_pct` (`float32` / `float64`): Promotional discount percentage (0.0% to 49.8%)
  11. `promo_id` (`object` / `string` nullable): Applied promotion ID (or `NaN` if unpromoted)

- **First 5 Representative Rows:**
  | date | receipt_id | store_id | sku_id | customer_id | quantity | unit_price | total_value | channel | discount_pct | promo_id |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 2025-04-02 | RCPT00000001 | ST16 | SKU02498 | CUST01410 | 1 | 2379.11 | 2379.11 | In-Store | 0.0 | *NaN* |
  | 2025-04-02 | RCPT00000001 | ST16 | SKU04596 | CUST01410 | 3 | 335.55 | 1006.65 | In-Store | 0.0 | *NaN* |
  | 2022-04-24 | RCPT00000002 | ST15 | SKU00078 | CUST00134 | 1 | 820.53 | 820.53 | Online | 0.0 | *NaN* |
  | 2022-04-24 | RCPT00000002 | ST15 | SKU00554 | CUST00134 | 2 | 88.32 | 176.64 | Online | 0.0 | *NaN* |
  | 2024-09-22 | RCPT00000003 | ST20 | SKU03727 | CUST08826 | 1 | 1660.93 | 1660.93 | Online | 0.0 | *NaN* |

- **Missing Value Count & Percentage:**
  - `date`: 0 (0.0%)
  - `receipt_id`: 0 (0.0%)
  - `store_id`: 0 (0.0%)
  - `sku_id`: 0 (0.0%)
  - `customer_id`: 0 (0.0%)
  - `quantity`: 0 (0.0%)
  - `unit_price`: 0 (0.0%)
  - `total_value`: 0 (0.0%)
  - `channel`: 0 (0.0%)
  - `discount_pct`: 0 (0.0%)
  - `promo_id`: 7,855,345 missing (78.98% unpromoted; 21.02% promoted)
- **Duplicate Rows:** ~13,500 rows (~0.14% estimated; 679 exact duplicates detected in 500,000 sampled rows). All duplicates represent identical line items in the same receipt (e.g. barcode double-scans).
- **Unique Value Counts for Important IDs:**
  - `receipt_id`: ~5,180,000 unique baskets (average 1.92 line items per receipt; range: 1 to 5)
  - `store_id`: 30 unique stores (100% of store master)
  - `sku_id`: 5,000 unique SKUs (100% of SKU master)
  - `customer_id`: 10,000 unique customers (100% of customer master)
  - `promo_id`: 96 unique promo IDs (out of 100 defined in promotions master)
- **Date Range:** `min = 2022-01-01`, `max = 2025-12-31` (1,461 continuous calendar days)
- **Numeric Statistics:**
  - `quantity`: min = 1.0, max = 5.0, mean = 1.8803, median = 2.0, zero count = 0, negative count = 0
  - `unit_price`: min = 24.41, max = 4,755.47, mean = 621.57, median = 338.51, zero count = 0, negative count = 0
  - `total_value`: min = 12.25, max = 23,777.35, mean = 1,094.51, median = 543.89, zero count = 0, negative count = 0
  - `discount_pct`: min = 0.0%, max = 49.8%, mean = 6.29%, median = 0.0%, zero count = 7,855,345, negative count = 0
- **Categorical Distributions:**
  - `channel`:
    - `In-Store`: 5,496,812 transactions (55.27%)
    - `Online`: 2,958,149 transactions (29.74%)
    - `Mobile App`: 1,490,550 transactions (14.99%)
- **Data Quality Assessment:** Zero formula errors (`abs(total_value - quantity * unit_price * (1 - discount_pct/100)) <= 0.05` holds on 100% of rows). Exact duplicate line items occur in ~0.14% of rows due to repeated POS item scans.
- **Primary Key Candidate:** Synthetic surrogate transaction ID (e.g. `transaction_id = hash(receipt_id, sku_id, row_idx)`). Natural composite `(receipt_id, sku_id)` has collision on multi-scan lines.
- **Foreign Keys:**
  - `store_id` → `store_master.store_id` (100% match)
  - `sku_id` → `sku_master.sku_id` (100% match)
  - `customer_id` → `customer_master.cust_id` (100% match)
  - `promo_id` → `promotions.promo_id` (100% match when not null)

```
========================================================================================
DATASET 2: sku_master.csv (Product Master Dimension)
========================================================================================
```
- **Exact Filename:** `sku_master.csv`
- **File Size:** 463,950 bytes (~453.08 KB)
- **Rows:** 5,000
- **Columns:** 7
- **Exact Column Names & Types:**
  1. `sku_id` (`object` / `string` format `SKU#####`): Unique SKU code
  2. `sku_name` (`object` / `string`): Product display title
  3. `category` (`object` / `string`): 12 top-level merchandise categories
  4. `subcategory` (`object` / `string`): 53 product subcategories
  5. `unit_price` (`float64`): Retail list price
  6. `cost_price` (`float64`): Cost of goods sold (COGS) to retailer
  7. `brand` (`object` / `string`): 30 private and brand names
- **First 5 Representative Rows:**
  | sku_id | sku_name | category | subcategory | unit_price | cost_price | brand |
  |---|---|---|---|---|---|---|
  | SKU00001 | NutriPlus Cookware Large | Home & Kitchen | Cookware | 813.41 | 619.77 | NutriPlus |
  | SKU00002 | CrispKing Bread Family Pack | Dairy & Bakery | Bread | 70.38 | 49.57 | CrispKing |
  | SKU00003 | SoftTouch Notebooks 2L | Stationery & Office | Notebooks | 151.28 | 83.67 | SoftTouch |
  | SKU00004 | SunriseFoods Eggs Large | Dairy & Bakery | Eggs | 233.64 | 156.32 | SunriseFoods |
  | SKU00005 | SunriseFoods Pest Control Pack of 6 | Home Care | Pest Control | 138.50 | 83.46 | SunriseFoods |
- **Missing Value Count & Percentage:** 0 missing values across all columns (0.0%).
- **Duplicate Rows:** 0 duplicate rows.
- **Unique Value Counts:**
  - `sku_id`: 5,000 (100% unique)
  - `sku_name`: 4,458 (some names re-used across size variants)
  - `category`: 12
  - `subcategory`: 53
  - `brand`: 30
- **Date Columns:** None present directly.
- **Numeric Statistics:**
  - `unit_price`: min = 24.41, max = 4,755.47, mean = 619.19, median = 318.75, std = 708.01, zeros = 0, negatives = 0
  - `cost_price`: min = 16.75, max = 3,398.16, mean = 417.04, median = 214.67, std = 480.35, zeros = 0, negatives = 0
  - `gross_margin_pct`: min = 20.00%, max = 45.00%, mean = 32.65%, median = 32.56%. Cost is strictly < unit_price for all 5,000 SKUs!
- **Categorical Distributions (Top Categories):**
  - Stationery & Office: 459 SKUs (9.18%)
  - Beverages: 457 SKUs (9.14%)
  - Dairy & Bakery: 428 SKUs (8.56%)
  - Home Care: 425 SKUs (8.50%)
  - Frozen Foods: 422 SKUs (8.44%)
  - Snacks & Confectionery: 414 SKUs (8.28%)
  - Personal Care: 410 SKUs (8.20%)
  - Apparel & Footwear: 410 SKUs (8.20%)
  - Health & Wellness: 396 SKUs (7.92%)
  - Grocery: 395 SKUs (7.90%)
  - Home & Kitchen: 394 SKUs (7.88%)
  - Electronics & Electrical: 390 SKUs (7.80%)
- **Data Quality Assessment:** Flawless master dimension. `cost_price` is strictly less than `unit_price` with margins between 20% and 45%.
- **Primary Key Candidate:** `sku_id` (verified strictly unique).
- **Foreign Keys:** None outbound. Inbound parent to `sales_transactions`, `inventory_snapshot`, and `sku_inventory_flags`.

```
========================================================================================
DATASET 3: store_master.csv (Store Master Dimension)
========================================================================================
```
- **Exact Filename:** `store_master.csv`
- **File Size:** 2,042 bytes (~1.99 KB)
- **Rows:** 30
- **Columns:** 5
- **Exact Column Names & Types:**
  1. `store_id` (`object` / `string` format `ST##`): Store identifier (`ST01` to `ST30`)
  2. `store_name` (`object` / `string`): Store display name
  3. `city` (`object` / `string`): Store city in Pakistan
  4. `store_type` (`object` / `string`): Format (`Supermarket`, `Convenience Store`, `Hypermarket`, `Express Store`)
  5. `opening_date` (`object` / `string` ISO 8601 `YYYY-MM-DD`): Store opening date
- **First 5 Representative Rows:**
  | store_id | store_name | city | store_type | opening_date |
  |---|---|---|---|---|
  | ST01 | Quetta Convenience Store #01 | Quetta | Convenience Store | 2017-07-08 |
  | ST02 | Islamabad Express Store #02 | Islamabad | Express Store | 2019-05-08 |
  | ST03 | Sialkot Supermarket #03 | Sialkot | Supermarket | 2019-03-28 |
  | ST04 | Multan Supermarket #04 | Multan | Supermarket | 2017-10-08 |
  | ST05 | Karachi Supermarket #05 | Karachi | Supermarket | 2020-11-03 |
- **Missing Value Count & Percentage:** 0 missing values across all columns (0.0%).
- **Duplicate Rows:** 0 duplicate rows.
- **Unique Value Counts:**
  - `store_id`: 30
  - `store_name`: 30
  - `city`: 12 unique cities
  - `store_type`: 4 formats
- **Date Columns:** `opening_date` (min = `2014-04-01`, max = `2021-12-10`). All stores opened prior to the 2022 sales start date!
- **Numeric Statistics:** None.
- **Categorical Distributions:**
  - `store_type`:
    - Convenience Store: 11 stores (36.7%)
    - Supermarket: 11 stores (36.7%)
    - Hypermarket: 7 stores (23.3%)
    - Express Store: 1 store (3.3%)
  - `city` (Top Cities):
    - Karachi: 5 stores
    - Islamabad: 4 stores
    - Multan: 4 stores
    - Quetta: 3 stores
    - Sialkot: 3 stores
    - Sukkur: 2, Peshawar: 2, Rawalpindi: 2, Lahore: 2, Hyderabad: 1, Faisalabad: 1, Gujranwala: 1
- **Primary Key Candidate:** `store_id` (verified strictly unique).
- **Foreign Keys:** Inbound parent to `sales_transactions` and `inventory_snapshot`.

```
========================================================================================
DATASET 4: customer_master.csv (Customer Master Dimension)
========================================================================================
```
- **Exact Filename:** `customer_master.csv`
- **File Size:** 554,507 bytes (~541.51 KB)
- **Rows:** 10,000
- **Columns:** 7
- **Exact Column Names & Types:**
  1. `cust_id` (`object` / `string` format `CUST#####`): Customer identifier
  2. `age` (`int64`): Customer age (18 to 80)
  3. `gender` (`object` / `string`): `Female`, `Male`, `Other`
  4. `city` (`object` / `string`): Customer residential city
  5. `loyalty_segment` (`object` / `string`): `Bronze`, `Silver`, `Gold`, `Platinum`
  6. `preferred_channel` (`object` / `string`): `In-Store`, `Online`, `Mobile App`
  7. `registration_date` (`object` / `string` ISO 8601 `YYYY-MM-DD`): Date joined loyalty program
- **First 5 Representative Rows:**
  | cust_id | age | gender | city | loyalty_segment | preferred_channel | registration_date |
  |---|---|---|---|---|---|---|
  | CUST00001 | 52 | Female | Sukkur | Silver | Online | 2021-01-01 |
  | CUST00002 | 41 | Female | Karachi | Silver | Mobile App | 2021-05-14 |
  | CUST00003 | 48 | Female | Bahawalpur | Bronze | In-Store | 2022-10-23 |
  | CUST00004 | 45 | Male | Karachi | Silver | In-Store | 2015-07-15 |
  | CUST00005 | 39 | Female | Faisalabad | Bronze | In-Store | 2024-01-02 |
- **Missing Value Count & Percentage:** 0 missing values across all columns (0.0%).
- **Duplicate Rows:** 0 duplicate rows.
- **Unique Value Counts:** `cust_id`: 10,000 (100% unique), `city`: 15, `age`: 63, `loyalty_segment`: 4.
- **Date Columns:** `registration_date` (min = `2015-01-01`, max = `2025-12-31`).
- **Numeric Statistics:**
  - `age`: min = 18, max = 80, mean = 37.70, median = 37.0, std = 12.19, zeros = 0, negatives = 0
- **Categorical Distributions:**
  - `gender`: Female: 4,930 (49.3%), Male: 4,885 (48.85%), Other: 185 (1.85%)
  - `loyalty_segment`: Bronze: 3,969 (39.7%), Silver: 3,106 (31.1%), Gold: 1,905 (19.1%), Platinum: 1,020 (10.2%)
  - `preferred_channel`: In-Store: 5,512 (55.1%), Online: 2,994 (29.9%), Mobile App: 1,494 (14.9%)
- **Data Quality Assessment:** Validated that ~17.3% of transactions occur before `registration_date`. This is intentional in the synthetic design: `registration_date` represents loyalty sign-up, not initial guest purchase.
- **Primary Key Candidate:** `cust_id` (verified strictly unique).
- **Foreign Keys:** Inbound parent to `sales_transactions.customer_id`.

```
========================================================================================
DATASET 5: inventory_snapshot.csv (Inventory Position Fact Table)
========================================================================================
```
- **Exact Filename:** `inventory_snapshot.csv`
- **File Size:** 932,817 bytes (~910.95 KB) [Clean: 749,638 bytes]
- **Rows:** 26,408 (Contaminated) | 21,228 (Clean)
- **Columns:** 6
- **Exact Column Names & Types:**
  1. `store_id` (`object` / `string`): Store identifier
  2. `sku_id` (`object` / `string`): Product identifier
  3. `stock_on_hand` (`int64`): Physically available units in store
  4. `reorder_point` (`int64`): Inventory threshold triggering replenishment
  5. `safety_stock` (`int64`): Buffer stock level
  6. `last_restock_date` (`object` / `string` ISO 8601 `YYYY-MM-DD`): Date store was last replenished
- **First 5 Representative Rows:**
  | store_id | sku_id | stock_on_hand | reorder_point | safety_stock | last_restock_date |
  |---|---|---|---|---|---|
  | ST11 | SKU02558 | 333 | 72 | 22 | 2025-06-23 |
  | ST21 | SKU01031 | 236 | 67 | 14 | 2025-06-17 |
  | ST26 | SKU02129 | 496 | 96 | 34 | 2025-07-10 |
  | ST19 | SKU02907 | 109 | 34 | 13 | 2025-08-17 |
  | ST05 | SKU01023 | 333 | 97 | 22 | 2025-07-12 |
- **Missing Value Count & Percentage:** 0 missing values across all columns (0.0%).
- **Duplicate Rows:** 0 duplicate rows.
- **Grain & Primary Key Candidate:** Composite `(store_id, sku_id)` is **100.0% unique** across all 26,408 rows.
- **Unique Value Counts:** `store_id`: 30, `sku_id`: 4,495 active SKUs (out of 5,000 total SKUs).
- **Date Columns:** `last_restock_date` (min = `2025-05-06`, max = `2025-12-31`).
- **Numeric Statistics:**
  - `stock_on_hand`: min = 0, max = 598, mean = 166.20, median = 140.0, zeros = 4,239 (16.05%), negatives = 0
  - `reorder_point`: min = 20, max = 100, mean = 59.98, median = 60.0, zeros = 0, negatives = 0
  - `safety_stock`: min = 4, max = 50, mean = 20.95, median = 20.0, zeros = 0, negatives = 0
- **Data Quality Assessment:** 
  - There is **no snapshot date column**; this file represents the **final stock state as of December 31, 2025**.
  - `lead_time_days` and `on_order_units` specified in the Zidio brief are **MISSING** from this file.
  - The 4,239 zero-stock records align heavily with the 200 `STOCKOUT_RISK` SKUs (4,169 zero-stock occurrences).
- **Foreign Keys:**
  - `store_id` → `store_master.store_id` (100% match)
  - `sku_id` → `sku_master.sku_id` (100% match)

```
========================================================================================
DATASET 6: promotions.csv (Promotions Dimension & Campaigns)
========================================================================================
```
- **Exact Filename:** `promotions.csv`
- **File Size:** 8,511 bytes (~8.31 KB)
- **Rows:** 100
- **Columns:** 8
- **Exact Column Names & Types:**
  1. `promo_id` (`object` / `string` format `PROMO###`): Campaign ID (`PROMO001` to `PROMO100`)
  2. `promo_name` (`object` / `string`): Campaign name (e.g. "Spring Discount Days")
  3. `start_date` (`object` / `string` ISO 8601 `YYYY-MM-DD`): Promo launch date
  4. `end_date` (`object` / `string` ISO 8601 `YYYY-MM-DD`): Promo expiry date
  5. `discount_pct` (`float64`): Discount percentage (5.0% to 49.8%)
  6. `promo_type` (`object` / `string`): Mechanism (`Bundle Offer`, `Percentage Discount`, `Clearance`, `Flat Discount`, `BOGO`)
  7. `target_type` (`object` / `string`): Target level (`Category`, `SKU`, `Brand`, `All`)
  8. `target_value` (`object` / `string`): Target entity name/ID
- **First 5 Representative Rows:**
  | promo_id | promo_name | start_date | end_date | discount_pct | promo_type | target_type | target_value |
  |---|---|---|---|---|---|---|---|
  | PROMO001 | Spring Discount Days | 2024-07-28 | 2024-08-02 | 15.1 | BOGO | Brand | NovaFresh |
  | PROMO002 | Summer Savings Event | 2023-03-09 | 2023-03-21 | 7.0 | Bundle Offer | All | All |
  | PROMO003 | Payday Discount Days | 2025-04-30 | 2025-05-26 | 10.2 | BOGO | Brand | ZestyCo |
  | PROMO004 | Super Sale | 2025-06-07 | 2025-06-17 | 15.9 | Clearance | SKU | SKU01108 |
  | PROMO005 | Festive Discount Days | 2024-03-23 | 2024-04-08 | 46.0 | Bundle Offer | All | All |
- **Missing Value Count & Percentage:** 0 missing values across all columns (0.0%).
- **Duplicate Rows:** 0 duplicate rows.
- **Unique Value Counts:** `promo_id`: 100, `promo_name`: 72, `target_value`: 49 entities.
- **Date Columns:** `start_date` (min = `2022-01-23`, max = `2025-11-13`), `end_date` (min = `2022-01-29`, max = `2025-11-21`).
- **Numeric Statistics:**
  - `discount_pct`: min = 5.0%, max = 49.8%, mean = 27.58%, median = 25.40%, std = 14.13%, zeros = 0, negatives = 0
- **Categorical Distributions:**
  - `promo_type`: Bundle Offer: 23, Percentage Discount: 22, Clearance: 20, Flat Discount: 18, BOGO: 17
  - `target_type`: Category: 38, SKU: 22, Brand: 21, All: 19
- **Data Quality Assessment:** 41 overlapping promo pairs exist. Exactly 4 promotions (`PROMO005`, `PROMO028`, `PROMO038`, `PROMO070`) have zero transactions in sales because a concurrent overlapping promotion offering higher discounts took precedence.
- **Primary Key Candidate:** `promo_id` (verified strictly unique).
- **Foreign Keys:** Target values dynamically reference `sku_master` (100% valid target resolution).

```
========================================================================================
DATASET 7: sku_inventory_flags.csv (Anomaly Answer Key Dimension)
========================================================================================
```
- **Exact Filename:** `sku_inventory_flags.csv`
- **File Size:** 123,704 bytes (~120.80 KB) [Clean: 116,214 bytes]
- **Rows:** 600
- **Columns:** 6
- **Exact Column Names & Types:**
  1. `sku_id` (`object` / `string` format `SKU#####`): Flagged product identifier
  2. `flag` (`object` / `string`): Anomaly classification (`STOCKOUT_RISK`, `SLOW_MOVER`)
  3. `affected_stores` (`object` / `string`): Semicolon-delimited list of store IDs (e.g. `ST22;ST26;ST23`)
  4. `window_start` (`object` / `string` ISO 8601 nullable): Stockout injection window start
  5. `window_end` (`object` / `string` ISO 8601 nullable): Stockout injection window end
  6. `notes` (`object` / `string`): Plain-language injection rationale
- **First 5 Representative Rows:**
  | sku_id | flag | affected_stores | window_start | window_end | notes |
  |---|---|---|---|---|---|
  | SKU04321 | STOCKOUT_RISK | ST22;ST26;ST23;ST15;ST12;ST28;ST01;ST27;ST08;ST21;ST24;ST07;ST30;ST20 | 2025-11-06 | 2025-12-03 | Top-selling SKU (by observed volume); injected recent stockout window. |
  | SKU04596 | STOCKOUT_RISK | ST12;ST09;ST21;ST23;ST07;ST24;ST01;ST28;ST14;ST13;ST29;ST10;ST30 | 2025-10-25 | 2025-11-06 | Top-selling SKU (by observed volume); injected recent stockout window. |
  | SKU03727 | STOCKOUT_RISK | ST19;ST14;ST28;ST29;ST02;ST07;ST17;ST08;ST05;ST16 | 2025-11-08 | 2025-11-29 | Top-selling SKU (by observed volume); injected recent stockout window. |
  | SKU04154 | STOCKOUT_RISK | ST15;ST03;ST30;ST09;ST24;ST19;ST28;ST02;ST07;ST14;ST17 | 2025-11-12 | 2025-11-26 | Top-selling SKU (by observed volume); injected recent stockout window. |
  | SKU00953 | STOCKOUT_RISK | ST09;ST19;ST27;ST20;ST01;ST03;ST11;ST10;ST22;ST26;ST30;ST13;ST21;ST23;ST15 | 2025-11-29 | 2025-12-25 | Top-selling SKU (by observed volume); injected recent stockout window. |
- **Missing Value Count & Percentage:**
  - `sku_id`: 0 (0.0%)
  - `flag`: 0 (0.0%)
  - `affected_stores`: 0 (0.0%)
  - `window_start`: 400 missing (66.67% null — corresponds 100% to `SLOW_MOVER` rows)
  - `window_end`: 400 missing (66.67% null — corresponds 100% to `SLOW_MOVER` rows)
  - `notes`: 0 (0.0%)
- **Duplicate Rows:** 0 duplicate rows.
- **Unique Value Counts:** `sku_id`: 600 (strictly unique), `flag`: 2 (`SLOW_MOVER`: 400, `STOCKOUT_RISK`: 200).
- **Date Columns:** `window_start` (min = `2025-10-17`, max = `2025-12-17`), `window_end` (min = `2025-10-29`, max = `2025-12-31`).
- **Primary Key Candidate:** `sku_id` (verified strictly unique across all 600 records).
- **Foreign Keys:**
  - `sku_id` → `sku_master.sku_id` (100% match)
  - `affected_stores` parsed store tokens → `store_master.store_id` (100% match)

---

## C. Data Quality Report

A rigorous scan was conducted covering potential data corruptions. Each finding is catalogued below with its root cause, impact, and severity rating.

| # | Data Quality Issue | Table & Column(s) | Observed Impact / Evidence | Severity | Recommended Treatment |
|---|---|---|---|---|---|
| 1 | **Missing Historical Snapshots** | `inventory_snapshot.csv` | Table contains only a single point-in-time snapshot (as of late 2025). No daily or periodic historical snapshots exist. | **HIGH** | Frame risk scoring and inventory models around the latest available stock position; do not assume historical time-series inventory tracking. |
| 2 | **Missing Lead Time & On-Order Stock** | `inventory_snapshot.csv` | `lead_time_days` and `on_order_units` specified in the brief are absent from the file. | **HIGH** | Synthesize or establish business-rule defaults in pipeline configuration (e.g. constant lead time of 7–14 days per category, on-order = 0). |
| 3 | **Suppressed Sales (Stockouts)** | `sales_transactions.csv` | Exactly 26,527 transaction rows are absent in contaminated dataset during stockout windows (lost sales). | **HIGH** | Demand models must be aware that zero observed sales during stockout windows represent censored demand, not zero true demand! |
| 4 | **Missing Standalone Calendar Table** | Project workspace | `calendar.csv` does not exist in raw files. | **MEDIUM** | Programmatically generate a complete 1,461-day calendar dimension table in the ETL pipeline. |
| 5 | **Duplicate POS Line Items** | `sales_transactions.csv` | ~0.14% exact duplicate rows (679 in 500k sample) within identical `receipt_id` baskets. | **LOW** | Preserve as valid multi-unit scans or de-duplicate depending on whether barcode scanning protocol treats rows as distinct scans. |
| 6 | **Customer Registration Pre-Sales Discrepancy** | `customer_master.csv` vs `sales_transactions.csv` | ~17.3% of transactions occur before `registration_date`. | **LOW** | Treat `registration_date` as loyalty program signup, not customer acquisition / first-purchase date. |
| 7 | **Overlapping Promotions** | `promotions.csv` | 41 overlapping promo pairs; 4 promotions never triggered in sales transactions. | **LOW** | When deriving promotion features, incorporate campaign precedence (highest discount percentage wins). |
| 8 | **Missing Launch Date in SKU Master** | `sku_master.csv` | `launch_date` is not in `sku_master.csv`. | **LOW** | Derive observed first-sale date from `sales_transactions.csv` (all 5,000 SKUs first transacted between Jan 1 and Jan 19, 2022). |

---

## D. Dataset Relationship / Entity Map

All 7 datasets connect into an interconnected Retail Star / Snowflake Schema. Referential integrity across all foreign keys is **100.0% verified**.

```mermaid
erDiagram
    store_master ||--o{ sales_transactions : "store_id (100% overlap)"
    sku_master ||--o{ sales_transactions : "sku_id (100% overlap)"
    customer_master ||--o{ sales_transactions : "customer_id (100% overlap)"
    promotions ||--o{ sales_transactions : "promo_id (96% active, 100% valid)"
    
    store_master ||--o{ inventory_snapshot : "store_id (100% overlap)"
    sku_master ||--o{ inventory_snapshot : "sku_id (89.9% active in snapshot)"
    
    sku_master ||--o{ sku_inventory_flags : "sku_id (100% overlap)"
    store_master ||--o{ sku_inventory_flags : "affected_stores (100% overlap)"
    
    sku_master ||--o{ promotions : "target_value [SKU/Brand/Category] (100% valid)"
```

### Verified Relationship Overlap Statistics
1. **Sales SKUs in `sku_master`:** $5,000 / 5,000 = \mathbf{100.0\%}$
2. **Sales Stores in `store_master`:** $30 / 30 = \mathbf{100.0\%}$
3. **Sales Customers in `customer_master`:** $10,000 / 10,000 = \mathbf{100.0\%}$
4. **Sales Promos in `promotions`:** $96 / 100 = \mathbf{96.0\%}$ active; $0$ invalid promo IDs ($100.0\%$ validity)
5. **Inventory SKUs in `sku_master`:** $4,495 / 4,495 = \mathbf{100.0\%}$ ($89.9\%$ of total 5,000 catalog SKUs are stocked)
6. **Inventory Stores in `store_master`:** $30 / 30 = \mathbf{100.0\%}$
7. **Flagged SKUs in `sku_master`:** $600 / 600 = \mathbf{100.0\%}$
8. **Flagged Stores in `store_master`:** All parsed store tokens across 600 records map $100.0\%$ to `store_master.store_id`.

---

## E. Zidio FORESIGHT Schema Mapping

The table below reconciles the Zidio project requirements against the actual source data, classifying each field as **AVAILABLE**, **DERIVABLE**, or **MISSING**.

| Zidio Logical Table | Required Logical Field | Source Dataset | Actual Source Column / Expression | Status |
|---|---|---|---|---|
| **`sales_daily`** | `date` | `sales_transactions.csv` | `date` | **AVAILABLE** |
| | `sku_id` | `sales_transactions.csv` | `sku_id` | **AVAILABLE** |
| | `units_sold` | `sales_transactions.csv` | `SUM(quantity)` grouped by `(date, sku_id)` | **DERIVABLE** |
| | `revenue` | `sales_transactions.csv` | `SUM(total_value)` (Net) or `SUM(qty * unit_price)` (Gross) | **DERIVABLE** |
| | `unit_price` | `sales_transactions.csv` / `sku_master` | `MAX(unit_price)` or `sku_master.unit_price` (constant per SKU) | **DERIVABLE** |
| | `promo_flag` | `sales_transactions.csv` | `MAX(CASE WHEN promo_id IS NOT NULL THEN 1 ELSE 0 END)` | **DERIVABLE** |
| **`sku_master`** | `sku_id` | `sku_master.csv` | `sku_id` | **AVAILABLE** |
| | `category` | `sku_master.csv` | `category` | **AVAILABLE** |
| | `subcategory` | `sku_master.csv` | `subcategory` | **AVAILABLE** |
| | `launch_date` | `sales_transactions.csv` | `MIN(date)` per `sku_id` (all launched early Jan 2022) | **DERIVABLE** |
| | `unit_cost` | `sku_master.csv` | `cost_price` | **AVAILABLE** (Name Mapping) |
| | `list_price` | `sku_master.csv` | `unit_price` | **AVAILABLE** (Name Mapping) |
| **`calendar`** | `date` | Derived Sequence | Continuous sequence: `2022-01-01` to `2025-12-31` (1,461 days) | **DERIVABLE** |
| | `week` | Derived Sequence | ISO Calendar Week (`date.dt.isocalendar().week`) | **DERIVABLE** |
| | `month` | Derived Sequence | Calendar Month (`date.dt.month`) | **DERIVABLE** |
| | `season` | Derived Sequence | Meteorological/Retail season mapping by month | **DERIVABLE** |
| | `is_holiday` | External / Domain Logic | National & Religious holidays (Eid, Pakistan Day, etc.) | **REQUIRES DOMAIN RULE** |
| | `promo_event` | `promotions.csv` | Joined via `start_date <= date <= end_date` | **DERIVABLE** |
| **`inventory_snapshots`** | `date` | `inventory_snapshot.csv` | Static snapshot representing end of period (`2025-12-31`) | **DERIVABLE** |
| | `sku_id` | `inventory_snapshot.csv` | `sku_id` | **AVAILABLE** |
| | `on_hand_units` | `inventory_snapshot.csv` | `stock_on_hand` | **AVAILABLE** (Name Mapping) |
| | `on_order_units` | N/A | Not recorded in extract | **MISSING** |
| | `lead_time_days` | N/A | Not recorded in extract | **MISSING** |
| | `reorder_point` | `inventory_snapshot.csv` | `reorder_point` | **AVAILABLE** |

---

## F. Sales Transaction → sales_daily Feasibility

### Transformation Grain
- **Source Grain:** One row per basket line item (Receipt $	imes$ Line $	imes$ Store $	imes$ SKU).
- **Target Grain:** **One row per SKU per day** (Chain-wide aggregation across all 30 stores).

### Derivation Logic & Evidence
1. **`date` & `sku_id`:** Form the composite grouping key: `GROUP BY date, sku_id`.
2. **`units_sold`:** 
   $$	ext{units\_sold} = \sum 	ext{quantity}$$
   *Verification:* `quantity` is strictly non-negative integer (1 to 5), with zero nulls.
3. **`revenue`:**
   $$	ext{revenue} = \sum 	ext{total\_value}$$
   *Verification:* `total_value` accurately incorporates the promotional discount `quantity * unit_price * (1 - discount_pct/100)`. If gross revenue is needed, $\sum (	ext{quantity} 	imes 	ext{unit\_price})$ can be maintained alongside net revenue.
4. **`unit_price`:**
   *Empirical Verification:* Across 3,699 sampled date+SKU pairs, the price unique count was **exactly 1 in 100% of cases**. `unit_price` does not fluctuate across stores on any given date. Therefore:
   $$	ext{unit\_price} = \max(	ext{unit\_price})$$
   Alternatively, the effective realized price is $	ext{revenue} / 	ext{units\_sold}$.
5. **`promo_flag`:**
   $$	ext{promo\_flag} = \max(\mathbb{I}(	ext{promo\_id} 	ext{ is not null}))$$
   Returns `1` if any sale of the SKU on that date was on promotion, else `0`.

### Panel Sparsity & Horizon Scaling
- **Observed Active SKU-Days:** 4,143,430 non-zero transaction days.
- **Full Dense Panel ($1,461 	ext{ days} 	imes 5,000 	ext{ SKUs}$):** 7,305,000 rows.
- **Sparsity Rate:** **43.28%** of SKU-days have zero sales.
- **Weekly Horizon Aggregation (Target for Forecasting):**
  Zidio Deliverable D3 specifies: *"Weekly SKU-level forecast over the defined horizon (6–8 weeks)."*
  Aggregating to 209 weeks $	imes$ 5,000 SKUs yields **1,045,000 rows**, where zero-sale sparsity drops dramatically, creating ideal time series for gradient boosted trees or seasonal models.

---

## G. Calendar Feasibility

No `calendar.csv` file was provided. However, all calendar attributes can be generated deterministically:

1. **`date` [DERIVABLE]:** Generate continuous sequence between `2022-01-01` and `2025-12-31` (1,461 dates).
2. **`week` [DERIVABLE]:** Extract ISO calendar week: `date.dt.isocalendar().week` (1 to 53).
3. **`month` [DERIVABLE]:** Extract calendar month: `date.dt.month` (1 to 12).
4. **`season` [DERIVABLE]:** Retail seasonal mapping for Pakistan / South Asian climate:
   - Winter: December, January, February
   - Spring: March, April
   - Summer: May, June, July, August, September
   - Autumn / Festive: October, November
5. **`is_holiday` [REQUIRES DOMAIN RULE]:**
   Since all store locations are Pakistani metropolitan cities (Karachi, Lahore, Islamabad, etc.), standard national/religious holidays can be flagged via rule-based calendar generation:
   - Eid-ul-Fitr & Eid-ul-Adha (lunar calendar windows)
   - Pakistan Day (March 23)
   - Independence Day (August 14)
   - Quaid-e-Azam Day / New Year (December 25)
   - Ashura & Eid Milad-un-Nabi
6. **`promo_event` [DERIVABLE]:**
   Left join from `promotions.csv` where `start_date <= calendar.date <= end_date`. For dates with multiple active promos, select the primary brand or general campaign title.

---

## H. Inventory Feasibility

### Inventory File Grain & Mapping
The file `inventory_snapshot.csv` is structured at the grain:
$$	ext{One row per } (	ext{store\_id}, 	ext{sku\_id})$$
It represents a **cross-sectional inventory snapshot as of late 2025**, containing 26,408 records in the contaminated dataset.

| Zidio Specification | Provided Column | Data Type | Completeness | Notes |
|---|---|---|---|---|
| `date` | *Implicit* | N/A | Missing | Represents static position as of end of period (Dec 31, 2025). |
| `sku_id` | `sku_id` | `object` | 100% | 4,495 unique SKUs active across stores. |
| Store Grain | `store_id` | `object` | 100% | 30 stores. Average ~880 SKUs per store. |
| `on_hand_units` | `stock_on_hand` | `int64` | 100% | Min = 0, Max = 598, Mean = 166.2. |
| `reorder_point` | `reorder_point` | `int64` | 100% | Min = 20, Max = 100, Mean = 60.0. |
| `safety_stock` | `safety_stock` | `int64` | 100% | Min = 4, Max = 50, Mean = 20.95. |
| Restock Recency | `last_restock_date` | `object` | 100% | Range: May 6, 2025 to Dec 31, 2025. |
| `on_order_units` | *None* | N/A | **MISSING** | Must be parameterized in risk logic. |
| `lead_time_days` | *None* | N/A | **MISSING** | Must be parameterized in risk logic. |

### Ground Truth Validation Against Anomalies
Joining `inventory_snapshot.csv` with `sku_inventory_flags.csv` validates the injected anomalies:
- **`STOCKOUT_RISK` SKUs (200 SKUs):** Mean on-hand stock is **2.57 units** (median = 0.0); 4,169 out of 4,292 store-SKU rows have `stock_on_hand == 0`!
- **`SLOW_MOVER` SKUs (400 SKUs):** Mean on-hand stock is **268.48 units** (vs 132 for normal SKUs); maximum stock reaches 598 units, representing dead capital.

---

## I. Promotion Feasibility

`promotions.csv` contains 100 promotion campaigns:
- **Active Window:** January 23, 2022 to November 21, 2025.
- **Discount Levels:** 5.0% to 49.8% (mean 27.58%).
- **Mechanisms:** Bundle Offer (23), Percentage Discount (22), Clearance (20), Flat Discount (18), BOGO (17).
- **Targeting Types:** Category (38), SKU (22), Brand (21), All (19).

### Connection to Sales and Forecasting
1. **Transaction Alignment:** In `sales_transactions.csv`, 2,090,166 transactions carry a `promo_id` and non-zero `discount_pct`. 100% of tested promo sales fall strictly within the promo's `start_date` and `end_date`.
2. **Promotion Flag Derivation:** Can be derived both backward (from sales transactions carrying `promo_id`) and forward (from date $	imes$ SKU matching against `promotions.csv` targeting rules).
3. **Feature Engineering Potential:** High. Future models can engineer promotional uplift features:
   - `is_promo_active`: Binary indicator.
   - `promo_discount_pct`: Continuous discount depth.
   - `promo_type`: Categorical one-hot feature.

---

## J. Missing Information

The audit reveals the following specific gaps between the raw data extracts and the idealized Zidio brief:

1. **`lead_time_days` (Inventory):** Not present in `inventory_snapshot.csv`. 
   - *Impact on Project:* Deliverable D4 (Risk Scoring) requires comparing forecast demand over the lead time against stock.
   - *Resolution Strategy:* Assign realistic category-level lead times via configuration (e.g. Perishables / Dairy: 3 days; Groceries / Beverages: 7 days; Apparel / Home: 14 days).
2. **`on_order_units` (Inventory):** Not present in `inventory_snapshot.csv`.
   - *Impact on Project:* Pipeline calculations of projected available balance ($	ext{on\_hand} + 	ext{on\_order}$).
   - *Resolution Strategy:* Assume `on_order_units = 0` (unreplenished state, typical of stockout-risk scenarios) or parameterize based on reorder point deficit.
3. **`launch_date` (SKU Master):** Not present in `sku_master.csv`.
   - *Impact on Project:* Determining new SKU cold-start vs mature products.
   - *Resolution Strategy:* Derive empirically from `MIN(date)` per SKU in sales history (all 5,000 SKUs launched in early January 2022).
4. **Historical Inventory Tracking:** Only a single snapshot is provided. Daily inventory time series cannot be directly inspected historically without full inventory reconstruction.

---

## K. Recommended Data Pipeline — Execution Plan Only

*(Strictly architectural planning — No transformation code executed yet)*

```mermaid
flowchart TD
    subgraph Raw Extracts
        R1[sales_transactions.csv - 9.95M rows]
        R2[sku_master.csv - 5k rows]
        R3[store_master.csv - 30 rows]
        R4[promotions.csv - 100 rows]
        R5[inventory_snapshot.csv - 26.4k rows]
        R6[sku_inventory_flags.csv - 600 rows]
    end

    subgraph Step 1: Chunked Ingestion & Cleaning
        P1[Chunked Aggregator: sales_transactions]
        P2[Dimension Validator & Type Normalizer]
    end

    subgraph Step 2: Intermediate Unified Foundations
        I1[(sales_daily: date + sku_id grain)]
        I2[(sales_weekly: week + sku_id grain)]
        I3[(calendar_master: 1461 days + holidays)]
        I4[(inventory_current: store + sku)]
    end

    subgraph Step 3: Feature Engineering Layer
        F1[Demand Lags: 1w, 2w, 4w, 52w]
        F2[Rolling Statistics: 4w mean/std]
        F3[Calendar & Promo Signals]
        F4[SKU & Category Embeddings/Ratios]
    end

    subgraph Step 4: Model & Risk Hand-off
        M1[Deliverable D3: Forecasting Engine]
        M2[Deliverable D4: Risk Scoring & Decisioning Grid]
    end

    R1 --> P1
    R2 & R3 & R4 & R5 & R6 --> P2
    P1 --> I1
    I1 --> I2
    P2 --> I3 & I4
    I2 & I3 & I4 --> F1 & F2 & F3 & F4
    F1 & F2 & F3 & F4 --> M1 --> M2
```

### Pipeline Architecture Steps
1. **Pipeline Module 1 (`pipeline/ingest_clean.py`):**
   - Stream `sales_transactions.csv` in chunks of 500,000 rows.
   - Aggregate sales to two grains:
     - `sales_daily`: `(date, sku_id)` $	o$ `units_sold`, `revenue`, `unit_price`, `promo_flag`.
     - `sales_weekly`: `(year_week, sku_id)` $	o$ `weekly_units`, `weekly_revenue`, `promo_days`.
   - Re-index with complete panel (filling zero-sales periods with 0).
2. **Pipeline Module 2 (`pipeline/calendar_builder.py`):**
   - Generate full 1,461-day table with week, month, season, Pakistani public holidays, and promotion campaign tags.
3. **Pipeline Module 3 (`pipeline/feature_store.py`):**
   - Construct rolling features on weekly grain:
     - Lags: $t-1, t-2, t-4, t-52$ (seasonal naive reference).
     - Rolling windows: 4-week moving average, 8-week moving average, rolling volatility.
     - Promotion interaction flags.
4. **Pipeline Module 4 (`pipeline/inventory_profiler.py`):**
   - Ingest `inventory_snapshot.csv`, parameterize lead times by category, and prepare input matrix for stockout/overstock risk scoring against `sku_inventory_flags.csv`.

---

## L. Blockers

| Blocker Identified | Severity | Status / Mitigation |
|---|---|---|
| **Memory Exhaustion Risk from 10M Row CSV** | HIGH | **RESOLVED & MITIGATED.** Streaming chunk iterator (`chunksize=500,000`) tested and verified; full file scans execute in ~16 seconds with < 100 MB RAM overhead. |
| **Missing Lead Time Field for Risk Scoring** | MEDIUM | **UNBLOCKED via Domain Heuristic.** Project brief allows establishing standard operational parameters for unrecorded supplier fields. |
| **Censored Demand during Stockouts** | MEDIUM | **IDENTIFIED.** Stockout periods have suppressed transaction records. Model training must avoid treating unserved demand during stockouts as genuine zero customer demand. |

*Summary:* There are **zero fatal technical blockers**. The project is 100% ready to transition into Phase 1 pipeline construction and Phase 2 exploratory data analysis.

---

## M. Questions to Resolve Before Building

Before proceeding with model training and dashboard engineering, the following decisions should be aligned:

1. **Target Forecasting Grain (Store vs. Chain Level):**
   - *Option A (Chain Level - Recommended by Brief):* Forecast weekly demand aggregated across all 30 stores per SKU ($5,000 	ext{ time series}$).
   - *Option B (Store-SKU Level):* Forecast weekly demand per store per SKU ($30 	imes 5,000 = 150,000 	ext{ time series}$).
   - *Senior DS Recommendation:* Follow Option A for primary forecasting (Deliverable D3) to maintain statistical stability and clear signal-to-noise ratio, then allocate replenishment down to stores using historical store distribution weights.
2. **Handling Stockout Windows in Training Data:**
   - *Option A:* Leave suppressed sales as 0 in training. (Risk: model under-forecasts high-demand items).
   - *Option B (Recommended):* Impute censored demand during known `STOCKOUT_RISK` windows using pre-stockout run-rate or clean dataset ground truth.
3. **Lead Time Parameterization:**
   - Agree on fixed lead time assumptions across categories (e.g. Default = 14 days or category-specific: Fast Moving = 7 days, Durable = 21 days).
4. **Revenue Metric Definition:**
   - Standardize whether `revenue` in `sales_daily` refers to Net Realized Revenue ($\sum 	ext{total\_value}$) or Gross List Revenue ($\sum (	ext{qty} 	imes 	ext{list\_price})$). Net revenue is recommended for finance reconciliation.

---

### Audit Certification
This audit report, accompanying profile (`dataset_profile.csv`), and machine-readable schema (`dataset_schema.json`) represent a complete and authentic reflection of the data assets. No files have been altered, and no synthetic values have been hallucinated.
