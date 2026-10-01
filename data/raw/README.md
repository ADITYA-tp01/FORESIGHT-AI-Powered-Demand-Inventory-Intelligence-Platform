# Data lake — raw

Do not copy the 801 MB `sales_transactions.csv` into this folder.

The operational extract lives at the sibling path:

    ../Dataset/retail_contaminated_dataset/
    ../Dataset/retail_clean_dataset/

`configs/config.yaml` `paths.dataset_root` and env `FORESIGHT_DATASET_ROOT` point there.
Phase 1 ingestion streams from those files into `data/interim` and `data/processed` as Parquet.
