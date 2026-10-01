"""Schema validation contracts."""

from __future__ import annotations

import pandas as pd


def assert_no_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    missing = df[columns].isnull().sum()
    bad = missing[missing > 0]
    if not bad.empty:
        raise ValueError(f"Nulls in mandatory columns: {bad.to_dict()}")


def assert_positive_strict(df: pd.DataFrame, column: str) -> None:
    if (df[column] <= 0).any():
        raise ValueError(f"{column} must be strictly positive")
