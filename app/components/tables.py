"""Interactive tables with one-click CSV export (docs/Plan.md 7.2 Page 3)."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def show_table(frame: pd.DataFrame) -> None:
    st.dataframe(frame, hide_index=True)


def csv_download_button(frame: pd.DataFrame, filename: str, label: str, key: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )
