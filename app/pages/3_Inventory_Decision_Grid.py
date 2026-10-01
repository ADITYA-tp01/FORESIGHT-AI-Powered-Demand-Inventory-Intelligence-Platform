"""Page 3 — Inventory Decision Grid: quadrant scatter + triage tables (Plan 7.2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT / "src", ROOT / "app"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import charts, loaders, tables  # noqa: E402

REORDER_COLUMNS = [
    "sku_id", "category", "stock_on_hand", "lead_time_days", "ltd",
    "stockout_risk", "rupees_at_risk",
]
MARKDOWN_COLUMNS = [
    "sku_id", "category", "stock_on_hand", "wos", "restock_age_days",
    "forward_8w_demand", "locked_capital",
]


def _present_quadrants(grid: pd.DataFrame) -> list[str]:
    present = set(grid.loc[grid["scored"], "quadrant"])
    return [q for q in charts.QUADRANT_ORDER if q in present]


def main() -> None:
    st.set_page_config(page_title="Inventory Decision Grid", layout="wide")
    st.title("Inventory Decision Grid")

    grid = loaders.decision_grid()
    quadrant_docs = loaders.settings().risk["quadrants"]

    col1, col2 = st.columns([2, 1])
    quadrants = col1.multiselect(
        "Quadrant",
        options=_present_quadrants(grid),
        default=_present_quadrants(grid),
    )
    categories = col2.multiselect(
        "Category", options=sorted(grid["category"].unique())
    )

    filtered = grid.loc[grid["quadrant"].isin(quadrants)]
    if categories:
        filtered = filtered.loc[filtered["category"].isin(categories)]

    scored = filtered.loc[filtered["scored"]]
    st.plotly_chart(charts.decision_scatter(scored))
    st.caption(
        "Y = stockout risk (0-1), X = weeks of supply (log scale; zero-cover SKUs "
        "plot at half the smallest positive WOS), marker size = total rupee impact "
        "(rupees at risk + locked capital). Hover for SKU detail."
    )

    reorder = (
        filtered.loc[filtered["quadrant"] == "reorder_now"]
        .sort_values("rupees_at_risk", ascending=False)
    )
    markdown = (
        filtered.loc[filtered["quadrant"] == "markdown_clear"]
        .sort_values("locked_capital", ascending=False)
    )

    tab_reorder, tab_markdown = st.tabs(
        [
            f"Immediate Reorders ({len(reorder)})",
            f"Markdown Candidates ({len(markdown)})",
        ]
    )
    with tab_reorder:
        st.caption(quadrant_docs.get("reorder_now", ""))
        display = reorder.loc[:, REORDER_COLUMNS].round({"stockout_risk": 3})
        tables.show_table(display)
        tables.csv_download_button(
            display, "reorder_candidates.csv", "Download CSV", key="dl_reorder"
        )
    with tab_markdown:
        st.caption(quadrant_docs.get("markdown_clear", ""))
        display = markdown.loc[:, MARKDOWN_COLUMNS].round({"wos": 1})
        tables.show_table(display)
        tables.csv_download_button(
            display, "markdown_candidates.csv", "Download CSV", key="dl_markdown"
        )


if __name__ == "__main__":
    main()
