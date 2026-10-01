"""Page 4 — SKU Drilldown: telemetry, store allocation, per-SKU history (Plan 7.2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT / "src", ROOT / "app"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import charts, kpi_cards, loaders, tables  # noqa: E402


def _metric(value: object, render) -> str:
    return "—" if pd.isna(value) else render(value)


def main() -> None:
    st.set_page_config(page_title="SKU Drilldown", layout="wide")
    st.title("SKU Drilldown")

    grid = loaders.decision_grid()
    names = grid.set_index("sku_id")["sku_name"]
    sku = st.selectbox(
        "SKU", options=grid["sku_id"], format_func=lambda s: f"{s} — {names[s]}"
    )
    row = grid.loc[grid["sku_id"] == sku].iloc[0]
    scored = bool(row["scored"])

    st.subheader(f"{row['sku_name']} · {row['category']} · {row['subcategory']}")
    if not scored:
        st.warning(
            "No inventory snapshot rows for this SKU — risk scoring skipped "
            "(quadrant `not_scored`)."
        )

    row1 = st.columns(4)
    row1[0].metric("On-hand units", _metric(row["stock_on_hand"], kpi_cards.format_units))
    row1[1].metric("Lead time (days)", _metric(row["lead_time_days"], lambda v: int(v)))
    row1[2].metric("Reorder point", _metric(row["reorder_point"], lambda v: int(v)))
    row1[3].metric("Safety stock", _metric(row["safety_stock"], lambda v: int(v)))

    row2 = st.columns(4)
    row2[0].metric(
        "Last restock",
        _metric(row["last_restock_date"], lambda v: str(pd.Timestamp(v).date())),
    )
    row2[1].metric(
        "Restock age (days)", _metric(row["restock_age_days"], lambda v: int(v))
    )
    row2[2].metric("Weeks of supply", _metric(row["wos"], lambda v: f"{v:.1f}"))
    row2[3].metric(
        "Stockout risk",
        _metric(row["stockout_risk"], lambda v: f"{v * 100:.1f}%"),
    )

    action = loaders.settings().risk["quadrants"].get(
        str(row["quadrant"]),
        "No inventory data — excluded from risk scoring.",
    )
    st.info(f"**{row['quadrant']}** — {action}")

    row3 = st.columns(4)
    row3[0].metric(
        "LTD (lead-time demand)", _metric(row["ltd"], kpi_cards.format_units)
    )
    row3[1].metric(
        "8-week demand", _metric(row["forward_8w_demand"], kpi_cards.format_units)
    )
    row3[2].metric(
        "Rupees at risk", _metric(row["rupees_at_risk"], kpi_cards.format_inr)
    )
    row3[3].metric(
        "Locked capital", _metric(row["locked_capital"], kpi_cards.format_inr)
    )

    if scored:
        st.subheader("Store-level allocation")
        store_rows = loaders.store_risk()
        store_rows = store_rows.loc[store_rows["sku_id"] == sku]
        if store_rows.empty:
            st.caption("No store rows for this SKU in the inventory snapshot.")
        else:
            table = store_rows.loc[
                :,
                [
                    "store_id", "store_stock_on_hand", "share",
                    "store_ltd", "store_pab", "store_outage",
                ],
            ].round({"share": 3, "store_ltd": 1, "store_pab": 1})
            left, right = st.columns([1, 1])
            with left:
                tables.show_table(table)
            with right:
                st.plotly_chart(charts.store_allocation_bar(store_rows))

        st.subheader("Weekly demand history")
        history = loaders.category_history(str(row["category"]))
        history = history.loc[history["sku_id"] == sku, ["year_week", "units_sold"]]
        st.plotly_chart(charts.trajectory_chart(history, None, len(history)))


if __name__ == "__main__":
    main()
