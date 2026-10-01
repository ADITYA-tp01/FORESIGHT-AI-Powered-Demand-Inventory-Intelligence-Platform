"""Page 2 — Forecast Explorer: history vs ML forecast, intervals, promos (Plan 7.2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT / "src", ROOT / "app"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import streamlit as st  # noqa: E402

from components import charts, loaders  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Forecast Explorer", layout="wide")
    st.title("Forecast Explorer")

    grid = loaders.decision_grid()
    dashboard_cfg = loaders.settings().dashboard

    category = st.selectbox("Category", sorted(grid["category"].unique()))
    in_category = grid.loc[grid["category"] == category]

    col1, col2 = st.columns(2)
    subcategories = col1.multiselect(
        "Subcategory", sorted(in_category["subcategory"].unique())
    )
    brands = col2.multiselect("Brand", sorted(in_category["brand"].unique()))

    candidates = in_category
    if subcategories:
        candidates = candidates.loc[candidates["subcategory"].isin(subcategories)]
    if brands:
        candidates = candidates.loc[candidates["brand"].isin(brands)]

    names = grid.set_index("sku_id")["sku_name"]
    max_skus = int(dashboard_cfg["max_explorer_skus"])
    selected = st.multiselect(
        f"SKU (up to {max_skus})",
        options=sorted(candidates["sku_id"]),
        format_func=lambda sku: f"{sku} — {names[sku]}",
        max_selections=max_skus,
    )

    if not selected:
        st.info("Select at least one SKU to plot history, forecast, intervals and promos.")
        return

    history = loaders.category_history(category)
    history = history.loc[history["sku_id"].isin(selected)]
    forecast = loaders.forecast_summary()
    forecast = forecast.loc[forecast["sku_id"].isin(selected)]
    promos = loaders.promo_windows()
    promos = promos.loc[promos["sku_id"].isin(selected)]

    st.plotly_chart(charts.sku_forecast_chart(history, forecast, promos))
    st.caption(
        "Gold bands = active promotional windows. The 80% band is the sum of "
        "per-SKU intervals (envelope when multiple SKUs are selected). "
        "Baseline = seasonal-naive with 3-tier fallback."
    )


if __name__ == "__main__":
    main()
