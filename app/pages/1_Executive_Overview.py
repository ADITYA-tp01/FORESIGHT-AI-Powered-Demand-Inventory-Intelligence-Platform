"""Page 1 — Executive Overview: portfolio KPIs, trajectory, quadrant mix (Plan 7.2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _extra in (ROOT / "src", ROOT / "app"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import streamlit as st  # noqa: E402

from components import charts, kpi_cards, loaders  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Executive Overview", layout="wide")
    st.title("Executive Overview")

    meta = loaders.manifest()
    k = loaders.kpi_values()
    grid = loaders.decision_grid()

    kpi_cards.kpi_row(
        [
            {
                "label": "Total Active SKUs",
                "value": kpi_cards.format_units(k["active_skus"]),
                "help": f"{k['scored_skus']:.0f} SKUs carry inventory rows and are scored for risk",
            },
            {
                "label": "8-Week Forecast Demand",
                "value": kpi_cards.format_units(k["forecast_8w_units"]),
                "help": f"Units, {meta['forecast_weeks'][0]} .. {meta['forecast_weeks'][1]}",
            },
            {
                "label": "Stockout Rupee Exposure",
                "value": kpi_cards.format_inr(k["stockout_rupee_exposure"]),
                "help": "Rupees at risk: forecast demand beyond stock + on-order",
            },
            {
                "label": "Locked Capital (Dead Stock)",
                "value": kpi_cards.format_inr(k["locked_capital"]),
                "help": "Inventory value beyond 8-week demand + safety stock",
            },
            {
                "label": "Model Backtest WAPE",
                "value": kpi_cards.format_pct(k["ml_wape"]),
                "help": (
                    f"Baseline {kpi_cards.format_pct(k['baseline_wape'])} — "
                    f"{k['wape_reduction_pct']:.1f}% relative reduction (rolling CV)"
                ),
            },
        ]
    )

    history = (
        loaders.history_weekly().groupby("year_week", as_index=False)["units_sold"].sum()
    )
    forecast = loaders.forecast_summary()
    forecast_totals = forecast.groupby("year_week", as_index=False)[
        ["forecast", "baseline"]
    ].sum()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Portfolio demand trajectory")
        st.plotly_chart(
            charts.trajectory_chart(
                history,
                forecast_totals,
                int(loaders.settings().dashboard["trajectory_history_weeks"]),
            )
        )
    with right:
        st.subheader("Quadrant distribution")
        st.plotly_chart(
            charts.quadrant_pie(grid["quadrant"].value_counts().to_dict())
        )

    st.subheader("Stockout rupee exposure by category")
    exposure = loaders.category_metrics("rupees_at_risk").reset_index()
    exposure.columns = ["category", "value"]
    st.plotly_chart(charts.category_risk_bar(exposure, "Rupees at risk (INR)"))

    st.caption(
        f"Serving layer generated {meta['generated_at']} · forecast "
        f"{meta['forecast_weeks'][0]} .. {meta['forecast_weeks'][1]} · "
        f"{meta['scored_skus']} of {meta['active_skus']} SKUs scored"
    )


if __name__ == "__main__":
    main()
