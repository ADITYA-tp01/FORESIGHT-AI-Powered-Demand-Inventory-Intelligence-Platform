"""Pure Plotly chart builders — no Streamlit import, unit-testable (Plan 7.2)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

QUADRANT_COLORS = {
    "reorder_now": "#d62728",
    "watch_volatile": "#9467bd",
    "healthy": "#2ca02c",
    "markdown_clear": "#ff7f0e",
    "not_scored": "#7f7f7f",
}
QUADRANT_ORDER = ("reorder_now", "watch_volatile", "healthy", "markdown_clear", "not_scored")

# Presentation-only marker sizing for the decision scatter (px, area scale).
_SIZE_MAX = 48

_LAYOUT = dict(margin=dict(t=36, b=12, l=12, r=12))
_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)


def week_monday(week_label: str) -> pd.Timestamp:
    """ISO week label -> Monday date (stable time axis for charts)."""
    year, week = map(int, str(week_label).split("-W"))
    return pd.Timestamp(date.fromisocalendar(year, week, 1))


def trajectory_chart(
    history: pd.DataFrame, forecast_totals: pd.DataFrame | None, history_weeks: int
) -> go.Figure:
    """Portfolio demand trajectory: recent history vs 8-week forecast + baseline.

    `history`: columns year_week, units_sold (already chain-summed).
    `forecast_totals`: columns year_week, forecast, baseline.
    """
    hist = history.sort_values("year_week").tail(history_weeks)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["year_week"], y=hist["units_sold"], mode="lines",
            name="Historical demand", line=dict(color="#1f77b4"),
        )
    )
    if forecast_totals is not None and not forecast_totals.empty:
        frame = forecast_totals.sort_values("year_week")
        fig.add_trace(
            go.Scatter(
                x=frame["year_week"], y=frame["forecast"], mode="lines+markers",
                name="ML forecast (8w)", line=dict(color="#d62728", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["year_week"], y=frame["baseline"], mode="lines",
                name="Seasonal-naive baseline", line=dict(color="#7f7f7f", dash="dot"),
            )
        )
    fig.update_layout(
        hovermode="x unified", yaxis_title="Units / week",
        legend=_LEGEND, **_LAYOUT,
    )
    return fig


def quadrant_pie(counts: dict[str, int]) -> go.Figure:
    labels = [q for q in QUADRANT_ORDER if counts.get(q, 0) > 0]
    values = [int(counts[q]) for q in labels]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels, values=values,
                marker=dict(colors=[QUADRANT_COLORS[q] for q in labels]),
                hole=0.35, textinfo="label+value",
            )
        ]
    )
    fig.update_layout(showlegend=False, **_LAYOUT)
    return fig


def category_risk_bar(frame: pd.DataFrame, value_title: str) -> go.Figure:
    """Horizontal bar: `frame` has columns category, value (pre-sorted)."""
    fig = px.bar(
        frame, x="value", y="category", orientation="h",
        labels={"value": value_title, "category": ""},
        color_discrete_sequence=["#d62728"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, **_LAYOUT)
    return fig


def _promo_spans(promo_windows: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Union of active-promo weeks as merged Monday-date spans."""
    if promo_windows.empty:
        return []
    dates = sorted({week_monday(week) for week in promo_windows["year_week"]})
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = dates[0]
    for current in dates[1:]:
        if (current - previous).days <= 7:  # consecutive ISO weeks
            previous = current
        else:
            spans.append((start, previous))
            start = previous = current
    spans.append((start, previous))
    return spans


def sku_forecast_chart(
    history: pd.DataFrame, forecast: pd.DataFrame, promo_windows: pd.DataFrame
) -> go.Figure:
    """Single chart per selection: history, ML point, 80% band, baseline, promos.

    `history`: sku_id, year_week, units_sold (selected SKUs, summed to chain).
    `forecast`: forecast_summary rows for the selected SKUs.
    `promo_windows`: active promo weeks for the selected SKUs.
    The band is the sum of per-SKU 80% intervals (an envelope, stated in caption).
    """
    hist = (
        history.groupby("year_week", as_index=False)["units_sold"].sum()
        .sort_values("year_week")
    )
    metrics = ["forecast", "forecast_q10", "forecast_q90", "baseline"]
    frame = forecast.groupby("year_week", as_index=False)[metrics].sum().sort_values(
        "year_week"
    )
    hist["week_date"] = hist["year_week"].map(week_monday)
    frame["week_date"] = frame["year_week"].map(week_monday)

    fig = go.Figure()
    for start, end in _promo_spans(promo_windows):
        fig.add_vrect(
            x0=start, x1=end, fillcolor="gold", opacity=0.18, line_width=0
        )
    fig.add_trace(
        go.Scatter(
            x=hist["week_date"], y=hist["units_sold"], mode="lines",
            name="Historical sales", line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["week_date"], y=frame["forecast_q90"], mode="lines",
            name="80% interval (upper)", line=dict(width=0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["week_date"], y=frame["forecast_q10"], mode="lines",
            name="80% interval", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.20)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["week_date"], y=frame["forecast"], mode="lines+markers",
            name="ML forecast", line=dict(color="#d62728"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["week_date"], y=frame["baseline"], mode="lines",
            name="Seasonal-naive baseline", line=dict(color="#7f7f7f", dash="dot"),
        )
    )
    fig.update_layout(
        hovermode="x unified", xaxis_title="Week", yaxis_title="Units",
        legend=_LEGEND, **_LAYOUT,
    )
    return fig


def decision_scatter(grid: pd.DataFrame) -> go.Figure:
    """Stockout risk vs overstock cover, sized by rupee impact (Plan 7.2 Page 3).

    Zero-cover SKUs plot at half the smallest positive WOS so the log axis
    keeps them visible (stated in the page caption).
    """
    scored = grid.loc[grid["scored"]].copy()
    positives = scored.loc[scored["wos"] > 0, "wos"]
    floor = float(positives.min()) / 2 if len(positives) else 1.0
    scored["wos_plot"] = scored["wos"].clip(lower=floor)
    fig = px.scatter(
        scored,
        x="wos_plot",
        y="stockout_risk",
        color="quadrant",
        size="rupee_impact",
        hover_name="sku_id",
        hover_data={
            "wos_plot": False,
            "wos": ":.1f",
            "stockout_risk": ":.3f",
            "category": True,
            "stock_on_hand": True,
            "rupees_at_risk": ":,.0f",
            "locked_capital": ":,.0f",
        },
        color_discrete_map=QUADRANT_COLORS,
        log_x=True,
        size_max=_SIZE_MAX,
        labels={"wos_plot": "Weeks of supply (log scale)", "stockout_risk": "Stockout risk"},
    )
    fig.update_yaxes(range=[0, 1.05])
    fig.update_layout(hovermode="closest", legend=_LEGEND, **_LAYOUT)
    return fig


def store_allocation_bar(store_rows: pd.DataFrame) -> go.Figure:
    """On-hand units per store, red where the store is flagged as an outage."""
    frame = store_rows.sort_values("store_stock_on_hand", ascending=False)
    fig = px.bar(
        frame,
        x="store_id",
        y="store_stock_on_hand",
        color="store_outage",
        color_discrete_map={True: "#d62728", False: "#2ca02c"},
        labels={
            "store_id": "Store",
            "store_stock_on_hand": "Units on hand",
            "store_outage": "Outage alert",
        },
    )
    fig.update_layout(**_LAYOUT)
    return fig
