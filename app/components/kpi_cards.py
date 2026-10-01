"""Styled KPI metric cards for the executive overview (docs/Plan.md 7.2)."""

from __future__ import annotations

from typing import Any

import streamlit as st


def format_inr(value: float) -> str:
    """Indian-grouped rupees: 32785603 -> 'INR 32,785,603'."""
    return f"INR {value:,.0f}"


def format_units(value: float) -> str:
    return f"{value:,.0f}"


def format_pct(fraction: float) -> str:
    """0.3081 -> '30.8%' (inputs are WAPE fractions)."""
    return f"{fraction * 100:.1f}%"


def kpi_row(cards: list[dict[str, Any]]) -> None:
    """Render metric cards across equal-width columns."""
    if not cards:
        return
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=True):
        column.metric(
            card["label"],
            card["value"],
            help=card.get("help"),
        )
