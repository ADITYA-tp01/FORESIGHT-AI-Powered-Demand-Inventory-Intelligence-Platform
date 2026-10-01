"""Retail calendar, South Asian seasons, Pakistani holidays."""

from __future__ import annotations

import pandas as pd
from data.external.pakistan_holidays import (  # type: ignore[import-untyped]
    resolved_holiday_name,
)

SEASON_BY_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Summer",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Summer",
    10: "Autumn/Festive",
    11: "Autumn/Festive",
}


def season_for_month(month: int) -> str:
    return SEASON_BY_MONTH[month]


def build_calendar(start: str = "2022-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for d in dates:
        iso = d.isocalendar()
        year_week = f"{iso.year:04d}-W{iso.week:02d}"
        holiday_name = resolved_holiday_name(d.month, d.day, d.year)
        rows.append(
            {
                "date": d,
                "year": d.year,
                "month": d.month,
                "quarter": (d.month - 1) // 3 + 1,
                "iso_week": iso.week,
                "year_week": year_week,
                "day_of_week": d.weekday(),
                "is_weekend": d.weekday() >= 5,
                "season": season_for_month(d.month),
                "is_holiday": bool(holiday_name),
                "holiday_name": holiday_name,
            }
        )
    return pd.DataFrame(rows)
