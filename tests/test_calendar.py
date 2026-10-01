"""Retail calendar generation."""

from __future__ import annotations

import pandas as pd


def test_calendar_covers_1461_days_and_tags_pakistan_day():
    from foresight.preprocessing.calendar import build_calendar

    cal = build_calendar("2022-01-01", "2025-12-31")
    assert len(cal) == 1461
    assert cal["date"].min() == pd.Timestamp("2022-01-01")
    assert cal["date"].max() == pd.Timestamp("2025-12-31")
    pakistan_day = cal.loc[cal["date"] == pd.Timestamp("2022-03-23")].iloc[0]
    assert bool(pakistan_day["is_holiday"]) is True
    assert pakistan_day["holiday_name"] == "Pakistan Day"
    assert set(["year", "month", "quarter", "iso_week", "year_week", "day_of_week", "is_weekend", "season"]).issubset(
        cal.columns
    )


def test_eid_ul_fitr_2025_is_tagged():
    from foresight.preprocessing.calendar import build_calendar

    cal = build_calendar("2025-01-01", "2025-12-31")
    eid = cal.loc[cal["date"] == pd.Timestamp("2025-03-31")].iloc[0]
    assert bool(eid["is_holiday"]) is True
    assert eid["holiday_name"] == "Eid-ul-Fitr"


def test_retail_season_mapping():
    from foresight.preprocessing.calendar import season_for_month

    assert season_for_month(1) == "Winter"
    assert season_for_month(4) == "Spring"
    assert season_for_month(8) == "Summer"
    assert season_for_month(11) == "Autumn/Festive"
