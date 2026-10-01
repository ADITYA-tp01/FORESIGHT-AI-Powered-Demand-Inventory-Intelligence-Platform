"""Ex-ante calendar features: Fourier seasonality + holiday flags.

Calendar features are known before the week happens, so they are allowed to
describe week `t` itself (unlike demand-derived features, which are shifted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAY_WEEK_COL = "is_holiday_week"
PROXIMITY_COL = "holiday_proximity"


def weekly_calendar_features(
    calendar: pd.DataFrame,
    proximity_days: int,
) -> pd.DataFrame:
    """Collapse the day-level calendar to one row per year_week.

    - month / iso_week: taken from the week's first available day
    - is_holiday_week: any day of the week is a holiday
    - holiday_proximity: any day within +/- proximity_days of a holiday
    """
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    prox = pd.Series(False, index=cal.index)
    holiday_dates = cal.loc[cal["is_holiday"], "date"]
    for offset in range(-int(proximity_days), int(proximity_days) + 1):
        near = holiday_dates + pd.Timedelta(days=int(offset))
        prox |= cal["date"].isin(set(near))
    cal["_holiday_proximity"] = prox

    weekly = cal.groupby("year_week", observed=True).agg(
        month=("month", "min"),
        iso_week=("iso_week", "min"),
        **{
            HOLIDAY_WEEK_COL: ("is_holiday", "max"),
            PROXIMITY_COL: ("_holiday_proximity", "max"),
        },
    )
    weekly = weekly.reset_index()
    weekly[HOLIDAY_WEEK_COL] = weekly[HOLIDAY_WEEK_COL].astype("int8")
    weekly[PROXIMITY_COL] = weekly[PROXIMITY_COL].astype("int8")
    return weekly


def build_calendar_features(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    week_period: int,
    month_period: int,
    proximity_days: int,
) -> pd.DataFrame:
    """Merge weekly calendar flags and add Fourier cyclical encodings."""
    weekly = weekly_calendar_features(calendar, proximity_days)
    missing = set(df["year_week"].unique()) - set(weekly["year_week"].unique())
    if missing:
        raise ValueError(
            f"Calendar has no rows for year_week values: {sorted(missing)}. "
            "Extend the calendar range before forecasting those weeks."
        )
    df = df.merge(weekly, on="year_week", how="left")

    week_angle = 2.0 * np.pi * df["iso_week"] / float(week_period)
    month_angle = 2.0 * np.pi * df["month"] / float(month_period)
    df["week_sin"] = np.sin(week_angle)
    df["week_cos"] = np.cos(week_angle)
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    return df
