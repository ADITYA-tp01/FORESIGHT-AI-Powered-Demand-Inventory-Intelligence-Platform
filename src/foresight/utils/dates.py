"""ISO week and retail calendar helpers (stubs used from Phase 1)."""

from __future__ import annotations

from datetime import date


def iso_year_week(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def advance_week(week_label: str, steps: int) -> list[str]:
    """Advance an ISO week label by `steps` weeks (assumes 52-week years)."""
    year, week = map(int, week_label.split("-W"))
    out = []
    for i in range(1, steps + 1):
        y, w = year, week + i
        while w > 52:
            w -= 52
            y += 1
        out.append(f"{y:04d}-W{w:02d}")
    return out
