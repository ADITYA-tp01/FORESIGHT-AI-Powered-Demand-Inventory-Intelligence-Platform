"""Pakistani public & retail holidays (2022–2025). Lunar dates are empirical, not computed."""

from __future__ import annotations

# Fixed Gregorian national holidays
FIXED_HOLIDAYS = {
    "03-23": "Pakistan Day",
    "05-01": "Labor Day",
    "08-14": "Independence Day",
    "11-09": "Iqbal Day",
    "12-25": "Quaid-e-Azam Day / Christmas",
}

# Empirical lunar Islamic holidays 2022–2025 (Pakistan observed dates).
# Source: documented domain heuristic — configs/features.yaml holiday_proximity_days.
LUNAR_HOLIDAYS = {
    # year: list of (YYYY-MM-DD, name)
    2022: [
        ("2022-05-03", "Eid-ul-Fitr"),
        ("2022-07-10", "Eid-ul-Adha"),
        ("2022-08-08", "Ashura"),
        ("2022-10-09", "Milad-un-Nabi"),
    ],
    2023: [
        ("2023-04-22", "Eid-ul-Fitr"),
        ("2023-06-29", "Eid-ul-Adha"),
        ("2023-07-28", "Ashura"),
        ("2023-09-28", "Milad-un-Nabi"),
    ],
    2024: [
        ("2024-04-10", "Eid-ul-Fitr"),
        ("2024-06-17", "Eid-ul-Adha"),
        ("2024-07-16", "Ashura"),
        ("2024-09-16", "Milad-un-Nabi"),
    ],
    2025: [
        ("2025-03-31", "Eid-ul-Fitr"),
        ("2025-06-07", "Eid-ul-Adha"),
        ("2025-07-06", "Ashura"),
        ("2025-09-05", "Milad-un-Nabi"),
    ],
}

_LUNAR_BY_MONTH_DAY = {
    f"{ld[5:7]}-{ld[8:10]}": name
    for _year, entries in LUNAR_HOLIDAYS.items()
    for ld, name in entries
}


def resolved_holiday_name(month: int, day: int, year: int = 0) -> str:
    key = f"{month:02d}-{day:02d}"
    fixed = FIXED_HOLIDAYS.get(key, "")
    lunar = _LUNAR_BY_MONTH_DAY.get(key, "")
    return lunar or fixed
