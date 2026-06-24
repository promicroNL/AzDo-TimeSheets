from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

REPORTING_HOUR_INCREMENT = Decimal("0.25")


def round_report_hours(hours: float) -> float:
    value = Decimal(str(hours))
    rounded_units = (value / REPORTING_HOUR_INCREMENT).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )
    return float(rounded_units * REPORTING_HOUR_INCREMENT)


def sum_report_hours(hours: Iterable[float]) -> float:
    return sum(round_report_hours(value) for value in hours)


def format_report_hours(hours: float) -> str:
    return f"{round_report_hours(hours):.2f}"
