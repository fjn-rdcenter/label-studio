"""Minimal 5-field cron ("minute hour day-of-month month day-of-week") for the long-term training schedule.

Supports `*`, numbers, lists (`1,3`), ranges (`1-5`) and steps (`*/15`, `0-30/10`). Day-of-week: 0 or 7 = Sunday.
Like cron, when both day fields are restricted a day matches if either matches. Times are local (container TZ).
"""
from __future__ import annotations

import datetime as dt

FIELDS = (("minute", 0, 59), ("hour", 0, 23), ("day", 1, 31), ("month", 1, 12), ("weekday", 0, 7))


def _parse_field(text: str, lo: int, hi: int) -> set[int]:
    values: set[int] = set()
    for part in text.split(","):
        base, _, step = part.partition("/")
        step_n = int(step) if step else 1
        if step_n < 1:
            raise ValueError(f"bad step in {part!r}")
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            start, end = (int(v) for v in base.split("-", 1))
        else:
            start = end = int(base)
            if step:
                end = hi
        if not (lo <= start <= end <= hi):
            raise ValueError(f"{part!r} is outside {lo}-{hi}")
        values.update(range(start, end + 1, step_n))
    return values


class Cron:
    def __init__(self, expr: str):
        parts = expr.split()
        if len(parts) != 5:
            raise ValueError(f"cron needs 5 fields (minute hour day month weekday), got {expr!r}")
        self.expr = expr
        sets = [_parse_field(p, lo, hi) for p, (_, lo, hi) in zip(parts, FIELDS)]
        self.minute, self.hour, self.day, self.month, weekday = sets
        self.weekday = {d % 7 for d in weekday}  # 7 = Sunday
        self.day_any, self.weekday_any = parts[2] == "*", parts[4] == "*"

    def date_matches(self, d: dt.date) -> bool:
        day_ok = d.day in self.day
        weekday_ok = (d.weekday() + 1) % 7 in self.weekday  # Python Monday=0 -> cron Sunday=0
        date_ok = (day_ok and weekday_ok) if (self.day_any or self.weekday_any) else (day_ok or weekday_ok)
        return d.month in self.month and date_ok

    def matches(self, t: dt.datetime) -> bool:
        return self.date_matches(t.date()) and t.hour in self.hour and t.minute in self.minute

    def next_after(self, t: dt.datetime) -> dt.datetime:
        """First matching minute strictly after `t` (searches up to ~4 years of days)."""
        times = sorted((h, m) for h in self.hour for m in self.minute)
        day = t.date()
        for _ in range(4 * 366):
            if self.date_matches(day):
                for h, m in times:
                    cand = dt.datetime(day.year, day.month, day.day, h, m, tzinfo=t.tzinfo)
                    if cand > t:
                        return cand
            day += dt.timedelta(days=1)
        raise ValueError(f"cron {self.expr!r} never fires")
