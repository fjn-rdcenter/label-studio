import datetime as dt

import pytest

from tasdf_online.schedule import Cron


def test_weekly_sunday_1am():
    c = Cron("0 1 * * 0")
    friday = dt.datetime(2026, 11, 13, 20, 0)  # Friday
    assert c.next_after(friday) == dt.datetime(2026, 11, 15, 1, 0)  # Sunday 01:00
    assert c.next_after(dt.datetime(2026, 11, 15, 1, 0)) == dt.datetime(2026, 11, 22, 1, 0)  # strictly after


def test_lists_ranges_steps_and_sunday_as_7():
    c = Cron("*/30 22-23 * * 1-5")
    assert c.next_after(dt.datetime(2026, 11, 13, 22, 10)) == dt.datetime(2026, 11, 13, 22, 30)
    assert c.next_after(dt.datetime(2026, 11, 13, 23, 30)) == dt.datetime(2026, 11, 16, 22, 0)  # skips the weekend
    assert Cron("0 1 * * 7").next_after(dt.datetime(2026, 11, 13)) == dt.datetime(2026, 11, 15, 1, 0)


def test_day_of_month_or_weekday_when_both_restricted():
    c = Cron("0 0 1 * 1")  # the 1st of the month OR any Monday (cron semantics)
    assert c.next_after(dt.datetime(2026, 11, 13)) == dt.datetime(2026, 11, 16, 0, 0)  # Monday comes first
    assert c.matches(dt.datetime(2026, 12, 1, 0, 0))


@pytest.mark.parametrize("bad", ["0 1 * *", "61 * * * *", "0 25 * * *", "*/0 * * * *", "0 0 31 2 *"])
def test_bad_expressions_fail_fast(bad):
    with pytest.raises(ValueError):
        Cron(bad).next_after(dt.datetime(2026, 1, 1))
