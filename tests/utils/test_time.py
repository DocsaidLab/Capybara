from __future__ import annotations

import time as py_time
from datetime import datetime, timezone

import numpy as np
import pytest

import capybara.utils.time as time_mod


def test_timer_requires_tic_before_toc():
    timer = time_mod.Timer()
    with pytest.raises(ValueError, match="has not been started"):
        timer.toc()


def test_timer_records_and_stats(monkeypatch, capsys):
    perf = iter([10.0, 10.5, 20.0, 20.25])

    monkeypatch.setattr(time_mod.time, "perf_counter", lambda: next(perf))
    timer = time_mod.Timer(precision=2, desc="bench", verbose=True)

    timer.tic()
    dt1 = timer.toc(verbose=True)
    assert dt1 == 0.5

    timer.tic()
    dt2 = timer.toc()
    assert dt2 == 0.25

    assert timer.mean == np.array([0.5, 0.25]).mean().round(2)
    assert timer.max == 0.5
    assert timer.min == 0.25
    assert timer.std == np.array([0.5, 0.25]).std().round(2)

    out = capsys.readouterr().out
    assert "bench" in out
    assert "Cost:" in out

    timer.clear_record()
    assert timer.mean is None


def test_timer_as_context_manager(monkeypatch):
    perf = iter([1.0, 1.2])
    monkeypatch.setattr(time_mod.time, "perf_counter", lambda: next(perf))
    timer = time_mod.Timer(precision=3)
    with timer:
        pass

    assert timer.dt == 0.2


def test_timer_context_manager_returns_self(monkeypatch):
    perf = iter([1.0, 1.1])
    monkeypatch.setattr(time_mod.time, "perf_counter", lambda: next(perf))

    with time_mod.Timer() as timer:
        assert isinstance(timer, time_mod.Timer)


def test_timer_as_decorator(monkeypatch):
    perf = iter([5.0, 5.1])
    monkeypatch.setattr(time_mod.time, "perf_counter", lambda: next(perf))

    timer = time_mod.Timer()

    @timer
    def add1(x: int) -> int:
        return x + 1

    assert add1(2) == 3
    assert timer.mean == 0.1


def test_now_supports_timestamp_datetime_time_and_fmt(monkeypatch):
    monkeypatch.setattr(time_mod.time, "time", lambda: 123.0)
    assert time_mod.now("timestamp") == 123.0

    assert isinstance(time_mod.now("datetime"), datetime)
    assert isinstance(time_mod.now("time"), py_time.struct_time)

    # fmt takes precedence and returns a string.
    monkeypatch.setattr(time_mod.time, "localtime", py_time.gmtime)
    assert time_mod.now(fmt="%Y-%m-%d") == "1970-01-01"

    with pytest.raises(ValueError, match="Unsupported input"):
        time_mod.now("invalid")


def test_time_and_datetime_converters_validate_types(monkeypatch):
    fixed = py_time.gmtime(0)

    assert time_mod.time2datetime(fixed).year == 1970
    assert isinstance(time_mod.timestamp2datetime(0), datetime)

    with pytest.raises(TypeError):
        time_mod.time2datetime("not-a-time")  # type: ignore[arg-type]

    monkeypatch.setattr(time_mod.time, "mktime", lambda _: 42.0)
    assert time_mod.time2timestamp(fixed) == 42.0
    with pytest.raises(TypeError):
        time_mod.time2timestamp("not-a-time")  # type: ignore[arg-type]

    assert time_mod.time2str(fixed, "%Y") == "1970"
    with pytest.raises(TypeError):
        time_mod.time2str("not-a-time", "%Y")  # type: ignore[arg-type]

    dt = datetime(2000, 1, 1, tzinfo=timezone.utc)
    assert isinstance(time_mod.datetime2time(dt), py_time.struct_time)
    assert time_mod.datetime2timestamp(dt) == 946684800.0
    assert time_mod.datetime2str(dt, "%Y") == "2000"

    with pytest.raises(TypeError):
        time_mod.datetime2time("not-a-dt")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        time_mod.datetime2timestamp("not-a-dt")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        time_mod.datetime2str("not-a-dt", "%Y")  # type: ignore[arg-type]


def test_str_converters_validate_types(monkeypatch):
    monkeypatch.setattr(time_mod.time, "mktime", lambda _: 99.0)
    assert time_mod.str2time("1970-01-01", "%Y-%m-%d").tm_year == 1970
    assert time_mod.str2datetime("1970-01-01", "%Y-%m-%d").year == 1970
    assert time_mod.str2timestamp("1970-01-01", "%Y-%m-%d") == 99.0

    with pytest.raises(TypeError):
        time_mod.str2time(123, "%Y")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        time_mod.str2datetime(123, "%Y")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        time_mod.str2timestamp(123, "%Y")  # type: ignore[arg-type]
