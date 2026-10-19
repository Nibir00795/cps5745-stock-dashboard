"""Hand-checkable cases for metrics.py:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics as M  # noqa: E402


def frame(values, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(next(iter(values.values()))))
    return pd.DataFrame(values, index=idx)


def test_rebased_starts_at_100():
    c = frame({"A": [50, 75, 100], "B": [10, 5, 20]})
    r = M.rebased(c)
    assert r.iloc[0].tolist() == [100, 100]
    assert r["A"].iloc[-1] == 200 and r["B"].iloc[-1] == 200


def test_drawdown_from_running_peak():
    c = frame({"A": [100, 120, 90, 60, 130]})
    assert M.drawdown(c)["A"].round(6).tolist() == [0, 0, -25, -50, 0]


def test_daily_returns_drop_first_row():
    c = frame({"A": [100, 110, 99]})
    r = M.daily_returns(c)["A"]
    assert len(r) == 2 and r.iloc[0] == pytest.approx(0.10) and r.iloc[1] == pytest.approx(-0.10)


def test_summary_total_return_and_cagr():
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])  # 731 days
    c = pd.DataFrame({"A": [100.0, 121.0]}, index=idx)
    s = M.summary(c).loc["A"]
    assert s["total_return"] == pytest.approx(0.21)
    assert s["cagr"] == pytest.approx(1.21 ** (365.25 / 731) - 1)
    assert s["max_drawdown"] == 0


def test_volatility_is_annualised():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 2000)
    c = frame({"A": 100 * np.cumprod(1 + r)})
    vol = M.summary(c).loc["A", "volatility"]
    assert vol == pytest.approx(0.02 * np.sqrt(252), rel=0.05)
    assert M.rolling_volatility(c, 30)["A"].dropna().mean() == pytest.approx(2 * np.sqrt(252), rel=0.1)


def test_calendar_returns_flag_partial_years():
    idx = pd.bdate_range("2023-06-01", "2025-03-31")
    c = pd.DataFrame({"A": np.linspace(100, 200, len(idx))}, index=idx)
    cal = M.calendar_returns(c).set_index("Year")
    assert cal.loc[2023, "partial"] and cal.loc[2025, "partial"] and not cal.loc[2024, "partial"]
    y23_end = c.loc[c.index.year == 2023, "A"].iloc[-1]
    y24_end = c.loc[c.index.year == 2024, "A"].iloc[-1]
    assert cal.loc[2024, "return"] == pytest.approx(y24_end / y23_end - 1)


def test_correlation_of_identical_and_opposite_moves():
    c = frame({"A": [100, 110, 99, 120, 108], "B": [50, 55, 49.5, 60, 54]})
    corr = M.correlation(c)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_wide_close_filters_dates_and_keeps_ticker_order():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
        "Ticker": ["X", "X", "Y", "Y"], "Close": [1, 2, 3, 4], "Volume": [10, 10, 10, 10]})
    w = M.wide_close(df, ["Y", "X"], "2024-01-03", "2024-01-03")
    assert list(w.columns) == ["Y", "X"] and w.iloc[0].tolist() == [4, 2]
    dv = M.wide_dollar_volume(df, ["X"], "2024-01-02", "2024-01-03")
    assert dv["X"].tolist() == [10, 20]


def test_published_data_is_sane():
    path = M.DATA
    if not path.exists():
        pytest.skip("data/prices.csv not fetched yet")
    df = M.load()
    assert set(["NVDA", "AMD", "INTC"]) <= set(df["Ticker"])
    assert (df[["Open", "High", "Low", "Close"]] > 0).all().all()
    assert not df.duplicated(["Date", "Ticker"]).any()
