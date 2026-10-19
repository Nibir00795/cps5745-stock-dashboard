"""
Comparison metrics for the dashboard. Pure pandas, no Dash, so tests/test_metrics.py can
check every number on its own.

All prices are adjusted closes, so a return computed from Close includes dividends and is not
broken by stock splits. One trading year = 252 trading days.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DATA = Path(__file__).resolve().parent / "data" / "prices.csv"


def load(path=DATA):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def wide_close(df, tickers, start, end):
    """Close prices, one column per ticker, rows = trading days inside [start, end]."""
    m = df["Ticker"].isin(tickers) & (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    w = df[m].pivot(index="Date", columns="Ticker", values="Close")
    return w[[t for t in tickers if t in w.columns]].dropna(how="all")


def wide_dollar_volume(df, tickers, start, end):
    m = df["Ticker"].isin(tickers) & (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    d = df[m].assign(DV=lambda x: x["Close"] * x["Volume"])
    w = d.pivot(index="Date", columns="Ticker", values="DV")
    return w[[t for t in tickers if t in w.columns]]


def rebased(close):
    """Each series divided by its first value in the range, times 100."""
    return close.div(close.bfill().iloc[0]).mul(100)


def daily_returns(close):
    return close.pct_change(fill_method=None).iloc[1:]


def drawdown(close):
    """Percent below the highest close so far in the range (0 at a new high, negative otherwise)."""
    return close.div(close.cummax()).sub(1).mul(100)


def rolling_volatility(close, window=30):
    """Annualised standard deviation of daily returns over a trailing window, in percent."""
    return daily_returns(close).rolling(window).std().mul(np.sqrt(TRADING_DAYS) * 100)


def summary(close):
    """One row per ticker: total return, annualised return, annualised volatility, max drawdown."""
    rows = []
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 2:
            continue
        days = (s.index[-1] - s.index[0]).days
        total = s.iloc[-1] / s.iloc[0] - 1
        cagr = (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1 if days > 0 else np.nan
        vol = s.pct_change().std() * np.sqrt(TRADING_DAYS)
        mdd = (s / s.cummax() - 1).min()
        rows.append({"Ticker": t, "first": s.iloc[0], "last": s.iloc[-1], "total_return": total,
                     "cagr": cagr, "volatility": vol, "max_drawdown": mdd,
                     "start": s.index[0], "end": s.index[-1]})
    return pd.DataFrame(rows).set_index("Ticker")


def calendar_returns(close):
    """Return in each calendar year inside the range. The first and last years can be partial;
    each is measured from the first to the last close available in that year (or from the
    previous year's last close when it is inside the range), and flagged as partial."""
    out = []
    years = sorted(close.index.year.unique())
    for t in close.columns:
        s = close[t].dropna()
        for y in years:
            in_y = s[s.index.year == y]
            if in_y.empty:
                continue
            prev = s[s.index.year == y - 1]
            base = prev.iloc[-1] if not prev.empty else in_y.iloc[0]
            full_start = not prev.empty
            full_end = in_y.index[-1].month == 12 and in_y.index[-1].day >= 24
            out.append({"Ticker": t, "Year": y, "return": in_y.iloc[-1] / base - 1,
                        "partial": not (full_start and full_end)})
    return pd.DataFrame(out)


def correlation(close):
    return daily_returns(close).corr()
