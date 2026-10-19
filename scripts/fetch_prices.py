#!/usr/bin/env python3
"""
Download daily prices for the dashboard's companies from Yahoo Finance and write
data/prices.csv. Run by .github/workflows/update-data.yml every weekday after the US close,
so the live dashboard always shows the latest available trading day.

    python scripts/fetch_prices.py

Prices are split- and dividend-adjusted (auto_adjust=True), so returns computed from Close
are total returns and a stock split never shows up as a crash.
"""
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TICKERS = ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM"]
REQUIRED = ["NVDA", "AMD", "INTC"]  # the three the dashboard compares by default
START = "2016-01-01"
OUT = Path(__file__).resolve().parent.parent / "data" / "prices.csv"


def fetch_one(ticker, attempts=3):
    """One ticker at a time, with retries: a batch download can silently return an empty
    column for a ticker when Yahoo throttles, which is how TSM went missing in the first run."""
    for i in range(attempts):
        df = yf.download(ticker, start=START, auto_adjust=True, progress=False, multi_level_index=False)
        df = df.dropna(subset=["Close"])
        if len(df) >= 1000:
            df = df.reset_index()
            df.insert(1, "Ticker", ticker)
            return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
        time.sleep(5 * (i + 1))
    if ticker in REQUIRED:
        sys.exit(f"No usable data for {ticker} after {attempts} attempts; leaving the old file in place.")
    print(f"Warning: skipping {ticker}, no usable data after {attempts} attempts.")
    return None


def main():
    frames = [f for f in (fetch_one(t) for t in TICKERS) if f is not None]
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = out[c].round(4)
    out["Volume"] = out["Volume"].astype("int64")

    # A run during market hours returns today's bar as it stands so far. Drop it: the
    # 22:30 UTC run adds the finished day.
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.time() < dtime(16, 30):
        out = out[out["Date"] < ny.strftime("%Y-%m-%d")]

    # Sanity checks before overwriting the published file.
    counts = out.groupby("Ticker").size().reindex(REQUIRED, fill_value=0)
    if counts.min() < 1000:
        sys.exit(f"Too few rows for some tickers: {counts.to_dict()}")
    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        sys.exit("Non-positive prices found; refusing to write.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["Ticker", "Date"]).to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows, {out['Date'].min()} to {out['Date'].max()}, to {OUT}")


if __name__ == "__main__":
    main()
