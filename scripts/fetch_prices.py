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
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM"]
START = "2016-01-01"
OUT = Path(__file__).resolve().parent.parent / "data" / "prices.csv"


def main():
    raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    frames = []
    for t in TICKERS:
        if t not in raw.columns.get_level_values(0):
            sys.exit(f"No data returned for {t}; leaving the old file in place.")
        df = raw[t].dropna(subset=["Close"]).reset_index()
        df.insert(1, "Ticker", t)
        frames.append(df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = out[c].round(4)
    out["Volume"] = out["Volume"].astype("int64")

    # Sanity checks before overwriting the published file.
    counts = out.groupby("Ticker").size()
    if counts.min() < 1000:
        sys.exit(f"Too few rows for some tickers: {counts.to_dict()}")
    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        sys.exit("Non-positive prices found; refusing to write.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["Ticker", "Date"]).to_csv(OUT, index=False)
    print(f"Wrote {len(out):,} rows, {out['Date'].min()} to {out['Date'].max()}, to {OUT}")


if __name__ == "__main__":
    main()
