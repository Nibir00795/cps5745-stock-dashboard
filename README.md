# Chip Stocks Compared

An interactive Plotly Dash dashboard comparing NVIDIA (NVDA), AMD and Intel (INTC), with TSMC, Broadcom and Qualcomm available in the company filter.

Md Jonayed Hossain Chowdhury. Developed for CPS*5745, Kean University, Fall 2026.

**Live:** https://chip-stocks-jonayed.vercel.app (no sign-in)

## What it shows

One company filter and one date range drive everything on the page. The quick-range buttons (1Y, 3Y, 5Y, YTD, MAX) set the date picker; Reset returns to NVDA, AMD and INTC over five years.

Under the filters are a price card per company and a short "Key differences" panel whose sentences are recomputed for whatever is selected. Then eight charts:

1. Growth of $100, log or linear scale
2. Drawdown from the running peak (%)
3. 30-day rolling volatility, annualised (%)
4. Return by calendar year (%), partial years marked with *
5. Risk versus return: annualised volatility against annualised return
6. Distribution of daily returns (box plot)
7. Correlation of daily returns (diverging scale, -1 to 1)
8. Trading activity: average daily dollar volume by month ($ billions)

Every company keeps the same colour in every chart, whatever else is selected.

## Data

Daily prices from Yahoo Finance via `yfinance`, adjusted for splits and dividends, from January 2016. `.github/workflows/update-data.yml` runs `scripts/fetch_prices.py` every weekday at 22:30 UTC (after the New York close) and commits `data/prices.csv` when it changes. Vercel redeploys on that commit, so the live page always shows the latest trading day. The workflow can also be run by hand from the Actions tab.

## Run locally

    pip install -r requirements.txt
    python scripts/fetch_prices.py   # needs yfinance; or pull the file the workflow committed
    python app.py                    # http://127.0.0.1:8050
    python -m pytest -q tests        # metric checks

## Files

- `app.py` layout, figures and callbacks. Vercel loads the Flask server `app` from here (`[tool.vercel]` in `pyproject.toml`).
- `metrics.py` the calculations, kept free of Dash so `tests/test_metrics.py` can check them.
- `assets/style.css` page styling, modelled on the Kean enrollment dashboard: navy header, one filter bar, summary cards above the charts.
