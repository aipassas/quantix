# Quantix

[![Tests](https://github.com/aipassas/quantix/actions/workflows/tests.yml/badge.svg)](https://github.com/aipassas/quantix/actions/workflows/tests.yml)

Institutional-grade stock analysis and simulation engine, built as a
Streamlit app on top of free Yahoo Finance data.

Quantix pulls price history and financial statements for a ticker, runs
them through a validation/standardization/quality-scoring pipeline, and
surfaces fundamental analysis (ratios, scorecard, Altman Z-Score, DCF,
Monte Carlo simulation), technical analysis (SMA, RSI, MACD, Bollinger
Bands, ATR) and an interactive charting dashboard — all from one search box.

## Features

- **Fundamental analysis** — valuation ratios, profitability/liquidity/
  solvency metrics, a weighted scorecard, Altman Z-Score, and a DCF model
  with Monte Carlo sensitivity simulation.
- **Technical analysis** — SMA crossover signals, RSI with overbought/
  oversold interpretation, MACD, Bollinger Band breakouts, and ATR-based
  stop-loss suggestions.
- **Interactive charting** — candlestick chart with volume overlay,
  togglable indicator panels, zoom/pan, built on Plotly.
- **Data quality scoring** — every ticker gets a 0-100 quality score based
  on field completeness, data freshness, and fetch reliability, shown
  before any derived metric.
- **Peer comparison & watchlist scanning**, plus a benchmark/VIX/10-year
  Treasury macro overlay.

No API keys required — everything is sourced from Yahoo Finance via
[`yfinance`](https://github.com/ranaroussi/yfinance).

## Getting started

```bash
pip install -r requirements.txt
streamlit run finance.py
```

The app opens at `http://localhost:8501`. Enter a ticker in the sidebar to
get started.

## Testing

Every technical indicator and risk metric is covered by a `pytest` suite —
cross-validated against `pandas_ta` (technical indicators) and hand-worked
reference computations (risk metrics), plus edge cases (insufficient data,
zero-variance series, missing financial statement fields).

```bash
pip install -r requirements-dev.txt
pytest
```

The default run makes zero network calls and finishes in a few seconds. A
small set of opt-in sanity checks against real Yahoo Finance data (marked
`@pytest.mark.live`) are skipped by default — run them explicitly with:

```bash
pytest -m live
```

## Architecture

Quantix is split into small, single-purpose modules rather than one large
script — see [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full data flow
(fetch → validate → standardize → quality-score → render), caching
strategy, and design principles.

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how data moves through the app,
  from the Yahoo Finance API to the screen.
- [`FUNDAMENTALS.md`](FUNDAMENTALS.md) — every fundamental analysis
  formula, null-handling rule, and assumption.
- [`TECHNICAL_ANALYSIS.md`](TECHNICAL_ANALYSIS.md) — every technical
  indicator's formula, smoothing convention, and how it was cross-validated
  against `pandas_ta`.
- [`RISK_ANALYTICS.md`](RISK_ANALYTICS.md) — every risk metric's formula
  (volatility, VaR/CVaR, Maximum Drawdown, Sharpe/Sortino/Calmar), the
  composite Risk Score's methodology, and how each was validated.

## Disclaimer

Quantix is a personal research/educational project. Nothing it displays is
financial advice.
