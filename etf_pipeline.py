"""Fund identity, lifecycle and performance — the ingestion layer.

WHAT WAS ALREADY BUILT. This is PHASE 1.1 and it ran last, so most of its
list already exists: expense ratio, AUM, top ten holdings, sector
weightings, category and family in etf_analysis; the 250-fund universe
with returns and yield in etf_screener; volume and relative volume in
etf_technicals; volatility and Sharpe in etf_comparison; and the whole
validation section in tests/test_etf_validation.py, where three of the
task's five rules were measured to be false. None of that is rebuilt.

What was never fetched is the fund's IDENTITY and LIFECYCLE — ISIN,
inception, dividend cadence — and a performance record that does not
inherit the provider's unit confusion. That is what this module adds.

THREE PROVIDER FIELDS ARE UNRELIABLE AND ARE REPLACED, not passed
through. The pattern across this project holds again: the raw data is
sound and the DERIVED analytics are not.

  1. `fundInceptionDate` contradicts the provider's own price history.
     VTI reports 2016-06-27; Yahoo's price series for VTI starts
     2001-06-15 and the fund really launched 2001-05-24 — a fifteen-year
     error, from the same API. The cross-check is free, so a reported
     inception later than the first price bar is reported as suspect
     rather than shown.

  2. `beta3Year` is wrong for anything that is not an equity fund. TLT
     reports 2.40 against a regressed three-year beta of +0.13 — an
     eighteen-fold overstatement that would tell someone a long treasury
     fund moves twice as hard as the market in the same direction, which
     is the opposite of why anyone holds one. QQQ (1.26 vs 1.27) and
     ARKK (2.46 vs 2.02) are roughly right, so the field is not
     uniformly broken — which is worse, because it looks reliable.
     Beta is regressed from prices here.

  3. THE RETURN FIELDS MIX UNITS. `ytdReturn` is a PERCENT (10.0863 for
     SPY) while `threeYearAverageReturn` (0.2185) and
     `fiveYearAverageReturn` (0.1307) are FRACTIONS. Exactly 100x apart,
     in the same dict, on the same fund. Rather than convert and hope the
     convention never flips, every window here is computed from the price
     series, which has one unit and no convention to remember.

WHAT IS GENUINELY ABSENT, checked rather than assumed:
  - Geographic allocation. There is no field for it anywhere in
    funds_data — not under region, country or domicile. It is the one
    item on the task's list with no substitute at all.
  - SHARES held per holding. top_holdings carries Name and Holding
    Percent and nothing else, so "ticker, % weight, shares" is two of
    three.
  - Custodian and manager name. Absent from info.
  - ISIN for some funds: SPY, VTI, ARKK, TLT and GLD report one; QQQ and
    every European listing checked do not.
  - Each fund's OWN stated benchmark. PHASE 1.3 measures tracking error
    against the benchmark the READER chooses instead, which is a real
    figure as long as the screen names it — see etf_risk. What stays
    unavailable is the mapping from a fund to the index its prospectus
    names.

MORNINGSTAR NEEDS A KEY. There is no free tier and no keyless endpoint,
and obtaining one means creating an account, which this build does not do
on the user's behalf. The connector is wired the way bond_data wires
FRED — read a key from st.secrets then the environment, and say what it
would add — so supplying one is a configuration change rather than a code
change.
"""
import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger("etf_pipeline")

CACHE_TTL_SECONDS = 86400        # the task asks for a daily refresh

MORNINGSTAR_ENV_VAR = "QUANTIX_MORNINGSTAR_API_KEY"

MORNINGSTAR_UNCONFIGURED = (
    "Morningstar is not configured. It would add the fields this data "
    "source has no substitute for — geographic allocation, share counts "
    "per holding, custodian and manager, and a stated benchmark to "
    "measure tracking error against. It needs a paid API key: add it "
    "under [morningstar] api_key in .streamlit/secrets.toml, or as the "
    f"{MORNINGSTAR_ENV_VAR} environment variable."
)

GEOGRAPHIC_ALLOCATION_UNAVAILABLE = (
    "Geographic allocation is not available. Checked field by field: this "
    "data source exposes asset classes, sector weightings, bond holdings, "
    "bond ratings, equity holdings, fund operations and the top ten — and "
    "nothing for region, country or domicile. It is the one item with no "
    "substitute here, rather than something computed approximately."
)

SHARE_COUNTS_UNAVAILABLE = (
    "Share counts per holding are not available — the holdings table "
    "carries a name and a weight and nothing else. The weight is the "
    "figure that matters for allocation; the share count would only add "
    "the fund's own scale."
)

# The task's performance windows, in TRADING days. Approximate by
# construction — a month is not 21 days every month — so the label says
# what was actually measured and `days_used` reports the bars behind it.
PERFORMANCE_WINDOWS: Tuple[Tuple[str, int], ...] = (
    ("1D", 1),
    ("1W", 5),
    ("1M", 21),
    ("3M", 63),
    ("1Y", 252),
    ("3Y", 756),
    ("5Y", 1260),
)

# Windows longer than a year are reported ANNUALISED, because a raw
# five-year number sitting beside a one-year number invites reading them
# as the same kind of thing.
ANNUALISE_ABOVE_DAYS = 252

BETA_BENCHMARK = "SPY"
BETA_MIN_DAYS = 60

DIVIDEND_FREQUENCY_LABELS: Tuple[Tuple[int, str], ...] = (
    (1, "Annual"), (2, "Semiannual"), (4, "Quarterly"),
    (12, "Monthly"), (52, "Weekly"),
)
DIVIDEND_LOOKBACK_DAYS = 1100     # about three years of payments
DIVIDEND_MIN_PAYMENTS = 3


def morningstar_api_key() -> Optional[str]:
    """st.secrets first, then the environment — the shape every other
    credential in this app uses. st.secrets raises when no secrets file
    exists, which is the normal state, so unconfigured stays quiet."""
    import os

    try:
        section = st.secrets.get("morningstar", {})
        value = section.get("api_key") if hasattr(section, "get") else None
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(MORNINGSTAR_ENV_VAR) or "").strip() or None


def morningstar_is_configured() -> bool:
    return bool(morningstar_api_key())


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


# --- identity and lifecycle ---------------------------------------------------

@dataclass(frozen=True)
class PerformanceWindow:
    label: str
    return_pct: Optional[float]
    annualised: bool
    days_used: int


@dataclass(frozen=True)
class FundIdentity:
    symbol: str
    name: str = ""
    isin: Optional[str] = None
    inception: Optional[datetime.date] = None
    inception_is_suspect: bool = False
    first_price_bar: Optional[datetime.date] = None
    dividend_frequency: Optional[str] = None
    dividends_per_year: Optional[float] = None
    beta: Optional[float] = None
    beta_r_squared: Optional[float] = None
    beta_days: int = 0
    reported_beta: Optional[float] = None
    performance: Tuple[PerformanceWindow, ...] = ()
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def age_years(self) -> Optional[float]:
        if self.inception is None:
            return None
        return (datetime.date.today() - self.inception).days / 365.25

    @property
    def beta_disagrees_with_reported(self) -> bool:
        """Whether the provider's beta and the regressed one tell
        different stories. TLT: reported 2.40, regressed 0.13."""
        if self.beta is None or self.reported_beta is None:
            return False
        return abs(self.beta - self.reported_beta) > 0.5


def dividend_frequency(dividends: Optional["pd.Series"],
                       ) -> Tuple[Optional[str], Optional[float]]:
    """(label, payments per year) from the MEDIAN gap between payments.

    Counting payments in a trailing year over-counts whenever a boundary
    payment falls inside the window — measured, that read quarterly funds
    as five a year and a monthly fund as thirteen. The median spacing has
    no such edge, and gets SPY, QQQ, VYM (quarterly) and TLT (monthly)
    right.
    """
    if dividends is None or len(dividends) < DIVIDEND_MIN_PAYMENTS:
        return None, None
    try:
        recent = dividends[dividends.index
                           >= dividends.index.max()
                           - pd.Timedelta(days=DIVIDEND_LOOKBACK_DAYS)]
        if len(recent) < DIVIDEND_MIN_PAYMENTS:
            return None, None
        gaps = np.diff(recent.index.values).astype("timedelta64[D]").astype(int)
        median_gap = float(np.median(gaps))
    except Exception:                              # noqa: BLE001
        return None, None
    if median_gap <= 0:
        return None, None
    per_year = 365.0 / median_gap
    label = min(DIVIDEND_FREQUENCY_LABELS,
                key=lambda kv: abs(kv[0] - per_year))[1]
    return label, per_year


def performance_windows(closes: Optional["pd.Series"]
                        ) -> Tuple[PerformanceWindow, ...]:
    """Every window the task asks for, from the price series.

    Computed rather than read, because the provider's own return fields
    mix percent and fraction in the same dict. A window with fewer bars
    than it needs is reported as unavailable rather than measured over a
    shorter span — a "5Y return" over two years is not one.
    """
    if closes is None:
        return ()
    series = pd.to_numeric(closes, errors="coerce").dropna()
    out: List[PerformanceWindow] = []
    for label, days in PERFORMANCE_WINDOWS:
        if len(series) <= days:
            out.append(PerformanceWindow(label, None, False, len(series)))
            continue
        first, last = _number(series.iloc[-1 - days]), _number(series.iloc[-1])
        if first is None or last is None or first <= 0:
            out.append(PerformanceWindow(label, None, False, len(series)))
            continue
        total = last / first - 1.0
        annualise = days > ANNUALISE_ABOVE_DAYS
        if annualise:
            years = days / 252.0
            value = ((1.0 + total) ** (1.0 / years) - 1.0) * 100.0
        else:
            value = total * 100.0
        out.append(PerformanceWindow(label, value, annualise, days))
    return tuple(out)


def regressed_beta(fund_closes: Optional["pd.Series"],
                   benchmark_closes: Optional["pd.Series"]
                   ) -> Tuple[Optional[float], Optional[float], int]:
    """(beta, r-squared, days) against the benchmark, from prices.

    Exists because `beta3Year` is wrong for non-equity funds: TLT reports
    2.40 where this returns +0.13. The r-squared travels with it, because
    a beta explaining nothing is a number a reader should discount — for
    a bond or commodity fund it will be near zero, and that is the useful
    signal rather than a defect.
    """
    if fund_closes is None or benchmark_closes is None:
        return None, None, 0
    pair = pd.concat([pd.to_numeric(fund_closes, errors="coerce"),
                      pd.to_numeric(benchmark_closes, errors="coerce")],
                     axis=1).dropna()
    if len(pair) < BETA_MIN_DAYS:
        return None, None, len(pair)
    returns = pair.pct_change().dropna()
    if len(returns) < BETA_MIN_DAYS or returns.iloc[:, 1].std() == 0:
        return None, None, len(returns)
    y = returns.iloc[:, 0].to_numpy(dtype=float)
    x = returns.iloc[:, 1].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), r_squared, len(returns)


def _epoch_to_date(value) -> Optional[datetime.date]:
    """Unix seconds -> a date, in UTC.

    UTC rather than local time: a timestamp near midnight resolves to the
    previous day in a western timezone, which would silently move an
    inception date by one.
    """
    number = _number(value)
    if number is None:
        return None
    try:
        return datetime.datetime.fromtimestamp(
            number, datetime.timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_identity(symbol: str, lookback_period: str = "max") -> FundIdentity:
    """A fund's identity, lifecycle and performance record. Never raises.

    The full history, not a window, for two reasons found by getting it
    wrong. The inception cross-check compares the reported date against
    the FIRST price bar, so a five-year window would only ever see bars
    from the last five years and could never contradict a date older than
    that — VTI's bad 2016 date sailed through until this was widened.
    And the 5Y performance window needs 1260 trading days, which a "5y"
    request does not quite supply: SPY came back with 1258 and reported
    its five-year return as unavailable.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return FundIdentity(symbol="", error="No symbol given.")
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # ISIN lives on the Ticker, not in info — and is absent for some
        # funds (QQQ and every European listing checked).
        isin: Optional[str] = None
        try:
            raw_isin = ticker.isin
            if raw_isin and str(raw_isin).strip() not in ("-", "None"):
                isin = str(raw_isin).strip().upper()
        except Exception:                          # noqa: BLE001
            log_exception(logger, "etf_pipeline.isin_unreadable", ticker=symbol)

        frame = yf.download([symbol, BETA_BENCHMARK], period=lookback_period,
                            progress=False, auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else None
        fund_closes = (closes[symbol]
                       if closes is not None and symbol in closes else None)
        bench_closes = (closes[BETA_BENCHMARK]
                        if closes is not None and BETA_BENCHMARK in closes
                        else None)

        first_bar = None
        if fund_closes is not None:
            clean = pd.to_numeric(fund_closes, errors="coerce").dropna()
            if len(clean):
                first_bar = clean.index[0].date()

        inception = _epoch_to_date(info.get("fundInceptionDate"))
        # The cross-check that caught VTI: a fund cannot have priced
        # before it existed, so a reported inception AFTER the first bar
        # means the field is wrong. The window ignores a few days of
        # slack, since the first bar is whatever history the source keeps.
        suspect = bool(inception and first_bar
                       and inception > first_bar + datetime.timedelta(days=5))

        frequency, per_year = dividend_frequency(_dividends(ticker))
        beta, r_squared, beta_days = regressed_beta(fund_closes, bench_closes)

        identity = FundIdentity(
            symbol=symbol,
            name=str(info.get("longName") or info.get("shortName") or "").strip(),
            isin=isin,
            inception=inception,
            inception_is_suspect=suspect,
            first_price_bar=first_bar,
            dividend_frequency=frequency,
            dividends_per_year=per_year,
            beta=beta,
            beta_r_squared=r_squared,
            beta_days=beta_days,
            reported_beta=_number(info.get("beta3Year")),
            performance=performance_windows(fund_closes),
        )
        log_event(logger, logging.INFO, "etf_pipeline.identity_loaded",
                  ticker=symbol, isin=bool(isin),
                  inception_suspect=suspect,
                  beta=None if beta is None else round(beta, 2))
        return identity
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "etf_pipeline.identity_failed", ticker=symbol,
                      error=f"{type(exc).__name__}: {exc}")
        return FundIdentity(symbol=symbol,
                            error=f"Fund identity is unavailable for {symbol} "
                                  f"({type(exc).__name__}).")


def _dividends(ticker) -> Optional["pd.Series"]:
    try:
        return ticker.dividends
    except Exception:                              # noqa: BLE001
        return None


def describe_inception(identity: FundIdentity) -> str:
    """One line about the fund's age, flagging a date the price history
    contradicts rather than printing it as fact."""
    if identity.inception is None:
        return "Inception date not reported."
    if identity.inception_is_suspect:
        return (f"Reported inception {identity.inception}, but this source's "
                f"own price history for {identity.symbol} starts "
                f"{identity.first_price_bar} — the fund cannot have traded "
                "before it existed, so treat the reported date as wrong.")
    age = identity.age_years
    return (f"Launched {identity.inception}"
            + (f", about {age:.0f} years ago." if age is not None else "."))
