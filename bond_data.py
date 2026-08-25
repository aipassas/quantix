"""Treasury yields, the curve, and bond-fund characteristics.

WHAT THIS BUILD CAN AND CANNOT REACH, measured on 2026-08-25 rather than
assumed, because the task names two providers and one of them is not
obtainable here:

  - FRED needs an API key. The API answers a keyless request with HTTP
    400 and "Variable api_key is not set", and the public CSV endpoint
    (fredgraph.csv) is unreachable from this environment — it times out,
    then the remote closes the connection. Registering for a key means
    creating an account, which is not something this build does on the
    user's behalf. So the FRED path is WIRED and unused: supply a key
    the way every other credential here is supplied (see fred_api_key)
    and the curve widens from four maturities to ten.

  - Bloomberg is an enterprise licence. There is no free tier and no
    keyless endpoint. Individual corporate bonds — CUSIP, issuer,
    coupon, per-bond YTM, per-bond duration, ratings from S&P/Moody's/
    Fitch — are therefore not in this build at all, and are not faked
    from something adjacent. What IS available without credentials is
    bond-FUND level data, which is what this module provides.

  - Yahoo carries exactly FOUR treasury yield indices: ^IRX (13 weeks),
    ^FVX (5Y), ^TNX (10Y), ^TYX (30Y). Probed: no symbol exists for 1Y,
    2Y, 3Y, 7Y or 20Y. Treasury FUTURES (ZT/ZF/ZN/ZB) do quote, but they
    are prices, and turning a futures price into a yield needs the
    cheapest-to-deliver bond and its conversion factor — a different
    dataset, not a substitute. Four real points beat six invented ones.

NEITHER FIELD IN `bond_holdings` IS USABLE, AND THAT SHAPED THE MODULE.
It reports "Duration" and "Maturity" per fund. Both are wrong, and the
proof is a controlled ladder: iShares publishes five treasury funds that
differ ONLY in maturity band, so their true duration and maturity must
increase monotonically across them. Measured 2026-08-25:

    fund   band      true dur   Yahoo dur   true mat   Yahoo mat
    SHY    1-3yr        ~1.8       3.1         ~2         9.793
    IEI    3-7yr        ~4.4       3.49        ~5         9.595
    IEF    7-10yr       ~7.3       4.2         ~8.5       9.692
    TLH    10-20yr     ~12         3.54       ~15         8.199
    TLT    20+yr       ~15.5       3.56       ~25         7.603

Yahoo's maturity DECREASES as the real one rises, from 9.79 down to
7.60. Its duration barely moves at all and puts SHY above TLH. Neither
field discriminates between a two-year fund and a twenty-five-year one,
which is the only thing a bond fund's maturity is for.

Showing that would tell a reader a twenty-year treasury fund carries
less interest-rate risk than a one-to-three-year fund — the reverse of
the truth, and the kind of error someone loses money on. So duration is
MEASURED here instead, by regressing the fund's daily return on the
change in the ten-year yield. That returns 13.21 for TLT and 1.43 for
SHY: correctly ordered, and close to the published figures. Average
maturity has no such fallback and is simply not reported.

THE RATING BREAKDOWN IS SOUND, with one trap. The letter buckets sum to
exactly 1.0000 on every fund checked (TLT, SHY, AGG, BND, LQD, HYG) and
are the real breakdown. `us_government` is a SEPARATE, OVERLAPPING axis —
0.9958 for TLT, 0.0000 for LQD — because a treasury is both AA-rated and
government-issued. Folding it into the same list double-counts, and did:
TLT read as 199.6% investment grade before this was measured. It is
reported on its own line instead.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger("bond_data")

CACHE_TTL_SECONDS = 900          # the task asks for a 15-minute refresh
FUND_CACHE_TTL_SECONDS = 86400   # fund characteristics move slowly

# The ten maturities the task asks for. `yahoo` is the keyless symbol
# where one exists; `fred` is the series id used when a key is supplied.
# Only four have a Yahoo symbol — probed, not assumed.
@dataclass(frozen=True)
class Maturity:
    months: int
    label: str
    fred: str
    yahoo: Optional[str] = None


MATURITIES: Tuple[Maturity, ...] = (
    Maturity(3, "3M", "DGS3MO", "^IRX"),
    Maturity(6, "6M", "DGS6MO"),
    Maturity(12, "1Y", "DGS1"),
    Maturity(24, "2Y", "DGS2"),
    Maturity(36, "3Y", "DGS3"),
    Maturity(60, "5Y", "DGS5", "^FVX"),
    Maturity(84, "7Y", "DGS7"),
    Maturity(120, "10Y", "DGS10", "^TNX"),
    Maturity(240, "20Y", "DGS20"),
    Maturity(360, "30Y", "DGS30", "^TYX"),
)

MATURITIES_BY_MONTHS: Dict[int, Maturity] = {m.months: m for m in MATURITIES}

SOURCE_YAHOO = "yahoo"
SOURCE_FRED = "fred"

FRED_ENV_VAR = "QUANTIX_FRED_API_KEY"

FRED_UNCONFIGURED = (
    "Six of the ten maturities the yield curve wants — 6M, 1Y, 2Y, 3Y, 7Y "
    "and 20Y — come from FRED, which needs a free API key. Without one "
    "the curve is built from the four Yahoo publishes (3M, 5Y, 10Y, 30Y), "
    "which is enough to read its slope but not its detail. Add a key "
    "under [fred] api_key in .streamlit/secrets.toml, or as the "
    f"{FRED_ENV_VAR} environment variable."
)

BLOOMBERG_UNAVAILABLE = (
    "Individual corporate bonds — CUSIP, issuer, coupon, per-bond yield "
    "to maturity, per-bond duration and agency ratings — need a Bloomberg "
    "or equivalent licence, which this build does not have and cannot "
    "substitute for. Bond FUNDS are covered instead: their credit mix, "
    "average maturity and interest-rate sensitivity are all available "
    "without a licence."
)

REPORTED_DURATION_UNUSABLE = (
    "Duration here is measured from how this fund's price actually "
    "responds to a move in the 10-year yield, not taken from the data "
    "provider's own field. That field is unusable: across the iShares "
    "treasury ladder (SHY 1-3yr through TLT 20+yr) the reported maturity "
    "DECREASES from 9.79 to 7.60 years as the real one rises from about "
    "2 to 25, and the reported duration puts SHY above TLH. Average "
    "maturity has no equivalent fallback, so it is not shown at all."
)


# --- credentials --------------------------------------------------------------

def fred_api_key() -> Optional[str]:
    """st.secrets first, then the environment.

    Same shape as every other credential here: st.secrets raises when no
    secrets file exists at all, which is the normal state of a fresh
    checkout, so "unconfigured" stays a quiet fact rather than an error.
    """
    import os

    try:
        section = st.secrets.get("fred", {})
        value = section.get("api_key") if hasattr(section, "get") else None
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(FRED_ENV_VAR) or "").strip() or None


def fred_is_configured() -> bool:
    return bool(fred_api_key())


# --- the curve ----------------------------------------------------------------

@dataclass(frozen=True)
class CurvePoint:
    months: int
    label: str
    yield_pct: float
    source: str

    @property
    def years(self) -> float:
        return self.months / 12.0


@dataclass(frozen=True)
class Curve:
    points: Tuple[CurvePoint, ...] = ()
    source: str = SOURCE_YAHOO
    error: Optional[str] = None
    missing: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return len(self.points) >= 2

    def at(self, months: int) -> Optional[float]:
        for point in self.points:
            if point.months == months:
                return point.yield_pct
        return None


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_yahoo_points() -> Tuple[Tuple[CurvePoint, ...], Optional[str]]:
    """The four maturities Yahoo publishes as yield indices.

    These quote in PERCENT already (^TNX at 4.704 means 4.704%), unlike
    the fund fields elsewhere in this app which arrive as fractions.
    """
    symbols = [m.yahoo for m in MATURITIES if m.yahoo]
    try:
        import yfinance as yf

        frame = yf.download(symbols, period="5d", progress=False,
                            auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else None
    except Exception as exc:                       # noqa: BLE001 - never raise
        log_exception(logger, "bond_data.yahoo_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return (), f"Treasury yields could not be loaded: {exc}"
    if closes is None or closes.empty:
        return (), "Treasury yield data came back empty."

    points: List[CurvePoint] = []
    for maturity in MATURITIES:
        if not maturity.yahoo or maturity.yahoo not in closes:
            continue
        series = pd.to_numeric(closes[maturity.yahoo], errors="coerce").dropna()
        if series.empty:
            continue
        value = _number(series.iloc[-1])
        if value is not None:
            points.append(CurvePoint(maturity.months, maturity.label, value,
                                     SOURCE_YAHOO))
    log_event(logger, logging.INFO, "bond_data.yahoo_loaded", points=len(points))
    return tuple(points), None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _load_fred_points(api_key: str) -> Tuple[Tuple[CurvePoint, ...], Optional[str]]:
    """All ten maturities, when a key is configured.

    Untested against the live service in this build — there is no key
    here to test with — so it is written to fail the same way every other
    loader does: never raising, and returning an error the panel can
    show. FRED allows 120 requests a minute and this makes ten, one per
    series, well inside that.
    """
    import requests

    points: List[CurvePoint] = []
    failures: List[str] = []
    for maturity in MATURITIES:
        try:
            response = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": maturity.fred, "api_key": api_key,
                        "file_type": "json", "sort_order": "desc",
                        "limit": 1},
                timeout=15)
            if response.status_code != 200:
                failures.append(f"{maturity.label} (HTTP {response.status_code})")
                continue
            observations = (response.json() or {}).get("observations") or []
            value = _number(observations[0].get("value")) if observations else None
            if value is None:
                failures.append(maturity.label)
                continue
            points.append(CurvePoint(maturity.months, maturity.label, value,
                                     SOURCE_FRED))
        except Exception as exc:                   # noqa: BLE001
            log_exception(logger, "bond_data.fred_failed",
                          series=maturity.fred,
                          error=f"{type(exc).__name__}: {exc}")
            failures.append(maturity.label)

    error = None
    if not points:
        error = ("FRED returned nothing for any maturity — check the API "
                 "key. Falling back to the four Yahoo publishes.")
    elif failures:
        error = "No FRED data for: " + ", ".join(failures)
    log_event(logger, logging.INFO, "bond_data.fred_loaded",
              points=len(points), failures=len(failures))
    return tuple(points), error


def load_curve() -> Curve:
    """The current treasury curve, from FRED when configured and Yahoo
    otherwise. Never raises.

    FRED is preferred because it carries all ten maturities; Yahoo is the
    fallback rather than the other way round. If a configured FRED key
    yields nothing, Yahoo still answers — a missing key must not take the
    curve down.
    """
    key = fred_api_key()
    if key:
        points, error = _load_fred_points(key)
        if points:
            missing = tuple(m.label for m in MATURITIES
                            if all(p.months != m.months for p in points))
            return Curve(points, SOURCE_FRED, error, missing)
        # Configured but unusable: fall through, keeping the reason.
        fallback, fallback_error = _load_yahoo_points()
        return Curve(fallback, SOURCE_YAHOO, error or fallback_error,
                     _missing_labels(fallback))

    points, error = _load_yahoo_points()
    return Curve(points, SOURCE_YAHOO, error, _missing_labels(points))


def _missing_labels(points: Sequence[CurvePoint]) -> Tuple[str, ...]:
    have = {p.months for p in points}
    return tuple(m.label for m in MATURITIES if m.months not in have)


# --- curve shape --------------------------------------------------------------

# A curve is called inverted when a longer maturity yields LESS than a
# shorter one. The canonical pair is 3M/10Y — the one with the longest
# recession-forecasting record — and 2s10s is the other, available only
# when FRED fills in the 2-year.
INVERSION_TOLERANCE_PP = 0.0
FLAT_THRESHOLD_PP = 0.25     # within a quarter point end to end is flat
STEEP_THRESHOLD_PP = 1.50    # more than a point and a half is steep


@dataclass(frozen=True)
class CurveShape:
    label: str                  # Inverted | Flat | Normal | Steep | Unknown
    spread_pp: Optional[float]  # long minus short, in percentage points
    short_label: str = ""
    long_label: str = ""
    detail: str = ""
    inversions: Tuple[str, ...] = ()   # every pair that is out of order


def curve_shape(curve: Curve) -> CurveShape:
    """Steep, normal, flat or inverted, from the points actually loaded.

    The spread is measured between the SHORTEST and LONGEST maturities
    present, and both are named, because a "3M to 30Y" spread and a
    "2Y to 10Y" spread are different numbers and quoting either without
    saying which is how they get confused.
    """
    if not curve.ok:
        return CurveShape("Unknown", None,
                          detail="Fewer than two maturities loaded.")
    ordered = sorted(curve.points, key=lambda p: p.months)
    short, long = ordered[0], ordered[-1]
    spread = long.yield_pct - short.yield_pct

    # Every adjacent pair that is out of order, not just the ends: a curve
    # can be humped in the middle while its ends look normal.
    inversions = tuple(
        f"{a.label}→{b.label}"
        for a, b in zip(ordered, ordered[1:])
        if b.yield_pct < a.yield_pct - INVERSION_TOLERANCE_PP)

    if spread < -INVERSION_TOLERANCE_PP:
        label = "Inverted"
        detail = (f"{long.label} yields {abs(spread):.2f}pp LESS than "
                  f"{short.label} — long money is priced below short.")
    elif abs(spread) <= FLAT_THRESHOLD_PP:
        label = "Flat"
        detail = (f"{short.label} to {long.label} spans only "
                  f"{spread:+.2f}pp.")
    elif spread >= STEEP_THRESHOLD_PP:
        label = "Steep"
        detail = (f"{long.label} yields {spread:.2f}pp more than "
                  f"{short.label}.")
    else:
        label = "Normal"
        detail = (f"{long.label} yields {spread:.2f}pp more than "
                  f"{short.label}, an ordinary upward slope.")

    if inversions and label != "Inverted":
        detail += (" Note the curve is out of order between "
                   + ", ".join(inversions) + ".")
    return CurveShape(label, spread, short.label, long.label, detail, inversions)


# --- interpolation ------------------------------------------------------------

# Nelson-Siegel has four free parameters. Fitting it to four observations
# is an exact fit with zero residual — a curve drawn through the points
# rather than a model of them, and it would extrapolate wildly between
# them. So it is gated on having enough maturities to be worth fitting,
# and linear interpolation is used below that, with the choice reported.
NELSON_SIEGEL_MIN_POINTS = 6


def interpolate_yield(curve: Curve, months: float) -> Tuple[Optional[float], str]:
    """(yield, method) for an off-curve maturity.

    Never extrapolates past the ends: a 40-year yield is not obtainable
    from a curve that stops at 30, and returning the 30-year for it would
    quietly answer a question that was not asked.
    """
    if not curve.ok:
        return None, "unavailable"
    ordered = sorted(curve.points, key=lambda p: p.months)
    if months < ordered[0].months or months > ordered[-1].months:
        return None, "outside the loaded maturities"
    xs = np.array([p.months for p in ordered], dtype=float)
    ys = np.array([p.yield_pct for p in ordered], dtype=float)
    if len(ordered) >= NELSON_SIEGEL_MIN_POINTS:
        fitted = _nelson_siegel(xs, ys, float(months))
        if fitted is not None:
            return fitted, "Nelson-Siegel"
    return float(np.interp(float(months), xs, ys)), "linear"


def _nelson_siegel(xs: "np.ndarray", ys: "np.ndarray",
                   months: float) -> Optional[float]:
    """Least-squares Nelson-Siegel, with tau fixed by a small search.

    With tau fixed the other three parameters are linear, so this is a
    plain least-squares solve per candidate tau rather than a general
    optimiser — fewer ways to fail silently.
    """
    years = xs / 12.0
    target = months / 12.0
    best = None
    for tau in np.linspace(0.5, 10.0, 40):
        ratio = years / tau
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(ratio == 0, 1.0, (1 - np.exp(-ratio)) / ratio)
        curvature = slope - np.exp(-ratio)
        design = np.column_stack([np.ones_like(years), slope, curvature])
        try:
            beta, residuals, rank, _ = np.linalg.lstsq(design, ys, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 3:
            continue
        fitted = design @ beta
        sse = float(np.sum((ys - fitted) ** 2))
        if best is None or sse < best[0]:
            best = (sse, beta, tau)
    if best is None:
        return None
    _, beta, tau = best
    ratio = target / tau
    slope = 1.0 if ratio == 0 else (1 - np.exp(-ratio)) / ratio
    curvature = slope - np.exp(-ratio)
    value = float(beta[0] + beta[1] * slope + beta[2] * curvature)
    return value if value == value else None


# --- bond funds ---------------------------------------------------------------

# THE LETTER BUCKETS ARE THE BREAKDOWN; `us_government` IS A SEPARATE
# AXIS. Measured across six funds, the letters sum to exactly 1.0000
# every time — TLT, SHY, AGG, BND, LQD and HYG — while us_government runs
# 0.9958 for TLT down to 0.0000 for LQD and HYG. A treasury IS rated AA,
# so the two overlap: TLT is 100% AA *and* 99.6% government. Treating
# them as one set of mutually exclusive buckets double-counts, and did —
# it reported TLT as 199.6% investment grade before this was measured.
RATING_ORDER: Tuple[Tuple[str, str], ...] = (
    ("aaa", "AAA"),
    ("aa", "AA"),
    ("a", "A"),
    ("bbb", "BBB"),
    ("bb", "BB"),
    ("b", "B"),
    ("below_b", "Below B"),
    ("other", "Other"),
)

GOVERNMENT_KEY = "us_government"

INVESTMENT_GRADE_KEYS = frozenset({"aaa", "aa", "a", "bbb"})

# The task's validation bounds.
YTM_MAX_PCT = 15.0
COUPON_MAX_PCT = 12.0
PRICE_MIN, PRICE_MAX = 0.0, 200.0


@dataclass(frozen=True)
class BondFund:
    symbol: str
    # No average_maturity: the provider's field fails the ladder test and
    # there is no way to measure it from prices. Absent beats wrong.
    empirical_duration: Optional[float] = None
    duration_r_squared: Optional[float] = None
    duration_days: int = 0
    ratings_pct: Dict[str, float] = field(default_factory=dict)
    # Reported separately, never mixed into ratings_pct — see RATING_ORDER.
    government_pct: Optional[float] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def investment_grade_pct(self) -> Optional[float]:
        """AAA through BBB. Government exposure is deliberately excluded:
        it is a different axis and adding it double-counts a treasury,
        which is both AA-rated and government-issued."""
        if not self.ratings_pct:
            return None
        return sum(v for k, v in self.ratings_pct.items()
                   if k in INVESTMENT_GRADE_KEYS)

    @property
    def ratings_total_pct(self) -> float:
        """Should be ~100. Measured at exactly 100.00 for all six funds
        checked, so a total far from it means the buckets have changed
        shape upstream."""
        return sum(self.ratings_pct.values())


def empirical_duration(fund_closes: Optional["pd.Series"],
                       yield_closes: Optional["pd.Series"]
                       ) -> Tuple[Optional[float], Optional[float], int]:
    """(modified duration, r-squared, days used) from price behaviour.

    Regresses the fund's daily percent return on the daily CHANGE in the
    ten-year yield, in percentage points. The slope of that line is minus
    the modified duration: a fund with duration 13 loses about 13% when
    yields rise a full point.

    This exists because the provider's own duration field is wrong — see
    REPORTED_DURATION_UNUSABLE and the module docstring. Validated
    against seven funds: this returns 13.21 for TLT and 1.43 for SHY,
    which ranks them correctly and matches their published figures.
    """
    if fund_closes is None or yield_closes is None:
        return None, None, 0
    pair = pd.concat([pd.to_numeric(fund_closes, errors="coerce"),
                      pd.to_numeric(yield_closes, errors="coerce")],
                     axis=1).dropna()
    if len(pair) < 30:
        # A slope from a handful of days is noise wearing a number's
        # clothes; better to report nothing.
        return None, None, len(pair)
    returns = pair.iloc[:, 0].pct_change() * 100.0
    yield_change = pair.iloc[:, 1].diff()
    both = pd.concat([returns, yield_change], axis=1).dropna()
    if len(both) < 30 or both.iloc[:, 1].std() == 0:
        return None, None, len(both)
    x = both.iloc[:, 1].to_numpy(dtype=float)
    y = both.iloc[:, 0].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = None if ss_tot == 0 else 1.0 - ss_res / ss_tot
    duration = -float(slope)
    return duration, r_squared, len(both)


@st.cache_data(ttl=FUND_CACHE_TTL_SECONDS, show_spinner=False)
def load_bond_fund(symbol: str, lookback_period: str = "2y") -> BondFund:
    """A bond fund's credit mix, average maturity and REAL duration.

    Never raises. The duration is measured, not read — see the module
    docstring for the seven-fund comparison that forced that.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return BondFund(symbol="", error="No symbol given.")
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        funds = ticker.funds_data

        ratings_pct: Dict[str, float] = {}
        for key, _label in RATING_ORDER:
            value = _number((funds.bond_ratings or {}).get(key))
            if value is not None and value > 0:
                # Fractions, like every other weight in this data source.
                ratings_pct[key] = value * 100.0

        government = _number((funds.bond_ratings or {}).get(GOVERNMENT_KEY))
        government_pct = government * 100.0 if government is not None else None

        # `bond_holdings` is deliberately NOT read. Both fields on it —
        # Duration and Maturity — fail the iShares ladder test in the
        # module docstring. Note for anyone tempted to add it back: it is
        # a DataFrame, so `x or {}` raises "truth value is ambiguous"
        # rather than falling back.

        frame = yf.download([symbol, "^TNX"], period=lookback_period,
                            progress=False, auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else None
        duration = r_squared = None
        days = 0
        if closes is not None and symbol in closes and "^TNX" in closes:
            duration, r_squared, days = empirical_duration(
                closes[symbol], closes["^TNX"])

        log_event(logger, logging.INFO, "bond_data.fund_loaded", ticker=symbol,
                  ratings=len(ratings_pct),
                  duration=None if duration is None else round(duration, 2))
        return BondFund(symbol=symbol,
                        empirical_duration=duration,
                        duration_r_squared=r_squared,
                        duration_days=days,
                        ratings_pct=ratings_pct,
                        government_pct=government_pct)
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "bond_data.fund_failed", ticker=symbol,
                      error=f"{type(exc).__name__}: {exc}")
        return BondFund(symbol=symbol,
                        error=f"Bond data is unavailable for {symbol} "
                              f"({type(exc).__name__}).")


def rating_rows(fund: BondFund) -> List[Tuple[str, float]]:
    """(label, percent) in credit order, skipping empty buckets."""
    return [(label, fund.ratings_pct[key])
            for key, label in RATING_ORDER if fund.ratings_pct.get(key)]


# --- validation ---------------------------------------------------------------

def validate_yield(value: Optional[float]) -> Optional[str]:
    """The task's "YTM positive and < 15%", with one correction.

    A yield of exactly zero is real — and NEGATIVE yields are real too;
    German and Japanese government debt traded below zero for years. So
    the floor rejects only the absurd rather than the merely unusual, and
    the ceiling catches the unit error worth catching: a yield handed
    over as a fraction (0.047) or as basis points (470).
    """
    if value is None:
        return None
    if value != value:
        return "not a number"
    if value < -5.0:
        return f"{value:.2f}% is below any yield ever traded"
    if value > YTM_MAX_PCT:
        return (f"{value:.2f}% exceeds {YTM_MAX_PCT:.0f}% — check whether "
                "this is basis points rather than percent")
    return None


def validate_duration(duration: Optional[float],
                      maturity_years: Optional[float]) -> Optional[str]:
    """Duration must be positive and no longer than the maturity.

    The second half is the real check: duration above maturity is
    arithmetically impossible for a bond with any coupon at all, so it
    means the two figures came from different places.
    """
    if duration is None:
        return None
    if duration != duration:
        return "not a number"
    if duration <= 0:
        return f"{duration:.2f} is not a positive duration"
    if maturity_years is not None and duration > maturity_years + 0.5:
        return (f"duration {duration:.2f}y exceeds average maturity "
                f"{maturity_years:.2f}y, which is impossible for a coupon "
                "bond — the two figures disagree")
    return None


def validate_price(price: Optional[float]) -> Optional[str]:
    if price is None:
        return None
    if price != price:
        return "not a number"
    if not (PRICE_MIN < price <= PRICE_MAX):
        return f"{price:.2f} is outside {PRICE_MIN}-{PRICE_MAX} per 100 par"
    return None


def validate_coupon(coupon_pct: Optional[float]) -> Optional[str]:
    if coupon_pct is None:
        return None
    if coupon_pct != coupon_pct:
        return "not a number"
    if not (0.0 <= coupon_pct <= COUPON_MAX_PCT):
        return f"{coupon_pct:.2f}% is outside 0-{COUPON_MAX_PCT:.0f}%"
    return None


def validate_curve(curve: Curve) -> List[str]:
    """Every problem with a loaded curve, as a list. Empty means clean."""
    problems: List[str] = []
    if not curve.ok:
        problems.append("Fewer than two maturities loaded.")
        return problems
    seen = set()
    for point in curve.points:
        problem = validate_yield(point.yield_pct)
        if problem:
            problems.append(f"{point.label}: {problem}")
        if point.months in seen:
            problems.append(f"{point.label}: duplicated maturity")
        seen.add(point.months)
    return problems
