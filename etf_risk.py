"""Fund risk: tracking, capture, concentration, liquidity and stress.

TRACKING ERROR IS BUILDABLE AFTER ALL, AND EARLIER PHASES SAID IT WAS
NOT. That claim needs correcting rather than quietly dropping. What is
genuinely unavailable is the fund's OWN STATED benchmark — this data
source returns no mapping from a fund to the index its prospectus names,
and PHASE 1.2 was right that inventing one would be a different number
under the same name. But the app has always had a user-chosen Market
Benchmark in the sidebar, and tracking error against a benchmark the
READER picked is a real, well-defined figure as long as the screen says
whose benchmark it is. So it is computed here, labelled with the symbol
it was measured against, and never presented as the fund's official
tracking error.

THE DISCRIMINATING TEST, measured over three years of daily returns
against ^GSPC: S&P 500 index funds track tightly — VOO 0.78%, IVV 0.94%,
SPY 1.15% — while funds tracking something else do not: QQQ 7.79%,
TLT 18.74%, ARKK 29.35%. A tracking-error implementation that cannot
separate those three from those three is not measuring tracking.

DAILY TRACKING ERROR AND CUMULATIVE DRAG ARE DIFFERENT THINGS, and the
difference caught me out. SPY's 1.15% against ^GSPC looks far too high
for a fund that famously tracks to a few basis points, and the obvious
explanation — that ^GSPC is a price index while SPY's series is
total-return adjusted, so dividends leak into the difference — is WRONG:
measured against ^SP500TR, the total-return index, the tracking error is
1.148%, essentially identical. The daily figure is dominated by the
closing-price mismatch between an ETF's last trade and an index's
official close, which is noise rather than a management failure.

The CUMULATIVE gap is where the real tracking story is: over the same
three years SPY returned 80.62% against ^SP500TR's 81.23%, a 0.61pp
shortfall — about 0.20% a year against a 0.0945% fee. Both numbers are
reported, because either alone misleads.

CONCENTRATION IS MEASURED OVER THE TOP TEN AND SAYS SO. The Herfindahl
index is the sum of squared weights, so the holdings this source omits —
37-46% of a typical fund, in many small positions — contribute very
little to it: a 0.1% position adds 0.000001. The top-ten figure is
therefore a close LOWER BOUND on the fund's true concentration rather
than a wild guess, which is worth stating precisely rather than either
hiding it or pretending it is the whole fund.

WHAT IS NOT AVAILABLE: bid-ask spread for most funds. Measured in PHASE
1.7, bid and ask are both 0.0 for six of ten funds INCLUDING SPY, so the
liquidity reading falls back to dollar volume against assets, which is
reported for everything.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from logging_setup import get_logger, log_event

logger = get_logger("etf_risk")

TRADING_DAYS = 252
MIN_DAYS = 60

# Tracking bands, from the measurement in the module docstring. A fund
# tracking a broad index lands under 2%; anything above 10% is not
# tracking that benchmark at all.
TIGHT_TRACKING_PCT = 2.0
LOOSE_TRACKING_PCT = 10.0

BENCHMARK_IS_YOUR_CHOICE = (
    "Measured against the Market Benchmark set in the sidebar, not "
    "against the index this fund's prospectus names — that mapping is "
    "not in this data source. A large figure against a benchmark the "
    "fund never claimed to track says nothing about the fund."
)

PRICE_INDEX_FLATTERS_A_FUND = (
    "A positive gap against a PRICE index is mostly dividends, not "
    "outperformance: ^GSPC excludes them and a fund's series does not. "
    "Measured over three years, SPY shows +6.44% against ^GSPC but "
    "-0.61% against ^SP500TR, the same index computed on a total-return "
    "basis. Compare against a total-return benchmark before reading a "
    "gap as skill."
)

BID_ASK_MOSTLY_ABSENT = (
    "Bid and ask are not reported for most funds here — measured, both "
    "come back as 0.00 for six of ten checked, including SPY. Where they "
    "are absent, liquidity is read from dollar volume against fund size "
    "instead, which is reported for everything."
)

TOP_TEN_CONCENTRATION_NOTE = (
    "Concentration is computed over the ten holdings this source "
    "discloses, which are 37-46% of a typical fund. Because the "
    "Herfindahl index squares each weight, the many small positions left "
    "out contribute almost nothing to it — a 0.1% holding adds 0.000001 "
    "— so this is a close lower bound rather than a guess."
)


def _clean(series: Optional["pd.Series"]) -> Optional["pd.Series"]:
    if series is None:
        return None
    out = pd.to_numeric(series, errors="coerce").dropna()
    return out if len(out) else None


def _aligned_returns(fund_closes, benchmark_closes
                     ) -> Optional[Tuple["pd.Series", "pd.Series"]]:
    """Daily returns on the dates BOTH series have.

    Aligning on the intersection matters: a fund listed on a venue with
    different holidays would otherwise contribute a spurious two-day
    return against the benchmark's one-day return, which reads as
    tracking error that is really a calendar mismatch.
    """
    fund, bench = _clean(fund_closes), _clean(benchmark_closes)
    if fund is None or bench is None:
        return None
    # sort=True explicitly: pandas warns that the default is changing
    # when concatenating DatetimeIndexes, and a silently UNSORTED join
    # would put the returns in the wrong order — which is not something
    # a tracking-error number would look wrong for.
    pair = pd.concat([fund, bench], axis=1, sort=True).dropna()
    if len(pair) < MIN_DAYS + 1:
        return None
    returns = pair.pct_change().dropna()
    if len(returns) < MIN_DAYS:
        return None
    return returns.iloc[:, 0], returns.iloc[:, 1]


@dataclass(frozen=True)
class TrackingResult:
    benchmark: str
    tracking_error_pct: Optional[float] = None      # annualised
    information_ratio: Optional[float] = None
    fund_total_return_pct: Optional[float] = None
    benchmark_total_return_pct: Optional[float] = None
    cumulative_gap_pct: Optional[float] = None      # fund minus benchmark
    annualised_gap_pct: Optional[float] = None
    days: int = 0

    @property
    def ok(self) -> bool:
        return self.tracking_error_pct is not None

    @property
    def band(self) -> str:
        """Tight / Loose / Not tracking, or Unknown."""
        if self.tracking_error_pct is None:
            return "Unknown"
        if self.tracking_error_pct <= TIGHT_TRACKING_PCT:
            return "Tight"
        if self.tracking_error_pct <= LOOSE_TRACKING_PCT:
            return "Loose"
        return "Not tracking this benchmark"


def tracking(fund_closes, benchmark_closes, benchmark: str = "") -> TrackingResult:
    """Tracking error, information ratio and the cumulative gap.

    Tracking error is the annualised standard deviation of the daily
    return DIFFERENCE — the task's own formula. The cumulative gap is
    reported beside it because the two answer different questions: the
    daily figure is dominated by closing-price mismatch (SPY reads 1.15%
    against an index it tracks to basis points), while the cumulative
    shortfall is the fee and the real drag.
    """
    aligned = _aligned_returns(fund_closes, benchmark_closes)
    if aligned is None:
        return TrackingResult(benchmark=benchmark)
    fund_returns, bench_returns = aligned
    excess = fund_returns - bench_returns
    std = float(excess.std())
    tracking_error = std * np.sqrt(TRADING_DAYS) * 100.0

    # Information ratio: annualised excess return over tracking error.
    # The task divides the excess return by 252, which would give a
    # DAILY excess over an ANNUAL tracking error — two different clocks,
    # the same mistake the bond reference made. Annualising both keeps
    # the ratio dimensionless.
    information = None
    if std > 0:
        information = float(excess.mean() * TRADING_DAYS) / (std * np.sqrt(TRADING_DAYS))

    fund_total = (float((1 + fund_returns).prod()) - 1.0) * 100.0
    bench_total = (float((1 + bench_returns).prod()) - 1.0) * 100.0
    gap = fund_total - bench_total
    years = len(fund_returns) / TRADING_DAYS
    annualised_gap = None
    if years > 0:
        fund_annual = (1 + fund_total / 100.0) ** (1 / years) - 1
        bench_annual = (1 + bench_total / 100.0) ** (1 / years) - 1
        annualised_gap = (fund_annual - bench_annual) * 100.0

    log_event(logger, logging.INFO, "etf_risk.tracking",
              benchmark=benchmark, te=round(tracking_error, 3),
              days=len(fund_returns))
    return TrackingResult(
        benchmark=benchmark,
        tracking_error_pct=tracking_error,
        information_ratio=information,
        fund_total_return_pct=fund_total,
        benchmark_total_return_pct=bench_total,
        cumulative_gap_pct=gap,
        annualised_gap_pct=annualised_gap,
        days=len(fund_returns))


@dataclass(frozen=True)
class CaptureRatios:
    up_pct: Optional[float] = None
    down_pct: Optional[float] = None
    up_days: int = 0
    down_days: int = 0

    @property
    def ok(self) -> bool:
        return self.up_pct is not None or self.down_pct is not None

    @property
    def asymmetry(self) -> Optional[float]:
        """Up capture minus down capture. Positive is the good kind: more
        of the rise than of the fall."""
        if self.up_pct is None or self.down_pct is None:
            return None
        return self.up_pct - self.down_pct


def capture_ratios(fund_closes, benchmark_closes) -> CaptureRatios:
    """How much of the benchmark's up days and down days the fund catches.

    Measured on the days the BENCHMARK rose or fell, not the days the
    fund did — that is what makes it a capture ratio rather than a
    restatement of the fund's own returns.
    """
    aligned = _aligned_returns(fund_closes, benchmark_closes)
    if aligned is None:
        return CaptureRatios()
    fund_returns, bench_returns = aligned
    up_mask, down_mask = bench_returns > 0, bench_returns < 0

    def ratio(mask) -> Optional[float]:
        if mask.sum() < 5:
            return None
        denominator = float(bench_returns[mask].mean())
        if denominator == 0:
            return None
        return (float(fund_returns[mask].mean()) / denominator) * 100.0

    return CaptureRatios(up_pct=ratio(up_mask), down_pct=ratio(down_mask),
                         up_days=int(up_mask.sum()), down_days=int(down_mask.sum()))


# --- concentration ------------------------------------------------------------

@dataclass(frozen=True)
class Concentration:
    herfindahl: Optional[float] = None          # 0-1, over the disclosed slice
    top_ten_pct: Optional[float] = None
    max_holding_pct: Optional[float] = None
    max_holding_symbol: str = ""
    effective_holdings: Optional[float] = None  # 1 / HHI
    disclosed_count: int = 0

    @property
    def ok(self) -> bool:
        return self.herfindahl is not None


def concentration(holdings: Sequence) -> Concentration:
    """Herfindahl and friends over the disclosed holdings.

    HHI is computed on FRACTIONS, not percents: squaring a percent gives
    a number a hundred times too large and breaks the 0-1 scale the index
    is defined on. `effective_holdings` is 1/HHI — the number of
    equally-weighted positions that would be this concentrated, which is
    the reading most people can act on.
    """
    weights = []
    top_symbol, top_weight = "", 0.0
    for holding in holdings or ():
        weight = getattr(holding, "weight_pct", None)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            continue
        if weight != weight or weight <= 0:
            continue
        weights.append(weight)
        if weight > top_weight:
            top_weight = weight
            top_symbol = getattr(holding, "symbol", "") or ""
    if not weights:
        return Concentration()
    fractions = [w / 100.0 for w in weights]
    hhi = float(sum(f * f for f in fractions))
    return Concentration(
        herfindahl=hhi,
        top_ten_pct=float(sum(weights)),
        max_holding_pct=top_weight,
        max_holding_symbol=top_symbol,
        effective_holdings=(1.0 / hhi) if hhi > 0 else None,
        disclosed_count=len(weights))


# --- liquidity ----------------------------------------------------------------

@dataclass(frozen=True)
class Liquidity:
    spread_bps: Optional[float] = None
    dollar_volume: Optional[float] = None
    turnover_pct: Optional[float] = None    # daily dollar volume / AUM
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.dollar_volume is not None or self.spread_bps is not None


def liquidity(price: Optional[float], volume: Optional[float],
              assets: Optional[float], bid: Optional[float] = None,
              ask: Optional[float] = None) -> Liquidity:
    """Spread where it is reported, and dollar volume against size always.

    The spread is quoted in BASIS POINTS because that is how a trading
    cost is quoted, and a spread in percent gets misread by a hundred.
    """
    spread_bps = None
    if bid and ask and bid > 0 and ask > 0 and ask >= bid:
        mid = (ask + bid) / 2.0
        if mid > 0:
            spread_bps = ((ask - bid) / mid) * 10000.0

    dollar_volume = None
    if price and volume and price > 0 and volume > 0:
        dollar_volume = float(price) * float(volume)

    turnover = None
    if dollar_volume and assets and assets > 0:
        turnover = (dollar_volume / float(assets)) * 100.0

    if spread_bps is None and dollar_volume is None:
        detail = "Neither a quoted spread nor a volume was reported."
    elif spread_bps is None:
        detail = ("No quoted spread reported, which is the normal case "
                  "here. Dollar volume is the liquidity reading instead.")
    else:
        detail = f"Quoted spread {spread_bps:.1f}bp."
    return Liquidity(spread_bps=spread_bps, dollar_volume=dollar_volume,
                     turnover_pct=turnover, detail=detail)


# --- historical extremes and stress -------------------------------------------

@dataclass(frozen=True)
class Extremes:
    worst_day_pct: Optional[float] = None
    worst_day_date: Optional[str] = None
    worst_month_pct: Optional[float] = None
    worst_month_label: Optional[str] = None
    days: int = 0


def historical_extremes(closes) -> Extremes:
    """The worst single day and worst calendar month actually suffered.

    A historical extreme is not a forecast, but it is the one risk figure
    that needs no model — it happened.
    """
    series = _clean(closes)
    if series is None or len(series) < 2:
        return Extremes()
    daily = series.pct_change().dropna()
    if daily.empty:
        return Extremes()
    worst_day = float(daily.min())
    worst_date = str(daily.idxmin().date()) if hasattr(daily.idxmin(), "date") else None

    worst_month_pct = worst_month_label = None
    try:
        monthly = series.resample("ME").last().pct_change().dropna()
        if len(monthly):
            worst_month_pct = float(monthly.min()) * 100.0
            worst_month_label = monthly.idxmin().strftime("%Y-%m")
    except Exception:                              # noqa: BLE001
        pass
    return Extremes(worst_day_pct=worst_day * 100.0, worst_day_date=worst_date,
                    worst_month_pct=worst_month_pct,
                    worst_month_label=worst_month_label, days=len(daily))


@dataclass(frozen=True)
class SectorShock:
    sector: str
    weight_pct: float
    shock_pct: float
    fund_impact_pct: float


def sector_stress(sector_weights: Optional[Dict[str, float]],
                  shock_pct: float,
                  labels: Optional[Dict[str, str]] = None) -> List[SectorShock]:
    """What a shock to each sector does to the whole fund.

    The task asks "what if tech, the biggest holding, fell 30%". The
    answer is weight x shock, which is the ONLY honest first-order answer
    — it assumes every other sector holds still, and a real sector rout
    never does. Sorted by impact so the sector that actually matters
    leads.

    Weights arrive as FRACTIONS from this data source (0.374 = 37.4%),
    the same convention etf_technicals documents.
    """
    if not sector_weights:
        return []
    labels = labels or {}
    rows: List[SectorShock] = []
    for key, raw in sector_weights.items():
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            continue
        if weight != weight or weight <= 0:
            continue
        weight_pct = weight * 100.0
        rows.append(SectorShock(
            sector=labels.get(key, key.replace("_", " ").title()),
            weight_pct=weight_pct,
            shock_pct=shock_pct,
            fund_impact_pct=weight * shock_pct))
    rows.sort(key=lambda r: r.fund_impact_pct)
    return rows


def total_shock_impact(rows: Sequence[SectorShock]) -> Optional[float]:
    """Every sector shocked at once — the whole-fund figure."""
    if not rows:
        return None
    return float(sum(r.fund_impact_pct for r in rows))
