"""Actual holdings, and portfolio return measured against a benchmark.

WHAT A HOLDING KNOWS, AND WHY IT MATTERS. Ticker, share count, cost basis
per share, and PURCHASE DATE. That last field is the one carrying weight:
with only ticker+shares there is no way to know what was held a year ago,
so a "return over time" chart would have to assume today's basket was
held for the whole window. That back-projects today's winners onto the
past and flatters the result — a portfolio that added NVDA last month
would appear to have owned it through the run-up. Recording when each
position started lets each holding enter the series on its own date.

THE TWO RETURN NUMBERS, AND WHY BOTH ARE SHOWN.

  Time-weighted (TWR) chains daily returns and is unaffected by when
  money went in. It answers "did my picks beat the index" and is the
  only figure it is fair to put next to the S&P 500's return, which is
  why fund factsheets quote it.

  Money-weighted (IRR) is the discount rate that makes the actual cash
  flows sum to today's value. It answers "what did my money actually
  do", including whether buying more happened at good or bad moments.

They can differ substantially, and each is misleading in the other's
context. Someone who bought heavily right before a rally has a great IRR
and an unremarkable TWR; reporting only IRR against an index would claim
skill that was timing, and reporting only TWR would ignore what actually
happened to their money. So both appear, labelled.

TWR IS COMPUTED BY DAILY VALUATION, not Modified Dietz. Each purchase is
an external cash flow, and the daily formula

    r(t) = (V(t) - V(t-1) - C(t)) / V(t-1)

removes it exactly rather than approximating it as Dietz does. With
daily prices available there is no reason to accept the approximation.

NO PRICES ARE INVENTED. A holding whose price history can't be fetched
is excluded from the series and named in the result, never silently
dropped and never zero-filled — a ticker missing from a portfolio chart
reads as "it didn't move" exactly like it does in the digest.
"""
import datetime
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PORTFOLIO
from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("portfolio_holdings")


@dataclass(frozen=True)
class Holding:
    ticker: str
    shares: float
    cost_basis: float               # per share, in the position's currency
    purchase_date: datetime.date

    @property
    def cost_total(self) -> float:
        return self.shares * self.cost_basis


@dataclass(frozen=True)
class PortfolioStore:
    """active/portfolios mirrors WatchlistStore deliberately — see
    config.PortfolioConfig. Multiple portfolios become a UI change
    rather than a migration."""
    active: str = PORTFOLIO.default_portfolio_name
    portfolios: Dict[str, Tuple[Holding, ...]] = field(default_factory=dict)

    def holdings(self, name: Optional[str] = None) -> Tuple[Holding, ...]:
        return self.portfolios.get(name or self.active, ())


@dataclass(frozen=True)
class HoldingPerformance:
    """One position's contribution, priced as of the period end."""
    ticker: str
    shares: float
    cost_basis: float
    purchase_date: datetime.date
    current_price: Optional[float] = None
    unavailable: str = ""

    @property
    def ok(self) -> bool:
        return self.current_price is not None

    @property
    def market_value(self) -> Optional[float]:
        return None if self.current_price is None else self.shares * self.current_price

    @property
    def cost_total(self) -> float:
        return self.shares * self.cost_basis

    @property
    def gain(self) -> Optional[float]:
        mv = self.market_value
        return None if mv is None else mv - self.cost_total

    @property
    def gain_pct(self) -> Optional[float]:
        if self.market_value is None or self.cost_total == 0:
            return None
        return (self.gain / self.cost_total) * 100


@dataclass(frozen=True)
class PortfolioPerformance:
    period_start: datetime.date
    period_end: datetime.date
    value_series: pd.Series                     # daily portfolio market value
    benchmark_series: Optional[pd.Series] = None  # rebased to the portfolio's first value
    holdings: Tuple[HoldingPerformance, ...] = ()
    twr_pct: Optional[float] = None
    mwr_pct: Optional[float] = None
    benchmark_return_pct: Optional[float] = None
    excluded: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def market_value(self) -> float:
        return float(sum(h.market_value or 0.0 for h in self.holdings))

    @property
    def cost_total(self) -> float:
        return float(sum(h.cost_total for h in self.holdings if h.ok))

    @property
    def total_gain(self) -> float:
        return self.market_value - self.cost_total

    @property
    def excess_vs_benchmark_pct(self) -> Optional[float]:
        """TWR minus the benchmark's return — the like-for-like gap.

        Deliberately built from TWR and not from money-weighted return:
        the whole reason TWR exists here is that it is the figure it is
        fair to difference against an index.
        """
        if self.twr_pct is None or self.benchmark_return_pct is None:
            return None
        return self.twr_pct - self.benchmark_return_pct


# --- persistence --------------------------------------------------------------

def _store_path() -> Path:
    return store_path(PORTFOLIO.store_filename)


def _parse_date(value) -> Optional[datetime.date]:
    if isinstance(value, datetime.date):
        return value
    try:
        return datetime.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def load_store(path: Optional[Path] = None) -> PortfolioStore:
    """Never raises. A malformed holding is dropped rather than
    discarding the whole portfolio — but note a dropped holding silently
    changes the reported return, so each one is logged."""
    path = path or _store_path()
    if not path.exists():
        return PortfolioStore()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "portfolio.store_corrupt", section="portfolio")
        return PortfolioStore()
    if not isinstance(raw, dict):
        return PortfolioStore()

    portfolios: Dict[str, Tuple[Holding, ...]] = {}
    for name, items in (raw.get("portfolios") or {}).items():
        if not isinstance(name, str) or not isinstance(items, list):
            continue
        good: List[Holding] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).strip().upper()
            purchase = _parse_date(item.get("purchase_date"))
            try:
                shares = float(item.get("shares"))
                cost = float(item.get("cost_basis"))
            except (TypeError, ValueError):
                log_event(logger, logging.WARNING, "portfolio.holding_dropped", reason="unparseable")
                continue
            if not ticker or purchase is None or shares <= 0 or cost < 0:
                log_event(logger, logging.WARNING, "portfolio.holding_dropped", reason="invalid")
                continue
            good.append(Holding(ticker, shares, cost, purchase))
        portfolios[name] = tuple(good)

    active = str(raw.get("active") or PORTFOLIO.default_portfolio_name)
    if portfolios and active not in portfolios:
        active = next(iter(portfolios))
    return PortfolioStore(active=active, portfolios=portfolios)


def save_store(store: PortfolioStore, path: Optional[Path] = None) -> None:
    path = path or _store_path()
    payload = {
        "active": store.active,
        "portfolios": {
            name: [{
                "ticker": h.ticker, "shares": h.shares,
                "cost_basis": h.cost_basis, "purchase_date": h.purchase_date.isoformat(),
            } for h in holdings]
            for name, holdings in store.portfolios.items()
        },
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


# --- editing ------------------------------------------------------------------

def add_holding(store: PortfolioStore, ticker: str, shares: float, cost_basis: float,
                purchase_date: datetime.date,
                portfolio: Optional[str] = None) -> Tuple[PortfolioStore, Optional[str]]:
    """Add a position. Returns (store, error)."""
    name = portfolio or store.active
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return store, "Enter a ticker."
    try:
        shares = float(shares)
        cost_basis = float(cost_basis)
    except (TypeError, ValueError):
        return store, "Shares and cost basis must be numbers."
    if shares <= 0:
        return store, "Shares must be greater than zero."
    if cost_basis < 0:
        return store, "Cost basis can't be negative."
    if purchase_date is None:
        return store, "Pick a purchase date."
    if purchase_date > datetime.date.today():
        return store, "That purchase date is in the future."

    existing = store.portfolios.get(name, ())
    if len(existing) >= PORTFOLIO.max_holdings:
        return store, f"This portfolio is full ({PORTFOLIO.max_holdings} holdings)."

    holding = Holding(ticker, shares, cost_basis, purchase_date)
    return replace(store, portfolios={**store.portfolios, name: existing + (holding,)}), None


def remove_holding(store: PortfolioStore, index: int,
                   portfolio: Optional[str] = None) -> PortfolioStore:
    """Removed by POSITION, not by ticker — the same ticker can legitimately
    be held twice at different cost bases and dates, and matching on ticker
    would delete the wrong lot."""
    name = portfolio or store.active
    existing = store.portfolios.get(name, ())
    if not 0 <= index < len(existing):
        return store
    remaining = existing[:index] + existing[index + 1:]
    return replace(store, portfolios={**store.portfolios, name: remaining})


# --- managing several portfolios ----------------------------------------------
#
# Mirrors watchlist_panel's create/rename/delete/set_active API on purpose:
# the two are the same shape of problem, and a reader who knows one should
# not have to learn a second set of conventions. Every operation returns
# (store, reason) UNCHANGED on refusal, so the caller can say why a click
# did nothing rather than leaving it silently inert.

def portfolio_names(store: PortfolioStore) -> Tuple[str, ...]:
    """Names in insertion order, guaranteed non-empty.

    Always returns at least the default name even for an empty store,
    because the UI renders a selectbox from this and an empty options
    list has nothing to select.
    """
    return tuple(store.portfolios) or (PORTFOLIO.default_portfolio_name,)


def create_portfolio(store: PortfolioStore, name: str) -> Tuple[PortfolioStore, Optional[str]]:
    """Add an empty portfolio and make it active — you just created it to
    put something in it."""
    name = (name or "").strip()
    if not name:
        return store, "Enter a name first."
    if len(name) > PORTFOLIO.max_name_chars:
        return store, f"Names are capped at {PORTFOLIO.max_name_chars} characters."
    if name in store.portfolios:
        return store, f'A portfolio named "{name}" already exists.'
    if len(store.portfolios) >= PORTFOLIO.max_portfolios:
        return store, (
            f"Portfolio limit reached ({PORTFOLIO.max_portfolios} max) — delete one first."
        )
    return replace(store, active=name,
                   portfolios={**store.portfolios, name: ()}), None


def rename_portfolio(store: PortfolioStore, old_name: str,
                     new_name: str) -> Tuple[PortfolioStore, Optional[str]]:
    """Rename in place, PRESERVING ORDER.

    Rebuilding the dict rather than popping and re-adding matters: the UI
    lists portfolios in insertion order, and a rename that quietly moved
    a client to the bottom of an advisor's list would look like a bug.
    """
    new_name = (new_name or "").strip()
    if old_name not in store.portfolios:
        return store, f'"{old_name}" does not exist.'
    if not new_name:
        return store, "Enter a new name."
    if len(new_name) > PORTFOLIO.max_name_chars:
        return store, f"Names are capped at {PORTFOLIO.max_name_chars} characters."
    if new_name == old_name:
        return store, None
    if new_name in store.portfolios:
        return store, f'A portfolio named "{new_name}" already exists.'

    renamed = {(new_name if key == old_name else key): value
               for key, value in store.portfolios.items()}
    active = new_name if store.active == old_name else store.active
    return replace(store, active=active, portfolios=renamed), None


def delete_portfolio(store: PortfolioStore, name: str) -> Tuple[PortfolioStore, Optional[str]]:
    """Refuses to delete the last one — something must be active, and
    silently recreating a default afterwards would be a surprise.

    Deleting a portfolio destroys its holdings, so the caller is expected
    to confirm first; this function does not second-guess a deliberate
    call.
    """
    if name not in store.portfolios:
        return store, f'"{name}" does not exist.'
    if len(store.portfolios) <= 1:
        return store, "Can't delete your last portfolio."
    remaining = {k: v for k, v in store.portfolios.items() if k != name}
    active = store.active if store.active != name else next(iter(remaining))
    return replace(store, active=active, portfolios=remaining), None


def set_active_portfolio(store: PortfolioStore, name: str) -> Tuple[PortfolioStore, Optional[str]]:
    if name not in store.portfolios:
        return store, f'"{name}" does not exist.'
    return replace(store, active=name), None


# --- the maths ----------------------------------------------------------------

def build_value_series(holdings: Tuple[Holding, ...],
                       prices: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """Daily portfolio market value, and the daily external cash flow.

    A holding contributes shares * price from its purchase date onward and
    nothing before it — which is what stops the chart back-projecting
    today's basket onto a past that didn't contain it.

    The cash-flow series records each purchase's cost on its purchase
    date. build_value_series returns both because time_weighted_return
    cannot be computed without knowing which value changes were market
    movement and which were money arriving.
    """
    if not holdings or not prices:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    index = None
    for series in prices.values():
        index = series.index if index is None else index.union(series.index)
    if index is None or len(index) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    index = index.sort_values()

    value = pd.Series(0.0, index=index)
    flows = pd.Series(0.0, index=index)

    for holding in holdings:
        series = prices.get(holding.ticker)
        if series is None or series.empty:
            continue
        aligned = series.reindex(index).ffill()
        held = pd.Series(
            [_as_date(ts) >= holding.purchase_date for ts in index], index=index)
        value = value.add((aligned * holding.shares).where(held, 0.0), fill_value=0.0)

        # The purchase lands on the first trading day on/after the stated
        # date — a date on a weekend or holiday must still be counted,
        # not silently discarded.
        entry = [ts for ts in index if _as_date(ts) >= holding.purchase_date]
        if entry:
            flows.loc[entry[0]] += holding.cost_total

    return value, flows


def _as_date(timestamp) -> datetime.date:
    try:
        return timestamp.date()
    except AttributeError:
        return timestamp


def time_weighted_return(value: pd.Series, flows: pd.Series) -> Optional[float]:
    """Time-weighted return as a percentage, by daily valuation.

    r(t) = (V(t) - V(t-1) - C(t)) / V(t-1), chained. Subtracting the cash
    flow is the entire point: without it, money arriving would register
    as a gain and a portfolio could show a spectacular "return" purely by
    depositing. Days where V(t-1) is zero — before anything was owned —
    contribute no return rather than an infinite one.
    """
    if value is None or value.empty or len(value) < 2:
        return None

    factors: List[float] = []
    previous = None
    for timestamp, current in value.items():
        if previous is None:
            previous = current
            continue
        flow = float(flows.get(timestamp, 0.0)) if flows is not None else 0.0
        if previous > 0:
            factors.append(1.0 + (current - previous - flow) / previous)
        previous = current

    if not factors:
        return None
    return (float(np.prod(factors)) - 1.0) * 100.0


def money_weighted_return(holdings: Tuple[Holding, ...], final_value: float,
                          end: datetime.date) -> Optional[float]:
    """Annualised money-weighted return (XIRR) as a percentage.

    Each purchase is a negative cash flow on its date; the portfolio's
    current value is a positive flow today. Solved with brentq over a
    wide bracket rather than Newton, because Newton on an IRR polynomial
    diverges readily and returning a wrong rate would be worse than
    returning none.

    None when there is nothing to solve or no sign change — an IRR is
    undefined without both an outflow and an inflow.
    """
    priced = [h for h in holdings if h.cost_total > 0]
    if not priced or final_value <= 0:
        return None

    flows: List[Tuple[datetime.date, float]] = [(h.purchase_date, -h.cost_total) for h in priced]
    flows.append((end, float(final_value)))
    first = min(d for d, _ in flows)
    if all(d == first for d, _ in flows):
        return None

    def npv(rate: float) -> float:
        total = 0.0
        for when, amount in flows:
            years = (when - first).days / 365.25
            total += amount / ((1.0 + rate) ** years)
        return total

    try:
        from scipy.optimize import brentq
        low, high = -0.9999, 10.0
        if npv(low) * npv(high) > 0:
            return None
        return float(brentq(npv, low, high, maxiter=200)) * 100.0
    except Exception:
        log_exception(logger, "portfolio.irr_failed", section="portfolio")
        return None


def rebase_benchmark(benchmark: pd.Series, to_value: float,
                     index: pd.Index) -> Optional[pd.Series]:
    """The benchmark scaled so it starts at the portfolio's opening value.

    Rebasing rather than plotting raw index levels is what makes the two
    lines comparable at a glance — otherwise a portfolio worth £8,000 and
    an index at 5,900 share an axis and say nothing.
    """
    if benchmark is None or benchmark.empty:
        return None
    aligned = benchmark.reindex(index).ffill().bfill()
    first = aligned.dropna()
    if first.empty or first.iloc[0] == 0:
        return None
    return aligned / first.iloc[0] * to_value


def build_performance(holdings: Tuple[Holding, ...],
                      price_loader: Callable[[str, datetime.date, datetime.date], pd.Series],
                      benchmark: str = None,
                      end: Optional[datetime.date] = None) -> PortfolioPerformance:
    """Assemble the whole dashboard payload.

    `price_loader(ticker, start, end) -> pd.Series of closes` is injected
    so this is testable without the network.
    """
    benchmark = benchmark or PORTFOLIO.default_benchmark
    end = end or datetime.date.today()
    notes: List[str] = []

    if not holdings:
        return PortfolioPerformance(
            period_start=end, period_end=end, value_series=pd.Series(dtype=float),
            notes=("No holdings yet — add a position to see performance.",))

    start = min(h.purchase_date for h in holdings)

    prices: Dict[str, pd.Series] = {}
    performances: List[HoldingPerformance] = []
    excluded: List[str] = []

    for holding in holdings:
        try:
            series = price_loader(holding.ticker, start, end)
        except Exception as e:
            series = None
            reason = f"price fetch failed ({type(e).__name__})"
        else:
            reason = "no price history available"

        if series is None or getattr(series, "empty", True):
            excluded.append(holding.ticker)
            performances.append(replace(
                HoldingPerformance(
                    ticker=holding.ticker, shares=holding.shares,
                    cost_basis=holding.cost_basis, purchase_date=holding.purchase_date),
                unavailable=reason))
            continue

        prices[holding.ticker] = series
        performances.append(HoldingPerformance(
            ticker=holding.ticker, shares=holding.shares, cost_basis=holding.cost_basis,
            purchase_date=holding.purchase_date, current_price=float(series.dropna().iloc[-1]),
        ))

    priced = tuple(h for h in holdings if h.ticker in prices)
    value, flows = build_value_series(priced, prices)

    twr = time_weighted_return(value, flows)
    final_value = float(value.iloc[-1]) if not value.empty else 0.0
    mwr = money_weighted_return(priced, final_value, end)

    benchmark_series = None
    benchmark_return = None
    if not value.empty:
        try:
            raw = price_loader(benchmark, start, end)
        except Exception:
            raw = None
        if raw is not None and not getattr(raw, "empty", True):
            benchmark_series = rebase_benchmark(raw, float(value.iloc[0]), value.index)
            clean = raw.dropna()
            if len(clean) >= 2 and clean.iloc[0] != 0:
                benchmark_return = float((clean.iloc[-1] / clean.iloc[0] - 1.0) * 100.0)
        else:
            notes.append(
                f"Couldn't fetch {benchmark}, so there's no benchmark comparison for this period.")

    if excluded:
        notes.append(
            "Excluded from the return because no prices could be fetched: "
            + ", ".join(excluded)
            + ". They're listed below with their cost, but aren't counted in any figure above."
        )
    if len(value) < PORTFOLIO.min_observations and not value.empty:
        notes.append(
            f"Only {len(value)} trading days of history — too short for these figures to mean "
            f"much yet."
        )

    log_event(logger, logging.INFO, "portfolio.performance_built",
              holdings=len(holdings), priced=len(priced), excluded=len(excluded))

    return PortfolioPerformance(
        period_start=start, period_end=end, value_series=value,
        benchmark_series=benchmark_series, holdings=tuple(performances),
        twr_pct=twr, mwr_pct=mwr, benchmark_return_pct=benchmark_return,
        excluded=tuple(excluded), notes=tuple(notes),
    )
