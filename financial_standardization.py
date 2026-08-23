"""Standardized financial data model for Quantix.

Yahoo Finance's raw shapes are inconsistent in ways that go beyond missing
fields (handled by financial_validation.py): the same concept is
represented in different units depending on the field, the ticker, or which
yfinance version answered the request — e.g. debtToEquity has been observed
both as a ratio (0.78) and as a percent-like number (78.0), and "current
price" is available from two different sources (info['currentPrice'], which
can be stale, and the freshly-fetched price history) that can disagree.

This module converts a TickerBundle into one StandardizedFinancials object
with canonical field names and consistent units, so every section of
finance.py reads the same normalized data instead of each doing its own
ad-hoc extraction and conversion:
  - Percentages and margins are always decimals (0.276, not 27.6). Display
    code multiplies by 100 only at render time.
  - Ratios (P/E, current ratio, debt/equity) are plain numbers.
  - Currency figures are raw dollar amounts.
  - A field that's genuinely unavailable is None — never a fabricated
    default like treating missing beta as 1.0. The exceptions are Total
    Debt, Retained Earnings, Inventory, Cash And Cash Equivalents, and
    Depreciation And Amortization, which default to 0 when absent because
    their absence conventionally means "not reported because there isn't
    any," not "unknown" (matches financial_validation.py's optional-field
    handling). P/E and Price-to-Book are similarly guarded but to None, not
    0: a non-positive value isn't a fabricated default to avoid, it's a
    genuinely meaningless valuation multiple (negative/zero earnings or book
    value) that would mislead if shown as-is.
"""
import datetime
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from data_loader import TickerBundle
from financial_validation import ValidationReport, get_field, validate_financials
from logging_setup import get_logger, log_event

logger = get_logger("standardization")


def _field_history(df: Optional[pd.DataFrame], aliases: Sequence[str]) -> Tuple[Tuple[datetime.date, float], ...]:
    """Every (period-end date, value) pair for the first matching alias,
    oldest first, NaN dropped — the full multi-year row get_field() only
    ever reduces to its most recent value. yfinance statement columns are
    most-recent-first, so this reverses them to chronological order."""
    if df is None or df.empty:
        return ()
    for alias in aliases:
        if alias in df.index:
            row = df.loc[alias]
            pairs = [(col, row[col]) for col in df.columns if pd.notna(row[col])]
            if pairs:
                return tuple(reversed(pairs))
    return ()

# A real-world debt/equity ratio is virtually never >= 5. Yahoo has reported
# this field both as a ratio (e.g. 0.78) and as a percent-like number (e.g.
# 78.0) across versions/tickers, so anything at or above this threshold is
# treated as percent-scaled and converted; otherwise it's assumed to already
# be a ratio. Best-effort given Yahoo's inconsistent reporting. Not private —
# fundamental_analysis.py reuses this to normalize Yahoo's debtToEquity for
# the Leverage Validation Report's cross-check.
_DEBT_TO_EQUITY_PERCENT_THRESHOLD = 5


def normalize_debt_to_equity(raw: Optional[float]) -> Optional[float]:
    if raw is None:
        return None
    return raw / 100 if abs(raw) >= _DEBT_TO_EQUITY_PERCENT_THRESHOLD else raw


def _positive_or_none(raw: Optional[float]) -> Optional[float]:
    """A non-positive P/E or Price-to-Book isn't a meaningful valuation
    multiple (negative/zero earnings or book value), so it's treated as
    missing rather than passed through as a misleading negative number."""
    return raw if raw and raw > 0 else None


def _compute_peg_ratio(pe_ratio: Optional[float], earnings_growth: Optional[float]) -> Optional[float]:
    """PEG proxy: P/E divided by the annual growth rate expressed as a whole
    number (e.g. 15 for 15% growth). pegRatio itself is deprecated/unreliable
    in modern yfinance responses (frequently None), so this is the canonical
    PEG source everywhere in the app instead of each caller reimplementing it.
    """
    if not pe_ratio or not earnings_growth or earnings_growth <= 0:
        return None
    return pe_ratio / (earnings_growth * 100)


@dataclass
class StandardizedFinancials:
    ticker: str

    # --- Company profile ---
    long_name: Optional[str]
    business_summary: Optional[str]
    website: Optional[str]
    sector: Optional[str]  # Yahoo's broad GICS-style sector (e.g. "Technology", "Financial Services"); drives sector-adjusted Scorecard thresholds

    # --- Valuation & ratios (decimals for percentages, plain numbers for ratios) ---
    pe_ratio: Optional[float]     # None for negative/zero earnings — a negative P/E is not a meaningful valuation multiple
    peg_ratio: Optional[float]
    price_to_book: Optional[float]  # None for negative/zero book value, same reasoning as pe_ratio
    net_margin: Optional[float]
    return_on_equity: Optional[float]
    debt_to_equity: Optional[float]   # statement-computed (Total Debt / Stockholders Equity); Yahoo's debtToEquity only as a fallback when Stockholders Equity is missing
    current_ratio: Optional[float]
    beta: Optional[float]
    earnings_growth: Optional[float]

    # --- Ownership (decimals) ---
    held_pct_insiders: Optional[float]
    held_pct_institutions: Optional[float]

    # --- Market data ---
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    current_price: Optional[float]

    # --- Balance sheet (canonical names, raw currency units) ---
    total_assets: Optional[float]
    current_assets: Optional[float]
    current_liabilities: Optional[float]
    stockholders_equity: Optional[float]
    total_liabilities: Optional[float]
    total_debt: float          # optional field: 0 means "no reported debt", not "unknown"
    total_debt_from_statement: Optional[float]  # raw balance-sheet-only value, pre-fallback; None if the statement doesn't report it. Exists purely so the Leverage Validation Report can compare it against Yahoo's info['totalDebt'] independently of `total_debt`'s resolved value
    retained_earnings: float   # optional field: defaults to 0
    inventory: float           # optional field: defaults to 0 (many companies genuinely carry none)
    cash_and_equivalents: float  # optional field: defaults to 0; used for Enterprise Value = Market Cap + Total Debt − Cash

    # --- Income statement ---
    total_revenue: Optional[float]
    ebit: Optional[float]
    interest_expense: Optional[float]
    net_income: Optional[float]
    gross_profit: Optional[float]        # optional: absent for banks/financials (no cost of revenue)
    operating_income: Optional[float]    # distinct from `ebit`, which falls back to this when EBIT itself is absent
    pretax_income: Optional[float]       # for effective tax rate; optional
    tax_provision: Optional[float]       # for effective tax rate; optional

    # --- Cash flow ---
    free_cash_flow: Optional[float]
    depreciation_and_amortization: float  # optional field: defaults to 0; used for EBITDA = EBIT + D&A

    # --- Freshness ---
    most_recent_quarter: Optional[datetime.date]  # from info['mostRecentQuarter'], used for staleness checks

    # --- Field-level validation of the source statements ---
    validation: ValidationReport

    # --- Data provenance ---
    data_fallbacks: Tuple[str, ...]  # human-readable notes on every field that fell back to a secondary source; see standardize_financials()

    # --- Multi-year history, for the DCF's revenue/margin trajectory ---
    # Every other field above is the single most-recent value get_field()
    # reduces a statement row to; these are the FULL multi-year row instead
    # (typically ~4-5 fiscal years from yfinance), as (period-end date, value)
    # pairs, oldest first, NaN dropped. Empty tuple when the statement or
    # field isn't available — the DCF degrades to a flat-margin assumption
    # in that case rather than fabricating a trend. Each value is already
    # signed exactly as the source statement reports it: Capital Expenditure
    # and Change In Working Capital both come from yfinance already negative
    # when they're a cash use, so callers add them directly rather than
    # subtracting a magnitude (confirmed by reconstructing reported Free
    # Cash Flow = Operating Cash Flow + Capital Expenditure from real data).
    revenue_history: Tuple[Tuple[datetime.date, float], ...] = ()
    ebit_history: Tuple[Tuple[datetime.date, float], ...] = ()
    depreciation_history: Tuple[Tuple[datetime.date, float], ...] = ()
    capex_history: Tuple[Tuple[datetime.date, float], ...] = ()
    change_in_working_capital_history: Tuple[Tuple[datetime.date, float], ...] = ()
    # Percent-valued (2.33 means 2.33%), NOT a fraction. Measured against
    # this yfinance version: AAPL 0.35, KO 2.33, JNJ 1.98 — each matching
    # dividendRate/price computed independently. Assuming a fraction, which
    # is the common expectation for this field, would report KO at 233%.
    dividend_yield_pct: Optional[float] = None


@st.cache_data(ttl=3600)
def _dividend_yield_pct(info: dict, current_price: Optional[float]) -> Optional[float]:
    """Trailing dividend yield as a percentage, or None.

    Yahoo's `dividendYield` is already percent-valued in this client
    version. Rather than trust that across versions, the value is
    sanity-checked against dividendRate/price — the same figure derived
    from two independent fields. A `dividendYield` that looks like a
    fraction (0.023 where the rate implies 2.3) is scaled; one that looks
    like a percentage is left alone. Falls back to the derived figure when
    Yahoo reports no yield at all, and returns None rather than guessing
    when neither is available.
    """
    reported = info.get('dividendYield')
    rate = info.get('dividendRate')
    derived = None
    if rate is not None and current_price:
        try:
            derived = float(rate) / float(current_price) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            derived = None

    if reported is None:
        return derived
    try:
        reported = float(reported)
    except (TypeError, ValueError):
        return derived
    if reported <= 0:
        return derived

    if derived is not None and derived > 0:
        # Whichever interpretation lands closer to the independently
        # derived figure is the right one.
        as_percent = abs(reported - derived)
        as_fraction = abs(reported * 100.0 - derived)
        return reported * 100.0 if as_fraction < as_percent else reported

    # No cross-check available. A "yield" under 0.5 is far more likely to
    # be a fraction (0.023 = 2.3%) than a real 0.02% payout.
    return reported * 100.0 if reported < 0.5 else reported


def standardize_financials(bundle: TickerBundle) -> StandardizedFinancials:
    """Convert a TickerBundle's raw, Yahoo-shaped data into one canonical,
    unit-consistent object. Safe to call on deep=False bundles (watchlist
    scans, peer comparisons) — statement-derived fields simply come back
    None since those bundles never fetch financials/balance sheet/cashflow.
    """
    info = bundle.info
    income_stmt, balance_sheet, cash_flow = bundle.income_stmt, bundle.balance_sheet, bundle.cash_flow

    # _positive_or_none here also automatically makes PEG (which divides by
    # pe_ratio) come back None for negative-earnings companies, with no extra
    # guard needed in _compute_peg_ratio.
    pe_ratio = _positive_or_none(info.get('trailingPE'))
    # earningsGrowth/revenueGrowth are already decimals in yfinance (0.15 = 15%)
    earnings_growth = info.get('earningsGrowth') or info.get('revenueGrowth') or None

    # Tracks every place a preferred data source was unavailable and the
    # code fell through to a secondary source or an assumption — surfaced by
    # the Financial Metrics Validation Report so "this number used a fallback"
    # is visible instead of silently indistinguishable from a fully-sourced one.
    data_fallbacks = []

    current_price = None
    if not bundle.price_history.empty:
        current_price = bundle.price_history['Close'].iloc[-1]
    elif info.get('currentPrice'):
        current_price = info.get('currentPrice')
        data_fallbacks.append("Current Price: sourced from Yahoo's quote (info['currentPrice']) because price history was unavailable")

    # Total Debt and Interest Expense each have two possible sources in raw
    # Yahoo data (the statement line item and the info-dict summary field),
    # which can disagree. The statement is preferred as the more detailed,
    # multi-period source; the info field is only a fallback so every
    # section of the app converges on one value instead of silently using
    # whichever source it happened to read from.
    total_debt_from_statement = get_field(balance_sheet, ("Total Debt",))
    total_debt = total_debt_from_statement
    if total_debt is None:
        total_debt = info.get('totalDebt')
        if total_debt is not None:
            data_fallbacks.append("Total Debt: sourced from Yahoo's info feed because the balance sheet didn't report it")
    interest_expense = get_field(income_stmt, ("Interest Expense",))
    if interest_expense is None and info.get('interestExpense'):
        interest_expense = abs(info.get('interestExpense'))
        data_fallbacks.append("Interest Expense: sourced from Yahoo's info feed because the income statement didn't report it")

    # Total Liabilities: normally one consolidated line ("Total Liabilities
    # Net Minority Interest"), but some tickers report its two real
    # components separately without the consolidated total. Verified
    # against real AAPL data that Current Liabilities + Total Non Current
    # Liabilities Net Minority Interest sums to exactly the consolidated
    # figure (checked directly against yfinance's own field-name catalog —
    # no genuinely different SINGLE label for this concept exists there, so
    # a derived sum of real components is the fallback, not a guessed
    # second alias). Gives Total Liabilities the same "don't trip straight
    # to Insufficient Data over a reporting-format quirk" treatment Total
    # Debt/Interest Expense already have above.
    total_liabilities = get_field(balance_sheet, ("Total Liabilities Net Minority Interest",))
    if total_liabilities is None:
        current_liabilities_component = get_field(balance_sheet, ("Current Liabilities",))
        non_current_liabilities_component = get_field(balance_sheet, ("Total Non Current Liabilities Net Minority Interest",))
        if current_liabilities_component is not None and non_current_liabilities_component is not None:
            total_liabilities = current_liabilities_component + non_current_liabilities_component
            data_fallbacks.append("Total Liabilities: derived as Current Liabilities + Total Non-Current Liabilities because the balance sheet didn't report a single consolidated total")

    # Debt-to-Equity: computed from statements (Total Debt / Stockholders
    # Equity) is the canonical value everywhere in the app — this avoids
    # relying on Yahoo's debtToEquity field, whose scale (ratio vs percent)
    # is inconsistent across tickers/versions. Yahoo's figure is only a
    # fallback for the rare case Stockholders Equity itself isn't reported.
    stockholders_equity = get_field(balance_sheet, ("Stockholders Equity",))
    # total_debt defaults to 0 (not None) below when genuinely unreported, so
    # a debt-free company correctly computes to 0.0 here rather than falling
    # through to the Yahoo fallback.
    resolved_total_debt = total_debt if total_debt is not None else 0
    debt_to_equity = (resolved_total_debt / stockholders_equity) if stockholders_equity else None
    if debt_to_equity is None:
        debt_to_equity = normalize_debt_to_equity(info.get('debtToEquity'))
        if debt_to_equity is not None:
            data_fallbacks.append("Debt-to-Equity: sourced from Yahoo's own (scale-normalized) figure because Stockholders Equity wasn't reported, so it couldn't be statement-computed")

    most_recent_quarter = None
    if info.get('mostRecentQuarter'):
        most_recent_quarter = datetime.date.fromtimestamp(info['mostRecentQuarter'])

    validation = validate_financials(income_stmt, balance_sheet, cash_flow)

    # A deep=False bundle (watchlist scan, peer comparison) never requests the
    # financial statements at all, so every statement field is trivially
    # "missing" — logging those would bury real signal under ~8 warnings per
    # scanned ticker. Only report field gaps when the statements were actually
    # fetched and came back incomplete.
    statements_requested = not (income_stmt.empty and balance_sheet.empty and cash_flow.empty)
    if statements_requested:
        # A missing required field is what forces a downstream metric to "N/A",
        # so it's logged at WARNING; optional fields are ordinary for many
        # tickers (ETFs, banks) and stay at DEBUG.
        for statement in validation.statements:
            for name in statement.missing_required:
                log_event(
                    logger, logging.WARNING, "data.missing_required",
                    ticker=bundle.ticker, statement=statement.statement_name, field=name,
                )
            for name in statement.missing_optional:
                log_event(
                    logger, logging.DEBUG, "data.missing_optional",
                    ticker=bundle.ticker, statement=statement.statement_name, field=name,
                )

    log_event(
        logger, logging.DEBUG if not statements_requested else logging.INFO,
        "standardized.built", ticker=bundle.ticker, statements=statements_requested,
        statements_valid=validation.is_valid if statements_requested else "n/a",
        missing_required=sum(len(s.missing_required) for s in validation.statements) if statements_requested else 0,
    )

    return StandardizedFinancials(
        ticker=bundle.ticker,

        long_name=info.get('longName') or info.get('shortName'),
        business_summary=info.get('longBusinessSummary'),
        website=info.get('website'),
        sector=info.get('sector'),

        pe_ratio=pe_ratio,
        peg_ratio=_compute_peg_ratio(pe_ratio, earnings_growth),
        price_to_book=_positive_or_none(info.get('priceToBook')),
        net_margin=info.get('profitMargins') if info.get('profitMargins') is not None else None,
        return_on_equity=info.get('returnOnEquity') if info.get('returnOnEquity') is not None else None,
        debt_to_equity=debt_to_equity,
        current_ratio=info.get('currentRatio') or None,
        beta=info.get('beta') if info.get('beta') is not None else None,
        earnings_growth=earnings_growth,

        held_pct_insiders=info.get('heldPercentInsiders') if info.get('heldPercentInsiders') is not None else None,
        held_pct_institutions=info.get('heldPercentInstitutions') if info.get('heldPercentInstitutions') is not None else None,

        market_cap=info.get('marketCap') or None,
        shares_outstanding=info.get('sharesOutstanding') or None,
        current_price=current_price,
        dividend_yield_pct=_dividend_yield_pct(info, current_price),

        total_assets=get_field(balance_sheet, ("Total Assets",)),
        current_assets=get_field(balance_sheet, ("Current Assets",)),
        current_liabilities=get_field(balance_sheet, ("Current Liabilities",)),
        stockholders_equity=stockholders_equity,
        total_liabilities=total_liabilities,
        total_debt=resolved_total_debt,
        total_debt_from_statement=total_debt_from_statement,
        retained_earnings=get_field(balance_sheet, ("Retained Earnings",), default=0),
        inventory=get_field(balance_sheet, ("Inventory",), default=0),
        cash_and_equivalents=get_field(balance_sheet, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"), default=0),

        total_revenue=get_field(income_stmt, ("Total Revenue",)),
        ebit=get_field(income_stmt, ("EBIT", "Operating Income")),
        interest_expense=interest_expense,
        net_income=get_field(income_stmt, ("Net Income",)),
        gross_profit=get_field(income_stmt, ("Gross Profit",)),
        operating_income=get_field(income_stmt, ("Operating Income",)),
        pretax_income=get_field(income_stmt, ("Pretax Income",)),
        tax_provision=get_field(income_stmt, ("Tax Provision",)),

        free_cash_flow=get_field(cash_flow, ("Free Cash Flow",)),
        depreciation_and_amortization=get_field(cash_flow, ("Depreciation And Amortization", "Depreciation Amortization Depletion"), default=0),

        most_recent_quarter=most_recent_quarter,

        validation=validation,

        data_fallbacks=tuple(data_fallbacks),

        revenue_history=_field_history(income_stmt, ("Total Revenue",)),
        ebit_history=_field_history(income_stmt, ("EBIT", "Operating Income")),
        depreciation_history=_field_history(cash_flow, ("Depreciation And Amortization", "Depreciation Amortization Depletion")),
        capex_history=_field_history(cash_flow, ("Capital Expenditure",)),
        change_in_working_capital_history=_field_history(cash_flow, ("Change In Working Capital",)),
    )
