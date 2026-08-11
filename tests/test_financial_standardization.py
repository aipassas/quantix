"""Tests for financial_standardization.py's Total Liabilities field sourcing
— standardize_financials() now falls back to deriving Total Liabilities from
its two real components when the balance sheet doesn't report a single
consolidated line, the same "don't trip straight to Insufficient Data over
a reporting-format quirk" treatment Total Debt/Interest Expense already had.
"""
import pandas as pd
import pytest

from data_loader import TickerBundle
from financial_standardization import standardize_financials

PERIOD = pd.Timestamp("2024-12-31")


def _bundle(balance_sheet_rows: dict) -> TickerBundle:
    balance_sheet = pd.DataFrame({PERIOD: balance_sheet_rows})
    return TickerBundle(
        ticker="TEST",
        info={"marketCap": 1_000_000_000.0, "sharesOutstanding": 100_000_000.0},
        balance_sheet=balance_sheet,
    )


def test_total_liabilities_uses_primary_label_when_present():
    bundle = _bundle({
        "Total Liabilities Net Minority Interest": 285_508_000_000.0,
        "Current Liabilities": 165_631_000_000.0,
        "Total Non Current Liabilities Net Minority Interest": 119_877_000_000.0,
    })
    standardized = standardize_financials(bundle)
    assert standardized.total_liabilities == 285_508_000_000.0
    assert not any("Total Liabilities" in f for f in standardized.data_fallbacks)


def test_total_liabilities_falls_back_to_derived_sum_when_primary_label_missing():
    """The primary consolidated line is absent, but both real components
    are present — must derive the sum rather than report Insufficient Data,
    and must disclose the fallback like every other one in this module."""
    bundle = _bundle({
        "Current Liabilities": 165_631_000_000.0,
        "Total Non Current Liabilities Net Minority Interest": 119_877_000_000.0,
    })
    standardized = standardize_financials(bundle)
    assert standardized.total_liabilities == pytest.approx(285_508_000_000.0)
    assert any("Total Liabilities" in f and "derived" in f for f in standardized.data_fallbacks)


def test_total_liabilities_none_when_neither_source_available():
    """No consolidated line AND no components — the existing fail-safe
    (None, never a fabricated number) must still hold."""
    bundle = _bundle({"Total Assets": 1_000_000.0})
    standardized = standardize_financials(bundle)
    assert standardized.total_liabilities is None


def test_total_liabilities_none_when_only_one_component_available():
    """Partial data (only one of the two components) must not silently
    produce a wrong (understated) total."""
    bundle = _bundle({"Current Liabilities": 165_631_000_000.0})
    standardized = standardize_financials(bundle)
    assert standardized.total_liabilities is None


@pytest.mark.live
def test_derived_sum_formula_matches_real_aapl_reported_total():
    """AAPL currently reports the consolidated line directly, so the
    fallback path isn't exercised for it — but this proves the derivation
    formula itself (Current Liabilities + Total Non Current Liabilities Net
    Minority Interest) is empirically correct against real data, which is
    exactly what the fallback would compute if the consolidated line were
    ever absent."""
    import datetime

    from data_loader import load_ticker_bundle

    end = datetime.date.today()
    start = end - datetime.timedelta(days=400)
    bundle = load_ticker_bundle("AAPL", start, end, deep=True)
    bs = bundle.balance_sheet

    reported_total = bs.loc["Total Liabilities Net Minority Interest"].iloc[0]
    current = bs.loc["Current Liabilities"].iloc[0]
    non_current = bs.loc["Total Non Current Liabilities Net Minority Interest"].iloc[0]

    assert current + non_current == pytest.approx(reported_total)
