"""ETF decomposition.

Almost everything here guards a unit or a denominator, because that is
where this data source is treacherous:

  * Yahoo labels a row "Price/Earnings" and reports its RECIPROCAL. Taken
    at face value every fund trades at 0.04x earnings.
  * The expense ratio exists in two places 100x apart — a percent in
    `info`, a fraction in `fund_operations`.
  * The task's own formula weights the top ten holdings, which are 37.6%
    of SPY. That produces a P/E of 9.46 against a true 25.15 — an error
    that reads like a cheap fund rather than like a bug.
"""
import pathlib

import pytest

import etf_analysis as ea


ROOT = pathlib.Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


def profile(**kw) -> ea.EtfProfile:
    base = dict(symbol="TEST", category="Large Blend",
                price_earnings=25.0, category_price_earnings=20.0,
                expense_ratio_pct=0.09, category_expense_ratio_pct=0.72,
                turnover_pct=3.0, category_turnover_pct=95.0,
                top_holdings=tuple(
                    ea.Holding(f"H{i}", f"Holding {i}", 3.0) for i in range(10)))
    base.update(kw)
    return ea.EtfProfile(**base)


# --- the reciprocal ------------------------------------------------------------

def test_the_mislabelled_ratio_is_inverted():
    """0.03976 is an earnings YIELD; the multiple is 25.15."""
    assert ea._invert(0.03976) == pytest.approx(25.15, abs=0.01)
    assert ea._invert(0.12003) == pytest.approx(8.33, abs=0.01)


def test_a_zero_or_negative_yield_has_no_multiple():
    """A fund with no earnings has no P/E, and 1/0 is not an answer."""
    for bad in (0, 0.0, -0.05, None, "", float("nan")):
        assert ea._invert(bad) is None


def test_an_absurd_multiple_is_refused_rather_than_printed():
    """A near-zero yield inverts to thousands, which is a data artefact."""
    assert ea._invert(1e-9) is None


# --- the expense ratio ---------------------------------------------------------

def test_the_expense_ratio_is_carried_as_a_percent():
    """The source frame holds a FRACTION (0.000945 = 0.0945%); `info`
    holds a percent. One conversion, in the loader, so no caller has to
    remember which it got."""
    source = (ROOT / "etf_analysis.py").read_text(encoding="utf-8")
    # Exact assignments, not substrings: "expense * 100.0" also matches
    # inside "category_expense * 100.0", so a poisoned copy that dropped
    # the fund's own conversion still passed.
    assert "expense_ratio_pct=expense * 100.0" in source
    assert "category_expense_ratio_pct=(category_expense * 100.0" in source
    assert "turnover_pct=turnover * 100.0" in source


def test_a_cheaper_fund_is_not_flagged():
    assert not ea.expense_is_high(profile())
    assert ea.expense_gap_pct(profile()) == pytest.approx(-0.63, abs=0.01)


def test_a_fund_more_than_half_a_point_over_its_peers_is_flagged():
    assert ea.expense_is_high(profile(expense_ratio_pct=1.5,
                                      category_expense_ratio_pct=0.72))


def test_a_missing_expense_ratio_is_not_treated_as_free():
    assert ea.expense_gap_pct(profile(expense_ratio_pct=None)) is None
    assert not ea.expense_is_high(profile(expense_ratio_pct=None))
    assert ea.expense_drag(None, 30) is None


# --- the fee drag --------------------------------------------------------------

def test_the_drag_compounds_rather_than_multiplying():
    """A 0.5% fee over 30 years costs far more than 15%, because it is
    charged against the whole balance every year."""
    thirty = ea.expense_drag(0.5, 30)
    assert thirty > 0.5 * 30 * 0.3
    assert 12 < thirty < 15, thirty


def test_a_longer_horizon_always_costs_more():
    drags = [ea.expense_drag(0.5, y) for y in ea.DRAG_YEARS]
    assert drags == sorted(drags)


def test_a_zero_fee_costs_nothing():
    assert ea.expense_drag(0.0, 30) == pytest.approx(0.0, abs=1e-9)


def test_the_assumed_return_is_declared_not_hidden():
    """It is an illustration of what a fee costs, not a forecast."""
    assert ea.DRAG_ASSUMED_GROSS_RETURN_PCT > 0
    assert "DRAG_ASSUMED_GROSS_RETURN_PCT" in FINANCE
    assert "not a forecast" in FINANCE


# --- concentration is not valuation --------------------------------------------

def test_concentration_is_the_sum_of_the_listed_weights():
    assert ea.concentration_pct(profile().top_holdings) == pytest.approx(30.0)


def test_concentration_of_nothing_is_unknown_not_zero():
    assert ea.concentration_pct(()) is None


def test_valuation_does_not_come_from_the_top_holdings():
    """The task specifies sum(weight x pe) over the top ten. For SPY those
    are 37.6% of the fund, so that sum gives 9.46 against a true 25.15 —
    a 62% understatement that looks like a cheap fund. The whole-fund
    figure is used instead."""
    source = (ROOT / "etf_analysis.py").read_text(encoding="utf-8")
    assert "price_earnings=_invert(_cell(equity" in source
    # No weighted-average-of-holdings valuation anywhere.
    assert "weight_pct * " not in source


# --- style ---------------------------------------------------------------------

def test_the_providers_own_category_wins():
    """My P/E cutoffs and Yahoo disagreed on a real fund: VTV is
    categorised Large Value and prices at 20.7x, which a cutoff of 18
    calls Blend. The category also carries the size band."""
    assert ea.style_label(profile(category="Large Value", price_earnings=20.7)) == "Large Value"
    assert ea.style_label(profile(category="Small Blend")) == "Small Blend"


def test_the_multiple_is_only_a_fallback():
    labelled = ea.style_label(profile(category="", price_earnings=15.0))
    assert labelled.startswith("Value")
    assert "inferred" in labelled, "an inferred label must say so"


def test_nothing_to_classify_says_so():
    assert "Not classified" in ea.style_label(
        ea.EtfProfile(symbol="X", category="", price_earnings=None))


def test_a_category_without_a_style_word_falls_through():
    """"Ultrashort Bond" carries no style, so the multiple decides."""
    out = ea.style_label(profile(category="Ultrashort Bond", price_earnings=15.0))
    assert "inferred" in out


# --- the scorecard -------------------------------------------------------------

def test_the_scorecard_says_how_much_of_it_could_be_scored():
    """A five-part scorecard secretly computed from three parts is a
    worse lie than an honest three."""
    parts, overall, how = ea.quality_scorecard(profile())
    assert overall is not None
    assert "3 of 5" in how


def test_unscoreable_components_are_listed_not_dropped():
    parts, _, _ = ea.quality_scorecard(profile())
    unscored = [p for p in parts if p.score is None]
    assert len(unscored) == 2
    for part in unscored:
        assert "not available" in part.detail.lower()


def test_a_profile_with_nothing_scoreable_returns_no_score():
    parts, overall, how = ea.quality_scorecard(
        ea.EtfProfile(symbol="X"))
    assert overall is None
    assert "Nothing could be scored" in how


def test_scores_stay_inside_the_scale():
    for expense, turnover in ((5.0, 900.0), (0.0, 0.0), (-1.0, 500.0)):
        parts, overall, _ = ea.quality_scorecard(
            profile(expense_ratio_pct=expense, turnover_pct=turnover))
        for part in parts:
            if part.score is not None:
                assert 0.0 <= part.score <= 10.0, part


def test_a_cheaper_lower_turnover_fund_scores_higher():
    cheap, _, _ = ea.quality_scorecard(profile(expense_ratio_pct=0.03, turnover_pct=2.0))
    dear, _, _ = ea.quality_scorecard(profile(expense_ratio_pct=1.2, turnover_pct=200.0))
    cheap_score = sum(p.score for p in cheap if p.score is not None)
    dear_score = sum(p.score for p in dear if p.score is not None)
    assert cheap_score > dear_score


# --- failure ------------------------------------------------------------------

def test_a_bad_symbol_reports_rather_than_raising():
    result = ea.load_profile("")
    assert not result.ok and result.error


# --- the panel ----------------------------------------------------------------

def test_the_panel_renders_only_where_holdings_are_supported():
    assert "asset_class.supports(asset_kind, asset_class.HOLDINGS)" in FINANCE


def test_the_panel_tells_the_reader_the_top_ten_are_not_the_valuation_basis():
    at = FINANCE.index("Top holdings")
    block = FINANCE[at:at + 700]
    assert "CONCENTRATION" in block
    assert "whole" in block


def test_the_inversion_is_explained_where_it_is_shown():
    at = FINANCE.index('st.metric("Price / Earnings"')
    assert "reciprocal" in FINANCE[at:at + 500]


# --- a bug this task surfaced -------------------------------------------------

def test_a_sub_one_percent_yield_is_not_multiplied_by_a_hundred():
    """Found while verifying this panel: QQQ's strip read "Dividend ·
    44.00%". It genuinely yields 0.44% and reports no dividendRate, so
    the cross-check could not run and the fallback heuristic ("a yield
    under 0.5 is more likely a fraction") scaled it. Every fund or low
    payer under 1% with no rate reported had the same 100x error."""
    from financial_standardization import _dividend_yield_pct

    # No dividendRate -> no cross-check -> must NOT be scaled.
    assert _dividend_yield_pct({"dividendYield": 0.44}, 705.0) == 0.44
    assert _dividend_yield_pct({"dividendYield": 0.02}, 100.0) == 0.02


def test_the_cross_check_still_corrects_a_genuine_fraction():
    """The hedge against a client version that flips convention has to
    survive the fix: where a rate IS reported, the two readings are
    compared and the closer one wins."""
    from financial_standardization import _dividend_yield_pct

    # rate/price implies 2.33%; a reported 0.0233 is the fraction form.
    assert _dividend_yield_pct(
        {"dividendYield": 0.0233, "dividendRate": 2.33}, 100.0) == pytest.approx(2.33)
    # ...and a reported 2.33 is already the percent form.
    assert _dividend_yield_pct(
        {"dividendYield": 2.33, "dividendRate": 2.33}, 100.0) == pytest.approx(2.33)


def test_no_yield_at_all_is_none_not_zero():
    from financial_standardization import _dividend_yield_pct

    assert _dividend_yield_pct({}, 100.0) is None
    assert _dividend_yield_pct({"dividendYield": 0}, 100.0) is None
