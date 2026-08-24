"""The asset-class spine.

This exists because nothing in the analysis path used to ask what kind of
instrument it was looking at — `quoteType` appeared twice in the whole
codebase, both times for display. BTC-USD therefore loaded as a valid
bundle with pe=None and sector=None, and then a discounted cash flow, an
eight-point company scorecard and a sector-percentile ranking were run
against it.

The distinction these tests protect is between a metric that is MISSING
and one that cannot exist. This app's whole convention is to disclose the
former; the latter is a different statement and has to read differently,
or a reader goes looking for data that was never going to be there.
"""
import ast
import pathlib

import pytest

import asset_class as ac


ROOT = pathlib.Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


# --- classification -----------------------------------------------------------

@pytest.mark.parametrize("quote_type,expected", [
    ("EQUITY", ac.EQUITY), ("ETF", ac.ETF), ("MUTUALFUND", ac.ETF),
    ("CRYPTOCURRENCY", ac.CRYPTO), ("CURRENCY", ac.FOREX),
    ("FUTURE", ac.FUTURE), ("INDEX", ac.INDEX),
])
def test_yahoos_quote_types_map_to_a_class(quote_type, expected):
    assert ac.classify({"quoteType": quote_type}) == expected


def test_classification_is_case_insensitive():
    """Yahoo is not consistent about case across endpoints."""
    assert ac.classify({"quoteType": "equity"}) == ac.EQUITY
    assert ac.classify({"quoteType": "Equity"}) == ac.EQUITY


def test_an_unknown_type_is_never_treated_as_a_stock():
    """Defaulting to equity is exactly what produced the original bug: an
    unrecognised instrument gets the full company treatment on the
    strength of a missing field."""
    for info in ({}, None, {"quoteType": ""}, {"quoteType": "OPTION"}, "nonsense"):
        assert ac.classify(info) == ac.UNKNOWN


def test_an_unknown_instrument_gets_no_company_analysis():
    assert not ac.supports(ac.UNKNOWN, ac.DCF)
    assert not ac.supports(ac.UNKNOWN, ac.FUNDAMENTALS)
    assert not ac.supports(ac.UNKNOWN, ac.SECTOR_PERCENTILE)


# --- the capability matrix ----------------------------------------------------

def test_every_class_declares_every_capability_one_way_or_the_other():
    """A capability nobody declared would silently read as unsupported."""
    for spec in ac.SPECS:
        for capability in spec.supports:
            assert capability in ac.ALL_CAPABILITIES, (spec.key, capability)


def test_only_equities_get_a_discounted_cash_flow():
    """Nothing without cash flows can have them discounted."""
    for spec in ac.SPECS:
        if spec.key == ac.EQUITY:
            assert ac.DCF in spec.supports
        else:
            assert ac.DCF not in spec.supports, spec.key


def test_only_equities_get_the_company_scorecard_and_sector_ranking():
    for spec in ac.SPECS:
        expected = spec.key == ac.EQUITY
        assert (ac.FUNDAMENTALS in spec.supports) is expected, spec.key
        assert (ac.SECTOR_PERCENTILE in spec.supports) is expected, spec.key


def test_price_derived_analysis_applies_to_everything():
    """Moving averages and value-at-risk need prices, nothing more. A
    currency pair has those, so withholding them would be as wrong as
    running a DCF on it."""
    for spec in ac.SPECS:
        for capability in (ac.TECHNICALS, ac.RISK, ac.SIMULATION):
            assert capability in spec.supports, (spec.key, capability)


def test_only_a_basket_has_holdings():
    for spec in ac.SPECS:
        assert (ac.HOLDINGS in spec.supports) is (spec.key == ac.ETF), spec.key


# --- what the reader is told --------------------------------------------------

def test_every_class_that_loses_a_panel_explains_why():
    """"Not available" alone reads as a fetch that failed."""
    for spec in ac.SPECS:
        if set(spec.supports) == set(ac.ALL_CAPABILITIES):
            continue
        if spec.key == ac.EQUITY:
            continue
        assert len(spec.absence_reason) > 40, spec.key
        assert spec.absence_reason.endswith("."), spec.key


def test_the_note_names_the_class_rather_than_being_generic():
    note = ac.unavailable_note(ac.CRYPTO, ac.DCF)
    assert "cryptocurrency" in note.lower()
    assert "cash flows" in note.lower()


def test_the_note_does_not_claim_the_data_is_merely_missing():
    """The whole point: this is not a gap to be filled later."""
    for key in (ac.CRYPTO, ac.FOREX, ac.INDEX, ac.ETF):
        note = ac.unavailable_note(key, ac.DCF).lower()
        assert "not applicable" in note
        assert "not reported" not in note


def test_gaps_in_this_build_are_recorded_per_class():
    """So a later phase does not rediscover which sources are missing."""
    for key in (ac.ETF, ac.CRYPTO, ac.FUTURE, ac.INDEX, ac.FOREX):
        assert ac.missing_sources(key), key
    assert ac.missing_sources(ac.EQUITY) == ()


# --- the app honours it -------------------------------------------------------

def test_the_app_classifies_once_from_the_bundle():
    assert "asset_kind = asset_class.classify(ticker_bundle.info" in FINANCE
    assert FINANCE.count("asset_class.classify(") == 1


def test_the_classification_happens_before_the_panels_that_use_it():
    """finance.py is a script; name order is execution order."""
    classified = FINANCE.index("asset_kind = asset_class.classify(")
    for use in ("asset_class.supports(asset_kind, asset_class.FUNDAMENTALS)",
                "asset_class.supports(asset_kind, asset_class.DCF)"):
        assert FINANCE.index(use) > classified, use


@pytest.mark.parametrize("capability", ["FUNDAMENTALS", "DCF"])
def test_the_equity_only_panels_are_gated(capability):
    assert f"asset_class.supports(asset_kind, asset_class.{capability})" in FINANCE


def test_the_gates_are_inside_the_tab_not_around_it():
    """A guard placed around `with tab_fundamentals:` would render the
    fallback notice into the page body instead of the tab."""
    tree = ast.parse(FINANCE)
    gated = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(ast.unparse(i.context_expr) == "tab_fundamentals" for i in node.items):
            continue
        first = node.body[0]
        if isinstance(first, ast.If) and "asset_class.supports" in ast.unparse(first.test):
            gated += 1
    assert gated == 2, f"{gated} of the two fundamentals blocks open with the guard"


def test_the_fallback_says_something_rather_than_rendering_nothing():
    """An empty tab reads as a broken page."""
    at = FINANCE.index("asset_class.supports(asset_kind, asset_class.DCF)")
    block = FINANCE[at:at + 400]
    assert "unavailable_note" in block
    assert "missing_sources" in block
