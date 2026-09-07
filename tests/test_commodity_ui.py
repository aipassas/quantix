"""What finance.py must render for a futures contract.

Source-reading, because the acceptance criteria are "every X has a Y" —
and because PHASE 3 shipped a header strip whose stats were declared,
unit-tested and still printed "Not reported" on screen, since nothing
ever handed it the object to read.
"""
import ast
import pathlib

import pytest

import asset_class
import asset_views
import button_roles
import commodity_curve
import commodity_data
import commodity_screener
import quick_stats


FINANCE = pathlib.Path(__file__).resolve().parent.parent / "finance.py"


@pytest.fixture(scope="module")
def source():
    return FINANCE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source):
    return ast.parse(source)


# --- the capability -----------------------------------------------------------

def test_curve_is_a_declared_capability():
    assert asset_class.CURVE in asset_class.ALL_CAPABILITIES
    assert asset_class.supports(asset_class.FUTURE, asset_class.CURVE)


def test_no_other_class_claims_a_term_structure():
    """An equity has no delivery months. Declaring it widely would put a
    contango card on a stock."""
    for spec in asset_class.SPECS:
        if spec.key == asset_class.FUTURE:
            continue
        assert asset_class.CURVE not in spec.supports, spec.key


def test_a_futures_contract_still_claims_no_fundamentals():
    for capability in (asset_class.FUNDAMENTALS, asset_class.DCF,
                       asset_class.SECTOR_PERCENTILE, asset_class.PEERS,
                       asset_class.ON_CHAIN):
        assert not asset_class.supports(asset_class.FUTURE, capability)


def test_both_panels_are_gated_on_the_capability(source):
    assert source.count(
        "asset_class.supports(asset_kind, asset_class.CURVE)") == 2


# --- the corrected facts ------------------------------------------------------

def test_the_gap_note_no_longer_claims_the_curve_is_unreachable():
    """It said "Yahoo returns only the front month per symbol". Probed
    false: GCZ26.CMX quotes, with open interest. Leaving that in would
    send a reader looking for a data source they already have."""
    note = " ".join(asset_class.missing_sources(asset_class.FUTURE))
    assert "front month" not in note
    assert "spot price" in note
    assert "USDA" in note


def test_the_absence_reason_no_longer_says_the_curve_is_unsourced():
    reason = asset_class.spec(asset_class.FUTURE).absence_reason
    assert "does not yet source" not in reason
    assert "term structure" in reason


def test_every_gap_entry_describes_an_absence():
    """Each is rendered as "Not sourced in this build: {entry}." — one
    that opens by listing what IS available contradicts its own prefix."""
    for gap in asset_class.missing_sources(asset_class.FUTURE):
        assert "ARE available" not in gap
        assert not gap.strip().endswith(".")


# --- the header ---------------------------------------------------------------

def test_the_commodity_header_offers_the_curve_and_the_range():
    stats = asset_views.header_stats(asset_class.FUTURE)
    for key in ("price", "change_pct", "range_52w_pct", "curve_shape",
                "roll_yield_pct"):
        assert key in stats, key


def test_every_commodity_header_stat_is_a_real_spec():
    for key in asset_views.header_stats(asset_class.FUTURE):
        assert key in quick_stats.STATS_BY_KEY, key


def test_the_futures_tab_is_no_longer_labelled_not_applicable():
    labels = asset_views.tab_labels(asset_class.FUTURE)
    assert "Valuation (n/a)" not in labels
    assert "Futures Curve" in labels


def test_the_first_eight_tabs_keep_their_count():
    """finance.py unpacks tab objects positionally and Cmd-1..8 bind to
    those positions."""
    assert len(asset_views.tab_labels(asset_class.FUTURE)) == \
        len(asset_views.BASE_TABS)


def test_the_header_strip_is_actually_given_a_commodity_row(source):
    """THE PHASE 3 BUG, not repeated. Declaring a stat and sourcing it
    are two separate steps, and only a source-reading test catches the
    second."""
    fragment = source[source.index("def _render_quick_stats"):][:5000]
    assert "commodity_screener" in fragment
    assert "asset_class.CURVE" in fragment


# --- the screener wiring ------------------------------------------------------

def test_the_commodity_remove_button_is_marked_destructive():
    assert "commodity_remove_" in button_roles.DANGER_PREFIXES


def test_the_screener_labels_are_numbered_per_row(source):
    """Streamlit hashes (label, options, index, help) to identify an
    unkeyed widget and label_visibility is NOT in that hash."""
    assert '_ms_suffix = "" if _ms_i == 0 else f" {_ms_i + 1}"' in source
    assert 'f"Commodity metric{_ms_suffix}"' in source


def test_the_criteria_widgets_take_no_streamlit_key(source):
    """The operator list changes with the metric — is/is not against
    < > — so a stored value outside the new options raises."""
    block = source[source.index("COMMODITY SCREENER"):]
    block = block[:block.index("st.header(\"ETF Screener\")")]
    for widget in ("Commodity metric", "Commodity op", "Commodity value"):
        index = block.index(widget)
        assert "key=" not in block[index:index + 300].split(")")[0], widget


# --- nothing fabricated -------------------------------------------------------

def test_the_unavailable_notes_reach_the_page(source):
    for constant in ("SPOT_UNAVAILABLE", "GEOPOLITICAL_UNAVAILABLE",
                     "POLYGON_UNCONFIGURED", "EIA_API_UNCONFIGURED"):
        assert constant in source, constant
    # The agricultural gap reaches the page through the per-commodity
    # accessor rather than the constant — what matters is that the
    # sentence is rendered for the commodity it applies to, not which
    # name carried it there.
    assert "inventory_note(" in source
    for constant in ("SPEC_FORMULA_NOTE", "ROLL_ANNUALISATION_CAVEAT",
                     "ANNUALISATION_NOTE", "BASIS_RISK_NOTE", "SHOCK_NOTE"):
        assert constant in source, constant


def test_the_curve_panel_shows_open_interest_per_contract(source):
    """A thin month has to be visible as thin, or the liquidity filter is
    a silent judgement."""
    block = source[source.index('st.header("Futures Curve"'):]
    block = block[:block.index("Commodity Risk & Season")]
    assert "Open interest" in block
    assert "Used for shape" in block


def test_every_commodity_metric_on_screen_carries_a_tooltip(source, tree):
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "metric":
            continue
        rendered = ast.unparse(node)
        if not any(tag in rendered for tag in ("_fc_", "_cm_", "_fi")):
            continue
        if not any(kw.arg == "help" for kw in node.keywords):
            missing.append(rendered[:90])
    assert not missing, f"commodity metrics with no help=: {missing}"


def test_the_agricultural_gap_is_not_printed_twice(source):
    """It reached the page from inventory_note() AND from an
    unconditional caption, so a corn page showed the same paragraph
    twice in a row — and a crude page carried a note about corn."""
    block = source[source.index('st.header("Futures Curve"'):]
    block = block[:block.index("Commodity Risk & Season")]
    assert block.count("AGRICULTURAL_INVENTORY_UNAVAILABLE") == 0
    # It still reaches the reader, through the per-commodity accessor.
    assert "inventory_note(" in block


def test_the_inventory_panel_compares_against_the_same_week(source):
    """A flat mean would report the season as news."""
    block = source[source.index("US crude inventory"):]
    block = block[:block.index("Commodity Risk & Season")]
    assert "SAME WEEK" in block


def test_the_curve_chart_and_the_seasonality_chart_are_explained(source):
    import metric_help

    assert "futures_curve" in metric_help.CHART_HELP
    assert "commodity_seasonality" in metric_help.CHART_HELP
    assert 'chart_help("futures_curve")' in source
    assert 'chart_help("commodity_seasonality")' in source
