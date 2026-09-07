"""What finance.py must render for a currency pair.

Source-reading, because the acceptance criteria are "every X has a Y" —
and because the header strip has now been declared-but-not-sourced once
(crypto) and caught pre-ship once (commodities) by exactly this kind of
test.
"""
import ast
import pathlib

import pytest

import asset_class
import asset_views
import button_roles
import forex_data
import forex_risk
import forex_screener
import forex_valuation
import quick_stats


FINANCE = pathlib.Path(__file__).resolve().parent.parent / "finance.py"


@pytest.fixture(scope="module")
def source():
    return FINANCE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source):
    return ast.parse(source)


# --- the capability -----------------------------------------------------------

def test_rate_parity_is_a_declared_capability():
    assert asset_class.RATE_PARITY in asset_class.ALL_CAPABILITIES
    assert asset_class.supports(asset_class.FOREX, asset_class.RATE_PARITY)


def test_no_other_class_claims_rate_parity():
    """An equity has no second policy rate. Declaring it widely would
    put a carry card on a stock."""
    for spec in asset_class.SPECS:
        if spec.key == asset_class.FOREX:
            continue
        assert asset_class.RATE_PARITY not in spec.supports, spec.key


def test_a_currency_pair_still_claims_no_fundamentals():
    for capability in (asset_class.FUNDAMENTALS, asset_class.DCF,
                       asset_class.SECTOR_PERCENTILE, asset_class.PEERS,
                       asset_class.ON_CHAIN, asset_class.CURVE):
        assert not asset_class.supports(asset_class.FOREX, capability)


def test_both_panels_are_gated_on_the_capability(source):
    assert source.count(
        "asset_class.supports(asset_kind, asset_class.RATE_PARITY)") == 2


# --- the corrected record -----------------------------------------------------

def test_the_gap_note_no_longer_claims_rates_are_unsourced():
    """It said "no rates provider is wired up". The BIS publishes policy
    rates for 49 economies free and keyless, so the note would have sent
    a reader looking for a credential they do not need."""
    note = " ".join(asset_class.missing_sources(asset_class.FOREX))
    assert "no rates provider" not in note
    assert "bid and ask" in note
    assert "implied volatility" in note


def test_the_absence_reason_names_what_a_pair_DOES_have():
    reason = asset_class.spec(asset_class.FOREX).absence_reason
    assert "policy rates" in reason
    assert "purchasing power" in reason


def test_every_gap_entry_describes_an_absence():
    """Each renders as "Not sourced in this build: {entry}." — an entry
    that opens by listing what IS available contradicts its prefix."""
    for gap in asset_class.missing_sources(asset_class.FOREX):
        assert "ARE available" not in gap
        assert not gap.strip().endswith(".")


# --- the header ---------------------------------------------------------------

def test_the_forex_header_offers_the_differential_and_the_ratio():
    stats = asset_views.header_stats(asset_class.FOREX)
    for key in ("price", "change_pct", "range_52w_pct",
                "rate_differential_pct", "carry_ratio"):
        assert key in stats, key


def test_every_forex_header_stat_is_a_real_spec():
    for key in asset_views.header_stats(asset_class.FOREX):
        assert key in quick_stats.STATS_BY_KEY, key


def test_no_bid_ask_stat_is_offered():
    """Yahoo quotes an ask BELOW the bid on four of fourteen majors, so
    a spread in the header would be a fabricated number."""
    stats = asset_views.header_stats(asset_class.FOREX)
    assert not [s for s in stats
                if "bid" in s or "ask" in s or "spread" in s]


def test_the_rates_tab_is_no_longer_labelled_not_applicable():
    labels = asset_views.tab_labels(asset_class.FOREX)
    assert "Valuation (n/a)" not in labels
    assert "Rates & Carry" in labels


def test_the_peers_tab_still_says_not_applicable():
    """A currency has no competitors to benchmark. The correlated-pair
    read lives with the risk it describes rather than earning a tab
    label it would only half fill."""
    assert "Peers (n/a)" in asset_views.tab_labels(asset_class.FOREX)


def test_the_first_eight_tabs_keep_their_count():
    assert len(asset_views.tab_labels(asset_class.FOREX)) == \
        len(asset_views.BASE_TABS)


def test_the_header_strip_is_actually_given_a_pair_row(source):
    """Declared and sourced are two different steps."""
    fragment = source[source.index("def _render_quick_stats"):][:6000]
    assert "forex_screener" in fragment
    assert "asset_class.RATE_PARITY" in fragment
    assert "forex_data.parse_pair(" in fragment


# --- the screener wiring ------------------------------------------------------

def test_the_forex_remove_button_is_marked_destructive():
    assert "forex_remove_" in button_roles.DANGER_PREFIXES


def test_the_screener_labels_are_numbered_per_row(source):
    assert '_fs_suffix = "" if _fs_i == 0 else f" {_fs_i + 1}"' in source
    assert 'f"Currency metric{_fs_suffix}"' in source


def test_the_criteria_widgets_take_no_streamlit_key(source):
    """The operator list changes with the metric — is/is not against
    < > — so a stored value outside the new options raises."""
    block = source[source.index("FOREX SCREENER"):]
    block = block[:block.index('st.header("ETF Screener")')]
    for widget in ("Currency metric", "Currency op", "Currency value"):
        index = block.index(widget)
        assert "key=" not in block[index:index + 300].split(")")[0], widget


# --- nothing fabricated -------------------------------------------------------

def test_the_unavailable_notes_reach_the_page(source):
    for constant in ("BID_ASK_UNUSABLE", "BROKER_FEEDS_UNCONFIGURED",
                     "FORWARDS_ARE_DERIVED", "CALENDAR_IS_THIRD_PARTY"):
        assert constant in source, constant
    for constant in ("PARITY_IS_NOT_A_FORECAST", "ANNUALISATION_NOTE",
                     "NO_TYPICAL_BAND", "IMPLIED_VOL_UNAVAILABLE",
                     "INTERVENTION_UNAVAILABLE",
                     "POLITICAL_RISK_UNAVAILABLE"):
        assert constant in source, constant


def test_the_forward_table_is_labelled_as_derived(source):
    """No free source publishes an FX forward curve, so presenting these
    as quotes would be the fabrication this app exists to avoid."""
    block = source[source.index('st.header("Rates & Carry"'):]
    block = block[:block.index("Valuation scorecard")]
    assert "FORWARDS_ARE_DERIVED" in block
    assert "not quoted" in block


def test_a_scheduled_rate_decision_warns_that_the_carry_will_change(source):
    block = source[source.index('st.header("Rates & Carry"'):]
    block = block[:block.index("Valuation scorecard")]
    assert "is_rate_decision" in block
    assert "about to" in block


def test_every_forex_metric_on_screen_carries_a_tooltip(source, tree):
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "metric":
            continue
        rendered = ast.unparse(node)
        if not any(tag in rendered for tag in ("_fx", "_fr_")):
            continue
        if not any(kw.arg == "help" for kw in node.keywords):
            missing.append(rendered[:90])
    assert not missing, f"forex metrics with no help=: {missing}"


def test_the_skew_metric_shows_its_extremes(source):
    """Skew is dominated by single days, so the number alone misleads."""
    block = source[source.index('st.header("Currency Risk & Carry Unwind"'):]
    block = block[:block.index("Risks this build does not score")]
    assert "worst_day_pct" in block and "best_day_pct" in block


def test_the_risk_panel_loads_its_own_long_history(source):
    """Skew needs years and the sidebar's range is usually one — the
    same trap that gave crypto and commodities a blank 1-year
    volatility."""
    block = source[source.index('st.header("Currency Risk & Carry Unwind"'):]
    block = block[:block.index("Risks this build does not score")]
    assert "load_price_history_only" in block
    assert "365 * 5" in block
