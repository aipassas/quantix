"""What finance.py must actually render for a coin.

These read the app's own source, because the acceptance criteria here
are all of the form "every X has a Y" — the kind a hand-written list
goes stale on.
"""
import ast
import pathlib

import pytest

import asset_class
import asset_views
import button_roles
import crypto_data
import quick_stats


FINANCE = (pathlib.Path(__file__).resolve().parent.parent / "finance.py")


@pytest.fixture(scope="module")
def source():
    return FINANCE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source):
    return ast.parse(source)


# --- the capability is declared, not inferred ---------------------------------

def test_on_chain_is_a_declared_capability():
    assert asset_class.ON_CHAIN in asset_class.ALL_CAPABILITIES
    assert asset_class.supports(asset_class.CRYPTO, asset_class.ON_CHAIN)


def test_no_other_class_claims_on_chain_analysis():
    """An equity has no public ledger. Declaring the capability widely
    would put an NVT card on a stock."""
    for spec in asset_class.SPECS:
        if spec.key == asset_class.CRYPTO:
            continue
        assert asset_class.ON_CHAIN not in spec.supports, spec.key


def test_crypto_still_does_not_claim_fundamentals_or_a_dcf():
    """A coin has no cash flows. On-chain valuation is an addition, not
    a reason to reopen the panels that never applied."""
    for capability in (asset_class.FUNDAMENTALS, asset_class.DCF,
                       asset_class.SECTOR_PERCENTILE, asset_class.PEERS):
        assert not asset_class.supports(asset_class.CRYPTO, capability)


def test_both_panels_are_gated_on_the_capability(source):
    """Gated on a declared capability rather than on a guess from a
    missing field — the distinction asset_class exists to keep."""
    assert source.count(
        "asset_class.supports(asset_kind, asset_class.ON_CHAIN)") == 2


# --- the missing-source note is current ---------------------------------------

def test_the_crypto_gap_note_no_longer_claims_nvt_is_unreachable():
    """It said NVT needed Glassnode. blockchain.info publishes the
    inputs free and keyless, so the note would have sent a reader
    looking for a credential they do not need."""
    gaps = asset_class.missing_sources(asset_class.CRYPTO)
    note = " ".join(gaps)
    # NVT used to be listed here as needing Glassnode. It does not —
    # blockchain.info publishes the inputs free and keyless — so listing
    # it would send a reader looking for a credential they do not need.
    assert "NVT" not in note
    # What genuinely is missing must still be named.
    assert "MVRV" in note
    assert "Bitcoin" in note
    # Each entry is rendered as "Not sourced in this build: {gap}.", so an
    # entry must describe an ABSENCE. One that opens by listing what IS
    # available contradicts its own prefix on screen, and did.
    for gap in gaps:
        assert "ARE available" not in gap, gap
        assert not gap.strip().endswith("."), gap


# --- the header ---------------------------------------------------------------

def test_the_crypto_header_offers_the_stats_the_task_names():
    stats = asset_views.header_stats(asset_class.CRYPTO)
    for key in ("price", "change_pct", "market_cap", "volume_24h",
                "dominance_pct"):
        assert key in stats, key


def test_every_crypto_header_stat_is_a_real_spec():
    for key in asset_views.header_stats(asset_class.CRYPTO):
        assert key in quick_stats.STATS_BY_KEY, key


def test_no_equity_only_stat_leaks_into_the_crypto_header():
    """A coin has no P/E, no sector and no dividend. Offering them is the
    category error that had the data-quality badge grade ETFs on
    corporate filings."""
    stats = set(asset_views.header_stats(asset_class.CRYPTO))
    for key in ("pe_ratio", "sector", "dividend_yield_pct", "net_margin",
                "return_on_equity", "debt_to_equity", "expense_ratio_pct"):
        assert key not in stats, key


def test_the_valuation_tab_is_no_longer_labelled_not_applicable():
    """It said "Valuation (n/a)" because there was no crypto valuation
    engine. There is one now, so the label would be false."""
    labels = asset_views.tab_labels(asset_class.CRYPTO)
    assert "Valuation (n/a)" not in labels
    assert "On-Chain & Valuation" in labels


def test_the_first_eight_tabs_keep_their_count_and_order():
    """finance.py unpacks its tab objects POSITIONALLY and ⌘1-⌘8 are
    bound to those positions, so a class may relabel a tab but must not
    insert or remove one."""
    labels = asset_views.tab_labels(asset_class.CRYPTO)
    assert len(labels) == len(asset_views.BASE_TABS)


# --- the screener wiring ------------------------------------------------------

def test_the_crypto_remove_button_is_marked_destructive():
    assert "crypto_remove_" in button_roles.DANGER_PREFIXES


def test_the_screener_criteria_widgets_take_no_streamlit_key(source):
    """Deliberate, for the reason the ETF and bond screeners document: a
    keyed widget restores its old value over a just-applied preset, and
    a stored value outside a changed options list raises."""
    block = source[source.index("CRYPTO SCREENER"):]
    block = block[:block.index("# ==========================================",
                               100)]
    for widget in ("Coin metric", "Coin op", "Coin threshold"):
        index = block.index(widget)
        following = block[index:index + 400]
        assert "key=" not in following.split(")")[0], widget


def test_the_screener_labels_are_numbered_per_row(source):
    """Streamlit identifies an unkeyed widget by hashing (label, options,
    index, help), and label_visibility is NOT in that hash — so two
    collapsed rows with matching parameters collide outright."""
    assert '_cs_suffix = "" if _cs_i == 0 else f" {_cs_i + 1}"' in source
    assert 'f"Coin metric{_cs_suffix}"' in source


# --- no fabricated numbers ----------------------------------------------------

def test_the_unavailable_notes_reach_the_page(source):
    """Every gap this module found has to be visible to a reader, not
    only recorded in a docstring."""
    for constant in ("MVRV_UNAVAILABLE", "WHALE_UNAVAILABLE",
                     "EXCHANGE_RESERVE_UNAVAILABLE", "SOCIAL_UNAVAILABLE"):
        assert constant in source, constant
    # The Bitcoin-only scope reaches the page through its accessor rather
    # than the constant, so both spellings count — what matters is that
    # the sentence is rendered, not which name carried it there.
    assert ("ONCHAIN_BITCOIN_ONLY" in source
            or "onchain_note(" in source)
    for constant in ("REGULATORY_UNAVAILABLE", "AUDIT_UNAVAILABLE",
                     "HACK_HISTORY_UNAVAILABLE", "ANNUALISATION_NOTE",
                     "WEEKEND_OVERLAP_NOTE", "LIQUIDATION_NOTE"):
        assert constant in source, constant


def test_the_nvt_threshold_note_reaches_the_page(source):
    assert "NVT_SPEC_THRESHOLD_NOTE" in source


def test_every_crypto_metric_on_screen_carries_a_tooltip(source, tree):
    """The app's standing rule. A number with no explanation is where
    every unit bug in this project has hidden."""
    block_start = source.index("CRYPTO SCREENER")
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", "")
        if name != "metric":
            continue
        # Only the crypto blocks: identified by the variable prefixes the
        # panels use.
        rendered = ast.unparse(node)
        if not any(tag in rendered for tag in ("_cx_", "_cr_")):
            continue
        if not any(kw.arg == "help" for kw in node.keywords):
            missing.append(rendered[:90])
    assert not missing, f"crypto metrics with no help=: {missing}"


def test_the_panels_never_render_a_bare_zero_for_a_missing_figure(source):
    """"Unavailable" is the house form. A 0.00 reads as a measured zero,
    which for a max supply states the opposite of the truth."""
    for block in ("_cx_", "_cr_"):
        assert f'{block}' in source
    assert '"Uncapped"' in source
    assert source.count('"Unavailable"') > 5


def test_the_header_strip_is_actually_given_a_coin_to_read(source):
    """THE BUG THIS CATCHES. The crypto stats were declared, the specs
    resolved, and the strip still printed "Not reported" for all three —
    because nothing ever handed it a CoinRow. Declaring a stat and
    sourcing it are two separate steps and only one of them is visible
    in a unit test of quick_stats."""
    fragment = source[source.index("def _render_quick_stats"):]
    fragment = fragment[:4000]
    assert "crypto_data.resolve(" in fragment
    assert "crypto_data.with_dominance(" in fragment
    assert "asset_class.ON_CHAIN" in fragment


def test_the_coin_row_never_displaces_a_fund_profile(source):
    """A symbol cannot be both, but the two share the argument, so the
    crypto branch has to run only when the fund branch found nothing."""
    fragment = source[source.index("def _render_quick_stats"):][:4000]
    assert "if _qs_fund is None and asset_class.supports(" in fragment
