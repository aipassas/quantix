"""The asset-class selector and the views it drives.

Two things carry the weight here. First that the selector is a LENS and
not a mode: it must never be able to make the page analyse a fund as a
stock, because the classification comes from the data. Second that a
header stat offered for a class is one that class can actually answer —
an ETF header full of "Not reported" is the same category error that had
the data-quality badge grading funds on corporate filings.
"""
import ast
from pathlib import Path

import pytest

import asset_class
import asset_views as av
import keyboard_shortcuts
import quick_stats


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


# --- the pills ----------------------------------------------------------------

def test_the_selector_offers_the_six_classes_the_task_asks_for():
    assert av.pill_labels() == (
        "Stocks", "ETFs & Funds", "Bonds", "Crypto", "Commodities", "Forex")


def test_every_pill_round_trips_through_its_key():
    for label in av.pill_labels():
        key = av.key_for_pill(label)
        assert av.view(key).pill == label
    # An unknown label falls back rather than raising.
    assert av.view("nonsense").key == av.DEFAULT_VIEW
    assert av.key_for_pill("nonsense") == av.DEFAULT_VIEW


def test_every_pill_resolves_to_a_real_asset_class():
    """A pill whose class the app cannot classify would light up and then
    analyse nothing."""
    known = {spec.key for spec in asset_class.SPECS}
    for view in av.VIEWS:
        assert view.asset_class_key in known, view.pill


def test_the_bonds_pill_says_what_is_not_built():
    """There is no bond asset class — a bond FUND is an ETF and a Treasury
    yield is an index level. A pill that silently did nothing would be
    worse than no pill, so it routes to instruments that work and states
    the gap."""
    bond = av.view("bond")
    assert bond.asset_class_key == asset_class.ETF
    assert "not built yet" in bond.note
    assert "duration" in bond.note
    assert set(bond.examples) == {"TLT", "AGG", "BND"}


def test_every_view_carries_examples_and_a_note():
    for view in av.VIEWS:
        assert view.examples, view.pill
        assert view.note, view.pill
        assert view.placeholder, view.pill
        assert len(view.badge) == 1, view.pill


def test_the_search_placeholder_follows_the_selection():
    assert "stock" in av.search_placeholder(asset_class.EQUITY).lower()
    assert "etf" in av.search_placeholder(asset_class.ETF).lower()
    assert "bond" in av.search_placeholder("bond").lower()
    assert "crypto" in av.search_placeholder(asset_class.CRYPTO).lower()


def test_the_selector_cannot_override_what_a_symbol_actually_is():
    """THE LOAD-BEARING PROPERTY. asset_kind must come from
    asset_class.classify on the fetched info, never from the pill —
    otherwise leaving "Stocks" lit and typing SPY runs company analysis
    on a fund."""
    assert "asset_kind = asset_class.classify(ticker_bundle.info" in FINANCE
    # The pill's session key must never be read into asset_kind.
    tree = ast.parse(FINANCE)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "asset_kind" for t in node.targets)):
            source = ast.dump(node)
            assert "asset_view" not in source, "the pill must not set asset_kind"


# --- badges -------------------------------------------------------------------

def test_each_class_has_a_distinct_badge():
    badges = [av.badge(spec.key) for spec in asset_class.SPECS]
    real = [b for b in badges if b != "?"]
    assert len(real) == len(set(real)), badges


def test_a_bond_fund_badges_as_a_fund():
    """It IS one. Badging it B would promise a bond analysis the app does
    not have."""
    assert av.badge(asset_class.ETF) == "E"
    assert av.badge("bond") == "?", "bond is a pill, not a classification"


def test_an_unclassified_symbol_gets_a_badge_rather_than_a_crash():
    assert av.badge(asset_class.UNKNOWN) == "?"
    assert av.badge("nonsense") == "?"
    assert av.badge_title(asset_class.ETF) == asset_class.label(asset_class.ETF)


def test_the_watchlist_badges_what_the_symbol_is_not_what_was_selected():
    assert "asset_views.badge(_wl_snap.asset_class_key)" in FINANCE
    # ...and the class rides along on the snapshot, which already had the
    # info dict, so no extra fetch was added.
    panel = (ROOT / "watchlist_panel.py").read_text(encoding="utf-8")
    assert "asset_class_key" in panel
    assert "klass = asset_class.classify(info, ticker)" in panel


# --- header stats -------------------------------------------------------------

def test_every_header_stat_key_is_a_real_stat():
    """A key with no StatSpec renders nothing and fails silently."""
    for klass, keys in av.HEADER_STATS.items():
        for key in keys:
            assert key in quick_stats.STATS_BY_KEY, (klass, key)


def test_a_fund_is_not_offered_stats_only_a_company_has():
    """SPY's header used to read "Market Cap · Not reported" beside "Net
    Margin · Not reported" — questions a fund cannot be asked, presented
    as data it failed to supply."""
    etf = av.header_stats(asset_class.ETF)
    for company_only in ("market_cap", "net_margin", "return_on_equity",
                         "debt_to_equity", "current_ratio", "sector",
                         "pe_ratio", "price_to_book"):
        assert company_only not in etf, company_only
    # ...and it IS offered what a fund reports.
    for fund_stat in ("expense_ratio_pct", "net_assets", "fund_category",
                      "fund_pe"):
        assert fund_stat in etf


def test_an_equity_keeps_every_stat_it_had():
    equity = av.header_stats(asset_class.EQUITY)
    for key in ("price", "change_pct", "pe_ratio", "market_cap",
                "dividend_yield_pct", "beta", "price_to_book", "net_margin",
                "return_on_equity", "debt_to_equity", "current_ratio",
                "sector"):
        assert key in equity, key


def test_a_class_with_no_issuer_gets_only_price_shaped_stats():
    for klass in (asset_class.FOREX, asset_class.FUTURE, asset_class.INDEX,
                  asset_class.UNKNOWN):
        assert set(av.header_stats(klass)) <= {"price", "change_pct"}, klass


def test_beta_is_left_out_of_the_fund_header_on_purpose():
    """standardized.beta never resolves for a fund — measured, SPY and TLT
    both read "Not reported" — so offering it would guarantee a
    permanently blank stat, which is what this mapping exists to remove."""
    assert "beta" not in av.header_stats(asset_class.ETF)
    source = (ROOT / "asset_views.py").read_text(encoding="utf-8")
    assert "beta3Year" in source, "the reason is recorded where it applies"


def test_every_class_offers_at_least_price_and_change():
    for klass in av.HEADER_STATS:
        assert "price" in av.header_stats(klass)
        assert "change_pct" in av.header_stats(klass)


def test_applies_agrees_with_header_stats():
    assert av.applies("market_cap", asset_class.EQUITY)
    assert not av.applies("market_cap", asset_class.ETF)
    assert av.applies("expense_ratio_pct", asset_class.ETF)
    assert not av.applies("expense_ratio_pct", asset_class.EQUITY)


# --- the saved selection must survive switching class -------------------------

def test_customising_on_one_class_does_not_erase_another_classes_stats():
    """THE BUG THIS PREVENTS. The picker only offers the current class's
    stats, so saving its result verbatim would wipe every stat belonging
    to the others — customise while looking at an ETF and a stock's
    market cap, ROE and net margin vanish from a selection the reader set
    up deliberately."""
    saved = ("price", "market_cap", "return_on_equity", "sector")
    applicable = ("price", "change_pct", "expense_ratio_pct")
    merged = quick_stats.merge_selection(saved, ("price", "expense_ratio_pct"),
                                         applicable)
    # The equity-only keys are untouched...
    for kept in ("market_cap", "return_on_equity", "sector"):
        assert kept in merged, kept
    # ...and the fund choice is in.
    assert "expense_ratio_pct" in merged


def test_merging_never_duplicates_a_key():
    merged = quick_stats.merge_selection(
        ("price", "market_cap"), ("price", "price"), ("price", "change_pct"))
    assert len(merged) == len(set(merged))


def test_merging_drops_a_choice_that_does_not_apply():
    merged = quick_stats.merge_selection(
        ("price",), ("price", "market_cap"), ("price", "change_pct"))
    assert "market_cap" not in merged


def test_the_page_merges_rather_than_overwrites():
    assert "quick_stats.merge_selection(" in FINANCE
    # The naive save must not come back.
    assert "quick_stats.set_selected(_qs_choice)" not in FINANCE


def test_the_picker_is_not_keyed_now_that_its_options_change():
    """A keyed widget whose options list changes can raise when its stored
    value falls outside the new list — and this list now changes with the
    asset class. The choice is read from the return value, so the key was
    never needed."""
    assert 'key="quick_stats_choice"' not in FINANCE


# --- tabs ---------------------------------------------------------------------

def test_the_eight_panels_keep_their_positions_for_every_class():
    """finance.py unpacks its tab objects positionally and ⌘1–⌘8 are bound
    to those positions, so a class may re-LABEL a tab but never reorder
    or drop one."""
    for spec in asset_class.SPECS:
        labels = av.tab_labels(spec.key)
        assert len(labels) >= len(av.BASE_TABS), spec.key
        assert labels[0] == "Overview", spec.key
        assert labels[:len(av.BASE_TABS)][1] == "Chart Workspace", spec.key


def test_the_comparison_tab_is_appended_and_only_for_funds():
    fund = av.tab_labels(asset_class.ETF)
    assert fund[-1] == av.COMPARISON_TAB
    assert len(fund) == len(av.BASE_TABS) + 1
    for other in (asset_class.EQUITY, asset_class.CRYPTO, asset_class.FOREX,
                  asset_class.INDEX, asset_class.UNKNOWN):
        assert av.COMPARISON_TAB not in av.tab_labels(other), other
        assert len(av.tab_labels(other)) == len(av.BASE_TABS), other
    assert av.has_comparison(asset_class.ETF)
    assert not av.has_comparison(asset_class.EQUITY)


def test_a_fund_is_told_its_third_tab_is_about_holdings():
    labels = av.tab_labels(asset_class.ETF)
    assert labels[2] == "Holdings & Fund Profile"
    assert av.tab_labels(asset_class.EQUITY)[2] == "Fundamentals & Valuation"


def test_the_page_builds_its_tabs_from_the_asset_class():
    assert "asset_views.tab_labels(asset_kind)" in FINANCE
    assert "asset_views.has_comparison(asset_kind)" in FINANCE
    # The old hardcoded strip must not come back.
    assert '"Overview", "Chart Workspace", "Fundamentals & Valuation", "Risk & Technicals",' not in FINANCE


# --- keyboard -----------------------------------------------------------------

def test_alt_digits_are_matched_on_position_not_on_the_character():
    """macOS composes Option+1 into "¡", so event.key is not "1" there. A
    key-based match would fire on Windows and never on a Mac — probed on
    the running page, event.code is Digit1 and the event is preventable."""
    source = (ROOT / "keyboard_shortcuts.py").read_text(encoding="utf-8")
    assert "/^Digit([1-9])$/.exec(event.code" in source
    assert "event.altKey" in source


def test_there_is_one_alt_trigger_button_per_pill():
    assert keyboard_shortcuts.ASSET_TRIGGER_PREFIX
    assert f'f"{{keyboard_shortcuts.ASSET_TRIGGER_PREFIX}}{{_kbd_i}}"' in FINANCE
    assert "for _kbd_i, _kbd_view in enumerate(asset_views.VIEWS):" in FINANCE


def test_the_alt_triggers_are_hidden_by_the_existing_rule():
    """They are real buttons so the click drives a real rerun, but they
    must not be visible. The hiding rule matches on the shared prefix."""
    assert keyboard_shortcuts.ASSET_TRIGGER_PREFIX.startswith("kbd_trigger_")
    assert '[class*="st-key-kbd_trigger_"]' in FINANCE


def test_the_palette_and_panel_follow_the_visible_tab_labels():
    """A palette offering "Go to Fundamentals & Valuation" while the tab
    reads "Holdings & Fund Profile" sends the reader looking for
    something that is not there."""
    fund_tabs = av.tab_labels(asset_class.ETF)
    labels = [c.label for c in keyboard_shortcuts.commands(fund_tabs)]
    assert "Go to Holdings & Fund Profile" in labels
    assert "Go to Fundamentals & Valuation" not in labels
    # ...and with no argument it still describes the equity strip.
    default_labels = [c.label for c in keyboard_shortcuts.commands()]
    assert "Go to Fundamentals & Valuation" in default_labels


def test_the_shortcut_panel_lists_the_tabs_actually_on_screen():
    fund_tabs = av.tab_labels(asset_class.ETF)
    nav = keyboard_shortcuts.shortcuts_by_category(fund_tabs)["Navigation"]
    tab_shortcut = next(s for s in nav if s.keys.startswith("⌘1"))
    assert "Holdings & Fund Profile" in tab_shortcut.note
    assert tab_shortcut.keys == "⌘1 – ⌘9"
    # The module constant is untouched by that call.
    assert keyboard_shortcuts.MAIN_TABS[2] == "Fundamentals & Valuation"


def test_the_alt_shortcut_is_documented_with_its_caveat():
    listed = [s for s in keyboard_shortcuts.SHORTCUTS if "Alt" in s.keys]
    assert listed, "an undocumented binding is an undiscoverable one"
    assert "event.code" in listed[0].note


def test_the_page_passes_the_live_tabs_to_both_panels():
    assert "keyboard_shortcuts.commands(_kbd_tabs)" in FINANCE
    assert "keyboard_shortcuts.shortcuts_by_category(\n                _kbd_tabs)" in FINANCE


def test_the_keyboard_block_reads_the_class_with_a_default():
    """finance.py is a script: the palette renders ~2400 lines before
    asset_kind exists, so it must read session_state with a fallback
    rather than raising NameError for every user on their first run."""
    assert 'st.session_state.get("asset_kind", asset_class.EQUITY)' in FINANCE
    assert 'st.session_state["asset_kind"] = asset_kind' in FINANCE


def test_the_pill_widget_takes_no_streamlit_key():
    """The same session entry is written by the Alt+N shortcut and by the
    deferred switch; a keyed widget restores its own stored value over an
    externally applied one on the next run."""
    block = FINANCE[FINANCE.index("_view_pick = st.sidebar.pills("):]
    block = block[:block.index(")")]
    assert "key=" not in block
