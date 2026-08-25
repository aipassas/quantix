"""Tests for the quick-stats strip.

This strip sits at the top of every screen, so a wrong number here is the
most-seen wrong number in the app. The properties that matter:

  * A metric the ticker doesn't report renders as "Not reported" — never
    0.00, never blank.
  * Units are converted once, from a declaration on the spec.
    StandardizedFinancials genuinely mixes fractions (net_margin) with
    already-percent values (dividend_yield_pct), and conflating them is
    the bug class this codebase keeps hitting.
  * Every spec's help_key resolves. metric_help.help_for RAISES on an
    unknown key, so a typo would crash the header rather than degrade.
  * An unreadable preferences file is never silently overwritten.
"""
import pytest

import local_store


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def qs():
    import quick_stats as module
    return module


class Quote:
    def __init__(self, price=309.35, change_pct=-0.63, pe_ratio=32.3):
        self.price, self.change_pct, self.pe_ratio = price, change_pct, pe_ratio


class Std:
    def __init__(self, **kw):
        defaults = dict(current_price=309.35, market_cap=3.24e12,
                        dividend_yield_pct=0.35, beta=1.09, price_to_book=61.2,
                        net_margin=0.2762, return_on_equity=1.49,
                        debt_to_equity=1.34, current_ratio=1.0, sector="Technology")
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


# --- the spec table is internally consistent ----------------------------------

def test_every_help_key_resolves(qs):
    """help_for RAISES on an unknown key, so a typo crashes the header."""
    import metric_help

    for spec in qs.STATS:
        if spec.help_key:
            metric_help.help_for(spec.help_key)   # must not raise


def test_every_spec_reads_a_field_that_exists(qs):
    import dataclasses

    import financial_standardization as fs
    import watchlist_panel as wp

    import etf_analysis

    std_fields = {f.name for f in dataclasses.fields(fs.StandardizedFinancials)}
    quote_fields = {f.name for f in dataclasses.fields(wp.QuoteSnapshot)}
    # A third source: fund stats read off an EtfProfile, via the alias map
    # (the stat key and the profile attribute differ where the stat key
    # would otherwise collide with an equity one).
    fund_fields = {f.name for f in dataclasses.fields(etf_analysis.EtfProfile)}
    for spec in qs.STATS:
        if spec.source == "fund":
            attr = qs._FUND_ATTRS.get(spec.key, spec.key)
            assert attr in fund_fields, f"{spec.key} -> {attr} is not on EtfProfile"
            continue
        have = quote_fields if spec.source == "quote" else std_fields
        assert spec.key in have, f"{spec.key} is not on {spec.source}"


def test_every_fund_stat_is_reachable_through_the_alias_map(qs):
    """A fund key with no alias entry falls through to getattr on the raw
    key, which silently returns None for every fund."""
    for spec in qs.STATS:
        if spec.source == "fund":
            assert spec.key in qs._FUND_ATTRS, spec.key


def test_a_fund_stat_with_no_profile_is_not_reported(qs):
    """It must not fall through to an equity field of the same name."""
    for spec in qs.STATS:
        if spec.source == "fund":
            assert qs.raw_value(spec, None, None, None) is None


def test_defaults_are_all_real_keys(qs):
    for key in qs.DEFAULT_KEYS:
        assert key in qs.STATS_BY_KEY
    assert len(qs.DEFAULT_KEYS) <= qs.MAX_SELECTED


# --- formatting ---------------------------------------------------------------

def test_compact_money_scales(qs):
    assert qs.compact_money(3.24e12) == "$3.24T"
    assert qs.compact_money(4.125e11) == "$412.50B"
    assert qs.compact_money(9.802e8) == "$980.20M"
    assert qs.compact_money(12345.0) == "$12.35K"
    assert qs.compact_money(950.0) == "$950.00"


def test_compact_money_scales_negatives_the_same_way(qs):
    """Equity can be negative; it must not fall through to plain digits."""
    assert qs.compact_money(-3.24e12) == "$-3.24T"


def test_a_missing_value_is_never_a_zero(qs):
    for spec in qs.STATS:
        assert qs.format_value(spec, None) == qs.NOT_REPORTED
    assert qs.compact_money(None) == qs.NOT_REPORTED


def test_nan_is_treated_as_missing(qs):
    spec = qs.STATS_BY_KEY["pe_ratio"]
    assert qs.format_value(spec, float("nan")) == qs.NOT_REPORTED


def test_fractions_and_percents_are_not_conflated(qs):
    """net_margin is a fraction (0.2762 -> 27.62%); dividend_yield_pct is
    already a percentage (0.35 -> 0.35%). Treating either as the other is
    the unit bug this codebase keeps hitting."""
    assert qs.format_value(qs.STATS_BY_KEY["net_margin"], 0.2762) == "27.62%"
    assert qs.format_value(qs.STATS_BY_KEY["dividend_yield_pct"], 0.35) == "0.35%"


def test_text_stats_pass_through(qs):
    assert qs.format_value(qs.STATS_BY_KEY["sector"], "Technology") == "Technology"
    assert qs.format_value(qs.STATS_BY_KEY["sector"], "  ") == qs.NOT_REPORTED


def test_display_reads_from_the_right_object(qs):
    quote, std = Quote(), Std()
    assert qs.display(qs.STATS_BY_KEY["price"], quote, std) == "$309.35"
    assert qs.display(qs.STATS_BY_KEY["market_cap"], quote, std) == "$3.24T"
    assert qs.display(qs.STATS_BY_KEY["sector"], quote, std) == "Technology"


def test_price_falls_back_to_the_standardized_close(qs):
    """Better a slightly older real price than "Not reported" on a ticker
    that plainly has one."""
    assert qs.display(qs.STATS_BY_KEY["price"], Quote(price=None), Std()) == "$309.35"
    assert qs.display(qs.STATS_BY_KEY["price"], Quote(price=None),
                      Std(current_price=None)) == qs.NOT_REPORTED


def test_a_ticker_reporting_nothing_renders_entirely_as_unavailable(qs):
    quote = Quote(price=None, change_pct=None, pe_ratio=None)
    std = Std(**{k: None for k in ("current_price", "market_cap", "dividend_yield_pct",
                                   "beta", "price_to_book", "net_margin",
                                   "return_on_equity", "debt_to_equity",
                                   "current_ratio", "sector")})
    for spec in qs.STATS:
        assert qs.display(spec, quote, std) == qs.NOT_REPORTED, spec.key


def test_raw_value_never_raises_on_a_junk_object(qs):
    class Nothing:
        pass

    for spec in qs.STATS:
        assert qs.raw_value(spec, Nothing(), Nothing()) is None


# --- the saved selection ------------------------------------------------------

def test_defaults_when_nothing_is_saved(qs):
    assert qs.selected() == qs.DEFAULT_KEYS


def test_selection_round_trips_in_order(qs):
    ok, error = qs.set_selected(["market_cap", "pe_ratio", "beta"])
    assert ok, error
    assert qs.selected() == ("market_cap", "pe_ratio", "beta")


def test_duplicates_and_unknown_keys_are_dropped(qs):
    qs.set_selected(["pe_ratio", "pe_ratio", "not_a_stat", "beta"])
    assert qs.selected() == ("pe_ratio", "beta")


def test_an_empty_selection_is_respected_not_reset(qs):
    """Clearing the strip is a real choice; springing back to defaults
    would make the control look broken."""
    ok, _ = qs.set_selected([])
    assert ok
    assert qs.selected() == ()


def test_too_many_is_refused(qs):
    ok, error = qs.set_selected([s.key for s in qs.STATS])
    assert not ok and error
    assert qs.selected() == qs.DEFAULT_KEYS      # unchanged


def test_reset_restores_the_defaults(qs):
    qs.set_selected(["beta"])
    qs.reset()
    assert qs.selected() == qs.DEFAULT_KEYS


def test_a_corrupt_preferences_file_is_never_overwritten(qs, sandbox):
    path = sandbox / qs.STORE_FILENAME
    path.write_text("{ not json")

    assert qs.selected() == qs.DEFAULT_KEYS       # degrade to defaults in the UI
    ok, error = qs.set_selected(["beta"])
    assert not ok and error
    assert path.read_text() == "{ not json"       # untouched


def test_the_refresh_interval_matches_the_quote_cache(qs):
    """Polling faster than the cache TTL re-renders an identical number."""
    import inspect

    import watchlist_panel

    source = inspect.getsource(watchlist_panel)
    assert f"ttl={qs.REFRESH_SECONDS}" in source or str(qs.REFRESH_SECONDS) in source


def test_price_and_day_are_offered_but_not_defaults(qs):
    """The symbol header already shows both in larger type immediately
    above the strip, so defaulting to them repeats what the eye has just
    read. They stay available in the picker."""
    assert "price" not in qs.DEFAULT_KEYS
    assert "change_pct" not in qs.DEFAULT_KEYS
    assert "price" in qs.STATS_BY_KEY
    assert "change_pct" in qs.STATS_BY_KEY


def test_the_default_strip_is_the_brief_minus_the_duplicates(qs):
    assert qs.DEFAULT_KEYS == ("pe_ratio", "market_cap", "dividend_yield_pct")
