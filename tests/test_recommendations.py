"""Tests for recommendations.py.

The properties that matter here are about not overstating what a
suggestion is:

  - "CONSERVATIVE" MUST BE HARDER ON EVERY CRITERION. Ceilings and floors
    scale in opposite directions, so a single missing inversion would
    make a conservative profile quietly LOOSEN half its criteria while
    appearing to tighten them.
  - THRESHOLDS MUST DERIVE FROM THE APP'S CONFIGURED VALUES, not a second
    set living alongside them and drifting.
  - AN UNEVALUABLE CRITERION MUST NOT COUNT EITHER WAY, matching the
    scorecard, and a candidate judged on too few must not outrank one
    judged on many.
  - NOTHING HERE PREDICTS. The module must expose no probability, score
    or forecast that could be read as one.
"""
import pytest

import recommendations as rec
from config import RECOMMENDATIONS, SCORECARD
from recommendations import (
    CriterionOutcome,
    Preferences,
    Suggestion,
    as_screen_criteria,
    available_sectors,
    criteria_for,
    rank,
)


class _FakeResult:
    def __init__(self, ticker, values, status="ok", detail=""):
        self.ticker = ticker
        self.values = values
        self.criteria_passes = {}
        self.status = status
        self.detail = detail


def screener_for(table):
    def run(tickers, criteria, *a, **k):
        return [_FakeResult(t, table.get(t, {})) for t in tickers]
    return run


def sectors_for(table):
    return lambda ticker: table.get(ticker, "")


def by_metric(prefs):
    return {o.metric: o for o in criteria_for(prefs)}


# --- translating preferences --------------------------------------------------

def test_balanced_matches_the_apps_shipped_thresholds_exactly():
    """The multiplier is 1.0, so a balanced profile must reproduce the
    configured values rather than approximate them."""
    criteria = by_metric(Preferences(risk_profile="Balanced"))
    assert criteria["beta"].threshold == pytest.approx(SCORECARD.max_beta)
    assert criteria["debt_to_equity"].threshold == pytest.approx(SCORECARD.max_debt_to_equity)
    assert criteria["net_margin_pct"].threshold == pytest.approx(SCORECARD.min_net_margin * 100)
    assert criteria["current_ratio"].threshold == pytest.approx(SCORECARD.min_current_ratio)


@pytest.mark.parametrize("metric", ["beta", "debt_to_equity", "annual_volatility_pct"])
def test_conservative_tightens_every_ceiling(metric):
    conservative = by_metric(Preferences(risk_profile="Conservative"))[metric]
    balanced = by_metric(Preferences(risk_profile="Balanced"))[metric]
    assert conservative.operator == "<="
    assert conservative.threshold < balanced.threshold


@pytest.mark.parametrize("metric", ["altman_z", "net_margin_pct", "current_ratio"])
def test_conservative_raises_every_floor(metric):
    """THE INVERSION, WHICH IS EASY TO GET BACKWARDS. Floors must scale
    the opposite way to ceilings — without that, "conservative" would
    demand a LOWER minimum margin, quietly relaxing half the criteria
    while appearing to tighten them."""
    conservative = by_metric(Preferences(risk_profile="Conservative"))[metric]
    balanced = by_metric(Preferences(risk_profile="Balanced"))[metric]
    assert conservative.operator == ">="
    assert conservative.threshold > balanced.threshold


@pytest.mark.parametrize("metric", ["beta", "debt_to_equity", "annual_volatility_pct"])
def test_aggressive_loosens_every_ceiling(metric):
    assert (by_metric(Preferences(risk_profile="Aggressive"))[metric].threshold
            > by_metric(Preferences(risk_profile="Balanced"))[metric].threshold)


@pytest.mark.parametrize("metric", ["altman_z", "net_margin_pct", "current_ratio"])
def test_aggressive_lowers_every_floor(metric):
    assert (by_metric(Preferences(risk_profile="Aggressive"))[metric].threshold
            < by_metric(Preferences(risk_profile="Balanced"))[metric].threshold)


def test_conservative_is_strictly_harder_than_aggressive_everywhere():
    """The whole-profile version of the property, so a criterion added
    later can't be scaled the wrong way unnoticed."""
    conservative = by_metric(Preferences(risk_profile="Conservative"))
    aggressive = by_metric(Preferences(risk_profile="Aggressive"))
    for metric, strict in conservative.items():
        loose = aggressive[metric]
        if strict.operator == "<=":
            assert strict.threshold < loose.threshold, metric
        else:
            assert strict.threshold > loose.threshold, metric


def test_an_unknown_profile_falls_back_to_the_default():
    assert by_metric(Preferences(risk_profile="Nonsense"))["beta"].threshold == pytest.approx(
        by_metric(Preferences(risk_profile=RECOMMENDATIONS.default_risk_profile))["beta"].threshold)


def test_profitability_can_be_switched_off():
    assert "net_margin_pct" not in by_metric(Preferences(require_profitable=False))
    assert "net_margin_pct" in by_metric(Preferences(require_profitable=True))


def test_value_leaning_adds_valuation_ceilings():
    criteria = by_metric(Preferences(valuation="Value-leaning"))
    assert criteria["pe_ratio"].operator == "<="
    assert "price_to_book" in criteria


def test_growth_leaning_uses_peg_not_a_raw_pe_ceiling():
    """A growth company with a high multiple and the earnings growth to
    match is exactly what this preference means; a P/E ceiling would
    exclude the thing being asked for."""
    criteria = by_metric(Preferences(valuation="Growth-leaning"))
    assert "peg_ratio" in criteria
    assert "pe_ratio" not in criteria


def test_any_valuation_adds_no_valuation_criteria():
    criteria = by_metric(Preferences(valuation="Any"))
    assert not {"pe_ratio", "price_to_book", "peg_ratio"} & set(criteria)


def test_criteria_convert_to_screener_filters():
    """The screener is reused for metric computation rather than
    duplicated, so the translation has to be exact."""
    outcomes = criteria_for(Preferences())
    converted = as_screen_criteria(outcomes)
    assert len(converted) == len(outcomes)
    assert {c.metric for c in converted} == {o.metric for o in outcomes}


def test_criteria_are_returned_as_data_so_the_ui_can_show_them():
    """A preference whose effect is invisible is indistinguishable from
    one that does nothing."""
    for outcome in criteria_for(Preferences()):
        assert outcome.label and outcome.metric
        assert outcome.operator in ("<=", ">=")
        assert isinstance(outcome.threshold, float)


# --- scoring a candidate ------------------------------------------------------

def outcome(metric="beta", operator="<=", threshold=1.5, value=None):
    return CriterionOutcome("label", metric, operator, threshold, value)


def test_a_ceiling_passes_at_or_below_the_threshold():
    assert outcome(operator="<=", threshold=1.5, value=1.4).passed is True
    assert outcome(operator="<=", threshold=1.5, value=1.5).passed is True
    assert outcome(operator="<=", threshold=1.5, value=1.6).passed is False


def test_a_floor_passes_at_or_above_the_threshold():
    assert outcome(operator=">=", threshold=10.0, value=12.0).passed is True
    assert outcome(operator=">=", threshold=10.0, value=10.0).passed is True
    assert outcome(operator=">=", threshold=10.0, value=8.0).passed is False


def test_a_missing_value_is_neither_pass_nor_fail():
    assert outcome(value=None).passed is None
    assert outcome(value=None).evaluable is False


def test_unevaluable_criteria_are_excluded_from_both_sides():
    """Matches the Blueprint scorecard's convention: a company is not
    marked down for a figure it structurally cannot report. Two of three
    met, with the third unavailable, is 100% — not 67%."""
    suggestion = Suggestion(ticker="X", outcomes=(
        outcome(metric="a", operator="<=", threshold=1, value=0.5),
        outcome(metric="b", operator="<=", threshold=1, value=0.5),
        outcome(metric="c", operator="<=", threshold=1, value=None),
    ))
    assert len(suggestion.evaluable) == 2
    assert len(suggestion.unavailable) == 1
    assert suggestion.match_pct == pytest.approx(100.0)


def test_match_percentage_counts_only_what_was_met():
    suggestion = Suggestion(ticker="X", outcomes=(
        outcome(metric="a", operator="<=", threshold=1, value=0.5),
        outcome(metric="b", operator="<=", threshold=1, value=5.0),
    ))
    assert suggestion.match_pct == pytest.approx(50.0)
    assert [o.metric for o in suggestion.matched] == ["a"]
    assert [o.metric for o in suggestion.missed] == ["b"]


def test_a_candidate_with_nothing_evaluable_has_no_score():
    suggestion = Suggestion(ticker="X", outcomes=(outcome(value=None),))
    assert suggestion.match_pct is None
    assert suggestion.comparable is False


def test_too_few_evaluable_criteria_is_not_comparable():
    """A perfect score on two criteria describes missing data, not a good
    fit."""
    few = Suggestion(ticker="X", outcomes=tuple(
        outcome(metric=str(i), operator="<=", threshold=1, value=0.5)
        for i in range(RECOMMENDATIONS.min_evaluable_criteria - 1)))
    assert few.match_pct == pytest.approx(100.0)
    assert few.comparable is False


# --- ranking ------------------------------------------------------------------

def test_a_better_match_ranks_higher():
    prefs = Preferences(risk_profile="Balanced", valuation="Any")
    good = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
            "altman_z": 6.0, "net_margin_pct": 25.0, "current_ratio": 2.0}
    poor = {"beta": 2.5, "debt_to_equity": 6.0, "annual_volatility_pct": 70.0,
            "altman_z": 0.5, "net_margin_pct": 1.0, "current_ratio": 0.4}
    ranked, _ = rank(prefs, ("GOOD", "POOR"),
                     screener=screener_for({"GOOD": good, "POOR": poor}),
                     sector_lookup=sectors_for({"GOOD": "Technology", "POOR": "Technology"}))
    assert [s.ticker for s in ranked] == ["GOOD", "POOR"]
    assert ranked[0].match_pct == pytest.approx(100.0)


def test_better_evidenced_candidates_win_a_tie():
    """At the same match percentage, the company judged on more criteria
    ranks first — that figure is better supported."""
    prefs = Preferences(risk_profile="Balanced", require_profitable=False, valuation="Any")
    full = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
            "altman_z": 6.0, "current_ratio": 2.0}
    partial = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0}
    ranked, _ = rank(prefs, ("PARTIAL", "FULL"),
                     screener=screener_for({"FULL": full, "PARTIAL": partial}),
                     sector_lookup=sectors_for({"FULL": "Technology", "PARTIAL": "Technology"}))
    assert [s.ticker for s in ranked] == ["FULL", "PARTIAL"]


def test_ranking_is_stable_between_runs():
    """Ties break alphabetically so the shortlist doesn't reshuffle on
    every rerun, which would make it look unreliable."""
    prefs = Preferences(risk_profile="Balanced", require_profitable=False, valuation="Any")
    same = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
            "altman_z": 6.0, "current_ratio": 2.0}
    table = {t: dict(same) for t in ("CCC", "AAA", "BBB")}
    ranked, _ = rank(prefs, ("CCC", "AAA", "BBB"), screener=screener_for(table),
                     sector_lookup=lambda t: "Technology")
    assert [s.ticker for s in ranked] == ["AAA", "BBB", "CCC"]


def test_sector_preferences_filter_the_universe():
    prefs = Preferences(sectors=("Technology",), require_profitable=False, valuation="Any")
    values = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
              "altman_z": 6.0, "current_ratio": 2.0}
    ranked, notes = rank(prefs, ("TECH", "BANK"),
                         screener=screener_for({"TECH": values, "BANK": values}),
                         sector_lookup=sectors_for({"TECH": "Technology",
                                                    "BANK": "Financial Services"}))
    assert [s.ticker for s in ranked] == ["TECH"]
    assert any("sector" in n for n in notes)


def test_no_sector_preference_means_every_sector():
    values = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
              "altman_z": 6.0, "current_ratio": 2.0}
    ranked, _ = rank(Preferences(sectors=(), require_profitable=False, valuation="Any"),
                     ("TECH", "BANK"),
                     screener=screener_for({"TECH": values, "BANK": values}),
                     sector_lookup=sectors_for({"TECH": "Technology", "BANK": "Financials"}))
    assert len(ranked) == 2


def test_thin_candidates_are_held_back_and_explained():
    prefs = Preferences(risk_profile="Balanced", require_profitable=False, valuation="Any")
    full = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
            "altman_z": 6.0, "current_ratio": 2.0}
    ranked, notes = rank(prefs, ("FULL", "THIN"),
                         screener=screener_for({"FULL": full, "THIN": {"beta": 0.1}}),
                         sector_lookup=lambda t: "Technology")
    assert [s.ticker for s in ranked] == ["FULL"]
    assert any("evaluable criteria" in n for n in notes)


def test_results_are_capped():
    values = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
              "altman_z": 6.0, "current_ratio": 2.0}
    tickers = tuple(f"T{i:02d}" for i in range(RECOMMENDATIONS.max_suggestions + 8))
    ranked, _ = rank(Preferences(require_profitable=False, valuation="Any"), tickers,
                     screener=screener_for({t: values for t in tickers}),
                     sector_lookup=lambda t: "Technology")
    assert len(ranked) <= RECOMMENDATIONS.max_suggestions


def test_an_empty_universe_says_so():
    ranked, notes = rank(Preferences(), (), screener=screener_for({}), sector_lookup=lambda t: "")
    assert ranked == () and notes


def test_a_failing_screener_degrades_to_a_note():
    def boom(tickers, criteria, *a, **k):
        raise RuntimeError("data source down")
    ranked, notes = rank(Preferences(), ("AAPL",), screener=boom, sector_lookup=lambda t: "")
    assert ranked == ()
    assert any("couldn't evaluate" in n.lower() for n in notes)


def test_available_sectors_lists_only_what_is_present():
    """Offering a sector nothing belongs to lets someone build a
    preference set that can only ever return nothing."""
    found = available_sectors(("A", "B", "C"),
                              sector_lookup=sectors_for({"A": "Technology", "B": "Energy"}))
    assert found == ("Energy", "Technology")


# --- the non-prediction contract ----------------------------------------------

def test_the_module_exposes_no_prediction_surface():
    """The task asked for a model that suggests stocks to buy. The
    momentum model measured below chance, so nothing here predicts, and
    no API should suggest otherwise."""
    banned = {"predict", "forecast", "probability", "expected_return", "target_price", "signal"}
    exposed = {name.lower() for name in dir(rec) if not name.startswith("_")}
    assert not (banned & exposed), f"prediction-shaped API exposed: {banned & exposed}"


def test_a_suggestion_carries_no_verdict_field():
    """No buy/sell/hold anywhere in the data model — a badge like that is
    read as a recommendation regardless of the disclaimer beneath it."""
    fields = set(Suggestion.__dataclass_fields__)
    assert not ({"verdict", "rating", "recommendation", "action"} & fields)
    assert "match_pct" in dir(Suggestion)


# --- the small-denominator problem ---------------------------------------------

def _suggestion(matched, total, ticker="X"):
    return Suggestion(ticker=ticker, outcomes=tuple(
        outcome(metric=str(i), operator="<=", threshold=1,
                value=0.5 if i < matched else 5.0)
        for i in range(total)))


def test_the_adjustment_penalises_a_small_basis():
    """OBSERVED LIVE. Under a conservative profile JPM ranked first at
    75% on four criteria — it's a bank, so Altman Z and current ratio are
    structurally unavailable — while AVGO, judged on all six, ranked
    below at 67%. Ranking by raw percentage rewards having fewer criteria
    apply to you."""
    few = _suggestion(3, 4)          # 75% on four
    many = _suggestion(9, 12)        # 75% on twelve — same rate, more evidence
    assert few.match_pct == pytest.approx(many.match_pct)          # same raw rate
    assert few.confidence_adjusted_pct < many.confidence_adjusted_pct


def test_the_adjustment_never_exceeds_the_raw_rate():
    """It is a lower bound: it may only ever revise a match downward."""
    for matched, total in ((3, 4), (4, 6), (8, 8), (1, 3), (0, 5)):
        s = _suggestion(matched, total)
        assert s.confidence_adjusted_pct <= s.match_pct + 1e-9


def test_a_perfect_match_on_many_criteria_beats_one_on_few():
    assert (_suggestion(10, 10).confidence_adjusted_pct
            > _suggestion(3, 3).confidence_adjusted_pct)


def test_the_adjustment_shrinks_as_evidence_grows():
    """The gap between raw and adjusted should close with more criteria —
    that is the whole point of the correction."""
    small = _suggestion(3, 4)
    large = _suggestion(30, 40)
    assert (small.match_pct - small.confidence_adjusted_pct) > \
           (large.match_pct - large.confidence_adjusted_pct)


def test_the_displayed_figure_is_still_the_raw_one():
    """The adjustment orders the list; it must not silently become the
    number shown, or the row would disagree with its own criteria count."""
    s = _suggestion(3, 4)
    assert s.match_pct == pytest.approx(75.0)


def test_no_adjusted_figure_without_anything_evaluable():
    assert Suggestion(ticker="X", outcomes=(outcome(value=None),)).confidence_adjusted_pct is None


def test_ranking_uses_the_adjusted_figure():
    """A high rate on few criteria must not outrank a slightly lower rate
    on many."""
    prefs = Preferences(risk_profile="Balanced", require_profitable=False, valuation="Any")
    thin = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0, "altman_z": 0.1}
    broad = {"beta": 0.8, "debt_to_equity": 0.5, "annual_volatility_pct": 15.0,
             "altman_z": 6.0, "current_ratio": 2.0}
    ranked, _ = rank(prefs, ("THIN", "BROAD"),
                     screener=screener_for({"THIN": thin, "BROAD": broad}),
                     sector_lookup=lambda t: "Technology")
    assert [s.ticker for s in ranked] == ["BROAD", "THIN"]
