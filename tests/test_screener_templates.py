"""Tests for saved screeners and the metrics they needed.

The properties worth defending are the ones that make a saved screen
reproducible and honest:

  * A template stores its ticker universe, because this screener filters
    a list rather than scanning the market. Criteria alone would produce a
    different answer every time.
  * A criterion this version cannot evaluate is KEPT and reported, never
    dropped. Silently discarding a filter makes a screen return more
    matches than the screen it claims to be.
  * The starter set seeds once. Deleting them all keeps them gone.

Also covers the metrics added for the brief's four example screens —
particularly dividend yield, whose Yahoo field is percent-valued in this
client version despite the widespread assumption that it is a fraction.
"""
import pytest

import local_store


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(local_store, "app_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def templates():
    import screener_templates as module
    return module


# --- the new metrics ----------------------------------------------------------

def test_the_brief_examples_are_all_expressible():
    """Each of the four named screens must map onto real metrics."""
    from screener import METRICS_BY_KEY, operators_for

    for metric, operator in (("sector", "is"), ("price", "<"),
                             ("pe_ratio", "<"), ("dividend_yield_pct", ">"),
                             ("revenue_growth_pct", ">")):
        assert metric in METRICS_BY_KEY, metric
        assert operator in operators_for(metric), (metric, operator)


def test_sector_is_categorical_and_numeric_operators_are_refused():
    """Ordering industries would be meaningless, so > and < are not offered."""
    from screener import operators_for

    assert set(operators_for("sector")) == {"is", "is not"}
    assert ">" not in operators_for("sector")
    assert "is" not in operators_for("pe_ratio")


def test_sector_matching_ignores_case_and_padding():
    from screener import CATEGORICAL_OPERATORS

    assert CATEGORICAL_OPERATORS["is"]("Technology", " technology ")
    assert CATEGORICAL_OPERATORS["is not"]("Technology", "Healthcare")


def test_dividend_yield_is_percent_valued_either_way():
    """Yahoo's field is percent-valued in this client version, but the
    common assumption is that it is a fraction. Both readings must land on
    the same answer, cross-checked against dividendRate/price."""
    from financial_standardization import _dividend_yield_pct

    as_percent = _dividend_yield_pct({"dividendYield": 2.33, "dividendRate": 2.12}, 91.1)
    as_fraction = _dividend_yield_pct({"dividendYield": 0.0233, "dividendRate": 2.12}, 91.1)
    assert as_percent == pytest.approx(2.33, abs=0.01)
    assert as_fraction == pytest.approx(2.33, abs=0.05)


def test_dividend_yield_falls_back_to_rate_over_price():
    from financial_standardization import _dividend_yield_pct

    assert _dividend_yield_pct({"dividendRate": 2.12}, 91.1) == pytest.approx(2.327, abs=0.01)


def test_a_non_payer_reports_nothing_rather_than_zero():
    from financial_standardization import _dividend_yield_pct

    assert _dividend_yield_pct({"dividendYield": 0}, 100) is None
    assert _dividend_yield_pct({}, None) is None


def test_revenue_growth_refuses_a_non_positive_base():
    """Growth from zero or negative revenue is not a percentage, and
    dividing by it yields a large positive number that reads as health."""
    import datetime

    from screener import _revenue_growth_pct

    class Std:
        def __init__(self, history):
            self.revenue_history = history

    d = datetime.date
    value, note = _revenue_growth_pct(Std(((d(2025, 1, 1), 120.0), (d(2024, 1, 1), 100.0))))
    assert value == pytest.approx(20.0)
    assert note is None

    for bad in (((d(2025, 1, 1), 120.0), (d(2024, 1, 1), 0.0)),
                ((d(2025, 1, 1), 120.0), (d(2024, 1, 1), -50.0))):
        value, note = _revenue_growth_pct(Std(bad))
        assert value is None and note


def test_revenue_growth_needs_two_periods():
    import datetime

    from screener import _revenue_growth_pct

    class Std:
        revenue_history = ((datetime.date(2025, 1, 1), 120.0),)

    value, note = _revenue_growth_pct(Std())
    assert value is None and "two periods" in note


def test_revenue_growth_uses_the_two_most_recent_periods():
    """Order in the stored history is not relied upon."""
    import datetime

    from screener import _revenue_growth_pct

    d = datetime.date

    class Oldest_first:
        revenue_history = ((d(2023, 1, 1), 50.0), (d(2024, 1, 1), 100.0), (d(2025, 1, 1), 120.0))

    value, _ = _revenue_growth_pct(Oldest_first())
    assert value == pytest.approx(20.0)


# --- the template store -------------------------------------------------------

def test_starters_seed_once_and_all_run(templates):
    loaded = templates.load()
    assert [t.name for t in loaded] == [
        "Tech Stocks Under $100", "Value Plays: P/E < 15",
        "Dividend Stocks: Yield > 2%", "Growth: Revenue Growth > 20%"]
    for template in loaded:
        assert template.unknown_parts() == [], template.name
        assert template.universe, f"{template.name} has no universe"


def test_a_template_carries_its_universe(templates):
    """The whole point: the screener filters a list, so the list is part
    of the saved screen or it does not reproduce."""
    tech = templates.get("Tech Stocks Under $100")
    assert "INTC" in tech.universe
    assert len(tech.universe) >= 5


def test_deleting_every_template_does_not_re_seed(templates):
    for name in [t.name for t in templates.load()]:
        templates.delete(name)
    assert templates.load() == []
    assert templates.load() == []          # and still empty on a second read


def test_save_dedupes_and_normalises_the_universe(templates):
    ok, error = templates.save("Mine", [{"metric": "roe_pct", "operator": ">", "threshold": 15.0}],
                               [" aapl ", "MSFT", "aapl", ""])
    assert ok, error
    assert templates.get("Mine").universe == ("AAPL", "MSFT")


def test_saving_over_a_name_keeps_its_position(templates):
    templates.save("Mine", [{"metric": "roe_pct", "operator": ">", "threshold": 15.0}], ["AAPL"])
    templates.move("Mine", -1)
    before = [t.name for t in templates.load()]

    templates.save("Mine", [{"metric": "pe_ratio", "operator": "<", "threshold": 9.0}], ["KO"])
    assert [t.name for t in templates.load()] == before
    assert templates.get("Mine").universe == ("KO",)


def test_reordering(templates):
    names = [t.name for t in templates.load()]
    assert templates.move(names[1], -1)
    assert [t.name for t in templates.load()][0] == names[1]
    # Moving past an end is a no-op, not an error.
    assert templates.move([t.name for t in templates.load()][0], -1) is False
    assert templates.move([t.name for t in templates.load()][-1], +1) is False


def test_save_validation(templates):
    assert templates.save("", [{"metric": "pe_ratio", "operator": "<", "threshold": 9}], ["A"])[0] is False
    assert templates.save("X", [], ["A"])[0] is False
    assert templates.save("y" * 200, [{"metric": "pe_ratio", "operator": "<", "threshold": 9}], ["A"])[0] is False


def test_an_unknown_criterion_is_kept_and_reported(templates):
    """Dropping it would make the screen return more matches than it claims."""
    templates.save("Future", [{"metric": "quantum_alpha", "operator": "<", "threshold": 1.0},
                              {"metric": "pe_ratio", "operator": "<", "threshold": 15.0}], ["AAPL"])
    saved = templates.get("Future")
    assert len(saved.criteria) == 2
    problems = saved.unknown_parts()
    assert any("quantum_alpha" in p for p in problems)


def test_an_operator_invalid_for_its_metric_is_reported(templates):
    templates.save("Odd", [{"metric": "sector", "operator": "<", "threshold": "Technology"}], ["AAPL"])
    assert templates.get("Odd").unknown_parts()


def test_an_unevaluable_criterion_never_silently_passes():
    """A template from a newer version must not be treated as a match."""
    from screener import ScreenCriterion, operators_for

    assert "~=" not in operators_for("pe_ratio")


def test_summary_renders_currency_as_a_prefix(templates):
    assert "< $100" in templates.get("Tech Stocks Under $100").summary


def test_a_corrupt_store_is_never_overwritten(templates, sandbox):
    """A single bad byte must not cost someone their saved screeners.

    Treating an unreadable file as "empty" is the dangerous reading: the
    next load would seed the starter set straight over the top of it.
    """
    path = sandbox / templates.STORE_FILENAME
    path.write_text("{ not json")

    assert templates.load() == []
    assert templates.store_is_corrupt()
    assert path.read_text() == "{ not json"      # untouched

    # And nothing that writes may touch it either.
    ok, error = templates.save("X", [{"metric": "pe_ratio", "operator": "<", "threshold": 9.0}], ["AAPL"])
    assert not ok and error
    assert templates.delete("anything") is False
    assert templates.move("anything", -1) is False
    assert path.read_text() == "{ not json"


def test_a_store_of_the_wrong_shape_is_also_treated_as_corrupt(templates, sandbox):
    (sandbox / templates.STORE_FILENAME).write_text('["not", "a", "dict"]')
    assert templates.load() == []
    assert templates.store_is_corrupt()


def test_a_healthy_store_is_not_flagged_corrupt(templates):
    templates.load()
    assert not templates.store_is_corrupt()


def test_reset_restores_the_starters(templates):
    for name in [t.name for t in templates.load()]:
        templates.delete(name)
    templates.reset_to_starters()
    assert len(templates.load()) == 4


def test_a_categorical_threshold_survives_the_run_path():
    """Regression: the screen-execution path coerced every threshold with
    float(), which is right for all fourteen original metrics and raises
    "could not convert string to float: 'Technology'" the moment a Sector
    criterion reaches it. Caught by running a saved screener, not by a
    unit test — so this pins the engine end of it."""
    from screener import ScreenCriterion, operators_for

    criterion = ScreenCriterion(metric="sector", operator="is", threshold="Technology")
    assert isinstance(criterion.threshold, str)
    operator = operators_for(criterion.metric)[criterion.operator]
    assert operator("Technology", criterion.threshold)
    assert not operator("Healthcare", criterion.threshold)


def test_mixed_numeric_and_categorical_criteria_evaluate_together():
    """The "Tech Stocks Under $100" shape: one of each."""
    from screener import operators_for

    sector_ok = operators_for("sector")["is"]("Technology", "Technology")
    price_ok = operators_for("price")["<"](90.07, 100.0)
    assert sector_ok and price_ok
    assert not operators_for("price")["<"](309.35, 100.0)
