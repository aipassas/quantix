"""The bond fund screener.

The finding worth pinning: the ETF universe alone does not contain the
bond market. Yahoo's top-250 screen returned 40 bond funds and was
missing 19 of the 24 largest bond ETFs, including AGG, BND, LQD, HYG and
TLT — so "Treasuries for Safety" returned nothing at all, because the
shortest treasury fund in that universe was IEI at 4.09 years.
"""
import pytest

import bond_screener as bs


def _row(symbol="X", name="Test Bond ETF", **over):
    fields = dict(fund_type=bs.classify(name), yield_pct=4.0, duration=5.0,
                  spread_bps=50.0, expense_ratio_pct=0.10, assets=1e9,
                  return_1y_pct=3.0)
    fields.update(over)
    return bs.BondFundRow(symbol=symbol, name=name, **fields)


# --- classification -----------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("iShares 7-10 Year Treasury Bond ETF", "Treasury"),
    ("iShares iBoxx $ High Yield Corporate Bond ETF", "High Yield"),
    ("iShares iBoxx $ Investment Grade Corporate Bond ETF", "Corporate (IG)"),
    ("iShares TIPS Bond ETF", "Inflation-Protected"),
    ("iShares National Muni Bond ETF", "Municipal"),
    ("iShares Core U.S. Aggregate Bond ETF", "Aggregate"),
    ("iShares MBS ETF", "Mortgage"),
    ("iShares J.P. Morgan USD Emerging Markets Bond ETF", "Emerging Market"),
    ("iShares Convertible Bond ETF", "Convertible"),
    ("Some Unlabelled Fund", "Other"),
])
def test_fund_type_is_matched_on_the_name(name, expected):
    assert bs.classify(name) == expected


def test_high_yield_wins_over_corporate():
    """A fund named "High Yield Corporate" is high yield. Ordering the
    rules most-specific-first is what makes that true."""
    assert bs.classify("iShares iBoxx $ High Yield Corporate Bond ETF") \
        == "High Yield"
    assert bs.classify("VanEck Fallen Angel High Yield Bond ETF") == "High Yield"


def test_bond_funds_are_recognised_and_equity_funds_are_not():
    assert bs.looks_like_bond_fund("iShares Core U.S. Aggregate Bond ETF")
    assert bs.looks_like_bond_fund("Vanguard Short-Term Treasury Index Fund")
    assert not bs.looks_like_bond_fund("SPDR S&P 500 ETF Trust")
    assert not bs.looks_like_bond_fund("Invesco QQQ Trust")


# --- the seeded universe ------------------------------------------------------

def test_the_core_bond_funds_are_seeded_in():
    """THE FINDING. The ETF universe was missing 19 of the 24 largest
    bond ETFs, so the screener could not see its own subject."""
    for symbol in ("AGG", "BND", "LQD", "HYG", "TLT", "SHY", "MUB", "TIP"):
        assert symbol in bs.CORE_BOND_FUNDS, symbol
    assert len(set(bs.CORE_BOND_FUNDS)) == len(bs.CORE_BOND_FUNDS), "no dupes"


def test_the_seed_list_spans_the_whole_credit_and_duration_range():
    """A seed list that was all treasuries would leave the same hole in a
    different place."""
    types = {bs.classify(n) for n in (
        "iShares Core U.S. Aggregate Bond ETF",
        "iShares 20+ Year Treasury Bond ETF",
        "iShares iBoxx $ High Yield Corporate Bond ETF",
        "iShares National Muni Bond ETF")}
    assert {"Aggregate", "Treasury", "High Yield", "Municipal"} <= types
    # Short AND long treasuries, so a duration filter has both ends.
    assert "SGOV" in bs.CORE_BOND_FUNDS and "TLT" in bs.CORE_BOND_FUNDS


def test_a_seeded_yield_is_normalised_from_a_fraction(monkeypatch):
    """`yield` arrives as 0.0466 from info while the screen's own rows are
    percent-valued — the 100x trap, in one function."""
    import sys
    import types

    class _Ticker:
        def __init__(self, symbol):
            self.info = {"longName": "Test Bond ETF", "yield": 0.0466,
                         "netExpenseRatio": 0.14, "totalAssets": 3.2e10}

    fake = types.ModuleType("yfinance")
    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    rows = bs._load_seeded(("LQD",))
    assert len(rows) == 1
    assert rows[0].yield_pct == pytest.approx(4.66)
    assert rows[0].expense_ratio_pct == pytest.approx(0.14)


def test_a_seeded_yield_already_in_percent_is_left_alone(monkeypatch):
    import sys
    import types

    class _Ticker:
        def __init__(self, symbol):
            self.info = {"longName": "Test Bond ETF", "yield": 4.66}

    fake = types.ModuleType("yfinance")
    fake.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    assert bs._load_seeded(("X",))[0].yield_pct == pytest.approx(4.66)


def test_a_seed_that_fails_is_skipped_not_fatal(monkeypatch):
    import sys
    import types

    def _ticker(symbol):
        if symbol == "BAD":
            raise RuntimeError("no such fund")
        return type("T", (), {"info": {"longName": f"{symbol} Bond ETF"}})()

    fake = types.ModuleType("yfinance")
    fake.Ticker = _ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    rows = bs._load_seeded(("GOOD", "BAD", "ALSOGOOD"))
    assert [r.symbol for r in rows] == ["GOOD", "ALSOGOOD"]


def test_seeding_nothing_is_not_an_error():
    assert bs._load_seeded(()) == []


# --- filtering ----------------------------------------------------------------

def test_a_numeric_filter_selects_on_the_right_side():
    rows = [_row("A", duration=1.0), _row("B", duration=5.0),
            _row("C", duration=12.0)]
    passed, _ = bs.run(rows, [bs.BondCriterion("duration", "<", 3.0)])
    assert [m.row.symbol for m in passed] == ["A"]


def test_a_text_filter_selects_on_the_fund_type():
    rows = [_row("A", name="iShares 20+ Year Treasury Bond ETF"),
            _row("B", name="iShares iBoxx $ High Yield Corporate Bond ETF")]
    passed, _ = bs.run(rows, [bs.BondCriterion("fund_type", "is", "Treasury")])
    assert [m.row.symbol for m in passed] == ["A"]
    passed, _ = bs.run(rows, [bs.BondCriterion("fund_type", "is not", "Treasury")])
    assert [m.row.symbol for m in passed] == ["B"]


def test_a_fund_that_does_not_report_a_metric_is_set_aside_not_failed():
    """None is NOT False: a fund whose duration could not be measured has
    not failed a duration filter."""
    rows = [_row("A", duration=2.0), _row("B", duration=None)]
    passed, unjudged = bs.run(rows, [bs.BondCriterion("duration", "<", 3.0)])
    assert [m.row.symbol for m in passed] == ["A"]
    assert [m.row.symbol for m in unjudged] == ["B"]
    assert unjudged[0].unmeasured == ("Duration (years)",)


def test_an_unknown_metric_counts_as_unmeasured_rather_than_passing():
    """A typo'd filter must match nothing, not everything."""
    rows = [_row("A")]
    passed, unjudged = bs.run(rows, [bs.BondCriterion("nonsense", ">", 1)])
    assert passed == []
    assert unjudged and "unknown filter" in unjudged[0].unmeasured[0]


def test_every_criterion_must_pass():
    rows = [_row("A", duration=2.0, yield_pct=3.0),
            _row("B", duration=2.0, yield_pct=5.0)]
    passed, _ = bs.run(rows, [bs.BondCriterion("duration", "<", 3.0),
                              bs.BondCriterion("yield_pct", ">", 4.0)])
    assert [m.row.symbol for m in passed] == ["B"]


def test_no_criteria_passes_everything():
    rows = [_row("A"), _row("B")]
    passed, unjudged = bs.run(rows, [])
    assert len(passed) == 2 and unjudged == []


def test_text_metrics_get_categorical_operators():
    assert set(bs.operators_for("fund_type")) == set(
        __import__("etf_screener").CATEGORICAL_OPERATORS)
    assert set(bs.operators_for("duration")) == set(
        __import__("etf_screener").OPERATORS)


# --- presets ------------------------------------------------------------------

def test_every_preset_uses_metrics_that_exist():
    """A preset naming a metric the screener does not have would silently
    match nothing."""
    for preset in bs.PRESETS:
        assert preset.criteria, preset.name
        assert preset.detail, preset.name
        for criterion in preset.criteria:
            assert criterion.metric in bs.METRICS_BY_KEY, (
                preset.name, criterion.metric)
            assert criterion.operator in bs.operators_for(criterion.metric)


def test_the_treasuries_preset_finds_short_treasuries():
    """It returned ZERO against the unseeded universe, whose shortest
    treasury fund was IEI at 4.09 years."""
    rows = [_row("SGOV", name="iShares 0-3 Month Treasury Bond ETF",
                 duration=-0.02),
            _row("SHY", name="iShares 1-3 Year Treasury Bond ETF",
                 duration=1.43),
            _row("TLT", name="iShares 20+ Year Treasury Bond ETF",
                 duration=13.2),
            _row("HYG", name="iShares High Yield Corporate Bond ETF",
                 duration=1.85)]
    preset = next(p for p in bs.PRESETS if p.name == "Treasuries for Safety")
    passed, _ = bs.run(rows, preset.criteria)
    assert {m.row.symbol for m in passed} == {"SGOV", "SHY"}


def test_the_high_yield_preset_finds_only_high_yield():
    rows = [_row("HYG", name="iShares High Yield Corporate Bond ETF"),
            _row("LQD", name="iShares Investment Grade Corporate Bond ETF")]
    preset = next(p for p in bs.PRESETS if p.name.startswith("High Yield"))
    passed, _ = bs.run(rows, preset.criteria)
    assert [m.row.symbol for m in passed] == ["HYG"]


def test_the_ladder_preset_is_absent_and_says_why():
    """A ladder is individual bonds maturing on known dates. Funds do not
    mature, so a mix of funds at different durations is a different
    instrument — calling it a ladder would be the wrong word."""
    assert not any("ladder" in p.name.lower() for p in bs.PRESETS)
    assert "do not mature" in bs.LADDER_UNAVAILABLE
    assert "individual bonds" in bs.LADDER_UNAVAILABLE


# --- formatting ---------------------------------------------------------------

def test_money_is_compact_and_percents_carry_their_unit():
    assert bs.format_value("assets", 3.2e10) == "$32.0B"
    assert bs.format_value("yield_pct", 4.66) == "4.66%"
    assert bs.format_value("spread_bps", 202.0) == "202bp"
    assert bs.format_value("duration", 13.21) == "13.21"


def test_a_missing_value_is_never_a_zero():
    for key in ("yield_pct", "duration", "spread_bps", "assets"):
        assert bs.format_value(key, None) == "Not reported"
    assert bs.format_value("nonsense", 1.0) == "Not reported"


def test_a_criterion_describes_itself_readably():
    assert bs.describe(bs.BondCriterion("duration", "<", 3.0)) \
        == "Duration (years) < 3.00"
    assert bs.describe(bs.BondCriterion("fund_type", "is", "Treasury")) \
        == "Fund type is Treasury"


def test_the_rate_loss_shortcut_is_minus_the_duration():
    """The task's "max loss if rates +100bps" filter."""
    assert _row(duration=5.0).rate_loss_100bp_pct == pytest.approx(-5.0)
    assert _row(duration=None).rate_loss_100bp_pct is None


def test_individual_bonds_are_declared_out_of_scope():
    assert "CUSIP" in bs.INDIVIDUAL_BONDS_NOT_SCREENED
    assert "FUNDS" in bs.INDIVIDUAL_BONDS_NOT_SCREENED


# --- live ---------------------------------------------------------------------

@pytest.mark.live
def test_the_real_universe_contains_the_major_bond_funds():
    """The regression guard for the seeding fix."""
    rows, error = bs.load_bond_universe()
    assert error is None, error
    have = {r.symbol for r in rows}
    for symbol in ("AGG", "BND", "LQD", "HYG", "TLT", "SHY"):
        assert symbol in have, f"{symbol} missing from the bond universe"
    assert len(rows) > 50, len(rows)


@pytest.mark.live
def test_measured_durations_rank_the_treasury_ladder():
    """SGOV (0-3 month bills) through TLT (20+ years) must increase."""
    rows, _ = bs.load_bond_universe()
    by_symbol = {r.symbol: r for r in rows}
    ladder = [by_symbol[s].duration
              for s in ("SGOV", "SHY", "IEI", "IEF", "TLT")
              if s in by_symbol and by_symbol[s].duration is not None]
    assert len(ladder) >= 4
    assert ladder == sorted(ladder), ladder


@pytest.mark.live
def test_a_rate_hedged_fund_measures_a_negative_duration():
    """Not a bug — these funds short treasuries to strip out rate risk,
    and a measured duration catches that where a positive-only field
    never could. Measured: HYHG -1.62, IGBH -0.08, SGOV -0.02."""
    rows, _ = bs.load_bond_universe()
    negatives = [r for r in rows if r.duration is not None and r.duration < 0]
    assert negatives, "no fund measured a negative duration"


@pytest.mark.live
def test_every_preset_returns_something_against_the_real_universe():
    """A preset that finds nothing is broken UX, and that is exactly what
    the unseeded universe produced."""
    rows, _ = bs.load_bond_universe()
    for preset in bs.PRESETS:
        passed, _ = bs.run(rows, preset.criteria)
        assert passed, f"{preset.name} matched nothing"


# --- how the bond panel is wired ---------------------------------------------

def test_the_bond_panel_reads_only_fields_that_exist_on_the_profile():
    """This shipped broken once: the panel read `EtfProfile.name`, which
    does not exist — the profile carries a symbol, category, family and
    legal type, and the fund's NAME lives on the bundle's info. An
    AttributeError there takes the whole Chart Workspace down."""
    import ast
    import dataclasses
    from pathlib import Path

    import etf_analysis

    finance = (Path(__file__).resolve().parent.parent
               / "finance.py").read_text(encoding="utf-8")
    fields = {f.name for f in dataclasses.fields(etf_analysis.EtfProfile)}
    fields |= {"ok", "error"}

    tree = ast.parse(finance)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "_rk_prof"):
            assert node.attr in fields, (
                f"finance.py reads _rk_prof.{node.attr}, which is not on "
                f"EtfProfile")


def test_the_bond_panel_is_gated_on_the_fund_actually_holding_bonds():
    """A duration and a credit spread on an equity fund would be
    arithmetic performed on the wrong instrument."""
    from pathlib import Path

    finance = (Path(__file__).resolve().parent.parent
               / "finance.py").read_text(encoding="utf-8")
    assert "bond_screener.looks_like_bond_fund(_bd_name)" in finance
    assert "if _bd_is_bond:" in finance
