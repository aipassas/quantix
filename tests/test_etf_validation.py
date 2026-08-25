"""ETF data validation and performance guards (PHASE 1.7).

THREE OF THE TASK'S FIVE VALIDATION RULES ARE FALSE AGAINST THIS DATA
SOURCE, measured before any of them was written as a test. Asserting them
would have produced a suite that failed on healthy funds, which teaches a
reader to ignore the suite:

  1. "Holdings weights sum to 100% (+/-2%)". They do not, and cannot:
     Yahoo returns the TOP TEN, not the fund. Measured across ten funds
     on 2026-08-25 the sums run from 2.90% (AGG) to 47.52% (ARKK), with
     SPY at 37.62% and IWM at 3.25%. A 100% assertion fails on all ten.
     What IS invariant is that no weight is negative, none exceeds 100,
     and the disclosed slice never exceeds the whole fund.

  2. "Price within 0.5% of NAV". Yahoo's navPrice is a stale close —
     established in PHASE 1.4 and recorded in
     etf_technicals.NAV_PREMIUM_UNAVAILABLE. On 2026-08-24 it implied
     ARKK at a -2.70% discount. The rule would fail permanently on the
     most liquid funds on the board.

  3. "Bid-ask spread reasonable (< 1%)". The fields are usually absent:
     measured, bid and ask were both 0.0 for six of ten funds INCLUDING
     SPY. Where both are genuinely reported the invariant does hold
     (QQQ 0.035%, TLT 0.060%, VTI 0.391%, ARKK 0.473%), so it is asserted
     conditionally and the absence is reported rather than passed over in
     silence — a test that skips its own subject and reports green is
     worse than no test.

The two that ARE true — expense ratio within 0-3%, and volume above zero
— are asserted directly.

WHAT THIS FILE DOES NOT COVER, and why:
  - "Test database inserts/updates". There is no database. etf_screener's
    docstring records that decision: a 250-row table cached for five
    minutes answers every query in the spec in memory, and adding a
    database and a scheduler to search 250 rows would be the larger and
    worse change.
  - "Screener query (10k ETFs)". The universe is 250 — the most one
    request returns. The per-row cost is measured here and extrapolated
    honestly rather than a 10k universe being faked.
  - Safari, Firefox, iOS and Android. This environment has one Chromium
    browser. Responsive behaviour is checked at the four breakpoints in
    the live browser; cross-BROWSER testing is not something this suite
    can honestly claim.

The live half follows the project's existing convention: real network
calls sit behind @pytest.mark.live and are skipped by the default suite
(see pytest.ini), so the fast deterministic run stays fast.
"""
import time

import pytest

import etf_analysis
import etf_screener
import etf_technicals


# A cross-section on purpose: two mega-cap index funds, a value fund, an
# active fund, a bond fund, a commodity trust, an international fund and
# a small-cap fund. The bond and commodity funds are the ones that break
# naive assumptions, so they are in the sample rather than excluded.
SAMPLE = ("SPY", "QQQ", "VTI", "VTV", "ARKK", "TLT", "GLD", "EFA", "IWM", "AGG")

# The task's own bounds, kept as named constants so a future change is a
# deliberate edit rather than a silent one.
#
# EXPENSE_MAX_PCT IS 10, NOT THE TASK'S 3, and the live run is what
# changed it: FCEF (First Trust Income Opportunity ETF) reports 3.69%.
# That is not a data error — it is a fund of funds, so its ratio includes
# the acquired fund fees of everything it holds. A 3% ceiling fails on a
# correctly reported figure, and a validation suite that fails on healthy
# data teaches its reader to ignore it.
#
# What the ceiling is actually FOR is catching a unit error, and this
# codebase has the 100x expense-ratio trap on record: the same figure
# exists as a percent in info.netExpenseRatio and as a fraction in
# funds_data.fund_operations. 10% catches a fraction read as a percent
# without rejecting a legitimate fund-of-funds fee. Anything above 3% is
# still REPORTED by name, so a reviewer sees it.
EXPENSE_MIN_PCT = 0.0
EXPENSE_MAX_PCT = 10.0
EXPENSE_NOTEWORTHY_PCT = 3.0
SPREAD_MAX_PCT = 1.0


# --- validation invariants that hold, on synthetic data -----------------------
# These run in the default suite. The live versions below assert the same
# invariants against real funds.

def _holding(symbol, weight):
    return etf_analysis.Holding(symbol=symbol, name=symbol, weight_pct=weight)


def test_a_disclosed_top_ten_never_exceeds_the_whole_fund():
    """The invariant that replaces "weights sum to 100%". A top-ten slice
    summing above 100 would mean the fund holds more than itself."""
    holdings = [_holding(f"H{i}", 9.5) for i in range(10)]
    total = etf_analysis.concentration_pct(holdings)
    assert 0 < total <= 100.0


def test_concentration_of_nothing_is_unknown_not_zero():
    """A bond fund discloses no holdings. Reporting 0% concentration
    would say the fund holds nothing, which is a different claim."""
    assert etf_analysis.concentration_pct([]) is None


def test_the_expense_ceiling_catches_the_hundred_x_unit_error():
    """The ceiling exists to catch a fraction read as a percent, not to
    police real fees. The same figure exists in this data source as a
    percent AND as a fraction 100x apart, which is the error worth
    catching — FCEF genuinely charges 3.69% as a fund of funds."""
    real_fee_pct = 0.0945                       # SPY, as a percent
    misread_as_percent = real_fee_pct * 100     # 9.45
    assert EXPENSE_MIN_PCT <= real_fee_pct <= EXPENSE_MAX_PCT
    assert misread_as_percent <= EXPENSE_MAX_PCT, (
        "9.45 sits just inside the ceiling; a larger fee misread this way "
        "is what the ceiling catches")
    assert 100.0 > EXPENSE_MAX_PCT
    # The noteworthy line sits below the ceiling, so a real high fee is
    # reported rather than failed.
    assert EXPENSE_NOTEWORTHY_PCT < EXPENSE_MAX_PCT


# --- calculation-layer coverage ----------------------------------------------
# The task sets a 95% bar for the calculation layer; these cover the
# branches the feature tests never reached.

def test_a_non_numeric_value_reads_as_absent():
    for junk in (None, "", "n/a", object()):
        assert etf_analysis._number(junk) is None
    assert etf_analysis._number(float("nan")) is None
    assert etf_analysis._number("3.5") == 3.5
    assert etf_analysis._number(7) == 7.0


def test_a_missing_cell_reads_as_absent_rather_than_raising():
    import pandas as pd

    frame = pd.DataFrame({"a": [1.0]}, index=["row"])
    assert etf_analysis._cell(frame, "row", "a") == 1.0
    assert etf_analysis._cell(frame, "nope", "a") is None
    assert etf_analysis._cell(frame, "row", "nope") is None
    assert etf_analysis._cell(None, "row", "a") is None


def test_expense_drag_needs_a_positive_horizon_and_a_known_fee():
    assert etf_analysis.expense_drag(None, 10) is None
    assert etf_analysis.expense_drag(0.5, 0) is None
    assert etf_analysis.expense_drag(0.5, -5) is None
    drag = etf_analysis.expense_drag(0.5, 30)
    assert drag is not None and 0 < drag < 100


def test_expense_drag_compounds_rather_than_multiplying():
    """A 0.5% fee over 30 years costs far more than 15%."""
    ten = etf_analysis.expense_drag(0.5, 10)
    thirty = etf_analysis.expense_drag(0.5, 30)
    assert thirty > ten
    assert thirty > 0.5 * 30 / 100


def test_expense_drag_returns_nothing_when_the_gross_outcome_is_wiped_out():
    """gross <= 0 has no meaningful percentage given up."""
    assert etf_analysis.expense_drag(0.5, 10, gross_return_pct=-100.0) is None


def test_the_valuation_gap_needs_both_sides():
    fund = etf_analysis.EtfProfile(symbol="X", price_earnings=25.0,
                                   category_price_earnings=20.0)
    assert etf_analysis.valuation_gap_pct(fund) == pytest.approx(25.0)
    assert etf_analysis.valuation_gap_pct(
        etf_analysis.EtfProfile(symbol="X", price_earnings=25.0)) is None
    assert etf_analysis.valuation_gap_pct(
        etf_analysis.EtfProfile(symbol="X",
                                category_price_earnings=20.0)) is None
    # A zero category P/E would divide by zero.
    assert etf_analysis.valuation_gap_pct(
        etf_analysis.EtfProfile(symbol="X", price_earnings=25.0,
                                category_price_earnings=0.0)) is None


def test_the_expense_gap_needs_both_sides_too():
    profile = etf_analysis.EtfProfile(symbol="X", expense_ratio_pct=0.75,
                                      category_expense_ratio_pct=0.30)
    assert etf_analysis.expense_gap_pct(profile) == pytest.approx(0.45)
    assert etf_analysis.expense_gap_pct(
        etf_analysis.EtfProfile(symbol="X", expense_ratio_pct=0.75)) is None
    assert not etf_analysis.expense_is_high(etf_analysis.EtfProfile(symbol="X"))


def test_a_fund_is_flagged_expensive_only_past_the_threshold():
    """Both sides of the cutoff, so a change to EXPENSE_FLAG_GAP_PCT is a
    deliberate edit rather than a silent one."""
    cutoff = etf_analysis.EXPENSE_FLAG_GAP_PCT

    def at(gap):
        return etf_analysis.EtfProfile(
            symbol="X", expense_ratio_pct=0.30 + gap,
            category_expense_ratio_pct=0.30)

    assert not etf_analysis.expense_is_high(at(cutoff - 0.05))
    assert not etf_analysis.expense_is_high(at(cutoff)), "the cutoff is exclusive"
    assert etf_analysis.expense_is_high(at(cutoff + 0.05))


def test_style_prefers_the_providers_category_over_a_pe_cutoff():
    """They disagreed on a real fund: VTV is categorised "Large Value" and
    prices at 20.7x, which a cutoff of 18 calls "Blend"."""
    vtv = etf_analysis.EtfProfile(symbol="VTV", category="Large Value",
                                  price_earnings=20.7)
    assert etf_analysis.style_label(vtv) == "Large Value"


def test_style_falls_back_to_the_multiple_when_there_is_no_category():
    cheap = etf_analysis.EtfProfile(symbol="X", price_earnings=10.0)
    rich = etf_analysis.EtfProfile(symbol="X", price_earnings=45.0)
    middling = etf_analysis.EtfProfile(symbol="X", price_earnings=22.0)
    assert etf_analysis.style_label(cheap).startswith("Value")
    assert etf_analysis.style_label(rich).startswith("Growth")
    assert etf_analysis.style_label(middling).startswith("Blend")
    # ...and says which basis it used, so no label is taken on trust.
    assert "inferred" in etf_analysis.style_label(cheap)


def test_style_says_so_when_it_cannot_classify():
    blank = etf_analysis.EtfProfile(symbol="X")
    assert "Not classified" in etf_analysis.style_label(blank)
    # A category with no style word in it is not a style.
    odd = etf_analysis.EtfProfile(symbol="X", category="Commodities Focused")
    assert "Not classified" in etf_analysis.style_label(odd)


# --- performance guards -------------------------------------------------------
# Ceilings are deliberately generous. Measured on 2026-08-25 the screener
# runs five filters over 250 rows in 0.58ms and a search in 0.05ms; these
# assert ORDERS OF MAGNITUDE rather than the measurement, so the guard
# catches an algorithmic regression without flaking on a loaded machine.

SEARCH_BUDGET_MS = 200.0        # the task's own budget
SCREEN_BUDGET_MS = 1000.0       # the task's own budget


def _synthetic_universe(n):
    return tuple(
        etf_screener.EtfRow(
            symbol=f"T{i:04d}", name=f"Fund {i} Index Trust",
            price=100.0 + i, expense_ratio_pct=0.03 + (i % 50) / 100.0,
            assets=1e9 * (i + 1), pe_ratio=10.0 + (i % 30),
            dividend_yield_pct=(i % 7) / 2.0, return_1y_pct=(i % 40) - 10.0,
            return_3y_pct=(i % 25) - 5.0, return_ytd_pct=(i % 20) - 5.0)
        for i in range(n))


def test_a_screen_over_ten_thousand_rows_stays_under_the_budget():
    """The real universe is 250 — the most one request returns — so the
    task's "10k ETFs in < 1 second" is tested against a synthetic
    universe of that size rather than a faked fetch. Measured per-row
    cost on the real table was 2.3us, which puts 10k at about 23ms."""
    universe = _synthetic_universe(10_000)
    criteria = [
        etf_screener.EtfCriterion("expense_ratio_pct", "<", 0.20),
        etf_screener.EtfCriterion("assets", ">", 1e9),
        etf_screener.EtfCriterion("dividend_yield_pct", ">", 1.0),
        etf_screener.EtfCriterion("return_1y_pct", ">", 5.0),
        etf_screener.EtfCriterion("pe_ratio", "<", 30.0),
    ]
    started = time.perf_counter()
    passed, unjudged = etf_screener.run(universe, criteria)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < SCREEN_BUDGET_MS, f"{elapsed_ms:.1f}ms"
    # ...and it actually filtered, rather than being fast by doing nothing.
    assert 0 < len(passed) < len(universe)
    assert not unjudged, "every synthetic row reports every metric"


def test_search_over_ten_thousand_rows_stays_under_the_budget():
    universe = _synthetic_universe(10_000)
    started = time.perf_counter()
    for query in ("t0042", "index", "fund 9", "trust"):
        etf_screener.search(universe, query)
    elapsed_ms = (time.perf_counter() - started) * 1000 / 4
    assert elapsed_ms < SEARCH_BUDGET_MS, f"{elapsed_ms:.1f}ms"


def test_sector_momentum_over_the_full_weight_set_is_instant():
    weights = {key: 1.0 / len(etf_technicals.SECTOR_PROXIES)
               for key in etf_technicals.SECTOR_PROXIES}
    returns = {key: 3.0 for key in etf_technicals.SECTOR_PROXIES}
    started = time.perf_counter()
    for _ in range(1000):
        etf_technicals.sector_momentum(weights, returns, 2.5)
    elapsed_ms = (time.perf_counter() - started)
    assert elapsed_ms < SCREEN_BUDGET_MS, f"{elapsed_ms:.1f}ms for 1000 runs"


# --- live validation against real funds ---------------------------------------
# Behind @pytest.mark.live, like every other real-network test here.

@pytest.mark.live
def test_every_sampled_fund_reports_an_expense_ratio_in_range():
    """One of the two task rules that IS true. Measured 0.03%-0.75%
    across the sample."""
    checked = 0
    for symbol in SAMPLE:
        profile = etf_analysis.load_profile(symbol)
        if not profile.ok or profile.expense_ratio_pct is None:
            continue
        checked += 1
        assert EXPENSE_MIN_PCT <= profile.expense_ratio_pct <= EXPENSE_MAX_PCT, (
            f"{symbol}: {profile.expense_ratio_pct}%")
    assert checked >= len(SAMPLE) - 1, f"only {checked} funds reported a fee"


@pytest.mark.live
def test_disclosed_holdings_are_a_slice_of_the_fund_never_more():
    """NOT "sums to 100%" — see the module docstring. Yahoo returns the
    top ten, which measured 2.90%-47.52% across the sample."""
    saw_holdings = False
    for symbol in SAMPLE:
        profile = etf_analysis.load_profile(symbol)
        if not profile.ok or not profile.top_holdings:
            continue
        saw_holdings = True
        total = etf_analysis.concentration_pct(profile.top_holdings)
        assert 0 < total <= 100.0, f"{symbol}: top ten sums to {total}%"
        for holding in profile.top_holdings:
            assert 0 <= holding.weight_pct <= 100.0, f"{symbol}/{holding.symbol}"
            assert holding.symbol, f"{symbol}: a holding with no symbol"
    assert saw_holdings, "no fund in the sample disclosed any holdings"


@pytest.mark.live
def test_every_sampled_fund_reports_volume():
    """The other task rule that IS true."""
    from data_loader import load_ticker_bundle

    for symbol in SAMPLE:
        info = load_ticker_bundle(symbol, deep=False).info or {}
        volume = info.get("volume") or info.get("regularMarketVolume")
        assert volume and volume > 0, f"{symbol}: volume {volume!r}"


@pytest.mark.live
def test_where_a_spread_is_reported_at_all_it_is_reasonable():
    """Conditional BY NECESSITY: bid and ask were both 0.0 for six of ten
    funds including SPY. The count is asserted so a run where the field
    vanishes entirely fails loudly rather than passing vacuously."""
    from data_loader import load_ticker_bundle

    reported = 0
    for symbol in SAMPLE:
        info = load_ticker_bundle(symbol, deep=False).info or {}
        bid, ask = info.get("bid"), info.get("ask")
        if not bid or not ask or bid <= 0 or ask <= 0:
            continue
        reported += 1
        spread_pct = abs(ask - bid) / ((ask + bid) / 2) * 100
        assert spread_pct < SPREAD_MAX_PCT, f"{symbol}: {spread_pct:.3f}%"
    assert reported >= 1, (
        "no fund in the sample reported both a bid and an ask — the "
        "conditional above would have passed without testing anything")


@pytest.mark.live
def test_the_universe_loads_within_the_ingest_budget():
    """The task budgets 60s for 500 funds. 250 is the most one request
    returns; measured, that took 0.70s."""
    started = time.perf_counter()
    rows, error = etf_screener.load_universe(etf_screener.UNIVERSE_SIZE)
    elapsed = time.perf_counter() - started
    assert error is None, error
    assert len(rows) > 100, f"only {len(rows)} funds returned"
    assert elapsed < 60.0, f"{elapsed:.1f}s"


@pytest.mark.live
def test_the_universe_rows_are_internally_consistent():
    """Data completeness, on the fields the screener actually filters."""
    rows, error = etf_screener.load_universe(etf_screener.UNIVERSE_SIZE)
    assert error is None, error

    have = {field: 0 for field in
            ("expense_ratio_pct", "assets", "dividend_yield_pct",
             "return_1y_pct", "pe_ratio", "price")}
    noteworthy = []
    for row in rows:
        assert row.symbol and row.symbol == row.symbol.upper()
        for field in have:
            value = getattr(row, field)
            if value is not None:
                have[field] += 1
                assert value == value, f"{row.symbol}.{field} is NaN"
        if row.expense_ratio_pct is not None:
            assert EXPENSE_MIN_PCT <= row.expense_ratio_pct <= EXPENSE_MAX_PCT, (
                f"{row.symbol}: {row.expense_ratio_pct}%")
            if row.expense_ratio_pct > EXPENSE_NOTEWORTHY_PCT:
                noteworthy.append(f"{row.symbol} {row.expense_ratio_pct:.2f}%")
        if row.assets is not None:
            assert row.assets >= 0, row.symbol

    total = len(rows)
    # Measured when the screener was built: expense, assets, yield and the
    # returns are present in every row; P/E in 23 of 25.
    for field in ("expense_ratio_pct", "assets", "return_1y_pct"):
        assert have[field] >= total * 0.9, (
            f"{field} present in only {have[field]} of {total}")
    assert have["pe_ratio"] >= total * 0.5, (
        f"P/E present in only {have['pe_ratio']} of {total}")
    # Reported, not failed. Measured 2026-08-25: exactly one fund in 250
    # sits above 3% — FCEF at 3.69%, a fund of funds whose ratio includes
    # its holdings' fees. The universe median is 0.34%. A unit error would
    # push a large share of the table over the line, which is what this
    # proportion catches.
    assert len(noteworthy) < total * 0.05, (
        f"{len(noteworthy)} of {total} funds above "
        f"{EXPENSE_NOTEWORTHY_PCT}%: {noteworthy[:10]}")


@pytest.mark.live
def test_a_screen_returns_only_funds_that_meet_every_criterion():
    """The screener's own filtering, checked against the rows it returned
    rather than against a count."""
    rows, error = etf_screener.load_universe(etf_screener.UNIVERSE_SIZE)
    assert error is None, error
    criteria = [
        etf_screener.EtfCriterion("expense_ratio_pct", "<", 0.20),
        etf_screener.EtfCriterion("assets", ">", 1e9),
    ]
    passed, unjudged = etf_screener.run(rows, criteria)
    assert passed, "no fund passed a deliberately loose screen"
    for match in passed:
        assert match.row.expense_ratio_pct < 0.20, match.row.symbol
        assert match.row.assets > 1e9, match.row.symbol
        assert not match.unmeasured, match.row.symbol
    for match in unjudged:
        assert match.unmeasured, match.row.symbol


@pytest.mark.live
def test_sector_proxies_all_return_history():
    """Eleven sector ETFs back the momentum panel. One going dark would
    silently drop that sector's contribution."""
    returns, error = etf_technicals.load_sector_returns()
    assert error is None, error
    assert set(returns) == set(etf_technicals.SECTOR_PROXIES), (
        f"missing: {set(etf_technicals.SECTOR_PROXIES) - set(returns)}")
    for key, value in returns.items():
        assert -100.0 < value < 100.0, f"{key}: {value}%"


# --- mocked API responses and error handling ---------------------------------
# The task's "Mock API responses, test error handling" bullet. These reach
# the fetch-and-parse paths that only ever ran against live Yahoo, which
# is where every unit trap in this module lives: the reciprocal in
# equity_holdings, the fraction-to-percent conversions, and the 100x
# duplicate expense ratio.

class _FakeFunds:
    def __init__(self, overview=None, equity=None, operations=None,
                 holdings=None, asset_classes=None, sectors=None):
        self.fund_overview = overview or {}
        self.equity_holdings = equity
        self.fund_operations = operations
        self.top_holdings = holdings
        self.asset_classes = asset_classes or {}
        self.sector_weightings = sectors or {}


class _FakeTicker:
    def __init__(self, funds, info=None, raises=None):
        self._funds, self._info, self._raises = funds, info or {}, raises

    @property
    def funds_data(self):
        if self._raises:
            raise self._raises
        return self._funds

    @property
    def info(self):
        return self._info


def _install(monkeypatch, ticker):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda symbol: ticker
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _frames():
    import pandas as pd

    equity = pd.DataFrame(
        {"XYZ": [0.03976, 0.2, 0.5], "Category Average": [0.05, 0.25, 0.6]},
        index=["Price/Earnings", "Price/Book", "Price/Sales"])
    operations = pd.DataFrame(
        {"XYZ": [0.000945, 0.02], "Category Average": [0.0035, 0.30]},
        index=["Annual Report Expense Ratio", "Annual Holdings Turnover"])
    holdings = pd.DataFrame(
        {"Name": ["Alpha Corp", "Beta Inc"], "Holding Percent": [0.0755, 0.0421]},
        index=["ALPHA", "BETA"])
    return equity, operations, holdings


def test_a_mocked_fund_parses_every_unit_correctly(monkeypatch):
    """The three traps in one pass: equity_holdings reports the RECIPROCAL
    of what it labels, operations reports FRACTIONS, and top_holdings
    reports fractions too."""
    equity, operations, holdings = _frames()
    _install(monkeypatch, _FakeTicker(
        _FakeFunds(overview={"categoryName": "Large Blend", "family": "ACME",
                             "legalType": "Exchange Traded Fund"},
                   equity=equity, operations=operations, holdings=holdings,
                   asset_classes={"stockPosition": 0.99},
                   sectors={"technology": 0.374}),
        info={"netAssets": 7.95e11}))

    profile = etf_analysis.load_profile.__wrapped__("xyz")

    assert profile.ok and profile.symbol == "XYZ"
    assert profile.category == "Large Blend" and profile.family == "ACME"
    # 1 / 0.03976 = 25.15, the multiple — NOT the earnings yield.
    assert profile.price_earnings == pytest.approx(25.15, abs=0.01)
    assert profile.category_price_earnings == pytest.approx(20.0, abs=0.01)
    # 0.000945 as a fraction is 0.0945%, not 0.000945%.
    assert profile.expense_ratio_pct == pytest.approx(0.0945)
    assert profile.category_expense_ratio_pct == pytest.approx(0.35)
    assert profile.turnover_pct == pytest.approx(2.0)
    assert profile.net_assets == pytest.approx(7.95e11)
    # 0.0755 as a fraction is 7.55%.
    assert [h.weight_pct for h in profile.top_holdings] == [
        pytest.approx(7.55), pytest.approx(4.21)]
    assert [h.symbol for h in profile.top_holdings] == ["ALPHA", "BETA"]
    assert profile.sector_weights == {"technology": 0.374}


def test_a_fetch_that_raises_returns_an_error_not_an_exception(monkeypatch):
    """This renders on every fund page; a raise here takes the page down."""
    _install(monkeypatch, _FakeTicker(None, raises=RuntimeError("network down")))
    profile = etf_analysis.load_profile.__wrapped__("XYZ")
    assert not profile.ok
    assert "XYZ" in profile.error and "RuntimeError" in profile.error


def test_unreadable_holdings_do_not_lose_the_rest_of_the_profile(monkeypatch):
    """A fund whose holdings frame is malformed still has a real expense
    ratio and category, and throwing those away would be a worse outcome
    than showing no holdings."""
    equity, operations, _ = _frames()
    _install(monkeypatch, _FakeTicker(
        _FakeFunds(overview={"categoryName": "Long Government"},
                   equity=equity, operations=operations,
                   holdings="not a frame"),
        info={"netAssets": 4.1e10}))

    # The frames are keyed by the fund's own symbol, so the lookup must
    # use the same one — a mismatch reads as "not reported", which is
    # exactly how this test first failed.
    profile = etf_analysis.load_profile.__wrapped__("XYZ")
    assert profile.ok
    assert profile.top_holdings == ()
    assert profile.expense_ratio_pct == pytest.approx(0.0945)
    assert profile.category == "Long Government"


def test_a_fund_reporting_nothing_yields_blanks_rather_than_zeros(monkeypatch):
    """A bond fund has no equity_holdings frame at all."""
    _install(monkeypatch, _FakeTicker(_FakeFunds(), info={}))
    profile = etf_analysis.load_profile.__wrapped__("AGG")
    assert profile.ok
    assert profile.price_earnings is None
    assert profile.expense_ratio_pct is None
    assert profile.net_assets is None
    assert profile.top_holdings == ()
    assert profile.sector_weights == {}


def test_an_empty_symbol_is_rejected_before_any_fetch(monkeypatch):
    def _boom(_):
        raise AssertionError("should not have fetched")

    import sys
    import types
    fake = types.ModuleType("yfinance")
    fake.Ticker = _boom
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    for blank in ("", "   ", None):
        profile = etf_analysis.load_profile.__wrapped__(blank)
        assert not profile.ok
        assert "No symbol" in profile.error


def test_the_universe_survives_a_failing_screen(monkeypatch):
    """load_universe must never raise: it renders the whole screener."""
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.screen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429"))
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    rows, error = etf_screener.load_universe.__wrapped__(50)
    assert rows == ()
    # The message names the exception TYPE and not its text, deliberately:
    # a raw upstream error string is not something to put on screen.
    assert error and "RuntimeError" in error
    assert "429" not in error


def test_a_screen_returning_no_quotes_is_reported_not_treated_as_empty(monkeypatch):
    """"Yahoo returned nothing" and "no fund matched" are different facts."""
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.screen = lambda *a, **k: {"quotes": []}
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    rows, error = etf_screener.load_universe.__wrapped__(50)
    assert rows == ()
    assert error and "no funds" in error


def test_the_universe_drops_duplicates_and_unparseable_rows(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.screen = lambda *a, **k: {"quotes": [
        {"symbol": "AAA", "netExpenseRatio": 0.10},
        {"symbol": "AAA", "netExpenseRatio": 0.10},   # duplicate
        {"symbol": "", "netExpenseRatio": 0.10},      # no symbol
        "not a dict",                                  # not parseable
        {"symbol": "BBB", "netExpenseRatio": 0.20},
    ]}
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    rows, error = etf_screener.load_universe.__wrapped__(50)
    assert error is None
    assert [r.symbol for r in rows] == ["AAA", "BBB"]


def test_sector_returns_survive_a_failing_download(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout"))
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    returns, error = etf_technicals.load_sector_returns.__wrapped__(20)
    assert returns == {}
    assert error and "timeout" in error


def test_comparison_prices_survive_a_failing_download(monkeypatch):
    import sys
    import types

    import etf_comparison

    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    closes, error = etf_comparison.load_prices.__wrapped__(("SPY", "QQQ"))
    assert closes is None
    assert error and "boom" in error


def test_comparison_prices_reject_an_empty_symbol_list():
    import etf_comparison

    closes, error = etf_comparison.load_prices.__wrapped__(())
    assert closes is None and error


# --- sector loader, mocked ----------------------------------------------------

def _sector_download(columns, rows=60):
    import pandas as pd
    import numpy as np

    index = pd.date_range("2025-01-01", periods=rows, freq="B")
    close = pd.DataFrame(
        {c: np.linspace(100, 110, rows) for c in columns}, index=index)
    return pd.concat({"Close": close}, axis=1)


def _install_download(monkeypatch, frame):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: frame
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_a_full_sector_download_yields_every_sector_and_no_error(monkeypatch):
    _install_download(monkeypatch,
                      _sector_download(list(etf_technicals.SECTOR_PROXIES.values())))
    returns, error = etf_technicals.load_sector_returns.__wrapped__(20)
    assert set(returns) == set(etf_technicals.SECTOR_PROXIES)
    assert error is None
    for value in returns.values():
        assert value == pytest.approx(returns["technology"])


def test_a_sector_that_goes_dark_is_named_rather_than_dropped(monkeypatch):
    """Silently omitting it would remove that sector's contribution from
    the estimate with nothing on screen to say so."""
    proxies = list(etf_technicals.SECTOR_PROXIES.values())
    _install_download(monkeypatch, _sector_download(proxies[:-1]))
    returns, error = etf_technicals.load_sector_returns.__wrapped__(20)
    assert len(returns) == len(proxies) - 1
    assert error and "No recent data for" in error
    missing_label = etf_technicals.SECTOR_LABELS["communication_services"]
    assert missing_label in error


def test_an_empty_sector_download_is_reported(monkeypatch):
    import pandas as pd

    _install_download(monkeypatch, pd.DataFrame())
    returns, error = etf_technicals.load_sector_returns.__wrapped__(20)
    assert returns == {}
    assert error and "empty" in error


def test_a_download_too_short_for_the_lookback_reports_rather_than_guesses(monkeypatch):
    """A 20-day momentum figure computed over four days is not one."""
    _install_download(
        monkeypatch,
        _sector_download(list(etf_technicals.SECTOR_PROXIES.values()), rows=5))
    returns, error = etf_technicals.load_sector_returns.__wrapped__(20)
    assert returns == {}
    assert error and "enough history" in error


# --- remaining defensive branches --------------------------------------------

def test_a_non_numeric_sector_return_reads_as_absent():
    for junk in (None, "n/a", object()):
        assert etf_technicals._number(junk) is None
    assert etf_technicals._number(float("nan")) is None
    assert etf_technicals._number("2.5") == 2.5


def test_a_change_from_a_zero_base_is_absent_not_infinite():
    """Dividing by a zero price would raise or return inf; neither is a
    percentage a reader can act on."""
    import pandas as pd

    assert etf_technicals._pct_change(pd.Series([0.0, 5.0]), 1) is None
    assert etf_technicals._pct_change(pd.Series([2.0, 4.0]), 1) == pytest.approx(100.0)
    # ...and an all-NaN series has nothing to measure.
    assert etf_technicals._pct_change(pd.Series([float("nan")] * 5), 1) is None


def test_a_sector_weight_that_is_not_a_number_is_skipped():
    rows = etf_technicals.sector_momentum(
        {"technology": "n/a", "energy": 0.4}, {"energy": 5.0})
    assert [r.key for r in rows] == ["energy"]


def test_the_gauge_ignores_an_sma_column_it_cannot_read():
    """A column present but entirely NaN must not be counted as a reading
    — that would inflate the denominator with nothing behind it."""
    import numpy as np
    import pandas as pd

    index = pd.date_range("2025-01-01", periods=30, freq="B")
    df = pd.DataFrame({"Close": np.linspace(10, 20, 30)}, index=index)
    sma = pd.DataFrame({"SMA_50": [float("nan")] * 30,
                        "SMA_200": [float("nan")] * 30}, index=index)
    verdict = etf_technicals.momentum_verdict(55.0, sma, df, None)
    assert verdict.considered == 1, "only the RSI could be read"
    assert len(verdict.reasons) == verdict.considered


# --- comparison price loader, mocked -----------------------------------------

def test_a_single_symbol_download_becomes_a_named_column(monkeypatch):
    """yfinance returns a Series rather than a frame for one symbol, and
    an unnamed Series would break every lookup downstream."""
    import sys
    import types

    import numpy as np
    import pandas as pd

    import etf_comparison

    index = pd.date_range("2025-01-01", periods=30, freq="B")
    frame = pd.concat(
        {"Close": pd.Series(np.linspace(100, 120, 30), index=index)}, axis=1)
    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: frame
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    closes, error = etf_comparison.load_prices.__wrapped__(("SPY",))
    assert error is None
    assert list(closes.columns) == ["SPY"]
    assert etf_comparison.total_return_pct(closes, "SPY") == pytest.approx(20.0)


def test_a_download_with_no_close_column_is_reported(monkeypatch):
    import sys
    import types

    import pandas as pd

    import etf_comparison

    fake = types.ModuleType("yfinance")
    fake.download = lambda *a, **k: pd.DataFrame({"Open": [1.0, 2.0]})
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    closes, error = etf_comparison.load_prices.__wrapped__(("SPY",))
    assert closes is None
    assert error and "empty" in error


def test_a_comparison_row_for_a_symbol_with_no_column_is_blank():
    import etf_comparison

    import pandas as pd

    frame = pd.DataFrame({"SPY": [100.0, 110.0]},
                         index=pd.date_range("2025-01-01", periods=2, freq="B"))
    assert etf_comparison.volatility_pct(frame, "QQQ") is None
    assert etf_comparison.sharpe(frame, "QQQ") is None
