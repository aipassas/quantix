"""Fund risk: tracking, capture, concentration, liquidity, stress.

The load-bearing test is the DISCRIMINATOR: a tracking-error
implementation that cannot separate an S&P index fund from a bond fund
is not measuring tracking. Measured over three years against ^GSPC —
VOO 0.78%, IVV 0.94%, SPY 1.15% against QQQ 7.79%, TLT 18.74%,
ARKK 29.35%.
"""
import numpy as np
import pandas as pd
import pytest

import etf_risk as er


def _prices(returns, start="2021-01-01"):
    index = pd.date_range(start, periods=len(returns) + 1, freq="B")
    values = 100 * np.cumprod(np.concatenate([[1.0], 1 + np.asarray(returns)]))
    return pd.Series(values, index=index)


def _pair(n=400, noise=0.0, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n)
    fund = beta * market + rng.normal(0, noise, n)
    return _prices(fund), _prices(market)


# --- tracking error -----------------------------------------------------------

def test_a_fund_that_replicates_its_benchmark_has_almost_no_tracking_error():
    fund, bench = _pair(noise=0.0)
    result = er.tracking(fund, bench, "^GSPC")
    assert result.ok
    assert result.tracking_error_pct < 0.01
    assert result.band == "Tight"


def test_tracking_error_rises_with_the_noise_between_the_two():
    """The whole point of the statistic."""
    tight = er.tracking(*_pair(noise=0.0005), benchmark="B").tracking_error_pct
    loose = er.tracking(*_pair(noise=0.005), benchmark="B").tracking_error_pct
    assert loose > tight * 5


def test_the_bands_separate_an_index_fund_from_something_else():
    """Measured live: VOO 0.78% / SPY 1.15% are Tight; QQQ 7.79% is
    Loose; TLT 18.74% and ARKK 29.35% are not tracking ^GSPC at all."""
    assert er.TIGHT_TRACKING_PCT == 2.0
    assert er.LOOSE_TRACKING_PCT == 10.0
    for value, expected in ((0.78, "Tight"), (1.15, "Tight"),
                            (7.79, "Loose"), (18.74, "Not tracking this benchmark"),
                            (29.35, "Not tracking this benchmark")):
        result = er.TrackingResult(benchmark="^GSPC", tracking_error_pct=value)
        assert result.band == expected, value
    assert er.TrackingResult(benchmark="X").band == "Unknown"


def test_tracking_error_is_annualised():
    """A daily standard deviation quoted as an annual figure is out by
    sqrt(252) — a factor of nearly sixteen."""
    fund, bench = _pair(noise=0.01, seed=3)
    result = er.tracking(fund, bench, "B")
    daily = 0.01
    assert result.tracking_error_pct == pytest.approx(
        daily * np.sqrt(252) * 100, rel=0.15)


def test_the_information_ratio_puts_both_halves_on_the_same_clock():
    """The task divides excess return by 252 while tracking error is
    annualised — a daily numerator over an annual denominator, the same
    two-clocks mistake the bond reference made. That is a factor of
    252 x sqrt(252), about 4000x, and a magnitude-only assertion did not
    catch it: this pins the formula itself."""
    fund, bench = _pair(n=400, noise=0.002, seed=5)
    result = er.tracking(fund, bench, "B")

    # Recomputed independently from the same two series.
    fund_returns = fund.pct_change().dropna()
    bench_returns = bench.pct_change().dropna()
    excess = (fund_returns - bench_returns).dropna()
    expected = (excess.mean() * 252) / (excess.std() * np.sqrt(252))

    assert result.information_ratio == pytest.approx(expected, rel=1e-6)
    # And the wrong clock would be about 4000x smaller.
    wrong = (excess.mean() / 252) / (excess.std() * np.sqrt(252))
    assert abs(result.information_ratio - wrong) > abs(wrong) * 100


def test_the_cumulative_gap_is_reported_beside_the_daily_figure():
    """They answer different questions: the daily number is dominated by
    closing-price mismatch, the cumulative one is the fee and real drag."""
    fund, bench = _pair(noise=0.0)
    result = er.tracking(fund, bench, "^GSPC")
    assert result.fund_total_return_pct is not None
    assert result.benchmark_total_return_pct is not None
    assert result.cumulative_gap_pct == pytest.approx(
        result.fund_total_return_pct - result.benchmark_total_return_pct,
        abs=1e-9)
    assert result.annualised_gap_pct is not None


def test_a_fund_that_lags_shows_a_negative_gap():
    rng = np.random.default_rng(7)
    market = rng.normal(0.0005, 0.01, 500)
    fund = market - 0.0002          # a steady daily drag
    result = er.tracking(_prices(fund), _prices(market), "B")
    assert result.cumulative_gap_pct < 0
    assert result.annualised_gap_pct < 0


def test_too_little_overlap_yields_no_tracking():
    short = _prices(np.zeros(10))
    assert not er.tracking(short, short, "B").ok
    assert not er.tracking(None, short, "B").ok
    assert not er.tracking(short, None, "B").ok


def test_returns_are_aligned_on_the_dates_both_series_have():
    """A venue with different holidays would otherwise contribute a
    two-day fund return against a one-day benchmark return, which reads
    as tracking error but is a calendar mismatch."""
    fund, bench = _pair(n=300, noise=0.0)
    holed = fund.drop(fund.index[50:60])
    result = er.tracking(holed, bench, "B")
    assert result.ok
    assert result.days < 300
    # Still essentially perfect tracking — the hole did not manufacture
    # error.
    assert result.tracking_error_pct < 0.05


def test_the_benchmark_is_named_and_declared_the_readers_choice():
    """A tracking error against a benchmark the fund never claimed to
    track says nothing about the fund."""
    result = er.tracking(*_pair(), benchmark="^GSPC")
    assert result.benchmark == "^GSPC"
    assert "sidebar" in er.BENCHMARK_IS_YOUR_CHOICE
    assert "prospectus" in er.BENCHMARK_IS_YOUR_CHOICE


def test_a_price_index_flatters_a_fund_and_the_note_says_so():
    """SPY shows +6.44% against ^GSPC but -0.61% against ^SP500TR. The
    difference is dividends, not skill."""
    assert "dividends" in er.PRICE_INDEX_FLATTERS_A_FUND
    assert "^SP500TR" in er.PRICE_INDEX_FLATTERS_A_FUND


# --- capture ------------------------------------------------------------------

def test_a_replicating_fund_captures_all_of_both_directions():
    fund, bench = _pair(noise=0.0)
    ratios = er.capture_ratios(fund, bench)
    assert ratios.up_pct == pytest.approx(100.0, abs=1.0)
    assert ratios.down_pct == pytest.approx(100.0, abs=1.0)
    assert ratios.asymmetry == pytest.approx(0.0, abs=2.0)


def test_a_leveraged_fund_captures_more_of_both():
    """Measured live: ARKK 215% up, 228% down — it amplifies the fall
    slightly more than the rise, which is the shape that erodes capital."""
    fund, bench = _pair(beta=2.0, noise=0.0)
    ratios = er.capture_ratios(fund, bench)
    assert ratios.up_pct == pytest.approx(200.0, abs=2.0)
    assert ratios.down_pct == pytest.approx(200.0, abs=2.0)


def test_a_defensive_fund_captures_less_of_the_fall():
    """The asymmetry is the point: catching more of the rise than of the
    fall is the only free lunch in the table."""
    rng = np.random.default_rng(11)
    market = rng.normal(0, 0.01, 600)
    fund = np.where(market > 0, market * 1.0, market * 0.5)
    ratios = er.capture_ratios(_prices(fund), _prices(market))
    assert ratios.up_pct == pytest.approx(100.0, abs=2.0)
    assert ratios.down_pct == pytest.approx(50.0, abs=3.0)
    assert ratios.asymmetry > 40


def test_capture_is_measured_on_the_benchmarks_direction():
    """THE DISCRIMINATOR for this statistic, and a weaker version of this
    test let a poisoned build pass. An INVERSE fund falls when the
    benchmark rises, so its up-capture must be about MINUS 100%. Measured
    on the fund's own direction instead it would read +100% — the right
    magnitude with the wrong sign, which is the reading that matters."""
    rng = np.random.default_rng(21)
    market = rng.normal(0, 0.01, 600)
    inverse = -market
    ratios = er.capture_ratios(_prices(inverse), _prices(market))
    assert ratios.up_pct == pytest.approx(-100.0, abs=2.0)
    assert ratios.down_pct == pytest.approx(-100.0, abs=2.0)
    # The day counts come from the BENCHMARK, so they must match the
    # benchmark's own up/down split rather than the fund's.
    assert ratios.up_days == int((market > 0).sum())
    assert ratios.down_days == int((market < 0).sum())


def test_capture_of_nothing_is_absent():
    assert not er.capture_ratios(None, None).ok
    assert er.capture_ratios(None, None).asymmetry is None


# --- concentration ------------------------------------------------------------

class _Holding:
    def __init__(self, symbol, weight_pct):
        self.symbol, self.weight_pct = symbol, weight_pct


def test_the_herfindahl_index_is_computed_on_fractions():
    """Squaring percents gives a number a hundred times too large and
    breaks the 0-1 scale the index is defined on."""
    result = er.concentration([_Holding("A", 100.0)])
    assert result.herfindahl == pytest.approx(1.0)
    assert result.effective_holdings == pytest.approx(1.0)


def test_equal_weights_give_an_effective_count_equal_to_the_holdings():
    """Ten equal 10% positions is an effective ten."""
    holdings = [_Holding(f"H{i}", 10.0) for i in range(10)]
    result = er.concentration(holdings)
    assert result.herfindahl == pytest.approx(0.10)
    assert result.effective_holdings == pytest.approx(10.0)
    assert result.top_ten_pct == pytest.approx(100.0)


def test_concentration_rises_as_weight_gathers_in_one_name():
    spread = er.concentration([_Holding(f"H{i}", 10.0) for i in range(10)])
    lumpy = er.concentration([_Holding("BIG", 60.0)]
                             + [_Holding(f"H{i}", 40.0 / 9) for i in range(9)])
    assert lumpy.herfindahl > spread.herfindahl
    assert lumpy.effective_holdings < spread.effective_holdings
    assert lumpy.max_holding_symbol == "BIG"
    assert lumpy.max_holding_pct == pytest.approx(60.0)


def test_the_top_ten_caveat_is_stated():
    """Live: SPY's HHI over its top ten is 0.0186, an effective 53.8
    holdings. That is a lower bound on the real concentration, not the
    whole fund."""
    assert "lower bound" in er.TOP_TEN_CONCENTRATION_NOTE
    assert "37-46%" in er.TOP_TEN_CONCENTRATION_NOTE


def test_a_fund_with_no_disclosed_holdings_has_no_concentration():
    """Normal for a bond or commodity fund — not a zero."""
    assert not er.concentration([]).ok
    assert er.concentration(None).herfindahl is None
    assert not er.concentration([_Holding("A", 0.0)]).ok


def test_nonsense_weights_are_skipped_rather_than_crashing():
    result = er.concentration([_Holding("A", 50.0), _Holding("B", None),
                               _Holding("C", float("nan")), _Holding("D", -5.0)])
    assert result.disclosed_count == 1
    assert result.max_holding_symbol == "A"


# --- liquidity ----------------------------------------------------------------

def test_the_spread_is_quoted_in_basis_points():
    """A trading cost in percent gets misread by a hundred."""
    result = er.liquidity(price=100.0, volume=1e6, assets=1e10,
                          bid=99.95, ask=100.05)
    assert result.spread_bps == pytest.approx(10.0, abs=0.1)
    assert "bp" in result.detail


def test_a_missing_spread_falls_back_to_dollar_volume():
    """Measured: bid and ask are both 0.00 for six of ten funds,
    including SPY."""
    result = er.liquidity(price=600.0, volume=5e7, assets=8e11,
                          bid=0.0, ask=0.0)
    assert result.spread_bps is None
    assert result.dollar_volume == pytest.approx(3e10)
    assert result.turnover_pct == pytest.approx(3.75, rel=0.01)
    assert "volume" in result.detail
    assert "0.00" in er.BID_ASK_MOSTLY_ABSENT


def test_liquidity_with_nothing_reported_says_so():
    result = er.liquidity(price=None, volume=None, assets=None)
    assert not result.ok
    assert "Neither" in result.detail


def test_an_inverted_quote_is_not_treated_as_a_spread():
    """An ask below the bid is bad data, not a negative trading cost."""
    result = er.liquidity(price=100.0, volume=1e6, assets=1e9,
                          bid=100.5, ask=99.5)
    assert result.spread_bps is None


# --- extremes -----------------------------------------------------------------

def test_the_worst_day_and_month_are_found():
    returns = [0.01] * 40
    returns[20] = -0.07
    result = er.historical_extremes(_prices(returns))
    assert result.worst_day_pct == pytest.approx(-7.0, abs=0.01)
    assert result.worst_day_date is not None
    assert result.days == 40


def test_extremes_of_a_rising_series_are_still_reported():
    """The worst day of a fund that only rose is its smallest gain, not
    None — the statistic is the minimum, and it exists."""
    result = er.historical_extremes(_prices([0.01] * 60))
    assert result.worst_day_pct is not None
    assert result.worst_day_pct > 0


def test_extremes_of_nothing_are_absent():
    assert er.historical_extremes(None).worst_day_pct is None
    assert er.historical_extremes(pd.Series([], dtype="float64")).days == 0


# --- sector stress ------------------------------------------------------------

def test_a_sector_shock_scales_by_its_weight():
    """Tech at 37.4% falling 30% costs the fund 11.22%."""
    rows = er.sector_stress({"technology": 0.374}, -30.0)
    assert len(rows) == 1
    assert rows[0].weight_pct == pytest.approx(37.4)
    assert rows[0].fund_impact_pct == pytest.approx(-11.22, abs=0.01)


def test_shocking_every_sector_moves_the_fund_by_the_shock():
    """The internal consistency check: weights summing to one means a
    uniform shock must move the whole fund by exactly that shock.
    Verified live on SPY at -30.00%."""
    weights = {"technology": 0.4, "financial_services": 0.35, "energy": 0.25}
    rows = er.sector_stress(weights, -30.0)
    assert er.total_shock_impact(rows) == pytest.approx(-30.0, abs=1e-9)


def test_the_biggest_loss_sorts_first():
    """The sector that actually matters should lead, not the alphabet."""
    rows = er.sector_stress({"energy": 0.05, "technology": 0.40}, -30.0)
    assert rows[0].sector.lower().startswith("tech")


def test_a_positive_shock_is_supported_too():
    rows = er.sector_stress({"technology": 0.40}, 20.0)
    assert rows[0].fund_impact_pct == pytest.approx(8.0)


def test_a_fund_with_no_sector_weights_has_no_stress_rows():
    """A bond fund reports none — an absent capability, not a zero."""
    assert er.sector_stress({}, -30.0) == []
    assert er.sector_stress(None, -30.0) == []
    assert er.total_shock_impact([]) is None


def test_weights_arrive_as_fractions():
    """0.374 means 37.4%, the convention this data source uses
    everywhere and etf_technicals already documents."""
    rows = er.sector_stress({"technology": 0.374}, -10.0)
    assert rows[0].weight_pct == pytest.approx(37.4)
    assert rows[0].fund_impact_pct == pytest.approx(-3.74, abs=0.01)


def test_sector_labels_are_used_when_supplied():
    rows = er.sector_stress({"financial_services": 0.2}, -10.0,
                            {"financial_services": "Financials"})
    assert rows[0].sector == "Financials"
    # ...and a key with no label is still readable.
    bare = er.sector_stress({"consumer_cyclical": 0.2}, -10.0)
    assert bare[0].sector == "Consumer Cyclical"


# --- live ---------------------------------------------------------------------

@pytest.mark.live
def test_index_funds_track_and_other_funds_do_not():
    """THE DISCRIMINATOR. An implementation that cannot separate these
    two groups is not measuring tracking."""
    import yfinance as yf

    px = yf.download(["SPY", "VOO", "QQQ", "TLT", "^GSPC"], period="3y",
                     progress=False, auto_adjust=True)["Close"].dropna()
    for symbol in ("SPY", "VOO"):
        result = er.tracking(px[symbol], px["^GSPC"], "^GSPC")
        assert result.ok, symbol
        assert result.tracking_error_pct < er.TIGHT_TRACKING_PCT, (
            f"{symbol} {result.tracking_error_pct}")
    for symbol in ("QQQ", "TLT"):
        result = er.tracking(px[symbol], px["^GSPC"], "^GSPC")
        assert result.tracking_error_pct > er.TIGHT_TRACKING_PCT, symbol
    # And a bond fund is not tracking an equity index at all.
    assert er.tracking(px["TLT"], px["^GSPC"], "^GSPC").band.startswith("Not")


@pytest.mark.live
def test_real_capture_ratios_rank_the_way_the_funds_behave():
    import yfinance as yf

    px = yf.download(["SPY", "QQQ", "TLT", "^GSPC"], period="3y",
                     progress=False, auto_adjust=True)["Close"].dropna()
    spy = er.capture_ratios(px["SPY"], px["^GSPC"])
    qqq = er.capture_ratios(px["QQQ"], px["^GSPC"])
    tlt = er.capture_ratios(px["TLT"], px["^GSPC"])
    assert spy.up_pct == pytest.approx(100.0, abs=5.0)
    assert qqq.up_pct > spy.up_pct
    assert tlt.up_pct < 50.0


@pytest.mark.live
def test_a_real_funds_concentration_is_ordered_the_way_it_should_be():
    import etf_analysis

    spy = er.concentration(etf_analysis.load_profile("SPY").top_holdings)
    qqq = er.concentration(etf_analysis.load_profile("QQQ").top_holdings)
    assert spy.ok and qqq.ok
    # QQQ holds a hundred names against SPY's five hundred, so it must be
    # the more concentrated of the two.
    assert qqq.herfindahl > spy.herfindahl
    assert 0 < spy.herfindahl < 1
    assert spy.effective_holdings > qqq.effective_holdings


@pytest.mark.live
def test_a_uniform_sector_shock_reconciles_on_a_real_fund():
    import etf_analysis

    profile = etf_analysis.load_profile("SPY")
    rows = er.sector_stress(profile.sector_weights, -30.0)
    assert rows
    total = er.total_shock_impact(rows)
    # Weights sum to about one, so the whole-fund move is about the shock.
    assert total == pytest.approx(-30.0, abs=1.5)
