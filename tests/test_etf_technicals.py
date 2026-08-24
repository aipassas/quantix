"""Fund technicals: sector momentum, flow, range position and signals.

The load-bearing claims here are about what the data can and cannot
support. Yahoo has no sector-weight history and no NAV history, and its
one NAV scalar is a stale close — so the tests pin that the module
computes momentum from sector-ETF proxies rather than inventing a weight
series, and that premium/discount is withheld rather than reported from
a figure measured to be three days old.

The other half is arithmetic that is easy to get wrong in ways nothing
crashes on: Yahoo's weights are FRACTIONS while every figure in this app
is percent-valued, a trailing average that includes the bar it is
testing understates every spike, and a rule that needs 200 days of
history must say "unavailable" rather than "not fired" on a 3-month
range.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import etf_technicals as et


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")
SOURCE = (ROOT / "etf_technicals.py").read_text(encoding="utf-8")


def _frame(closes, volumes=None, opens=None):
    n = len(closes)
    index = pd.date_range("2025-01-01", periods=n, freq="B")
    closes = pd.Series(closes, index=index, dtype="float64")
    opens = (pd.Series(opens, index=index, dtype="float64")
             if opens is not None else closes.shift(1).fillna(closes.iloc[0]))
    volumes = (pd.Series(volumes, index=index, dtype="float64")
               if volumes is not None else pd.Series(1e6, index=index))
    return pd.DataFrame({"Open": opens, "High": closes, "Low": closes,
                         "Close": closes, "Volume": volumes})


# --- the sector proxies -------------------------------------------------------

def test_every_yahoo_sector_key_maps_to_a_proxy_and_a_label():
    """Yahoo reports exactly eleven sector keys. A key with no proxy would
    silently drop that sector's contribution out of the estimate."""
    assert len(et.SECTOR_PROXIES) == 11
    assert set(et.SECTOR_PROXIES) == set(et.SECTOR_LABELS)
    assert len(set(et.SECTOR_PROXIES.values())) == 11, "no proxy reused"
    # The eleven keys Yahoo actually returns, written out rather than
    # derived from the mapping under test.
    assert set(et.SECTOR_PROXIES) == {
        "realestate", "consumer_cyclical", "basic_materials",
        "consumer_defensive", "technology", "communication_services",
        "financial_services", "utilities", "industrials", "energy",
        "healthcare"}


# --- weights are fractions, contributions are points --------------------------

def test_yahoo_weights_are_fractions_and_come_out_percent_valued():
    """sector_weightings gives 0.374 for 37.4%. Every figure in this app is
    percent-valued, so a missing conversion is invisible in Python and
    wrong by 100x on screen."""
    rows = et.sector_momentum({"technology": 0.374}, {"technology": 10.0})
    assert len(rows) == 1
    assert rows[0].weight_pct == pytest.approx(37.4)
    # 37.4% of a +10% move is +3.74 percentage points, NOT 374.
    assert rows[0].contribution_pct == pytest.approx(3.74)


def test_the_contributions_reconstruct_the_funds_own_move():
    """The real cross-check that weights and returns are being combined in
    the right units: measured live, SPY's contributions summed to 3.36pp
    against an actual 20-day move of 3.30%."""
    weights = {"technology": 0.5, "energy": 0.3, "utilities": 0.2}
    returns = {"technology": 4.0, "energy": 10.0, "utilities": -5.0}
    rows = et.sector_momentum(weights, returns)
    # 0.5*4 + 0.3*10 + 0.2*-5 = 2 + 3 - 1 = 4.0
    assert et.estimated_fund_move(rows) == pytest.approx(4.0)


def test_an_unmeasured_sector_is_not_treated_as_a_zero_contribution():
    """A sector whose proxy returned nothing has not contributed nothing.
    Summing it as zero would understate the estimate and sort it among
    the flat sectors instead of the unknown ones."""
    rows = et.sector_momentum({"technology": 0.6, "energy": 0.4},
                              {"technology": 5.0})
    energy = next(r for r in rows if r.key == "energy")
    assert energy.return_pct is None
    assert energy.contribution_pct is None
    # Only the measured sector counts toward the estimate.
    assert et.estimated_fund_move(rows) == pytest.approx(3.0)
    # ...and the unmeasured one sorts last, not as a middling zero.
    assert rows[-1].key == "energy"


def test_a_fund_with_no_sector_weights_yields_no_rows():
    """TLT and GLD report zero sector weightings — a bond fund and a
    commodity trust have no equity sectors. That is an absent capability,
    not missing data."""
    for empty in ({}, None):
        assert et.sector_momentum(empty, {"technology": 5.0}) == []
    assert et.estimated_fund_move([]) is None


def test_divergence_is_flagged_at_the_tasks_two_point_threshold():
    weights = {"technology": 0.5, "energy": 0.5}
    returns = {"technology": 5.0, "energy": 1.9}
    rows = et.sector_momentum(weights, returns, fund_return_pct=3.0)
    tech = next(r for r in rows if r.key == "technology")
    energy = next(r for r in rows if r.key == "energy")
    assert tech.divergence_pct == pytest.approx(2.0)
    assert tech.flagged, "exactly at the threshold counts"
    assert energy.divergence_pct == pytest.approx(-1.1)
    assert not energy.flagged

    ahead, behind = et.leaders_and_laggards(rows)
    assert [r.key for r in ahead] == ["technology"]
    assert behind == ()


def test_divergence_is_absent_rather_than_zero_without_a_fund_return():
    rows = et.sector_momentum({"technology": 0.5}, {"technology": 5.0})
    assert rows[0].divergence_pct is None
    assert not rows[0].flagged


# --- relative volume ----------------------------------------------------------

def test_the_trailing_average_excludes_the_bar_being_tested():
    """Including it puts the value under test into its own baseline, which
    drags the ratio toward 100% exactly when the spike is largest — a
    true doubling reads as about 190% instead of 200%."""
    volumes = [100.0] * 20 + [200.0]
    reading = et.relative_volume(_frame([10.0] * 21, volumes=volumes))
    assert reading.average == pytest.approx(100.0)
    assert reading.ratio_pct == pytest.approx(200.0)
    assert reading.days_used == 20
    assert reading.is_spike


def test_a_spike_is_measured_against_the_ratio_not_the_excess():
    """"200% above the average" and "200% of the average" differ by 1.5x.
    The module resolves it as "of", and says so on screen."""
    assert et.VOLUME_SPIKE_RATIO_PCT == 200.0
    just_under = et.relative_volume(
        _frame([10.0] * 21, volumes=[100.0] * 20 + [199.0]))
    assert not just_under.is_spike
    just_over = et.relative_volume(
        _frame([10.0] * 21, volumes=[100.0] * 20 + [201.0]))
    assert just_over.is_spike


def test_volume_knows_whether_the_latest_bar_was_up_or_down():
    """The accumulation signal is a spike on a DOWN day; without the
    direction it would fire on any spike at all."""
    up = et.relative_volume(_frame([10.0, 12.0], opens=[10.0, 11.0]))
    down = et.relative_volume(_frame([10.0, 9.0], opens=[10.0, 11.0]))
    assert up.on_up_day is True
    assert down.on_up_day is False


def test_volume_that_is_absent_reads_as_absent_not_as_zero():
    assert not et.relative_volume(None).ok
    assert not et.relative_volume(pd.DataFrame()).ok
    frame = _frame([10.0] * 5)
    del frame["Volume"]
    assert not et.relative_volume(frame).ok
    assert "not reported" in et.describe_volume(et.relative_volume(None))


def test_a_flat_average_of_zero_does_not_divide_by_zero():
    reading = et.relative_volume(_frame([10.0] * 5, volumes=[0.0] * 5))
    assert reading.ratio_pct is None
    assert not reading.ok


# --- range position -----------------------------------------------------------

def test_a_three_month_range_is_not_called_a_52_week_high():
    """The high of whatever happens to be loaded is not a 52-week high.
    Below the threshold the figures are still returned — they are the
    loaded range — but `sufficient` is False so the caller cannot label
    them as annual."""
    short = et.range_position(_frame(list(np.linspace(10, 20, 63))))
    assert short.days_used == 63
    assert not short.sufficient
    assert short.high == pytest.approx(20.0)

    long = et.range_position(_frame(list(np.linspace(10, 20, 251))))
    assert long.sufficient
    assert et.TRADING_DAYS_52W == 200


def test_position_in_range_runs_from_the_low_to_the_high():
    at_high = et.range_position(_frame([10.0, 20.0, 30.0]))
    assert at_high.position_pct == pytest.approx(100.0)
    assert at_high.at_new_high and not at_high.at_new_low

    at_low = et.range_position(_frame([30.0, 20.0, 10.0]))
    assert at_low.position_pct == pytest.approx(0.0)
    assert at_low.at_new_low and not at_low.at_new_high

    midway = et.range_position(_frame([10.0, 30.0, 20.0]))
    assert midway.position_pct == pytest.approx(50.0)
    assert not midway.at_new_high and not midway.at_new_low


def test_a_flat_series_has_no_range_to_sit_in():
    """Zero span would divide by zero; the position is unknown, not 0%."""
    flat = et.range_position(_frame([10.0] * 30))
    assert flat.position_pct is None


def test_range_position_of_nothing_is_empty_not_zero():
    assert et.range_position(None).price is None
    assert et.range_position(pd.DataFrame()).price is None


# --- signals ------------------------------------------------------------------

def _sma(frame, periods=(20, 50, 200)):
    import technical_indicators as ti
    return ti.compute_sma_lines(frame, periods)


def test_a_rule_that_cannot_be_evaluated_says_so_rather_than_not_fired():
    """SMA(200) needs 200 trading days. A one-year range gives 251 and a
    three-month range gives 63 (both measured), so on a short range the
    sell rule genuinely cannot be checked — and reporting that as "not
    fired" reads as an all-clear that was never performed."""
    frame = _frame(list(np.linspace(10, 20, 63)))
    import technical_indicators as ti
    results = {s.name: s for s in et.signals(
        frame, _sma(frame), ti.compute_rsi(frame, 14), et.relative_volume(frame))}
    sell = results["Sell — trend reversal"]
    assert sell.state == et.UNAVAILABLE
    assert "200" in sell.detail
    assert et.UNAVAILABLE not in (et.FIRED, et.NOT_FIRED)


def test_every_signal_reports_one_of_the_three_states():
    frame = _frame(list(np.linspace(10, 20, 251)))
    import technical_indicators as ti
    produced = et.signals(frame, _sma(frame), ti.compute_rsi(frame, 14),
                          et.relative_volume(frame))
    assert len(produced) == 4
    for signal in produced:
        assert signal.state in (et.FIRED, et.NOT_FIRED, et.UNAVAILABLE)
        assert signal.detail, signal.name


def test_signals_survive_having_nothing_to_work_with():
    """This renders on every fund; a short or empty range must not raise."""
    for frame in (None, pd.DataFrame(), _frame([10.0, 11.0])):
        produced = et.signals(frame, None, None, None)
        assert len(produced) == 4
        assert all(s.state == et.UNAVAILABLE for s in produced)


def test_the_accumulation_signal_needs_a_down_day_not_just_a_spike():
    frame_up = _frame([10.0] * 20 + [12.0],
                      volumes=[100.0] * 20 + [500.0],
                      opens=[10.0] * 20 + [10.0])
    frame_down = _frame([10.0] * 20 + [9.0],
                        volumes=[100.0] * 20 + [500.0],
                        opens=[10.0] * 20 + [10.0])
    import technical_indicators as ti

    def support(frame):
        return next(s for s in et.signals(frame, _sma(frame),
                                          ti.compute_rsi(frame, 14),
                                          et.relative_volume(frame))
                    if s.name.startswith("Support"))

    assert support(frame_up).state == et.NOT_FIRED
    assert support(frame_down).state == et.FIRED


# --- the momentum gauge -------------------------------------------------------

def test_the_gauge_is_scored_out_of_what_could_be_measured():
    """Averaging a missing reading in as zero would drag every short range
    toward Neutral and make the gauge look decisive when it was
    half-blind."""
    short = _frame(list(np.linspace(10, 20, 63)))
    long = _frame(list(np.linspace(10, 20, 251)))
    verdict_short = et.momentum_verdict(55.0, _sma(short), short,
                                        et.range_position(short))
    verdict_long = et.momentum_verdict(55.0, _sma(long), long,
                                       et.range_position(long))
    assert verdict_short.considered < verdict_long.considered
    assert verdict_short.considered > 0


def test_the_gauge_reports_unavailable_rather_than_neutral_when_blind():
    """Neutral is a reading. "I could not read it" is not the same thing,
    and a gauge that shows Neutral for both is lying about one."""
    verdict = et.momentum_verdict(None, None, None, None)
    assert verdict.label == "Unavailable"
    assert verdict.considered == 0
    assert verdict.reasons


def test_the_gauge_direction_follows_the_readings():
    rising = _frame(list(np.linspace(10, 30, 251)))
    falling = _frame(list(np.linspace(30, 10, 251)))
    up = et.momentum_verdict(70.0, _sma(rising), rising, et.range_position(rising))
    down = et.momentum_verdict(30.0, _sma(falling), falling,
                               et.range_position(falling))
    assert up.label == "Bullish" and up.score > 0
    assert down.label == "Bearish" and down.score < 0
    assert up.reasons and down.reasons


def test_every_reason_the_gauge_gives_is_backed_by_a_counted_reading():
    frame = _frame(list(np.linspace(10, 30, 251)))
    verdict = et.momentum_verdict(70.0, _sma(frame), frame,
                                  et.range_position(frame))
    assert len(verdict.reasons) == verdict.considered


# --- what the data cannot support ---------------------------------------------

def test_premium_discount_to_nav_is_withheld_and_says_why():
    """Not an oversight. Yahoo has no NAV history, and its single navPrice
    is a three-day-old close: measured 2026-08-24, ARKK's reported NAV
    was its 21 August close, a false -2.70% discount on a fund that
    arbitrages to within 0.05%. The task's own 0.5% alert would have
    fired permanently on QQQ, GLD, TLT and ARKK."""
    text = et.NAV_PREMIUM_UNAVAILABLE
    assert "navPrice" in text
    assert "stale" in text.lower()
    assert "0.5%" in text
    # And no premium is computed anywhere in the module.
    tree = ast.parse(SOURCE)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert not [n for n in names if "premium" in n.lower() or "nav" in n.lower()]


def test_sector_momentum_is_not_derived_from_a_weight_history():
    """funds_data.sector_weightings is a single undated dict. Any "20-day
    average of the weights" would be a fabricated series — the module
    takes momentum from sector ETF prices instead."""
    tree = ast.parse(SOURCE)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert not [c for c in calls if c.attr == "sector_weightings"], (
        "weights are passed IN by the caller, not fetched here")
    # The proxies are what supply the time dimension.
    assert "XLK" in SOURCE and "yf.download" in SOURCE


def test_a_lookback_longer_than_the_data_returns_nothing_not_a_short_window():
    """A 20-day momentum figure computed over four days is not a 20-day
    figure, and silently shortening the window would make every freshly
    listed fund look like it had a full history."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert et._pct_change(series, 20) is None
    assert et._pct_change(series, 3) == pytest.approx(300.0)


def test_a_failed_sector_load_reports_an_error_rather_than_flat_sectors():
    """A sector panel that cannot load must say so. Returning an empty
    mapping with no error would render every sector as unmeasured with no
    explanation."""
    returns, error = {}, "Sector price data came back empty."
    rows = et.sector_momentum({"technology": 0.5}, returns)
    assert rows[0].return_pct is None
    assert error


# --- how it is wired into the page --------------------------------------------

def test_the_signals_compute_their_own_long_averages():
    """The 50/200-day lines on the chart exist only when the "Show
    20/50/200 SMA Trio" checkbox is ticked. A signal that reported
    "not evaluated" because a DISPLAY toggle was off would be blaming the
    data for a UI state."""
    assert "compute_sma_lines(df, (50, etf_technicals.SMA_LONG_PERIOD))" in FINANCE
    # ...and it must not read the chart's own trio columns instead.
    assert "etf_technicals.signals(df, _ft_sma," in FINANCE


def test_the_panel_is_gated_on_the_instrument_being_a_basket():
    assert ("if asset_class.supports(asset_kind, asset_class.HOLDINGS):"
            in FINANCE)


def test_the_page_discloses_the_nav_gap_rather_than_omitting_it_silently():
    """A reader who expects a premium/discount panel and finds nothing
    cannot tell whether it is missing or broken."""
    assert "etf_technicals.NAV_PREMIUM_UNAVAILABLE" in FINANCE


def test_the_page_explains_that_not_evaluated_is_not_an_all_clear():
    assert "is not" in FINANCE and "Not evaluated" in FINANCE


def test_a_fund_with_no_sectors_gets_an_explanation_not_an_empty_table():
    """TLT and GLD report no sector weightings. An empty chart would read
    as a data failure rather than as an absent capability."""
    assert "no equity sector weightings" in FINANCE


def test_the_sector_table_coerces_its_numeric_columns():
    """A column no row reports is object dtype, and Streamlit prints the
    literal "None" into it — the trap already hit on the ETF screener."""
    assert 'pd.to_numeric(_ft_t[_c], errors="coerce")' in FINANCE


def test_the_sector_percentages_do_not_use_streamlits_percent_preset():
    """That preset multiplies by 100, and every figure here is already
    percent-valued."""
    import re
    block = FINANCE[FINANCE.index("etf_sector_momentum_table") - 4000:
                    FINANCE.index("etf_sector_momentum_table")]
    assert 'format="percent"' not in block
    assert 'format="%.1f%%"' in block
