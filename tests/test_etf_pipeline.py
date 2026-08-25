"""Fund identity, lifecycle and performance ingestion.

Three provider fields are replaced rather than passed through, and the
tests pin the REPLACEMENTS — a test written against `beta3Year` would
lock in a number that calls a long treasury fund twice as volatile as
the market in the same direction.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

import etf_pipeline as ep


def _series(values, start="2020-01-01"):
    index = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=index, dtype="float64")


# --- credentials --------------------------------------------------------------

def test_morningstar_is_unconfigured_by_default(monkeypatch):
    monkeypatch.delenv(ep.MORNINGSTAR_ENV_VAR, raising=False)
    assert ep.morningstar_api_key() is None
    assert not ep.morningstar_is_configured()


def test_a_key_in_the_environment_is_picked_up(monkeypatch):
    monkeypatch.setenv(ep.MORNINGSTAR_ENV_VAR, "  key123  ")
    assert ep.morningstar_api_key() == "key123"
    assert ep.morningstar_is_configured()


def test_the_unconfigured_note_says_what_a_key_would_add():
    """A connector that just says "not configured" gives no basis to
    decide whether the key is worth buying."""
    text = ep.MORNINGSTAR_UNCONFIGURED
    for promised in ("geographic", "custodian", "tracking error"):
        assert promised.lower() in text.lower(), promised
    assert ep.MORNINGSTAR_ENV_VAR in text


def test_what_is_absent_is_named_rather_than_approximated():
    assert "region, country or domicile" in ep.GEOGRAPHIC_ALLOCATION_UNAVAILABLE
    assert "weight" in ep.SHARE_COUNTS_UNAVAILABLE


# --- dividend frequency -------------------------------------------------------

def _dividends(gap_days, count=12):
    dates = [pd.Timestamp("2023-01-15") + pd.Timedelta(days=gap_days * i)
             for i in range(count)]
    return pd.Series([0.5] * count, index=pd.DatetimeIndex(dates))


@pytest.mark.parametrize("gap,expected", [
    (91, "Quarterly"), (30, "Monthly"), (182, "Semiannual"),
    (365, "Annual"), (7, "Weekly"),
])
def test_frequency_comes_from_the_median_gap(gap, expected):
    """Counting payments in a trailing year over-counts whenever a
    boundary payment falls inside the window — measured, that read
    quarterly funds as five a year and a monthly fund as thirteen."""
    label, per_year = ep.dividend_frequency(_dividends(gap))
    assert label == expected
    assert per_year == pytest.approx(365.0 / gap, rel=0.01)


def test_an_irregular_payer_is_still_classified_by_its_median():
    """One missed payment must not reclassify a quarterly fund."""
    dates = [pd.Timestamp("2023-01-15"), pd.Timestamp("2023-04-15"),
             pd.Timestamp("2023-07-15"), pd.Timestamp("2024-01-15"),
             pd.Timestamp("2024-04-15"), pd.Timestamp("2024-07-15")]
    series = pd.Series([0.5] * 6, index=pd.DatetimeIndex(dates))
    label, _ = ep.dividend_frequency(series)
    assert label == "Quarterly"


def test_a_fund_that_pays_nothing_has_no_frequency():
    """GLD pays no dividend. That is a fact about the fund, not a gap."""
    assert ep.dividend_frequency(None) == (None, None)
    assert ep.dividend_frequency(pd.Series([], dtype="float64")) == (None, None)
    # Two payments is not enough to establish a cadence.
    assert ep.dividend_frequency(_dividends(91, count=2)) == (None, None)


# --- performance windows ------------------------------------------------------

def test_every_window_the_task_asks_for_is_covered():
    assert [label for label, _ in ep.PERFORMANCE_WINDOWS] == [
        "1D", "1W", "1M", "3M", "1Y", "3Y", "5Y"]


def test_a_window_with_too_few_bars_is_unavailable_not_shortened():
    """A "5Y return" measured over two years is not one."""
    windows = {w.label: w for w in ep.performance_windows(
        _series(list(np.linspace(100, 120, 300))))}
    assert windows["1Y"].return_pct is not None
    assert windows["3Y"].return_pct is None
    assert windows["5Y"].return_pct is None


def test_short_windows_are_total_and_long_ones_are_annualised():
    """A raw five-year number beside a one-year number invites reading
    them as the same kind of thing."""
    windows = {w.label: w for w in ep.performance_windows(
        _series(list(np.linspace(100, 200, 1400))))}
    assert not windows["1Y"].annualised
    assert windows["3Y"].annualised and windows["5Y"].annualised
    # Annualising must shrink a multi-year gain below its total.
    assert windows["5Y"].return_pct < 100.0


def test_a_doubling_over_one_year_reads_as_a_hundred_percent():
    values = list(np.linspace(100, 200, 253))
    windows = {w.label: w for w in ep.performance_windows(_series(values))}
    assert windows["1Y"].return_pct == pytest.approx(100.0, rel=0.01)


def test_annualising_a_known_compound_return_is_exact():
    """Three years of exactly 10% a year annualises back to 10%."""
    days = 757
    values = [100 * (1.10 ** (i / 252.0)) for i in range(days)]
    windows = {w.label: w for w in ep.performance_windows(_series(values))}
    assert windows["3Y"].return_pct == pytest.approx(10.0, abs=0.1)


def test_windows_of_nothing_are_empty_rather_than_zero():
    assert ep.performance_windows(None) == ()
    empty = ep.performance_windows(pd.Series([], dtype="float64"))
    assert all(w.return_pct is None for w in empty)


def test_a_flat_series_returns_zero_not_none():
    """Zero IS the answer for a flat price — distinct from unavailable."""
    windows = {w.label: w for w in ep.performance_windows(_series([100.0] * 300))}
    assert windows["1M"].return_pct == pytest.approx(0.0)


# --- beta ---------------------------------------------------------------------

def test_beta_is_recovered_from_returns():
    rng = np.random.default_rng(0)
    market = rng.normal(0, 0.01, 500)
    for target in (0.2, 1.0, 2.0):
        fund = target * market + rng.normal(0, 0.0005, 500)
        bench_prices = _series(list(100 * np.cumprod(1 + market)))
        fund_prices = _series(list(100 * np.cumprod(1 + fund)))
        beta, r2, days = ep.regressed_beta(fund_prices, bench_prices)
        assert beta == pytest.approx(target, rel=0.05), target
        assert r2 > 0.9
        assert days > 400


def test_a_fund_uncorrelated_with_the_market_reports_a_low_r_squared():
    """The r-squared travels with the beta because a beta explaining
    nothing is a number to discount — for a bond fund it is near zero,
    and that IS the signal. Live: TLT beta 0.08 at r-squared 0.01."""
    rng = np.random.default_rng(1)
    market = _series(list(100 * np.cumprod(1 + rng.normal(0, 0.01, 400))))
    unrelated = _series(list(100 * np.cumprod(1 + rng.normal(0, 0.01, 400))))
    beta, r2, _ = ep.regressed_beta(unrelated, market)
    assert abs(beta) < 0.3
    assert r2 < 0.1


def test_too_few_days_yields_no_beta():
    short = _series(list(np.linspace(100, 110, 30)))
    beta, r2, _ = ep.regressed_beta(short, short)
    assert beta is None and r2 is None


def test_a_flat_benchmark_has_no_beta_rather_than_a_division_by_zero():
    flat = _series([100.0] * 200)
    moving = _series(list(np.linspace(100, 150, 200)))
    assert ep.regressed_beta(moving, flat)[0] is None


def test_missing_inputs_yield_no_beta():
    series = _series(list(np.linspace(100, 110, 200)))
    assert ep.regressed_beta(None, series)[0] is None
    assert ep.regressed_beta(series, None)[0] is None


def test_a_reported_beta_far_from_the_regressed_one_is_flagged():
    """TLT reports 2.40 against a regressed 0.13 — an eighteen-fold
    overstatement that would sell a long treasury fund as a leveraged
    equity bet."""
    identity = ep.FundIdentity(symbol="TLT", beta=0.13, reported_beta=2.40)
    assert identity.beta_disagrees_with_reported
    agreeing = ep.FundIdentity(symbol="QQQ", beta=1.27, reported_beta=1.26)
    assert not agreeing.beta_disagrees_with_reported
    # Nothing to disagree with when one side is absent.
    assert not ep.FundIdentity(symbol="X", beta=1.0).beta_disagrees_with_reported


# --- inception ----------------------------------------------------------------

def test_an_inception_after_the_first_price_bar_is_flagged():
    """THE VTI CASE. It reports 2016-06-27 while this source's own price
    history starts 2001-06-15 — a fifteen-year error from the same API.
    A fund cannot have traded before it existed."""
    identity = ep.FundIdentity(
        symbol="VTI",
        inception=datetime.date(2016, 6, 27),
        first_price_bar=datetime.date(2001, 6, 15),
        inception_is_suspect=True)
    text = ep.describe_inception(identity)
    assert "2016-06-27" in text and "2001-06-15" in text
    assert "wrong" in text


def test_pricing_a_few_days_after_launch_is_normal_and_not_flagged():
    """SPY reports 1993-01-22 and first prices 1993-01-29. A fund pricing
    shortly AFTER its stated inception is the ordinary case."""
    identity = ep.FundIdentity(
        symbol="SPY",
        inception=datetime.date(1993, 1, 22),
        first_price_bar=datetime.date(1993, 1, 29))
    assert not identity.inception_is_suspect
    assert "Launched 1993-01-22" in ep.describe_inception(identity)


def test_a_missing_inception_says_so():
    assert "not reported" in ep.describe_inception(ep.FundIdentity(symbol="X"))


def test_age_is_derived_from_inception():
    identity = ep.FundIdentity(
        symbol="X", inception=datetime.date.today() - datetime.timedelta(days=3653))
    assert identity.age_years == pytest.approx(10.0, abs=0.1)
    assert ep.FundIdentity(symbol="X").age_years is None


def test_an_epoch_is_read_in_utc():
    """A timestamp near midnight resolves to the previous day in a
    western timezone, which would silently move an inception by one."""
    # 1993-01-22T00:00:00Z — SPY's reported inception.
    assert ep._epoch_to_date(727660800) == datetime.date(1993, 1, 22)
    assert ep._epoch_to_date(None) is None
    assert ep._epoch_to_date("nonsense") is None


# --- the loader ---------------------------------------------------------------

def _install(monkeypatch, *, info, isin, closes, dividends=None):
    import sys
    import types

    class _Ticker:
        def __init__(self, symbol):
            self.info = info
            self.isin = isin
            self.dividends = dividends

    fake = types.ModuleType("yfinance")
    fake.Ticker = _Ticker
    fake.download = lambda *a, **k: pd.concat({"Close": closes}, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _closes(symbol="VTI", n=600, start="2001-06-15"):
    index = pd.date_range(start, periods=n, freq="B")
    values = np.linspace(100, 160, n)
    return pd.DataFrame({symbol: values, "SPY": np.linspace(100, 150, n)},
                        index=index)


def test_a_full_load_reports_identity_and_flags_the_bad_inception(monkeypatch):
    _install(monkeypatch,
             info={"longName": "Vanguard Total Stock Market ETF",
                   "fundInceptionDate": 1466985600,      # 2016-06-27
                   "beta3Year": 2.40},
             isin="US9229087690",
             closes=_closes(),
             dividends=_dividends(91))

    identity = ep.load_identity.__wrapped__("vti")
    assert identity.ok and identity.symbol == "VTI"
    assert identity.isin == "US9229087690"
    assert identity.inception == datetime.date(2016, 6, 27)
    assert identity.first_price_bar == datetime.date(2001, 6, 15)
    assert identity.inception_is_suspect
    assert identity.dividend_frequency == "Quarterly"
    assert identity.beta is not None
    assert identity.reported_beta == 2.40
    assert identity.performance


def test_a_missing_isin_is_absent_rather_than_a_dash(monkeypatch):
    """Some funds report "-" instead of nothing; QQQ and every European
    listing checked report no ISIN at all."""
    for raw in ("-", "", None, "None"):
        _install(monkeypatch, info={}, isin=raw, closes=_closes("QQQ"))
        assert ep.load_identity.__wrapped__("QQQ").isin is None


def test_an_isin_is_normalised(monkeypatch):
    _install(monkeypatch, info={}, isin="  us78462f1030  ", closes=_closes("SPY"))
    assert ep.load_identity.__wrapped__("SPY").isin == "US78462F1030"


def test_an_unreadable_isin_does_not_lose_the_rest(monkeypatch):
    import sys
    import types

    class _Ticker:
        info = {"fundInceptionDate": 727660800}
        dividends = None

        @property
        def isin(self):
            raise RuntimeError("no isin")

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda s: _Ticker()
    fake.download = lambda *a, **k: pd.concat({"Close": _closes("SPY")}, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    identity = ep.load_identity.__wrapped__("SPY")
    assert identity.ok
    assert identity.isin is None
    assert identity.inception == datetime.date(1993, 1, 22)


def test_a_raising_load_returns_an_error_not_an_exception(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda s: (_ for _ in ()).throw(RuntimeError("boom"))
    fake.download = lambda *a, **k: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    identity = ep.load_identity.__wrapped__("SPY")
    assert not identity.ok
    assert "SPY" in identity.error and "RuntimeError" in identity.error


def test_an_empty_symbol_is_rejected_before_any_fetch():
    for blank in ("", "   ", None):
        identity = ep.load_identity.__wrapped__(blank)
        assert not identity.ok and "No symbol" in identity.error


def test_the_full_history_is_requested_not_a_window():
    """Found by getting it wrong twice: a five-year window can never
    contradict an inception older than five years (VTI's bad 2016 date
    sailed through), and it supplies 1258 trading days where the 5Y
    window needs 1260, so SPY's five-year return read as unavailable."""
    import inspect

    signature = inspect.signature(ep.load_identity.__wrapped__)
    assert signature.parameters["lookback_period"].default == "max"


# --- live ---------------------------------------------------------------------

@pytest.mark.live
def test_real_funds_report_an_identity():
    for symbol in ("SPY", "TLT", "GLD"):
        identity = ep.load_identity(symbol)
        assert identity.ok, identity.error
        assert identity.inception is not None, symbol
        assert identity.first_price_bar is not None, symbol
        assert not identity.inception_is_suspect, symbol


@pytest.mark.live
def test_vti_still_reports_an_inception_its_own_prices_contradict():
    """The regression guard for the cross-check. If this ever passes
    cleanly the provider has fixed the field, and the check can go."""
    identity = ep.load_identity("VTI")
    assert identity.ok
    assert identity.inception_is_suspect, (
        "VTI's reported inception no longer contradicts its price history")


@pytest.mark.live
def test_a_bond_funds_measured_beta_contradicts_the_reported_one():
    """TLT: reported 2.40, regressed near zero. The reported field would
    sell a long treasury fund as a leveraged equity bet."""
    identity = ep.load_identity("TLT")
    assert identity.ok
    assert identity.beta is not None
    assert abs(identity.beta) < 0.6, identity.beta
    if identity.reported_beta is not None:
        assert identity.beta_disagrees_with_reported


@pytest.mark.live
def test_computed_performance_agrees_with_the_providers_own_fractions():
    """A cross-check on both the maths and the unit reading: SPY's
    computed 5Y annualised came to 13.01% against a reported
    fiveYearAverageReturn of 0.1307, and 3Y to 22.08% against 0.2185."""
    import yfinance as yf

    identity = ep.load_identity("SPY")
    windows = {w.label: w for w in identity.performance}
    info = yf.Ticker("SPY").info
    for label, key in (("3Y", "threeYearAverageReturn"),
                       ("5Y", "fiveYearAverageReturn")):
        reported = info.get(key)
        if reported is None or windows[label].return_pct is None:
            continue
        # Reported as a FRACTION; ours is a percent.
        assert windows[label].return_pct == pytest.approx(
            reported * 100.0, abs=2.0), label


@pytest.mark.live
def test_dividend_cadence_is_recognised_on_real_funds():
    assert ep.load_identity("SPY").dividend_frequency == "Quarterly"
    assert ep.load_identity("TLT").dividend_frequency == "Monthly"
    # GLD pays nothing at all.
    assert ep.load_identity("GLD").dividend_frequency is None
