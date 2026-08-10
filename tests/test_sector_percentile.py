"""Tests for sector_percentile.py — the sector-relative percentile engine.

Percentile math and the peer-count guard are tested against mocked peer
data (no network calls). The "a known outlier lands in a high percentile
against its actual peers" acceptance criterion is a live test against real
data, same convention as the rest of this suite (see test_live_sanity.py).
"""
import pytest

import sector_percentile as sp
from financial_standardization import StandardizedFinancials


def _std(ticker="TEST", sector="Technology", **overrides) -> StandardizedFinancials:
    """A minimal StandardizedFinancials with every field defaulted to None/0
    except what a test explicitly overrides — only the fields
    sector_percentile.py actually reads need real values."""
    fields = dict(
        ticker=ticker, long_name=None, business_summary=None, website=None, sector=sector,
        pe_ratio=None, peg_ratio=None, price_to_book=None, net_margin=None, return_on_equity=None,
        debt_to_equity=None, current_ratio=None, beta=None, earnings_growth=None,
        held_pct_insiders=None, held_pct_institutions=None, market_cap=None, shares_outstanding=None,
        current_price=None, total_assets=None, current_assets=None, current_liabilities=None,
        stockholders_equity=None, total_liabilities=None, total_debt=0.0, total_debt_from_statement=None,
        retained_earnings=0.0, inventory=0.0, cash_and_equivalents=0.0, total_revenue=None, ebit=None,
        interest_expense=None, net_income=None, gross_profit=None, operating_income=None,
        pretax_income=None, tax_provision=None, free_cash_flow=None, depreciation_and_amortization=0.0,
        most_recent_quarter=None, validation=None, data_fallbacks=(),
    )
    fields.update(overrides)
    return StandardizedFinancials(**fields)


def test_format_percentile_ordinal_suffixes():
    assert sp.format_percentile(1) == "1st percentile"
    assert sp.format_percentile(2) == "2nd percentile"
    assert sp.format_percentile(3) == "3rd percentile"
    assert sp.format_percentile(4) == "4th percentile"
    assert sp.format_percentile(11) == "11th percentile"
    assert sp.format_percentile(12) == "12th percentile"
    assert sp.format_percentile(13) == "13th percentile"
    assert sp.format_percentile(21) == "21st percentile"
    assert sp.format_percentile(85) == "85th percentile"
    assert sp.format_percentile(None) == "N/A"


def test_candidate_universe_excludes_target_and_is_deduplicated():
    universe = sp._candidate_universe("AAPL")
    assert "AAPL" not in universe
    assert len(universe) == len(set(universe))
    assert "MSFT" in universe  # from WATCHLIST.tech_basket


def test_returns_none_when_sector_unknown():
    target = _std(sector=None)
    assert sp.compute_sector_percentiles(target) is None


def test_returns_none_when_fewer_than_min_peers(monkeypatch):
    fake_peers = {
        "PEER1": {"sector": "Technology", "net_margin": 0.20, "roe": None, "debt_to_equity": None,
                   "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None},
        "PEER2": {"sector": "Technology", "net_margin": 0.25, "roe": None, "debt_to_equity": None,
                   "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None},
        "PEER3": {"sector": "Healthcare", "net_margin": 0.30, "roe": None, "debt_to_equity": None,
                   "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None},
    }
    monkeypatch.setattr(sp, "_fetch_peer_sector_metrics", lambda candidates: fake_peers)
    target = _std(sector="Technology", net_margin=0.22)
    assert sp.MIN_PEERS == 3  # only 2 Technology peers above — this test assumes that
    assert sp.compute_sector_percentiles(target) is None


def test_computes_correct_percentile_with_enough_peers(monkeypatch):
    fake_peers = {
        f"PEER{i}": {"sector": "Technology", "net_margin": margin, "roe": None, "debt_to_equity": None,
                      "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None}
        for i, margin in enumerate([0.05, 0.10, 0.15, 0.20, 0.25])
    }
    monkeypatch.setattr(sp, "_fetch_peer_sector_metrics", lambda candidates: fake_peers)
    # Net margin of 0.30 beats all 5 peers -> 100th percentile (kind="mean": (5 strict + 5 weak)/2/5*100 = 100)
    target = _std(sector="Technology", net_margin=0.30)
    result = sp.compute_sector_percentiles(target)
    assert result is not None
    assert result.peer_count == 5
    assert result.percentiles["net_margin"] == pytest.approx(100.0)

    # Net margin tied with the middle peer (0.15) -> 50th percentile under kind="mean"
    target_tied = _std(sector="Technology", net_margin=0.15)
    result_tied = sp.compute_sector_percentiles(target_tied)
    assert result_tied.percentiles["net_margin"] == pytest.approx(50.0)


def test_metric_specific_none_does_not_block_other_metrics(monkeypatch):
    """A metric missing for enough peers should come back None for THAT
    metric only, without preventing percentiles for metrics that do have
    enough peer data — the sector-level MIN_PEERS guard and the
    per-metric one are independent."""
    fake_peers = {
        "PEER1": {"sector": "Technology", "net_margin": 0.10, "roe": None, "debt_to_equity": 1.0,
                   "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None},
        "PEER2": {"sector": "Technology", "net_margin": 0.20, "roe": None, "debt_to_equity": None,
                   "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None},
        "PEER3": {"sector": "Technology", "net_margin": 0.30, "roe": None, "debt_to_equity": None,
                   "pe_ratio": None, "peg_ratio": None, "beta": None, "current_ratio": None, "price_to_book": None},
    }
    monkeypatch.setattr(sp, "_fetch_peer_sector_metrics", lambda candidates: fake_peers)
    target = _std(sector="Technology", net_margin=0.25, debt_to_equity=0.5)
    result = sp.compute_sector_percentiles(target)
    assert result is not None
    assert result.percentiles["net_margin"] is not None       # 3 peers have net_margin -> enough
    assert result.percentiles["debt_to_equity"] is None        # only 1 peer has debt_to_equity -> not enough


@pytest.mark.live
def test_known_high_margin_outlier_lands_in_high_percentile_real_data():
    """Acceptance criterion: a company with unusually high margins for its
    sector lands in a high percentile against its actual peers. NVDA's net
    margin is a genuine, verified standout in this app's configured
    Technology peer universe — confirmed by inspecting the actual peer
    values directly (NVDA ~63% vs. the next-highest peer, TSM, at ~50%;
    AAPL was tried first and rejected as the example here specifically
    because real data showed it does NOT lead this particular peer set —
    MSFT/AVGO/ASML/NVDA/TSM all report higher margins, a good reminder that
    "known outlier" claims need checking against real data, not assumed)."""
    import datetime
    from data_loader import load_ticker_bundle
    from financial_standardization import standardize_financials

    end = datetime.date.today()
    start = end - datetime.timedelta(days=30)
    bundle = load_ticker_bundle("NVDA", start, end, deep=False)
    std = standardize_financials(bundle)
    assert std.sector, "NVDA should have a reported sector"

    result = sp.compute_sector_percentiles(std)
    assert result is not None, f"expected enough {std.sector} peers in the configured universe for this test to be meaningful"
    assert result.peer_count >= sp.MIN_PEERS
    net_margin_pct = result.percentiles["net_margin"]
    assert net_margin_pct is not None
    assert net_margin_pct >= 90.0, f"NVDA's net margin should rank at the top of its {result.peer_count} {std.sector} peers, got {net_margin_pct}"
