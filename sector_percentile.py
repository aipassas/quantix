"""Sector-relative percentile ranking — where a ticker's fundamental
metrics rank against same-sector peers, shown ALONGSIDE (never instead of)
the existing fixed-threshold Scorecard/Master Matrix/Quality Classification
checks in fundamental_analysis.py. Those answer "is this healthy in
absolute terms"; this answers "how does it compare to its actual peers."

The peer universe is deliberately not a new hardcoded sector-mapping table
— it's every ticker already configured elsewhere in the app (the
Institutional Watchlist baskets and the Peer Comparison defaults),
regrouped by each ticker's own Yahoo-reported sector field. Every peer
fetch is shallow (deep=False, info-only), matching the same efficiency
principle the Stock Screener uses — this module needs quote-level ratios,
never full financial statements, for any peer.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
from scipy.stats import percentileofscore

from config import PEER_DEFAULTS, WATCHLIST
from data_loader import load_ticker_bundle
from financial_standardization import StandardizedFinancials, standardize_financials
from logging_setup import get_logger, log_event

logger = get_logger("sector_percentile")

MIN_PEERS = 3  # fewer same-sector peers than this and a percentile is noise, not signal — see compute_sector_percentiles()

# Every metric here is sourced directly from Yahoo's `.info` in
# financial_standardization.py, so it's available on a shallow (deep=False)
# bundle for every peer. ROIC, Interest Coverage, and FCF Yield are
# deliberately NOT offered — they need full financial statements, and
# fetching those for a whole peer universe would defeat the "avoid a full
# deep fetch per peer" requirement this module was built to respect.
SUPPORTED_METRICS: Dict[str, str] = {
    "net_margin": "Net Margin",
    "roe": "Return on Equity",
    "debt_to_equity": "Debt-to-Equity",
    "pe_ratio": "P/E Ratio",
    "peg_ratio": "PEG Ratio",
    "beta": "Beta",
    "current_ratio": "Current Ratio",
    "price_to_book": "Price-to-Book",
}

# Maps a percentile metric key to the StandardizedFinancials attribute that
# holds it. "roe" reads `return_on_equity` (Yahoo's own reported figure,
# available shallow) rather than the statement-computed
# roe_pct_computed() the Quality Classification panel shows elsewhere —
# necessarily, since peers are only ever shallow-fetched here. Labeled
# "Return on Equity (Yahoo-reported)" wherever displayed so it's never
# mistaken for that other, independently-computed figure.
_METRIC_FIELD: Dict[str, str] = {
    "net_margin": "net_margin",
    "roe": "return_on_equity",
    "debt_to_equity": "debt_to_equity",
    "pe_ratio": "pe_ratio",
    "peg_ratio": "peg_ratio",
    "beta": "beta",
    "current_ratio": "current_ratio",
    "price_to_book": "price_to_book",
}


def _candidate_universe(exclude_ticker: str) -> Tuple[str, ...]:
    """Every ticker already configured elsewhere in the app — the
    Institutional Watchlist's two baskets plus every Peer Comparison
    default — deduplicated, uppercased, with the target ticker itself
    excluded. Reused as-is rather than inventing a new sector->tickers
    table to maintain."""
    tickers = set(WATCHLIST.tech_basket) | set(WATCHLIST.diversified_basket)
    for peer_str in PEER_DEFAULTS.by_ticker.values():
        tickers.update(t.strip().upper() for t in peer_str.split(",") if t.strip())
    tickers.update(t.strip().upper() for t in PEER_DEFAULTS.fallback.split(",") if t.strip())
    tickers.discard(exclude_ticker.upper())
    return tuple(sorted(tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_peer_sector_metrics(candidate_tickers: Tuple[str, ...]) -> Dict[str, Dict[str, Optional[float]]]:
    """Shallow-fetch every candidate's sector plus every SUPPORTED_METRICS
    value. Cached for 24h — sector membership and these ratios move slowly
    compared to price data, and this avoids re-fetching the whole
    configured universe on every rerun (this function's own underlying
    info fetch is separately cached at 30 min in data_loader.py; this
    outer cache is about not re-running the fetch LOOP itself, not about
    serving stale quote data past its own freshness window).
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for t in candidate_tickers:
        try:
            bundle = load_ticker_bundle(t, deep=False)
            if not bundle.is_valid:
                continue
            std = standardize_financials(bundle)
            out[t] = {"sector": std.sector, **{key: getattr(std, field) for key, field in _METRIC_FIELD.items()}}
        except Exception:
            log_event(logger, logging.WARNING, "peer_fetch.skipped", ticker=t)
            continue
    return out


@dataclass
class SectorPercentileResult:
    sector: str
    peer_tickers: Tuple[str, ...]                    # same-sector peers actually used
    percentiles: Dict[str, Optional[float]]           # metric key -> 0-100, None if that specific metric couldn't be ranked
    target_values: Dict[str, Optional[float]]         # the analyzed ticker's own values, for display alongside the percentile

    @property
    def peer_count(self) -> int:
        return len(self.peer_tickers)


def compute_sector_percentiles(target: StandardizedFinancials) -> Optional[SectorPercentileResult]:
    """Where `target` ranks against same-sector peers, metric by metric.

    Returns None (never a fabricated result) when the sector is unknown or
    fewer than MIN_PEERS peers in the configured universe share it — the
    risk this task's notes call out explicitly: a percentile computed from
    1-2 peers reads as precise but is actually just "above/below one
    specific company," not a real distribution.
    """
    if not target.sector:
        return None

    candidates = _candidate_universe(target.ticker)
    peer_data = _fetch_peer_sector_metrics(candidates)
    same_sector = {t: d for t, d in peer_data.items() if d.get("sector") == target.sector}

    if len(same_sector) < MIN_PEERS:
        log_event(logger, logging.INFO, "sector_percentile.insufficient_peers",
                   ticker=target.ticker, sector=target.sector, peer_count=len(same_sector), min_required=MIN_PEERS)
        return None

    target_values: Dict[str, Optional[float]] = {key: getattr(target, field) for key, field in _METRIC_FIELD.items()}
    percentiles: Dict[str, Optional[float]] = {}
    for key in SUPPORTED_METRICS:
        target_value = target_values.get(key)
        peer_values = [d[key] for d in same_sector.values() if d.get(key) is not None]
        if target_value is None or len(peer_values) < MIN_PEERS:
            percentiles[key] = None
            continue
        percentiles[key] = float(percentileofscore(peer_values, target_value, kind="mean"))

    log_event(logger, logging.INFO, "sector_percentile.computed",
              ticker=target.ticker, sector=target.sector, peer_count=len(same_sector))
    return SectorPercentileResult(
        sector=target.sector, peer_tickers=tuple(sorted(same_sector.keys())),
        percentiles=percentiles, target_values=target_values,
    )


def format_percentile(value: Optional[float]) -> str:
    """'85th percentile' style, matching ordinal phrasing throughout this
    task's own spec. 'N/A' when this specific metric couldn't be ranked
    (missing peer data for it specifically, even though the sector overall
    had enough peers for other metrics)."""
    if value is None:
        return "N/A"
    value_int = round(value)
    suffix = "th" if 11 <= value_int % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(value_int % 10, "th")
    return f"{value_int}{suffix} percentile"
