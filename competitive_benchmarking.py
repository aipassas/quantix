"""Competitive Benchmarking — outperform/laggard flagging for a
user-defined peer group.

EXTENDS, RATHER THAN DUPLICATES, THE EXISTING PEER COMPETITOR MATRIX:
finance.py's "Path 5: Peer Competitor Matrix" section already lets a user
define/edit a peer group (a comma-separated ticker input) and already
recomputes dynamically whenever that group changes — two of this task's
three acceptance criteria were already satisfied by that section before
this module existed. Building a second, parallel comparison view next to
it would have meant two UIs answering the same question with possibly
different numbers. This module instead adds the ONE thing that section
was missing: turning its existing per-ticker metrics into an explicit
outperform/laggard verdict per peer per metric, plus growth and momentum
— the two metric categories the task named that the matrix didn't have
yet (it already had valuation and margins).

MOMENTUM is computed identically to the existing "Relative Strength &
Alpha Generation" section's own headline return figure — Close[-1] /
Close[0] - 1 over the user's currently-SELECTED date range, not a
separately-invented lookback window — so a peer's momentum number means
the same thing the main ticker's own return number already means
elsewhere on this exact page.

FLAGGING METHODOLOGY: a peer's value on a metric is flagged "outperform"
or "laggard" only once it sits at least
COMPETITIVE_BENCHMARKING.outperform_threshold_pct percent away from the
group average, in the favorable or unfavorable direction for THAT metric
(some metrics are lower-is-better, e.g. P/E and Debt/Equity — see
METRICS below). This is a disclosed distance threshold, not a
statistical test: peer groups here are typically 2-6 names, far too few
for a significance test to mean anything, and presenting one would be
fabricating rigor the sample size can't support.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import COMPETITIVE_BENCHMARKING
from financial_standardization import StandardizedFinancials


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str  # "" | "%"
    decimals: int
    higher_is_better: bool


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("pe_ratio", "P/E Ratio", "", 1, False),
    MetricSpec("peg_ratio", "PEG Ratio", "", 2, False),
    MetricSpec("price_to_book", "Price/Book", "", 2, False),
    MetricSpec("debt_to_equity", "Debt/Equity", "", 2, False),
    MetricSpec("return_on_equity", "ROE", "%", 1, True),
    MetricSpec("net_margin", "Net Margin", "%", 1, True),
    MetricSpec("earnings_growth", "Earnings Growth", "%", 1, True),
    MetricSpec("momentum_pct", "Period Momentum", "%", 1, True),
)
METRICS_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in METRICS}


@dataclass
class PeerMetrics:
    ticker: str
    is_target: bool
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    status: str = "ok"          # "ok" | "unavailable"
    detail: str = ""


def build_peer_metrics(ticker: str, std: StandardizedFinancials, is_target: bool, momentum_pct: Optional[float]) -> PeerMetrics:
    """One row's worth of metric values from an already-standardized
    ticker plus a separately-supplied momentum figure (momentum needs
    price history, which standardize_financials() doesn't fetch for a
    shallow/deep=False peer bundle — the caller supplies it from whatever
    it already fetched, so this module never fetches data of its own)."""
    values = {
        "pe_ratio": std.pe_ratio,
        "peg_ratio": std.peg_ratio,
        "price_to_book": std.price_to_book,
        "debt_to_equity": std.debt_to_equity,
        "return_on_equity": std.return_on_equity * 100 if std.return_on_equity is not None else None,
        "net_margin": std.net_margin * 100 if std.net_margin is not None else None,
        "earnings_growth": std.earnings_growth * 100 if std.earnings_growth is not None else None,
        "momentum_pct": momentum_pct,
    }
    return PeerMetrics(ticker=ticker, is_target=is_target, values=values)


def group_average(rows: List[PeerMetrics], metric_key: str) -> Optional[float]:
    """Mean over whichever rows actually HAVE this metric — never
    fabricated by imputing a value for a ticker that's missing it, and
    None (not 0 or NaN) when nobody in the group has it."""
    available = [r.values[metric_key] for r in rows if r.values.get(metric_key) is not None]
    if not available:
        return None
    return sum(available) / len(available)


@dataclass(frozen=True)
class MetricFlag:
    metric: str
    value: Optional[float]
    group_average: Optional[float]
    verdict: str  # "outperform" | "laggard" | "in_line" | "unavailable"

    @property
    def icon(self) -> str:
        return {"outperform": "Outperform", "laggard": "Laggard",
                "in_line": "In line", "unavailable": "n/a"}[self.verdict]


def flag_metric(value: Optional[float], average: Optional[float], higher_is_better: bool) -> MetricFlag:
    """Never fabricates a verdict when either the value or the group
    average is missing — "unavailable", not a silently-defaulted
    "in_line"."""
    if value is None or average is None or average == 0:
        return MetricFlag(metric="", value=value, group_average=average, verdict="unavailable")

    pct_distance = ((value - average) / abs(average)) * 100
    is_favorable_direction = pct_distance > 0 if higher_is_better else pct_distance < 0

    if abs(pct_distance) < COMPETITIVE_BENCHMARKING.outperform_threshold_pct:
        verdict = "in_line"
    else:
        verdict = "outperform" if is_favorable_direction else "laggard"

    return MetricFlag(metric="", value=value, group_average=average, verdict=verdict)


@dataclass
class BenchmarkRow:
    metrics: PeerMetrics
    flags: Dict[str, MetricFlag]
    outperform_count: int
    laggard_count: int
    evaluable_count: int
    overall_verdict: str  # "Outperformer" | "Laggard" | "Mixed" | "Not Enough Data"

    @property
    def ticker(self) -> str:
        return self.metrics.ticker

    @property
    def overall_icon(self) -> str:
        return {"Outperformer": "Outperform", "Laggard": "Laggard",
                "Mixed": "Mixed", "Not Enough Data": "n/a"}[self.overall_verdict]


def _overall_verdict(outperform_count: int, laggard_count: int, evaluable_count: int) -> str:
    """A peer needs at least half its evaluable metrics to lean one way
    to earn a clean Outperformer/Laggard label; anything closer is
    "Mixed" rather than a forced call on a narrow majority (e.g. 2 of 3
    metrics) that a peer group this small shouldn't be read as decisive."""
    if evaluable_count == 0:
        return "Not Enough Data"
    if outperform_count >= (evaluable_count / 2) + 1:
        return "Outperformer"
    if laggard_count >= (evaluable_count / 2) + 1:
        return "Laggard"
    return "Mixed"


def build_benchmark_rows(rows: List[PeerMetrics]) -> List[BenchmarkRow]:
    """Flags every row against the GROUP AVERAGE (computed across all
    rows, target included — the target's own value pulls the average
    exactly as much as any peer's) for every declared metric, then rolls
    each row's per-metric flags into one overall verdict."""
    averages = {m.key: group_average(rows, m.key) for m in METRICS}

    result: List[BenchmarkRow] = []
    for row in rows:
        flags: Dict[str, MetricFlag] = {}
        outperform_count = laggard_count = evaluable_count = 0
        for m in METRICS:
            flag = flag_metric(row.values.get(m.key), averages[m.key], m.higher_is_better)
            flags[m.key] = flag
            if flag.verdict != "unavailable":
                evaluable_count += 1
                if flag.verdict == "outperform":
                    outperform_count += 1
                elif flag.verdict == "laggard":
                    laggard_count += 1
        result.append(BenchmarkRow(
            metrics=row, flags=flags,
            outperform_count=outperform_count, laggard_count=laggard_count, evaluable_count=evaluable_count,
            overall_verdict=_overall_verdict(outperform_count, laggard_count, evaluable_count),
        ))
    return result
