"""In-app help and support: searchable self-serve answers, plus an
explicit outbound report for what search can't answer.

NO LIVE CHAT, AND THAT IS THE CENTRAL DECISION. The originating task
asked for an "embedded chat/help widget" to "reduce support response
time". A chat box is a promise that somebody is reading it. Quantix runs
locally with no support organisation behind it, so that promise would be
false, and a message typed into an unanswered box is worse than no box —
it costs the user time and their problem still isn't solved. What
genuinely reduces time-to-answer here is making the answer findable
immediately, so search comes first and the contact form is the fallback.

A third-party chat widget (Intercom, Crisp, Zendesk) was considered and
rejected on three counts. It needs an account and workspace id that only
the person running this app can create. It would route whatever a user
types to a third party. And it cannot render as the floating bubble those
products assume: st.markdown(unsafe_allow_html=True) inserts via
innerHTML and browsers never execute a <script> inserted that way (proven
in this codebase — see onboarding.py), while st.components.v1.html does
run scripts but only inside a sandboxed iframe, which is a fixed box in
the page flow and cannot overlay anything.

THE HELP CORPUS IS ASSEMBLED, NOT DUPLICATED. metric_help.py already
holds 57 metric definitions and 13 chart explanations, written to be read
by a non-specialist and already shown as tooltips throughout the app.
Re-typing any of that here would create a second copy to drift out of
sync with the first, so build_index() pulls those in and adds only the
task-oriented FAQ that has no other home. One fact, one source.

DIAGNOSTICS ARE OPT-IN AND SHOWN BEFORE SENDING. Recent log lines make a
bug actionable, but they also reveal which tickers someone has been
researching — that is exactly the kind of thing a person should decide
about deliberately rather than discover afterwards. The checkbox is off
by default and the exact text that would be sent is rendered first.
"""
import datetime
import platform
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from config import SUPPORT
from metric_help import CHART_HELP, GLOSSARY

# Glossary keys are terse slugs used as lookup ids, not display text
# (`altman_z`, `var_historical`). Titles live at the call sites in
# finance.py, which search results have no access to — so the readable
# name is reconstructed here. Only the keys that don't title-case
# sensibly need an entry; a test asserts every key resolves to something
# clean, so this can't silently fall behind the glossary.
_TITLE_OVERRIDES: Dict[str, str] = {
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "calmar": "Calmar Ratio",
    "cvar": "Expected Shortfall (CVaR)",
    "var_historical": "Historical VaR",
    "var_parametric": "Parametric VaR",
    "altman_z": "Altman Z-Score",
    "rsi": "RSI (Relative Strength Index)",
    "atr": "ATR (Average True Range)",
    "vix": "VIX (Fear Index)",
    "treasury_10y": "10-Year Treasury Yield",
    "roc_auc": "ROC AUC",
    "kelly_half": "Half-Kelly Position Size",
    "oos_return": "Out-of-Sample Return",
    "price_z_score": "Price Z-Score",
    "hurst": "Hurst Exponent",
    "beta_systematic": "Beta (Systematic Risk)",
    "alpha_generated": "Alpha Generated",
    "alpha_selection": "Alpha from Selection",
    "volatility_rolling": "Rolling Volatility",
    "volatility_full_range": "Full-Range Volatility",
    "max_drawdown": "Maximum Drawdown",
    "peak_to_trough": "Peak-to-Trough Decline",
    "strategy_return_gross": "Strategy Return (Gross)",
    "strategy_return_net": "Strategy Return (Net of Costs)",
    "buy_hold_baseline": "Buy-and-Hold Baseline",
    "weighted_avg_volatility": "Weighted-Average Volatility",
    "yahoo_disagreements": "Yahoo Disagreements",
    "model_accuracy": "Model Accuracy",
    "price_technicals": "Price & Technical Indicators chart",
    "relative_strength": "Relative Strength chart",
    "risk_gauge": "Risk Gauge",
    "var_distribution": "VaR Distribution chart",
    "drawdown_underwater": "Underwater Drawdown chart",
    "strategy_equity": "Strategy Equity Curve",
    "walk_forward_equity": "Walk-Forward Equity Curve",
    "portfolio_equity": "Portfolio Equity Curve",
    "monte_carlo_paths": "Monte Carlo Paths chart",
    "seasonality_surface": "Seasonality Surface",
    "peer_radar": "Peer Radar chart",
    "correlation_matrix": "Correlation Matrix",
    "efficient_frontier": "Efficient Frontier chart",
}


def title_for(key: str) -> str:
    if key in _TITLE_OVERRIDES:
        return _TITLE_OVERRIDES[key]
    return key.replace("_", " ").title()


@dataclass(frozen=True)
class HelpArticle:
    id: str
    title: str
    body: str
    category: str            # "Getting started" | "Metric" | "Chart" | ...
    keywords: Tuple[str, ...] = ()

    @property
    def haystack(self) -> str:
        return " ".join((self.title, self.body, " ".join(self.keywords))).lower()


# The FAQ. Every answer here is about how THIS app actually behaves, and
# each one exists because it is a question the app's own design provokes
# — a metric that renders as unavailable, a setting that needs a restart,
# data that appears to vanish on first sign-in. Generic filler ("How do I
# use the app?") is deliberately absent; it would dilute search results
# without answering anything.
FAQ: Tuple[HelpArticle, ...] = (
    HelpArticle(
        id="faq_unavailable_metric",
        title="Why does a metric show as unavailable instead of a number?",
        category="Data",
        keywords=("missing", "n/a", "blank", "empty", "null", "zero", "not reported"),
        body=(
            "Because the figure genuinely wasn't reported, or can't be computed for that "
            "company. Quantix never substitutes a zero or a guess for missing data — a "
            "fabricated number that looks real is worse than an honest gap.\n\n"
            "This is common and expected for Financials: banks don't report a meaningful "
            "current ratio, for instance. A metric that can't be evaluated is excluded from "
            "both halves of the Blueprint Alignment score rather than counted as a failure, "
            "so a company isn't penalised for something it structurally cannot report."
        ),
    ),
    HelpArticle(
        id="faq_restart_needed",
        title="I changed a setting or added a file and the app didn't pick it up",
        category="Troubleshooting",
        keywords=("restart", "reload", "importerror", "module", "config", "not applied"),
        body=(
            "Restart Streamlit rather than just rerunning the page. The running process "
            "caches imported modules, so a brand-new module or a new constant in config.py "
            "isn't visible to a rerun and usually surfaces as an ImportError.\n\n"
            "Restarting is safe: everything Quantix persists — watchlists, favourites, theme, "
            "thresholds, alert rules, notes — is written to disk and survives."
        ),
    ),
    HelpArticle(
        id="faq_settings_vanished",
        title="My watchlists and settings disappeared after I signed in",
        category="Accounts",
        keywords=("lost", "gone", "missing", "empty", "sign in", "login", "oauth", "account"),
        body=(
            "Nothing was lost. Signing in gives you your own private profile, which starts "
            "empty; what you built up before signing in still belongs to the signed-out "
            "profile and reappears the moment you sign out.\n\n"
            "To bring it across, open the Account panel while signed in and use "
            "\"Copy them into my account\". It copies rather than moves, so the signed-out "
            "profile stays intact and the whole thing is reversible."
        ),
    ),
    HelpArticle(
        id="faq_enable_signin",
        title="How do I turn on sign-in?",
        category="Accounts",
        keywords=("oauth", "google", "microsoft", "github", "login", "sso", "authentication"),
        body=(
            "Install the dependencies (`pip install -r requirements.txt`, which adds Authlib), "
            "copy the [auth] block from .streamlit/secrets.toml.example into "
            ".streamlit/secrets.toml with your own client id and secret, then RESTART "
            "Streamlit — a rerun won't pick up a new dependency.\n\n"
            "You register the OAuth application yourself at Google Cloud Console or the Azure "
            "portal; the example file lists exactly what to create and where."
        ),
    ),
    HelpArticle(
        id="faq_github_signin",
        title="Why isn't GitHub offered as a sign-in option?",
        category="Accounts",
        keywords=("github", "oauth", "provider", "missing", "login"),
        body=(
            "Because GitHub doesn't support the protocol Streamlit's sign-in requires. "
            "Streamlit needs an OpenID Connect discovery document; GitHub implements plain "
            "OAuth 2.0 for user login, issues no id_token, and returns 404 for its "
            "well-known OIDC configuration URL.\n\n"
            "You can still sign in with GitHub through an OIDC broker — Auth0, Okta, Keycloak "
            "or Entra External ID — configured with GitHub as an upstream connection. Quantix "
            "reads its provider list from your secrets file rather than a fixed list, so "
            "pointing an [auth.github] table at a broker makes the button appear with no code "
            "change."
        ),
    ),
    HelpArticle(
        id="faq_email_setup",
        title="How do I email a report, and why does emailing say it isn't configured?",
        category="Setup",
        keywords=("smtp", "mail", "send", "tear sheet", "pdf", "notification", "gmail"),
        body=(
            "Quantix stores no credentials of its own, so it reads mail settings at send time "
            "from .streamlit/secrets.toml (an [smtp] table) or QUANTIX_SMTP_* environment "
            "variables. Until one of those exists, anything that sends mail explains that it "
            "isn't configured instead of failing.\n\n"
            "See .streamlit/secrets.toml.example for the exact keys. For Gmail specifically "
            "you need a 16-character App Password, not your normal login password."
        ),
    ),
    HelpArticle(
        id="faq_alerts_tab_open",
        title="Why do my alerts only fire while the browser tab is open?",
        category="Alerts",
        keywords=("alert", "notification", "background", "monitoring", "closed", "polling"),
        body=(
            "Because monitoring runs inside the page, polling on a timer while the tab is "
            "open. There is no always-on background worker — that would be a second "
            "permanently-running process, a much larger piece of architecture than the "
            "in-tab monitoring this provides.\n\n"
            "Your rules and trigger history are saved to disk and survive a restart; only the "
            "checking pauses when the tab closes."
        ),
    ),
    HelpArticle(
        id="faq_change_thresholds",
        title="Can I change the thresholds Quantix judges a company against?",
        category="Getting started",
        keywords=("threshold", "scorecard", "custom", "tune", "criteria", "sector pe"),
        body=(
            "Yes — open Custom Thresholds in the sidebar. You can retune the scorecard's "
            "margin, leverage, coverage and valuation cut-offs, the risk thresholds, and the "
            "sector P/E bands.\n\n"
            "Only your changes are saved, not a full snapshot, so any threshold you haven't "
            "touched keeps tracking the shipped default if that default is ever revised. "
            "Every tooltip and verdict reads your effective values, so the explanation and "
            "the number it describes can't disagree."
        ),
    ),
    HelpArticle(
        id="faq_alignment_score",
        title="What does the Blueprint Alignment score actually measure?",
        category="Getting started",
        keywords=("alignment", "score", "blueprint", "green flags", "verdict", "high", "moderate"),
        body=(
            "It's a weighted pass rate across the financial-health checks that could be "
            "evaluated for this company. Core signals — profitability, leverage, capital "
            "efficiency — count for more than secondary ones like valuation multiples and "
            "volatility.\n\n"
            "Checks that can't be computed are left out of the calculation entirely rather "
            "than scored as failures, so a company with sparse data isn't punished for the "
            "gaps. That also means a very high score on a company with few evaluable metrics "
            "is a weaker signal than the same score on one with many."
        ),
    ),
    HelpArticle(
        id="faq_historical_comparison",
        title="How do I compare a stock to how it looked at an earlier date?",
        category="Getting started",
        keywords=("historical", "compare", "then", "now", "past", "as of", "backtest"),
        body=(
            "Use the Historical Comparison panel. It rebuilds the analysis as it would have "
            "looked on a chosen past date, using only information available by then — price "
            "history is truncated and financial statements filed after that date are "
            "excluded, so the comparison isn't quietly informed by hindsight.\n\n"
            "Some metrics may be unavailable in the historical column where the underlying "
            "statements hadn't been filed yet. That's the honest answer rather than a "
            "back-filled one."
        ),
    ),
    HelpArticle(
        id="faq_api_access",
        title="Can a script or bot read my Quantix data?",
        category="Setup",
        keywords=("api", "key", "programmatic", "script", "automation", "bot", "curl", "robot"),
        body=(
            "Yes. Create a scoped key in the API Keys panel, then start the API with "
            "`python3 api_server.py`. It listens on 127.0.0.1:8787 and `GET /v1` lists every "
            "endpoint and scope.\n\n"
            "The API is read-only — quotes, fundamentals, risk metrics and your watchlists. "
            "There is deliberately no endpoint that places trades or changes anything, "
            "because Quantix has no brokerage connection. The key's secret is shown once at "
            "creation and only its hash is stored, so it can't be looked up again; if you "
            "lose it, revoke it and issue another."
        ),
    ),
    HelpArticle(
        id="faq_where_is_my_data",
        title="Where is my data stored, and does anything leave my machine?",
        category="Privacy",
        keywords=("privacy", "data", "stored", "local", "cloud", "gdpr", "security", "files"),
        body=(
            "Everything Quantix saves is a plain JSON file in the application folder on this "
            "machine, and all of them are gitignored. Signed in, your personal settings live "
            "under a users/ subfolder keyed to your account.\n\n"
            "Nothing is uploaded anywhere. The app makes outbound requests only to fetch "
            "market data for the tickers you look at, and to send mail if you have configured "
            "SMTP and explicitly ask it to. There is no analytics or telemetry."
        ),
    ),
    HelpArticle(
        id="faq_note_not_posting",
        title="Why did my team note post twice, or seem not to post at all?",
        category="Troubleshooting",
        keywords=("note", "duplicate", "posted", "twice", "collaboration", "mention"),
        body=(
            "That was a bug in an earlier version where the compose box kept its text after "
            "posting, which read as \"nothing happened\" and invited a second click. It's "
            "fixed — the box now clears on a successful post.\n\n"
            "If a note is rejected the text is deliberately kept so you don't lose what you "
            "wrote; check for a warning message explaining what was missing, usually an "
            "author name."
        ),
    ),
)


def build_index() -> Tuple[HelpArticle, ...]:
    """The full searchable corpus: the FAQ above plus every metric and
    chart explanation already written for the tooltips.

    Assembled at call time from metric_help rather than copied, so a
    definition edited for a tooltip is the same text search returns.
    """
    articles: List[HelpArticle] = list(FAQ)
    for key, text in GLOSSARY.items():
        articles.append(HelpArticle(
            id=f"metric_{key}", title=title_for(key), body=text,
            category="Metric", keywords=(key,),
        ))
    for key, text in CHART_HELP.items():
        articles.append(HelpArticle(
            id=f"chart_{key}", title=title_for(key), body=text,
            category="Chart", keywords=(key, "chart", "graph"),
        ))
    return tuple(articles)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _terms(query: str) -> List[str]:
    return _WORD_RE.findall((query or "").lower())


def search(query: str, index: Optional[Tuple[HelpArticle, ...]] = None,
           limit: Optional[int] = None) -> Tuple[HelpArticle, ...]:
    """Rank the corpus against `query`. Empty query returns nothing.

    Scoring is intentionally simple and explainable rather than clever: a
    term in the title outweighs one in the body, and an article must match
    EVERY term to appear at all. Requiring all terms is what stops
    "dividend yield" from returning every article mentioning "yield" —
    with a corpus this small, precision matters far more than recall.
    """
    index = index if index is not None else build_index()
    limit = limit if limit is not None else SUPPORT.search_results_shown
    terms = _terms(query)
    if not terms:
        return ()

    scored: List[Tuple[float, int, HelpArticle]] = []
    for position, article in enumerate(index):
        title = article.title.lower()
        haystack = article.haystack
        if not all(term in haystack for term in terms):
            continue
        score = 0.0
        for term in terms:
            if term in title:
                score += 3.0
            if term in article.keywords:
                score += 2.0
            score += haystack.count(term) * 0.1
        # Whole-phrase title hit is the strongest signal there is.
        if query.strip().lower() in title:
            score += 5.0
        # FAQ entries answer questions; metric definitions define terms.
        # On a tie the question-shaped article is likelier to be wanted.
        if article.category not in ("Metric", "Chart"):
            score += 0.5
        scored.append((-score, position, article))

    scored.sort()
    return tuple(article for _, _, article in scored[:limit])


def categories(index: Optional[Tuple[HelpArticle, ...]] = None) -> Tuple[str, ...]:
    index = index if index is not None else build_index()
    seen: List[str] = []
    for article in index:
        if article.category not in seen:
            seen.append(article.category)
    return tuple(seen)


def browse(category: str, index: Optional[Tuple[HelpArticle, ...]] = None) -> Tuple[HelpArticle, ...]:
    index = index if index is not None else build_index()
    return tuple(a for a in index if a.category == category)


# --- support reports ----------------------------------------------------------

@dataclass(frozen=True)
class SupportReport:
    category: str
    subject: str
    body: str
    reply_to: str = ""
    diagnostics: str = ""

    def as_email_body(self) -> str:
        parts = [
            f"Category: {self.category}",
            f"Reply to: {self.reply_to or '(not supplied)'}",
            "",
            self.body,
        ]
        if self.diagnostics:
            parts += ["", "--- diagnostics (attached by the sender) ---", self.diagnostics]
        return "\n".join(parts)


def diagnostics_snapshot(log_lines: Optional[int] = None,
                         log_reader: Optional[Callable[[int], List[str]]] = None,
                         extra: Optional[Dict[str, str]] = None) -> str:
    """Environment details plus recent log lines, as plain text.

    Returned as text rather than assembled inside the sender so the UI can
    show the user EXACTLY what they are about to disclose. Log lines name
    the tickers someone has been looking at, which is theirs to decide
    about — it must never be possible to send this without having seen it.

    `log_reader` is injectable so tests don't depend on a log file.
    """
    log_lines = log_lines if log_lines is not None else SUPPORT.diagnostics_log_lines
    rows = [
        f"Quantix diagnostics — {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.system()} {platform.release()}",
    ]
    try:
        import streamlit
        rows.append(f"Streamlit: {streamlit.__version__}")
    except Exception:
        rows.append("Streamlit: unknown")
    for key, value in (extra or {}).items():
        rows.append(f"{key}: {value}")

    if log_reader is None:
        def log_reader(limit):
            from logging_setup import recent_logs
            return recent_logs(limit=limit)
    try:
        recent = log_reader(log_lines)
    except Exception as e:
        recent = [f"(could not read the log: {type(e).__name__})"]

    rows.append("")
    rows.append(f"Last {len(recent)} log lines:")
    rows.extend(str(line).rstrip() for line in recent)
    return "\n".join(rows)


def compose_report(category: str, subject: str, body: str, reply_to: str = "",
                   diagnostics: str = "") -> Tuple[Optional[SupportReport], Optional[str]]:
    """Validate and assemble a report. Returns (report, error)."""
    category = (category or "").strip() or SUPPORT.categories[0]
    subject = (subject or "").strip()
    body = (body or "").strip()
    reply_to = (reply_to or "").strip()

    if not subject:
        return None, "Give the report a short subject."
    if len(subject) > SUPPORT.max_subject_chars:
        return None, f"Subjects are capped at {SUPPORT.max_subject_chars} characters."
    if not body:
        return None, "Describe the problem or question first."
    if len(body) > SUPPORT.max_body_chars:
        return None, f"Reports are capped at {SUPPORT.max_body_chars} characters."
    if reply_to and "@" not in reply_to:
        return None, "That reply-to address doesn't look like an email address."

    return SupportReport(category=category, subject=subject, body=body,
                         reply_to=reply_to, diagnostics=diagnostics), None


def is_destination_configured() -> bool:
    return bool((SUPPORT.support_address or "").strip())


def send_report(report: SupportReport, sender: Optional[Callable] = None,
                to_address: str = "") -> Tuple[bool, Optional[str]]:
    """Email a composed report. Returns (ok, error); never raises.

    `sender` is injected (email_report.send_notification_email in the app,
    a fake in tests) so this is testable without a mail server.

    A missing destination is reported as a plain fact rather than an
    error, because on a stock install there ISN'T one — the UI's job then
    is to show the composed text for the user to copy somewhere useful,
    not to imply something broke.
    """
    to_address = (to_address or SUPPORT.support_address or "").strip()
    if not to_address:
        return False, (
            "No support address is configured for this Quantix instance, so there's nowhere "
            "to send this. Copy the report below into an email or a GitHub issue instead."
        )
    if sender is None:
        from email_report import send_notification_email
        sender = send_notification_email
    subject = f"[Quantix {report.category}] {report.subject}"
    return sender(to_address, subject, report.as_email_body())
