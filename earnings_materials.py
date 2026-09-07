"""Earnings materials from SEC filings, searchable across quarters.

THE TASK ASKED FOR CALL TRANSCRIPTS. THEY DO NOT EXIST HERE, AND THAT IS
A MEASUREMENT, NOT AN ASSUMPTION. Sampling the latest earnings 8-K of 25
large caps on 2026-09-07:

    file a text earnings exhibit ...... 25 of 25
    file management prepared remarks ..  1 of 25  (HUM, exhibit 99.3)
    contain analyst Q&A ...............  0 of 25

Zero. Not a minority — none. A company files its written earnings
release with the SEC because Regulation FD requires the numbers to reach
everyone at once; the CALL itself, where analysts ask questions, is not
a filed document. Verbatim transcripts are a licensed product (FactSet,
Refinitiv, S&P) with no free tier and no keyless endpoint, and this
build does not register for accounts on the user's behalf.

So this module is named for what it actually holds. Calling it
"Transcripts" would promise the analyst Q&A — the half people actually
want — and deliver a press release. The panel says the same thing on
screen rather than leaving the disappointment to be discovered.

WHAT IS HERE IS NOT NOTHING. The filed exhibits run 10,000 to 113,000
characters per quarter and include the CEO's quoted commentary, the
segment discussion, and the guidance language, going back to whenever
the issuer's filing index starts (2015 for AAPL, 2020 for MSFT). Some
issuers file real management commentary alongside the numbers: HUM posts
its prepared remarks, NVDA files a CFO Commentary exhibit. Searching
that corpus for "tariff" or "China" across twelve quarters is the
research task the ticket described, and it is fully supported.

TWO KEYWORD TRAPS, BOTH FOUND BY MEASUREMENT AND BOTH GUARDED HERE.

First: "prepared remarks" in an exhibit usually means the company is
ANNOUNCING them, not filing them. CSCO's release says "Text of the
conference call's prepared remarks will be available within 24 hours";
CVX's says "Prepared remarks for today's call ... on Chevron's website".
A naive substring test classified both as management commentary. Only a
SELF-REFERENTIAL mention ("Certain of the matters discussed in these
prepared remarks are forward-looking" — HUM) means the document is the
remarks. `_is_announcement_only` is that guard.

Second: "operator" is not a Q&A marker. It is an oil-field operator, a
network operator, a plant operator. Matching it flagged CVX and CSCO as
containing analyst Q&A when neither does.

Classification therefore reads three signals — the filer's exhibit
description, the document filename, and the body — because none is
sufficient alone. Descriptions are usually the literal string "EX-99.1"
(measured: 6 of 8 issuers), so the filename carries most of the signal:
`q2fy27cfocommentary.htm`, `a2q2026humanaincpostedrema.htm`,
`a2q26erfex992supplement.htm` all say what they are.

SEC ACCESS RULES. data.sec.gov requires a declared User-Agent naming a
real contact, and asks for no more than ten requests a second. Both are
honoured: the contact is a config constant a licensee sets to their own
address, and fetches are throttled and cached. One quarter costs one
index request plus one per exhibit, so the default of eight quarters is
roughly twenty requests — cached for a day afterwards.
"""
import datetime
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from config import EARNINGS_MATERIALS
from logging_setup import get_logger, log_exception

logger = get_logger("earnings_materials")

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
SEC_ROOT = "https://www.sec.gov"

# 8-K Item 2.02 is "Results of Operations and Financial Condition" — the
# item a company files its quarterly numbers under. Filtering on it is
# what separates an earnings 8-K from the 100+ other 8-Ks an issuer files
# (TSLA files 126 in the recent window; 68 are earnings).
EARNINGS_ITEM = "2.02"

REQUEST_TIMEOUT_SECONDS = 20

# The ticker->CIK map changes when a company lists or renames; a day is
# plenty. Filings change quarterly. Documents, once filed, never change:
# an accession number addresses an immutable document, so its TTL is
# long and exists only to bound the cache, not to catch updates.
TICKER_MAP_TTL_SECONDS = 24 * 60 * 60
FILINGS_TTL_SECONDS = 6 * 60 * 60
DOCUMENT_TTL_SECONDS = 7 * 24 * 60 * 60

# Document kinds. MANAGEMENT_COMMENTARY is the one worth surfacing
# separately — it is the closest thing in the filing record to what a
# transcript would give you.
MANAGEMENT_COMMENTARY = "Management commentary"
RESULTS_RELEASE = "Results release"
SUPPLEMENT = "Financial supplement"

# Shown wherever the panel could be mistaken for a transcript archive.
TRANSCRIPTS_UNAVAILABLE = (
    "These are the earnings documents the company filed with the SEC — "
    "the results release, and whatever commentary or supplement it filed "
    "alongside. They are not call transcripts: the analyst Q&A from the "
    "earnings call is not a filed document and is not available from any "
    "free source. Sampling 25 large caps, all 25 filed a text earnings "
    "exhibit, one filed its prepared remarks, and none contained Q&A. "
    "For the call itself, use the company's investor-relations site."
)


@dataclass(frozen=True)
class Filing:
    """One earnings 8-K."""
    ticker: str
    cik: str
    accession: str
    filed_on: str           # YYYY-MM-DD, as EDGAR reports it
    items: str

    @property
    def index_url(self) -> str:
        bare = self.accession.replace("-", "")
        return f"{SEC_ARCHIVE_BASE}/{int(self.cik)}/{bare}/{self.accession}-index.htm"


@dataclass(frozen=True)
class Document:
    """One text exhibit from an earnings 8-K, with its extracted body."""
    ticker: str
    accession: str
    filed_on: str
    exhibit: str            # "EX-99.1"
    filename: str
    url: str
    description: str
    text: str
    kind: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def label(self) -> str:
        """How the document is named on screen: the date, the kind and the
        exhibit number, because two exhibits in one filing are routinely
        the same kind and the number is the only thing telling them
        apart."""
        return f"{self.filed_on} · {self.kind} · {self.exhibit}"


@dataclass(frozen=True)
class Hit:
    """One keyword match, with enough context to read without opening the
    document."""
    document: Document
    snippet: str
    position: int


@dataclass(frozen=True)
class SearchResult:
    hits: Tuple[Hit, ...]
    documents_searched: int
    documents_matched: int
    query: str

    @property
    def total_hits(self) -> int:
        return len(self.hits)


# --- HTML to text -------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Body text from a filed exhibit.

    Deliberately stdlib rather than a parser dependency: EDGAR exhibits
    are plain, table-heavy HTML with no scripting, and the app already
    carries enough of a dependency surface.

    Block-level tags emit a space so that a table cell does not weld
    itself onto the next one — "$1,234Revenue" would otherwise be one
    token and would break both search and word counts.
    """

    _BLOCK = {"p", "div", "br", "tr", "td", "th", "li", "table",
              "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._suppress = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("script", "style"):
            self._suppress += 1
        elif tag in self._BLOCK:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._suppress = max(0, self._suppress - 1)
        elif tag in self._BLOCK:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._suppress:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def html_to_text(html: str) -> str:
    """Readable text from an EDGAR exhibit. Never raises: a malformed
    document degrades to whatever parsed, because one bad quarter must
    not remove eleven good ones from the search."""
    try:
        parser = _TextExtractor()
        parser.feed(html or "")
        parser.close()
        return parser.text
    except Exception:
        log_exception(logger, "earnings.parse_failed", section="earnings_materials")
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


# --- classification -----------------------------------------------------------

# Filename and description fragments, checked before the body because the
# filer's own naming is the most reliable signal available. Measured on
# real filings: NVDA "q2fy27cfocommentary", HUM "…postedremarks",
# JPM "…ex992supplement", WMT "earningspresentationfy27".
_COMMENTARY_NAME_HINTS = (
    "cfocommentary", "cfo-commentary", "cfo_commentary",
    "postedrema", "postedremarks", "preparedrema", "prepared-remarks",
    "commentary", "remarks",
)
_SUPPLEMENT_NAME_HINTS = (
    "supplement", "supplemental", "statistical", "financialtables",
    "presentation", "slides", "deck", "infographic",
)
# Checked LAST of the three, and only to confirm a press release, because
# "EARNINGS RELEASE FINANCIAL SUPPLEMENT" (JPM's exhibit 99.2 description)
# contains both this and the supplement hint. Specific beats general.
_RELEASE_NAME_HINTS = (
    "earningsrelease", "pressrelease", "newsrelease", "earningspr",
    "-pr.", "results",
)

# A self-referential mention: the document is TALKING ABOUT ITSELF, which
# an announcement of remarks posted elsewhere never does.
_SELF_REFERENTIAL = re.compile(
    r"\b(?:these|this|the\s+following|the\s+accompanying|contained\s+in\s+these)\s+"
    r"(?:prepared\s+)?(?:remarks|commentary)\b",
    re.IGNORECASE,
)

# The announcement shape: remarks that WILL BE, or ARE, available
# somewhere else. CSCO and CVX both match this and must not be
# classified as commentary.
_ANNOUNCEMENT = re.compile(
    r"(?:prepared\s+remarks|commentary)[^.]{0,120}?"
    r"(?:will\s+be\s+available|are\s+available|is\s+available|"
    r"will\s+be\s+posted|can\s+be\s+found|available\s+(?:on|at))",
    re.IGNORECASE,
)


def _is_announcement_only(text: str) -> bool:
    """True when every mention of remarks is the company announcing them
    rather than filing them.

    This is the guard the naive version lacked. Measured false positives
    it removes: CSCO ("Text of the conference call's prepared remarks
    will be available within 24 hours of completion of the call") and CVX
    ("Prepared remarks for today's call ... on Chevron's website").
    """
    if _SELF_REFERENTIAL.search(text):
        return False
    return bool(_ANNOUNCEMENT.search(text))


def classify_document(text: str, filename: str = "", description: str = "") -> str:
    """Which of the three kinds this exhibit is.

    Order matters, and it is most-specific-first rather than any
    intuitive reading order. The filer's naming wins when it is
    informative, because a document called `cfocommentary.htm` is
    management commentary whether or not it happens to use the phrase
    "prepared remarks" — NVDA's does not.

    Supplement is tested before release because JPM describes its tables
    as "EARNINGS RELEASE FINANCIAL SUPPLEMENT", which matches both.
    Release is tested after commentary for the same reason and in the
    other direction: JPM's press release is filed as `…narrative.htm`,
    which read as commentary until its description ("EARNINGS RELEASE —
    SECOND QUARTER 2026 RESULTS") was given the deciding vote.
    """
    haystack = f"{filename} {description}".lower().replace(" ", "")

    if any(hint in haystack for hint in _SUPPLEMENT_NAME_HINTS):
        return SUPPLEMENT
    if any(hint in haystack for hint in _COMMENTARY_NAME_HINTS):
        return MANAGEMENT_COMMENTARY
    if any(hint in haystack for hint in _RELEASE_NAME_HINTS):
        return RESULTS_RELEASE

    body = text or ""
    if _SELF_REFERENTIAL.search(body) and not _is_announcement_only(body):
        return MANAGEMENT_COMMENTARY

    # A supplement is mostly numbers. Measured: JPM's 88k-character
    # supplement is ~34% digits, its narrative release ~11%. The
    # threshold sits between them with room on both sides.
    if body and _digit_share(body) > EARNINGS_MATERIALS.supplement_digit_share:
        return SUPPLEMENT

    return RESULTS_RELEASE


def _digit_share(text: str) -> float:
    """Fraction of non-space characters that are digits. A financial
    supplement is a table; a narrative release is prose."""
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    return sum(c.isdigit() for c in stripped) / len(stripped)


# --- SEC fetching -------------------------------------------------------------

def _headers() -> dict:
    """SEC requires a User-Agent that names a real contact. Sending a
    browser-shaped one, or none, gets the request refused — and doing it
    anyway would be misrepresenting who is calling."""
    return {
        "User-Agent": EARNINGS_MATERIALS.user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _sec_get(url: str) -> str:
    """One throttled GET against SEC, returning body text. Raises on a
    non-200 so the caller can turn it into a message."""
    import requests

    time.sleep(EARNINGS_MATERIALS.request_interval_seconds)
    response = requests.get(url, headers=_headers(),
                            timeout=REQUEST_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} from {url}")
    return response.text


@st.cache_data(ttl=TICKER_MAP_TTL_SECONDS, show_spinner=False)
def load_ticker_map() -> Tuple[dict, Optional[str]]:
    """Every ticker the SEC knows, mapped to its zero-padded CIK.

    One file, ~10,400 entries, cached for a day. Fetching the whole map
    is cheaper than a per-ticker lookup endpoint and means a failed
    lookup can say whether the ticker is unknown to the SEC or merely
    has no earnings filings — two different answers.
    """
    import json

    try:
        raw = json.loads(_sec_get(SEC_TICKERS_URL))
    except Exception as exc:
        log_exception(logger, "earnings.ticker_map_failed",
                      section="earnings_materials")
        return {}, f"Could not reach the SEC ticker index: {exc}"
    mapping = {}
    for entry in (raw or {}).values():
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("ticker") or "").upper()
        cik = str(entry.get("cik_str") or "").strip()
        if symbol and cik.isdigit():
            mapping[symbol] = cik.zfill(10)
    return mapping, None


def resolve_cik(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """The SEC CIK for a ticker, or an explanation.

    A miss here is usually a non-US listing: the SEC index carries US
    registrants only, so VWCE.DE or a crypto pair will never be in it.
    Saying that is more useful than "not found".
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None, "No ticker to look up."
    mapping, error = load_ticker_map()
    if error:
        return None, error
    cik = mapping.get(symbol)
    if cik:
        return cik, None
    return None, (
        f"{symbol} is not in the SEC's registrant index. Only companies that "
        "file with the SEC appear there, so non-US listings, funds and "
        "crypto have no filings to search."
    )


@st.cache_data(ttl=FILINGS_TTL_SECONDS, show_spinner=False)
def load_filings(ticker: str, limit: int = 40) -> Tuple[Tuple[Filing, ...], Optional[str]]:
    """Earnings 8-Ks for a ticker, newest first.

    Reads the `recent` block of the submissions API, which holds the last
    1,000 filings. For a large cap that reaches back five to ten years —
    AAPL's earliest is 2015, MSFT's 2020. It is NOT the issuer's complete
    history, and `earliest_filing_date` is surfaced so the panel can say
    how far back the search actually goes rather than implying "all".
    """
    import json

    cik, error = resolve_cik(ticker)
    if error or not cik:
        return (), error

    try:
        raw = json.loads(_sec_get(SEC_SUBMISSIONS_URL.format(cik=cik)))
    except Exception as exc:
        log_exception(logger, "earnings.submissions_failed",
                      section="earnings_materials")
        return (), f"Could not reach the SEC filing index: {exc}"

    recent = ((raw or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    items = recent.get("items") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []

    out: List[Filing] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        item_text = items[i] if i < len(items) else ""
        if EARNINGS_ITEM not in (item_text or ""):
            continue
        out.append(Filing(
            ticker=(ticker or "").strip().upper(),
            cik=cik,
            accession=accessions[i] if i < len(accessions) else "",
            filed_on=dates[i] if i < len(dates) else "",
            items=item_text or "",
        ))
        if len(out) >= max(1, int(limit)):
            break

    if not out:
        return (), (
            f"{(ticker or '').strip().upper()} has no 8-K filings reporting "
            "results (item 2.02) in the SEC's recent index."
        )
    return tuple(out), None


_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

# Only these extensions carry readable text. TSLA files its shareholder
# deck as 33 JPEGs in the same filing; an image exhibit is skipped rather
# than counted as a document with no text, which would render as an empty
# search result the reader cannot explain.
_TEXT_EXTENSIONS = (".htm", ".html", ".txt")


def _cell_text(html: str) -> str:
    return _TAG_RE.sub("", html).replace("\xa0", " ").replace("&amp;", "&").strip()


def parse_filing_index(html: str) -> Tuple[Tuple[str, str, str], ...]:
    """(exhibit type, description, href) for each EX-99 text document.

    Reads the declared TYPE column rather than pattern-matching
    filenames. That distinction is load-bearing: KO's earnings exhibit is
    `a2026q2earningsreleaseex-9.htm` and TSLA's is buried among 33 JPEGs,
    and a filename regex found neither while the type column finds both.
    """
    found: List[Tuple[str, str, str]] = []
    for row in _ROW_RE.findall(html or ""):
        cells = [_cell_text(c) for c in _CELL_RE.findall(row)]
        if len(cells) < 4:
            continue
        exhibit = cells[3].upper().replace("EXHIBIT", "EX-").replace(" ", "")
        if not exhibit.startswith("EX-99"):
            continue
        href_match = _HREF_RE.search(row)
        if not href_match:
            continue
        href = href_match.group(1)
        if not href.lower().endswith(_TEXT_EXTENSIONS):
            continue
        found.append((cells[3].strip(), cells[1].strip(), href))
    return tuple(found)


@st.cache_data(ttl=DOCUMENT_TTL_SECONDS, show_spinner=False)
def load_documents(ticker: str, quarters: int = 0
                   ) -> Tuple[Tuple[Document, ...], Tuple[str, ...]]:
    """Text exhibits for the most recent `quarters` earnings filings.

    Returns (documents, warnings). A quarter that fails to fetch produces
    a warning and is skipped: eleven readable quarters with a note about
    the twelfth is a better answer than an error page.
    """
    quarters = int(quarters or EARNINGS_MATERIALS.default_quarters)
    quarters = max(1, min(quarters, EARNINGS_MATERIALS.max_quarters))

    filings, error = load_filings(ticker)
    if error:
        return (), (error,)

    documents: List[Document] = []
    warnings: List[str] = []
    for filing in filings[:quarters]:
        try:
            index_html = _sec_get(filing.index_url)
        except Exception as exc:
            warnings.append(f"{filing.filed_on}: could not read the filing index ({exc}).")
            continue
        entries = parse_filing_index(index_html)
        if not entries:
            warnings.append(
                f"{filing.filed_on}: this filing has no text exhibit — the "
                "company filed its results as images or data only."
            )
            continue
        for exhibit, description, href in entries:
            url = href if href.startswith("http") else f"{SEC_ROOT}{href}"
            try:
                body = html_to_text(_sec_get(url))
            except Exception as exc:
                warnings.append(f"{filing.filed_on} {exhibit}: could not read the exhibit ({exc}).")
                continue
            if len(body) < EARNINGS_MATERIALS.min_document_chars:
                continue
            filename = url.rsplit("/", 1)[-1]
            documents.append(Document(
                ticker=filing.ticker,
                accession=filing.accession,
                filed_on=filing.filed_on,
                exhibit=exhibit,
                filename=filename,
                url=url,
                description=description,
                text=body,
                kind=classify_document(body, filename, description),
            ))
    return tuple(documents), tuple(warnings)


# --- search -------------------------------------------------------------------

def _snippet(text: str, start: int, end: int, context: int) -> str:
    """Keyword in context, cut at word boundaries so a snippet never
    begins mid-word."""
    left = max(0, start - context)
    right = min(len(text), end + context)
    fragment = text[left:right]
    if left > 0:
        space = fragment.find(" ")
        fragment = ("… " + fragment[space + 1:]) if space != -1 else ("… " + fragment)
    if right < len(text):
        space = fragment.rfind(" ")
        fragment = (fragment[:space] + " …") if space != -1 else (fragment + " …")
    return fragment.strip()


def search(documents: Sequence[Document], query: str,
           context_chars: int = 0, max_hits_per_document: int = 0) -> SearchResult:
    """Every occurrence of `query` across the given documents.

    Case-insensitive substring search on a whole-word boundary where the
    query looks like a word, so searching "AI" does not match "said" or
    "China" — a plain substring search made the feature useless on short
    queries, which are the common case.

    Hits per document are capped: a search for "revenue" against a
    financial supplement returns hundreds, and a reader wants to know
    the document is full of them, not scroll through every one. The cap
    is reported so the count is never silently wrong.
    """
    query = (query or "").strip()
    context_chars = int(context_chars or EARNINGS_MATERIALS.snippet_context_chars)
    cap = int(max_hits_per_document or EARNINGS_MATERIALS.max_hits_per_document)
    if not query:
        return SearchResult((), len(documents), 0, query)

    # A word-boundary pattern when the query is alphanumeric; a literal
    # one otherwise, so "q4 2026" or "$1.2 billion" still works.
    escaped = re.escape(query)
    if re.fullmatch(r"[\w\s'-]+", query):
        pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
    else:
        pattern = re.compile(escaped, re.IGNORECASE)

    hits: List[Hit] = []
    matched = 0
    for document in documents:
        found = 0
        for match in pattern.finditer(document.text):
            if found >= cap:
                break
            hits.append(Hit(
                document=document,
                snippet=_snippet(document.text, match.start(), match.end(), context_chars),
                position=match.start(),
            ))
            found += 1
        if found:
            matched += 1
    return SearchResult(tuple(hits), len(documents), matched, query)


def coverage_note(documents: Sequence[Document]) -> str:
    """What the corpus actually contains, in one sentence.

    Named quarters and a real date range, because "8 documents" does not
    tell a reader whether the search covered the period they care about.
    """
    if not documents:
        return "No earnings documents were retrieved."
    dates = sorted({d.filed_on for d in documents if d.filed_on})
    quarters = len(dates)
    words = sum(d.word_count for d in documents)
    commentary = sum(1 for d in documents if d.kind == MANAGEMENT_COMMENTARY)
    span = f"{dates[0]} to {dates[-1]}" if dates else "an unknown period"
    note = (
        f"Searching {len(documents)} document(s) from {quarters} filing(s), "
        f"{span} — about {words:,} words."
    )
    if commentary:
        note += (
            f" {commentary} of them {'is' if commentary == 1 else 'are'} "
            "management commentary rather than a numbers release."
        )
    else:
        note += (
            " None of them is management commentary — this issuer files "
            "results only, and discusses them on the call instead."
        )
    return note


def earliest_filing_date(filings: Sequence[Filing]) -> Optional[datetime.date]:
    """The oldest earnings filing the index reached, so the panel can
    state the real limit of the archive instead of implying it is
    complete."""
    dates = []
    for filing in filings:
        try:
            dates.append(datetime.date.fromisoformat(filing.filed_on))
        except Exception:
            continue
    return min(dates) if dates else None
