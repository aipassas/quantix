"""Turn a spoken or typed sentence into screener criteria.

WHAT THIS IS AND IS NOT. It maps a bounded English grammar onto
`screener.ScreenCriterion` — the SAME criteria the Screener's filter
engine already runs. No new query logic, no model, no API: nineteen
metrics, six operators and eleven sectors make a closed vocabulary, and
a closed vocabulary is parseable exactly rather than approximately. A
language model would be less predictable here, not more capable.

THE FAILURE THAT MATTERS IS THE SILENT ONE. "Show me cheap tech stocks
with high ROE" contains one thing this can screen on (Technology) and
two it cannot: "cheap" names no metric, and "high ROE" names a metric
with no threshold to compare against. Returning just the sector filter
would run a screen the reader believes is narrower than it is, and hand
back a list they would read as cheap and high-return. So every span
that did not become a criterion is REPORTED, and a query that yielded
nothing usable says so rather than screening the whole universe.

That is the same rule the rest of this app follows for missing data: an
absent answer is stated, never rendered as a permissive default.

NUMBERS ARE REQUIRED, COMPARATIVES ARE NOT ENOUGH. "high", "low",
"strong", "good" are deliberately unmapped. There is no defensible
threshold for a high ROE — 15% is a different bar in utilities than in
software, and this app already refuses to invent bounds that fail on
correct data. A comparative without a number is reported as needing
one.

ON DICTATION. The speech half is browser-side and optional: the query
box works typed. Availability is DETECTED at runtime rather than
assumed, because a microphone button that is advertised and never fires
is worse than none — the same rule the keyboard shortcuts follow. And
the browser's speech API sends captured audio to the browser vendor's
servers, which is a departure from "analysis runs on this machine" and
is disclosed where the button is, not buried here.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import screener

# --- vocabulary ---------------------------------------------------------------

# Spoken forms for each metric, longest matched first so "price to book"
# never loses to "price". Every key must exist in screener.METRICS_BY_KEY;
# a test enforces that, because a synonym pointing at a renamed metric
# would silently stop matching rather than fail.
METRIC_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "pe_ratio": ("p/e ratio", "pe ratio", "price to earnings", "price earnings",
                 "p e ratio", "p/e", "pe"),
    "peg_ratio": ("peg ratio", "peg"),
    "price_to_book": ("price to book", "price/book", "p/b ratio", "p/b",
                      "book value ratio"),
    "net_margin_pct": ("net profit margin", "net margin", "profit margin",
                       "margins", "margin"),
    "roe_pct": ("return on equity", "roe"),
    "debt_to_equity": ("debt to equity", "debt/equity", "d/e ratio", "d/e",
                       "debt equity"),
    "current_ratio": ("current ratio",),
    "beta": ("beta",),
    "rsi": ("rsi", "relative strength index"),
    "sharpe_ratio": ("sharpe ratio", "sharpe"),
    "sortino_ratio": ("sortino ratio", "sortino"),
    "annual_volatility_pct": ("annualized volatility", "annualised volatility",
                              "volatility", "vol"),
    "max_drawdown_pct": ("maximum drawdown", "max drawdown", "drawdown"),
    "altman_z": ("altman z-score", "altman z score", "altman z", "altman",
                 "z-score", "z score"),
    "price": ("share price", "stock price", "price"),
    "dividend_yield_pct": ("dividend yield", "dividend", "yield"),
    "revenue_growth_pct": ("revenue growth", "sales growth", "top line growth",
                           "revenue"),
    "sector": ("sector", "industry"),
}

# Spoken forms for each sector. The Yahoo labels are the target; these
# are what people actually say.
SECTOR_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "Technology": ("technology", "tech", "software", "it"),
    "Healthcare": ("healthcare", "health care", "health", "pharma",
                   "pharmaceutical", "biotech"),
    "Financial Services": ("financial services", "financials", "financial",
                           "finance", "banks", "banking", "bank"),
    "Energy": ("energy", "oil and gas", "oil & gas", "oil"),
    "Utilities": ("utilities", "utility"),
    "Industrials": ("industrials", "industrial"),
    "Real Estate": ("real estate", "reits", "reit", "property"),
    "Basic Materials": ("basic materials", "materials", "mining"),
    "Communication Services": ("communication services", "communications",
                               "communication", "telecom",
                               "telecommunications", "media"),
    "Consumer Cyclical": ("consumer cyclical", "consumer discretionary",
                          "discretionary", "cyclicals"),
    "Consumer Defensive": ("consumer defensive", "consumer staples",
                           "staples", "defensive"),
}

# Comparison words. Longest first for the same reason as the metrics:
# "no more than" must beat "more than", which inverts the operator.
OPERATOR_SYNONYMS: Tuple[Tuple[str, str], ...] = (
    ("no less than", ">="), ("no more than", "<="),
    ("greater than or equal to", ">="), ("less than or equal to", "<="),
    ("at least", ">="), ("at most", "<="),
    ("greater than", ">"), ("less than", "<"),
    ("higher than", ">"), ("lower than", "<"),
    ("more than", ">"), ("fewer than", "<"),
    ("bigger than", ">"), ("smaller than", "<"),
    ("above", ">"), ("below", "<"),
    ("over", ">"), ("under", "<"),
    ("exceeding", ">"), ("exceeds", ">"),
    ("minimum", ">="), ("maximum", "<="),
    ("up to", "<="),
    (">=", ">="), ("<=", "<="), (">", ">"), ("<", "<"),
)

# Words that ask for a comparison without giving a number. Recognised so
# the reader is told a threshold is missing, rather than the phrase
# being silently dropped.
COMPARATIVE_WITHOUT_NUMBER: Tuple[str, ...] = (
    "high", "higher", "highest", "low", "lower", "lowest", "strong",
    "strongest", "weak", "good", "great", "best", "worst", "cheap",
    "cheapest", "expensive", "undervalued", "overvalued", "profitable",
    "safe", "risky", "solid", "healthy", "attractive", "reasonable",
)

# Negation for a categorical metric.
NEGATION_WORDS: Tuple[str, ...] = ("not", "excluding", "exclude", "except",
                                   "other than", "outside")

# Words carrying no filter. Stripped only when deciding whether anything
# was left unparsed, never before matching.
FILLER_WORDS: Tuple[str, ...] = (
    "show", "me", "find", "get", "list", "search", "for", "give", "a",
    "an", "the", "please", "with", "and", "that", "have", "having", "has",
    "stocks", "stock", "shares", "companies", "company", "names", "all",
    "any", "of", "in", "is", "are", "which", "where", "whose", "i", "want",
    "looking", "look", "to", "at", "from", "on", "up", "it", "their",
    "there", "some", "only", "please", "can", "you", "us", "%", "percent",
    "dollars", "dollar", "x", "ratio", "ratios",
)

SPEECH_SENDS_AUDIO_OFF_DEVICE = (
    "Dictation uses your browser's own speech recognition, which sends "
    "the captured audio to the browser vendor's servers to transcribe. "
    "Everything else in Quantix runs on this machine — this one step "
    "does not. Typing the same query keeps it local."
)

NO_CRITERIA_FOUND = (
    "Nothing in that could be turned into a filter. Name a metric and a "
    "number — \"P/E under 20\", \"ROE above 15%\" — or a sector, like "
    "\"technology\"."
)


# --- parse results ------------------------------------------------------------

@dataclass(frozen=True)
class ParsedCriterion:
    metric: str
    operator: str
    threshold: object
    source: str = ""            # the span of the query it came from

    def to_screen_criterion(self) -> "screener.ScreenCriterion":
        return screener.ScreenCriterion(self.metric, self.operator,
                                        self.threshold)

    @property
    def text(self) -> str:
        return screener.criterion_text(self.metric, self.operator,
                                       self.threshold)


@dataclass(frozen=True)
class Unparsed:
    span: str
    reason: str


@dataclass(frozen=True)
class ParseResult:
    query: str = ""
    criteria: Tuple[ParsedCriterion, ...] = ()
    unparsed: Tuple[Unparsed, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.criteria)

    @property
    def complete(self) -> bool:
        """Everything meaningful in the query became a criterion."""
        return self.ok and not self.unparsed

    def screen_criteria(self) -> List["screener.ScreenCriterion"]:
        return [c.to_screen_criterion() for c in self.criteria]

    @property
    def summary(self) -> str:
        if not self.criteria:
            return NO_CRITERIA_FOUND
        text = "Screening for " + " and ".join(c.text for c in self.criteria)
        if self.unparsed:
            text += (". Not used: "
                     + "; ".join(f"\"{u.span}\" — {u.reason}"
                                 for u in self.unparsed))
        return text + "."


# --- parsing ------------------------------------------------------------------

_NUMBER = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*(%|percent|dollars?)?", re.IGNORECASE)

# Spoken numbers the recogniser sometimes returns as words.
_WORD_NUMBERS: Dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty five": 25, "thirty": 30, "forty": 40, "fifty": 50,
    "one hundred": 100, "hundred": 100,
}


def _normalise(text: str) -> str:
    """Lower-cased, with the punctuation a recogniser sprinkles in
    removed — but NOT the characters that carry meaning here.

    A speech transcript arrives as "show me tech stocks with ROE over
    15%." and a typed one as "roe>15". Both have to land in the same
    place, so comparison symbols and the decimal point survive and
    sentence punctuation does not.
    """
    text = str(text or "").strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"[.,;!?]+(?=\s|$)", " ", text)
    text = re.sub(r"[()\"']", " ", text)
    # Keep >, <, =, %, $, /, - and word characters; drop the rest.
    text = re.sub(r"[^\w\s><=%$/\-.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _longest_first(pairs) -> List[Tuple[str, object]]:
    return sorted(pairs, key=lambda kv: -len(kv[0]))


_METRIC_LOOKUP: List[Tuple[str, str]] = _longest_first(
    [(syn, key) for key, syns in METRIC_SYNONYMS.items() for syn in syns])
_SECTOR_LOOKUP: List[Tuple[str, str]] = _longest_first(
    [(syn, name) for name, syns in SECTOR_SYNONYMS.items() for syn in syns])
_OPERATOR_LOOKUP: List[Tuple[str, str]] = _longest_first(
    list(OPERATOR_SYNONYMS))


def _find_number(text: str) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
    """The first number in `text`, and where it sat.

    A percent sign is consumed but does NOT scale the value: every
    percent-valued metric in this app is stored percent-valued, so
    "15%" is 15.0. Dividing by a hundred here is the same 100x error the
    Excel and Streamlit percent formats produce, from the other
    direction.
    """
    match = _NUMBER.search(text)
    if match:
        raw = match.group(1).replace(",", ".")
        try:
            return float(raw), match.span()
        except ValueError:
            pass
    for word, value in _longest_first(list(_WORD_NUMBERS.items())):
        index = text.find(word)
        if index >= 0:
            return float(value), (index, index + len(word))
    return None, None


def _match_at(text: str, lookup: Sequence[Tuple[str, object]]
              ) -> List[Tuple[int, int, object, str]]:
    """Every non-overlapping vocabulary hit, longest first.

    Word-boundary anchored so "vol" does not fire inside "volatility"
    and "it" does not fire inside "with" — the second of which turned
    every query containing "with" into a Technology screen.
    """
    hits: List[Tuple[int, int, object, str]] = []
    taken: List[Tuple[int, int]] = []
    for phrase, value in lookup:
        for match in re.finditer(
                r"(?<![\w/])" + re.escape(phrase) + r"(?![\w/])", text):
            start, end = match.span()
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            hits.append((start, end, value, match.group(0)))
    hits.sort(key=lambda h: h[0])
    return hits


def parse(query: str) -> ParseResult:
    """A sentence into criteria, plus whatever could not be used.

    Works left to right over the metric mentions: each metric claims the
    text up to the next metric, and its operator and number are looked
    for inside that window. Without the window, "P/E under 20 and ROE
    above 15" lets the first metric grab the last number.
    """
    original = str(query or "").strip()
    text = _normalise(original)
    if not text:
        return ParseResult(original)

    criteria: List[ParsedCriterion] = []
    unparsed: List[Unparsed] = []
    consumed: List[Tuple[int, int]] = []

    def claim(start: int, end: int) -> None:
        consumed.append((start, end))

    # --- sectors, which need no number ---------------------------------
    metric_hits = _match_at(text, _METRIC_LOOKUP)
    sector_hits = _match_at(text, _SECTOR_LOOKUP)
    # A word that is BOTH a sector name and a metric ("energy" is only a
    # sector; "real estate" only a sector) never collides, but "sector"
    # itself is a metric synonym and must not eat the sector name after
    # it.
    metric_spans = [(s, e) for s, e, _, _ in metric_hits]
    for start, end, sector_name, span_text in sector_hits:
        # Skip a sector word sitting inside a metric phrase.
        if any(start >= m_start and end <= m_end
               for m_start, m_end in metric_spans):
            continue
        window_start = max(0, start - 30)
        preceding = text[window_start:start]
        negated = any(word in preceding for word in NEGATION_WORDS)
        criteria.append(ParsedCriterion(
            "sector", "is not" if negated else "is", sector_name, span_text))
        claim(start, end)
        for word in NEGATION_WORDS:
            index = preceding.rfind(word)
            if index >= 0:
                claim(window_start + index, window_start + index + len(word))

    # --- numeric metrics ------------------------------------------------
    numeric_hits = [h for h in metric_hits if h[2] != "sector"]
    for position, (start, end, metric_key, span_text) in enumerate(numeric_hits):
        window_end = (numeric_hits[position + 1][0]
                      if position + 1 < len(numeric_hits) else len(text))
        window = text[end:window_end]

        operator = None
        operator_span = None
        for phrase, symbol in _OPERATOR_LOOKUP:
            match = re.search(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])",
                              window)
            if match:
                operator, operator_span = symbol, match.span()
                break

        value, value_span = _find_number(window)

        if value is None:
            lead_start = max(0, start - 25)
            comparative = None
            comparative_span = None
            for word in COMPARATIVE_WITHOUT_NUMBER:
                match = re.search(r"(?<![\w])" + word + r"(?![\w])",
                                  text[lead_start:start])
                if match:
                    comparative = word
                    comparative_span = (lead_start + match.start(),
                                        lead_start + match.end())
                    break
            if comparative is None:
                for word in COMPARATIVE_WITHOUT_NUMBER:
                    match = re.search(r"(?<![\w])" + word + r"(?![\w])",
                                      window)
                    if match:
                        comparative = word
                        comparative_span = (end + match.start(),
                                            end + match.end())
                        break
            spec = screener.METRICS_BY_KEY.get(metric_key)
            label = spec.label if spec else metric_key
            if comparative:
                # The hint shows the SHAPE, not a threshold. Suggesting
                # "Debt/Equity above 15" would be nonsense for a ratio
                # that runs under 2, and picking a sensible number per
                # metric would mean inventing the bounds this app
                # refuses to invent.
                unparsed.append(Unparsed(
                    f"{comparative} {span_text}".strip(),
                    f"\"{comparative}\" is not a threshold — give a "
                    f"number, as in \"{label} above <n>\" or "
                    f"\"{label} below <n>\""))
            else:
                unparsed.append(Unparsed(
                    span_text, f"no threshold was given for {label}"))
            claim(start, end)
            # Claim the comparative too, or the leftover scan reports
            # "high" a second time on its own.
            if comparative_span:
                claim(*comparative_span)
            continue

        if operator is None:
            operator = ">"      # "ROE 15%" reads as a floor in practice
        criteria.append(ParsedCriterion(metric_key, operator, value,
                                        span_text))
        claim(start, end)
        if operator_span:
            claim(end + operator_span[0], end + operator_span[1])
        if value_span:
            claim(end + value_span[0], end + value_span[1])

    # --- what was left --------------------------------------------------
    leftover = []
    cursor = 0
    for start, end in sorted(consumed):
        if start > cursor:
            leftover.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        leftover.append(text[cursor:])

    for chunk in leftover:
        words = [w for w in chunk.split()
                 if w and w not in FILLER_WORDS
                 and not re.fullmatch(r"[\d.,%$><=\-]+", w)]
        if not words:
            continue
        phrase = " ".join(words)
        if all(w in COMPARATIVE_WITHOUT_NUMBER for w in words):
            unparsed.append(Unparsed(
                phrase, "names no metric this screener measures"))
        else:
            unparsed.append(Unparsed(phrase, "not recognised"))

    return ParseResult(original, tuple(criteria), tuple(unparsed))


# --- examples, for the empty state --------------------------------------------

EXAMPLE_QUERIES: Tuple[str, ...] = (
    "Show me tech stocks with ROE above 15%",
    "P/E under 20 and dividend yield above 3%",
    "Healthcare companies with revenue growth over 10%",
    "Financials with debt to equity below 1 and price under $100",
    "Sharpe ratio above 1 excluding energy",
)


def describe_vocabulary() -> str:
    """What the parser can actually hear, for the help text.

    Listed rather than summarised because a user guessing at the
    vocabulary is the main way this feature disappoints: they say
    something reasonable, nothing matches, and the panel looks broken.
    """
    metrics = ", ".join(
        screener.METRICS_BY_KEY[key].label
        for key in METRIC_SYNONYMS if key in screener.METRICS_BY_KEY)
    return (f"Metrics: {metrics}. Sectors: "
            f"{', '.join(SECTOR_SYNONYMS)}. Comparisons: above, below, "
            f"at least, at most, over, under — each needs a number.")


# --- the dictation component --------------------------------------------------
#
# Runs inside a st.components.v1.html iframe. Two facts make this work,
# both measured rather than assumed:
#
#   - Streamlit's component iframe carries "microphone" in its
#     permissions-policy `allow` attribute and `allow-same-origin` in
#     its sandbox, so getUserMedia and the speech API are permitted
#     there. Probed on the running app.
#   - The iframe can reach window.parent.document, which is how it hands
#     the transcript back: there is no other channel from a plain HTML
#     component to Python. It writes into the query box and clicks the
#     apply button, exactly as keyboard_shortcuts.py drives tabs.
#
# AVAILABILITY IS DETECTED, NEVER ASSUMED. Chromium builds without the
# speech backend, a denied microphone, and Firefox (no webkitSpeechRecognition
# at all) each fail differently, and a button that is advertised and
# never fires is worse than none. The component renders its own state:
# a working mic, or the reason there isn't one, with the typed box still
# working either way.

_COMPONENT_HEIGHT = 78

_LISTENER_TEMPLATE = """
<style>
  .vs-wrap { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
             sans-serif; display: flex; align-items: center; gap: 10px; }
  .vs-btn { border: 1px solid %(border)s; background: %(bg)s;
            color: %(fg)s; border-radius: 8px; padding: 8px 14px;
            font-size: 14px; cursor: pointer; display: flex;
            align-items: center; gap: 8px; }
  .vs-btn:disabled { opacity: .5; cursor: not-allowed; }
  .vs-btn.listening { border-color: %(accent)s; color: %(accent)s; }
  .vs-dot { width: 9px; height: 9px; border-radius: 50%%;
            background: %(fg)s; }
  .vs-btn.listening .vs-dot { background: %(accent)s;
                              animation: vs-pulse 1s infinite; }
  @keyframes vs-pulse { 0%%,100%% { opacity: 1 } 50%% { opacity: .25 } }
  .vs-status { font-size: 12px; color: %(muted)s; }
</style>
<div class="vs-wrap">
  <button class="vs-btn" id="vs-btn"><span class="vs-dot"></span>
    <span id="vs-label">Dictate</span></button>
  <span class="vs-status" id="vs-status"></span>
</div>
<script>
(function () {
  const QUERY_KEY = %(query_key)s;
  const APPLY_KEY = %(apply_key)s;
  const btn = document.getElementById("vs-btn");
  const label = document.getElementById("vs-label");
  const status = document.getElementById("vs-status");

  function say(text) { status.textContent = text; }
  function disable(text) { btn.disabled = true; say(text); }

  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    disable("This browser has no speech recognition — type the query instead.");
    return;
  }
  if (!window.isSecureContext) {
    disable("Dictation needs a secure context (https or localhost).");
    return;
  }

  let pdoc = null;
  try { pdoc = window.parent.document; } catch (err) { pdoc = null; }
  if (!pdoc) {
    disable("Can't reach the page to deliver the transcript.");
    return;
  }

  // The query box and the apply button, found by their Streamlit keys.
  function parentInput() {
    const host = pdoc.querySelector('[class*="st-key-' + QUERY_KEY + '"]');
    return host ? host.querySelector("input, textarea") : null;
  }
  function parentApply() {
    const host = pdoc.querySelector('[class*="st-key-' + APPLY_KEY + '"]');
    return host ? host.querySelector("button") : null;
  }

  // React tracks the input's value on the DOM node, so assigning .value
  // directly is invisible to it. The native setter plus an input event
  // is what Streamlit actually notices.
  function deliver(text) {
    const input = parentInput();
    if (!input) { say("Couldn't find the query box."); return; }
    const proto = input.tagName === "TEXTAREA"
      ? window.parent.HTMLTextAreaElement.prototype
      : window.parent.HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    setter.call(input, text);
    input.dispatchEvent(new window.parent.Event("input", { bubbles: true }));
    input.dispatchEvent(new window.parent.Event("change", { bubbles: true }));
    const apply = parentApply();
    if (apply) { apply.click(); }
    else { say("Transcribed — press Search."); }
  }

  // Ask the browser up front rather than making the reader press a
  // button to discover it is refused. Chrome answers this; Safari does
  // not implement the query for microphone, in which case the click
  // path's onerror still reports it.
  if (navigator.permissions && navigator.permissions.query) {
    navigator.permissions.query({ name: "microphone" }).then(function (p) {
      if (p.state === "denied") {
        disable("Microphone access is blocked for this site — allow it in "
                + "the browser, or type the query.");
      }
    }).catch(function () { /* query unsupported; the click path reports */ });
  }

  const recognition = new Recognition();
  recognition.lang = %(lang)s;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  // Single-utterance: a continuous stream would keep the microphone
  // open across reruns, and this iframe is destroyed on every one.
  recognition.continuous = false;

  let listening = false;
  function stop() {
    listening = false;
    btn.classList.remove("listening");
    label.textContent = "Dictate";
  }

  recognition.onstart = function () {
    listening = true;
    btn.classList.add("listening");
    label.textContent = "Listening…";
    say("Speak your screen — say \\u201cstop\\u201d style phrasing is not needed.");
  };
  recognition.onresult = function (event) {
    const said = event.results[0][0].transcript;
    say("Heard: \\u201c" + said + "\\u201d");
    deliver(said);
  };
  recognition.onerror = function (event) {
    stop();
    const map = {
      "not-allowed": "Microphone permission was refused — allow it in the "
                   + "browser, or type the query.",
      "service-not-allowed": "This browser blocked its speech service — "
                           + "type the query instead.",
      "no-speech": "Nothing was heard. Try again, or type the query.",
      "audio-capture": "No microphone was found — type the query instead.",
      "network": "The browser's speech service could not be reached — "
               + "type the query instead."
    };
    say(map[event.error] || ("Dictation failed (" + event.error
                             + ") — type the query instead."));
  };
  recognition.onend = stop;

  btn.addEventListener("click", function () {
    if (listening) { recognition.stop(); return; }
    say("");
    try { recognition.start(); }
    catch (err) { say("Could not start dictation — type the query instead."); }
  });
})();
</script>
"""


def listener_html(query_key: str = "voice_query",
                  apply_key: str = "voice_apply",
                  lang: str = "en-US",
                  palette=None) -> str:
    """The dictation button, as a self-contained component document.

    `query_key` and `apply_key` are the Streamlit keys of the text box
    the transcript is written into and the button clicked afterwards —
    the only channel a plain HTML component has back to Python.
    """
    import json

    def _colour(name: str, fallback: str) -> str:
        return str(getattr(palette, name, fallback) or fallback)

    return _LISTENER_TEMPLATE % {
        "query_key": json.dumps(str(query_key)),
        "apply_key": json.dumps(str(apply_key)),
        "lang": json.dumps(str(lang)),
        "bg": _colour("button_bg", "#111111"),
        "fg": _colour("button_text", "#e6e6e6"),
        "border": _colour("button_border", "#333333"),
        "muted": _colour("metric_label", "#8a8a8a"),
        "accent": _colour("card_accent", "#22d3ee"),
    }


def component_height() -> int:
    return _COMPONENT_HEIGHT
