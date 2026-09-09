"""Import holdings or a watchlist from a spreadsheet, or from pasted text.

WHAT THIS EXISTS TO REPLACE: retyping forty tickers one at a time. So the
bar is not "it can parse a CSV" — pandas does that in one line — it is
"it survives what a broker actually exports", which is a different and
much messier thing. Everything below was measured against real file
shapes on 2026-09-09 rather than assumed.

WHAT WAS MEASURED, INCLUDING ONE THING THAT TURNED OUT NOT TO BE A
PROBLEM:

  - A BROKER PREAMBLE CRASHES THE PARSER OUTRIGHT. Schwab and Fidelity
    exports open with account name, generation date and a blank line
    before the real header. `pd.read_csv` on that raises
    "ParserError: Expected 1 fields in line 4, saw 3" — not a bad
    result, a traceback. `find_header_row` scans for the first row whose
    cell count matches the body instead.

  - EVERY NUMBER FORMAT A BROKER WRITES FAILS `float()`. Measured:
    "$1,234.56", "1,234.56", "(123.45)" (an accounting negative), "--",
    "N/A" — all raise. `parse_number` handles them, and returns None for
    the genuinely-absent ones rather than 0.0, per this app's standing
    rule that a fabricated number is worse than a gap.

  - A MIXED COLUMN COMES BACK AS TEXT. One "Total" row at the bottom of
    a Shares column makes pandas type the whole column `str` (measured
    on pandas 3.0.3), so per-cell coercion is the only safe reading.

  - THE UTF-8 BOM IS NOT A PROBLEM HERE, and I checked before writing
    code for it. Excel's "CSV UTF-8" writes a BOM that classically
    turns the first header into "﻿Symbol"; pandas 3.0.3 already
    strips it, giving 'Symbol' with no `encoding=` argument at all.
    Noted so nobody re-adds the workaround.

TWO AMBIGUITIES THAT MUST BE ASKED ABOUT, NEVER GUESSED. Both are silent
when wrong — they produce a plausible number, not an error.

  1. COST BASIS: PER SHARE OR TOTAL? Brokers export both under similar
     headings ("Cost Basis" is usually a total, "Cost/Share" per share).
     `Holding.cost_basis` is PER SHARE, so reading a total into it
     overstates the position by the share count — 10 shares at $150.25
     becomes a $15,025 cost instead of $1,502.50. `ColumnMapping.
     cost_is_total` carries the answer, guessed from the header wording
     and shown to the user as a control they can flip.

  2. DATE ORDER: 01/02/2024 IS JANUARY 2ND OR FEBRUARY 1ST. Measured
     both readings from the same string. But a column can usually settle
     itself: any value whose first part exceeds 12 forces day-first, any
     whose second part does forces month-first. `date_convention` infers
     from the whole column and returns UNDECIDABLE only when no row
     disambiguates — and then the UI asks rather than picking.

TICKERS ARE NORMALISED BUT NEVER "CORRECTED". Decoration comes off ($,
quotes, asterisks, a trailing dot, a trailing "(Apple Inc.)"), but the
INTERIOR of a symbol is left exactly as written, because a dot or hyphen
inside one is load-bearing: BRK.B is a share class, VWCE.DE is a
listing venue, and this codebase already learned that stripping a
venue suffix turns a working symbol into a 404. A row that does not look
like a ticker at all — "Cash & Cash Investments", "Account Total" — is
REPORTED as skipped with the reason, never silently dropped, because a
row that vanishes without explanation is how an import quietly loses a
position.

NOTHING IS WRITTEN UNTIL THE USER CONFIRMS. `build_preview` is pure and
does no I/O; it returns every row with a verdict attached, and the panel
renders that table. Applying is a separate call. An importer that writes
first and reports afterwards cannot be reviewed, and this one is aimed
at a store the user has spent real effort building.

LEGACY .xls IS NOT SUPPORTED AND SAYS SO. Reading it needs `xlrd`, which
is not installed here — measured: pandas raises ImportError naming it.
An unsupported format that fails with a stack trace teaches nothing, so
the suffix is refused up front with the fix.
"""
import datetime
import io
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import SPREADSHEET_IMPORT
from logging_setup import get_logger, log_exception

logger = get_logger("spreadsheet_import")


# The four things an import can populate. Watchlist needs only `ticker`;
# a portfolio needs all four.
ROLE_TICKER = "ticker"
ROLE_SHARES = "shares"
ROLE_COST = "cost_basis"
ROLE_DATE = "purchase_date"

ROLE_LABELS: Dict[str, str] = {
    ROLE_TICKER: "Ticker",
    ROLE_SHARES: "Shares",
    ROLE_COST: "Cost basis",
    ROLE_DATE: "Purchase date",
}

# Header synonyms, lowercased and stripped of punctuation before matching.
# Ordered longest-first within each role so "cost per share" wins over
# "cost" — the shorter one would otherwise claim the column and take the
# per-share/total question with it.
_HEADER_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    ROLE_TICKER: ("ticker symbol", "stock symbol", "security symbol", "symbol",
                  "ticker", "security", "instrument", "stock", "asset"),
    ROLE_SHARES: ("share quantity", "number of shares", "quantity owned",
                  "quantity", "shares", "units", "qty", "position", "amount held"),
    ROLE_COST: ("cost per share", "average cost per share", "cost basis per share",
                "avg cost per share", "price per share", "purchase price",
                "average cost", "avg cost", "unit cost", "cost share",
                "cost basis total", "total cost basis", "market value cost",
                "cost basis", "cost", "price paid", "price"),
    ROLE_DATE: ("date acquired", "acquisition date", "purchase date",
                "trade date", "buy date", "date bought", "opened", "acquired",
                "date"),
}

# A header matching one of these means the column holds a TOTAL, not a
# per-share figure. Checked against the raw header, because the
# distinction is a silent 10x error either way — see the module
# docstring.
_TOTAL_COST_MARKERS: Tuple[str, ...] = ("total", "market value", "book value")
_PER_SHARE_MARKERS: Tuple[str, ...] = ("per share", "/share", "per unit",
                                       "unit cost", "average cost", "avg cost",
                                       "price paid", "price per")

# Rows a broker adds that are not positions. Matched on the whole
# normalised cell, not as a substring, so a real ticker containing these
# letters is unaffected.
_NON_POSITION_ROWS: Tuple[str, ...] = (
    "cash", "cash & cash investments", "cash and cash investments",
    "account total", "total", "totals", "grand total", "subtotal",
    "pending activity", "money market", "sweep", "n/a", "--", "",
)

# Decoration that comes off a symbol. The INTERIOR is never touched —
# BRK.B and VWCE.DE must survive intact.
_TICKER_DECORATION = re.compile(r"""^[\s"'$*]+|[\s"'*.,]+$""")
_TICKER_TRAILING_NAME = re.compile(r"\s*\(.*\)\s*$")
# What a symbol may contain once decoration is off. Dots and hyphens are
# in here deliberately, and so is a LEADING caret: ^GSPC and ^TNX are
# index symbols this app uses as benchmarks and treasury yields, so a
# shape check that rejected them would refuse a legitimate watchlist row.
_TICKER_SHAPE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$")

_NUMBER_CLEAN = re.compile(r"[^\d.\-]")

DATE_DAY_FIRST = "day_first"
DATE_MONTH_FIRST = "month_first"
DATE_UNDECIDABLE = "undecidable"

XLS_UNSUPPORTED = (
    "Legacy .xls files need the `xlrd` package, which is not installed here. "
    "Open the file in Excel or Numbers and save it as .xlsx or .csv — both "
    "import fine."
)


@dataclass(frozen=True)
class ColumnMapping:
    """Which spreadsheet column feeds which field.

    `cost_is_total` is not a formatting detail: it decides whether the
    cost column is divided by the share count. Guessed from the header
    and then shown as a control, because guessing it silently is a 10x
    error in whichever direction the guess was wrong.
    """
    ticker: str = ""
    shares: str = ""
    cost_basis: str = ""
    purchase_date: str = ""
    cost_is_total: bool = False
    day_first: bool = False
    # Used ONLY when the file carries no date column at all — which is
    # the common case, not an edge one: a broker's positions export
    # lists what you hold today and says nothing about when you bought
    # it. Without this the most ordinary file in the world imports zero
    # rows. It is a fact the USER supplies, not one invented from the
    # data, and the panel states what it costs: performance is measured
    # from this date.
    #
    # A file that DOES have a date column and an unreadable cell is a
    # different situation and stays a problem — there the file claims a
    # date and we failed to read it, which is worth surfacing rather
    # than papering over.
    fallback_date: Optional[datetime.date] = None

    def column_for(self, role: str) -> str:
        return getattr(self, role, "") or ""


@dataclass(frozen=True)
class RowResult:
    """One spreadsheet row and what will happen to it.

    `status` is one of "ok", "skipped", "problem". The distinction
    matters on screen: "skipped" is a row that was never a position (a
    cash line, a totals row), while "problem" is a row that looks like
    one and could not be read. Collapsing them would hide a real loss
    inside a list of expected noise.
    """
    row_number: int
    ticker: str = ""
    shares: Optional[float] = None
    cost_basis: Optional[float] = None
    purchase_date: Optional[datetime.date] = None
    status: str = "ok"
    reason: str = ""
    raw_ticker: str = ""

    @property
    def importable(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class Preview:
    rows: Tuple[RowResult, ...] = ()
    warnings: Tuple[str, ...] = ()
    truncated: bool = False

    @property
    def importable(self) -> Tuple[RowResult, ...]:
        return tuple(r for r in self.rows if r.importable)

    @property
    def problems(self) -> Tuple[RowResult, ...]:
        return tuple(r for r in self.rows if r.status == "problem")

    @property
    def skipped(self) -> Tuple[RowResult, ...]:
        return tuple(r for r in self.rows if r.status == "skipped")

    def summary(self) -> str:
        """One line. Names the problems separately from the skips —
        see RowResult."""
        if not self.rows:
            return "Nothing to import from this file."
        parts = [f"{len(self.importable)} row(s) ready"]
        if self.problems:
            parts.append(f"{len(self.problems)} could not be read")
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped (not positions)")
        return " · ".join(parts) + "."


# --- reading ------------------------------------------------------------------

def _decode(data: bytes) -> Tuple[str, str]:
    """Bytes to text, trying the encodings a spreadsheet actually gets
    saved in. Returns (text, error).

    `chardet` is not installed and is not worth a dependency for this:
    utf-8 covers almost everything, cp1252 covers the Windows exports
    that are not utf-8, and latin-1 cannot fail, so the ladder always
    terminates with SOMETHING readable rather than an exception.
    """
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), ""
        except UnicodeDecodeError:
            continue
    return "", "That file is not text Quantix can read."


def find_header_row(text: str, sample_lines: int = 0) -> int:
    """Index of the line that looks like the header, skipping a preamble.

    THIS IS THE ONE THAT CRASHES WITHOUT IT. A broker export begins with
    account name, a generated-on date and a blank line; feeding that to
    pandas raises a ParserError about inconsistent field counts rather
    than returning anything.

    The header is taken to be the first line whose delimiter count
    matches the MODE of the lines after it — i.e. the first line shaped
    like the table's body.
    """
    sample_lines = sample_lines or SPREADSHEET_IMPORT.header_scan_lines
    lines = [ln for ln in text.splitlines()[:sample_lines]]
    counts = [ln.count(",") for ln in lines]
    populated = [c for c in counts if c > 0]
    if not populated:
        return 0
    # The body's shape: the most common non-zero field count.
    body_width = max(set(populated), key=populated.count)
    for index, count in enumerate(counts):
        if count == body_width:
            return index
    return 0


def read_table(data: bytes, filename: str = "") -> Tuple[Optional[pd.DataFrame], str]:
    """A DataFrame from an uploaded file, or an explanation.

    Never raises. Every failure returns a sentence a non-technical
    reader can act on, because "ParserError: Expected 1 fields" is not
    one.
    """
    name = (filename or "").lower().strip()
    if len(data or b"") > SPREADSHEET_IMPORT.max_upload_bytes:
        limit = SPREADSHEET_IMPORT.max_upload_bytes // (1024 * 1024)
        return None, f"That file is larger than {limit} MB."
    if not data:
        return None, "That file is empty."

    if name.endswith(".xls"):
        return None, XLS_UNSUPPORTED

    try:
        if name.endswith((".xlsx", ".xlsm")):
            frame = pd.read_excel(io.BytesIO(data))
        else:
            text, error = _decode(data)
            if error:
                return None, error
            skip = find_header_row(text)
            frame = pd.read_csv(io.StringIO(text), skiprows=skip,
                                skip_blank_lines=True)
    except Exception as exc:
        log_exception(logger, "spreadsheet_import.read_failed",
                      section="spreadsheet_import")
        return None, (
            "Quantix could not read that file as a table "
            f"({type(exc).__name__}). A plain CSV or .xlsx export works best."
        )

    if frame is None or frame.empty:
        return None, "That file has no rows under its header."
    frame = frame.rename(columns=lambda c: str(c).strip())
    return frame, ""


def read_pasted(text: str) -> Tuple[Optional[pd.DataFrame], str]:
    """A DataFrame from pasted text.

    Two shapes are accepted because both are what people actually paste:
    a bare list of symbols ("AAPL, MSFT, NVDA" or one per line), and a
    block copied straight out of a spreadsheet, which arrives
    TAB-separated. Sniffing the delimiter keeps that second case from
    landing as one column of junk.
    """
    text = (text or "").strip()
    if not text:
        return None, "Paste something first."

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if any("\t" in ln for ln in lines):
        try:
            frame = pd.read_csv(io.StringIO(text), sep="\t")
            return frame.rename(columns=lambda c: str(c).strip()), ""
        except Exception:
            pass

    # A bare ticker list — the Batch Watchlist Import case. Reuse the
    # watchlist panel's own splitting so pasted text and the sidebar box
    # accept exactly the same thing.
    symbols: List[str] = []
    for line in lines:
        for part in line.replace(",", " ").replace(";", " ").split():
            symbols.append(part)
    if not symbols:
        return None, "No symbols found in that text."
    return pd.DataFrame({"Ticker": symbols}), ""


# --- cell parsing -------------------------------------------------------------

def parse_number(value) -> Optional[float]:
    """A float from whatever a spreadsheet put in the cell, or None.

    Handles the measured set: "$1,234.56", "1,234.56", "(123.45)" as an
    accounting negative, "12.5%", "--", "N/A", "". Returns None rather
    than 0.0 for the absent ones — a zero is silently absorbed into any
    total the reader later builds, which is the same reasoning the
    Excel export follows for blank cells.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)

    text = str(value).strip()
    if not text or text.lower() in ("--", "-", "n/a", "na", "none", "nan", "—"):
        return None

    negative = text.startswith("(") and text.endswith(")")
    cleaned = _NUMBER_CLEAN.sub("", text)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -number if negative else number


def normalise_ticker(raw) -> str:
    """A symbol with decoration removed and NOTHING ELSE CHANGED.

    The interior is untouched on purpose: BRK.B is a share class and
    VWCE.DE is a listing venue, and this codebase has already learned
    that dropping a venue suffix turns a working symbol into a 404. So
    only leading/trailing junk goes — $, quotes, asterisks, a trailing
    dot, and a trailing "(Apple Inc.)".
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = _TICKER_TRAILING_NAME.sub("", text)
    text = _TICKER_DECORATION.sub("", text)
    return text.upper().strip()


def is_non_position_row(raw) -> bool:
    """True for the rows brokers add that were never positions."""
    text = str(raw or "").strip().lower()
    text = text.replace("&", "and")
    return text in _NON_POSITION_ROWS


def looks_like_ticker(symbol: str) -> bool:
    return bool(_TICKER_SHAPE.match(symbol or ""))


def date_convention(values: Sequence) -> str:
    """Whether a column of dates is day-first, month-first, or unknowable.

    A single unambiguous row settles the whole column: a first part above
    12 can only be a day, a second part above 12 can only be a day in the
    other position. Only when NO row disambiguates does this return
    UNDECIDABLE — and then the caller must ask, because 01/02/2024 really
    is two different dates and picking one silently moves a purchase by a
    month.
    """
    first_over_12 = second_over_12 = False
    for value in values:
        if isinstance(value, (datetime.date, datetime.datetime, pd.Timestamp)):
            continue
        text = str(value or "").strip()
        parts = re.split(r"[/\-.]", text)
        if len(parts) < 2 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        first, second = int(parts[0]), int(parts[1])
        if first > 12 and first <= 31:
            first_over_12 = True
        if second > 12 and second <= 31:
            second_over_12 = True
    if first_over_12 and not second_over_12:
        return DATE_DAY_FIRST
    if second_over_12 and not first_over_12:
        return DATE_MONTH_FIRST
    # Contradictory columns land here too — some rows day-first, some
    # month-first — and are just as unknowable.
    return DATE_UNDECIDABLE


def parse_date(value, day_first: bool = False) -> Optional[datetime.date]:
    """A date from a cell, or None. Never raises."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        # An Excel serial date, if it is in a plausible range. Excel's
        # epoch is 1899-12-30 (its deliberate 1900 leap-year bug).
        number = int(value)
        if 20000 <= number <= 60000:
            return datetime.date(1899, 12, 30) + datetime.timedelta(days=number)
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        stamp = pd.to_datetime(text, dayfirst=day_first, format="mixed")
    except Exception:
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return stamp.date()


# --- column mapping -----------------------------------------------------------

def _normalise_header(header: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(header or "").lower()).strip()


def guess_mapping(columns: Sequence[str], values: Optional[pd.DataFrame] = None
                  ) -> ColumnMapping:
    """Best guess at which column is which, from the header names.

    A guess, deliberately shown to the user as editable controls rather
    than applied silently — a wrong column here imports the wrong
    numbers with no error anywhere.
    """
    normalised = {str(c): _normalise_header(c) for c in columns}
    chosen: Dict[str, str] = {}
    taken: set = set()

    for role, synonyms in _HEADER_SYNONYMS.items():
        for synonym in synonyms:
            for column, header in normalised.items():
                if column in taken:
                    continue
                if header == synonym or synonym in header.split():
                    chosen[role] = column
                    taken.add(column)
                    break
            if role in chosen:
                break
        if role in chosen:
            continue
        # Second pass: substring, which is looser and therefore only
        # reached when no whole-word match existed.
        for synonym in synonyms:
            for column, header in normalised.items():
                if column in taken:
                    continue
                if synonym in header:
                    chosen[role] = column
                    taken.add(column)
                    break
            if role in chosen:
                break

    cost_column = chosen.get(ROLE_COST, "")
    mapping = ColumnMapping(
        ticker=chosen.get(ROLE_TICKER, ""),
        shares=chosen.get(ROLE_SHARES, ""),
        cost_basis=cost_column,
        purchase_date=chosen.get(ROLE_DATE, ""),
        cost_is_total=cost_is_total_guess(cost_column),
    )

    if values is not None and mapping.purchase_date:
        convention = date_convention(values[mapping.purchase_date].tolist())
        mapping = replace(mapping, day_first=(convention == DATE_DAY_FIRST))
    return mapping


def cost_is_total_guess(header: str) -> bool:
    """Whether a cost column holds a TOTAL rather than a per-share price.

    Per-share markers are checked FIRST, because "total cost basis per
    share" would otherwise match "total" and invert the reading. When
    neither marker appears — a bare "Cost Basis", which is the common
    case — the answer is total, since that is what the bare heading
    almost always means in a broker export. Either way the user sees the
    control and can flip it.
    """
    text = _normalise_header(header)
    if not text:
        return False
    if any(marker in text for marker in
           (m.replace("/", " ") for m in _PER_SHARE_MARKERS)):
        return False
    if any(marker in text for marker in _TOTAL_COST_MARKERS):
        return True
    return "cost basis" in text


# --- preview ------------------------------------------------------------------

def build_preview(frame: pd.DataFrame, mapping: ColumnMapping,
                  need_position: bool = False,
                  existing: Sequence[str] = ()) -> Preview:
    """Every row with a verdict. Pure — no I/O, nothing written.

    `need_position` is True for a portfolio import, where shares, cost
    and date are all required; a watchlist import needs only the ticker.
    `existing` are symbols already held, so a re-import of the same
    spreadsheet reports duplicates instead of doubling the position.
    """
    warnings: List[str] = []
    if not mapping.ticker:
        return Preview((), ("No column has been chosen for the ticker.",))
    if mapping.ticker not in frame.columns:
        return Preview((), (f"Column '{mapping.ticker}' is not in this file.",))

    limit = SPREADSHEET_IMPORT.max_rows
    working = frame.head(limit)
    truncated = len(frame) > limit
    if truncated:
        warnings.append(
            f"This file has {len(frame):,} rows; only the first {limit:,} were read.")

    seen: set = {str(t).upper() for t in existing}
    results: List[RowResult] = []

    for offset, (_, row) in enumerate(working.iterrows()):
        number = offset + 1
        raw = row.get(mapping.ticker)
        symbol = normalise_ticker(raw)

        if is_non_position_row(raw) or not symbol:
            results.append(RowResult(number, raw_ticker=str(raw or ""),
                                     status="skipped",
                                     reason="Not a position row."))
            continue
        if not looks_like_ticker(symbol):
            results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                     status="skipped",
                                     reason="Does not look like a ticker symbol."))
            continue
        if symbol in seen:
            results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                     status="skipped",
                                     reason="Already present — not imported twice."))
            continue

        shares = cost = None
        purchase = None
        if need_position:
            shares = parse_number(row.get(mapping.shares)) if mapping.shares else None
            cost = parse_number(row.get(mapping.cost_basis)) if mapping.cost_basis else None
            if mapping.purchase_date:
                purchase = parse_date(row.get(mapping.purchase_date),
                                      mapping.day_first)
            else:
                purchase = mapping.fallback_date

            if shares is None:
                results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                         status="problem",
                                         reason="No share count could be read."))
                continue
            if shares < 0:
                # A short position. add_holding refuses it, so importing
                # it would fail silently at apply time; say so here.
                results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                         shares=shares, status="problem",
                                         reason="Short position — Quantix tracks long positions only."))
                continue
            if shares == 0:
                results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                         shares=0.0, status="skipped",
                                         reason="Zero shares — a closed position."))
                continue
            if cost is None:
                results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                         shares=shares, status="problem",
                                         reason="No cost basis could be read."))
                continue
            if mapping.cost_is_total and shares:
                cost = cost / shares
            if purchase is None:
                reason = ("No purchase date could be read."
                          if mapping.purchase_date else
                          "This file has no date column — set a purchase "
                          "date to use for every row.")
                results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                         shares=shares, cost_basis=cost,
                                         status="problem", reason=reason))
                continue
            if purchase > datetime.date.today():
                results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                         shares=shares, cost_basis=cost,
                                         purchase_date=purchase, status="problem",
                                         reason="Purchase date is in the future."))
                continue

        seen.add(symbol)
        results.append(RowResult(number, ticker=symbol, raw_ticker=str(raw),
                                 shares=shares, cost_basis=cost,
                                 purchase_date=purchase, status="ok"))

    if need_position and mapping.purchase_date:
        convention = date_convention(working[mapping.purchase_date].tolist())
        if convention == DATE_UNDECIDABLE:
            warnings.append(
                "Every date in this file could be read either way round "
                "(01/02 is both 1 February and 2 January). Check the "
                "day/month setting below — nothing in the file settles it."
            )
    return Preview(tuple(results), tuple(warnings), truncated)


def fit_to_capacity(preview: Preview, room: int) -> Tuple[Tuple[RowResult, ...], str]:
    """Split the importable rows into what fits and a note about the rest.

    Truncating silently is the failure this prevents: someone importing
    forty tickers into a ten-slot watchlist must be told which thirty did
    not go, by name, rather than discovering the list is short later.
    """
    ready = preview.importable
    if room <= 0:
        return (), ("There is no room left — remove something first, or "
                    "import into a new list.")
    if len(ready) <= room:
        return ready, ""
    kept, dropped = ready[:room], ready[room:]
    names = ", ".join(r.ticker for r in dropped[:12])
    more = f" and {len(dropped) - 12} more" if len(dropped) > 12 else ""
    return kept, (
        f"Only {room} more would fit, so {len(dropped)} were left out: "
        f"{names}{more}."
    )
