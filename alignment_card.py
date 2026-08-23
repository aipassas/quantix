"""The alignment cards, and the hover tooltip that expands one.

WHY THESE CARDS STOPPED BEING st.info(). The brief asks for "hover to show
tooltip with EPS, Growth, etc." on the alignment cards. st.info in
Streamlit 1.58 takes (body, icon, width, title) — there is no help= and no
hover surface of any kind, so the tooltip is simply not expressible while
the card is a native alert. These cards are pure display: nothing about
them calls back into Streamlit, so replacing them with markup costs
nothing that a widget was providing.

THE EXTRA FIGURES ARE FREE. EPS, earnings growth, revenue growth, ROE,
dividend yield and market cap all come off the SAME shallow bundle the
card's score was already computed from — process_ticker_data fetches it
once per ticker and caches for an hour. The tooltip therefore adds no
network request; it surfaces numbers the page had already paid for and
was throwing away.

HOVER IS NOT THE ONLY WAY IN. The panel is keyboard-reachable
(tabindex="0") and opens on :focus-within as well as :hover, because a
tooltip that exists only under a mouse pointer hides these figures from
keyboard and touch users entirely. aria-describedby ties the panel to its
card so a screen reader reads the detail rather than a stray list of
numbers.

UNITS COME FROM quick_stats. StandardizedFinancials mixes fractions
(return_on_equity, earnings_growth) with already-percent values
(dividend_yield_pct), and this codebase has shipped that confusion more
than once. Reusing StatSpec/format_value means the conversion lives in one
place rather than being re-derived here — and inherits "Not reported"
for a missing figure instead of a fabricated 0.00.
"""
import html
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from quick_stats import NOT_REPORTED, StatSpec, format_value

# The rows of the hover panel, in display order. `kind` drives the unit
# conversion — see the module docstring; do not "simplify" a
# fraction_percent to a percent.
TIP_FIELDS: Tuple[StatSpec, ...] = (
    StatSpec("eps", "EPS (TTM)", "fundamental", "money"),
    StatSpec("earnings_growth", "Earnings growth", "fundamental", "fraction_percent"),
    StatSpec("revenue_growth", "Revenue growth", "fundamental", "fraction_percent"),
    StatSpec("return_on_equity", "ROE", "fundamental", "fraction_percent"),
    StatSpec("dividend_yield_pct", "Dividend yield", "fundamental", "percent"),
    StatSpec("market_cap", "Market cap", "fundamental", "money"),
)

# The rows on the face of the card — what st.info showed, unchanged. The
# brief asks to ADD a tooltip, not to rearrange what is already legible at
# rest.
FACE_FIELDS: Tuple[StatSpec, ...] = (
    StatSpec("pe", "P/E", "fundamental", "number", decimals=1),
    StatSpec("margin", "Margin", "fundamental", "percent", decimals=1),
)


def _slug(ticker: str) -> str:
    """A DOM-id-safe form of the ticker.

    Tickers reach this from the user's own watchlist text box, so they are
    not guaranteed to be bare letters — an id is interpolated into markup
    and must not be able to carry anything else.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "-", (ticker or "").strip()) or "unknown"


def tip_rows(data: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """(label, formatted value) for the hover panel.

    Every field is emitted even when absent, rendering as "Not reported".
    Dropping missing rows would make two cards show different shapes and
    invite the reader to assume the figure was zero rather than unknown.
    """
    return tuple((spec.label, format_value(spec, data.get(spec.key)))
                 for spec in TIP_FIELDS)


def face_rows(data: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple((spec.label, format_value(spec, data.get(spec.key)))
                 for spec in FACE_FIELDS)


def alignment_line(data: Mapping[str, Any]) -> str:
    """"Strong (86%)" — or just the status when there is no score."""
    status = str(data.get("status") or "").strip() or NOT_REPORTED
    score = data.get("score")
    try:
        return f"{status} ({float(score):.0f}%)"
    except (TypeError, ValueError):
        return status


def card_html(data: Mapping[str, Any]) -> str:
    """One alignment card, with its hover/focus detail panel."""
    ticker = str(data.get("ticker") or "").strip() or "—"
    tip_id = f"qx-tip-{_slug(ticker)}"

    face = "".join(
        f'<div class="qx-acard-row"><span class="qx-acard-k">{html.escape(label)}</span>'
        f'<span class="qx-acard-v">{html.escape(value)}</span></div>'
        for label, value in
        (("Alignment", alignment_line(data)),) + face_rows(data)
    )
    tip = "".join(
        f'<div class="qx-acard-row"><span class="qx-acard-k">{html.escape(label)}</span>'
        f'<span class="qx-acard-v">{html.escape(value)}</span></div>'
        for label, value in tip_rows(data)
    )

    return (
        f'<div class="qx-acard" tabindex="0" aria-describedby="{tip_id}">'
        f'<div class="qx-acard-head">{html.escape(ticker)}</div>'
        f'{face}'
        f'<div class="qx-acard-more">Hover for detail</div>'
        f'<div class="qx-acard-tip" id="{tip_id}" role="tooltip">'
        f'<div class="qx-acard-tiphead">{html.escape(ticker)} · detail</div>'
        f'{tip}</div>'
        f'</div>'
    )


def css(palette, accent: str) -> str:
    """The card's styles, including the 200ms transitions the brief asks for.

    The overflow rule matters as much as the tooltip's own styling:
    Streamlit's column and element wrappers clip their children, so an
    absolutely-positioned panel is invisible below the card's own bottom
    edge until those ancestors are opened up. :has() scopes that to the
    blocks that actually contain a card rather than unclipping the page.
    """
    return f"""
    <style>
    [data-testid="stElementContainer"]:has(.qx-acard),
    [data-testid="stVerticalBlock"]:has(.qx-acard),
    [data-testid="stColumn"]:has(.qx-acard),
    [data-testid="stHorizontalBlock"]:has(.qx-acard) {{
        overflow: visible !important;
    }}

    .qx-acard {{
        position: relative;
        background: {palette.card_bg};
        border: 1px solid {palette.card_border};
        border-left: 3px solid {accent};
        border-radius: 6px;
        padding: 14px 16px;
        outline: none;
        cursor: default;
        transition: background-color 200ms ease, border-color 200ms ease,
                    transform 200ms ease, box-shadow 200ms ease;
    }}
    .qx-acard:hover, .qx-acard:focus-visible {{
        background: {palette.tab_selected_bg};
        border-color: {accent};
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
    }}
    .qx-acard-head {{
        color: {palette.header_text};
        font-weight: 700; font-size: 1.15rem; letter-spacing: 0.02em;
        margin-bottom: 8px;
    }}
    .qx-acard-row {{
        display: flex; justify-content: space-between; gap: 12px;
        font-size: 0.92rem; line-height: 1.65;
    }}
    .qx-acard-tip .qx-acard-k, .qx-acard-tip .qx-acard-v {{
        white-space: nowrap;
    }}
    .qx-acard-k {{ color: {palette.metric_label}; }}
    .qx-acard-v {{ color: {palette.app_text}; font-variant-numeric: tabular-nums; }}
    .qx-acard-more {{
        margin-top: 8px; font-size: 0.75rem; letter-spacing: 0.04em;
        text-transform: uppercase; color: {palette.symbol_header_meta};
        transition: color 200ms ease;
    }}
    .qx-acard:hover .qx-acard-more, .qx-acard:focus-visible .qx-acard-more {{
        color: {accent};
    }}

    .qx-acard-tip {{
        position: absolute; left: 0; top: calc(100% + 8px);
        /* Sized to its content, not to the card. The columns are narrow
           enough that "Earnings growth" wrapped under its own value and
           read as a rendering fault; the panel is an overlay, so it is
           free to be wider than the card that opens it. */
        width: max-content;
        min-width: 100%;
        max-width: min(300px, calc(100vw - 2rem));
        background: {palette.card_bg};
        border: 1px solid {accent};
        border-radius: 6px;
        padding: 12px 14px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.6);
        z-index: 60;
        opacity: 0;
        transform: translateY(-6px);
        pointer-events: none;
        transition: opacity 200ms ease, transform 200ms ease;
    }}
    .qx-acard:hover .qx-acard-tip, .qx-acard:focus-within .qx-acard-tip {{
        opacity: 1;
        transform: translateY(0);
    }}
    /* The rightmost column has no room to the right, so its panel hangs
       from the other edge rather than off the page. */
    [data-testid="stColumn"]:last-of-type .qx-acard-tip {{
        left: auto; right: 0;
    }}
    .qx-acard-tiphead {{
        color: {accent}; font-weight: 700; font-size: 0.78rem;
        letter-spacing: 0.06em; text-transform: uppercase;
        margin-bottom: 6px;
    }}

    /* Motion is decoration here; the colour and the panel still do the
       work without it. */
    @media (prefers-reduced-motion: reduce) {{
        .qx-acard, .qx-acard-tip, .qx-acard-more {{
            transition-duration: 1ms !important;
        }}
        .qx-acard:hover, .qx-acard:focus-visible {{ transform: none !important; }}
        .qx-acard-tip {{ transform: none !important; }}
    }}
    </style>
    """
