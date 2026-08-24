"""Skeletons, a live-data pulse, and honest progress.

WHERE A SKELETON IS REAL AND WHERE IT IS THEATRE. Streamlit executes the
script top to bottom, so content below the current line does not exist
yet — there is nothing to put a placeholder in front of. A skeleton is
only meaningful where the app has RESERVED a slot and fills it later, and
this app does that in exactly two places: the symbol header (reserved at
the top, filled after the ticker bundle loads ~10 seconds in) and the
Executive Digest (reserved early, filled once every signal it synthesises
has been computed). Those two sit visibly empty while the page works,
which is the gap worth covering. Sprinkling skeletons anywhere else would
be decoration in front of content that was never going to be late.

A RESERVED SLOT HAS TO BE st.empty(), NOT st.container(). A container
APPENDS, so a skeleton written into one stays on screen underneath the
real content forever. st.empty() holds one thing and replaces it, and
calling .container() on it gives a replaceable slot that can hold several
elements — which is what both of these need.

THE PULSE MEANS "THIS IS STILL CHECKING", not "something happened". It
marks the two panels that genuinely re-run on a timer (the real-time
alert engine and the quick-stats strip). A pulse anywhere else would
imply a liveness the app does not have — it is a stateless script with no
background worker, and every other number on the page is as old as the
last rerun.

MOTION IS OPTIONAL. Everything here honours prefers-reduced-motion: the
shimmer stops and the pulse becomes a static dot, both of which still
say what they need to say.
"""
from typing import Optional, Sequence

# Widths of the bars in a default skeleton, as percentages. Uneven on
# purpose: equal bars read as a table, ragged ones read as prose that has
# not arrived yet.
DEFAULT_ROWS: Sequence[int] = (35, 70, 55)

SHIMMER_SECONDS = 1.4
PULSE_SECONDS = 2.0


def css(palette) -> str:
    """One stylesheet for both effects. Injected once."""
    base = getattr(palette, "card_bg", "#0a0a0a")
    edge = getattr(palette, "card_border", "#1a1a1a")
    glow = getattr(palette, "tab_selected_bg", "#161616")
    accent = getattr(palette, "card_accent", "#00ea77")
    return f"""
    <style>
    .qx-skeleton {{
        border: 1px solid {edge};
        border-radius: 8px;
        padding: 16px 18px;
        background: {base};
    }}
    .qx-skeleton-bar {{
        height: 12px;
        border-radius: 6px;
        margin: 10px 0;
        background: linear-gradient(90deg, {edge} 25%, {glow} 37%, {edge} 63%);
        background-size: 400% 100%;
        animation: qx-shimmer {SHIMMER_SECONDS}s ease-in-out infinite;
    }}
    .qx-skeleton-bar.tall {{ height: 22px; }}
    .qx-skeleton-label {{
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {getattr(palette, "symbol_header_meta", "#6b7280")};
        margin-bottom: 4px;
    }}
    @keyframes qx-shimmer {{
        0%   {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    .qx-pulse {{
        display: inline-flex; align-items: center; gap: 7px;
        font-size: 0.78rem; color: {getattr(palette, "symbol_header_meta", "#6b7280")};
    }}
    .qx-pulse-dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: {accent};
        animation: qx-pulse {PULSE_SECONDS}s ease-in-out infinite;
    }}
    @keyframes qx-pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50%      {{ opacity: 0.35; transform: scale(0.82); }}
    }}

    /* The shimmer and the pulse are both decoration; the shape of the
       skeleton and the words beside the dot carry the meaning without
       them. */
    @media (prefers-reduced-motion: reduce) {{
        .qx-skeleton-bar {{ animation: none; background: {edge}; }}
        .qx-pulse-dot {{ animation: none; }}
    }}
    </style>
    """


def skeleton(label: str = "", rows: Optional[Sequence[int]] = None,
             tall_first: bool = False) -> str:
    """A placeholder block. `label` names what is coming, because a bare
    grey rectangle does not tell anyone whether to wait or to worry."""
    widths = list(rows if rows is not None else DEFAULT_ROWS)
    bars = "".join(
        f'<div class="qx-skeleton-bar{" tall" if tall_first and i == 0 else ""}"'
        f' style="width:{max(5, min(100, int(width)))}%"></div>'
        for i, width in enumerate(widths)
    )
    heading = f'<div class="qx-skeleton-label">{label}</div>' if label else ""
    return f'<div class="qx-skeleton" aria-busy="true" aria-live="polite">{heading}{bars}</div>'


def pulse(label: str) -> str:
    """A live indicator for a panel that genuinely re-runs on a timer."""
    return f'<span class="qx-pulse"><span class="qx-pulse-dot"></span>{label}</span>'


def progress_text(done: int, total: int, current: str = "") -> str:
    """"7 of 16 · NVDA". Never reports more done than there are."""
    done = max(0, min(int(done), int(total)))
    text = f"{done} of {total}"
    return f"{text} · {current}" if current else text


def progress_fraction(done: int, total: int) -> float:
    """0.0-1.0 for st.progress, safe when total is 0."""
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, done / total))
