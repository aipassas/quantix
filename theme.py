"""Dark/Light theme — palette definitions and local persistence.

finance.py renders the actual toggle (a sidebar radio in the System tab)
and builds the injected CSS/Plotly template from whichever ThemePalette
this module resolves; this module only holds the two palettes and
which one is currently selected.

Scope: this covers the app's own chrome — the "Professional UI Injection
(OLED Edition)" CSS block and every chart's `template=` — NOT the CIO Tear
Sheet, which is a deliberately-white printed-report facsimile (styled to
look like a physical document regardless of app theme, the same way a
PDF export doesn't follow your OS dark mode) and is intentionally left
alone here.

Persisted with the same atomic-write local-JSON-file pattern every other
cross-restart preference in this app already uses (onboarding.py,
watchlist_panel.py, risk_alerts.py, ...). Quantix has no accounts, so
this is "the theme this locally-run instance was last set to," not a
per-visitor preference.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import THEME
from local_store import atomic_write_text
from logging_setup import get_logger, log_exception

logger = get_logger("theme")


@dataclass(frozen=True)
class ThemePalette:
    name: str
    label: str
    plotly_template: str

    sidebar_icon: str  # sidebar-reopen control icon/fill color

    app_bg: str
    app_text: str
    header_text: str

    card_bg: str
    card_border: str
    card_accent: str  # kept identical across themes — the brand's neon-green accent
    card_hover_accent: str
    metric_value: str
    metric_label: str  # st.metric's caption line, above the value

    # Secondary (non-primary) buttons in the MAIN area. These need explicit
    # colors because Streamlit's own base theme is light: left alone, its
    # widget chrome keeps a white button face while the app-level `color`
    # set below cascades near-white text onto it — measured at 1.23:1 in
    # the browser, i.e. effectively invisible until hovered. Deliberately
    # NOT applied to the sidebar, which keeps Streamlit's light chrome and
    # already measures ~11.9:1 there.
    button_bg: str
    button_border: str
    button_text: str
    button_hover_bg: str
    button_hover_border: str
    button_hover_text: str

    table_head_bg: str
    table_head_text: str
    table_head_border: str
    table_body_bg: str
    table_body_text: str
    table_body_border: str

    expander_bg: str
    expander_border: str
    expander_text: str

    tab_rail_border: str
    tab_bg: str
    tab_border: str
    tab_inactive_text: str
    tab_hover_bg: str
    tab_hover_text: str
    tab_selected_bg: str
    tab_selected_border: str
    tab_selected_text: str

    symbol_header_bg: str
    symbol_header_border: str
    symbol_header_shadow: str
    symbol_header_name: str
    symbol_header_meta: str
    symbol_header_flat: str

    chart_fg: str  # a primary chart line/marker-outline that must read against the plot background
    chart_faint_line: str  # a subtle dashed reference/threshold line


DARK = ThemePalette(
    name="dark", label="Dark (OLED)", plotly_template="plotly_dark",
    sidebar_icon="#ffffff",
    app_bg="#000000", app_text="#e2e8f0", header_text="#ffffff",
    card_bg="#0a0a0a", card_border="#1a1a1a", card_accent="#00ea77", card_hover_accent="#ffffff",
    metric_value="#ffffff", metric_label="#9ca3af",
    button_bg="#0a0a0a", button_border="#2a2a2a", button_text="#e2e8f0",
    button_hover_bg="#1a1a1a", button_hover_border="#00ea77", button_hover_text="#ffffff",
    table_head_bg="#0a0a0a", table_head_text="#ffffff", table_head_border="#333333",
    table_body_bg="#000000", table_body_text="#cccccc", table_body_border="#1a1a1a",
    expander_bg="#0a0a0a", expander_border="#1a1a1a", expander_text="#ffffff",
    tab_rail_border="#1f1f1f", tab_bg="#0a0a0a", tab_border="#1a1a1a",
    tab_inactive_text="#9ca3af", tab_hover_bg="#141414", tab_hover_text="#e5e7eb",
    tab_selected_bg="#161616", tab_selected_border="#2a2a2a", tab_selected_text="#ffffff",
    symbol_header_bg="#0a0a0a", symbol_header_border="#1f1f1f",
    symbol_header_shadow="rgba(0, 0, 0, 0.85)",
    symbol_header_name="#9ca3af", symbol_header_meta="#6b7280", symbol_header_flat="#9ca3af",
    chart_fg="#ffffff", chart_faint_line="rgba(255, 255, 255, 0.3)",
)

LIGHT = ThemePalette(
    name="light", label="Light", plotly_template="plotly_white",
    sidebar_icon="#0f172a",
    app_bg="#ffffff", app_text="#1e293b", header_text="#0f172a",
    card_bg="#f8fafc", card_border="#e2e8f0", card_accent="#00ea77", card_hover_accent="#0f172a",
    metric_value="#0f172a", metric_label="#475569",
    button_bg="#ffffff", button_border="#cbd5e1", button_text="#0f172a",
    button_hover_bg="#f1f5f9", button_hover_border="#0f172a", button_hover_text="#0f172a",
    table_head_bg="#f1f5f9", table_head_text="#0f172a", table_head_border="#cbd5e1",
    table_body_bg="#ffffff", table_body_text="#334155", table_body_border="#e2e8f0",
    expander_bg="#f8fafc", expander_border="#e2e8f0", expander_text="#0f172a",
    tab_rail_border="#e2e8f0", tab_bg="#f8fafc", tab_border="#e2e8f0",
    tab_inactive_text="#64748b", tab_hover_bg="#f1f5f9", tab_hover_text="#1e293b",
    tab_selected_bg="#e2e8f0", tab_selected_border="#cbd5e1", tab_selected_text="#0f172a",
    symbol_header_bg="#f8fafc", symbol_header_border="#e2e8f0",
    symbol_header_shadow="rgba(15, 23, 42, 0.12)",
    symbol_header_name="#64748b", symbol_header_meta="#94a3b8", symbol_header_flat="#64748b",
    chart_fg="#0f172a", chart_faint_line="rgba(15, 23, 42, 0.25)",
)

PALETTES = {"dark": DARK, "light": LIGHT}


def _state_path() -> Path:
    return Path(__file__).resolve().parent / THEME.state_filename


def load_theme(path: Optional[Path] = None) -> str:
    """The name of the last-persisted theme ("dark" or "light"). Never
    raises: a missing file, a corrupt file, or a persisted name that
    isn't a known palette all fall back to THEME.default_theme rather
    than crashing the app on load."""
    path = path or _state_path()
    if not path.exists():
        return THEME.default_theme
    try:
        name = json.loads(path.read_text()).get("theme")
    except Exception:
        log_exception(logger, "theme.state_corrupt", section="theme")
        return THEME.default_theme
    return name if name in PALETTES else THEME.default_theme


def save_theme(name: str, path: Optional[Path] = None) -> None:
    """Atomic write (temp file + rename), same pattern as every other
    local store in this app."""
    path = path or _state_path()
    atomic_write_text(path, json.dumps({"theme": name}, indent=2))
