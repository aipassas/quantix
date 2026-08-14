"""Tests for theme.py — dark/light palette definitions and local
persistence for the app's chrome theme.
"""
from theme import DARK, LIGHT, PALETTES, load_theme, save_theme


# --- palette definitions -----------------------------------------------------------------

def test_both_palettes_registered_under_their_own_name():
    assert PALETTES["dark"] is DARK
    assert PALETTES["light"] is LIGHT
    assert DARK.name == "dark"
    assert LIGHT.name == "light"


def test_dark_and_light_use_different_plotly_templates():
    assert DARK.plotly_template == "plotly_dark"
    assert LIGHT.plotly_template == "plotly_white"


def test_every_dark_field_has_a_light_counterpart_and_vice_versa():
    """Both palettes must define the same set of fields — finance.py
    reads every field by name off whichever palette is active, so a
    field present on one and missing on the other would only surface as
    an AttributeError at runtime, on whichever theme wasn't exercised."""
    dark_fields = set(DARK.__dataclass_fields__)
    light_fields = set(LIGHT.__dataclass_fields__)
    assert dark_fields == light_fields


def test_dark_and_light_backgrounds_are_meaningfully_different():
    """Sanity check that this is a real dark/light pair, not two
    near-identical palettes."""
    assert DARK.app_bg != LIGHT.app_bg
    assert DARK.app_text != LIGHT.app_text


# --- accessibility: WCAG contrast ratios -----------------------------------------------------------------

def _relative_luminance(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def channel(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = channel(r), channel(g), channel(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(hex_a, hex_b):
    """WCAG 2.x contrast ratio between two colors, always >= 1.0."""
    l1, l2 = _relative_luminance(hex_a), _relative_luminance(hex_b)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

WCAG_AA_NORMAL_TEXT = 4.5
WCAG_AA_LARGE_TEXT = 3.0  # headers, metric values, tab labels — all bold/large per the CSS


def test_body_text_meets_wcag_aa_normal_text_contrast():
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.app_bg, palette.app_text)
        assert ratio >= WCAG_AA_NORMAL_TEXT, f"{palette.name}: app_text on app_bg = {ratio:.2f}:1"


def test_table_body_text_meets_wcag_aa_normal_text_contrast():
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.table_body_bg, palette.table_body_text)
        assert ratio >= WCAG_AA_NORMAL_TEXT, f"{palette.name}: table_body_text on table_body_bg = {ratio:.2f}:1"


def test_metric_value_meets_wcag_aa_large_text_contrast():
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.card_bg, palette.metric_value)
        assert ratio >= WCAG_AA_LARGE_TEXT, f"{palette.name}: metric_value on card_bg = {ratio:.2f}:1"


def test_headers_meet_wcag_aa_large_text_contrast():
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.app_bg, palette.header_text)
        assert ratio >= WCAG_AA_LARGE_TEXT, f"{palette.name}: header_text on app_bg = {ratio:.2f}:1"


def test_tab_selected_text_meets_wcag_aa_large_text_contrast():
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.tab_selected_bg, palette.tab_selected_text)
        assert ratio >= WCAG_AA_LARGE_TEXT, f"{palette.name}: tab_selected_text on tab_selected_bg = {ratio:.2f}:1"


def test_chart_fg_meets_wcag_aa_large_text_contrast_against_app_bg():
    """The ADX line / marker outline color — the plot area itself is
    transparent (paper_bgcolor/plot_bgcolor are 'rgba(0,0,0,0)' throughout
    finance.py), so it composites against the page background."""
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.app_bg, palette.chart_fg)
        assert ratio >= WCAG_AA_LARGE_TEXT, f"{palette.name}: chart_fg on app_bg = {ratio:.2f}:1"


# --- persistence -----------------------------------------------------------------

def test_fresh_instance_defaults_to_dark(tmp_path):
    """Preserves the existing OLED look for anyone already running this
    app before the toggle existed — a fresh/missing state file must not
    silently switch them to light."""
    assert load_theme(tmp_path / "nope.json") == "dark"


def test_save_and_load_theme_round_trip(tmp_path):
    path = tmp_path / "theme.json"
    save_theme("light", path)
    assert load_theme(path) == "light"


def test_load_theme_corrupt_file_degrades_to_default(tmp_path):
    path = tmp_path / "theme.json"
    path.write_text("{not valid json")
    assert load_theme(path) == "dark"


def test_load_theme_unknown_name_degrades_to_default(tmp_path):
    """A hand-edited or future-version file naming a palette this build
    doesn't know must not crash the app — fall back to the default."""
    path = tmp_path / "theme.json"
    path.write_text('{"theme": "solarized"}')
    assert load_theme(path) == "dark"


def test_save_theme_writes_atomically_no_tmp_left_behind(tmp_path):
    path = tmp_path / "theme.json"
    save_theme("light", path)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
