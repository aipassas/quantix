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


def test_secondary_button_text_meets_wcag_aa_normal_text_contrast():
    """Regression test for a real bug seen in the running app: Streamlit's
    own base theme is light, so main-area secondary buttons kept a WHITE
    face while the app-level text colour cascaded near-white (#e2e8f0)
    onto them — measured at 1.23:1 in-browser, i.e. blank-looking boxes
    that only became readable on hover. Both the resting and hover states
    must now clear AA."""
    for palette in (DARK, LIGHT):
        resting = _contrast_ratio(palette.button_bg, palette.button_text)
        assert resting >= WCAG_AA_NORMAL_TEXT, f"{palette.name}: button_text on button_bg = {resting:.2f}:1"
        hover = _contrast_ratio(palette.button_hover_bg, palette.button_hover_text)
        assert hover >= WCAG_AA_NORMAL_TEXT, f"{palette.name}: button_hover_text on button_hover_bg = {hover:.2f}:1"


def test_secondary_button_face_is_distinguishable_from_the_page():
    """The button has to look like a button. Its face and the surrounding
    canvas being near-identical would be a different flavour of the same
    bug — technically readable text on an invisible control."""
    for palette in (DARK, LIGHT):
        assert palette.button_bg != palette.app_bg or palette.button_border != palette.app_bg, (
            f"{palette.name}: button face and border both match the page background"
        )
        edge = _contrast_ratio(palette.app_bg, palette.button_border)
        assert edge >= 1.3, f"{palette.name}: button border barely separates from the page ({edge:.2f}:1)"


def test_metric_label_meets_wcag_aa_normal_text_contrast():
    """The other half of the same bug: st.metric's caption line ("RSI (14)",
    "Required Fields") kept Streamlit's light-theme label colour and
    measured 1.68:1 against the black canvas."""
    for palette in (DARK, LIGHT):
        ratio = _contrast_ratio(palette.app_bg, palette.metric_label)
        assert ratio >= WCAG_AA_NORMAL_TEXT, f"{palette.name}: metric_label on app_bg = {ratio:.2f}:1"


def test_metric_label_is_dimmer_than_its_value():
    """Label and value must stay visually ranked — the label is secondary
    to the number it captions, so fixing its contrast must not flatten the
    hierarchy by making both equally loud."""
    for palette in (DARK, LIGHT):
        label = _contrast_ratio(palette.app_bg, palette.metric_label)
        value = _contrast_ratio(palette.card_bg, palette.metric_value)
        assert label < value, f"{palette.name}: metric_label ({label:.2f}) is not dimmer than metric_value ({value:.2f})"


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
