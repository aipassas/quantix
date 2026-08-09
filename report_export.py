"""PDF report generation for Quantix.

Renders the exact same Tear Sheet HTML/CSS fragment finance.py already
builds for the browser's Print-to-PDF flow — via WeasyPrint — rather than
a second, parallel PDF layout implementation that could drift from what's
actually shown on screen.

WeasyPrint needs a native Pango/GObject library it doesn't bundle as a
Python wheel. The import is deferred into generate_tear_sheet_pdf() (never
at module load time) so a machine without that system library doesn't
crash the whole app on startup — the PDF button just reports itself
unavailable instead, and everything else (including the existing browser
Print-to-PDF, which needs no extra dependency at all) keeps working.
"""
import logging
import os
import platform
from typing import Optional, Tuple

from logging_setup import get_logger, log_event, log_exception

logger = get_logger("report_export")

# Homebrew's lib directory isn't on macOS's default dynamic-linker search
# path, so WeasyPrint's cffi-based Pango bindings fail to dlopen without
# it even when Pango is actually installed. Harmless to set on Linux —
# these paths simply won't exist there, and WeasyPrint finds Pango via the
# normal system library path instead (e.g. after `apt-get install
# libpango-1.0-0`).
_HOMEBREW_LIB_CANDIDATES = ("/opt/homebrew/lib", "/usr/local/lib")


def _ensure_weasyprint_can_load_native_libs() -> None:
    if platform.system() != "Darwin":
        return
    existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    parts = existing.split(":") if existing else []
    for candidate in _HOMEBREW_LIB_CANDIDATES:
        if os.path.isdir(candidate) and candidate not in parts:
            parts.append(candidate)
    if parts:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)


def generate_tear_sheet_pdf(html_fragment: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Render the Tear Sheet's existing HTML fragment to PDF bytes via WeasyPrint.

    Returns (pdf_bytes, None) on success, or (None, error_message) if
    WeasyPrint (or its native Pango dependency) isn't available in this
    environment — never raises, so a missing system library degrades to
    an informative message rather than crashing the app.
    """
    _ensure_weasyprint_can_load_native_libs()
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        log_event(logger, logging.WARNING, "pdf.unavailable", reason=str(e))
        return None, (
            "PDF generation isn't available in this environment — WeasyPrint "
            "requires the native Pango library, which isn't installed here "
            f"({type(e).__name__}). Use your browser's Print-to-PDF "
            "(Cmd/Ctrl+P) instead, or install Pango (e.g. `brew install pango` "
            "on macOS, `apt-get install libpango-1.0-0` on Debian/Ubuntu)."
        )

    document = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{html_fragment}</body></html>"
    try:
        return HTML(string=document).write_pdf(), None
    except Exception as e:
        log_exception(logger, "pdf.render_error")
        return None, f"PDF rendering failed: {type(e).__name__}: {e}"
