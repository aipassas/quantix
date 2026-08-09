"""Centralized logging configuration for Quantix.

Every module logs through a child of one `quantix` logger, so all output
shares a format and can be filtered by origin (`quantix.data_loader`,
`quantix.finance`, ...).

Output goes three places at once:
  - A rotating file (quantix.log next to this module), so errors persist
    after the terminal is closed — this is the durable "error log."
  - The console running `streamlit run`, for live tailing during development.
  - An in-memory ring buffer, which powers the in-app log viewer so debug
    information is available without leaving the browser.

Events are logged in a structured `event key=value` style via log_event(),
e.g.

    2026-08-08 15:30:02 INFO    [quantix.data_loader] fetch.ok ticker=AAPL dataset="price history" ms=412

which stays readable by eye while remaining greppable and parseable.
"""
import logging
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, List

LOGGER_NAME = "quantix"
LOG_FILENAME = "quantix.log"
LOG_MAX_BYTES = 1_000_000   # ~1 MB per file
LOG_BACKUP_COUNT = 3        # keep 3 rotations → ~4 MB ceiling total
IN_MEMORY_CAPACITY = 500    # records retained for the in-app viewer

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_log_buffer: Deque[str] = deque(maxlen=IN_MEMORY_CAPACITY)


class _InMemoryHandler(logging.Handler):
    """Retains the most recent formatted records so the app can display them
    in-page, instead of the user having to tail a file or watch the terminal."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append(self.format(record))
        except Exception:  # pragma: no cover - handler of last resort
            self.handleError(record)


def log_file_path() -> Path:
    """Absolute path of the rotating log file (kept next to the source, so it
    doesn't depend on the working directory Streamlit was launched from)."""
    return Path(__file__).resolve().parent / LOG_FILENAME


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the `quantix` logger tree and return it.

    Idempotent by design: Streamlit re-executes the entire script on every
    widget interaction, so this must never stack duplicate handlers. The
    level is applied on every call so a debug toggle can raise/lower
    verbosity at runtime.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # avoid double-printing through the root logger

    if not getattr(logger, "_quantix_configured", False):
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        # File handler first, but never let a filesystem problem (read-only
        # dir, permissions) take down the app — degrade to console instead.
        try:
            file_handler = RotatingFileHandler(
                log_file_path(), maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            logging.getLogger(LOGGER_NAME).warning(
                f"logging.file_unavailable path={log_file_path()} error={e}"
            )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        memory_handler = _InMemoryHandler()
        memory_handler.setFormatter(formatter)
        logger.addHandler(memory_handler)

        logger._quantix_configured = True  # type: ignore[attr-defined]

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Child logger for one module, e.g. get_logger("data_loader")."""
    return logging.getLogger(f"{LOGGER_NAME}.{module_name}")


def _format_value(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if (" " in text or "=" in text) else text


def log_event(logger: logging.Logger, level: int, event: str, /, **fields: Any) -> None:
    """Emit one structured line: an `event.name` followed by key=value context.

    Values containing spaces or '=' are quoted so the line stays unambiguous
    to parse. The first three parameters are positional-only so a context
    field may be named `logger`, `level`, or `event` without colliding with
    them.
    """
    if not fields:
        logger.log(level, event)
        return
    context = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
    logger.log(level, f"{event} {context}")


def log_exception(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """Structured ERROR entry that also captures the active traceback."""
    context = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
    logger.exception(f"{event} {context}".strip())


def recent_logs(limit: int = 200) -> List[str]:
    """Most recent formatted log lines, newest last (for the in-app viewer)."""
    return list(_log_buffer)[-limit:]


def clear_log_buffer() -> None:
    _log_buffer.clear()
