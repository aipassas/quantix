"""Read-only HTTP API for programmatic access to Quantix.

RUN IT:  python3 api_server.py            (127.0.0.1:8787)
         python3 api_server.py --port 9000 --host 127.0.0.1

This is a SEPARATE PROCESS from the Streamlit app, and has to be.
Streamlit owns its own HTTP server and offers no way to mount extra
routes on it, so "add an API to the Streamlit app" is not a thing that
exists. Nothing here is started automatically — an API that silently
begins listening on a port because you opened a dashboard would be a
genuinely bad surprise, so starting it is always an explicit act.

WHY STDLIB AND NOT FASTAPI. The endpoint set is tiny, entirely read-only,
and bound to loopback by default. http.server covers that with zero new
dependencies and keeps the authentication path short enough to read in
one sitting, which matters more for a credential check than framework
ergonomics do. The tradeoff is real and worth naming: no automatic
OpenAPI schema, and no battle-tested request parsing. If this ever needs
to face a network rather than a laptop, that tradeoff should be revisited
in favour of a real framework behind a real reverse proxy.

READ-ONLY, AND NO TRADING. The originating task mentions "headless
trading bots". Quantix has no brokerage integration and this API exposes
no write path of any kind — no order placement, no portfolio mutation,
nothing that moves money. `GET /v1` says so explicitly so an integrator
finds out from the API itself rather than by discovering an endpoint
doesn't exist.

BINDS TO LOOPBACK BY DEFAULT. --host 0.0.0.0 publishes your financial
analysis to the whole local network; the flag exists but warns loudly.
There is no TLS here — over anything but loopback, put it behind a
reverse proxy that terminates TLS, because an API key sent over plain
HTTP is a key you have given away.
"""
import argparse
import datetime
import json
import logging
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from api_keys import SCOPES, ApiKey, load_store, touch_last_used, verify_key
from config import API_KEYS
from logging_setup import get_logger, log_event, log_exception, setup_logging

logger = get_logger("api_server")

def _brand_name() -> str:
    """Imported lazily — api_server runs as a separate process and must
    not depend on Streamlit being importable at module load."""
    try:
        from branding import brand
        return brand().name
    except Exception:
        return "Quantix"


API_VERSION = "v1"

# Endpoint table: path -> (required scope, handler). Every route is a GET
# and every scope is a read; see the module docstring.
_ROUTES: Dict[str, Tuple[Optional[str], str]] = {}


def route(path: str, scope: Optional[str]):
    def decorator(fn: Callable):
        _ROUTES[path] = (scope, fn.__name__)
        globals()[f"_handler_{fn.__name__}"] = fn
        return fn
    return decorator


class ApiError(Exception):
    """An error with an HTTP status attached. Raised by handlers and
    turned into a JSON body by the request handler, so no handler has to
    format an error response itself."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _ticker_arg(query: Dict[str, list]) -> str:
    values = query.get("ticker") or query.get("symbol") or []
    ticker = (values[0] if values else "").strip().upper()
    if not ticker:
        raise ApiError(400, "Missing required parameter: ticker")
    if len(ticker) > 20 or not all(c.isalnum() or c in "-.:^" for c in ticker):
        raise ApiError(400, "That doesn't look like a ticker symbol.")
    return ticker


def _int_arg(query: Dict[str, list], name: str, default: int, low: int, high: int) -> int:
    values = query.get(name) or []
    if not values:
        return default
    try:
        value = int(values[0])
    except (TypeError, ValueError):
        raise ApiError(400, f"{name} must be a whole number.")
    if not low <= value <= high:
        raise ApiError(400, f"{name} must be between {low} and {high}.")
    return value


def _round(value: Any, places: int = 4) -> Any:
    """Numbers are rounded on the way out and None is preserved as null.

    Preserving None matters: this app's rule is never to fabricate a
    number, and a JSON `0.0` where the real answer is "not reported"
    would be exactly that — a robot consuming it has no way to tell the
    difference.
    """
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
        return round(float(value), places)
    except (TypeError, ValueError):
        return value


# --- endpoints ----------------------------------------------------------------

@route("/v1", None)
def index(key: Optional[ApiKey], query: Dict[str, list]) -> dict:
    """Discovery. Unauthenticated on purpose so an integrator can see what
    exists — and what deliberately does not — before wiring a key up."""
    return {
        "service": f"{_brand_name()} API",
        "version": API_VERSION,
        "read_only": True,
        "trading_supported": False,
        "notice": (
            "This API is read-only. Quantix performs analysis and has no brokerage "
            "integration, so there is no endpoint that places orders, moves money, or "
            "mutates any account state. Every route below is a GET."
        ),
        "authentication": (
            "Send your key as 'Authorization: Bearer qtx_...' or 'X-API-Key: qtx_...'. "
            "Keys are created in the Quantix sidebar under API Keys."
        ),
        "scopes": SCOPES,
        "endpoints": {
            path: {"scope": scope, "method": "GET"}
            for path, (scope, _) in sorted(_ROUTES.items())
        },
    }


@route("/v1/quote", "quote:read")
def quote(key: Optional[ApiKey], query: Dict[str, list]) -> dict:
    from data_loader import load_ticker_bundle

    ticker = _ticker_arg(query)
    bundle = load_ticker_bundle(ticker, deep=False)
    info = bundle.info or {}
    if not info:
        raise ApiError(404, f"No data available for {ticker}.")
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous = info.get("previousClose")
    change = None
    change_pct = None
    if price is not None and previous:
        change = float(price) - float(previous)
        change_pct = (change / float(previous)) * 100
    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "currency": info.get("currency"),
        "price": _round(price),
        "previous_close": _round(previous),
        "change": _round(change),
        "change_pct": _round(change_pct, 2),
        "market_cap": info.get("marketCap"),
        "as_of": datetime.datetime.now().isoformat(timespec="seconds"),
    }


@route("/v1/fundamentals", "fundamentals:read")
def fundamentals(key: Optional[ApiKey], query: Dict[str, list]) -> dict:
    from data_loader import load_ticker_bundle
    from financial_standardization import standardize_financials
    from fundamental_analysis import FundamentalAnalysisEngine

    ticker = _ticker_arg(query)
    bundle = load_ticker_bundle(ticker, deep=True)
    if not (bundle.info or {}):
        raise ApiError(404, f"No data available for {ticker}.")
    engine = FundamentalAnalysisEngine(standardize_financials(bundle), bundle.info)
    metrics = engine.analyze()

    return {
        "ticker": ticker,
        "sector": (bundle.info or {}).get("sector"),
        "alignment": {
            "score_pct": _round(metrics.score_pct, 1),
            "verdict": metrics.alignment_verdict,
        },
        "profitability": {
            "gross_margin_pct": _round(engine.gross_margin_pct(), 2),
            "operating_margin_pct": _round(engine.operating_margin_pct(), 2),
            "net_margin_pct": _round(engine.net_margin_pct_computed(), 2),
            "roa_pct": _round(engine.roa_pct(), 2),
            "roe_pct": _round(engine.roe_pct_computed(), 2),
            "roic_pct": _round(engine.roic_pct(), 2),
        },
        "liquidity": {
            "current_ratio": _round(engine.current_ratio_computed(), 2),
            "quick_ratio": _round(engine.quick_ratio_computed(), 2),
        },
        "leverage": {
            "interest_coverage": _round(engine.interest_coverage(), 2),
            "altman_z": _round(engine.altman_z_score(), 2),
        },
        "valuation": {
            "pe_ratio": _round(engine.pe_ratio_computed(), 2),
            "price_to_book": _round(engine.price_to_book_computed(), 2),
            "ev_to_ebitda": _round(engine.ev_to_ebitda_computed(), 2),
            "fcf_yield_pct": _round(engine.fcf_yield_pct(), 2),
        },
        "disclosure": (
            "A null value means the figure was not reported or is not computable for this "
            "company, never that it is zero. Metrics that cannot be evaluated are excluded "
            "from the alignment score rather than counted as failures."
        ),
    }


# Every risk_analytics function below returns a FRACTION, not a
# percentage — finance.py multiplies by 100 at render time. Emitting the
# raw fraction under a field named *_pct would report AAPL's ~33%
# annualized volatility as 0.33%, which is a fabricated number by
# mislabelling. _pct fields are scaled here, ratios are not.
def _as_pct(value):
    return None if value is None else _round(value * 100, 2)


# Below this, an annualized figure is being extrapolated from too little
# data to mean anything. Matches historical_comparison.py's own floor.
_MIN_RISK_BARS = 30

# yfinance's history() defaults to ONE MONTH when start/end are omitted —
# about 24 trading days. Annualizing volatility off that is misleading, so
# the window is always explicit here.
_DEFAULT_RISK_DAYS = 365
_MAX_RISK_DAYS = 1825


@route("/v1/risk", "risk:read")
def risk(key: Optional[ApiKey], query: Dict[str, list]) -> dict:
    import risk_analytics as ra
    from config import RISK
    from data_loader import load_ticker_bundle

    ticker = _ticker_arg(query)
    days = _int_arg(query, "days", _DEFAULT_RISK_DAYS, 1, _MAX_RISK_DAYS)
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)

    bundle = load_ticker_bundle(ticker, start=start, end=end, deep=True)
    prices = bundle.price_history
    if prices is None or prices.empty:
        raise ApiError(404, f"No price history available for {ticker}.")
    if len(prices) < _MIN_RISK_BARS:
        raise ApiError(422, (
            f"Only {len(prices)} trading days of history available for {ticker} over the last "
            f"{days} days — too few to annualize from. Request a longer window with ?days=."
        ))

    close = prices["Close"]
    drawdown = ra.compute_max_drawdown(close)
    confidence = RISK.var_confidence_default
    return {
        "ticker": ticker,
        "window_days": days,
        "observations": int(len(prices)),
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
        "annualized_return_pct": _as_pct(ra.compute_annualized_return(prices)),
        "annualized_volatility_pct": _as_pct(ra.compute_annualized_volatility(prices)),
        "sharpe_ratio": _round(ra.compute_sharpe_ratio(prices), 2),
        "sortino_ratio": _round(ra.compute_sortino_ratio(prices), 2),
        "historical_var_pct": _as_pct(ra.compute_historical_var(prices, confidence)),
        "expected_shortfall_pct": _as_pct(ra.compute_expected_shortfall(prices, confidence)),
        # max_drawdown is a NEGATIVE FRACTION (-0.32 for a 32% decline).
        "max_drawdown_pct": _as_pct(None if drawdown is None else drawdown.max_drawdown),
        "var_confidence_level": confidence,
        "risk_free_rate_assumed": RISK.risk_free_rate,
        "disclosure": (
            "Computed from the stated window of daily closes. VaR and Expected Shortfall are "
            "1-day figures at the given confidence level; return and volatility are annualized. "
            "Past volatility is not a forecast, and a null value means the metric was not "
            "computable from the available history — never that it is zero."
        ),
    }


@route("/v1/watchlists", "watchlist:read")
def watchlists(key: Optional[ApiKey], query: Dict[str, list]) -> dict:
    """Owner-scoped. Resolves the namespace from the KEY's owner_key, not
    from any session — the server has none. This is the one endpoint where
    getting the namespace wrong would show one user another's data, so the
    namespace comes from the credential itself and nowhere else."""
    import local_store
    from config import FAVORITES, WATCHLIST_PANEL

    owner = key.owner_key if key else ""

    import favorites as favorites_module
    import watchlist_panel

    wl = watchlist_panel.load_watchlist_store(
        local_store.store_path(WATCHLIST_PANEL.store_filename, namespace=owner or None))
    quick = favorites_module.load_store(
        local_store.store_path(FAVORITES.store_filename, namespace=owner or None))

    return {
        "active": wl.active,
        "lists": {name: list(entry.tickers) for name, entry in wl.lists.items()},
        "favorites": list(quick.favorites),
        "recents": list(quick.recents),
        "owner_scoped": bool(owner),
    }


# --- server -------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "Quantix"
    sys_version = ""  # don't advertise the Python version

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Read-only API, no cookies, no browser session to protect — but
        # say so rather than leaving it ambiguous.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 401:
            self.send_header("WWW-Authenticate", 'Bearer realm="Quantix"')
        self.end_headers()
        self.wfile.write(body)

    def _presented_key(self) -> str:
        auth_header = self.headers.get("Authorization", "") or ""
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return (self.headers.get("X-API-Key", "") or "").strip()

    def do_GET(self) -> None:  # noqa: N802  (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/v1"
        query = parse_qs(parsed.query)

        entry = _ROUTES.get(path)
        if entry is None:
            self._send(404, {"error": "Unknown endpoint.",
                             "hint": f"GET /{API_VERSION} lists every available endpoint."})
            return
        required_scope, handler_name = entry
        handler = globals()[f"_handler_{handler_name}"]

        key: Optional[ApiKey] = None
        if required_scope is not None:
            presented = self._presented_key()
            if not presented:
                self._send(401, {"error": "Missing API key.",
                                 "hint": "Send 'Authorization: Bearer qtx_...' or 'X-API-Key: qtx_...'."})
                return
            key, error = verify_key(load_store(), presented)
            if key is None:
                # Uniform 401 regardless of whether the key was malformed,
                # unknown, revoked or expired at the transport level; the
                # message distinguishes them for a legitimate owner
                # debugging their own robot, but the status never lets an
                # attacker enumerate which key ids exist.
                log_event(logger, logging.WARNING, "api_server.auth_failed", path=path)
                self._send(401, {"error": error or "Invalid API key."})
                return
            if not key.has_scope(required_scope):
                self._send(403, {
                    "error": f"This key does not carry the '{required_scope}' scope.",
                    "key_scopes": list(key.scopes),
                    "required_scope": required_scope,
                })
                return

        try:
            payload = handler(key, query)
        except ApiError as e:
            self._send(e.status, {"error": e.message})
            return
        except Exception:
            # Never leak a traceback to a caller; log it in full instead.
            log_exception(logger, "api_server.handler_failed", section="api_server")
            self._send(500, {"error": "Internal error while producing this response."})
            return

        if key is not None:
            threading.Thread(target=touch_last_used, args=(key.id,), daemon=True).start()
            log_event(logger, logging.INFO, "api_server.request",
                      path=path, key_id=key.id, scope=required_scope)
        self._send(200, payload)

    def log_message(self, fmt: str, *args) -> None:
        """Route stdlib's access log into the app's logger instead of
        stderr, so API access lands in the same quantix.log as everything
        else rather than in a terminal nobody is reading."""
        logger.info("api_server.access %s", fmt % args)


def serve(host: str = None, port: int = None) -> None:
    host = host or API_KEYS.default_host
    port = port or API_KEYS.default_port

    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "api_server.exposed_beyond_loopback host=%s — this publishes Quantix "
            "analysis beyond this machine, over plain HTTP with no TLS. Put a "
            "TLS-terminating reverse proxy in front of it or an API key sent to it "
            "is a key given away.", host,
        )
        print(
            f"\n  WARNING: binding to {host}, not loopback.\n"
            f"  There is no TLS here. Any API key sent over plain HTTP is exposed.\n"
            f"  Use a reverse proxy that terminates TLS if this is reachable by anyone else.\n",
            file=sys.stderr,
        )

    server = ThreadingHTTPServer((host, port), _Handler)
    keys = load_store().keys
    active = sum(1 for k in keys if k.is_usable())
    print(f"Quantix API — read-only, no trading. http://{host}:{port}/{API_VERSION}")
    print(f"{active} active key(s) of {len(keys)} issued. Create keys in the Quantix sidebar.")
    if active == 0:
        print("No usable keys yet — every endpoint except /v1 will return 401 until you make one.")
    log_event(logger, logging.INFO, "api_server.started", host=host, port=port, active_keys=active)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only HTTP API for Quantix.")
    parser.add_argument("--host", default=API_KEYS.default_host,
                        help="Interface to bind. Defaults to loopback; anything else is exposed.")
    parser.add_argument("--port", type=int, default=API_KEYS.default_port)
    args = parser.parse_args()
    setup_logging()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
