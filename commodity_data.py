"""Futures contracts, the forward curve, and what inventory is reachable.

THE RECORDED NOTE WAS WRONG, and correcting it is most of this module.
`asset_class.MISSING_SOURCES[FUTURE]` said "the forward curve needs
quotes for every contract month; Yahoo returns only the front month per
symbol". Probed on 2026-08-26, that is false. Yahoo quotes DATED
contracts, and the exchange suffix is what makes them resolve:

    GCZ26         -> HTTP 404, "Quote not found"
    GCZ26.CMX     -> Gold Dec 26, 4683.20, open interest 330,763

So the forward curve IS buildable, for every commodity here, along with
volume and open interest per contract. The bare-symbol 404 is the same
shape as the VWCE lookup failure: the venue is the missing part, not the
symbol.

THE CONTRACT CALENDAR IS PROBED, NEVER ASSUMED. Different commodities
list different months and the difference is large. Over an 18-month
window from September 2026:

    Gold      18 of 18 months quote
    Crude     17 of 18
    Nat gas   18 of 18
    Copper    18 of 18
    Corn       7 of 18   (Sep, Dec, Mar, May, Jul — the CBOT cycle)

Hard-coding "all months" invents eleven phantom corn contracts;
hard-coding the classical gold cycle (Feb/Apr/Jun/Aug/Oct/Dec) discards
twelve real quotes. Neither guess survives contact with the data, so
`load_curve` generates candidates and keeps whatever answers.

QUOTED IS NOT TRADED. Gold quotes every month and only the even ones are
held. Measured open interest across the gold strip:

    Oct 26  55,855   Nov 26     544
    Dec 26 330,763   Jan 27     481
    Feb 27  26,244   Mar 27      30
    Apr 27   7,656   May 27      13
    Jun 27   3,203   Jul 27       8

A contract with eight open lots has a price, not a market, and giving it
equal weight in a curve fit is how a shape gets misread. Points are kept
for SHAPE only above `LIQUIDITY_FLOOR_FRACTION` of the curve's own
largest open interest — a relative test, so a thin commodity is not
wholly discarded by an absolute one. At 0.3% that splits gold exactly
along the even/odd line while keeping every crude month (20-100% of max)
and every corn month (5-100%).

THERE IS NO SPOT PRICE, so basis is not computable. XAUUSD=X and
XAGUSD=X both return nothing. The task's "basis = futures - spot" has no
second term, and calling the front contract "spot" would relabel a
dated claim as an undated one. The calendar spread between the two
nearest contracts measures the same tightness out of quotes that exist,
and is what `commodity_curve` reports instead.

INVENTORY: CRUDE YES, AGRICULTURE NO. EIA's v2 API answers a keyless
request with HTTP 403 API_KEY_MISSING, but its own history page is open
and machine-readable — 573 rows of weekly US crude stocks back to 1982,
enough for the seasonal comparison that makes an inventory number mean
anything. USDA NASS returns 401 unauthorized, so agricultural stocks,
yields and weather-driven supply are not in this build at all.
"""
import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger(__name__)

CURVE_TTL_SECONDS = 900
INVENTORY_TTL_SECONDS = 21600      # weekly data; six hours is generous
REQUEST_TIMEOUT_SECONDS = 30

# CME month codes, January to December. Index+1 is the calendar month.
MONTH_CODES = "FGHJKMNQUVXZ"

# How far out to look for contracts, in months. Eighteen reaches a full
# year beyond the front for every commodity here without spending fetches
# on a tail that quotes but does not trade.
CURVE_HORIZON_MONTHS = 18

# A contract is used for SHAPE only if its open interest is at least this
# fraction of the largest on the curve. Measured: 0.3% separates gold's
# even months (>=0.32% of max) from its odd ones (<=0.16%) and keeps
# every listed crude and corn month.
LIQUIDITY_FLOOR_FRACTION = 0.003

ENERGY, METALS, AGRICULTURE, LIVESTOCK = (
    "Energy", "Metals", "Agriculture", "Livestock")


@dataclass(frozen=True)
class Commodity:
    key: str
    root: str            # CME root, e.g. "GC"
    exchange: str        # Yahoo venue suffix, e.g. "CMX"
    name: str
    unit: str
    sector: str
    continuous: str      # Yahoo's front-month symbol, e.g. "GC=F"
    note: str = ""


# Sixteen contracts, every one probed live. The `continuous` symbol is
# what a user types; the dated symbols are constructed from root and
# exchange.
COMMODITIES: Tuple[Commodity, ...] = (
    Commodity("gold", "GC", "CMX", "Gold", "USD/troy oz", METALS, "GC=F",
              "Quotes every month; only the even ones carry open "
              "interest."),
    Commodity("silver", "SI", "CMX", "Silver", "USD/troy oz", METALS, "SI=F"),
    Commodity("copper", "HG", "CMX", "Copper", "USD/lb", METALS, "HG=F"),
    Commodity("platinum", "PL", "NYM", "Platinum", "USD/troy oz", METALS,
              "PL=F"),
    Commodity("palladium", "PA", "NYM", "Palladium", "USD/troy oz", METALS,
              "PA=F"),
    Commodity("wti", "CL", "NYM", "WTI Crude Oil", "USD/barrel", ENERGY,
              "CL=F", "The only commodity here with inventory data."),
    Commodity("brent", "BZ", "NYM", "Brent Crude Oil", "USD/barrel", ENERGY,
              "BZ=F"),
    Commodity("natgas", "NG", "NYM", "Natural Gas", "USD/MMBtu", ENERGY,
              "NG=F",
              "Its curve carries the heating season: measured Oct 2.86 "
              "against Jan 3.93, a 37% winter premium that is a real "
              "price, not a distortion."),
    Commodity("corn", "ZC", "CBT", "Corn", "US cents/bushel", AGRICULTURE,
              "ZC=F",
              "Lists only the Mar/May/Jul/Sep/Dec cycle — seven of "
              "eighteen months."),
    Commodity("wheat", "ZW", "CBT", "Chicago Wheat", "US cents/bushel",
              AGRICULTURE, "ZW=F"),
    Commodity("soybeans", "ZS", "CBT", "Soybeans", "US cents/bushel",
              AGRICULTURE, "ZS=F"),
    Commodity("coffee", "KC", "NYB", "Coffee", "US cents/lb", AGRICULTURE,
              "KC=F"),
    Commodity("sugar", "SB", "NYB", "Sugar #11", "US cents/lb", AGRICULTURE,
              "SB=F"),
    Commodity("cotton", "CT", "NYB", "Cotton", "US cents/lb", AGRICULTURE,
              "CT=F"),
    Commodity("cattle", "LE", "CME", "Live Cattle", "US cents/lb", LIVESTOCK,
              "LE=F"),
    Commodity("hogs", "HE", "CME", "Lean Hogs", "US cents/lb", LIVESTOCK,
              "HE=F"),
)
COMMODITIES_BY_KEY: Dict[str, Commodity] = {c.key: c for c in COMMODITIES}
COMMODITIES_BY_SYMBOL: Dict[str, Commodity] = {
    c.continuous.upper(): c for c in COMMODITIES}

SECTORS: Tuple[str, ...] = (ENERGY, METALS, AGRICULTURE, LIVESTOCK)

SPOT_UNAVAILABLE = (
    "There is no spot price in this build, so basis (futures minus spot) "
    "is not reported. XAUUSD=X and XAGUSD=X both return nothing, and "
    "calling the front contract \"spot\" would relabel a dated claim as "
    "an undated one. The calendar spread between the two nearest "
    "contracts measures the same tightness out of quotes that exist."
)

AGRICULTURAL_INVENTORY_UNAVAILABLE = (
    "Agricultural stocks, yields and acreage are not sourced. USDA's "
    "QuickStats API answers a keyless request with HTTP 401, and "
    "registering for a key is not something this build does on anyone's "
    "behalf."
)

GEOPOLITICAL_UNAVAILABLE = (
    "Geopolitical risk is not scored. There is no feed of conflict "
    "exposure per commodity, and a percentage assembled by hand would "
    "be an opinion presented as a measurement — on a subject where the "
    "number would look authoritative precisely because it is a number."
)

WEATHER_UNAVAILABLE = (
    "Weather-driven yield risk is not modelled. It needs USDA condition "
    "and yield data, which is behind the same key as the stocks above."
)

POLYGON_UNCONFIGURED = (
    "Polygon.io is not wired. Everything here comes from the exchange "
    "quotes Yahoo already publishes, which cover price, volume and open "
    "interest per dated contract — the inputs the curve actually needs."
)


# --- contract symbols ---------------------------------------------------------

def contract_symbol(commodity: Commodity, year: int, month: int) -> str:
    """The Yahoo symbol for one dated contract.

    The venue suffix is load-bearing: GCZ26 is a 404 and GCZ26.CMX is a
    quote. A symbol built without it looks correct and resolves to
    nothing.
    """
    if not 1 <= int(month) <= 12:
        raise ValueError(f"month must be 1-12, got {month!r}")
    return (f"{commodity.root}{MONTH_CODES[int(month) - 1]}"
            f"{int(year) % 100:02d}.{commodity.exchange}")


def candidate_months(start: Tuple[int, int],
                     horizon: int = CURVE_HORIZON_MONTHS
                     ) -> Tuple[Tuple[int, int], ...]:
    """(year, month) pairs to probe, starting from `start` inclusive."""
    year, month = int(start[0]), int(start[1])
    out: List[Tuple[int, int]] = []
    for _ in range(int(horizon)):
        out.append((year, month))
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return tuple(out)


# --- the curve ----------------------------------------------------------------

@dataclass(frozen=True)
class ContractPoint:
    symbol: str
    year: int
    month: int
    price: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None

    @property
    def label(self) -> str:
        return f"{MONTH_CODES[self.month - 1]}{self.year % 100:02d}"

    @property
    def expiry(self) -> "pd.Timestamp":
        return pd.Timestamp(year=self.year, month=self.month, day=1)

    def months_out(self, reference: "pd.Timestamp") -> float:
        """Distance to this contract in months, from a reference date."""
        return ((self.expiry.year - reference.year) * 12
                + (self.expiry.month - reference.month))

    def years_out(self, reference: "pd.Timestamp") -> float:
        return self.months_out(reference) / 12.0


@dataclass(frozen=True)
class Curve:
    commodity: Optional[Commodity] = None
    points: Tuple[ContractPoint, ...] = ()
    as_of: Optional["pd.Timestamp"] = None
    probed: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return len(self.points) >= 2 and not self.error

    @property
    def front(self) -> Optional[ContractPoint]:
        return self.points[0] if self.points else None

    @property
    def max_open_interest(self) -> int:
        return max((p.open_interest or 0) for p in self.points) if self.points else 0

    @property
    def liquid_points(self) -> Tuple[ContractPoint, ...]:
        """Contracts with enough open interest to carry a shape read.

        Falls back to every point when NO contract reports open interest
        — an absent field is not evidence of an illiquid market, and
        discarding the whole curve over it would be the same error as
        grading an ETF on filings it never makes.
        """
        largest = self.max_open_interest
        if largest <= 0:
            return self.points
        floor = largest * LIQUIDITY_FLOOR_FRACTION
        kept = tuple(p for p in self.points
                     if (p.open_interest or 0) >= floor)
        return kept if len(kept) >= 2 else self.points

    @property
    def thin_points(self) -> Tuple[ContractPoint, ...]:
        liquid = set(p.symbol for p in self.liquid_points)
        return tuple(p for p in self.points if p.symbol not in liquid)


def _int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        number = float(value)
        return None if number != number else int(number)
    except (TypeError, ValueError):
        return None


def _float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        number = float(value)
        return None if number != number else number
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=CURVE_TTL_SECONDS, show_spinner=False)
def load_curve(commodity_key: str,
               start: Optional[Tuple[int, int]] = None,
               horizon: int = CURVE_HORIZON_MONTHS) -> Curve:
    """Every dated contract that answers, in expiry order.

    Candidates are generated and the ones that quote are kept — the
    contract calendar is never assumed, because corn lists seven months
    of eighteen and gold lists all of them.
    """
    commodity = COMMODITIES_BY_KEY.get(commodity_key)
    if commodity is None:
        return Curve(error=f"No commodity named {commodity_key!r}.")

    import yfinance as yf

    today = pd.Timestamp.today().normalize()
    if start is None:
        start = (today.year, today.month)
    candidates = candidate_months(start, horizon)

    points: List[ContractPoint] = []
    for year, month in candidates:
        symbol = contract_symbol(commodity, year, month)
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:                   # noqa: BLE001
            log_exception(logger, "commodity_data.contract_failed",
                          symbol=symbol,
                          error=f"{type(exc).__name__}: {exc}")
            continue
        price = _float(info.get("regularMarketPrice"))
        if price is None or price <= 0:
            continue
        points.append(ContractPoint(
            symbol=symbol, year=year, month=month, price=price,
            open_interest=_int(info.get("openInterest")),
            volume=_int(info.get("regularMarketVolume"))))

    if len(points) < 2:
        return Curve(commodity=commodity, points=tuple(points), as_of=today,
                     probed=len(candidates),
                     error=("Fewer than two contracts quoted, so there is "
                            "no curve to read. The front month alone is a "
                            "price, not a term structure."))
    log_event(logger, logging.INFO, "commodity_data.curve_loaded",
              commodity=commodity_key, contracts=len(points),
              probed=len(candidates))
    return Curve(commodity=commodity, points=tuple(points), as_of=today,
                 probed=len(candidates))


# --- inventory ----------------------------------------------------------------

EIA_CRUDE_HISTORY_URL = (
    "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTUS1&f=W")

EIA_API_UNCONFIGURED = (
    "EIA's v2 API is not used: a keyless request returns HTTP 403 "
    "API_KEY_MISSING. The weekly series below comes from EIA's own "
    "public history page instead, which needs no key and carries the "
    "full record back to 1982."
)

INVENTORY_COMMODITIES = ("wti",)


def inventory_available(commodity_key: str) -> bool:
    return commodity_key in INVENTORY_COMMODITIES


def inventory_note(commodity_key: str) -> str:
    """Why there is no inventory series for this commodity."""
    if inventory_available(commodity_key):
        return ""
    commodity = COMMODITIES_BY_KEY.get(commodity_key)
    if commodity is not None and commodity.sector in (AGRICULTURE, LIVESTOCK):
        return AGRICULTURAL_INVENTORY_UNAVAILABLE
    return ("Inventory is sourced for US crude oil only. EIA publishes "
            "weekly petroleum stocks without a key; there is no "
            "equivalent open series for this commodity here.")


@st.cache_data(ttl=INVENTORY_TTL_SECONDS, show_spinner=False)
def load_crude_inventory() -> Tuple[Optional["pd.Series"], Optional[str]]:
    """Weekly US commercial crude stocks, in thousand barrels.

    EIA lays the history out as one row per month with five (End Date,
    Value) column pairs, so it is unpivoted here rather than read as a
    time series. The dates carry no year of their own — the row does —
    which is why the year comes from the Year-Month column and not from
    parsing the cell.
    """
    import requests

    try:
        response = requests.get(EIA_CRUDE_HISTORY_URL,
                                timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
        tables = pd.read_html(io.StringIO(response.text))
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "commodity_data.inventory_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return None, ("EIA's weekly crude stocks page did not answer. "
                      "Inventory is unavailable this run.")

    frame = None
    for table in sorted(tables, key=lambda t: -(t.shape[0] * t.shape[1])):
        flat = [" ".join(str(part) for part in col) if isinstance(col, tuple)
                else str(col) for col in table.columns]
        if any("Year-Month" in name for name in flat):
            table = table.copy()
            table.columns = flat
            frame = table
            break
    if frame is None:
        return None, "EIA's page did not contain the expected history table."

    year_month_col = next(c for c in frame.columns if "Year-Month" in c)
    date_cols = [c for c in frame.columns if "End Date" in c]
    value_cols = [c for c in frame.columns if c.endswith("Value")]

    records: List[Tuple[pd.Timestamp, float]] = []
    for _, row in frame.iterrows():
        stamp = str(row[year_month_col])
        if "-" not in stamp:
            continue
        year_text = stamp.split("-")[0].strip()
        if not year_text.isdigit():
            continue
        year = int(year_text)
        for date_col, value_col in zip(date_cols, value_cols):
            raw_date, raw_value = row.get(date_col), row.get(value_col)
            value = _float(raw_value)
            if value is None or not isinstance(raw_date, str):
                continue
            parts = raw_date.strip().split("/")
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                continue
            month, day = int(parts[0]), int(parts[1])
            # December weeks belonging to a January row roll back a year;
            # the row's month is the anchor, the cell's is the truth.
            row_month = stamp.split("-")[-1].strip()
            anchor = {"Jan": 1, "Dec": 12}.get(row_month)
            stamp_year = year
            if anchor == 1 and month == 12:
                stamp_year = year - 1
            elif anchor == 12 and month == 1:
                stamp_year = year + 1
            try:
                records.append((pd.Timestamp(year=stamp_year, month=month,
                                             day=day), value))
            except ValueError:
                continue

    if not records:
        return None, "EIA's history table held no readable observations."
    series = pd.Series(dict(records)).sort_index()
    series.name = "crude_stocks_kbbl"
    log_event(logger, logging.INFO, "commodity_data.inventory_loaded",
              observations=int(series.count()),
              first=str(series.index[0].date()),
              last=str(series.index[-1].date()))
    return series, None


# --- validation ---------------------------------------------------------------

def validate_price(price: Optional[float]) -> Optional[str]:
    """Only impossibilities are flagged.

    Commodity prices span four orders of magnitude in one universe —
    natural gas near 2.86 and gold near 4,683 — so a range bound would
    fail on correct data. Note a futures price CAN legitimately be
    negative: WTI settled at -37.63 in April 2020. Zero and NaN are the
    real errors.
    """
    if price is None:
        return None
    if price != price:
        return "Price is not a number."
    if price == 0:
        return "Price is exactly zero, which no listed contract quotes."
    return None


def validate_inventory(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value != value:
        return "Inventory is not a number."
    if value < 0:
        return f"Inventory is negative ({value:,.0f}); stocks cannot be."
    return None


def validate_curve(curve: Curve) -> List[str]:
    notes: List[str] = []
    if not curve.points:
        return ["No contracts quoted."]
    for point in curve.points:
        note = validate_price(point.price)
        if note:
            notes.append(f"{point.symbol}: {note}")
    expiries = [p.expiry for p in curve.points]
    if expiries != sorted(expiries):
        notes.append("Contracts are not in expiry order.")
    if len(set(expiries)) != len(expiries):
        notes.append("The curve contains duplicate expiries.")
    return notes
