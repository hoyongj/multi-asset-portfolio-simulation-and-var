"""Data and pricing foundation for the approved multi-asset portfolio brief.

Sources: Federal Reserve fitted zero-coupon yields (SVENY, percentage units,
continuously compounded), and Yahoo Finance ETF prices accessed via yfinance.
https://www.federalreserve.gov/data/nominal-yield-curve.htm
https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html

Internally yields and their changes use decimal-yield units; multiply changes
by 10,000 to report basis points. ETF log returns use Adj Close exclusively.
All factors and both ETF price fields are aligned BEFORE taking differences.
No missing values are filled. Changes span consecutive retained dates; their
start dates and every dropped source date remain available for inspection.

USD 1,000,000 is a reporting normalization, not an estimated portfolio size.
Quantities and dollar P&L scale linearly with it; return and risk percentages
are invariant to this scale. Holdings are hypothetical, fractional, buy-and-hold
positions with distributions reinvested and no transaction costs or rebalancing.

Bond prices are exp(-decimal_yield * remaining_years) per USD 1 face value.
Historical aging uses actual elapsed calendar days / 365 (ACT/365 Fixed).
Explicit elapsed_years can instead express a simulation horizon: 252 modeled
trading days / 252 = 1 year, leaving maturities of 1, 4, and 9 years. Yields
are linearly interpolated in maturity; extrapolation outside 1--10 is rejected.

Importing does not retrieve data, seed global random state, or write files.
Executing retrieves and reports this foundation in memory; it writes no outputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ANALYSIS_START = "2022-09-01"
ANALYSIS_END = "2026-08-31"  # Inclusive, subject to source availability.
TREASURY_TENORS = (1, 2, 4, 5, 9, 10)
YIELD_COLUMNS = tuple(f"SVENY{tenor:02d}" for tenor in TREASURY_TENORS)
TREASURY_HOLDINGS = ("Treasury_2Y", "Treasury_5Y", "Treasury_10Y")
INITIAL_MATURITIES = (2.0, 5.0, 10.0)
ETF_TICKERS = ("LQD", "SPY")
ETF_PRICE_FIELDS = ("Close", "Adj Close")
HOLDINGS = TREASURY_HOLDINGS + ETF_TICKERS
WEIGHTS = (0.1, 0.1, 0.1, 0.3, 0.4)
INITIAL_PORTFOLIO_VALUE = 1_000_000.0
CALIBRATION_WINDOW = 500
BACKTEST_FORECASTS = 250
VAR_CONFIDENCE = 0.99
TRADING_DAYS = 252
SIMULATION_PATHS = 10_000
CALENDAR_DAYS_PER_YEAR = 365.0
BASIS_POINTS_PER_DECIMAL = 10_000.0
MASTER_SEED = 42
# Append new names only: these fixed positions identify the reserved streams.
RANDOM_STREAM_NAMES = (
    "standalone_terminal_simulation",
    "correlated_terminal_simulation",
    "rolling_parametric_var",
    "risk_attribution",
)
FED_YIELD_URL = (
    "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
)
REQUEST_TIMEOUT_SECONDS = 30
FED_MISSING_VALUES = ("NA", "N/A", "ND", ".")
# Keep yfinance's timezone/cookie caches inside this project's ignored environment.
YFINANCE_CACHE_DIRECTORY = Path(__file__).resolve().parent / ".venv" / "yfinance-cache"
LEVEL_COLUMNS = YIELD_COLUMNS + tuple(
    f"{ticker}_{field}" for ticker in ETF_TICKERS for field in ETF_PRICE_FIELDS
)


class DataValidationError(ValueError):
    """A source observation or valuation input violates the data contract."""


class DataRetrievalError(RuntimeError):
    """A required remote source could not be retrieved."""


def create_child_seeds(master_seed: int = MASTER_SEED) -> dict[str, int]:
    """Recreate deterministic, distinct named seeds without shared RNG state."""
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(
        master_seed, (int, np.integer)
    ) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    children = np.random.SeedSequence(int(master_seed)).spawn(len(RANDOM_STREAM_NAMES))
    return {
        name: int(child.generate_state(1, dtype=np.uint64)[0])
        for name, child in zip(RANDOM_STREAM_NAMES, children)
    }


CHILD_SEEDS = MappingProxyType(create_child_seeds())


@dataclass(frozen=True)
class SourceCoverage:
    """Coverage diagnostics; absent weekdays may include legitimate holidays.

    Boundary shortfalls are disclosed rather than silently extending the sample.
    No exchange or Federal Reserve holiday calendar is inferred from weekdays.
    """

    source: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    first_valid_date: pd.Timestamp
    last_valid_date: pd.Timestamp
    observation_count: int
    incomplete_dates: tuple[pd.Timestamp, ...]
    absent_weekdays: tuple[pd.Timestamp, ...]
    boundary_shortfalls: tuple[str, ...]


@dataclass(frozen=True)
class MarketData:
    """Aligned levels and changes, with complete source/alignment diagnostics."""

    levels: pd.DataFrame
    yield_changes: pd.DataFrame
    etf_log_returns: pd.DataFrame
    change_start_dates: pd.Series
    coverage: tuple[SourceCoverage, ...]
    dropped_dates: tuple[pd.Timestamp, ...]

    @property
    def risk_factor_changes(self) -> pd.DataFrame:
        """Six decimal-yield changes followed by LQD and SPY log returns."""
        return pd.concat([self.yield_changes, self.etf_log_returns], axis=1)

    @property
    def yield_changes_bp(self) -> pd.DataFrame:
        """A reporting view; internal changes remain in decimal-yield units."""
        return self.yield_changes * BASIS_POINTS_PER_DECIMAL


@dataclass(frozen=True)
class PortfolioAllocation:
    """Fixed initial dollar allocations, Treasury face amounts, and ETF shares."""

    allocation_date: pd.Timestamp
    initial_value: float
    initial_holding_values: pd.Series
    treasury_face_quantities: pd.Series
    etf_shares: pd.Series
    initial_etf_close: pd.Series
    initial_etf_adj_close: pd.Series


def _as_date(value: object, label: str) -> pd.Timestamp:
    """Parse ISO dates or datetime objects, preserving their local session date."""
    if isinstance(value, str):
        try:
            if len(value) != 10 or value[4] != "-" or value[7] != "-":
                raise ValueError("expected YYYY-MM-DD")
            value = date.fromisoformat(value)
        except ValueError as exc:
            raise DataValidationError(f"{label}: invalid ISO date {value!r}") from exc
    if not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        raise DataValidationError(f"{label}: expected a date, got {value!r}")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("missing date")
        return stamp.tz_localize(None).normalize().as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError(f"{label}: invalid date {value!r}") from exc


def _date_bounds(start: object, end: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = _as_date(start, "start"), _as_date(end, "end")
    if first > last:
        raise DataValidationError("start date must not follow end date")
    return first, last


def _numeric_columns(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Parse numeric text, retain missing cells, and reject invalid/complex data."""
    converted = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        forbidden = series.dropna().map(
            lambda x: isinstance(
                x, (bool, np.bool_, complex, np.complexfloating, date, np.datetime64)
            )
        )
        if forbidden.any():
            raise DataValidationError(f"{source}: {column} contains non-real numeric data")
        try:
            values = pd.to_numeric(series, errors="raise").to_numpy(
                dtype=float, na_value=np.nan
            )
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataValidationError(f"{source}: {column} must be numeric") from exc
        if np.isinf(values).any():
            raise DataValidationError(f"{source}: {column} contains infinite values")
        converted[column] = values
    return converted


def validate_source_frame(
    frame: pd.DataFrame,
    required_columns: Sequence[str],
    source: str,
    start: object = ANALYSIS_START,
    end: object = ANALYSIS_END,
    *,
    price_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, SourceCoverage]:
    """Validate a daily source, clip inclusively, and disclose coverage/missingness.

    Invalid dates, duplicate dates/columns, malformed numbers, infinities, and
    nonpositive observed prices fail. Missing observations remain for joint
    alignment, with their dates recorded. Empty or wholly incomplete sources fail.
    """
    first, last = _date_bounds(start, end)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError(f"{source}: no observations")
    if frame.columns.has_duplicates:
        raise DataValidationError(f"{source}: duplicate columns")
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise DataValidationError(f"{source}: missing required columns {sorted(missing)}")
    data = frame.loc[:, list(required_columns)].copy()
    data.index = pd.DatetimeIndex([_as_date(x, source) for x in data.index], name="Date")
    if data.index.has_duplicates:
        raise DataValidationError(f"{source}: duplicate dates")
    data = data.sort_index().loc[first:last]
    if data.empty:
        raise DataValidationError(f"{source}: no observations in requested date range")
    data = _numeric_columns(data, source)
    for column in price_columns:
        if column not in data.columns:
            raise DataValidationError(f"{source}: missing price column {column}")
        if (data[column].dropna() <= 0).any():
            raise DataValidationError(f"{source}: {column} must be strictly positive")
    incomplete = data.isna().any(axis=1)
    valid = data.loc[~incomplete]
    if valid.empty:
        raise DataValidationError(f"{source}: no complete valid observations")
    weekdays = pd.bdate_range(first, last)
    shortfalls = []
    if len(weekdays):
        if valid.index[0] > weekdays[0]:
            shortfalls.append("start")
        if valid.index[-1] < weekdays[-1]:
            shortfalls.append("end")
    coverage = SourceCoverage(
        source, first, last, valid.index[0], valid.index[-1], len(data),
        tuple(data.index[incomplete]), tuple(weekdays.difference(data.index)),
        tuple(shortfalls),
    )
    return data, coverage


def parse_fed_csv(text: str) -> pd.DataFrame:
    """Find the actual Date header after the Federal Reserve CSV's prose preamble."""
    lines = text.lstrip("\ufeff").splitlines()
    for offset, line in enumerate(lines):
        header = next(csv.reader([line]), [])
        if header and header[0].strip() == "Date":
            if len(header) != len(set(header)):
                raise DataValidationError("Federal Reserve: duplicate CSV columns")
            try:
                frame = pd.read_csv(
                    StringIO("\n".join(lines[offset:])), na_values=list(FED_MISSING_VALUES)
                )
            except (ValueError, pd.errors.ParserError) as exc:
                raise DataValidationError("Federal Reserve: malformed CSV") from exc
            missing = {"Date", *YIELD_COLUMNS}.difference(frame.columns)
            if missing:
                raise DataValidationError(f"Federal Reserve: missing columns {sorted(missing)}")
            return frame.set_index("Date").loc[:, list(YIELD_COLUMNS)]
    raise DataValidationError("Federal Reserve: Date header not found in CSV")


def fetch_treasury_yields(
    start: object = ANALYSIS_START, end: object = ANALYSIS_END
) -> pd.DataFrame:
    """Retrieve required fitted yields with requests; return source percentage units."""
    _date_bounds(start, end)
    try:
        response = requests.get(FED_YIELD_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataRetrievalError(f"Federal Reserve retrieval failed: {exc}") from exc
    frame, _ = validate_source_frame(
        parse_fed_csv(response.text), YIELD_COLUMNS, "Federal Reserve", start, end
    )
    return frame


def fetch_etf_prices(
    start: object = ANALYSIS_START, end: object = ANALYSIS_END
) -> dict[str, pd.DataFrame]:
    """Retrieve unadjusted daily downloads retaining BOTH Close and Adj Close.

    Yahoo's end is exclusive, so request the calendar day after our inclusive end.
    Single-symbol downloads use flat columns and preserve missing source rows.
    """
    first, last = _date_bounds(start, end)
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIRECTORY))
    result = {}
    for ticker in ETF_TICKERS:
        try:
            frame = yf.download(
                ticker, start=first.strftime("%Y-%m-%d"),
                end=(last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=False, back_adjust=False, repair=False,
                keepna=True, actions=False, threads=False, progress=False,
                ignore_tz=True, multi_level_index=False, timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise DataRetrievalError(f"Yahoo Finance {ticker} retrieval failed: {exc}") from exc
        result[ticker], _ = validate_source_frame(
            frame, ETF_PRICE_FIELDS, f"Yahoo Finance {ticker}", first, last,
            price_columns=ETF_PRICE_FIELDS,
        )
    return result


def prepare_market_data(
    yields_percent: pd.DataFrame,
    etf_prices: Mapping[str, pd.DataFrame],
    start: object = ANALYSIS_START,
    end: object = ANALYSIS_END,
) -> MarketData:
    """Validate raw source units, align every level, then build common-date changes.

    No returns or dependence may be estimated from pre-alignment observations.
    Dropped dates are the union of supplied in-range dates minus common valid
    dates; absent weekdays and incomplete source rows are separately disclosed.
    """
    if set(etf_prices) != set(ETF_TICKERS):
        raise DataValidationError("ETF sources must contain exactly LQD and SPY")
    yields, yield_coverage = validate_source_frame(
        yields_percent, YIELD_COLUMNS, "Federal Reserve", start, end
    )
    frames = [yields / 100.0]
    reports = [yield_coverage]
    for ticker in ETF_TICKERS:
        frame, report = validate_source_frame(
            etf_prices[ticker], ETF_PRICE_FIELDS, f"Yahoo Finance {ticker}", start, end,
            price_columns=ETF_PRICE_FIELDS,
        )
        frames.append(frame.add_prefix(f"{ticker}_"))
        reports.append(report)
    joined = pd.concat(frames, axis=1).sort_index()
    levels = joined.dropna().loc[:, list(LEVEL_COLUMNS)]
    if len(levels) < 2:
        raise DataValidationError("At least two common valid dates are required for changes")
    yield_changes = levels.loc[:, list(YIELD_COLUMNS)].diff().iloc[1:]
    adjusted = pd.DataFrame(
        {ticker: levels[f"{ticker}_Adj Close"] for ticker in ETF_TICKERS}
    )
    etf_log_returns = np.log(adjusted).diff().iloc[1:]
    return MarketData(
        levels, yield_changes, etf_log_returns,
        pd.Series(levels.index[:-1], index=levels.index[1:], name="previous_valid_date"),
        tuple(reports), tuple(joined.index.difference(levels.index)),
    )


def retrieve_market_data(
    start: object = ANALYSIS_START, end: object = ANALYSIS_END
) -> MarketData:
    """Retrieve only the specified sources and construct the validated foundation."""
    yields = fetch_treasury_yields(start, end)
    prices = fetch_etf_prices(start, end)
    return prepare_market_data(yields, prices, start, end)


def _real_array(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise DataValidationError(f"{label} must contain finite real numbers")
    return array.astype(float)


def zero_coupon_price(yield_decimal: object, remaining_maturity: object) -> float | np.ndarray:
    """Continuous-compounding price per USD 1 face; scalar or broadcastable arrays."""
    yields = _real_array(yield_decimal, "yield_decimal")
    maturity = _real_array(remaining_maturity, "remaining_maturity")
    if (maturity < 0).any():
        raise DataValidationError("remaining_maturity must be nonnegative")
    try:
        with np.errstate(over="raise", invalid="raise"):
            prices = np.exp(-yields * maturity)
    except (FloatingPointError, ValueError) as exc:
        raise DataValidationError("Zero-coupon price is not representable") from exc
    if (prices <= 0).any():
        raise DataValidationError("Zero-coupon price underflowed to zero")
    return float(prices) if prices.ndim == 0 else prices


def interpolate_yield(curve: Mapping[int, float], remaining_maturity: float) -> float:
    """Interpolate decimal yields between adjacent fitted tenors; never extrapolate.

    In particular, aging our holdings by up to one year uses the 1--2, 4--5,
    and 9--10 pairs, reaching exactly the 1-, 4-, and 9-year yields at year one.
    """
    missing = set(TREASURY_TENORS).difference(curve)
    if missing:
        raise DataValidationError(f"Missing fitted tenors: {sorted(missing)}")
    maturity = _real_array(remaining_maturity, "remaining_maturity")
    if maturity.ndim != 0 or not TREASURY_TENORS[0] <= maturity <= TREASURY_TENORS[-1]:
        raise DataValidationError("Maturity must be a scalar within fitted tenors [1, 10]")
    yields = _real_array([curve[tenor] for tenor in TREASURY_TENORS], "curve")
    return float(np.interp(float(maturity), TREASURY_TENORS, yields))


def _valuation_row(levels: pd.DataFrame, valuation_date: object) -> tuple[pd.Timestamp, pd.Series]:
    stamp = _as_date(valuation_date, "valuation_date")
    if not isinstance(levels.index, pd.DatetimeIndex) or levels.index.hasnans:
        raise DataValidationError("Valuation levels require a nonmissing DatetimeIndex")
    if levels.index.tz is not None or not levels.index.equals(levels.index.normalize()):
        raise DataValidationError("Valuation levels require timezone-naive daily dates")
    if levels.index.has_duplicates or levels.columns.has_duplicates:
        raise DataValidationError("Valuation levels contain duplicate dates or columns")
    if set(LEVEL_COLUMNS).difference(levels.columns):
        raise DataValidationError("Valuation levels lack required yield or ETF columns")
    if stamp not in levels.index:
        raise DataValidationError(f"No common valid observation on {stamp.date()}")
    row = _numeric_columns(levels.loc[[stamp], list(LEVEL_COLUMNS)], "valuation").iloc[0]
    if row.isna().any():
        raise DataValidationError("Valuation date has missing observations")
    if (row.loc[[x for x in LEVEL_COLUMNS if x not in YIELD_COLUMNS]] <= 0).any():
        raise DataValidationError("ETF valuation prices must be strictly positive")
    return stamp, row


def treasury_unit_prices(
    levels: pd.DataFrame,
    valuation_date: object,
    remaining_maturities: Sequence[float] = INITIAL_MATURITIES,
) -> pd.Series:
    """Price each Treasury on an exact common date using supplied remaining years."""
    _, row = _valuation_row(levels, valuation_date)
    maturities = _real_array(remaining_maturities, "remaining_maturities")
    if maturities.shape != (len(TREASURY_HOLDINGS),):
        raise DataValidationError("Provide three remaining Treasury maturities")
    curve = dict(zip(TREASURY_TENORS, row.loc[list(YIELD_COLUMNS)]))
    yields = [interpolate_yield(curve, maturity) for maturity in maturities]
    return pd.Series(
        zero_coupon_price(yields, maturities), index=TREASURY_HOLDINGS, name="unit_price"
    )


def allocate_portfolio(
    levels: pd.DataFrame,
    allocation_date: object,
    initial_value: float = INITIAL_PORTFOLIO_VALUE,
) -> PortfolioAllocation:
    """Allocate fixed weights using Treasury unit prices and initial ETF Close."""
    stamp, row = _valuation_row(levels, allocation_date)
    amount = _real_array(initial_value, "initial_value")
    if amount.ndim != 0 or amount <= 0:
        raise DataValidationError("initial_value must be a positive scalar")
    if len(WEIGHTS) != len(HOLDINGS) or not np.isclose(sum(WEIGHTS), 1.0):
        raise DataValidationError("Configured weights must match holdings and sum to one")
    initial = pd.Series(np.asarray(WEIGHTS) * float(amount), index=HOLDINGS, name="USD")
    close = pd.Series({ticker: row[f"{ticker}_Close"] for ticker in ETF_TICKERS})
    adjusted = pd.Series({ticker: row[f"{ticker}_Adj Close"] for ticker in ETF_TICKERS})
    return PortfolioAllocation(
        stamp, float(amount), initial,
        initial.loc[list(TREASURY_HOLDINGS)] / treasury_unit_prices(levels, stamp),
        initial.loc[list(ETF_TICKERS)] / close, close, adjusted,
    )


def holding_values(
    allocation: PortfolioAllocation,
    levels: pd.DataFrame,
    valuation_date: object,
    *,
    elapsed_years: float | None = None,
) -> pd.Series:
    """Value fixed holdings on an exact date, with no rebalancing or date fallback.

    Default aging is ACT/365 Fixed from the allocation date. A supplied nonnegative
    elapsed_years supports explicit modeled horizons. Remaining maturities must
    stay within fitted coverage. Subsequent ETF Close never enters a holding value:
    initial shares * initial Close * (current Adj Close / initial Adj Close).
    """
    stamp, row = _valuation_row(levels, valuation_date)
    if stamp < allocation.allocation_date:
        raise DataValidationError("valuation_date must not precede allocation_date")
    if elapsed_years is None:
        elapsed_years = (stamp - allocation.allocation_date).days / CALENDAR_DAYS_PER_YEAR
    age = _real_array(elapsed_years, "elapsed_years")
    if age.ndim != 0 or age < 0:
        raise DataValidationError("elapsed_years must be a nonnegative scalar")
    maturities = np.asarray(INITIAL_MATURITIES) - float(age)
    bonds = allocation.treasury_face_quantities * treasury_unit_prices(levels, stamp, maturities)
    current_adjusted = pd.Series(
        {ticker: row[f"{ticker}_Adj Close"] for ticker in ETF_TICKERS}
    )
    etfs = (
        allocation.etf_shares * allocation.initial_etf_close
        * (current_adjusted / allocation.initial_etf_adj_close)
    )
    return pd.concat([bonds, etfs]).reindex(HOLDINGS).rename("USD")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the in-memory data/pricing foundation; no simulations or files are produced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation-date", help="Exact common date; default: first valid date")
    parser.add_argument("--valuation-date", help="Exact common date; default: allocation date")
    args = parser.parse_args(argv)
    try:
        market = retrieve_market_data()
        allocation_date = args.allocation_date or market.levels.index[0]
        allocation = allocate_portfolio(market.levels, allocation_date)
        valuation_date = args.valuation_date or allocation.allocation_date
        values = holding_values(allocation, market.levels, valuation_date)
    except (DataValidationError, DataRetrievalError) as exc:
        parser.exit(1, f"Data/pricing foundation failed: {exc}\n")
    print("Sources: Federal Reserve fitted zero-coupon yields; Yahoo Finance via yfinance.")
    print(f"Requested inclusive dates: {ANALYSIS_START} through {ANALYSIS_END}")
    print(
        f"Common valid dates: {market.levels.index[0].date()} through "
        f"{market.levels.index[-1].date()}; {len(market.levels)} levels, "
        f"{len(market.risk_factor_changes)} aligned changes."
    )
    for report in market.coverage:
        print(
            f"{report.source}: valid coverage {report.first_valid_date.date()} through "
            f"{report.last_valid_date.date()}; boundary shortfalls: "
            f"{', '.join(report.boundary_shortfalls) or 'none'}"
        )
        for label, dates in (
            ("Incomplete observation dates", report.incomplete_dates),
            ("Absent weekdays (may be holidays)", report.absent_weekdays),
        ):
            print(f"  {label}: {', '.join(str(x.date()) for x in dates) or 'none'}")
    print("Dates dropped in alignment: " + (
        ", ".join(str(x.date()) for x in market.dropped_dates) or "none"
    ))
    print(
        f"Reporting normalization: USD {allocation.initial_value:,.2f}; "
        "return and risk percentages are invariant to this scale."
    )
    print(f"Allocation date: {allocation.allocation_date.date()}")
    print(f"Holding values on {_as_date(valuation_date, 'valuation_date').date()} (USD):")
    print(values.to_string(float_format=lambda value: f"{value:,.2f}"))
    print(f"Portfolio value: USD {values.sum():,.2f}")
    print("Latest aligned yield changes (basis points):")
    print(market.yield_changes_bp.tail(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
