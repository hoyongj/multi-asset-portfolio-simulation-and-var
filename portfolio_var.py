"""Data, pricing, and Gaussian terminal simulation for the approved project brief.

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

Gaussian calibration uses the latest 500 common-date changes through the chosen
allocation date, capped at 2026-08-31. Means, sample standard deviations, and sample
covariance use the same observations (ddof=1). Daily draws are IID through time.
Standalone draws are independent marginal analyses; their summed portfolio is an
independence diagnostic. The portfolio analysis uses the full eight-factor model.

Rolling backtests allocate once at the first forecast origin and age bonds by
aligned steps/252 for both scenario and realized valuations. Each forecast excludes
its realized change from the immediately preceding 500-change estimation window.
VaR is a signed 99% dollar loss quantile, and exceptions strictly exceed it.
Latest-forecast Euler attribution uses common full-revaluation scenarios and
central proportional position bumps, reporting the unnormalized residual.

Importing does not retrieve data, seed global random state, or write analysis files.
Executing runs the complete analysis and writes the nine configured result files
only after analysis, table validation, and in-memory figure rendering succeed.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.special import xlog1py, xlogy
from scipy.stats import chi2


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
RISK_FACTOR_COLUMNS = YIELD_COLUMNS + ETF_TICKERS
TERMINAL_HORIZON_YEARS = TRADING_DAYS / TRADING_DAYS
TERMINAL_MATURITIES = tuple(maturity - TERMINAL_HORIZON_YEARS for maturity in INITIAL_MATURITIES)
TERMINAL_YIELD_COLUMNS = tuple(f"SVENY{int(tenor):02d}" for tenor in TERMINAL_MATURITIES)
STANDALONE_FACTOR_COLUMNS = TERMINAL_YIELD_COLUMNS + ETF_TICKERS
ESTIMATION_DDOF = 1
PSD_ROUNDOFF_MULTIPLIER = 10.0
SUMMARY_PERCENTILES = (0.01, 0.5, 0.99)
VAR_METHODS = ("parametric", "historical")
RISK_IDENTIFIERS = HOLDINGS + ("Portfolio",)
VAR_QUANTILE_METHOD = "linear"
ATTRIBUTION_RELATIVE_STEP = 1e-4
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
PROJECT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
CSV_FILENAMES = ("data_quality.csv", "terminal_paths.csv", "terminal_summary.csv",
                 "var_forecasts.csv", "var_summary.csv", "risk_summary.csv")
FIGURE_FILENAMES = ("terminal_distributions.png", "var_backtests.png", "risk_contributions.png")
OUTPUT_FILENAMES = CSV_FILENAMES + FIGURE_FILENAMES
FIGURE_DPI = 150
PLOT_HOLDING_LABELS = ("2-year Treasury", "5-year Treasury", "10-year Treasury", "LQD", "SPY", "Portfolio")


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


def create_rng(purpose: str, master_seed: int = MASTER_SEED) -> np.random.Generator:
    """Create a fresh reproducible stream, including reserved VaR/attribution streams.

    Consumption in one purpose never advances another purpose's generator.
    Reproducibility assumes the same pinned numerical libraries/build.
    """
    seeds = create_child_seeds(master_seed)
    if purpose not in seeds:
        raise ValueError(f"Unknown random stream {purpose!r}; expected {tuple(seeds)}")
    return np.random.default_rng(seeds[purpose])


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
    missing_value_count: int


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


@dataclass(frozen=True)
class CovarianceAdjustment:
    """Covariance used for simulation and the complete numerical correction record."""

    covariance: np.ndarray
    factor: np.ndarray
    original_min_eigenvalue: float
    tolerance: float
    clipped_eigenvalues: int
    correction_frobenius_norm: float


@dataclass(frozen=True)
class GaussianCalibration:
    """Eight ordered Gaussian marginals and joint covariance from one 500-row sample."""

    cutoff_date: pd.Timestamp
    sample: pd.DataFrame
    mean: pd.Series
    std: pd.Series
    sample_covariance: pd.DataFrame
    covariance: pd.DataFrame
    covariance_adjustment: CovarianceAdjustment


@dataclass(frozen=True)
class TerminalSimulation:
    """Daily changes and terminal results, without filesystem output.

    daily_changes has (path, day) rows. paths has one row per path and (holding,
    metric) columns. unit_price is per USD 1 Treasury face or per initial ETF
    share (initial Close times the adjusted-price ratio). adjusted_price separately
    records each ETF's simulated allocation-date Adj Close times that ratio.
    value and return are holding dollars and simple returns; Portfolio has their
    sum and total return. summary describes each metric and records its initial
    value. The standalone Portfolio sum assumes independent marginal draws.
    """

    purpose: str
    seed: int
    horizon_days: int
    allocation: PortfolioAllocation
    quantities: pd.Series
    initial_unit_prices: pd.Series
    terminal_maturities: pd.Series
    daily_changes: pd.DataFrame
    terminal_yields: pd.DataFrame
    paths: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class TerminalAnalysis:
    """Common calibration with standalone and correlated terminal results."""

    calibration: GaussianCalibration
    standalone: TerminalSimulation
    correlated: TerminalSimulation


@dataclass(frozen=True)
class VaRAttribution:
    """Common-scenario Euler finite differences, in USD, with an unscaled residual.

    residual = portfolio_var - contribution_sum. Contributions are never rescaled
    to force reconciliation; a finite bump can cross empirical quantile ranks.
    """

    contributions: pd.DataFrame
    portfolio_var: float
    contribution_sum: float
    residual: float
    relative_step: float


@dataclass(frozen=True)
class BacktestResult:
    """Rolling forecasts and diagnostics; all tables remain in memory.

    Backtest aging uses aligned steps / 252 from the first forecast origin,
    including for realized valuations. This explicit trading clock differs from
    holding_values' default ACT/365 clock: source-calendar gaps can make 250
    aligned changes span more than 365 days, beyond the supplied 1-year tenor.
    Each forecast ages all bonds by exactly 1/252 year; no extrapolation occurs.
    """

    allocation: PortfolioAllocation
    quantities: pd.Series
    valuations: pd.DataFrame
    weights: pd.DataFrame
    remaining_maturities: pd.DataFrame
    windows: pd.DataFrame
    records: pd.DataFrame
    statistics: pd.DataFrame
    latest_scenarios: Mapping[str, pd.DataFrame]
    latest_losses: Mapping[str, pd.DataFrame]
    diversification: pd.DataFrame
    attribution: VaRAttribution


@dataclass(frozen=True)
class CompleteAnalysis:
    """One source snapshot shared by every terminal, backtest, and reporting result."""

    market: MarketData
    terminal: TerminalAnalysis
    backtest: BacktestResult


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
        tuple(shortfalls), int(data.isna().sum().sum()),
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


def stabilize_covariance(covariance: object) -> CovarianceAdjustment:
    """Factor a PSD covariance, correcting only floating-point roundoff.

    Symmetry and negative-eigenvalue tolerances are 10 * n * machine epsilon
    times the matrix/eigenvalue scale, respectively. Material asymmetry or
    indefiniteness fails. Average roundoff asymmetry, then clip only negative
    eigenvalues within tolerance to zero: this is the smallest Frobenius-norm
    PSD correction of the symmetric matrix. No diagonal jitter, positive
    eigenvalue floor, shrinkage, or statistical regularization is applied.
    Eigen factor A simulates mean + z @ A.T, including singular covariances.
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
    """
    original = _real_array(covariance, "covariance")
    if original.ndim != 2 or original.shape[0] != original.shape[1] or not original.size:
        raise DataValidationError("covariance must be a nonempty square matrix")
    relative_tolerance = PSD_ROUNDOFF_MULTIPLIER * len(original) * np.finfo(float).eps
    scale = float(np.max(np.abs(original)))
    if np.max(np.abs(original - original.T)) > relative_tolerance * scale:
        raise DataValidationError("Covariance has material asymmetry")
    symmetric = original / 2 + original.T / 2
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise DataValidationError("Covariance eigen decomposition failed") from exc
    tolerance = relative_tolerance * float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -tolerance:
        raise DataValidationError("Covariance is materially indefinite; numerical repair refused")
    negative_count = int(np.count_nonzero(eigenvalues < 0))
    clipped = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors * np.sqrt(clipped)
    corrected = factor @ factor.T if negative_count else symmetric
    return CovarianceAdjustment(
        corrected, factor, float(eigenvalues[0]), tolerance, negative_count,
        float(np.linalg.norm(corrected - original, ord="fro")),
    )


def calibrate_gaussian(
    aligned_changes: pd.DataFrame, valuation_date: object = ANALYSIS_END
) -> GaussianCalibration:
    """Fit all eight factors on exactly the latest 500 aligned changes by cutoff.

    Input units are decimal-yield changes and adjusted-price log returns. Sample
    std/covariance use ddof=1 consistently. Missing changes fail rather than being
    filled or silently removed. Changes after the inclusive cutoff are excluded.
    """
    cutoff = _as_date(valuation_date, "valuation_date")
    if not _as_date(ANALYSIS_START, "start") <= cutoff <= _as_date(ANALYSIS_END, "end"):
        raise DataValidationError("Calibration cutoff must be within the analysis period")
    if not isinstance(aligned_changes, pd.DataFrame) or set(aligned_changes.columns) != set(RISK_FACTOR_COLUMNS):
        raise DataValidationError("Calibration requires exactly the eight specified risk factors")
    changes, _ = validate_source_frame(
        aligned_changes, RISK_FACTOR_COLUMNS, "calibration", ANALYSIS_START, cutoff
    )
    if changes.isna().any().any():
        raise DataValidationError("Calibration changes must already be aligned and complete")
    if len(changes) < CALIBRATION_WINDOW:
        raise DataValidationError(f"Calibration requires {CALIBRATION_WINDOW} aligned daily changes")
    sample = changes.tail(CALIBRATION_WINDOW).copy()
    mean = sample.mean().rename("daily_mean")
    std = sample.std(ddof=ESTIMATION_DDOF).rename("daily_std")
    sample_covariance = sample.cov(ddof=ESTIMATION_DDOF)
    adjustment = stabilize_covariance(sample_covariance.to_numpy())
    used_covariance = pd.DataFrame(
        adjustment.covariance, index=RISK_FACTOR_COLUMNS, columns=RISK_FACTOR_COLUMNS
    )
    return GaussianCalibration(
        cutoff, sample, mean, std, sample_covariance, used_covariance, adjustment
    )


def _simulate_terminal(
    calibration: GaussianCalibration,
    allocation: PortfolioAllocation,
    levels: pd.DataFrame,
    purpose: str,
    path_count: int,
    master_seed: int,
) -> TerminalSimulation:
    if isinstance(path_count, (bool, np.bool_)) or not isinstance(path_count, (int, np.integer)) or path_count <= 0:
        raise DataValidationError("path_count must be a positive integer")
    stamp, row = _valuation_row(levels, allocation.allocation_date)
    if stamp > _as_date(ANALYSIS_END, "end"):
        raise DataValidationError("Allocation must not exceed the analysis end date")
    if calibration.sample.index[-1] > stamp:
        raise DataValidationError("Calibration uses changes after the allocation date")
    correlated = purpose == "correlated_terminal_simulation"
    factors = RISK_FACTOR_COLUMNS if correlated else STANDALONE_FACTOR_COLUMNS
    mean = calibration.mean.loc[list(factors)].to_numpy()
    std = calibration.std.loc[list(factors)].to_numpy()
    rng = create_rng(purpose, master_seed)
    daily = np.empty((path_count, TRADING_DAYS, len(factors)), dtype=float)
    # Generate one day at a time to avoid a second full 10,000 x 252 x 8 array.
    for day in range(TRADING_DAYS):
        innovations = rng.standard_normal((path_count, len(factors)))
        if correlated:
            daily[:, day, :] = innovations @ calibration.covariance_adjustment.factor.T + mean
        else:
            daily[:, day, :] = innovations * std + mean
    if not np.isfinite(daily).all():
        raise DataValidationError("Simulated daily changes are not finite")
    path_index = pd.RangeIndex(1, path_count + 1, name="path")
    daily_index = pd.MultiIndex.from_product(
        [path_index, range(1, TRADING_DAYS + 1)], names=["path", "day"]
    )
    daily_changes = pd.DataFrame(daily.reshape(-1, len(factors)), index=daily_index, columns=factors)
    cumulative = pd.DataFrame(daily.sum(axis=1), index=path_index, columns=factors)
    yield_factors = [factor for factor in factors if factor in YIELD_COLUMNS]
    terminal_yields = cumulative.loc[:, yield_factors].add(row.loc[yield_factors], axis=1)
    bond_prices = zero_coupon_price(
        terminal_yields.loc[:, list(TERMINAL_YIELD_COLUMNS)].to_numpy(),
        np.asarray(TERMINAL_MATURITIES),
    )
    # Initial Adj Close anchors simulated adjusted prices. Converting back to a
    # value per initial observed share retains the original Close-based quantity.
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            adjusted_prices = np.exp(cumulative.loc[:, list(ETF_TICKERS)]).mul(
                allocation.initial_etf_adj_close, axis=1
            )
            etf_unit_prices = adjusted_prices.div(allocation.initial_etf_adj_close, axis=1).mul(
                allocation.initial_etf_close, axis=1
            )
            unit_prices = pd.DataFrame(
                np.column_stack([bond_prices, etf_unit_prices.to_numpy()]),
                index=path_index, columns=HOLDINGS,
            )
            quantities = pd.concat([
                allocation.treasury_face_quantities, allocation.etf_shares
            ]).reindex(HOLDINGS).rename("fixed_quantity")
            values = unit_prices.mul(quantities, axis=1)
            returns = values.div(allocation.initial_holding_values, axis=1) - 1.0
    except FloatingPointError as exc:
        raise DataValidationError("Terminal values are not representable; paths were not clipped") from exc
    initial_prices = pd.concat([
        treasury_unit_prices(levels, stamp), allocation.initial_etf_close
    ]).reindex(HOLDINGS).rename("initial_unit_price")
    columns, initial = {}, {}
    for holding in HOLDINGS:
        for metric, series, initial_value in (
            ("unit_price", unit_prices[holding], initial_prices[holding]),
            ("value", values[holding], allocation.initial_holding_values[holding]),
            ("return", returns[holding], 0.0),
        ):
            columns[holding, metric] = series
            initial[holding, metric] = initial_value
        if holding in ETF_TICKERS:
            columns[holding, "adjusted_price"] = adjusted_prices[holding]
            initial[holding, "adjusted_price"] = allocation.initial_etf_adj_close[holding]
    columns["Portfolio", "value"] = values.sum(axis=1)
    columns["Portfolio", "return"] = columns["Portfolio", "value"] / allocation.initial_value - 1.0
    initial["Portfolio", "value"], initial["Portfolio", "return"] = allocation.initial_value, 0.0
    paths = pd.DataFrame(columns, index=path_index)
    paths.columns = pd.MultiIndex.from_tuples(paths.columns, names=["holding", "metric"])
    if not np.isfinite(paths.to_numpy()).all() or (unit_prices.to_numpy() <= 0).any():
        raise DataValidationError("Terminal prices, values, or returns are invalid")
    summary = paths.describe(percentiles=list(SUMMARY_PERCENTILES)).T
    summary.insert(0, "initial", pd.Series(initial))
    return TerminalSimulation(
        purpose, create_child_seeds(master_seed)[purpose], TRADING_DAYS, allocation,
        quantities, initial_prices, pd.Series(TERMINAL_MATURITIES, index=TREASURY_HOLDINGS,
                                             name="remaining_years"),
        daily_changes, terminal_yields, paths, summary,
    )


def simulate_standalone_terminal(
    calibration: GaussianCalibration,
    allocation: PortfolioAllocation,
    levels: pd.DataFrame,
    *,
    path_count: int = SIMULATION_PATHS,
    master_seed: int = MASTER_SEED,
) -> TerminalSimulation:
    """Simulate five independent marginals for 252 days, then fully reprice.

    Treasury shocks use the 1-, 4-, and 9-year yield marginals and their initial
    curve levels, while returns retain the initial 2-, 5-, and 10-year prices.
    Standalone Portfolio totals are the sum of these independent draws.
    """
    return _simulate_terminal(
        calibration, allocation, levels, "standalone_terminal_simulation", path_count, master_seed
    )


def simulate_correlated_terminal(
    calibration: GaussianCalibration,
    allocation: PortfolioAllocation,
    levels: pd.DataFrame,
    *,
    path_count: int = SIMULATION_PATHS,
    master_seed: int = MASTER_SEED,
) -> TerminalSimulation:
    """Simulate all six yield factors and both ETF log returns jointly for 252 days."""
    return _simulate_terminal(
        calibration, allocation, levels, "correlated_terminal_simulation", path_count, master_seed
    )


def run_terminal_analysis(
    market: MarketData,
    allocation_date: object | None = None,
    *,
    path_count: int = SIMULATION_PATHS,
    master_seed: int = MASTER_SEED,
) -> TerminalAnalysis:
    """Calibrate and run both terminal analyses in memory at one common date.

    Default: latest 500 changes available through 2026-08-31, allocated at the
    last common valid observation. Explicit allocation dates use only changes
    through that date and must themselves be common valid observations.
    """
    cutoff = ANALYSIS_END if allocation_date is None else allocation_date
    calibration = calibrate_gaussian(market.risk_factor_changes, cutoff)
    stamp = calibration.sample.index[-1] if allocation_date is None else allocation_date
    allocation = allocate_portfolio(market.levels, stamp)
    standalone = simulate_standalone_terminal(
        calibration, allocation, market.levels, path_count=path_count, master_seed=master_seed
    )
    correlated = simulate_correlated_terminal(
        calibration, allocation, market.levels, path_count=path_count, master_seed=master_seed
    )
    return TerminalAnalysis(calibration, standalone, correlated)


def _integer_count(value: int, label: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise DataValidationError(f"{label} must be an integer >= {minimum}")
    return int(value)


def rolling_forecast_seed(forecast_ordinal: int, master_seed: int = MASTER_SEED) -> int:
    """Key each forecast to its global eligible ordinal, independent of call order.

    Ordinal zero is the forecast immediately following the first 500 changes;
    later selection of the latest 250 forecasts does not renumber these streams.
    """
    ordinal = _integer_count(forecast_ordinal, "forecast_ordinal", minimum=0)
    parent = create_child_seeds(master_seed)["rolling_parametric_var"]
    child = np.random.SeedSequence(parent, spawn_key=(ordinal,))
    return int(child.generate_state(1, dtype=np.uint64)[0])


def create_forecast_rng(forecast_ordinal: int, master_seed: int = MASTER_SEED) -> np.random.Generator:
    """A fresh one-day Gaussian scenario generator for a particular forecast."""
    return np.random.default_rng(rolling_forecast_seed(forecast_ordinal, master_seed))


def _binary_sequence(exceptions: object) -> np.ndarray:
    values = np.asarray(exceptions)
    if (values.ndim != 1 or not values.size or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all()):
        raise DataValidationError("Exceptions must be a nonempty one-dimensional binary sequence")
    return values.astype(int)


def _bernoulli_loglik(zeros: int, ones: int, probability: float) -> float:
    # xlogy/xlog1py implement the continuous boundary convention 0 * log(0) = 0.
    return float(xlogy(ones, probability) + xlog1py(zeros, -probability))


def kupiec_test(exceptions: object) -> dict[str, float | int]:
    """Unconditional coverage LR using all forecasts and nominal exception rate 1%.

    Log likelihoods use xlogy/xlog1py, so all-zero/all-one samples remain finite.
    P-values use chi-square survival functions (one degree of freedom), avoiding
    subtraction of a CDF near one. These are asymptotic, including at boundaries.
    """
    sequence = _binary_sequence(exceptions)
    n, hits = len(sequence), int(sequence.sum())
    null = _bernoulli_loglik(n - hits, hits, 1.0 - VAR_CONFIDENCE)
    alternative = _bernoulli_loglik(n - hits, hits, hits / n)
    lr = max(0.0, 2.0 * (alternative - null))  # LR cannot be negative; remove roundoff.
    return {"forecasts": n, "exception_count": hits, "exception_rate": hits / n,
            "kupiec_lr": lr, "kupiec_p_value": float(chi2.sf(lr, 1))}


def christoffersen_test(exceptions: object) -> dict[str, float | int | bool]:
    """First-order independence and LRcc = LRuc + LRind (df 1 and 2).

    Independence conditions on the first observation: transition likelihoods use
    T-1 transitions, with a pooled transition MLE under independence. A transition
    row never visited contributes zero log likelihood; its unidentified probability
    is NaN. Boundary probabilities use 0*log(0)=0 without smoothing/pseudocounts.
    An all-zero or all-one sequence thus has LRind=0. Standard asymptotic p-values
    are retained, with independence_identified=False when a row is unobserved.
    For a singleton there are no transitions: independence/CC statistics and their
    p-values are NaN. UC still uses all T observations. No observation is invented.
    Christoffersen (1998), Evaluating Interval Forecasts, doi:10.2307/2527341.
    """
    sequence = _binary_sequence(exceptions)
    counts = np.bincount(2 * sequence[:-1] + sequence[1:], minlength=4)
    n00, n01, n10, n11 = (int(count) for count in counts)
    p01 = n01 / (n00 + n01) if n00 + n01 else np.nan
    p11 = n11 / (n10 + n11) if n10 + n11 else np.nan
    lr_ind = lr_cc = p_ind = p_cc = np.nan
    if len(sequence) > 1:
        pooled = (n01 + n11) / (len(sequence) - 1)
        null = _bernoulli_loglik(n00 + n10, n01 + n11, pooled)
        alternative = 0.0
        if n00 + n01:
            alternative += _bernoulli_loglik(n00, n01, p01)
        if n10 + n11:
            alternative += _bernoulli_loglik(n10, n11, p11)
        lr_ind = max(0.0, 2.0 * (alternative - null))
        lr_cc = float(kupiec_test(sequence)["kupiec_lr"]) + lr_ind
        p_ind, p_cc = float(chi2.sf(lr_ind, 1)), float(chi2.sf(lr_cc, 2))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "p01": p01, "p11": p11, "independence_identified": bool(n00 + n01 and n10 + n11),
            "christoffersen_ind_lr": lr_ind, "christoffersen_ind_p_value": p_ind,
            "christoffersen_cc_lr": lr_cc, "christoffersen_cc_p_value": p_cc}


def loss_var(losses: pd.DataFrame) -> pd.Series:
    """99% dollar loss quantiles, using linear empirical quantile interpolation.

    Loss = origin value - next-day value = -P&L. VaR is signed and is not floored
    at zero: a sufficiently profitable distribution can have negative loss VaR.
    """
    if not isinstance(losses, pd.DataFrame) or losses.empty or losses.columns.has_duplicates:
        raise DataValidationError("Loss scenarios require a nonempty DataFrame with unique columns")
    values = _real_array(losses.to_numpy(), "losses")
    quantiles = np.quantile(values, VAR_CONFIDENCE, axis=0, method=VAR_QUANTILE_METHOD)
    return pd.Series(quantiles, index=losses.columns, name="forecast_var")


def diversification_effect(allocated_vars: pd.DataFrame) -> pd.DataFrame:
    """Sum already-allocated holding dollar VaRs less portfolio VaR; no extra weights."""
    if not set(RISK_IDENTIFIERS).issubset(allocated_vars.columns):
        raise DataValidationError("Diversification requires all holdings and portfolio VaR")
    values = _real_array(allocated_vars.loc[:, list(RISK_IDENTIFIERS)].to_numpy(), "VaRs")
    standalone_sum = values[:, :len(HOLDINGS)].sum(axis=1)
    return pd.DataFrame({"allocated_standalone_var_sum": standalone_sum,
                         "portfolio_var": values[:, -1],
                         "diversification_effect": standalone_sum - values[:, -1]},
                        index=allocated_vars.index)


def euler_var_attribution(
    holding_losses: pd.DataFrame, quantities: pd.Series,
    relative_step: float = ATTRIBUTION_RELATIVE_STEP,
) -> VaRAttribution:
    """Euler contributions q_i*dVaR/dq_i from central proportional position bumps.

    Reuse the latest parametric holding losses and the SAME scenarios and quantile
    rule for all bumps. Full-revaluation losses are linear in fixed position size,
    so bumping q_i by +/- h*q_i changes each portfolio loss by +/- h*L_i exactly.
    The default h=1e-4 is documented, fixed, and uses no new random draws. A finite
    empirical quantile can switch order statistics across bumps; disclose the
    portfolio VaR, contribution sum, and residual without normalizing contributions.
    """
    if not set(HOLDINGS).issubset(holding_losses.columns) or not quantities.index.is_unique:
        raise DataValidationError("Attribution requires the five holding losses and quantities")
    losses = _real_array(holding_losses.loc[:, list(HOLDINGS)].to_numpy(), "holding losses")
    positions = _real_array(quantities.reindex(HOLDINGS).to_numpy(), "quantities")
    step = _real_array(relative_step, "relative_step")
    if not len(losses) or (positions <= 0).any() or step.ndim != 0 or not 0 < step < 1:
        raise DataValidationError("Attribution requires nonempty losses, positive quantities, and 0 < step < 1")
    total = losses.sum(axis=1)
    portfolio_var = float(np.quantile(total, VAR_CONFIDENCE, method=VAR_QUANTILE_METHOD))
    rows = []
    for column, holding in enumerate(HOLDINGS):
        bump = float(step) * positions[column]
        plus = float(np.quantile(total + float(step) * losses[:, column], VAR_CONFIDENCE,
                                 method=VAR_QUANTILE_METHOD))
        minus = float(np.quantile(total - float(step) * losses[:, column], VAR_CONFIDENCE,
                                  method=VAR_QUANTILE_METHOD))
        marginal = (plus - minus) / (2 * bump)
        rows.append({"holding": holding, "quantity": positions[column], "quantity_bump": bump,
                     "var_plus": plus, "var_minus": minus,
                     "marginal_var_per_unit": marginal,
                     "euler_contribution": positions[column] * marginal})
    contributions = pd.DataFrame(rows).set_index("holding")
    contribution_sum = float(contributions["euler_contribution"].sum())
    return VaRAttribution(contributions, portfolio_var, contribution_sum,
                          portfolio_var - contribution_sum, float(step))


def scenario_holding_values(
    allocation: PortfolioAllocation, levels: pd.DataFrame, forecast_origin: object,
    factor_changes: pd.DataFrame, remaining_maturities: Sequence[float],
) -> pd.DataFrame:
    """Full next-day revaluation at supplied exact remaining maturities.

    All eight shocks are applied to origin levels, with vectorized linear yield
    interpolation between the bracketing fitted tenors. ETF shocks are log returns
    applied to origin Adj Close, retaining the initial Close-based share quantity.
    This function never reads a realized forecast-day level.
    """
    _, row = _valuation_row(levels, forecast_origin)
    if (not isinstance(factor_changes, pd.DataFrame) or factor_changes.empty
            or factor_changes.columns.has_duplicates or set(factor_changes.columns) != set(RISK_FACTOR_COLUMNS)):
        raise DataValidationError("Scenarios require the eight specified risk factors")
    shocks = _real_array(factor_changes.loc[:, list(RISK_FACTOR_COLUMNS)].to_numpy(), "factor changes")
    maturities = _real_array(remaining_maturities, "remaining maturities")
    if maturities.shape != (3,) or (maturities < TREASURY_TENORS[0]).any() or (maturities > TREASURY_TENORS[-1]).any():
        raise DataValidationError("Three remaining maturities must stay within fitted coverage [1, 10]")
    curves = row.loc[list(YIELD_COLUMNS)].to_numpy() + shocks[:, :len(YIELD_COLUMNS)]
    fitted = np.empty((len(shocks), len(TREASURY_HOLDINGS)))
    for column, maturity in enumerate(maturities):
        upper = int(np.searchsorted(TREASURY_TENORS, maturity))
        if TREASURY_TENORS[upper] == maturity:
            fitted[:, column] = curves[:, upper]
        else:
            lower = upper - 1
            weight = (maturity - TREASURY_TENORS[lower]) / (TREASURY_TENORS[upper] - TREASURY_TENORS[lower])
            fitted[:, column] = curves[:, lower] * (1 - weight) + curves[:, upper] * weight
    bonds = zero_coupon_price(fitted, maturities) * allocation.treasury_face_quantities.to_numpy()
    origin_adjusted = np.array([row[f"{ticker}_Adj Close"] for ticker in ETF_TICKERS])
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            adjusted = origin_adjusted * np.exp(shocks[:, len(YIELD_COLUMNS):])
            etfs = (allocation.etf_shares.to_numpy() * allocation.initial_etf_close.to_numpy()
                    * (adjusted / allocation.initial_etf_adj_close.to_numpy()))
    except FloatingPointError as exc:
        raise DataValidationError("One-day ETF scenario values are not representable") from exc
    values = np.column_stack([bonds, etfs])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataValidationError("One-day holding values must be finite and positive")
    return pd.DataFrame(values, index=factor_changes.index, columns=HOLDINGS)


def _backtest_inputs(market: MarketData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check chronological alignment, including the explicit start date of each change."""
    for frame in (market.levels, market.yield_changes, market.etf_log_returns):
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
                or frame.index.has_duplicates or not frame.index.is_monotonic_increasing):
            raise DataValidationError("Backtest inputs require unique chronological daily dates")
    if not market.yield_changes.index.equals(market.etf_log_returns.index):
        raise DataValidationError("Backtest yield and ETF change indices must match")
    price_columns = [f"{ticker}_{field}" for ticker in ETF_TICKERS for field in ETF_PRICE_FIELDS]
    levels, _ = validate_source_frame(market.levels, LEVEL_COLUMNS, "backtest levels",
                                       price_columns=price_columns)
    changes, _ = validate_source_frame(market.risk_factor_changes, RISK_FACTOR_COLUMNS, "backtest changes")
    if not changes.index.equals(levels.index[1:]):
        raise DataValidationError("Each backtest change must end at the next aligned level date")
    if (market.change_start_dates.index.has_duplicates
            or not market.change_start_dates.index.is_monotonic_increasing):
        raise DataValidationError("Change-start metadata must have unique chronological dates")
    starts = market.change_start_dates.reindex(changes.index)
    if starts.isna().any() or not pd.DatetimeIndex(starts.to_numpy()).equals(levels.index[:-1]):
        raise DataValidationError("Each change must start at the immediately preceding aligned date")
    if not np.isfinite(levels.to_numpy()).all() or not np.isfinite(changes.to_numpy()).all():
        raise DataValidationError("Backtest levels and changes must be finite and complete")
    return levels, changes


def estimation_window(
    changes: pd.DataFrame, forecast_change_position: int, forecast_origin: object,
    forecast_date: object,
) -> pd.DataFrame:
    """Exactly 500 immediately preceding changes, with explicit no-look-ahead checks."""
    position = _integer_count(forecast_change_position, "forecast_change_position", CALIBRATION_WINDOW)
    origin, target = _as_date(forecast_origin, "forecast_origin"), _as_date(forecast_date, "forecast_date")
    if changes.index.has_duplicates or not changes.index.is_monotonic_increasing or position >= len(changes):
        raise DataValidationError("Invalid chronological forecast-change index")
    window = changes.iloc[position - CALIBRATION_WINDOW:position].copy()
    if (len(window) != CALIBRATION_WINDOW or changes.index[position] != target
            or window.index[-1] != origin or origin >= target
            or not (window.index < target).all() or target in window.index):
        raise DataValidationError("Estimation window violates the forecast origin or no-look-ahead boundary")
    return window


def backtest_var(
    market: MarketData, *, scenario_count: int = SIMULATION_PATHS,
    forecast_count: int = BACKTEST_FORECASTS, master_seed: int = MASTER_SEED,
) -> BacktestResult:
    """Latest 250 eligible one-day forecasts; smaller counts support offline checks.

    Allocate USD 1m at the first selected forecast origin, then hold all quantities
    fixed. Backtest ages are aligned steps/252 for origin, scenarios, and realized
    observations alike, rather than the standalone historical ACT/365 default.
    Each Gaussian uses precisely the immediately preceding 500 eight-factor
    changes. Historical scenarios apply those same 500 changes to current levels.
    Exceptions are strict: realized P&L < -VaR. Dollar loss VaRs are not clipped.
    Only the latest scenario distributions are retained for diversification and
    common-scenario Euler attribution; no files or publication outputs are written.
    """
    scenario_count = _integer_count(scenario_count, "scenario_count")
    forecast_count = _integer_count(forecast_count, "forecast_count")
    if forecast_count > BACKTEST_FORECASTS:
        raise DataValidationError(f"At most {BACKTEST_FORECASTS} forecasts are in scope")
    levels, changes = _backtest_inputs(market)
    if len(changes) < CALIBRATION_WINDOW + forecast_count:
        raise DataValidationError(
            f"Backtest requires {CALIBRATION_WINDOW + forecast_count} aligned changes; found {len(changes)}"
        )
    first = len(changes) - forecast_count
    allocation = allocate_portfolio(levels, levels.index[first])
    quantities = pd.concat([allocation.treasury_face_quantities, allocation.etf_shares]).reindex(HOLDINGS)
    dates = levels.index[first:]
    ages = np.arange(forecast_count + 1) / TRADING_DAYS
    remaining = pd.DataFrame(np.asarray(INITIAL_MATURITIES) - ages[:, None],
                             index=dates, columns=TREASURY_HOLDINGS)
    valuations = pd.DataFrame([
        holding_values(allocation, levels, stamp, elapsed_years=float(age))
        for stamp, age in zip(dates, ages)
    ], index=dates).reindex(columns=HOLDINGS)
    weights = valuations.div(valuations.sum(axis=1), axis=0)
    records, window_rows = [], []
    latest_scenarios, latest_losses, latest_vars = {}, {}, {}
    for step, position in enumerate(range(first, len(changes))):
        origin, forecast_date = levels.index[position], changes.index[position]
        window = estimation_window(changes, position, origin, forecast_date)
        model = calibrate_gaussian(window, origin)
        if not model.sample.index.equals(window.index):
            raise DataValidationError("Rolling calibration changed the checked estimation window")
        ordinal = position - CALIBRATION_WINDOW
        generator = create_forecast_rng(ordinal, master_seed)
        simulated = generator.standard_normal((scenario_count, len(RISK_FACTOR_COLUMNS)))
        simulated = simulated @ model.covariance_adjustment.factor.T + model.mean.to_numpy()
        scenarios = {
            "parametric": pd.DataFrame(simulated, columns=RISK_FACTOR_COLUMNS,
                                        index=pd.RangeIndex(1, scenario_count + 1, name="scenario")),
            "historical": window,
        }
        current = valuations.iloc[step]
        realized = valuations.iloc[step + 1] - current
        realized["Portfolio"] = realized.sum()
        for method, factor_changes in scenarios.items():
            future = scenario_holding_values(allocation, levels, origin, factor_changes, remaining.iloc[step + 1])
            losses = future.rsub(current, axis=1)
            losses["Portfolio"] = losses.sum(axis=1)
            forecasts = loss_var(losses)
            for holding in RISK_IDENTIFIERS:
                var, pnl = float(forecasts[holding]), float(realized[holding])
                records.append({"forecast_date": forecast_date, "method": method, "holding": holding,
                                "forecast_var": var, "realized_pnl": pnl, "exception": pnl < -var,
                                "estimation_window_start": window.index[0],
                                "estimation_window_end": window.index[-1], "forecast_origin": origin,
                                "forecast_ordinal": ordinal})
            if step == forecast_count - 1:
                latest_scenarios[method] = factor_changes.copy()
                latest_losses[method] = losses
                latest_vars[method] = forecasts
        correction = model.covariance_adjustment
        window_rows.append({"forecast_date": forecast_date, "forecast_origin": origin,
                            "forecast_change_position": position, "forecast_ordinal": ordinal,
                            "estimation_count": len(window),
                            "estimation_window_start": window.index[0], "estimation_window_end": window.index[-1],
                            "parametric_seed": rolling_forecast_seed(ordinal, master_seed),
                            "covariance_clipped_eigenvalues": correction.clipped_eigenvalues,
                            "covariance_correction_frobenius_norm": correction.correction_frobenius_norm})
    record_frame = pd.DataFrame(records)
    statistics = []
    for (method, holding), group in record_frame.groupby(["method", "holding"], sort=False):
        exceptions = group.sort_values("forecast_date")["exception"].to_numpy()
        statistics.append({"method": method, "holding": holding,
                           **kupiec_test(exceptions), **christoffersen_test(exceptions)})
    stats = pd.DataFrame(statistics).set_index(["method", "holding"])
    diversification = diversification_effect(pd.DataFrame.from_dict(latest_vars, orient="index"))
    diversification.index.name = "method"
    diversification["forecast_date"] = dates[-1]
    diversification["forecast_origin"] = dates[-2]
    attribution = euler_var_attribution(latest_losses["parametric"], quantities)
    return BacktestResult(allocation, quantities, valuations, weights, remaining,
                          pd.DataFrame(window_rows).set_index("forecast_date"), record_frame, stats,
                          MappingProxyType(latest_scenarios), MappingProxyType(latest_losses),
                          diversification, attribution)


def run_full_analysis() -> CompleteAnalysis:
    """Complete the fixed analysis from one live snapshot before creating outputs/."""
    market = retrieve_market_data()
    _backtest_inputs(market)
    required = CALIBRATION_WINDOW + BACKTEST_FORECASTS
    if len(market.risk_factor_changes) < required:
        raise DataValidationError(
            f"At least {required} aligned changes are required; live sources supplied "
            f"{len(market.risk_factor_changes)}. No result files were generated."
        )
    terminal = run_terminal_analysis(market)
    backtest = backtest_var(market)
    return CompleteAnalysis(market, terminal, backtest)


def _report_metadata(result: CompleteAnalysis) -> dict[str, object]:
    return {
        "requested_start": ANALYSIS_START, "requested_end": ANALYSIS_END,
        "sample_start": result.market.levels.index[0].strftime("%Y-%m-%d"),
        "sample_end": result.market.levels.index[-1].strftime("%Y-%m-%d"),
        "confidence_level": VAR_CONFIDENCE, "reporting_notional_usd": INITIAL_PORTFOLIO_VALUE,
        "master_seed": MASTER_SEED,
    }


def _with_metadata(frame: pd.DataFrame, metadata: Mapping[str, object]) -> pd.DataFrame:
    result = frame.copy()
    for key, value in metadata.items():
        result[key] = value
    return result


def data_quality_table(result: CompleteAnalysis) -> pd.DataFrame:
    """Tidy diagnostics, units, conventions, source deviations, positions, and seeds."""
    rows = []

    def add(section, subject, check, value, unit="text", details="none"):
        rows.append({"section": section, "subject": subject, "check": check,
                     "value": str(value), "unit": unit, "details": str(details)})

    def dates(values):
        return ";".join(stamp.strftime("%Y-%m-%d") for stamp in values) or "none"

    for source in result.market.coverage:
        add("source", source.source, "first_valid_date", source.first_valid_date.date(), "date")
        add("source", source.source, "last_valid_date", source.last_valid_date.date(), "date")
        add("source", source.source, "in_period_rows", source.observation_count, "rows")
        add("source", source.source, "complete_rows", source.observation_count - len(source.incomplete_dates), "rows")
        add("source", source.source, "alignment_dropped_rows", source.observation_count - len(result.market.levels), "rows")
        add("source", source.source, "duplicate_dates", 0, "count", "Duplicates are rejected before alignment")
        add("source", source.source, "missing_values", source.missing_value_count, "cells")
        add("source", source.source, "incomplete_rows", len(source.incomplete_dates), "rows", dates(source.incomplete_dates))
        add("source", source.source, "absent_weekdays", len(source.absent_weekdays), "days",
            dates(source.absent_weekdays) + "; absent weekdays can be legitimate holidays")
        add("source", source.source, "source_period_deviation", ";".join(source.boundary_shortfalls) or "none",
            details="Boundary shortfalls compare complete observations with requested weekdays; no dates are fabricated")
    market = result.market
    for check, value, unit in (
        ("aligned_levels", len(market.levels), "rows"),
        ("aligned_changes", len(market.risk_factor_changes), "changes"),
        ("required_backtest_changes", CALIBRATION_WINDOW + BACKTEST_FORECASTS, "changes"),
        ("duplicate_dates", int(market.levels.index.duplicated().sum()), "count"),
        ("missing_required_values", int(market.levels.isna().sum().sum()), "cells"),
        ("nonfinite_changes", int((~np.isfinite(market.risk_factor_changes.to_numpy())).sum()), "cells"),
        ("dropped_dates", len(market.dropped_dates), "days"),
        ("maximum_calendar_gap", int((market.levels.index[1:] - market.levels.index[:-1]).days.max()), "days"),
    ):
        add("alignment", "all eight factors", check, value, unit,
            dates(market.dropped_dates) if check == "dropped_dates" else "none")
    add("alignment", "all eight factors", "policy", "Common valid dates before differences; no filling")
    add("alignment", "all eight factors", "required_level_columns", ";".join(market.levels.columns))
    add("alignment", "all eight factors", "ordered_change_factors", ";".join(market.risk_factor_changes.columns))
    add("alignment", "all eight factors", "date_spans", "Changes span consecutive retained dates, including source-calendar gaps")
    add("validation", "all sources", "checks", "PASS",
        details="ISO dates, required columns, unique dates/columns, real numeric types, finite values, positive ETF prices, source coverage")
    for subject, convention in (
        ("Treasury yields", "Source percentage yields / 100 = decimal yields"),
        ("Treasury changes", "Decimal-yield units internally; multiply by 10000 for basis points"),
        ("ETF changes", "Natural log of consecutive aligned Adj Close ratios; decimal log return"),
        ("Treasury pricing", "exp(-decimal yield * remaining years), USD per USD 1 face"),
        ("Maturity interpolation", "Linear yield interpolation between bracketing fitted tenors, including 1-2, 4-5, 9-10; no extrapolation"),
        ("Terminal aging", "252 trading days = 1 year; remaining maturities 1, 4, 9 years"),
        ("Backtest aging", "Aligned forecast steps / 252 from first forecast origin, in scenarios AND realized valuations"),
        ("Historical helper default", "ACT/365 Fixed; complete backtest explicitly supplies the aligned-step clock"),
        ("ETF valuation", "Initial allocation / Close gives shares; all later values use initial allocation * Adj Close ratio"),
        ("Returns and risk", "CSV returns are decimal; dollar loss VaR and P&L are USD; charts explicitly label percentages"),
    ):
        add("convention", subject, "units_and_method", convention)
    for purpose, seed in CHILD_SEEDS.items():
        add("randomness", purpose, "child_seed", seed, "integer seed")
    add("randomness", "rolling VaR", "forecast_seed_rule",
        "SeedSequence(rolling child seed, spawn_key=(global eligible forecast ordinal,)); uint64 state",
        details="Each parametric forecast seed is recorded in var_forecasts.csv")
    add("randomness", "attribution", "draws", "No new draws: reuse latest parametric full-revaluation losses")
    for analysis, allocation in (("terminal", result.terminal.correlated.allocation),
                                 ("backtest", result.backtest.allocation)):
        add("allocation", analysis, "allocation_date", allocation.allocation_date.date(), "date")
        quantities = pd.concat([allocation.treasury_face_quantities, allocation.etf_shares]).reindex(HOLDINGS)
        for holding, weight in zip(HOLDINGS, WEIGHTS):
            add("allocation", analysis + ":" + holding, "initial_weight", weight, "decimal")
            add("allocation", analysis + ":" + holding, "fixed_quantity", quantities[holding],
                "USD face" if holding in TREASURY_HOLDINGS else "initial shares")
    correction = result.terminal.calibration.covariance_adjustment
    add("numerics", "terminal covariance", "clipped_eigenvalues", correction.clipped_eigenvalues, "count")
    add("numerics", "terminal covariance", "correction_frobenius_norm", correction.correction_frobenius_norm, "native covariance units")
    add("numerics", "rolling covariance", "clipped_eigenvalues_total",
        result.backtest.windows["covariance_clipped_eigenvalues"].sum(), "count")
    add("numerics", "rolling covariance", "maximum_correction_frobenius_norm",
        result.backtest.windows["covariance_correction_frobenius_norm"].max(), "native covariance units")
    add("numerics", "covariance", "policy", "Reject material indefiniteness; clip only negative eigenvalues within 10*n*eps*eigenvalue scale to zero; no jitter")
    return _with_metadata(pd.DataFrame(rows), _report_metadata(result))


def build_output_tables(result: CompleteAnalysis) -> dict[str, pd.DataFrame]:
    """Build six finite, explicitly labelled tables; never touch the filesystem."""
    metadata = _report_metadata(result)
    terminal = result.terminal
    backtest = result.backtest
    terminal_context = {**metadata, "allocation_date": terminal.correlated.allocation.allocation_date,
                        "calibration_start": terminal.calibration.sample.index[0],
                        "calibration_end": terminal.calibration.sample.index[-1],
                        "calibration_changes": len(terminal.calibration.sample),
                        "horizon_trading_days": TRADING_DAYS, "path_count": len(terminal.correlated.paths)}
    paths = pd.DataFrame({"path": terminal.correlated.paths.index})
    summaries = []
    for method, simulation in (("standalone", terminal.standalone), ("correlated", terminal.correlated)):
        selected = simulation.paths.loc[:, [column for column in simulation.paths.columns
                                             if method == "correlated" or column[0] != "Portfolio"]]
        for holding, metric in selected.columns:
            suffix = {"value": "value_usd", "return": "return_decimal", "adjusted_price": "adjusted_price_usd"}.get(metric)
            if metric == "unit_price":
                suffix = "unit_price_usd_per_face" if holding in TREASURY_HOLDINGS else "unit_price_usd_per_initial_share"
            paths[f"{method}_{holding}_{suffix}"] = selected[holding, metric].to_numpy()
        summary = simulation.summary.loc[selected.columns].reset_index()
        summary.insert(0, "analysis", method)
        summary["unit"] = [
            "decimal return" if metric == "return" else
            "USD per USD 1 face" if metric == "unit_price" and holding in TREASURY_HOLDINGS else
            "USD per initial ETF share" if metric == "unit_price" else
            "USD adjusted-price basis" if metric == "adjusted_price" else "USD"
            for holding, metric in zip(summary["holding"], summary["metric"])
        ]
        summary["count"] = summary["count"].astype(int)
        summary["seed"] = str(simulation.seed)
        summaries.append(summary.rename(columns={"1%": "p01", "50%": "p50", "99%": "p99"}))
    paths["standalone_seed"] = str(terminal.standalone.seed)
    paths["correlated_seed"] = str(terminal.correlated.seed)
    paths = _with_metadata(paths, terminal_context)
    terminal_summary = _with_metadata(pd.concat(summaries, ignore_index=True), terminal_context)
    forecast_count = len(backtest.windows)
    backtest_context = {**metadata, "allocation_date": backtest.allocation.allocation_date,
                        "rolling_window_changes": CALIBRATION_WINDOW, "forecast_count": forecast_count,
                        "horizon_trading_days": 1, "aging_clock": "aligned steps / 252", "monetary_unit": "USD"}
    forecasts = _with_metadata(backtest.records, backtest_context)
    forecasts["scenario_count"] = forecasts["method"].map(
        {method: len(backtest.latest_scenarios[method]) for method in VAR_METHODS})
    forecast_seeds = backtest.windows["parametric_seed"].map(str)
    forecasts["forecast_seed"] = [forecast_seeds.loc[date] if method == "parametric" else "not_used"
                                   for method, date in zip(forecasts["method"], forecasts["forecast_date"])]
    var_summary = backtest.statistics.reset_index()
    for column in ("p01", "p11"):
        var_summary[column] = var_summary[column].astype(object).where(var_summary[column].notna(), "not_identified")
    var_summary["transition_probability_status"] = np.where(var_summary["independence_identified"], "identified", "unobserved_state")
    var_summary["forecast_start"] = backtest.windows.index[0]
    var_summary["forecast_end"] = backtest.windows.index[-1]
    var_summary["scenario_count"] = var_summary["method"].map(
        {method: len(backtest.latest_scenarios[method]) for method in VAR_METHODS})
    var_summary = _with_metadata(var_summary, backtest_context)
    risk_rows = []
    latest = backtest.records.loc[backtest.records["forecast_date"] == backtest.windows.index[-1]]
    for method in VAR_METHODS:
        holding_vars = latest.loc[latest["method"] == method].set_index("holding")["forecast_var"]
        for holding in HOLDINGS:
            risk_rows.append({"method": method, "holding": holding, "metric": "allocated_standalone_var", "value_usd": holding_vars[holding]})
        for metric, column in (("standalone_var_sum", "allocated_standalone_var_sum"),
                                ("portfolio_var", "portfolio_var"), ("diversification_effect", "diversification_effect")):
            risk_rows.append({"method": method, "holding": "Portfolio", "metric": metric,
                              "value_usd": backtest.diversification.loc[method, column]})
    attribution = backtest.attribution
    for holding, row in attribution.contributions.iterrows():
        risk_rows.append({"method": "parametric", "holding": holding, "metric": "euler_var_contribution", "value_usd": row["euler_contribution"]})
    for metric, value in (("contribution_sum", attribution.contribution_sum), ("attribution_residual", attribution.residual)):
        risk_rows.append({"method": "parametric", "holding": "Portfolio", "metric": metric, "value_usd": value})
    risk_summary = _with_metadata(pd.DataFrame(risk_rows), backtest_context)
    risk_summary["forecast_date"] = backtest.windows.index[-1]
    risk_summary["forecast_origin"] = backtest.windows.iloc[-1]["forecast_origin"]
    risk_summary["scenario_count"] = risk_summary["method"].map(
        {method: len(backtest.latest_scenarios[method]) for method in VAR_METHODS})
    risk_summary["position_bump_relative"] = [str(attribution.relative_step) if metric.startswith("euler") or metric in ("contribution_sum", "attribution_residual") else "not_applicable"
                                               for metric in risk_summary["metric"]]
    tables = {"data_quality.csv": data_quality_table(result), "terminal_paths.csv": paths,
              "terminal_summary.csv": terminal_summary, "var_forecasts.csv": forecasts,
              "var_summary.csv": var_summary, "risk_summary.csv": risk_summary}
    for name, frame in tables.items():
        if frame.empty or frame.columns.has_duplicates or frame.isna().any().any():
            raise DataValidationError(f"Incomplete report table: {name}")
        numeric = frame.select_dtypes(include="number").to_numpy()
        if not np.isfinite(numeric).all():
            raise DataValidationError(f"Nonfinite report values: {name}")
    return tables


def render_figures(result: CompleteAnalysis) -> dict[str, bytes]:
    """Render all three Matplotlib figures in memory and close every figure."""
    os.environ["MPLCONFIGDIR"] = str(PROJECT_DIRECTORY / ".venv" / "matplotlib-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    terminal, backtest = result.terminal, result.backtest
    source_caption = (f"Sources: Federal Reserve fitted yields; Yahoo Finance via yfinance. "
                      f"Sample: {result.market.levels.index[0].date()} to {result.market.levels.index[-1].date()}.")
    images = {}

    def save(figure, name, description):
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=FIGURE_DPI,
                       metadata={"Title": name, "Description": description})
        images[name] = buffer.getvalue()

    navy, gold, teal = "#234b70", "#d68c27", "#16887d"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    try:
        fig.subplots_adjust(left=.07, right=.98, top=.79, bottom=.11, hspace=.38, wspace=.25)
        fig.suptitle("One-year terminal return distributions", fontsize=19, y=.97)
        caption = (f"Allocation: {terminal.correlated.allocation.allocation_date.date()} | "
                   f"{len(terminal.correlated.paths):,} paths | {TRADING_DAYS} trading days\n"
                   f"Calibration: {len(terminal.calibration.sample)} changes, "
                   f"{terminal.calibration.sample.index[0].date()} to {terminal.calibration.sample.index[-1].date()}")
        fig.text(.5, .89, caption, ha="center", va="center", fontsize=10)
        for ax, holding, label in zip(axes.flat, RISK_IDENTIFIERS, PLOT_HOLDING_LABELS):
            if holding != "Portfolio":
                ax.hist(terminal.standalone.paths[holding, "return"] * 100, bins=55,
                        density=True, alpha=.58, color=gold, label="Standalone marginal")
            ax.hist(terminal.correlated.paths[holding, "return"] * 100, bins=55,
                    density=True, alpha=.64, color=navy, label="Correlated model")
            ax.axvline(0, color="#737373", linewidth=.8, linestyle=":")
            ax.set(title=label, xlabel="Terminal return (%)", ylabel="Density")
            ax.grid(axis="y", alpha=.18)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .85), ncol=2, frameon=False)
        fig.text(.5, .035, source_caption + "\nTables report 1st/50th/99th percentiles; returns include bond aging and reinvested ETF distributions.",
                 ha="center", fontsize=9)
        save(fig, "terminal_distributions.png", caption + "\n" + source_caption)
    finally:
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    try:
        fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.09, hspace=.48, wspace=.24)
        fig.suptitle("One-day VaR backtests: realized P&L and loss thresholds", fontsize=18, y=.97)
        caption = (f"99% loss VaR; plotted thresholds = -VaR | {len(backtest.windows)} forecasts, "
                   f"{backtest.windows.index[0].date()} to {backtest.windows.index[-1].date()}\n"
                   f"{CALIBRATION_WINDOW}-change rolling windows | "
                   f"{len(backtest.latest_scenarios['parametric']):,} Gaussian / 500 historical scenarios per forecast")
        fig.text(.5, .91, caption, ha="center", va="center", fontsize=10)
        for ax, holding, label in zip(axes.flat, RISK_IDENTIFIERS, PLOT_HOLDING_LABELS):
            forecasts = backtest.records.loc[backtest.records["holding"] == holding]
            parametric = forecasts.loc[forecasts["method"] == "parametric"].sort_values("forecast_date")
            historical = forecasts.loc[forecasts["method"] == "historical"].sort_values("forecast_date")
            ax.plot(parametric["forecast_date"], parametric["realized_pnl"], color=navy, linewidth=.8, label="Realized P&L")
            for frame, color, label_method, marker in ((parametric, gold, "Parametric", "o"),
                                                        (historical, teal, "Historical", "x")):
                ax.plot(frame["forecast_date"], -frame["forecast_var"], color=color, linewidth=1,
                        linestyle="--", label=label_method + " -VaR")
                breaches = frame.loc[frame["exception"]]
                ax.scatter(breaches["forecast_date"], breaches["realized_pnl"], marker=marker,
                           color=color, s=30, zorder=4, label=label_method + " exception")
            ax.axhline(0, color="#737373", linewidth=.6)
            ax.set_title(label)
            ax.set_ylabel("One-day P&L / threshold (USD)")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.grid(alpha=.16)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .865), ncol=5, frameon=False, fontsize=9)
        fig.text(.5, .025, source_caption + f"\nBuy-and-hold from {backtest.allocation.allocation_date.date()}; "
                 "aligned steps/252 for bond aging; no rebalancing.", ha="center", fontsize=9)
        save(fig, "var_backtests.png", caption + "\n" + source_caption)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    try:
        fig.subplots_adjust(left=.09, right=.71, top=.78, bottom=.18)
        contribution = backtest.attribution
        values = contribution.contributions["euler_contribution"].to_numpy()
        bars = ax.bar(PLOT_HOLDING_LABELS[:5], values, color=[navy if value >= 0 else gold for value in values])
        ax.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=4, fontsize=10)
        ax.axhline(0, color="#737373", linewidth=.8)
        ax.set_ylabel("Euler VaR contribution (USD)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", alpha=.18)
        ax.margins(y=.15)
        fig.suptitle("Latest parametric 99% one-day VaR contributions", fontsize=18, y=.97)
        caption = (f"Forecast: {backtest.windows.index[-1].date()} | Origin: {backtest.windows.iloc[-1]['forecast_origin'].date()}\n"
                   f"{CALIBRATION_WINDOW}-change window | {len(backtest.windows)} forecasts | "
                   f"{len(backtest.latest_scenarios['parametric']):,} common Gaussian scenarios")
        fig.text(.5, .88, caption, ha="center", fontsize=10)
        reconciliation = (f"Portfolio VaR\nUSD {contribution.portfolio_var:,.2f}\n\n"
                          f"Contribution sum\nUSD {contribution.contribution_sum:,.2f}\n\n"
                          f"Residual (VaR - sum)\nUSD {contribution.residual:+.6g}\n\n"
                          "Residual is disclosed;\ncontributions are not rescaled.")
        fig.text(.76, .74, reconciliation, va="top", fontsize=12)
        fig.text(.5, .045, source_caption + f"\nCentral position bumps: +/-{contribution.relative_step:.2%}; "
                 "fixed positions; full revaluation; common scenarios.", ha="center", fontsize=9)
        save(fig, "risk_contributions.png", caption + "\n" + reconciliation + "\n" + source_caption)
    finally:
        plt.close(fig)
    return images


def write_outputs(tables: Mapping[str, pd.DataFrame], figures: Mapping[str, bytes]) -> tuple[Path, ...]:
    """Write exactly the configured files after every in-memory result is complete."""
    if set(tables) != set(CSV_FILENAMES) or set(figures) != set(FIGURE_FILENAMES):
        raise DataValidationError("Output set must contain exactly the six CSVs and three PNGs")
    payloads = {name: frame.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g").encode("utf-8")
                for name, frame in tables.items()}
    payloads.update(figures)
    if OUTPUT_DIRECTORY.exists():
        unexpected = [path.name for path in OUTPUT_DIRECTORY.iterdir()
                      if not path.is_file() or path.name not in OUTPUT_FILENAMES]
        if unexpected:
            raise DataValidationError(f"Unexpected entries in outputs/: {unexpected}")
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    for name in OUTPUT_FILENAMES:
        (OUTPUT_DIRECTORY / name).write_bytes(payloads[name])
    return tuple(OUTPUT_DIRECTORY / name for name in OUTPUT_FILENAMES)


def main(argv: Sequence[str] | None = None) -> int:
    """The single entry point: finish all analyses and prepare reports before writing."""
    parser = argparse.ArgumentParser(description="Run the complete fixed multi-asset simulation and VaR analysis.")
    parser.parse_args(argv)
    try:
        print(f"Analyzing {ANALYSIS_START} through {ANALYSIS_END}; retrieving and validating sources...", flush=True)
        result = run_full_analysis()
        print(f"Analysis complete: {len(result.market.risk_factor_changes)} aligned changes; "
              f"{len(result.terminal.correlated.paths):,} terminal paths; {len(result.backtest.windows)} forecasts.", flush=True)
        tables = build_output_tables(result)
        figures = render_figures(result)
        paths = write_outputs(tables, figures)
    except (DataValidationError, DataRetrievalError, OSError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Wrote {len(paths)} result files to {OUTPUT_DIRECTORY}")
    print(f"Latest parametric portfolio VaR: USD {result.backtest.attribution.portfolio_var:,.2f}; "
          f"attribution residual: USD {result.backtest.attribution.residual:+.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
