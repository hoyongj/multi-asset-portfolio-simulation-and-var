"""STA2540 group project: 1-day and 5-day VaR and ES by historical and Monte Carlo simulation.

Portfolio (USD market values at the valuation date, 2020-12-31):
    SPX        4,000   S&P 500 index exposure
    Govt_10Y   2,000   10-year government bond, 4% coupon, semiannual
    Corp_A_10Y 4,000   10-year A-rated bond, 5% coupon, semiannual

Sources (FRED, https://fred.stlouisfed.org): SP500 (index level), DGS10 (10-year
constant-maturity Treasury yield, percent), BAA10Y (Moody's Baa yield minus DGS10,
percent). The course brief names BAA10Y as the corporate spread; it is a Baa
spread and therefore a conservative proxy for an A-rated issuer.

Risk factors: SPX daily log return, absolute change in DGS10, and absolute change
in BAA10Y (both in decimal yield units). Govt_10Y is priced at DGS10; Corp_A_10Y
at DGS10 + BAA10Y. Each bond is a newly issued 10-year bullet discounted at a
single yield to maturity with semiannual compounding (bond-equivalent basis,
matching DGS10's quoting convention). Scenarios are applied instantaneously to
valuation-date levels and every position is fully repriced; bond carry, coupon
accrual and roll-down over the 1- or 5-day horizon are ignored.

Historical simulation replays the observed 2019-12-31 .. 2020-12-31 changes:
1-day uses every daily change, 5-day uses overlapping 5-day changes. Monte Carlo
calibrates a multivariate normal to the same daily changes (sample mean and
covariance, ddof=1) and sums IID daily draws over the horizon, so SPX follows a
geometric Brownian motion and yields and spreads follow arithmetic Brownian
motions. Only dates on which all three series are present are retained; nothing
is filled.

VaR is the empirical loss quantile (NumPy "linear"); ES is the mean loss at or
beyond VaR. Losses are positive numbers in USD.

Importing does not retrieve data or write files. Executing runs the analysis and
writes its CSVs and figure to group_project_outputs/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES = ("SP500", "DGS10", "BAA10Y")
REQUEST_TIMEOUT_SECONDS = 30

HISTORY_START = "2019-12-31"
HISTORY_END = "2020-12-31"  # Inclusive; also the valuation date.

HOLDINGS = ("SPX", "Govt_10Y", "Corp_A_10Y")
POSITION_VALUES = {"SPX": 4_000.0, "Govt_10Y": 2_000.0, "Corp_A_10Y": 4_000.0}
COUPON_RATES = {"Govt_10Y": 0.04, "Corp_A_10Y": 0.05}
BOND_MATURITY_YEARS = 10
COUPONS_PER_YEAR = 2
FACE_PER_BOND = 100.0

HORIZONS = (1, 5)
CONFIDENCE_LEVELS = (0.95, 0.99)
MC_PATHS = 100_000
MASTER_SEED = 42
ESTIMATION_DDOF = 1
QUANTILE_METHOD = "linear"
RISK_IDS = HOLDINGS + ("Portfolio",)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "group_project_outputs"
OUTPUT_FILENAMES = ("market_data.csv", "positions.csv", "risk_measures.csv", "pnl_distributions.png")


class DataError(RuntimeError):
    """Raised when source data cannot be retrieved or fails validation."""


@dataclass(frozen=True)
class Position:
    holding: str
    market_value: float
    unit_price: float
    quantity: float
    yield_decimal: float | None
    modified_duration: float | None
    dv01: float | None


# ---------------------------------------------------------------- data

def fetch_fred_series(series_id: str, start: str = HISTORY_START, end: str = HISTORY_END) -> pd.Series:
    """Download one FRED series as a float Series indexed by date (missing kept as NaN)."""
    try:
        response = requests.get(
            FRED_CSV_URL, params={"id": series_id, "cosd": start, "coed": end},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataError(f"FRED {series_id} retrieval failed: {exc}") from exc
    return parse_fred_csv(response.text, series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    frame = pd.read_csv(StringIO(text), na_values=["."])
    date_column = frame.columns[0]
    if series_id not in frame.columns:
        raise DataError(f"FRED {series_id}: column missing from CSV")
    series = pd.to_numeric(frame[series_id], errors="coerce")
    series.index = pd.to_datetime(frame[date_column])
    series.index.name = "Date"
    if series.index.has_duplicates:
        raise DataError(f"FRED {series_id}: duplicate dates")
    return series.rename(series_id).sort_index()


def align_levels(raw: Mapping[str, pd.Series], start: str = HISTORY_START, end: str = HISTORY_END) -> pd.DataFrame:
    """Keep dates where every series is observed; convert percent yields to decimals."""
    frame = pd.concat({name: raw[name] for name in FRED_SERIES}, axis=1).sort_index().loc[start:end]
    levels = frame.dropna()
    if levels.empty or levels.index[0] != pd.Timestamp(start) or levels.index[-1] != pd.Timestamp(end):
        raise DataError(f"Aligned data must cover {start} through {end} on both endpoints")
    if (levels["SP500"] <= 0).any():
        raise DataError("SP500 levels must be positive")
    levels = levels.assign(DGS10=levels["DGS10"] / 100.0, BAA10Y=levels["BAA10Y"] / 100.0)
    return levels.rename(columns={"SP500": "SPX"})


def daily_changes(levels: pd.DataFrame) -> pd.DataFrame:
    """SPX log returns and absolute decimal changes in the government yield and spread."""
    return pd.DataFrame({
        "SPX": np.log(levels["SPX"]).diff(),
        "DGS10": levels["DGS10"].diff(),
        "BAA10Y": levels["BAA10Y"].diff(),
    }).iloc[1:]


def horizon_changes(changes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Overlapping h-day changes; log returns and yield changes are both additive."""
    return changes.rolling(horizon).sum().iloc[horizon - 1:]


# ---------------------------------------------------------------- pricing

def bond_price(yield_decimal: object, coupon_rate: float, years: int = BOND_MATURITY_YEARS,
               frequency: int = COUPONS_PER_YEAR, face: float = FACE_PER_BOND) -> np.ndarray | float:
    """Price per `face` of a bullet bond at a yield to maturity compounded `frequency` times a year."""
    y = np.asarray(yield_decimal, dtype=float)
    periods = np.arange(1, years * frequency + 1)
    discount = (1.0 + y[..., None] / frequency) ** -periods
    price = face * (coupon_rate / frequency * discount.sum(axis=-1) + discount[..., -1])
    return price if price.ndim else float(price)


def modified_duration(yield_decimal: float, coupon_rate: float, bump: float = 1e-6) -> float:
    up, down = bond_price(yield_decimal + bump, coupon_rate), bond_price(yield_decimal - bump, coupon_rate)
    return -(up - down) / (2 * bump) / bond_price(yield_decimal, coupon_rate)


def bond_yields(levels: Mapping[str, object]) -> dict[str, object]:
    return {"Govt_10Y": levels["DGS10"],
            "Corp_A_10Y": np.asarray(levels["DGS10"]) + np.asarray(levels["BAA10Y"])}


def build_positions(valuation_levels: pd.Series) -> dict[str, Position]:
    spx = valuation_levels["SPX"]
    positions = {"SPX": Position("SPX", POSITION_VALUES["SPX"], spx, POSITION_VALUES["SPX"] / spx,
                                 None, None, None)}
    for holding, y in bond_yields(valuation_levels).items():
        y = float(y)
        price = bond_price(y, COUPON_RATES[holding])
        value = POSITION_VALUES[holding]
        duration = modified_duration(y, COUPON_RATES[holding])
        positions[holding] = Position(holding, value, price, value / price, y, duration,
                                      duration * value * 1e-4)
    return positions


def scenario_pnl(positions: Mapping[str, Position], valuation_levels: pd.Series,
                 shocks: pd.DataFrame | Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Full-revaluation USD P&L per holding and for the portfolio under factor shocks."""
    spx = np.asarray(shocks["SPX"], dtype=float)
    shocked = {"DGS10": valuation_levels["DGS10"] + np.asarray(shocks["DGS10"], dtype=float),
               "BAA10Y": valuation_levels["BAA10Y"] + np.asarray(shocks["BAA10Y"], dtype=float)}
    pnl = {"SPX": positions["SPX"].market_value * np.expm1(spx)}
    for holding, y in bond_yields(shocked).items():
        p = positions[holding]
        pnl[holding] = p.quantity * bond_price(y, COUPON_RATES[holding]) - p.market_value
    frame = pd.DataFrame(pnl, columns=list(HOLDINGS))
    frame["Portfolio"] = frame[list(HOLDINGS)].sum(axis=1)
    return frame


# ---------------------------------------------------------------- risk measures

def var_es(pnl: object, confidence: float) -> tuple[float, float]:
    losses = -np.asarray(pnl, dtype=float)
    var = float(np.quantile(losses, confidence, method=QUANTILE_METHOD))
    return var, float(losses[losses >= var].mean())


def simulate_shocks(changes: pd.DataFrame, horizon: int, paths: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sum `horizon` IID multivariate-normal daily draws calibrated to `changes`."""
    mean = changes.mean().to_numpy()
    covariance = changes.cov(ddof=ESTIMATION_DDOF).to_numpy()
    draws = rng.multivariate_normal(mean, covariance, size=(paths, horizon), method="cholesky")
    return pd.DataFrame(draws.sum(axis=1), columns=changes.columns)


@dataclass(frozen=True)
class Analysis:
    levels: pd.DataFrame
    changes: pd.DataFrame
    positions: dict[str, Position]
    scenarios: dict[tuple[str, int], pd.DataFrame]  # (method, horizon) -> P&L frame
    risk: pd.DataFrame


def run_analysis(levels: pd.DataFrame, paths: int = MC_PATHS, seed: int = MASTER_SEED) -> Analysis:
    changes = daily_changes(levels)
    valuation = levels.iloc[-1]
    positions = build_positions(valuation)
    horizon_seeds = np.random.SeedSequence(seed).spawn(len(HORIZONS))
    scenarios: dict[tuple[str, int], pd.DataFrame] = {}
    for horizon, child in zip(HORIZONS, horizon_seeds):
        scenarios["historical", horizon] = scenario_pnl(positions, valuation, horizon_changes(changes, horizon))
        shocks = simulate_shocks(changes, horizon, paths, np.random.default_rng(child))
        scenarios["monte_carlo", horizon] = scenario_pnl(positions, valuation, shocks)
    rows = []
    for (method, horizon), pnl in scenarios.items():
        for confidence in CONFIDENCE_LEVELS:
            var = {}
            for risk_id in RISK_IDS:
                var[risk_id], es = var_es(pnl[risk_id], confidence)
                rows.append({"method": method, "horizon_days": horizon, "confidence": confidence,
                             "risk_id": risk_id, "scenarios": len(pnl), "var_usd": var[risk_id],
                             "es_usd": es})
            rows.append({"method": method, "horizon_days": horizon, "confidence": confidence,
                         "risk_id": "Diversification", "scenarios": len(pnl),
                         "var_usd": sum(var[h] for h in HOLDINGS) - var["Portfolio"], "es_usd": np.nan})
    return Analysis(levels, changes, positions, scenarios, pd.DataFrame(rows))


# ---------------------------------------------------------------- reporting

def positions_table(positions: Mapping[str, Position]) -> pd.DataFrame:
    return pd.DataFrame([vars(p) for p in positions.values()])


def render_figure(analysis: Analysis) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    confidence = max(CONFIDENCE_LEVELS)
    for row, horizon in enumerate(HORIZONS):
        for col, method in enumerate(("historical", "monte_carlo")):
            ax = axes[row, col]
            pnl = analysis.scenarios[method, horizon]["Portfolio"]
            var, es = var_es(pnl, confidence)
            ax.hist(pnl, bins=60, color="#4C72B0", alpha=0.75)
            ax.axvline(-var, color="#C44E52", label=f"-VaR {confidence:.0%}: {var:,.0f}")
            ax.axvline(-es, color="#8172B2", linestyle="--", label=f"-ES {confidence:.0%}: {es:,.0f}")
            ax.set_title(f"{method.replace('_', ' ').title()} - {horizon}-day portfolio P&L")
            ax.set_xlabel("P&L (USD)")
            ax.legend(fontsize=8)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def write_outputs(analysis: Analysis, directory: Path = OUTPUT_DIRECTORY) -> tuple[Path, ...]:
    payloads = {
        "market_data.csv": analysis.levels.reset_index(),
        "positions.csv": positions_table(analysis.positions),
        "risk_measures.csv": analysis.risk,
    }
    directory.mkdir(exist_ok=True)
    for name, frame in payloads.items():
        frame.to_csv(directory / name, index=False, date_format="%Y-%m-%d", float_format="%.10g")
    (directory / "pnl_distributions.png").write_bytes(render_figure(analysis))
    return tuple(directory / name for name in OUTPUT_FILENAMES)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the STA2540 group-project VaR/ES analysis.")
    parser.parse_args(argv)
    try:
        levels = align_levels({name: fetch_fred_series(name) for name in FRED_SERIES})
        analysis = run_analysis(levels)
        paths = write_outputs(analysis)
    except (DataError, OSError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"{len(levels)} aligned dates {HISTORY_START}..{HISTORY_END}; wrote {len(paths)} files to {OUTPUT_DIRECTORY}")
    portfolio = analysis.risk[analysis.risk["risk_id"] == "Portfolio"]
    print(portfolio[["method", "horizon_days", "confidence", "var_usd", "es_usd"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
