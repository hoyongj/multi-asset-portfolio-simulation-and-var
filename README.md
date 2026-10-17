# Multi-Asset Portfolio Simulation and VaR

This local Python project studies one-year terminal outcomes and one-day market-risk forecasts for a fixed five-holding portfolio. It retrieves observed market data, calibrates Gaussian risk-factor models, fully reprices hypothetical positions, backtests two VaR methods, and produces six CSV tables and three Matplotlib figures. It is a reproducible research analysis, not an investment recommendation or a trading system.

| Holding | Initial weight | Instrument represented |
| --- | ---: | --- |
| Treasury_2Y | 10% | Hypothetical 2-year zero-coupon US Treasury |
| Treasury_5Y | 10% | Hypothetical 5-year zero-coupon US Treasury |
| Treasury_10Y | 10% | Hypothetical 10-year zero-coupon US Treasury |
| LQD | 30% | iShares iBoxx $ Investment Grade Corporate Bond ETF |
| SPY | 40% | SPDR S&P 500 ETF Trust |

USD 1,000,000 is the initial reporting notional: USD 100,000 in each Treasury, USD 300,000 in LQD, and USD 400,000 in SPY. Dollar values, P&L, VaR, and contributions scale with this notional. Returns and risk expressed as percentages of portfolio value are invariant to it. Fractional positions are allowed.

## Sources and sample

The requested period is **2022-09-01 through 2026-08-31, inclusive**, subject to source availability. No local market-data inputs are required.

- The [Federal Reserve Board's fitted nominal yield curve](https://www.federalreserve.gov/data/nominal-yield-curve.htm), described by Gürkaynak, Sack, and Wright, supplies fitted continuously compounded zero-coupon yields. The module retrieves the Board's [CSV](https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv) with `requests` and uses `SVENY01`, `SVENY02`, `SVENY04`, `SVENY05`, `SVENY09`, and `SVENY10` as six separate factors.
- [Yahoo Finance LQD history](https://finance.yahoo.com/quote/LQD/history/) and [SPY history](https://finance.yahoo.com/quote/SPY/history/) supply observed ETF prices, accessed with the open-source [`yfinance` download interface](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html). Downloads explicitly use `auto_adjust=False` and retain distinct `Close` and `Adj Close` columns. The exclusive download end is 2026-09-01 to include 2026-08-31. This project is independent of Yahoo Finance and `yfinance`.

Observed fitted yields and ETF histories are source data. The zero-coupon holdings are hypothetical instruments valued from the fitted curve; they are not histories of purchased Treasury securities. Terminal paths and forecast scenarios are simulated or replayed model outcomes, not observations or promises of future performance. Historical realized backtest P&L is also a hypothetical portfolio valuation.

Dates, required columns, real numeric types, duplicate dates, missing values, finite values, source coverage, and positive ETF prices are checked. Negative yields are permitted. All six yields and both ETF price fields are aligned on common complete dates **before** constructing changes. Missing observations are dropped, never filled. Changes span consecutive retained dates, so source-calendar gaps can span several calendar days; they are not rescaled. `data_quality.csv` records missing cells, dropped dates, absent weekdays, coverage shortfalls, and the maximum calendar gap. An absent weekday can be a legitimate holiday.

At least 751 aligned levels, hence **750 aligned changes**, are required for the complete analysis. Missing required series, failed downloads, or insufficient history produce an explicit error before result files are written. No forecasts are fabricated and the window requirements are not reduced.

## Valuation and allocations

Source yields are percentages; dividing by 100 gives decimal yields. Daily yield changes remain in decimal-yield units internally; multiplying by 10,000 reports basis points. ETF changes are natural logarithms of consecutive aligned **Adj Close** ratios, never `Close` returns.

For remaining maturity `T` years and decimal continuous yield `y(T)`, a Treasury costs `exp(-y(T) * T)` USD per USD 1 face. Yields are linearly interpolated between the bracketing fitted tenors, including the 1-2, 4-5, and 9-10 year intervals. There is no extrapolation outside 1-10 years. Initial face quantities equal allocated dollars divided by the allocation-date price at the original 2-, 5-, or 10-year maturity.

The terminal analysis allocates at the latest common valid date on or before 2026-08-31 and calibrates on the latest 500 changes ending there. The backtest makes a separate allocation at the origin immediately preceding its first selected forecast. These dates and fixed quantities appear in `data_quality.csv`; terminal and VaR tables also carry allocation dates. Quantities stay fixed within each analysis, with no interim rebalancing; market weights drift.

For the completed sample, the terminal allocation date is **2026-08-31**, the backtest allocation date is **2025-08-28**, and the 250 forecast dates run from **2025-08-29 through 2026-08-31**. The recorded output dates remain authoritative if source availability changes.

An ETF's initial shares equal its allocated dollars divided by allocation-date `Close`. Every later holding value equals its initial allocation times `Adj Close(t) / Adj Close(allocation)`, representing reinvested distributions and split adjustments. Terminal adjusted prices equal allocation-date `Adj Close` times the exponential of cumulative simulated log returns. The separate terminal `unit_price` is the total-return-equivalent value per initial share: initial `Close` times that same adjusted-price ratio. It need not equal an observed future raw closing price.

The terminal horizon ages each Treasury exactly one year, leaving maturities **1, 4, and 9 years**. Backtest aging uses elapsed aligned forecast steps divided by 252, consistently for origin, scenario, and realized valuations. Thus its final maturities are initial maturity minus 250/252 years, even when the calendar span exceeds a year. This explicit trading-time convention keeps the short bond inside the available fitted-tenor range. The reusable historical valuation helper defaults to actual calendar days/365; the complete backtest supplies its trading-time age explicitly. Every scenario uses full exponential Treasury repricing and ETF adjusted-price scaling, with no duration approximation.

## Fixed settings and randomness

All configuration is in `portfolio_var.py`.

| Setting | Value |
| --- | --- |
| Calibration / rolling estimation window | 500 aligned daily changes |
| Backtest forecasts | Latest 250 eligible one-day forecasts per holding and method |
| Loss VaR confidence | 99%; expected exception probability 1% |
| Trading days per year / terminal horizon | 252 / 252 daily steps |
| Terminal paths per analysis | 10,000 |
| Parametric scenarios per forecast | 10,000 |
| Historical scenarios per forecast | 500 |
| Sample standard deviation and covariance | `ddof=1` |
| Empirical loss quantile | NumPy `method="linear"` |
| Attribution proportional position bump | 0.0001 (0.01%) in each direction |
| Master seed | 42 |

`numpy.random.SeedSequence(42).spawn(4)` generates the following fixed named child seeds, in this order, using one unsigned 64-bit state value per child:

| Purpose | Child seed |
| --- | ---: |
| Standalone terminal simulation | 16138347438539916964 |
| Correlated terminal simulation | 134183728835869882 |
| Rolling parametric VaR | 11601846009883706861 |
| Risk attribution, reserved | 3747978530954135749 |

Each rolling forecast obtains its own seed from `SeedSequence(rolling_child_seed, spawn_key=(ordinal,)).generate_state(1, dtype=np.uint64)[0]`. The ordinal is its position among all eligible forecasts after the first 500 changes, rather than its position in a selected subset. Execution order therefore does not change that forecast's scenarios. Actual forecast seeds are recorded in `var_forecasts.csv`. Attribution reuses the latest parametric scenarios and consumes no additional randomness; its child stream remains reserved. No global NumPy random state is used. Identical source data, configuration, pinned libraries, and numerical builds give reproducible results; upstream data revisions can change later runs.

## Terminal simulations

Univariate Gaussian means and sample standard deviations are fitted for individual daily factors. Each standalone holding receives 10,000 paths of 252 independent daily draws from its applicable marginal. Treasury standalone models use the 1-, 4-, or 9-year yield-change marginal and the allocation-date yield at that terminal maturity, while preserving the actual initial 2-, 5-, or 10-year purchase price. ETFs use their respective adjusted-price log-return marginals. Standalone holdings are analyzed individually; their independent sum is not published as the portfolio result.

The correlated simulation fits an eight-factor Gaussian model ordered as the six Treasury tenors above, LQD, then SPY. Daily draws preserve the fitted cross-factor covariance; draws are IID through time. All eight factors evolve separately over 252 days. Treasury terminal values use their aged maturities and terminal fitted curve, while ETF values accumulate simulated log returns. Both analyses preserve initial face amounts and ETF positions.

Covariance handling only clips negative eigenvalues attributable to numerical roundoff: their magnitude must be within `10 * number_of_factors * machine_epsilon * eigenvalue_scale`. Materially indefinite covariance is rejected. There is no arbitrary diagonal jitter or variance floor. Correction counts and norms are disclosed in the quality table.

## VaR, coverage, diversification, and attribution

For each forecast, exactly the immediately preceding 500 aligned changes are available at its origin. Explicit index checks require that the estimation window ends at the origin and before the realized P&L date. Parametric VaR fits all eight Gaussian means and covariances and generates 10,000 one-day full-revaluation scenarios. Historical VaR applies each of those 500 observed eight-factor changes to current levels and fully reprices the same fixed holdings at the next step's maturity.

Loss is current holding value minus scenario value. VaR is the signed 99th percentile of that USD loss, without flooring at zero; sufficiently consistent gains can yield negative loss VaR. Realized P&L is actual next-day value minus current value on the same valuation basis. An exception is strictly `realized_pnl < -forecast_var`. Each method reports all five allocated holdings and the total portfolio: 250 forecasts each, or 3,000 forecast rows overall. Figures plot realized P&L against **negative VaR** thresholds.

Kupiec unconditional coverage compares the observed exception rate with 1%, using a likelihood-ratio statistic and a one-degree-of-freedom chi-squared p-value. Christoffersen independence uses consecutive exception transitions (249 transitions for 250 forecasts); conditional coverage adds the Kupiec and independence statistics and uses two degrees of freedom. Stable log-likelihood calculations implement `0 * log(0) = 0`, including zero or all exceptions, without pseudocounts. If a conditioning state is never observed, its transition probability is not identified and is written as `not_identified`, with an explicit status column. The independence likelihood-ratio contribution is zero for a constant sequence. Likelihood statistics and p-values remain finite for the full backtest; an unidentified probability is not fabricated.

At the latest forecast, each method's diversification effect equals the sum of the five allocated standalone holding VaRs minus portfolio VaR. These holding VaRs already include their allocated positions: weights are not applied a second time.

Latest parametric Euler contributions use the **same** full-revaluation scenario losses for all five holdings. For each position, a central proportional finite difference approximates the derivative of portfolio VaR, multiplied by position size. Both perturbed portfolios retain the common scenarios and fixed quantities for other positions. The result reports five contributions, their sum, portfolio VaR, and the residual `portfolio_VaR - contribution_sum`. Contributions are not normalized to force reconciliation. Empirical quantile changes and finite differences can produce a residual or negative contributions.

Low coverage-test p-values suggest incompatibility with the specified coverage or independence hypothesis. Large p-values do not establish correctness, and the analysis does not require a particular statistical-test outcome to succeed. Terminal percentiles describe simulated distributions; they are not confidence intervals for estimated parameters.

## Run locally

Run these exact PowerShell commands from the project root. The tested interpreter is Python 3.14.7; verify that `python --version` reports that version before creating the environment. The six direct dependencies are pinned in `requirements.txt`. Internet access is needed for the analysis, while the synthetic `unittest` suite runs offline.

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest -v
.venv\Scripts\python.exe portfolio_var.py
```

Environment creation and installation are setup steps; reuse the existing environment for subsequent tests and analyses. The single analysis command performs retrieval, validation, alignment, simulation, rolling forecasts, coverage tests, diversification, attribution, and reporting with the fixed settings. It renders figures in memory, closes them after rendering, and creates `outputs/` only after successful analysis and report preparation. Subsequent successful runs replace the same nine result files. No additional output files are produced.

## Generated results

| File in `outputs/` | Contents |
| --- | --- |
| `data_quality.csv` | Tidy source coverage, raw missing-cell and duplicate checks, row counts, dropped dates, alignment counts, units, maturity and aging conventions, source-period deviations, fixed allocations, seeds, and numerical covariance diagnostics. |
| `terminal_paths.csv` | Exactly 10,000 path rows. Prefixed columns distinguish five standalone holding distributions from correlated holding and portfolio results. Includes terminal prices or per-unit values, adjusted ETF prices, allocated holding values and decimal returns, correlated portfolio value and return, seeds, and calibration/horizon metadata. |
| `terminal_summary.csv` | Initial levels, observation counts, means, sample standard deviations, minima, 1st/50th/99th percentiles, and maxima for all 36 published terminal holding/portfolio metrics, with explicit units and analysis context. |
| `var_forecasts.csv` | All 3,000 method/holding forecasts, USD VaR and realized P&L, exception indicators, estimation dates, origins, forecast dates and ordinals, scenario counts, and forecast seeds. |
| `var_summary.csv` | Twelve method/holding rows: forecast and exception counts/rates, Kupiec and Christoffersen statistics and p-values, transition counts/probabilities and identification status, sample and forecast context. |
| `risk_summary.csv` | Latest-forecast holding VaRs, their sum, portfolio VaR, and diversification for both methods; five parametric Euler contributions, their total, explicit residual, and position-bump setting. All risk amounts are USD. |
| `terminal_distributions.png` | Six labelled return-distribution panels: standalone and correlated holding marginals, plus the correlated portfolio, with source, calibration, path, and horizon labels. |
| `var_backtests.png` | Six labelled realized-P&L panels with parametric and historical negative-VaR thresholds and exception markers, dates, window/scenario counts, and valuation context. |
| `risk_contributions.png` | Five holding-level parametric Euler contributions, alongside portfolio VaR, contribution sum, and the unscaled reconciliation residual. |

CSV return columns are **decimal** returns; terminal return plots use **percentages**. Dollar prices, values, P&L, and VaR have labelled USD units; Treasury unit prices are per USD 1 face. Tables carry requested and actual sample periods, reporting notional, confidence level, and the relevant allocation, calibration/rolling window, forecast count, scenario/path count, and horizon. Historical replay scenarios are observations, not random simulation paths.

## Interpretation and limitations

The sample is finite and may omit stress regimes. Its aligned observations, fitted parameters, and results depend on the source vintage and availability. Complete-date filtering can remove information and turn an adjacent retained-date change into a multi-calendar-day change. Boundary shortfalls are disclosed instead of silently extending the requested period.

Gaussian factors impose light tails, stable estimated dependence, and IID daily changes; negative yields are possible. Long-horizon accumulation extrapolates daily means and covariances over a year, and estimates from 500 changes are uncertain. Fitted Treasury yields and linear maturity interpolation approximate a curve and do not represent executable bond quotes. Trading-time aging is a modeling convention, especially across calendar gaps. Adjusted ETF prices are vendor-provided total-return proxies that may be revised.

Only about 2.5 exceptions are expected in 250 forecasts at 99% confidence under correct coverage. Coverage and independence tests therefore have limited tail-event evidence, and their chi-squared p-values are asymptotic, especially fragile at boundary counts. Ten thousand Monte Carlo paths and 500 historical scenarios provide finite tail resolution; empirical VaR and numerical Euler attribution are sensitive to tail order statistics.

The analysis excludes transaction costs, bid-ask spreads, slippage, taxes, funding and financing, margin, liquidity, default and recovery, FX, intraday risk, operational risk, and explicit corporate-credit-spread factors. LQD's observed return factor bundles its market exposures rather than decomposing them. No interim rebalancing, investment advice, or guarantee of future risk coverage is implied.
