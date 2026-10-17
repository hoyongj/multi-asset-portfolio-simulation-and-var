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

Source yields are divided by 100 to convert percentages to decimal yields. Daily yield changes use decimal units; multiply by 10,000 for basis points. ETF changes are log returns calculated only from **Adj Close** ratios.

A Treasury with remaining maturity `T` and decimal yield `y(T)` costs `exp(-y(T) * T)` USD per USD 1 face. Yields are linearly interpolated between bracketing fitted tenors, including 1-2, 4-5, and 9-10 years, with no extrapolation outside 1-10 years. Initial face quantities equal allocated dollars divided by the price at the original 2-, 5-, or 10-year maturity.

The terminal analysis allocates at the latest common valid date on or before 2026-08-31 and uses the latest 500 changes ending there. The backtest allocates separately at its first forecast origin. For the saved results, these dates are **2026-08-31** and **2025-08-28**, respectively; forecasts run from **2025-08-29 through 2026-08-31**. Allocation dates and quantities are recorded in [data_quality.csv](outputs/data_quality.csv). Each analysis holds quantities fixed without rebalancing, so market weights drift.

Initial ETF shares equal allocated dollars divided by allocation-date `Close`. Later holding values equal the initial allocation times `Adj Close(t) / Adj Close(allocation)`, including reinvested distributions and split adjustments. Simulated terminal `Adj Close` equals its allocation-date value times the exponential of cumulative log returns. Terminal `unit_price` equals initial `Close` times the adjusted-price ratio: a total-return-equivalent value per initial share.

After 252 trading days, Treasury maturities are **1, 4, and 9 years**. Backtest aging uses elapsed aligned steps/252 for both scenario and realized valuations, leaving initial maturity minus 250/252 years at its end. This convention keeps the short bond within the fitted-tenor range even if the calendar span exceeds a year. The general valuation helper defaults to calendar days/365; the backtest supplies its trading-time age explicitly. All scenarios fully reprice holdings without a duration approximation.

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

Each forecast uses exactly the preceding 500 aligned changes. Index checks ensure that the window ends at the forecast origin and excludes the realized forecast-day change. Parametric VaR fits the eight-factor Gaussian model and draws 10,000 one-day scenarios. Historical VaR replays the same window's 500 observed changes. Both methods apply changes to current levels and fully reprice fixed holdings at the next step's maturity.

VaR is the 99th percentile of USD loss: current value minus scenario value. It is not floored at zero, so consistent modeled gains can produce negative VaR. Realized P&L is actual next-day value minus current value; an exception occurs when `realized_pnl < -forecast_var`. Each method produces 250 forecasts for all five holdings and the portfolio, giving 3,000 rows. Backtest plots compare P&L with **negative VaR** thresholds.

Kupiec tests whether the exception rate matches 1%, using a likelihood-ratio statistic with a one-degree-of-freedom chi-squared p-value. Christoffersen independence tests whether exceptions cluster, using the 249 transitions between 250 forecasts and one degree of freedom. Conditional coverage combines the Kupiec and independence statistics and uses two degrees of freedom. Zero counts use `0 * log(0) = 0`, without pseudocounts. An unobserved conditioning state's probability is reported as `not_identified`; constant sequences have an independence statistic of zero.

At the latest forecast, diversification equals the sum of the five allocated holding VaRs minus portfolio VaR. Holding VaRs already reflect position sizes, so weights are not applied again.

Euler attribution uses the latest parametric loss scenarios for all five holdings. Each position is bumped by +/-0.01%, with other positions fixed; central finite differences estimate the VaR derivative, multiplied by position size. The five contributions, their sum, portfolio VaR, and residual `portfolio_VaR - contribution_sum` are reported without rescaling. Quantile changes can produce a residual or negative contributions.

Low test p-values suggest a coverage or independence mismatch. Large p-values do not establish model correctness, and no particular test outcome is required for the analysis to succeed.

## Simulation and backtest results

These saved results use the **2022-09-01 through 2026-08-31** sample and the documented seeds. Terminal simulations allocate on **2026-08-31**, calibrate on 500 changes from **2024-08-27 through 2026-08-31**, and generate **10,000 paths over 252 trading days**. Returns below are percentages; the underlying CSV stores decimal returns.

| Terminal distribution | Mean return | 1st percentile | Median | 99th percentile |
| --- | ---: | ---: | ---: | ---: |
| Standalone 2-year Treasury | 4.51% | 3.17% | 4.51% | 5.92% |
| Standalone 5-year Treasury | 3.33% | -4.30% | 3.29% | 11.45% |
| Standalone 10-year Treasury | 1.16% | -14.41% | 0.95% | 19.28% |
| Standalone LQD | 2.04% | -12.41% | 1.84% | 18.21% |
| Standalone SPY | 19.83% | -20.01% | 18.13% | 74.52% |
| Correlated portfolio | 9.55% | -8.89% | 8.95% | 33.25% |

The correlated portfolio's mean terminal value is **USD 1,095,458** from the USD 1,000,000 initial notional; its 1st percentile is **USD 911,129**. These are conditional simulated outcomes, not confidence intervals for estimated parameters. Full price, value, and return statistics are in [terminal_summary.csv](outputs/terminal_summary.csv).

The **99% one-day** portfolio backtest covers **250 forecasts from 2025-08-29 through 2026-08-31**, each using a 500-change rolling window. VaR below is for the latest forecast, **2026-08-31**; exception counts and test p-values cover the full backtest.

| Method | Latest portfolio VaR (USD) | Exceptions / 250 | Exception rate | Kupiec p-value | Christoffersen conditional-coverage p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parametric | 12,930.35 | 3 | 1.2% | 0.758 | 0.919 |
| Historical | 15,302.50 | 1 | 0.4% | 0.278 | 0.553 |

Only 2.5 exceptions are expected over 250 forecasts at 99% confidence, so these tests have limited tail evidence. Individual holdings can show different coverage: the historical 5-year Treasury had no exceptions and a Kupiec p-value of 0.025. See [var_summary.csv](outputs/var_summary.csv) for all holding and portfolio results.

At the latest forecast, diversification effects are **USD 2,670.14** for parametric VaR and **USD 2,552.84** for historical VaR. SPY has the largest parametric Euler contribution, **USD 11,399.24**; the 2-year Treasury has a small negative contribution of **USD -6.37**. The five contributions sum to **USD 12,930.35**, with a disclosed residual of approximately **USD -3.29e-08**. Full reconciliation is in [risk_summary.csv](outputs/risk_summary.csv). These sample results do not guarantee future returns or risk coverage.

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

## Repository and outputs

| Location | Contents |
| --- | --- |
| [portfolio_var.py](portfolio_var.py) | All configuration and functions for retrieval, validation, pricing, simulation, VaR backtesting, coverage tests, diversification, attribution, and output generation. Run this module to perform the complete analysis. |
| [test_portfolio_var.py](test_portfolio_var.py) | Consolidated offline `unittest` suite using synthetic data. |
| [outputs/data_quality.csv](outputs/data_quality.csv) | Source coverage, missing and dropped dates, alignment, units, conventions, allocations, seeds, and covariance corrections. |
| [outputs/terminal_paths.csv](outputs/terminal_paths.csv) | 10,000 path rows containing standalone holding distributions and correlated holding and portfolio results. |
| [outputs/terminal_summary.csv](outputs/terminal_summary.csv) | Initial levels, units, and distribution statistics for all 36 terminal metrics. |
| [outputs/var_forecasts.csv](outputs/var_forecasts.csv) | 3,000 forecasts with VaR, realized P&L, exceptions, estimation-window dates, scenario counts, and seeds. |
| [outputs/var_summary.csv](outputs/var_summary.csv) | Forecast and exception counts and rates, plus Kupiec and Christoffersen coverage tests. |
| [outputs/risk_summary.csv](outputs/risk_summary.csv) | Latest holding and portfolio VaRs, diversification effects, five Euler contributions, their sum, and the attribution residual. |
| [outputs/terminal_distributions.png](outputs/terminal_distributions.png) | Six panels comparing standalone and correlated terminal-return distributions. |
| [outputs/var_backtests.png](outputs/var_backtests.png) | Six panels showing realized P&L, negative-VaR thresholds, and exceptions for both methods. |
| [outputs/risk_contributions.png](outputs/risk_contributions.png) | Five holding-level contributions and their reconciliation to portfolio VaR. |

CSV return columns are **decimal** returns; terminal return plots use **percentages**. Dollar prices, values, P&L, and VaR have labelled USD units; Treasury unit prices are per USD 1 face. Tables carry requested and actual sample periods, reporting notional, confidence level, and the relevant allocation, calibration/rolling window, forecast count, scenario/path count, and horizon. Historical replay scenarios are observations, not random simulation paths.

## Interpretation and limitations

The sample is finite and may omit stress regimes. Its aligned observations, fitted parameters, and results depend on the source vintage and availability. Complete-date filtering can remove information and turn an adjacent retained-date change into a multi-calendar-day change. Boundary shortfalls are disclosed instead of silently extending the requested period.

Gaussian factors impose light tails, stable estimated dependence, and IID daily changes; negative yields are possible. Long-horizon accumulation extrapolates daily means and covariances over a year, and estimates from 500 changes are uncertain. Fitted Treasury yields and linear maturity interpolation approximate a curve and do not represent executable bond quotes. Trading-time aging is a modeling convention, especially across calendar gaps. Adjusted ETF prices are vendor-provided total-return proxies that may be revised.

Only about 2.5 exceptions are expected in 250 forecasts at 99% confidence under correct coverage. Coverage and independence tests therefore have limited tail-event evidence, and their chi-squared p-values are asymptotic, especially fragile at boundary counts. Ten thousand Monte Carlo paths and 500 historical scenarios provide finite tail resolution; empirical VaR and numerical Euler attribution are sensitive to tail order statistics.

The analysis excludes transaction costs, bid-ask spreads, slippage, taxes, funding and financing, margin, liquidity, default and recovery, FX, intraday risk, operational risk, and explicit corporate-credit-spread factors. LQD's observed return factor bundles its market exposures rather than decomposing them. No interim rebalancing, investment advice, or guarantee of future risk coverage is implied.
