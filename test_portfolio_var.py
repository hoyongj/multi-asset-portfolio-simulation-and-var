r"""Synthetic, offline tests for the data and pricing foundation.

Run with: .venv\Scripts\python.exe -m unittest -v
All retrieval calls are mocked; no market data or external local files are used.
"""

import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import portfolio_var as pv


def synthetic_sources():
    dates = pd.bdate_range("2022-09-01", periods=6, name="Date").as_unit("ns")
    yields = pd.DataFrame(
        {column: 2.0 + tenor / 10 + np.arange(6) / 100
         for tenor, column in zip(pv.TREASURY_TENORS, pv.YIELD_COLUMNS)},
        index=dates,
    )
    etfs = {
        "LQD": pd.DataFrame(
            {"Close": [100., 105., 90., 115., 95., 110.],
             "Adj Close": [80., 81., 83., 82., 84., 85.]}, index=dates,
        ),
        "SPY": pd.DataFrame(
            {"Close": [200., 210., 220., 180., 190., 230.],
             "Adj Close": [150., 153., 154., 156., 155., 159.]}, index=dates,
        ),
    }
    return yields, etfs


class SyntheticTestCase(unittest.TestCase):
    def setUp(self):
        self.yields, self.etfs = synthetic_sources()
        self.start, self.end = self.yields.index[[0, -1]]

    def prepare(self):
        return pv.prepare_market_data(self.yields, self.etfs, self.start, self.end)


class ConfigurationAndSeedTests(unittest.TestCase):
    def test_fixed_holdings_and_weights(self):
        self.assertEqual(pv.HOLDINGS, ("Treasury_2Y", "Treasury_5Y", "Treasury_10Y", "LQD", "SPY"))
        self.assertEqual(pv.WEIGHTS, (0.1, 0.1, 0.1, 0.3, 0.4))
        self.assertAlmostEqual(sum(pv.WEIGHTS), 1.0)
        self.assertAlmostEqual(sum(pv.WEIGHTS[:3]), 0.3)
        self.assertEqual(pv.INITIAL_PORTFOLIO_VALUE, 1_000_000)

    def test_fixed_analysis_settings(self):
        self.assertEqual((pv.ANALYSIS_START, pv.ANALYSIS_END), ("2022-09-01", "2026-08-31"))
        self.assertEqual(pv.TREASURY_TENORS, (1, 2, 4, 5, 9, 10))
        self.assertEqual(
            (pv.CALIBRATION_WINDOW, pv.BACKTEST_FORECASTS, pv.VAR_CONFIDENCE,
             pv.TRADING_DAYS, pv.SIMULATION_PATHS, pv.MASTER_SEED),
            (500, 250, 0.99, 252, 10_000, 42),
        )

    def test_child_seeds_are_reproducible_named_and_distinct(self):
        seeds = pv.create_child_seeds()
        self.assertEqual(tuple(seeds), (
            "standalone_terminal_simulation", "correlated_terminal_simulation",
            "rolling_parametric_var", "risk_attribution",
        ))
        self.assertEqual(seeds, pv.create_child_seeds(42))
        self.assertEqual(seeds, dict(pv.CHILD_SEEDS))
        self.assertEqual(len(set(seeds.values())), 4)
        self.assertNotEqual(seeds, pv.create_child_seeds(43))
        before = np.random.get_state()
        pv.create_child_seeds()
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_named_streams_do_not_share_consumption(self):
        seeds = pv.create_child_seeds()
        np.random.default_rng(seeds["standalone_terminal_simulation"]).normal(size=100)
        actual = np.random.default_rng(seeds["correlated_terminal_simulation"]).normal(size=5)
        expected = np.random.default_rng(
            pv.create_child_seeds()["correlated_terminal_simulation"]
        ).normal(size=5)
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_master_seeds(self):
        for seed in (-1, 1.5, True, "42"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                pv.create_child_seeds(seed)


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.curve = {1: .01, 2: .03, 4: .08, 5: .12, 9: .20, 10: .30}

    def test_continuous_compounding_per_unit_face(self):
        self.assertAlmostEqual(pv.zero_coupon_price(.05, 5), math.exp(-.25))
        self.assertAlmostEqual(pv.zero_coupon_price(-.01, 2), math.exp(.02))
        self.assertEqual(pv.zero_coupon_price(.05, 0), 1.0)

    def test_vector_prices_and_broadcasting(self):
        actual = pv.zero_coupon_price(np.array([[.01], [.02]]), [1., 4., 9.])
        np.testing.assert_allclose(actual, np.exp(-np.array([[.01], [.02]]) * [1., 4., 9.]))

    def test_prices_reject_invalid_numbers_and_maturities(self):
        for yield_value, maturity in ((np.inf, 2), (.01, np.nan), (.01, -1), (True, 2), (1j, 2)):
            with self.subTest(yield_value=yield_value, maturity=maturity):
                with self.assertRaises(pv.DataValidationError):
                    pv.zero_coupon_price(yield_value, maturity)
        for yield_value in (-1_000., 1_000.):
            with self.assertRaises(pv.DataValidationError):
                pv.zero_coupon_price(yield_value, 10.)

    def test_interpolation_uses_applicable_pairs(self):
        for maturity, expected in ((1.5, .02), (4.5, .10), (9.5, .25)):
            with self.subTest(maturity=maturity):
                self.assertAlmostEqual(pv.interpolate_yield(self.curve, maturity), expected)
        for maturity, expected in self.curve.items():
            self.assertAlmostEqual(pv.interpolate_yield(self.curve, maturity), expected)

    def test_interpolation_rejects_extrapolation_and_missing_tenors(self):
        for maturity in (.99, 10.01, np.nan, [1., 2.]):
            with self.subTest(maturity=maturity), self.assertRaises(pv.DataValidationError):
                pv.interpolate_yield(self.curve, maturity)
        del self.curve[4]
        with self.assertRaisesRegex(pv.DataValidationError, "Missing fitted tenors"):
            pv.interpolate_yield(self.curve, 4.5)


class ValidationTests(SyntheticTestCase):
    def test_missing_required_yield_and_etf_columns(self):
        for column in pv.YIELD_COLUMNS:
            with self.subTest(column=column), self.assertRaisesRegex(pv.DataValidationError, "required columns"):
                pv.prepare_market_data(self.yields.drop(columns=column), self.etfs, self.start, self.end)
        for ticker in pv.ETF_TICKERS:
            for field in pv.ETF_PRICE_FIELDS:
                prices = {**self.etfs, ticker: self.etfs[ticker].drop(columns=field)}
                with self.subTest(ticker=ticker, field=field):
                    with self.assertRaisesRegex(pv.DataValidationError, "required columns"):
                        pv.prepare_market_data(self.yields, prices, self.start, self.end)

    def test_invalid_dates_and_numeric_index_are_rejected(self):
        for bad_date in ("2022-02-30", "09/01/2022", "bad", None, pd.NaT, 20220901):
            yields = self.yields.copy()
            yields.index = [bad_date, *self.yields.index[1:]]
            with self.subTest(date=bad_date), self.assertRaises(pv.DataValidationError):
                pv.prepare_market_data(yields, self.etfs, self.start, self.end)
        with self.assertRaises(pv.DataValidationError):
            pv.prepare_market_data(self.yields.reset_index(drop=True), self.etfs, self.start, self.end)

    def test_duplicate_dates_detected_after_session_date_normalization(self):
        for duplicate in (self.yields.index[0], self.yields.index[0] + pd.Timedelta(hours=12)):
            yields = self.yields.copy()
            yields.index = [self.yields.index[0], duplicate, *self.yields.index[2:]]
            with self.assertRaisesRegex(pv.DataValidationError, "duplicate dates"):
                pv.prepare_market_data(yields, self.etfs, self.start, self.end)
        self.etfs["SPY"] = pd.concat([self.etfs["SPY"], self.etfs["SPY"].iloc[[0]]])
        with self.assertRaisesRegex(pv.DataValidationError, "duplicate dates"):
            self.prepare()

    def test_duplicate_columns_are_rejected(self):
        self.yields = pd.concat([self.yields, self.yields.iloc[:, [0]]], axis=1)
        with self.assertRaisesRegex(pv.DataValidationError, "duplicate columns"):
            self.prepare()

    def test_bad_numeric_types_and_infinities_are_rejected(self):
        for value in ("bad", True, 1 + 2j, np.inf, -np.inf, pd.Timestamp("2022-09-01")):
            for target in ("yield", "ETF"):
                yields, etfs = synthetic_sources()
                frame = yields if target == "yield" else etfs["LQD"]
                column = pv.YIELD_COLUMNS[0] if target == "yield" else "Adj Close"
                frame[column] = frame[column].astype(object)
                frame.iloc[0, frame.columns.get_loc(column)] = value
                with self.subTest(value=value, target=target), self.assertRaises(pv.DataValidationError):
                    pv.prepare_market_data(yields, etfs, self.start, self.end)

    def test_numeric_text_is_parsed_to_floats(self):
        self.yields = self.yields.astype(str)
        self.etfs = {ticker: frame.astype(str) for ticker, frame in self.etfs.items()}
        market = self.prepare()
        self.assertTrue(all(dtype == np.dtype(float) for dtype in market.levels.dtypes))

    def test_nonpositive_close_and_adjusted_prices_are_rejected(self):
        for ticker in pv.ETF_TICKERS:
            for field in pv.ETF_PRICE_FIELDS:
                for value in (0., -1.):
                    yields, etfs = synthetic_sources()
                    etfs[ticker].loc[self.start, field] = value
                    with self.subTest(ticker=ticker, field=field, value=value):
                        with self.assertRaisesRegex(pv.DataValidationError, "strictly positive"):
                            pv.prepare_market_data(yields, etfs, self.start, self.end)

    def test_no_complete_source_and_empty_range_fail(self):
        self.etfs["LQD"].loc[:, "Adj Close"] = np.nan
        with self.assertRaisesRegex(pv.DataValidationError, "no complete"):
            self.prepare()
        for start, end in ((self.end, self.start), ("2026-08-01", "2026-08-31")):
            with self.assertRaises(pv.DataValidationError):
                pv.prepare_market_data(self.yields, self.etfs, start, end)
        with self.assertRaises(pv.DataValidationError):
            pv.prepare_market_data(self.yields.iloc[:0], self.etfs, self.start, self.end)

    def test_at_least_two_common_dates_and_exact_etf_set_required(self):
        for count in (0, 1):
            etfs = {ticker: frame.iloc[2:] for ticker, frame in self.etfs.items()}
            with self.assertRaisesRegex(pv.DataValidationError, "two common valid dates"):
                pv.prepare_market_data(self.yields.iloc[:2 + count], etfs, self.start, self.end)
        for etfs in ({"LQD": self.etfs["LQD"]}, {**self.etfs, "EXTRA": self.etfs["SPY"]}):
            with self.assertRaisesRegex(pv.DataValidationError, "exactly LQD and SPY"):
                pv.prepare_market_data(self.yields, etfs, self.start, self.end)

    def test_source_coverage_shortfalls_are_explicit(self):
        self.etfs["LQD"] = self.etfs["LQD"].iloc[1:-1]
        market = self.prepare()
        report = market.coverage[1]
        self.assertEqual(report.boundary_shortfalls, ("start", "end"))
        self.assertEqual(report.first_valid_date, self.yields.index[1])
        self.assertEqual(report.last_valid_date, self.yields.index[-2])
        self.assertEqual(report.absent_weekdays, (self.start, self.end))
        self.assertEqual(market.dropped_dates, (self.start, self.end))

    def test_inclusive_bounds_and_unsorted_sources(self):
        first, last = self.yields.index[[1, -2]]
        market = pv.prepare_market_data(self.yields.iloc[::-1], self.etfs, first, last)
        self.assertEqual(market.levels.index[0], first)
        self.assertEqual(market.levels.index[-1], last)
        self.assertEqual(len(market.levels), 4)
        self.assertEqual(market.coverage[0].boundary_shortfalls, ())


class AlignmentTests(SyntheticTestCase):
    def test_aligned_changes_units_order_and_adjusted_return_basis(self):
        market = self.prepare()
        np.testing.assert_allclose(market.levels[list(pv.YIELD_COLUMNS)], self.yields / 100)
        np.testing.assert_allclose(market.yield_changes, .0001, atol=1e-16)
        np.testing.assert_allclose(market.yield_changes_bp, 1., atol=1e-12)
        for ticker in pv.ETF_TICKERS:
            expected = np.log(self.etfs[ticker]["Adj Close"]).diff().iloc[1:]
            assert_series_equal(market.etf_log_returns[ticker], expected, check_names=False, check_freq=False)
        self.assertEqual(tuple(market.risk_factor_changes.columns), pv.YIELD_COLUMNS + pv.ETF_TICKERS)
        self.assertTrue(market.yield_changes.index.equals(market.etf_log_returns.index))
        self.assertFalse(market.risk_factor_changes.isna().any().any())

    def test_all_six_yields_and_both_price_fields_align_before_differencing(self):
        missing_yield_date, missing_price_date = self.yields.index[[1, 3]]
        self.yields.loc[missing_yield_date, "SVENY09"] = np.nan
        self.etfs["SPY"].loc[missing_price_date, "Close"] = np.nan
        self.etfs["LQD"] = self.etfs["LQD"].drop(self.yields.index[-1])
        market = self.prepare()
        expected_dates = self.yields.index[[0, 2, 4]]
        self.assertTrue(market.levels.index.equals(expected_dates))
        expected_yields = (self.yields.loc[expected_dates, list(pv.YIELD_COLUMNS)] / 100).diff().iloc[1:]
        assert_frame_equal(market.yield_changes, expected_yields, check_freq=False)
        for ticker in pv.ETF_TICKERS:
            expected = np.log(self.etfs[ticker].loc[expected_dates, "Adj Close"]).diff().iloc[1:]
            assert_series_equal(market.etf_log_returns[ticker], expected, check_names=False, check_freq=False)
        self.assertEqual(market.dropped_dates, tuple(self.yields.index[[1, 3, 5]]))
        self.assertEqual(market.coverage[0].incomplete_dates, (missing_yield_date,))
        self.assertEqual(market.coverage[2].incomplete_dates, (missing_price_date,))
        self.assertEqual(tuple(market.change_start_dates), tuple(expected_dates[:-1]))


class AllocationTests(SyntheticTestCase):
    def setUp(self):
        super().setUp()
        self.market = self.prepare()
        self.allocation = pv.allocate_portfolio(self.market.levels, self.start)

    def test_initial_quantities_reconcile_to_fixed_allocations(self):
        allocation = self.allocation
        expected = np.asarray(pv.WEIGHTS) * 1_000_000
        np.testing.assert_allclose(allocation.initial_holding_values, expected)
        np.testing.assert_allclose(
            allocation.treasury_face_quantities * pv.treasury_unit_prices(self.market.levels, self.start),
            expected[:3],
        )
        self.assertEqual(allocation.etf_shares["LQD"], 300_000 / 100)
        self.assertEqual(allocation.etf_shares["SPY"], 400_000 / 200)
        np.testing.assert_allclose(pv.holding_values(allocation, self.market.levels, self.start), expected)

    def test_etfs_use_initial_close_and_subsequent_adjusted_ratios(self):
        values = pv.holding_values(self.allocation, self.market.levels, self.end)
        self.assertAlmostEqual(values["LQD"], 300_000 * 85 / 80)
        self.assertAlmostEqual(values["SPY"], 400_000 * 159 / 150)
        changed = self.market.levels.copy()
        changed.loc[self.end, ["LQD_Close", "SPY_Close"]] = [5., 5_000.]
        assert_series_equal(values, pv.holding_values(self.allocation, changed, self.end))

    def test_adjusted_price_level_rescaling_does_not_change_holding_values(self):
        changed = self.market.levels.copy()
        changed["LQD_Adj Close"] *= 7.5
        changed["SPY_Adj Close"] *= .3
        rescaled = pv.allocate_portfolio(changed, self.start)
        assert_series_equal(rescaled.etf_shares, self.allocation.etf_shares)
        np.testing.assert_allclose(
            pv.holding_values(rescaled, changed, self.end),
            pv.holding_values(self.allocation, self.market.levels, self.end),
        )

    def test_reporting_normalization_changes_dollars_only(self):
        larger = pv.allocate_portfolio(self.market.levels, self.start, 2_000_000)
        baseline_values = pv.holding_values(self.allocation, self.market.levels, self.end)
        larger_values = pv.holding_values(larger, self.market.levels, self.end)
        np.testing.assert_allclose(larger_values, 2 * baseline_values)
        np.testing.assert_allclose(larger.treasury_face_quantities, 2 * self.allocation.treasury_face_quantities)
        np.testing.assert_allclose(larger.etf_shares, 2 * self.allocation.etf_shares)
        np.testing.assert_allclose(
            larger_values / larger.initial_holding_values - 1,
            baseline_values / self.allocation.initial_holding_values - 1,
        )
        self.assertAlmostEqual(larger_values.sum() / larger.initial_value,
                               baseline_values.sum() / self.allocation.initial_value)

    def test_calendar_aging_and_explicit_one_year_terminal_tenors(self):
        age = (self.end - self.start).days / 365
        row = self.market.levels.loc[self.end]
        curve = dict(zip(pv.TREASURY_TENORS, row[list(pv.YIELD_COLUMNS)]))
        expected = [
            self.allocation.treasury_face_quantities[holding]
            * math.exp(-pv.interpolate_yield(curve, maturity - age) * (maturity - age))
            for holding, maturity in zip(pv.TREASURY_HOLDINGS, pv.INITIAL_MATURITIES)
        ]
        values = pv.holding_values(self.allocation, self.market.levels, self.end)
        np.testing.assert_allclose(values.iloc[:3], expected)
        terminal = pv.holding_values(self.allocation, self.market.levels, self.end, elapsed_years=1.)
        expected_terminal = self.allocation.treasury_face_quantities.to_numpy() * np.exp(
            -row[["SVENY01", "SVENY04", "SVENY09"]].to_numpy() * [1., 4., 9.]
        )
        np.testing.assert_allclose(terminal.iloc[:3], expected_terminal)

    def test_explicit_allocation_dates_and_invalid_valuation_requests(self):
        later = pv.allocate_portfolio(self.market.levels, self.end)
        self.assertEqual(later.allocation_date, self.end)
        np.testing.assert_allclose(pv.holding_values(later, self.market.levels, self.end),
                                   later.initial_holding_values)
        with self.assertRaisesRegex(pv.DataValidationError, "precede"):
            pv.holding_values(later, self.market.levels, self.start)
        with self.assertRaisesRegex(pv.DataValidationError, "No common valid"):
            pv.allocate_portfolio(self.market.levels, "2022-09-03")
        for age in (-1., np.inf, 1.01):
            with self.subTest(age=age), self.assertRaises(pv.DataValidationError):
                pv.holding_values(self.allocation, self.market.levels, self.end, elapsed_years=age)
        for amount in (0., -1., np.nan, True, [1_000_000.]):
            with self.subTest(amount=amount), self.assertRaises(pv.DataValidationError):
                pv.allocate_portfolio(self.market.levels, self.start, amount)


class RetrievalAndExecutionTests(SyntheticTestCase):
    def fed_text(self):
        return "Federal Reserve fitted yields\nA synthetic CSV preamble\n" + self.yields.to_csv()

    def test_fed_preamble_missing_marker_and_required_header(self):
        parsed = pv.parse_fed_csv(self.fed_text())
        self.assertEqual(tuple(parsed.columns), pv.YIELD_COLUMNS)
        text = self.fed_text().replace("2.1,", "ND,", 1)
        self.assertTrue(pd.isna(pv.parse_fed_csv(text).iloc[0, 0]))
        for text in ("<html>blocked</html>", "Date,SVENY02\n2022-09-01,3.0"):
            with self.assertRaises(pv.DataValidationError):
                pv.parse_fed_csv(text)
        with self.assertRaisesRegex(pv.DataValidationError, "duplicate CSV columns"):
            pv.parse_fed_csv("Date,SVENY01,SVENY01\n2022-09-01,1,1")

    @patch("portfolio_var.requests.get")
    def test_fed_requests_retrieval_preserves_source_percentage_units(self, get):
        response = Mock(text=self.fed_text())
        get.return_value = response
        actual = pv.fetch_treasury_yields(self.start, self.end)
        get.assert_called_once_with(pv.FED_YIELD_URL, timeout=pv.REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status.assert_called_once_with()
        assert_frame_equal(actual, self.yields, check_freq=False)

    @patch("portfolio_var.requests.get", side_effect=pv.requests.ConnectionError("offline"))
    def test_fed_network_failures_are_explicit(self, get):
        with self.assertRaises(pv.DataRetrievalError):
            pv.fetch_treasury_yields()

    @patch("portfolio_var.yf.set_tz_cache_location")
    @patch("portfolio_var.yf.download")
    def test_etf_download_flags_and_inclusive_end(self, download, cache):
        download.side_effect = lambda ticker, **kwargs: self.etfs[ticker]
        result = pv.fetch_etf_prices()
        cache.assert_called_once_with(str(pv.YFINANCE_CACHE_DIRECTORY))
        self.assertEqual(tuple(result), pv.ETF_TICKERS)
        self.assertEqual(download.call_count, 2)
        for call, ticker in zip(download.call_args_list, pv.ETF_TICKERS):
            self.assertEqual(call.args, (ticker,))
            self.assertEqual(call.kwargs["start"], "2022-09-01")
            self.assertEqual(call.kwargs["end"], "2026-09-01")
            for flag in ("auto_adjust", "back_adjust", "repair", "multi_level_index"):
                self.assertIs(call.kwargs[flag], False)
            self.assertIs(call.kwargs["keepna"], True)
            self.assertEqual(tuple(result[ticker].columns), ("Close", "Adj Close"))

    @patch("portfolio_var.yf.set_tz_cache_location")
    @patch("portfolio_var.yf.download")
    def test_etf_empty_and_failed_downloads_fail_without_fallback(self, download, cache):
        download.return_value = pd.DataFrame()
        with self.assertRaises(pv.DataValidationError):
            pv.fetch_etf_prices()
        download.side_effect = RuntimeError("offline")
        with self.assertRaises(pv.DataRetrievalError):
            pv.fetch_etf_prices()

    @patch("portfolio_var.fetch_etf_prices")
    @patch("portfolio_var.fetch_treasury_yields")
    def test_retrieval_pipeline_and_executable_report(self, yields, etfs):
        yields.return_value = self.yields
        etfs.return_value = self.etfs
        market = pv.retrieve_market_data(self.start, self.end)
        self.assertEqual(len(market.risk_factor_changes), 5)
        yields.assert_called_once_with(self.start, self.end)
        etfs.assert_called_once_with(self.start, self.end)
        output = io.StringIO()
        with patch("portfolio_var.retrieve_market_data", return_value=market), redirect_stdout(output):
            self.assertEqual(pv.main([]), 0)
        self.assertIn("1,000,000.00", output.getvalue())
        self.assertIn("invariant to this scale", output.getvalue())
        self.assertIn("basis points", output.getvalue())


if __name__ == "__main__":
    unittest.main()
