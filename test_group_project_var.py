"""Synthetic, offline tests for group_project_var. Run with: .venv/bin/python -m unittest -v"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import group_project_var as gp


def synthetic_levels(days=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame({
        "SPX": 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, days))),
        "DGS10": 0.015 + np.cumsum(rng.normal(0, 0.0005, days)),
        "BAA10Y": 0.02 + np.cumsum(rng.normal(0, 0.0003, days)),
    }, index=dates.rename("Date"))


class PricingTests(unittest.TestCase):
    def test_par_bond_prices_at_face(self):
        self.assertAlmostEqual(gp.bond_price(0.04, 0.04), 100.0, places=10)
        self.assertAlmostEqual(gp.bond_price(0.05, 0.05), 100.0, places=10)

    def test_premium_and_discount(self):
        self.assertGreater(gp.bond_price(0.01, 0.04), 100.0)
        self.assertLess(gp.bond_price(0.06, 0.04), 100.0)

    def test_vectorized_matches_scalar(self):
        ys = np.array([0.01, 0.02, 0.03])
        np.testing.assert_allclose(gp.bond_price(ys, 0.05), [gp.bond_price(y, 0.05) for y in ys])

    def test_modified_duration_of_par_bond(self):
        # Closed form for a par bond: (1 - (1 + y/2)^-20) / y
        y = 0.04
        self.assertAlmostEqual(gp.modified_duration(y, y), (1 - (1 + y / 2) ** -20) / y, places=6)


class DataTests(unittest.TestCase):
    def test_parse_fred_csv_handles_missing(self):
        text = "observation_date,DGS10\n2020-01-01,\n2020-01-02,1.88\n2020-01-03,.\n"
        series = gp.parse_fred_csv(text, "DGS10")
        self.assertEqual(series.notna().sum(), 1)
        self.assertEqual(series.loc["2020-01-02"], 1.88)

    def test_parse_fred_csv_requires_column(self):
        with self.assertRaises(gp.DataError):
            gp.parse_fred_csv("observation_date,X\n2020-01-02,1\n", "DGS10")

    def test_align_drops_incomplete_dates_and_converts_percent(self):
        dates = pd.to_datetime(["2019-12-31", "2020-06-01", "2020-12-31"])
        raw = {"SP500": pd.Series([3000.0, np.nan, 3700.0], dates),
               "DGS10": pd.Series([1.9, 0.7, 0.93], dates),
               "BAA10Y": pd.Series([2.0, 3.0, 2.18], dates)}
        levels = gp.align_levels(raw)
        self.assertEqual(list(levels.index), [dates[0], dates[2]])
        self.assertAlmostEqual(levels.loc[dates[2], "DGS10"], 0.0093)
        self.assertEqual(list(levels.columns), ["SPX", "DGS10", "BAA10Y"])

    def test_align_requires_endpoints(self):
        dates = pd.to_datetime(["2020-01-02", "2020-12-31"])
        raw = {name: pd.Series([1.0, 1.0], dates) for name in gp.FRED_SERIES}
        with self.assertRaises(gp.DataError):
            gp.align_levels(raw)

    def test_horizon_changes_are_overlapping_sums(self):
        changes = gp.daily_changes(synthetic_levels())
        five = gp.horizon_changes(changes, 5)
        self.assertEqual(len(five), len(changes) - 4)
        np.testing.assert_allclose(five.iloc[0], changes.iloc[:5].sum())


class RiskTests(unittest.TestCase):
    def test_var_es_on_known_losses(self):
        pnl = -np.arange(1, 101, dtype=float)  # losses 1..100
        var, es = gp.var_es(pnl, 0.95)
        self.assertAlmostEqual(var, np.quantile(np.arange(1, 101), 0.95))
        self.assertAlmostEqual(es, np.arange(1, 101)[np.arange(1, 101) >= var].mean())
        self.assertGreaterEqual(es, var)

    def test_zero_shock_gives_zero_pnl(self):
        levels = synthetic_levels()
        valuation = levels.iloc[-1]
        positions = gp.build_positions(valuation)
        zero = pd.DataFrame({c: [0.0] for c in ("SPX", "DGS10", "BAA10Y")})
        np.testing.assert_allclose(gp.scenario_pnl(positions, valuation, zero).to_numpy(), 0.0, atol=1e-9)

    def test_positions_match_market_values(self):
        positions = gp.build_positions(synthetic_levels().iloc[-1])
        for holding, p in positions.items():
            self.assertAlmostEqual(p.quantity * p.unit_price, gp.POSITION_VALUES[holding])

    def test_rate_and_spread_shocks_hit_the_right_bonds(self):
        valuation = synthetic_levels().iloc[-1]
        positions = gp.build_positions(valuation)
        shocks = pd.DataFrame({"SPX": [0.0, 0.0], "DGS10": [0.001, 0.0], "BAA10Y": [0.0, 0.001]})
        pnl = gp.scenario_pnl(positions, valuation, shocks)
        self.assertLess(pnl.loc[0, "Govt_10Y"], 0)
        self.assertLess(pnl.loc[0, "Corp_A_10Y"], 0)
        self.assertAlmostEqual(pnl.loc[1, "Govt_10Y"], 0, places=9)
        self.assertLess(pnl.loc[1, "Corp_A_10Y"], 0)
        np.testing.assert_allclose(pnl["Portfolio"], pnl[list(gp.HOLDINGS)].sum(axis=1))

    def test_run_analysis_is_reproducible_and_complete(self):
        levels = synthetic_levels()
        first = gp.run_analysis(levels, paths=2_000)
        second = gp.run_analysis(levels, paths=2_000)
        pd.testing.assert_frame_equal(first.risk, second.risk)
        expected_rows = 2 * len(gp.HORIZONS) * len(gp.CONFIDENCE_LEVELS) * (len(gp.RISK_IDS) + 1)
        self.assertEqual(len(first.risk), expected_rows)
        measures = first.risk[first.risk["risk_id"] != "Diversification"]
        self.assertTrue((measures["es_usd"] >= measures["var_usd"] - 1e-9).all())

    def test_write_outputs(self):
        analysis = gp.run_analysis(synthetic_levels(), paths=500)
        with tempfile.TemporaryDirectory() as tmp:
            paths = gp.write_outputs(analysis, Path(tmp))
            self.assertEqual({p.name for p in paths}, set(gp.OUTPUT_FILENAMES))
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == "__main__":
    unittest.main()
