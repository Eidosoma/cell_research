import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import e02_s14_strong_statistics as s14


class StrongStatisticsTests(unittest.TestCase):
    def test_empirical_high_tail_uses_plus_one_correction(self):
        nulls = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(s14.empirical_p_value(0.35, nulls, "high"), 2 / 5)
        self.assertAlmostEqual(s14.empirical_p_value(0.05, nulls, "low"), 1 / 5)

    def test_paired_signflip_all_zero_is_one(self):
        p_value, reps, mode = s14.paired_signflip_p_value(np.array([0.0, 0.0, 0.0]), "two-sided", seed=1)
        self.assertEqual(p_value, 1.0)
        self.assertEqual(reps, 0)
        self.assertEqual(mode, "all_zero_differences")

    def test_paired_signflip_detects_consistent_positive_shift(self):
        diffs = np.array([1.0, 2.0, 3.0, 4.0])
        p_value, reps, mode = s14.paired_signflip_p_value(diffs, "high", seed=1)
        self.assertLessEqual(p_value, 0.125)
        self.assertEqual(reps, 16)
        self.assertEqual(mode, "exact_signflip")

    def test_bh_fdr_is_monotone_within_family(self):
        df = pd.DataFrame(
            {
                "fdrFamilyId": ["a", "a", "a", "b"],
                "pValue": [0.001, 0.02, 0.5, 0.04],
            }
        )
        out = s14.apply_bh_fdr(df)
        a = out[out["fdrFamilyId"] == "a"].sort_values("pValue")
        self.assertTrue(a["qValue"].is_monotonic_increasing)
        self.assertTrue(bool(out.loc[0, "significantFdr05"]))
        self.assertTrue(bool(out.loc[3, "significantFdr05"]))

    def test_validate_sources_reports_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            steps = root / "research_steps" / "S14"
            results.mkdir(parents=True)
            steps.mkdir(parents=True)
            table_name = "e02_scheduler_regimes.parquet"
            df = pd.DataFrame({"schedulerRegime": ["s01_random_sequential"]})
            df.to_parquet(results / table_name, index=False)
            tables = {table_name: df}
            validation = s14.validate_sources(results, tables, steps)
            row = validation[validation["artifactName"] == table_name].iloc[0]
            self.assertFalse(bool(row["validationPassed"]))
            self.assertIn("mixtureId", row["missingRequiredColumns"])


if __name__ == "__main__":
    unittest.main()
