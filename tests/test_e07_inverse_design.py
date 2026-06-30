from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from morphospace import parse_rule_program
from platonic_space.inverse_design import (
    PROFILE_IDS,
    DesignRunPanel,
    build_inverse_design_target_profiles,
    generate_inverse_design_pool,
    holdout_panel,
    run_design_panel,
    score_candidate_pool,
    select_designed_policies,
    summarize_profile_validation,
    validation_checks,
    write_designed_policy_files,
)


class TestE07InverseDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        target_path = Path("/artifacts/research_steps/S11/target_performance_summary.parquet")
        direct_path = Path("/artifacts/research_steps/S11/direct_validation_results.parquet")
        if not target_path.exists() or not direct_path.exists():
            raise unittest.SkipTest("S11 target-performance and direct-validation artifacts are required for S12 inverse-design tests")
        cls.s11_targets = pd.read_parquet(target_path)
        cls.s11_direct = pd.read_parquet(direct_path)

    def test_target_profiles_use_supported_s11_families(self) -> None:
        profiles = build_inverse_design_target_profiles(self.s11_targets, self.s11_direct)

        self.assertEqual(set(profiles["profileId"]), set(PROFILE_IDS))
        primary_support = "\n".join(profiles["s11SupportFamiliesJson"].astype(str))
        self.assertNotIn("high_error_reduction", primary_support)
        self.assertNotIn("high_compatibility", primary_support)
        self.assertNotIn("high_repair_success", primary_support)
        self.assertIn("low_swap_count", "\n".join(profiles["deferredOrSecondaryFamiliesJson"].astype(str)))

    def test_generated_pool_roundtrips_as_dsl(self) -> None:
        pool = generate_inverse_design_pool(reference_size=64)
        valid = pool[pool["poolStatus"].eq("valid")]

        self.assertGreaterEqual(len(valid), 20)
        self.assertTrue(valid["policyId"].is_unique)
        for payload in valid["dslProgramJson"].head(10):
            program = parse_rule_program(payload)
            self.assertEqual(parse_rule_program(program.to_json()).to_dict(), program.to_dict())

    def test_mini_search_and_holdout_validation_are_direct(self) -> None:
        profiles = build_inverse_design_target_profiles(self.s11_targets, self.s11_direct)
        pool = generate_inverse_design_pool(reference_size=64)
        valid_pool = pool[pool["poolStatus"].eq("valid")].head(8).copy()
        train_panel = DesignRunPanel(tasks=holdout_panel().tasks[:3], seed_count=1, stage="unit_training")
        _train_runs, train_vectors, _train_traces = run_design_panel(valid_pool, train_panel, include_traces=False)
        scored, _summary = score_candidate_pool(valid_pool, train_vectors, profiles)
        designs = select_designed_policies(scored, per_profile=1)

        mini_holdout = DesignRunPanel(tasks=holdout_panel().tasks, seed_count=1, stage="unit_holdout")
        holdout_runs, holdout_vectors, _traces = run_design_panel(designs, mini_holdout, include_traces=False)
        design_validation, profile_summary = summarize_profile_validation(designs, holdout_vectors, profiles)
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_files = write_designed_policy_files(designs, Path(tmpdir))
            checks = validation_checks(profiles, valid_pool, designs, holdout_runs, design_validation, policy_files)

        hard_failures = checks[checks["severity"].eq("error") & ~checks["success"]]
        self.assertEqual(len(designs), len(PROFILE_IDS))
        self.assertFalse(holdout_runs.empty)
        self.assertFalse(design_validation.empty)
        self.assertFalse(profile_summary.empty)
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
