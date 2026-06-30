from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from platonic_space.counterfactual_tests import (
    CLASS_STRATIFICATION_NOTE,
    attach_universality_classes,
    error_limit_table,
    replay_records_for_candidates,
    score_heldout_predictions,
    select_counterfactual_candidates,
    validate_candidates,
    validation_checks,
)


class TestE07CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            Path("/artifacts/research_steps/S05/modeling_frame.parquet"),
            Path("/artifacts/research_steps/S05/evaluation_metrics.parquet"),
            Path("/artifacts/research_steps/S05/error_limits.parquet"),
            Path("/artifacts/research_steps/S10/universality_classes.parquet"),
            Path("/artifacts/research_steps/S04/unified_behavior_corpus.parquet"),
        )
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("S04/S05/S10 artifacts are required for S11 counterfactual tests")
        cls.frame = pd.read_parquet(required[0])
        cls.metrics = pd.read_parquet(required[1])
        cls.error_limits = error_limit_table(pd.read_parquet(required[2]))
        cls.classes = pd.read_parquet(required[3])
        cls.s04 = pd.read_parquet(required[4])

    def test_split_predictions_are_excluded_from_training_groups(self) -> None:
        predictions, diagnostics = score_heldout_predictions(
            self.frame,
            self.metrics,
            target_columns=["aggregation"],
            split_names=["heldout_policy"],
            min_train_rows=50,
            min_test_rows=20,
        )

        self.assertFalse(predictions.empty)
        self.assertFalse(diagnostics.empty)
        self.assertEqual(set(predictions["partition"]), {"test"})
        self.assertFalse(predictions["trainContainsExcludedGroup"].any())
        self.assertGreater(int(diagnostics.loc[diagnostics["status"].eq("evaluated"), "testRows"].sum()), 100)

    def test_candidate_selection_uses_classes_only_as_strata(self) -> None:
        predictions, _diagnostics = score_heldout_predictions(
            self.frame,
            self.metrics,
            target_columns=["aggregation"],
            split_names=["heldout_policy"],
            min_train_rows=50,
            min_test_rows=20,
        )
        predictions = attach_universality_classes(predictions, self.classes)
        candidates, selection = select_counterfactual_candidates(
            predictions,
            [
                {
                    "counterfactualFamily": "high_aggregation",
                    "target": "aggregation",
                    "direction": "max",
                    "allowedSplits": ("heldout_policy",),
                    "competenceAxis": "aggregation",
                }
            ],
            max_candidates_per_spec=8,
            max_candidates_per_class=1,
        )

        self.assertFalse(candidates.empty)
        self.assertFalse(selection.empty)
        self.assertTrue(candidates["candidateId"].is_unique)
        self.assertEqual(set(candidates["classUse"]), {"stratification_only"})
        self.assertFalse(candidates["classCausalInvariantAssumption"].any())
        self.assertIn("stratification", CLASS_STRATIFICATION_NOTE)

    def test_replay_validation_has_source_table_verification_and_intervals(self) -> None:
        predictions, diagnostics = score_heldout_predictions(
            self.frame,
            self.metrics,
            target_columns=["aggregation"],
            split_names=["heldout_policy"],
            min_train_rows=50,
            min_test_rows=20,
        )
        predictions = attach_universality_classes(predictions, self.classes)
        candidates, _selection = select_counterfactual_candidates(
            predictions,
            [
                {
                    "counterfactualFamily": "high_aggregation",
                    "target": "aggregation",
                    "direction": "max",
                    "allowedSplits": ("heldout_policy",),
                    "competenceAxis": "aggregation",
                }
            ],
            max_candidates_per_spec=4,
            max_candidates_per_class=1,
        )
        replay_records = replay_records_for_candidates(candidates, self.s04)
        validation = validate_candidates(candidates, replay_records, self.error_limits)
        checks = validation_checks(predictions, candidates, validation, replay_records, diagnostics, min_evaluated_models=1)
        hard_failures = checks[(checks["severity"].eq("error")) & (~checks["success"])]

        self.assertFalse(replay_records.empty)
        self.assertFalse(validation.empty)
        self.assertTrue((validation["sourceTableVerifiedFraction"] > 0).all())
        self.assertTrue(validation["s05AbsErrorP95"].notna().all())
        self.assertTrue(hard_failures.empty, hard_failures.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
