"""E04 S13 overfitting and transfer tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e04.evolutionary_search import EvolutionGenome, heuristic_parameter_vector, params_from_vector, policy_id_for
from src.e04.overfitting_transfer import (
    S13_GENERALIZATION_AXES,
    assert_s13_design,
    build_s13_transfer_schedule,
    default_s13_transfer_configs,
    infer_s13_outcome,
    run_s13_transfer_matrix,
    s13_config_hash,
    s13_config_records,
    s13_memory_specs,
    summarize_transfer_gaps,
    summarize_transfer_results,
    transfer_gaps,
)


class OverfittingTransferTests(unittest.TestCase):
    def _genome(self) -> EvolutionGenome:
        params = params_from_vector(heuristic_parameter_vector())
        return EvolutionGenome(
            policy_id=policy_id_for(0, 0, params, ()),
            generation=0,
            population_index=0,
            parent_ids=(),
            mutation_seed=1,
            mutation_scale=0.0,
            parameters=params,
            lineage_note="unit_test_heuristic",
        )

    def test_default_configs_predefine_heldouts_and_design(self) -> None:
        configs = default_s13_transfer_configs(train_seeds=(18001,), heldout_seeds=(52001,), max_events=80, stress_max_events=96)
        specs = s13_memory_specs()
        audit = assert_s13_design(configs, specs)
        records = s13_config_records(configs)
        digest = s13_config_hash(configs)

        self.assertTrue(audit["success"])
        self.assertEqual(set(audit["generalizationAxisCounts"]), set(S13_GENERALIZATION_AXES))
        self.assertEqual(tuple(spec.name for spec in specs), ("no_memory", "neighbor_memory"))
        self.assertTrue(all(record["predefined_before_evaluation"] for record in records))
        self.assertEqual(digest, s13_config_hash(configs))

    def test_s13_schedules_cover_dense_and_frozen_variants(self) -> None:
        configs = default_s13_transfer_configs(train_seeds=(18001,), heldout_seeds=(52001,), max_events=120, stress_max_events=180)
        by_axis = {config.generalization_axis: config for config in configs if config.split == "heldout"}
        dense = build_s13_transfer_schedule(by_axis["unseen_perturbation_rate"])
        frozen = build_s13_transfer_schedule(by_axis["unseen_frozen_placement"])
        combined = build_s13_transfer_schedule(by_axis["combined_stress"])

        self.assertGreaterEqual(len(dense), 8)
        self.assertGreaterEqual(sum(1 for item in dense if item.perturbation_type == "delete_insert"), 2)
        self.assertGreaterEqual(sum(1 for item in frozen if item.perturbation_type == "freeze"), 4)
        self.assertTrue(all(item.safe_representation.startswith("s13_predefined_") for item in combined))

    def test_small_s13_matrix_has_no_oracle_hits_and_gaps(self) -> None:
        configs = default_s13_transfer_configs(train_seeds=(18001,), heldout_seeds=(52001,), max_events=32, stress_max_events=36)
        configs = tuple(config for config in configs if config.generalization_axis in {"s08_train_reference", "unseen_perturbation_rate"})
        specs = s13_memory_specs()
        frozen_hash = s13_config_hash(configs)
        rows = run_s13_transfer_matrix(
            genomes=(self._genome(),),
            configs=configs,
            specs=specs,
            frozen_config_hash=frozen_hash,
        )
        df = pd.DataFrame(rows)
        summary = summarize_transfer_results(df)
        gap_df = transfer_gaps(df)
        gap_summary = summarize_transfer_gaps(gap_df)
        outcome = infer_s13_outcome(gap_summary)

        self.assertEqual(len(rows), len(configs) * len(specs))
        self.assertEqual(set(df["uses_global_oracle"]), {False})
        self.assertEqual(set(df["predefined_before_evaluation"]), {True})
        self.assertEqual(set(df["selected_policy_was_retuned"]), {False})
        self.assertFalse(summary.empty)
        self.assertFalse(gap_df.empty)
        self.assertIn(outcome["outcome"], {"supportive", "null", "constraining/contradictory"})
        for value in df["feature_audit_summary_json"]:
            audit = json.loads(value)
            self.assertFalse(audit["usesGlobalOracle"])
            self.assertEqual(audit["excludedSignalHits"], {})


if __name__ == "__main__":
    unittest.main()
