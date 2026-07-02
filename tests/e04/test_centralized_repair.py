"""E04 S14 centralized repair baseline tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.e04.centralized_repair import (
    CENTRALIZED_COMPARATOR_GROUP,
    LOCAL_COMPARATOR_GROUP,
    load_s13_configs,
    paired_comparison_deltas,
    s13_config_from_record,
    schedule_from_record,
    select_best_local_s13_rows,
    simulate_centralized_repair,
    validate_s14_outputs,
)
from src.e04.overfitting_transfer import (
    default_s13_transfer_configs,
    s13_config_records,
)


class CentralizedRepairTests(unittest.TestCase):
    def _small_record(self) -> dict[str, object]:
        config = default_s13_transfer_configs(
            train_seeds=(18001,),
            heldout_seeds=(),
            max_events=28,
            stress_max_events=28,
        )[1]
        record = s13_config_records((config,))[0]
        record["schedule_json"] = json.dumps(
            [
                {
                    "event_step": 3,
                    "perturbation_type": "freeze",
                    "params": {"position": 2},
                    "safe_representation": "unit_test_freeze",
                },
                {
                    "event_step": 5,
                    "perturbation_type": "swap",
                    "params": {"left_position": 1, "right_position": 4},
                    "safe_representation": "unit_test_swap",
                },
            ],
            sort_keys=True,
        )
        return record

    def test_centralized_row_is_global_and_excluded_from_local_claims(self) -> None:
        record = self._small_record()
        row = simulate_centralized_repair(
            s13_config_from_record(record),
            schedule_from_record(record),
            s13_config_id=str(record["config_id"]),
        )

        self.assertEqual(row["comparison_group"], CENTRALIZED_COMPARATOR_GROUP)
        self.assertTrue(row["centralized_baseline"])
        self.assertTrue(row["uses_global_oracle"])
        self.assertTrue(row["global_state_access"])
        self.assertFalse(row["eligible_for_local_only_claims"])
        self.assertGreaterEqual(row["global_repair_action_count"], 1)
        audit = json.loads(row["global_access_audit_json"])
        self.assertIn("target_sorted_order", audit["accessed_fields"])

    def test_best_local_selection_and_validation_keep_claim_groups_separate(self) -> None:
        local_rows = pd.DataFrame(
            [
                {
                    "s13_config_id": "cfg-a",
                    "split": "heldout",
                    "benchmark_family": "overfitting_transfer",
                    "generalization_axis": "combined_stress",
                    "scenario_name": "s13_unit",
                    "policy_id": "p-low",
                    "memory_ablation": "neighbor_memory",
                    "eligible_for_local_only_claims": True,
                    "uses_global_oracle": False,
                    "centralized_baseline": False,
                    "fitness_score": 0.4,
                    "schedule_sha256": "sha",
                    "freeze_perturbation_count": 1,
                    "unfreeze_count": 1,
                    "mean_recovery_events": 2.0,
                    "max_events": 10,
                    "unrecovered_perturbations": 0,
                    "mean_sortedness_percent": 80.0,
                    "min_sortedness_percent": 60.0,
                    "energy_total": 3.0,
                    "final_sortedness_percent": 90.0,
                    "time_in_target_fraction": 0.6,
                },
                {
                    "s13_config_id": "cfg-a",
                    "split": "heldout",
                    "benchmark_family": "overfitting_transfer",
                    "generalization_axis": "combined_stress",
                    "scenario_name": "s13_unit",
                    "policy_id": "p-high",
                    "memory_ablation": "neighbor_memory",
                    "eligible_for_local_only_claims": True,
                    "uses_global_oracle": False,
                    "centralized_baseline": False,
                    "fitness_score": 0.7,
                    "schedule_sha256": "sha",
                    "freeze_perturbation_count": 1,
                    "unfreeze_count": 1,
                    "mean_recovery_events": 1.0,
                    "max_events": 10,
                    "unrecovered_perturbations": 0,
                    "mean_sortedness_percent": 85.0,
                    "min_sortedness_percent": 70.0,
                    "energy_total": 2.0,
                    "final_sortedness_percent": 95.0,
                    "time_in_target_fraction": 0.7,
                },
            ]
        )
        selected = select_best_local_s13_rows(local_rows)
        self.assertEqual(selected.iloc[0]["policy_id"], "p-high")
        self.assertEqual(selected.iloc[0]["comparison_group"], LOCAL_COMPARATOR_GROUP)
        self.assertTrue(selected.iloc[0]["eligible_for_local_only_claims"])

        central = selected.iloc[[0]].copy()
        central["policy_id"] = "central"
        central["comparison_group"] = CENTRALIZED_COMPARATOR_GROUP
        central["centralized_baseline"] = True
        central["uses_global_oracle"] = True
        central["global_state_access"] = True
        central["eligible_for_local_only_claims"] = False
        central["baseline_type"] = "centralized_global_oracle_baseline"
        central["fitness_score"] = 0.8
        df = pd.concat([selected, central], ignore_index=True)
        paired = paired_comparison_deltas(df)
        checks = validate_s14_outputs(df, paired)

        self.assertEqual(len(paired), 1)
        self.assertTrue(checks["all_passed"])

    def test_load_s13_configs_validates_json_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = Path(tmpdir) / "valid.json"
            empty_path = Path(tmpdir) / "empty.json"
            valid_path.write_text(json.dumps({"configs": [{"config_id": "cfg"}]}), encoding="utf-8")
            empty_path.write_text(json.dumps({"configs": []}), encoding="utf-8")

            self.assertEqual(load_s13_configs(valid_path), [{"config_id": "cfg"}])
            with self.assertRaises(ValueError):
                load_s13_configs(empty_path)


if __name__ == "__main__":
    unittest.main()
