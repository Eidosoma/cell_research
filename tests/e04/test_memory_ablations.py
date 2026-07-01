"""E04 S09 memory-ablation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.e04.evolutionary_search import heuristic_parameter_vector, params_from_vector, policy_id_for
from src.e04.memory_ablations import (
    REQUIRED_MEMORY_ABLATIONS,
    S09EvaluationConfig,
    apply_memory_ablation,
    assert_matched_seed_design,
    default_memory_ablation_specs,
    default_s09_eval_configs,
    load_selected_s08_policies,
    run_memory_ablation_matrix,
)
from src.e04.no_oracle_protocol import LOCAL_ONLY_PROTOCOL_ID, LocalTrainingObservation


class MemoryAblationTests(unittest.TestCase):
    def _projected_view(self) -> LocalTrainingObservation:
        return LocalTrainingObservation(
            protocol_id=LOCAL_ONLY_PROTOCOL_ID,
            behavior="bubble",
            actor_value=5,
            actor_status="ACTIVE",
            reverse_direction=False,
            left_present=True,
            left_value=7,
            left_status="ACTIVE",
            right_present=True,
            right_value=3,
            right_status="FREEZE",
            at_left_boundary=False,
            at_right_boundary=False,
            last_move_success=False,
            last_action_type="blocked_swap_attempt",
            time_since_movement=99,
            local_frustration=99,
            failed_swap_count=99,
            neighbor_identity_count=99,
            signal_blocked=0.75,
            signal_frustrated=0.5,
            signal_scope="local_window_plus_explicit_allowed_diffusive_fields",
        )

    def test_specs_cover_required_memory_ladder_and_limits(self) -> None:
        specs = default_memory_ablation_specs()

        self.assertEqual(tuple(spec.name for spec in specs), REQUIRED_MEMORY_ABLATIONS)
        self.assertEqual(specs[0].capacity_limit()["allowedSignalFieldsVisible"], [])
        self.assertFalse(specs[0].memory_config.enabled)
        self.assertTrue(specs[1].expose_last_move_success)
        self.assertEqual(specs[1].failed_swap_capacity, 0)
        self.assertEqual(specs[2].failed_swap_capacity, 2)
        self.assertEqual(specs[3].neighbor_capacity, 2)
        self.assertEqual(set(specs[4].capacity_limit()["allowedSignalFieldsVisible"]), {"blocked", "frustrated"})

    def test_feature_masking_enforces_policy_visible_capacity(self) -> None:
        specs = {spec.name: spec for spec in default_memory_ablation_specs()}
        projected = self._projected_view()

        no_memory = apply_memory_ablation(projected, specs["no_memory"])
        self.assertIsNone(no_memory.last_move_success)
        self.assertIsNone(no_memory.last_action_type)
        self.assertEqual(no_memory.time_since_movement, 0)
        self.assertEqual(no_memory.local_frustration, 0)
        self.assertEqual(no_memory.failed_swap_count, 0)
        self.assertEqual(no_memory.neighbor_identity_count, 0)
        self.assertEqual(no_memory.signal_scope, "none")

        one_bit = apply_memory_ablation(projected, specs["one_bit_memory"])
        self.assertFalse(one_bit.last_move_success)
        self.assertIsNone(one_bit.last_action_type)
        self.assertEqual(one_bit.failed_swap_count, 0)
        self.assertEqual(one_bit.neighbor_identity_count, 0)

        counter = apply_memory_ablation(projected, specs["bounded_counter_memory"])
        self.assertEqual(counter.time_since_movement, 7)
        self.assertEqual(counter.local_frustration, 3)
        self.assertEqual(counter.failed_swap_count, 2)
        self.assertEqual(counter.neighbor_identity_count, 0)

        neighbor = apply_memory_ablation(projected, specs["neighbor_memory"])
        self.assertEqual(neighbor.neighbor_identity_count, 2)

        signal_field = apply_memory_ablation(projected, specs["signal_field_memory"])
        self.assertIsNone(signal_field.last_move_success)
        self.assertEqual(signal_field.signal_blocked, 0.75)
        self.assertEqual(signal_field.signal_frustrated, 0.5)

    def test_selected_policy_loader_validates_parameter_names(self) -> None:
        params = params_from_vector(heuristic_parameter_vector())
        record = {
            "policy_id": policy_id_for(0, 0, params, ()),
            "generation": 0,
            "population_index": 0,
            "parent_ids": [],
            "mutation_seed": 1,
            "mutation_scale": 0.0,
            "parameters": params,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policies.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = load_selected_s08_policies(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].parameters, params)

    def test_default_design_has_repair_frozen_and_homeostatic_families(self) -> None:
        configs = default_s09_eval_configs(max_events=20, heldout_max_events=24, seeds=(19001,))
        audit = assert_matched_seed_design(configs)

        self.assertTrue(audit["success"])
        self.assertEqual(audit["benchmarkFamilyCounts"]["repair"], 1)
        self.assertEqual(audit["benchmarkFamilyCounts"]["frozen_cell"], 1)
        self.assertEqual(audit["benchmarkFamilyCounts"]["homeostatic"], 3)

    def test_small_matrix_has_matched_seeds_and_no_oracle_hits(self) -> None:
        params = params_from_vector(heuristic_parameter_vector())
        record = {
            "policy_id": policy_id_for(0, 0, params, ()),
            "generation": 0,
            "population_index": 0,
            "parent_ids": (),
            "mutation_seed": 1,
            "mutation_scale": 0.0,
            "parameters": params,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policies.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            genomes = load_selected_s08_policies(path)
        specs = default_memory_ablation_specs()[:2]
        configs = (
            S09EvaluationConfig(
                benchmark_family="repair",
                values=(1, 2, 3, 4),
                task_name="frozen_damage",
                reliability_mode="no_fatigue_control",
                activation_seed=41,
                policy_seed=42,
                schedule_seed=43,
                max_events=12,
            ),
        )

        rows = run_memory_ablation_matrix(genomes=genomes, configs=configs, specs=specs)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["memory_ablation"] for row in rows}, {"no_memory", "one_bit_memory"})
        self.assertEqual(len({row["schedule_sha256"] for row in rows}), 1)
        self.assertTrue(all(not row["uses_global_oracle"] for row in rows))
        self.assertTrue(all(row["protocol_id"].endswith("e04_s09_memory_ablation") for row in rows))


if __name__ == "__main__":
    unittest.main()
