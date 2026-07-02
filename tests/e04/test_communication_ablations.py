"""E04 S10 communication-ablation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.e04.communication_ablations import (
    REQUIRED_COMMUNICATION_ABLATIONS,
    apply_communication_ablation,
    default_communication_ablation_specs,
    default_s10_eval_configs,
    neighbor_memory_spec,
    run_communication_ablation_matrix,
)
from src.e04.evolutionary_search import heuristic_parameter_vector, params_from_vector, policy_id_for
from src.e04.memory_ablations import S09EvaluationConfig, assert_matched_seed_design, load_selected_s08_policies
from src.e04.no_oracle_protocol import LOCAL_ONLY_PROTOCOL_ID, LocalTrainingObservation


class CommunicationAblationTests(unittest.TestCase):
    def _projected_view(self) -> LocalTrainingObservation:
        return LocalTrainingObservation(
            protocol_id=LOCAL_ONLY_PROTOCOL_ID,
            behavior="bubble",
            actor_value=4,
            actor_status="ACTIVE",
            reverse_direction=False,
            left_present=True,
            left_value=5,
            left_status="ACTIVE",
            right_present=True,
            right_value=2,
            right_status="FREEZE",
            at_left_boundary=False,
            at_right_boundary=False,
            last_move_success=False,
            last_action_type="blocked_swap_attempt",
            time_since_movement=99,
            local_frustration=99,
            failed_swap_count=99,
            neighbor_identity_count=99,
            signal_blocked=0.8,
            signal_frustrated=0.4,
            signal_scope="explicit_allowed_diffusive_fields",
        )

    def _genomes(self):
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
            return load_selected_s08_policies(path)

    def test_specs_cover_required_modes_and_noise(self) -> None:
        specs = default_communication_ablation_specs()

        self.assertEqual(tuple(spec.name for spec in specs), REQUIRED_COMMUNICATION_ABLATIONS)
        self.assertFalse(specs[0].signal_enabled)
        self.assertEqual(specs[1].access_mode, "nearest_neighbor")
        self.assertTrue(specs[2].diffusion_enabled)
        self.assertTrue(specs[3].less_local)
        self.assertGreater(specs[4].noise_std, 0.0)

    def test_mask_keeps_neighbor_memory_and_switches_signal_visibility(self) -> None:
        projected = self._projected_view()
        specs = {spec.name: spec for spec in default_communication_ablation_specs()}
        memory_spec = neighbor_memory_spec()

        no_signal = apply_communication_ablation(projected, specs["memory_only_no_signal"])
        self.assertFalse(no_signal.last_move_success)
        self.assertEqual(no_signal.time_since_movement, memory_spec.time_since_capacity)
        self.assertEqual(no_signal.local_frustration, memory_spec.frustration_capacity)
        self.assertEqual(no_signal.failed_swap_count, memory_spec.failed_swap_capacity)
        self.assertEqual(no_signal.neighbor_identity_count, memory_spec.neighbor_capacity)
        self.assertEqual(no_signal.signal_scope, "none")
        self.assertEqual(no_signal.signal_blocked, 0.0)

        diffusive = apply_communication_ablation(projected, specs["diffusive_signaling"])
        self.assertEqual(diffusive.signal_blocked, 0.8)
        self.assertEqual(diffusive.signal_frustrated, 0.4)
        self.assertIsNone(diffusive.last_action_type)

    def test_default_s10_grid_reuses_s09_matched_seed_design(self) -> None:
        configs = default_s10_eval_configs(max_events=20, heldout_max_events=24, seeds=(19001,))
        audit = assert_matched_seed_design(configs)

        self.assertTrue(audit["success"])
        self.assertEqual(audit["benchmarkFamilyCounts"]["repair"], 1)
        self.assertEqual(audit["benchmarkFamilyCounts"]["frozen_cell"], 1)
        self.assertEqual(audit["benchmarkFamilyCounts"]["homeostatic"], 3)

    def test_small_signal_runs_have_no_oracle_hits_and_validate_noise(self) -> None:
        specs = [
            spec
            for spec in default_communication_ablation_specs()
            if spec.name in {"diffusive_signaling", "noisy_diffusive_signaling"}
        ]
        configs = (
            S09EvaluationConfig(
                benchmark_family="repair",
                values=(1, 2, 3, 4),
                task_name="frozen_damage",
                reliability_mode="no_fatigue_control",
                activation_seed=51,
                policy_seed=52,
                schedule_seed=53,
                max_events=12,
            ),
        )

        rows = run_communication_ablation_matrix(genomes=self._genomes(), configs=configs, specs=specs)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(not row["uses_global_oracle"] for row in rows))
        self.assertEqual(len({row["schedule_sha256"] for row in rows}), 1)
        by_mode = {row["communication_ablation"]: row for row in rows}
        noisy = json.loads(by_mode["noisy_diffusive_signaling"]["signal_observed_json"])
        diffusive = json.loads(by_mode["diffusive_signaling"]["signal_observed_json"])
        self.assertGreater(noisy["noiseAppliedCount"], 0)
        self.assertEqual(diffusive["noiseAppliedCount"], 0)
        self.assertTrue(noisy["targetDerivedSignalFieldsZeroed"])


if __name__ == "__main__":
    unittest.main()
