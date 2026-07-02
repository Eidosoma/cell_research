"""E05 S11 GPU-batched tissue simulation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.gpu_tissue import (
    DEFAULT_GPU_EVENT_MULTIPLIERS,
    DEFAULT_GPU_SWEEP_SEEDS,
    default_gpu_task_templates,
    initial_state_for_template,
    make_deterministic_schedules,
    neighbor_index_tensor,
    replay_reference_cpu_from_schedule,
    run_gpu_tissue_sweep,
    run_tensor_batch,
    schedule_seed_for_group,
    state_to_occupant_indices,
    target_distance_matrix,
    validate_cpu_gpu_agreement,
    validate_reference_cpu_agreement,
    vectorization_scope_rows,
)
from src.e05.scrambled_embryo import RUNNABLE_POLICY_IDS


class GpuTissueTests(unittest.TestCase):
    def test_default_templates_cover_s07_to_s10_square_grid_tasks(self) -> None:
        templates = default_gpu_task_templates()
        source_steps = {template.source_research_step_id for template in templates}

        self.assertEqual(source_steps, {"S07", "S08", "S09", "S10"})
        self.assertEqual(len(templates), 8)
        self.assertTrue(all(template.target.substrate.kind == "square_grid_2d" for template in templates))
        self.assertTrue(all(template.site_count > 1 for template in templates))

    def test_vectorization_scope_documents_blockers(self) -> None:
        rows = vectorization_scope_rows()
        blocked = [row for row in rows if not row["vectorized_in_s11"]]

        self.assertTrue(any(row["scope_id"] == "s08_missing_patch" for row in blocked))
        self.assertTrue(any(row["scope_id"] == "graph_or_irregular_substrates" for row in blocked))
        self.assertTrue(all(row["reason"] for row in blocked))

    def test_tensor_cpu_kernel_matches_reference_replay(self) -> None:
        template = default_gpu_task_templates()[0]
        seed = 6101
        policy_id = "s07_local_target_neighbor_descent"
        state, _metadata = initial_state_for_template(template, seed)
        distance = target_distance_matrix(template.target)
        neighbors = neighbor_index_tensor(template.target)
        event_multiplier = 3
        schedules = make_deterministic_schedules(
            batch_size=1,
            event_cap=event_multiplier * template.site_count,
            site_count=template.site_count,
            max_degree=int(neighbors.shape[1]),
            schedule_seed=schedule_seed_for_group(
                template=template,
                policy_id=policy_id,
                event_multiplier=event_multiplier,
                seeds=(seed,),
            ),
        )
        tensor_output = run_tensor_batch(
            initial_occupants=torch.tensor([state_to_occupant_indices(template.target, state)], dtype=torch.long),
            distance_matrix=distance,
            neighbor_indices=neighbors,
            policy_id=policy_id,
            schedules=schedules,
            records_per_run=3,
            device="cpu",
        )
        replay = replay_reference_cpu_from_schedule(
            target=template.target,
            initial_state=state,
            policy_id=policy_id,
            actor_schedule=schedules.actor_site_indices[0].tolist(),
            neighbor_order_schedule=schedules.neighbor_order_indices[0].tolist(),
        )

        self.assertEqual(
            tuple(tensor_output.final_occupants[0].tolist()),
            state_to_occupant_indices(template.target, replay["final_state"]),
        )
        self.assertEqual(int(tensor_output.accepted_swaps[0].item()), replay["accepted_swaps"])
        self.assertEqual(int(tensor_output.wait_actions[0].item()), replay["wait_actions"])

    def test_reference_validation_rows_pass(self) -> None:
        rows = validate_reference_cpu_agreement(
            templates=default_gpu_task_templates()[:1],
            policy_ids=RUNNABLE_POLICY_IDS,
            seeds=(6201,),
            event_multiplier=2,
        )

        self.assertEqual(len(rows), len(RUNNABLE_POLICY_IDS))
        self.assertTrue(all(row["success"] for row in rows))

    def test_cpu_gpu_validation_rows_pass_when_cuda_available(self) -> None:
        rows = validate_cpu_gpu_agreement(
            templates=default_gpu_task_templates()[:1],
            policy_ids=RUNNABLE_POLICY_IDS,
            seeds=(6301,),
            event_multiplier=2,
        )

        if torch.cuda.is_available():
            self.assertTrue(all(row["success"] for row in rows))
            self.assertTrue(all(row["final_occupants_equal"] for row in rows))
        else:
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["success"])

    def test_small_sweep_records_accounting_and_memory_rows(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = run_gpu_tissue_sweep(
            templates=default_gpu_task_templates()[:1],
            policy_ids=RUNNABLE_POLICY_IDS,
            seeds=DEFAULT_GPU_SWEEP_SEEDS[:2],
            event_multipliers=DEFAULT_GPU_EVENT_MULTIPLIERS[:1],
            records_per_run=2,
            device=device,
        )

        self.assertEqual(len(result.summary_rows), 1 * len(RUNNABLE_POLICY_IDS) * 2 * 1)
        self.assertTrue(result.trace_rows)
        self.assertEqual(len(result.memory_rows), len(RUNNABLE_POLICY_IDS))
        self.assertTrue(all(row["population_delta_total"] == 0 for row in result.summary_rows))
        self.assertTrue(all(row["accepted_swaps"] == row["attempted_swaps"] for row in result.summary_rows))
        self.assertTrue(all(row["total_energy_cost"] == float(row["accepted_swaps"]) for row in result.summary_rows))


if __name__ == "__main__":
    unittest.main()
