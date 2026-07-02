"""E05 S07 scrambled-embryo benchmark tests."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e05.actions import ActionExecutor
from src.e05.scrambled_embryo import (
    LocalTargetNeighborDescentPolicy,
    adjacent_swap_target_error_delta,
    policy_source_mapping_rows,
    run_scrambled_benchmark,
    scramble_displacement_fraction,
    scramble_target_state,
    simulate_scrambled_run,
    target_coverage_rows,
    two_dimensional_targets,
)
from src.e05.targets import default_target_gallery, gradient_target


class ScrambledEmbryoTests(unittest.TestCase):
    def test_scramble_is_deterministic_and_positive_error(self) -> None:
        target = gradient_target(4, 3)
        first = scramble_target_state(target, 17)
        second = scramble_target_state(target, 17)

        self.assertEqual(first.signature(), second.signature())
        self.assertGreater(target.target_error(first), 0.0)
        self.assertGreater(scramble_displacement_fraction(target, first), 0.2)

    def test_local_target_policy_proposes_only_local_legal_swap_or_wait(self) -> None:
        target = gradient_target(4, 3)
        state = scramble_target_state(target, 19)
        policy = LocalTargetNeighborDescentPolicy()
        executor = ActionExecutor()
        actor_site_id = target.substrate.site_ids[0]
        request = policy.propose(state=state, target=target, actor_site_id=actor_site_id, rng=random.Random(3))

        self.assertIn(request.action_type, {"swap", "wait"})
        if request.action_type == "swap":
            self.assertTrue(state.is_adjacent(request.source_site_id, request.target_site_id))
            before_delta = adjacent_swap_target_error_delta(target, state, request.source_site_id, request.target_site_id)
            self.assertLess(before_delta, 0.0)
        result = executor.execute(state, request)
        self.assertTrue(result.allowed)

    def test_local_target_policy_reduces_error_on_gradient_fixture(self) -> None:
        target = gradient_target(4, 3)
        result = simulate_scrambled_run(
            target=target,
            policy_id="s07_local_target_neighbor_descent",
            seed=23,
            event_multiplier=50,
            records_per_run=10,
        )

        self.assertGreater(result.summary_row["initial_target_error"], 0.0)
        self.assertLessEqual(result.summary_row["final_target_error"], result.summary_row["initial_target_error"])
        self.assertGreater(result.summary_row["target_recovery_fraction"], 0.0)
        self.assertEqual(result.summary_row["population_delta_total"], 0)

    def test_policy_source_mapping_blocks_1d_sources_with_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classic = root / "e03_classic_policy_library.json"
            classic.write_text(
                json.dumps({"policies": [{"policyId": "classic_one"}]}),
                encoding="utf-8",
            )
            evolved = root / "e04_evolved_repair_policies.jsonl"
            evolved.write_text(json.dumps({"policy_id": "repair_one"}) + "\n", encoding="utf-8")

            rows = policy_source_mapping_rows((classic, evolved))

        blocked = [row for row in rows if row["source_experiment"] in {"E03", "E04"}]
        self.assertEqual(len(blocked), 2)
        self.assertTrue(all(row["mapping_status"] == "blocked_without_expanding_information_access" for row in blocked))
        self.assertTrue(all(row["blocker"] for row in blocked))

    def test_target_coverage_simulates_only_2d_gallery_targets(self) -> None:
        targets = default_target_gallery()
        coverage = target_coverage_rows(targets)
        simulated_ids = {row["target_id"] for row in coverage if row["simulated_in_s07"]}
        two_d_ids = {target.target_id for target in two_dimensional_targets(targets)}

        self.assertEqual(simulated_ids, two_d_ids)
        self.assertNotIn("sorted_row", simulated_ids)

    def test_compact_benchmark_returns_expected_run_matrix(self) -> None:
        targets = (gradient_target(4, 3),)
        summaries, traces, metrics, _runs = run_scrambled_benchmark(
            targets=targets,
            policy_ids=("s07_local_target_neighbor_descent", "s07_random_adjacent_swap_control"),
            seeds=(31, 32),
            event_multiplier=10,
            records_per_run=5,
        )

        self.assertEqual(len(summaries), 4)
        self.assertTrue(traces)
        self.assertEqual(len(metrics), 4 * 2 * 8)
        self.assertTrue(all(row["population_delta_total"] == 0 for row in summaries))


if __name__ == "__main__":
    unittest.main()
