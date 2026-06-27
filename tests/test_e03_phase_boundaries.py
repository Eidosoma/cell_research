from __future__ import annotations

import json
import unittest

import pandas as pd

from morphospace import (
    detect_phase_boundaries,
    near_boundary_variance_table,
    phase_boundary_config_dict,
    phase_family_metadata,
    select_phase_boundary_candidates,
    summarize_repeated_seed_runs,
)
from morphospace.phase_boundaries import PhaseBoundaryTask
from morphospace.rule_dsl import DSL_VERSION


def program(policy_id: str, action: str = "swap_right", probability: float = 1.0) -> str:
    then: dict[str, object] = {"action": action}
    if probability != 1.0:
        then["probability"] = probability
    return json.dumps(
        {
            "version": DSL_VERSION,
            "policy_id": policy_id,
            "name": policy_id,
            "rules": [{"name": "r", "when": [{"op": "always"}], "then": then}],
            "default": {"action": "wait"},
            "initial_state": {},
        },
        sort_keys=True,
    )


def corpus_rows() -> pd.DataFrame:
    rows = [
        {
            "policyId": "pc_elite0000000001",
            "generationMethod": "single_rule_sweep",
            "lineageId": "single_compare_right_gt_swap_right_1",
            "parentPolicyIdsJson": "[]",
            "dslProgramJson": program("pc_elite0000000001"),
            "generationIndex": 1,
            "complexityScore": 2,
            "stateKeyCount": 0,
            "stochasticActionCount": 0,
            "targetUpdateCount": 0,
        },
        {
            "policyId": "pc_bubble00000001",
            "generationMethod": "classic_parameter_sweep",
            "lineageId": "bubble_increasing_right_first_0.5",
            "parentPolicyIdsJson": "[]",
            "dslProgramJson": program("pc_bubble00000001", probability=0.5),
            "generationIndex": 2,
            "complexityScore": 6,
            "stateKeyCount": 0,
            "stochasticActionCount": 2,
            "targetUpdateCount": 0,
        },
        {
            "policyId": "pc_prob0000000001",
            "generationMethod": "probability_mutation",
            "lineageId": "mut_prob_pc_elite0000000001_0.25",
            "parentPolicyIdsJson": '["pc_elite0000000001"]',
            "dslProgramJson": program("pc_prob0000000001", probability=0.25),
            "generationIndex": 3,
            "complexityScore": 3,
            "stateKeyCount": 0,
            "stochasticActionCount": 1,
            "targetUpdateCount": 0,
        },
        {
            "policyId": "pc_memory00000001",
            "generationMethod": "memory_mutation",
            "lineageId": "mut_memory_pc_elite0000000001",
            "parentPolicyIdsJson": '["pc_elite0000000001"]',
            "dslProgramJson": program("pc_memory00000001"),
            "generationIndex": 4,
            "complexityScore": 4,
            "stateKeyCount": 1,
            "stochasticActionCount": 0,
            "targetUpdateCount": 0,
        },
        {
            "policyId": "pc_target00000001",
            "generationMethod": "target_memory_mutation",
            "lineageId": "mut_target_pc_elite0000000001",
            "parentPolicyIdsJson": '["pc_elite0000000001"]',
            "dslProgramJson": program("pc_target00000001", "swap_target"),
            "generationIndex": 5,
            "complexityScore": 4,
            "stateKeyCount": 1,
            "stochasticActionCount": 0,
            "targetUpdateCount": 1,
        },
    ]
    for row in rows:
        row.setdefault("family", "generated")
        row.setdefault("lineageDepth", 0)
        row.setdefault("ruleCount", 1)
        row.setdefault("predicateCount", 1)
        row.setdefault("actionCountsJson", "{}")
    return pd.DataFrame(rows)


class TestE03PhaseBoundaries(unittest.TestCase):
    def test_phase_family_metadata_parses_core_axes(self) -> None:
        classic = phase_family_metadata(
            {
                "policyId": "pc_bubble00000001",
                "generationMethod": "classic_parameter_sweep",
                "lineageId": "bubble_increasing_right_first_0.5",
                "dslProgramJson": program("pc_bubble00000001", probability=0.5),
                "generationIndex": 1,
            }
        )
        self.assertEqual(classic["familyKey"], "classic_bubble_increasing_right_first")
        self.assertEqual(classic["parameterAxis"], "swap_probability")
        self.assertAlmostEqual(classic["parameterValue"], 0.5)

        single = phase_family_metadata(
            {
                "policyId": "pc_single0000001",
                "generationMethod": "single_rule_sweep",
                "lineageId": "single_compare_left_lteq_swap_target_0.5",
                "dslProgramJson": program("pc_single0000001", "swap_target", 0.5),
                "generationIndex": 1,
            }
        )
        self.assertEqual(single["familyKey"], "single_rule_compare_left_lteq")
        self.assertEqual(single["parameterAxis"], "swap_preference_and_wait_probability")

    def test_candidate_selection_includes_elite_and_parameterized_neighbors(self) -> None:
        corpus = corpus_rows()
        elites = pd.DataFrame([{"policyId": "pc_elite0000000001", "eliteRank": 1}])
        candidates = select_phase_boundary_candidates(corpus, elites, max_policies=10)
        self.assertIn("pc_elite0000000001", set(candidates["policyId"]))
        self.assertIn("classic_bubble_increasing_right_first", set(candidates["familyKey"]))
        self.assertIn("probability_mutation_pc_elite0000000001", set(candidates["familyKey"]))
        self.assertIn("memory_mutation_pc_elite0000000001", set(candidates["familyKey"]))
        self.assertIn("target_memory_mutation_pc_elite0000000001", set(candidates["familyKey"]))

    def test_repeated_seed_summary_and_boundary_detection(self) -> None:
        base = {
            "taskId": "task",
            "familyKey": "classic_bubble_increasing_right_first",
            "parameterAxis": "swap_probability",
            "generationMethod": "classic_parameter_sweep",
            "taskFamily": "sorting",
            "taskPanel": "test",
            "inputProfile": "toy",
            "frozenVariant": "none",
            "isS08Elite": False,
            "s08EliteRank": None,
        }
        rows = []
        for seed in range(5):
            rows.append(
                {
                    **base,
                    "policyId": "left",
                    "lineageId": "bubble_increasing_right_first_0.25",
                    "parameterValue": 0.25,
                    "parameterLabel": "p=0.25",
                    "candidateRank": 1,
                    "seedIndex": seed,
                    "completedNumeric": 0.0,
                    "finalSortednessScore": 0.3,
                    "delayedGratification": 0.0,
                    "dgEventCount": 0,
                    "oscillationProxy": 0.0,
                    "finalAggregation": 0.5,
                    "robustnessScore": None,
                    "swapCount": 1,
                    "comparisonCount": 1,
                    "activationCount": 10,
                    "stopReason": "cap",
                }
            )
            rows.append(
                {
                    **base,
                    "policyId": "right",
                    "lineageId": "bubble_increasing_right_first_1",
                    "parameterValue": 1.0,
                    "parameterLabel": "p=1",
                    "candidateRank": 2,
                    "seedIndex": seed,
                    "completedNumeric": 1.0,
                    "finalSortednessScore": 1.0,
                    "delayedGratification": 0.1,
                    "dgEventCount": 1,
                    "oscillationProxy": 0.0,
                    "finalAggregation": 0.5,
                    "robustnessScore": None,
                    "swapCount": 3,
                    "comparisonCount": 3,
                    "activationCount": 12,
                    "stopReason": "sorted",
                }
            )
        summary = summarize_repeated_seed_runs(pd.DataFrame(rows))
        self.assertEqual(len(summary), 2)
        self.assertEqual(int(summary["seedCount"].min()), 5)

        boundaries = detect_phase_boundaries(summary)
        self.assertEqual(len(boundaries), 1)
        self.assertTrue(bool(boundaries.iloc[0]["sortingSuccessTransition"]))
        variance = near_boundary_variance_table(boundaries, summary)
        self.assertEqual(len(variance), 2)
        self.assertEqual(int(variance["seedCount"].min()), 5)

    def test_config_records_tasks_and_seed_count(self) -> None:
        config = phase_boundary_config_dict(
            seed_count=5,
            max_policies=10,
            tasks=[
                PhaseBoundaryTask(
                    task_id="task",
                    task_family="sorting",
                    task_panel="panel",
                    input_profile="toy",
                    initial_values=(3, 1, 2),
                    scheduler_seed_base=1,
                    tie_seed_base=2,
                )
            ],
        )
        self.assertEqual(config["seedCountPerPolicyTask"], 5)
        self.assertEqual(config["tasks"][0]["taskId"], "task")
        self.assertIn("transitionRules", config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
