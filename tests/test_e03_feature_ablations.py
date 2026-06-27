from __future__ import annotations

import unittest

import pandas as pd

from morphospace import (
    ablate_program,
    build_ablation_policy_table,
    feature_effect_summary,
    local_inversion_program,
    paired_delta_table,
    parse_rule_program,
    select_s12_source_policies,
    validate_ablation_outputs,
)


def target_memory_program() -> str:
    return parse_rule_program(
        {
            "version": "e03_s02_rule_dsl.v1",
            "policy_id": "toy_target_memory",
            "name": "toy target memory",
            "initial_state": {"target_position": 0, "seen": 0},
            "rules": [
                {
                    "name": "target_then_swap",
                    "when": [{"op": "always"}],
                    "then": {
                        "action": "swap_target",
                        "updates": [
                            {"op": "estimate_target_position", "key": "target_position", "direction": "increasing"},
                            {"op": "increment", "key": "seen", "amount": 1, "min": 0, "max": 4},
                        ],
                    },
                }
            ],
            "default": {"action": "wait", "reason": "target_missing"},
        }
    ).to_json()


def stochastic_tie_program() -> str:
    return parse_rule_program(
        {
            "version": "e03_s02_rule_dsl.v1",
            "policy_id": "toy_stochastic_tie",
            "name": "toy stochastic tie",
            "initial_state": {},
            "rules": [
                {
                    "name": "left_tie",
                    "when": [{"op": "compare_left", "operator": "<="}],
                    "then": {"action": "swap_left", "probability": 0.5},
                },
                {
                    "name": "right_tie",
                    "when": [{"op": "compare_right", "operator": ">="}],
                    "then": {"action": "swap_right", "probability": 0.5},
                },
            ],
            "default": {"action": "wait"},
        }
    ).to_json()


def corpus_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": "classic_p",
                "family": "hand_designed",
                "generationMethod": "classic_parameter_sweep",
                "lineageId": "bubble_increasing",
                "dslProgramJson": local_inversion_program("classic_p").to_json(),
                "ruleCount": 2,
                "predicateCount": 2,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 0,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 4,
                "actionCountsJson": '{"swap_left":1,"swap_right":1}',
            },
            {
                "policyId": "elite_p",
                "family": "generated",
                "generationMethod": "single_rule_sweep",
                "lineageId": "elite_lineage",
                "dslProgramJson": stochastic_tie_program(),
                "ruleCount": 2,
                "predicateCount": 2,
                "updateCount": 0,
                "stateKeyCount": 0,
                "stochasticActionCount": 2,
                "targetUpdateCount": 0,
                "signalCount": 0,
                "complexityScore": 5,
                "actionCountsJson": '{"swap_left":1,"swap_right":1}',
            },
            {
                "policyId": "generated_p",
                "family": "generated",
                "generationMethod": "target_memory_mutation",
                "lineageId": "generated_lineage",
                "dslProgramJson": target_memory_program(),
                "ruleCount": 1,
                "predicateCount": 1,
                "updateCount": 2,
                "stateKeyCount": 2,
                "stochasticActionCount": 0,
                "targetUpdateCount": 1,
                "signalCount": 0,
                "complexityScore": 6,
                "actionCountsJson": '{"swap_target":1}',
            },
        ]
    )


def assignments_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": "classic_p",
                "className": "UC_classic",
                "classLabel": "classic_local_inversion",
                "primaryRole": "classic",
                "isClassicPolicy": True,
                "isNullPolicy": False,
                "isS08Elite": False,
            },
            {
                "policyId": "elite_p",
                "className": "UC_elite",
                "classLabel": "elite",
                "primaryRole": "elite",
                "isClassicPolicy": False,
                "isNullPolicy": False,
                "isS08Elite": True,
            },
            {
                "policyId": "generated_p",
                "className": "UC_generated",
                "classLabel": "generated",
                "primaryRole": "generated",
                "isClassicPolicy": False,
                "isNullPolicy": False,
                "isS08Elite": False,
            },
        ]
    )


class TestE03FeatureAblations(unittest.TestCase):
    def test_ablation_rewrites_remain_parseable(self) -> None:
        program = stochastic_tie_program()
        left = ablate_program(program, "suppress_compare_left")
        self.assertTrue(left.changed)
        self.assertIn("position_compare", left.program.to_json())

        deterministic = ablate_program(program, "remove_stochasticity")
        self.assertTrue(deterministic.changed)
        self.assertNotIn("probability", deterministic.program.to_json())

        strict = ablate_program(program, "strict_tie_rules")
        self.assertTrue(strict.changed)
        strict_program = parse_rule_program(strict.program.to_json())
        operators = [predicate.operator for rule in strict_program.rules for predicate in rule.when]
        self.assertIn("<", operators)
        self.assertIn(">", operators)

        no_target = ablate_program(target_memory_program(), "remove_target_position")
        self.assertTrue(no_target.changed)
        self.assertNotIn("swap_target", no_target.program.to_json())

        no_memory = ablate_program(target_memory_program(), "remove_memory_updates")
        self.assertTrue(no_memory.changed)
        self.assertNotIn('"seen"', no_memory.program.to_json())

    def test_policy_selection_and_ablation_table(self) -> None:
        corpus = corpus_fixture()
        selected = select_s12_source_policies(
            corpus_df=corpus,
            elites_df=pd.DataFrame([{"policyId": "elite_p", "eliteRank": 1}]),
            assignments_df=assignments_fixture(),
            exemplars_df=pd.DataFrame(
                [{"policyId": "generated_p", "className": "UC_generated", "exemplarRank": 1}]
            ),
            max_generated_per_class=1,
        )
        roles = ",".join(selected["sourceRole"].astype(str))
        self.assertIn("classic", roles)
        self.assertIn("elite", roles)
        self.assertIn("generated", roles)

        policy_table = build_ablation_policy_table(selected)
        self.assertTrue(policy_table["parseSuccess"].all())
        self.assertTrue(policy_table["roundtripSuccess"].all())
        self.assertIn("original", set(policy_table["ablationType"]))
        self.assertIn("remove_target_position", set(policy_table["ablationType"]))
        self.assertIn("remove_stochasticity", set(policy_table["ablationType"]))

    def test_paired_delta_and_validation(self) -> None:
        selected = select_s12_source_policies(
            corpus_df=corpus_fixture(),
            elites_df=pd.DataFrame([{"policyId": "elite_p", "eliteRank": 1}]),
            assignments_df=assignments_fixture(),
            exemplars_df=pd.DataFrame(
                [{"policyId": "generated_p", "className": "UC_generated", "exemplarRank": 1}]
            ),
            max_generated_per_class=1,
        )
        policy_table = build_ablation_policy_table(selected)
        source_id = "classic_p"
        ablation_id = policy_table[
            policy_table["sourcePolicyId"].eq(source_id) & policy_table["ablationType"].eq("suppress_compare_left")
        ]["ablationPolicyId"].iloc[0]
        vector_rows = pd.DataFrame(
            [
                {
                    "vectorId": "v_original",
                    "policyId": source_id,
                    "taskId": "toy_task",
                    "taskFamily": "sorting",
                    "taskPanel": "toy",
                    "inputProfile": "toy",
                    "frozenVariant": "none",
                    "frozenCount": 0,
                    "replicateIndex": 0,
                    "sourceRunId": "run_original",
                    "completionSuccess": 1.0,
                    "finalSortednessScore": 1.0,
                    "energyScore": 0.8,
                },
                {
                    "vectorId": "v_ablated",
                    "policyId": ablation_id,
                    "taskId": "toy_task",
                    "taskFamily": "sorting",
                    "taskPanel": "toy",
                    "inputProfile": "toy",
                    "frozenVariant": "none",
                    "frozenCount": 0,
                    "replicateIndex": 0,
                    "sourceRunId": "run_ablated",
                    "completionSuccess": 0.0,
                    "finalSortednessScore": 0.5,
                    "energyScore": 0.9,
                },
            ]
        )
        deltas = paired_delta_table(vector_rows, policy_table)
        target = deltas[deltas["ablationType"].eq("suppress_compare_left")].iloc[0]
        self.assertAlmostEqual(float(target["finalSortednessScoreDelta"]), -0.5)
        summary = feature_effect_summary(deltas)
        self.assertFalse(summary.empty)

        run_df = pd.DataFrame(
            [
                {
                    "policyId": row["ablationPolicyId"],
                    "taskId": "toy_task",
                    "seedIndex": 0,
                    "finalSortednessPercent": 100.0,
                    "finalMonotonicityError": 0,
                    "valueCountsConserved": True,
                }
                for _, row in policy_table.iterrows()
            ]
        )
        validation = validate_ablation_outputs(
            source_policies_df=selected,
            policy_table_df=policy_table,
            run_df=run_df,
            vector_df=pd.DataFrame(
                {
                    "vectorId": [f"v{i}" for i in range(len(run_df))],
                    "policyId": run_df["policyId"],
                }
            ),
            paired_delta_df=pd.DataFrame(
                {
                    "sourcePolicyId": ["classic_p"] * (len(policy_table) - 3),
                    "ablationType": ["suppress_compare_left"] * (len(policy_table) - 3),
                    "taskId": ["toy_task"] * (len(policy_table) - 3),
                    "replicateIndex": [0] * (len(policy_table) - 3),
                }
            ),
            effect_summary_df=summary,
            parser_validation_df=policy_table[["parseSuccess", "roundtripSuccess"]].copy(),
            upstream_statuses={step: {"success": True, "status": "completed"} for step in ["S08", "S09", "S10", "S11"]},
            repo_test_payload={"success": True, "returnCode": 0, "command": ["unittest"]},
            expected_task_count=1,
            seed_count=1,
            report_exists=True,
            figure_paths=["a.png", "b.png"],
            s13_dir_exists=False,
        )
        failed = validation[~validation["success"]]
        self.assertTrue(failed.empty, failed.to_string())


if __name__ == "__main__":
    unittest.main(verbosity=2)
