"""S14 frontier-candidate selection tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.e03.frontier_candidates import (
    FrontierConfig,
    build_policy_source_table,
    candidate_policy_json_records,
    evaluate_candidate_pool,
    finalize_candidate_set,
    initial_values_for_profile,
    pareto_front_mask,
    recheck_summary_frame,
    run_cpu_frontier_config,
    select_frontier_candidates,
    validation_frame,
)
from src.e03.policy_generation import semantic_hash
from src.e03.rule_dsl import SIMPLE_SWAP_LEFT_POLICY, SIMPLE_SWAP_RIGHT_POLICY, parse_policy
from src.e03.coarse_sweep import PolicyRecord


def source_record(source: str, source_kind: str = "unit") -> dict[str, object]:
    policy = parse_policy(source)
    return {
        "schema": "unit",
        "policyId": policy.policy_id,
        "policyName": policy.name,
        "sourceKind": source_kind,
        "semanticHash": semantic_hash(policy),
        "dslSha256": policy.sha256,
        "dslSource": policy.to_source(),
    }


def unit_universe(source_table: pd.DataFrame) -> pd.DataFrame:
    policies = [
        (parse_policy(SIMPLE_SWAP_LEFT_POLICY), "UC01", 0.92, 0.88, 0.35, 12.0, False),
        (parse_policy(SIMPLE_SWAP_RIGHT_POLICY), "UC02", 0.86, 0.82, 0.30, 6.0, False),
    ]
    rows = []
    for idx, (policy, class_id, score, heldout, improve, work, classic) in enumerate(policies):
        rows.append(
            {
                "policy_id": policy.policy_id,
                "policy_name": policy.name,
                "source_kind": "unit",
                "route": "cpu_fallback",
                "classic_dsl_seed": False,
                "classic_landmark": classic,
                "qd_candidate": idx == 0,
                "s08_archive_winner": idx == 0,
                "screen_score": score,
                "screen_heldout_final_sortedness_mean": heldout,
                "screen_heldout_improvement_mean": improve,
                "screen_heldout_work_mean": work,
                "best_final_sortedness": heldout,
                "n100_final_sortedness_mean": 0.60 + idx * 0.05,
                "invalid_run_count": 0,
                "timeout_run_count": 0,
                "embedding_x": float(idx),
                "embedding_y": 0.0,
                "embedding_z": 0.0,
                "class_id": class_id,
                "cautious_label": f"{class_id} label",
                "cluster_distance": 0.1,
            }
        )
    universe = pd.DataFrame(rows).merge(source_table, on="policy_id", how="left")
    universe["source_hash_valid"] = True
    universe["candidate_eligible"] = True
    universe["nonclassic_policy"] = True
    universe["basic_sorting_prior_pass"] = True
    universe["pareto_frontier_labels_json"] = ["[\"screen_pareto\"]", "[\"screen_pareto\"]"]
    universe["pareto_frontier_count"] = 1
    universe["selection_score"] = [0.95, 0.85]
    universe["s09_boundary_count"] = [1, 0]
    universe["s13_classic_neighbor_hit_count"] = [0, 1]
    universe["s12_claim_count_local_necessary_candidate"] = [1, 0]
    return universe


class FrontierCandidateTests(unittest.TestCase):
    def test_pareto_front_mask_marks_non_dominated_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "policy_id": ["a", "b", "c"],
                "quality": [0.9, 0.8, 0.7],
                "work": [10.0, 4.0, 20.0],
            }
        )
        mask = pareto_front_mask(frame, maximize=("quality",), minimize=("work",))
        self.assertEqual(frame.loc[mask, "policy_id"].tolist(), ["a", "b"])

    def test_source_table_verifies_policy_id_sha_and_semantic_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policies.jsonl"
            path.write_text(json.dumps(source_record(SIMPLE_SWAP_LEFT_POLICY)) + "\n", encoding="utf-8")
            table = build_policy_source_table(
                generated_policy_library=path,
                qd_candidates=pd.DataFrame(),
                qd_discovered_policy_library=None,
            )
        self.assertEqual(len(table), 1)
        self.assertTrue(bool(table.iloc[0]["source_hash_valid"]))
        self.assertEqual(table.iloc[0]["computed_policy_id"], table.iloc[0]["policy_id"])

    def test_input_profiles_are_deterministic_unique_arrays(self) -> None:
        profiles = ["random_permutation", "nearly_sorted", "reverse_sorted", "block_reversed", "alternating_high_low"]
        for profile in profiles:
            first = initial_values_for_profile(12, 1234, profile)
            second = initial_values_for_profile(12, 1234, profile)
            self.assertEqual(first, second)
            self.assertEqual(sorted(first), list(range(1, 13)))
        self.assertEqual(initial_values_for_profile(5, 1, "reverse_sorted"), (5, 4, 3, 2, 1))

    def test_cpu_recheck_and_candidate_json_records(self) -> None:
        record_data = source_record(SIMPLE_SWAP_LEFT_POLICY)
        policy = parse_policy(SIMPLE_SWAP_LEFT_POLICY)
        record = PolicyRecord(
            policy=policy,
            policy_id=policy.policy_id,
            policy_name=policy.name,
            source_kind="unit",
            semantic_hash=semantic_hash(policy),
            dsl_sha256=policy.sha256,
            features={},
            compiles_for_batch=False,
            requires_cpu_fallback=True,
            fallback_reasons=("unit",),
            route="cpu_fallback",
        )
        config = FrontierConfig("unit", "heldout", 5, 17, 10, "random_permutation")
        row = run_cpu_frontier_config(record, config)
        self.assertEqual(row["research_step_id"], "S14")
        self.assertFalse(row["invalid"])

        candidate = pd.DataFrame(
            [
                {
                    "policy_id": policy.policy_id,
                    "policy_name": policy.name,
                    "source_kind": "unit",
                    "class_id": "UC01",
                    "cautious_label": "unit",
                    "selection_score": 0.9,
                    "selection_reasons_json": "[\"unit\"]",
                    "pareto_frontier_labels_json": "[\"screen_pareto\"]",
                    "source_hash_valid": True,
                    "dsl_source": record_data["dslSource"],
                    "candidate_pool_rank": 1,
                }
            ]
        )
        run_df = evaluate_candidate_pool(candidate, [config])
        summary = recheck_summary_frame(run_df, candidate)
        final = finalize_candidate_set(summary, final_count=10, min_count=1)
        final["curated_rank"] = [1]
        records = candidate_policy_json_records(final)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["policyId"], policy.policy_id)
        self.assertTrue(records[0]["sourceHashVerified"])

    def test_selection_and_validation_accept_small_unit_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policies.jsonl"
            path.write_text(
                json.dumps(source_record(SIMPLE_SWAP_LEFT_POLICY)) + "\n" + json.dumps(source_record(SIMPLE_SWAP_RIGHT_POLICY)) + "\n",
                encoding="utf-8",
            )
            source_table = build_policy_source_table(
                generated_policy_library=path,
                qd_candidates=pd.DataFrame(),
                qd_discovered_policy_library=None,
            )
        universe = unit_universe(source_table)
        selected = select_frontier_candidates(universe, target_count=2, final_count=10)
        self.assertEqual(len(selected), 2)
        self.assertFalse(selected["classic_landmark"].any())
        configs = [FrontierConfig("unit_random", "heldout", 5, 17, 10, "random_permutation")]
        run_df = evaluate_candidate_pool(selected, configs)
        rechecked = recheck_summary_frame(run_df, selected)
        final = finalize_candidate_set(rechecked, final_count=10, min_count=1)
        records = candidate_policy_json_records(final)
        validation = validation_frame(
            universe=universe,
            selected_pool=selected,
            recheck_runs=run_df,
            rechecked_pool=rechecked,
            final_candidates=final,
            json_records=records,
            profiles_written=True,
            figure_written=True,
            unit_success=True,
            target_count=2,
            config_count=1,
        )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertTrue(cases["policy_jsonl_matches_final"])
        self.assertTrue(cases["final_candidates_nonclassic"])
        self.assertTrue(cases["final_sources_hash_verified"])


if __name__ == "__main__":
    unittest.main()
