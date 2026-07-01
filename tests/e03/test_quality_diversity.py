"""S08 quality-diversity search tests."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.e03.coarse_sweep import PolicyRecord
from src.e03.gpu_batch_simulator import compile_policy
from src.e03.policy_generation import policy_feature_flags, semantic_hash
from src.e03.quality_diversity import (
    add_descriptor_columns,
    add_novelty_columns,
    archive_digest,
    build_archive,
    generate_qd_candidates,
    seed_archive_frame,
    validation_frame,
)
from src.e03.rule_dsl import SIMPLE_SWAP_LEFT_POLICY, SIMPLE_SWAP_RIGHT_POLICY, parse_policy


def policy_record(source: str, *, source_kind: str = "unit_seed") -> PolicyRecord:
    policy = parse_policy(source)
    compiled = compile_policy(policy)
    return PolicyRecord(
        policy=policy,
        policy_id=policy.policy_id,
        policy_name=policy.name,
        source_kind=source_kind,
        semantic_hash=semantic_hash(policy),
        dsl_sha256=policy.sha256,
        features=policy_feature_flags(policy),
        compiles_for_batch=True,
        requires_cpu_fallback=bool(compiled.requires_cpu_fallback),
        fallback_reasons=compiled.fallback_reasons,
        route="cpu_fallback" if compiled.requires_cpu_fallback else "jax_batch",
    )


def summary_row(record: PolicyRecord, *, score: float, final: float, work: float, classic: bool = False) -> dict[str, object]:
    return {
        "schema": "unit",
        "experiment_id": "E03",
        "research_step_id": "S07",
        "policy_id": record.policy_id,
        "policy_name": record.policy_name,
        "source_kind": "classic_dsl_seed" if classic else record.source_kind,
        "route": record.route,
        "requires_cpu_fallback": record.requires_cpu_fallback,
        "screen_run_count": 1,
        "heldout_run_count": 1,
        "scale_run_count": 0,
        "invalid_run_count": 0,
        "timeout_run_count": 0,
        "screen_train_final_sortedness_mean": final,
        "screen_heldout_final_sortedness_mean": final,
        "screen_heldout_sorted_run_fraction": 0.0,
        "screen_heldout_improvement_mean": final - 0.45,
        "screen_heldout_work_mean": work,
        "screen_score": score,
        "n100_final_sortedness_mean": float("nan"),
        "n1000_final_sortedness_mean": float("nan"),
        "best_final_sortedness": final,
        "classic_dsl_seed": classic,
    }


class QualityDiversityTests(unittest.TestCase):
    def test_archive_keeps_best_policy_per_descriptor_cell(self) -> None:
        left = policy_record(SIMPLE_SWAP_LEFT_POLICY, source_kind="classic_dsl_seed")
        right = policy_record(SIMPLE_SWAP_RIGHT_POLICY)
        frame = pd.DataFrame(
            [
                summary_row(left, score=0.4, final=0.62, work=4, classic=True),
                summary_row(right, score=0.6, final=0.63, work=4),
            ]
        )
        seed = seed_archive_frame(frame, [left, right])
        seed = add_novelty_columns(seed, s07_reference=seed, classic_reference=seed[seed["classic_dsl_seed"] == True])  # noqa: E712
        archive = build_archive(seed)
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.iloc[0]["policy_id"], right.policy_id)
        self.assertEqual(archive_digest(archive), archive_digest(build_archive(seed)))

    def test_qd_generation_is_deterministic_and_lineage_is_valid(self) -> None:
        parents = [
            policy_record(SIMPLE_SWAP_LEFT_POLICY),
            policy_record(SIMPLE_SWAP_RIGHT_POLICY),
        ]
        seen_a = {record.semantic_hash: record.policy_id for record in parents}
        seen_b = {record.semantic_hash: record.policy_id for record in parents}
        first = generate_qd_candidates(parent_records=parents, seen_semantics=seen_a, target_count=4, generation=1, seed=99)
        second = generate_qd_candidates(parent_records=parents, seen_semantics=seen_b, target_count=4, generation=1, seed=99)
        self.assertEqual([item.policy_id for item in first], [item.policy_id for item in second])
        self.assertEqual(len(first), 4)
        parent_ids = {record.policy_id for record in parents}
        for item in first:
            self.assertTrue(set(item.parent_policy_ids).issubset(parent_ids))
            self.assertIn(item.source_kind, {"qd_mutation", "qd_recombination"})
            self.assertTrue(item.tiny_execution_signature)

    def test_validation_requires_archive_and_lineage_consistency(self) -> None:
        left = policy_record(SIMPLE_SWAP_LEFT_POLICY, source_kind="classic_dsl_seed")
        right = policy_record(SIMPLE_SWAP_RIGHT_POLICY)
        base = pd.DataFrame(
            [
                summary_row(left, score=0.4, final=0.62, work=4, classic=True),
                summary_row(right, score=0.6, final=0.72, work=10),
            ]
        )
        seed = add_descriptor_columns(base)
        seed["generation"] = [0, 1]
        seed["qd_candidate"] = [False, True]
        seed["semantic_hash"] = [left.semantic_hash, right.semantic_hash]
        seed["novelty_score"] = [0.0, 0.2]
        seed["novelty_to_classics"] = [0.0, 0.2]
        archive = build_archive(seed)
        discovered = seed[seed["qd_candidate"] == True].copy()  # noqa: E712
        discovered["nonclassic_cell"] = True
        discovered["archive_winner"] = discovered["policy_id"].isin(set(archive["policy_id"]))
        lineage = pd.DataFrame(
            [
                {
                    "child_policy_id": right.policy_id,
                    "parent_policy_ids_json": json.dumps([left.policy_id]),
                }
            ]
        )
        validation = validation_frame(
            archive=archive,
            discovered=discovered,
            lineage=lineage,
            candidate_summary=discovered,
            candidate_count_target=1,
            known_policy_ids={left.policy_id, right.policy_id},
            recomputed_archive=build_archive(seed),
            figure_exists=True,
        )
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
