"""S05 policy generation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.e03.policy_generation import (
    CandidatePolicy,
    _accept_candidate,
    accepted_policy_frame,
    generate_policy_library,
    semantic_hash,
    validate_generated_library,
    write_policy_jsonl,
)
from src.e03.rule_dsl import SIMPLE_SWAP_LEFT_POLICY, parse_policy


class PolicyGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_validated(self) -> None:
        first, first_audit = generate_policy_library(target_unique=120, seed=12345, max_attempts=2500)
        second, _second_audit = generate_policy_library(target_unique=120, seed=12345, max_attempts=2500)
        self.assertEqual([item.policy.policy_id for item in first], [item.policy.policy_id for item in second])
        self.assertGreaterEqual(len(first), 120)
        validation = validate_generated_library(first, 120)
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))
        self.assertGreaterEqual(int((first_audit["accepted"] == False).sum()), 1)

    def test_duplicate_filter_uses_semantics_not_candidate_name(self) -> None:
        seen: dict[str, str] = {}
        first = CandidatePolicy(SIMPLE_SWAP_LEFT_POLICY.replace("compare_swap_left", "candidate_a"), "unit", 0, 1)
        second = CandidatePolicy(SIMPLE_SWAP_LEFT_POLICY.replace("compare_swap_left", "candidate_b"), "unit", 1, 1)
        accepted, rejected = _accept_candidate(first, seen)
        self.assertIsNotNone(accepted)
        self.assertIsNone(rejected)
        accepted_second, rejected_second = _accept_candidate(second, seen)
        self.assertIsNone(accepted_second)
        self.assertIsNotNone(rejected_second)
        self.assertEqual(rejected_second.reason, "duplicate")

    def test_jsonl_writer_outputs_parseable_sources(self) -> None:
        policies, _audit = generate_policy_library(target_unique=25, seed=7, max_attempts=1000)
        frame = accepted_policy_frame(policies)
        self.assertEqual(frame["semantic_hash"].nunique(), len(frame))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policies.jsonl"
            write_policy_jsonl(path, policies)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(policies))
        for item in policies[:5]:
            parsed = parse_policy(item.policy.to_source())
            self.assertEqual(semantic_hash(parsed), item.semantic_hash)


if __name__ == "__main__":
    unittest.main()
