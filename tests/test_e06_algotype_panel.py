from __future__ import annotations

import unittest
from pathlib import Path

from chimera.panel import (
    E04_HANDOFF_PATH,
    build_algotype_panel,
    direct_policy_records,
    handoff_policy_from_competence_spec,
    instantiate_panel_policy,
    run_validation_panel,
)


class TestE06AlgotypePanel(unittest.TestCase):
    def test_direct_records_roundtrip_and_validate(self) -> None:
        panel = build_algotype_panel(frontier_limit=0, discovered_limit=0, handoff_limit=0)
        direct_ids = {record["panelPolicyId"] for record in direct_policy_records()}
        direct_panel = panel[panel["panelPolicyId"].isin(direct_ids)].head(3)
        validation, summary = run_validation_panel(
            direct_panel,
            input_seeds=(6101,),
            goal_directions=("increasing",),
            max_activations=500,
        )
        self.assertEqual(len(validation), len(direct_panel))
        self.assertTrue(summary["serializationRoundtripSuccess"].all())
        self.assertTrue(summary["canRunCommonInterface"].all())
        self.assertTrue(validation["reproducibleSeedReplay"].all())

    def test_frontier_records_instantiate_when_artifacts_exist(self) -> None:
        if not Path("/previous-artifacts/E03/research_steps/S14/frontier_candidates.parquet").exists():
            self.skipTest("E03 frontier artifacts are not mounted")
        panel = build_algotype_panel(frontier_limit=2, discovered_limit=0, handoff_limit=0)
        frontier = panel[panel["panelGroup"] == "e03_frontier"]
        self.assertEqual(len(frontier), 2)
        for row in frontier.to_dict(orient="records"):
            policy = instantiate_panel_policy(row)
            self.assertEqual(policy.family, "dsl")

    def test_e04_handoff_policy_reconstructs_when_available(self) -> None:
        if not E04_HANDOFF_PATH.exists():
            self.skipTest("E04 handoff artifacts are not mounted")
        panel = build_algotype_panel(frontier_limit=0, discovered_limit=0, handoff_limit=1)
        handoff = panel[panel["panelGroup"] == "e04_handoff"]
        self.assertEqual(len(handoff), 1)
        row = handoff.iloc[0].to_dict()
        policy = handoff_policy_from_competence_spec(row["constructorPayloadJson"])
        self.assertTrue(policy.policy_id.startswith("e04_signal_"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
