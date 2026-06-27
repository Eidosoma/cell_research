from __future__ import annotations

import json
import unittest

import pandas as pd

from memory_repair import (
    SYNTHESIS_PROXY_SCOPE_NOTE,
    aggregate_mechanism_metrics,
    build_mechanism_conclusions,
    build_traceability_matrix,
    policy_evidence_table,
    required_s15_evidence_sources,
    select_handoff_candidates,
    validate_synthesis_outputs,
)


class TestE04MinimalSynthesis(unittest.TestCase):
    def test_conclusions_trace_to_declared_sources_and_are_proxy_scoped(self) -> None:
        metrics = {
            "meanCellMemoryScoreDelta": 0.0,
            "maxCellMemoryScoreDelta": 0.0,
            "meanSignalFieldMemoryScoreDelta": 1.2,
            "bestCommunicationVariant": "randomized_control",
            "bestCommunicationMeanScoreDelta": 2.0,
            "diffusiveCommunicationMeanScoreDelta": -1.0,
            "nearestNeighborCommunicationMeanScoreDelta": -0.5,
            "learnedPolicyHoldoutMeanScore": 0.8,
            "learnedPolicyRelativeDrop": 0.01,
            "meanCentralizedGlobalMinusLocalScore": 0.1,
            "minCentralizedGlobalMinusLocalScore": 0.05,
        }
        conclusions = build_mechanism_conclusions(metrics)
        sources = pd.DataFrame(required_s15_evidence_sources())
        sources["exists"] = True
        trace = build_traceability_matrix(conclusions, sources)
        self.assertGreaterEqual(len(conclusions), 6)
        self.assertTrue(conclusions["claimBoundary"].eq(SYNTHESIS_PROXY_SCOPE_NOTE).all())
        self.assertTrue(trace["sourceExists"].all(), trace.to_string(index=False))
        cited = set()
        for payload in conclusions["evidenceSourceIds"]:
            cited.update(json.loads(payload))
        self.assertTrue(cited.issubset(set(sources["evidenceSourceId"])))

    def test_policy_evidence_selects_enhanced_handoff_candidates(self) -> None:
        profiles = pd.DataFrame(
            [
                {
                    "policyId": "baseline_classic_bubble_open_loop",
                    "policyGroup": "baseline",
                    "familyKind": "classic",
                    "overallCompetenceProxy": 0.9,
                },
                {
                    "policyId": "enhanced_local_memory_signal_adaptive",
                    "policyGroup": "enhanced",
                    "familyKind": "learned",
                    "overallCompetenceProxy": 0.82,
                },
                {
                    "policyId": "enhanced_s08_rank_1_e04_s08_candidate_abc",
                    "policyGroup": "enhanced",
                    "familyKind": "evolved",
                    "overallCompetenceProxy": 0.78,
                },
            ]
        )
        gaps = pd.DataFrame(
            [
                {"policyId": "enhanced_local_memory_signal_adaptive", "holdoutMeanScore": 0.84, "relativeSelectionToHoldoutDrop": 0.0},
                {"policyId": "enhanced_s08_rank_1_e04_s08_candidate_abc", "holdoutMeanScore": 0.61, "relativeSelectionToHoldoutDrop": 0.2},
            ]
        )
        central = pd.DataFrame(
            [
                {"localPolicyId": "enhanced_local_memory_signal_adaptive", "meanLocalControllerScore": 0.83, "meanGlobalMinusLocalScore": 0.06},
                {"localPolicyId": "enhanced_s08_rank_1_e04_s08_candidate_abc", "meanLocalControllerScore": 0.82, "meanGlobalMinusLocalScore": 0.08},
            ]
        )
        audits = pd.DataFrame([{"candidateId": "e04_s08_candidate_abc", "success": True}])
        evidence = policy_evidence_table(profiles, gaps, central, audits)
        selected = select_handoff_candidates(evidence, max_evolved=1)
        self.assertEqual(selected.iloc[0]["policyId"], "enhanced_local_memory_signal_adaptive")
        self.assertEqual(selected.iloc[0]["handoffTier"], "primary")
        self.assertTrue(selected["policyGroup"].eq("enhanced").all())
        self.assertEqual(len(selected), 2)

    def test_aggregate_metrics_and_validation_accept_complete_synthesis(self) -> None:
        memory = pd.DataFrame(
            [
                {"ablationAxis": "cell_memory_capacity", "ablationVariant": "one_bit", "meanScoreDelta": 0.0},
                {"ablationAxis": "signal_field_memory_capacity", "ablationVariant": "signal_field_memory", "meanScoreDelta": 1.4},
            ]
        )
        communication = pd.DataFrame(
            [
                {"ablationVariant": "randomized_control", "meanScoreDelta": 3.0},
                {"ablationVariant": "diffusive", "meanScoreDelta": -3.7},
                {"ablationVariant": "nearest_neighbor", "meanScoreDelta": -3.6},
            ]
        )
        profiles = pd.DataFrame(
            [{"policyId": "enhanced_local_memory_signal_adaptive", "policyGroup": "enhanced", "overallCompetenceProxy": 0.81}]
        )
        gaps = pd.DataFrame(
            [
                {
                    "policyId": "enhanced_local_memory_signal_adaptive",
                    "policyGroup": "enhanced",
                    "familyKind": "learned",
                    "holdoutMeanScore": 0.84,
                    "relativeSelectionToHoldoutDrop": 0.0,
                }
            ]
        )
        central = pd.DataFrame(
            [
                {
                    "localPolicyId": "enhanced_local_memory_signal_adaptive",
                    "localPolicyGroup": "enhanced",
                    "localFamilyKind": "learned",
                    "meanLocalControllerScore": 0.83,
                    "meanGlobalMinusLocalScore": 0.06,
                }
            ]
        )
        metrics = aggregate_mechanism_metrics(memory, communication, profiles, gaps, central)
        self.assertEqual(metrics["bestCommunicationVariant"], "randomized_control")
        self.assertGreater(metrics["meanSignalFieldMemoryScoreDelta"], 0.0)

        conclusions = build_mechanism_conclusions(metrics)
        sources = pd.DataFrame(required_s15_evidence_sources())
        sources["exists"] = True
        trace = build_traceability_matrix(conclusions, sources)
        handoff = select_handoff_candidates(policy_evidence_table(profiles, gaps, central, pd.DataFrame()))
        handoff["oracleAccessAllowed"] = False
        replay = pd.DataFrame(
            [
                {
                    "policyId": handoff.iloc[0]["policyId"],
                    "completed": True,
                    "handoffReplayScore": 0.75,
                    "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE,
                }
            ]
        )
        statuses = pd.DataFrame(
            [
                {"researchStepId": f"S{index:02d}", "success": True, "stepNumber": index, "claimBoundary": SYNTHESIS_PROXY_SCOPE_NOTE}
                for index in range(1, 15)
            ]
        )
        validation = validate_synthesis_outputs(
            statuses,
            sources,
            conclusions,
            trace,
            handoff,
            replay,
            report_exists=True,
            report_bundle_exists=True,
            deterministic_replay_match=True,
        )
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
