from __future__ import annotations

import json
import unittest

import pandas as pd

from chimera.history import (
    NO_MUTANT_HISTORY_PROTOCOL,
    SIMULTANEOUS_HISTORY_PROTOCOL,
    build_developmental_history_condition_matrix,
    history_protocol_table,
    select_s12_contexts,
)
from chimera.mixtures import compact_json
from chimera.mutants import mutant_clone_config_by_variant


def _s11_clone_row(
    index: int,
    *,
    outcome: str,
    behavior: str,
    governance: str,
    interface: str,
    graft_outcome: str,
) -> dict[str, object]:
    host = "host_policy"
    donor = "donor_policy"
    values = list(range(100, 0, -1))
    positions = [0, 1, 2]
    assignment = [donor if pos in positions else host for pos in range(100)]
    return {
        "conditionId": f"s11_mutant_{index}",
        "s11ConditionKind": "mutant_clone",
        "matchedNoMutantConditionId": f"s11_control_{index}",
        "s11MatchedNoGraftConditionId": f"s10_control_{index}",
        "mutantOutcomeClass": outcome,
        "mutantBehaviorVariant": behavior,
        "mutantObjectiveFamily": "neutral" if behavior == "neutral_clone_control" else "selfish",
        "mutantCloneSize": 3,
        "mutantClonePositionName": "left_edge_clone",
        "hostPolicyId": host,
        "donorPolicyId": donor,
        "leftPanelPolicyId": host,
        "rightPanelPolicyId": donor,
        "governanceVariant": governance,
        "interfaceRuleVariant": interface,
        "graftOutcomeClass": graft_outcome,
        "initialValuesJson": compact_json(values),
        "initialPanelPolicyIdsJson": compact_json(assignment),
        "mutantLineageCellIdsJson": compact_json(positions),
        "mutantInitialPositionsJson": compact_json(positions),
        "mutantCloneConfigJson": compact_json(mutant_clone_config_by_variant(behavior).to_dict()),
        "policyIdsJson": compact_json([host, donor]),
        "goalMode": "same_goal",
        "goalCompatibilityClass": "same",
        "finalStateHash": f"clone_hash_{index}",
        "finalValuesJson": compact_json(values),
        "finalPanelPolicyIdsJson": compact_json(assignment),
        "noMutantFinalStateHash": f"control_hash_{index}",
        "noMutantFinalValuesJson": compact_json(values),
        "noMutantFinalPanelPolicyIdsJson": compact_json([host] * 100),
        "schedulerSeed": 100 + index,
        "tieBreakerSeed": 200 + index,
        "n": 100,
        "runSucceeded": True,
        "takeoverProxyScore": 0.8 if "disruptive" in outcome else 0.1,
        "containmentScore": 0.2 if "disruptive" in outcome else 0.9,
        "deltaVsNoMutantFinalTargetQualityScore": -0.2 if "disruptive" in outcome else 0.0,
        "mutantChangedMosaicClass": "disruptive" in outcome,
        "broadControlLike": governance == "organizer_global_upper_bound",
        "metricBoundary": "unit test computational proxy boundary",
        "ratioLabel": "97:3",
        "pairCategory": "unit",
        "valueProfile": "reversed_unique",
        "perturbationType": "none",
        "seedIndex": 0,
    }


class TestE06DevelopmentalHistory(unittest.TestCase):
    def test_protocol_table_separates_reset_carryover_and_timing(self) -> None:
        protocols = history_protocol_table()
        self.assertIn(NO_MUTANT_HISTORY_PROTOCOL, set(protocols["historyProtocol"]))
        self.assertIn(SIMULTANEOUS_HISTORY_PROTOCOL, set(protocols["historyProtocol"]))
        self.assertTrue(protocols["memoryCarryoverMode"].str.contains("reset").any())
        self.assertTrue(protocols["memoryCarryoverMode"].str.contains("carryover").any())
        self.assertIn("early", set(protocols["historyTimingLabel"]))
        self.assertIn("late", set(protocols["historyTimingLabel"]))

    def test_select_contexts_uses_s11_outcomes_and_controls(self) -> None:
        rows = [
            _s11_clone_row(
                0,
                outcome="neutral_clone_reference",
                behavior="neutral_clone_control",
                governance="no_governance_control",
                interface="disabled_baseline",
                graft_outcome="absorbed_or_mixed",
            ),
            _s11_clone_row(
                1,
                outcome="disruptive_takeover_proxy",
                behavior="selfish_aggregation",
                governance="conflict_resolution_range1",
                interface="combined_self_boundary",
                graft_outcome="segregated_patch",
            ),
        ]
        selected = select_s12_contexts(pd.DataFrame(rows), max_contexts=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["mutantOutcomeClass"]), {"neutral_clone_reference", "disruptive_takeover_proxy"})
        self.assertTrue(selected["s12MatchedNoMutantConditionId"].astype(str).str.startswith("s11_control_").all())
        self.assertFalse(selected["memoryEligiblePolicyPresent"].any())

    def test_condition_matrix_records_protocol_targets_and_timelines(self) -> None:
        selected = select_s12_contexts(
            pd.DataFrame(
                [
                    _s11_clone_row(
                        0,
                        outcome="neutral_clone_reference",
                        behavior="neutral_clone_control",
                        governance="no_governance_control",
                        interface="disabled_baseline",
                        graft_outcome="absorbed_or_mixed",
                    )
                ]
            ),
            max_contexts=1,
        )
        conditions, timeline = build_developmental_history_condition_matrix(selected, total_activation_cap=3500)
        protocols = set(conditions["historyProtocol"])
        self.assertEqual(len(conditions), len(history_protocol_table()))
        self.assertIn(NO_MUTANT_HISTORY_PROTOCOL, protocols)
        self.assertIn(SIMULTANEOUS_HISTORY_PROTOCOL, protocols)
        self.assertTrue(any("staged_introduction" in value for value in protocols))
        self.assertTrue(any("transient_rule_release" in value for value in protocols))
        no_mutant = conditions[conditions["historyProtocol"] == NO_MUTANT_HISTORY_PROTOCOL].iloc[0]
        self.assertEqual(json.loads(str(no_mutant["postHistoryTargetCountsJson"])), {"host_policy": 100})
        self.assertEqual(int(no_mutant["mutantCloneSize"]), 0)
        staged = conditions[conditions["historyProtocol"] == "early_staged_introduction_reset"].iloc[0]
        self.assertEqual(json.loads(str(staged["actualCountsJson"])) if "actualCountsJson" in staged else {"not": "run"}, {"not": "run"})
        self.assertEqual(json.loads(str(staged["postHistoryTargetCountsJson"])), {"donor_policy": 3, "host_policy": 97})
        self.assertTrue(str(staged["initialPanelPolicyIdsJson"]).count("host_policy") >= 100)
        self.assertFalse(timeline.empty)
        self.assertTrue(timeline["plannedEvent"].astype(str).str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
