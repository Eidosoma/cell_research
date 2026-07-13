from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from reference_simulator import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    InvariantViolation,
    Policy,
    Scenario,
    audit_reference_result,
)
from reference_simulator.api import create_scenario, run_scenario
from shared_events import (
    HistoricalInvariantViolation,
    adapt_historical_run,
    audit_historical_trace,
)
from shared_events.model import bundle_with_events
from shared_events.schema import assign_event_id


S04_RUN = Path("/artifacts/research_steps/S04/smoke_outputs/generated_raw_smoke/run.json")


class ReferenceInvariantTests(unittest.TestCase):
    def test_all_policy_and_architecture_profiles_pass_independent_audit(self):
        for architecture in Architecture:
            for policy in Policy:
                with self.subTest(architecture=architecture.value, policy=policy.value):
                    scenario = create_scenario(
                        [4, 1, 3, 2],
                        architecture=architecture.value,
                        policy=policy.value,
                        generation_key=f"S07/unit/{architecture.value}/{policy.value}",
                        permute=False,
                        seed=707,
                        max_activations=500,
                    )
                    result = run_scenario(scenario, trace_mode="full")
                    self.assertTrue(audit_reference_result(result)["success"])

    def test_fault_duplicate_conflict_and_terminal_profiles_are_observed(self):
        scenarios = [
            create_scenario(
                [3, 1, 4, 2], policy="Bubble", faults={1: "stuck"},
                generation_key="S07/mutation/stuck", permute=False, seed=10,
                max_activations=100,
            ),
            create_scenario(
                [5, 4, 3, 2, 1], policy="Bubble", batch_width=4,
                generation_key="S07/unit/conflict", permute=False, seed=77,
                max_activations=200,
            ),
            create_scenario(
                [1, 1, 2], policy="Bubble", generation_key="S07/unit/duplicate",
                permute=False,
            ),
            Scenario.create(
                (
                    Cell("a", 1, Policy.BUBBLE, Direction.ASCENDING, FaultMode.NORMAL),
                    Cell("b", 2, Policy.BUBBLE, Direction.DESCENDING, FaultMode.NORMAL),
                ),
                initial_occupancy=("a", "b"), seed=5, max_activations=5,
                generation_key="S07/unit/mixed",
            ),
        ]
        coverage: dict[str, int] = {}
        stops = set()
        for scenario in scenarios:
            result = run_scenario(scenario, trace_mode="full")
            audit = audit_reference_result(result)
            stops.add(result.summary["stopReason"])
            for key, value in audit["coverage"].items():
                coverage[key] = coverage.get(key, 0) + value
        self.assertGreater(coverage.get("stuckTargetRejections", 0), 0)
        self.assertGreater(coverage.get("conflictLosses", 0), 0)
        self.assertGreater(coverage.get("duplicateScenarios", 0), 0)
        self.assertTrue({"complete", "quiescent", "event_budget"}.issubset(stops))

    def test_reference_audit_kills_position_hash_ledger_and_terminal_mutants(self):
        result = run_scenario(
            create_scenario(
                [4, 1, 3, 2], policy="Bubble", generation_key="S07/unit/mutants",
                permute=False, seed=23, max_activations=200,
            ),
            trace_mode="full",
        )
        mutations = []
        position = deepcopy(result.to_dict())
        position["events"][0]["proposal"]["actorPos"] = 999
        mutations.append(position)
        hash_mutant = deepcopy(result.to_dict())
        hash_mutant["events"][0]["preStateHash"] = "0" * 64
        mutations.append(hash_mutant)
        ledger = deepcopy(result.to_dict())
        ledger["events"][0]["ledgerDelta"]["acceptedSwaps"] = 1
        mutations.append(ledger)
        terminal = deepcopy(result.to_dict())
        terminal["summary"]["stopReason"] = "event_budget"
        mutations.append(terminal)
        for mutant in mutations:
            with self.subTest():
                with self.assertRaises(InvariantViolation):
                    audit_reference_result(mutant)


class HistoricalObservableInvariantTests(unittest.TestCase):
    def test_historical_audit_is_recorded_swap_only_and_detects_hash_tamper(self):
        raw = json.loads(S04_RUN.read_text(encoding="utf-8"))
        bundle = adapt_historical_run(raw, source_path=S04_RUN)
        summary = audit_historical_trace(bundle, raw)
        self.assertEqual(summary["sequenceBasis"], "recorded_swap")
        self.assertFalse(summary["activationParityClaimed"])

        events = [deepcopy(event) for event in bundle.events]
        events[0]["state"]["observablePreHash"] = "sha256:" + "0" * 64
        events[0] = assign_event_id(events[0])
        with self.assertRaises(HistoricalInvariantViolation):
            audit_historical_trace(bundle_with_events(bundle, events), raw)


if __name__ == "__main__":
    unittest.main()
