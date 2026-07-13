from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from reference_simulator import Architecture, Cell, Direction, FaultMode, Policy, Scenario
from reference_simulator.api import create_scenario, exact_replay, run_many, run_scenario
from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import ProposalKind, RunResult
from reference_simulator.policies import cell_view_proposal
from reference_simulator.rng import u64
from reference_simulator.scheduler import resolve_conflicts


S03_FIXTURES = Path("/artifacts/research_steps/S03/toy_fixtures.json")


def fixture_scenario(fixture: dict, *, max_activations: int = 10) -> Scenario:
    cells = []
    cursors = {}
    for item in fixture["pre"]:
        cells.append(
            Cell(
                item["id"],
                item["value"],
                Policy(item["policy"]),
                Direction(item["direction"]),
                FaultMode(item["fault"]),
            )
        )
        if "cursor" in item:
            cursors[item["id"]] = item["cursor"]
    return Scenario.create(
        cells,
        initial_occupancy=[item["id"] for item in fixture["pre"]],
        initial_selection_cursors=cursors,
        max_activations=max_activations,
        generation_key="S03/" + fixture["fixtureId"],
    )


def apply_fixture_proposal(scenario: Scenario, state, proposal):
    decision = "no_op"
    if proposal.kind == ProposalKind.MEMORY_UPDATE:
        state.selection_cursors[proposal.actor_id] = proposal.new_cursor
        decision = "accepted"
    elif proposal.kind == ProposalKind.SWAP:
        target = scenario.cell_map[state.occupancy[proposal.target_pos]]
        if target.fault == FaultMode.STUCK:
            decision = "rejected_target_stuck"
        else:
            state.occupancy[proposal.actor_pos], state.occupancy[proposal.target_pos] = (
                state.occupancy[proposal.target_pos], state.occupancy[proposal.actor_pos]
            )
            decision = "accepted"
    return decision


class S03FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(S03_FIXTURES.read_text())["fixtures"]

    def test_all_17_frozen_fixtures(self):
        for fixture in self.fixtures:
            with self.subTest(fixture=fixture["fixtureId"]):
                scenario = fixture_scenario(
                    fixture,
                    max_activations=fixture.get("terminal", {}).get("maxActivations", 10),
                )
                state = initial_state(scenario)
                if fixture["fixtureId"] == "T14":
                    proposals = []
                    for ordinal, activation in enumerate(fixture["batch"]):
                        proposal = cell_view_proposal(
                            scenario, state, activation["actorId"], side=activation["neighborChoice"]
                        )
                        from dataclasses import replace
                        proposals.append(replace(proposal, ordinal=ordinal, priority=activation["priority"]))
                    accepted, lost = resolve_conflicts(proposals)
                    self.assertEqual(accepted, {1})
                    self.assertEqual(lost, {0})
                    apply_fixture_proposal(scenario, state, proposals[1])
                    self.assertEqual(state.occupancy, fixture["expected"]["postOrder"])
                    continue
                if "terminal" in fixture:
                    state.activation_count = fixture["terminal"]["activationCount"]
                    self.assertEqual(evaluate_terminal(scenario, state), fixture["expected"]["stopReason"])
                    continue
                activation = fixture["activation"]
                proposal = cell_view_proposal(
                    scenario,
                    state,
                    activation["actorId"],
                    side=activation.get("neighborChoice"),
                )
                self.assertEqual(proposal.kind.value, fixture["expected"]["proposal"])
                if "reason" in fixture["expected"]:
                    self.assertEqual(proposal.reason, fixture["expected"]["reason"])
                decision = apply_fixture_proposal(scenario, state, proposal)
                if "decision" in fixture["expected"]:
                    self.assertEqual(decision, fixture["expected"]["decision"])
                if "newCursor" in fixture["expected"]:
                    self.assertEqual(state.selection_cursors[activation["actorId"]], fixture["expected"]["newCursor"])
                self.assertEqual(state.occupancy, fixture["expected"]["postOrder"])

    def test_descending_selection_numeric_comparison_is_not_reversed(self):
        scenario = Scenario.create(
            [
                Cell("actor", 5, Policy.SELECTION, Direction.DESCENDING),
                Cell("target", 4, Policy.BUBBLE, Direction.DESCENDING),
            ],
            initial_occupancy=("actor", "target"),
            initial_selection_cursors={"actor": 1},
            generation_key="selection-descending-contract",
        )
        proposal = cell_view_proposal(scenario, initial_state(scenario), "actor")
        self.assertEqual(proposal.kind, ProposalKind.MEMORY_UPDATE)
        self.assertEqual(proposal.new_cursor, 0)


class DeterminismTests(unittest.TestCase):
    def test_legacy_with_replacement_profile_preserves_requested_and_realized_counts(self):
        scenario = Scenario.create(
            (
                Cell("a", 1, Policy.BUBBLE, fault=FaultMode.PASSIVE),
                Cell("b", 2, Policy.BUBBLE),
            ),
            requested_fault_count=3,
            fault_placement="legacy_with_replacement",
            generation_key="legacy-placement-test",
        )
        self.assertEqual(scenario.requested_fault_count, 3)
        self.assertEqual(scenario.realized_fault_count, 1)
        self.assertEqual(Scenario.from_dict(scenario.to_dict()), scenario)

    def test_rng_known_vector(self):
        self.assertEqual(u64(7, "scenario-A", "actor_activation", 3, 0), 2584083456120664033)

    def test_scenario_stable_roundtrip_and_tamper_rejection(self):
        scenario = create_scenario(
            [3, 1, 2], policy="Bubble", generation_key="serialization", seed=19
        )
        self.assertEqual(Scenario.from_json_bytes(scenario.to_json_bytes()), scenario)
        decoded = json.loads(scenario.to_json_bytes())
        decoded["seed"] = "20"
        with self.assertRaisesRegex(ValueError, "scenario ID"):
            Scenario.from_dict(decoded)

    def test_all_six_policies_replay_and_result_roundtrip(self):
        for architecture in Architecture:
            for policy in Policy:
                with self.subTest(architecture=architecture, policy=policy):
                    scenario = create_scenario(
                        [4, 1, 3, 2],
                        policy=policy,
                        architecture=architecture,
                        seed=23,
                        max_activations=50_000,
                        generation_key=f"six/{architecture.value}/{policy.value}",
                        permute=False,
                    )
                    result = run_scenario(scenario, trace_mode="full")
                    self.assertEqual(result.summary["stopReason"], "complete")
                    exact_replay(result)
                    self.assertEqual(
                        RunResult.from_json_bytes(result.to_json_bytes()).to_json_bytes(),
                        result.to_json_bytes(),
                    )

    def test_replicate_parallelism_is_byte_identical(self):
        scenarios = [
            create_scenario(
                list(range(8)),
                policy=Policy.BUBBLE,
                seed=index,
                generation_key=f"parallel/{index}",
            )
            for index in range(8)
        ]
        serial = [result.to_json_bytes() for result in run_many(scenarios, workers=1)]
        parallel = [result.to_json_bytes() for result in run_many(scenarios, workers=4)]
        self.assertEqual(parallel, serial)

    def test_worker_guard(self):
        with self.assertRaisesRegex(ValueError, r"\[1, 8\]"):
            run_many([], workers=9)

    def test_reference_fault_sampling_is_exact_and_reproducible(self):
        first = create_scenario(
            list(range(20)), policy=Policy.BUBBLE, seed=71,
            generation_key="faults", fault_count=7, fault_mode=FaultMode.STUCK,
        )
        second = create_scenario(
            list(range(20)), policy=Policy.BUBBLE, seed=71,
            generation_key="faults", fault_count=7, fault_mode=FaultMode.STUCK,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.requested_fault_count, 7)
        self.assertEqual(first.realized_fault_count, 7)
        self.assertEqual(sum(cell.fault == FaultMode.STUCK for cell in first.cells), 7)

    def test_batch_conflicts_are_reproducible(self):
        scenario = create_scenario(
            [5, 4, 3, 2, 1],
            policy=Policy.BUBBLE,
            seed=77,
            generation_key="batch",
            permute=False,
            batch_width=4,
            max_activations=100_000,
        )
        first = run_scenario(scenario, trace_mode="full")
        second = run_scenario(scenario, trace_mode="full")
        self.assertEqual(first.to_json_bytes(), second.to_json_bytes())
        self.assertGreaterEqual(first.summary["ledger"]["conflictLosses"], 0)


if __name__ == "__main__":
    unittest.main()
