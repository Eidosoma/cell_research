#!/usr/bin/env python3
"""Run S05 acceptance gates and write compact validation/profile artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference_simulator import Architecture, Cell, Direction, FaultMode, Policy, Scenario
from reference_simulator.api import create_scenario, exact_replay, run_many, run_scenario
from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import ProposalKind, RunResult, canonical_json_bytes
from reference_simulator.policies import cell_view_proposal
from reference_simulator.rng import u64
from reference_simulator.scheduler import resolve_conflicts


S03_FIXTURES = Path("/artifacts/research_steps/S03/toy_fixtures.json")
SOURCE_PATHS = [
    *sorted((ROOT / "reference_simulator").glob("*.py")),
    *sorted((ROOT / "reference_simulator").glob("*.md")),
    ROOT / "scripts" / "validate_reference_backend.py",
    ROOT / "tests" / "test_reference_simulator.py",
]


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fixture_scenario(fixture: dict[str, Any]) -> Scenario:
    cells = []
    cursors = {}
    for item in fixture["pre"]:
        cells.append(
            Cell(
                item["id"], item["value"], Policy(item["policy"]),
                Direction(item["direction"]), FaultMode(item["fault"]),
            )
        )
        if "cursor" in item:
            cursors[item["id"]] = item["cursor"]
    return Scenario.create(
        cells,
        initial_occupancy=[item["id"] for item in fixture["pre"]],
        initial_selection_cursors=cursors,
        max_activations=fixture.get("terminal", {}).get("maxActivations", 10),
        generation_key="S03/" + fixture["fixtureId"],
    )


def validate_fixtures() -> list[dict[str, Any]]:
    source = json.loads(S03_FIXTURES.read_text())
    outcomes = []
    for fixture in source["fixtures"]:
        scenario = fixture_scenario(fixture)
        state = initial_state(scenario)
        fixture_id = fixture["fixtureId"]
        passed = False
        observed: dict[str, Any] = {}
        if fixture_id == "T14":
            proposals = []
            for ordinal, activation in enumerate(fixture["batch"]):
                proposal = cell_view_proposal(
                    scenario, state, activation["actorId"], side=activation["neighborChoice"]
                )
                proposals.append(replace(proposal, ordinal=ordinal, priority=activation["priority"]))
            accepted, lost = resolve_conflicts(proposals)
            winner = next(proposal for proposal in proposals if proposal.ordinal in accepted)
            state.occupancy[winner.actor_pos], state.occupancy[winner.target_pos] = (
                state.occupancy[winner.target_pos], state.occupancy[winner.actor_pos]
            )
            observed = {
                "winnerActorId": winner.actor_id,
                "loserActorIds": sorted(proposal.actor_id for proposal in proposals if proposal.ordinal in lost),
                "postOrder": state.occupancy,
            }
            passed = observed == fixture["expected"]
        elif "terminal" in fixture:
            state.activation_count = fixture["terminal"]["activationCount"]
            observed = {"stopReason": evaluate_terminal(scenario, state)}
            if fixture_id == "T15":
                values = [scenario.cell_map[item].value for item in state.occupancy]
                observed.update(
                    referenceSortedness=sum(a <= b for a, b in zip(values, values[1:])) / (len(values) - 1),
                    paperSortednessStrict=(1 + sum(a < b for a, b in zip(values, values[1:]))) / len(values),
                )
            passed = all(abs(observed[key] - value) < 1e-12 if isinstance(value, float) else observed[key] == value for key, value in fixture["expected"].items())
        else:
            activation = fixture["activation"]
            proposal = cell_view_proposal(
                scenario, state, activation["actorId"], side=activation.get("neighborChoice")
            )
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
            observed = {"proposal": proposal.kind.value, "postOrder": state.occupancy}
            if proposal.kind == ProposalKind.NO_OP:
                observed["reason"] = proposal.reason
            if "decision" in fixture["expected"]:
                observed["decision"] = decision
            if "newCursor" in fixture["expected"]:
                observed["newCursor"] = state.selection_cursors[proposal.actor_id]
            passed = observed == fixture["expected"]
        outcomes.append({"fixtureId": fixture_id, "passed": passed, "observed": observed})
    if not all(item["passed"] for item in outcomes):
        raise AssertionError("one or more S03 fixtures failed")
    return outcomes


def smoke_validation(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs = []
    results = []
    for architecture in Architecture:
        for policy in Policy:
            config_id = f"smoke-{architecture.value}-{policy.value}"
            scenario = create_scenario(
                [4, 1, 3, 2],
                policy=policy,
                architecture=architecture,
                seed=23,
                max_activations=50_000,
                generation_key=config_id,
                permute=False,
            )
            configs.append({
                "configId": config_id,
                "scenarioId": scenario.scenario_id,
                "architecture": architecture.value,
                "policy": policy.value,
                "seed": "23",
                "values": [4, 1, 3, 2],
                "maxActivations": 50_000,
                "traceMode": "full",
            })
            result = run_scenario(scenario, trace_mode="full")
            replay = exact_replay(result)
            roundtrip = RunResult.from_json_bytes(result.to_json_bytes())
            passed = (
                result.summary["stopReason"] == "complete"
                and replay.to_json_bytes() == result.to_json_bytes()
                and roundtrip.to_json_bytes() == result.to_json_bytes()
                and Scenario.from_json_bytes(scenario.to_json_bytes()) == scenario
            )
            results.append({
                "configId": config_id,
                "passed": passed,
                "stopReason": result.summary["stopReason"],
                "activationCount": result.summary["activationCount"],
                "eventCount": len(result.events),
                "eventDigest": result.event_digest,
                "finalStateHash": result.final_state_hash,
                "exactReplay": True,
                "scenarioRoundtrip": True,
                "resultRoundtrip": True,
            })
            if not passed:
                raise AssertionError(f"smoke failed: {config_id}")
            if config_id == "smoke-cell_view-Bubble":
                write_json(output / "smoke_sample_scenario.json", scenario.to_dict())
                write_json(output / "smoke_sample_result.json", result.to_dict())
    return configs, results


def parallel_validation() -> dict[str, Any]:
    scenarios = [
        create_scenario(
            list(range(8)), policy=Policy.BUBBLE, seed=index,
            generation_key=f"parallel/{index}", max_activations=100_000,
        )
        for index in range(16)
    ]
    serial = [item.to_json_bytes() for item in run_many(scenarios, workers=1)]
    four = [item.to_json_bytes() for item in run_many(scenarios, workers=4)]
    eight = [item.to_json_bytes() for item in run_many(scenarios, workers=8)]
    passed = serial == four == eight
    if not passed:
        raise AssertionError("replicate outputs changed with worker count")
    return {
        "passed": passed,
        "replicates": len(scenarios),
        "workerCounts": [1, 4, 8],
        "orderedAggregateSha256": hashlib.sha256(b"".join(serial)).hexdigest(),
    }


def profile_backend() -> list[dict[str, Any]]:
    rows = []
    for n in (20, 100):
        for architecture in Architecture:
            for policy in Policy:
                config_id = f"profile-n{n}-{architecture.value}-{policy.value}"
                scenario = create_scenario(
                    list(range(n)),
                    policy=policy,
                    architecture=architecture,
                    seed=101,
                    generation_key=config_id,
                    max_activations=2_000_000,
                )
                started = perf_counter()
                result = run_scenario(scenario, trace_mode="digest")
                elapsed = perf_counter() - started
                if result.summary["stopReason"] != "complete":
                    raise AssertionError(f"profiling scenario did not complete: {config_id}")
                rows.append({
                    "configId": config_id,
                    "n": n,
                    "architecture": architecture.value,
                    "policy": policy.value,
                    "seed": 101,
                    "traceMode": "digest",
                    "stopReason": result.summary["stopReason"],
                    "activationCount": result.summary["activationCount"],
                    "acceptedSwaps": result.summary["ledger"]["acceptedSwaps"],
                    "valueComparisons": result.summary["ledger"]["valueComparisons"],
                    "elapsedSeconds": round(elapsed, 6),
                    "activationsPerSecond": round(result.summary["activationCount"] / elapsed, 3),
                    "eventDigest": result.event_digest,
                    "finalStateHash": result.final_state_hash,
                })
                print(
                    f"profile {config_id}: {result.summary['activationCount']} activations in {elapsed:.3f}s",
                    flush=True,
                )
    return rows


def write_benchmark_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def api_documentation() -> str:
    return """# Reference simulator API (S05)\n\n## Contract\n\nThe repository package `reference_simulator` implements `E01-reference-v1`. It is a clean-room reference backend and imports no frozen-public-commit source. The conventional `traditional` controllers are separately named R controls, not reconstructed publication code. Events are marked `pre-S06`; S06 has not begun.\n\n## Public objects\n\n- `create_scenario(values, policy=..., architecture=..., generation_key=..., seed=...) -> Scenario` creates homogeneous scenarios. `generation_key` is mandatory and keys the pre-ID permutation; the final post-permutation content determines `scenarioId`.\n- `run_scenario(scenario, trace_mode='full'|'digest'|'none') -> RunResult` executes terminal precedence `invariant_error > complete > quiescent > event_budget`.\n- `exact_replay(result) -> RunResult` reruns and requires byte-identical stable result JSON.\n- `run_many(scenarios, workers=1..8, trace_mode=...) -> list[RunResult]` parallelizes replicates only and preserves input order.\n- `Scenario.to_json_bytes()` / `Scenario.from_json_bytes()` use canonical UTF-8 JSON and validate the content-derived ID.\n- `RunResult.to_json_bytes()` / `RunResult.from_json_bytes()` provide stable result round trips.\n\n## Counter streams\n\nRuntime draws use SHA-256 addresses `(uint128 seed, final scenarioId, ASCII stream, uint64 eventIndex, uint32 drawIndex)`. Required streams are `actor_activation`, `bubble_side`, `conflict_priority`, and `scenario_permutation`; unbiased bounded integers use rejection sampling. Replicate worker count never enters an address.\n\n## Selection decision\n\nS03's target-state ambiguity is preserved explicitly: `target.value <= actor.value` advances the identity-owned cursor in both directions; otherwise the actor swaps into the cursor. Descending changes cursor start/advance only. This is intentionally not replaced with a conventional descending comparator.\n\n## Command line\n\n`python -m reference_simulator.cli SCENARIO.json --trace-mode full --output RESULT.json`\n"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    started_at = datetime.now(timezone.utc).isoformat()

    fixture_results = validate_fixtures()
    write_json(output / "toy_fixture_results.json", {
        "schema": "E01.S05.toy-fixture-results.v1",
        "source": str(S03_FIXTURES),
        "sourceSha256": sha256(S03_FIXTURES),
        "passed": sum(item["passed"] for item in fixture_results),
        "total": len(fixture_results),
        "fixtures": fixture_results,
    })
    known_rng = u64(7, "scenario-A", "actor_activation", 3, 0)
    if known_rng != 2584083456120664033:
        raise AssertionError("counter RNG known vector failed")
    smoke_configs, smoke_results = smoke_validation(output)
    write_json(output / "smoke_configurations.json", {
        "schema": "E01.S05.smoke-configurations.v1", "configurations": smoke_configs,
    })
    write_json(output / "smoke_results.json", {
        "schema": "E01.S05.smoke-results.v1", "results": smoke_results,
    })
    parallel = parallel_validation()
    benchmarks = profile_backend()
    write_benchmark_csv(output / "benchmark.csv", benchmarks)
    write_json(output / "benchmark_summary.json", {
        "schema": "E01.S05.benchmark.v1",
        "clock": "time.perf_counter",
        "replicatesPerCondition": 1,
        "conditions": 12,
        "rows": benchmarks,
        "interpretation": "Single-run environment-specific engineering profile, not inferential evidence.",
    })
    (output / "api_documentation.md").write_text(api_documentation())
    write_json(output / "semantic_decisions.json", {
        "schema": "E01.S05.semantic-decisions.v1",
        "researchStepId": "S05",
        "decisions": [
            {
                "id": "S05-D01",
                "topic": "Selection target-state ambiguity",
                "decision": "Preserve S03 exactly: target.value <= actor.value advances in both directions; direction only changes cursor start and delta.",
                "status": "frozen_contract_preserved",
            },
            {
                "id": "S05-D02",
                "topic": "Scenario-permutation self-reference",
                "decision": "Use mandatory stable generation_key for pre-ID permutation, then derive final scenarioId from post-permutation canonical content; runtime streams use final scenarioId.",
                "status": "explicit_clean_room_elaboration",
            },
            {
                "id": "S05-D03",
                "topic": "Traditional controller source gap",
                "decision": "Implement separately named traditional_global_primary_action_v1 controllers; passive targets move, swaps touching stuck cells reject; do not attribute this behavior to P or C.",
                "status": "clean_room_control_only",
            },
            {
                "id": "S05-D04",
                "topic": "Event schema boundary",
                "decision": "Emit minimum S03 semantic events labeled pre-S06; do not claim the shared cross-backend schema is complete.",
                "status": "boundary_preserved",
            },
        ],
    })
    commit = git("rev-parse", "HEAD")
    status = git("status", "--short")
    source_files = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in SOURCE_PATHS
    ]
    write_json(output / "source_package_manifest.json", {
        "schema": "E01.S05.source-package.v1",
        "repository": "https://github.com/Eidosoma/cell_research",
        "branch": "eidosoma/groups/28",
        "commitAtValidation": commit,
        "workingTreeDirtyAtValidation": bool(status),
        "packageRoot": "reference_simulator/",
        "sourceFiles": source_files,
        "sourceArchiveCreated": False,
        "sourceArchiveReason": "The stricter GitHub-workspace rule prohibits copying repository source into ARTIFACTS_DIR; this manifest and the release Git pointer replace the planned tar.zst duplicate.",
        "historicalSourceIncluded": False,
    })
    write_json(output / "environment_provenance.json", {
        "schema": "E01.S05.environment.v1",
        "startedAtUtc": started_at,
        "finishedAtUtc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logicalCpuCount": os.cpu_count(),
        "maximumPermittedWorkers": 8,
        "parallelismValidated": parallel,
        "threadEnvironment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "dependencies": "Python standard library only",
        "gitCommitAtValidation": commit,
        "gitStatusAtValidation": status.splitlines(),
    })
    write_json(output / "validation_summary.json", {
        "schema": "E01.S05.validation.v1",
        "researchStepId": "S05",
        "success": True,
        "checks": {
            "s03ToyFixtures": {"passed": 17, "total": 17},
            "counterRngKnownVector": {"passed": True, "observed": known_rng},
            "sixPolicySmoke": {"passed": 6, "total": 6},
            "exactReplay": {"passed": 6, "total": 6},
            "scenarioSerialization": {"passed": 6, "total": 6},
            "resultSerialization": {"passed": 6, "total": 6},
            "replicateParallelism": parallel,
            "profiling": {"passed": len(benchmarks), "total": 12, "n": [20, 100]},
            "selectionDescendingContract": "covered by T12 plus repository test",
        },
        "caveats": [
            "Traditional controls are new R specifications, not recovered publication behavior.",
            "The unusual direction-invariant Selection numeric comparison is preserved from S03.",
            "Benchmarks are single-run engineering profiles and environment-specific.",
            "Events are pre-S06 and not yet the shared cross-backend schema.",
        ],
    })
    print(f"wrote S05 validation outputs to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
