#!/usr/bin/env python3
"""Build and validate the E05 S03 lesion-operator library."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract  # noqa: E402
from causal_simulator.schedulers import (  # noqa: E402
    SchedulerExecutionContract,
    SchedulerFamily,
    run_scheduled_architecture,
)
from reference_simulator.model import (  # noqa: E402
    Direction,
    Policy,
    canonical_json_bytes,
    sha256_json,
)
from src.regeneration.lesions import (  # noqa: E402
    BENCHMARK_VERSION,
    LESION_FIXTURE_SCHEMA,
    LESION_SPEC_SCHEMA,
    OPERATOR_IDS,
    LesionState,
    apply_lesion,
    validate_application,
    validate_lesion_spec,
    validate_pairing_rows,
)
from src.regeneration.tasks import (  # noqa: E402
    _hash,
    _seed,
    initial_checkpoint,
    replay_checkpoints,
    stabilize_achieved_checkpoint,
)
from src.regeneration.timing import (  # noqa: E402
    _prefix_digest,
    _s01_pairing_id,
    _source_scenario,
    locate_trigger,
)


CONFIG = REPOSITORY / "configs/regeneration/s03_lesions.json"
S02_CONFIG = REPOSITORY / "configs/regeneration/s02_timing.json"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
INPUTS: tuple[Path, ...] = (
    Path("/workspace/AGENTS.md"),
    Path("/workspace/FULL_PLAN.md"),
    Path("/workspace/RESEARCH_PLAN.md"),
    Path("/workspace/input-attachments/MANIFEST.json"),
    S01_DIR / "research_step_full_results.md",
    S01_DIR / "task_spec.md",
    S01_DIR / "task_spec.json",
    S01_DIR / "baseline_scenarios.parquet",
    S01_DIR / "validation_summary.json",
    S02_DIR / "research_step_full_results.md",
    S02_DIR / "timing_spec.md",
    S02_DIR / "timing_spec.json",
    S02_DIR / "preinjury_states.parquet",
    S02_DIR / "damage_timing_scenarios.parquet",
    S02_DIR / "validation_summary.json",
    Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
    Path("/previous-artifacts/E02/release/causal_simulator_extension/release_manifest.json"),
    Path("/previous-artifacts/E02/research_steps/S04/scheduler_package/scheduler_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_prespecification.json"),
    Path("/previous-artifacts/E02/research_steps/S05/fault_package/fault_semantics_contract.md"),
    Path("/previous-artifacts/E02/research_steps/S08/semantic_random_stream_specification.json"),
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True
    ).strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs() -> dict[str, Any]:
    missing = [str(path) for path in INPUTS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required S01/S02/E01/E02 inputs: {missing}")
    s01 = _load_json(S01_DIR / "validation_summary.json")
    s02 = _load_json(S02_DIR / "validation_summary.json")
    e01 = _load_json(INPUTS[-6])
    e02 = _load_json(INPUTS[-5])
    if not s01.get("success") or not s02.get("success"):
        raise RuntimeError("S01 and S02 validation gates must both pass")
    if not e01.get("validationSuccess"):
        raise RuntimeError("E01 release validation gate did not pass")
    if not e02.get("smokeValidation", {}).get("success"):
        raise RuntimeError("E02 release smoke gate did not pass")
    return {"s01": s01, "s02": s02, "e01": e01, "e02": e02}


def _json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _scalar_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                clean[key + "Json"] = json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            else:
                clean[key] = value
        normalized.append(clean)
    return pd.DataFrame(normalized)


def _checkpoint_expectations() -> dict[tuple[str, str], dict[str, Any]]:
    frame = pd.read_parquet(S02_DIR / "preinjury_states.parquet")
    return {
        (str(row["s01PairingBlockId"]), str(row["timingConditionId"])): row
        for row in frame.to_dict("records")
    }


def _source_expectations() -> tuple[dict[str, str], dict[str, str]]:
    s01 = pd.read_parquet(
        S01_DIR / "baseline_scenarios.parquet",
        columns=["pairingBlockId", "executableScenarioId"],
    ).drop_duplicates()
    s02 = pd.read_parquet(
        S02_DIR / "damage_timing_scenarios.parquet",
        columns=["s01PairingBlockId", "executableScenarioId"],
    ).drop_duplicates()
    return (
        dict(zip(s01["pairingBlockId"], s01["executableScenarioId"], strict=True)),
        dict(
            zip(
                s02["s01PairingBlockId"],
                s02["executableScenarioId"],
                strict=True,
            )
        ),
    )


def _fixture_row(
    application: Any,
    *,
    n: int,
    policy: Policy,
    direction: Direction,
    replicate: int,
    clock: str,
    nominal_fraction: float,
    budget: int,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    fixture = application.to_dict()
    return {
        "schemaVersion": fixture["schemaVersion"],
        "benchmarkVersion": fixture["benchmarkVersion"],
        "lesionCaseId": fixture["lesionCaseId"],
        "lesionPairId": fixture["lesionPairId"],
        "operatorId": fixture["operatorId"],
        "family": fixture["family"],
        "sourceScenarioId": fixture["sourceScenarioId"],
        "s01PairingBlockId": fixture["s01PairingBlockId"],
        "timingConditionId": fixture["timingConditionId"],
        "clock": clock,
        "nominalFraction": nominal_fraction,
        "n": n,
        "policy": policy.value,
        "direction": direction.value,
        "replicateOrdinal": replicate,
        "developmentBudget": budget,
        "recoveryBudget": budget,
        "eventBudgetProfile": "profile_scaled_frozen_s07_v1",
        "preInjuryStateHash": fixture["preInjuryStateHash"],
        "preLesionStateHash": fixture["preLesionStateHash"],
        "postLesionStateHash": fixture["postLesionStateHash"],
        "activationCount": application.pre_state.activation_count,
        "preIdentityCount": len(application.pre_state.occupancy),
        "postIdentityCount": len(application.post_state.occupancy),
        "identityContract": fixture["identityContract"],
        "targetCorrespondence": fixture["targetCorrespondence"],
        "severity": fixture["severity"],
        "reversibility": fixture["reversibility"],
        "runtimeCompatibility": fixture["runtimeCompatibility"],
        "parameters": fixture["parameters"],
        "tombstone": fixture.get("tombstone"),
        "validation": dict(audit),
    }


def _paired_rows(
    application: Any,
    fixture: Mapping[str, Any],
) -> list[dict[str, Any]]:
    pre = application.pre_state
    common = {
        "schemaVersion": "e05.s03.lesion-scenario.v1",
        "benchmarkVersion": BENCHMARK_VERSION,
        "lesionPairId": application.lesion_pair_id,
        "operatorId": application.operator_id,
        "family": application.family,
        "sourceScenarioId": pre.source_scenario_id,
        "s01PairingBlockId": application.s01_pairing_block_id,
        "timingConditionId": application.timing_condition_id,
        "clock": fixture["clock"],
        "nominalFraction": fixture["nominalFraction"],
        "n": fixture["n"],
        "policy": fixture["policy"],
        "direction": fixture["direction"],
        "replicateOrdinal": fixture["replicateOrdinal"],
        "preInjuryStateHash": pre.source_checkpoint_hash,
        "preLesionStateHash": pre.state_hash,
        "activationCount": pre.activation_count,
        "streamCountersSha256": _json_digest(dict(pre.stream_counters)),
        "ledgerSha256": _json_digest(dict(pre.ledger)),
        "developmentBudget": fixture["developmentBudget"],
        "recoveryBudget": fixture["recoveryBudget"],
        "eventBudgetProfile": fixture["eventBudgetProfile"],
    }
    rows: list[dict[str, Any]] = []
    for arm, post_hash, identity_count, pairing_status in (
        ("matched_sham", pre.state_hash, len(pre.occupancy), "exact_checkpoint_sham"),
        (
            "active_lesion",
            application.post_state.state_hash,
            len(application.post_state.occupancy),
            application.runtime_compatibility["postInjuryPairingStatus"],
        ),
    ):
        content = {
            "lesionPairId": application.lesion_pair_id,
            "arm": arm,
            "postLesionStateHash": post_hash,
        }
        rows.append(
            {
                **common,
                "lesionScenarioId": "e05s03:" + sha256_json(content),
                "arm": arm,
                "postLesionStateHash": post_hash,
                "postIdentityCount": identity_count,
                "postInjuryPairingStatus": pairing_status,
            }
        )
    return rows


def build_panel(specification: Mapping[str, Any]) -> dict[str, Any]:
    validate_lesion_spec(specification)
    timing_spec = _load_json(S02_CONFIG)
    panel = specification["validationPanel"]
    architecture = ArchitectureExecutionContract.distributed_local()
    scheduler = SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION)
    expected_checkpoints = _checkpoint_expectations()
    expected_s01_sources, expected_s02_sources = _source_expectations()
    fixtures: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    reverse_rows: list[dict[str, Any]] = []
    selected_states: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    application_failures: list[str] = []
    formation_replay_failures: list[str] = []
    operator_replay_failures: list[str] = []
    formation_replays = 0
    operator_replays = 0

    for n in panel["sizes"]:
        for policy_name in panel["policies"]:
            policy = Policy(policy_name)
            for direction_name in panel["directions"]:
                direction = Direction(direction_name)
                for replicate in range(panel["replicatesPerCell"]):
                    scenario, _, budget = _source_scenario(
                        n, policy, direction, replicate
                    )
                    pairing_id = _s01_pairing_id(
                        scenario,
                        n=n,
                        policy=policy,
                        direction=direction,
                        replicate=replicate,
                        budget=budget,
                    )
                    source_rows.append(
                        {
                            "s01PairingBlockId": pairing_id,
                            "sourceScenarioId": scenario.scenario_id,
                            "s01SourceExpected": expected_s01_sources.get(pairing_id),
                            "s02SourceExpected": expected_s02_sources.get(pairing_id),
                            "s01IdentityPass": expected_s01_sources.get(pairing_id)
                            == scenario.scenario_id,
                            "s02IdentityPass": expected_s02_sources.get(pairing_id)
                            == scenario.scenario_id,
                        }
                    )
                    development = run_scheduled_architecture(
                        scenario, architecture, scheduler, trace_mode="full"
                    ).result
                    replayed = run_scheduled_architecture(
                        scenario, architecture, scheduler, trace_mode="full"
                    ).result
                    formation_replays += 1
                    if development.to_json_bytes() != replayed.to_json_bytes():
                        formation_replay_failures.append(scenario.scenario_id)
                    phase_events = tuple(development.events[:budget])
                    checkpoints = replay_checkpoints(scenario, phase_events)
                    initial = initial_checkpoint(scenario)
                    completed = bool(development.summary["completed"]) and int(
                        development.summary["activationCount"]
                    ) <= budget
                    completion_events = (
                        int(development.summary["activationCount"])
                        if completed
                        else None
                    )
                    development_stop = (
                        str(development.summary["stopReason"])
                        if int(development.summary["activationCount"]) <= budget
                        else "phase_event_budget"
                    )
                    stabilized = None
                    stabilization_audit = None
                    if completed and completion_events is not None:
                        stabilized, stabilization_audit = stabilize_achieved_checkpoint(
                            scenario, checkpoints[completion_events - 1]
                        )
                        if not stabilization_audit["success"]:
                            stabilized = None
                    injury_seed = _seed("injury", pairing_id)
                    for condition in timing_spec["timingConditions"]:
                        trigger = locate_trigger(
                            condition,
                            initial=initial,
                            development_checkpoints=checkpoints,
                            development_stop_reason=development_stop,
                            completion_events=completion_events,
                            stabilized=stabilized,
                        )
                        if trigger.status != "triggered" or trigger.checkpoint is None:
                            raise RuntimeError(
                                f"validated S02 checkpoint became unreachable: "
                                f"{pairing_id}/{trigger.condition_id}/{trigger.status}"
                            )
                        checkpoint = trigger.checkpoint
                        expected = expected_checkpoints.get(
                            (pairing_id, trigger.condition_id)
                        )
                        if expected is None:
                            raise RuntimeError("S02 checkpoint row is missing")
                        selection = dict(checkpoint.selection_cursors)
                        streams = dict(checkpoint.stream_counters)
                        ledger = dict(checkpoint.ledger)
                        actual_internal_hash = _hash(
                            {
                                "selectionCursors": selection,
                                "streamCounters": streams,
                                "ledger": ledger,
                            }
                        )
                        actual_prefix = _prefix_digest(
                            phase_events,
                            min(checkpoint.activation_count, len(phase_events)),
                        )
                        checks = {
                            "stateHashPass": checkpoint.state_hash
                            == expected["stateHash"],
                            "occupancyHashPass": _hash(list(checkpoint.occupancy))
                            == expected["occupancyHash"],
                            "internalStateHashPass": actual_internal_hash
                            == expected["internalStateHash"],
                            "prefixDigestPass": actual_prefix
                            == expected["prefixDigest"],
                            "activationCountPass": checkpoint.activation_count
                            == int(expected["activationCount"]),
                            "distancePass": checkpoint.distance
                            == int(expected["distance"]),
                            "selectionCursorsPass": selection
                            == json.loads(expected["selectionCursorsJson"]),
                            "streamCountersPass": streams
                            == json.loads(expected["streamCountersJson"]),
                            "ledgerPass": ledger == json.loads(expected["ledgerJson"]),
                        }
                        checkpoint_rows.append(
                            {
                                "s01PairingBlockId": pairing_id,
                                "timingConditionId": trigger.condition_id,
                                "clock": trigger.clock.value,
                                "n": n,
                                "policy": policy.value,
                                "direction": direction.value,
                                "replicateOrdinal": replicate,
                                "activationCount": checkpoint.activation_count,
                                "stateHash": checkpoint.state_hash,
                                **checks,
                                "success": all(checks.values()),
                            }
                        )
                        state = LesionState.from_checkpoint(scenario, checkpoint)
                        for operator_id in OPERATOR_IDS:
                            application = apply_lesion(
                                operator_id,
                                state,
                                s01_pairing_block_id=pairing_id,
                                timing_condition_id=trigger.condition_id,
                                injury_seed=injury_seed,
                            )
                            replay = apply_lesion(
                                operator_id,
                                state,
                                s01_pairing_block_id=pairing_id,
                                timing_condition_id=trigger.condition_id,
                                injury_seed=injury_seed,
                            )
                            operator_replays += 1
                            if application.to_dict() != replay.to_dict():
                                operator_replay_failures.append(
                                    f"{pairing_id}/{trigger.condition_id}/{operator_id}"
                                )
                            audit = validate_application(application)
                            if not audit["success"]:
                                application_failures.append(
                                    f"{pairing_id}/{trigger.condition_id}/{operator_id}"
                                )
                            fixture = _fixture_row(
                                application,
                                n=n,
                                policy=policy,
                                direction=direction,
                                replicate=replicate,
                                clock=trigger.clock.value,
                                nominal_fraction=trigger.nominal_fraction,
                                budget=budget,
                                audit=audit,
                            )
                            fixtures.append(fixture)
                            paired_rows.extend(_paired_rows(application, fixture))
                            reverse_rows.append(
                                {
                                    "lesionCaseId": application.lesion_case_id,
                                    "operatorId": operator_id,
                                    "n": n,
                                    "policy": policy.value,
                                    "direction": direction.value,
                                    "timingConditionId": trigger.condition_id,
                                    "reversibilityClass": application.reversibility[
                                        "class"
                                    ],
                                    "inverseExact": audit["checks"]["inverseExact"],
                                    "tombstoneRequired": application.reversibility[
                                        "exactStateRestorationRequiresTombstone"
                                    ],
                                }
                            )
                            if (
                                n == 20
                                and policy == Policy.BUBBLE
                                and replicate == 0
                                and trigger.condition_id == "post_completion"
                            ):
                                selected_states.append(
                                    {
                                        "fixture": application.to_dict(),
                                        "preState": application.pre_state.to_dict(),
                                        "postState": application.post_state.to_dict(),
                                    }
                                )

    pairing = validate_pairing_rows(paired_rows)
    fixture_count = len(fixtures)
    checkpoint_count = len(checkpoint_rows)
    expected_blocks = (
        len(panel["sizes"])
        * len(panel["policies"])
        * len(panel["directions"])
        * panel["replicatesPerCell"]
    )
    expected_checkpoints_count = expected_blocks * panel["timingConditionsPerBlock"]
    expected_fixtures = expected_checkpoints_count * panel["operatorsPerCheckpoint"]
    run_accounting = {
        "schemaVersion": "e05.s03.run-accounting.v1",
        "sourceBlockExpected": expected_blocks,
        "sourceBlockObserved": len(source_rows),
        "checkpointExpected": expected_checkpoints_count,
        "checkpointObserved": checkpoint_count,
        "activeFixtureExpected": expected_fixtures,
        "activeFixtureObserved": fixture_count,
        "pairedRowExpected": 2 * expected_fixtures,
        "pairedRowObserved": len(paired_rows),
        "operatorCounts": {
            operator: sum(item["operatorId"] == operator for item in fixtures)
            for operator in OPERATOR_IDS
        },
        "timingConditionCounts": {
            condition: sum(
                item["timingConditionId"] == condition for item in fixtures
            )
            for condition in specification["inherits"]["timingConditions"]
        },
        "unreachableCheckpointCount": expected_checkpoints_count - checkpoint_count,
        "substitutionCount": 0,
        "silentExclusionCount": expected_fixtures - fixture_count,
    }
    run_accounting["success"] = (
        run_accounting["sourceBlockExpected"]
        == run_accounting["sourceBlockObserved"]
        and run_accounting["checkpointExpected"]
        == run_accounting["checkpointObserved"]
        and run_accounting["activeFixtureExpected"]
        == run_accounting["activeFixtureObserved"]
        and run_accounting["pairedRowExpected"]
        == run_accounting["pairedRowObserved"]
        and run_accounting["silentExclusionCount"] == 0
    )

    postcompletion = [
        item for item in fixtures if item["timingConditionId"] == "post_completion"
    ]
    calibration_failures: list[str] = []
    for item in postcompletion:
        operator = item["operatorId"]
        n = item["n"]
        delta = item["severity"]["validPostTargetOrderDistanceDelta"]
        parameters = item["parameters"]
        if operator == "segment_reversal_central_v1":
            expected_delta = parameters["length"] * (parameters["length"] - 1) // 2
        elif operator == "block_transposition_adjacent_equal_v1":
            expected_delta = parameters["blockLength"] ** 2
        elif operator == "insertion_out_of_place_max_v1":
            expected_delta = n
        elif operator == "local_scramble_sattolo_v1":
            expected_delta = None
            if delta <= 0:
                calibration_failures.append(
                    f"{item['lesionCaseId']}: scramble did not disorder sorted target"
                )
        else:
            expected_delta = 0
        if expected_delta is not None and delta != expected_delta:
            calibration_failures.append(
                f"{item['lesionCaseId']}: expected delta {expected_delta}, got {delta}"
            )
    checkpoint_success = all(item["success"] for item in checkpoint_rows)
    source_success = all(
        item["s01IdentityPass"] and item["s02IdentityPass"] for item in source_rows
    )
    target_success = all(
        item["targetCorrespondence"]["postLesionTargetSequenceFeasible"]
        for item in fixtures
    )
    timing_state_success = all(
        item["identityContract"]["activationCountPreserved"]
        and item["identityContract"]["streamCountersPreserved"]
        and item["identityContract"]["ledgerPreserved"]
        for item in fixtures
    )
    validation_summary = {
        "schemaVersion": "e05.s03.validation-summary.v1",
        "researchStepId": "S03",
        "operatorValidationPass": not application_failures,
        "sourceIdentityPass": source_success,
        "s02CheckpointIdentityPass": checkpoint_success,
        "formationReplayPass": not formation_replay_failures,
        "operatorReplayPass": not operator_replay_failures,
        "pairingPass": pairing["success"],
        "targetFeasibilityPass": target_success,
        "timingStatePreservationPass": timing_state_success,
        "severityCalibrationPass": not calibration_failures,
        "runAccountingPass": run_accounting["success"],
        "formationReplayCount": formation_replays,
        "operatorReplayCount": operator_replays,
        "fixtureCount": fixture_count,
        "pairedRowCount": len(paired_rows),
        "checkpointCount": checkpoint_count,
        "applicationFailureCount": len(application_failures),
        "formationReplayFailureCount": len(formation_replay_failures),
        "operatorReplayFailureCount": len(operator_replay_failures),
        "calibrationFailureCount": len(calibration_failures),
    }
    validation_summary["success"] = all(
        value
        for key, value in validation_summary.items()
        if key.endswith("Pass")
    )
    return {
        "fixtures": fixtures,
        "pairedRows": paired_rows,
        "checkpointRows": checkpoint_rows,
        "sourceRows": source_rows,
        "reverseRows": reverse_rows,
        "selectedStates": selected_states,
        "pairingValidation": pairing,
        "runAccounting": run_accounting,
        "validationSummary": validation_summary,
        "applicationFailures": application_failures,
        "formationReplayFailures": formation_replay_failures,
        "operatorReplayFailures": operator_replay_failures,
        "calibrationFailures": calibration_failures,
    }


def _severity_rows(fixtures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lesionCaseId": item["lesionCaseId"],
            "operatorId": item["operatorId"],
            "family": item["family"],
            "timingConditionId": item["timingConditionId"],
            "clock": item["clock"],
            "n": item["n"],
            "policy": item["policy"],
            "direction": item["direction"],
            "replicateOrdinal": item["replicateOrdinal"],
            **item["severity"],
        }
        for item in fixtures
    ]


def _target_rows(fixtures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lesionCaseId": item["lesionCaseId"],
            "operatorId": item["operatorId"],
            "timingConditionId": item["timingConditionId"],
            "n": item["n"],
            **item["targetCorrespondence"],
        }
        for item in fixtures
    ]


def _calibration_summary(severity: pd.DataFrame) -> pd.DataFrame:
    return (
        severity.groupby(["operatorId", "n"], sort=True)
        .agg(
            fixtureCount=("lesionCaseId", "size"),
            medianOrderDelta=("validPostTargetOrderDistanceDelta", "median"),
            minimumOrderDelta=("validPostTargetOrderDistanceDelta", "min"),
            maximumOrderDelta=("validPostTargetOrderDistanceDelta", "max"),
            medianNormalizedOrderDelta=("normalizedOrderDistanceDelta", "median"),
            medianAffectedFraction=("directlyAffectedIdentityFraction", "median"),
            identityEditDistance=("identityCardinalityEditDistance", "median"),
            frozenIdentityCount=("frozenIdentityCount", "median"),
        )
        .reset_index()
    )


def _write_report(
    output: Path,
    panel: Mapping[str, Any],
    git_commit: str,
    artifact_paths: Sequence[str],
) -> None:
    validation = panel["validationSummary"]
    accounting = panel["runAccounting"]
    pairing = panel["pairingValidation"]
    report = f"""# S03 Research Step Full Results — Apply multiple lesion types

## Top summary

- **Research step ID:** S03
- **Completion status:** Complete on 2026-07-17; stopped before S04.
- **Artifacts written:** {len(artifact_paths) + 2} compact report, specification, fixture, validation, accounting, provenance, and manifest files under `/artifacts/research_steps/S03/`, including the canonical `lesion_library/`.
- **Validation result:** PASS — {validation['checkpointCount']}/{accounting['checkpointExpected']} exact S02 checkpoints matched; {validation['fixtureCount']}/{accounting['activeFixtureExpected']} active fixtures and {validation['pairedRowCount']}/{accounting['pairedRowExpected']} paired rows were accounted; {validation['formationReplayCount']}/{accounting['sourceBlockExpected']} formation replays and {validation['operatorReplayCount']}/{accounting['activeFixtureExpected']} operator replays were exact; all invariants, inverses, targets, timing-state fields, severity anchors, and pairs passed.
- **Outcome classification:** Supportive — the predeclared S03 completion criterion was met: all seven lesion types now have explicit, deterministic, fixture-tested identity, target, metric, severity, feasibility, and reversibility semantics.
- **Caveats or blockers:** S03 validates the lesion library, not recovery outcomes. Duplication, deletion, and insertion are deliberately outside the fixed-identity E01 runner; freezing requires a faulted-scenario rebuild and therefore cannot claim exact post-injury scenario-ID RNG coupling. Severity is a vector of descriptors, not a biologically commensurate scalar. No blocker remains within S03.
- **Lay summary:** The simulator can now create seven clearly different kinds of damage at every previously validated damage time. It keeps track of which cells remain, which are copied, removed, or newly inserted, what the repaired pattern would mean afterward, and how to undo each test lesion. Deletion and insertion are not mislabeled as ordinary sorting problems.
- **Recommended next action:** Chief Scientist review of S03. If accepted, separately authorize S04 to add dynamic fault processes; do not start S04 from this handoff.

## Frozen question

Can freezing, central segment reversal, local scrambling, block transposition, tandem duplication, central deletion, and direction-wrong insertion be defined and replayed over the exact S02 timing checkpoints without changing the inherited task, checkpoint, timing, phase-budget, target, or pairing contracts—and with valid non-permutation metrics for count-changing tasks?

## Outcome and anchor results

S03 is **supportive** at the operator-library level. The panel reconstructed all {accounting['sourceBlockObserved']} S01 construction blocks and all {accounting['checkpointObserved']} S02 trigger checkpoints. It applied seven active operators at each checkpoint, yielding {accounting['activeFixtureObserved']} active fixtures and {accounting['pairedRowObserved']} sham/active scenario rows. There were zero substitutions, silent exclusions, operator failures, replay mismatches, inverse failures, or target-feasibility failures.

The lesion library contains exactly these operators:

1. `freeze_stuck_one_central_v1`: one central occupied identity becomes E02 `stuck`; identity and occupancy are conserved.
2. `segment_reversal_central_v1`: a centered 20%-scale segment is reversed.
3. `local_scramble_sattolo_v1`: a centered 20%-scale window receives a counter-addressed Sattolo single-cycle derangement, so every selected identity changes position.
4. `block_transposition_adjacent_equal_v1`: two centered, adjacent, equally sized blocks exchange positions.
5. `duplication_tandem_v1`: a provenance-linked copy of the central identity is inserted immediately after its source.
6. `deletion_central_v1`: the central identity, cell record, and owned cursor are removed and captured in a complete tombstone.
7. `insertion_out_of_place_max_v1`: a novel maximum-valued identity is inserted at the direction-wrong boundary.

The S01/S02 `s01_validation_adjacent_swap_v1` fixture is explicitly prohibited and never appears as an S03 operator.

## Inputs and inherited contracts

- S01 task specification, baseline scenarios, checkpoint evidence, validation summary, and full-results report from `/artifacts/research_steps/S01/`.
- S02 timing specification, all 384 pre-injury records, 768 timing scenarios, validation summary, and full-results report from `/artifacts/research_steps/S02/`.
- E01 reference simulator release and E02 scheduler, fault, and random-stream contracts mounted under `/previous-artifacts/E01/` and `/previous-artifacts/E02/`.
- Workspace `AGENTS.md`, `FULL_PLAN.md`, `RESEARCH_PLAN.md`, and `input-attachments/MANIFEST.json`.
- No dataset was used and no dependency was installed.

The exact inherited phase budget remained `100*n^2` charged opportunities for development and `100*n^2` after intervention. The source scenario envelope remained `2*budget + 20*n`; post-completion checkpoints retained the absorbing-target certificate and at-least-two-opportunities-per-identity stabilization probe capped at `20*n`. Every lesion is instantaneous: it advances neither activation/global event clock, stream counters, nor ledger. Surviving Selection cursors are preserved without reset or remapping.

## Detailed methods

### Checkpoint reconstruction

For every combination of `n ∈ {{20,50}}`, Bubble/Insertion/Selection, ascending/descending, and four replicates, the S01 construction seed, generation key, uniform scheduler, and full development execution were reconstructed. Initialization, 25%/50%/75% first progress crossings, 25%/50%/75% paired-completion event clocks, and stabilized post-completion checkpoints were located with the S02 implementation. Each reconstructed checkpoint was compared with S02 for state hash, occupancy hash, internal-state hash, prefix digest, event index, distance, Selection cursors, stream counters, and ledger.

### Identity and state semantics

Freezing, reversal, scrambling, and transposition conserve the identity set. Duplication adds one derived identity linked to its source. Deletion removes one identity and retains its cell, index, and cursor tombstone. Insertion adds one novel identity. All surviving cell records and policy state are unchanged except the intentionally frozen cell's fault field. A generated Selection identity starts at the direction-specific boundary of the new state; existing Selection cursors remain byte-for-byte unchanged even if the identity count changes.

### Targets and metrics

Identity-conserving lesions retain S01's direction-aware strict unequal inversion target. Duplication uses the sorted augmented multiset plus an identity-addition descriptor. Deletion uses the sorted survivor target plus an identity-deficit descriptor. Out-of-place insertion uses the sorted augmented set plus an identity-addition descriptor. Thus no deletion or insertion row uses `permutation_inversion_distance_v1`.

All targets are sequence-feasible by construction, but sequence feasibility is separated from runtime reachability. The three count-changing operators are not executable in the current fixed-identity E01 engine. Freezing is executable only after rebuilding canonical scenario content, which changes the scenario ID and prevents an exact post-injury scenario-ID RNG-pairing claim. Occupancy-only lesions preserve the inherited shared-prefix status.

### Severity calibration

Severity is not collapsed to one cross-family number. Each fixture records valid pre/post order distance and delta, normalized order-distance change, directly affected count/fraction, moved-survivor count, normalized existing-identity displacement, identity additions/removals and edit distance, and newly frozen count. Calibration is stratified by operator and `n` in `severity_calibration_summary.parquet`.

On every stabilized sorted fixture, the analytical anchors passed: reversal adds `C(k,2)` inversions; equal-block transposition adds `b^2`; direction-wrong insertion of a novel maximum adds exactly `n`; freezing, tandem duplication, and deletion add zero order inversions; each Sattolo fixture is a fixed-point-free derangement and introduces positive disorder. These anchors validate implementation, not biological equivalence.

### Reversibility

Reversal and equal-block transposition are self-inverse. Scrambling restores the recorded pre-window. Generated duplication and insertion identities are removed by ID. Freezing restores the original fault record. Deletion is conditionally exactly reversible only with the recorded tombstone. All {validation['fixtureCount']} inversions restored the complete pre-lesion S03 state hash, including identities, occupancy, cells, cursor state, event clock, streams, and ledger.

### Pairing and timing leakage

Every active fixture has one canonical sham at the exact same pre-injury state. The pair validator compared S01 block, timing condition, pre-injury and pre-lesion hashes, source scenario, event index, stream/ledger digests, budgets, and event-budget profile. All {pairing['pairCount']} pairs passed. Operators receive only the checkpoint, declared injury seed, source scenario ID, S01 pairing ID, and timing-condition label; they receive no downstream recovery result or future policy state. Only local scrambling consumes a dedicated counter-addressed injury stream.

## Commands

```text
python -m pytest -q tests/test_regeneration_lesions.py
python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py
ruff check src/regeneration/lesions.py tests/test_regeneration_lesions.py scripts/build_regeneration_s03.py src/regeneration/__init__.py
python scripts/build_regeneration_s03.py --artifacts-dir /artifacts/research_steps/S03
```

Execution was intentionally serial. The counter-addressed panel is only 48 source trajectories and 2,688 deterministic transforms; avoiding nested/process parallelism makes the replay audit direct and leaves shared-machine headroom.

## Results and validation

| Validation | Result |
| --- | ---: |
| S01/S02 source identity | {accounting['sourceBlockObserved']}/{accounting['sourceBlockExpected']} pass |
| Exact S02 checkpoint identity | {validation['checkpointCount']}/{accounting['checkpointExpected']} pass |
| Active operator invariants | {validation['fixtureCount']}/{accounting['activeFixtureExpected']} pass |
| Sham/active pairing | {pairing['pairCount']}/{accounting['activeFixtureExpected']} pass |
| Formation deterministic replay | {validation['formationReplayCount']}/{accounting['sourceBlockExpected']} exact |
| Operator deterministic replay | {validation['operatorReplayCount']}/{accounting['activeFixtureExpected']} exact |
| Exact inverse restoration | {validation['fixtureCount']}/{accounting['activeFixtureExpected']} pass |
| Target sequence feasibility | {validation['fixtureCount']}/{accounting['activeFixtureExpected']} pass |
| Timing state preservation | {validation['fixtureCount']}/{accounting['activeFixtureExpected']} pass |
| Stabilized severity anchors | all pass |
| Substitutions / silent exclusions | 0 / 0 |

The run-accounting table records 384 fixtures for each operator and 336 fixtures for each timing condition. Count-conserving operators contributed 1,536 active fixtures; duplication, deletion, and insertion contributed 1,152 count-changing fixtures. Runtime compatibility is therefore explicit rather than inferred.

## Artifacts

The principal machine-readable outputs are:

- `lesion_library/lesion_spec.json` and schemas: frozen semantics.
- `lesion_library/operator_fixtures.parquet`: all 2,688 active fixtures.
- `lesion_library/severity_descriptors.parquet`: complete severity vectors.
- `lesion_library/target_correspondence.parquet`: target and metric profile per fixture.
- `lesion_library/reversibility_validation.parquet`: inverse audit per fixture.
- `lesion_library/selected_fixture_states.jsonl`: 14 full hand-inspectable post-completion states, one per operator and direction.
- `lesion_scenarios.parquet`: all 5,376 sham/active rows.
- checkpoint, source, pairing, target, timing-state, replay, severity, run-accounting, validation, provenance, and artifact manifests.

## Caveats, failed assumptions, and limitations

- S03 does not run recovery trajectories, compare repair capacities, or estimate lesion effects. Those are downstream experiments.
- Order-distance deltas can be negative at partially formed states because a reversal or deletion can accidentally reduce current disorder. That is evidence about the intervention at that checkpoint, not a fixture failure.
- Severity descriptors are deliberately multidimensional. Equal affected fractions, inversion deltas, or identity edits do not make different lesion families biologically equivalent.
- Tandem duplication copies value, policy, direction, fault, and analysis label but creates a new identity and fresh Selection cursor where applicable. It is not the same object duplicated by reference.
- Deletion's original-identity target is infeasible until the tombstone is restored. Insertion and duplication require augmented targets.
- All S02 main-panel thresholds were reachable. The inherited explicit-unreachable/no-substitution rule remains tested in S02 but has no populated S03 main-panel rows.
- Freezing validates a single central E02 `stuck` identity. It does not calibrate multiple frozen counts or passive faults.
- The Sattolo scramble is deterministic for the declared injury address and fixed-point-free within its window, but its exact inversion delta varies by checkpoint and seed.

## Provenance

- Repository: `Eidosoma/cell_research`
- Branch: `eidosoma/groups/28`
- Source commit: `{git_commit}`
- Benchmark version: `{BENCHMARK_VERSION}`
- RNG: E01 SHA-256 counter-addressed runtime profile plus dedicated `lesion_local_scramble_s03_v1` injury stream.
- Runtime: Python {platform.python_version()}, pandas {package_version('pandas')}, pyarrow {package_version('pyarrow')}, jsonschema {package_version('jsonschema')}.
- Generated UTC: {datetime.now(timezone.utc).isoformat()}

Input and output SHA-256 hashes are recorded in `input_provenance.json` and `artifact_manifest.json`. Repository source remains in Git and was not copied into the artifact directory.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")


def write_outputs(output: Path, specification: Mapping[str, Any], panel: Mapping[str, Any]) -> None:
    library = output / "lesion_library"
    library.mkdir(parents=True, exist_ok=True)
    fixtures = panel["fixtures"]
    severity_rows = _severity_rows(fixtures)
    severity = pd.DataFrame(severity_rows)
    _write_json(library / "lesion_spec.json", specification)
    _write_json(library / "lesion_spec.schema.json", LESION_SPEC_SCHEMA)
    _write_json(library / "lesion_fixture.schema.json", LESION_FIXTURE_SCHEMA)
    _scalar_frame(fixtures).to_parquet(
        library / "operator_fixtures.parquet", index=False
    )
    severity.to_parquet(library / "severity_descriptors.parquet", index=False)
    _scalar_frame(_target_rows(fixtures)).to_parquet(
        library / "target_correspondence.parquet", index=False
    )
    pd.DataFrame(panel["reverseRows"]).to_parquet(
        library / "reversibility_validation.parquet", index=False
    )
    _calibration_summary(severity).to_parquet(
        library / "severity_calibration_summary.parquet", index=False
    )
    with (library / "selected_fixture_states.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in panel["selectedStates"]:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    pd.DataFrame(panel["pairedRows"]).to_parquet(
        output / "lesion_scenarios.parquet", index=False
    )
    pd.DataFrame(panel["checkpointRows"]).to_parquet(
        output / "checkpoint_compatibility.parquet", index=False
    )
    pd.DataFrame(panel["sourceRows"]).to_parquet(
        output / "source_identity_validation.parquet", index=False
    )

    validations = {
        "operator_validation.json": {
            "schemaVersion": "e05.s03.operator-validation.v1",
            "fixtureCount": len(fixtures),
            "failureCount": len(panel["applicationFailures"]),
            "failures": panel["applicationFailures"],
            "success": not panel["applicationFailures"],
        },
        "replay_validation.json": {
            "schemaVersion": "e05.s03.replay-validation.v1",
            "formationReplayCount": panel["validationSummary"][
                "formationReplayCount"
            ],
            "formationFailures": panel["formationReplayFailures"],
            "operatorReplayCount": panel["validationSummary"][
                "operatorReplayCount"
            ],
            "operatorFailures": panel["operatorReplayFailures"],
            "success": not panel["formationReplayFailures"]
            and not panel["operatorReplayFailures"],
        },
        "s02_checkpoint_identity_validation.json": {
            "schemaVersion": "e05.s03.s02-checkpoint-identity-validation.v1",
            "checkpointCount": len(panel["checkpointRows"]),
            "failureCount": sum(
                not row["success"] for row in panel["checkpointRows"]
            ),
            "success": all(row["success"] for row in panel["checkpointRows"]),
        },
        "pairing_validation.json": panel["pairingValidation"],
        "target_feasibility_validation.json": {
            "schemaVersion": "e05.s03.target-feasibility-validation.v1",
            "fixtureCount": len(fixtures),
            "countChangingFixtureCount": sum(
                not item["identityContract"]["identityConserving"]
                for item in fixtures
            ),
            "invalidPermutationMetricCount": sum(
                "permutation_inversion_distance_v1"
                in item["targetCorrespondence"]["metricProfile"]
                for item in fixtures
                if not item["identityContract"]["identityConserving"]
            ),
            "targetFeasibilityFailureCount": sum(
                not item["targetCorrespondence"][
                    "postLesionTargetSequenceFeasible"
                ]
                for item in fixtures
            ),
            "success": all(
                item["targetCorrespondence"]["postLesionTargetSequenceFeasible"]
                for item in fixtures
            )
            and all(
                "permutation_inversion_distance_v1"
                not in item["targetCorrespondence"]["metricProfile"]
                for item in fixtures
                if not item["identityContract"]["identityConserving"]
            ),
        },
        "timing_state_preservation_validation.json": {
            "schemaVersion": "e05.s03.timing-state-preservation-validation.v1",
            "fixtureCount": len(fixtures),
            "activationCountFailureCount": sum(
                not item["identityContract"]["activationCountPreserved"]
                for item in fixtures
            ),
            "streamCounterFailureCount": sum(
                not item["identityContract"]["streamCountersPreserved"]
                for item in fixtures
            ),
            "ledgerFailureCount": sum(
                not item["identityContract"]["ledgerPreserved"] for item in fixtures
            ),
            "success": panel["validationSummary"]["timingStatePreservationPass"],
        },
        "severity_calibration_validation.json": {
            "schemaVersion": "e05.s03.severity-calibration-validation.v1",
            "postCompletionFixtureCount": sum(
                item["timingConditionId"] == "post_completion" for item in fixtures
            ),
            "failureCount": len(panel["calibrationFailures"]),
            "failures": panel["calibrationFailures"],
            "crossFamilyScalarEquivalenceClaimed": False,
            "success": not panel["calibrationFailures"],
        },
        "run_accounting.json": panel["runAccounting"],
        "validation_summary.json": panel["validationSummary"],
    }
    for name, content in validations.items():
        _write_json(output / name, content)


def _provenance(output: Path, git_commit: str) -> None:
    _write_json(
        output / "input_provenance.json",
        {
            "schemaVersion": "e05.s03.input-provenance.v1",
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in (CONFIG, S02_CONFIG, *INPUTS)
            ],
        },
    )
    _write_json(
        output / "environment_provenance.json",
        {
            "schemaVersion": "e05.s03.environment-provenance.v1",
            "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "repository": str(REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "gitCommit": git_commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCountVisible": os.cpu_count(),
            "workerCount": 1,
            "packages": {
                name: package_version(name)
                for name in ("pandas", "pyarrow", "jsonschema", "pytest")
            },
        },
    )
    (output / "execution_commands.log").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_regeneration_lesions.py",
                "python -m pytest -q tests/test_regeneration_tasks.py tests/test_regeneration_timing.py tests/test_regeneration_lesions.py",
                "ruff check src/regeneration/lesions.py tests/test_regeneration_lesions.py scripts/build_regeneration_s03.py src/regeneration/__init__.py",
                "python scripts/build_regeneration_s03.py --artifacts-dir /artifacts/research_steps/S03",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _manifest(output: Path, git_commit: str) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e05.s03.artifact-manifest.v1",
            "researchStepId": "S03",
            "benchmarkVersion": BENCHMARK_VERSION,
            "repositoryCommit": git_commit,
            "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "files": [
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in files
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/artifacts/research_steps/S03"),
    )
    args = parser.parse_args()
    _validate_inputs()
    specification = _load_json(CONFIG)
    validate_lesion_spec(specification)
    panel = build_panel(specification)
    if not panel["validationSummary"]["success"]:
        raise RuntimeError(
            "S03 validation failed: "
            + json.dumps(panel["validationSummary"], sort_keys=True)
        )
    output = args.artifacts_dir
    output.mkdir(parents=True, exist_ok=True)
    write_outputs(output, specification, panel)
    git_commit = _git("rev-parse", "HEAD")
    _provenance(output, git_commit)
    prospective = [
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file()
    ]
    _write_report(output, panel, git_commit, prospective)
    _manifest(output, git_commit)
    print(json.dumps(panel["validationSummary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
