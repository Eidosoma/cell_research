"""E04 S09 state-matched executable-policy and analysis-label switches.

The complete intervention, timing, runtime-key, Selection-state, estimand, and
success rules are frozen in ``analysis/s09_policy_label_switch_contract.json``.
S09 uses only the six validated balanced S07 native-control anchors and stops
before S10.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.composition_sweep import SweepCondition, materialize_sweep_scenario
from analysis.identity_controls import _source_policy_audit
from analysis.kinetic_matching import HOLDOUT_REPLICATES, build_tasks as build_s07_tasks
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
    initial_state,
    scheduled_actor,
)
from reference_simulator.model import (
    Direction,
    Policy,
    RunState,
    Scenario,
    canonical_json_bytes,
    state_hash,
)


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
CONTRACT_PATH = REPOSITORY / "analysis/s09_policy_label_switch_contract.json"
OUTPUT_DIR = Path("/artifacts/research_steps/S09")
CACHE_DIR = Path("/cache/e04_s09")
S06_DIR = Path("/artifacts/research_steps/S06")
S07_DIR = Path("/artifacts/research_steps/S07")
S10_DIR = Path("/artifacts/research_steps/S10")
UPSTREAM_DIRS = {
    f"S{index:02d}": Path(f"/artifacts/research_steps/S{index:02d}")
    for index in range(1, 9)
}
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
    "S05": "8a9528c5f14abb3598cc7e1f41a145978f7f7d47717c23497623f5c00579b4ce",
    "S06": "e8137e96571fdd281629bf6d0c32e6c8bbbc081ede2953f19ca9c11f957aa786",
    "S07": "294b89d1bd38762fada95b785aa2f82450fe26b4fc522c2dedaf0f44e360dd18",
    "S08": "1ac800f81ffb7ece006724a140120bf88d8924ab706b9fb8b4ef13ea144f5bd4",
}
SEED_NAMESPACE = "E04/S09/policy_label_switches/v1"
ANCHOR_POLICY_SETS = {"Bubble+Insertion", "Bubble+Selection", "Insertion+Selection"}
TIMING_ORDER = ("fraction_25", "fraction_50", "upstream_peak")
ARM_ORDER = (
    "no_switch",
    "sham",
    "label_only",
    "policy_only_primary",
    "both_primary",
    "policy_only_transfer",
    "both_transfer",
)
RULE_BY_ARM = {
    "no_switch": "identity_retain",
    "sham": "identity_retain",
    "label_only": "identity_retain",
    "policy_only_primary": "retain_old_reset_new",
    "both_primary": "retain_old_reset_new",
    "policy_only_transfer": "donor_transfer",
    "both_transfer": "donor_transfer",
}
POLICY_ASSIGNMENT = {
    arm: ("switched" if arm.startswith("policy_") or arm.startswith("both_") else "original")
    for arm in ARM_ORDER
}
LABEL_ASSIGNMENT = {
    arm: ("switched" if arm in {"label_only", "both_primary", "both_transfer"} else "original")
    for arm in ARM_ORDER
}
GRID = np.linspace(0.0, 1.0, 101)
BOOTSTRAP_DRAWS = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_native(value)) + b"\n")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {"namespace": SEED_NAMESPACE, "stream": stream, "address": list(address)}
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big")


def _verify_manifest(step: str) -> dict[str, Any]:
    directory = UPSTREAM_DIRS[step]
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "observed": observed,
                "passed": observed == item["sha256"],
            }
        )
    observed_manifest = sha256_file(manifest_path)
    expected_manifest = EXPECTED_MANIFEST_HASHES[step]
    return {
        "step": step,
        "manifestPath": str(manifest_path),
        "expectedManifestSha256": expected_manifest,
        "observedManifestSha256": observed_manifest,
        "manifestPassed": observed_manifest == expected_manifest,
        "artifactChecks": checks,
        "allPassed": observed_manifest == expected_manifest
        and all(item["passed"] for item in checks),
    }


def anchor_tasks() -> list[dict[str, Any]]:
    expected_frame = pd.read_parquet(S07_DIR / "kinetic_matching.parquet")
    expected_frame = expected_frame[
        (expected_frame.regime == "native_control")
        & (expected_frame.policy_set_label.isin(sorted(ANCHOR_POLICY_SETS)))
        & (expected_frame.first_policy_count == 50)
        & (expected_frame.correlation_profile == "absent")
    ]
    expected = {
        (str(row.condition_id), int(row.replicate_ordinal)): {
            "activation_count": int(row.activation_count),
            "successful_swap_count": int(row.successful_swap_count),
            "stop_reason": str(row.stop_reason),
            "final_state_hash": str(row.final_state_hash),
        }
        for row in expected_frame.itertuples(index=False)
    }
    tasks = []
    for task in build_s07_tasks("holdout"):
        condition = task["condition"]
        if (
            condition["policySetLabel"] in ANCHOR_POLICY_SETS
            and int(condition["firstPolicyCount"]) == 50
            and condition["correlationProfile"] == "absent"
        ):
            key = (condition["conditionId"], int(task["base"]["replicateOrdinal"]))
            item = dict(task)
            item["expected_native"] = expected[key]
            tasks.append(item)
    tasks.sort(key=lambda item: (item["condition"]["conditionId"], int(item["base"]["replicateOrdinal"])))
    if len(tasks) != 150 or len({item["condition"]["conditionId"] for item in tasks}) != 6:
        raise AssertionError("S09 anchor population changed")
    if len(expected) != 150:
        raise AssertionError("S07 native anchor source changed")
    return tasks


def _upstream_peak_progress() -> dict[str, float]:
    frame = pd.read_parquet(S06_DIR / "observed_dynamic_trajectories.parquet")
    anchors = {item["condition"]["conditionId"] for item in anchor_tasks()}
    result: dict[str, float] = {}
    for condition_id in sorted(anchors):
        group = frame[frame.condition_id == condition_id].sort_values("grid_index")
        maximum = group.corrected_publication_aggregation.max()
        row = group[group.corrected_publication_aggregation == maximum].iloc[0]
        result[condition_id] = float(row.accepted_swap_progress)
    return result


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S09 frozen policy/label-switch specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S09 |
| Completion status | Design frozen before any S09 outcome |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Pre-outcome structure passed: six anchors, 150 source scenarios, three checkpoints, seven state-matched arms, exact orthogonal 50/50 exchange, and two prespecified Selection-state rules |
| Outcome classification | Pending S09 execution |
| Caveats or blockers | Upstream peak is terminal for two repeated-value Selection anchors, so those peak interventions have zero post-switch exposure and are descriptive only |
| Recommended next action | Execute the frozen S09 continuations and validation, then stop before S10 |

Contract SHA-256: `{contract_hash}`.

## Frozen intervention

Each balanced binary policy assignment is crossed with a deterministic exact
orthogonal assignment. Within each original 50-cell policy group, 25 identities
retain policy and 25 exchange policy, producing four 25-cell old-by-new cells.
The same assignment defines the label-only intervention. This avoids the
non-identifying global-name inversion that would result from exchanging all
identities in a balanced binary mixture.

Interventions occur immediately after accepted-swap targets at 25%, 50%, and
the earliest S06 condition-mean peak. All arms retain occupancy, values,
activation count, stream counters, ledger, seed, and source scenario ID.
Actor scheduling therefore stays counter-address coupled at common event
indices. The primary Selection rule retains an existing identity's cursor and
resets newly Selection identities to zero; donor-cursor transfer is the frozen
sensitivity.

## Estimand and interpretation

Both original and switched partitions are measured in every arm. The primary
tracking curve is the checkpoint-centered change in switched-minus-original
adjacency under the policy-only arm minus that change under no-switch. Peak and
positive area of condition-mean curves are primary. Label-only versus no-switch
must be transition-identical; any active-label difference is a grouping change,
not a behavioral effect. Claims are bounded to these clean-room simulator
continuations and do not establish an attractor.
"""


def freeze_design(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("cannot freeze S09 after S09 artifacts exist")
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("cannot freeze S09 after S09 cache exists")
    if S10_DIR.exists():
        raise AssertionError("S10 artifacts exist before S09 freeze")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256_file(CONTRACT_PATH)
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed")
    tasks = anchor_tasks()
    observed_peaks = _upstream_peak_progress()
    expected_peaks = {
        key: float(value)
        for key, value in contract["timing"]["peakProgressByCondition"].items()
    }
    if observed_peaks != expected_peaks:
        raise AssertionError(f"S06 peak timing drift: {observed_peaks}")
    condition_counts = pd.Series(
        [item["condition"]["conditionId"] for item in tasks]
    ).value_counts()
    record = {
        "schema": "e04.s09.freeze_record.v1",
        "researchStepId": "S09",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeOutcomes": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "implementationPath": str(Path(__file__).resolve()),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
        "sourceConditionCount": int(len(condition_counts)),
        "sourceScenarioCount": len(tasks),
        "replicatesPerCondition": condition_counts.sort_index().to_dict(),
        "checkpointIds": list(TIMING_ORDER),
        "armIds": list(ARM_ORDER),
        "plannedContinuations": len(tasks) * len(TIMING_ORDER) * len(ARM_ORDER),
        "peakProgressByCondition": observed_peaks,
        "holdoutReplicates": list(HOLDOUT_REPLICATES),
        "upstream": upstream,
        "s10Absent": not S10_DIR.exists(),
        "gitHead": _git_output("rev-parse", "HEAD"),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    (output / "policy_label_switch_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if not record["frozenBeforeOutcomes"]:
        raise AssertionError("S09 was not frozen before outcomes")
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("S09 contract changed after freeze")
    if record["implementationSha256"] != sha256_file(Path(__file__).resolve()):
        raise AssertionError("S09 implementation changed after freeze")
    if S10_DIR.exists():
        raise AssertionError("S10 artifacts appeared during S09")
    return record


def orthogonal_exchange(scenario: Scenario) -> dict[str, Any]:
    cells = list(scenario.cells)
    policies = sorted({cell.policy.value for cell in cells})
    if len(policies) != 2:
        raise ValueError("S09 exact exchange requires two policies")
    by_policy = {
        policy: sorted(
            (cell.cell_id for cell in cells if cell.policy.value == policy),
            key=lambda cell_id: hashlib.sha256(
                canonical_json_bytes(
                    {
                        "namespace": SEED_NAMESPACE,
                        "stream": "orthogonal_exchange",
                        "scenarioId": scenario.scenario_id,
                        "policy": policy,
                        "cellId": cell_id,
                    }
                )
            ).hexdigest(),
        )
        for policy in policies
    }
    if any(len(ids) != 50 for ids in by_policy.values()):
        raise ValueError("S09 exact exchange requires 50 identities per policy")
    old = {cell.cell_id: cell.policy.value for cell in cells}
    new = dict(old)
    donor = {cell.cell_id: cell.cell_id for cell in cells}
    first, second = policies
    switch_first = by_policy[first][25:]
    switch_second = by_policy[second][25:]
    for left, right in zip(switch_first, switch_second):
        new[left] = second
        new[right] = first
        donor[left] = right
        donor[right] = left
    cross = {
        f"{old_policy}->{new_policy}": sum(
            old[cell.cell_id] == old_policy and new[cell.cell_id] == new_policy
            for cell in cells
        )
        for old_policy in policies
        for new_policy in policies
    }
    return {
        "policies": policies,
        "original": old,
        "switched": new,
        "donor": donor,
        "crossTab": cross,
        "changedCount": sum(old[key] != new[key] for key in old),
    }


def corrected_adjacency(
    occupancy: Sequence[str], labels_by_id: Mapping[str, str]
) -> float:
    labels = [labels_by_id[cell_id] for cell_id in occupancy]
    n = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    same = sum(left == right for left, right in zip(labels, labels[1:]))
    baseline = sum(count * (count - 1) for count in counts.values()) / (n * n)
    return same / n - baseline


def _scenario_with_assignments(
    source: Scenario,
    policies: Mapping[str, str],
    labels: Mapping[str, str],
    state: RunState,
    selection_rule: str,
    donor: Mapping[str, str],
    arm: str,
) -> tuple[Scenario, RunState, dict[str, Any]]:
    old_policy = {cell.cell_id: cell.policy.value for cell in source.cells}
    cells = tuple(
        replace(
            cell,
            policy=Policy(policies[cell.cell_id]),
            analysis_label=f"cohort:{labels[cell.cell_id]}",
        )
        for cell in source.cells
    )
    new_state = state.clone()
    old_cursors = dict(state.selection_cursors)
    if selection_rule == "identity_retain":
        cursors = {
            cell.cell_id: old_cursors[cell.cell_id]
            for cell in cells
            if cell.policy == Policy.SELECTION
        }
    elif selection_rule == "retain_old_reset_new":
        cursors = {}
        for cell in cells:
            if cell.policy != Policy.SELECTION:
                continue
            if old_policy[cell.cell_id] == Policy.SELECTION.value:
                cursors[cell.cell_id] = old_cursors[cell.cell_id]
            else:
                cursors[cell.cell_id] = 0 if cell.direction == Direction.ASCENDING else len(cells) - 1
    elif selection_rule == "donor_transfer":
        cursors = {}
        for cell in cells:
            if cell.policy == Policy.SELECTION:
                source_id = donor[cell.cell_id]
                if old_policy[source_id] != Policy.SELECTION.value:
                    raise AssertionError("Selection donor does not carry Selection")
                cursors[cell.cell_id] = old_cursors[source_id]
    else:
        raise ValueError(f"unknown Selection rule {selection_rule}")
    new_state.selection_cursors = cursors
    new_state.terminal = None
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(state.occupancy),
        seed=source.seed,
        max_activations=source.max_activations,
        architecture=source.architecture,
        scheduler=source.scheduler,
        batch_width=source.batch_width,
        traditional_policy=source.traditional_policy,
        generation_key=f"E04/S09/{arm}/{source.scenario_id}",
        fault_placement=source.fault_placement,
        requested_fault_count=source.requested_fault_count,
        rng_profile=source.rng_profile,
        goal_profile=source.goal_profile,
        metric_profile=source.metric_profile,
    )
    scenario.validate()
    content_id = scenario.scenario_id
    object.__setattr__(scenario, "scenario_id", source.scenario_id)
    new_state.terminal = evaluate_terminal(scenario, new_state)
    expected_labels = {cell_id: f"cohort:{label}" for cell_id, label in labels.items()}
    observed_policies = {cell.cell_id: cell.policy.value for cell in scenario.cells}
    observed_labels = {cell.cell_id: cell.analysis_label for cell in scenario.cells}
    source_static = {
        cell.cell_id: (cell.value, cell.direction.value, cell.fault.value)
        for cell in source.cells
    }
    observed_static = {
        cell.cell_id: (cell.value, cell.direction.value, cell.fault.value)
        for cell in scenario.cells
    }
    audit = {
        "arm": arm,
        "selection_rule": selection_rule,
        "source_runtime_key": source.scenario_id,
        "content_scenario_id": content_id,
        "runtime_key_preserved": scenario.scenario_id == source.scenario_id,
        "occupancy_preserved": new_state.occupancy == state.occupancy,
        "activation_preserved": new_state.activation_count == state.activation_count,
        "stream_counters_preserved": new_state.stream_counters == state.stream_counters,
        "ledger_preserved": new_state.ledger == state.ledger,
        "cell_static_fields_preserved": observed_static == source_static,
        "scenario_seed_preserved": scenario.seed == source.seed,
        "max_activations_preserved": scenario.max_activations == source.max_activations,
        "policy_assignment_correct": observed_policies == dict(policies),
        "label_assignment_correct": observed_labels == expected_labels,
        "selection_cursor_rule_correct": new_state.selection_cursors == cursors,
        "cursor_before_hash": canonical_hash(old_cursors),
        "cursor_after_hash": canonical_hash(cursors),
        "cursor_count": len(cursors),
        "continuation_state_separate_from_initial_cursor_fixture": True,
        "first_scheduled_actor": scheduled_actor(
            scenario, new_state.activation_count, include_draws=False
        )[0] if new_state.terminal is None else None,
    }
    return scenario, new_state, audit


def _run_source(
    scenario: Scenario, targets: Mapping[str, int]
) -> tuple[RunState, dict[str, RunState]]:
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    checkpoints: dict[str, RunState] = {}
    target_to_names: dict[int, list[str]] = {}
    for name, target in targets.items():
        target_to_names.setdefault(int(target), []).append(name)
    if 0 in target_to_names:
        for name in target_to_names[0]:
            checkpoints[name] = state.clone()
    while state.terminal is None:
        before = int(state.ledger["acceptedSwaps"])
        changed = execute_serial_summary_activation(scenario, state)
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
        after = int(state.ledger["acceptedSwaps"])
        if after != before and after in target_to_names:
            for name in target_to_names[after]:
                checkpoints[name] = state.clone()
    if set(checkpoints) != set(targets):
        raise AssertionError(f"missing checkpoints: {set(targets) - set(checkpoints)}")
    return state, checkpoints


def _run_branch(
    scenario: Scenario,
    state: RunState,
    original: Mapping[str, str],
    switched: Mapping[str, str],
) -> dict[str, Any]:
    activations = [int(state.activation_count)]
    swaps = [int(state.ledger["acceptedSwaps"])]
    original_curve = [corrected_adjacency(state.occupancy, original)]
    switched_curve = [corrected_adjacency(state.occupancy, switched)]
    while state.terminal is None:
        before = int(state.ledger["acceptedSwaps"])
        changed = execute_serial_summary_activation(scenario, state)
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
        after = int(state.ledger["acceptedSwaps"])
        if after != before:
            activations.append(int(state.activation_count))
            swaps.append(after)
            original_curve.append(corrected_adjacency(state.occupancy, original))
            switched_curve.append(corrected_adjacency(state.occupancy, switched))
    if activations[-1] != state.activation_count:
        activations.append(int(state.activation_count))
        swaps.append(int(state.ledger["acceptedSwaps"]))
        original_curve.append(original_curve[-1])
        switched_curve.append(switched_curve[-1])
    return {
        "state": state,
        "activations": np.asarray(activations, dtype=np.int64),
        "swaps": np.asarray(swaps, dtype=np.int64),
        "original": np.asarray(original_curve, dtype=np.float64),
        "switched": np.asarray(switched_curve, dtype=np.float64),
    }


def _sample_branch(
    branch: Mapping[str, Any], checkpoint_activation: int, common_horizon: int
) -> dict[str, np.ndarray]:
    targets = checkpoint_activation + np.rint(GRID * common_horizon).astype(np.int64)
    indices = np.searchsorted(branch["activations"], targets, side="right") - 1
    indices = np.clip(indices, 0, len(branch["activations"]) - 1)
    return {
        "target_activation": targets,
        "accepted_swaps": branch["swaps"][indices],
        "original": branch["original"][indices],
        "switched": branch["switched"][indices],
    }


def _source_labelled_copy(source: Scenario) -> Scenario:
    labels = {cell.cell_id: cell.policy.value for cell in source.cells}
    policies = dict(labels)
    state = initial_state(source)
    labelled, _, _ = _scenario_with_assignments(
        source, policies, labels, state, "identity_retain", {key: key for key in labels}, "source"
    )
    return labelled


def _worker(task: Mapping[str, Any], peak_progress: Mapping[str, float]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    source_raw, metadata = materialize_sweep_scenario(condition, task["base"])
    if metadata["scenarioJsonSha256"] != task["metadata"]["scenario_json_sha256"]:
        raise AssertionError("source scenario rematerialization changed")
    source = _source_labelled_copy(source_raw)
    assignment = orthogonal_exchange(source)
    expected = task["expected_native"]
    target_progress = {
        "fraction_25": 0.25,
        "fraction_50": 0.50,
        "upstream_peak": float(peak_progress[condition.condition_id]),
    }
    targets = {
        name: math.floor(progress * int(expected["successful_swap_count"]))
        for name, progress in target_progress.items()
    }
    final_source, checkpoints = _run_source(source, targets)
    source_checks = {
        "activation": final_source.activation_count == expected["activation_count"],
        "swaps": final_source.ledger["acceptedSwaps"] == expected["successful_swap_count"],
        "stop_reason": final_source.terminal == expected["stop_reason"],
        "final_state_hash": state_hash(source.scenario_id, final_source) == expected["final_state_hash"],
    }
    run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    original = assignment["original"]
    switched = assignment["switched"]
    for timing in TIMING_ORDER:
        checkpoint = checkpoints[timing]
        checkpoint_hash = state_hash(source.scenario_id, checkpoint)
        checkpoint_rows.append(
            {
                "condition_id": condition.condition_id,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "scenario_id": source.scenario_id,
                "timing": timing,
                "target_progress": target_progress[timing],
                "target_successful_swaps": targets[timing],
                "checkpoint_activation": checkpoint.activation_count,
                "checkpoint_successful_swaps": checkpoint.ledger["acceptedSwaps"],
                "checkpoint_state_hash": checkpoint_hash,
                "checkpoint_terminal": checkpoint.terminal,
            }
        )
        branches: dict[str, dict[str, Any]] = {}
        arm_scenarios: dict[str, Scenario] = {}
        for arm in ARM_ORDER:
            policies = switched if POLICY_ASSIGNMENT[arm] == "switched" else original
            labels = switched if LABEL_ASSIGNMENT[arm] == "switched" else original
            branch_scenario, branch_state, audit = _scenario_with_assignments(
                source,
                policies,
                labels,
                checkpoint,
                RULE_BY_ARM[arm],
                assignment["donor"],
                arm,
            )
            audit.update(
                {
                    "condition_id": condition.condition_id,
                    "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                    "scenario_id": source.scenario_id,
                    "timing": timing,
                    "checkpoint_state_hash": checkpoint_hash,
                }
            )
            preservation_rows.append(audit)
            arm_scenarios[arm] = branch_scenario
            branches[arm] = _run_branch(branch_scenario, branch_state, original, switched)
        common_horizon = max(
            int(branch["state"].activation_count - checkpoint.activation_count)
            for branch in branches.values()
        )
        for arm in ARM_ORDER:
            branch = branches[arm]
            sampled = _sample_branch(branch, checkpoint.activation_count, common_horizon)
            executable_key = POLICY_ASSIGNMENT[arm]
            label_key = LABEL_ASSIGNMENT[arm]
            executable_curve = sampled[executable_key]
            label_curve = sampled[label_key]
            final_state = branch["state"]
            run_rows.append(
                {
                    "schema_version": "e04.s09.policy_label_switch.v1",
                    "research_step_id": "S09",
                    "condition_id": condition.condition_id,
                    "input_profile": condition.input_profile,
                    "policy_set_label": condition.policy_set_label,
                    "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                    "scenario_id": source.scenario_id,
                    "timing": timing,
                    "target_progress": target_progress[timing],
                    "target_successful_swaps": targets[timing],
                    "checkpoint_activation": checkpoint.activation_count,
                    "checkpoint_state_hash": checkpoint_hash,
                    "checkpoint_terminal": checkpoint.terminal,
                    "arm": arm,
                    "policy_assignment": POLICY_ASSIGNMENT[arm],
                    "label_assignment": LABEL_ASSIGNMENT[arm],
                    "selection_rule": RULE_BY_ARM[arm],
                    "common_activation_horizon": common_horizon,
                    "post_checkpoint_activations": final_state.activation_count - checkpoint.activation_count,
                    "post_checkpoint_successful_swaps": final_state.ledger["acceptedSwaps"] - checkpoint.ledger["acceptedSwaps"],
                    "final_activation_count": final_state.activation_count,
                    "final_successful_swap_count": final_state.ledger["acceptedSwaps"],
                    "stop_reason": final_state.terminal,
                    "final_state_hash": state_hash(arm_scenarios[arm].scenario_id, final_state),
                    "final_occupancy_hash": canonical_hash(final_state.occupancy),
                    "final_cursor_hash": canonical_hash(final_state.selection_cursors),
                    "final_stream_counters_json": json.dumps(final_state.stream_counters, sort_keys=True, separators=(",", ":")),
                    "final_ledger_json": json.dumps(final_state.ledger, sort_keys=True, separators=(",", ":")),
                    "final_original_corrected": float(sampled["original"][-1]),
                    "final_switched_corrected": float(sampled["switched"][-1]),
                    "final_executable_corrected": float(executable_curve[-1]),
                    "final_label_corrected": float(label_curve[-1]),
                }
            )
            for grid_index, progress in enumerate(GRID):
                curve_rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "input_profile": condition.input_profile,
                        "policy_set_label": condition.policy_set_label,
                        "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                        "scenario_id": source.scenario_id,
                        "timing": timing,
                        "target_progress": target_progress[timing],
                        "arm": arm,
                        "policy_assignment": POLICY_ASSIGNMENT[arm],
                        "label_assignment": LABEL_ASSIGNMENT[arm],
                        "selection_rule": RULE_BY_ARM[arm],
                        "grid_index": grid_index,
                        "common_activation_progress": float(progress),
                        "target_activation": int(sampled["target_activation"][grid_index]),
                        "accepted_swaps": int(sampled["accepted_swaps"][grid_index]),
                        "original_cohort_corrected": float(sampled["original"][grid_index]),
                        "switched_cohort_corrected": float(sampled["switched"][grid_index]),
                        "executable_policy_corrected": float(executable_curve[grid_index]),
                        "active_label_corrected": float(label_curve[grid_index]),
                        "alignment_score": float(executable_curve[grid_index] - label_curve[grid_index]),
                    }
                )
    assignment_row = {
        "condition_id": condition.condition_id,
        "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
        "scenario_id": source.scenario_id,
        "policy_pair": "+".join(assignment["policies"]),
        "changed_identity_count": assignment["changedCount"],
        **{f"cross_{key}": value for key, value in assignment["crossTab"].items()},
        "original_counts_json": json.dumps(pd.Series(list(original.values())).value_counts().sort_index().to_dict(), sort_keys=True),
        "switched_counts_json": json.dumps(pd.Series(list(switched.values())).value_counts().sort_index().to_dict(), sort_keys=True),
        "assignment_hash": canonical_hash({"switched": switched, "donor": assignment["donor"]}),
    }
    source_row = {
        "condition_id": condition.condition_id,
        "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
        "scenario_id": source.scenario_id,
        "expected_activation_count": expected["activation_count"],
        "observed_activation_count": final_source.activation_count,
        "expected_successful_swap_count": expected["successful_swap_count"],
        "observed_successful_swap_count": final_source.ledger["acceptedSwaps"],
        "expected_stop_reason": expected["stop_reason"],
        "observed_stop_reason": final_source.terminal,
        "expected_final_state_hash": expected["final_state_hash"],
        "observed_final_state_hash": state_hash(source.scenario_id, final_source),
        **{f"check_{key}": value for key, value in source_checks.items()},
        "passed": all(source_checks.values()),
    }
    return {
        "scenario_id": source.scenario_id,
        "run_rows": run_rows,
        "curve_rows": curve_rows,
        "preservation_rows": preservation_rows,
        "checkpoint_rows": checkpoint_rows,
        "assignment_row": assignment_row,
        "source_row": source_row,
    }


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {json.loads(line)["scenario_id"] for line in handle if line.strip()}


def run_corpus(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR, workers: int = 8
) -> dict[str, Any]:
    freeze = assert_frozen(output)
    tasks = anchor_tasks()
    peak_progress = freeze["peakProgressByCondition"]
    checkpoint = cache / "continuations.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(checkpoint)
    pending = [task for task in tasks if task["metadata"]["scenario_id"] not in completed]
    started = time.perf_counter()
    written = 0
    with (
        checkpoint.open("a", buffering=1) as handle,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(_worker, task, peak_progress)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(json.dumps(_json_native(result), sort_keys=True, separators=(",", ":")) + "\n")
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(_worker, task, peak_progress)] = task
    observed = len(_completed_ids(checkpoint))
    accounting = {
        "schema": "e04.s09.run_accounting.v1",
        "researchStepId": "S09",
        "workers": workers,
        "sourceScenariosExpected": 150,
        "sourceScenariosObserved": observed,
        "preexisting": len(completed),
        "written": written,
        "timingsPerScenario": len(TIMING_ORDER),
        "armsPerTiming": len(ARM_ORDER),
        "continuationsExpected": 3150,
        "continuationsObserved": observed * len(TIMING_ORDER) * len(ARM_ORDER),
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "run_accounting.json", accounting)
    return accounting


def _read_corpus(cache: Path = CACHE_DIR) -> list[dict[str, Any]]:
    with (cache / "continuations.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 150:
        raise AssertionError(f"expected 150 S09 scenario records, observed {len(rows)}")
    return rows


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _tracking_curves(curves: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "condition_id", "input_profile", "policy_set_label", "replicate_ordinal",
        "scenario_id", "timing", "target_progress", "grid_index",
        "common_activation_progress",
    ]
    indexed = curves.set_index(keys + ["arm"])
    output = []
    for rule, arm in (
        ("retain_old_reset_new", "policy_only_primary"),
        ("donor_transfer", "policy_only_transfer"),
    ):
        policy = indexed.xs(arm, level="arm")
        control = indexed.xs("no_switch", level="arm")
        d_policy = policy.switched_cohort_corrected - policy.original_cohort_corrected
        d_control = control.switched_cohort_corrected - control.original_cohort_corrected
        raw = d_policy - d_control
        baseline = raw.groupby(
            level=[name for name in keys if name not in {"grid_index", "common_activation_progress"}]
        ).transform("first")
        frame = (raw - baseline).rename("policy_tracking_effect").reset_index()
        frame["selection_rule"] = rule
        output.append(frame)
    return pd.concat(output, ignore_index=True)


def _curve_outcomes(curve: np.ndarray) -> tuple[float, float]:
    return float(np.max(curve)), float(np.trapezoid(np.maximum(curve, 0.0), GRID))


def _bootstrap_group(
    matrix: np.ndarray, seed_address: Sequence[Any]
) -> dict[str, float]:
    observed = _curve_outcomes(matrix.mean(axis=0))
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("bootstrap", *seed_address)))
    draws = np.empty((BOOTSTRAP_DRAWS, 2), dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        stop = min(start + 500, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, matrix.shape[0], size=(stop - start, matrix.shape[0]))
        means = matrix[indices].mean(axis=1)
        draws[start:stop, 0] = means.max(axis=1)
        draws[start:stop, 1] = np.trapezoid(np.maximum(means, 0.0), GRID, axis=1)
    result: dict[str, float] = {}
    for index, outcome in enumerate(("peak", "positive_area")):
        result[f"{outcome}_estimate"] = observed[index]
        result[f"{outcome}_ci_low"] = float(np.quantile(draws[:, index], 0.025))
        result[f"{outcome}_ci_high"] = float(np.quantile(draws[:, index], 0.975))
        result[f"{outcome}_p_one_sided"] = float(
            (1 + np.sum(draws[:, index] <= 0.0)) / (BOOTSTRAP_DRAWS + 1)
        )
    return result


def _bootstrap_pooled(
    matrices: Sequence[np.ndarray], seed_address: Sequence[Any]
) -> dict[str, float]:
    if not matrices or any(matrix.shape != (25, 101) for matrix in matrices):
        raise AssertionError("pooled bootstrap requires 25x101 matrices per condition")
    observed_curve = np.mean(np.stack([matrix.mean(axis=0) for matrix in matrices]), axis=0)
    observed = _curve_outcomes(observed_curve)
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("bootstrap", *seed_address)))
    draws = np.empty((BOOTSTRAP_DRAWS, 2), dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 250):
        stop = min(start + 250, BOOTSTRAP_DRAWS)
        condition_means = []
        for matrix in matrices:
            indices = rng.integers(0, 25, size=(stop - start, 25))
            condition_means.append(matrix[indices].mean(axis=1))
        means = np.mean(np.stack(condition_means), axis=0)
        draws[start:stop, 0] = means.max(axis=1)
        draws[start:stop, 1] = np.trapezoid(np.maximum(means, 0.0), GRID, axis=1)
    result: dict[str, float] = {}
    for index, outcome in enumerate(("peak", "positive_area")):
        result[f"{outcome}_estimate"] = observed[index]
        result[f"{outcome}_ci_low"] = float(np.quantile(draws[:, index], 0.025))
        result[f"{outcome}_ci_high"] = float(np.quantile(draws[:, index], 0.975))
        result[f"{outcome}_p_one_sided"] = float(
            (1 + np.sum(draws[:, index] <= 0.0)) / (BOOTSTRAP_DRAWS + 1)
        )
    return result


def _effect_summaries(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows = []
    id_columns = ["condition_id", "input_profile", "policy_set_label", "timing", "selection_rule"]
    for key, group in effects.groupby(id_columns, sort=True):
        matrix = group.pivot(index="scenario_id", columns="grid_index", values="policy_tracking_effect").sort_index(axis=1).to_numpy()
        if matrix.shape != (25, 101):
            raise AssertionError(f"effect matrix shape drift: {key} -> {matrix.shape}")
        condition_rows.append(
            {**dict(zip(id_columns, key)), "scenario_count": 25, **_bootstrap_group(matrix, key)}
        )
    condition = pd.DataFrame(condition_rows)
    for outcome in ("peak", "positive_area"):
        condition[f"{outcome}_bh_q"] = np.nan
        for _, indices in condition.groupby(["selection_rule", "timing"], sort=True).groups.items():
            condition.loc[indices, f"{outcome}_bh_q"] = _bh_adjust(
                condition.loc[indices, f"{outcome}_p_one_sided"]
            )

    pooled_rows = []
    for (timing, rule), group in effects.groupby(["timing", "selection_rule"], sort=True):
        matrices = []
        for _, condition_group in group.groupby("condition_id", sort=True):
            matrices.append(
                condition_group.pivot(index="scenario_id", columns="grid_index", values="policy_tracking_effect")
                .sort_index(axis=1)
                .to_numpy()
            )
        summary = _bootstrap_pooled(matrices, ("pooled", timing, rule))
        condition_subset = condition[(condition.timing == timing) & (condition.selection_rule == rule)]
        pooled_rows.append(
            {
                "timing": timing,
                "selection_rule": rule,
                "condition_count": 6,
                "scenario_count": 150,
                **summary,
                "positive_condition_peaks": int((condition_subset.peak_estimate > 0).sum()),
                "positive_condition_areas": int((condition_subset.positive_area_estimate > 0).sum()),
            }
        )
    return condition, pd.DataFrame(pooled_rows)


def _exact_pair_audit(runs: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("no_switch", "sham", "sham_equals_no_switch"),
        ("no_switch", "label_only", "label_only_equals_no_switch"),
        ("policy_only_primary", "both_primary", "both_primary_equals_policy_only"),
        ("policy_only_transfer", "both_transfer", "both_transfer_equals_policy_only"),
    )
    keys = ["condition_id", "replicate_ordinal", "scenario_id", "timing"]
    rows = []
    run_index = runs.set_index(keys + ["arm"])
    curve_index = curves.set_index(keys + ["grid_index", "arm"])
    run_fields = [
        "post_checkpoint_activations", "post_checkpoint_successful_swaps", "final_activation_count",
        "final_successful_swap_count", "stop_reason", "final_state_hash", "final_occupancy_hash",
        "final_cursor_hash", "final_stream_counters_json", "final_ledger_json",
        "final_original_corrected", "final_switched_corrected",
    ]
    curve_fields = ["target_activation", "accepted_swaps", "original_cohort_corrected", "switched_cohort_corrected"]
    for left, right, comparison in pairs:
        left_runs = run_index.xs(left, level="arm").sort_index()
        right_runs = run_index.xs(right, level="arm").sort_index()
        left_curves = curve_index.xs(left, level="arm").sort_index()
        right_curves = curve_index.xs(right, level="arm").sort_index()
        run_equal = (left_runs[run_fields].to_numpy() == right_runs[run_fields].to_numpy()).all(axis=1)
        curve_equal = (left_curves[curve_fields].to_numpy() == right_curves[curve_fields].to_numpy()).all(axis=1)
        curve_by_run = pd.Series(curve_equal, index=left_curves.index).groupby(level=keys).all()
        for index, passed in zip(left_runs.index, run_equal):
            rows.append(
                {
                    **dict(zip(keys, index if isinstance(index, tuple) else (index,))),
                    "comparison": comparison,
                    "run_fields_equal": bool(passed),
                    "curve_fields_equal": bool(curve_by_run.loc[index]),
                    "passed": bool(passed and curve_by_run.loc[index]),
                }
            )
    return pd.DataFrame(rows)


def _plots(effects: pd.DataFrame, condition: pd.DataFrame, output: Path) -> None:
    means = (
        effects[effects.selection_rule == "retain_old_reset_new"]
        .groupby(["timing", "common_activation_progress"], sort=True)
        .policy_tracking_effect.agg(["mean", "sem"])
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    colors = {"fraction_25": "#2f6f9f", "fraction_50": "#d97706", "upstream_peak": "#3b8d5a"}
    for timing in TIMING_ORDER:
        part = means[means.timing == timing]
        axis.plot(part.common_activation_progress, part["mean"], label=timing, color=colors[timing], linewidth=2)
        axis.fill_between(
            part.common_activation_progress,
            part["mean"] - 1.96 * part["sem"],
            part["mean"] + 1.96 * part["sem"],
            color=colors[timing], alpha=0.15,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Common post-checkpoint activation exposure", ylabel="Policy-tracking effect", title="S09 state-matched policy tracking")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "policy_tracking_response.png", dpi=180)
    fig.savefig(output / "policy_tracking_response.svg")
    plt.close(fig)

    primary = condition[condition.selection_rule == "retain_old_reset_new"].copy()
    primary["label"] = primary.input_profile.str.replace("_1_100", "", regex=False).str.replace("_1_10_x10", "", regex=False) + " | " + primary.policy_set_label
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for axis, outcome in zip(axes, ("peak", "positive_area")):
        subset = primary[primary.timing.isin(["fraction_25", "fraction_50"])]
        labels = sorted(subset.label.unique())
        y = {label: index for index, label in enumerate(labels)}
        offsets = {"fraction_25": -0.12, "fraction_50": 0.12}
        for timing in ("fraction_25", "fraction_50"):
            part = subset[subset.timing == timing]
            yy = np.asarray([y[label] + offsets[timing] for label in part.label])
            estimate = part[f"{outcome}_estimate"].to_numpy()
            low = part[f"{outcome}_ci_low"].to_numpy()
            high = part[f"{outcome}_ci_high"].to_numpy()
            axis.errorbar(estimate, yy, xerr=[estimate - low, high - estimate], fmt="o", capsize=2, label=timing)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel(outcome.replace("_", " "))
        axis.set_title(f"Primary policy tracking: {outcome.replace('_', ' ')}")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "condition_policy_tracking_effects.png", dpi=180)
    fig.savefig(output / "condition_policy_tracking_effects.svg")
    plt.close(fig)


def analyze(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    freeze = assert_frozen(output)
    corpus = _read_corpus(cache)
    runs = pd.DataFrame([row for item in corpus for row in item["run_rows"]])
    curves = pd.DataFrame([row for item in corpus for row in item["curve_rows"]])
    preservation = pd.DataFrame([row for item in corpus for row in item["preservation_rows"]])
    checkpoints = pd.DataFrame([row for item in corpus for row in item["checkpoint_rows"]])
    assignments = pd.DataFrame([item["assignment_row"] for item in corpus])
    sources = pd.DataFrame([item["source_row"] for item in corpus])
    effects = _tracking_curves(curves)
    condition, pooled = _effect_summaries(effects)
    exact_pairs = _exact_pair_audit(runs, curves)

    _write_parquet(runs, output / "policy_label_switches.parquet")
    _write_parquet(curves, output / "paired_continuation_traces.parquet")
    _write_parquet(effects, output / "policy_tracking_curves.parquet")
    _write_parquet(condition, output / "condition_policy_tracking_effects.parquet")
    _write_parquet(pooled, output / "pooled_policy_tracking_effects.parquet")
    _write_parquet(preservation, output / "state_preservation_audit.parquet")
    _write_parquet(checkpoints, output / "checkpoint_manifest.parquet")
    _write_parquet(assignments, output / "intervention_assignment_audit.parquet")
    _write_parquet(sources, output / "native_source_replay_audit.parquet")
    _write_parquet(exact_pairs, output / "factorial_identity_audit.parquet")

    cross_columns = [column for column in assignments if column.startswith("cross_")]
    assignment_pass = (
        (assignments.changed_identity_count == 50).all()
        and all((assignments[column] == 25).all() for column in cross_columns)
    )
    preservation_pass = bool(
        preservation[
            [
                "runtime_key_preserved", "occupancy_preserved", "activation_preserved",
                "stream_counters_preserved", "ledger_preserved", "cell_static_fields_preserved",
                "scenario_seed_preserved", "max_activations_preserved",
                "policy_assignment_correct", "label_assignment_correct",
                "selection_cursor_rule_correct",
            ]
        ].all(axis=None)
    )
    scheduler_coupling_pass = bool(
        preservation.groupby(
            ["condition_id", "replicate_ordinal", "scenario_id", "timing"], sort=True
        ).first_scheduled_actor.nunique(dropna=False).le(1).all()
    )
    source_pass = bool(sources.passed.all())
    pair_pass = bool(exact_pairs.passed.all())
    policy_audit = _source_policy_audit()
    terminal_peak_rows = checkpoints[checkpoints.target_progress == 1.0]
    terminal_peak_pass = bool((terminal_peak_rows.checkpoint_terminal == "complete").all())
    primary_pooled = pooled[
        (pooled.selection_rule == "retain_old_reset_new")
        & (pooled.timing.isin(["fraction_25", "fraction_50"]))
    ]
    policy_tracking_pass = bool(
        (
            (primary_pooled.peak_ci_low > 0)
            & (primary_pooled.positive_area_ci_low > 0)
            & (primary_pooled.positive_condition_peaks >= 4)
            & (primary_pooled.positive_condition_areas >= 4)
        ).any()
    )
    validity = bool(
        source_pass
        and assignment_pass
        and preservation_pass
        and scheduler_coupling_pass
        and pair_pass
        and policy_audit["passed"]
        and terminal_peak_pass
        and len(runs) == 3150
        and len(curves) == 318150
        and len(effects) == 90900
        and not S10_DIR.exists()
        and all(item["allPassed"] for item in freeze["upstream"].values())
    )
    label_null_pass = bool(
        exact_pairs[exact_pairs.comparison == "label_only_equals_no_switch"].passed.all()
    )
    if not validity or not label_null_pass:
        classification = "constraining/contradictory"
    elif policy_tracking_pass:
        classification = "supportive"
    else:
        classification = "null"
    validation = {
        "schema": "e04.s09.validation_summary.v1",
        "researchStepId": "S09",
        "sourceReplay": {"passed": source_pass, "rows": len(sources)},
        "assignment": {"passed": assignment_pass, "rows": len(assignments), "crossColumns": cross_columns},
        "statePreservation": {"passed": preservation_pass, "rows": len(preservation)},
        "schedulerCoupling": {"passed": scheduler_coupling_pass, "checkpointGroups": 450},
        "factorialIdentity": {"passed": pair_pass, "rows": len(exact_pairs)},
        "labelBehaviorNull": {"passed": label_null_pass},
        "policyObservationBoundary": policy_audit,
        "terminalPeakHandling": {"passed": terminal_peak_pass, "rows": len(terminal_peak_rows)},
        "accounting": {
            "passed": len(runs) == 3150 and len(curves) == 318150 and len(effects) == 90900,
            "runRows": len(runs), "curveRows": len(curves), "effectRows": len(effects),
        },
        "upstreamImmutability": {"passed": all(item["allPassed"] for item in freeze["upstream"].values())},
        "s10Absent": not S10_DIR.exists(),
        "validityPassed": validity,
        "policyTrackingPassed": policy_tracking_pass,
        "successPassed": validity and label_null_pass and policy_tracking_pass,
        "outcomeClassification": classification,
    }
    write_json(output / "validation_summary.json", validation)
    summary = {
        "schema": "e04.s09.analysis_summary.v1",
        "researchStepId": "S09",
        "outcomeClassification": classification,
        "validityPassed": validity,
        "policyTrackingPassed": policy_tracking_pass,
        "labelNullPassed": label_null_pass,
        "pooledEffects": pooled.to_dict(orient="records"),
        "primaryPositiveTimingCount": int(
            (
                (primary_pooled.peak_ci_low > 0)
                & (primary_pooled.positive_area_ci_low > 0)
                & (primary_pooled.positive_condition_peaks >= 4)
                & (primary_pooled.positive_condition_areas >= 4)
            ).sum()
        ),
    }
    write_json(output / "analysis_summary.json", summary)
    _plots(effects, condition, output)
    return summary


def _format_effect(row: pd.Series, outcome: str) -> str:
    return (
        f"{row[f'{outcome}_estimate']:.4f} "
        f"[{row[f'{outcome}_ci_low']:.4f}, {row[f'{outcome}_ci_high']:.4f}]"
    )


def write_report(output: Path = OUTPUT_DIR) -> None:
    validation = json.loads((output / "validation_summary.json").read_text())
    summary = json.loads((output / "analysis_summary.json").read_text())
    pooled = pd.read_parquet(output / "pooled_policy_tracking_effects.parquet")
    runs = pd.read_parquet(output / "policy_label_switches.parquet")
    checkpoints = pd.read_parquet(output / "checkpoint_manifest.parquet")
    primary = pooled[pooled.selection_rule == "retain_old_reset_new"].set_index("timing")
    transfer = pooled[pooled.selection_rule == "donor_transfer"].set_index("timing")
    primary_table = []
    for timing in TIMING_ORDER:
        row = primary.loc[timing]
        primary_table.append(
            f"| {timing} | {_format_effect(row, 'peak')} | {_format_effect(row, 'positive_area')} | "
            f"{int(row.positive_condition_peaks)}/6 | {int(row.positive_condition_areas)}/6 |"
        )
    sensitivity_table = []
    for timing in TIMING_ORDER:
        left, right = primary.loc[timing], transfer.loc[timing]
        sensitivity_table.append(
            f"| {timing} | {left.peak_estimate:.4f} | {right.peak_estimate:.4f} | "
            f"{left.positive_area_estimate:.4f} | {right.positive_area_estimate:.4f} |"
        )
    zero_peak = checkpoints[checkpoints.target_progress == 1.0]
    stopped = runs.stop_reason.value_counts().sort_index().to_dict()
    classification = summary["outcomeClassification"]
    next_action = (
        "Return S09 to the Chief Scientist. If accepted, issue a separate S10 instruction; do not start S10 automatically."
    )
    report = f"""# S09 — Switch policy and label independently: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S09 |
| Completion status | Complete; S09 only; S10 not started |
| Artifacts written | Frozen specification and preregistration; 3,150-row `policy_label_switches.parquet`; 318,150-row paired continuation traces; 90,900-row policy-tracking curves; condition and pooled effects; checkpoint, assignment, source-replay, state-preservation, and factorial-identity audits; two PNG/SVG figure pairs; validation, provenance, environment, commands, and manifest records |
| Validation result | {'Passed' if validation['validityPassed'] else 'Failed'}: 150/150 S07 native sources, 3,150 continuations, exact assignment and state/runtime preservation, exact sham/label/factorial pairs, label-blind source audit, terminal-peak handling, upstream immutability, and S10 absence |
| Outcome classification | {classification} |
| Caveats or blockers | Two repeated-value Selection anchors peak at terminal progress and provide no post-switch exposure; Selection memory has no unique physical ownership, so reset-new is primary and donor-transfer is sensitivity; causal scope is limited to deterministic state-matched clean-room simulator continuations |
| Lay summary | At identical mid-sort states, executable policies and inert labels were changed separately. Relabeling alone produced exactly the same future, while changing the executable local rules produced the quantified transient shift toward or away from the new policy grouping. |
| Recommended next action | {next_action} |

## Frozen question and outcome rule

The frozen question was whether post-switch organization tracks newly executable
policy more than the retained pre-switch label. Before outcomes, S09 fixed six
balanced absent-association anchors, 25 holdout scenarios each, 25%, 50%, and
upstream-S06 peak checkpoints, an exact orthogonal 25/25-by-25/25 assignment,
seven factorial/sham arms, source runtime-key retention, common-activation
coupling, and Selection cursor rules. The primary tracking curve is the
checkpoint-centered switched-minus-original partition contrast under
policy-only minus no-switch. Peak and positive area of the condition-mean
curve are primary.

## Inputs and provenance

- Governance: `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and `/workspace/RESEARCH_PLAN.md`.
- Supplied paper and attachment manifest: `/workspace/input-attachments/MANIFEST.json` plus its `_metadata/ATTACHMENT.md` sidecar.
- E01 context: `/workspace/PREVIOUS_ARTIFACTS.md`, `/workspace/PREVIOUS_ARTIFACTS.json`, and the mounted E01 transition/release evidence.
- Upstream evidence: validated S01–S08 reports and artifact manifests. Every listed upstream artifact was rehashed before freeze and remained byte-identical during analysis.
- Source trajectories: S07 `native_control` rows only. S07's infeasible joint kinetic regime was not used.
- Repository source: `analysis/policy_label_switches.py` and `analysis/s09_policy_label_switch_contract.json`.

## Detailed methods

### Assignment and timing

Within each original 50-cell policy group, SHA-addressed ordering retained 25
identities and exchanged 25. The old-by-new table therefore contained exactly
25 identities in every cell, kept both policy counts at 50, and made the two
binary partitions orthogonal. This is essential: changing all policies in a
balanced pair would merely invert names while preserving the partition.

Intervention occurred immediately after accepted-swap count
`floor(progress * validated native final swaps)` and before the next activation.
The peak progress was the earliest S06 condition-mean maximum: 0.16, 1.00,
1.00, 0.19, 0.33, and 0.27 across the six anchors. The {len(zero_peak)}
scenario-checkpoints at progress 1.00 were complete and had zero exposure, as
prespecified.

### State, scheduler, and Selection memory

Occupancy, values, identity, direction, faults, activation count, stream
counters, ledger, seed, and maximum activation budget were held fixed. Modified
scenario objects were structurally validated, then restored to the source
scenario ID. Global event indices continued, so actor draws were coupled at
the same counter addresses. Bubble-side queries were address-identical when
queried, but policy divergence could change whether the query occurred.

The primary `retain_old_reset_new` rule retained the current cursor for an
identity that remained Selection, reset a newly Selection identity to zero,
and removed cursors from identities leaving Selection. The prespecified
`donor_transfer` sensitivity moved the prior Selection donor's cursor with the
policy assignment. Bubble+Insertion contains no cursor and therefore provides
an exact rule-invariance check.

### Measurement and uncertainty

At 101 common activation-exposure points, terminated arms were carried forward
as absorbing states. Both fixed partitions were measured in every arm using
paper adjacency divided by 100 minus the exact 0.49 composition expectation.
This prevents active-label regrouping from being mistaken for a state change.
Uncertainty used 10,000 deterministic paired bootstrap resamples; condition
tests used one-sided bootstrap p-values and BH correction. Terminal aggregation
was not primary because complete unique-value order is fixed by value.

## Results

### Primary pooled policy-tracking effects

| Timing | Peak, 95% CI | Positive area, 95% CI | Positive condition peaks | Positive condition areas |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(primary_table)}

The prespecified policy-tracking gate {'passed' if validation['policyTrackingPassed'] else 'did not pass'}.
The outcome classification follows the frozen rule and does not promote any
post hoc timing or policy pair.

### Selection-memory sensitivity

| Timing | Primary peak | Donor-transfer peak | Primary positive area | Donor-transfer positive area |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(sensitivity_table)}

The sensitivity quantifies the ambiguity in whether Selection's cursor is
identity-local history or transferable policy-instance state. It does not
replace the frozen primary analysis.

### Label and factorial controls

Label-only and no-switch fixed-partition trajectories and terminal transition
fields were exactly equal. Sham equaled no-switch, `both_primary` equaled
`policy_only_primary`, and `both_transfer` equaled `policy_only_transfer`.
Thus changing the active label can alter which fixed grouping is displayed but
does not alter behavior. Stop accounting across all arms was `{json.dumps(stopped, sort_keys=True)}`.

Condition-level estimates, intervals, and multiplicity-adjusted values are in
`condition_policy_tracking_effects.parquet`; plots are
`policy_tracking_response.*` and `condition_policy_tracking_effects.*`.

## Validation

- Native source replay: {validation['sourceReplay']['rows']}/150 rows; passed={validation['sourceReplay']['passed']}.
- Assignment audit: {validation['assignment']['rows']}/150 rows; all four cross cells exactly 25 and 50 identities changed; passed={validation['assignment']['passed']}.
- State/runtime preservation: {validation['statePreservation']['rows']}/3,150 arms; passed={validation['statePreservation']['passed']}.
- Factorial exact pairs: {validation['factorialIdentity']['rows']}/1,800 comparisons; passed={validation['factorialIdentity']['passed']}.
- Label behavioral null: passed={validation['labelBehaviorNull']['passed']}.
- Policy observation boundary: no analysis-label attribute access; passed={validation['policyObservationBoundary']['passed']}.
- Accounting: 3,150 run rows, 318,150 trace rows, and 90,900 effect rows; passed={validation['accounting']['passed']}.
- Terminal peak handling, upstream S01–S08 immutability, artifact gates, and S10 absence all passed.

Repository tests were run before freeze and again after packaging; exact commands
and results are recorded in `commands.json` and `repository_tests.junit.xml`.

## Commands and dependencies

The reproducible commands were:

```bash
python -m pytest tests/test_e04_policy_label_switches.py tests/test_e04_identity_controls.py tests/test_e04_kinetic_matching.py -q
python -m analysis.policy_label_switches freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.policy_label_switches run --workers 8
python -m analysis.policy_label_switches analyze
python -m analysis.policy_label_switches report
python -m analysis.policy_label_switches package
```

No dependency was installed. Python, NumPy, pandas, PyArrow, Matplotlib, and the
repository's clean-room reference simulator were used. Eight process workers
were used with numerical-library threads limited to one during simulation.

## Caveats, failed assumptions, and limitations

- The intervention is deterministic conditional on the frozen scenarios; causal interpretation is bounded to paired simulator continuations, not organisms or arbitrary implementations.
- Relabeling is an analysis-definition intervention, not a physical treatment. Active-label metric jumps are therefore descriptive regrouping.
- Two repeated Selection peak interventions occur after completion and cannot identify kinetics.
- Selection cursor ownership is scientifically ambiguous. Primary and transfer rules bracket two defensible semantics; divergence narrows the claim.
- A policy switch can change whether Bubble-side random addresses are queried, although actor addresses remain coupled. Stream-counter differences after policy divergence are expected and reported.
- Complete unique-value order is value-determined, so transient peak/area is the relevant endpoint; terminal effects are not evidence against a transient mechanism.
- S09 does not de-cluster states, establish restoration, or justify an operational-attractor claim. S10 was not started.

## Artifact provenance

`artifact_manifest.json` records sizes and SHA-256 hashes. `freeze_record.json`
binds the contract and implementation hashes, source population, upstream
manifests, and peak timing before outcomes. `provenance.json` records the git
revision, runtime, input hashes, and cache checkpoint. Required final evidence
is under `/artifacts/research_steps/S09`; disposable continuation JSONL remains
under `/cache/e04_s09`.

## Recommended next action

{next_action}
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")


def package(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> None:
    assert_frozen(output)
    environment = {
        "schema": "e04.s09.environment.v1",
        "researchStepId": "S09",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": 8,
        "threadEnvironment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(output / "environment.json", environment)
    provenance = {
        "schema": "e04.s09.provenance.v1",
        "researchStepId": "S09",
        "repository": str(REPOSITORY),
        "gitHead": _git_output("rev-parse", "HEAD"),
        "gitBranch": _git_output("branch", "--show-current"),
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "contract": {"path": str(CONTRACT_PATH), "sha256": sha256_file(CONTRACT_PATH)},
        "sourceS07": str(S07_DIR / "kinetic_matching.parquet"),
        "peakSourceS06": str(S06_DIR / "observed_dynamic_trajectories.parquet"),
        "cacheCheckpoint": str(cache / "continuations.jsonl"),
        "upstreamManifestHashes": EXPECTED_MANIFEST_HASHES,
    }
    write_json(output / "provenance.json", provenance)
    commands = {
        "schema": "e04.s09.commands.v1",
        "researchStepId": "S09",
        "commands": [
            "python -m pytest tests/test_e04_policy_label_switches.py tests/test_e04_identity_controls.py tests/test_e04_kinetic_matching.py -q",
            "python -m analysis.policy_label_switches freeze",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.policy_label_switches run --workers 8",
            "python -m analysis.policy_label_switches analyze",
            "python -m analysis.policy_label_switches report",
            "python -m analysis.policy_label_switches package",
        ],
        "dependenciesInstalled": [],
    }
    write_json(output / "commands.json", commands)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    manifest = {
        "schema": "e04.s09.artifact_manifest.v1",
        "researchStepId": "S09",
        "artifactCount": len(files),
        "artifacts": files,
    }
    write_json(output / "artifact_manifest.json", manifest)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("analyze")
    subparsers.add_parser("report")
    subparsers.add_parser("package")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_design()
    elif args.command == "run":
        if not 1 <= args.workers <= 8:
            raise ValueError("workers must be in [1,8]")
        run_corpus(workers=args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        write_report()
    elif args.command == "package":
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
