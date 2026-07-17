"""E04 S11 state-matched value-position and policy-position interventions.

The complete intervention, preservation, pairing, cursor, estimand, and claim
boundaries are frozen in ``s11_value_policy_contract.json``.  S11 stops before
S12.
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
from analysis.declustering_intervention import (
    _run_source,
    _source_labelled_copy,
    matched_optimum_patterns,
    optimal_policy_pattern,
)
from analysis.identity_controls import _source_policy_audit
from analysis.policy_label_switches import anchor_tasks, corrected_adjacency
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
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
from reference_simulator.scheduler import scheduled_side


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "analysis/s11_value_policy_contract.json"
OUTPUT_DIR = Path("/artifacts/research_steps/S11")
CACHE_DIR = Path("/cache/e04_s11")
S06_DIR = Path("/artifacts/research_steps/S06")
S12_DIR = Path("/artifacts/research_steps/S12")
UPSTREAM_DIRS = {
    f"S{index:02d}": Path(f"/artifacts/research_steps/S{index:02d}")
    for index in range(1, 11)
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
    "S09": "00fffe60a61a4707d98b99a257a1df2a42a14d26af2ed4e20155c1bdd0103840",
    "S10": "5e9d15d43c74805c6edbf53d86691ba296315be11ccf16584a627d6bfa3a78f6",
}
SEED_NAMESPACE = "E04/S11/value-policy/v1"
CONDITIONS = (
    "S03-REP-BUB-INS-P50-ABS",
    "S03-REP-BUB-SEL-P50-ABS",
    "S03-REP-INS-SEL-P50-ABS",
    "S03-UNQ-BUB-INS-P50-ABS",
    "S03-UNQ-BUB-SEL-P50-ABS",
    "S03-UNQ-INS-SEL-P50-ABS",
)
PEAK_PROGRESS = {
    "S03-REP-BUB-INS-P50-ABS": 0.16,
    "S03-REP-BUB-SEL-P50-ABS": 0.36,
    "S03-REP-INS-SEL-P50-ABS": 0.75,
    "S03-UNQ-BUB-INS-P50-ABS": 0.19,
    "S03-UNQ-BUB-SEL-P50-ABS": 0.33,
    "S03-UNQ-INS-SEL-P50-ABS": 0.27,
}
ARMS = (
    "no_switch",
    "sham",
    "value_primary",
    "value_matched",
    "policy_primary",
    "policy_matched",
    "both_primary",
    "both_matched",
    "policy_transfer",
    "both_transfer",
)
GRID = np.linspace(0.0, 1.0, 101)
VALUE_CANDIDATES = 128
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
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {"namespace": SEED_NAMESPACE, "stream": stream, "address": list(address)}
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big"
    )


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


def source_tasks() -> list[dict[str, Any]]:
    tasks = [
        item
        for item in anchor_tasks()
        if item["condition"]["conditionId"] in CONDITIONS
    ]
    counts = pd.Series(
        [item["condition"]["conditionId"] for item in tasks]
    ).value_counts()
    if len(tasks) != 150 or set(counts.tolist()) != {25}:
        raise AssertionError("S11 source population changed")
    return tasks


def upstream_eligible_peaks() -> dict[str, float]:
    frame = pd.read_parquet(S06_DIR / "observed_dynamic_trajectories.parquet")
    result: dict[str, float] = {}
    for condition_id in CONDITIONS:
        group = frame[
            (frame.condition_id == condition_id)
            & (frame.accepted_swap_progress <= 0.75)
        ].sort_values("grid_index")
        maximum = group.corrected_publication_aggregation.max()
        row = group[group.corrected_publication_aggregation == maximum].iloc[0]
        result[condition_id] = float(row.accepted_swap_progress)
    return result


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S11 frozen value-position / policy-position specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S11 |
| Completion status | Design frozen before any S11 continuation outcome |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Pre-outcome structure passed: six anchors, 150 paired sources, ten state-matched arms, S10-compatible 33-assignment policy support, matched spatial-distance value derangements, primary and donor-transfer Selection rules, and 101-point common-exposure sampling |
| Outcome classification | Pending S11 execution |
| Caveats or blockers | Value reassignment changes explicit sorting progress and is not a pure mediator intervention; unique inputs cross S10's physical-feasibility boundary through direct field reassignment; strong policy-pair heterogeneity is retained rather than pooled away |
| Recommended next action | Execute and validate the frozen S11 design, report bounded intervention-specific component estimates, and stop before S12 |

Contract SHA-256: `{contract_hash}`.

## Frozen interventions

At the earliest validated S06 aggregation maximum at or below 75% accepted-swap
progress, occupancy and every RunState counter are fixed. The value-only arm
reassigns values on a prespecified support while leaving the executable policy
at every position unchanged. The policy-only arm reassigns executable policies
while leaving the exact positional value sequence unchanged. The joint arm
applies both independently frozen assignments. No arm moves an identity.

Policy targets use ten value-rank strata, S10's exact MILP audit, and its fixed
250,000-draw support rule. Primary and matched targets have the same supported
edge count and Hamming distance. Value targets derange exactly the corresponding
changed positions; the primary/matched pair is chosen without outcomes to match
inversion distance and total value change.

## State, cursor, and scheduler semantics

Direction, fault, analysis label, occupancy, identity set, activation count,
stream counters, ledger, seed, maximum budget, scheduler, and source runtime key
are exact. Retain-old/reset-new is primary for Selection; deterministic donor
transfer is sensitivity. Actor scheduling remains exactly counter-address
coupled. Policy-dependent Bubble-side queries remain coupled for unchanged
actors and may differ only when their executable policy changes.

## Estimand and claim boundary

The 2x2 primary factorial defines value and policy Shapley curves, their
interaction, and their joint effect for composition-corrected adjacency and
reference Sortedness. Signed, positive, negative, final, and peak-absolute
effects are reported with paired bootstrap uncertainty and matched-assignment
bounds. These are component effects of the specified simulator interventions,
not natural causal mediation. Value shuffling changes task progress by design.
"""


def freeze_design(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("cannot freeze S11 after S11 artifacts exist")
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("cannot freeze S11 after S11 cache exists")
    if S12_DIR.exists():
        raise AssertionError("S12 artifacts exist before S11 freeze")
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract["researchStepId"] != "S11":
        raise AssertionError("S11 contract identity drift")
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed")
    observed_peaks = upstream_eligible_peaks()
    if observed_peaks != PEAK_PROGRESS:
        raise AssertionError(f"S06 eligible peak drift: {observed_peaks}")
    tasks = source_tasks()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256_file(CONTRACT_PATH)
    record = {
        "schema": "e04.s11.freeze_record.v1",
        "researchStepId": "S11",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeOutcomes": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "implementationPath": str(Path(__file__).resolve()),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
        "sourceConditionCount": 6,
        "sourceScenarioCount": len(tasks),
        "replicatesPerCondition": 25,
        "armIds": list(ARMS),
        "plannedContinuations": len(tasks) * len(ARMS),
        "peakProgressByCondition": observed_peaks,
        "upstream": upstream,
        "s12Absent": not S12_DIR.exists(),
        "gitHead": _git_output("rev-parse", "HEAD"),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    (output / "value_policy_shuffle_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if not record["frozenBeforeOutcomes"]:
        raise AssertionError("S11 was not frozen before outcomes")
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("S11 contract changed after freeze")
    if record["implementationSha256"] != sha256_file(Path(__file__).resolve()):
        raise AssertionError("S11 implementation changed after freeze")
    if S12_DIR.exists():
        raise AssertionError("S12 artifacts appeared during S11")
    return record


def value_rank_strata(source: Scenario) -> dict[str, int]:
    ordered = sorted(source.cells, key=lambda cell: (float(cell.value), cell.cell_id))
    if len(ordered) != 100:
        raise ValueError("S11 rank strata require n=100")
    return {cell.cell_id: rank // 10 for rank, cell in enumerate(ordered)}


def positional_sequences(
    source: Scenario, checkpoint: RunState
) -> tuple[list[int], list[str], list[int]]:
    values = [int(source.cell_map[cell_id].value) for cell_id in checkpoint.occupancy]
    policies = [source.cell_map[cell_id].policy.value for cell_id in checkpoint.occupancy]
    strata_by_id = value_rank_strata(source)
    strata = [strata_by_id[cell_id] for cell_id in checkpoint.occupancy]
    return values, policies, strata


def select_policy_pair(
    strata: Sequence[int], policies: Sequence[str], scenario_id: str
) -> dict[str, Any]:
    optimum = optimal_policy_pattern(strata, policies, (scenario_id, "S11"))
    first, rest, support = matched_optimum_patterns(
        strata, optimum, f"S11/{scenario_id}", channels=32
    )
    patterns = np.vstack([first, rest])
    original = np.asarray(
        [optimum["policyNames"].index(policy) for policy in policies], dtype=np.uint8
    )
    hammings = np.sum(patterns != original, axis=1)
    groups: dict[int, list[int]] = {}
    for index, hamming in enumerate(hammings):
        groups.setdefault(int(hamming), []).append(index)
    eligible = [(len(indices), -hamming, hamming, indices) for hamming, indices in groups.items() if len(indices) >= 2]
    if not eligible:
        raise RuntimeError("supported policy bank has no equal-Hamming pair")
    _, _, selected_hamming, indices = max(eligible)
    primary_index, matched_index = indices[:2]
    primary = patterns[primary_index].copy()
    matched = patterns[matched_index].copy()
    return {
        "optimum": optimum,
        "support": support,
        "originalBinary": original,
        "primaryBinary": primary,
        "matchedBinary": matched,
        "policyNames": optimum["policyNames"],
        "hamming": int(selected_hamming),
        "primaryBankIndex": int(primary_index),
        "matchedBankIndex": int(matched_index),
        "equalHammingGroupSize": len(indices),
        "bankHash": hashlib.sha256(patterns.tobytes()).hexdigest(),
    }


def _inversion_count(values: Sequence[int | float]) -> int:
    array = np.asarray(values)
    return int(np.sum(array[:, None] > array[None, :]) // 2)


def _sortedness(values: Sequence[int | float]) -> tuple[float, float]:
    n = len(values)
    reference = (1 + sum(left <= right for left, right in zip(values, values[1:]))) / n
    paper = (1 + sum(left < right for left, right in zip(values, values[1:]))) / n
    return float(reference), float(paper)


def _association(values: Sequence[int | float], policies_binary: Sequence[int]) -> float:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(policies_binary, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def value_derangement_bank(
    values: Sequence[int], support: Sequence[int], scenario_id: str, target_id: str
) -> list[np.ndarray]:
    support_array = np.asarray(sorted(support), dtype=np.int64)
    if len(support_array) < 2:
        raise ValueError("value derangement support is too small")
    values_array = np.asarray(values, dtype=np.int64)
    old = values_array[support_array]
    if int(np.max(np.unique(old, return_counts=True)[1])) * 2 > len(old):
        raise RuntimeError("value-level derangement is infeasible on target support")
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("value_derangements", scenario_id, target_id))
    )
    bank: list[np.ndarray] = []
    seen: set[bytes] = set()
    attempts = 0
    while len(bank) < VALUE_CANDIDATES and attempts < 1_000_000:
        attempts += 1
        candidate = rng.permutation(old)
        if np.any(candidate == old):
            continue
        full = values_array.copy()
        full[support_array] = candidate
        key = full.tobytes()
        if key not in seen:
            seen.add(key)
            bank.append(full)
    if len(bank) != VALUE_CANDIDATES:
        raise RuntimeError(
            f"only {len(bank)} value derangements found after {attempts} attempts"
        )
    return bank


def select_value_pair(
    values: Sequence[int],
    primary_support: Sequence[int],
    matched_support: Sequence[int],
    scenario_id: str,
) -> dict[str, Any]:
    primary_bank = value_derangement_bank(
        values, primary_support, scenario_id, "primary"
    )
    matched_bank = value_derangement_bank(
        values, matched_support, scenario_id, "matched"
    )
    original = np.asarray(values, dtype=np.int64)

    def signatures(bank: Sequence[np.ndarray]) -> list[tuple[int, int, str]]:
        return [
            (
                _inversion_count(candidate),
                int(np.abs(candidate - original).sum()),
                hashlib.sha256(candidate.tobytes()).hexdigest(),
            )
            for candidate in bank
        ]

    primary_signatures = signatures(primary_bank)
    matched_signatures = signatures(matched_bank)
    choices = []
    for left, left_signature in enumerate(primary_signatures):
        for right, right_signature in enumerate(matched_signatures):
            choices.append(
                (
                    abs(left_signature[0] - right_signature[0]),
                    abs(left_signature[1] - right_signature[1]),
                    left_signature[2],
                    right_signature[2],
                    left,
                    right,
                )
            )
    choice = min(choices)
    left, right = int(choice[-2]), int(choice[-1])
    return {
        "primaryValues": primary_bank[left],
        "matchedValues": matched_bank[right],
        "primaryCandidateIndex": left,
        "matchedCandidateIndex": right,
        "inversionDifference": int(choice[0]),
        "totalValueChangeDifference": int(choice[1]),
        "primaryBankHash": hashlib.sha256(
            np.stack(primary_bank).tobytes()
        ).hexdigest(),
        "matchedBankHash": hashlib.sha256(
            np.stack(matched_bank).tobytes()
        ).hexdigest(),
    }


def donor_map(
    source: Scenario, policy_by_id: Mapping[str, str], scenario_id: str
) -> dict[str, str]:
    old = {cell.cell_id: cell.policy.value for cell in source.cells}
    donors = {
        cell_id: cell_id
        for cell_id in old
        if old[cell_id] == Policy.SELECTION.value
        and policy_by_id[cell_id] == Policy.SELECTION.value
    }
    outgoing = [
        cell_id
        for cell_id in old
        if old[cell_id] == Policy.SELECTION.value
        and policy_by_id[cell_id] != Policy.SELECTION.value
    ]
    incoming = [
        cell_id
        for cell_id in old
        if old[cell_id] != Policy.SELECTION.value
        and policy_by_id[cell_id] == Policy.SELECTION.value
    ]

    def order(cell_id: str, role: str) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "namespace": SEED_NAMESPACE,
                    "stream": "selection_donor",
                    "scenarioId": scenario_id,
                    "role": role,
                    "cellId": cell_id,
                }
            )
        ).hexdigest()

    outgoing.sort(key=lambda item: order(item, "outgoing"))
    incoming.sort(key=lambda item: order(item, "incoming"))
    if len(outgoing) != len(incoming):
        raise AssertionError("Selection policy exchange is unbalanced")
    donors.update({target: source_id for target, source_id in zip(incoming, outgoing)})
    return donors


def build_continuation(
    source: Scenario,
    checkpoint: RunState,
    value_by_id: Mapping[str, int],
    policy_by_id: Mapping[str, str],
    arm: str,
    cursor_rule: str,
) -> tuple[Scenario, RunState, dict[str, Any]]:
    old_values = {cell.cell_id: int(cell.value) for cell in source.cells}
    old_policies = {cell.cell_id: cell.policy.value for cell in source.cells}
    cells = tuple(
        replace(
            cell,
            value=int(value_by_id[cell.cell_id]),
            policy=Policy(policy_by_id[cell.cell_id]),
        )
        for cell in source.cells
    )
    state = checkpoint.clone()
    old_cursors = dict(checkpoint.selection_cursors)
    if cursor_rule == "identity_retain":
        cursors = dict(old_cursors)
    elif cursor_rule == "retain_old_reset_new":
        cursors = {}
        for cell in cells:
            if cell.policy != Policy.SELECTION:
                continue
            if old_policies[cell.cell_id] == Policy.SELECTION.value:
                cursors[cell.cell_id] = old_cursors[cell.cell_id]
            else:
                cursors[cell.cell_id] = (
                    0 if cell.direction == Direction.ASCENDING else len(cells) - 1
                )
    elif cursor_rule == "donor_transfer":
        donors = donor_map(source, policy_by_id, source.scenario_id)
        cursors = {
            cell.cell_id: old_cursors[donors[cell.cell_id]]
            for cell in cells
            if cell.policy == Policy.SELECTION
        }
    else:
        raise ValueError(f"unknown cursor rule {cursor_rule}")
    state.selection_cursors = cursors
    state.terminal = None
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(checkpoint.occupancy),
        seed=source.seed,
        max_activations=source.max_activations,
        architecture=source.architecture,
        scheduler=source.scheduler,
        batch_width=source.batch_width,
        traditional_policy=source.traditional_policy,
        generation_key=f"E04/S11/{arm}/{source.scenario_id}",
        fault_placement=source.fault_placement,
        requested_fault_count=source.requested_fault_count,
        rng_profile=source.rng_profile,
        goal_profile=source.goal_profile,
        metric_profile=source.metric_profile,
    )
    scenario.validate()
    content_id = scenario.scenario_id
    object.__setattr__(scenario, "scenario_id", source.scenario_id)
    state.terminal = evaluate_terminal(scenario, state)

    new_values = {cell.cell_id: int(cell.value) for cell in scenario.cells}
    new_policies = {cell.cell_id: cell.policy.value for cell in scenario.cells}
    old_static = {
        cell.cell_id: (cell.direction.value, cell.fault.value, cell.analysis_label)
        for cell in source.cells
    }
    new_static = {
        cell.cell_id: (cell.direction.value, cell.fault.value, cell.analysis_label)
        for cell in scenario.cells
    }
    value_changed = {cell_id for cell_id in old_values if old_values[cell_id] != new_values[cell_id]}
    policy_changed = {cell_id for cell_id in old_policies if old_policies[cell_id] != new_policies[cell_id]}
    audit = {
        "arm": arm,
        "scenario_id": source.scenario_id,
        "cursor_rule": cursor_rule,
        "content_scenario_id": content_id,
        "runtime_key_preserved": scenario.scenario_id == source.scenario_id,
        "seed_preserved": scenario.seed == source.seed,
        "max_activations_preserved": scenario.max_activations == source.max_activations,
        "occupancy_preserved": state.occupancy == checkpoint.occupancy,
        "identity_bijection_preserved": sorted(state.occupancy) == sorted(checkpoint.occupancy),
        "activation_preserved": state.activation_count == checkpoint.activation_count,
        "stream_counters_preserved": state.stream_counters == checkpoint.stream_counters,
        "ledger_preserved": state.ledger == checkpoint.ledger,
        "direction_fault_label_preserved": old_static == new_static,
        "global_value_multiset_preserved": sorted(old_values.values()) == sorted(new_values.values()),
        "global_policy_multiset_preserved": sorted(old_policies.values()) == sorted(new_policies.values()),
        "value_changed_count": len(value_changed),
        "policy_changed_count": len(policy_changed),
        "value_changed_ids_hash": canonical_hash(sorted(value_changed)),
        "policy_changed_ids_hash": canonical_hash(sorted(policy_changed)),
        "value_fields_preserved": old_values == new_values,
        "policy_fields_preserved": old_policies == new_policies,
        "cursor_count_before": len(old_cursors),
        "cursor_count_after": len(cursors),
        "cursor_before_hash": canonical_hash(old_cursors),
        "cursor_after_hash": canonical_hash(cursors),
        "identity_cursor_preserved": cursors == old_cursors,
        "terminal_immediately_after_intervention": state.terminal,
    }
    return scenario, state, audit


def run_branch(scenario: Scenario, state: RunState) -> dict[str, Any]:
    id_to_index = {cell.cell_id: index for index, cell in enumerate(scenario.cells)}
    policy_by_id = {cell.cell_id: cell.policy.value for cell in scenario.cells}
    activations = [int(state.activation_count)]
    swaps = [int(state.ledger["acceptedSwaps"])]
    metrics = [corrected_adjacency(state.occupancy, policy_by_id)]
    occupancies = [
        np.fromiter(
            (id_to_index[cell_id] for cell_id in state.occupancy),
            dtype=np.uint8,
            count=len(state.occupancy),
        )
    ]
    while state.terminal is None:
        before = int(state.ledger["acceptedSwaps"])
        changed = execute_serial_summary_activation(scenario, state)
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
        after = int(state.ledger["acceptedSwaps"])
        if after != before:
            activations.append(int(state.activation_count))
            swaps.append(after)
            metrics.append(corrected_adjacency(state.occupancy, policy_by_id))
            occupancies.append(
                np.fromiter(
                    (id_to_index[cell_id] for cell_id in state.occupancy),
                    dtype=np.uint8,
                    count=len(state.occupancy),
                )
            )
    if activations[-1] != state.activation_count:
        activations.append(int(state.activation_count))
        swaps.append(int(state.ledger["acceptedSwaps"]))
        metrics.append(metrics[-1])
        occupancies.append(occupancies[-1].copy())
    return {
        "state": state,
        "activations": np.asarray(activations, dtype=np.int64),
        "swaps": np.asarray(swaps, dtype=np.int64),
        "metrics": np.asarray(metrics, dtype=np.float64),
        "occupancies": np.stack(occupancies),
    }


def sample_branch(
    scenario: Scenario,
    branch: Mapping[str, Any],
    checkpoint_activation: int,
    common_horizon: int,
) -> dict[str, np.ndarray]:
    targets = checkpoint_activation + np.rint(GRID * common_horizon).astype(np.int64)
    indices = np.searchsorted(branch["activations"], targets, side="right") - 1
    indices = np.clip(indices, 0, len(branch["activations"]) - 1)
    occupancy = branch["occupancies"][indices]
    values = np.asarray([cell.value for cell in scenario.cells], dtype=np.float64)
    arranged = values[occupancy]
    reference = (
        1 + np.sum(arranged[:, :-1] <= arranged[:, 1:], axis=1)
    ) / arranged.shape[1]
    paper = (
        1 + np.sum(arranged[:, :-1] < arranged[:, 1:], axis=1)
    ) / arranged.shape[1]
    return {
        "target_activation": targets,
        "accepted_swaps": branch["swaps"][indices],
        "corrected_aggregation": branch["metrics"][indices],
        "reference_sortedness": reference,
        "paper_sortedness": paper,
        "occupancy": occupancy,
    }


def scheduler_coupling_audit(
    reference: Scenario,
    comparison: Scenario,
    checkpoint_activation: int,
    event_count: int,
) -> dict[str, Any]:
    if event_count == 0:
        offsets = np.asarray([], dtype=np.int64)
    else:
        offsets = np.unique(
            np.rint(
                np.linspace(0, event_count - 1, min(257, event_count))
            ).astype(np.int64)
        )
    actor_equal = []
    unchanged_query_equal = []
    changed_policy_events = 0
    records = []
    for offset in offsets:
        event_index = checkpoint_activation + int(offset)
        left_actor, left_draws, left_consumed = scheduled_actor(
            reference, event_index, include_draws=True
        )
        right_actor, right_draws, right_consumed = scheduled_actor(
            comparison, event_index, include_draws=True
        )
        same_actor = (
            left_actor == right_actor
            and left_draws == right_draws
            and left_consumed == right_consumed
        )
        actor_equal.append(same_actor)
        left_policy = reference.cell_map[left_actor].policy
        right_policy = comparison.cell_map[right_actor].policy
        if left_policy == right_policy:
            left_side = (
                scheduled_side(reference, event_index)[0]
                if left_policy == Policy.BUBBLE
                else None
            )
            right_side = (
                scheduled_side(comparison, event_index)[0]
                if right_policy == Policy.BUBBLE
                else None
            )
            unchanged_query_equal.append(left_side == right_side)
        else:
            changed_policy_events += 1
        records.append(
            (
                event_index,
                left_actor,
                right_actor,
                left_policy.value,
                right_policy.value,
            )
        )
    passed = bool(all(actor_equal) and all(unchanged_query_equal))
    return {
        "common_event_count": event_count,
        "sampled_event_count": len(offsets),
        "actor_addresses_equal": bool(all(actor_equal)),
        "unchanged_policy_random_addresses_equal": bool(all(unchanged_query_equal)),
        "changed_policy_sampled_events": changed_policy_events,
        "qualified_policy_query_divergence": changed_policy_events > 0,
        "sample_hash": canonical_hash(records),
        "passed": passed,
    }


def _position_maps(
    source: Scenario,
    checkpoint: RunState,
    values: Sequence[int],
    policy_binary: Sequence[int],
    policy_names: Sequence[str],
) -> tuple[dict[str, int], dict[str, str]]:
    value_by_id = {cell.cell_id: int(cell.value) for cell in source.cells}
    policy_by_id = {cell.cell_id: cell.policy.value for cell in source.cells}
    for position, cell_id in enumerate(checkpoint.occupancy):
        value_by_id[cell_id] = int(values[position])
        policy_by_id[cell_id] = str(policy_names[int(policy_binary[position])])
    return value_by_id, policy_by_id


def _worker(task: Mapping[str, Any], peak_progress: Mapping[str, float]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    source_raw, metadata = materialize_sweep_scenario(condition, task["base"])
    if metadata["scenarioJsonSha256"] != task["metadata"]["scenario_json_sha256"]:
        raise AssertionError("source scenario rematerialization changed")
    source = _source_labelled_copy(source_raw)
    expected = task["expected_native"]
    progress = float(peak_progress[condition.condition_id])
    target_swap = math.floor(progress * int(expected["successful_swap_count"]))
    final_source, checkpoints = _run_source(source, {"eligible_peak": target_swap})
    checkpoint = checkpoints["eligible_peak"]
    source_checks = {
        "activation": final_source.activation_count == expected["activation_count"],
        "swaps": final_source.ledger["acceptedSwaps"]
        == expected["successful_swap_count"],
        "stop_reason": final_source.terminal == expected["stop_reason"],
        "final_state_hash": state_hash(source.scenario_id, final_source)
        == expected["final_state_hash"],
    }
    values, policies, strata = positional_sequences(source, checkpoint)
    policy_pair = select_policy_pair(strata, policies, source.scenario_id)
    original_binary = policy_pair["originalBinary"]
    primary_binary = policy_pair["primaryBinary"]
    matched_binary = policy_pair["matchedBinary"]
    primary_support = np.flatnonzero(primary_binary != original_binary)
    matched_support = np.flatnonzero(matched_binary != original_binary)
    value_pair = select_value_pair(
        values, primary_support, matched_support, source.scenario_id
    )
    value_primary = value_pair["primaryValues"]
    value_matched = value_pair["matchedValues"]
    policy_names = policy_pair["policyNames"]

    original_values, original_policies = _position_maps(
        source, checkpoint, values, original_binary, policy_names
    )
    primary_values, primary_policies = _position_maps(
        source, checkpoint, value_primary, primary_binary, policy_names
    )
    matched_values, matched_policies = _position_maps(
        source, checkpoint, value_matched, matched_binary, policy_names
    )
    arm_specifications = {
        "no_switch": (original_values, original_policies, "identity_retain"),
        "sham": (original_values, original_policies, "identity_retain"),
        "value_primary": (primary_values, original_policies, "identity_retain"),
        "value_matched": (matched_values, original_policies, "identity_retain"),
        "policy_primary": (original_values, primary_policies, "retain_old_reset_new"),
        "policy_matched": (original_values, matched_policies, "retain_old_reset_new"),
        "both_primary": (primary_values, primary_policies, "retain_old_reset_new"),
        "both_matched": (matched_values, matched_policies, "retain_old_reset_new"),
        "policy_transfer": (original_values, primary_policies, "donor_transfer"),
        "both_transfer": (primary_values, primary_policies, "donor_transfer"),
    }
    scenarios: dict[str, Scenario] = {}
    branches: dict[str, dict[str, Any]] = {}
    preservation_rows = []
    for arm in ARMS:
        value_by_id, policy_by_id, cursor_rule = arm_specifications[arm]
        scenario, state, audit = build_continuation(
            source,
            checkpoint,
            value_by_id,
            policy_by_id,
            arm,
            cursor_rule,
        )
        audit.update(
            {
                "condition_id": condition.condition_id,
                "input_profile": condition.input_profile,
                "policy_set_label": condition.policy_set_label,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "checkpoint_progress": progress,
                "checkpoint_state_hash": state_hash(source.scenario_id, checkpoint),
            }
        )
        scenarios[arm] = scenario
        branches[arm] = run_branch(scenario, state)
        preservation_rows.append(audit)
    common_horizon = max(
        int(branch["state"].activation_count - checkpoint.activation_count)
        for branch in branches.values()
    )
    sampled = {
        arm: sample_branch(
            scenarios[arm], branches[arm], checkpoint.activation_count, common_horizon
        )
        for arm in ARMS
    }

    scheduler_rows = []
    for arm in ARMS:
        common_events = min(
            int(branches["no_switch"]["state"].activation_count - checkpoint.activation_count),
            int(branches[arm]["state"].activation_count - checkpoint.activation_count),
        )
        audit = scheduler_coupling_audit(
            scenarios["no_switch"], scenarios[arm], checkpoint.activation_count, common_events
        )
        scheduler_rows.append(
            {
                "condition_id": condition.condition_id,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "scenario_id": source.scenario_id,
                "arm": arm,
                **audit,
            }
        )

    policy_names_tuple = tuple(policy_names)
    before_reference, before_paper = _sortedness(values)
    primary_reference, primary_paper = _sortedness(value_primary)
    matched_reference, matched_paper = _sortedness(value_matched)
    original_edges = int(np.sum(original_binary[:-1] == original_binary[1:]))
    primary_edges = int(np.sum(primary_binary[:-1] == primary_binary[1:]))
    matched_edges = int(np.sum(matched_binary[:-1] == matched_binary[1:]))
    expected_strata = {
        str(stratum): int(original_binary[np.asarray(strata) == stratum].sum())
        for stratum in sorted(set(strata))
    }
    primary_strata = {
        str(stratum): int(primary_binary[np.asarray(strata) == stratum].sum())
        for stratum in sorted(set(strata))
    }
    matched_strata = {
        str(stratum): int(matched_binary[np.asarray(strata) == stratum].sum())
        for stratum in sorted(set(strata))
    }
    repeated = condition.input_profile == "repeated_1_10_x10"
    assignment_row = {
        "condition_id": condition.condition_id,
        "input_profile": condition.input_profile,
        "policy_set_label": condition.policy_set_label,
        "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
        "scenario_id": source.scenario_id,
        "checkpoint_progress": progress,
        "stratum_semantics": "exact_value" if repeated else "value_rank_decile",
        "s10_physical_shuffle_feasible": repeated,
        "direct_field_reassignment": True,
        "original_same_edges": original_edges,
        "exact_minimum_same_edges": int(policy_pair["optimum"]["minimumSameEdges"]),
        "support_qualified_same_edges": int(policy_pair["support"]["supportQualifiedSameEdges"]),
        "primary_same_edges": primary_edges,
        "matched_same_edges": matched_edges,
        "primary_policy_hamming": int(np.sum(primary_binary != original_binary)),
        "matched_policy_hamming": int(np.sum(matched_binary != original_binary)),
        "equal_hamming_group_size": int(policy_pair["equalHammingGroupSize"]),
        "demonstrated_supported_assignments": int(policy_pair["support"]["demonstratedDistinctAssignments"]),
        "support_bank_draws": int(policy_pair["support"]["supportBankDraws"]),
        "policy_bank_hash": policy_pair["bankHash"],
        "primary_policy_pattern_hash": hashlib.sha256(primary_binary.tobytes()).hexdigest(),
        "matched_policy_pattern_hash": hashlib.sha256(matched_binary.tobytes()).hexdigest(),
        "stratum_counts_before_json": json.dumps(expected_strata, sort_keys=True),
        "stratum_counts_primary_json": json.dumps(primary_strata, sort_keys=True),
        "stratum_counts_matched_json": json.dumps(matched_strata, sort_keys=True),
        "stratum_policy_counts_preserved": expected_strata == primary_strata == matched_strata,
        "primary_value_changed_count": int(np.sum(value_primary != np.asarray(values))),
        "matched_value_changed_count": int(np.sum(value_matched != np.asarray(values))),
        "primary_support_exact": bool(
            np.array_equal(
                np.flatnonzero(value_primary != np.asarray(values)), primary_support
            )
        ),
        "matched_support_exact": bool(
            np.array_equal(
                np.flatnonzero(value_matched != np.asarray(values)), matched_support
            )
        ),
        "value_multiset_primary_preserved": sorted(value_primary.tolist()) == sorted(values),
        "value_multiset_matched_preserved": sorted(value_matched.tolist()) == sorted(values),
        "value_pair_inversion_difference": int(value_pair["inversionDifference"]),
        "value_pair_total_change_difference": int(value_pair["totalValueChangeDifference"]),
        "value_primary_bank_hash": value_pair["primaryBankHash"],
        "value_matched_bank_hash": value_pair["matchedBankHash"],
        "reference_sortedness_before": before_reference,
        "reference_sortedness_value_primary": primary_reference,
        "reference_sortedness_value_matched": matched_reference,
        "paper_sortedness_before": before_paper,
        "paper_sortedness_value_primary": primary_paper,
        "paper_sortedness_value_matched": matched_paper,
        "inversions_before": _inversion_count(values),
        "inversions_value_primary": _inversion_count(value_primary),
        "inversions_value_matched": _inversion_count(value_matched),
        "total_value_change_primary": int(
            np.abs(value_primary - np.asarray(values)).sum()
        ),
        "total_value_change_matched": int(
            np.abs(value_matched - np.asarray(values)).sum()
        ),
        "policy_value_correlation_before": _association(values, original_binary),
        "policy_value_correlation_value_primary": _association(value_primary, original_binary),
        "policy_value_correlation_value_matched": _association(value_matched, original_binary),
        "policy_value_correlation_policy_primary": _association(values, primary_binary),
        "policy_value_correlation_policy_matched": _association(values, matched_binary),
        "immediate_policy_drop_primary": (original_edges - primary_edges) / 100.0,
        "immediate_policy_drop_matched": (original_edges - matched_edges) / 100.0,
        "policy_names_json": json.dumps(policy_names_tuple),
    }

    identity = {
        "condition_id": condition.condition_id,
        "input_profile": condition.input_profile,
        "policy_set_label": condition.policy_set_label,
        "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
        "scenario_id": source.scenario_id,
        "checkpoint_progress": progress,
        "checkpoint_target_swaps": target_swap,
        "checkpoint_activation": checkpoint.activation_count,
        "checkpoint_successful_swaps": checkpoint.ledger["acceptedSwaps"],
        "checkpoint_state_hash": state_hash(source.scenario_id, checkpoint),
    }
    run_rows = []
    trace_rows = []
    for arm in ARMS:
        final = branches[arm]["state"]
        run_rows.append(
            {
                **identity,
                "arm": arm,
                "common_activation_horizon": common_horizon,
                "post_checkpoint_activations": final.activation_count
                - checkpoint.activation_count,
                "post_checkpoint_successful_swaps": final.ledger["acceptedSwaps"]
                - checkpoint.ledger["acceptedSwaps"],
                "stop_reason": final.terminal,
                "final_activation_count": final.activation_count,
                "final_successful_swap_count": final.ledger["acceptedSwaps"],
                "final_state_hash": state_hash(scenarios[arm].scenario_id, final),
                "final_occupancy_hash": canonical_hash(final.occupancy),
                "final_cursor_hash": canonical_hash(final.selection_cursors),
                "initial_corrected_aggregation": float(
                    sampled[arm]["corrected_aggregation"][0]
                ),
                "final_corrected_aggregation": float(
                    sampled[arm]["corrected_aggregation"][-1]
                ),
                "initial_reference_sortedness": float(
                    sampled[arm]["reference_sortedness"][0]
                ),
                "final_reference_sortedness": float(
                    sampled[arm]["reference_sortedness"][-1]
                ),
            }
        )
        for grid_index, exposure in enumerate(GRID):
            trace_rows.append(
                {
                    **identity,
                    "arm": arm,
                    "grid_index": grid_index,
                    "common_activation_progress": float(exposure),
                    "target_activation": int(
                        sampled[arm]["target_activation"][grid_index]
                    ),
                    "accepted_swaps": int(sampled[arm]["accepted_swaps"][grid_index]),
                    "corrected_aggregation": float(
                        sampled[arm]["corrected_aggregation"][grid_index]
                    ),
                    "reference_sortedness": float(
                        sampled[arm]["reference_sortedness"][grid_index]
                    ),
                    "paper_sortedness": float(
                        sampled[arm]["paper_sortedness"][grid_index]
                    ),
                }
            )

    exact_rows = []
    no_branch, sham_branch = branches["no_switch"], branches["sham"]
    exact_rows.append(
        {
            **identity,
            "comparison": "no_switch_equals_sham",
            "final_state_equal": state_hash(
                scenarios["no_switch"].scenario_id, no_branch["state"]
            )
            == state_hash(scenarios["sham"].scenario_id, sham_branch["state"]),
            "activations_equal": bool(
                np.array_equal(no_branch["activations"], sham_branch["activations"])
            ),
            "swaps_equal": bool(np.array_equal(no_branch["swaps"], sham_branch["swaps"])),
            "metrics_equal": bool(
                np.array_equal(no_branch["metrics"], sham_branch["metrics"])
            ),
            "occupancies_equal": bool(
                np.array_equal(no_branch["occupancies"], sham_branch["occupancies"])
            ),
        }
    )
    exact_rows[0]["passed"] = all(
        exact_rows[0][key]
        for key in (
            "final_state_equal",
            "activations_equal",
            "swaps_equal",
            "metrics_equal",
            "occupancies_equal",
        )
    )
    initial_identities = {
        "value_primary_initial_aggregation_equals_no": np.isclose(
            sampled["value_primary"]["corrected_aggregation"][0],
            sampled["no_switch"]["corrected_aggregation"][0],
            atol=0,
            rtol=0,
        ),
        "value_matched_initial_aggregation_equals_no": np.isclose(
            sampled["value_matched"]["corrected_aggregation"][0],
            sampled["no_switch"]["corrected_aggregation"][0],
            atol=0,
            rtol=0,
        ),
        "policy_primary_initial_equals_both_primary": np.isclose(
            sampled["policy_primary"]["corrected_aggregation"][0],
            sampled["both_primary"]["corrected_aggregation"][0],
            atol=0,
            rtol=0,
        ),
        "policy_matched_initial_equals_both_matched": np.isclose(
            sampled["policy_matched"]["corrected_aggregation"][0],
            sampled["both_matched"]["corrected_aggregation"][0],
            atol=0,
            rtol=0,
        ),
    }
    factorial_row = {**identity, **initial_identities, "passed": bool(all(initial_identities.values()))}
    source_row = {
        **identity,
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
        "trace_rows": trace_rows,
        "preservation_rows": preservation_rows,
        "scheduler_rows": scheduler_rows,
        "exact_rows": exact_rows,
        "factorial_row": factorial_row,
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
    tasks = source_tasks()
    checkpoint = cache / "value_policy_continuations.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(checkpoint)
    pending = [
        task for task in tasks if task["metadata"]["scenario_id"] not in completed
    ]
    started = time.perf_counter()
    written = 0
    with checkpoint.open("a", buffering=1) as handle, ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[
                executor.submit(_worker, task, freeze["peakProgressByCondition"])
            ] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(
                    json.dumps(
                        _json_native(result), sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[
                        executor.submit(
                            _worker, task, freeze["peakProgressByCondition"]
                        )
                    ] = task
    observed = len(_completed_ids(checkpoint))
    accounting = {
        "schema": "e04.s11.run_accounting.v1",
        "researchStepId": "S11",
        "workers": workers,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "sourceScenariosExpected": 150,
        "sourceScenariosObserved": observed,
        "continuationsPerScenario": len(ARMS),
        "continuationsExpected": 150 * len(ARMS),
        "continuationsObserved": observed * len(ARMS),
        "traceRowsExpected": 150 * len(ARMS) * len(GRID),
        "preexisting": len(completed),
        "written": written,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "run_accounting.json", accounting)
    return accounting


def _read_corpus(cache: Path = CACHE_DIR) -> list[dict[str, Any]]:
    with (cache / "value_policy_continuations.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 150:
        raise AssertionError(f"expected 150 S11 records, observed {len(rows)}")
    return rows


def effect_trajectories(traces: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "condition_id",
        "input_profile",
        "policy_set_label",
        "replicate_ordinal",
        "scenario_id",
        "checkpoint_progress",
        "grid_index",
        "common_activation_progress",
    ]
    indexed = traces.set_index(keys + ["arm"])
    rows = []
    for metric in ("corrected_aggregation", "reference_sortedness"):
        y00 = indexed.xs("no_switch", level="arm")[metric]
        y10 = indexed.xs("value_primary", level="arm")[metric]
        y01 = indexed.xs("policy_primary", level="arm")[metric]
        y11 = indexed.xs("both_primary", level="arm")[metric]
        factorial = {
            "value_shapley": 0.5 * ((y10 - y00) + (y11 - y01)),
            "policy_shapley": 0.5 * ((y01 - y00) + (y11 - y10)),
            "interaction": y11 - y10 - y01 + y00,
            "joint": y11 - y00,
        }
        for component, effect in factorial.items():
            frame = effect.rename("effect").reset_index()
            frame["effect_family"] = "primary_factorial"
            frame["metric"] = metric
            frame["component"] = component
            rows.append(frame)
        for arm in (
            "value_primary",
            "value_matched",
            "policy_primary",
            "policy_matched",
        ):
            effect = indexed.xs(arm, level="arm")[metric] - y00
            frame = effect.rename("effect").reset_index()
            frame["effect_family"] = "assignment_sensitivity"
            frame["metric"] = metric
            frame["component"] = arm
            rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result[
        keys + ["effect_family", "metric", "component", "effect"]
    ]


def _curve_outcomes(curve: np.ndarray) -> dict[str, float]:
    return {
        "signed_area": float(np.trapezoid(curve, GRID)),
        "positive_area": float(np.trapezoid(np.maximum(curve, 0.0), GRID)),
        "negative_area": float(np.trapezoid(np.minimum(curve, 0.0), GRID)),
        "final_effect": float(curve[-1]),
        "peak_absolute_effect": float(np.max(np.abs(curve))),
        "time_peak_absolute": float(GRID[int(np.argmax(np.abs(curve)))]),
    }


def scenario_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "condition_id",
        "input_profile",
        "policy_set_label",
        "scenario_id",
        "replicate_ordinal",
        "effect_family",
        "metric",
        "component",
    ]
    for keys, group in effects.groupby(group_columns, sort=True):
        curve = group.sort_values("grid_index").effect.to_numpy()
        rows.append({**dict(zip(group_columns, keys)), **_curve_outcomes(curve)})
    return pd.DataFrame(rows)


def _bootstrap_curves(
    matrices: Sequence[np.ndarray], address: Sequence[Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    if not matrices or any(matrix.shape[1] != 101 for matrix in matrices):
        raise ValueError("bootstrap matrices must be nonempty with 101 columns")
    observed = np.mean(np.stack([matrix.mean(axis=0) for matrix in matrices]), axis=0)
    rng = np.random.Generator(
        np.random.PCG64DXSM(derive_seed("bootstrap", *address))
    )
    curves = np.empty((BOOTSTRAP_DRAWS, 101), dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 250):
        stop = min(start + 250, BOOTSTRAP_DRAWS)
        parts = []
        for matrix in matrices:
            indices = rng.integers(
                0, matrix.shape[0], size=(stop - start, matrix.shape[0])
            )
            parts.append(matrix[indices].mean(axis=1))
        curves[start:stop] = np.mean(np.stack(parts), axis=0)
    signed = np.trapezoid(curves, GRID, axis=1)
    final = curves[:, -1]
    peak = np.max(np.abs(curves), axis=1)
    summary = {
        **_curve_outcomes(observed),
        "signed_area_ci_low": float(np.quantile(signed, 0.025)),
        "signed_area_ci_high": float(np.quantile(signed, 0.975)),
        "signed_area_p_two_sided": float(
            min(
                1.0,
                2
                * min(
                    (1 + np.sum(signed <= 0)) / (BOOTSTRAP_DRAWS + 1),
                    (1 + np.sum(signed >= 0)) / (BOOTSTRAP_DRAWS + 1),
                ),
            )
        ),
        "final_effect_ci_low": float(np.quantile(final, 0.025)),
        "final_effect_ci_high": float(np.quantile(final, 0.975)),
        "peak_absolute_effect_ci_low": float(np.quantile(peak, 0.025)),
        "peak_absolute_effect_ci_high": float(np.quantile(peak, 0.975)),
    }
    curve_summary = np.column_stack(
        [
            observed,
            np.quantile(curves, 0.025, axis=0),
            np.quantile(curves, 0.975, axis=0),
        ]
    )
    return curve_summary, summary


def summarize_effects(
    effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_rows = []
    pooled_rows = []
    condition_curve_rows = []
    pooled_curve_rows = []
    identities = ["effect_family", "metric", "component"]
    for identity, subset in effects.groupby(identities, sort=True):
        family, metric, component = identity
        matrices = []
        for condition_id in CONDITIONS:
            condition = subset[subset.condition_id == condition_id]
            matrix = condition.pivot(
                index="scenario_id", columns="grid_index", values="effect"
            ).sort_index(axis=1)
            if matrix.shape != (25, 101):
                raise AssertionError(f"effect matrix shape drift: {identity} {condition_id}")
            array = matrix.to_numpy()
            matrices.append(array)
            curve, summary = _bootstrap_curves(
                [array], ("condition", condition_id, *identity)
            )
            condition_rows.append(
                {
                    "condition_id": condition_id,
                    "input_profile": condition.input_profile.iloc[0],
                    "policy_set_label": condition.policy_set_label.iloc[0],
                    "effect_family": family,
                    "metric": metric,
                    "component": component,
                    "scenario_count": 25,
                    **summary,
                }
            )
            for grid_index, exposure in enumerate(GRID):
                condition_curve_rows.append(
                    {
                        "condition_id": condition_id,
                        "effect_family": family,
                        "metric": metric,
                        "component": component,
                        "grid_index": grid_index,
                        "common_activation_progress": float(exposure),
                        "effect": float(curve[grid_index, 0]),
                        "ci_low": float(curve[grid_index, 1]),
                        "ci_high": float(curve[grid_index, 2]),
                    }
                )
        curve, summary = _bootstrap_curves(matrices, ("pooled", *identity))
        pooled_rows.append(
            {
                "effect_family": family,
                "metric": metric,
                "component": component,
                "condition_count": 6,
                "scenario_count": 150,
                **summary,
            }
        )
        for grid_index, exposure in enumerate(GRID):
            pooled_curve_rows.append(
                {
                    "effect_family": family,
                    "metric": metric,
                    "component": component,
                    "grid_index": grid_index,
                    "common_activation_progress": float(exposure),
                    "effect": float(curve[grid_index, 0]),
                    "ci_low": float(curve[grid_index, 1]),
                    "ci_high": float(curve[grid_index, 2]),
                }
            )
    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(pooled_rows),
        pd.DataFrame(condition_curve_rows),
        pd.DataFrame(pooled_curve_rows),
    )


def deterministic_assignment_replay(
    assignments: pd.DataFrame, sample_count: int = 12
) -> pd.DataFrame:
    tasks = source_tasks()
    selected = sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            canonical_json_bytes(
                {
                    "namespace": SEED_NAMESPACE,
                    "stream": "assignment_replay_sample",
                    "scenarioId": task["metadata"]["scenario_id"],
                }
            )
        ).hexdigest(),
    )[:sample_count]
    indexed = assignments.set_index("scenario_id")
    rows = []
    for task in selected:
        condition = SweepCondition.from_dict(task["condition"])
        source_raw, _ = materialize_sweep_scenario(condition, task["base"])
        source = _source_labelled_copy(source_raw)
        expected = task["expected_native"]
        target = math.floor(
            PEAK_PROGRESS[condition.condition_id]
            * int(expected["successful_swap_count"])
        )
        _, checkpoints = _run_source(source, {"eligible_peak": target})
        checkpoint = checkpoints["eligible_peak"]
        values, policies, strata = positional_sequences(source, checkpoint)
        pair = select_policy_pair(strata, policies, source.scenario_id)
        primary_support = np.flatnonzero(
            pair["primaryBinary"] != pair["originalBinary"]
        )
        matched_support = np.flatnonzero(
            pair["matchedBinary"] != pair["originalBinary"]
        )
        value_pair = select_value_pair(
            values, primary_support, matched_support, source.scenario_id
        )
        expected_row = indexed.loc[source.scenario_id]
        checks = {
            "policy_bank": pair["bankHash"] == expected_row.policy_bank_hash,
            "policy_primary": hashlib.sha256(
                pair["primaryBinary"].tobytes()
            ).hexdigest()
            == expected_row.primary_policy_pattern_hash,
            "policy_matched": hashlib.sha256(
                pair["matchedBinary"].tobytes()
            ).hexdigest()
            == expected_row.matched_policy_pattern_hash,
            "value_primary_bank": value_pair["primaryBankHash"]
            == expected_row.value_primary_bank_hash,
            "value_matched_bank": value_pair["matchedBankHash"]
            == expected_row.value_matched_bank_hash,
        }
        rows.append(
            {
                "condition_id": condition.condition_id,
                "scenario_id": source.scenario_id,
                **{f"check_{key}": value for key, value in checks.items()},
                "passed": all(checks.values()),
            }
        )
    return pd.DataFrame(rows)


def _plots(
    condition_curves: pd.DataFrame,
    condition_effects: pd.DataFrame,
    assignments: pd.DataFrame,
    output: Path,
) -> None:
    factorial = condition_curves[
        (condition_curves.effect_family == "primary_factorial")
        & (condition_curves.metric == "corrected_aggregation")
    ]
    colors = {
        "value_shapley": "#4c78a8",
        "policy_shapley": "#e45756",
        "interaction": "#72b7b2",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for axis, condition_id in zip(axes.flat, CONDITIONS):
        for component, color in colors.items():
            group = factorial[
                (factorial.condition_id == condition_id)
                & (factorial.component == component)
            ]
            axis.plot(
                group.common_activation_progress,
                group.effect,
                color=color,
                linewidth=1.8,
                label=component.replace("_", " "),
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(
            condition_id.replace("S03-", "").replace("-P50-ABS", ""), fontsize=9
        )
        axis.set_xlabel("Common activation exposure")
    axes[0, 0].set_ylabel("Corrected aggregation effect")
    axes[1, 0].set_ylabel("Corrected aggregation effect")
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "value_policy_component_curves.png", dpi=180)
    fig.savefig(output / "value_policy_component_curves.svg")
    plt.close(fig)

    subset = condition_effects[
        (condition_effects.effect_family == "primary_factorial")
        & (condition_effects.metric == "corrected_aggregation")
        & condition_effects.component.isin(["value_shapley", "policy_shapley"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    labels = [item.replace("S03-", "").replace("-P50-ABS", "") for item in CONDITIONS]
    for axis, component in zip(axes, ("value_shapley", "policy_shapley")):
        group = subset[subset.component == component].set_index("condition_id").loc[list(CONDITIONS)]
        estimate = group.signed_area.to_numpy()
        low = group.signed_area_ci_low.to_numpy()
        high = group.signed_area_ci_high.to_numpy()
        y = np.arange(len(labels))
        axis.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="o",
            capsize=3,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, labels)
        axis.set_title(component.replace("_", " "))
        axis.set_xlabel("Signed area (95% paired bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output / "value_policy_component_effects.png", dpi=180)
    fig.savefig(output / "value_policy_component_effects.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(
        assignments.inversions_value_primary - assignments.inversions_before,
        assignments.reference_sortedness_value_primary
        - assignments.reference_sortedness_before,
        alpha=0.55,
        s=20,
        c=np.where(assignments.input_profile.str.startswith("repeated"), "#e45756", "#4c78a8"),
    )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("Immediate inversion-count change")
    axes[0].set_ylabel("Immediate reference-Sortedness change")
    axes[0].set_title("Value shuffle changes task progress")
    axes[1].scatter(
        assignments.immediate_policy_drop_primary,
        assignments.primary_policy_hamming / 100.0,
        alpha=0.55,
        s=20,
        c=np.where(assignments.input_profile.str.startswith("repeated"), "#e45756", "#4c78a8"),
    )
    axes[1].set_xlabel("Immediate corrected-adjacency drop")
    axes[1].set_ylabel("Changed-position fraction")
    axes[1].set_title("Support-qualified policy magnitude")
    fig.tight_layout()
    fig.savefig(output / "intervention_distance_and_progress.png", dpi=180)
    fig.savefig(output / "intervention_distance_and_progress.svg")
    plt.close(fig)


def analyze(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    freeze = assert_frozen(output)
    corpus = _read_corpus(cache)
    runs = pd.DataFrame([row for item in corpus for row in item["run_rows"]])
    traces = pd.DataFrame([row for item in corpus for row in item["trace_rows"]])
    preservation = pd.DataFrame(
        [row for item in corpus for row in item["preservation_rows"]]
    )
    scheduler = pd.DataFrame(
        [row for item in corpus for row in item["scheduler_rows"]]
    )
    exact = pd.DataFrame([row for item in corpus for row in item["exact_rows"]])
    factorial = pd.DataFrame([item["factorial_row"] for item in corpus])
    assignments = pd.DataFrame([item["assignment_row"] for item in corpus])
    sources = pd.DataFrame([item["source_row"] for item in corpus])
    effects = effect_trajectories(traces)
    scenarios = scenario_effects(effects)
    condition, pooled, condition_curves, pooled_curves = summarize_effects(effects)
    replay = deterministic_assignment_replay(assignments)

    _write_parquet(runs, output / "value_policy_shuffles.parquet")
    _write_parquet(traces, output / "continuation_trajectories.parquet")
    _write_parquet(effects, output / "component_effect_trajectories.parquet")
    _write_parquet(scenarios, output / "scenario_component_effects.parquet")
    _write_parquet(condition, output / "condition_component_effects.parquet")
    _write_parquet(pooled, output / "pooled_component_effects.parquet")
    _write_parquet(condition_curves, output / "condition_component_curves.parquet")
    _write_parquet(pooled_curves, output / "pooled_component_curves.parquet")
    _write_parquet(assignments, output / "intervention_assignment_audit.parquet")
    _write_parquet(preservation, output / "state_preservation_audit.parquet")
    _write_parquet(scheduler, output / "scheduler_and_random_stream_audit.parquet")
    _write_parquet(exact, output / "sham_replay_audit.parquet")
    _write_parquet(factorial, output / "factorial_identity_audit.parquet")
    _write_parquet(sources, output / "native_source_replay_audit.parquet")
    _write_parquet(replay, output / "deterministic_assignment_replay.parquet")

    source_pass = bool(len(sources) == 150 and sources.passed.all())
    exact_pass = bool(len(exact) == 150 and exact.passed.all())
    factorial_pass = bool(len(factorial) == 150 and factorial.passed.all())
    replay_pass = bool(len(replay) == 12 and replay.passed.all())
    common_preservation = [
        "runtime_key_preserved",
        "seed_preserved",
        "max_activations_preserved",
        "occupancy_preserved",
        "identity_bijection_preserved",
        "activation_preserved",
        "stream_counters_preserved",
        "ledger_preserved",
        "direction_fault_label_preserved",
        "global_value_multiset_preserved",
        "global_policy_multiset_preserved",
    ]
    preservation_common_pass = bool(preservation[common_preservation].all(axis=None))
    sham_rows = preservation[preservation.arm.isin(["no_switch", "sham"])]
    value_rows = preservation[
        preservation.arm.isin(["value_primary", "value_matched"])
    ]
    policy_rows = preservation[
        preservation.arm.isin(["policy_primary", "policy_matched", "policy_transfer"])
    ]
    both_rows = preservation[
        preservation.arm.isin(["both_primary", "both_matched", "both_transfer"])
    ]
    component_pass = bool(
        sham_rows.value_fields_preserved.all()
        and sham_rows.policy_fields_preserved.all()
        and sham_rows.identity_cursor_preserved.all()
        and value_rows.policy_fields_preserved.all()
        and value_rows.identity_cursor_preserved.all()
        and policy_rows.value_fields_preserved.all()
        and (~value_rows.value_fields_preserved).all()
        and (~policy_rows.policy_fields_preserved).all()
        and (~both_rows.value_fields_preserved).all()
        and (~both_rows.policy_fields_preserved).all()
    )
    assignment_pass = bool(
        len(assignments) == 150
        and (assignments.demonstrated_supported_assignments == 33).all()
        and (assignments.support_bank_draws == 250_000).all()
        and (
            assignments.primary_same_edges
            == assignments.support_qualified_same_edges
        ).all()
        and (
            assignments.matched_same_edges
            == assignments.support_qualified_same_edges
        ).all()
        and (
            assignments.primary_policy_hamming
            == assignments.matched_policy_hamming
        ).all()
        and (
            assignments.primary_value_changed_count
            == assignments.primary_policy_hamming
        ).all()
        and (
            assignments.matched_value_changed_count
            == assignments.matched_policy_hamming
        ).all()
        and assignments.primary_support_exact.all()
        and assignments.matched_support_exact.all()
        and assignments.value_multiset_primary_preserved.all()
        and assignments.value_multiset_matched_preserved.all()
        and assignments.stratum_policy_counts_preserved.all()
    )
    intervention_gate = bool(
        assignment_pass
        and assignments.immediate_policy_drop_primary.median() >= 0.10
        and (assignments.immediate_policy_drop_primary >= 0.05).mean() >= 0.90
    )
    scheduler_pass = bool(len(scheduler) == 1500 and scheduler.passed.all())
    accounting_pass = bool(
        len(runs) == 1500
        and len(traces) == 151_500
        and len(effects) == 242_400
        and len(scenarios) == 2400
        and len(condition) == 96
        and len(pooled) == 16
    )
    policy_audit = _source_policy_audit()
    upstream_pass = all(item["allPassed"] for item in freeze["upstream"].values())
    primary_arms = ["no_switch", "value_primary", "policy_primary", "both_primary"]
    primary_runs = runs[runs.arm.isin(primary_arms)].assign(
        complete=lambda frame: frame.stop_reason == "complete"
    )
    pooled_completion = primary_runs.groupby("arm").complete.mean()
    condition_completion = primary_runs.groupby(["arm", "condition_id"]).complete.mean()
    completion_gate = bool(
        runs.stop_reason.ne("invariant_error").all()
        and (pooled_completion >= 0.75).all()
        and (condition_completion >= 0.60).all()
    )
    validity = bool(
        source_pass
        and exact_pass
        and factorial_pass
        and replay_pass
        and preservation_common_pass
        and component_pass
        and scheduler_pass
        and accounting_pass
        and policy_audit["passed"]
        and upstream_pass
        and not S12_DIR.exists()
    )

    primary_factorial = pooled[
        (pooled.effect_family == "primary_factorial")
        & (pooled.metric == "corrected_aggregation")
    ].set_index("component")
    sensitivity = pooled[
        (pooled.effect_family == "assignment_sensitivity")
        & (pooled.metric == "corrected_aggregation")
    ].set_index("component")

    def excludes_zero(row: pd.Series) -> bool:
        return bool(row.signed_area_ci_low > 0 or row.signed_area_ci_high < 0)

    shapley_nonzero = all(
        excludes_zero(primary_factorial.loc[component])
        for component in ("value_shapley", "policy_shapley")
    )
    assignment_robust: dict[str, bool] = {}
    assignment_sign_flip: dict[str, bool] = {}
    for field in ("value", "policy"):
        left = sensitivity.loc[f"{field}_primary"]
        right = sensitivity.loc[f"{field}_matched"]
        assignment_robust[field] = bool(
            excludes_zero(left)
            and excludes_zero(right)
            and np.sign(left.signed_area) == np.sign(right.signed_area)
        )
        assignment_sign_flip[field] = bool(
            np.sign(left.signed_area) != np.sign(right.signed_area)
        )
    separable = bool(shapley_nonzero and all(assignment_robust.values()))
    supportive = bool(validity and intervention_gate and completion_gate and separable)
    contradictory = bool(
        not validity
        or not intervention_gate
        or not completion_gate
        or any(assignment_sign_flip.values())
        or (shapley_nonzero and not all(assignment_robust.values()))
    )
    classification = (
        "supportive"
        if supportive
        else ("constraining/contradictory" if contradictory else "null")
    )

    completion = (
        runs.groupby(["arm", "condition_id", "stop_reason"], sort=True)
        .size()
        .rename("runs")
        .reset_index()
    )
    _write_parquet(completion, output / "completion_accounting.parquet")
    validation = {
        "schema": "e04.s11.validation_summary.v1",
        "researchStepId": "S11",
        "sourceReplay": {"passed": source_pass, "rows": len(sources)},
        "shamReplay": {"passed": exact_pass, "rows": len(exact)},
        "factorialIdentities": {"passed": factorial_pass, "rows": len(factorial)},
        "deterministicAssignmentReplay": {
            "passed": replay_pass,
            "rows": len(replay),
        },
        "statePreservation": {
            "passed": preservation_common_pass and component_pass,
            "rows": len(preservation),
        },
        "assignmentAndTargetDistance": {
            "passed": assignment_pass,
            "rows": len(assignments),
        },
        "schedulerCoupling": {"passed": scheduler_pass, "rows": len(scheduler)},
        "policyObservationBoundary": policy_audit,
        "accounting": {
            "passed": accounting_pass,
            "runRows": len(runs),
            "traceRows": len(traces),
            "effectRows": len(effects),
        },
        "upstreamImmutability": {"passed": upstream_pass},
        "s12Absent": not S12_DIR.exists(),
        "validityPassed": validity,
        "interventionGatePassed": intervention_gate,
        "completionGatePassed": completion_gate,
        "separableContributionGatePassed": separable,
        "assignmentRobustness": assignment_robust,
        "assignmentSignFlip": assignment_sign_flip,
        "outcomeClassification": classification,
    }
    write_json(output / "validation_summary.json", validation)
    summary = {
        "schema": "e04.s11.analysis_summary.v1",
        "researchStepId": "S11",
        "outcomeClassification": classification,
        "gates": {
            "validity": validity,
            "intervention": intervention_gate,
            "completion": completion_gate,
            "separableContribution": separable,
        },
        "pooledCompletionByPrimaryArm": pooled_completion.to_dict(),
        "medianImmediatePolicyDrop": float(
            assignments.immediate_policy_drop_primary.median()
        ),
        "fractionPolicyDropAtLeast005": float(
            (assignments.immediate_policy_drop_primary >= 0.05).mean()
        ),
        "medianValueInversionChangePrimary": float(
            np.median(
                assignments.inversions_value_primary - assignments.inversions_before
            )
        ),
        "medianValueSortednessChangePrimary": float(
            np.median(
                assignments.reference_sortedness_value_primary
                - assignments.reference_sortedness_before
            )
        ),
        "factorialAggregation": primary_factorial.reset_index().to_dict(
            orient="records"
        ),
        "assignmentSensitivityAggregation": sensitivity.reset_index().to_dict(
            orient="records"
        ),
        "valueInterventionIsPureMediator": False,
        "uniqueResultsCrossS10PhysicalBoundary": True,
    }
    write_json(output / "analysis_summary.json", summary)
    _plots(condition_curves, condition, assignments, output)
    return summary


def report(output: Path = OUTPUT_DIR) -> None:
    validation = json.loads((output / "validation_summary.json").read_text())
    pooled = pd.read_parquet(output / "pooled_component_effects.parquet")
    condition = pd.read_parquet(output / "condition_component_effects.parquet")
    assignments = pd.read_parquet(output / "intervention_assignment_audit.parquet")
    runs = pd.read_parquet(output / "value_policy_shuffles.parquet")
    primary = pooled[
        (pooled.effect_family == "primary_factorial")
        & (pooled.metric == "corrected_aggregation")
    ].set_index("component")
    task = pooled[
        (pooled.effect_family == "primary_factorial")
        & (pooled.metric == "reference_sortedness")
    ].set_index("component")
    sensitivity = pooled[
        (pooled.effect_family == "assignment_sensitivity")
        & (pooled.metric == "corrected_aggregation")
    ].set_index("component")
    completion = (
        runs.assign(complete=runs.stop_reason == "complete")
        .groupby("arm")
        .complete.mean()
    )

    def estimate(row: pd.Series) -> str:
        return (
            f"{row.signed_area:.4f} "
            f"[{row.signed_area_ci_low:.4f}, {row.signed_area_ci_high:.4f}]"
        )

    component_rows = "\n".join(
        f"| {component.replace('_', ' ')} | {estimate(primary.loc[component])} | "
        f"{primary.loc[component].final_effect:.4f} | "
        f"{primary.loc[component].peak_absolute_effect:.4f} |"
        for component in ("value_shapley", "policy_shapley", "interaction", "joint")
    )
    sensitivity_rows = "\n".join(
        f"| {component.replace('_', ' ')} | {estimate(sensitivity.loc[component])} |"
        for component in (
            "value_primary",
            "value_matched",
            "policy_primary",
            "policy_matched",
        )
    )
    condition_primary = condition[
        (condition.effect_family == "primary_factorial")
        & (condition.metric == "corrected_aggregation")
        & condition.component.isin(["value_shapley", "policy_shapley"])
    ]
    condition_rows = []
    for condition_id in CONDITIONS:
        group = condition_primary[condition_primary.condition_id == condition_id].set_index(
            "component"
        )
        condition_rows.append(
            f"| {condition_id} | {estimate(group.loc['value_shapley'])} | "
            f"{estimate(group.loc['policy_shapley'])} |"
        )
    condition_table = "\n".join(condition_rows)
    policy_pairs = (
        condition_primary.groupby(["policy_set_label", "component"])
        .signed_area.mean()
        .unstack()
    )
    heterogeneity = "; ".join(
        f"{index}: value {row.value_shapley:.4f}, policy {row.policy_shapley:.4f}"
        for index, row in policy_pairs.iterrows()
    )
    classification = validation["outcomeClassification"]
    completion_text = ", ".join(
        f"{arm} {completion.get(arm, math.nan):.1%}"
        for arm in ("no_switch", "value_primary", "policy_primary", "both_primary")
    )
    caveat = (
        "Direct value reassignment changed sorting progress and policy–value association; "
        "it is not a pure mediator intervention. Unique-input arms cross S10's physical "
        "feasibility boundary by changing identity fields. Completion and policy-pair "
        "heterogeneity bound pooled interpretation."
    )
    recommended = (
        "Return S11 to the Chief Scientist for review. No S12 work has been started; "
        "issue a separate instruction only if the bounded intervention evidence warrants S12."
    )
    text = f"""# S11 — Separate value-position and policy-position effects: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S11 |
| Completion status | Complete; S11 only; S12 not started |
| Artifacts written | Frozen specification and preregistration; 1,500-row `value_policy_shuffles.parquet`; 151,500-row continuation corpus; 242,400-row component-effect corpus; scenario, condition, pooled, and curve summaries; intervention-distance, preservation, source-replay, sham, factorial, scheduler, deterministic-replay, completion, validation, provenance, test, and manifest records; three PNG/SVG figure pairs; this canonical report |
| Validation result | {'Passed' if validation['validityPassed'] else 'Failed'}: 150 source replays, 1,500 state-matched continuations, exact sham and factorial identities, component-specific preservation, target-distance pairing, 12 deterministic assignment replays, scheduler/runtime qualification, complete accounting, S01–S10 immutability, and S12 absence were audited |
| Outcome classification | **{classification}** |
| Caveats or blockers | {caveat} |
| Lay summary | Values and executable policies were independently rearranged at the same positions and followed under paired random schedules. The analysis can distinguish the response to these specified interventions, but the value intervention also moves the system backward or sideways on its sorting task, so it does not isolate a natural causal pathway. |
| Recommended next action | {recommended} |

## Frozen question and decision rule

S11 asked whether value arrangement and executable-policy arrangement make
separable contributions to aggregation persistence at state-matched,
exposure-qualified peaks. Before continuation outcomes, the design froze all
six balanced absent-association anchors, 25 holdout scenarios each, checkpoints,
rank strata, the S10-compatible 250,000-draw/33-assignment support rule,
primary/matched Hamming and edge distances, value derangement matching, ten
arms, Selection cursor rules, runtime-key semantics, common exposure, a 2x2
Shapley decomposition, bootstrap uncertainty, and decision gates.

The supportive rule required validity, intervention, and S10-informed completion
gates plus nonzero pooled value and policy Shapley signed areas whose direct
primary/matched assignment intervals excluded zero in the same direction.

## Inputs and provenance

- Governance: `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and `/workspace/RESEARCH_PLAN.md`.
- Supplied paper: attachment manifest, sidecar, and extracted paper Markdown. The paper keeps Value and Algotype identity fields fixed in natural runs; repeated values only partially dissociate sorting pressure from clustering and do not implement S11's field reassignments.
- E01: transition contract, release manifest, runtime-key fixture, identity-owned Selection cursor semantics, and counter-addressed scheduler.
- S01–S10: every canonical manifest and every listed artifact was rehashed before freeze. S10 contributed the unique-value physical-feasibility boundary, support-qualified magnitude, 76% completion warning, and heterogeneous policy-pair response.
- Source population: S07 native-control holdout scenarios for six balanced pairwise absent-association anchors, 25 scenarios per condition.
- Reproducible repository source: `analysis/value_policy_shuffles.py`, `analysis/s11_value_policy_contract.json`, and focused tests at Git commit `{_git_output('rev-parse', 'HEAD')}` on `eidosoma/groups/28`.

## Detailed methods

### Checkpoints and pairing

Each source was replayed to its validated native endpoint and captured immediately
after the scenario-specific accepted-swap target at the earliest S06 corrected-
adjacency maximum no later than 75% progress. Checkpoints were 0.16, 0.36, and
0.75 for repeated Bubble+Insertion, Bubble+Selection, and Insertion+Selection,
and 0.19, 0.33, and 0.27 for their unique counterparts. All ten arms began from
the same occupancy, counters, ledger, seed, runtime key, and global event index.

### Policy-position intervention

Ten value-rank strata of ten identities were frozen from source values. For
repeated inputs these are exact-value groups; for unique inputs they are deciles,
because exact value conditioning would fix every policy and reproduce S10's
infeasibility. Within each stratum, policy counts were held exact. An MILP
recorded the exact minimum same-policy edge count. A deterministic 250,000-draw
bank selected the smallest edge bin containing 33 distinct assignments; primary
and matched assignments came from an equal-Hamming subgroup. The median
immediate corrected-adjacency drop was
{assignments.immediate_policy_drop_primary.median():.4f};
{(assignments.immediate_policy_drop_primary >= 0.05).mean():.1%} dropped by at
least 0.05.

### Value-position intervention and target distance

For each policy target, values were permuted only on its changed-position
support, and every supported position had to receive a different value. The
global value multiset and executable policy at every position remained exact.
Primary and matched derangement banks contained 128 candidates; the chosen pair
minimized inversion-count difference and then total value-change difference.
Thus value and policy arms changed the same number of positions, while inversion
count, strict/non-strict Sortedness, total value change, and policy–value
correlation remained visible diagnostics.

The median primary value intervention changed inversion count by
{np.median(assignments.inversions_value_primary - assignments.inversions_before):.1f}
and reference Sortedness by
{np.median(assignments.reference_sortedness_value_primary - assignments.reference_sortedness_before):.4f}.
This is an explicit change in task progress, not a mediator held apart from the
sorting task.

### State, cursor, and random-stream semantics

No identity moved. Direction, fault, analysis label, occupancy, identity set,
activation count, stream counters, ledger, seed, maximum budget, scheduler, and
source scenario ID were exact. Value-only preserved every policy field;
policy-only preserved every value field; joint applied both assignments on the
frozen support. The primary Selection rule retained cursors for identities that
remained Selection and reset newly Selection identities to zero. Donor transfer
was a prespecified sensitivity.

Scheduled actor identity and counter draws matched exactly at every sampled
common event. Bubble-side random queries matched for unchanged-policy actors;
query changes were permitted only when the intervention changed that actor's
executable policy. Terminated arms were carried forward on the 101-point common
activation grid, while completion remained separately reported and gated.

### Metrics, decomposition, and uncertainty

Aggregation was paper adjacency minus the exact balanced expectation 0.49.
Task progress used non-strict reference Sortedness; strict paper Sortedness and
inversions were distance diagnostics. With no-switch `Y00`, value-only `Y10`,
policy-only `Y01`, and joint `Y11`, the value Shapley curve was one half of
`(Y10-Y00)+(Y11-Y01)`, the policy Shapley curve one half of
`(Y01-Y00)+(Y11-Y10)`, interaction was `Y11-Y10-Y01+Y00`, and joint was
`Y11-Y00`. Signed, positive, negative, final, and peak-absolute effects used
complete common-exposure curves. Ten-thousand deterministic paired bootstrap
resamples were used within condition and stratified across six conditions.

## Results

### Pooled aggregation component effects

| Component | Signed area, 95% CI | Final effect | Peak absolute effect |
| --- | ---: | ---: | ---: |
{component_rows}

The corresponding task-progress signed areas were value
{estimate(task.loc['value_shapley'])}, policy
{estimate(task.loc['policy_shapley'])}, interaction
{estimate(task.loc['interaction'])}, and joint
{estimate(task.loc['joint'])}.

### Assignment-sensitivity bounds

| Direct arm | Signed aggregation area, 95% CI |
| --- | ---: |
{sensitivity_rows}

The matched assignments have the same supported policy edge count, policy
Hamming distance, and value-support size as their primaries. The frozen
assignment-robustness gates were value=
{validation['assignmentRobustness']['value']} and policy=
{validation['assignmentRobustness']['policy']}; sign flips were value=
{validation['assignmentSignFlip']['value']} and policy=
{validation['assignmentSignFlip']['policy']}.

### Condition and policy-pair heterogeneity

| Condition | Value Shapley signed area, 95% CI | Policy Shapley signed area, 95% CI |
| --- | ---: | ---: |
{condition_table}

Policy-pair means (averaging unique and repeated anchors) were: {heterogeneity}.
This spread is retained as a primary limitation on the pooled estimate, not
treated as sampling noise to be averaged away.

### Completion

Primary completion was {completion_text}. The frozen completion gate was
{validation['completionGatePassed']}. Full arm-by-condition stop accounting is
in `completion_accounting.parquet`; absorbing curves do not count as completion.

## Validation

- Native source replay: {validation['sourceReplay']['rows']}/150; passed={validation['sourceReplay']['passed']}.
- State/component preservation: {validation['statePreservation']['rows']}/1,500; passed={validation['statePreservation']['passed']}.
- Assignment and target-distance constraints: {validation['assignmentAndTargetDistance']['rows']}/150; passed={validation['assignmentAndTargetDistance']['passed']}.
- Exact sham comparisons: {validation['shamReplay']['rows']}/150; passed={validation['shamReplay']['passed']}.
- Factorial identities at intervention: {validation['factorialIdentities']['rows']}/150; passed={validation['factorialIdentities']['passed']}.
- Deterministic assignment replays: {validation['deterministicAssignmentReplay']['rows']}/12; passed={validation['deterministicAssignmentReplay']['passed']}.
- Scheduler/runtime audits: {validation['schedulerCoupling']['rows']}/1,500; passed={validation['schedulerCoupling']['passed']}.
- Accounting: 1,500 run rows, 151,500 continuation rows, and 242,400 effect rows; passed={validation['accounting']['passed']}.
- Policy observation boundary, S01–S10 byte immutability, S12 absence, repository tests, compilation, lint, and final artifact checks are recorded separately.

## Commands and dependencies

```bash
python -m pytest tests/test_e04_value_policy_shuffles.py tests/test_e04_declustering_intervention.py tests/test_e04_policy_label_switches.py -q --junitxml=/artifacts/research_steps/S11/repository_tests.junit.xml
python -m ruff check analysis/value_policy_shuffles.py tests/test_e04_value_policy_shuffles.py
python -m py_compile analysis/value_policy_shuffles.py
python -m analysis.value_policy_shuffles freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.value_policy_shuffles run --workers 8
python -m analysis.value_policy_shuffles analyze
python -m analysis.value_policy_shuffles report
python -m analysis.value_policy_shuffles package
```

No dependency was installed. The preinstalled Python, NumPy, pandas, PyArrow,
Matplotlib, SciPy/HiGHS (through the validated S10 optimizer), and clean-room
reference simulator were used. Simulation used eight process workers and one
thread per numerical library.

## Caveats, failed assumptions, and limitations

- Value shuffling reassigns the explicit sort key. Its immediate progress shift and changed policy–value correlation make it a task perturbation, not a pure mediator intervention or natural indirect effect.
- S10's exact value-sequence-preserving physical shuffle remains impossible for unique values. S11 obtains unique results only by directly changing identity fields and preserving coarse decile counts; these are not physical continuations of the same intervention class.
- Policy reassignment changes Algotype and Selection state. Retain-old/reset-new is primary; donor transfer is a model-dependent sensitivity.
- The support-qualified target demonstrates 33 assignments in a fixed draw bank; it is not proof of the globally strongest bin with that support.
- Matched arms bound assignment sensitivity with one prespecified alternative each, not the full conditional distribution of outcomes over assignments.
- Common actor scheduling does not and should not force identical proposals after values or policies change. Policy-dependent Bubble queries diverge only when the actor's policy changes.
- Completion qualification is deliberately informed by S10's 76% primary completion, and all stop reasons remain visible. Carry-forward curves do not repair incomplete dynamics.
- Strong policy-pair and input-profile heterogeneity bounds pooled interpretation.
- All evidence is simulator-local and does not establish biological adhesion, affinity, cognition, intention, or a general causal decomposition.

## Artifact provenance

`freeze_record.json` binds the contract, implementation, eligible checkpoints,
source population, S01–S10 manifests, S12 absence, and repository commit before
outcomes. `artifact_manifest.json`, `provenance.json`, `environment.json`, and
`commands.json` bind final files, source hashes, runtime, workers, and commands.
Compact evidence is under `/artifacts/research_steps/S11`; disposable restartable
JSONL remains under `/cache/e04_s11`.

## Recommended next action

{recommended}
"""
    (output / "research_step_full_results.md").write_text(text, encoding="utf-8")


def package(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> None:
    freeze = assert_frozen(output)
    validation = json.loads((output / "validation_summary.json").read_text())
    environment = {
        "schema": "e04.s11.environment.v1",
        "researchStepId": "S11",
        "python": sys.version,
        "platform": platform.platform(),
        "cpuCount": os.cpu_count(),
        "workers": 8,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "newDependenciesInstalled": [],
    }
    commands = {
        "schema": "e04.s11.commands.v1",
        "researchStepId": "S11",
        "commands": [
            "python -m pytest tests/test_e04_value_policy_shuffles.py tests/test_e04_declustering_intervention.py tests/test_e04_policy_label_switches.py -q --junitxml=/artifacts/research_steps/S11/repository_tests.junit.xml",
            "python -m ruff check analysis/value_policy_shuffles.py tests/test_e04_value_policy_shuffles.py",
            "python -m py_compile analysis/value_policy_shuffles.py",
            "python -m analysis.value_policy_shuffles freeze",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.value_policy_shuffles run --workers 8",
            "python -m analysis.value_policy_shuffles analyze",
            "python -m analysis.value_policy_shuffles report",
            "python -m analysis.value_policy_shuffles package",
        ],
    }
    provenance = {
        "schema": "e04.s11.provenance.v1",
        "researchStepId": "S11",
        "repository": str(REPOSITORY),
        "branch": _git_output("branch", "--show-current"),
        "gitHead": _git_output("rev-parse", "HEAD"),
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": sha256_file(CONTRACT_PATH),
        "implementationPath": str(Path(__file__).resolve()),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
        "freezeRecordSha256": sha256_file(output / "freeze_record.json"),
        "upstreamManifestHashes": EXPECTED_MANIFEST_HASHES,
        "cachePath": str(cache / "value_policy_continuations.jsonl"),
        "cacheCollectible": False,
        "paperPath": "/workspace/input-attachments/21c74f06-0617-5f83-901f-4579974d05d4/pdf-markdown.md",
        "e01Context": "/previous-artifacts/E01/research_steps/S03/transition_spec.md",
    }
    write_json(output / "environment.json", environment)
    write_json(output / "commands.json", commands)
    write_json(output / "provenance.json", provenance)
    status = {
        "researchStepId": "S11",
        "stepNumber": 11,
        "success": bool(validation["validityPassed"]),
        "status": "complete",
        "artifactsWritten": sorted(
            path.name for path in output.iterdir() if path.is_file()
        ),
        "validationResult": "passed" if validation["validityPassed"] else "failed",
        "caveatsOrBlockers": [
            "Value reassignment changes task progress and is not a pure mediator intervention.",
            "Unique results cross S10's physical feasibility boundary by direct field reassignment.",
            "Completion and policy-pair heterogeneity bound pooled interpretation.",
        ],
        "recommendedNextAction": "Return S11 for Chief Scientist review; do not start S12 without a separate instruction.",
    }
    write_json(output / "status.json", status)
    manifest_entries = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        manifest_entries.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "e04.s11.artifact_manifest.v1",
        "researchStepId": "S11",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "artifactCount": len(manifest_entries),
        "artifacts": manifest_entries,
        "validationPassed": validation["validityPassed"],
        "outcomeClassification": validation["outcomeClassification"],
        "s12Started": False,
        "freeze": {
            "contractSha256": freeze["contractSha256"],
            "implementationSha256": freeze["implementationSha256"],
        },
    }
    write_json(output / "artifact_manifest.json", manifest)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("analyze")
    subparsers.add_parser("report")
    subparsers.add_parser("package")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_design()
    elif args.command == "run":
        run_corpus(workers=args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        report()
    elif args.command == "package":
        package()


if __name__ == "__main__":
    main()
