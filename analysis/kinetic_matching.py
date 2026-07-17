"""E04 S07 label-blind kinetic matching interventions and analysis.

The intervention, split, feasibility, and endpoint rules are frozen in
``analysis/s07_kinetic_matching_contract.json``.  Native policy proposals are
never rewritten: a state-blind scheduler may select the actor and a transparent
actuator may reject an otherwise valid swap.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
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

from analysis.chimeric_replication import (
    AdmissibilityTracker,
    GRID,
    MetricTracker,
    canonical_hash,
)
from analysis.composition_sweep import (
    SweepCondition,
    build_conditions,
    load_base_draws,
    materialize_sweep_scenario,
)
from analysis.dynamic_nulls import (
    _label_curves,
    _value_strata,
    mobility_strata,
    trajectory_outcomes,
)
from reference_simulator.engine import evaluate_terminal, initial_state
from reference_simulator.model import canonical_json_bytes, state_hash
from reference_simulator.rng import u64
from reference_simulator.scheduler import scheduled_actor, scheduled_side


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
CONTRACT_PATH = REPOSITORY / "analysis/s07_kinetic_matching_contract.json"
OUTPUT_DIR = Path("/artifacts/research_steps/S07")
S08_DIR = Path("/artifacts/research_steps/S08")
UPSTREAM_DIRS = {f"S{index:02d}": Path(f"/artifacts/research_steps/S{index:02d}") for index in range(1, 7)}
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
    "S05": "8a9528c5f14abb3598cc7e1f41a145978f7f7d47717c23497623f5c00579b4ce",
    "S06": "e8137e96571fdd281629bf6d0c32e6c8bbbc081ede2953f19ca9c11f957aa786",
}
CALIBRATION_REPLICATES = tuple(range(1, 250, 10))
HOLDOUT_REPLICATES = tuple(range(5, 250, 10))
ANCHOR_POLICY_SETS = {"Bubble+Insertion", "Bubble+Selection", "Insertion+Selection"}
REGIMES = (
    "native_control",
    "activation_equal",
    "displacement_equal",
    "target_range_equal",
    "success_equal",
    "joint_equal",
)
NULL_FAMILIES = (
    "label_permuted_global",
    "mobility_matched",
    "label_permuted_value_stratified",
)
OUTCOMES = ("peak", "positive_area", "duration")
PRIMARY_OUTCOMES = ("peak", "positive_area")
REFERENCE_DRAWS = 500
CALIBRATION_DRAWS = 12
TOTAL_DRAWS = REFERENCE_DRAWS + CALIBRATION_DRAWS
RETAINED_DRAWS = 20
POLICY_NAMES = ("Bubble", "Insertion", "Selection")
POLICY_CODES = {name: index for index, name in enumerate(POLICY_NAMES)}
EXPECTED_CONDITIONS = 186
EXPECTED_HOLDOUT_SCENARIOS = EXPECTED_CONDITIONS * len(HOLDOUT_REPLICATES)
BOOTSTRAP_DRAWS = 10_000
SEED_NAMESPACE = "E04/S07/kinetic_matching/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {"namespace": SEED_NAMESPACE, "stream": stream, "address": address}
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big")


def _verify_manifest(step: str) -> dict[str, Any]:
    directory = UPSTREAM_DIRS[step]
    path = directory / "artifact_manifest.json"
    manifest = json.loads(path.read_text())
    checks = []
    for item in manifest["artifacts"]:
        artifact = directory / item["path"]
        observed = sha256_file(artifact) if artifact.is_file() else None
        checks.append({"path": item["path"], "expected": item["sha256"], "observed": observed, "passed": observed == item["sha256"]})
    manifest_hash = sha256_file(path)
    return {
        "step": step,
        "manifestPath": str(path),
        "expectedManifestSha256": EXPECTED_MANIFEST_HASHES[step],
        "observedManifestSha256": manifest_hash,
        "manifestPassed": manifest_hash == EXPECTED_MANIFEST_HASHES[step],
        "artifactChecks": checks,
        "allPassed": manifest_hash == EXPECTED_MANIFEST_HASHES[step] and all(item["passed"] for item in checks),
    }


def _calibration_conditions() -> tuple[SweepCondition, ...]:
    selected = tuple(
        condition
        for condition in build_conditions()
        if condition.policy_set_label in ANCHOR_POLICY_SETS
        and condition.first_policy_count == 50
        and condition.correlation_profile == "absent"
    )
    if len(selected) != 6:
        raise AssertionError(f"expected six calibration anchors, found {len(selected)}")
    return selected


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S07 frozen kinetic-matching specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S07 |
| Completion status | Design and split frozen before calibration or holdout simulation |
| Artifacts written | `preregistration.json`, `freeze_record.json`, and this specification |
| Validation result | Pre-simulation structure passed: six calibration anchors, 150 calibration scenarios, 186 holdout conditions, 4,650 holdout scenarios, six regimes, and disjoint replicate ordinals |
| Outcome classification | Pending S07 execution |
| Caveats or blockers | Selection's long-range native targets may make target-range and four-way joint matching infeasible without censoring; feasibility is tested, not assumed |
| Recommended next action | Calibrate only on the frozen calibration split, freeze parameters, then open the holdout; do not start S08 |

Contract SHA-256: `{contract_hash}`.

## Intervention boundary

The runtime may use actor identity, executable policy, the unchanged native
proposal, actor/target positions, event index, frozen parameters, and a
counter-addressed intervention draw. It may not read `analysis_label`, any
neighbor's policy label, aggregation, condition association label, future state,
or a holdout result. Native policy observations, sorting direction, proposal
kind, target position, and Selection cursor update are never rewritten.

Activation matching uses identity-cycle permutations that are blind to policy.
The other interventions can reject a mechanically valid native swap according
to a policy-indexed frozen actuator probability and, for range matching, native
target distance. Rejection charges the opportunity and leaves the state and
native policy memory unchanged.

## Split, calibration, and feasibility

Calibration uses the six balanced pairwise absent-association anchors and
replicate ordinals `1,11,...,241` (150 scenarios). Holdout uses all 186 S03
conditions and ordinals `5,15,...,245` (4,650 scenarios). The sets are disjoint.
No holdout outcome is readable until `calibration_models.json` is written and
hashed.

Balance is audited for per-cell activation opportunity, experienced spatial
displacement, executed target range, and successful swaps per activation.
Completion must be at least 95% in every calibration anchor and no more than five
percentage points below native control. A regime is feasible only when its named
balance tolerances and completion gate pass. If no candidate is feasible, the
lowest frozen objective is retained and explicitly labelled infeasible.

## Aggregation endpoints

Primary endpoints are the peak and positive area of the complete 101-point
composition-corrected condition-mean trajectory. They are adjusted against 500
global identity-label and mobility-matched null trajectories. Value-stratified
nulls are mechanism-sensitive diagnostics. Duration is descriptive only and
cannot determine the S07 outcome. BH correction is within regime, null family,
and outcome across 186 conditions. The six primary anchor reductions use paired
whole-scenario bootstrap intervals with familywise coverage.
"""


def freeze_design(output: Path, cache: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.iterdir()):
        raise FileExistsError("cannot freeze S07 after cache output exists")
    if S08_DIR.exists():
        raise AssertionError("S08 artifacts exist before S07 freeze")
    contract = json.loads(CONTRACT_PATH.read_text())
    contract_hash = sha256_file(CONTRACT_PATH)
    calibration = set(CALIBRATION_REPLICATES)
    holdout = set(HOLDOUT_REPLICATES)
    if calibration & holdout:
        raise AssertionError("calibration and holdout overlap")
    conditions = build_conditions()
    anchors = _calibration_conditions()
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed")
    record = {
        "schema": "e04.s07.freeze_record.v1",
        "researchStepId": "S07",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeCalibration": True,
        "frozenBeforeHoldout": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "conditionCount": len(conditions),
        "calibrationAnchorCount": len(anchors),
        "calibrationScenarioCount": len(anchors) * len(CALIBRATION_REPLICATES),
        "holdoutScenarioCount": len(conditions) * len(HOLDOUT_REPLICATES),
        "calibrationReplicates": list(CALIBRATION_REPLICATES),
        "holdoutReplicates": list(HOLDOUT_REPLICATES),
        "splitOverlap": sorted(calibration & holdout),
        "regimes": list(REGIMES),
        "nullFamilies": list(NULL_FAMILIES),
        "upstream": upstream,
        "s08Absent": not S08_DIR.exists(),
    }
    if record["calibrationScenarioCount"] != 150 or record["holdoutScenarioCount"] != EXPECTED_HOLDOUT_SCENARIOS:
        raise AssertionError("frozen population size changed")
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    (output / "kinetic_intervention_specification.md").write_text(_specification_markdown(contract_hash), encoding="utf-8")
    return record


def assert_frozen(output: Path) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("S07 contract changed after freeze")
    if not record["frozenBeforeCalibration"] or record["splitOverlap"]:
        raise AssertionError("S07 split was not validly frozen")
    if S08_DIR.exists():
        raise AssertionError("S08 artifacts appeared during S07")
    return record


@dataclass(frozen=True, slots=True)
class RegimeParameters:
    regime: str
    gate_probabilities: Mapping[str, float]
    range_kappas: Mapping[str, float]
    identity_cycle: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "gateProbabilities": {name: float(self.gate_probabilities.get(name, 1.0)) for name in POLICY_NAMES},
            "rangeKappas": {name: float(self.range_kappas.get(name, 0.0)) for name in POLICY_NAMES},
            "identityCycle": bool(self.identity_cycle),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RegimeParameters":
        return cls(
            regime=str(value["regime"]),
            gate_probabilities={str(k): float(v) for k, v in value["gateProbabilities"].items()},
            range_kappas={str(k): float(v) for k, v in value["rangeKappas"].items()},
            identity_cycle=bool(value["identityCycle"]),
        )


def _empty_policy_ledger() -> dict[str, dict[str, float]]:
    return {
        policy: {
            "actor_activations": 0.0,
            "valid_swap_proposals": 0.0,
            "accepted_swaps": 0.0,
            "native_range_sum": 0.0,
            "accepted_range_sum": 0.0,
            "actor_displacement_sum": 0.0,
            "experienced_displacement_sum": 0.0,
            "gate_rejections": 0.0,
            "memory_updates": 0.0,
            "no_ops": 0.0,
        }
        for policy in POLICY_NAMES
    }


class _IdentityCycleScheduler:
    def __init__(self, scenario: Any, regime: str) -> None:
        self.ids = tuple(cell.cell_id for cell in scenario.cells)
        self.scenario_id = scenario.scenario_id
        self.regime = regime
        self.cached_cycle = -1
        self.cached_order: np.ndarray | None = None

    def actor(self, event_index: int) -> str:
        cycle, offset = divmod(event_index, len(self.ids))
        if cycle != self.cached_cycle:
            rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("identity_cycle", self.scenario_id, self.regime, cycle)))
            self.cached_order = rng.permutation(len(self.ids))
            self.cached_cycle = cycle
        assert self.cached_order is not None
        return self.ids[int(self.cached_order[offset])]


def _gate_accept(scenario: Any, params: RegimeParameters, policy: str, distance: int, event_index: int) -> bool:
    probability = float(params.gate_probabilities.get(policy, 1.0))
    kappa = float(params.range_kappas.get(policy, 0.0))
    probability *= math.exp(-kappa * max(distance - 1, 0))
    probability = min(max(probability, 0.0), 1.0)
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    draw = u64(derive_seed("swap_gate", params.regime), scenario.scenario_id, "s07_swap_gate", event_index, 0)
    return draw < int(probability * (1 << 64))


def _encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _decode_array(value: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(value), dtype=dtype)
    if raw.size != math.prod(shape):
        raise ValueError("encoded array shape mismatch")
    return raw.reshape(shape).copy()


def _kinetic_values(ledger: Mapping[str, Mapping[str, float]], counts: Mapping[str, int], total_activations: int) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    population = sum(counts.values())
    for policy, count in counts.items():
        values = ledger[policy]
        actor_activations = float(values["actor_activations"])
        accepted = float(values["accepted_swaps"])
        valid = float(values["valid_swap_proposals"])
        expected_per_cell = total_activations / population if total_activations and population else math.nan
        per_cell = actor_activations / count if count else math.nan
        result[policy] = {
            **{key: float(value) for key, value in values.items()},
            "policy_count": float(count),
            "activation_opportunity": per_cell / expected_per_cell if expected_per_cell else None,
            "experienced_displacement_rate": (
                float(values["experienced_displacement_sum"]) / count / (total_activations / 10000)
                if total_activations and count else None
            ),
            "actor_displacement_rate": float(values["actor_displacement_sum"]) / actor_activations if actor_activations else None,
            "executed_target_range": float(values["accepted_range_sum"]) / accepted if accepted else None,
            "native_target_range": float(values["native_range_sum"]) / valid if valid else None,
            "successful_action_rate": accepted / actor_activations if actor_activations else None,
            "structural_identity_error": (
                abs(float(values["actor_displacement_sum"]) / actor_activations - (accepted / actor_activations) * (float(values["accepted_range_sum"]) / accepted))
                if actor_activations and accepted else 0.0
            ),
        }
    return result


def execute_kinetic_run(scenario: Any, params: RegimeParameters, *, validate_scenario: bool = True) -> dict[str, Any]:
    """Run one exact serial scenario with an optional label-blind kinetic gate."""
    if validate_scenario:
        scenario.validate()
    state = initial_state(scenario)
    state.terminal = evaluate_terminal(scenario, state)
    tracker = MetricTracker(scenario, False)
    admissibility = AdmissibilityTracker(scenario, state)
    policy_counts = {policy: sum(cell.policy.value == policy for cell in scenario.cells) for policy in POLICY_NAMES if any(cell.policy.value == policy for cell in scenario.cells)}
    policy_ledger = _empty_policy_ledger()
    points: list[dict[str, Any]] = [{"swap_index": 0, **tracker.metrics()}]
    initial_occupancy = tuple(state.occupancy)
    swaps: list[tuple[int, int]] = []
    cycle_scheduler = _IdentityCycleScheduler(scenario, params.regime) if params.identity_cycle else None
    start = time.perf_counter()

    while state.terminal is None:
        if state.activation_count >= scenario.max_activations:
            state.terminal = "event_budget"
            break
        event_index = state.activation_count
        if cycle_scheduler is None:
            actor_id, _, consumed = scheduled_actor(scenario, event_index, include_draws=False)
            state.stream_counters["actor_activation"] = state.stream_counters.get("actor_activation", 0) + consumed
        else:
            actor_id = cycle_scheduler.actor(event_index)
            state.stream_counters["s07_identity_cycle"] = state.stream_counters.get("s07_identity_cycle", 0) + 1
        actor = scenario.cell_map[actor_id]
        policy = actor.policy.value
        values = policy_ledger[policy]
        values["actor_activations"] += 1
        position = admissibility.positions[actor_id]
        reads = 0
        comparisons = 0
        outcome = "noop"
        target_position: int | None = None
        new_cursor: int | None = None

        if policy == "Bubble":
            side, _ = scheduled_side(scenario, event_index)
            state.stream_counters["bubble_side"] = state.stream_counters.get("bubble_side", 0) + 1
            target_position = position + (-1 if side == "left" else 1)
            if not 0 <= target_position < len(state.occupancy):
                reads = 1
                target_position = None
            else:
                reads, comparisons = 2, 1
                target = scenario.cell_map[state.occupancy[target_position]]
                inversion = actor.value < target.value if side == "left" else actor.value > target.value
                if inversion:
                    outcome = "swap"
        elif policy == "Insertion":
            if position == 0:
                reads = 1
            else:
                first_invalid = admissibility._first_invalid(True)
                if first_invalid is not None and first_invalid < position - 1:
                    reads = first_invalid + 3
                    comparisons = first_invalid + 1
                else:
                    reads = position + 2
                    comparisons = position
                    target_position = position - 1
                    target = scenario.cell_map[state.occupancy[target_position]]
                    if actor.value < target.value:
                        outcome = "swap"
        else:
            cursor = state.selection_cursors[actor_id]
            if not 0 <= cursor < len(state.occupancy) or cursor == position:
                reads = 1
            else:
                reads, comparisons = 2, 1
                target_position = cursor
                target = scenario.cell_map[state.occupancy[cursor]]
                if target.value <= actor.value:
                    outcome = "memory"
                    new_cursor = cursor + 1
                else:
                    outcome = "swap"

        state.ledger["activations"] += 1
        state.ledger["observationReads"] += reads
        state.ledger["valueComparisons"] += comparisons
        state.ledger["proposals"] += 1
        changed = False
        if outcome == "noop":
            state.ledger["noOps"] += 1
            values["no_ops"] += 1
        elif outcome == "memory":
            state.ledger["memoryUpdates"] += 1
            values["memory_updates"] += 1
            state.selection_cursors[actor_id] = int(new_cursor)
            admissibility.after_memory_update(state)
            changed = True
        else:
            assert target_position is not None
            distance = abs(target_position - position)
            values["valid_swap_proposals"] += 1
            values["native_range_sum"] += distance
            if _gate_accept(scenario, params, policy, distance, event_index):
                target_id = state.occupancy[target_position]
                target_policy = scenario.cell_map[target_id].policy.value
                state.occupancy[position], state.occupancy[target_position] = target_id, actor_id
                tracker.after_swap(state, position, target_position)
                admissibility.after_swap(state, position, target_position)
                state.ledger["acceptedSwaps"] += 1
                state.ledger["displacedCells"] += 2
                values["accepted_swaps"] += 1
                values["accepted_range_sum"] += distance
                values["actor_displacement_sum"] += distance
                values["experienced_displacement_sum"] += distance
                policy_ledger[target_policy]["experienced_displacement_sum"] += distance
                swaps.append((position, target_position))
                points.append({"swap_index": int(state.ledger["acceptedSwaps"]), **tracker.metrics()})
                changed = True
            else:
                state.ledger["rejections"] += 1
                values["gate_rejections"] += 1

        state.activation_count += 1
        if changed:
            state.terminal = admissibility.terminal_after_change(state)
        elif state.activation_count >= scenario.max_activations:
            state.terminal = "event_budget"

    elapsed = time.perf_counter() - start
    final_values = [scenario.cell_map[cell_id].value for cell_id in state.occupancy]
    final_swaps = len(swaps)
    target_indices = [math.floor(progress * final_swaps) for progress in GRID]
    occupancy = np.empty((len(GRID), len(scenario.cells)), dtype=np.uint8)
    id_to_index = {cell.cell_id: index for index, cell in enumerate(scenario.cells)}
    working = list(initial_occupancy)
    cursor = 0
    for grid_index, target_index in enumerate(target_indices):
        while cursor < target_index:
            first, second = swaps[cursor]
            working[first], working[second] = working[second], working[first]
            cursor += 1
        occupancy[grid_index] = np.fromiter((id_to_index[cell_id] for cell_id in working), dtype=np.uint8, count=len(working))
    grid_points = [points[index] for index in target_indices]
    baseline = sum(count * (count - 1) for count in policy_counts.values()) / (len(scenario.cells) ** 2)
    corrected = np.asarray([float(point["publication_aggregation"]) - baseline for point in grid_points], dtype=np.float64)
    kinetic = _kinetic_values(policy_ledger, policy_counts, state.activation_count)
    labels = np.asarray([POLICY_CODES[cell.policy.value] for cell in scenario.cells], dtype=np.uint8)
    identity_values = np.asarray([int(cell.value) for cell in scenario.cells], dtype=np.int16)
    final_ordered = all(left <= right for left, right in zip(final_values, final_values[1:]))
    return {
        "schema_version": "e04.s07.kinetic_run.v1",
        "scenario_id": scenario.scenario_id,
        "regime": params.regime,
        "stop_reason": state.terminal,
        "completed": state.terminal == "complete",
        "censored": state.terminal in {"event_budget", "invariant_error"},
        "activation_count": state.activation_count,
        "successful_swap_count": state.ledger["acceptedSwaps"],
        "gate_rejection_count": state.ledger["rejections"],
        "final_state_hash": state_hash(scenario.scenario_id, state),
        "final_consensus_ordered": final_ordered,
        "value_multiset_conserved": sorted(final_values) == sorted(cell.value for cell in scenario.cells),
        "trajectory_sha256": canonical_hash(points),
        "corrected_curve_sha256": hashlib.sha256(corrected.tobytes()).hexdigest(),
        "publication_aggregation_null": baseline,
        "corrected_peak": float(np.max(corrected)),
        "corrected_positive_area": float(trajectory_outcomes(corrected)["positive_area"][0]),
        "corrected_duration": float(trajectory_outcomes(corrected)["duration"][0]),
        "policy_counts_json": json.dumps(policy_counts, sort_keys=True, separators=(",", ":")),
        "policy_kinetics_json": json.dumps(_json_native(kinetic), sort_keys=True, separators=(",", ":")),
        "ledger_json": json.dumps(state.ledger, sort_keys=True, separators=(",", ":")),
        "elapsed_seconds": elapsed,
        "occupancy_b64": _encode_array(occupancy),
        "corrected_curve_b64": _encode_array(corrected),
        "labels_b64": _encode_array(labels),
        "values_b64": _encode_array(identity_values),
    }


def _task_metadata(condition: SweepCondition, base: Mapping[str, Any], metadata: Mapping[str, Any], scenario: Any) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "input_profile": condition.input_profile,
        "policy_set_label": condition.policy_set_label,
        "composition_class": condition.composition_class,
        "composition_profile": condition.composition_profile,
        "first_policy_count": condition.first_policy_count,
        "rare_policy": condition.rare_policy,
        "correlation_profile": condition.correlation_profile,
        "replicate_ordinal": int(base["replicateOrdinal"]),
        "base_draw_id": base["baseDrawId"],
        "pairing_block_id": base["pairingBlockId"],
        "scenario_json_sha256": metadata["scenarioJsonSha256"],
        "scenario_id": scenario.scenario_id,
    }


def build_tasks(split: str) -> list[dict[str, Any]]:
    bases = load_base_draws()
    if split == "calibration":
        conditions = _calibration_conditions()
        replicates = CALIBRATION_REPLICATES
    elif split == "holdout":
        conditions = build_conditions()
        replicates = HOLDOUT_REPLICATES
    else:
        raise ValueError("split must be calibration or holdout")
    tasks = []
    for condition in conditions:
        for replicate in replicates:
            base = bases[(condition.input_profile, replicate)]
            scenario, metadata = materialize_sweep_scenario(condition, base)
            tasks.append({"condition": condition.to_dict(), "base": dict(base), "metadata": _task_metadata(condition, base, metadata, scenario)})
    tasks.sort(key=lambda item: (item["condition"]["conditionId"], int(item["base"]["replicateOrdinal"])))
    return tasks


def _run_worker(task: Mapping[str, Any], params_dict: Mapping[str, Any]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    scenario, metadata = materialize_sweep_scenario(condition, task["base"])
    if metadata["scenarioJsonSha256"] != task["metadata"]["scenario_json_sha256"]:
        raise AssertionError("scenario rematerialization changed")
    result = execute_kinetic_run(scenario, RegimeParameters.from_dict(params_dict))
    result.update(task["metadata"])
    result["run_id"] = "e04s07:" + hashlib.sha256(f"{result['regime']}|{scenario.scenario_id}".encode()).hexdigest()
    return result


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ids.add(str(row["scenario_id"]))
    return ids


def run_tasks(tasks: Sequence[Mapping[str, Any]], params: RegimeParameters, checkpoint: Path, workers: int = 8) -> dict[str, Any]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(checkpoint)
    pending = [task for task in tasks if str(task["metadata"]["scenario_id"]) not in completed]
    started = time.perf_counter()
    written = 0
    with checkpoint.open("a", buffering=1) as handle, ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(_run_worker, task, params.to_dict())] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(json.dumps(_json_native(result), sort_keys=True, separators=(",", ":")) + "\n")
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(_run_worker, task, params.to_dict())] = task
    return {"regime": params.regime, "requested": len(tasks), "preexisting": len(completed), "written": written, "elapsedSeconds": time.perf_counter() - started, "checkpoint": str(checkpoint)}


def read_checkpoint(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _policy_long(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    output = []
    for row in rows:
        kinetics = json.loads(row["policy_kinetics_json"])
        for policy, values in kinetics.items():
            output.append({
                "regime": row["regime"],
                "condition_id": row["condition_id"],
                "input_profile": row["input_profile"],
                "policy_set_label": row["policy_set_label"],
                "replicate_ordinal": int(row["replicate_ordinal"]),
                "scenario_id": row["scenario_id"],
                "completed": bool(row["completed"]),
                "policy": policy,
                **values,
            })
    return pd.DataFrame(output)


def _relative_gap(left: float, right: float) -> float:
    denominator = (abs(left) + abs(right)) / 2
    return abs(left - right) / denominator if denominator > 0 else 0.0


def balance_audit(rows: Sequence[Mapping[str, Any]], named_targets: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    long = _policy_long(rows)
    metric_map = {
        "activationOpportunity": ("activation_opportunity", "absolute", 0.02, 0.04),
        "experiencedDisplacement": ("experienced_displacement_rate", "relative", 0.10, 0.20),
        "executedTargetRange": ("executed_target_range", "absolute", 0.25, 0.50),
        "successfulActionRate": ("successful_action_rate", "relative", 0.10, 0.20),
    }
    grouped = long.groupby(["condition_id", "policy"], sort=True).mean(numeric_only=True).reset_index()
    gap_rows = []
    for condition_id, group in grouped.groupby("condition_id", sort=True):
        policies = sorted(group.policy.tolist())
        for i, first in enumerate(policies):
            for second in policies[i + 1 :]:
                left = group[group.policy == first].iloc[0]
                right = group[group.policy == second].iloc[0]
                for target in named_targets:
                    column, kind, _, _ = metric_map[target]
                    a, b = float(left[column]), float(right[column])
                    gap = abs(a - b) if kind == "absolute" else _relative_gap(a, b)
                    gap_rows.append({"condition_id": condition_id, "first_policy": first, "second_policy": second, "target": target, "gap": gap, "first_value": a, "second_value": b})
    gaps = pd.DataFrame(gap_rows)
    target_results = {}
    for target in named_targets:
        _, _, median_tolerance, p90_tolerance = metric_map[target]
        values = gaps[gaps.target == target].gap.to_numpy(dtype=float)
        median = float(np.median(values)) if len(values) else math.inf
        p90 = float(np.quantile(values, 0.9)) if len(values) else math.inf
        target_results[target] = {"medianGap": median, "p90Gap": p90, "medianTolerance": median_tolerance, "p90Tolerance": p90_tolerance, "passed": median <= median_tolerance and p90 <= p90_tolerance}
    run_frame = pd.DataFrame(rows)
    completion = run_frame.groupby("condition_id", sort=True).completed.mean()
    completion_pass = bool((completion >= 0.95).all())
    structural_error = float(pd.to_numeric(long.structural_identity_error, errors="coerce").fillna(0).max())
    feasible = completion_pass and all(item["passed"] for item in target_results.values())
    objective = sum((item["medianGap"] / max(item["medianTolerance"], 1e-12)) ** 2 for item in target_results.values()) + 100 * max(0.0, 0.95 - float(completion.min()))
    summary = {
        "namedTargets": list(named_targets),
        "targetResults": target_results,
        "minimumConditionCompletion": float(completion.min()),
        "completionPassed": completion_pass,
        "structuralIdentityMaxError": structural_error,
        "feasible": feasible,
        "objective": float(objective),
    }
    return gaps, summary


def _pooled_native_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    long = _policy_long(rows)
    columns = ["experienced_displacement_rate", "executed_target_range", "successful_action_rate"]
    result = {}
    for policy, group in long.groupby("policy", sort=True):
        result[policy] = {column: float(group[column].mean()) for column in columns}
    return result


def _gate_from_means(means: Mapping[str, Mapping[str, float]], metric: str) -> dict[str, float]:
    target = min(float(values[metric]) for values in means.values() if float(values[metric]) > 0)
    return {policy: float(np.clip(target / float(values[metric]), 0.02, 1.0)) for policy, values in means.items()}


def _candidate_checkpoint(cache: Path, label: str) -> Path:
    return cache / "calibration" / f"{label}.jsonl"


def _calibration_run(tasks: Sequence[Mapping[str, Any]], params: RegimeParameters, cache: Path, workers: int, label: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _candidate_checkpoint(cache, label or params.regime)
    accounting = run_tasks(tasks, params, path, workers)
    rows = read_checkpoint(path)
    if len(rows) != len(tasks):
        raise AssertionError("calibration checkpoint incomplete")
    return rows, accounting


def calibrate_interventions(output: Path, cache: Path, workers: int = 8) -> dict[str, Any]:
    assert_frozen(output)
    if (output / "calibration_models.json").exists():
        raise FileExistsError("calibration models already frozen")
    tasks = build_tasks("calibration")
    accounting = []
    candidates: list[dict[str, Any]] = []
    gap_frames = []

    native = RegimeParameters("native_control", {p: 1.0 for p in POLICY_NAMES}, {p: 0.0 for p in POLICY_NAMES}, False)
    native_rows, item = _calibration_run(tasks, native, cache, workers)
    accounting.append(item)
    native_means = _pooled_native_means(native_rows)
    chosen: dict[str, dict[str, Any]] = {}

    def assess(params: RegimeParameters, rows: list[dict[str, Any]], targets: Sequence[str], label: str) -> dict[str, Any]:
        gaps, summary = balance_audit(rows, targets)
        gaps.insert(0, "candidate_label", label)
        gaps.insert(0, "regime", params.regime)
        gap_frames.append(gaps)
        record = {"candidateLabel": label, "parameters": params.to_dict(), **summary}
        candidates.append(record)
        return record

    native_record = assess(native, native_rows, [], "native_control")
    chosen["native_control"] = native_record

    activation = RegimeParameters("activation_equal", {p: 1.0 for p in POLICY_NAMES}, {p: 0.0 for p in POLICY_NAMES}, True)
    rows, item = _calibration_run(tasks, activation, cache, workers)
    accounting.append(item)
    chosen["activation_equal"] = assess(activation, rows, ["activationOpportunity"], "activation_equal")

    for regime, metric, target_name in (
        ("displacement_equal", "experienced_displacement_rate", "experiencedDisplacement"),
        ("success_equal", "successful_action_rate", "successfulActionRate"),
    ):
        probabilities = _gate_from_means(native_means, metric)
        records = []
        for iteration in range(3):
            params = RegimeParameters(regime, probabilities, {p: 0.0 for p in POLICY_NAMES}, False)
            label = f"{regime}_iter{iteration}"
            rows, item = _calibration_run(tasks, params, cache, workers, label)
            accounting.append(item)
            record = assess(params, rows, [target_name], label)
            records.append(record)
            if record["feasible"]:
                break
            observed = _pooled_native_means(rows)
            common = min(float(values[metric]) for values in native_means.values() if float(values[metric]) > 0)
            probabilities = {policy: float(np.clip(probabilities.get(policy, 1.0) * common / max(float(observed[policy][metric]), 1e-12), 0.02, 1.0)) for policy in observed}
        chosen[regime] = next((record for record in records if record["feasible"]), min(records, key=lambda record: record["objective"]))

    range_records = []
    for kappa in json.loads(CONTRACT_PATH.read_text())["calibration"]["targetRangeGrid"]:
        params = RegimeParameters("target_range_equal", {p: 1.0 for p in POLICY_NAMES}, {"Bubble": 0.0, "Insertion": 0.0, "Selection": float(kappa)}, False)
        label = f"target_range_equal_k{kappa:g}"
        rows, item = _calibration_run(tasks, params, cache, workers, label)
        accounting.append(item)
        range_records.append(assess(params, rows, ["executedTargetRange"], label))
    feasible_range = [record for record in range_records if record["feasible"]]
    completion_range = [record for record in range_records if record["completionPassed"]]
    if feasible_range:
        chosen["target_range_equal"] = feasible_range[0]
    elif completion_range:
        chosen["target_range_equal"] = min(completion_range, key=lambda record: record["objective"])
    else:
        chosen["target_range_equal"] = min(range_records, key=lambda record: record["objective"])

    success_probs = chosen["success_equal"]["parameters"]["gateProbabilities"]
    range_kappas = chosen["target_range_equal"]["parameters"]["rangeKappas"]
    probabilities = {policy: float(success_probs.get(policy, 1.0)) for policy in POLICY_NAMES}
    joint_records = []
    native_disp_target = min(values["experienced_displacement_rate"] for values in native_means.values())
    native_success_target = min(values["successful_action_rate"] for values in native_means.values())
    for iteration in range(3):
        params = RegimeParameters("joint_equal", probabilities, range_kappas, True)
        label = f"joint_equal_iter{iteration}"
        rows, item = _calibration_run(tasks, params, cache, workers, label)
        accounting.append(item)
        record = assess(params, rows, ["activationOpportunity", "experiencedDisplacement", "executedTargetRange", "successfulActionRate"], label)
        joint_records.append(record)
        if record["feasible"]:
            break
        observed = _pooled_native_means(rows)
        probabilities = {
            policy: float(np.clip(
                probabilities.get(policy, 1.0)
                * math.sqrt(
                    (native_disp_target / max(observed[policy]["experienced_displacement_rate"], 1e-12))
                    * (native_success_target / max(observed[policy]["successful_action_rate"], 1e-12))
                ),
                0.02,
                1.0,
            ))
            for policy in observed
        }
    chosen["joint_equal"] = next((record for record in joint_records if record["feasible"]), min(joint_records, key=lambda record: record["objective"]))

    candidates_frame = pd.json_normalize(candidates, sep="__")
    _write_parquet(candidates_frame, output / "calibration_candidates.parquet")
    gaps = pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    _write_parquet(gaps, output / "calibration_balance_gaps.parquet")
    models = {
        "schema": "e04.s07.calibration_models.v1",
        "researchStepId": "S07",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeHoldout": True,
        "contractSha256": sha256_file(CONTRACT_PATH),
        "calibrationReplicates": list(CALIBRATION_REPLICATES),
        "holdoutReplicates": list(HOLDOUT_REPLICATES),
        "nativeCalibrationMeans": native_means,
        "selected": chosen,
        "candidateCount": len(candidates),
        "accounting": accounting,
    }
    write_json(output / "calibration_models.json", models)
    model_hash = sha256_file(output / "calibration_models.json")
    record = {"schema": "e04.s07.parameter_freeze.v1", "researchStepId": "S07", "frozenBeforeHoldout": True, "calibrationModelsSha256": model_hash, "selectedFeasibility": {regime: bool(record["feasible"]) for regime, record in chosen.items()}, "holdoutRowsRead": 0}
    write_json(output / "parameter_freeze_record.json", record)
    return record


def assert_parameters_frozen(output: Path) -> dict[str, Any]:
    record = json.loads((output / "parameter_freeze_record.json").read_text())
    if record["calibrationModelsSha256"] != sha256_file(output / "calibration_models.json"):
        raise AssertionError("calibration models changed after parameter freeze")
    if not record["frozenBeforeHoldout"]:
        raise AssertionError("parameters were not frozen before holdout")
    return record


def run_holdout(output: Path, cache: Path, workers: int = 8) -> dict[str, Any]:
    assert_frozen(output)
    assert_parameters_frozen(output)
    tasks = build_tasks("holdout")
    models = json.loads((output / "calibration_models.json").read_text())
    accounting = []
    for regime in REGIMES:
        params = RegimeParameters.from_dict(models["selected"][regime]["parameters"])
        checkpoint = cache / "holdout" / f"{regime}.jsonl"
        accounting.append(run_tasks(tasks, params, checkpoint, workers))
    record = {"schema": "e04.s07.holdout_accounting.v1", "researchStepId": "S07", "expectedPerRegime": len(tasks), "regimes": accounting, "totalExpected": len(tasks) * len(REGIMES), "totalObserved": sum(len(read_checkpoint(cache / "holdout" / f"{regime}.jsonl")) for regime in REGIMES)}
    write_json(output / "holdout_accounting.json", record)
    return record


def _decode_run(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "occupancy": _decode_array(str(row["occupancy_b64"]), np.dtype(np.uint8), (len(GRID), 100)),
        "corrected_curve": _decode_array(str(row["corrected_curve_b64"]), np.dtype(np.float64), (len(GRID),)),
        "labels": _decode_array(str(row["labels_b64"]), np.dtype(np.uint8), (100,)),
        "values": _decode_array(str(row["values_b64"]), np.dtype(np.int16), (100,)),
    }


def _permuted_labels(labels: np.ndarray, strata: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    output = np.broadcast_to(labels, (TOTAL_DRAWS, len(labels))).copy()
    if strata is None:
        return rng.permuted(output, axis=1)
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        output[:, indices] = rng.permuted(output[:, indices], axis=1)
    return output


def _support_log(labels: np.ndarray, strata: np.ndarray) -> tuple[float, int]:
    log_support = 0.0
    mutable = 0
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        counts = np.bincount(labels[indices], minlength=len(POLICY_NAMES))
        log_support += math.lgamma(len(indices) + 1) - sum(math.lgamma(int(count) + 1) for count in counts)
        mutable += int(sum(count > 0 for count in counts) > 1)
    return log_support, mutable


def _null_file(null_dir: Path, regime: str, condition_id: str) -> Path:
    digest = hashlib.sha256(f"{regime}|{condition_id}".encode()).hexdigest()[:24]
    return null_dir / f"{digest}.npz"


def _null_worker(regime: str, condition_id: str, raw_rows: Sequence[Mapping[str, Any]], null_dir: Path) -> dict[str, Any]:
    rows = sorted((_decode_run(row) for row in raw_rows), key=lambda row: int(row["replicate_ordinal"]))
    if len(rows) != len(HOLDOUT_REPLICATES):
        raise AssertionError("S07 null group lacks 25 holdout scenarios")
    family_curves = {family: np.zeros((TOTAL_DRAWS, len(GRID)), dtype=np.float64) for family in NULL_FAMILIES}
    observed = np.mean(np.stack([row["corrected_curve"] for row in rows]), axis=0)
    count_violations = 0
    stratum_violations = 0
    value_support = []
    mobility_support = []
    pairing = []
    for row in rows:
        labels = row["labels"]
        values = row["values"]
        occupancy = row["occupancy"]
        baseline = float(row["publication_aggregation_null"])
        value_strata = _value_strata(values, str(row["input_profile"]))
        mobility, _, _ = mobility_strata(occupancy)
        value_support.append(_support_log(labels, value_strata))
        mobility_support.append(_support_log(labels, mobility))
        pairing.append(str(row["scenario_id"]))
        for family, strata in (
            ("label_permuted_global", None),
            ("mobility_matched", mobility),
            ("label_permuted_value_stratified", value_strata),
        ):
            rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("null_labels", regime, condition_id, row["scenario_id"], family)))
            permuted = _permuted_labels(labels, strata, rng)
            expected = np.bincount(labels, minlength=len(POLICY_NAMES))
            observed_counts = np.stack([np.sum(permuted == code, axis=1) for code in range(len(POLICY_NAMES))], axis=1)
            count_violations += int(np.sum(observed_counts != expected))
            if strata is not None:
                for stratum in np.unique(strata):
                    indices = np.flatnonzero(strata == stratum)
                    target = np.bincount(labels[indices], minlength=len(POLICY_NAMES))
                    realized = np.stack([np.sum(permuted[:, indices] == code, axis=1) for code in range(len(POLICY_NAMES))], axis=1)
                    stratum_violations += int(np.sum(realized != target))
            family_curves[family] += _label_curves(permuted, occupancy, baseline)
    for family in NULL_FAMILIES:
        family_curves[family] /= len(rows)
    path = _null_file(null_dir, regime, condition_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        regime=np.asarray(regime),
        condition_id=np.asarray(condition_id),
        families=np.asarray(NULL_FAMILIES),
        curves=np.stack([family_curves[family] for family in NULL_FAMILIES]),
        observed=observed,
        scenario_ids=np.asarray(pairing),
        replicate_ordinals=np.asarray([int(row["replicate_ordinal"]) for row in rows], dtype=np.int16),
        metadata_json=np.asarray(json.dumps({key: rows[0][key] for key in ("input_profile", "policy_set_label", "composition_class", "composition_profile", "first_policy_count", "rare_policy", "correlation_profile")}, sort_keys=True)),
        value_support=np.asarray(value_support, dtype=np.float64),
        mobility_support=np.asarray(mobility_support, dtype=np.float64),
        count_violations=np.asarray(count_violations, dtype=np.int64),
        stratum_violations=np.asarray(stratum_violations, dtype=np.int64),
    )
    return {"regime": regime, "conditionId": condition_id, "path": str(path), "sha256": sha256_file(path), "scenarioCount": len(rows), "groupTrajectories": len(NULL_FAMILIES) * TOTAL_DRAWS, "countViolations": count_violations, "stratumViolations": stratum_violations}


def run_dynamic_nulls(output: Path, cache: Path, workers: int = 8) -> dict[str, Any]:
    assert_parameters_frozen(output)
    null_dir = cache / "nulls"
    manifest_rows = []
    started = time.perf_counter()
    for regime in REGIMES:
        rows = read_checkpoint(cache / "holdout" / f"{regime}.jsonl")
        if len(rows) != EXPECTED_HOLDOUT_SCENARIOS:
            raise AssertionError(f"holdout checkpoint incomplete for {regime}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["condition_id"])].append(row)
        pending = [(condition_id, group) for condition_id, group in sorted(grouped.items()) if not _null_file(null_dir, regime, condition_id).exists()]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_null_worker, regime, condition_id, group, null_dir): condition_id for condition_id, group in pending}
            for future in futures:
                manifest_rows.append(future.result())
        for condition_id in sorted(grouped):
            path = _null_file(null_dir, regime, condition_id)
            if not any(item["regime"] == regime and item["conditionId"] == condition_id for item in manifest_rows):
                with np.load(path, allow_pickle=False) as data:
                    manifest_rows.append({"regime": regime, "conditionId": condition_id, "path": str(path), "sha256": sha256_file(path), "scenarioCount": len(data["scenario_ids"]), "groupTrajectories": len(NULL_FAMILIES) * TOTAL_DRAWS, "countViolations": int(data["count_violations"]), "stratumViolations": int(data["stratum_violations"])})
    manifest_rows.sort(key=lambda row: (row["regime"], row["conditionId"]))
    expected_files = len(REGIMES) * EXPECTED_CONDITIONS
    record = {
        "schema": "e04.s07.dynamic_null_accounting.v1",
        "researchStepId": "S07",
        "groupFiles": len(manifest_rows),
        "expectedGroupFiles": expected_files,
        "groupNullTrajectories": sum(int(row["groupTrajectories"]) for row in manifest_rows),
        "expectedGroupNullTrajectories": expected_files * len(NULL_FAMILIES) * TOTAL_DRAWS,
        "countViolations": sum(int(row["countViolations"]) for row in manifest_rows),
        "stratumViolations": sum(int(row["stratumViolations"]) for row in manifest_rows),
        "elapsedSeconds": time.perf_counter() - started,
        "files": manifest_rows,
    }
    if record["groupFiles"] != expected_files or record["countViolations"] or record["stratumViolations"]:
        raise AssertionError("S07 dynamic null accounting failed")
    write_json(output / "dynamic_null_accounting.json", record)
    return record


def _bh_adjust(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate((ranked * len(array) / np.arange(1, len(array) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


class _ParquetAppender:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def package_results(output: Path, cache: Path) -> dict[str, Any]:
    assert_parameters_frozen(output)
    run_rows = []
    policy_frames = []
    completion_rows = []
    observed_curve_rows = []
    for regime in REGIMES:
        rows = read_checkpoint(cache / "holdout" / f"{regime}.jsonl")
        policy_frames.append(_policy_long(rows))
        for row in rows:
            compact = {key: value for key, value in row.items() if not key.endswith("_b64")}
            run_rows.append(compact)
        frame = pd.DataFrame(rows)
        for condition_id, group in frame.groupby("condition_id", sort=True):
            completion_rows.append({
                "regime": regime,
                "condition_id": condition_id,
                "runs": len(group),
                "completed": int(group.completed.sum()),
                "completion_fraction": float(group.completed.mean()),
                "event_budget": int((group.stop_reason == "event_budget").sum()),
                "mean_activations": float(group.activation_count.mean()),
                "mean_accepted_swaps": float(group.successful_swap_count.mean()),
                "all_value_conserved": bool(group.value_multiset_conserved.all()),
                "all_complete_runs_ordered": bool(group.loc[group.completed, "final_consensus_ordered"].all()),
            })
            curves = np.stack([_decode_array(value, np.dtype(np.float64), (len(GRID),)) for value in group.corrected_curve_b64])
            mean = curves.mean(axis=0)
            for grid_index, progress in enumerate(GRID):
                observed_curve_rows.append({"regime": regime, "condition_id": condition_id, "progress": float(progress), "corrected_mean": float(mean[grid_index]), "corrected_sd": float(curves[:, grid_index].std(ddof=1)), "runs": len(group)})
    runs = pd.DataFrame(run_rows)
    _write_parquet(runs, output / "kinetic_matching.parquet")
    policy = pd.concat(policy_frames, ignore_index=True)
    _write_parquet(policy, output / "policy_kinetic_ledger.parquet")
    completion = pd.DataFrame(completion_rows)
    _write_parquet(completion, output / "completion_comparability.parquet")
    _write_parquet(pd.DataFrame(observed_curve_rows), output / "observed_kinetic_trajectories.parquet")

    statistics_rows = []
    response_rows = []
    effect_rows = []
    preservation_rows = []
    retained = _ParquetAppender(output / "null_distributions" / "retained_null_trajectories.parquet")
    for regime in REGIMES:
        for condition in build_conditions():
            path = _null_file(cache / "nulls", regime, condition.condition_id)
            with np.load(path, allow_pickle=False) as data:
                families = [str(item) for item in data["families"]]
                curves = data["curves"].astype(np.float64)
                observed = data["observed"].astype(np.float64)
                metadata = json.loads(str(data["metadata_json"]))
                scenario_ids = [str(item) for item in data["scenario_ids"]]
                pairing_hash = canonical_hash(scenario_ids)
                observed_outcomes = {name: float(values[0]) for name, values in trajectory_outcomes(observed).items()}
                preservation_rows.append({"regime": regime, "condition_id": condition.condition_id, "scenario_count": len(scenario_ids), "pairing_sha256": pairing_hash, "count_violations": int(data["count_violations"]), "stratum_violations": int(data["stratum_violations"]), "value_log_support_median": float(np.median(data["value_support"][:, 0])), "mobility_log_support_median": float(np.median(data["mobility_support"][:, 0]))})
                for family_index, family in enumerate(families):
                    family_curves = curves[family_index]
                    outcomes = trajectory_outcomes(family_curves)
                    for draw in range(TOTAL_DRAWS):
                        statistics_rows.append({"regime": regime, "condition_id": condition.condition_id, "null_family": family, "draw_index": draw, "draw_role": "reference" if draw < REFERENCE_DRAWS else "calibration", **{name: float(values[draw]) if math.isfinite(float(values[draw])) else None for name, values in outcomes.items()}})
                    retained_rows = []
                    for draw in range(RETAINED_DRAWS):
                        for grid_index, progress in enumerate(GRID):
                            retained_rows.append({"regime": regime, "condition_id": condition.condition_id, "null_family": family, "draw_index": draw, "progress": float(progress), "corrected_aggregation": float(family_curves[draw, grid_index])})
                    retained.write(pd.DataFrame(retained_rows))
                    for grid_index, progress in enumerate(GRID):
                        values = family_curves[:REFERENCE_DRAWS, grid_index]
                        response_rows.append({"regime": regime, "condition_id": condition.condition_id, "null_family": family, "progress": float(progress), "observed": float(observed[grid_index]), "null_mean": float(values.mean()), "null_sd": float(values.std(ddof=1)), "null_q025": float(np.quantile(values, 0.025)), "null_q975": float(np.quantile(values, 0.975)), **metadata})
                    for outcome in OUTCOMES:
                        reference = outcomes[outcome][:REFERENCE_DRAWS]
                        observed_value = observed_outcomes[outcome]
                        effect_rows.append({"regime": regime, "condition_id": condition.condition_id, "null_family": family, "outcome": outcome, "observed": observed_value, "null_mean": float(reference.mean()), "null_q025": float(np.quantile(reference, 0.025)), "null_q975": float(np.quantile(reference, 0.975)), "adjusted_effect": observed_value - float(reference.mean()), "effect_ci_low": observed_value - float(np.quantile(reference, 0.975)), "effect_ci_high": observed_value - float(np.quantile(reference, 0.025)), "p_value": float((1 + np.sum(reference >= observed_value)) / (REFERENCE_DRAWS + 1)), **metadata})
    retained.close()
    statistics = pd.DataFrame(statistics_rows)
    _write_parquet(statistics, output / "null_distributions" / "dynamic_null_statistics.parquet")
    _write_parquet(pd.DataFrame(response_rows), output / "null_distributions" / "null_response_surface.parquet")
    effects = pd.DataFrame(effect_rows)
    effects["q_value"] = np.nan
    for _, indices in effects.groupby(["regime", "null_family", "outcome"], sort=True).groups.items():
        effects.loc[list(indices), "q_value"] = _bh_adjust(effects.loc[list(indices), "p_value"])
    _write_parquet(effects, output / "null_adjusted_effects.parquet")
    preservation = pd.DataFrame(preservation_rows)
    _write_parquet(preservation, output / "null_preservation_audit.parquet")
    record = {
        "schema": "e04.s07.packaging_accounting.v1",
        "researchStepId": "S07",
        "kineticRunRows": len(runs),
        "policyKineticRows": len(policy),
        "completionRows": len(completion),
        "observedTrajectoryRows": len(observed_curve_rows),
        "nullStatisticRows": len(statistics),
        "nullResponseRows": len(response_rows),
        "effectRows": len(effects),
        "preservationRows": len(preservation),
    }
    write_json(output / "packaging_accounting.json", record)
    return record


def _randomized_p(reference: np.ndarray, observed: float, seed: int) -> float:
    greater = int(np.sum(reference > observed))
    tied = int(np.sum(reference == observed)) + 1
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return float((greater + rng.uniform(0.0, tied)) / (len(reference) + 1))


def null_calibration(output: Path) -> pd.DataFrame:
    frame = pq.read_table(output / "null_distributions" / "dynamic_null_statistics.parquet").to_pandas()
    rows = []
    for (regime, family), group in frame.groupby(["regime", "null_family"], sort=True):
        for outcome in OUTCOMES:
            randomized = []
            conservative = []
            for condition_id, condition in group.groupby("condition_id", sort=True):
                reference = condition[condition.draw_role == "reference"][outcome].dropna().to_numpy(dtype=float)
                pseudo = condition[condition.draw_role == "calibration"][outcome].dropna().to_numpy(dtype=float)
                if len(reference) != REFERENCE_DRAWS:
                    raise AssertionError("S07 null calibration reference count changed")
                for index, observed in enumerate(pseudo):
                    randomized.append(_randomized_p(reference, float(observed), derive_seed("calibration_tie", regime, family, outcome, condition_id, index)))
                    conservative.append(float((1 + np.sum(reference >= observed)) / (REFERENCE_DRAWS + 1)))
            randomized_array = np.asarray(randomized)
            conservative_array = np.asarray(conservative)
            for alpha in (0.01, 0.05, 0.1):
                standard_error = math.sqrt(alpha * (1 - alpha) / len(randomized_array))
                tolerance = max(0.015, 5 * standard_error)
                randomized_rate = float(np.mean(randomized_array <= alpha))
                conservative_rate = float(np.mean(conservative_array <= alpha))
                rows.append({
                    "regime": regime,
                    "null_family": family,
                    "outcome": outcome,
                    "alpha": alpha,
                    "pseudo_observations": len(randomized_array),
                    "randomized_rate": randomized_rate,
                    "conservative_rate": conservative_rate,
                    "tolerance": tolerance,
                    "randomized_passed": abs(randomized_rate - alpha) <= tolerance,
                    "conservative_passed": conservative_rate <= alpha + tolerance,
                })
    result = pd.DataFrame(rows)
    _write_parquet(result, output / "dynamic_null_calibration.parquet")
    return result


def _regime_targets(regime: str) -> list[str]:
    return {
        "native_control": [],
        "activation_equal": ["activationOpportunity"],
        "displacement_equal": ["experiencedDisplacement"],
        "target_range_equal": ["executedTargetRange"],
        "success_equal": ["successfulActionRate"],
        "joint_equal": ["activationOpportunity", "experiencedDisplacement", "executedTargetRange", "successfulActionRate"],
    }[regime]


def holdout_balance(output: Path, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_gaps = []
    summaries = []
    models = json.loads((output / "calibration_models.json").read_text())
    for regime in REGIMES:
        rows = read_checkpoint(cache / "holdout" / f"{regime}.jsonl")
        gaps, summary = balance_audit(rows, _regime_targets(regime))
        gaps.insert(0, "regime", regime)
        all_gaps.append(gaps)
        summaries.append({
            "regime": regime,
            "calibration_feasible": bool(models["selected"][regime]["feasible"]),
            "holdout_balance_passed": bool(summary["feasible"]),
            "holdout_completion_passed": bool(summary["completionPassed"]),
            "minimum_condition_completion": float(summary["minimumConditionCompletion"]),
            "objective": float(summary["objective"]),
            "structural_identity_max_error": float(summary["structuralIdentityMaxError"]),
            "target_results_json": json.dumps(summary["targetResults"], sort_keys=True, separators=(",", ":")),
        })
    gaps_frame = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.DataFrame()
    summary_frame = pd.DataFrame(summaries)
    _write_parquet(gaps_frame, output / "holdout_balance_gaps.parquet")
    summary_frame.to_csv(output / "residual_imbalance_summary.csv", index=False)
    return gaps_frame, summary_frame


def _s03_expected_holdout() -> dict[str, dict[str, Any]]:
    columns = ["scenario_id", "condition_id", "replicate_ordinal", "activation_count", "successful_swap_count", "final_state_hash", "trajectory_sha256", "stop_reason", "scenario_json_sha256"]
    frame = pq.read_table(Path("/artifacts/research_steps/S03/composition_sweep.parquet"), columns=columns).to_pandas()
    frame = frame[frame.replicate_ordinal.isin(HOLDOUT_REPLICATES)]
    return {str(row.scenario_id): row._asdict() for row in frame.itertuples(index=False)}


def native_replay_audit(output: Path, cache: Path) -> pd.DataFrame:
    expected = _s03_expected_holdout()
    rows = read_checkpoint(cache / "holdout" / "native_control.jsonl")
    audit = []
    for row in rows:
        reference = expected[str(row["scenario_id"])]
        checks = {
            "condition": row["condition_id"] == reference["condition_id"],
            "replicate": int(row["replicate_ordinal"]) == int(reference["replicate_ordinal"]),
            "activation": int(row["activation_count"]) == int(reference["activation_count"]),
            "swaps": int(row["successful_swap_count"]) == int(reference["successful_swap_count"]),
            "final_hash": row["final_state_hash"] == reference["final_state_hash"],
            "trajectory_hash": row["trajectory_sha256"] == reference["trajectory_sha256"],
            "stop_reason": row["stop_reason"] == reference["stop_reason"],
            "scenario_json": row["scenario_json_sha256"] == reference["scenario_json_sha256"],
        }
        audit.append({"scenario_id": row["scenario_id"], "condition_id": row["condition_id"], "replicate_ordinal": int(row["replicate_ordinal"]), **{f"check_{key}": value for key, value in checks.items()}, "passed": all(checks.values())})
    frame = pd.DataFrame(audit)
    _write_parquet(frame, output / "native_control_replay_audit.parquet")
    return frame


def deterministic_and_label_blind_audit(output: Path) -> dict[str, Any]:
    from dataclasses import replace
    from reference_simulator.model import Scenario

    base = load_base_draws()[("unique_1_100", HOLDOUT_REPLICATES[0])]
    condition = next(item for item in build_conditions() if item.input_profile == "unique_1_100" and item.policy_set_label == "Bubble+Selection" and item.first_policy_count == 50 and item.correlation_profile == "absent")
    scenario, _ = materialize_sweep_scenario(condition, base)
    models = json.loads((output / "calibration_models.json").read_text())
    parameters = RegimeParameters.from_dict(models["selected"]["joint_equal"]["parameters"])
    first = execute_kinetic_run(scenario, parameters)
    second = execute_kinetic_run(scenario, parameters)
    ignored = {"elapsed_seconds"}
    deterministic_fields = sorted(set(first) - ignored)
    deterministic = all(first[key] == second[key] for key in deterministic_fields)
    secret_cells = tuple(replace(cell, analysis_label=f"secret-{index % 7}") for index, cell in enumerate(scenario.cells))
    secret = Scenario.create(
        secret_cells,
        initial_occupancy=scenario.initial_occupancy,
        seed=scenario.seed,
        max_activations=scenario.max_activations,
        architecture=scenario.architecture,
        generation_key=scenario.generation_key + "/secret-label-audit",
        fault_placement=scenario.fault_placement,
        requested_fault_count=scenario.requested_fault_count,
    )
    object.__setattr__(secret, "scenario_id", scenario.scenario_id)
    labelled = execute_kinetic_run(secret, parameters, validate_scenario=False)
    label_fields = [
        "stop_reason", "completed", "activation_count", "successful_swap_count",
        "gate_rejection_count", "final_state_hash", "final_consensus_ordered",
        "value_multiset_conserved", "trajectory_sha256", "corrected_curve_sha256",
        "policy_kinetics_json", "ledger_json", "occupancy_b64", "corrected_curve_b64",
    ]
    label_blind = all(first[key] == labelled[key] for key in label_fields)
    record = {
        "schema": "e04.s07.deterministic_label_blind_audit.v1",
        "researchStepId": "S07",
        "conditionId": condition.condition_id,
        "scenarioId": scenario.scenario_id,
        "regime": parameters.regime,
        "deterministicFields": deterministic_fields,
        "deterministicPassed": deterministic,
        "labelBlindFields": label_fields,
        "labelBlindPassed": label_blind,
        "allPassed": deterministic and label_blind,
    }
    write_json(output / "deterministic_label_blind_audit.json", record)
    return record


def _bootstrap_anchor_contrasts(output: Path, cache: Path) -> pd.DataFrame:
    effects = pq.read_table(output / "null_adjusted_effects.parquet").to_pandas()
    conditions = [
        condition
        for condition in build_conditions()
        if condition.policy_set_label in ANCHOR_POLICY_SETS
        and condition.first_policy_count == 50
        and condition.correlation_profile == "absent"
    ]
    checkpoint = {regime: read_checkpoint(cache / "holdout" / f"{regime}.jsonl") for regime in REGIMES}
    grouped = {
        regime: {
            condition_id: sorted([row for row in rows if row["condition_id"] == condition_id], key=lambda row: int(row["replicate_ordinal"]))
            for condition_id in [condition.condition_id for condition in conditions]
        }
        for regime, rows in checkpoint.items()
    }
    confidence = 1 - 0.05 / len(conditions)
    alpha = 1 - confidence
    rows = []
    for regime in REGIMES[1:]:
        for family in ("label_permuted_global", "mobility_matched"):
            for outcome in PRIMARY_OUTCOMES:
                for condition in conditions:
                    control_records = grouped["native_control"][condition.condition_id]
                    intervention_records = grouped[regime][condition.condition_id]
                    control_curves = np.stack([_decode_array(row["corrected_curve_b64"], np.dtype(np.float64), (len(GRID),)) for row in control_records])
                    intervention_curves = np.stack([_decode_array(row["corrected_curve_b64"], np.dtype(np.float64), (len(GRID),)) for row in intervention_records])
                    control_null = float(effects[(effects.regime == "native_control") & (effects.condition_id == condition.condition_id) & (effects.null_family == family) & (effects.outcome == outcome)].null_mean.iloc[0])
                    intervention_null = float(effects[(effects.regime == regime) & (effects.condition_id == condition.condition_id) & (effects.null_family == family) & (effects.outcome == outcome)].null_mean.iloc[0])
                    control_observed = float(trajectory_outcomes(control_curves.mean(axis=0))[outcome][0])
                    intervention_observed = float(trajectory_outcomes(intervention_curves.mean(axis=0))[outcome][0])
                    control_effect = control_observed - control_null
                    intervention_effect = intervention_observed - intervention_null
                    reduction = control_effect - intervention_effect
                    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("anchor_bootstrap", regime, family, outcome, condition.condition_id)))
                    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
                    for start in range(0, BOOTSTRAP_DRAWS, 500):
                        stop = min(start + 500, BOOTSTRAP_DRAWS)
                        indices = rng.integers(0, len(control_curves), size=(stop - start, len(control_curves)))
                        control_mean = control_curves[indices].mean(axis=1)
                        intervention_mean = intervention_curves[indices].mean(axis=1)
                        control_values = trajectory_outcomes(control_mean)[outcome] - control_null
                        intervention_values = trajectory_outcomes(intervention_mean)[outcome] - intervention_null
                        draws[start:stop] = control_values - intervention_values
                    rows.append({
                        "regime": regime,
                        "condition_id": condition.condition_id,
                        "input_profile": condition.input_profile,
                        "policy_set_label": condition.policy_set_label,
                        "null_family": family,
                        "outcome": outcome,
                        "control_adjusted_effect": control_effect,
                        "intervention_adjusted_effect": intervention_effect,
                        "reduction": reduction,
                        "percent_reduction": reduction / control_effect if control_effect > 0 else None,
                        "ci_low": float(np.quantile(draws, alpha / 2)),
                        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
                        "confidence": confidence,
                        "reduction_significant": float(np.quantile(draws, alpha / 2)) > 0,
                    })
    frame = pd.DataFrame(rows)
    _write_parquet(frame, output / "primary_anchor_kinetic_contrasts.parquet")
    return frame


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=180)
    fig.savefig(output / f"{stem}.svg")
    plt.close(fig)


def _plots(output: Path, balance: pd.DataFrame, effects: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot = balance.copy()
    ax.bar(np.arange(len(plot)), plot.objective, color=["#4c78a8" if value else "#e45756" for value in plot.calibration_feasible])
    ax.set_xticks(np.arange(len(plot)), plot.regime, rotation=30, ha="right")
    ax.set_ylabel("Frozen residual-balance objective")
    ax.set_title("Calibration feasibility (blue feasible; red infeasible)")
    _save_figure(fig, output, "kinetic_balance")

    completion = pq.read_table(output / "completion_comparability.parquet").to_pandas()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    summary = completion.groupby("regime", sort=False).completion_fraction.agg(["mean", "min"]).reindex(REGIMES)
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["mean"], width=0.36, label="mean")
    ax.bar(x + 0.18, summary["min"], width=0.36, label="minimum condition")
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, summary.index, rotation=30, ha="right")
    ax.set_ylabel("Completion fraction")
    ax.legend()
    _save_figure(fig, output, "completion_comparability")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    primary = contrasts[(contrasts.null_family == "label_permuted_global")]
    for axis, outcome in zip(axes, PRIMARY_OUTCOMES):
        data = primary[primary.outcome == outcome]
        regimes = list(REGIMES[1:])
        medians = [float(data[data.regime == regime].percent_reduction.median()) for regime in regimes]
        axis.bar(np.arange(len(regimes)), medians, color="#59a14f")
        axis.axhline(0.25, color="black", linestyle="--", linewidth=1)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(regimes)), regimes, rotation=35, ha="right")
        axis.set_title(outcome.replace("_", " "))
        axis.set_ylabel("Median primary-anchor reduction fraction")
    _save_figure(fig, output, "null_adjusted_endpoint_reductions")

    fig, ax = plt.subplots(figsize=(8, 6))
    primary_effects = effects[effects.outcome == "peak"]
    merged = primary_effects.pivot_table(index=["regime", "condition_id"], columns="null_family", values="adjusted_effect").reset_index()
    ax.scatter(merged["label_permuted_global"], merged["label_permuted_value_stratified"], s=9, alpha=0.4)
    limits = [float(min(merged["label_permuted_global"].min(), merged["label_permuted_value_stratified"].min())), float(max(merged["label_permuted_global"].max(), merged["label_permuted_value_stratified"].max()))]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Global-null adjusted peak")
    ax.set_ylabel("Value-stratified adjusted peak")
    ax.set_title("Value conditioning remains a mechanism-sensitive diagnostic")
    _save_figure(fig, output, "value_conditioning_diagnostic")


def analyze_results(output: Path, cache: Path) -> dict[str, Any]:
    calibration = null_calibration(output)
    _, balance = holdout_balance(output, cache)
    replay = native_replay_audit(output, cache)
    deterministic = deterministic_and_label_blind_audit(output)
    contrasts = _bootstrap_anchor_contrasts(output, cache)
    effects = pq.read_table(output / "null_adjusted_effects.parquet").to_pandas()
    models = json.loads((output / "calibration_models.json").read_text())
    joint_feasible = bool(models["selected"]["joint_equal"]["feasible"])
    joint_global = contrasts[(contrasts.regime == "joint_equal") & (contrasts.null_family == "label_permuted_global")]
    outcome_rules = {}
    for outcome in PRIMARY_OUTCOMES:
        group = joint_global[joint_global.outcome == outcome]
        outcome_rules[outcome] = {
            "medianPercentReduction": float(group.percent_reduction.median()),
            "significantReductions": int(group.reduction_significant.sum()),
            "passed": float(group.percent_reduction.median()) >= 0.25 and int(group.reduction_significant.sum()) >= 4,
        }
    joint_mobility = contrasts[(contrasts.regime == "joint_equal") & (contrasts.null_family == "mobility_matched")]
    mobility_direction = {outcome: float(joint_mobility[joint_mobility.outcome == outcome].reduction.median()) > 0 for outcome in PRIMARY_OUTCOMES}
    supportive = joint_feasible and all(item["passed"] for item in outcome_rules.values()) and all(mobility_direction.values())
    if not joint_feasible:
        classification = "constraining/contradictory"
    elif supportive:
        classification = "supportive"
    else:
        classification = "null"
    value_sensitivity = effects[effects.outcome.isin(PRIMARY_OUTCOMES)].pivot_table(index=["regime", "condition_id", "outcome"], columns="null_family", values="adjusted_effect").reset_index()
    value_sensitivity["value_minus_global"] = value_sensitivity["label_permuted_value_stratified"] - value_sensitivity["label_permuted_global"]
    _write_parquet(value_sensitivity, output / "value_conditioning_sensitivity.parquet")
    analysis = {
        "schema": "e04.s07.analysis_summary.v1",
        "researchStepId": "S07",
        "outcomeClassification": classification,
        "jointCalibrationFeasible": joint_feasible,
        "selectedRegimeFeasibility": {regime: bool(models["selected"][regime]["feasible"]) for regime in REGIMES},
        "primaryOutcomeRules": outcome_rules,
        "mobilityReductionDirection": mobility_direction,
        "supportiveRulePassed": supportive,
        "allNullCalibrationPassed": bool((calibration.randomized_passed & calibration.conservative_passed).all()),
        "nativeReplayPassed": bool(replay.passed.all()),
        "deterministicLabelBlindPassed": bool(deterministic["allPassed"]),
        "durationUsedForDecision": False,
        "holdoutBalance": balance.to_dict(orient="records"),
    }
    write_json(output / "analysis_summary.json", analysis)
    write_json(output / "success_criteria.json", {"schema": "e04.s07.success_criteria.v1", "researchStepId": "S07", "classification": classification, "jointFeasible": joint_feasible, "outcomeRules": outcome_rules, "mobilityDirection": mobility_direction, "passed": supportive})
    _plots(output, balance, effects, contrasts)
    return analysis


def write_environment(output: Path, workers: int = 8) -> dict[str, Any]:
    import matplotlib as mpl
    import pyarrow
    import scipy

    record = {
        "schema": "e04.s07.environment.v1",
        "researchStepId": "S07",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "workers": workers,
        "cpuCount": os.cpu_count(),
        "threadEnvironment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "gpuUsed": False,
        "dependenciesInstalled": [],
    }
    write_json(output / "environment.json", record)
    return record


def write_commands(output: Path) -> dict[str, Any]:
    commands = [
        "python scripts/run_kinetic_matching.py freeze",
        "python scripts/run_kinetic_matching.py calibrate --workers 8",
        "python scripts/run_kinetic_matching.py holdout --workers 8",
        "python scripts/run_kinetic_matching.py nulls --workers 8",
        "python scripts/run_kinetic_matching.py package",
        "python scripts/run_kinetic_matching.py analyze",
        "python -m pytest -q tests/test_e04_aggregation.py tests/test_e04_composition_baselines.py tests/test_e04_composition_sweep.py tests/test_e04_aggregation_metrics.py tests/test_e04_static_nulls.py tests/test_e04_dynamic_nulls.py tests/test_e04_kinetic_matching.py --junitxml=/artifacts/research_steps/S07/repository_tests.junit.xml",
        "ruff check analysis/kinetic_matching.py scripts/run_kinetic_matching.py tests/test_e04_kinetic_matching.py",
        "python -m py_compile analysis/kinetic_matching.py scripts/run_kinetic_matching.py tests/test_e04_kinetic_matching.py",
        "git diff --check",
        "python scripts/run_kinetic_matching.py provenance",
        "python scripts/run_kinetic_matching.py validate",
        "python scripts/run_kinetic_matching.py manifest",
    ]
    record = {"schema": "e04.s07.commands.v1", "researchStepId": "S07", "workingDirectory": str(REPOSITORY), "commands": commands}
    write_json(output / "commands.json", record)
    return record


def write_provenance(output: Path, cache: Path) -> dict[str, Any]:
    upstream = {step: _verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    paths = [
        WORKSPACE / "AGENTS.md",
        WORKSPACE / "FULL_PLAN.md",
        WORKSPACE / "RESEARCH_PLAN.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        WORKSPACE / "input-attachments/MANIFEST.json",
        WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
        Path("/previous-artifacts/E01/specification/transition_spec.md"),
        Path("/previous-artifacts/E01/release/reference_simulator/release_manifest.json"),
        Path("/previous-artifacts/E01/research_steps/S13/research_step_full_results.md"),
        CONTRACT_PATH,
        REPOSITORY / "analysis/kinetic_matching.py",
        REPOSITORY / "scripts/run_kinetic_matching.py",
        REPOSITORY / "tests/test_e04_kinetic_matching.py",
    ]
    inputs = []
    for path in paths:
        inputs.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    record = {
        "schema": "e04.s07.provenance.v1",
        "researchStepId": "S07",
        "repository": str(REPOSITORY),
        "branch": _git_output("branch", "--show-current"),
        "head": _git_output("rev-parse", "HEAD"),
        "gitStatusShort": _git_output("status", "--short"),
        "contractSha256": sha256_file(CONTRACT_PATH),
        "parameterFreezeSha256": sha256_file(output / "parameter_freeze_record.json"),
        "calibrationModelsSha256": sha256_file(output / "calibration_models.json"),
        "inputs": inputs,
        "upstreamImmutability": upstream,
        "outputDirectory": str(output),
        "cacheDirectory": str(cache),
        "evidenceLayer": "E01 clean-room deterministic reference simulator",
        "historicalPublicationBytesClaimed": False,
    }
    write_json(output / "upstream_immutability_audit.json", {"schema": "e04.s07.upstream_immutability.v1", "researchStepId": "S07", "steps": upstream, "allPassed": all(item["allPassed"] for item in upstream.values())})
    write_json(output / "provenance.json", record)
    return record


def _table_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def validate_artifacts(output: Path) -> dict[str, Any]:
    required = [
        "preregistration.json", "freeze_record.json", "kinetic_intervention_specification.md",
        "calibration_models.json", "parameter_freeze_record.json", "calibration_candidates.parquet",
        "calibration_balance_gaps.parquet", "kinetic_matching.parquet", "policy_kinetic_ledger.parquet",
        "completion_comparability.parquet", "holdout_balance_gaps.parquet", "residual_imbalance_summary.csv",
        "observed_kinetic_trajectories.parquet", "null_adjusted_effects.parquet",
        "primary_anchor_kinetic_contrasts.parquet", "value_conditioning_sensitivity.parquet",
        "dynamic_null_calibration.parquet", "null_preservation_audit.parquet",
        "null_distributions/dynamic_null_statistics.parquet",
        "null_distributions/retained_null_trajectories.parquet",
        "null_distributions/null_response_surface.parquet", "analysis_summary.json",
        "success_criteria.json", "native_control_replay_audit.parquet",
        "deterministic_label_blind_audit.json", "holdout_accounting.json",
        "dynamic_null_accounting.json", "packaging_accounting.json", "environment.json",
        "commands.json", "provenance.json", "upstream_immutability_audit.json",
        "repository_tests.junit.xml", "research_step_full_results.md",
        "kinetic_balance.png", "kinetic_balance.svg", "completion_comparability.png",
        "completion_comparability.svg", "null_adjusted_endpoint_reductions.png",
        "null_adjusted_endpoint_reductions.svg", "value_conditioning_diagnostic.png",
        "value_conditioning_diagnostic.svg",
    ]
    checks = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": _json_native(detail)})

    missing = [path for path in required if not (output / path).is_file() or (output / path).stat().st_size == 0]
    add("required_files", not missing, {"missing": missing, "required": len(required)})
    freeze = json.loads((output / "freeze_record.json").read_text())
    parameter = json.loads((output / "parameter_freeze_record.json").read_text())
    add("frozen_split_and_parameters", freeze["contractSha256"] == sha256_file(CONTRACT_PATH) and not freeze["splitOverlap"] and parameter["calibrationModelsSha256"] == sha256_file(output / "calibration_models.json"), {"contract": freeze["contractSha256"], "models": parameter["calibrationModelsSha256"]})
    add("calibration_holdout_disjoint", not (set(CALIBRATION_REPLICATES) & set(HOLDOUT_REPLICATES)), {"calibration": list(CALIBRATION_REPLICATES), "holdout": list(HOLDOUT_REPLICATES)})
    expected_rows = {
        "kinetic_matching": len(REGIMES) * EXPECTED_HOLDOUT_SCENARIOS,
        "completion": len(REGIMES) * EXPECTED_CONDITIONS,
        "observed": len(REGIMES) * EXPECTED_CONDITIONS * len(GRID),
        "null_statistics": len(REGIMES) * EXPECTED_CONDITIONS * len(NULL_FAMILIES) * TOTAL_DRAWS,
        "null_response": len(REGIMES) * EXPECTED_CONDITIONS * len(NULL_FAMILIES) * len(GRID),
        "effects": len(REGIMES) * EXPECTED_CONDITIONS * len(NULL_FAMILIES) * len(OUTCOMES),
        "preservation": len(REGIMES) * EXPECTED_CONDITIONS,
    }
    observed_rows = {
        "kinetic_matching": _table_rows(output / "kinetic_matching.parquet"),
        "completion": _table_rows(output / "completion_comparability.parquet"),
        "observed": _table_rows(output / "observed_kinetic_trajectories.parquet"),
        "null_statistics": _table_rows(output / "null_distributions/dynamic_null_statistics.parquet"),
        "null_response": _table_rows(output / "null_distributions/null_response_surface.parquet"),
        "effects": _table_rows(output / "null_adjusted_effects.parquet"),
        "preservation": _table_rows(output / "null_preservation_audit.parquet"),
    }
    add("complete_table_accounting", observed_rows == expected_rows, {"expected": expected_rows, "observed": observed_rows})
    runs = pq.read_table(output / "kinetic_matching.parquet").to_pandas()
    add("run_identity_and_invariants", len(runs) == len(REGIMES) * EXPECTED_HOLDOUT_SCENARIOS and runs.run_id.nunique() == len(runs) and bool(runs.value_multiset_conserved.all()) and not bool((runs.stop_reason == "invariant_error").any()), {"rows": len(runs), "uniqueRuns": runs.run_id.nunique(), "stopReasons": runs.stop_reason.value_counts().to_dict()})
    replay = pq.read_table(output / "native_control_replay_audit.parquet").to_pandas()
    add("native_control_replay", len(replay) == EXPECTED_HOLDOUT_SCENARIOS and bool(replay.passed.all()), {"rows": len(replay), "passed": int(replay.passed.sum())})
    deterministic = json.loads((output / "deterministic_label_blind_audit.json").read_text())
    add("deterministic_label_blind", deterministic["allPassed"], deterministic)
    policy = pq.read_table(output / "policy_kinetic_ledger.parquet").to_pandas()
    add("structural_identity", float(policy.structural_identity_error.fillna(0).max()) <= 1e-12, float(policy.structural_identity_error.fillna(0).max()))
    completion = pq.read_table(output / "completion_comparability.parquet").to_pandas()
    add("completion_audit", bool(completion.all_value_conserved.all()) and bool(completion.all_complete_runs_ordered.all()) and len(completion) == len(REGIMES) * EXPECTED_CONDITIONS, completion.groupby("regime").completion_fraction.agg(["mean", "min"]).to_dict())
    preservation = pq.read_table(output / "null_preservation_audit.parquet").to_pandas()
    add("null_preservation", int(preservation.count_violations.sum()) == 0 and int(preservation.stratum_violations.sum()) == 0 and bool((preservation.scenario_count == len(HOLDOUT_REPLICATES)).all()), {"countViolations": int(preservation.count_violations.sum()), "stratumViolations": int(preservation.stratum_violations.sum())})
    calibration = pq.read_table(output / "dynamic_null_calibration.parquet").to_pandas()
    add("null_calibration", len(calibration) == len(REGIMES) * len(NULL_FAMILIES) * len(OUTCOMES) * 3 and bool((calibration.randomized_passed & calibration.conservative_passed).all()), {"rows": len(calibration), "passed": int((calibration.randomized_passed & calibration.conservative_passed).sum())})
    effects = pq.read_table(output / "null_adjusted_effects.parquet").to_pandas()
    add("multiplicity_complete", bool(effects.q_value.notna().all()) and bool(effects.q_value.between(0, 1).all()), {"rows": len(effects)})
    analysis = json.loads((output / "analysis_summary.json").read_text())
    add("duration_excluded_from_decision", analysis["durationUsedForDecision"] is False, analysis["durationUsedForDecision"])
    upstream = json.loads((output / "upstream_immutability_audit.json").read_text())
    add("upstream_immutability", upstream["allPassed"], list(upstream["steps"]))
    report = (output / "research_step_full_results.md").read_text()
    fields = ["Research step ID", "Completion status", "Artifacts written", "Validation result", "Outcome classification", "Caveats or blockers", "Lay summary", "Recommended next action", "Detailed methods", "Commands", "Provenance"]
    add("report_required_fields", all(field in report for field in fields), fields)
    add("s08_absent", not S08_DIR.exists(), str(S08_DIR))
    record = {"schema": "e04.s07.validation_summary.v1", "researchStepId": "S07", "passed": all(item["passed"] for item in checks), "checks": checks}
    write_json(output / "validation_summary.json", record)
    if not record["passed"]:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise AssertionError(f"S07 artifact validation failed: {failed}")
    return record


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    record = {"schema": "e04.s07.artifact_manifest.v1", "researchStepId": "S07", "artifactCount": len(artifacts), "artifacts": artifacts}
    write_json(output / "artifact_manifest.json", record)
    return record
