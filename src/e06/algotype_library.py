"""Curate and validate the E06 S01 chimeric Algotype library.

S01 is a bridge step: it does not introduce a new chimeric simulator.  It
normalizes upstream E01/E03/E04 policy records into a stable library and runs
small pure-policy smoke tasks under each policy's native execution contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.e02.deterministic_simulator import SimulatorConfig, simulate
from src.e03.coarse_sweep import actor_schedule, initial_values, sortedness_metrics
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, parse_policy
from src.e04.evolutionary_search import EvolutionGenome, S08LocalEvolutionPolicy
from src.e04.no_oracle_protocol import LOCAL_ONLY_PROTOCOL_ID, LocalTrainingObservation


ALGOTYPE_SCHEMA = "eidosoma.e06.chimeric_algotype.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s01_policy_validation.v1"
METADATA_SCHEMA = "eidosoma.e06.algotype_metadata.v1"
EXPERIMENT_ID = "E06"
STEP_ID = "S01"


NULL_WAIT_DSL = """policy e06_null_wait v1
state ideal_position=none
rule else wait
end
"""

NULL_COMPARE_ONLY_DSL = """policy e06_null_compare_only v1
state ideal_position=none
rule if target_exists(left) then compare(left), wait
rule if target_exists(right) then compare(right), wait
rule else wait
end
"""

RANDOM_ADJACENT_WALK_DSL = """policy e06_random_adjacent_walk v1
state ideal_position=none
rule if random_lt(0.5) and target_exists(left) and target_movable(left) then swap(left)
rule if target_exists(right) and target_movable(right) then swap(right)
rule else wait
end
"""

RANDOM_NOISY_INVERSION_DSL = """policy e06_random_noisy_inversion_cleaner v1
state ideal_position=none
rule if random_lt(0.35) and target_exists(left) and target_movable(left) then swap(left)
rule if target_exists(left) and target_movable(left) and self_lt(left) then compare(left), swap(left)
rule if target_exists(right) and target_movable(right) and self_gt(right) then compare(right), swap(right)
rule else wait
end
"""


@dataclass(frozen=True)
class S01SourcePaths:
    """Input artifact locations used by E06 S01."""

    e03_classic_library: Path = Path("/previous-artifacts/E03/policies/e03_classic_policy_library.json")
    e03_frontier_candidates: Path = Path("/previous-artifacts/E03/policies/e03_frontier_candidate_policies.jsonl")
    e03_frontier_table: Path = Path("/previous-artifacts/E03/tables/e03_frontier_candidates.csv")
    e04_repair_capable: Path = Path("/previous-artifacts/E04/policies/e04_repair_capable_algotypes.jsonl")
    e04_evolved_repair: Path = Path("/previous-artifacts/E04/policies/e04_evolved_repair_policies.jsonl")


@dataclass(frozen=True)
class S01ValidationConfig:
    """Small pure-policy validation grid for curated Algotypes."""

    seeds: tuple[int, ...] = (2026070201, 2026070202)
    array_size: int = 8
    dsl_event_cap: int = 192
    original_max_events: int = 50_000
    memory_event_cap: int = 192


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def source_file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "sizeBytes": path.stat().st_size if path.exists() else None,
    }


def _algotype_id(payload: Mapping[str, Any]) -> str:
    return f"e06:{stable_hash(payload)[:16]}"


def _record(
    *,
    source_category: str,
    source_policy_id: str,
    display_name: str,
    policy_family: str,
    representation_type: str,
    execution_backend: str,
    algorithm: str,
    source_artifact: str,
    direction_support: str = "increasing_default",
    dsl_source: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    upstream_metrics: Mapping[str, Any] | None = None,
    source_payload: Mapping[str, Any] | None = None,
    compatibility: Mapping[str, Any] | None = None,
    evidence_caveat: str = "",
) -> dict[str, Any]:
    id_payload = {
        "schema": ALGOTYPE_SCHEMA,
        "sourceCategory": source_category,
        "sourcePolicyId": source_policy_id,
        "displayName": display_name,
        "representationType": representation_type,
        "executionBackend": execution_backend,
        "dslSha256": hashlib.sha256(dsl_source.encode("utf-8")).hexdigest() if dsl_source else None,
        "parametersHash": stable_hash(parameters or {}) if parameters else None,
    }
    algotype_id = _algotype_id(id_payload)
    compatibility_metadata = {
        "supportsPureArrayValidation": True,
        "supportsSameGoalChimeraCandidate": source_category != "null_control",
        "supportsOppositeGoalCandidate": source_category == "original",
        "requiresMemory": source_category == "memory_repair",
        "requiresSignaling": False,
        "usesGlobalOracle": False,
        "informationAccess": "local_neighbors" if source_category == "memory_repair" else "local_array_policy",
        "executionContract": execution_backend,
    }
    compatibility_metadata.update(dict(compatibility or {}))
    return {
        "schema": ALGOTYPE_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "algotypeId": algotype_id,
        "policyId": algotype_id,
        "sourcePolicyId": source_policy_id,
        "displayName": display_name,
        "sourceCategory": source_category,
        "policyFamily": policy_family,
        "algorithm": algorithm,
        "representationType": representation_type,
        "executionBackend": execution_backend,
        "directionSupport": direction_support,
        "sourceArtifact": source_artifact,
        "dslSource": dsl_source,
        "dslSha256": hashlib.sha256(dsl_source.encode("utf-8")).hexdigest() if dsl_source else None,
        "parameters": dict(parameters or {}),
        "upstreamMetrics": dict(upstream_metrics or {}),
        "compatibilityMetadata": compatibility_metadata,
        "sourcePayloadDigest": stable_hash(dict(source_payload or {}))[:24],
        "evidenceCaveat": evidence_caveat,
    }


def load_original_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exact E03 interface-wrapper records for Bubble/Insertion/Selection."""

    records: list[dict[str, Any]] = []
    status = {"source": source_file_entry(path), "loaded": 0, "fallbackUsed": False}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("policies", []):
            if entry.get("representationType") != "interface_wrapper":
                continue
            if entry.get("exactness") != "exact_public_method":
                continue
            algorithm = str(entry["algorithm"])
            records.append(
                _record(
                    source_category="original",
                    source_policy_id=str(entry["policyId"]),
                    display_name=f"original_{algorithm}",
                    policy_family=f"original_{algorithm}",
                    representation_type="interface_wrapper",
                    execution_backend="e02_public_cell_simulator",
                    algorithm=algorithm,
                    source_artifact=str(path),
                    direction_support=str(entry.get("direction", "parameterized_by_cell_reverse_direction")),
                    upstream_metrics={"exactness": entry.get("exactness")},
                    source_payload=entry,
                    compatibility={
                        "supportsOppositeGoalCandidate": True,
                        "supportsReverseDirectionParameter": True,
                        "paperOriginal": True,
                    },
                    evidence_caveat="Exact public-method wrapper inherited from E03 classic policy mapping.",
                )
            )
    if not records:
        status["fallbackUsed"] = True
        for algorithm in ("bubble", "insertion", "selection"):
            records.append(
                _record(
                    source_category="original",
                    source_policy_id=f"fallback_original_{algorithm}",
                    display_name=f"original_{algorithm}",
                    policy_family=f"original_{algorithm}",
                    representation_type="interface_wrapper",
                    execution_backend="e02_public_cell_simulator",
                    algorithm=algorithm,
                    source_artifact="fallback_hardcoded_original_algorithms",
                    direction_support="parameterized_by_cell_reverse_direction",
                    compatibility={"paperOriginal": True, "fallbackGenerated": True},
                    evidence_caveat="Fallback original record generated because E03 classic policy library was unavailable.",
                )
            )
    status["loaded"] = len(records)
    return records, status


def builtin_control_records() -> list[dict[str, Any]]:
    """Return E06 null and randomized control policies."""

    specs = [
        ("null_control", "null_wait", "null_wait", NULL_WAIT_DSL, "Null policy that never moves."),
        ("null_control", "null_compare_only", "null_compare_only", NULL_COMPARE_ONLY_DSL, "Null policy that observes adjacent cells but never moves."),
        ("randomized", "random_adjacent_walk", "random_adjacent_walk", RANDOM_ADJACENT_WALK_DSL, "Random local walk control, value-agnostic."),
        (
            "randomized",
            "random_noisy_inversion_cleaner",
            "random_noisy_inversion_cleaner",
            RANDOM_NOISY_INVERSION_DSL,
            "Randomized inversion-cleaning control with occasional value-agnostic swaps.",
        ),
    ]
    records: list[dict[str, Any]] = []
    for source_category, source_policy_id, name, source, caveat in specs:
        policy = parse_policy(source)
        records.append(
            _record(
                source_category=source_category,
                source_policy_id=f"e06_builtin:{source_policy_id}",
                display_name=name,
                policy_family=source_category,
                representation_type="dsl",
                execution_backend="e03_dsl_cpu_interpreter",
                algorithm="control",
                source_artifact="src.e06.algotype_library builtin controls",
                dsl_source=policy.to_source(),
                upstream_metrics={"policyName": policy.name, "policyId": policy.policy_id},
                source_payload={"sourcePolicyId": source_policy_id, "dslSha256": policy.sha256},
                compatibility={
                    "supportsSameGoalChimeraCandidate": source_category == "randomized",
                    "supportsOppositeGoalCandidate": False,
                    "controlPolicy": True,
                    "controlRole": source_category,
                },
                evidence_caveat=caveat,
            )
        )
    return records


def load_e03_frontier_records(path: Path, *, max_records: int = 16) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load E03 discovered frontier policies with stable source hashes."""

    raw = read_jsonl(path)
    status = {"source": source_file_entry(path), "loaded": 0, "fallbackUsed": False, "parseFailures": 0}
    records: list[dict[str, Any]] = []
    if raw:
        raw = sorted(raw, key=lambda item: int(item.get("curatedRank", 10_000)))[:max_records]
    for entry in raw:
        try:
            policy = parse_policy(str(entry["dslSource"]))
        except Exception:
            status["parseFailures"] += 1
            continue
        records.append(
            _record(
                source_category="discovered",
                source_policy_id=str(entry["policyId"]),
                display_name=f"e03_frontier_rank_{int(entry.get('curatedRank', len(records) + 1)):02d}",
                policy_family=str(entry.get("sourceKind", "e03_discovered")),
                representation_type="dsl",
                execution_backend="e03_dsl_cpu_interpreter",
                algorithm="discovered_local_rule",
                source_artifact=str(path),
                dsl_source=policy.to_source(),
                upstream_metrics={
                    "curatedRank": entry.get("curatedRank"),
                    "classId": entry.get("classId"),
                    "cautiousLabel": entry.get("cautiousLabel"),
                    "selectionScore": entry.get("selectionScore"),
                    "s14MeanFinalSortedness": entry.get("s14MeanFinalSortedness"),
                    "s14PerturbationMeanFinalSortedness": entry.get("s14PerturbationMeanFinalSortedness"),
                    "sourceHashVerified": entry.get("sourceHashVerified"),
                    "selectionReasons": entry.get("selectionReasons", []),
                },
                source_payload=entry,
                compatibility={
                    "supportsOppositeGoalCandidate": "unknown",
                    "supportsReverseDirectionParameter": "partial_or_unknown",
                    "frontierCandidate": True,
                    "e03ClassId": entry.get("classId"),
                },
                evidence_caveat="E03 frontier candidate validated upstream as a discovered DSL policy; reverse-goal compatibility remains untested for E06.",
            )
        )
    if not records:
        status["fallbackUsed"] = True
    status["loaded"] = len(records)
    return records, status


def load_e04_memory_records(path: Path, *, max_records: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load E04 repair-capable memory policy records."""

    raw = read_jsonl(path)
    status = {"source": source_file_entry(path), "loaded": 0, "fallbackUsed": False, "parseFailures": 0}
    records: list[dict[str, Any]] = []
    for entry in raw[:max_records]:
        parameters = dict(entry.get("parameters", {}))
        if not parameters:
            status["parseFailures"] += 1
            continue
        records.append(
            _record(
                source_category="memory_repair",
                source_policy_id=str(entry["policy_id"]),
                display_name=f"e04_memory_repair_{len(records) + 1:02d}",
                policy_family=str(entry.get("policy_family", "e04_memory_repair")),
                representation_type="local_training_weight_policy",
                execution_backend="e04_local_training_policy",
                algorithm="memory_repair_adjacent",
                source_artifact=str(path),
                parameters=parameters,
                upstream_metrics={
                    "mechanismId": entry.get("mechanism_id"),
                    "minimalMemory": entry.get("minimal_memory"),
                    "repairCapableStatus": entry.get("repair_capable_status"),
                    "requiredSignalMode": entry.get("required_signal_mode"),
                    "s13NeighborHeldoutMeanFitness": entry.get("s13_neighbor_heldout_mean_fitness"),
                    "s13NeighborMinusNoMemoryHeldoutFitness": entry.get("s13_neighbor_minus_no_memory_heldout_fitness"),
                    "s14LocalMeanComparisonRepairScore": entry.get("s14_local_mean_comparison_repair_score"),
                    "eligibleForLocalOnlyClaims": entry.get("eligible_for_local_only_claims"),
                    "limitations": entry.get("limitations", []),
                },
                source_payload=entry,
                compatibility={
                    "requiresMemory": True,
                    "supportsSignaling": False,
                    "allowedTrainingSignalFields": entry.get("allowed_training_signal_fields", []),
                    "excludedTrainingSignalFields": entry.get("excluded_training_signal_fields", []),
                    "policyInputContract": entry.get("policy_input_contract"),
                    "eligibleForLocalOnlyClaims": entry.get("eligible_for_local_only_claims"),
                    "supportsOppositeGoalCandidate": False,
                },
                evidence_caveat=(
                    "E04 local-only memory repair policy. S01 pure-array validation uses a simplified adjacent-action "
                    "harness; E04 repair competence remains the upstream evidence layer."
                ),
            )
        )
    if not records:
        status["fallbackUsed"] = True
    status["loaded"] = len(records)
    return records, status


def curate_algotype_records(paths: S01SourcePaths, *, max_e03_frontier: int = 16, max_e04_memory: int = 4) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals, original_status = load_original_records(paths.e03_classic_library)
    controls = builtin_control_records()
    discovered, discovered_status = load_e03_frontier_records(paths.e03_frontier_candidates, max_records=max_e03_frontier)
    memory, memory_status = load_e04_memory_records(paths.e04_repair_capable, max_records=max_e04_memory)
    records = [*originals, *controls, *discovered, *memory]
    ids = [record["algotypeId"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Curated Algotype IDs are not unique")
    status = {
        "originals": original_status,
        "builtinControls": {"loaded": len(controls), "fallbackUsed": False},
        "discovered": discovered_status,
        "memoryRepair": memory_status,
        "totalRecords": len(records),
        "sourceFiles": {
            "e03ClassicLibrary": source_file_entry(paths.e03_classic_library),
            "e03FrontierCandidates": source_file_entry(paths.e03_frontier_candidates),
            "e03FrontierTable": source_file_entry(paths.e03_frontier_table),
            "e04RepairCapable": source_file_entry(paths.e04_repair_capable),
            "e04EvolvedRepair": source_file_entry(paths.e04_evolved_repair),
        },
    }
    return records, status


def _validation_base(record: Mapping[str, Any], case_id: str, seed: int, array_size: int, values: Sequence[int]) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "algotype_id": record["algotypeId"],
        "source_policy_id": record["sourcePolicyId"],
        "display_name": record["displayName"],
        "source_category": record["sourceCategory"],
        "representation_type": record["representationType"],
        "execution_backend": record["executionBackend"],
        "validation_case": case_id,
        "array_size": int(array_size),
        "seed": int(seed),
        "initial_values_json": json.dumps(list(values), separators=(",", ":")),
    }


def _finalize_validation_row(
    base: Mapping[str, Any],
    *,
    final_values: Sequence[int],
    event_cap: int,
    events_executed: int,
    compare_count: int,
    swap_count: int,
    update_count: int = 0,
    wait_count: int = 0,
    execution_status: str = "ok",
    stop_reason: str = "fixed_horizon_complete",
    error_message: str = "",
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    initial = json.loads(str(base["initial_values_json"]))
    initial_metrics = sortedness_metrics(initial)
    final_metrics = sortedness_metrics(final_values)
    final_is_sorted = bool(final_metrics["is_sorted"])
    invalid = execution_status != "ok"
    row = dict(base)
    row.update(
        {
            "event_cap": int(event_cap),
            "events_executed": int(events_executed),
            "execution_status": execution_status,
            "run_status": "invalid" if invalid else "ok_sorted" if final_is_sorted else "ok_unsorted_fixed_horizon",
            "stop_reason": stop_reason,
            "invalid": bool(invalid),
            "error_message": error_message[:500],
            "compare_count": int(compare_count),
            "swap_count": int(swap_count),
            "update_count": int(update_count),
            "wait_count": int(wait_count),
            "work_count": int(compare_count + swap_count + update_count),
            "initial_inversion_count": int(initial_metrics["inversion_count"]),
            "final_inversion_count": int(final_metrics["inversion_count"]),
            "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
            "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
            "inversion_sortedness_delta": float(final_metrics["inversion_sortedness"] - initial_metrics["inversion_sortedness"]),
            "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
            "final_is_sorted": final_is_sorted,
            "final_values_json": json.dumps(list(final_values), separators=(",", ":")),
            "elapsed_seconds": float(elapsed_seconds),
        }
    )
    return row


def validate_original_record(record: Mapping[str, Any], config: S01ValidationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    algorithm = str(record["algorithm"]).replace("original_", "")
    for seed in config.seeds:
        values = initial_values(config.array_size, seed)
        base = _validation_base(record, f"pure_original_n{config.array_size}_seed{seed}", seed, config.array_size, values)
        started = time.perf_counter()
        try:
            result = simulate(
                SimulatorConfig(
                    values=values,
                    algorithm=algorithm,
                    activation_seed=seed,
                    policy_seed=seed + 10_000,
                    max_events=config.original_max_events,
                    stop_when_sorted=True,
                )
            )
            rows.append(
                _finalize_validation_row(
                    base,
                    final_values=result.final_values,
                    event_cap=config.original_max_events,
                    events_executed=result.event_count,
                    compare_count=result.comparison_count,
                    swap_count=result.swap_count,
                    stop_reason=result.stop_reason,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
        except Exception as exc:  # pragma: no cover - preserved in validation artifacts.
            rows.append(
                _finalize_validation_row(
                    base,
                    final_values=values,
                    event_cap=config.original_max_events,
                    events_executed=0,
                    compare_count=0,
                    swap_count=0,
                    execution_status="invalid",
                    stop_reason="exception",
                    error_message=repr(exc),
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
    return rows


def run_dsl_pure_config(record: Mapping[str, Any], *, seed: int, array_size: int, event_cap: int) -> dict[str, Any]:
    policy = parse_policy(str(record["dslSource"]))
    values = initial_values(array_size, seed)
    schedule = actor_schedule(array_size, event_cap, seed)
    interpreter = DSLInterpreter(policy)
    rng = random.Random((seed * 1_000_003) ^ int(policy.sha256[:12], 16))
    statuses: tuple[str, ...] = tuple("ACTIVE" for _ in values)
    ideal_position: int | None = None
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    current_values = values
    started = time.perf_counter()
    for actor_index in schedule:
        state = DSLArrayState(
            values=current_values,
            statuses=statuses,
            actor_index=int(actor_index),
            ideal_position=ideal_position,
        )
        result = interpreter.step_state(state, rng)
        current_values = result.state_after.values
        statuses = tuple(result.state_after.statuses or statuses)
        ideal_position = result.state_after.ideal_position
        compare_count += int(bool(result.action.compare_counted))
        swap_count += int(result.action.action_type == "swap")
        update_count += int(result.action.action_type == "update_state")
        wait_count += int(result.action.action_type == "wait")
    return {
        "values": tuple(current_values),
        "events_executed": int(event_cap),
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "elapsed_seconds": time.perf_counter() - started,
    }


def validate_dsl_record(record: Mapping[str, Any], config: S01ValidationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        values = initial_values(config.array_size, seed)
        case_id = f"e06_s01_pure_dsl_n{config.array_size}_seed{seed}"
        base = _validation_base(record, case_id, seed, config.array_size, values)
        try:
            result = run_dsl_pure_config(record, seed=seed, array_size=config.array_size, event_cap=config.dsl_event_cap)
            rows.append(
                _finalize_validation_row(
                    base,
                    final_values=result["values"],
                    event_cap=config.dsl_event_cap,
                    events_executed=result["events_executed"],
                    compare_count=result["compare_count"],
                    swap_count=result["swap_count"],
                    update_count=result["update_count"],
                    wait_count=result["wait_count"],
                    stop_reason="fixed_horizon_complete",
                    elapsed_seconds=result["elapsed_seconds"],
                )
            )
        except Exception as exc:  # pragma: no cover - preserved in validation artifacts.
            rows.append(
                _finalize_validation_row(
                    base,
                    final_values=values,
                    event_cap=config.dsl_event_cap,
                    events_executed=0,
                    compare_count=0,
                    swap_count=0,
                    execution_status="invalid",
                    stop_reason="exception",
                    error_message=repr(exc),
                )
            )
    return rows


def _memory_observation(
    *,
    values: Sequence[int],
    actor_index: int,
    memory: Mapping[str, Any],
) -> LocalTrainingObservation:
    left_idx = actor_index - 1
    right_idx = actor_index + 1
    left_present = left_idx >= 0
    right_present = right_idx < len(values)
    return LocalTrainingObservation(
        protocol_id=LOCAL_ONLY_PROTOCOL_ID,
        behavior="bubble",
        actor_value=int(values[actor_index]),
        actor_status="ACTIVE",
        reverse_direction=False,
        left_present=left_present,
        left_value=int(values[left_idx]) if left_present else None,
        left_status="ACTIVE" if left_present else None,
        right_present=right_present,
        right_value=int(values[right_idx]) if right_present else None,
        right_status="ACTIVE" if right_present else None,
        at_left_boundary=not left_present,
        at_right_boundary=not right_present,
        last_move_success=memory.get("last_move_success"),
        last_action_type=memory.get("last_action_type"),
        time_since_movement=int(memory.get("time_since_movement", 0)),
        local_frustration=int(memory.get("local_frustration", 0)),
        failed_swap_count=int(memory.get("failed_swap_count", 0)),
        neighbor_identity_count=int(memory.get("neighbor_identity_count", 0)),
        signal_blocked=0.0,
        signal_frustrated=0.0,
        signal_scope="none",
    )


def run_memory_pure_config(record: Mapping[str, Any], *, seed: int, array_size: int, event_cap: int) -> dict[str, Any]:
    values = list(initial_values(array_size, seed))
    identities = list(range(array_size))
    schedule = actor_schedule(array_size, event_cap, seed)
    parameters = {str(key): float(value) for key, value in dict(record["parameters"]).items()}
    genome = EvolutionGenome(
        policy_id=str(record["sourcePolicyId"]),
        generation=0,
        population_index=0,
        parent_ids=(),
        mutation_seed=seed,
        mutation_scale=0.0,
        parameters=parameters,
        lineage_note="e06_s01_replay_from_e04_artifact",
    )
    policy = S08LocalEvolutionPolicy(genome)
    memory_by_identity: dict[int, dict[str, Any]] = {identity: {"time_since_movement": 0, "local_frustration": 0} for identity in identities}
    compare_count = 0
    swap_count = 0
    wait_count = 0
    action_errors = 0
    started = time.perf_counter()
    for actor_index in schedule:
        actor_id = identities[actor_index]
        memory = memory_by_identity.setdefault(actor_id, {"time_since_movement": 0, "local_frustration": 0})
        observation = _memory_observation(values=values, actor_index=actor_index, memory=memory)
        decision = policy.select_action(observation)
        target_index: int | None = None
        if decision.action == "swap_left":
            target_index = actor_index - 1
        elif decision.action == "swap_right":
            target_index = actor_index + 1
        moved = False
        if target_index is not None:
            compare_count += 1
            if 0 <= target_index < len(values):
                values[actor_index], values[target_index] = values[target_index], values[actor_index]
                identities[actor_index], identities[target_index] = identities[target_index], identities[actor_index]
                moved = True
                swap_count += 1
            else:
                action_errors += 1
        else:
            wait_count += 1
        memory_by_identity[actor_id] = {
            "last_move_success": moved if target_index is not None else None,
            "last_action_type": decision.action,
            "time_since_movement": 0 if moved else min(255, int(memory.get("time_since_movement", 0)) + 1),
            "local_frustration": max(0, min(255, int(memory.get("local_frustration", 0)) + (0 if moved else 1))),
            "failed_swap_count": 0 if moved else min(4, int(memory.get("failed_swap_count", 0)) + int(target_index is not None)),
            "neighbor_identity_count": int(observation.left_present) + int(observation.right_present),
        }
    return {
        "values": tuple(values),
        "events_executed": int(event_cap),
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "wait_count": int(wait_count),
        "action_errors": int(action_errors),
        "elapsed_seconds": time.perf_counter() - started,
    }


def validate_memory_record(record: Mapping[str, Any], config: S01ValidationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        values = initial_values(config.array_size, seed)
        base = _validation_base(record, f"pure_memory_n{config.array_size}_seed{seed}", seed, config.array_size, values)
        try:
            result = run_memory_pure_config(record, seed=seed, array_size=config.array_size, event_cap=config.memory_event_cap)
            rows.append(
                _finalize_validation_row(
                    base,
                    final_values=result["values"],
                    event_cap=config.memory_event_cap,
                    events_executed=result["events_executed"],
                    compare_count=result["compare_count"],
                    swap_count=result["swap_count"],
                    wait_count=result["wait_count"],
                    execution_status="ok" if result["action_errors"] == 0 else "invalid",
                    stop_reason="fixed_horizon_complete",
                    error_message=f"action_errors={result['action_errors']}" if result["action_errors"] else "",
                    elapsed_seconds=result["elapsed_seconds"],
                )
            )
        except Exception as exc:  # pragma: no cover - preserved in validation artifacts.
            rows.append(
                _finalize_validation_row(
                    base,
                    final_values=values,
                    event_cap=config.memory_event_cap,
                    events_executed=0,
                    compare_count=0,
                    swap_count=0,
                    execution_status="invalid",
                    stop_reason="exception",
                    error_message=repr(exc),
                )
            )
    return rows


def validate_algotype_records(records: Sequence[Mapping[str, Any]], config: S01ValidationConfig | None = None) -> pd.DataFrame:
    config = config or S01ValidationConfig()
    rows: list[dict[str, Any]] = []
    for record in records:
        backend = str(record["executionBackend"])
        if backend == "e02_public_cell_simulator":
            rows.extend(validate_original_record(record, config))
        elif backend == "e03_dsl_cpu_interpreter":
            rows.extend(validate_dsl_record(record, config))
        elif backend == "e04_local_training_policy":
            rows.extend(validate_memory_record(record, config))
        else:
            raise ValueError(f"Unsupported E06 S01 execution backend: {backend}")
    return pd.DataFrame(rows)


def summarize_validation(records: Sequence[Mapping[str, Any]], validation: pd.DataFrame) -> pd.DataFrame:
    record_by_id = {str(record["algotypeId"]): record for record in records}
    rows: list[dict[str, Any]] = []
    grouped = validation.groupby("algotype_id", sort=False)
    for algotype_id, group in grouped:
        record = record_by_id[str(algotype_id)]
        rows.append(
            {
                "schema": METADATA_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "algotype_id": algotype_id,
                "source_policy_id": record["sourcePolicyId"],
                "display_name": record["displayName"],
                "source_category": record["sourceCategory"],
                "policy_family": record["policyFamily"],
                "algorithm": record["algorithm"],
                "representation_type": record["representationType"],
                "execution_backend": record["executionBackend"],
                "requires_memory": bool(record["compatibilityMetadata"].get("requiresMemory", False)),
                "requires_signaling": bool(record["compatibilityMetadata"].get("requiresSignaling", False)),
                "uses_global_oracle": bool(record["compatibilityMetadata"].get("usesGlobalOracle", False)),
                "validation_run_count": int(len(group)),
                "invalid_run_count": int(group["invalid"].sum()),
                "pure_execution_success": bool(not group["invalid"].any()),
                "sorted_run_fraction": float(group["final_is_sorted"].mean()),
                "mean_initial_inversion_sortedness": float(group["initial_inversion_sortedness"].mean()),
                "mean_final_inversion_sortedness": float(group["final_inversion_sortedness"].mean()),
                "mean_inversion_sortedness_delta": float(group["inversion_sortedness_delta"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "mean_swap_count": float(group["swap_count"].mean()),
                "baseline_competence_score": float(
                    0.75 * group["final_inversion_sortedness"].mean()
                    + 0.25 * max(0.0, group["inversion_sortedness_delta"].mean())
                ),
                "upstream_metrics_json": stable_json(record["upstreamMetrics"]),
                "compatibility_metadata_json": stable_json(record["compatibilityMetadata"]),
                "evidence_caveat": record.get("evidenceCaveat", ""),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(["source_category", "display_name"], kind="mergesort").reset_index(drop=True)


def attach_competence_vectors(records: Sequence[Mapping[str, Any]], metadata: pd.DataFrame) -> list[dict[str, Any]]:
    meta_by_id = {str(row["algotype_id"]): dict(row) for row in metadata.to_dict(orient="records")}
    enriched: list[dict[str, Any]] = []
    for record in records:
        row = meta_by_id[str(record["algotypeId"])]
        competence = {
            "schema": "eidosoma.e06.s01_baseline_competence_vector.v1",
            "validationRunCount": int(row["validation_run_count"]),
            "pureExecutionSuccess": bool(row["pure_execution_success"]),
            "invalidRunCount": int(row["invalid_run_count"]),
            "sortedRunFraction": float(row["sorted_run_fraction"]),
            "meanInitialInversionSortedness": float(row["mean_initial_inversion_sortedness"]),
            "meanFinalInversionSortedness": float(row["mean_final_inversion_sortedness"]),
            "meanInversionSortednessDelta": float(row["mean_inversion_sortedness_delta"]),
            "meanWorkCount": float(row["mean_work_count"]),
            "meanSwapCount": float(row["mean_swap_count"]),
            "baselineCompetenceScore": float(row["baseline_competence_score"]),
        }
        enriched.append({**dict(record), "baselineCompetenceVector": competence})
    return enriched


def validation_checks(records: Sequence[Mapping[str, Any]], metadata: pd.DataFrame, validation: pd.DataFrame, source_status: Mapping[str, Any]) -> pd.DataFrame:
    categories = set(str(record["sourceCategory"]) for record in records)
    original = metadata[metadata["source_category"] == "original"]
    nulls = metadata[metadata["source_category"] == "null_control"]
    checks = [
        {
            "validation_case": "stable_ids_unique",
            "success": len({record["algotypeId"] for record in records}) == len(records),
            "observed": f"{len(records)} records",
            "expected": "all algotype IDs unique",
        },
        {
            "validation_case": "required_source_categories_present",
            "success": {"original", "null_control", "randomized", "discovered", "memory_repair"}.issubset(categories),
            "observed": ",".join(sorted(categories)),
            "expected": "original,null_control,randomized,discovered,memory_repair",
        },
        {
            "validation_case": "e03_frontier_candidates_available",
            "success": int(source_status.get("discovered", {}).get("loaded", 0)) > 0,
            "observed": str(source_status.get("discovered", {}).get("loaded", 0)),
            "expected": "at least one E03 discovered frontier candidate",
        },
        {
            "validation_case": "e04_memory_policies_available",
            "success": int(source_status.get("memoryRepair", {}).get("loaded", 0)) > 0,
            "observed": str(source_status.get("memoryRepair", {}).get("loaded", 0)),
            "expected": "at least one E04 memory or repair-capable policy",
        },
        {
            "validation_case": "all_policies_execute_pure_arrays",
            "success": bool(metadata["pure_execution_success"].all()),
            "observed": str(int(metadata["invalid_run_count"].sum())),
            "expected": "zero invalid pure-policy validation runs",
        },
        {
            "validation_case": "originals_sort_small_pure_arrays",
            "success": bool((original["sorted_run_fraction"] == 1.0).all()) and len(original) == 3,
            "observed": original[["display_name", "sorted_run_fraction"]].to_dict(orient="records"),
            "expected": "three original policies with sorted_run_fraction=1.0",
        },
        {
            "validation_case": "null_controls_do_not_swap",
            "success": bool((nulls["mean_swap_count"] == 0.0).all()) and len(nulls) >= 2,
            "observed": nulls[["display_name", "mean_swap_count"]].to_dict(orient="records"),
            "expected": "all null controls have zero swaps in pure validation",
        },
        {
            "validation_case": "metadata_matches_library",
            "success": len(metadata) == len(records) and metadata["algotype_id"].nunique() == len(records),
            "observed": f"metadata={len(metadata)} records={len(records)} unique_metadata={metadata['algotype_id'].nunique()}",
            "expected": "one metadata row per Algotype",
        },
        {
            "validation_case": "validation_rows_complete",
            "success": len(validation) == 2 * len(records),
            "observed": f"{len(validation)} validation rows for {len(records)} records",
            "expected": "two validation rows per Algotype",
        },
    ]
    return pd.DataFrame(checks)


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                clean[key] = None
            elif hasattr(value, "item"):
                item = value.item()
                clean[key] = None if isinstance(item, float) and math.isnan(item) else item
            else:
                clean[key] = value
        out.append(clean)
    return out
