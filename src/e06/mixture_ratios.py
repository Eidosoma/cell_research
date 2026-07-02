"""E06 S02 bounded mixture-ratio sweeps for chimeric Algotypes."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import actor_schedule, initial_values, sortedness_metrics
from src.e03.policy_interface import PolicyAction
from src.e03.rule_dsl import DSLArrayState, DSLInterpreter, parse_policy
from src.e04.evolutionary_search import EvolutionGenome, S08LocalEvolutionPolicy
from src.e04.no_oracle_protocol import LOCAL_ONLY_PROTOCOL_ID, LocalTrainingObservation


EXPERIMENT_ID = "E06"
STEP_ID = "S02"
MIXTURE_SCHEMA = "eidosoma.e06.s02_mixture_ratio_run.v1"
SUMMARY_SCHEMA = "eidosoma.e06.s02_mixture_ratio_summary.v1"
VALIDATION_SCHEMA = "eidosoma.e06.s02_validation.v1"
DEFAULT_SEEDS = (2026070201, 2026070202, 2026070203, 2026070204)
PAIR_RATIOS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75)
THREE_WAY_RATIOS = (
    (0.01, 0.01, 0.98),
    (0.10, 0.10, 0.80),
    (0.25, 0.25, 0.50),
    (0.34, 0.33, 0.33),
)


@dataclass(frozen=True)
class S02Config:
    """Configuration for the bounded S02 sweep."""

    array_size: int = 100
    event_cap: int = 4_000
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    top_discovered_count: int = 4
    include_all_originals_and_controls: bool = True
    arrangement: str = "random_permutation_labels"
    scheduler: str = "cyclic_scan_seed_offset"


@dataclass(frozen=True)
class MixtureCondition:
    """One mixture condition before expansion over seeds."""

    condition_id: str
    condition_kind: str
    panel: str
    policy_ids: tuple[str, ...]
    ratio_targets: tuple[float, ...]
    label: str


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_s01_library(library_path: Path, metadata_path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if not library_path.exists():
        raise FileNotFoundError(f"S01 Algotype library is missing: {library_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"S01 metadata table is missing: {metadata_path}")
    records = read_jsonl(library_path)
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    if not records:
        raise ValueError("S01 Algotype library is empty")
    if metadata.empty:
        raise ValueError("S01 metadata table is empty")
    record_ids = {str(record["algotypeId"]) for record in records}
    metadata_ids = set(metadata["algotype_id"].astype(str))
    missing = record_ids - metadata_ids
    if missing:
        raise ValueError(f"S01 metadata lacks {len(missing)} library IDs")
    return records, metadata


def _rank_number(display_name: str) -> int:
    try:
        return int(str(display_name).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10_000


def select_s02_panel(records: Sequence[Mapping[str, Any]], metadata: pd.DataFrame, *, top_discovered_count: int = 4) -> list[str]:
    """Select a bounded S02 policy panel from the S01 library."""

    meta = metadata.copy()
    meta["algotype_id"] = meta["algotype_id"].astype(str)
    selected: list[str] = []

    def add(ids: Sequence[str]) -> None:
        for policy_id in ids:
            if policy_id not in selected:
                selected.append(policy_id)

    add(meta[meta["source_category"] == "original"].sort_values("display_name")["algotype_id"].tolist())
    add(meta[meta["source_category"] == "null_control"].sort_values("display_name")["algotype_id"].tolist())
    add(meta[meta["source_category"] == "randomized"].sort_values("display_name")["algotype_id"].tolist())
    discovered = meta[meta["source_category"] == "discovered"].copy()
    if not discovered.empty:
        discovered["rank_number"] = discovered["display_name"].map(_rank_number)
        discovered["baseline_competence_score"] = pd.to_numeric(discovered["baseline_competence_score"], errors="coerce")
        discovered = discovered.sort_values(["rank_number", "baseline_competence_score", "display_name"], ascending=[True, False, True])
        add(discovered.head(top_discovered_count)["algotype_id"].tolist())
    add(meta[meta["source_category"] == "memory_repair"].sort_values("display_name")["algotype_id"].tolist())

    record_ids = {str(record["algotypeId"]) for record in records}
    return [policy_id for policy_id in selected if policy_id in record_ids]


def _by_display(metadata: pd.DataFrame) -> dict[str, str]:
    return {str(row["display_name"]): str(row["algotype_id"]) for row in metadata.to_dict(orient="records")}


def _condition_id(kind: str, policy_ids: Sequence[str], ratios: Sequence[float]) -> str:
    payload = {
        "kind": kind,
        "policyIds": list(policy_ids),
        "ratios": [round(float(value), 6) for value in ratios],
    }
    return f"s02_{kind}_{stable_hash(payload)[:14]}"


def build_bounded_conditions(panel_ids: Sequence[str], metadata: pd.DataFrame) -> list[MixtureCondition]:
    """Build prioritized pairwise and three-way S02 conditions."""

    display_to_id = _by_display(metadata)
    panel_set = set(panel_ids)

    def pid(name: str) -> str | None:
        value = display_to_id.get(name)
        return value if value in panel_set else None

    pairs_by_panel: list[tuple[str, str, str]] = []
    for left, right in (
        ("original_bubble", "original_insertion"),
        ("original_bubble", "original_selection"),
        ("original_insertion", "original_selection"),
        ("original_bubble", "null_wait"),
        ("original_bubble", "null_compare_only"),
        ("original_bubble", "random_adjacent_walk"),
        ("original_bubble", "random_noisy_inversion_cleaner"),
    ):
        if pid(left) and pid(right):
            pairs_by_panel.append(("original_control", pid(left), pid(right)))

    discovered = (
        metadata[(metadata["source_category"] == "discovered") & (metadata["algotype_id"].isin(panel_set))]
        .assign(rank_number=lambda frame: frame["display_name"].map(_rank_number))
        .sort_values(["rank_number", "display_name"], kind="mergesort")
    )
    top_discovered = discovered["algotype_id"].tolist()
    originals = [pid("original_bubble"), pid("original_insertion"), pid("original_selection")]
    originals = [item for item in originals if item is not None]
    if top_discovered:
        for original_id in originals:
            pairs_by_panel.append(("frontier_vs_original", original_id, top_discovered[0]))
        for original_id in originals[:2]:
            if len(top_discovered) > 1:
                pairs_by_panel.append(("frontier_vs_original", original_id, top_discovered[1]))
        if len(top_discovered) > 1:
            pairs_by_panel.append(("frontier_frontier", top_discovered[0], top_discovered[1]))

    memory = (
        metadata[(metadata["source_category"] == "memory_repair") & (metadata["algotype_id"].isin(panel_set))]
        .sort_values("display_name")["algotype_id"]
        .tolist()
    )
    for mem_id in memory:
        if pid("original_bubble"):
            pairs_by_panel.append(("memory_vs_original", pid("original_bubble"), mem_id))
        if top_discovered:
            pairs_by_panel.append(("memory_vs_frontier", top_discovered[0], mem_id))

    seen_pairs: set[tuple[str, str, tuple[float, ...]]] = set()
    conditions: list[MixtureCondition] = []
    for panel, left, right in pairs_by_panel:
        if left is None or right is None or left == right:
            continue
        for ratio in PAIR_RATIOS:
            ratios = (float(ratio), float(1.0 - ratio))
            key = (left, right, ratios)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            condition_id = _condition_id("pair", (left, right), ratios)
            conditions.append(
                MixtureCondition(
                    condition_id=condition_id,
                    condition_kind="pair",
                    panel=panel,
                    policy_ids=(left, right),
                    ratio_targets=ratios,
                    label=f"{panel}:{left[:10]}:{right[:10]}:{int(round(ratio * 100))}-{int(round((1-ratio) * 100))}",
                )
            )

    three_way_specs = [
        ("three_originals", (pid("original_bubble"), pid("original_insertion"), pid("original_selection"))),
        ("original_control_random", (pid("original_bubble"), pid("null_wait"), pid("random_noisy_inversion_cleaner"))),
        (
            "original_frontier_memory",
            (pid("original_bubble"), top_discovered[0] if top_discovered else None, memory[0] if memory else None),
        ),
    ]
    for panel, ids in three_way_specs:
        policy_ids = tuple(item for item in ids if item is not None)
        if len(policy_ids) != 3 or len(set(policy_ids)) != 3:
            continue
        for ratios in THREE_WAY_RATIOS:
            condition_id = _condition_id("threeway", policy_ids, ratios)
            conditions.append(
                MixtureCondition(
                    condition_id=condition_id,
                    condition_kind="threeway",
                    panel=panel,
                    policy_ids=policy_ids,
                    ratio_targets=tuple(float(value) for value in ratios),
                    label=f"{panel}:{'-'.join(str(int(round(r*100))) for r in ratios)}",
                )
            )
    return conditions


def counts_for_ratios(ratios: Sequence[float], total: int) -> tuple[int, ...]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not ratios:
        raise ValueError("at least one ratio is required")
    if any(float(value) < 0.0 for value in ratios):
        raise ValueError("ratios must be non-negative")
    ratio_sum = sum(float(value) for value in ratios)
    if ratio_sum <= 0:
        raise ValueError("ratio sum must be positive")
    normalized = [float(value) / ratio_sum for value in ratios]
    raw = [value * total for value in normalized]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - counts[idx], normalized[idx]), reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    for idx, ratio in enumerate(normalized):
        if ratio > 0 and counts[idx] == 0:
            donor = max(range(len(counts)), key=lambda item: counts[item])
            if counts[donor] <= 1:
                raise ValueError("total is too small to allocate positive ratios")
            counts[donor] -= 1
            counts[idx] = 1
    if sum(counts) != total:
        raise RuntimeError("ratio allocation failed")
    return tuple(counts)


def build_initial_policy_ids(policy_ids: Sequence[str], counts: Sequence[int], *, seed: int, condition_id: str) -> tuple[str, ...]:
    labels: list[str] = []
    for policy_id, count in zip(policy_ids, counts, strict=True):
        labels.extend([str(policy_id)] * int(count))
    rng = random.Random(seed ^ int(stable_hash({"condition": condition_id, "labels": labels})[:12], 16))
    rng.shuffle(labels)
    return tuple(labels)


def contiguous_run_count(labels: Sequence[str]) -> int:
    if not labels:
        return 0
    return 1 + sum(1 for idx in range(1, len(labels)) if labels[idx] != labels[idx - 1])


def aggregation_metrics(labels: Sequence[str]) -> dict[str, Any]:
    n = len(labels)
    if n == 0:
        return {
            "aggregation_left_neighbor_percent": 0.0,
            "expected_random_left_neighbor_percent": 0.0,
            "aggregation_delta_percent": 0.0,
            "interface_count": 0,
            "contiguous_run_count": 0,
            "largest_block_fraction": 0.0,
            "label_entropy": 0.0,
        }
    same_left = sum(1 for idx in range(1, n) if labels[idx] == labels[idx - 1])
    counts = Counter(labels)
    expected = 100.0 * sum(count * (count - 1) for count in counts.values()) / (n * n)
    runs: list[int] = []
    current = 1
    for idx in range(1, n):
        if labels[idx] == labels[idx - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log(p, 2)
    max_entropy = math.log(max(len(counts), 1), 2) if counts else 1.0
    return {
        "aggregation_left_neighbor_percent": 100.0 * same_left / n,
        "expected_random_left_neighbor_percent": expected,
        "aggregation_delta_percent": 100.0 * same_left / n - expected,
        "interface_count": int((n - 1) - same_left),
        "contiguous_run_count": contiguous_run_count(labels),
        "largest_block_fraction": max(runs) / n,
        "label_entropy": 0.0 if max_entropy == 0 else entropy / max_entropy,
    }


def _safe_mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def dominance_metrics(labels: Sequence[str], policy_ids: Sequence[str]) -> dict[str, Any]:
    positions_by_policy: dict[str, list[float]] = {policy_id: [] for policy_id in policy_ids}
    denom = max(1, len(labels) - 1)
    for idx, policy_id in enumerate(labels):
        positions_by_policy.setdefault(policy_id, []).append(idx / denom)
    means = {policy_id: _safe_mean(values) for policy_id, values in positions_by_policy.items()}
    present = {policy_id: mean for policy_id, mean in means.items() if mean is not None}
    if not present:
        return {"leftmost_policy_id": None, "rightmost_policy_id": None, "position_bias_margin": 0.0, "mean_position_by_policy_json": "{}"}
    leftmost = min(present, key=lambda key: present[key])
    rightmost = max(present, key=lambda key: present[key])
    return {
        "leftmost_policy_id": leftmost,
        "rightmost_policy_id": rightmost,
        "position_bias_margin": float(present[rightmost] - present[leftmost]),
        "mean_position_by_policy_json": json.dumps(present, sort_keys=True, separators=(",", ":")),
    }


def classify_final_state(final_sortedness: float, aggregation_delta: float, largest_block_fraction: float, interface_count: int) -> str:
    if final_sortedness >= 0.98 and aggregation_delta >= 10.0:
        return "sorted_aggregated"
    if final_sortedness >= 0.98:
        return "sorted_mixed"
    if final_sortedness >= 0.85 and aggregation_delta >= 10.0:
        return "partial_sort_aggregated"
    if final_sortedness >= 0.85:
        return "partial_sort_mixed"
    if largest_block_fraction >= 0.40 and interface_count <= 20:
        return "segregated_low_sort"
    if final_sortedness < 0.50:
        return "low_progress"
    return "mixed_intermediate"


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


class MixtureRuntime:
    """Stateful policy dispatcher for one chimeric 1D array run."""

    def __init__(self, records: Sequence[Mapping[str, Any]], seed: int) -> None:
        self.records = {str(record["algotypeId"]): dict(record) for record in records}
        self.seed = int(seed)
        self.dsl: dict[str, DSLInterpreter] = {}
        self.memory_policies: dict[str, S08LocalEvolutionPolicy] = {}
        for record in records:
            policy_id = str(record["algotypeId"])
            if record["executionBackend"] == "e03_dsl_cpu_interpreter":
                self.dsl[policy_id] = DSLInterpreter(parse_policy(str(record["dslSource"])))
            elif record["executionBackend"] == "e04_local_training_policy":
                parameters = {str(key): float(value) for key, value in dict(record["parameters"]).items()}
                genome = EvolutionGenome(
                    policy_id=str(record["sourcePolicyId"]),
                    generation=0,
                    population_index=0,
                    parent_ids=(),
                    mutation_seed=seed,
                    mutation_scale=0.0,
                    parameters=parameters,
                    lineage_note="e06_s02_mixture_replay_from_e04_artifact",
                )
                self.memory_policies[policy_id] = S08LocalEvolutionPolicy(genome)

    @staticmethod
    def _target_exists(values: Sequence[int], idx: int | None) -> bool:
        return idx is not None and 0 <= int(idx) < len(values)

    @staticmethod
    def _prefix_sorted(values: Sequence[int], actor_index: int) -> bool:
        return all(values[idx - 1] <= values[idx] for idx in range(1, actor_index))

    def _original_action(
        self,
        record: Mapping[str, Any],
        values: Sequence[int],
        actor_index: int,
        cell_id: int,
        memory: dict[int, dict[str, Any]],
        rng: random.Random,
    ) -> PolicyAction:
        algorithm = str(record["algorithm"])
        if algorithm == "bubble":
            target = actor_index + 1 if rng.random() < 0.5 else actor_index - 1
            compare = self._target_exists(values, target)
            should_swap = compare and (
                (target > actor_index and values[actor_index] > values[target])
                or (target < actor_index and values[actor_index] < values[target])
            )
            return PolicyAction("swap" if should_swap else "wait", target_index=target, compare_counted=compare)
        if algorithm == "insertion":
            target = actor_index - 1
            compare = self._target_exists(values, target) and self._prefix_sorted(values, actor_index)
            should_swap = compare and values[actor_index] < values[target]
            return PolicyAction("swap" if should_swap else "wait", target_index=target, compare_counted=compare)
        if algorithm == "selection":
            state = memory.setdefault(cell_id, {})
            ideal = state.get("ideal_position")
            if ideal is None or int(ideal) < 0 or int(ideal) >= len(values):
                ideal = 0
                state["ideal_position"] = ideal
            target = int(ideal)
            if actor_index == target:
                return PolicyAction("wait", target_index=target, compare_counted=False)
            compare = self._target_exists(values, target)
            if compare and values[actor_index] < values[target]:
                return PolicyAction("swap", target_index=target, compare_counted=True)
            next_ideal = min(len(values) - 1, target + 1)
            return PolicyAction("update_state", target_index=target, compare_counted=compare, state_update={"ideal_position": next_ideal})
        return PolicyAction("wait")

    def _dsl_action(
        self,
        policy_id: str,
        values: Sequence[int],
        labels: Sequence[str],
        actor_index: int,
        cell_id: int,
        memory: dict[int, dict[str, Any]],
        rng: random.Random,
    ) -> PolicyAction:
        state = memory.setdefault(cell_id, {})
        ideal = state.get("ideal_position")
        dsl_state = DSLArrayState(
            values=tuple(values),
            labels=tuple(labels),
            statuses=tuple("ACTIVE" for _ in values),
            actor_index=int(actor_index),
            ideal_position=None if ideal is None else int(ideal),
        )
        result = self.dsl[policy_id].step_state(dsl_state, rng)
        if state.get("ideal_position") is None and result.state_before.ideal_position is not None:
            state["ideal_position"] = int(result.state_before.ideal_position)
        if "ideal_position" in result.action.state_update:
            updated = result.action.state_update["ideal_position"]
            state["ideal_position"] = None if updated is None else int(updated)
        return result.action

    def _memory_action(
        self,
        policy_id: str,
        values: Sequence[int],
        actor_index: int,
        cell_id: int,
        memory: dict[int, dict[str, Any]],
    ) -> PolicyAction:
        state = memory.setdefault(cell_id, {})
        observation = _memory_observation(values=values, actor_index=actor_index, memory=state)
        decision = self.memory_policies[policy_id].select_action(observation)
        if decision.action == "swap_left":
            target = actor_index - 1
        elif decision.action == "swap_right":
            target = actor_index + 1
        else:
            return PolicyAction("wait", metadata={"decision": decision.action})
        return PolicyAction("swap", target_index=target, compare_counted=True, metadata={"decision": decision.action})

    def action_for(
        self,
        policy_id: str,
        values: Sequence[int],
        labels: Sequence[str],
        actor_index: int,
        cell_id: int,
        memory: dict[int, dict[str, Any]],
        rng: random.Random,
    ) -> PolicyAction:
        record = self.records[policy_id]
        backend = str(record["executionBackend"])
        if backend == "e02_public_cell_simulator":
            return self._original_action(record, values, actor_index, cell_id, memory, rng)
        if backend == "e03_dsl_cpu_interpreter":
            return self._dsl_action(policy_id, values, labels, actor_index, cell_id, memory, rng)
        if backend == "e04_local_training_policy":
            return self._memory_action(policy_id, values, actor_index, cell_id, memory)
        return PolicyAction("wait", metadata={"unsupported_backend": backend})


def simulate_mixture_condition(
    condition: MixtureCondition,
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    config: S02Config,
) -> dict[str, Any]:
    counts = counts_for_ratios(condition.ratio_targets, config.array_size)
    values = list(initial_values(config.array_size, seed))
    labels = list(build_initial_policy_ids(condition.policy_ids, counts, seed=seed, condition_id=condition.condition_id))
    cell_ids = list(range(config.array_size))
    memory: dict[int, dict[str, Any]] = {}
    runtime = MixtureRuntime([records_by_id[policy_id] for policy_id in condition.policy_ids], seed)
    schedule = actor_schedule(config.array_size, config.event_cap, seed)
    rng = random.Random(seed ^ int(stable_hash(condition.condition_id)[:12], 16))
    compare_count = 0
    swap_count = 0
    update_count = 0
    wait_count = 0
    invalid_action_count = 0
    started = time.perf_counter()
    initial_metrics = sortedness_metrics(values)
    initial_agg = aggregation_metrics(labels)
    initial_counts = Counter(labels)
    for actor_index in schedule:
        actor_index = int(actor_index)
        policy_id = labels[actor_index]
        cell_id = cell_ids[actor_index]
        action = runtime.action_for(policy_id, values, labels, actor_index, cell_id, memory, rng)
        compare_count += int(bool(action.compare_counted))
        if action.action_type == "swap" and action.target_index is not None:
            target = int(action.target_index)
            if 0 <= target < len(values):
                values[actor_index], values[target] = values[target], values[actor_index]
                labels[actor_index], labels[target] = labels[target], labels[actor_index]
                cell_ids[actor_index], cell_ids[target] = cell_ids[target], cell_ids[actor_index]
                swap_count += 1
                memory.setdefault(cell_id, {})["last_move_success"] = True
                memory.setdefault(cell_id, {})["last_action_type"] = "swap"
                memory.setdefault(cell_id, {})["time_since_movement"] = 0
            else:
                invalid_action_count += 1
                wait_count += 1
                state = memory.setdefault(cell_id, {})
                state["last_move_success"] = False
                state["last_action_type"] = "invalid_swap"
                state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
        elif action.action_type == "update_state":
            update_count += 1
            state = memory.setdefault(cell_id, {})
            if "ideal_position" in action.state_update:
                updated = action.state_update["ideal_position"]
                state["ideal_position"] = None if updated is None else int(updated)
            state["last_action_type"] = "update_state"
        else:
            wait_count += 1
            state = memory.setdefault(cell_id, {})
            state["last_move_success"] = None
            state["last_action_type"] = "wait"
            state["time_since_movement"] = min(255, int(state.get("time_since_movement", 0)) + 1)
    elapsed = time.perf_counter() - started
    final_metrics = sortedness_metrics(values)
    final_agg = aggregation_metrics(labels)
    final_counts = Counter(labels)
    dominance = dominance_metrics(labels, condition.policy_ids)
    realized_ratios = tuple(final_counts.get(policy_id, 0) / config.array_size for policy_id in condition.policy_ids)
    expected_counts = dict(zip(condition.policy_ids, counts, strict=True))
    final_state_class = classify_final_state(
        float(final_metrics["inversion_sortedness"]),
        float(final_agg["aggregation_delta_percent"]),
        float(final_agg["largest_block_fraction"]),
        int(final_agg["interface_count"]),
    )
    display_names = [str(records_by_id[policy_id]["displayName"]) for policy_id in condition.policy_ids]
    categories = [str(records_by_id[policy_id]["sourceCategory"]) for policy_id in condition.policy_ids]
    return {
        "schema": MIXTURE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "research_step_id": STEP_ID,
        "condition_id": condition.condition_id,
        "condition_kind": condition.condition_kind,
        "panel": condition.panel,
        "condition_label": condition.label,
        "array_size": int(config.array_size),
        "event_cap": int(config.event_cap),
        "events_executed": int(config.event_cap),
        "seed": int(seed),
        "scheduler": config.scheduler,
        "arrangement": config.arrangement,
        "policy_ids_json": json.dumps(list(condition.policy_ids), separators=(",", ":")),
        "display_names_json": json.dumps(display_names, separators=(",", ":")),
        "source_categories_json": json.dumps(categories, separators=(",", ":")),
        "ratio_targets_json": json.dumps(list(condition.ratio_targets), separators=(",", ":")),
        "expected_counts_json": json.dumps(expected_counts, sort_keys=True, separators=(",", ":")),
        "initial_counts_json": json.dumps(dict(sorted(initial_counts.items())), sort_keys=True, separators=(",", ":")),
        "final_counts_json": json.dumps(dict(sorted(final_counts.items())), sort_keys=True, separators=(",", ":")),
        "realized_ratios_json": json.dumps(list(realized_ratios), separators=(",", ":")),
        "realized_ratio_max_abs_error": float(max(abs(realized_ratios[idx] - (counts[idx] / config.array_size)) for idx in range(len(counts)))),
        "count_preservation_success": dict(initial_counts) == dict(final_counts) == expected_counts,
        "compare_count": int(compare_count),
        "swap_count": int(swap_count),
        "update_count": int(update_count),
        "wait_count": int(wait_count),
        "work_count": int(compare_count + swap_count + update_count),
        "invalid_action_count": int(invalid_action_count),
        "initial_inversion_count": int(initial_metrics["inversion_count"]),
        "final_inversion_count": int(final_metrics["inversion_count"]),
        "initial_inversion_sortedness": float(initial_metrics["inversion_sortedness"]),
        "final_inversion_sortedness": float(final_metrics["inversion_sortedness"]),
        "inversion_sortedness_delta": float(final_metrics["inversion_sortedness"] - initial_metrics["inversion_sortedness"]),
        "initial_adjacent_sortedness": float(initial_metrics["adjacent_sortedness"]),
        "final_adjacent_sortedness": float(final_metrics["adjacent_sortedness"]),
        "final_is_sorted": bool(final_metrics["is_sorted"]),
        "initial_aggregation_left_neighbor_percent": float(initial_agg["aggregation_left_neighbor_percent"]),
        "final_aggregation_left_neighbor_percent": float(final_agg["aggregation_left_neighbor_percent"]),
        "expected_random_left_neighbor_percent": float(final_agg["expected_random_left_neighbor_percent"]),
        "aggregation_delta_percent": float(final_agg["aggregation_delta_percent"]),
        "interface_count": int(final_agg["interface_count"]),
        "contiguous_run_count": int(final_agg["contiguous_run_count"]),
        "largest_block_fraction": float(final_agg["largest_block_fraction"]),
        "label_entropy": float(final_agg["label_entropy"]),
        "leftmost_policy_id": dominance["leftmost_policy_id"],
        "rightmost_policy_id": dominance["rightmost_policy_id"],
        "position_bias_margin": float(dominance["position_bias_margin"]),
        "mean_position_by_policy_json": dominance["mean_position_by_policy_json"],
        "final_state_class": final_state_class,
        "final_values_head_json": json.dumps(values[:20], separators=(",", ":")),
        "final_values_tail_json": json.dumps(values[-20:], separators=(",", ":")),
        "final_labels_head_json": json.dumps(labels[:20], separators=(",", ":")),
        "final_labels_tail_json": json.dumps(labels[-20:], separators=(",", ":")),
        "elapsed_seconds": float(elapsed),
    }


def run_s02_sweep(
    records: Sequence[Mapping[str, Any]],
    metadata: pd.DataFrame,
    config: S02Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or S02Config()
    panel_ids = select_s02_panel(records, metadata, top_discovered_count=config.top_discovered_count)
    conditions = build_bounded_conditions(panel_ids, metadata)
    records_by_id = {str(record["algotypeId"]): dict(record) for record in records}
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in config.seeds:
            rows.append(simulate_mixture_condition(condition, records_by_id, seed=int(seed), config=config))
    run_df = pd.DataFrame(rows)
    condition_rows = [
        {
            "condition_id": condition.condition_id,
            "condition_kind": condition.condition_kind,
            "panel": condition.panel,
            "condition_label": condition.label,
            "policy_ids_json": json.dumps(list(condition.policy_ids), separators=(",", ":")),
            "display_names_json": json.dumps([records_by_id[pid]["displayName"] for pid in condition.policy_ids], separators=(",", ":")),
            "source_categories_json": json.dumps([records_by_id[pid]["sourceCategory"] for pid in condition.policy_ids], separators=(",", ":")),
            "ratio_targets_json": json.dumps(list(condition.ratio_targets), separators=(",", ":")),
            "expected_counts_json": json.dumps(
                dict(zip(condition.policy_ids, counts_for_ratios(condition.ratio_targets, config.array_size), strict=True)),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for condition in conditions
    ]
    condition_df = pd.DataFrame(condition_rows)
    return run_df, condition_df, pd.DataFrame({"algotype_id": panel_ids})


def summarize_s02_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_id, group in run_df.groupby("condition_id", sort=False):
        first = group.iloc[0]
        class_counts = group["final_state_class"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "research_step_id": STEP_ID,
                "condition_id": condition_id,
                "condition_kind": first["condition_kind"],
                "panel": first["panel"],
                "condition_label": first["condition_label"],
                "policy_ids_json": first["policy_ids_json"],
                "display_names_json": first["display_names_json"],
                "source_categories_json": first["source_categories_json"],
                "ratio_targets_json": first["ratio_targets_json"],
                "expected_counts_json": first["expected_counts_json"],
                "seed_count": int(group["seed"].nunique()),
                "run_count": int(len(group)),
                "mean_final_inversion_sortedness": float(group["final_inversion_sortedness"].mean()),
                "sd_final_inversion_sortedness": float(group["final_inversion_sortedness"].std(ddof=0)),
                "mean_inversion_sortedness_delta": float(group["inversion_sortedness_delta"].mean()),
                "sorted_run_fraction": float(group["final_is_sorted"].mean()),
                "mean_aggregation_delta_percent": float(group["aggregation_delta_percent"].mean()),
                "mean_interface_count": float(group["interface_count"].mean()),
                "mean_largest_block_fraction": float(group["largest_block_fraction"].mean()),
                "mean_position_bias_margin": float(group["position_bias_margin"].mean()),
                "mean_work_count": float(group["work_count"].mean()),
                "invalid_action_count": int(group["invalid_action_count"].sum()),
                "final_state_class_mode": str(group["final_state_class"].mode().iloc[0]),
                "final_state_class_counts_json": json.dumps(class_counts, sort_keys=True, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def detect_threshold_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Find coarse ratio-response candidates for follow-up."""

    rows: list[dict[str, Any]] = []
    pairwise = summary_df[summary_df["condition_kind"] == "pair"].copy()
    if pairwise.empty:
        return pd.DataFrame()
    pairwise["policy_pair"] = pairwise["policy_ids_json"]
    pairwise["first_ratio"] = pairwise["ratio_targets_json"].map(lambda text: json.loads(text)[0])
    for pair, group in pairwise.groupby("policy_pair", sort=False):
        group = group.sort_values("first_ratio")
        sortedness = group["mean_final_inversion_sortedness"].astype(float).to_numpy()
        aggregation = group["mean_aggregation_delta_percent"].astype(float).to_numpy()
        ratios = group["first_ratio"].astype(float).to_numpy()
        if len(group) < 2:
            continue
        sortedness_range = float(np.max(sortedness) - np.min(sortedness))
        aggregation_range = float(np.max(aggregation) - np.min(aggregation))
        max_jump_idx = int(np.argmax(np.abs(np.diff(sortedness)))) if len(sortedness) > 1 else 0
        rows.append(
            {
                "policy_ids_json": pair,
                "display_names_json": str(group["display_names_json"].iloc[0]),
                "panel": str(group["panel"].iloc[0]),
                "sortedness_range": sortedness_range,
                "aggregation_delta_range": aggregation_range,
                "largest_sortedness_jump": float(np.max(np.abs(np.diff(sortedness)))) if len(sortedness) > 1 else 0.0,
                "candidate_threshold_between_ratios_json": json.dumps(
                    [float(ratios[max_jump_idx]), float(ratios[max_jump_idx + 1])], separators=(",", ":")
                )
                if len(ratios) > 1
                else "[]",
                "minority_effect_candidate": bool(sortedness[0] > sortedness[-1] + 0.05 or sortedness[1] > sortedness[-1] + 0.05),
                "aggregation_sensitive_candidate": bool(aggregation_range >= 10.0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["largest_sortedness_jump", "aggregation_delta_range"],
        ascending=[False, False],
        kind="mergesort",
    )


def validate_s02_outputs(
    run_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    config: S02Config,
    *,
    figure_written: bool,
    unit_tests_success: bool,
) -> pd.DataFrame:
    seed_set = set(int(seed) for seed in config.seeds)
    required_run_columns = {
        "condition_id",
        "seed",
        "realized_ratio_max_abs_error",
        "count_preservation_success",
        "invalid_action_count",
    }
    required_condition_columns = {"condition_id", "condition_kind", "source_categories_json"}
    has_run_schema = required_run_columns.issubset(run_df.columns)
    has_condition_schema = required_condition_columns.issubset(condition_df.columns)
    if has_run_schema:
        seed_sets = run_df.groupby("condition_id")["seed"].agg(lambda values: set(int(value) for value in values)).tolist()
    else:
        seed_sets = []
    panel_categories: set[str] = set()
    if has_condition_schema:
        for text in condition_df["source_categories_json"].tolist():
            panel_categories.update(json.loads(text))
    ratio_success = False
    ratio_observed = "missing required run columns"
    if has_run_schema and not run_df.empty:
        ratio_success = bool((run_df["realized_ratio_max_abs_error"] <= 1e-12).all() and run_df["count_preservation_success"].all())
        ratio_observed = (
            f"max_error={run_df['realized_ratio_max_abs_error'].max():.3g}; "
            f"count_failures={int((~run_df['count_preservation_success']).sum())}"
        )
    invalid_success = False
    invalid_observed = "missing required run columns"
    if has_run_schema and not run_df.empty:
        invalid_count = int(run_df["invalid_action_count"].sum())
        invalid_success = invalid_count == 0
        invalid_observed = str(invalid_count)
    result_rows_success = False
    result_rows_observed = f"runs={len(run_df)} conditions={len(condition_df)} seeds={len(seed_set)}"
    if has_run_schema and has_condition_schema and not condition_df.empty:
        result_rows_success = len(run_df) == len(condition_df) * len(seed_set)
    checks = [
        {
            "validation_case": "realized_ratios_match_expected_counts",
            "success": ratio_success,
            "observed": ratio_observed,
            "expected": "zero realized-ratio error against integer counts and all counts preserved",
        },
        {
            "validation_case": "balanced_seed_sets_by_condition",
            "success": bool(seed_sets and all(item == seed_set for item in seed_sets)),
            "observed": f"{len(seed_sets)} conditions; expected_seeds={sorted(seed_set)}",
            "expected": "each condition uses the same configured seed set",
        },
        {
            "validation_case": "bounded_panel_includes_priority_categories",
            "success": {"original", "null_control", "randomized", "discovered", "memory_repair"}.issubset(panel_categories),
            "observed": ",".join(sorted(panel_categories)),
            "expected": "original,null_control,randomized,discovered,memory_repair",
        },
        {
            "validation_case": "pair_and_threeway_conditions_present",
            "success": has_condition_schema and {"pair", "threeway"}.issubset(set(condition_df["condition_kind"])),
            "observed": condition_df["condition_kind"].value_counts().sort_index().to_dict() if has_condition_schema else "missing required condition columns",
            "expected": "at least one pair and one threeway condition",
        },
        {
            "validation_case": "no_invalid_actions",
            "success": invalid_success,
            "observed": invalid_observed,
            "expected": "zero out-of-bounds or invalid actions",
        },
        {
            "validation_case": "result_rows_complete",
            "success": result_rows_success,
            "observed": result_rows_observed,
            "expected": "run rows equal condition count times seed count",
        },
        {
            "validation_case": "figure_written",
            "success": bool(figure_written),
            "observed": str(bool(figure_written)),
            "expected": "mixture-ratio phase figure exists",
        },
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests_success),
            "observed": str(bool(unit_tests_success)),
            "expected": "focused E06 unit tests pass",
        },
    ]
    return pd.DataFrame(checks)
