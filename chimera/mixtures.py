"""E06 S02 mixture-ratio sweeps for ready Algotype panel rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import aggregation, sortedness_percent, state_hash
from memory_repair import MemoryEventSimulator, SignalEventSimulator
from morphospace import PolicyEventSimulator

from .panel import CLAIM_BOUNDARY, instantiate_panel_policy, json_ready, parse_json_maybe


STEP_ID = "S02"
STEP_NUMBER = 2
MIXTURE_SWEEP_VERSION = "e06_s02_mixture_ratios.v1"
PAIR_RATIOS: tuple[tuple[int, int], ...] = ((1, 99), (5, 95), (10, 90), (25, 75), (50, 50), (75, 25))
THREE_WAY_PROFILES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("34_33_33", (34, 33, 33)),
    ("50_25_25", (50, 25, 25)),
)
PRIORITY_LIMITS = {
    "frontier": 4,
    "discovered": 2,
    "memory": 3,
    "signal": 3,
    "handoff": 2,
    "controls": 2,
}
DEFAULT_PANEL_PATH = Path("/artifacts/data/e06_algotype_panel.parquet")
_PANEL_LOOKUP: dict[str, dict[str, Any]] = {}


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: Sequence[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else out.stderr.strip()


def load_ready_panel(panel_path: Path = DEFAULT_PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    if "readyForS02Mixing" not in panel.columns:
        raise ValueError(f"panel lacks readyForS02Mixing column: {panel_path}")
    ready = panel[panel["readyForS02Mixing"].map(bool)].copy()
    if ready.empty:
        raise ValueError("no readyForS02Mixing=true policies available for S02")
    return ready.reset_index(drop=True)


def _summary_value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    payload = parse_json_maybe(row.get("competenceSummaryJson"), {})
    return payload.get(key, default) if isinstance(payload, Mapping) else default


def _sort_by_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["_rank"] = pd.to_numeric(out.get("sourceRank", np.nan), errors="coerce")
    return out.sort_values(["_rank", "panelPolicyId"], kind="mergesort").drop(columns=["_rank"])


def select_priority_policy_ids(ready_panel: pd.DataFrame) -> dict[str, list[str]]:
    """Select a bounded, auditable S02 priority panel from ready S01 rows."""

    groups: dict[str, list[str]] = {}
    groups["original"] = list(
        ready_panel[ready_panel["panelGroup"] == "original_classic"]
        .sort_values("panelPolicyId", kind="mergesort")["panelPolicyId"]
        .astype(str)
    )
    groups["frontier"] = list(
        _sort_by_rank(ready_panel[ready_panel["panelGroup"] == "e03_frontier"])
        .head(PRIORITY_LIMITS["frontier"])["panelPolicyId"]
        .astype(str)
    )
    groups["discovered"] = list(
        _sort_by_rank(ready_panel[ready_panel["panelGroup"] == "e03_selected_discovered"])
        .head(PRIORITY_LIMITS["discovered"])["panelPolicyId"]
        .astype(str)
    )

    memory = ready_panel[ready_panel["panelGroup"] == "e04_memory"].copy()
    if not memory.empty:
        memory["variant"] = memory.apply(lambda row: _summary_value(row, "memoryVariant", ""), axis=1)
        memory["variantRank"] = memory["variant"].map({"bounded_counter": 0, "neighbor_memory": 1, "one_bit": 2}).fillna(9)
        preferred_bases = [
            "classic_bubble",
            "classic_insertion",
            "classic_selection",
            "pc_b7934afa1e6b4306",
            "pc_86366d5224e8ff58",
            "pc_fcb4af2d6327c554",
        ]
        memory = memory[memory["classLabel"].isin(preferred_bases)]
        memory = memory.sort_values(["classLabel", "variantRank", "panelPolicyId"], kind="mergesort")
        memory = memory.groupby("classLabel", sort=False).head(1).head(PRIORITY_LIMITS["memory"])
    groups["memory"] = list(memory["panelPolicyId"].astype(str)) if not memory.empty else []

    signal = ready_panel[ready_panel["panelGroup"] == "e04_signal"].copy()
    if not signal.empty:
        signal["variant"] = signal.apply(lambda row: _summary_value(row, "signalVariant", ""), axis=1)
        signal["variantRank"] = signal["variant"].map({"diffusive": 0, "nearest_neighbor": 1, "inert": 2, "randomized": 3}).fillna(9)
        preferred_bases = [
            "classic_bubble",
            "classic_insertion",
            "classic_selection",
            "pc_b7934afa1e6b4306",
            "pc_86366d5224e8ff58",
            "pc_fcb4af2d6327c554",
        ]
        signal = signal[signal["classLabel"].isin(preferred_bases)]
        signal = signal.sort_values(["classLabel", "variantRank", "panelPolicyId"], kind="mergesort")
        signal = signal.groupby("classLabel", sort=False).head(1).head(PRIORITY_LIMITS["signal"])
    groups["signal"] = list(signal["panelPolicyId"].astype(str)) if not signal.empty else []

    groups["handoff"] = list(
        _sort_by_rank(ready_panel[ready_panel["panelGroup"] == "e04_handoff"])
        .head(PRIORITY_LIMITS["handoff"])["panelPolicyId"]
        .astype(str)
    )

    controls = ready_panel[ready_panel["panelGroup"].isin(["null_control", "randomized_control"])].copy()
    if not controls.empty:
        controls["_is_p50"] = controls["panelPolicyId"].astype(str).str.contains("p050")
        controls = controls.sort_values(["panelGroup", "_is_p50", "panelPolicyId"], ascending=[True, False, True])
    groups["controls"] = list(controls.head(PRIORITY_LIMITS["controls"])["panelPolicyId"].astype(str)) if not controls.empty else []
    return groups


def _add_pair(
    pairs: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    left: str,
    right: str,
    *,
    category: str,
    reason: str,
) -> None:
    if left == right:
        return
    key = (left, right, category)
    if key in seen:
        return
    seen.add(key)
    pairs.append(
        {
            "mixtureKind": "pair",
            "leftPanelPolicyId": left,
            "rightPanelPolicyId": right,
            "thirdPanelPolicyId": "",
            "pairCategory": category,
            "priorityReason": reason,
        }
    )


def priority_pair_records(priority: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    originals = list(priority.get("original", []))
    frontier = list(priority.get("frontier", []))
    discovered = list(priority.get("discovered", []))
    memory = list(priority.get("memory", []))
    signal = list(priority.get("signal", []))
    handoff = list(priority.get("handoff", []))
    controls = list(priority.get("controls", []))

    for i, left in enumerate(originals):
        for right in originals[i + 1 :]:
            _add_pair(pairs, seen, left, right, category="original_pair", reason="all original-classic pairings")

    for candidate in [*frontier, *discovered]:
        for original in originals:
            _add_pair(
                pairs,
                seen,
                candidate,
                original,
                category="e03_vs_original",
                reason="ready E03 frontier/discovered policy against each original policy",
            )

    for candidate in [*memory, *signal, *handoff]:
        for original in originals:
            _add_pair(
                pairs,
                seen,
                candidate,
                original,
                category="memory_signal_handoff_vs_original",
                reason="representative E04 memory/signaling/handoff policy against each original policy",
            )

    for i, left in enumerate(frontier):
        for right in frontier[i + 1 :]:
            _add_pair(
                pairs,
                seen,
                left,
                right,
                category="frontier_pair",
                reason="top ready E03 frontier pair",
            )

    for candidate in handoff:
        for target in frontier[:2]:
            _add_pair(
                pairs,
                seen,
                candidate,
                target,
                category="handoff_vs_frontier",
                reason="E04 handoff candidate against top frontier policies",
            )

    control_targets = [*originals, *frontier[:1]]
    for control in controls:
        for target in control_targets:
            _add_pair(
                pairs,
                seen,
                control,
                target,
                category="control_pair",
                reason="null/randomized control against original and top frontier policies",
            )
    return pairs


def priority_triplets(priority: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    originals = list(priority.get("original", []))
    frontier = list(priority.get("frontier", []))
    memory = list(priority.get("memory", []))
    signal = list(priority.get("signal", []))
    handoff = list(priority.get("handoff", []))
    triplets: list[dict[str, Any]] = []
    candidates = [
        (originals[:3], "original_three_way", "Bubble/Insertion/Selection three-way control"),
        ([frontier[0], frontier[1], originals[0]] if len(frontier) >= 2 and originals else [], "frontier_frontier_original", "top frontier pair with original policy"),
        ([memory[0], signal[0], originals[0]] if memory and signal and originals else [], "memory_signal_original", "representative memory and signal policies with original policy"),
        ([handoff[0], frontier[0], originals[0]] if handoff and frontier and originals else [], "handoff_frontier_original", "handoff candidate with frontier and original policy"),
    ]
    for ids, category, reason in candidates:
        if len(ids) == 3 and len(set(ids)) == 3:
            triplets.append(
                {
                    "mixtureKind": "three_way",
                    "leftPanelPolicyId": ids[0],
                    "rightPanelPolicyId": ids[1],
                    "thirdPanelPolicyId": ids[2],
                    "pairCategory": category,
                    "priorityReason": reason,
                }
            )
    return triplets


def seed_bundle(seed_index: int, base: int = 620_000) -> dict[str, int]:
    return {
        "seedIndex": int(seed_index),
        "valueSeed": int(base + seed_index * 1000 + 11),
        "arrangementSeed": int(base + seed_index * 1000 + 23),
        "schedulerSeed": int(base + seed_index * 1000 + 37),
        "tieBreakerSeed": int(base + seed_index * 1000 + 53),
    }


def build_condition_matrix(
    ready_panel: pd.DataFrame,
    *,
    n: int = 100,
    seed_count: int = 2,
    max_pairs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    priority = select_priority_policy_ids(ready_panel)
    pairs = priority_pair_records(priority)
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    triplets = priority_triplets(priority)
    rows: list[dict[str, Any]] = []

    for pair_index, pair in enumerate(pairs):
        for left_count, right_count in PAIR_RATIOS:
            if left_count + right_count != n:
                raise ValueError("pair ratios require n=100 for exact requested counts")
            ratio_label = f"{left_count}:{right_count}"
            for seed_index in range(seed_count):
                seeds = seed_bundle(seed_index, base=621_000 + pair_index * 100)
                ids = [pair["leftPanelPolicyId"], pair["rightPanelPolicyId"]]
                counts = [left_count, right_count]
                condition_key = compact_json({"kind": "pair", "ids": ids, "counts": counts, **seeds})
                rows.append(
                    {
                        "conditionId": f"s02_pair_{pair_index:04d}_{ratio_label.replace(':', '_')}_seed{seed_index}",
                        **pair,
                        "ratioLabel": ratio_label,
                        "targetCountsJson": compact_json(dict(zip(ids, counts, strict=True))),
                        "targetProportionsJson": compact_json({pid: count / n for pid, count in zip(ids, counts, strict=True)}),
                        "n": int(n),
                        "targetPolicyCount": len(ids),
                        "policyIdsJson": compact_json(ids),
                        "policyCountsJson": compact_json(counts),
                        "conditionHash": sha256_text(condition_key)[:16],
                        "s02Version": MIXTURE_SWEEP_VERSION,
                        **seeds,
                    }
                )

    for triplet_index, triplet in enumerate(triplets):
        ids = [triplet["leftPanelPolicyId"], triplet["rightPanelPolicyId"], triplet["thirdPanelPolicyId"]]
        for profile_name, counts in THREE_WAY_PROFILES:
            if sum(counts) != n:
                raise ValueError("three-way profiles require n=100")
            for seed_index in range(seed_count):
                seeds = seed_bundle(seed_index, base=741_000 + triplet_index * 100)
                condition_key = compact_json({"kind": "three_way", "ids": ids, "counts": counts, **seeds})
                rows.append(
                    {
                        "conditionId": f"s02_three_{triplet_index:02d}_{profile_name}_seed{seed_index}",
                        **triplet,
                        "ratioLabel": profile_name,
                        "targetCountsJson": compact_json(dict(zip(ids, counts, strict=True))),
                        "targetProportionsJson": compact_json({pid: count / n for pid, count in zip(ids, counts, strict=True)}),
                        "n": int(n),
                        "targetPolicyCount": len(ids),
                        "policyIdsJson": compact_json(ids),
                        "policyCountsJson": compact_json(list(counts)),
                        "conditionHash": sha256_text(condition_key)[:16],
                        "s02Version": MIXTURE_SWEEP_VERSION,
                        **seeds,
                    }
                )

    condition_df = pd.DataFrame(rows)
    selection_rows = [
        {"priorityGroup": group, "selectedCount": len(ids), "panelPolicyIdsJson": compact_json(list(ids))}
        for group, ids in priority.items()
    ]
    selection_rows.append(
        {
            "priorityGroup": "priority_limits",
            "selectedCount": int(sum(PRIORITY_LIMITS.values())),
            "panelPolicyIdsJson": compact_json(PRIORITY_LIMITS),
        }
    )
    selection_rows.append(
        {
            "priorityGroup": "pair_records",
            "selectedCount": int(len(pairs)),
            "panelPolicyIdsJson": compact_json([f"{row['leftPanelPolicyId']}|{row['rightPanelPolicyId']}" for row in pairs]),
        }
    )
    selection_rows.append(
        {
            "priorityGroup": "full_ready_panel",
            "selectedCount": int(len(ready_panel)),
            "panelPolicyIdsJson": compact_json(list(ready_panel["panelPolicyId"].astype(str))),
        }
    )
    return condition_df, pd.DataFrame(selection_rows)


def initial_values(seed: int, n: int, value_profile: str = "random_unique") -> list[int]:
    rng = np.random.default_rng(int(seed))
    n = int(n)
    profile = str(value_profile or "random_unique")
    if profile == "random_unique":
        values = np.arange(1, n + 1, dtype=np.int16)
        rng.shuffle(values)
    elif profile == "reversed_unique":
        values = np.arange(n, 0, -1, dtype=np.int16)
    elif profile == "duplicate_1_10_x10":
        if n != 100:
            raise ValueError("duplicate_1_10_x10 requires n=100")
        values = np.repeat(np.arange(1, 11, dtype=np.int16), 10)
        rng.shuffle(values)
    else:
        raise ValueError(f"unsupported valueProfile: {profile}")
    return [int(value) for value in values]


def policy_assignment(condition: Mapping[str, Any]) -> tuple[list[str], list[int], str]:
    ids = [str(item) for item in parse_json_maybe(condition["policyIdsJson"], [])]
    counts = [int(item) for item in parse_json_maybe(condition["policyCountsJson"], [])]
    explicit_assignment = parse_json_maybe(condition.get("arrangementPolicyIdsJson", ""), [])
    if explicit_assignment:
        assigned = [str(item) for item in explicit_assignment]
        if len(assigned) != int(condition["n"]):
            raise ValueError("arrangementPolicyIdsJson length must match n")
        actual = Counter(assigned)
        expected = dict(zip(ids, counts, strict=True))
        if dict(actual) != expected:
            raise ValueError("arrangementPolicyIdsJson counts must match policyCountsJson")
        label_by_id = {pid: label for label, pid in enumerate(ids)}
        labels = [int(label_by_id[pid]) for pid in assigned]
        assignment_hash = sha256_text(compact_json(assigned))
        return assigned, labels, assignment_hash
    labels: list[int] = []
    for label, count in enumerate(counts):
        labels.extend([label] * int(count))
    rng = np.random.default_rng(int(condition["arrangementSeed"]))
    permuted = list(rng.permutation(np.asarray(labels, dtype=np.int16)).astype(int))
    assigned = [ids[label] for label in permuted]
    assignment_hash = sha256_text(compact_json(assigned))
    return assigned, permuted, assignment_hash


def _worker_init(panel_records: Sequence[Mapping[str, Any]]) -> None:
    global _PANEL_LOOKUP
    _PANEL_LOOKUP = {str(row["panelPolicyId"]): dict(row) for row in panel_records}


def simulator_backend(policy_records: Sequence[Mapping[str, Any]]) -> str:
    kinds = {str(row["simulatorKind"]) for row in policy_records}
    if "signal_event" in kinds:
        return "signal_event"
    if "memory_event" in kinds:
        return "memory_event"
    return "policy_event"


def run_mixture_condition(
    condition: Mapping[str, Any],
    panel_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    max_activations: int = 15000,
    max_swaps: int = 8000,
    max_comparisons: int = 60000,
) -> dict[str, Any]:
    panel_lookup = panel_lookup or _PANEL_LOOKUP
    ids = [str(item) for item in parse_json_maybe(condition["policyIdsJson"], [])]
    records = [dict(panel_lookup[pid]) for pid in ids]
    assigned_ids, numeric_labels, arrangement_hash = policy_assignment(condition)
    policy_cache = {pid: instantiate_panel_policy(panel_lookup[pid]) for pid in ids}
    policies = [policy_cache[pid] for pid in assigned_ids]
    reverse_payload = parse_json_maybe(condition.get("goalReverseDirectionsJson", ""), [])
    reverse_directions = [bool(value) for value in reverse_payload] if reverse_payload else [False] * int(condition["n"])
    if len(reverse_directions) != int(condition["n"]):
        raise ValueError("goalReverseDirectionsJson length must match n")
    values = initial_values(
        int(condition["valueSeed"]),
        int(condition["n"]),
        str(condition.get("valueProfile", "random_unique")),
    )
    backend = simulator_backend(records)
    research_step_id = str(condition.get("researchStepId", STEP_ID))
    implementation_prefix = str(condition.get("implementationPrefix", "e06_s02"))
    common = {
        "labels": numeric_labels,
        "reverse_directions": reverse_directions,
        "scheduler_seed": int(condition["schedulerSeed"]),
        "tie_breaker_seed": int(condition["tieBreakerSeed"]),
        "condition_id": str(condition["conditionId"]),
        "research_step_id": research_step_id,
    }
    if backend == "signal_event":
        first_signal = next((policy for policy in policy_cache.values() if hasattr(policy, "signal_config")), None)
        signal_config = getattr(first_signal, "signal_config", "no_signal")
        simulator = SignalEventSimulator(
            values,
            policies,
            signal_config=signal_config,
            auto_wrap_policies=False,
            trace_signal_activations=False,
            trace_memory_activations=False,
            implementation=f"{implementation_prefix}_signal_mixture",
            **common,
        )
        signal_config_json = compact_json(signal_config.to_dict()) if hasattr(signal_config, "to_dict") else compact_json(signal_config)
    elif backend == "memory_event":
        simulator = MemoryEventSimulator(
            values,
            policies,
            trace_memory_activations=False,
            implementation=f"{implementation_prefix}_memory_mixture",
            **common,
        )
        signal_config_json = ""
    else:
        simulator = PolicyEventSimulator(values, policies, implementation=f"{implementation_prefix}_policy_mixture", **common)
        signal_config_json = ""

    started = time.perf_counter()
    try:
        result = simulator.run(
            max_activations=max_activations,
            max_swaps=max_swaps,
            max_comparisons=max_comparisons,
            no_move_check_interval=int(condition["n"]),
        )
        final_labels = [int(cell.label) for cell in simulator.cells]
        final_ids = [ids[label] for label in final_labels]
        initial_ids = list(assigned_ids)
        final_values = simulator.current_values()
        initial_counts = Counter(initial_ids)
        final_counts = Counter(final_ids)
        norm_positions = np.linspace(0.0, 1.0, num=len(final_ids)) if len(final_ids) > 1 else np.asarray([0.0])
        mean_positions = {
            pid: float(np.mean([norm_positions[i] for i, value in enumerate(final_ids) if value == pid]))
            for pid in ids
        }
        left_id = str(condition["leftPanelPolicyId"])
        right_id = str(condition["rightPanelPolicyId"])
        left_mean = mean_positions.get(left_id)
        right_mean = mean_positions.get(right_id)
        minority_count = min(initial_counts.values()) if initial_counts else 0
        minority_ids = sorted(pid for pid, count in initial_counts.items() if count == minority_count)
        final_aggregation = aggregation(final_ids)
        initial_aggregation = aggregation(initial_ids)
        completed = sortedness_percent(final_values) >= 100.0 - 1e-9
        if completed:
            final_class = "sorted"
        elif str(result.stop_reason).startswith("max_"):
            final_class = "resource_capped_partial"
        elif result.stop_reason == "no_cell_can_move_after_two_checks":
            final_class = "no_move_partial"
        else:
            final_class = "partial"
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": True,
            "runError": "",
            "simulationBackend": backend,
            "signalConfigJson": signal_config_json,
            "initialValuesJson": compact_json(values),
            "finalValuesJson": compact_json(final_values),
            "initialStateHash": state_hash(values),
            "finalStateHash": state_hash(final_values),
            "initialPolicyAssignmentHash": arrangement_hash,
            "initialPanelPolicyIdsJson": compact_json(initial_ids),
            "finalPanelPolicyIdsJson": compact_json(final_ids),
            "actualCountsJson": compact_json(dict(initial_counts)),
            "finalCountsJson": compact_json(dict(final_counts)),
            "actualProportionsJson": compact_json({pid: initial_counts[pid] / int(condition["n"]) for pid in ids}),
            "actualRatiosMatchTarget": compact_json(dict(initial_counts)) == str(condition["targetCountsJson"]),
            "valueCountsConserved": Counter(values) == Counter(final_values),
            "policyCountsConserved": initial_counts == final_counts,
            "completed": bool(completed),
            "stopReason": result.stop_reason,
            "finalStateClass": final_class,
            "initialSortednessPercent": sortedness_percent(values),
            "finalSortednessPercent": sortedness_percent(final_values),
            "sortednessGain": sortedness_percent(final_values) - sortedness_percent(values),
            "initialAggregation": float(initial_aggregation),
            "finalAggregation": float(final_aggregation),
            "aggregationDelta": float(final_aggregation - initial_aggregation),
            "aggregationClass": "increased" if final_aggregation > initial_aggregation + 0.05 else ("decreased" if final_aggregation < initial_aggregation - 0.05 else "stable"),
            "swapCount": int(result.swap_count),
            "comparisonCount": int(result.comparison_count),
            "activationCount": int(result.activation_count),
            "eventCount": int(result.event_count),
            "blockedMoveAttempts": int(result.blocked_move_attempts),
            "frozenSwapAttempts": int(result.frozen_swap_attempts),
            "minorityPanelPolicyIdsJson": compact_json(minority_ids),
            "minorityInitialCount": int(minority_count),
            "minorityPersistence": 1.0 if minority_count > 0 and all(final_counts[pid] == initial_counts[pid] for pid in minority_ids) else 0.0,
            "leftPolicyMeanFinalPosition": left_mean,
            "rightPolicyMeanFinalPosition": right_mean,
            "leftMinusRightPositionBias": None if left_mean is None or right_mean is None else float(right_mean - left_mean),
            "meanFinalPositionByPolicyJson": compact_json(mean_positions),
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }
    except Exception as exc:  # pragma: no cover - retained for artifact-level failure accounting
        return {
            **{key: json_ready(value) for key, value in condition.items()},
            "runSucceeded": False,
            "runError": f"{type(exc).__name__}: {exc}",
            "simulationBackend": backend,
            "runtimeSeconds": float(time.perf_counter() - started),
            "claimBoundary": CLAIM_BOUNDARY,
        }


def _run_worker(condition: Mapping[str, Any], max_activations: int, max_swaps: int, max_comparisons: int) -> dict[str, Any]:
    return run_mixture_condition(
        condition,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )


def run_conditions(
    condition_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    workers: int = 1,
    max_activations: int = 15000,
    max_swaps: int = 8000,
    max_comparisons: int = 60000,
) -> pd.DataFrame:
    records = [dict(row) for row in condition_df.to_dict(orient="records")]
    panel_records = [dict(row) for row in panel_df.to_dict(orient="records")]
    if workers <= 1:
        lookup = {str(row["panelPolicyId"]): row for row in panel_records}
        return pd.DataFrame(
            [
                run_mixture_condition(
                    record,
                    lookup,
                    max_activations=max_activations,
                    max_swaps=max_swaps,
                    max_comparisons=max_comparisons,
                )
                for record in records
            ]
        )
    with ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init, initargs=(panel_records,)) as pool:
        futures = [
            pool.submit(_run_worker, record, int(max_activations), int(max_swaps), int(max_comparisons))
            for record in records
        ]
        return pd.DataFrame([future.result() for future in futures])


def summarize_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if result_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = ["mixtureKind", "pairCategory", "leftPanelPolicyId", "rightPanelPolicyId", "thirdPanelPolicyId", "ratioLabel"]
    grouped = result_df.groupby(group_cols, dropna=False)
    pair_summary = grouped.agg(
        runCount=("conditionId", "size"),
        successCount=("runSucceeded", "sum"),
        completionRate=("completed", "mean"),
        meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
        minFinalSortednessPercent=("finalSortednessPercent", "min"),
        meanSortednessGain=("sortednessGain", "mean"),
        meanFinalAggregation=("finalAggregation", "mean"),
        meanAggregationDelta=("aggregationDelta", "mean"),
        meanSwapCount=("swapCount", "mean"),
        meanActivationCount=("activationCount", "mean"),
        meanRuntimeSeconds=("runtimeSeconds", "mean"),
        allRatiosMatchTarget=("actualRatiosMatchTarget", "all"),
        allValueCountsConserved=("valueCountsConserved", "all"),
        allPolicyCountsConserved=("policyCountsConserved", "all"),
    ).reset_index()
    category_summary = (
        result_df.groupby(["mixtureKind", "pairCategory", "ratioLabel"], dropna=False)
        .agg(
            conditionCount=("conditionId", "size"),
            completionRate=("completed", "mean"),
            meanFinalSortednessPercent=("finalSortednessPercent", "mean"),
            meanFinalAggregation=("finalAggregation", "mean"),
            meanAggregationDelta=("aggregationDelta", "mean"),
            meanActivationCount=("activationCount", "mean"),
        )
        .reset_index()
    )
    return pair_summary, category_summary


def validation_checks(
    condition_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ready_panel: pd.DataFrame,
    smoke_df: pd.DataFrame,
) -> pd.DataFrame:
    ready_ids = set(ready_panel["panelPolicyId"].astype(str))
    condition_ids = {
        pid
        for text in condition_df["policyIdsJson"].astype(str)
        for pid in parse_json_maybe(text, [])
    }
    seed_cols = ["seedIndex", "valueSeed", "arrangementSeed", "schedulerSeed", "tieBreakerSeed"]
    checks = [
        {
            "checkId": "only_ready_policies",
            "success": bool(condition_ids <= ready_ids),
            "detail": f"{len(condition_ids)} selected policies all came from readyForS02Mixing=true rows",
        },
        {
            "checkId": "requested_pair_ratios_present",
            "success": bool(set(f"{a}:{b}" for a, b in PAIR_RATIOS) <= set(condition_df["ratioLabel"].astype(str))),
            "detail": "all requested pair ratios are represented",
        },
        {
            "checkId": "actual_ratios_match_target",
            "success": bool(result_df["actualRatiosMatchTarget"].fillna(False).map(bool).all()),
            "detail": "all simulated policy counts match target count JSON",
        },
        {
            "checkId": "seed_columns_complete",
            "success": bool(condition_df[seed_cols].notna().all().all() and not condition_df.duplicated(seed_cols + ["conditionId"]).any()),
            "detail": "condition matrix records value, arrangement, scheduler, and tie-breaker seeds",
        },
        {
            "checkId": "one_result_per_condition",
            "success": bool(len(result_df) == len(condition_df) and result_df["conditionId"].is_unique),
            "detail": f"{len(result_df)} result rows for {len(condition_df)} conditions",
        },
        {
            "checkId": "value_and_policy_counts_conserved",
            "success": bool(result_df["valueCountsConserved"].fillna(False).map(bool).all() and result_df["policyCountsConserved"].fillna(False).map(bool).all()),
            "detail": "all completed runs conserved values and policy labels",
        },
        {
            "checkId": "smoke_replay_deterministic",
            "success": bool(not smoke_df.empty and smoke_df["replayMatch"].map(bool).all()),
            "detail": f"{len(smoke_df)} smoke replay checks matched",
        },
    ]
    return pd.DataFrame(checks)


def run_smoke_replays(
    condition_df: pd.DataFrame,
    ready_panel: pd.DataFrame,
    *,
    sample_size: int = 6,
    max_activations: int = 2000,
) -> pd.DataFrame:
    sample = condition_df.head(int(sample_size))
    lookup = {str(row["panelPolicyId"]): dict(row) for row in ready_panel.to_dict(orient="records")}
    rows: list[dict[str, Any]] = []
    for condition in sample.to_dict(orient="records"):
        first = run_mixture_condition(condition, lookup, max_activations=max_activations, max_swaps=1000, max_comparisons=8000)
        second = run_mixture_condition(condition, lookup, max_activations=max_activations, max_swaps=1000, max_comparisons=8000)
        keys = ["runSucceeded", "finalStateHash", "finalPanelPolicyIdsJson", "swapCount", "activationCount", "stopReason"]
        match = all(first.get(key) == second.get(key) for key in keys)
        rows.append(
            {
                "conditionId": condition["conditionId"],
                "replayMatch": bool(match),
                "firstFinalStateHash": first.get("finalStateHash"),
                "secondFinalStateHash": second.get("finalStateHash"),
                "firstStopReason": first.get("stopReason"),
                "secondStopReason": second.get("stopReason"),
                "checkedKeysJson": compact_json(keys),
            }
        )
    return pd.DataFrame(rows)


def write_ratio_plot(category_summary: pd.DataFrame, figure_dir: Path, step_dir: Path) -> list[Path]:
    if category_summary.empty:
        return []
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [f"{a}:{b}" for a, b in PAIR_RATIOS] + [name for name, _ in THREE_WAY_PROFILES]
    plot_df = category_summary.copy()
    plot_df["ratioLabel"] = pd.Categorical(plot_df["ratioLabel"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["pairCategory", "ratioLabel"], kind="mergesort")
    fig, ax = plt.subplots(figsize=(11, 6))
    for category, group in plot_df.groupby("pairCategory", observed=False):
        if group.empty:
            continue
        ax.plot(group["ratioLabel"].astype(str), group["meanFinalSortednessPercent"], marker="o", linewidth=1.5, label=str(category))
    ax.set_xlabel("Mixture ratio/profile")
    ax.set_ylabel("Mean final Sortedness (%)")
    ax.set_title("E06 S02 mixture-ratio sweep by priority category")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.tight_layout()
    paths = [
        figure_dir / "e06_s02_ratio_sortedness_by_category.png",
        figure_dir / "e06_s02_ratio_sortedness_by_category.pdf",
        step_dir / "ratio_sortedness_by_category.png",
        step_dir / "ratio_sortedness_by_category.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
    plt.close(fig)
    return paths


def compact_result_csv(result_df: pd.DataFrame) -> pd.DataFrame:
    """Drop verbose 100-cell JSON arrays from CSV; Parquet keeps full detail."""

    verbose_columns = [
        "initialValuesJson",
        "finalValuesJson",
        "initialPanelPolicyIdsJson",
        "finalPanelPolicyIdsJson",
    ]
    return result_df.drop(columns=[col for col in verbose_columns if col in result_df.columns])


def write_markdown_reports(
    *,
    step_dir: Path,
    status: Mapping[str, Any],
    category_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
    artifacts_written: Sequence[str],
) -> Path:
    top = category_summary.sort_values(["completionRate", "meanFinalSortednessPercent"], ascending=[False, False]).head(5)
    top_lines = "\n".join(
        f"- {row.pairCategory} at `{row.ratioLabel}`: completion {row.completionRate:.2f}, mean Sortedness {row.meanFinalSortednessPercent:.1f}%"
        for row in top.itertuples(index=False)
    ) or "- No category summaries available."
    validation_result = str(status["validationResult"])
    text = f"""# Research Step S02: Vary mixture ratios

## Completion status

{status['status']} on {status['completedAt']}. Outcome classification: {status['outcomeClassification']}.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{validation_result}. Ran {status['conditionCount']} prioritized mixture conditions covering requested pair ratios, selected three-way mixtures, and {status['selectedPolicyCount']} ready S01 policies. Validation checks passed: {int(validation_df['success'].sum())}/{len(validation_df)}.

## Caveats or blockers

The full 71-policy pairwise matrix was not run because it would require 2,485 unordered pairs before ratios and seeds. S02 prioritized originals, ready E03 frontier/discovered policies, and representative E04 memory/signaling/handoff policies. Dominance and minority-persistence fields are computational proxies; true goal-conflict dominance is reserved for later opposite-goal steps.

## Lay summary

S02 mixed selected ready policies at exact 100-cell ratios and measured whether mixed collectives still sorted, clustered by policy, or stalled. The sweep is a bounded phase-map seed for later arrangement and goal-compatibility work, not an exhaustive atlas of all 71 ready policies.

## Anchor results

{top_lines}

## Recommended next action

Chief Scientist review, then S03 should vary initial spatial arrangement using the S02 priority pairs and any high-contrast ratio/category combinations from `$ARTIFACTS_DIR/results/e06_mixture_ratios.parquet`.
"""
    path = step_dir / "summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append({"path": str(path), "sha256": sha256_file(path), "sizeBytes": int(path.stat().st_size)})
    return records


def run_s02_mixture_ratios(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    panel_path: Path = DEFAULT_PANEL_PATH,
    n: int = 100,
    seed_count: int = 2,
    max_pairs: int | None = None,
    workers: int | None = None,
    max_activations: int = 6000,
    max_swaps: int = 4000,
    max_comparisons: int = 30000,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    workers = int(workers if workers is not None else min(8, os.cpu_count() or 1))
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    figures_dir = artifacts_dir / "figures" / "e06"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, results_dir, figures_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ready_panel = load_ready_panel(panel_path)
    condition_df, selection_df = build_condition_matrix(ready_panel, n=n, seed_count=seed_count, max_pairs=max_pairs)
    smoke_df = run_smoke_replays(condition_df, ready_panel)
    result_df = run_conditions(
        condition_df,
        ready_panel,
        workers=workers,
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
    )
    pair_summary, category_summary = summarize_results(result_df)
    validation_df = validation_checks(condition_df, result_df, ready_panel, smoke_df)

    condition_csv = step_dir / "mixture_ratio_condition_matrix.csv"
    condition_parquet = step_dir / "mixture_ratio_condition_matrix.parquet"
    result_csv = results_dir / "e06_mixture_ratios.csv"
    result_parquet = results_dir / "e06_mixture_ratios.parquet"
    sweep_csv = results_dir / "e06_mixture_sweeps.csv"
    sweep_parquet = results_dir / "e06_mixture_sweeps.parquet"
    step_result_csv = step_dir / "mixture_ratio_runs.csv"
    step_result_parquet = step_dir / "mixture_ratio_runs.parquet"
    pair_summary_csv = step_dir / "mixture_ratio_pair_summary.csv"
    pair_summary_parquet = step_dir / "mixture_ratio_pair_summary.parquet"
    category_summary_csv = step_dir / "mixture_ratio_category_summary.csv"
    category_summary_parquet = step_dir / "mixture_ratio_category_summary.parquet"
    validation_csv = step_dir / "validation_checks.csv"
    validation_parquet = step_dir / "validation_checks.parquet"
    smoke_csv = step_dir / "smoke_replay_checks.csv"
    smoke_parquet = step_dir / "smoke_replay_checks.parquet"
    selection_csv = step_dir / "priority_policy_selection.csv"
    selection_parquet = step_dir / "priority_policy_selection.parquet"

    result_csv_df = compact_result_csv(result_df)
    for path, df in [
        (condition_csv, condition_df),
        (step_result_csv, result_csv_df),
        (result_csv, result_csv_df),
        (sweep_csv, result_csv_df),
        (pair_summary_csv, pair_summary),
        (category_summary_csv, category_summary),
        (validation_csv, validation_df),
        (smoke_csv, smoke_df),
        (selection_csv, selection_df),
    ]:
        df.to_csv(path, index=False)
    for path, df in [
        (condition_parquet, condition_df),
        (step_result_parquet, result_df),
        (result_parquet, result_df),
        (sweep_parquet, result_df),
        (pair_summary_parquet, pair_summary),
        (category_summary_parquet, category_summary),
        (validation_parquet, validation_df),
        (smoke_parquet, smoke_df),
        (selection_parquet, selection_df),
    ]:
        df.to_parquet(path, index=False)

    figure_paths = write_ratio_plot(category_summary, figures_dir, step_dir)
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_result = "passed" if validation_df["success"].map(bool).all() and result_df["runSucceeded"].fillna(False).map(bool).all() else "failed"
    selected_policy_ids = {
        pid for text in condition_df["policyIdsJson"].astype(str) for pid in parse_json_maybe(text, [])
    }
    artifacts = [
        condition_csv,
        condition_parquet,
        result_csv,
        result_parquet,
        sweep_csv,
        sweep_parquet,
        step_result_csv,
        step_result_parquet,
        pair_summary_csv,
        pair_summary_parquet,
        category_summary_csv,
        category_summary_parquet,
        validation_csv,
        validation_parquet,
        smoke_csv,
        smoke_parquet,
        selection_csv,
        selection_parquet,
        *figure_paths,
        step_dir / "summary.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "Full 71-policy pairwise matrix was not run; S02 used a prioritized condition matrix.",
            "All rows use same-goal increasing sorting; opposite-goal dominance is deferred to later E06 steps.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then run S03 initial spatial arrangement sweeps using S02 priority pairs "
            "and high-contrast ratio/category results."
        ),
        "outcomeClassification": "supportive" if validation_result == "passed" else "constraining/contradictory",
        "conditionCount": int(len(condition_df)),
        "resultCount": int(len(result_df)),
        "selectedPolicyCount": int(len(selected_policy_ids)),
        "readyPanelInputCount": int(len(ready_panel)),
        "pairConditionCount": int((condition_df["mixtureKind"] == "pair").sum()),
        "threeWayConditionCount": int((condition_df["mixtureKind"] == "three_way").sum()),
        "maxPairsLimit": max_pairs,
        "completionRate": float(result_df["completed"].mean()),
        "meanFinalSortednessPercent": float(result_df["finalSortednessPercent"].mean()),
        "workerCount": int(workers),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
    }
    summary_path = write_markdown_reports(
        step_dir=step_dir,
        status=status,
        category_summary=category_summary,
        validation_df=validation_df,
        artifacts_written=[str(path) for path in artifacts],
    )
    if summary_path not in artifacts:
        artifacts.append(summary_path)
    write_json(status_path, status)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "mixtureSweepVersion": MIXTURE_SWEEP_VERSION,
        "artifacts": artifact_records(artifacts),
    }
    write_json(manifest_path, manifest_payload)

    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.exists():
        try:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    research_steps = dict(existing_manifest.get("researchSteps", {}))
    research_steps[STEP_ID] = status
    run_manifest = {
        **existing_manifest,
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": "E06",
        "lastResearchStepId": STEP_ID,
        "lastStepNumber": STEP_NUMBER,
        "updatedAt": completed_at,
        "git": {
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "dirtyStatus": git_value(["status", "--short"], repo_root),
            "remote": git_value(["remote", "-v"], repo_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": int(workers),
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": research_steps,
        "artifacts": artifact_records(artifacts),
    }
    write_json(run_manifest_path, run_manifest)
    manifest_payload["artifacts"] = artifact_records(artifacts)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "conditions": condition_df,
        "results": result_df,
        "pairSummary": pair_summary,
        "categorySummary": category_summary,
        "validation": validation_df,
        "smoke": smoke_df,
        "selection": selection_df,
        "artifactPaths": [str(path) for path in artifacts],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E06 S02 mixture-ratio sweeps.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-activations", type=int, default=6000)
    parser.add_argument("--max-swaps", type=int, default=4000)
    parser.add_argument("--max-comparisons", type=int, default=30000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_s02_mixture_ratios(
        artifacts_dir=args.artifacts_dir,
        panel_path=args.panel_path,
        n=args.n,
        seed_count=args.seed_count,
        max_pairs=args.max_pairs,
        workers=args.workers,
        max_activations=args.max_activations,
        max_swaps=args.max_swaps,
        max_comparisons=args.max_comparisons,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['conditionCount']} conditions, "
        f"{status['selectedPolicyCount']} policies, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
