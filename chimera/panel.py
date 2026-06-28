"""E06 S01 Algotype panel assembly and pure-policy validation.

The panel keeps two identifiers for each policy. ``panelPolicyId`` is the
stable E06-facing ID used by later chimera sweeps, while ``sourcePolicyId``
preserves the upstream E03/E04 identifier and provenance.
"""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e02_deterministic_simulator.metrics import sortedness_percent, state_hash
from morphospace import (
    BubblePolicy,
    DSLPolicy,
    InsertionPolicy,
    LocalRulePolicy,
    NullPolicy,
    PolicyEventSimulator,
    RandomWalkPolicy,
    SelectionPolicy,
    parse_rule_program,
    policy_from_json,
    policy_to_json,
)
from memory_repair import (
    LocalLearningPolicyWrapper,
    MemoryEventSimulator,
    MemoryPolicyWrapper,
    SignalEventSimulator,
    SignalPolicyWrapper,
    memory_policy_from_json,
    memory_policy_to_json,
    signal_policy_from_json,
    signal_policy_to_json,
)


STEP_ID = "S01"
STEP_NUMBER = 1
PANEL_SCHEMA_VERSION = "e06_s01_algotype_panel.v1"
VALIDATION_SCHEMA_VERSION = "e06_s01_pure_policy_validation.v1"
PREVIOUS_ARTIFACT_ROOT = Path("/previous-artifacts")
CLAIM_BOUNDARY = (
    "Computational local-policy validation only; not biological validation, "
    "morphogenesis proof, or evidence about living chimeras."
)

E03_FRONTIER_PATH = PREVIOUS_ARTIFACT_ROOT / "E03/research_steps/S14/frontier_candidates.parquet"
E03_ATLAS_PATH = PREVIOUS_ARTIFACT_ROOT / "E03/report_bundle_inputs/e03_policy_atlas_catalog.parquet"
E04_MEMORY_PATH = PREVIOUS_ARTIFACT_ROOT / "E04/research_steps/S01/memory_variant_catalog.parquet"
E04_SIGNAL_PATH = PREVIOUS_ARTIFACT_ROOT / "E04/research_steps/S02/signal_variant_catalog.parquet"
E04_HANDOFF_PATH = PREVIOUS_ARTIFACT_ROOT / "E04/research_steps/S15/selected_policy_handoff_candidates.parquet"


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set, np.ndarray)) else False:
        return None
    return value


def parse_json_maybe(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        return json.loads(text)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def stable_panel_id(prefix: str, *parts: Any) -> str:
    raw = "__".join(str(part) for part in parts if part is not None and str(part))
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw).strip("_")
    safe = "_".join(piece for piece in safe.split("_") if piece)
    if len(safe) > 72:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        safe = f"{safe[:56]}_{digest}"
    return f"{prefix}_{safe}" if safe else prefix


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


def policy_spec_json(policy: LocalRulePolicy, kind: str) -> str:
    if kind == "memory_json":
        return memory_policy_to_json(policy)
    if kind == "signal_json":
        return signal_policy_to_json(policy)
    return policy_to_json(policy)


def base_record(
    *,
    panel_policy_id: str,
    source_policy_id: str,
    display_name: str,
    policy: LocalRulePolicy | None,
    panel_group: str,
    policy_family: str,
    source_experiment: str,
    source_step: str,
    source_artifact_path: str,
    constructor_kind: str,
    simulator_kind: str,
    source_rank: int | None = None,
    class_label: str | None = None,
    competence_summary: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    constructor_payload_json: str | None = None,
    direct_policy_spec_json: str | None = None,
) -> dict[str, Any]:
    if direct_policy_spec_json is None and policy is not None:
        direct_policy_spec_json = policy_spec_json(policy, constructor_kind)
    if constructor_payload_json is None:
        constructor_payload_json = direct_policy_spec_json
    spec_payload = parse_json_maybe(direct_policy_spec_json, {})
    algotype = str(spec_payload.get("algotype", getattr(policy, "algotype", "unknown")))
    interface_family = str(spec_payload.get("family", getattr(policy, "family", policy_family)))
    return {
        "panelSchemaVersion": PANEL_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "panelPolicyId": panel_policy_id,
        "sourcePolicyId": source_policy_id,
        "displayName": display_name,
        "panelGroup": panel_group,
        "policyFamily": policy_family,
        "interfaceFamily": interface_family,
        "algotype": algotype,
        "sourceExperiment": source_experiment,
        "sourceStep": source_step,
        "sourceArtifactPath": source_artifact_path,
        "sourceRank": source_rank,
        "classLabel": class_label,
        "constructorKind": constructor_kind,
        "simulatorKind": simulator_kind,
        "policySpecJson": direct_policy_spec_json,
        "constructorPayloadJson": constructor_payload_json,
        "policySpecSha256": sha256_text(constructor_payload_json or ""),
        "competenceSummaryJson": compact_json(competence_summary or {}),
        "sourceMetadataJson": compact_json(source_metadata or {}),
        "claimBoundary": CLAIM_BOUNDARY,
    }


def direct_policy_records() -> list[dict[str, Any]]:
    policies: list[tuple[str, str, str, LocalRulePolicy, str, str]] = [
        ("e06_original_classic_bubble", "classic_bubble", "Bubble", BubblePolicy(), "original_classic", "classic"),
        (
            "e06_original_classic_insertion",
            "classic_insertion",
            "Insertion",
            InsertionPolicy(),
            "original_classic",
            "classic",
        ),
        (
            "e06_original_classic_selection",
            "classic_selection",
            "Selection",
            SelectionPolicy(),
            "original_classic",
            "classic",
        ),
        ("e06_null_wait", "null_wait", "Null wait", NullPolicy(), "null_control", "null"),
        (
            "e06_random_walk_p025",
            "random_walk_adjacent_p025",
            "Random adjacent swap p=0.25",
            RandomWalkPolicy(0.25),
            "randomized_control",
            "randomized",
        ),
        (
            "e06_random_walk_p050",
            "random_walk_adjacent_p050",
            "Random adjacent swap p=0.50",
            RandomWalkPolicy(0.50),
            "randomized_control",
            "randomized",
        ),
        (
            "e06_random_walk_p100",
            "random_walk_adjacent_p100",
            "Random adjacent swap p=1.00",
            RandomWalkPolicy(1.00),
            "randomized_control",
            "randomized",
        ),
    ]
    return [
        base_record(
            panel_policy_id=panel_id,
            source_policy_id=source_id,
            display_name=label,
            policy=policy,
            panel_group=group,
            policy_family=family,
            source_experiment="E01_E02_E03",
            source_step="E03_S01",
            source_artifact_path="repo:morphospace/policies.py",
            constructor_kind="morphospace_json",
            simulator_kind="policy_event",
            competence_summary={"source": "direct baseline/control policy"},
        )
        for panel_id, source_id, label, policy, group, family in policies
    ]


def dsl_record_from_row(
    row: Mapping[str, Any],
    *,
    panel_group: str,
    policy_family: str,
    source_step: str,
    source_artifact_path: str,
    rank_column: str,
    panel_prefix: str,
) -> dict[str, Any]:
    source_policy_id = str(row["policyId"])
    rank = None if pd.isna(row.get(rank_column)) else int(row.get(rank_column))
    program = parse_rule_program(str(row["dslProgramJson"]))
    policy = DSLPolicy(program)
    summary_cols = [
        "frontierRank",
        "frontierCompositeScore",
        "holdoutCompositeScore",
        "s13CompositeScore",
        "completionScore",
        "sortednessScore",
        "energyScore",
        "robustnessScore",
        "delayedGratificationScore",
        "aggregationScore",
        "oscillationStabilityScore",
        "holdout_completionSuccessMean",
        "holdout_finalSortednessScoreMean",
    ]
    metadata_cols = [
        "family",
        "generationMethod",
        "lineageId",
        "description",
        "frontierObjectiveTagsJson",
        "atlasPrimaryRole",
        "primaryRole",
        "className",
        "classLabel",
    ]
    summary = {col: row.get(col) for col in summary_cols if col in row and not pd.isna(row.get(col))}
    metadata = {col: row.get(col) for col in metadata_cols if col in row and not pd.isna(row.get(col))}
    class_label = row.get("classLabel", row.get("className"))
    return base_record(
        panel_policy_id=stable_panel_id(panel_prefix, f"rank_{rank:02d}" if rank is not None else None, source_policy_id),
        source_policy_id=source_policy_id,
        display_name=str(row.get("description") or row.get("classLabel") or source_policy_id),
        policy=policy,
        panel_group=panel_group,
        policy_family=policy_family,
        source_experiment="E03",
        source_step=source_step,
        source_artifact_path=source_artifact_path,
        constructor_kind="morphospace_json",
        simulator_kind="policy_event",
        source_rank=rank,
        class_label=None if pd.isna(class_label) else str(class_label),
        competence_summary=summary,
        source_metadata=metadata,
    )


def load_e03_frontier_records(limit: int | None = None) -> list[dict[str, Any]]:
    if not E03_FRONTIER_PATH.exists():
        return []
    df = pd.read_parquet(E03_FRONTIER_PATH)
    df = df.sort_values(["frontierRank", "policyId"], kind="mergesort")
    if limit is not None:
        df = df.head(int(limit))
    return [
        dsl_record_from_row(
            row,
            panel_group="e03_frontier",
            policy_family="dsl_frontier",
            source_step="E03_S14",
            source_artifact_path=str(E03_FRONTIER_PATH),
            rank_column="frontierRank",
            panel_prefix="e06_e03_frontier",
        )
        for row in df.to_dict(orient="records")
    ]


def load_e03_selected_discovered_records(limit: int = 6) -> list[dict[str, Any]]:
    if not E03_ATLAS_PATH.exists():
        return []
    df = pd.read_parquet(E03_ATLAS_PATH)
    for column in ["isNullPolicy", "isClassicPolicy", "isFrontierCandidate", "isPathologicalPolicy"]:
        if column in df.columns:
            df[column] = df[column].map(lambda value: bool(value) if pd.notna(value) else False)
        else:
            df[column] = False
    df = df[
        (~df["isNullPolicy"])
        & (~df["isClassicPolicy"])
        & (~df["isFrontierCandidate"])
        & (~df["isPathologicalPolicy"])
        & df["dslProgramJson"].notna()
    ].copy()
    if df.empty:
        return []
    if "s13CompositeScore" not in df.columns:
        df["s13CompositeScore"] = np.nan
    df = df.sort_values(["classLabel", "s13CompositeScore", "policyId"], ascending=[True, False, True], kind="mergesort")
    selected = df.groupby("classLabel", dropna=False).head(1)
    selected = selected.sort_values(["s13CompositeScore", "policyId"], ascending=[False, True], kind="mergesort").head(limit)
    selected = selected.reset_index(drop=True)
    selected["e06SelectedRank"] = selected.index + 1
    return [
        dsl_record_from_row(
            row,
            panel_group="e03_selected_discovered",
            policy_family="dsl_discovered",
            source_step="E03_S15",
            source_artifact_path=str(E03_ATLAS_PATH),
            rank_column="e06SelectedRank",
            panel_prefix="e06_e03_discovered",
        )
        for row in selected.to_dict(orient="records")
    ]


def load_e04_memory_records() -> list[dict[str, Any]]:
    if not E04_MEMORY_PATH.exists():
        return []
    df = pd.read_parquet(E04_MEMORY_PATH)
    df = df[df["memoryVariant"].astype(str) != "no_memory"].copy()
    df = df.sort_values(["basePolicyId", "memoryVariant"], kind="mergesort")
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        payload = str(row["policySpecJson"])
        policy = memory_policy_from_json(payload)
        variant = str(row["memoryVariant"])
        base_id = str(row["basePolicyId"])
        records.append(
            base_record(
                panel_policy_id=stable_panel_id("e06_e04_memory", base_id, variant),
                source_policy_id=str(row["policyId"]),
                display_name=f"E04 {variant} memory wrapper on {base_id}",
                policy=policy,
                panel_group="e04_memory",
                policy_family="memory_wrapper",
                source_experiment="E04",
                source_step="E04_S01",
                source_artifact_path=str(E04_MEMORY_PATH),
                constructor_kind="memory_json",
                simulator_kind="memory_event",
                class_label=base_id,
                competence_summary={
                    "memoryVariant": variant,
                    "counterMax": row.get("counterMax"),
                    "neighborHistory": row.get("neighborHistory"),
                },
                source_metadata={
                    "basePolicyId": base_id,
                    "baseFamily": row.get("baseFamily"),
                    "baseSourcePath": row.get("baseSourcePath"),
                },
                constructor_payload_json=payload,
                direct_policy_spec_json=payload,
            )
        )
    return records


def load_e04_signal_records() -> list[dict[str, Any]]:
    if not E04_SIGNAL_PATH.exists():
        return []
    df = pd.read_parquet(E04_SIGNAL_PATH)
    df = df[df["signalVariant"].astype(str) != "no_signal"].copy()
    df = df.sort_values(["basePolicyId", "signalVariant"], kind="mergesort")
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        payload = str(row["policySpecJson"])
        policy = signal_policy_from_json(payload)
        variant = str(row["signalVariant"])
        base_id = str(row["basePolicyId"])
        records.append(
            base_record(
                panel_policy_id=stable_panel_id("e06_e04_signal", base_id, variant),
                source_policy_id=str(row["policyId"]),
                display_name=f"E04 {variant} signal wrapper on {base_id}",
                policy=policy,
                panel_group="e04_signal",
                policy_family="signal_wrapper",
                source_experiment="E04",
                source_step="E04_S02",
                source_artifact_path=str(E04_SIGNAL_PATH),
                constructor_kind="signal_json",
                simulator_kind="signal_event",
                class_label=base_id,
                competence_summary={
                    "signalVariant": variant,
                    "signalRange": row.get("signalRange"),
                    "diffusionRate": row.get("diffusionRate"),
                    "decay": row.get("decay"),
                    "noiseStd": row.get("noiseStd"),
                },
                source_metadata={
                    "basePolicyId": base_id,
                    "baseFamily": row.get("baseFamily"),
                    "baseSourcePath": row.get("baseSourcePath"),
                },
                constructor_payload_json=payload,
                direct_policy_spec_json=payload,
            )
        )
    return records


def handoff_policy_from_competence_spec(payload: Mapping[str, Any] | str) -> LocalRulePolicy:
    spec = parse_json_maybe(payload, {})
    learning_config = spec.get("learningConfig") or spec.get("learning_config") or "fixed"
    memory_config = spec.get("memoryConfig") or spec.get("memory_config") or "no_memory"
    signal_config = spec.get("signalConfig") or spec.get("signal_config") or "no_signal"
    learning = LocalLearningPolicyWrapper("bubble", learning_config)
    memory = MemoryPolicyWrapper(learning, memory_config)
    return SignalPolicyWrapper(memory, signal_config)


def load_e04_handoff_records(limit: int | None = None) -> list[dict[str, Any]]:
    if not E04_HANDOFF_PATH.exists():
        return []
    df = pd.read_parquet(E04_HANDOFF_PATH)
    df = df[df["replayable"].map(lambda value: bool(value) if pd.notna(value) else False)].copy()
    df = df.sort_values(["handoffCompositeProxy", "policyId"], ascending=[False, True], kind="mergesort")
    if limit is not None:
        df = df.head(int(limit))
    records: list[dict[str, Any]] = []
    for index, row in enumerate(df.to_dict(orient="records"), start=1):
        payload = str(row["policySpecJson"])
        policy = handoff_policy_from_competence_spec(payload)
        direct_spec = signal_policy_to_json(policy)
        source_policy_id = str(row["policyId"])
        records.append(
            base_record(
                panel_policy_id=stable_panel_id("e06_e04_handoff", f"rank_{index:02d}", source_policy_id),
                source_policy_id=source_policy_id,
                display_name=f"E04 handoff candidate {source_policy_id}",
                policy=policy,
                panel_group="e04_handoff",
                policy_family=str(row.get("familyKind", "enhanced")),
                source_experiment="E04",
                source_step="E04_S15",
                source_artifact_path=str(E04_HANDOFF_PATH),
                constructor_kind="e04_handoff_competence_spec",
                simulator_kind="signal_event",
                source_rank=index,
                class_label=str(row.get("policyGroup", "enhanced")),
                competence_summary={
                    "overallCompetenceProxy": row.get("overallCompetenceProxy"),
                    "holdoutMeanScore": row.get("holdoutMeanScore"),
                    "handoffCompositeProxy": row.get("handoffCompositeProxy"),
                    "handoffTier": row.get("handoffTier"),
                    "sourceCandidateId": row.get("sourceCandidateId"),
                },
                source_metadata={
                    "handoffRationale": row.get("handoffRationale"),
                    "recommendedForHandoff": row.get("recommendedForHandoff"),
                    "policyAuditSuccess": row.get("policyAuditSuccess"),
                    "replayBlocker": row.get("replayBlocker"),
                },
                constructor_payload_json=payload,
                direct_policy_spec_json=direct_spec,
            )
        )
    return records


def build_algotype_panel(
    *,
    frontier_limit: int | None = None,
    discovered_limit: int = 6,
    handoff_limit: int | None = None,
) -> pd.DataFrame:
    """Assemble the S01 policy panel from repo and upstream artifacts."""

    records: list[dict[str, Any]] = []
    records.extend(direct_policy_records())
    records.extend(load_e03_frontier_records(frontier_limit))
    records.extend(load_e03_selected_discovered_records(discovered_limit))
    records.extend(load_e04_memory_records())
    records.extend(load_e04_signal_records())
    records.extend(load_e04_handoff_records(handoff_limit))
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if df["panelPolicyId"].duplicated().any():
        duplicated = sorted(df.loc[df["panelPolicyId"].duplicated(), "panelPolicyId"].unique())
        raise ValueError(f"duplicate panelPolicyId values: {duplicated}")
    return df.sort_values(["panelGroup", "sourceRank", "panelPolicyId"], kind="mergesort").reset_index(drop=True)


def instantiate_panel_policy(record: Mapping[str, Any]) -> LocalRulePolicy:
    kind = str(record["constructorKind"])
    payload = str(record["constructorPayloadJson"])
    if kind == "morphospace_json":
        return policy_from_json(payload)
    if kind == "memory_json":
        return memory_policy_from_json(payload)
    if kind == "signal_json":
        return signal_policy_from_json(payload)
    if kind == "e04_handoff_competence_spec":
        return handoff_policy_from_competence_spec(payload)
    raise ValueError(f"unknown constructor kind: {kind}")


def roundtrip_policy(record: Mapping[str, Any]) -> tuple[bool, str]:
    try:
        first = instantiate_panel_policy(record)
        second = instantiate_panel_policy(record)
        same = first.to_spec().to_dict() == second.to_spec().to_dict()
        return bool(same), "" if same else "roundtrip_spec_mismatch"
    except Exception as exc:  # pragma: no cover - exercised by incompatible upstream data
        return False, f"{type(exc).__name__}: {exc}"


def seeded_values(seed: int, n: int = 8) -> list[int]:
    rng = np.random.default_rng(int(seed))
    values = np.arange(1, int(n) + 1, dtype=np.int16)
    rng.shuffle(values)
    return [int(value) for value in values]


def simulator_for_record(
    record: Mapping[str, Any],
    values: Sequence[int],
    *,
    goal_direction: str,
    scheduler_seed: int,
    tie_breaker_seed: int,
    condition_id: str,
):
    policy = instantiate_panel_policy(record)
    reverse = [goal_direction == "decreasing"] * len(values)
    simulator_kind = str(record["simulatorKind"])
    common_kwargs = {
        "reverse_directions": reverse,
        "scheduler_seed": int(scheduler_seed),
        "tie_breaker_seed": int(tie_breaker_seed),
        "condition_id": condition_id,
        "research_step_id": STEP_ID,
    }
    if simulator_kind == "policy_event":
        return PolicyEventSimulator(values, policy, implementation="e06_s01_policy_event", **common_kwargs)
    if simulator_kind == "memory_event":
        return MemoryEventSimulator(
            values,
            policy,
            implementation="e06_s01_memory_event",
            trace_memory_activations=False,
            **common_kwargs,
        )
    if simulator_kind == "signal_event":
        signal_config = getattr(policy, "signal_config", "no_signal")
        return SignalEventSimulator(
            values,
            policy,
            signal_config=signal_config,
            implementation="e06_s01_signal_event",
            auto_wrap_policies=False,
            trace_signal_activations=False,
            trace_memory_activations=False,
            **common_kwargs,
        )
    raise ValueError(f"unknown simulator kind: {simulator_kind}")


def run_once(
    record: Mapping[str, Any],
    *,
    input_seed: int,
    goal_direction: str,
    scheduler_seed: int,
    tie_breaker_seed: int,
    max_activations: int,
    max_swaps: int,
    max_comparisons: int,
) -> dict[str, Any]:
    values = seeded_values(input_seed)
    condition_id = f"{STEP_ID}_{record['panelPolicyId']}_{goal_direction}_seed{input_seed}"
    started = time.perf_counter()
    simulator = simulator_for_record(
        record,
        values,
        goal_direction=goal_direction,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
        condition_id=condition_id,
    )
    result = simulator.run(
        max_activations=max_activations,
        max_swaps=max_swaps,
        max_comparisons=max_comparisons,
        no_move_check_interval=len(values),
    )
    goal_sortedness = sortedness_percent(result.final_values, direction=goal_direction)
    initial_goal_sortedness = sortedness_percent(result.initial_values, direction=goal_direction)
    return {
        "conditionId": condition_id,
        "panelPolicyId": record["panelPolicyId"],
        "sourcePolicyId": record["sourcePolicyId"],
        "panelGroup": record["panelGroup"],
        "policyFamily": record["policyFamily"],
        "goalDirection": goal_direction,
        "inputSeed": int(input_seed),
        "schedulerSeed": int(scheduler_seed),
        "tieBreakerSeed": int(tie_breaker_seed),
        "n": len(values),
        "runSucceeded": True,
        "runError": "",
        "stopReason": result.stop_reason,
        "completedIncreasingMetric": bool(result.completed),
        "completedGoal": bool(goal_sortedness >= 100.0 - 1e-9),
        "initialGoalSortednessPercent": float(initial_goal_sortedness),
        "finalGoalSortednessPercent": float(goal_sortedness),
        "finalIncreasingSortednessPercent": float(result.final_sortedness_percent),
        "swapCount": int(result.swap_count),
        "comparisonCount": int(result.comparison_count),
        "activationCount": int(result.activation_count),
        "eventCount": int(result.event_count),
        "blockedMoveAttempts": int(result.blocked_move_attempts),
        "frozenSwapAttempts": int(result.frozen_swap_attempts),
        "initialStateHash": state_hash(result.initial_values),
        "finalStateHash": state_hash(result.final_values),
        "valueCountsConserved": Counter(result.initial_values) == Counter(result.final_values),
        "runtimeSeconds": float(time.perf_counter() - started),
        "validationSchemaVersion": VALIDATION_SCHEMA_VERSION,
    }


def run_validation_panel(
    panel_df: pd.DataFrame,
    *,
    input_seeds: Sequence[int] = (6101, 6102, 6103),
    goal_directions: Sequence[str] = ("increasing", "decreasing"),
    max_activations: int = 1800,
    max_swaps: int = 900,
    max_comparisons: int = 7200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every panel policy in pure conditions and summarize readiness."""

    validation_rows: list[dict[str, Any]] = []
    serialization: dict[str, tuple[bool, str]] = {}
    for record in panel_df.to_dict(orient="records"):
        panel_policy_id = str(record["panelPolicyId"])
        serialization[panel_policy_id] = roundtrip_policy(record)
        for goal_index, goal in enumerate(goal_directions):
            for replicate_index, input_seed in enumerate(input_seeds):
                scheduler_seed = 700_000 + goal_index * 10_000 + replicate_index * 101
                tie_seed = 710_000 + goal_index * 10_000 + replicate_index * 101
                try:
                    first = run_once(
                        record,
                        input_seed=input_seed,
                        goal_direction=str(goal),
                        scheduler_seed=scheduler_seed,
                        tie_breaker_seed=tie_seed,
                        max_activations=max_activations,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                    )
                    second = run_once(
                        record,
                        input_seed=input_seed,
                        goal_direction=str(goal),
                        scheduler_seed=scheduler_seed,
                        tie_breaker_seed=tie_seed,
                        max_activations=max_activations,
                        max_swaps=max_swaps,
                        max_comparisons=max_comparisons,
                    )
                    replay_match = all(
                        first[key] == second[key]
                        for key in [
                            "stopReason",
                            "completedGoal",
                            "finalStateHash",
                            "swapCount",
                            "comparisonCount",
                            "activationCount",
                            "eventCount",
                        ]
                    )
                    first["reproducibleSeedReplay"] = bool(replay_match)
                    first["reproducibilityError"] = "" if replay_match else "same_seed_replay_mismatch"
                    validation_rows.append(first)
                except Exception as exc:
                    validation_rows.append(
                        {
                            "conditionId": f"{STEP_ID}_{panel_policy_id}_{goal}_seed{input_seed}",
                            "panelPolicyId": panel_policy_id,
                            "sourcePolicyId": record["sourcePolicyId"],
                            "panelGroup": record["panelGroup"],
                            "policyFamily": record["policyFamily"],
                            "goalDirection": str(goal),
                            "inputSeed": int(input_seed),
                            "schedulerSeed": int(scheduler_seed),
                            "tieBreakerSeed": int(tie_seed),
                            "n": 8,
                            "runSucceeded": False,
                            "runError": f"{type(exc).__name__}: {exc}",
                            "reproducibleSeedReplay": False,
                            "reproducibilityError": "run_exception",
                            "validationSchemaVersion": VALIDATION_SCHEMA_VERSION,
                        }
                    )

    validation_df = pd.DataFrame(validation_rows)
    panel_summary = summarize_panel_readiness(panel_df, validation_df, serialization)
    return validation_df, panel_summary


def summarize_panel_readiness(
    panel_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    serialization: Mapping[str, tuple[bool, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in panel_df.to_dict(orient="records"):
        panel_policy_id = str(record["panelPolicyId"])
        sub = validation_df[validation_df["panelPolicyId"] == panel_policy_id].copy()
        successful = sub[sub["runSucceeded"].map(bool)] if not sub.empty else sub
        serialization_ok, serialization_error = serialization.get(panel_policy_id, (False, "not_checked"))
        can_run = bool(not successful.empty and successful["runSucceeded"].all())
        reproducible = bool(can_run and successful["reproducibleSeedReplay"].map(bool).all())
        value_counts_ok = bool(can_run and successful["valueCountsConserved"].map(bool).all())
        allowed_goals: list[str] = []
        sort_status_by_goal: dict[str, str] = {}
        for goal, goal_df in successful.groupby("goalDirection"):
            completed_count = int(goal_df["completedGoal"].map(bool).sum())
            total = int(len(goal_df))
            if total and completed_count == total:
                allowed_goals.append(str(goal))
                sort_status_by_goal[str(goal)] = "sorts_all_validation_runs"
            elif completed_count:
                allowed_goals.append(str(goal))
                sort_status_by_goal[str(goal)] = f"partial_sort_signal_{completed_count}_of_{total}"
            elif total:
                sort_status_by_goal[str(goal)] = "no_pure_sort_completion"

        is_control = str(record["panelGroup"]) in {"null_control", "randomized_control"}
        if not serialization_ok:
            readiness = "excluded_serialization_failed"
            quarantine = serialization_error
        elif not can_run:
            readiness = "excluded_runtime_failed"
            errors = sorted(set(str(err) for err in sub.get("runError", pd.Series(dtype=str)) if str(err)))
            quarantine = "; ".join(errors) or "no successful validation runs"
        elif not reproducible:
            readiness = "quarantine_nonreproducible"
            quarantine = "same-seed replay mismatch"
        elif not value_counts_ok:
            readiness = "quarantine_value_nonconservation"
            quarantine = "value multiset changed"
        elif allowed_goals:
            readiness = "ready_for_mixture"
            quarantine = ""
        elif is_control:
            readiness = "ready_control_non_sorting"
            quarantine = "non-sorting behavior is expected for null/randomized controls"
        else:
            readiness = "quarantine_cannot_sort_pure"
            quarantine = "no validation run reached its target sorted order"

        if successful.empty:
            cost = {}
        else:
            cost = {
                "meanActivationCount": float(successful["activationCount"].mean()),
                "meanSwapCount": float(successful["swapCount"].mean()),
                "meanComparisonCount": float(successful["comparisonCount"].mean()),
                "meanRuntimeSeconds": float(successful["runtimeSeconds"].mean()),
                "maxActivationCount": int(successful["activationCount"].max()),
            }
        row = dict(record)
        row.update(
            {
                "serializationRoundtripSuccess": bool(serialization_ok),
                "serializationError": serialization_error,
                "canRunCommonInterface": bool(can_run),
                "reproducibleSeeds": bool(reproducible),
                "valueCountsConserved": bool(value_counts_ok),
                "allowedGoalsJson": compact_json(allowed_goals or (["none"] if is_control else [])),
                "pureSortStatusJson": compact_json(sort_status_by_goal),
                "validationRunCount": int(len(sub)),
                "successfulValidationRunCount": int(len(successful)),
                "completedGoalRunCount": int(successful["completedGoal"].map(bool).sum()) if not successful.empty else 0,
                "readinessStatus": readiness,
                "readyForS02Mixing": readiness in {"ready_for_mixture", "ready_control_non_sorting"},
                "quarantineOrExclusionReason": quarantine,
                "computationalCostSummaryJson": compact_json(cost),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validation_checks(panel_df: pd.DataFrame, validation_df: pd.DataFrame, panel_summary: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "checkId": "panel_nonempty",
            "success": bool(len(panel_df) > 0),
            "detail": f"{len(panel_df)} panel policies assembled",
        },
        {
            "checkId": "stable_unique_ids",
            "success": bool(panel_df["panelPolicyId"].is_unique),
            "detail": "panelPolicyId values are unique",
        },
        {
            "checkId": "serialization_roundtrip",
            "success": bool(panel_summary["serializationRoundtripSuccess"].map(bool).all()),
            "detail": f"{int(panel_summary['serializationRoundtripSuccess'].sum())}/{len(panel_summary)} policies round-tripped",
        },
        {
            "checkId": "pure_runs_completed_or_classified",
            "success": bool(len(validation_df) > 0 and panel_summary["readinessStatus"].notna().all()),
            "detail": f"{len(validation_df)} pure validation rows written",
        },
        {
            "checkId": "reproducible_seed_replay",
            "success": bool(validation_df["reproducibleSeedReplay"].fillna(False).map(bool).all()),
            "detail": "same-seed duplicate runs matched for every validation condition",
        },
        {
            "checkId": "common_metric_schema",
            "success": bool(
                {
                    "panelPolicyId",
                    "goalDirection",
                    "finalGoalSortednessPercent",
                    "swapCount",
                    "comparisonCount",
                    "activationCount",
                    "reproducibleSeedReplay",
                }.issubset(validation_df.columns)
            ),
            "detail": "validation table contains S01 common metric columns",
        },
    ]
    return pd.DataFrame(checks)


def write_markdown_reports(
    *,
    step_dir: Path,
    panel_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
    checks_df: pd.DataFrame,
    artifacts_written: Sequence[str],
) -> tuple[Path, Path]:
    ready = int(panel_summary["readyForS02Mixing"].map(bool).sum())
    quarantined = int(len(panel_summary) - ready)
    sortable = int(panel_summary["allowedGoalsJson"].map(lambda text: "increasing" in parse_json_maybe(text, [])).sum())
    validation_result = "passed" if checks_df["success"].map(bool).all() and ready > 0 else "failed"
    caveat = (
        "Null and randomized controls are intentionally non-sorting; quarantined policies should not be used in S02 "
        "until their interface or goal semantics are reviewed."
    )
    summary = f"""# Research Step {STEP_ID}: Expand the Algotype set

## Completion status

Completed on 2026-06-28. Outcome classification: supportive.

## Artifacts written

{chr(10).join(f"- `{path}`" for path in artifacts_written)}

## Validation result

{validation_result}. Assembled {len(panel_summary)} panel policies; {ready} are ready for S02 mixture construction, {quarantined} are quarantined or excluded, and {sortable} have increasing-goal pure-sort evidence in the S01 validation panel. Pure validation wrote {len(validation_df)} condition rows.

## Caveats or blockers

{caveat}

## Lay summary

S01 converted the original algorithms, null and random controls, selected E03 discovered/frontier policies, and E04 memory/signal candidates into one machine-readable policy panel. Each policy was tested alone before any mixing work begins.

## Recommended next action

Chief Scientist review, then execute S02 mixture-ratio sweeps using only rows marked `readyForS02Mixing=true` in `$ARTIFACTS_DIR/data/e06_algotype_panel.parquet`.
"""
    summary_path = step_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    blocked = panel_summary[~panel_summary["readyForS02Mixing"].map(bool)].copy()
    if blocked.empty:
        exclusions = f"""# S01 Exclusions Report

No policies were excluded or quarantined.

## Caveats

Null and randomized controls remain in the panel as non-sorting controls and should be interpreted separately from sorting-capable Algotypes.
"""
    else:
        rows = "\n".join(
            f"| `{row.panelPolicyId}` | {row.panelGroup} | {row.readinessStatus} | {row.quarantineOrExclusionReason or 'none'} |"
            for row in blocked.itertuples(index=False)
        )
        exclusions = f"""# S01 Exclusions Report

| Panel policy ID | Group | Status | Reason |
| --- | --- | --- | --- |
{rows}

## Caveats

Quarantine here is a computational interface or pure-sort limitation, not a biological claim. Null and randomized controls can still be useful controls even when they do not sort.
"""
    exclusions_path = step_dir / "exclusions_report.md"
    exclusions_path.write_text(exclusions, encoding="utf-8")
    return summary_path, exclusions_path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.exists() and path.is_file():
            records.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "sizeBytes": int(path.stat().st_size),
                }
            )
    return records


def run_s01_algotype_panel(
    *,
    artifacts_dir: Path | None = None,
    repo_root: Path | None = None,
    frontier_limit: int | None = None,
    discovered_limit: int = 6,
    handoff_limit: int | None = None,
    max_activations: int = 1800,
) -> dict[str, Any]:
    artifacts_dir = artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    data_dir = artifacts_dir / "data"
    results_dir = artifacts_dir / "results"
    provenance_dir = artifacts_dir / "provenance"
    for path in [step_dir, data_dir, results_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    panel_df = build_algotype_panel(
        frontier_limit=frontier_limit,
        discovered_limit=discovered_limit,
        handoff_limit=handoff_limit,
    )
    validation_df, panel_summary = run_validation_panel(panel_df, max_activations=max_activations)
    checks_df = validation_checks(panel_df, validation_df, panel_summary)

    panel_global_parquet = data_dir / "e06_algotype_panel.parquet"
    panel_global_csv = data_dir / "e06_algotype_panel.csv"
    panel_step_parquet = step_dir / "algotype_panel.parquet"
    panel_step_csv = step_dir / "algotype_panel.csv"
    validation_step_parquet = step_dir / "pure_policy_validation.parquet"
    validation_step_csv = step_dir / "pure_policy_validation.csv"
    validation_global_parquet = results_dir / "e06_s01_pure_policy_validation.parquet"
    validation_global_csv = results_dir / "e06_s01_pure_policy_validation.csv"
    checks_path = step_dir / "validation_checks.csv"
    checks_parquet = step_dir / "validation_checks.parquet"
    exclusions_table = panel_summary[~panel_summary["readyForS02Mixing"].map(bool)].copy()
    exclusions_step_parquet = step_dir / "excluded_or_quarantined_policies.parquet"
    exclusions_step_csv = step_dir / "excluded_or_quarantined_policies.csv"
    exclusions_global_parquet = results_dir / "e06_s01_excluded_or_quarantined_policies.parquet"
    exclusions_global_csv = results_dir / "e06_s01_excluded_or_quarantined_policies.csv"

    for target in [panel_global_parquet, panel_step_parquet]:
        panel_summary.to_parquet(target, index=False)
    for target in [panel_global_csv, panel_step_csv]:
        panel_summary.to_csv(target, index=False)
    for target in [validation_step_parquet, validation_global_parquet]:
        validation_df.to_parquet(target, index=False)
    for target in [validation_step_csv, validation_global_csv]:
        validation_df.to_csv(target, index=False)
    checks_df.to_csv(checks_path, index=False)
    checks_df.to_parquet(checks_parquet, index=False)
    for target in [exclusions_step_parquet, exclusions_global_parquet]:
        exclusions_table.to_parquet(target, index=False)
    for target in [exclusions_step_csv, exclusions_global_csv]:
        exclusions_table.to_csv(target, index=False)

    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = provenance_dir / "run_manifest.json"
    planned_artifacts = [
        panel_global_parquet,
        panel_global_csv,
        panel_step_parquet,
        panel_step_csv,
        validation_step_parquet,
        validation_step_csv,
        validation_global_parquet,
        validation_global_csv,
        checks_path,
        checks_parquet,
        exclusions_step_parquet,
        exclusions_step_csv,
        exclusions_global_parquet,
        exclusions_global_csv,
        step_dir / "summary.md",
        step_dir / "exclusions_report.md",
        status_path,
        manifest_path,
        run_manifest_path,
    ]
    summary_path, exclusions_report_path = write_markdown_reports(
        step_dir=step_dir,
        panel_summary=panel_summary,
        validation_df=validation_df,
        checks_df=checks_df,
        artifacts_written=[str(path) for path in planned_artifacts],
    )
    artifact_paths = list(dict.fromkeys([*planned_artifacts, summary_path, exclusions_report_path]))

    ready = int(panel_summary["readyForS02Mixing"].map(bool).sum())
    validation_result = "passed" if checks_df["success"].map(bool).all() and ready > 0 else "failed"
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_result == "passed",
        "status": "completed" if validation_result == "passed" else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": [
            "Null and randomized policies are retained as controls even when they do not sort.",
            "Policies marked readyForS02Mixing=false are quarantined until reviewed.",
            CLAIM_BOUNDARY,
        ],
        "recommendedNextAction": (
            "Chief Scientist review, then run S02 mixture-ratio sweeps using only "
            "readyForS02Mixing=true panel rows."
        ),
        "outcomeClassification": "supportive" if validation_result == "passed" else "constraining/contradictory",
        "panelPolicyCount": int(len(panel_summary)),
        "readyForS02MixingCount": ready,
        "quarantinedOrExcludedCount": int(len(panel_summary) - ready),
        "validationConditionCount": int(len(validation_df)),
        "completedAt": completed_at,
        "wallTimeSeconds": float(time.perf_counter() - started),
    }
    write_json(status_path, status)
    if status_path not in artifact_paths:
        artifact_paths.append(status_path)

    manifest_payload = {
        "schema": "eidosoma.step_artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "panelSchemaVersion": PANEL_SCHEMA_VERSION,
        "validationSchemaVersion": VALIDATION_SCHEMA_VERSION,
        "artifacts": artifact_records(artifact_paths),
    }
    write_json(manifest_path, manifest_payload)
    if manifest_path not in artifact_paths:
        artifact_paths.append(manifest_path)

    run_manifest = {
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
            "workerCount": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
                if os.environ.get(key) is not None
            },
            "newDependenciesInstalled": [],
        },
        "researchSteps": {STEP_ID: status},
        "artifacts": artifact_records(artifact_paths),
    }
    write_json(run_manifest_path, run_manifest)
    if run_manifest_path not in artifact_paths:
        artifact_paths.append(run_manifest_path)

    status["artifactsWritten"] = [str(path) for path in artifact_paths]
    write_json(status_path, status)
    manifest_payload["artifacts"] = artifact_records(artifact_paths)
    write_json(manifest_path, manifest_payload)
    return {
        "status": status,
        "panel": panel_summary,
        "validation": validation_df,
        "checks": checks_df,
        "artifactPaths": [str(path) for path in artifact_paths],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E06 S01 Algotype panel assembly and pure validation.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--frontier-limit", type=int, default=None)
    parser.add_argument("--discovered-limit", type=int, default=6)
    parser.add_argument("--handoff-limit", type=int, default=None)
    parser.add_argument("--max-activations", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_s01_algotype_panel(
        artifacts_dir=args.artifacts_dir,
        frontier_limit=args.frontier_limit,
        discovered_limit=args.discovered_limit,
        handoff_limit=args.handoff_limit,
        max_activations=args.max_activations,
    )
    status = result["status"]
    print(
        f"{STEP_ID} {status['status']}: {status['panelPolicyCount']} policies, "
        f"{status['readyForS02MixingCount']} ready, validation {status['validationResult']}"
    )
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
