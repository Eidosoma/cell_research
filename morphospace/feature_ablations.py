"""Feature-ablation helpers for E03 S12.

S12 rewrites DSL policy records into paired ablations, then compares each
ablated policy against its source policy under identical simulator seeds.  These
helpers keep ablations inside the S02 DSL boundary and make every transformation
auditable as JSON.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .competence import canonical_json
from .rule_dsl import DSL_VERSION, RuleProgram, parse_rule_program


FEATURE_ABLATION_VERSION = "e03_s12_feature_ablations.v1"

ABLATION_TYPES = (
    "suppress_compare_left",
    "suppress_compare_right",
    "remove_target_position",
    "remove_memory_updates",
    "remove_waiting",
    "remove_stochasticity",
    "strict_tie_rules",
)

DELTA_METRICS = (
    "completionSuccess",
    "finalSortednessScore",
    "finalMonotonicityScore",
    "swapEfficiencyScore",
    "comparisonEfficiencyScore",
    "activationEfficiencyScore",
    "energyScore",
    "robustnessScore",
    "delayedGratification",
    "delayedGratificationScore",
    "aggregationFinal",
    "aggregationAucScore",
    "oscillationProxy",
    "oscillationScore",
    "failureOscillationScore",
)


@dataclass(frozen=True)
class AblationOutcome:
    """One DSL ablation result."""

    ablation_type: str
    changed: bool
    program: RuleProgram
    change_summary: dict[str, Any]


@dataclass(frozen=True)
class AblationTask:
    task_id: str
    task_family: str
    task_panel: str
    input_profile: str
    initial_values: tuple[int, ...]
    scheduler_seed_base: int
    tie_seed_base: int
    frozen_variant: str = "none"
    frozen_positions: tuple[int, ...] = ()
    chimera_with_null: bool = False
    candidate_positions: tuple[int, ...] = ()
    max_activations: int = 640
    max_swaps: int = 640
    max_comparisons: int = 2560

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskFamily": self.task_family,
            "taskPanel": self.task_panel,
            "inputProfile": self.input_profile,
            "initialValues": list(self.initial_values),
            "schedulerSeedBase": int(self.scheduler_seed_base),
            "tieSeedBase": int(self.tie_seed_base),
            "frozenVariant": self.frozen_variant,
            "frozenPositions": list(self.frozen_positions),
            "chimeraWithNull": bool(self.chimera_with_null),
            "candidatePositions": list(self.candidate_positions),
            "maxActivations": int(self.max_activations),
            "maxSwaps": int(self.max_swaps),
            "maxComparisons": int(self.max_comparisons),
        }


def _json_payload(program: RuleProgram | str | Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(parse_rule_program(program).to_dict())


def _false_predicate() -> dict[str, Any]:
    return {"op": "position_compare", "operator": "<", "value": 0}


def _stable_ablation_policy_id(source_policy_id: str, ablation_type: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json({"source": source_policy_id, "ablation": ablation_type, "payload": payload}).encode()).hexdigest()[:14]
    prefix = str(source_policy_id)
    if len(prefix) > 28:
        prefix = prefix[:28]
    return f"ab_{prefix}_{digest}"


def _program_with_identity(payload: dict[str, Any], *, source_policy_id: str, ablation_type: str) -> RuleProgram:
    payload = copy.deepcopy(payload)
    payload["policy_id"] = _stable_ablation_policy_id(source_policy_id, ablation_type, payload)
    payload["name"] = f"{payload.get('name', source_policy_id)} | {ablation_type}"
    return parse_rule_program(payload)


def _mark_reason(action: dict[str, Any], reason: str) -> None:
    action["reason"] = reason


def suppress_compare(program: RuleProgram | str | Mapping[str, Any], predicate_op: str) -> AblationOutcome:
    if predicate_op not in {"compare_left", "compare_right"}:
        raise ValueError("predicate_op must be compare_left or compare_right")
    payload = _json_payload(program)
    source_policy_id = str(payload["policy_id"])
    replaced = 0
    for rule in payload.get("rules", []):
        new_when = []
        for predicate in rule.get("when", []):
            if predicate.get("op") == predicate_op:
                new_when.append(_false_predicate())
                replaced += 1
            else:
                new_when.append(predicate)
        rule["when"] = new_when or [_false_predicate()]
    ablation_type = f"suppress_{predicate_op.removeprefix('compare_')}_comparison"
    # Public artifact naming keeps the original queue language.
    artifact_type = "suppress_compare_left" if predicate_op == "compare_left" else "suppress_compare_right"
    program_out = _program_with_identity(payload, source_policy_id=source_policy_id, ablation_type=artifact_type)
    return AblationOutcome(
        artifact_type,
        replaced > 0,
        program_out,
        {"disabledPredicateOp": predicate_op, "predicateReplacements": int(replaced), "internalName": ablation_type},
    )


def remove_target_position(program: RuleProgram | str | Mapping[str, Any]) -> AblationOutcome:
    payload = _json_payload(program)
    source_policy_id = str(payload["policy_id"])
    removed_updates = 0
    converted_actions = 0
    if "initial_state" in payload and "target_position" in payload["initial_state"]:
        payload["initial_state"].pop("target_position", None)
        removed_updates += 1
    for rule in payload.get("rules", []):
        then = rule.get("then", {})
        kept_updates = []
        for update in then.get("updates", []):
            if update.get("op") == "estimate_target_position" or update.get("key") == "target_position":
                removed_updates += 1
            else:
                kept_updates.append(update)
        if kept_updates:
            then["updates"] = kept_updates
        else:
            then.pop("updates", None)
        if then.get("action") == "swap_target":
            then.clear()
            then["action"] = "wait"
            _mark_reason(then, "ablate_target_position")
            converted_actions += 1
    default = payload.get("default", {})
    if default.get("action") == "swap_target":
        default.clear()
        default["action"] = "wait"
        _mark_reason(default, "ablate_target_position_default")
        converted_actions += 1
    program_out = _program_with_identity(payload, source_policy_id=source_policy_id, ablation_type="remove_target_position")
    return AblationOutcome(
        "remove_target_position",
        removed_updates > 0 or converted_actions > 0,
        program_out,
        {"removedTargetUpdates": int(removed_updates), "convertedSwapTargetActions": int(converted_actions)},
    )


def remove_memory_updates(program: RuleProgram | str | Mapping[str, Any]) -> AblationOutcome:
    payload = _json_payload(program)
    source_policy_id = str(payload["policy_id"])
    removed_updates = 0
    removed_initial_keys: list[str] = []
    state = dict(payload.get("initial_state", {}))
    for key in list(state):
        if key != "target_position":
            removed_initial_keys.append(str(key))
            state.pop(key, None)
    payload["initial_state"] = state
    for rule in payload.get("rules", []):
        then = rule.get("then", {})
        kept_updates = []
        for update in then.get("updates", []):
            if update.get("op") in {"set", "increment", "decrement", "signal"}:
                removed_updates += 1
            else:
                kept_updates.append(update)
        if kept_updates:
            then["updates"] = kept_updates
        else:
            then.pop("updates", None)
        if then.get("action") in {"remember", "signal"}:
            then["action"] = "wait"
            _mark_reason(then, "ablate_memory_or_signal")
    program_out = _program_with_identity(payload, source_policy_id=source_policy_id, ablation_type="remove_memory_updates")
    return AblationOutcome(
        "remove_memory_updates",
        removed_updates > 0 or bool(removed_initial_keys),
        program_out,
        {"removedStateUpdates": int(removed_updates), "removedInitialStateKeys": removed_initial_keys},
    )


def remove_waiting(program: RuleProgram | str | Mapping[str, Any]) -> AblationOutcome:
    payload = _json_payload(program)
    source_policy_id = str(payload["policy_id"])
    converted = 0
    for rule in payload.get("rules", []):
        then = rule.get("then", {})
        if then.get("action") == "wait":
            then.clear()
            then["action"] = "swap_right"
            _mark_reason(then, "ablate_wait_force_right")
            converted += 1
    default = payload.get("default", {})
    if default.get("action") == "wait":
        default.clear()
        default["action"] = "swap_right"
        _mark_reason(default, "ablate_wait_force_right_default")
        converted += 1
    program_out = _program_with_identity(payload, source_policy_id=source_policy_id, ablation_type="remove_waiting")
    return AblationOutcome(
        "remove_waiting",
        converted > 0,
        program_out,
        {"convertedWaitActions": int(converted), "replacementAction": "swap_right"},
    )


def remove_stochasticity(program: RuleProgram | str | Mapping[str, Any]) -> AblationOutcome:
    payload = _json_payload(program)
    source_policy_id = str(payload["policy_id"])
    changed = 0
    for action in [rule.get("then", {}) for rule in payload.get("rules", [])] + [payload.get("default", {})]:
        probability = action.get("probability")
        if probability is not None and float(probability) != 1.0:
            action.pop("probability", None)
            changed += 1
    program_out = _program_with_identity(payload, source_policy_id=source_policy_id, ablation_type="remove_stochasticity")
    return AblationOutcome(
        "remove_stochasticity",
        changed > 0,
        program_out,
        {"probabilitiesForcedToOne": int(changed)},
    )


def strict_tie_rules(program: RuleProgram | str | Mapping[str, Any]) -> AblationOutcome:
    payload = _json_payload(program)
    source_policy_id = str(payload["policy_id"])
    mapping = {"<=": "<", ">=": ">", "==": "!=", "!=": "=="}
    changed = 0
    changed_ops: dict[str, int] = {}
    for rule in payload.get("rules", []):
        for predicate in rule.get("when", []):
            operator = predicate.get("operator")
            if operator in mapping:
                predicate["operator"] = mapping[operator]
                changed += 1
                changed_ops[str(operator)] = changed_ops.get(str(operator), 0) + 1
    program_out = _program_with_identity(payload, source_policy_id=source_policy_id, ablation_type="strict_tie_rules")
    return AblationOutcome(
        "strict_tie_rules",
        changed > 0,
        program_out,
        {"operatorChanges": changed_ops, "operatorChangeCount": int(changed)},
    )


def ablate_program(program: RuleProgram | str | Mapping[str, Any], ablation_type: str) -> AblationOutcome:
    """Apply one named S12 ablation and return a valid DSL program."""

    if ablation_type == "suppress_compare_left":
        return suppress_compare(program, "compare_left")
    if ablation_type == "suppress_compare_right":
        return suppress_compare(program, "compare_right")
    if ablation_type == "remove_target_position":
        return remove_target_position(program)
    if ablation_type == "remove_memory_updates":
        return remove_memory_updates(program)
    if ablation_type == "remove_waiting":
        return remove_waiting(program)
    if ablation_type == "remove_stochasticity":
        return remove_stochasticity(program)
    if ablation_type == "strict_tie_rules":
        return strict_tie_rules(program)
    raise ValueError(f"unknown ablation type: {ablation_type}")


def available_ablation_outcomes(
    program: RuleProgram | str | Mapping[str, Any],
    *,
    ablation_types: Sequence[str] = ABLATION_TYPES,
) -> list[AblationOutcome]:
    outcomes: list[AblationOutcome] = []
    for ablation_type in ablation_types:
        outcome = ablate_program(program, ablation_type)
        if outcome.changed:
            outcomes.append(outcome)
    return outcomes


def _safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Mapping):
        return dict(value)
    return default


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda value: bool(value) if pd.notna(value) else False)


def select_s12_source_policies(
    *,
    corpus_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    exemplars_df: pd.DataFrame,
    max_generated_per_class: int = 3,
) -> pd.DataFrame:
    """Select classic, generated, and elite DSL source policies for S12."""

    corpus = corpus_df.copy()
    corpus["policyId"] = corpus["policyId"].astype(str)
    elites = elites_df.copy()
    elites["policyId"] = elites["policyId"].astype(str)
    assignments = assignments_df.copy()
    assignments["policyId"] = assignments["policyId"].astype(str)
    exemplars = exemplars_df.copy()
    exemplars["policyId"] = exemplars["policyId"].astype(str)
    elite_ids = set(elites["policyId"])

    assignment_cols = [
        "policyId",
        "className",
        "classLabel",
        "primaryRole",
        "isClassicPolicy",
        "isNullPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "distanceToCentroid",
    ]
    meta = assignments[[column for column in assignment_cols if column in assignments.columns]].copy()
    joined = corpus.merge(meta, on="policyId", how="left")
    joined["isS08Elite"] = _as_bool(joined.get("isS08Elite", pd.Series(False, index=joined.index))) | joined["policyId"].isin(elite_ids)
    joined["isClassicPolicy"] = _as_bool(joined.get("isClassicPolicy", pd.Series(False, index=joined.index))) | joined[
        "generationMethod"
    ].eq("classic_parameter_sweep")
    joined["isNullPolicy"] = _as_bool(joined.get("isNullPolicy", pd.Series(False, index=joined.index)))

    selected_ids: set[str] = set()
    selection_notes: dict[str, list[str]] = {}

    def add(policy_id: str, note: str) -> None:
        selected_ids.add(str(policy_id))
        selection_notes.setdefault(str(policy_id), []).append(note)

    for policy_id in joined[joined["isClassicPolicy"]]["policyId"].sort_values():
        add(policy_id, "all_classic_parameterized")
    for policy_id in elites.sort_values("eliteRank", kind="mergesort")["policyId"]:
        add(policy_id, "all_s08_elites")

    exemplar_join = exemplars.merge(
        joined[
            [
                "policyId",
                "isClassicPolicy",
                "isNullPolicy",
                "isS08Elite",
                "className",
                "classLabel",
                "generationMethod",
                "lineageId",
            ]
        ],
        on="policyId",
        how="inner",
        suffixes=("", "_corpus"),
    )
    generated = exemplar_join[
        ~_as_bool(exemplar_join["isClassicPolicy"])
        & ~_as_bool(exemplar_join["isNullPolicy"])
        & ~_as_bool(exemplar_join["isS08Elite"])
    ].copy()
    generated = generated.sort_values(["className", "exemplarRank", "policyId"], kind="mergesort")
    for _, group in generated.groupby("className", sort=True):
        for policy_id in group.head(int(max_generated_per_class))["policyId"]:
            add(str(policy_id), "generated_s11_exemplar")

    selected = joined[joined["policyId"].isin(selected_ids)].copy()
    selected["sourceSelectionNotesJson"] = selected["policyId"].map(lambda policy_id: canonical_json(sorted(selection_notes.get(policy_id, []))))

    def role(row: pd.Series) -> str:
        roles: list[str] = []
        if bool(row.get("isClassicPolicy", False)):
            roles.append("classic")
        if bool(row.get("isS08Elite", False)):
            roles.append("elite")
        if not roles:
            roles.append("generated")
        return "+".join(roles)

    selected["sourceRole"] = selected.apply(role, axis=1)
    selected["selectedForS12"] = True
    sort_cols = ["sourceRole", "className", "generationMethod", "lineageId", "policyId"]
    return selected.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def build_ablation_policy_table(
    source_policies_df: pd.DataFrame,
    *,
    include_original: bool = True,
    ablation_types: Sequence[str] = ABLATION_TYPES,
) -> pd.DataFrame:
    """Return original and ablated DSL records with parse/round-trip validation."""

    rows: list[dict[str, Any]] = []
    for _, source in source_policies_df.iterrows():
        source_policy_id = str(source["policyId"])
        program = parse_rule_program(source["dslProgramJson"])
        common = {
            "featureAblationVersion": FEATURE_ABLATION_VERSION,
            "sourcePolicyId": source_policy_id,
            "sourceRole": str(source.get("sourceRole", "unknown")),
            "sourceClassName": str(source.get("className", "missing")),
            "sourceClassLabel": str(source.get("classLabel", "missing")),
            "sourcePrimaryRole": str(source.get("primaryRole", "missing")),
            "sourceFamily": str(source.get("family", "unknown")),
            "sourceGenerationMethod": str(source.get("generationMethod", "unknown")),
            "sourceLineageId": str(source.get("lineageId", "unknown")),
            "sourceSelectionNotesJson": str(source.get("sourceSelectionNotesJson", "[]")),
        }
        if include_original:
            rows.append(
                {
                    **common,
                    "ablationPolicyId": source_policy_id,
                    "variantRole": "original",
                    "ablationType": "original",
                    "changed": False,
                    "changeSummaryJson": "{}",
                    "dslProgramJson": program.to_json(),
                    "dslVersion": program.version,
                    "parseSuccess": True,
                    "roundtripSuccess": True,
                    "programSha256": hashlib.sha256(program.to_json().encode()).hexdigest(),
                }
            )
        for outcome in available_ablation_outcomes(program, ablation_types=ablation_types):
            roundtrip = parse_rule_program(outcome.program.to_json())
            rows.append(
                {
                    **common,
                    "ablationPolicyId": outcome.program.policy_id,
                    "variantRole": "ablated",
                    "ablationType": outcome.ablation_type,
                    "changed": bool(outcome.changed),
                    "changeSummaryJson": canonical_json(outcome.change_summary),
                    "dslProgramJson": outcome.program.to_json(),
                    "dslVersion": outcome.program.version,
                    "parseSuccess": True,
                    "roundtripSuccess": canonical_json(roundtrip.to_dict()) == canonical_json(outcome.program.to_dict()),
                    "programSha256": hashlib.sha256(outcome.program.to_json().encode()).hexdigest(),
                }
            )
    return pd.DataFrame(rows).sort_values(["sourceRole", "sourcePolicyId", "variantRole", "ablationType"], kind="mergesort").reset_index(drop=True)


def paired_delta_table(run_vectors_df: pd.DataFrame, policy_table_df: pd.DataFrame) -> pd.DataFrame:
    """Compute same-task, same-seed ablation deltas against original policies."""

    vectors = run_vectors_df.copy()
    for column in ["policyId", "taskId", "replicateIndex"]:
        if column not in vectors.columns:
            raise ValueError(f"run vector table missing {column}")
    metadata = policy_table_df[
        [
            "ablationPolicyId",
            "sourcePolicyId",
            "sourceRole",
            "sourceClassName",
            "sourceClassLabel",
            "sourcePrimaryRole",
            "sourceGenerationMethod",
            "sourceLineageId",
            "variantRole",
            "ablationType",
        ]
    ].rename(columns={"ablationPolicyId": "policyId"})
    overlapping = [column for column in metadata.columns if column != "policyId" and column in vectors.columns]
    if overlapping:
        vectors = vectors.drop(columns=overlapping)
    vectors = vectors.merge(metadata, on="policyId", how="left", validate="many_to_one")
    originals = vectors[vectors["variantRole"].eq("original")].copy()
    ablated = vectors[vectors["variantRole"].eq("ablated")].copy()
    join_keys = ["sourcePolicyId", "taskId", "replicateIndex"]
    original_cols = join_keys + [
        "policyId",
        "sourceRunId",
        "taskFamily",
        "taskPanel",
        "inputProfile",
        "frozenVariant",
        "frozenCount",
    ] + [column for column in DELTA_METRICS if column in originals.columns]
    original_cols = list(dict.fromkeys(original_cols))
    merged = ablated.merge(
        originals[original_cols],
        on=join_keys,
        how="inner",
        suffixes=("", "_original"),
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        payload = {
            "featureAblationVersion": FEATURE_ABLATION_VERSION,
            "sourcePolicyId": str(row["sourcePolicyId"]),
            "ablationPolicyId": str(row["policyId"]),
            "originalPolicyId": str(row["policyId_original"]),
            "sourceRole": str(row["sourceRole"]),
            "sourceClassName": str(row["sourceClassName"]),
            "sourceClassLabel": str(row["sourceClassLabel"]),
            "sourcePrimaryRole": str(row["sourcePrimaryRole"]),
            "sourceGenerationMethod": str(row["sourceGenerationMethod"]),
            "sourceLineageId": str(row["sourceLineageId"]),
            "ablationType": str(row["ablationType"]),
            "taskId": str(row["taskId"]),
            "taskFamily": str(row["taskFamily"]),
            "taskPanel": str(row["taskPanel"]),
            "inputProfile": str(row["inputProfile"]),
            "frozenVariant": str(row["frozenVariant"]),
            "frozenCount": int(row["frozenCount"]) if pd.notna(row["frozenCount"]) else 0,
            "replicateIndex": int(row["replicateIndex"]) if pd.notna(row["replicateIndex"]) else -1,
            "sourceRunId": str(row["sourceRunId"]),
            "originalSourceRunId": str(row["sourceRunId_original"]),
        }
        for metric in DELTA_METRICS:
            if metric not in row.index or f"{metric}_original" not in row.index:
                continue
            ablated_value = row[metric]
            original_value = row[f"{metric}_original"]
            payload[f"{metric}Original"] = float(original_value) if pd.notna(original_value) else np.nan
            payload[f"{metric}Ablated"] = float(ablated_value) if pd.notna(ablated_value) else np.nan
            if pd.notna(ablated_value) and pd.notna(original_value):
                payload[f"{metric}Delta"] = float(ablated_value) - float(original_value)
            else:
                payload[f"{metric}Delta"] = np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def feature_effect_summary(paired_delta_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired ablation effects by feature and task family."""

    if paired_delta_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouping_columns = ["ablationType", "sourceRole", "taskFamily"]
    for keys, group in paired_delta_df.groupby(grouping_columns, dropna=False, sort=True):
        ablation_type, source_role, task_family = map(str, keys)
        payload: dict[str, Any] = {
            "featureAblationVersion": FEATURE_ABLATION_VERSION,
            "ablationType": ablation_type,
            "sourceRole": source_role,
            "taskFamily": task_family,
            "pairCount": int(len(group)),
            "sourcePolicyCount": int(group["sourcePolicyId"].nunique()),
        }
        for metric in DELTA_METRICS:
            delta_col = f"{metric}Delta"
            original_col = f"{metric}Original"
            if delta_col not in group.columns:
                continue
            deltas = pd.to_numeric(group[delta_col], errors="coerce")
            valid = deltas[np.isfinite(deltas)]
            payload[f"{metric}DeltaMean"] = float(valid.mean()) if len(valid) else np.nan
            payload[f"{metric}DeltaMedian"] = float(valid.median()) if len(valid) else np.nan
            payload[f"{metric}DeltaSd"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0 if len(valid) == 1 else np.nan
            payload[f"{metric}ImprovedFraction"] = float((valid > 1e-12).mean()) if len(valid) else np.nan
            payload[f"{metric}WorsenedFraction"] = float((valid < -1e-12).mean()) if len(valid) else np.nan
            originals = pd.to_numeric(group.get(original_col, pd.Series(dtype=float)), errors="coerce")
            payload[f"{metric}OriginalMean"] = float(originals[np.isfinite(originals)].mean()) if np.isfinite(originals).any() else np.nan
        rows.append(payload)
    summary = pd.DataFrame(rows)
    if "finalSortednessScoreDeltaMean" in summary.columns:
        summary["candidateNecessityCall"] = np.where(
            summary["finalSortednessScoreDeltaMean"].le(-0.05) | summary.get("completionSuccessDeltaMean", 0).le(-0.05),
            "candidate_necessary_for_observed_task_family_competence",
            np.where(
                summary["finalSortednessScoreDeltaMean"].ge(0.05) | summary.get("completionSuccessDeltaMean", 0).ge(0.05),
                "ablation_improved_or_removed_costly_feature",
                "no_clear_mean_effect_in_bounded_panel",
            ),
        )
    else:
        summary["candidateNecessityCall"] = "no_final_sortedness_delta_available"
    summary["sufficiencyCaveat"] = "Removal-only paired ablations test local necessity/contribution; they do not prove feature sufficiency."
    return summary


def validate_ablation_outputs(
    *,
    source_policies_df: pd.DataFrame,
    policy_table_df: pd.DataFrame,
    run_df: pd.DataFrame,
    vector_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
    effect_summary_df: pd.DataFrame,
    parser_validation_df: pd.DataFrame,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    expected_task_count: int,
    seed_count: int,
    report_exists: bool,
    figure_paths: Sequence[str],
    s13_dir_exists: bool,
) -> pd.DataFrame:
    """Return S12 validation rows."""

    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append({"checkId": check_id, "success": bool(success), "detail": str(detail)})

    for step in ["S08", "S09", "S10", "S11"]:
        status = upstream_statuses.get(step, {})
        add(f"upstream_{step.lower()}_success", bool(status.get("success")), f"{step} status={status.get('status')}")

    source_roles = set(",".join(source_policies_df.get("sourceRole", pd.Series(dtype=str)).astype(str)).replace("+", ",").split(","))
    add(
        "source_role_coverage",
        {"classic", "generated", "elite"}.issubset(source_roles) and len(source_policies_df) >= 3,
        f"sources={len(source_policies_df)} roles={sorted(source_roles)}",
    )
    add(
        "policy_specs_parse_roundtrip",
        not parser_validation_df.empty and bool(parser_validation_df["parseSuccess"].all()) and bool(parser_validation_df["roundtripSuccess"].all()),
        f"policy_specs={len(policy_table_df)} parser_rows={len(parser_validation_df)}",
    )
    add(
        "ablation_types_present",
        set(ABLATION_TYPES).intersection(set(policy_table_df["ablationType"])) >= {"suppress_compare_left", "suppress_compare_right", "remove_stochasticity", "remove_waiting"}
        and policy_table_df["ablationType"].nunique() >= 6,
        f"types={sorted(set(policy_table_df['ablationType']))}",
    )
    expected_runs = len(policy_table_df) * int(expected_task_count) * int(seed_count)
    add("run_rows_complete", len(run_df) == expected_runs, f"rows={len(run_df)} expected={expected_runs}")
    add(
        "run_value_counts_conserved",
        not run_df.empty and bool(run_df["valueCountsConserved"].all()),
        "all paired CPU runs preserved input value multisets",
    )
    final_sortedness = pd.to_numeric(run_df.get("finalSortednessPercent", pd.Series(dtype=float)), errors="coerce")
    final_error = pd.to_numeric(run_df.get("finalMonotonicityError", pd.Series(dtype=float)), errors="coerce")
    add(
        "run_metric_ranges",
        not run_df.empty and bool(final_sortedness.between(0.0, 100.0).all() and (final_error >= 0).all()),
        "final Sortedness in [0,100] and monotonicity error nonnegative",
    )
    add(
        "competence_vectors_match_runs",
        len(vector_df) == len(run_df) and vector_df["vectorId"].is_unique,
        f"vectors={len(vector_df)} runs={len(run_df)} uniqueVectors={vector_df['vectorId'].nunique() if 'vectorId' in vector_df else 0}",
    )
    expected_pairs = (len(policy_table_df) - int(policy_table_df["variantRole"].eq("original").sum())) * int(expected_task_count) * int(seed_count)
    add(
        "paired_same_seed_deltas",
        len(paired_delta_df) == expected_pairs and paired_delta_df[["sourcePolicyId", "ablationType", "taskId", "replicateIndex"]].notna().all().all(),
        f"pairs={len(paired_delta_df)} expected={expected_pairs}",
    )
    add(
        "effect_summary_nonempty",
        not effect_summary_df.empty and effect_summary_df["ablationType"].nunique() >= 1,
        f"rows={len(effect_summary_df)} ablationTypes={effect_summary_df['ablationType'].nunique() if not effect_summary_df.empty else 0}",
    )
    add("causal_report_exists", bool(report_exists), f"report_exists={report_exists}")
    add("figure_outputs", len(figure_paths) >= 2 and all(bool(path) for path in figure_paths), f"figures={len(figure_paths)}")
    add(
        "repo_unit_tests",
        bool(repo_test_payload.get("success")),
        f"returnCode={repo_test_payload.get('returnCode')} command={repo_test_payload.get('command')}",
    )
    add("no_s13_artifacts", not s13_dir_exists, f"s13_dir_exists={s13_dir_exists}")
    return pd.DataFrame(rows)
