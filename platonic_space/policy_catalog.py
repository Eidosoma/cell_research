from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .world_schema import SCHEMA_VERSION as WORLD_SCHEMA_VERSION


POLICY_SCHEMA_VERSION = "e07_s02_policy_abstract_catalog.v1"
POLICY_CATALOG_VERSION = "e07_s02_policy_catalog.v1"
POLICY_CLAIM_BOUNDARY = (
    "Computational local-policy catalog only; not direct evidence about biological morphogenesis, "
    "cognition, agency, sentience, clinical behavior, or living chimeras."
)
REQUIRED_POLICY_FIELDS = (
    "abstractPolicyId",
    "sourcePolicyId",
    "policyLabel",
    "sourceExperimentId",
    "sourceStepIds",
    "policyFamily",
    "policyKind",
    "representationType",
    "representationJson",
    "representationHash",
    "dslAvailable",
    "automatonAvailable",
    "decisionGraphAvailable",
    "neuralMetadataAvailable",
    "observationRequirementsJson",
    "memoryDepth",
    "signalingHorizon",
    "stochasticity",
    "lineageJson",
    "complexityScore",
    "actionSpaceJson",
    "linkedWorldIds",
    "sourceArtifactPaths",
    "roundTripStatus",
    "replayability",
    "caveatsOrBlockers",
    "claimBoundary",
)
NONEMPTY_POLICY_FIELDS = (
    "abstractPolicyId",
    "sourcePolicyId",
    "policyLabel",
    "sourceExperimentId",
    "sourceStepIds",
    "policyFamily",
    "policyKind",
    "representationType",
    "representationJson",
    "representationHash",
    "linkedWorldIds",
    "sourceArtifactPaths",
    "roundTripStatus",
    "replayability",
    "claimBoundary",
)
JSON_COLUMNS = (
    "sourceStepIds",
    "representationJson",
    "observationRequirementsJson",
    "lineageJson",
    "actionSpaceJson",
    "linkedWorldIds",
    "sourceArtifactPaths",
    "caveatsOrBlockers",
)
HARD_COVERAGE_REQUIREMENTS = {
    "classic_sorting_policy": "classic policies",
    "dsl_generated_policy": "generated DSL policies",
    "memory_augmented_policy": "memory policies",
    "signal_augmented_policy": "signaling policies",
    "morphology_policy": "2D/morphology policies",
    "governance_rule": "governance policies",
    "null_model": "null/control policies",
}


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(json_ready(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (list, tuple, dict, set, str, Path)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def json_hash(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def safe_json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_world_catalog(path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet") -> pd.DataFrame:
    worlds = read_table(path)
    if worlds.empty:
        raise FileNotFoundError(f"S01 world catalog not found or empty: {path}")
    return worlds


def _source_path(previous_artifacts_dir: Path, experiment_id: str, *parts: str) -> str:
    return str(previous_artifacts_dir.joinpath(experiment_id, *parts))


def _step_token(step: Any) -> str:
    token = str(step)
    if token.startswith("E") and "_S" in token:
        token = token.split("_")[-1]
    return token


def _linked_worlds(worlds: pd.DataFrame, experiment_id: str, steps: Sequence[str] | None = None) -> list[str]:
    subset = worlds[worlds["experimentId"].eq(experiment_id)]
    if not steps:
        return sorted(subset["worldId"].astype(str).unique())
    wanted = {_step_token(step) for step in steps}
    hits: list[str] = []
    for _, row in subset.iterrows():
        row_steps = {_step_token(item) for item in safe_json_loads(row.get("sourceStepIds"), []) or []}
        if row_steps & wanted:
            hits.append(str(row["worldId"]))
    if not hits:
        hits = list(subset["worldId"].astype(str).head(3))
    return sorted(set(hits))


def _truthy(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    out = _float_or_none(value)
    return int(out) if out is not None else None


def _action_counts_to_space(action_counts: Any) -> dict[str, Any]:
    parsed = safe_json_loads(action_counts, {})
    if isinstance(parsed, dict):
        return {"actions": sorted(str(key) for key in parsed), "actionCounts": parsed}
    return {"actions": [], "actionCounts": {}}


def _stochasticity_from_fields(*values: Any) -> str:
    for value in values:
        if _float_or_none(value) and _float_or_none(value) > 0:
            return "stochastic_policy_or_action"
        if _truthy(value):
            return "stochastic_policy_or_action"
    return "deterministic_policy_or_scheduler_only"


def _record(
    *,
    abstract_policy_id: str,
    source_policy_id: str,
    policy_label: str,
    source_experiment_id: str,
    source_step_ids: Sequence[str],
    policy_family: str,
    policy_kind: str,
    representation_type: str,
    representation: Mapping[str, Any],
    source_artifact_paths: Sequence[str],
    linked_world_ids: Sequence[str],
    dsl_available: bool = False,
    automaton_available: bool = False,
    decision_graph_available: bool = False,
    neural_metadata_available: bool = False,
    observation_requirements: Any = None,
    memory_depth: int | None = None,
    signaling_horizon: int | None = None,
    stochasticity: str = "unknown",
    lineage: Any = None,
    complexity_score: float | None = None,
    action_space: Any = None,
    round_trip_status: str = "not_applicable",
    replayability: str = "metadata",
    caveats_or_blockers: Sequence[str] | None = None,
    source_row_count: int = 1,
) -> dict[str, Any]:
    return {
        "abstractPolicyId": abstract_policy_id,
        "sourcePolicyId": source_policy_id,
        "policyLabel": policy_label,
        "sourceExperimentId": source_experiment_id,
        "sourceStepIds": list(source_step_ids),
        "policyFamily": policy_family,
        "policyKind": policy_kind,
        "representationType": representation_type,
        "representationJson": dict(representation),
        "representationHash": json_hash(representation),
        "dslAvailable": bool(dsl_available),
        "automatonAvailable": bool(automaton_available),
        "decisionGraphAvailable": bool(decision_graph_available),
        "neuralMetadataAvailable": bool(neural_metadata_available),
        "observationRequirementsJson": observation_requirements if observation_requirements is not None else [],
        "memoryDepth": memory_depth,
        "signalingHorizon": signaling_horizon,
        "stochasticity": stochasticity,
        "lineageJson": lineage if lineage is not None else {},
        "complexityScore": complexity_score,
        "actionSpaceJson": action_space if action_space is not None else {},
        "linkedWorldIds": sorted(set(str(item) for item in linked_world_ids)),
        "sourceArtifactPaths": sorted(set(str(item) for item in source_artifact_paths)),
        "sourceRowCount": int(source_row_count),
        "roundTripStatus": round_trip_status,
        "replayability": replayability,
        "caveatsOrBlockers": list(caveats_or_blockers or []),
        "claimBoundary": POLICY_CLAIM_BOUNDARY,
        "policySchemaVersion": POLICY_SCHEMA_VERSION,
        "policyCatalogVersion": POLICY_CATALOG_VERSION,
        "worldSchemaVersion": WORLD_SCHEMA_VERSION,
    }


def _add_e01(records: list[dict[str, Any]], previous: Path, worlds: pd.DataFrame) -> None:
    path = _source_path(previous, "E01", "research_steps", "S03", "condition_matrix.csv")
    df = read_table(path)
    if df.empty:
        return
    for _, row in df[["implementation", "algorithms"]].drop_duplicates().iterrows():
        algorithm = str(row["algorithms"])
        implementation = str(row["implementation"])
        if ";" in algorithm or "," in algorithm:
            continue
        policy_id = f"{implementation}_{algorithm}"
        records.append(
            _record(
                abstract_policy_id=f"E01::{policy_id}",
                source_policy_id=policy_id,
                policy_label=f"E01 {implementation} {algorithm}",
                source_experiment_id="E01",
                source_step_ids=["S03", "S04"],
                policy_family=implementation,
                policy_kind="classic_sorting_policy",
                representation_type="source_code_reference",
                representation={"algorithm": algorithm, "implementation": implementation, "source": "E01 condition matrix"},
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E01", ["S03", "S04"]),
                observation_requirements=["own_value", "neighbor_values"],
                memory_depth=1 if algorithm == "selection" else 0,
                signaling_horizon=0,
                stochasticity="deterministic_policy_or_scheduler_only",
                lineage={"source": "original_paper_code"},
                complexity_score={"bubble": 2, "insertion": 3, "selection": 5}.get(algorithm, 1),
                action_space={"actions": ["wait", "swap_left", "swap_right", "target_swap_if_selection"]},
                replayability="source_code_reference",
            )
        )


def _add_e02(records: list[dict[str, Any]], previous: Path, worlds: pd.DataFrame) -> None:
    sources = [
        ("S04", "label_shuffle", _source_path(previous, "E02", "results", "e02_label_shuffle_mixture_summary.csv"), ["trajectory_label_shuffle"]),
        ("S05", "dummy_algotype", _source_path(previous, "E02", "results", "e02_dummy_algotypes_summary.csv"), ["dummy_label_same_code"]),
        ("S06", "speed_matched", _source_path(previous, "E02", "results", "e02_speed_matched_summary.csv"), ["speed_matched_algotype"]),
    ]
    for step, family, path, names in sources:
        if not Path(path).exists():
            continue
        for name in names:
            records.append(
                _record(
                    abstract_policy_id=f"E02::null::{name}",
                    source_policy_id=name,
                    policy_label=f"E02 {name} control",
                    source_experiment_id="E02",
                    source_step_ids=[step],
                    policy_family=family,
                    policy_kind="null_model",
                    representation_type="null_control_metadata",
                    representation={"nullControl": name, "family": family, "sourceStep": step},
                    source_artifact_paths=[path],
                    linked_world_ids=_linked_worlds(worlds, "E02", [step]),
                    observation_requirements=[],
                    memory_depth=0,
                    signaling_horizon=0,
                    stochasticity="stochastic_null_resampling_or_scheduler",
                    action_space={"actions": ["matched_control_operation"]},
                    replayability="derived_control",
                    caveats_or_blockers=["Null/control policy may be a trajectory or label operation rather than an executable local-cell policy."],
                )
            )
    for path, step, column in [
        (_source_path(previous, "E02", "results", "e02_local_move_null_summary.csv"), "S07", "nullModel"),
        (_source_path(previous, "E02", "results", "e02_dg_null_summary.csv"), "S08", "nullModel"),
    ]:
        df = read_table(path)
        if df.empty or column not in df.columns:
            continue
        for null_model in sorted(str(item) for item in df[column].dropna().unique()):
            records.append(
                _record(
                    abstract_policy_id=f"E02::null::{null_model}",
                    source_policy_id=null_model,
                    policy_label=f"E02 {null_model} null",
                    source_experiment_id="E02",
                    source_step_ids=[step],
                    policy_family="null_model",
                    policy_kind="null_model",
                    representation_type="null_control_metadata",
                    representation={"nullModel": null_model, "sourceStep": step},
                    source_artifact_paths=[path],
                    linked_world_ids=_linked_worlds(worlds, "E02", [step]),
                    observation_requirements=[],
                    memory_depth=0,
                    signaling_horizon=0,
                    stochasticity="stochastic_null_resampling_or_scheduler",
                    action_space={"actions": ["random_or_resampled_local_move"]},
                    replayability="derived_control",
                    caveats_or_blockers=["Null/control policy is bounded proxy evidence, not a biological or causal model."],
                )
            )


def _add_e03(records: list[dict[str, Any]], previous: Path, worlds: pd.DataFrame) -> None:
    path = _source_path(previous, "E03", "research_steps", "S01", "policy_catalog.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["policy_id"])
        records.append(
            _record(
                abstract_policy_id=f"E03::classic::{policy_id}",
                source_policy_id=policy_id,
                policy_label=policy_id,
                source_experiment_id="E03",
                source_step_ids=["S01"],
                policy_family=str(row.get("family", "classic")),
                policy_kind="classic_sorting_policy" if row.get("family") == "classic" else "null_model",
                representation_type="policy_spec_metadata",
                representation=row.to_dict(),
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E03", ["S01"]),
                observation_requirements=["own_value", "neighbor_values"],
                memory_depth=0,
                signaling_horizon=0,
                stochasticity="stochastic_policy_or_action" if "random" in policy_id else "deterministic_policy_or_scheduler_only",
                complexity_score=1,
                action_space={"actions": ["wait", "swap_left", "swap_right", "swap_target_if_selection"]},
                round_trip_status="metadata_only",
                replayability="policy_event_simulator",
            )
        )

    path = _source_path(previous, "E03", "research_steps", "S03", "classic_policy_templates.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["policyId"])
        program = safe_json_loads(row.get("dslProgramJson"), {})
        records.append(
            _record(
                abstract_policy_id=f"E03::template::{policy_id}",
                source_policy_id=policy_id,
                policy_label=str(row.get("templateId", policy_id)),
                source_experiment_id="E03",
                source_step_ids=["S03"],
                policy_family="classic_template",
                policy_kind="classic_sorting_policy",
                representation_type="dsl_template",
                representation={"dslProgram": program, "parameterSpace": safe_json_loads(row.get("parameterSpaceJson"), {}), "exactBehavior": _truthy(row.get("exactBehavior"))},
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E03", ["S03"]),
                dsl_available=True,
                observation_requirements=safe_json_loads(row.get("minimalDslExtensionJson"), []),
                memory_depth=1 if "selection" in policy_id else 0,
                signaling_horizon=0,
                stochasticity="deterministic_policy_or_scheduler_only",
                lineage={"linkedS01PolicyId": row.get("linkedS01PolicyId"), "templateId": row.get("templateId")},
                complexity_score=_float_or_none(row.get("ruleCount")),
                action_space=_action_counts_to_space({"template": 1}),
                round_trip_status="json_roundtrip_ok" if program else "metadata_only",
                replayability="approximate_dsl",
                caveats_or_blockers=safe_json_loads(row.get("caveatsJson"), []),
            )
        )

    path = _source_path(previous, "E03", "research_steps", "S05", "policy_corpus.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["policyId"])
        program = safe_json_loads(row.get("dslProgramJson"), {})
        records.append(
            _record(
                abstract_policy_id=f"E03::dsl::{policy_id}",
                source_policy_id=policy_id,
                policy_label=str(row.get("description", policy_id)),
                source_experiment_id="E03",
                source_step_ids=["S05"],
                policy_family=str(row.get("family", "generated")),
                policy_kind="dsl_generated_policy",
                representation_type="dsl_program",
                representation={"dslProgram": program, "dslVersion": row.get("dslVersion"), "dslRelativePath": row.get("dslRelativePath")},
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E03", ["S05", "S06", "S07", "S08", "S09"]),
                dsl_available=True,
                observation_requirements=safe_json_loads(row.get("observationRequirementsJson"), []),
                memory_depth=_int_or_none(row.get("stateKeyCount")) or 0,
                signaling_horizon=_int_or_none(row.get("signalCount")) or 0,
                stochasticity=_stochasticity_from_fields(row.get("stochasticActionCount")),
                lineage={
                    "lineageId": row.get("lineageId"),
                    "lineageDepth": _int_or_none(row.get("lineageDepth")),
                    "parentPolicyIds": safe_json_loads(row.get("parentPolicyIdsJson"), []),
                    "mutationOperators": safe_json_loads(row.get("mutationOperatorsJson"), []),
                    "generationMethod": row.get("generationMethod"),
                },
                complexity_score=_float_or_none(row.get("complexityScore")),
                action_space=_action_counts_to_space(row.get("actionCountsJson")),
                round_trip_status="json_roundtrip_ok" if program else "failed_missing_dsl",
                replayability="executable_dsl",
            )
        )

    path = _source_path(previous, "E03", "research_steps", "S12", "ablation_policy_specs.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["ablationPolicyId"])
        program = safe_json_loads(row.get("dslProgramJson"), {})
        records.append(
            _record(
                abstract_policy_id=f"E03::ablation::{policy_id}",
                source_policy_id=policy_id,
                policy_label=f"{row.get('sourcePolicyId')} {row.get('ablationType')}",
                source_experiment_id="E03",
                source_step_ids=["S12"],
                policy_family="feature_ablation",
                policy_kind="dsl_generated_policy",
                representation_type="dsl_program",
                representation={"dslProgram": program, "ablationType": row.get("ablationType"), "changeSummary": safe_json_loads(row.get("changeSummaryJson"), {})},
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E03", ["S12"]),
                dsl_available=True,
                observation_requirements=[],
                memory_depth=0,
                signaling_horizon=0,
                stochasticity="deterministic_policy_or_scheduler_only",
                lineage={"sourcePolicyId": row.get("sourcePolicyId"), "variantRole": row.get("variantRole")},
                complexity_score=None,
                action_space={"actions": ["inferred_from_ablation_dsl"]},
                round_trip_status="json_roundtrip_ok" if _truthy(row.get("roundtripSuccess")) else "failed_source_roundtrip",
                replayability="executable_dsl",
                caveats_or_blockers=[] if _truthy(row.get("roundtripSuccess")) else ["Source ablation row failed roundtrip validation."],
            )
        )


def _add_e04(records: list[dict[str, Any]], previous: Path, worlds: pd.DataFrame) -> None:
    for step, path_tail, kind, representation_type in [
        ("S01", ("research_steps", "S01", "memory_variant_catalog.csv"), "memory_augmented_policy", "policy_spec_json"),
        ("S02", ("research_steps", "S02", "signal_variant_catalog.csv"), "signal_augmented_policy", "policy_spec_json"),
    ]:
        path = _source_path(previous, "E04", *path_tail)
        df = read_table(path)
        for _, row in df.iterrows():
            policy_id = str(row["policyId"])
            spec = safe_json_loads(row.get("policySpecJson"), row.to_dict())
            records.append(
                _record(
                    abstract_policy_id=f"E04::{step.lower()}::{policy_id}",
                    source_policy_id=policy_id,
                    policy_label=policy_id,
                    source_experiment_id="E04",
                    source_step_ids=[step],
                    policy_family=str(row.get("family", kind)),
                    policy_kind=kind,
                    representation_type=representation_type,
                    representation={"policySpec": spec},
                    source_artifact_paths=[path],
                    linked_world_ids=_linked_worlds(worlds, "E04", [step]),
                    observation_requirements=["own_value", "neighbor_values", "memory_or_signal_state"],
                    memory_depth=_int_or_none(row.get("neighborHistory")) if step == "S01" else 0,
                    signaling_horizon=_int_or_none(row.get("signalRange")) if step == "S02" else 0,
                    stochasticity="deterministic_policy_or_scheduler_only",
                    lineage={"basePolicyId": row.get("basePolicyId"), "baseFamily": row.get("baseFamily")},
                    complexity_score=None,
                    action_space={"actions": ["wait", "swap_left", "swap_right", "memory_or_signal_update"]},
                    round_trip_status="json_roundtrip_ok" if isinstance(spec, dict) else "failed_missing_spec",
                    replayability="policy_spec_replayable",
                )
            )

    path = _source_path(previous, "E04", "research_steps", "S06", "learning_variant_catalog.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["policyId"])
        records.append(
            _record(
                abstract_policy_id=f"E04::learning::{policy_id}",
                source_policy_id=policy_id,
                policy_label=policy_id,
                source_experiment_id="E04",
                source_step_ids=["S06"],
                policy_family=str(row.get("policyFamily", "local_learning_repair")),
                policy_kind="adaptive_learning_policy",
                representation_type="learning_parameter_metadata",
                representation=row.to_dict(),
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E04", ["S06"]),
                observation_requirements=["local_reward_terms", "neighbor_values"],
                memory_depth=1,
                signaling_horizon=0,
                stochasticity="stochastic_policy_or_action",
                lineage={"variant": row.get("variant"), "basePolicyId": "classic_bubble"},
                complexity_score=_float_or_none(row.get("learningRate")),
                action_space=safe_json_loads(row.get("initialActionProbabilitiesJson"), {}),
                round_trip_status="metadata_only",
                replayability="parameter_metadata",
            )
        )

    path = _source_path(previous, "E04", "results", "e04_evolved_repair_policies.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["candidateId"])
        spec = safe_json_loads(row.get("policySpecJson"), {})
        records.append(
            _record(
                abstract_policy_id=f"E04::evolved::{policy_id}",
                source_policy_id=policy_id,
                policy_label=f"E04 evolved rank {row.get('rank')}",
                source_experiment_id="E04",
                source_step_ids=["S08"],
                policy_family="evolved_repair",
                policy_kind="adaptive_learning_policy",
                representation_type="evolved_parameter_vector",
                representation={
                    "policySpec": spec,
                    "genome": safe_json_loads(row.get("genomeJson"), {}),
                    "learningConfig": safe_json_loads(row.get("learningConfigJson"), {}),
                    "memoryConfig": safe_json_loads(row.get("memoryConfigJson"), {}),
                    "signalConfig": safe_json_loads(row.get("signalConfigJson"), {}),
                    "repairConfig": safe_json_loads(row.get("repairConfigJson"), {}),
                },
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E04", ["S08"]),
                observation_requirements=["local_reward_terms", "memory_state", "signal_state"],
                memory_depth=_int_or_none(safe_json_loads(row.get("memoryConfigJson"), {}).get("neighborHistory")),
                signaling_horizon=_int_or_none(safe_json_loads(row.get("signalConfigJson"), {}).get("signalRange")),
                stochasticity="stochastic_policy_or_action",
                lineage={"genomeId": row.get("genomeId"), "generation": row.get("generation"), "rank": row.get("rank")},
                complexity_score=_float_or_none(row.get("proxyFitness")),
                action_space={"actions": ["wait", "swap_left", "swap_right", "repair_attempt", "signal_emit"]},
                round_trip_status="json_roundtrip_ok" if isinstance(spec, dict) and spec else "failed_missing_spec",
                replayability="parameter_metadata",
                caveats_or_blockers=["Evolved policy represented by parameter vectors and serialized policy spec; exact search history is summarized."],
            )
        )

    path = _source_path(previous, "E04", "research_steps", "S15", "handoff_policy_catalog.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["policyId"])
        spec = safe_json_loads(row.get("basePolicySpec"), {})
        records.append(
            _record(
                abstract_policy_id=f"E04::handoff::{policy_id}",
                source_policy_id=policy_id,
                policy_label=str(row.get("policyLabel", policy_id)),
                source_experiment_id="E04",
                source_step_ids=[str(row.get("sourceStep", "S15"))],
                policy_family=str(row.get("policyGroup", "handoff")),
                policy_kind="adaptive_learning_policy",
                representation_type="handoff_policy_spec",
                representation={
                    "basePolicySpec": spec,
                    "learningConfig": safe_json_loads(row.get("learningConfig"), {}),
                    "memoryConfig": safe_json_loads(row.get("memoryConfig"), {}),
                    "signalConfig": safe_json_loads(row.get("signalConfig"), {}),
                    "repairConfig": safe_json_loads(row.get("repairConfig"), {}),
                },
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E04", ["S15", "S06", "S08"]),
                observation_requirements=["memory_state", "signal_state", "local_reward_terms"],
                memory_depth=_int_or_none(safe_json_loads(row.get("memoryConfig"), {}).get("neighborHistory")),
                signaling_horizon=_int_or_none(safe_json_loads(row.get("signalConfig"), {}).get("signalRange")),
                stochasticity="stochastic_policy_or_action",
                lineage={"sourceCandidateId": row.get("sourceCandidateId"), "familyKind": row.get("familyKind")},
                complexity_score=None,
                action_space={"actions": ["wait", "swap_left", "swap_right", "repair_attempt", "signal_emit"]},
                round_trip_status="json_roundtrip_ok" if isinstance(spec, dict) and spec else "metadata_only",
                replayability="handoff_metadata",
                caveats_or_blockers=[] if _truthy(row.get("replayable")) else [str(row.get("replayBlocker"))],
            )
        )


def _add_e05(records: list[dict[str, Any]], previous: Path, worlds: pd.DataFrame) -> None:
    paths = [
        _source_path(previous, "E05", "results", "e05_morphology_benchmarks.csv"),
        _source_path(previous, "E05", "results", "e05_local_global_control.csv"),
        _source_path(previous, "E05", "results", "e05_morphospace_trajectories.csv"),
    ]
    frames = [read_table(path).assign(_source_path=path) for path in paths if Path(path).exists()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "policy_id" not in df.columns:
        return
    group_cols = ["policy_id"]
    for policy_id, sub in df.groupby(group_cols, dropna=True):
        source_policy_id = str(policy_id[0] if isinstance(policy_id, tuple) else policy_id)
        first = sub.iloc[0]
        families = sorted(str(item) for item in sub.get("policy_family", pd.Series(dtype=object)).dropna().unique())
        source_steps = sorted(str(item) for item in sub.get("source_step_id", pd.Series(dtype=object)).dropna().unique())
        uses_target = bool(sub.get("uses_target_map", pd.Series(dtype=object)).map(_truthy).any())
        uses_gradient = bool(sub.get("uses_global_gradient", pd.Series(dtype=object)).map(_truthy).any())
        uses_organizer = bool(sub.get("uses_organizer", pd.Series(dtype=object)).map(_truthy).any())
        global_baseline = bool(sub.get("is_global_information_baseline", pd.Series(dtype=object)).map(_truthy).any())
        random_null = "null" in source_policy_id or any("null" in family for family in families)
        policy_kind = "null_model" if random_null else "global_baseline_policy" if global_baseline else "morphology_policy"
        records.append(
            _record(
                abstract_policy_id=f"E05::morphology::{source_policy_id}",
                source_policy_id=source_policy_id,
                policy_label=source_policy_id,
                source_experiment_id="E05",
                source_step_ids=source_steps or ["S15"],
                policy_family=";".join(families) if families else "morphology_policy",
                policy_kind=policy_kind,
                representation_type="morphology_policy_reference",
                representation={
                    "policyId": source_policy_id,
                    "policyFamilies": families,
                    "informationScopes": sorted(str(item) for item in sub.get("information_scope", pd.Series(dtype=object)).dropna().unique()),
                    "controlClasses": sorted(str(item) for item in sub.get("control_class", pd.Series(dtype=object)).dropna().unique()),
                    "usesTargetMap": uses_target,
                    "usesGlobalGradient": uses_gradient,
                    "usesOrganizer": uses_organizer,
                    "isGlobalInformationBaseline": global_baseline,
                },
                source_artifact_paths=sorted(str(item) for item in sub["_source_path"].dropna().unique()),
                linked_world_ids=_linked_worlds(worlds, "E05", source_steps or ["S15"]),
                observation_requirements=sorted(str(item) for item in sub.get("information_scope", pd.Series(dtype=object)).dropna().unique()),
                memory_depth=1 if "memory" in source_policy_id else 0,
                signaling_horizon=1 if "signal" in source_policy_id else 0,
                stochasticity="stochastic_policy_or_action" if random_null else "deterministic_policy_or_scheduler_only",
                lineage={"sourceFamilies": families, "sourceStepIds": source_steps},
                complexity_score=None,
                action_space={"actions": ["grid_or_graph_local_action", "repair_action_if_policy_family_allows"]},
                round_trip_status="metadata_only",
                replayability="source_reference",
                caveats_or_blockers=["Global-information baseline, not local-only policy."] if global_baseline else [],
                source_row_count=len(sub),
            )
        )


def _add_e06(records: list[dict[str, Any]], previous: Path, worlds: pd.DataFrame) -> None:
    path = _source_path(previous, "E06", "research_steps", "S01", "algotype_panel.csv")
    df = read_table(path)
    for _, row in df.iterrows():
        policy_id = str(row["panelPolicyId"])
        spec = safe_json_loads(row.get("policySpecJson"), {})
        records.append(
            _record(
                abstract_policy_id=f"E06::panel::{policy_id}",
                source_policy_id=policy_id,
                policy_label=str(row.get("displayName", policy_id)),
                source_experiment_id="E06",
                source_step_ids=["S01"],
                policy_family=str(row.get("policyFamily", "panel_policy")),
                policy_kind="chimeric_panel_policy",
                representation_type="policy_spec_json",
                representation={"policySpec": spec, "constructorPayload": safe_json_loads(row.get("constructorPayloadJson"), {})},
                source_artifact_paths=[path],
                linked_world_ids=_linked_worlds(worlds, "E06", ["S01", "S02", "S03", "S04", "S05"]),
                dsl_available=str(row.get("interfaceFamily", "")).startswith("dsl"),
                observation_requirements=["own_value", "neighbor_values", "neighbor_policy_id"],
                memory_depth=1 if "memory" in str(row.get("policyFamily", "")) else 0,
                signaling_horizon=1 if "signal" in str(row.get("policyFamily", "")) else 0,
                stochasticity="stochastic_policy_or_action" if "random" in policy_id else "deterministic_policy_or_scheduler_only",
                lineage={"sourceExperiment": row.get("sourceExperiment"), "sourcePolicyId": row.get("sourcePolicyId"), "panelGroup": row.get("panelGroup")},
                complexity_score=None,
                action_space={"actions": ["wait", "swap_left", "swap_right", "policy_specific"]},
                round_trip_status="json_roundtrip_ok" if _truthy(row.get("serializationRoundtripSuccess")) else "failed_source_roundtrip",
                replayability="panel_policy_spec",
                caveats_or_blockers=[] if _truthy(row.get("readyForS02Mixing")) else [str(row.get("quarantineOrExclusionReason"))],
            )
        )

    path = _source_path(previous, "E06", "results", "e06_governance_mechanisms.csv")
    df = read_table(path)
    if not df.empty:
        cols = [
            "governanceVariant",
            "governanceFamily",
            "informationAccess",
            "localInformationOnly",
            "usesGlobalState",
            "usesTargetMap",
            "usesOrganizer",
            "broadControlLike",
            "governanceConfigHash",
            "governanceConfigJson",
        ]
        for _, row in df[[c for c in cols if c in df.columns]].drop_duplicates().iterrows():
            variant = str(row["governanceVariant"])
            config = safe_json_loads(row.get("governanceConfigJson"), {})
            records.append(
                _record(
                    abstract_policy_id=f"E06::governance::{variant}",
                    source_policy_id=variant,
                    policy_label=f"E06 governance {variant}",
                    source_experiment_id="E06",
                    source_step_ids=["S09"],
                    policy_family=str(row.get("governanceFamily", "governance")),
                    policy_kind="governance_rule",
                    representation_type="governance_config",
                    representation={"governanceConfig": config, "governanceConfigHash": row.get("governanceConfigHash")},
                    source_artifact_paths=[path],
                    linked_world_ids=_linked_worlds(worlds, "E06", ["S09"]),
                    observation_requirements=[row.get("informationAccess")],
                    memory_depth=0,
                    signaling_horizon=None,
                    stochasticity="deterministic_policy_or_scheduler_only",
                    lineage={"governanceFamily": row.get("governanceFamily")},
                    complexity_score=None,
                    action_space={"actions": ["allow_swap", "redirect_swap", "veto_swap", "inject_swap"]},
                    round_trip_status="json_roundtrip_ok" if isinstance(config, dict) and config else "metadata_only",
                    replayability="governance_config",
                    caveats_or_blockers=["Broad-control/global-information upper-bound controller."] if _truthy(row.get("broadControlLike")) else [],
                )
            )

    path = _source_path(previous, "E06", "results", "e06_intervention_search.csv")
    df = read_table(path)
    if not df.empty and "interventionRecipeId" in df.columns:
        cols = [
            "interventionRecipeId",
            "interventionType",
            "interventionFamily",
            "governanceVariant",
            "historyProtocol",
            "claimBoundary",
        ]
        for _, row in df[[c for c in cols if c in df.columns]].drop_duplicates().iterrows():
            recipe = str(row["interventionRecipeId"])
            governance = str(row.get("governanceVariant", "none"))
            records.append(
                _record(
                    abstract_policy_id=f"E06::intervention::{recipe}::{governance}",
                    source_policy_id=f"{recipe}::{governance}",
                    policy_label=f"{recipe} under {governance}",
                    source_experiment_id="E06",
                    source_step_ids=["S14"],
                    policy_family=str(row.get("interventionFamily", "intervention")),
                    policy_kind="governance_rule",
                    representation_type="intervention_recipe_metadata",
                    representation=row.to_dict(),
                    source_artifact_paths=[path],
                    linked_world_ids=_linked_worlds(worlds, "E06", ["S14"]),
                    observation_requirements=["history_protocol", "governance_variant"],
                    memory_depth=0,
                    signaling_horizon=None,
                    stochasticity="deterministic_policy_or_scheduler_only",
                    lineage={"historyProtocol": row.get("historyProtocol"), "governanceVariant": governance},
                    complexity_score=None,
                    action_space={"actions": ["timing_shift", "transient_rule_switch", "state_reset_or_carryover"]},
                    round_trip_status="metadata_only",
                    replayability="intervention_metadata",
                    caveats_or_blockers=["Intervention recipe is a computational control protocol, not a standalone local cell policy."],
                )
            )


def build_policy_catalog(
    previous_artifacts_dir: str | Path = "/previous-artifacts",
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
) -> list[dict[str, Any]]:
    previous = Path(previous_artifacts_dir)
    worlds = load_world_catalog(world_catalog_path)
    records: list[dict[str, Any]] = []
    _add_e01(records, previous, worlds)
    _add_e02(records, previous, worlds)
    _add_e03(records, previous, worlds)
    _add_e04(records, previous, worlds)
    _add_e05(records, previous, worlds)
    _add_e06(records, previous, worlds)
    return records


def catalog_to_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for column in JSON_COLUMNS:
            row[column] = compact_json(row.get(column, []))
        rows.append(row)
    df = pd.DataFrame(rows)
    columns = list(REQUIRED_POLICY_FIELDS) + [
        "sourceRowCount",
        "policySchemaVersion",
        "policyCatalogVersion",
        "worldSchemaVersion",
    ]
    columns = [column for column in columns if column in df.columns]
    return df[columns].sort_values(["sourceExperimentId", "abstractPolicyId"]).reset_index(drop=True)


def validate_policy_catalog(
    records: Sequence[Mapping[str, Any]],
    world_catalog_path: str | Path = "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
) -> pd.DataFrame:
    worlds = load_world_catalog(world_catalog_path)
    valid_world_ids = set(worlds["worldId"].astype(str))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        pid = str(record.get("abstractPolicyId", "UNKNOWN"))
        missing = [field for field in REQUIRED_POLICY_FIELDS if field not in record]
        missing.extend(field for field in NONEMPTY_POLICY_FIELDS if record.get(field) in (None, "", []))
        rows.append({"abstractPolicyId": pid, "checkId": "required_fields", "success": not missing, "severity": "error" if missing else "info", "detail": "missing=" + ",".join(missing) if missing else "all required fields populated"})

        duplicate = pid in seen
        seen.add(pid)
        rows.append({"abstractPolicyId": pid, "checkId": "unique_abstract_policy_id", "success": not duplicate, "severity": "error" if duplicate else "info", "detail": "duplicate" if duplicate else "unique"})

        paths = [Path(path) for path in record.get("sourceArtifactPaths", [])]
        missing_paths = [str(path) for path in paths if not path.exists()]
        rows.append({"abstractPolicyId": pid, "checkId": "source_artifacts_exist", "success": bool(paths) and not missing_paths, "severity": "error" if missing_paths or not paths else "info", "detail": f"{len(paths)} source artifacts resolved" if not missing_paths else compact_json(missing_paths)})

        linked = set(str(item) for item in record.get("linkedWorldIds", []))
        unresolved = sorted(linked - valid_world_ids)
        rows.append({"abstractPolicyId": pid, "checkId": "linked_world_ids_resolve", "success": bool(linked) and not unresolved, "severity": "error" if unresolved or not linked else "info", "detail": f"{len(linked)} linked worlds resolved" if not unresolved else compact_json(unresolved)})

        rep = record.get("representationJson")
        rep_ok = isinstance(rep, Mapping) and bool(rep)
        rows.append({"abstractPolicyId": pid, "checkId": "representation_json_parseable", "success": rep_ok, "severity": "error" if not rep_ok else "info", "detail": "representation object available" if rep_ok else "missing or non-object representation"})

        rt = str(record.get("roundTripStatus", ""))
        rt_ok = not rt.startswith("failed")
        rows.append({"abstractPolicyId": pid, "checkId": "roundtrip_or_metadata_status", "success": rt_ok, "severity": "warning" if not rt_ok else "info", "detail": rt})

    kind_counts = Counter(str(record.get("policyKind")) for record in records)
    for kind, label in HARD_COVERAGE_REQUIREMENTS.items():
        count = kind_counts.get(kind, 0)
        rows.append({"abstractPolicyId": "__coverage__", "checkId": f"coverage_{kind}", "success": count > 0, "severity": "error" if count == 0 else "info", "detail": f"{label}: {count} records"})

    experiment_counts = Counter(str(record.get("sourceExperimentId")) for record in records)
    for experiment_id in [f"E{i:02d}" for i in range(1, 7)]:
        count = experiment_counts.get(experiment_id, 0)
        rows.append({"abstractPolicyId": "__coverage__", "checkId": f"coverage_{experiment_id}", "success": count > 0, "severity": "error" if count == 0 else "info", "detail": f"{experiment_id}: {count} records"})

    return pd.DataFrame(rows).sort_values(["abstractPolicyId", "checkId"]).reset_index(drop=True)


def coverage_summary(records: Sequence[Mapping[str, Any]], validation: pd.DataFrame) -> dict[str, Any]:
    hard_failures = validation[(validation["severity"].eq("error")) & (~validation["success"])]
    warnings = validation[(validation["severity"].eq("warning")) & (~validation["success"])]
    return {
        "allHardChecksPassed": hard_failures.empty,
        "hardFailureCount": int(len(hard_failures)),
        "warningFailureCount": int(len(warnings)),
        "validationCheckCount": int(len(validation)),
        "policyRecordCount": int(len(records)),
        "sourceExperimentCounts": dict(Counter(str(record["sourceExperimentId"]) for record in records)),
        "policyKindCounts": dict(Counter(str(record["policyKind"]) for record in records)),
        "representationTypeCounts": dict(Counter(str(record["representationType"]) for record in records)),
        "roundTripStatusCounts": dict(Counter(str(record["roundTripStatus"]) for record in records)),
    }
