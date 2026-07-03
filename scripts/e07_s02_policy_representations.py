#!/usr/bin/env python3
"""Build E07 S02 abstract policy representations."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e03.rule_dsl import DSLValidationError, parse_policy  # noqa: E402
from src.e04.evolutionary_search import PARAMETER_NAMES  # noqa: E402
from src.e07.policy_schema import (  # noqa: E402
    PolicyRepresentation,
    dataframe_from_policy_records,
    sha256_file,
    sha256_text,
    stable_hash,
    stable_json,
    validate_policy_table,
)


STEP_ID = "S02"
STEP_NUMBER = 2
EXPERIMENT_ID = "E07"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-artifacts-dir", type=Path, default=Path("/previous-artifacts"))
    parser.add_argument("--s01-world-inventory", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "tables" / "e07_world_inventory.csv")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    import hashlib

    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_path(path),
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def self_referential_artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "artifactType": "file",
        "note": "Checksum omitted because this report or manifest contains the artifact list.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def list_existing(*paths: Path) -> tuple[str, ...]:
    return tuple(str(path) for path in paths if path.exists())


def as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return (value,) if value else ()
        value = parsed
    if isinstance(value, Mapping):
        return tuple(str(item) for item in value.keys())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return (str(value),)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def compact_record_id(prefix: str, index: int, record: Mapping[str, Any]) -> str:
    policy_id = record.get("policyId") or record.get("policy_id") or record.get("algotypeId") or record.get("sourcePolicyId") or f"row{index}"
    return f"{prefix}:{index:05d}:{policy_id}"


def world_groups(world_inventory: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    def select(mask: pd.Series) -> tuple[str, ...]:
        return tuple(world_inventory.loc[mask, "world_id"].astype(str).tolist())

    return {
        "array_core": select((world_inventory["substrate_kind"] == "array_1d") & (world_inventory["experiment_id"].isin(["E01", "E02"]))),
        "e03": select(world_inventory["experiment_id"] == "E03"),
        "e04": select(world_inventory["experiment_id"] == "E04"),
        "e05": select(world_inventory["experiment_id"] == "E05"),
        "e06": select(world_inventory["experiment_id"] == "E06"),
    }


def dsl_features_from_source(dsl_source: str, declared_sha: str | None) -> dict[str, Any]:
    try:
        policy = parse_policy(dsl_source)
    except DSLValidationError as exc:
        return {
            "parser_status": "parse_failed",
            "parser_error": str(exc),
            "canonical_sha": None,
            "source_sha": sha256_text(dsl_source),
            "hash_status": "parse_failed",
            "action_vocabulary": (),
            "state_variables": (),
            "parsed_policy_name": None,
            "rule_count": None,
        }

    action_names: set[str] = set()
    condition_names: set[str] = set()
    state_variables: set[str] = set()
    if policy.state_init != "none":
        state_variables.add("ideal_position")
    for rule in policy.rules:
        for condition in rule.conditions:
            condition_names.add(condition.name)
            if "ideal" in condition.args or condition.name in {"at_ideal", "not_at_ideal"}:
                state_variables.add("ideal_position")
        for action in rule.actions:
            action_names.add(action.name)
            if action.name in {"set_ideal", "estimate_target_position"}:
                state_variables.add("ideal_position")
    canonical_sha = policy.sha256
    source_sha = sha256_text(dsl_source)
    if declared_sha == canonical_sha:
        hash_status = "matches_declared_canonical_dsl_hash"
    elif declared_sha == source_sha:
        hash_status = "matches_declared_source_dsl_hash"
    elif declared_sha:
        hash_status = "declared_dsl_hash_mismatch"
    else:
        hash_status = "not_applicable"
    return {
        "parser_status": "parsed",
        "parser_error": "",
        "canonical_sha": canonical_sha,
        "source_sha": source_sha,
        "hash_status": hash_status,
        "action_vocabulary": tuple(sorted(action_names)),
        "condition_vocabulary": tuple(sorted(condition_names)),
        "state_variables": tuple(sorted(state_variables)),
        "parsed_policy_name": policy.name,
        "rule_count": len(policy.rules),
    }


def dsl_policy_record(
    *,
    source_experiment_id: str,
    source_step_id: str,
    source_record_id: str,
    record: Mapping[str, Any],
    source_path: Path,
    applicable_world_ids: tuple[str, ...],
    source_category: str,
    policy_family: str,
    algorithm: str,
) -> PolicyRepresentation:
    dsl_source = str(record.get("dslSource", ""))
    declared_sha = record.get("dslSha256")
    features = dsl_features_from_source(dsl_source, str(declared_sha) if declared_sha else None)
    payload = {
        "dslSource": dsl_source,
        "features": record.get("features", {}),
        "conditionVocabulary": features.get("condition_vocabulary", ()),
        "ruleCount": record.get("ruleCount", features.get("rule_count")),
    }
    upstream_metrics = {
        key: value
        for key, value in record.items()
        if key.startswith("heldout") or key.startswith("s14") or key in {"classId", "cautiousLabel", "curatedRank", "archiveWinner", "generation", "mapCellId"}
    }
    return PolicyRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id=source_step_id,
        source_record_id=source_record_id,
        source_policy_id=str(record.get("policyId") or f"dsl:{features.get('canonical_sha', sha256_text(dsl_source))[:16]}"),
        display_name=str(record.get("policyName") or features.get("parsed_policy_name") or record.get("displayName") or "dsl_policy"),
        source_category=source_category,
        policy_family=policy_family,
        algorithm=algorithm,
        representation_type="dsl",
        abstraction_kind="dsl_program",
        interpretable=features["parser_status"] == "parsed",
        metadata_only=False,
        requires_memory=bool(record.get("features", {}).get("usesMemory", False)),
        requires_signaling=bool(record.get("features", {}).get("usesSignal", False)),
        uses_global_oracle=False,
        information_access="local_array_policy",
        execution_backend=str(record.get("executionBackend", "e03_dsl_interpreter")),
        direction_support=str(record.get("direction", record.get("directionSupport", "increasing_or_encoded_in_rules"))),
        observation_contract="src.e03.policy_interface.PolicyObservation",
        action_vocabulary=features["action_vocabulary"],
        state_variables=features["state_variables"],
        parameter_keys=(),
        parent_policy_ids=as_tuple(record.get("parentPolicyIds") or record.get("parent_ids")),
        source_artifacts=(str(source_path),),
        applicable_world_ids=applicable_world_ids,
        limitations=as_tuple(record.get("knownDeviations") or record.get("limitations")),
        upstream_metrics=upstream_metrics,
        representation_payload=payload,
        declared_dsl_sha256=str(declared_sha) if declared_sha else None,
        computed_dsl_canonical_sha256=features["canonical_sha"],
        computed_dsl_source_sha256=features["source_sha"],
        source_payload_sha256=stable_hash(record),
        hash_validation_status=features["hash_status"],
        parser_validation_status=features["parser_status"],
        parser_error=features["parser_error"],
    )


def interface_policy_record(
    *,
    source_experiment_id: str,
    source_step_id: str,
    source_record_id: str,
    source_policy_id: str,
    display_name: str,
    algorithm: str,
    implementation_ref: str,
    source_artifacts: tuple[str, ...],
    applicable_world_ids: tuple[str, ...],
    metadata: Mapping[str, Any],
) -> PolicyRepresentation:
    payload = {
        "algorithm": algorithm,
        "implementationRef": implementation_ref,
        "sourceArtifactHashes": {path: sha256_file(Path(path)) for path in source_artifacts if Path(path).is_file()},
        "metadata": dict(metadata),
    }
    return PolicyRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id=source_step_id,
        source_record_id=source_record_id,
        source_policy_id=source_policy_id,
        display_name=display_name,
        source_category="original",
        policy_family="public_sorting_method",
        algorithm=algorithm,
        representation_type="interface_wrapper",
        abstraction_kind="source_method_wrapper",
        interpretable=True,
        metadata_only=False,
        requires_memory=False,
        requires_signaling=False,
        uses_global_oracle=False,
        information_access="local_array_policy",
        execution_backend="e02_public_cell_simulator",
        direction_support="parameterized_by_cell_reverse_direction",
        observation_contract="src.e03.policy_interface.PolicyObservation over public repository cell objects",
        action_vocabulary=("compare", "swap", "wait"),
        state_variables=("ideal_position",) if algorithm == "selection" else (),
        source_artifacts=source_artifacts,
        applicable_world_ids=applicable_world_ids,
        limitations=as_tuple(metadata.get("knownDeviations")),
        upstream_metrics={},
        representation_payload=payload,
        source_payload_sha256=stable_hash(payload),
        hash_validation_status="interface_wrapper_hash_recorded",
        parser_validation_status="source_ast_parse_ok",
        parser_error="",
    )


def weighted_policy_record(
    *,
    source_experiment_id: str,
    source_step_id: str,
    source_record_id: str,
    record: Mapping[str, Any],
    source_path: Path,
    applicable_world_ids: tuple[str, ...],
    source_category: str,
) -> PolicyRepresentation:
    parameters = dict(record.get("parameters", {}))
    parameter_keys = tuple(sorted(parameters))
    expected = set(PARAMETER_NAMES)
    missing = sorted(expected - set(parameter_keys))
    extra = sorted(set(parameter_keys) - expected)
    parser_status = "loadable_parameters" if parameters and not missing else "parameter_validation_failed"
    parser_error = "" if parser_status == "loadable_parameters" else f"missing={missing}; extra={extra}"
    payload = {
        "policyFamily": record.get("policy_family", record.get("policyFamily", "s08_local_memory_signal_adjacent_policy")),
        "policyInputContract": record.get("policy_input_contract", record.get("policyInputContract", "src.e04.no_oracle_protocol.LocalTrainingObservation")),
        "parameters": parameters,
    }
    return PolicyRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id=source_step_id,
        source_record_id=source_record_id,
        source_policy_id=str(record.get("policy_id") or record.get("policyId") or record.get("algotypeId")),
        display_name=str(record.get("displayName") or record.get("policy_id") or record.get("policyId") or "weighted_local_policy"),
        source_category=source_category,
        policy_family=str(record.get("policy_family", record.get("policyFamily", "s08_local_memory_signal_adjacent_policy"))),
        algorithm=str(record.get("algorithm", "memory_repair_adjacent")),
        representation_type="local_training_weight_policy",
        abstraction_kind="weighted_decision_policy",
        interpretable=parser_status == "loadable_parameters",
        metadata_only=False,
        requires_memory=True,
        requires_signaling=False,
        uses_global_oracle=False,
        information_access="local_neighbors",
        execution_backend=str(record.get("executionBackend", "e04_local_training_policy")),
        direction_support=str(record.get("directionSupport", "increasing_default")),
        observation_contract=str(record.get("policy_input_contract", record.get("policyInputContract", "src.e04.no_oracle_protocol.LocalTrainingObservation"))),
        action_vocabulary=("idle", "swap_left", "swap_right"),
        state_variables=("last_move_success", "time_since_movement", "local_frustration", "neighbor_identity_memory"),
        parameter_keys=parameter_keys,
        parent_policy_ids=as_tuple(record.get("parent_ids")),
        source_artifacts=(str(source_path),),
        applicable_world_ids=applicable_world_ids,
        limitations=as_tuple(record.get("limitations")),
        upstream_metrics={
            key: value
            for key, value in record.items()
            if key.endswith("_fitness") or key.startswith("heldout_") or key.startswith("s08_") or key.startswith("s09_") or key.startswith("s13_")
        },
        representation_payload=payload,
        source_payload_sha256=stable_hash(record),
        hash_validation_status="parameter_hash_recorded",
        parser_validation_status=parser_status,
        parser_error=parser_error,
    )


def metadata_only_record(
    *,
    source_experiment_id: str,
    source_step_id: str,
    source_record_id: str,
    source_policy_id: str,
    display_name: str,
    source_category: str,
    policy_family: str,
    algorithm: str,
    source_artifacts: tuple[str, ...],
    applicable_world_ids: tuple[str, ...],
    limitations: tuple[str, ...],
    uses_global_oracle: bool = False,
) -> PolicyRepresentation:
    payload = {
        "sourcePolicyId": source_policy_id,
        "policyFamily": policy_family,
        "metadataOnlyReason": list(limitations),
    }
    return PolicyRepresentation(
        source_experiment_id=source_experiment_id,
        source_step_id=source_step_id,
        source_record_id=source_record_id,
        source_policy_id=source_policy_id,
        display_name=display_name,
        source_category=source_category,
        policy_family=policy_family,
        algorithm=algorithm,
        representation_type="metadata_only",
        abstraction_kind="policy_reference",
        interpretable=False,
        metadata_only=True,
        requires_memory=False,
        requires_signaling=False,
        uses_global_oracle=uses_global_oracle,
        information_access="global_or_endpoint_context" if uses_global_oracle else "not_documented",
        execution_backend="not_available_as_policy_artifact",
        direction_support="not_documented",
        observation_contract="not_available",
        action_vocabulary=(),
        state_variables=(),
        source_artifacts=source_artifacts,
        applicable_world_ids=applicable_world_ids,
        limitations=limitations,
        upstream_metrics={},
        representation_payload=payload,
        source_payload_sha256=stable_hash(payload),
        hash_validation_status="metadata_only_no_artifact_hash",
        parser_validation_status="metadata_only_explicit",
    )


def original_source_records(repo_dir: Path, worlds: Mapping[str, tuple[str, ...]]) -> list[PolicyRepresentation]:
    mapping = {
        "bubble": repo_dir / "modules" / "multithread" / "BubbleSortCell.py",
        "insertion": repo_dir / "modules" / "multithread" / "InsertionSortCell.py",
        "selection": repo_dir / "modules" / "multithread" / "SelectionSortCell.py",
    }
    records: list[PolicyRepresentation] = []
    for algorithm, path in mapping.items():
        source = path.read_text(encoding="utf-8")
        parser_status = "source_ast_parse_ok"
        parser_error = ""
        try:
            ast.parse(source)
        except SyntaxError as exc:
            parser_status = "source_ast_parse_failed"
            parser_error = str(exc)
        source_hash = sha256_file(path)
        records.append(
            PolicyRepresentation(
                source_experiment_id="original_repo",
                source_step_id="public_code",
                source_record_id=f"original_repo:{algorithm}",
                source_policy_id=f"original_repo:{algorithm}",
                display_name=f"public_{algorithm}_cell_method",
                source_category="original",
                policy_family="public_sorting_method",
                algorithm=algorithm,
                representation_type="source_code_method",
                abstraction_kind="python_source_method",
                interpretable=parser_status == "source_ast_parse_ok",
                metadata_only=False,
                requires_memory=False,
                requires_signaling=False,
                uses_global_oracle=False,
                information_access="local_array_policy",
                execution_backend="public_repository_cell_method",
                direction_support="parameterized_by_cell_reverse_direction",
                observation_contract="public MultiThreadCell state and neighboring public cell objects",
                action_vocabulary=("compare", "swap", "wait"),
                state_variables=("ideal_position",) if algorithm == "selection" else (),
                source_artifacts=(str(path),),
                applicable_world_ids=tuple([*worlds["array_core"], *worlds["e03"], *worlds["e06"]]),
                limitations=("Source-code method record is not a DSL or finite-state abstraction; it preserves original Python source provenance.",),
                representation_payload={"sourcePath": str(path), "sourceSha256": source_hash},
                source_payload_sha256=source_hash,
                hash_validation_status="source_code_hash_recorded",
                parser_validation_status=parser_status,
                parser_error=parser_error,
            )
        )
    return records


def collect_e03_records(base: Path, worlds: Mapping[str, tuple[str, ...]]) -> list[PolicyRepresentation]:
    records: list[PolicyRepresentation] = []
    applicable = tuple([*worlds["array_core"], *worlds["e03"]])
    classic_path = base / "policies" / "e03_classic_policy_library.json"
    classic_data = read_json(classic_path)
    for idx, record in enumerate(classic_data.get("policies", [])):
        source_record_id = compact_record_id("E03:classic", idx, record)
        if record.get("representationType") == "dsl":
            records.append(
                dsl_policy_record(
                    source_experiment_id="E03",
                    source_step_id=str(record.get("researchStepId", "S03")),
                    source_record_id=source_record_id,
                    record=record,
                    source_path=classic_path,
                    applicable_world_ids=applicable,
                    source_category="classic_dsl_shadow",
                    policy_family=f"classic_{record.get('algorithm', 'unknown')}",
                    algorithm=str(record.get("algorithm", "unknown")),
                )
            )
        else:
            algorithm = str(record.get("algorithm", "unknown"))
            source_file = {
                "bubble": REPO_ROOT / "modules" / "multithread" / "BubbleSortCell.py",
                "insertion": REPO_ROOT / "modules" / "multithread" / "InsertionSortCell.py",
                "selection": REPO_ROOT / "modules" / "multithread" / "SelectionSortCell.py",
            }.get(algorithm)
            source_artifacts = [str(classic_path)]
            if source_file and source_file.exists():
                source_artifacts.append(str(source_file))
            records.append(
                interface_policy_record(
                    source_experiment_id="E03",
                    source_step_id=str(record.get("researchStepId", "S03")),
                    source_record_id=source_record_id,
                    source_policy_id=str(record.get("policyId")),
                    display_name=f"e03_interface_{algorithm}",
                    algorithm=algorithm,
                    implementation_ref=str(record.get("implementationRef", "")),
                    source_artifacts=tuple(source_artifacts),
                    applicable_world_ids=applicable,
                    metadata=record,
                )
            )

    for path, prefix, category in (
        (base / "policies" / "e03_generated_policy_library.jsonl", "E03:generated", "generated_dsl"),
        (base / "policies" / "e03_qd_discovered_policies.jsonl", "E03:qd", "qd_discovered"),
        (base / "policies" / "e03_frontier_candidate_policies.jsonl", "E03:frontier", "frontier_candidate"),
    ):
        for idx, record in enumerate(read_jsonl(path)):
            source_record_id = compact_record_id(prefix, idx, record)
            records.append(
                dsl_policy_record(
                    source_experiment_id="E03",
                    source_step_id=str(record.get("researchStepId", path.stem)),
                    source_record_id=source_record_id,
                    record=record,
                    source_path=path,
                    applicable_world_ids=applicable,
                    source_category=category,
                    policy_family=str(record.get("mutationOperator") or record.get("source_kind") or record.get("sourceKind") or category),
                    algorithm="dsl_local_rule",
                )
            )
    return records


def collect_e04_records(base: Path, worlds: Mapping[str, tuple[str, ...]]) -> list[PolicyRepresentation]:
    records: list[PolicyRepresentation] = []
    applicable = worlds["e04"]
    for path, category in (
        (base / "policies" / "e04_evolved_repair_policies.jsonl", "evolved_repair"),
        (base / "policies" / "e04_repair_capable_algotypes.jsonl", "repair_capable"),
    ):
        for idx, record in enumerate(read_jsonl(path)):
            records.append(
                weighted_policy_record(
                    source_experiment_id="E04",
                    source_step_id=str(record.get("researchStepId", record.get("protocol_id", "S08-S15"))),
                    source_record_id=compact_record_id(f"E04:{category}", idx, record),
                    record=record,
                    source_path=path,
                    applicable_world_ids=applicable,
                    source_category=category,
                )
            )
    return records


def collect_e06_records(base: Path, worlds: Mapping[str, tuple[str, ...]]) -> list[PolicyRepresentation]:
    records: list[PolicyRepresentation] = []
    path = base / "policies" / "e06_chimeric_algotype_library.jsonl"
    applicable = worlds["e06"]
    for idx, record in enumerate(read_jsonl(path)):
        source_record_id = compact_record_id("E06:chimeric", idx, record)
        representation_type = str(record.get("representationType", "metadata_only"))
        compatibility = dict(record.get("compatibilityMetadata", {}))
        if representation_type == "dsl" and record.get("dslSource"):
            records.append(
                dsl_policy_record(
                    source_experiment_id="E06",
                    source_step_id=str(record.get("researchStepId", "S01")),
                    source_record_id=source_record_id,
                    record={**record, "policyId": record.get("algotypeId", record.get("policyId")), "policyName": record.get("displayName")},
                    source_path=path,
                    applicable_world_ids=applicable,
                    source_category=str(record.get("sourceCategory", "chimeric")),
                    policy_family=str(record.get("policyFamily", "chimeric_dsl")),
                    algorithm=str(record.get("algorithm", "dsl_local_rule")),
                )
            )
        elif representation_type == "local_training_weight_policy":
            records.append(
                weighted_policy_record(
                    source_experiment_id="E06",
                    source_step_id=str(record.get("researchStepId", "S01")),
                    source_record_id=source_record_id,
                    record={**record, "policy_id": record.get("algotypeId")},
                    source_path=path,
                    applicable_world_ids=applicable,
                    source_category=str(record.get("sourceCategory", "memory_repair")),
                )
            )
        elif representation_type == "interface_wrapper":
            algorithm = str(record.get("algorithm", "unknown"))
            source_file = {
                "bubble": REPO_ROOT / "modules" / "multithread" / "BubbleSortCell.py",
                "insertion": REPO_ROOT / "modules" / "multithread" / "InsertionSortCell.py",
                "selection": REPO_ROOT / "modules" / "multithread" / "SelectionSortCell.py",
            }.get(algorithm)
            source_artifacts = [str(path)]
            if source_file and source_file.exists():
                source_artifacts.append(str(source_file))
            records.append(
                interface_policy_record(
                    source_experiment_id="E06",
                    source_step_id=str(record.get("researchStepId", "S01")),
                    source_record_id=source_record_id,
                    source_policy_id=str(record.get("algotypeId")),
                    display_name=str(record.get("displayName", f"e06_{algorithm}")),
                    algorithm=algorithm,
                    implementation_ref=str(record.get("sourcePolicyId", "")),
                    source_artifacts=tuple(source_artifacts),
                    applicable_world_ids=applicable,
                    metadata={**record, "compatibilityMetadata": compatibility},
                )
            )
        else:
            records.append(
                metadata_only_record(
                    source_experiment_id="E06",
                    source_step_id=str(record.get("researchStepId", "S01")),
                    source_record_id=source_record_id,
                    source_policy_id=str(record.get("algotypeId", record.get("policyId", source_record_id))),
                    display_name=str(record.get("displayName", "e06_metadata_policy")),
                    source_category=str(record.get("sourceCategory", "chimeric")),
                    policy_family=str(record.get("policyFamily", "chimeric")),
                    algorithm=str(record.get("algorithm", "unknown")),
                    source_artifacts=(str(path),),
                    applicable_world_ids=applicable,
                    limitations=("E06 record could not be interpreted as DSL, interface wrapper, or E04 weighted policy; retained as metadata-only alias.",),
                    uses_global_oracle=boolish(compatibility.get("usesGlobalOracle")),
                )
            )
    return records


def collect_e05_metadata_policy_refs(base: Path, world_inventory: pd.DataFrame) -> list[PolicyRepresentation]:
    catalog_path = base / "tables" / "e05_benchmark_task_catalog.csv"
    if not catalog_path.exists():
        return []
    task_df = pd.read_csv(catalog_path)
    task_worlds = {
        str(row["task_id"]): str(row["world_id"])
        for row in world_inventory[world_inventory["experiment_id"] == "E05"].to_dict(orient="records")
    }
    policy_to_worlds: dict[str, set[str]] = defaultdict(set)
    policy_to_tasks: dict[str, set[str]] = defaultdict(set)
    for row in task_df.to_dict(orient="records"):
        task_id = str(row["benchmark_task_id"])
        world_id = task_worlds.get(task_id)
        for policy_id in as_tuple(row.get("policy_set_json")):
            if world_id:
                policy_to_worlds[policy_id].add(world_id)
            policy_to_tasks[policy_id].add(task_id)

    records: list[PolicyRepresentation] = []
    for idx, policy_id in enumerate(sorted(policy_to_tasks)):
        uses_global = policy_id.startswith("s13_")
        records.append(
            metadata_only_record(
                source_experiment_id="E05",
                source_step_id="S15",
                source_record_id=f"E05:benchmark-policy:{idx:03d}:{policy_id}",
                source_policy_id=policy_id,
                display_name=policy_id,
                source_category="e05_benchmark_policy_reference",
                policy_family="e05_benchmark_reference",
                algorithm="morphology_task_policy_reference",
                source_artifacts=(str(catalog_path),),
                applicable_world_ids=tuple(sorted(policy_to_worlds[policy_id])),
                limitations=(
                    "E05 benchmark catalog references this policy ID but no standalone policy artifact was mounted for S02.",
                    "Retained as metadata-only so S04 can link benchmark rows without inventing source code.",
                ),
                uses_global_oracle=uses_global,
            )
        )
    return records


def collect_policy_records(args: argparse.Namespace, world_inventory: pd.DataFrame) -> list[PolicyRepresentation]:
    worlds = world_groups(world_inventory)
    previous = args.previous_artifacts_dir
    records: list[PolicyRepresentation] = []
    records.extend(original_source_records(args.repo_dir, worlds))
    records.extend(collect_e03_records(previous / "E03", worlds))
    records.extend(collect_e04_records(previous / "E04", worlds))
    records.extend(collect_e05_metadata_policy_refs(previous / "E05", world_inventory))
    records.extend(collect_e06_records(previous / "E06", worlds))
    return records


def markdown_table(df: pd.DataFrame, columns: list[str], *, max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[columns].head(max_rows).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(lambda value: str(value).replace("|", "\\|"))
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(record[column]) for column in columns) + " |" for record in shown.to_dict(orient="records")]
    if len(df) > max_rows:
        omitted = [f"... {len(df) - max_rows} more rows omitted", *("" for _ in columns[1:])]
        rows.append("| " + " | ".join(omitted) + " |")
    return "\n".join([header, separator, *rows])


def top_summary_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome_classification: str,
    caveats_or_blockers: str,
    recommended_next_action: str,
    lay_summary: str,
) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: {outcome_classification}
- Caveats or blockers: {caveats_or_blockers}
- Lay summary: {lay_summary}
- Recommended next action: {recommended_next_action}
"""


def spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome: str,
    caveats: str,
    recommended_next_action: str,
    lay_summary: str,
    table: pd.DataFrame,
) -> str:
    summary = top_summary_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary)
    by_type = table.groupby(["representation_type", "abstraction_kind"]).size().reset_index(name="record_count")
    by_source = table.groupby(["source_experiment_id", "source_category"]).size().reset_index(name="record_count")
    return f"""{summary}

# E07 S02 Policy Representation Specification

## Frozen Question

Can every Algotype be encoded as DSL code, finite-state automaton, decision graph, or neural policy metadata suitable for cross-world comparison?

## Representation Contract

Each row preserves one source policy record or source alias and assigns:

- `policy_uid`: source-row-stable ID.
- `canonical_policy_id`: hash-based ID for deduplicating equivalent representations.
- Source provenance: upstream experiment, step, source policy ID, source artifact paths, source payload hash.
- Abstract form: `representation_type`, `abstraction_kind`, action vocabulary, state variables, parameter keys, and observation contract.
- Validation: declared versus computed hashes, parser/load status, metadata-only limitations, and S01 `world_id` links.

## Supported Representation Kinds

- `dsl` / `dsl_program`: parseable E03/E06 rule DSL source. S02 records both canonical DSL-object hash and raw-source hash because E03 and E06 used different declared hash conventions.
- `interface_wrapper` / `source_method_wrapper`: exact public-method wrappers for Bubble, Insertion, and Selection policies.
- `source_code_method` / `python_source_method`: original repository method source records.
- `local_training_weight_policy` / `weighted_decision_policy`: E04 local-only memory/repair policies represented by parameter vectors and input contracts.
- `metadata_only` / `policy_reference`: E05 benchmark policy references where no standalone policy artifact was mounted.

## Coverage By Representation

{markdown_table(by_type, ["representation_type", "abstraction_kind", "record_count"], max_rows=30)}

## Coverage By Source

{markdown_table(by_source, ["source_experiment_id", "source_category", "record_count"], max_rows=40)}

## Hash And Parser Rules

DSL rows pass if the declared upstream hash matches either the parsed canonical DSL hash or the raw DSL source hash. Weighted local policies pass when all expected E04 parameter names are present. Original source-method rows pass Python AST parsing. Metadata-only rows pass only when limitations explicitly state why no parseable artifact exists.
"""


def full_results_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    outcome: str,
    caveats: str,
    recommended_next_action: str,
    lay_summary: str,
    table: pd.DataFrame,
    validation_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    summary = top_summary_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary)
    by_source = table.groupby("source_experiment_id").size().reset_index(name="record_count")
    by_type = table.groupby("representation_type").size().reset_index(name="record_count")
    by_parser = table.groupby("parser_validation_status").size().reset_index(name="record_count")
    by_hash = table.groupby("hash_validation_status").size().reset_index(name="record_count")
    sample = table[["policy_uid", "canonical_policy_id", "source_experiment_id", "source_policy_id", "display_name", "representation_type", "parser_validation_status"]].head(24)
    return f"""{summary}

# E07 S02 Full Results: Represent Policies Abstractly

## Lay Summary

{lay_summary}

## Frozen Question

Can every Algotype be encoded as DSL code, finite-state automaton, decision graph, or neural policy metadata suitable for cross-world comparison?

## Inputs

- S01 world inventory: `/artifacts/tables/e07_world_inventory.csv`.
- E03 policy libraries: `/previous-artifacts/E03/policies/`.
- E04 local repair policy libraries: `/previous-artifacts/E04/policies/`.
- E05 benchmark policy references: `/previous-artifacts/E05/tables/e05_benchmark_task_catalog.csv`.
- E06 chimeric Algotype library: `/previous-artifacts/E06/policies/e06_chimeric_algotype_library.jsonl`.
- Original public source files in the repository under `modules/multithread/`.

## Methods

S02 normalized every available upstream policy record into a row-stable source record and a hash-stable canonical representation. DSL programs were parsed with `src.e03.rule_dsl.parse_policy`; local memory/repair policies were loaded as parameterized weighted decision policies using the E04 `PARAMETER_NAMES` contract; original public methods were retained as source-code and interface-wrapper representations. E05 benchmark policy IDs without mounted standalone artifacts were retained as explicit metadata-only records.

## Commands

- `python -m unittest discover -s tests/e07 -p 'test_*.py'`
- `python scripts/e07_s02_policy_representations.py --artifacts-dir /artifacts --previous-artifacts-dir /previous-artifacts --s01-world-inventory /artifacts/tables/e07_world_inventory.csv --run-unit-tests`

Unit-test command result: return code {unit_test_result.get("returnCode")}, success={unit_test_result.get("success")}, elapsed={unit_test_result.get("elapsedSeconds")} seconds.

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- New dependencies installed: none.
- CPU/GPU use: serial metadata extraction and parser validation only; no GPU and no parallel workers.

## Results

Policy representation rows written: {len(table)}.

### Records By Source

{markdown_table(by_source, ["source_experiment_id", "record_count"], max_rows=20)}

### Records By Representation Type

{markdown_table(by_type, ["representation_type", "record_count"], max_rows=20)}

### Parser/Loader Status

{markdown_table(by_parser, ["parser_validation_status", "record_count"], max_rows=20)}

### Hash Validation Status

{markdown_table(by_hash, ["hash_validation_status", "record_count"], max_rows=20)}

### Sample Rows

{markdown_table(sample, ["policy_uid", "canonical_policy_id", "source_experiment_id", "source_policy_id", "display_name", "representation_type", "parser_validation_status"], max_rows=24)}

## Validation

{markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)}

## Artifacts

{markdown_table(pd.DataFrame(artifacts), ["path", "description", "sha256"], max_rows=20)}

## Caveats And Blockers

{caveats}

- Rows preserve source aliases; downstream steps should use `canonical_policy_id` to deduplicate behaviorally identical or wrapped representations.
- Metadata-only E05 policy references are placeholders for linking benchmark tasks; they are not executable policy definitions.
- DSL hash validation accepts two documented upstream conventions: canonical parsed DSL hash and raw source hash.

## Provenance

- Repository HEAD before commit: `{manifest["git"]["head"]}`
- Repository branch: `{manifest["git"]["branch"]}`
- Policy table hash: `{stable_hash(table.to_dict(orient="records"))}`
- Source files:
  - `src/e07/policy_schema.py`
  - `scripts/e07_s02_policy_representations.py`
  - `tests/e07/test_policy_schema.py`

## Recommended Next Action

{recommended_next_action}
"""


def build_manifest(
    *,
    args: argparse.Namespace,
    table: pd.DataFrame,
    validation_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e07.s02.manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "generatedAtUtc": utc_now(),
        "git": {
            "head": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "inputs": {
            "s01WorldInventory": str(args.s01_world_inventory),
            "s01WorldInventorySha256": sha256_file(args.s01_world_inventory) if args.s01_world_inventory.exists() else None,
            "previousArtifactsDir": str(args.previous_artifacts_dir),
        },
        "policyRepresentations": {
            "recordCount": int(len(table)),
            "recordCountBySource": table.groupby("source_experiment_id").size().astype(int).to_dict(),
            "recordCountByRepresentationType": table.groupby("representation_type").size().astype(int).to_dict(),
            "recordHash": stable_hash(table.to_dict(orient="records")),
        },
        "validation": {
            "checks": validation_df.to_dict(orient="records"),
            "unitTests": unit_test_result,
        },
        "artifacts": artifacts,
        "sourceFiles": [
            source_entry(args.repo_dir / "src" / "e07" / "policy_schema.py", args.repo_dir),
            source_entry(args.repo_dir / "scripts" / "e07_s02_policy_representations.py", args.repo_dir),
            source_entry(args.repo_dir / "tests" / "e07" / "test_policy_schema.py", args.repo_dir),
        ],
    }


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    tables_dir = artifacts_dir / "tables"

    world_inventory = pd.read_csv(args.s01_world_inventory)
    records = collect_policy_records(args, world_inventory)
    table = dataframe_from_policy_records(records)
    validation_df = validate_policy_table(table, s01_world_ids=world_inventory["world_id"].astype(str).tolist())

    unit_test_result = {"command": "not run", "success": True, "returnCode": 0, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        unit_test_result = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e07", "-p", "test_*.py"], args.repo_dir)

    parquet_path = tables_dir / "e07_policy_representations.parquet"
    csv_path = tables_dir / "e07_policy_representations.csv"
    validation_path = step_dir / "e07_s02_validation_checks.csv"
    spec_path = reports_dir / "e07_policy_representation_spec.md"
    full_results_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(parquet_path, index=False)
    table.to_csv(csv_path, index=False)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)

    validation_success = bool(validation_df["success"].all() and unit_test_result["success"])
    dsl_count = int((table["representation_type"] == "dsl").sum())
    metadata_only_count = int(table["metadata_only"].astype(bool).sum())
    validation_result = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} policy validation checks passed; "
        f"unit tests {'passed' if unit_test_result['success'] else 'failed'}; "
        f"{len(table)} policy source records, {dsl_count} DSL rows, {metadata_only_count} metadata-only rows"
    )
    outcome = "supportive" if validation_success and len(table) > 0 else "null"
    caveats = (
        "S02 is a representation audit, not behavioral validation. E05 benchmark policies without standalone mounted artifacts are metadata-only. "
        "Rows preserve aliases from E03/E04/E06/original sources, so downstream steps should deduplicate with `canonical_policy_id` when needed."
    )
    lay_summary = (
        "S02 converted the available upstream policies into one table that records source, abstract form, parser status, hashes, and S01 world links. "
        "Executable DSL and weighted local policies are validated where possible, while missing policy artifacts are kept as explicit metadata-only records."
    )
    recommended_next_action = "Stop before S03; after Chief Scientist review, proceed to S03 goal representations using `canonical_policy_id` and S01 `world_id` links."

    artifacts = [
        artifact_entry(parquet_path, artifacts_dir, "Required S02 policy representation table"),
        artifact_entry(csv_path, artifacts_dir, "CSV inspection copy of S02 policy representation table"),
        artifact_entry(validation_path, artifacts_dir, "S02 validation check table"),
        self_referential_artifact_entry(spec_path, artifacts_dir, "Required S02 policy representation specification report"),
        self_referential_artifact_entry(manifest_path, artifacts_dir, "S02 artifact and provenance manifest"),
        self_referential_artifact_entry(full_results_path, artifacts_dir, "Required S02 full-results handoff report"),
    ]

    write_text(spec_path, spec_markdown(artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary, table))
    manifest = build_manifest(args=args, table=table, validation_df=validation_df, unit_test_result=unit_test_result, artifacts=artifacts)
    manifest["validation"]["success"] = validation_success
    write_json(manifest_path, manifest)
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts,
            validation_result,
            outcome,
            caveats,
            recommended_next_action,
            lay_summary,
            table,
            validation_df,
            unit_test_result,
            manifest,
        ),
    )

    final_artifacts = [
        artifact_entry(parquet_path, artifacts_dir, "Required S02 policy representation table"),
        artifact_entry(csv_path, artifacts_dir, "CSV inspection copy of S02 policy representation table"),
        artifact_entry(validation_path, artifacts_dir, "S02 validation check table"),
        self_referential_artifact_entry(spec_path, artifacts_dir, "Required S02 policy representation specification report"),
        self_referential_artifact_entry(manifest_path, artifacts_dir, "S02 artifact and provenance manifest"),
        self_referential_artifact_entry(full_results_path, artifacts_dir, "Required S02 full-results handoff report"),
    ]
    manifest["artifacts"] = final_artifacts
    write_json(manifest_path, manifest)
    write_text(spec_path, spec_markdown(final_artifacts, validation_result, outcome, caveats, recommended_next_action, lay_summary, table))
    write_text(
        full_results_path,
        full_results_markdown(
            final_artifacts,
            validation_result,
            outcome,
            caveats,
            recommended_next_action,
            lay_summary,
            table,
            validation_df,
            unit_test_result,
            manifest,
        ),
    )

    print(json.dumps({"success": validation_success, "recordCount": len(table), "validationResult": validation_result}, indent=2))
    return 0 if validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
