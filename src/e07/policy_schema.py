"""Policy representation schema for E07 cross-world analysis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


POLICY_SCHEMA_VERSION = "eidosoma.e07.policy_representation.v1"

POLICY_COLUMNS = (
    "schema_version",
    "policy_uid",
    "canonical_policy_id",
    "source_experiment_id",
    "source_step_id",
    "source_record_id",
    "source_policy_id",
    "display_name",
    "source_category",
    "policy_family",
    "algorithm",
    "representation_type",
    "abstraction_kind",
    "interpretable",
    "metadata_only",
    "requires_memory",
    "requires_signaling",
    "uses_global_oracle",
    "information_access",
    "execution_backend",
    "direction_support",
    "observation_contract",
    "action_vocabulary_json",
    "state_variables_json",
    "parameter_keys_json",
    "parent_policy_ids_json",
    "source_artifacts_json",
    "applicable_world_ids_json",
    "limitations_json",
    "upstream_metrics_json",
    "representation_payload_json",
    "declared_dsl_sha256",
    "computed_dsl_canonical_sha256",
    "computed_dsl_source_sha256",
    "source_payload_sha256",
    "representation_hash",
    "hash_validation_status",
    "parser_validation_status",
    "parser_error",
    "record_hash",
)


def sanitize_json(value: Any) -> Any:
    """Convert pandas/numpy/NaN values into strict JSON-compatible values."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def stable_json(payload: Any) -> str:
    return json.dumps(sanitize_json(payload), sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_uid(source_experiment_id: str, source_record_id: str) -> str:
    digest = stable_hash({"sourceExperimentId": source_experiment_id, "sourceRecordId": source_record_id})
    return f"e07pol:{digest[:20]}"


def canonical_policy_id(representation_hash: str) -> str:
    return f"canon:{representation_hash[:20]}"


@dataclass(frozen=True)
class PolicyRepresentation:
    """One source policy or policy alias normalized for E07."""

    source_experiment_id: str
    source_step_id: str
    source_record_id: str
    source_policy_id: str
    display_name: str
    source_category: str
    policy_family: str
    algorithm: str
    representation_type: str
    abstraction_kind: str
    interpretable: bool
    metadata_only: bool
    requires_memory: bool
    requires_signaling: bool
    uses_global_oracle: bool
    information_access: str
    execution_backend: str
    direction_support: str
    observation_contract: str
    action_vocabulary: tuple[str, ...] = ()
    state_variables: tuple[str, ...] = ()
    parameter_keys: tuple[str, ...] = ()
    parent_policy_ids: tuple[str, ...] = ()
    source_artifacts: tuple[str, ...] = ()
    applicable_world_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    upstream_metrics: Mapping[str, Any] = field(default_factory=dict)
    representation_payload: Mapping[str, Any] = field(default_factory=dict)
    declared_dsl_sha256: str | None = None
    computed_dsl_canonical_sha256: str | None = None
    computed_dsl_source_sha256: str | None = None
    source_payload_sha256: str | None = None
    representation_hash: str | None = None
    hash_validation_status: str = "not_applicable"
    parser_validation_status: str = "not_attempted"
    parser_error: str = ""
    schema_version: str = POLICY_SCHEMA_VERSION

    def with_hashes(self) -> "PolicyRepresentation":
        representation_hash = self.representation_hash or stable_hash(
            {
                "representationType": self.representation_type,
                "abstractionKind": self.abstraction_kind,
                "payload": dict(self.representation_payload),
                "computedDslCanonicalSha256": self.computed_dsl_canonical_sha256,
                "computedDslSourceSha256": self.computed_dsl_source_sha256,
                "sourcePayloadSha256": self.source_payload_sha256,
            }
        )
        return PolicyRepresentation(**{**self.__dict__, "representation_hash": representation_hash})

    def to_flat_dict(self) -> dict[str, Any]:
        checked = self.with_hashes()
        payload = {
            "schema_version": checked.schema_version,
            "policy_uid": policy_uid(checked.source_experiment_id, checked.source_record_id),
            "canonical_policy_id": canonical_policy_id(str(checked.representation_hash)),
            "source_experiment_id": checked.source_experiment_id,
            "source_step_id": checked.source_step_id,
            "source_record_id": checked.source_record_id,
            "source_policy_id": checked.source_policy_id,
            "display_name": checked.display_name,
            "source_category": checked.source_category,
            "policy_family": checked.policy_family,
            "algorithm": checked.algorithm,
            "representation_type": checked.representation_type,
            "abstraction_kind": checked.abstraction_kind,
            "interpretable": bool(checked.interpretable),
            "metadata_only": bool(checked.metadata_only),
            "requires_memory": bool(checked.requires_memory),
            "requires_signaling": bool(checked.requires_signaling),
            "uses_global_oracle": bool(checked.uses_global_oracle),
            "information_access": checked.information_access,
            "execution_backend": checked.execution_backend,
            "direction_support": checked.direction_support,
            "observation_contract": checked.observation_contract,
            "action_vocabulary_json": stable_json(list(checked.action_vocabulary)),
            "state_variables_json": stable_json(list(checked.state_variables)),
            "parameter_keys_json": stable_json(list(checked.parameter_keys)),
            "parent_policy_ids_json": stable_json(list(checked.parent_policy_ids)),
            "source_artifacts_json": stable_json(list(checked.source_artifacts)),
            "applicable_world_ids_json": stable_json(list(checked.applicable_world_ids)),
            "limitations_json": stable_json(list(checked.limitations)),
            "upstream_metrics_json": stable_json(dict(checked.upstream_metrics)),
            "representation_payload_json": stable_json(dict(checked.representation_payload)),
            "declared_dsl_sha256": checked.declared_dsl_sha256,
            "computed_dsl_canonical_sha256": checked.computed_dsl_canonical_sha256,
            "computed_dsl_source_sha256": checked.computed_dsl_source_sha256,
            "source_payload_sha256": checked.source_payload_sha256,
            "representation_hash": checked.representation_hash,
            "hash_validation_status": checked.hash_validation_status,
            "parser_validation_status": checked.parser_validation_status,
            "parser_error": checked.parser_error,
        }
        payload["record_hash"] = stable_hash({key: payload[key] for key in payload if key != "record_hash"})
        return payload


def dataframe_from_policy_records(records: Sequence[PolicyRepresentation]) -> pd.DataFrame:
    rows = [record.to_flat_dict() for record in records]
    df = pd.DataFrame(rows, columns=list(POLICY_COLUMNS))
    if not df.empty:
        df = df.sort_values(["source_experiment_id", "source_record_id"], kind="mergesort").reset_index(drop=True)
    return df


def parse_json_list(text: Any) -> list[Any]:
    if text is None:
        return []
    try:
        return list(json.loads(str(text)))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def validate_policy_table(
    table: pd.DataFrame,
    *,
    required_sources: Iterable[str] = ("E03", "E04", "E05", "E06", "original_repo"),
    s01_world_ids: Iterable[str] = (),
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    missing_columns = [column for column in POLICY_COLUMNS if column not in table.columns]
    checks.append(
        {
            "validation_case": "required_columns_present",
            "success": not missing_columns,
            "detail": "all policy columns present" if not missing_columns else f"missing columns: {missing_columns}",
        }
    )

    duplicate_uid_count = int(table["policy_uid"].duplicated().sum()) if "policy_uid" in table else len(table)
    checks.append(
        {
            "validation_case": "policy_uids_unique",
            "success": duplicate_uid_count == 0,
            "detail": f"duplicate policy_uid rows: {duplicate_uid_count}",
        }
    )

    observed_sources = set(table.get("source_experiment_id", pd.Series(dtype=str)).astype(str))
    missing_sources = sorted(set(required_sources) - observed_sources)
    checks.append(
        {
            "validation_case": "required_sources_covered",
            "success": not missing_sources,
            "detail": "covered required policy sources" if not missing_sources else f"missing sources: {missing_sources}",
        }
    )

    parse_success_statuses = {"parsed", "loadable_parameters", "source_ast_parse_ok", "metadata_only_explicit"}
    parser_failures = table[~table["parser_validation_status"].isin(parse_success_statuses)] if "parser_validation_status" in table else table
    checks.append(
        {
            "validation_case": "representations_parse_or_are_explicit_metadata",
            "success": parser_failures.empty,
            "detail": "all representations parse/load or are explicit metadata-only records"
            if parser_failures.empty
            else f"parser/load failures: {len(parser_failures)}",
        }
    )

    acceptable_hash = {
        "matches_declared_canonical_dsl_hash",
        "matches_declared_source_dsl_hash",
        "source_code_hash_recorded",
        "parameter_hash_recorded",
        "metadata_only_no_artifact_hash",
        "interface_wrapper_hash_recorded",
        "not_applicable",
    }
    hash_failures = table[~table["hash_validation_status"].isin(acceptable_hash)] if "hash_validation_status" in table else table
    checks.append(
        {
            "validation_case": "hashes_match_or_are_documented",
            "success": hash_failures.empty,
            "detail": "all declared hashes match a known convention or are documented"
            if hash_failures.empty
            else f"hash failures: {len(hash_failures)}",
        }
    )

    metadata_rows = table[table["metadata_only"].astype(bool)] if "metadata_only" in table else table.iloc[0:0]
    undocumented_metadata = metadata_rows[metadata_rows["limitations_json"].map(lambda text: len(parse_json_list(text)) == 0)]
    checks.append(
        {
            "validation_case": "metadata_only_records_have_limitations",
            "success": undocumented_metadata.empty,
            "detail": "metadata-only rows document limitations" if undocumented_metadata.empty else f"undocumented metadata-only rows: {len(undocumented_metadata)}",
        }
    )

    world_id_set = set(str(item) for item in s01_world_ids)
    bad_world_rows: list[str] = []
    if world_id_set and "applicable_world_ids_json" in table:
        for row in table.to_dict(orient="records"):
            for world_id in parse_json_list(row.get("applicable_world_ids_json")):
                if str(world_id) not in world_id_set:
                    bad_world_rows.append(str(row.get("policy_uid")))
                    break
    checks.append(
        {
            "validation_case": "s01_world_links_valid",
            "success": not bad_world_rows,
            "detail": "all S01 world links resolve" if not bad_world_rows else f"bad world links: {len(bad_world_rows)}",
        }
    )

    source_missing: list[str] = []
    for row in table.to_dict(orient="records"):
        for path in parse_json_list(row.get("source_artifacts_json")):
            if path and not Path(path).exists():
                source_missing.append(str(path))
    checks.append(
        {
            "validation_case": "source_artifacts_exist",
            "success": not source_missing,
            "detail": "all listed source artifacts exist" if not source_missing else f"missing source artifacts: {len(source_missing)}",
        }
    )

    return pd.DataFrame(checks)
