#!/usr/bin/env python3
"""Run E05 S02 cell-identity schema validation and write artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace2d import (  # noqa: E402
    IDENTITY_SCHEMA_VERSION,
    NEIGHBOR_PREFERENCE_VERSION,
    actor_components,
    build_example_identity_catalog,
    catalog_rows,
    default_identity_schema,
    evaluate_neighbor_preferences,
    scalar_identity,
    identity_to_cell_state,
    observable_components,
    scalar_values_from_identities,
    validate_identity_catalog,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S02"
STEP_NUMBER = 2
STEP_TITLE = "Generalize cell identity"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    value = json_ready(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#{}[]\n,") or text.lower() in {"true", "false", "null"}:
        return json.dumps(text)
    return text


def to_yaml(value: Any, indent: int = 0) -> str:
    value = json_ready(value)
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(payload) + "\n", encoding="utf-8")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> list[Path]:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path_without_suffix.with_suffix(".csv")
    parquet_path = path_without_suffix.with_suffix(".parquet")
    safe = df.copy()
    for column in safe.columns:
        if safe[column].map(lambda item: isinstance(item, (Mapping, list, tuple))).any():
            safe[column] = safe[column].map(lambda item: json.dumps(json_ready(item), sort_keys=True) if isinstance(item, (Mapping, list, tuple)) else item)
    safe.to_csv(csv_path, index=False)
    safe.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def identity_schema_markdown(schema_record: Mapping[str, Any]) -> str:
    rows = [
        [
            field["name"],
            field["kind"],
            field["fixed"],
            field["mutable"],
            field["observableToNeighbors"],
            field["hidden"],
            field["description"],
        ]
        for field in schema_record["fields"]
    ]
    return f"""# E05 S02 Identity Schema

Research step ID: {STEP_ID}
Title: {STEP_TITLE}
Identity schema version: `{IDENTITY_SCHEMA_VERSION}`
Neighborhood preference version: `{NEIGHBOR_PREFERENCE_VERSION}`

S02 replaces scalar `Value` with identity vectors while keeping `scalar_value` as a reversible one-dimensional baseline component. Biological-sounding labels are computational role names only.

{markdown_table(['field', 'kind', 'fixed', 'mutable', 'neighbor-observable', 'hidden', 'description'], rows)}

## Locality Boundary

Neighbor observations include only fields marked `observableToNeighbors=true` and not `hidden`. Actor-local target-neighborhood preferences are available to the actor or evaluator, but are intentionally excluded from neighbor payloads. Hidden `internal_state` is excluded from both neighbor and actor public payloads.
"""


def build_validation_rows() -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    schema = default_identity_schema()
    scalar_values = [5, 1, 4, 2, 3]
    scalar_identities = tuple(
        scalar_identity(100 + index, value, n=len(scalar_values))
        for index, value in enumerate(scalar_values)
    )
    example_identities = build_example_identity_catalog()
    full_catalog = scalar_identities + example_identities

    catalog_errors = validate_identity_catalog(full_catalog, schema)
    restored_values = scalar_values_from_identities(scalar_identities)
    first_identity = example_identities[0]
    full_record = first_identity.to_record(schema=schema, include_hidden=True)
    restored_record = type(first_identity).from_record(full_record).to_record(schema=schema, include_hidden=True)
    neighbor_payload = observable_components(first_identity, schema)
    actor_payload = actor_components(first_identity, schema)
    cell_state = identity_to_cell_state(first_identity, schema)

    satisfied_pref = evaluate_neighbor_preferences(example_identities[0], [example_identities[1], example_identities[2], example_identities[4]])
    penalized_pref = evaluate_neighbor_preferences(example_identities[0], [example_identities[3]])

    validation_rows = [
        {
            "research_step_id": STEP_ID,
            "check_id": "schema_valid",
            "success": len(schema.validate()) == 0,
            "detail": "; ".join(schema.validate()),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "catalog_valid",
            "success": len(catalog_errors) == 0,
            "detail": "; ".join(catalog_errors),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "scalar_1d_roundtrip",
            "success": restored_values == scalar_values,
            "detail": f"input={scalar_values}; restored={restored_values}",
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "serialization_roundtrip",
            "success": restored_record == full_record,
            "detail": first_identity.identity_id,
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "neighbor_visibility_boundary",
            "success": "target_neighbor_preferences" not in neighbor_payload and "internal_state" not in neighbor_payload,
            "detail": json.dumps(json_ready(neighbor_payload), sort_keys=True),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "actor_payload_boundary",
            "success": "target_neighbor_preferences" in actor_payload and "internal_state" not in actor_payload,
            "detail": json.dumps(json_ready(actor_payload), sort_keys=True),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "cell_state_bridge_preserves_scalar",
            "success": cell_state.value == first_identity.scalar_value and "internal_state" not in cell_state.identity,
            "detail": json.dumps(json_ready(cell_state.to_record()), sort_keys=True),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "preference_satisfied_case",
            "success": satisfied_pref["penalty"] == 0.0 and satisfied_pref["score"] == 1.0,
            "detail": json.dumps(json_ready(satisfied_pref), sort_keys=True),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "preference_penalized_case",
            "success": penalized_pref["penalty"] > 0.0 and penalized_pref["score"] < 1.0,
            "detail": json.dumps(json_ready(penalized_pref), sort_keys=True),
        },
    ]
    preference_rows = [
        {
            "research_step_id": STEP_ID,
            "case_id": "satisfied",
            "actor_cell_id": satisfied_pref["actorCellId"],
            "neighbor_cell_ids_json": json.dumps(satisfied_pref["neighborCellIds"]),
            "rule_count": satisfied_pref["ruleCount"],
            "satisfied_rule_count": satisfied_pref["satisfiedRuleCount"],
            "penalty": satisfied_pref["penalty"],
            "score": satisfied_pref["score"],
            "rules_json": json.dumps(json_ready(satisfied_pref["rules"]), sort_keys=True),
        },
        {
            "research_step_id": STEP_ID,
            "case_id": "penalized",
            "actor_cell_id": penalized_pref["actorCellId"],
            "neighbor_cell_ids_json": json.dumps(penalized_pref["neighborCellIds"]),
            "rule_count": penalized_pref["ruleCount"],
            "satisfied_rule_count": penalized_pref["satisfiedRuleCount"],
            "penalty": penalized_pref["penalty"],
            "score": penalized_pref["score"],
            "rules_json": json.dumps(json_ready(penalized_pref["rules"]), sort_keys=True),
        },
    ]
    return validation_rows, pd.DataFrame(catalog_rows(full_catalog, schema)), pd.DataFrame(preference_rows), schema.to_record()


def write_outputs(artifacts_dir: Path) -> dict[str, Any]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    code_index_dir = artifacts_dir / "code" / "e05_morphospace_simulator"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    code_index_dir.mkdir(parents=True, exist_ok=True)

    validation_rows, catalog_df, preference_df, schema_record = build_validation_rows()
    validation_df = pd.DataFrame(validation_rows)

    artifact_paths: list[Path] = []
    artifact_paths.extend(write_dataframe(validation_df, step_dir / "identity_validation_results"))
    artifact_paths.extend(write_dataframe(validation_df, results_dir / "e05_s02_identity_validation"))
    artifact_paths.extend(write_dataframe(catalog_df, step_dir / "identity_catalog"))
    artifact_paths.extend(write_dataframe(preference_df, step_dir / "neighborhood_preference_examples"))

    catalog_jsonl = step_dir / "identity_catalog.jsonl"
    catalog_jsonl.write_text("\n".join(catalog_df["full_record_json"].tolist()) + "\n", encoding="utf-8")
    artifact_paths.append(catalog_jsonl)

    schema_json = step_dir / "identity_schema.json"
    write_json(schema_json, schema_record)
    artifact_paths.append(schema_json)

    schema_md = step_dir / "identity_schema.md"
    schema_md.write_text(identity_schema_markdown(schema_record), encoding="utf-8")
    artifact_paths.append(schema_md)

    config_payload = {
        "schemaVersion": "e05_s02_identity_target_specs.v1",
        "researchStepId": STEP_ID,
        "identitySchema": schema_record,
        "exampleCatalogPath": str(step_dir / "identity_catalog.parquet"),
        "targetNeighborhoodPreferenceVersion": NEIGHBOR_PREFERENCE_VERSION,
        "targetMorphologies": {
            "status": "deferred_to_S03",
            "note": "S02 defines identity fields and local target-neighborhood preference rule schema; geometric target morphologies are queued for S03.",
        },
    }
    config_yaml = configs_dir / "e05_identity_target_specs.yaml"
    write_yaml(config_yaml, config_payload)
    artifact_paths.append(config_yaml)

    code_index_path = code_index_dir / "README.md"
    code_index_path.write_text(
        "# E05 Morphospace Simulator Code Index\n\n"
        "Repository-backed source is kept in git per workspace instructions.\n\n"
        "- S01 package: `morphospace2d/substrates.py`\n"
        "- S02 package: `morphospace2d/identities.py`\n"
        "- S01 runner: `scripts/e05_s01_substrate_generalization.py`\n"
        "- S02 runner: `scripts/e05_s02_cell_identity.py`\n"
        "- S01 tests: `tests/test_e05_substrate_generalization.py`\n"
        "- S02 tests: `tests/test_e05_cell_identity.py`\n",
        encoding="utf-8",
    )
    artifact_paths.append(code_index_path)

    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_test_success = unit_log_path.exists() and "OK" in unit_log_path.read_text(encoding="utf-8", errors="replace")
    if unit_log_path.exists():
        artifact_paths.append(unit_log_path)

    success = bool(validation_df["success"].all())
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    reported_artifact_paths = artifact_paths + [status_path, summary_path, manifest_path]
    validation_result = (
        "Passed: identity schema is valid, scalar 1D values round-trip, identity records serialize, hidden/internal fields are excluded from neighbor payloads, "
        "cell-state bridging preserves scalar Value, and target-neighborhood preference examples distinguish satisfied from penalized local neighborhoods"
        + ("; focused repository unit tests passed." if unit_test_success else ".")
        if success
        else "Failed: at least one S02 identity validation check did not pass."
    )
    caveats = (
        "S02 defines computational identity fields and local preference rules only. Target morphologies, shape metrics, and biological interpretations remain deferred; "
        "organ-like labels are analogical computational roles."
    )
    recommended = "Proceed to S03 to define target morphologies only after Chief Scientist instruction."

    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in reported_artifact_paths],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "validationCommands": [
            {
                "args": "PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_e05_cell_identity tests.test_e05_substrate_generalization",
                "logPath": str(unit_log_path),
                "success": unit_test_success,
            }
        ],
        "generatedAt": utc_now(),
        "git": git_metadata(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadEnvironment": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            },
        },
    }
    write_json(status_path, status_payload)

    summary_rows = [
        [row["check_id"], row["success"], row["detail"][:120]]
        for row in validation_rows
    ]
    summary = f"""# E05 S02 Status Summary

- Step ID: {STEP_ID}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Artifacts written: {', '.join(str(path) for path in reported_artifact_paths)}
- Validation result: {validation_result}
- Outcome classification: {'supportive' if success else 'constraining/contradictory'}
- Caveats or blockers: {caveats}
- Lay summary: Cell identities now include scalar baseline value, axis coordinate, computational role, polarity, adhesion class, actor-local target-neighborhood preferences, and hidden internal state. The validation confirms that the scalar baseline is reversible and hidden fields do not leak into neighbor-observable payloads.
- Recommended next action: {recommended}

## Validation Table

{markdown_table(['check', 'success', 'detail'], summary_rows)}
"""
    summary_path.write_text(summary, encoding="utf-8")

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "schemaVersion": "eidosoma.step_artifact_manifest.v1",
        "generatedAt": utc_now(),
        "artifacts": [
            {"path": str(path), "sha256": sha256_path(path), "bytes": path.stat().st_size}
            for path in artifact_paths + [status_path, summary_path]
            if path.exists() and path.is_file()
        ],
        "sourceCode": {
            "repositoryPackage": "morphospace2d/",
            "runner": "scripts/e05_s02_cell_identity.py",
            "tests": "tests/test_e05_cell_identity.py",
            "note": "Repository source is committed in git and not copied into artifacts per GitHub workspace rule.",
        },
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
    }
    write_json(manifest_path, manifest_payload)

    return {
        "success": success,
        "stepDir": step_dir,
        "status": status_payload,
        "artifactPaths": [str(path) for path in reported_artifact_paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    result = write_outputs(args.artifacts_dir)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
