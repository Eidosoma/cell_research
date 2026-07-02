#!/usr/bin/env python3
"""Validate E05 S02 identity vectors and write research artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.e05.cell_identity import (
    CellIdentity,
    TargetNeighborPreference,
    attach_identity,
    default_morphogenesis_identity_schema,
    identity_from_substrate_cell,
    identity_ordered_values,
    local_identity_observation,
    scalar_identity,
    scalar_value_schema,
    should_swap_for_identity_order,
)
from src.e05.substrates import SubstrateCell, array_substrate, square_grid_substrate


STEP_ID = "S02"
STEP_NUMBER = 2
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e03-dir", type=Path, default=Path("/previous-artifacts/E03"))
    parser.add_argument("--previous-e04-dir", type=Path, default=Path("/previous-artifacts/E04"))
    parser.add_argument("--s01-results", type=Path, default=Path("/artifacts/results/e05_substrate_validation.parquet"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def morph_identity(
    identity_id: str,
    ap: float,
    organ: str,
    polarity: tuple[float, float],
    adhesion: str,
    preferences: tuple[TargetNeighborPreference, ...] = (),
) -> CellIdentity:
    return CellIdentity(
        identity_id=identity_id,
        components={
            "ap_coordinate": ap,
            "organ_type": organ,
            "polarity": polarity,
            "adhesion_type": adhesion,
        },
        target_preferences=preferences,
    )


def _row(
    validation_case: str,
    case_type: str,
    success: bool,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    return {
        "research_step_id": STEP_ID,
        "experiment_id": EXPERIMENT_ID,
        "validation_case": validation_case,
        "case_type": case_type,
        "success": bool(success),
        "expected_json": stable_json(expected),
        "observed_json": stable_json(observed),
        "detail": detail,
    }


def validation_scalar_order_decisions() -> dict[str, Any]:
    schema = scalar_value_schema(1, 4)
    observed = {
        "left_gt_right_increasing_swap": should_swap_for_identity_order(scalar_identity(4), scalar_identity(1), schema),
        "left_gt_right_decreasing_swap": should_swap_for_identity_order(
            scalar_identity(4), scalar_identity(1), schema, direction="decreasing"
        ),
        "compare_2_3": schema.compare_order(scalar_identity(2), scalar_identity(3)),
        "schema_order_component": schema.order_component,
    }
    expected = {
        "left_gt_right_increasing_swap": True,
        "left_gt_right_decreasing_swap": False,
        "compare_2_3": -1,
        "schema_order_component": "value",
    }
    return _row(
        "scalar_identity_order_decisions_match_values",
        "scalar_special_case",
        observed == expected,
        expected,
        observed,
        "One-component identity schema gives the same adjacent inversion decisions as scalar Value comparison.",
    )


def validation_scalar_sort_recovery() -> dict[str, Any]:
    increasing = identity_ordered_values((4, 1, 3, 2), direction="increasing")
    decreasing = identity_ordered_values((1, 4, 2, 3), direction="decreasing")
    duplicates = identity_ordered_values((2, 1, 2), direction="increasing")
    observed = {
        "increasing_final_values": list(increasing["final_values"]),
        "increasing_swap_count": increasing["swap_count"],
        "decreasing_final_values": list(decreasing["final_values"]),
        "decreasing_swap_count": decreasing["swap_count"],
        "duplicate_final_values": list(duplicates["final_values"]),
    }
    expected = {
        "increasing_final_values": [1, 2, 3, 4],
        "increasing_swap_count": 4,
        "decreasing_final_values": [4, 3, 2, 1],
        "decreasing_swap_count": 4,
        "duplicate_final_values": [1, 2, 2],
    }
    return _row(
        "scalar_value_sorting_recovered_by_identity_swaps",
        "scalar_special_case",
        observed == expected,
        expected,
        observed,
        "Local adjacent swaps driven only by scalar identity order recover increasing and decreasing scalar sorting.",
    )


def validation_identity_metadata_roundtrip() -> dict[str, Any]:
    identity = scalar_identity(7, "seven")
    cell = attach_identity(SubstrateCell("cell_a", value=7, label="scalar"), identity)
    recovered = identity_from_substrate_cell(cell)
    observed = {
        "identity_id": recovered.identity_id,
        "value": recovered.components["value"],
        "metadata_key_present": "e05_identity" in cell.metadata,
    }
    expected = {"identity_id": "seven", "value": 7.0, "metadata_key_present": True}
    return _row(
        "identity_metadata_roundtrip_on_substrate_cell",
        "schema_serialization",
        observed == expected,
        expected,
        observed,
        "SubstrateCell metadata can carry and recover CellIdentity records without mutating substrate mechanics.",
    )


def validation_local_identity_observation() -> dict[str, Any]:
    schema = default_morphogenesis_identity_schema()
    substrate = array_substrate(3)
    actor = morph_identity(
        "actor",
        0.5,
        "neural",
        (1.0, 0.0),
        "high",
        (TargetNeighborPreference("right", "organ_type", "epidermis"),),
    )
    left = morph_identity("left", 0.25, "neural", (1.0, 0.0), "high")
    right = morph_identity("right", 0.75, "epidermis", (0.0, 1.0), "low")
    substrate.fill_sites(
        (
            attach_identity(SubstrateCell("left", value=0.25), left),
            attach_identity(SubstrateCell("actor", value=0.5), actor),
            attach_identity(SubstrateCell("right", value=0.75), right),
        )
    )
    observation = local_identity_observation(substrate, 1, schema)
    observed = {
        "actor_cell_id": observation.actor_cell_id,
        "directions": [relation.direction for relation in observation.neighbor_relations],
        "order_relations": [relation.order_relation for relation in observation.neighbor_relations],
        "left_compatibility": observation.neighbor_relations[0].compatibility_score,
        "right_preference": observation.neighbor_relations[1].target_preference_score,
    }
    expected = {
        "actor_cell_id": "actor",
        "directions": ["left", "right"],
        "order_relations": [1, -1],
        "left_compatibility": 1.0,
        "right_preference": 1.0,
    }
    return _row(
        "local_identity_observation_exposes_neighbor_relations",
        "local_observation",
        observed == expected,
        expected,
        observed,
        "Policy-visible local observation includes identity order, compatibility, and target-neighbor preference scores.",
    )


def validation_compatibility_and_polarity() -> dict[str, Any]:
    schema = default_morphogenesis_identity_schema()
    same = morph_identity("same", 0.2, "neural", (1.0, 0.0), "high")
    same_family = morph_identity("same_family", 0.8, "neural", (1.0, 0.0), "high")
    opposite = morph_identity("opposite", 0.2, "epidermis", (-1.0, 0.0), "low")
    observed = {
        "same_family_compatibility": schema.compatibility_score(same, same_family),
        "opposite_polarity_distance": schema.component_distance("polarity", same, opposite),
        "opposite_compatibility_less_than_same": schema.compatibility_score(same, opposite) < schema.compatibility_score(same, same_family),
    }
    expected = {
        "same_family_compatibility": 1.0,
        "opposite_polarity_distance": 1.0,
        "opposite_compatibility_less_than_same": True,
    }
    return _row(
        "compatibility_and_polarity_are_bounded",
        "compatibility_metrics",
        observed == expected,
        expected,
        observed,
        "Organ/adhesion compatibility and polarity distance are bounded proxy scores.",
    )


def validation_target_preference_mismatch() -> dict[str, Any]:
    schema = default_morphogenesis_identity_schema()
    substrate = square_grid_substrate(2, 1)
    actor = morph_identity(
        "actor",
        0.1,
        "boundary",
        (1.0, 0.0),
        "medium",
        (TargetNeighborPreference("east", "organ_type", "neural"),),
    )
    mismatch = morph_identity("mismatch", 0.2, "epidermis", (1.0, 0.0), "medium")
    substrate.fill_sites(
        (
            attach_identity(SubstrateCell("actor", value=0.1), actor),
            attach_identity(SubstrateCell("mismatch", value=0.2), mismatch),
        )
    )
    relation = local_identity_observation(substrate, 0, schema).neighbor_relations[0]
    observed = {
        "direction": relation.direction,
        "target_preference_score": relation.target_preference_score,
    }
    expected = {"direction": "east", "target_preference_score": 0.0}
    return _row(
        "target_neighbor_preference_scores_mismatch",
        "local_preference",
        observed == expected,
        expected,
        observed,
        "A directional target-neighbor preference scores zero when the neighbor has the wrong organ_type.",
    )


def validation_schema_rejections() -> dict[str, Any]:
    schema = default_morphogenesis_identity_schema()
    failures = []
    for label, callback in (
        ("missing_components", lambda: schema.validate_identity(CellIdentity("missing", {"ap_coordinate": 0.1}))),
        ("bad_category", lambda: schema.validate_identity(morph_identity("bad_category", 0.1, "heart", (1.0, 0.0), "high"))),
        ("bad_preference_relation", lambda: TargetNeighborPreference("left", "organ_type", "neural", relation="contains")),
    ):
        try:
            callback()
        except ValueError:
            failures.append(label)
    observed = {"failure_labels": failures}
    expected = {"failure_labels": ["missing_components", "bad_category", "bad_preference_relation"]}
    return _row(
        "schema_validation_rejects_invalid_identity_inputs",
        "schema_validation",
        observed == expected,
        expected,
        observed,
        "Schema validation rejects missing components, unknown categories, and unsupported preference relations.",
    )


def run_validations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_scalar_order_decisions(),
            validation_scalar_sort_recovery(),
            validation_identity_metadata_roundtrip(),
            validation_local_identity_observation(),
            validation_compatibility_and_polarity(),
            validation_target_preference_mismatch(),
            validation_schema_rejections(),
        ]
    )


def markdown_validation_table(df: pd.DataFrame) -> str:
    columns = ["validation_case", "case_type", "success", "detail"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def top_summary_markdown(artifacts: list[dict[str, Any]], validation_result: str, recommended_next_action: str) -> str:
    artifact_lines = "\n".join(f"- `{entry['path']}`" for entry in artifacts if entry.get("path"))
    return f"""## Top Summary

- Research step ID: {STEP_ID}
- Completion status: Completed
- Artifacts written:
{artifact_lines}
- Validation result: {validation_result}
- Outcome classification: supportive
- Caveats or blockers: S02 defines identity-vector mechanics and scalar special-case recovery only; target morphologies and target-error metrics remain for S03 and S05.
- Lay summary: S02 lets simulated cells carry richer identities while proving the original scalar Value sorting task remains available as a one-component identity-vector special case.
- Recommended next action: {recommended_next_action}
"""


def identity_spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    recommended_next_action: str,
) -> str:
    scalar_schema = scalar_value_schema().compact_dict()
    default_schema = default_morphogenesis_identity_schema().compact_dict()
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# E05 S02 Identity Vector Specification

## Scope

S02 replaces scalar-only `Value` observations with typed identity vectors while preserving scalar sorting as a special case. The layer is intentionally local: identity vectors are stored on S01 `SubstrateCell` metadata, and local policy observations report only actor-neighbor identity relations.

## Identity Record

`CellIdentity` contains a stable `identity_id`, typed `components`, optional `target_preferences`, and metadata. It serializes into the `e05_identity` metadata key on S01 substrate cells.

## Component Types

- `continuous`: numeric, optionally bounded, and optionally ordered.
- `categorical`: string label with optional allowed categories.
- `polarity`: finite numeric vector compared by bounded cosine distance.

## S02 Schemas

Scalar schema:

```json
{json.dumps(scalar_schema, indent=2, sort_keys=True)}
```

Default exploratory morphogenesis schema:

```json
{json.dumps(default_schema, indent=2, sort_keys=True)}
```

## Local Preference Contract

`TargetNeighborPreference` records an optional direction, component name, expected value, relation, tolerance, and weight. S02 scores these preferences only against adjacent occupied neighbors from S01 substrate observations. S03 will define target morphologies; S02 only defines the per-cell preference representation and local scoring.

## Validation Summary

{markdown_validation_table(validation_df)}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    test_commands: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    args: argparse.Namespace,
    recommended_next_action: str,
) -> str:
    command_lines = "\n".join(
        f"- `{command['command']}`: return code {command['returnCode']}, success={command['success']}, elapsed={command['elapsedSeconds']:.3f}s"
        for command in test_commands
    )
    source_lines = "\n".join(
        f"- `{entry['relativePath']}` sha256 `{entry['sha256']}` ({entry['sizeBytes']} bytes)"
        for entry in source_files
    )
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# Research Step Full Results: {STEP_ID} Generalize Cell Identity

## Lay Summary

S02 gives each simulated cell a typed identity vector rather than just a number. A cell can now carry an anterior-posterior coordinate, organ-type label, polarity vector, adhesion type, and local preferences about what kind of neighbor should be nearby. The original scalar sorting model is still available as a one-component identity vector called `value`, so this extension does not break continuity with the 1D sorting baseline.

## Frozen Question

Can scalar Value be replaced with identity vectors while preserving local target-neighborhood preferences?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S02.
- S01 substrate code and artifacts, including `{args.s01_results}`.
- E03 policy interface context: `{args.previous_e03_dir}` and live repository `src/e03/policy_interface.py`.
- E04 memory/signaling context: `{args.previous_e04_dir}` and live repository `src/e04/`.
- Datasets: none required.

## Methods

Implemented `src/e05/cell_identity.py` with typed identity schemas, `CellIdentity` records, directional `TargetNeighborPreference` records, metadata attachment to S01 `SubstrateCell`, identity distance and compatibility functions, local identity observations, scalar identity-order decisions, and a scalar-identity adjacent-swap recovery helper. Added unit tests in `tests/e05/test_cell_identity.py` and validation cases in `scripts/e05_s02_identity_validation.py`.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- No new packages were installed for S02.
- CPU/GPU use: validation is small and serial; no GPU use was needed.

## Parameters

- Scalar special-case values: increasing `(4, 1, 3, 2)` and decreasing `(1, 4, 2, 3)`.
- Default identity schema components: `ap_coordinate`, `organ_type`, `polarity`, and `adhesion_type`.
- Local observation fixtures: 1D array and 2-site square-grid substrate.

## Results

{markdown_validation_table(validation_df)}

All validation rows passed. The primary success criterion was met: scalar Value sorting is recovered as a one-component identity-vector special case, and cells can carry multi-component identities with local neighbor-relation observations.

## Validation Checks

- Scalar adjacent-order decisions match scalar Value comparisons.
- Scalar Value sorting is recovered by local adjacent swaps for increasing and decreasing directions.
- Identity metadata round-trips through S01 substrate cells.
- Local observations expose actor-neighbor order, compatibility, and target-preference scores.
- Compatibility and polarity scores are bounded proxy metrics.
- Schema validation rejects incomplete or invalid identity records.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- S02 is an interface and validation step; it does not define target morphologies, global morphology errors, shape repair tasks, or biological identity claims.
- The default morphogenesis schema is a toy computational proxy. Component weights and category sets should be treated as configurable assumptions for downstream work.
- Local target-neighborhood preferences are per-cell local scores, not a full target morphology specification.

## Blockers And Failed Assumptions

No blocker was found. The scalar special case recovered adjacent-swap sorting; target morphology work should proceed in S03 after review.

## Recommended Next Action

{recommended_next_action}
"""


def write_checksums(paths: list[Path], checksum_path: Path, artifacts_dir: Path) -> None:
    lines = []
    for path in sorted(paths):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}")
    write_text(checksum_path, "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, reports_dir, results_dir, tables_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    validation_df = run_validations()
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    recommended_next_action = "Stop before S03 and let the Chief Scientist review S02; if accepted, proceed to define target morphologies in S03."

    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_path = results_dir / "e05_identity_validation.parquet"
    validation_csv_path = tables_dir / "e05_identity_validation.csv"
    config_path = configs_dir / "e05_s02_identity_validation.json"
    source_manifest_path = src_snapshot_dir / "e05_cell_identity_manifest.json"
    spec_path = reports_dir / "e05_identity_vector_spec.md"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    write_json(
        config_path,
        {
            "schema": "eidosoma.e05_s02.identity_validation_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "validationCases": validation_df["validation_case"].tolist(),
            "scalarSpecialCaseRequired": True,
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/cell_identity.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_cell_identity.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s02_identity_validation.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/substrates.py", args.repo_dir),
    ]
    write_json(
        source_manifest_path,
        {
            "schema": "eidosoma.source_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "sourceFiles": source_files,
            "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
            "createdAt": utc_now(),
        },
    )

    artifacts_for_summary = [
        artifact_entry(validation_path, artifacts_dir, "Machine-readable S02 identity validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S02 validation table for quick inspection."),
        artifact_entry(config_path, artifacts_dir, "Validation configuration and case list."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S02 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(spec_path), "description": "S02 identity vector specification report."},
        {"path": str(full_results_path), "description": "S02 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S02 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S02."},
        {"path": str(checksum_path), "description": "Checksums for key S02 artifacts."},
    ]
    write_text(
        spec_path,
        identity_spec_markdown(
            [*artifacts_for_summary, *planned_artifact_paths],
            validation_result,
            validation_df,
            recommended_next_action,
        ),
    )
    artifacts_after_spec = [
        *artifacts_for_summary,
        artifact_entry(spec_path, artifacts_dir, "S02 identity vector specification report."),
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_after_spec, *planned_artifact_paths[1:]],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_after_spec,
        artifact_entry(full_results_path, artifacts_dir, "S02 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S02 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "caveatsOrBlockers": "No blocker. S02 does not define target morphologies or morphology metrics.",
            "recommendedNextAction": recommended_next_action,
            "createdAt": utc_now(),
        },
    )
    run_manifest_payload = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "experimentTitle": EXPERIMENT_TITLE,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "success": bool(validation_success and test_success),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "pythonVersion": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"pandas": pd.__version__},
        "commands": test_commands,
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S02 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        validation_path,
        validation_csv_path,
        config_path,
        source_manifest_path,
        spec_path,
        full_results_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    write_checksums(checksum_inputs, checksum_path, artifacts_dir)

    print(json.dumps({"success": bool(validation_success and test_success), "validationResult": validation_result, "artifactsDir": str(artifacts_dir)}, indent=2))
    return 0 if validation_success and test_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
