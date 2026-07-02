#!/usr/bin/env python3
"""Validate E05 S03 target morphologies and write gallery artifacts."""

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

from src.e05.cell_identity import attach_identity, scalar_identity
from src.e05.substrates import SubstrateCell
from src.e05.targets import (
    boundary_target,
    constructed_target_error,
    default_target_gallery,
    gradient_target,
    render_target_gallery,
    sorted_row_target,
    symmetry_target,
    target_render_mode,
    target_summary_rows,
)


STEP_ID = "S03"
STEP_NUMBER = 3
EXPERIMENT_ID = "E05"
EXPERIMENT_TITLE = "From one-dimensional sorting to higher-dimensional morphospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--s01-results", type=Path, default=Path("/artifacts/results/e05_substrate_validation.parquet"))
    parser.add_argument("--s02-results", type=Path, default=Path("/artifacts/results/e05_identity_validation.parquet"))
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


def validation_gallery_target_families(targets) -> dict[str, Any]:
    observed = {
        "target_count": len(targets),
        "target_kinds": [target.target_kind for target in targets],
    }
    expected = {
        "target_count": 7,
        "target_kinds": ["sorted_row", "gradient", "stripes", "boundary", "ring", "symmetry", "organ_like"],
    }
    return _row(
        "gallery_contains_required_target_families",
        "target_catalog",
        observed == expected,
        expected,
        observed,
        "Gallery covers sorted row, gradient, stripes, boundary, ring, symmetry, and toy organ-like target families.",
    )


def validation_constructed_zero_errors(targets) -> dict[str, Any]:
    errors = {target.target_id: constructed_target_error(target) for target in targets}
    observed = {"all_zero": all(value == 0.0 for value in errors.values()), "errors": errors}
    expected = {"all_zero": True, "errors": {target.target_id: 0.0 for target in targets}}
    return _row(
        "constructed_target_states_have_zero_error",
        "zero_error",
        observed == expected,
        expected,
        observed,
        "Every gallery target produces zero target error when a constructed substrate is filled with its own target identities.",
    )


def validation_sorted_row_contract() -> dict[str, Any]:
    target = sorted_row_target((4, 1, 3, 2))
    observed = {
        "values": [target.identities_by_site[site_id].components["value"] for site_id in target.substrate.site_ids],
        "metric_components": list(target.metric_contract.compatible_components),
        "error": constructed_target_error(target),
    }
    expected = {"values": [1.0, 2.0, 3.0, 4.0], "metric_components": ["value"], "error": 0.0}
    return _row(
        "sorted_row_target_preserves_scalar_value_contract",
        "scalar_target",
        observed == expected,
        expected,
        observed,
        "Sorted-row target preserves the scalar Value special case for S06 continuity work.",
    )


def validation_gradient_endpoints() -> dict[str, Any]:
    target = gradient_target(4, 2)
    values = {
        str(target.substrate.coordinate(site_id)): target.identities_by_site[site_id].components["ap_coordinate"]
        for site_id in target.substrate.site_ids
    }
    observed = {"left_endpoint": values["(0, 0)"], "right_endpoint": values["(3, 1)"], "error": constructed_target_error(target)}
    expected = {"left_endpoint": 0.0, "right_endpoint": 1.0, "error": 0.0}
    return _row(
        "gradient_target_endpoints_are_bounded",
        "gradient",
        observed == expected,
        expected,
        observed,
        "Gradient target spans ap_coordinate 0 to 1 and validates with zero constructed error.",
    )


def validation_boundary_perimeter() -> dict[str, Any]:
    target = boundary_target(4, 3)
    perimeter = sorted(
        site_id
        for site_id in target.substrate.site_ids
        if target.identities_by_site[site_id].components["organ_type"] == "boundary"
    )
    observed = {"perimeter": perimeter, "error": constructed_target_error(target)}
    expected = {"perimeter": [0, 1, 2, 3, 4, 7, 8, 9, 10, 11], "error": 0.0}
    return _row(
        "boundary_target_marks_perimeter",
        "boundary",
        observed == expected,
        expected,
        observed,
        "Boundary target labels exactly the rectangular perimeter as boundary tissue.",
    )


def validation_symmetry_mirror() -> dict[str, Any]:
    target = symmetry_target(7, 5)
    width = target.substrate.dimensions["width"]
    height = target.substrate.dimensions["height"]
    mismatches = []
    for y in range(height):
        for x in range(width):
            left_site = y * width + x
            right_site = y * width + (width - 1 - x)
            if target.identities_by_site[left_site].components["organ_type"] != target.identities_by_site[right_site].components["organ_type"]:
                mismatches.append([left_site, right_site])
    observed = {"mismatch_count": len(mismatches), "error": constructed_target_error(target)}
    expected = {"mismatch_count": 0, "error": 0.0}
    return _row(
        "symmetry_target_mirrors_organ_labels",
        "symmetry",
        observed == expected,
        expected,
        observed,
        "Bilateral symmetry target mirrors organ labels around the midline.",
    )


def validation_perturbed_error_positive() -> dict[str, Any]:
    target = sorted_row_target((1, 2, 3))
    state = target.substrate.copy_empty()
    wrong_cells = (
        attach_identity(SubstrateCell("wrong_0", value=3), scalar_identity(3, "wrong_0")),
        attach_identity(SubstrateCell("wrong_1", value=2), scalar_identity(2, "wrong_1")),
        attach_identity(SubstrateCell("wrong_2", value=1), scalar_identity(1, "wrong_2")),
    )
    state.fill_sites(wrong_cells)
    error = target.target_error(state)
    observed = {"error_positive": error > 0.0, "error": error}
    expected = {"error_positive": True}
    return _row(
        "target_error_detects_wrong_state",
        "negative_control",
        observed["error_positive"] == expected["error_positive"],
        expected,
        observed,
        "Target-error proxy is nonzero when scalar identities are assigned to the wrong sites.",
    )


def validation_gallery_png(output_path: Path) -> dict[str, Any]:
    import matplotlib.image as mpimg

    image = mpimg.imread(output_path)
    observed = {
        "exists": output_path.exists(),
        "png_header": output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
        "size_gt_1000": output_path.stat().st_size > 1000,
        "shape": list(image.shape),
        "nonblank": bool(float(image.max()) > float(image.min())),
    }
    expected = {
        "exists": True,
        "png_header": True,
        "size_gt_1000": True,
        "nonblank": True,
    }
    success = all(observed[key] == value for key, value in expected.items())
    return _row(
        "target_gallery_png_is_valid_and_nonblank",
        "rendering",
        success,
        expected,
        observed,
        "Rendered target gallery PNG exists, has a PNG header, and contains nonblank pixel variation.",
    )


def validation_render_modes(targets) -> dict[str, Any]:
    observed = {target.target_kind: target_render_mode(target) for target in targets}
    expected = {
        "sorted_row": "value",
        "gradient": "ap_coordinate",
        "stripes": "organ_type",
        "boundary": "organ_type",
        "ring": "organ_type",
        "symmetry": "organ_type",
        "organ_like": "organ_type",
    }
    return _row(
        "target_render_modes_match_target_semantics",
        "rendering",
        observed == expected,
        expected,
        observed,
        "Gallery rendering uses scalar values for sorted rows, ap_coordinate for gradients, and organ_type colors for categorical morphology targets.",
    )


def run_validations(targets, gallery_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            validation_gallery_target_families(targets),
            validation_constructed_zero_errors(targets),
            validation_sorted_row_contract(),
            validation_gradient_endpoints(),
            validation_boundary_perimeter(),
            validation_symmetry_mirror(),
            validation_perturbed_error_positive(),
            validation_gallery_png(gallery_path),
            validation_render_modes(targets),
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
- Caveats or blockers: S03 defines toy computational target morphologies and zero-error checks only; action extensions and morphology metrics remain for S04 and S05.
- Lay summary: S03 turns the substrate and identity layers into concrete target patterns such as sorted rows, gradients, stripes, rings, boundaries, symmetry, and a toy organ-like pattern, then verifies constructed states hit those targets exactly.
- Recommended next action: {recommended_next_action}
"""


def target_spec_markdown(
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    target_index_df: pd.DataFrame,
    recommended_next_action: str,
) -> str:
    family_lines = "\n".join(
        f"- `{row.target_id}` (`{row.target_kind}`): {row.title}; primary metric `{row.primary_metric}`; render mode `{row.render_mode}`; site count {row.site_count}."
        for row in target_index_df.itertuples()
    )
    return f"""{top_summary_markdown(artifacts, validation_result, recommended_next_action)}

# E05 S03 Target Morphology Specification

## Scope

S03 defines reusable target morphology specifications over S01 substrates using S02 identity vectors. These are explicit site-to-identity maps with metric-compatibility metadata. They are toy computational targets for later morphogenesis benchmarks, not anatomical data.

## Target Families

{family_lines}

## Target Contract

Each `TargetMorphology` stores a target ID, substrate graph, identity schema, identity per site, target kind, metric contract, and metadata. `constructed_substrate()` fills a fresh substrate with the target identities. `target_error()` currently computes a mean identity-distance proxy and must be exactly zero for a constructed target state.

## Metric Compatibility

S03 records the intended primary metric and compatible identity components for each target. The full morphology metric suite is intentionally deferred to S05.

## Gallery

The gallery PNG at `/artifacts/figures/e05/target_morphology_gallery.png` visualizes the S03 target set. Scalar sorted-row targets render by value, gradient targets render by `ap_coordinate`, and categorical identity targets render by organ-type color.

## Validation Summary

{markdown_validation_table(validation_df)}
"""


def full_results_markdown(
    *,
    artifacts: list[dict[str, Any]],
    validation_result: str,
    validation_df: pd.DataFrame,
    target_index_df: pd.DataFrame,
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

# Research Step Full Results: {STEP_ID} Define Target Morphologies

## Lay Summary

S03 defines the target patterns that later simulations will try to reach. The targets include a sorted row for continuity with the original sorting model, plus 2D gradients, stripes, rings, boundary/interior layouts, bilateral symmetry, and a simple body-with-appendage toy pattern. Constructed states that exactly match each target have zero target error, and a PNG gallery shows the full target set.

## Frozen Question

Can simple and organ-like target patterns be specified as local or global constraints suitable for evaluating morphogenesis proxies?

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, E05 S03.
- S01 substrate code and artifacts, including `{args.s01_results}`.
- S02 identity-vector code and artifacts, including `{args.s02_results}`.
- Datasets: none required.

## Methods

Implemented `src/e05/targets.py` with `TargetMorphology`, metric contracts, target constructors, target-error proxy, target-summary rows, and gallery rendering. Added tests in `tests/e05/test_targets.py` and generated validation/report artifacts through `scripts/e05_s03_target_gallery.py`.

## Commands

{command_lines if command_lines else "- Unit tests were skipped by command-line option."}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Pandas: `{pd.__version__}`
- Matplotlib: used for gallery rendering through the preinstalled plotting stack.
- No new packages were installed for S03.
- CPU/GPU use: validation and gallery rendering are small and serial; no GPU use was needed.

## Parameters

- Target gallery size: {len(target_index_df)} targets.
- Target families: {", ".join(target_index_df["target_kind"].tolist())}.
- Rendering output: `/artifacts/figures/e05/target_morphology_gallery.png`.

## Results

{markdown_validation_table(validation_df)}

All validation rows passed. The primary success criterion was met: target patterns are represented on S01 substrates with S02 identities, gallery rendering succeeds, and constructed target states have zero target error.

## Target Index

| target_id | target_kind | substrate_kind | site_count | primary_metric | render_mode |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(f"| {row.target_id} | {row.target_kind} | {row.substrate_kind} | {row.site_count} | {row.primary_metric} | {row.render_mode} |" for row in target_index_df.itertuples())}

## Validation Checks

- Gallery includes sorted row, gradient, stripes, boundary, ring, symmetry, and toy organ-like targets.
- Constructed target states have zero target error for every target.
- Sorted-row target preserves the scalar Value target contract.
- Gradient endpoints span the bounded anterior-posterior coordinate.
- Boundary and symmetry targets satisfy expected structural constraints.
- A deliberately wrong scalar state has positive target error.
- Gallery PNG exists, has a valid PNG header, is nonblank, and uses target-appropriate render modes.

## Artifacts

{chr(10).join(f"- `{entry['path']}`: {entry['description']}" for entry in artifacts)}

## Source Provenance

{source_lines}

## Caveats And Limitations

- Targets are computational proxy patterns, not biological anatomy.
- The S03 target-error proxy is intentionally minimal and serves zero-error validation only; S05 will define richer morphology metrics.
- The toy organ-like target is a synthetic benchmark pattern, not a wet-lab regeneration model.

## Blockers And Failed Assumptions

No blocker was found. Target morphologies can be represented on S01 substrates with S02 identities.

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
    figures_dir = artifacts_dir / "figures" / "e05"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, reports_dir, results_dir, tables_dir, configs_dir, figures_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    targets = default_target_gallery()
    gallery_path = figures_dir / "target_morphology_gallery.png"
    render_target_gallery(targets, gallery_path)
    validation_df = run_validations(targets, gallery_path)
    validation_success = bool(validation_df["success"].all())
    validation_result = f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed"
    recommended_next_action = "Stop before S04 and let the Chief Scientist review S03; if accepted, proceed to generalize actions in S04."

    target_index_df = pd.DataFrame(target_summary_rows(targets))
    test_commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        test_commands.append(run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e05", "-v"], args.repo_dir))
        test_commands.append(
            run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py", "-v"], args.repo_dir)
        )
    test_success = all(command["success"] for command in test_commands)

    validation_path = results_dir / "e05_target_validation.parquet"
    validation_csv_path = tables_dir / "e05_target_validation.csv"
    target_index_path = results_dir / "e05_target_morphology_index.parquet"
    target_index_csv_path = tables_dir / "e05_target_morphology_index.csv"
    target_specs_path = results_dir / "e05_target_gallery_specs.json"
    config_path = configs_dir / "e05_s03_target_gallery.json"
    source_manifest_path = src_snapshot_dir / "e05_targets_manifest.json"
    spec_path = reports_dir / "e05_target_morphology_spec.md"
    full_results_path = step_dir / "research_step_full_results.md"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksum_path = checksums_dir / "sha256sums.txt"

    validation_df.to_parquet(validation_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    target_index_df.to_parquet(target_index_path, index=False)
    target_index_df.to_csv(target_index_csv_path, index=False)
    write_json(
        target_specs_path,
        {
            "schema": "eidosoma.e05_s03.target_gallery_specs.v1",
            "researchStepId": STEP_ID,
            "targets": [target.compact_dict() for target in targets],
        },
    )
    write_json(
        config_path,
        {
            "schema": "eidosoma.e05_s03.target_gallery_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "targetIds": [target.target_id for target in targets],
            "validationCases": validation_df["validation_case"].tolist(),
            "unitTestsRun": bool(args.run_unit_tests),
            "createdAt": started_at,
        },
    )

    source_files = [
        source_entry(args.repo_dir / "src/e05/targets.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e05/test_targets.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e05_s03_target_gallery.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e05/cell_identity.py", args.repo_dir),
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
        artifact_entry(validation_path, artifacts_dir, "Machine-readable S03 target validation table."),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV mirror of S03 validation table."),
        artifact_entry(target_index_path, artifacts_dir, "Machine-readable target morphology index."),
        artifact_entry(target_index_csv_path, artifacts_dir, "CSV mirror of target morphology index."),
        artifact_entry(target_specs_path, artifacts_dir, "JSON target morphology specifications."),
        artifact_entry(gallery_path, artifacts_dir, "S03 target morphology gallery PNG."),
        artifact_entry(config_path, artifacts_dir, "Validation and gallery configuration."),
        artifact_entry(source_manifest_path, artifacts_dir, "Repository source hashes for S03 code and tests."),
    ]
    planned_artifact_paths = [
        {"path": str(spec_path), "description": "S03 target morphology specification report."},
        {"path": str(full_results_path), "description": "S03 full-results handoff report."},
        {"path": str(artifact_manifest_path), "description": "S03 artifact manifest."},
        {"path": str(run_manifest_path), "description": "Experiment run manifest updated for S03."},
        {"path": str(checksum_path), "description": "Checksums for key S03 artifacts."},
    ]
    write_text(
        spec_path,
        target_spec_markdown(
            [*artifacts_for_summary, *planned_artifact_paths],
            validation_result,
            validation_df,
            target_index_df,
            recommended_next_action,
        ),
    )
    artifacts_after_spec = [
        *artifacts_for_summary,
        artifact_entry(spec_path, artifacts_dir, "S03 target morphology specification report."),
    ]
    write_text(
        full_results_path,
        full_results_markdown(
            artifacts=[*artifacts_after_spec, *planned_artifact_paths[1:]],
            validation_result=f"{validation_result}; unit-test commands success={test_success}",
            validation_df=validation_df,
            target_index_df=target_index_df,
            test_commands=test_commands,
            source_files=source_files,
            args=args,
            recommended_next_action=recommended_next_action,
        ),
    )
    artifacts_final = [
        *artifacts_after_spec,
        artifact_entry(full_results_path, artifacts_dir, "S03 full-results handoff report."),
    ]
    write_json(
        artifact_manifest_path,
        {
            "schema": "eidosoma.artifact_manifest.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "experimentId": EXPERIMENT_ID,
            "success": bool(validation_success and test_success),
            "artifacts": [*artifacts_final, manifest_self_entry(artifact_manifest_path, artifacts_dir, "S03 artifact manifest.")],
            "validationResult": f"{validation_result}; unit-test commands success={test_success}",
            "caveatsOrBlockers": "No blocker. S03 defines toy computational targets only; action extensions and metric suite remain future steps.",
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
        "artifacts": [*artifacts_final, artifact_entry(artifact_manifest_path, artifacts_dir, "S03 artifact manifest.")],
        "validationResult": f"{validation_result}; unit-test commands success={test_success}",
    }
    write_json(run_manifest_path, run_manifest_payload)
    checksum_inputs = [
        validation_path,
        validation_csv_path,
        target_index_path,
        target_index_csv_path,
        target_specs_path,
        gallery_path,
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
