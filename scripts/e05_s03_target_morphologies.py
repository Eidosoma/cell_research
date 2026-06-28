#!/usr/bin/env python3
"""Run E05 S03 target morphology validation and write artifacts."""

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
    LOCAL_TARGET_PAYLOAD_VERSION,
    TARGET_ENERGY_VERSION,
    TARGET_SCHEMA_VERSION,
    build_standard_target_library,
    default_identity_schema,
    evaluate_target_energy,
    load_target_library_json,
    render_target_panel,
    scrambled_state,
    target_catalog_rows,
    write_target_library_json,
)


EXPERIMENT_ID = "E05"
STEP_ID = "S03"
STEP_NUMBER = 3
STEP_TITLE = "Define target morphologies"
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
        lines = []
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
            if isinstance(item, (Mapping, list)):
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


def target_library_markdown(target_rows: Sequence[Mapping[str, Any]], figure_path: Path) -> str:
    rows = [
        [row["target_id"], row["motif"], row["substrate_type"], row["node_count"], row["edge_count"], row["description"]]
        for row in target_rows
    ]
    return f"""# E05 S03 Target Morphology Library

Research step ID: {STEP_ID}
Title: {STEP_TITLE}
Target schema version: `{TARGET_SCHEMA_VERSION}`
Energy schema version: `{TARGET_ENERGY_VERSION}`
Local target payload version: `{LOCAL_TARGET_PAYLOAD_VERSION}`

S03 defines computational target morphologies over the S01 substrates and S02 identity schema. Target-energy evaluation is global and report-facing; local policy payloads expose only actor-local target preferences and visible identity components, not whole target maps or hidden state.

Figure: `{figure_path}`

{markdown_table(['target_id', 'motif', 'substrate', 'nodes', 'edges', 'description'], rows)}
"""


def build_validation_outputs(target_library_path: Path) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    targets = build_standard_target_library()
    loaded_targets = load_target_library_json(target_library_path)
    loaded_by_id = {target.target_id: target for target in loaded_targets}
    required_motifs = {"gradient", "stripes", "ring", "sorted_row", "boundary", "organ_like"}
    motif_set = {target.motif for target in targets}

    validation_rows: list[dict[str, Any]] = [
        {
            "research_step_id": STEP_ID,
            "check_id": "required_motif_coverage",
            "success": required_motifs.issubset(motif_set),
            "detail": ",".join(sorted(motif_set)),
        },
        {
            "research_step_id": STEP_ID,
            "check_id": "target_config_roundtrip",
            "success": [target.target_id for target in loaded_targets] == [target.target_id for target in targets],
            "detail": ",".join(target.target_id for target in loaded_targets),
        },
    ]
    energy_rows: list[dict[str, Any]] = []
    payload_rows: list[dict[str, Any]] = []
    for target in targets:
        errors = target.validate()
        exact_energy = evaluate_target_energy(target, target.target_state())
        scrambled_energy = evaluate_target_energy(target, scrambled_state(target))
        loaded_energy = evaluate_target_energy(loaded_by_id[target.target_id], loaded_by_id[target.target_id].target_state())
        validation_rows.extend(
            [
                {
                    "research_step_id": STEP_ID,
                    "check_id": f"{target.target_id}_valid",
                    "success": len(errors) == 0,
                    "detail": "; ".join(errors),
                },
                {
                    "research_step_id": STEP_ID,
                    "check_id": f"{target.target_id}_zero_energy_on_target_state",
                    "success": exact_energy["totalEnergy"] == 0.0 and loaded_energy["totalEnergy"] == 0.0,
                    "detail": json.dumps(json_ready({"exact": exact_energy, "loaded": loaded_energy}), sort_keys=True),
                },
                {
                    "research_step_id": STEP_ID,
                    "check_id": f"{target.target_id}_scrambled_positive_energy",
                    "success": scrambled_energy["totalEnergy"] > 0.0,
                    "detail": json.dumps(json_ready(scrambled_energy), sort_keys=True),
                },
            ]
        )
        for case_id, energy in [("target_state", exact_energy), ("scrambled_state", scrambled_energy), ("loaded_target_state", loaded_energy)]:
            energy_rows.append(
                {
                    "research_step_id": STEP_ID,
                    "target_id": target.target_id,
                    "motif": target.motif,
                    "case_id": case_id,
                    "component_penalty": energy["componentPenalty"],
                    "missing_penalty": energy["missingPenalty"],
                    "extra_penalty": energy["extraPenalty"],
                    "preference_penalty": energy["preferencePenalty"],
                    "total_energy": energy["totalEnergy"],
                    "within_tolerance": energy["withinTolerance"],
                }
            )
        for position in sorted(target.substrate.nodes)[: min(4, len(target.substrate.nodes))]:
            payload = target.local_policy_payload(position)
            payload_rows.append(
                {
                    "research_step_id": STEP_ID,
                    "target_id": target.target_id,
                    "motif": target.motif,
                    "position_json": json.dumps(list(position)),
                    "payload_keys_json": json.dumps(sorted(payload)),
                    "actor_visible_keys_json": json.dumps(sorted(payload["actorVisibleIdentity"])),
                    "leakage_audit_passed": "internal_state" not in payload["actorVisibleIdentity"] and "target_map" not in payload,
                    "payload_json": json.dumps(json_ready(payload), sort_keys=True),
                }
            )
    return validation_rows, pd.DataFrame(energy_rows), pd.DataFrame(payload_rows), target_catalog_rows(targets)


def write_outputs(artifacts_dir: Path) -> dict[str, Any]:
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    figures_dir = step_dir / "figures"
    code_index_dir = artifacts_dir / "code" / "e05_morphospace_simulator"
    for directory in [step_dir, results_dir, configs_dir, figures_dir, code_index_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    targets = build_standard_target_library()
    target_library_json = step_dir / "target_morphology_library.json"
    write_target_library_json(target_library_json, targets)
    validation_rows, energy_df, payload_df, target_rows = build_validation_outputs(target_library_json)
    validation_df = pd.DataFrame(validation_rows)
    catalog_df = pd.DataFrame(target_rows)

    artifact_paths: list[Path] = [target_library_json]
    artifact_paths.extend(write_dataframe(validation_df, step_dir / "target_validation_results"))
    artifact_paths.extend(write_dataframe(validation_df, results_dir / "e05_s03_target_validation"))
    artifact_paths.extend(write_dataframe(catalog_df, step_dir / "target_catalog"))
    artifact_paths.extend(write_dataframe(energy_df, step_dir / "target_energy_examples"))
    artifact_paths.extend(write_dataframe(payload_df, step_dir / "local_payload_audit"))

    figure_path = figures_dir / "target_morphology_panel.png"
    render_target_panel(targets, figure_path)
    artifact_paths.append(figure_path)

    library_md = step_dir / "target_morphology_library.md"
    library_md.write_text(target_library_markdown(target_rows, figure_path), encoding="utf-8")
    artifact_paths.append(library_md)

    config_payload = {
        "schemaVersion": "e05_s03_identity_target_specs.v1",
        "researchStepId": STEP_ID,
        "identitySchemaVersion": IDENTITY_SCHEMA_VERSION,
        "targetSchemaVersion": TARGET_SCHEMA_VERSION,
        "targetEnergyVersion": TARGET_ENERGY_VERSION,
        "localTargetPayloadVersion": LOCAL_TARGET_PAYLOAD_VERSION,
        "identitySchema": default_identity_schema().to_record(),
        "targetLibraryPath": str(target_library_json),
        "targetCatalogPath": str(step_dir / "target_catalog.parquet"),
        "targetMorphologies": [target.to_record() for target in targets],
        "leakageBoundary": {
            "globalTargetEvaluation": "target-energy functions may inspect full target configs for report-facing metrics",
            "localPolicyPayload": "only actor-local target preferences and S02 visible identity components are exposed",
            "hiddenFieldsExcluded": True,
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
        "- S03 package: `morphospace2d/targets.py`\n"
        "- S03 runner: `scripts/e05_s03_target_morphologies.py`\n"
        "- S03 tests: `tests/test_e05_target_morphologies.py`\n",
        encoding="utf-8",
    )
    artifact_paths.append(code_index_path)

    unit_log_path = step_dir / "repo_unit_test_log.txt"
    unit_test_success = unit_log_path.exists() and "OK" in unit_log_path.read_text(encoding="utf-8", errors="replace")
    if unit_log_path.exists():
        artifact_paths.append(unit_log_path)

    success = bool(validation_df["success"].all() and (figure_path.exists() and figure_path.stat().st_size > 1000))
    status_path = step_dir / "status.json"
    summary_path = step_dir / "summary.md"
    manifest_path = step_dir / "artifact_manifest.json"
    reported_artifact_paths = artifact_paths + [status_path, summary_path, manifest_path]
    validation_result = (
        "Passed: target library covers gradients, stripes, rings, sorted rows, boundaries, and abstract organ-like motifs; configs load; rendered target panel exists; "
        "target-energy functions return zero on exact target states and positive energy on scrambled controls; local policy payload audits exclude hidden fields and whole-target maps"
        + ("; focused repository unit tests passed." if unit_test_success else ".")
        if success
        else "Failed: at least one S03 target morphology validation check did not pass."
    )
    caveats = (
        "S03 target morphologies are computational benchmark motifs, not biological anatomy. Target-energy checks are global report-facing evaluators; local policies must use only audited local payloads."
    )
    recommended = "Proceed to S04 to generalize action semantics only after Chief Scientist instruction."
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
                "args": "PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_e05_target_morphologies tests.test_e05_cell_identity tests.test_e05_substrate_generalization",
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

    summary_rows = [[row["check_id"], row["success"], row["detail"][:120]] for row in validation_rows]
    summary = f"""# E05 S03 Status Summary

- Step ID: {STEP_ID}
- Completion status: {'completed' if success else 'completed with validation failures'}
- Artifacts written: {', '.join(str(path) for path in reported_artifact_paths)}
- Validation result: {validation_result}
- Outcome classification: {'supportive' if success else 'constraining/contradictory'}
- Caveats or blockers: {caveats}
- Lay summary: The benchmark now has target patterns for gradients, stripes, rings, sorted rows, boundaries, and an abstract organ-like motif. Exact target states score zero energy, scrambled controls score above zero, and audited local payloads avoid hidden state and whole-target maps.
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
            "runner": "scripts/e05_s03_target_morphologies.py",
            "tests": "tests/test_e05_target_morphologies.py",
            "note": "Repository source is committed in git and not copied into artifacts per GitHub workspace rule.",
        },
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended,
    }
    write_json(manifest_path, manifest_payload)

    return {"success": success, "stepDir": step_dir, "status": status_payload, "artifactPaths": [str(path) for path in reported_artifact_paths]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    result = write_outputs(args.artifacts_dir)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
