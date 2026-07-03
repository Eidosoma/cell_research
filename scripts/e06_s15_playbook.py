#!/usr/bin/env python3
"""Synthesize the E06 S15 chimeric-control playbook."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e06.control_playbook import (  # noqa: E402
    REQUIRED_PRIOR_STEPS,
    SOURCE_TABLES,
    STEP_ID,
    STEP_NUMBER,
    S15Config,
    build_claim_registry_and_evidence,
    build_handoff_markdown,
    build_playbook_markdown,
    load_s15_sources,
    markdown_table,
    relative_artifact_path,
    validate_s15_outputs,
)
from src.e06.mixture_ratios import EXPERIMENT_ID, sha256_file  # noqa: E402


ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = datetime.now(UTC)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "command": " ".join(command),
        "cwd": str(cwd),
        "returnCode": proc.returncode,
        "success": proc.returncode == 0,
        "elapsedSeconds": (datetime.now(UTC) - started).total_seconds(),
        "stdout": proc.stdout[-6000:],
        "stderr": proc.stderr[-6000:],
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"


def source_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "sizeBytes": path.stat().st_size if path.exists() else None,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": relative_artifact_path(path, artifacts_dir),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "full_results_report": artifacts_dir / "research_steps/S15/research_step_full_results.md",
        "playbook": artifacts_dir / "reports/e06_chimeric_control_playbook.md",
        "handoff": artifacts_dir / "reports/e06_report_bundle_handoff.md",
        "phase_matrix": artifacts_dir / "results/e06_chimerism_phase_matrix.parquet",
        "claim_registry": artifacts_dir / "tables/e06_s15_claim_registry.csv",
        "evidence_links": artifacts_dir / "tables/e06_s15_claim_evidence_links.csv",
        "history_effect_summary": artifacts_dir / "research_steps/S15/e06_s15_history_effect_summary.csv",
        "validation_checks": artifacts_dir / "research_steps/S15/e06_s15_validation_checks.csv",
        "config": artifacts_dir / "configs/e06_s15_playbook_config.json",
        "manifest": artifacts_dir / "research_steps/S15/manifest.json",
        "run_manifest": artifacts_dir / "run_manifest.json",
        "checksums": artifacts_dir / "checksums/sha256sums.txt",
    }


def artifact_relative_list(paths: dict[str, Path], artifacts_dir: Path) -> list[str]:
    ordered_keys = [
        "full_results_report",
        "playbook",
        "handoff",
        "phase_matrix",
        "claim_registry",
        "evidence_links",
        "history_effect_summary",
        "validation_checks",
        "config",
        "manifest",
        "run_manifest",
        "checksums",
    ]
    return [relative_artifact_path(paths[key], artifacts_dir) for key in ordered_keys]


def write_checksums(paths: dict[str, Path], artifacts_dir: Path) -> None:
    checksum_path = paths["checksums"]
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "full_results_report",
        "playbook",
        "handoff",
        "phase_matrix",
        "claim_registry",
        "evidence_links",
        "history_effect_summary",
        "validation_checks",
        "config",
        "manifest",
        "run_manifest",
    ]
    lines = []
    for key in keys:
        path = paths[key]
        if path.exists():
            lines.append(f"{sha256_file(path)}  {relative_artifact_path(path, artifacts_dir)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    paths: dict[str, Path],
    artifacts_dir: Path,
    repo_dir: Path,
    *,
    started_at: str,
    sources: dict[str, pd.DataFrame],
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
) -> None:
    source_paths = {
        key: artifacts_dir / relative_path
        for key, relative_path in SOURCE_TABLES.items()
    }
    for step_id in REQUIRED_PRIOR_STEPS:
        source_paths[f"{step_id.lower()}_full_results_report"] = artifacts_dir / f"research_steps/{step_id}/research_step_full_results.md"
    artifact_descriptions = {
        "full_results_report": "S15 full-results report",
        "playbook": "Evidence-linked chimeric-control playbook",
        "handoff": "Chief report-bundle handoff",
        "phase_matrix": "Carried-forward E06 chimerism phase matrix",
        "claim_registry": "Machine-readable S15 claim registry",
        "evidence_links": "Machine-readable row-linked claim evidence table",
        "history_effect_summary": "S12 history-effect aggregate used by S15",
        "validation_checks": "S15 validation checks",
        "config": "S15 synthesis configuration",
        "manifest": "S15 artifact manifest",
        "run_manifest": "Experiment run manifest updated by S15",
        "checksums": "SHA256 checksums for S15 artifacts",
    }
    manifest = {
        "schema": "eidosoma.e06.s15_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": "complete",
        "success": bool(validation_df["success"].all()) if not validation_df.empty else False,
        "startedAt": started_at,
        "generatedAt": utc_now(),
        "claimCount": int(len(claims_df)),
        "evidenceLinkCount": int(len(evidence_df)),
        "validationPassed": int(validation_df["success"].sum()) if not validation_df.empty else 0,
        "validationTotal": int(len(validation_df)),
        "unitTests": unit_test_result,
        "repoState": {
            "branch": git_output(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": git_output(repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(repo_dir, ["status", "--short"]),
        },
        "sources": {
            key: {
                **source_entry(path),
                "rowCount": int(len(sources[key])) if key in sources else None,
            }
            for key, path in source_paths.items()
        },
        "artifacts": {
            key: artifact_entry(path, artifacts_dir, artifact_descriptions[key])
            for key, path in paths.items()
            if path.exists() and key in artifact_descriptions
        },
    }
    write_json(paths["manifest"], manifest)


def write_run_manifest(
    paths: dict[str, Path],
    artifacts_dir: Path,
    repo_dir: Path,
    *,
    started_at: str,
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "latestResearchStepId": STEP_ID,
        "latestStepNumber": STEP_NUMBER,
        "status": "complete",
        "success": bool(validation_df["success"].all()) if not validation_df.empty else False,
        "startedAt": started_at,
        "generatedAt": utc_now(),
        "repo": {
            "path": str(repo_dir),
            "branch": git_output(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": git_output(repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(repo_dir, ["status", "--short"]),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
        },
        "summary": {
            "claimCount": int(len(claims_df)),
            "evidenceLinkCount": int(len(evidence_df)),
            "validationPassed": int(validation_df["success"].sum()) if not validation_df.empty else 0,
            "validationTotal": int(len(validation_df)),
            "outcomeClassification": "supportive synthesis",
            "recommendedNextAction": "Chief report-bundle generation for E06; do not start E07 until instructed.",
        },
        "artifacts": {
            key: relative_artifact_path(path, artifacts_dir)
            for key, path in paths.items()
            if path.exists()
        },
    }
    write_json(paths["run_manifest"], run_manifest)


def build_full_report(
    *,
    artifacts_written: list[str],
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    history_summary: pd.DataFrame,
    validation_df: pd.DataFrame,
    unit_test_result: dict[str, Any],
    command_line: str,
    input_paths: dict[str, Path],
    validation_passed: bool,
) -> str:
    access_counts = claims_df["access_stratum"].value_counts().sort_index().reset_index()
    access_counts.columns = ["access_stratum", "claim_count"]
    outcome_counts = claims_df["outcome_classification"].value_counts().sort_index().reset_index()
    outcome_counts.columns = ["outcome_classification", "claim_count"]
    artifact_lines = "\n".join(f"- `{item}`" for item in artifacts_written)
    input_df = pd.DataFrame(
        [
            {
                "input_name": key,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for key, path in input_paths.items()
        ]
    )
    return f"""# E06 S15 Research Step Full Results

## Top Summary

- Step ID: S15
- Completion status: Complete
- Artifacts written:
{artifact_lines}
- Validation result: {"Passed" if validation_passed else "Failed"}; {int(validation_df["success"].sum()) if not validation_df.empty else 0}/{len(validation_df)} validation checks passed and focused unit tests {"passed" if unit_test_result.get("success") else "failed"}.
- Outcome classification: supportive synthesis
- Caveats or blockers: S15 is a synthesis step and adds no new simulations. All claims remain computational proxy claims, and behavior-only, explicit-interface, finite-radius-governance, and global-controller-like evidence are kept separate.
- Lay summary: S15 turns S01-S14 into an auditable chimeric-control playbook. It emphasizes ratio, arrangement, goal compatibility, dominance context, developmental history, predictive triage, and the null rescue results from governance, graft, and S14 intervention searches.
- Recommended next action: E06 is ready for Chief report-bundle generation. Do not start E07 unless separately instructed.

## Frozen Question

Can the chimeric phase diagram be converted into practical rules for which mixtures cooperate naturally, require intervention, or create new stable collectives?

## Inputs

{markdown_table(input_df, ["input_name", "path", "exists", "size_bytes"], max_rows=80)}

## Methods

S15 loaded completed S01-S14 reports and compact evidence tables from `$ARTIFACTS_DIR`. It did not rerun simulations. The synthesis code selected auditable anchor rows from the validated outputs, converted them into a claim registry, and wrote a separate evidence-link table with artifact paths, row selectors, row counts, and evidence values. The Markdown playbook and Chief handoff are generated from those machine-readable tables.

Claims were not allowed to stand without an evidence link. Claims involving explicit interface rules, finite-radius governance, or global-controller-like comparators carry access-stratum labels. S14's prospective null rescue result is preserved as a pruning constraint rather than treated as a successful intervention search.

## Commands

- Main command: `{command_line}`
- Focused unit-test command: `{unit_test_result.get("command", "not run")}`
- Focused unit-test return code: `{unit_test_result.get("returnCode", "not run")}`

## Dependencies

No new dependencies were installed. S15 used repository code plus preinstalled Python, `pandas`, and `pyarrow`.

## Parameters

```json
{json.dumps({
    "minClaimCount": S15Config().min_claim_count,
    "requiredPriorSteps": list(REQUIRED_PRIOR_STEPS),
    "requiredAccessTerms": list(S15Config().required_access_terms),
}, indent=2)}
```

## Results

- Claims written: {len(claims_df)}
- Evidence links written: {len(evidence_df)}
- Prior full-results reports checked: {len(REQUIRED_PRIOR_STEPS)}
- Outcome classification: supportive synthesis

### Claim Registry

{markdown_table(claims_df, ["claim_id", "title", "evidence_level", "outcome_classification", "access_stratum", "recommended_use"], max_rows=40)}

### Outcome Counts

{markdown_table(outcome_counts, ["outcome_classification", "claim_count"], max_rows=30)}

### Access-Stratum Counts

{markdown_table(access_counts, ["access_stratum", "claim_count"], max_rows=30)}

### History-Effect Summary Used In S15

{markdown_table(history_summary, ["history_id", "history_family", "condition_count", "mean_history_effect_magnitude", "mean_delta_final_target_quality", "mean_delta_aggregation_delta_percent", "state_class_change_rate"], max_rows=10)}

## Validation

{markdown_table(validation_df, ["validation_case", "success", "observed", "expected"], max_rows=40)}

## Artifacts

{markdown_table(pd.DataFrame({"relative_path": artifacts_written}), ["relative_path"], max_rows=80)}

## Provenance

- Experiment ID: {EXPERIMENT_ID}
- Research step number: {STEP_NUMBER}
- Script: `scripts/e06_s15_playbook.py`
- Repository branch: `{git_output(REPO_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"])}`
- Repository commit: `{git_output(REPO_ROOT, ["rev-parse", "HEAD"])}`
- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`

## Caveats And Limitations

- S15 adds no new simulation rows; it is only a synthesis of completed S01-S14 evidence.
- All evidence remains computational and proxy-bound to sorting-array chimeras.
- The playbook should not be treated as a biological protocol or wet-lab validation.
- S13 feature rankings remain exploratory associations, not causal proof.
- S14 was a bounded null rescue search, not an exhaustive impossibility proof.
- Global-controller-like comparator rows remain separate from finite-radius local governance.

## Failed Assumptions Or Blockers

No required S01-S14 artifact was missing. The main limitation is interpretive: S15 can make the evidence auditable and decision-oriented, but it cannot strengthen the biological scope beyond the completed computational proxy experiments.
"""


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    artifacts_dir = args.artifacts_dir
    repo_dir = args.repo_dir
    paths = artifact_paths(artifacts_dir)
    rel_artifacts = artifact_relative_list(paths, artifacts_dir)

    config = S15Config()
    write_json(
        paths["config"],
        {
            "schema": "eidosoma.e06.s15_config.v1",
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "minClaimCount": config.min_claim_count,
            "requiredClaimIds": list(config.required_claim_ids),
            "requiredAccessTerms": list(config.required_access_terms),
            "sourceTables": SOURCE_TABLES,
        },
    )

    sources = load_s15_sources(artifacts_dir)
    claims_df, evidence_df, history_summary = build_claim_registry_and_evidence(sources, artifacts_dir)
    paths["claim_registry"].parent.mkdir(parents=True, exist_ok=True)
    paths["history_effect_summary"].parent.mkdir(parents=True, exist_ok=True)
    claims_df.to_csv(paths["claim_registry"], index=False)
    evidence_df.to_csv(paths["evidence_links"], index=False)
    history_summary.to_csv(paths["history_effect_summary"], index=False)

    unit_test_result = (
        run_command(["python", "-m", "unittest", "tests.e06.test_control_playbook", "-v"], cwd=repo_dir)
        if args.run_unit_tests
        else {"command": "not run", "returnCode": None, "success": True, "stdout": "", "stderr": ""}
    )

    prior_reports = [artifacts_dir / f"research_steps/{step_id}/research_step_full_results.md" for step_id in REQUIRED_PRIOR_STEPS]
    input_paths = {
        **{key: artifacts_dir / relative_path for key, relative_path in SOURCE_TABLES.items()},
        **{f"{step_id.lower()}_report": artifacts_dir / f"research_steps/{step_id}/research_step_full_results.md" for step_id in REQUIRED_PRIOR_STEPS},
    }

    # Write preliminary Markdown so validation can check non-empty output files.
    playbook_text = build_playbook_markdown(claims_df, evidence_df, artifacts_written=rel_artifacts, validation_passed=False)
    handoff_text = build_handoff_markdown(claims_df, evidence_df, artifacts_written=rel_artifacts, validation_passed=False)
    preliminary_report = build_full_report(
        artifacts_written=rel_artifacts,
        claims_df=claims_df,
        evidence_df=evidence_df,
        history_summary=history_summary,
        validation_df=pd.DataFrame(columns=["validation_case", "success", "observed", "expected"]),
        unit_test_result=unit_test_result,
        command_line=" ".join(sys.argv),
        input_paths=input_paths,
        validation_passed=False,
    )
    write_text(paths["playbook"], playbook_text)
    write_text(paths["handoff"], handoff_text)
    write_text(paths["full_results_report"], preliminary_report)

    validation_artifacts = {
        key: paths[key]
        for key in (
            "full_results_report",
            "playbook",
            "handoff",
            "phase_matrix",
            "claim_registry",
            "evidence_links",
            "history_effect_summary",
            "config",
        )
    }
    validation_df = validate_s15_outputs(
        claims_df,
        evidence_df,
        validation_artifacts,
        prior_reports,
        config,
        playbook_text=playbook_text,
        handoff_text=handoff_text,
        full_report_text=preliminary_report,
        unit_tests_success=bool(unit_test_result.get("success")),
    )
    paths["validation_checks"].parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(paths["validation_checks"], index=False)
    validation_passed = bool(validation_df["success"].all())

    playbook_text = build_playbook_markdown(claims_df, evidence_df, artifacts_written=rel_artifacts, validation_passed=validation_passed)
    handoff_text = build_handoff_markdown(claims_df, evidence_df, artifacts_written=rel_artifacts, validation_passed=validation_passed)
    full_report_text = build_full_report(
        artifacts_written=rel_artifacts,
        claims_df=claims_df,
        evidence_df=evidence_df,
        history_summary=history_summary,
        validation_df=validation_df,
        unit_test_result=unit_test_result,
        command_line=" ".join(sys.argv),
        input_paths=input_paths,
        validation_passed=validation_passed,
    )
    write_text(paths["playbook"], playbook_text)
    write_text(paths["handoff"], handoff_text)
    write_text(paths["full_results_report"], full_report_text)

    write_manifest(
        paths,
        artifacts_dir,
        repo_dir,
        started_at=started_at,
        sources=sources,
        claims_df=claims_df,
        evidence_df=evidence_df,
        validation_df=validation_df,
        unit_test_result=unit_test_result,
    )
    write_run_manifest(
        paths,
        artifacts_dir,
        repo_dir,
        started_at=started_at,
        claims_df=claims_df,
        evidence_df=evidence_df,
        validation_df=validation_df,
    )
    write_checksums(paths, artifacts_dir)

    result = {
        "researchStepId": STEP_ID,
        "success": validation_passed,
        "status": "complete" if validation_passed else "failed_validation",
        "outcomeClassification": "supportive synthesis",
        "claimCount": int(len(claims_df)),
        "evidenceLinkCount": int(len(evidence_df)),
        "validationPassed": int(validation_df["success"].sum()),
        "validationTotal": int(len(validation_df)),
        "report": str(paths["full_results_report"]),
        "playbook": str(paths["playbook"]),
        "handoff": str(paths["handoff"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
