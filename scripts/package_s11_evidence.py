#!/usr/bin/env python3
"""Package compact S11 validation, provenance, and stable claim evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.frozen_cell_results import (  # noqa: E402
    C_PROFILE,
    CONFIRM_SPLIT,
    PAPER_SPLIT,
    R_PROFILE,
    canonical_json_bytes,
    primary_contrasts,
    preregistration_sha256,
    sha256_file,
    validate_run_table,
    write_json,
)


OUTPUT = Path("/artifacts/research_steps/S11")
RESULTS = Path("/artifacts/results/replication_summary.parquet")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    frame = pd.read_parquet(OUTPUT / "frozen_cell_results.parquet")
    primary = pd.read_csv(OUTPUT / "primary_architecture_contrasts.csv")
    reported = pd.read_csv(OUTPUT / "reported_mean_comparison.csv")
    replay = json.loads((OUTPUT / "exact_replay_samples.json").read_text())
    toys = json.loads((OUTPUT / "fault_semantics_toys.json").read_text())
    pre = json.loads((OUTPUT / "source_checksum_pre_run.json").read_text())
    post = json.loads((OUTPUT / "source_checksum_post_run.json").read_text())
    junit_root = ET.parse(OUTPUT / "repository_tests.junit.xml").getroot()
    junit = junit_root if junit_root.tag == "testsuite" else junit_root.find("testsuite")
    if junit is None:
        raise ValueError("JUnit report has no testsuite")

    accounting = (
        frame.groupby(["backend_profile", "split", "placement_profile"], sort=True)
        .agg(
            runs=("run_id", "size"),
            conditions=("condition_id", "nunique"),
            pairing_blocks=("pairing_block_id", "nunique"),
            completed=("completed", "sum"),
            censored=("censored", "sum"),
        )
        .reset_index()
    )
    accounting.to_csv(OUTPUT / "run_accounting.csv", index=False)

    availability_rows = []
    fields = {
        "final_values": ("observed", "observed", True, "unique-value endpoint"),
        "monotonicity_error": ("derived_exact", "derived_exact", True, "same adjacent-inversion formula"),
        "completion": ("observed_R_terminal", "derived_endpoint", False, "C native stop is no_cells_should_move"),
        "stop_reason": ("observed", "observed", False, "different backend stopping profiles"),
        "accepted_swaps": ("observed", "successful_recorded_swaps", True, "C sequence excludes non-swaps"),
        "logical_value_comparisons": ("observed", "unavailable", False, "C source probe is not a logical comparison count"),
        "observation_reads": ("observed", "unavailable", False, "not serialized by C"),
        "target_calculations": ("derived_exact", "unavailable", False, "C proposals unavailable"),
        "proposals": ("observed", "unavailable", False, "C activations unavailable"),
        "rejected_actions": ("derived_exact", "unavailable", False, "C rejection/conflict outcomes unavailable"),
        "memory_updates": ("observed", "unavailable", False, "C Selection cursor updates unavailable"),
        "activations": ("observed", "unavailable", False, "C sequence basis is recorded_swap"),
        "displacement": ("observed", "derived_successful_swap", True, "two endpoints per successful unique-value swap"),
        "native_state_hash": ("observed", "unavailable", False, "C has observable final hash only"),
        "fault_identity": ("observed", "adapter_input_only", False, "C events do not record actor/target identities"),
        "stuck_fault": ("observed", "implementation_absent", False, "no distinct stuck C mode"),
        "traditional_architecture": ("clean_room_control", "generator_absent", False, "not recovered historically"),
        "wall_time": ("observed_descriptive", "observed_descriptive", False, "backend/runtime specific"),
    }
    for field, (r_status, c_status, comparable, boundary) in fields.items():
        availability_rows.append(
            {
                "field": field,
                "reference_status": r_status,
                "historical_status": c_status,
                "directly_comparable": comparable,
                "boundary": boundary,
            }
        )
    pd.DataFrame(availability_rows).to_csv(OUTPUT / "backend_field_availability.csv", index=False)

    rerun_a = primary_contrasts(frame)
    rerun_b = primary_contrasts(frame)
    stable_columns = list(primary.columns)
    bootstrap_exact = canonical_json_bytes(rerun_a[stable_columns].to_dict(orient="records")) == canonical_json_bytes(rerun_b[stable_columns].to_dict(orient="records"))
    published_primary_match = np.allclose(
        rerun_a.sort_values(["policy", "fault_mode", "requested_fault_count"])[
            ["mean_cell_view_minus_traditional", "adjusted_ci_low", "adjusted_ci_high"]
        ].to_numpy(),
        primary.sort_values(["policy", "fault_mode", "requested_fault_count"])[
            ["mean_cell_view_minus_traditional", "adjusted_ci_low", "adjusted_ci_high"]
        ].to_numpy(),
        equal_nan=True,
    )
    means_recomputed = True
    for row in reported[reported.n.gt(0)].to_dict(orient="records"):
        selected = frame[
            frame.backend_profile.eq(row["backend_profile"])
            & frame.split.eq(PAPER_SPLIT)
            & frame.placement_profile.eq(row["placement_profile"])
            & frame.fault_mode.eq(row["fault_mode"])
            & frame.policy.eq(row["policy"])
            & frame.architecture.eq(row["architecture"])
            & frame.requested_fault_count.eq(row["requested_fault_count"])
        ]
        means_recomputed &= bool(np.isclose(selected.monotonicity_error.mean(), row["simulated_mean"]))

    run_validation = validate_run_table(frame)
    checks = {
        "runTable": run_validation["success"],
        "faultSemanticsToys": toys["success"] and toys["passed"] == 6,
        "exactReplay36": len(replay) == 36 and all(row["byteExactDigestReplay"] and row["fastSummaryMatchesDigest"] for row in replay),
        "sourcePreChecksum": pre["contentValid"] and pre["filesChecked"] == 91,
        "sourcePostChecksum": post["contentValid"] and post["filesChecked"] == 91,
        "sourceCommitStable": pre["expectedCommit"] == post["expectedCommit"] == "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
        "primaryPairing": len(primary) == 18 and primary.pairs.eq(1000).all(),
        "bootstrapDeterministic": bootstrap_exact,
        "primaryArtifactRecomputed": published_primary_match,
        "reportedMeansRecomputed": means_recomputed,
        "zeroCensoring": int(frame.censored.sum()) == 0,
        "onlyAllowedSplits": set(frame.split) == {PAPER_SPLIT, CONFIRM_SPLIT},
        "legacyConfirmatoryUnopened": not bool(
            (frame.split.eq(CONFIRM_SPLIT) & frame.placement_profile.eq("legacy_with_replacement")).any()
        ),
        "repositoryTests68": int(junit.attrib.get("tests", 0)) == 68
        and int(junit.attrib.get("failures", 0)) == 0
        and int(junit.attrib.get("errors", 0)) == 0,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    validation = {
        "schemaVersion": "e01.s11.validation.v1",
        "researchStepId": "S11",
        "success": all(checks.values()),
        "checks": checks,
        "runValidation": run_validation,
        "primaryContrasts": len(primary),
        "exactReplaySamples": len(replay),
        "repositoryTests": {
            "tests": int(junit.attrib.get("tests", 0)),
            "failures": int(junit.attrib.get("failures", 0)),
            "errors": int(junit.attrib.get("errors", 0)),
        },
        "faultSemanticsToys": toys,
    }
    write_json(OUTPUT / "validation_summary.json", validation)

    missing = {
        "researchStepId": "S11",
        "frozenPublicCommit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
        "publicationSnapshotClaimed": False,
        "missingEvidence": [
            "exact publication commit/tag/release",
            "Figure 5 raw replicate arrays and original fault maps/random streams",
            "Figure 5 error-bar definition, placement rule, and stopping rule",
            "historical traditional generator and fault semantics",
            "distinct historical stuck-fault implementation",
            "historical activations, actor/target identities, observations, logical comparisons, proposals, rejections, cursor updates, and native state hashes",
            "historical dependency lock and portable timing environment",
        ],
        "nonSubstitutionDecision": "No regenerated reference result is represented as a missing historical file or stream.",
    }
    write_json(OUTPUT / "missing_evidence.json", missing)

    inputs = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"),
        Path("/artifacts/research_steps/S01/claim_registry.parquet"),
        Path("/artifacts/research_steps/S03/transition_contract.json"),
        Path("/artifacts/research_steps/S04/patch_ledger.json"),
        Path("/artifacts/research_steps/S06/event_schema.json"),
        Path("/artifacts/research_steps/S07/invariant_coverage_matrix.csv"),
        Path("/artifacts/research_steps/S08/paired_scenario_bank.parquet"),
        Path("/artifacts/research_steps/S09/no_fault_runs.parquet"),
        Path("/artifacts/research_steps/S10/cost_ledger.parquet"),
        ROOT / "analysis/s11_frozen_cell_preregistration.json",
    ]
    provenance = {
        "schemaVersion": "e01.s11.provenance.v1",
        "researchStepId": "S11",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryHeadBeforeS11Commit": git("rev-parse", "HEAD"),
        "repositoryBranch": git("branch", "--show-current"),
        "preregistrationSha256": preregistration_sha256(),
        "inputs": [file_record(path) for path in inputs if path.exists()],
        "cacheCheckpoints": [
            file_record(Path("/cache/e01_s11/reference_paper.jsonl")),
            file_record(Path("/cache/e01_s11/reference_confirmatory.jsonl")),
            file_record(Path("/cache/e01_s11/historical_paper.jsonl")),
        ],
        "sourceCode": [
            "analysis/frozen_cell_results.py",
            "analysis/s11_frozen_cell_preregistration.json",
            "scripts/run_frozen_cell_results.py",
            "scripts/s11_historical_worker.py",
            "scripts/package_s11_evidence.py",
            "reference_simulator/engine.py",
            "tests/test_frozen_cell_results.py",
        ],
    }
    write_json(OUTPUT / "provenance.json", provenance)

    existing = pd.read_parquet(RESULTS)
    reported_index = reported[
        reported.backend_profile.eq(R_PROFILE)
        & reported.placement_profile.eq("reference_without_replacement")
        & reported.architecture.eq("cell_view")
    ].set_index(["policy", "fault_mode", "requested_fault_count"])
    additions = []
    for row in primary.to_dict(orient="records"):
        claim_id = (
            f"F05-{'B-PASSIVE' if row['fault_mode'] == 'passive' else 'C-STUCK'}-"
            f"{row['policy'].upper()}-F{row['requested_fault_count']}-CELL_VIEW"
        )
        endpoint = reported_index.loc[(row["policy"], row["fault_mode"], row["requested_fault_count"]), "classification"]
        additions.append(
            {
                "research_step_id": "S11",
                "figure": 5,
                "claim_id": claim_id,
                "endpoint_classification": endpoint,
                "qualitative_classification": row["claim_classification"],
                "outcome_classification": row["claim_classification"],
                "estimate": row["mean_cell_view_minus_traditional"],
                "adjusted_ci_low": row["adjusted_ci_low"],
                "adjusted_ci_high": row["adjusted_ci_high"],
                "analysis_split": CONFIRM_SPLIT,
            }
        )
    stable = pd.concat([existing[existing.research_step_id.ne("S11")], pd.DataFrame(additions)], ignore_index=True)
    stable.to_parquet(RESULTS, index=False, compression="zstd")

    commands = {
        "researchStepId": "S11",
        "threadEnvironment": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "commands": [
            "python scripts/run_frozen_cell_results.py validate",
            "python scripts/run_frozen_cell_results.py reference-paper --workers 8",
            "python scripts/run_frozen_cell_results.py historical-paper --workers 8",
            "python scripts/run_frozen_cell_results.py reference-confirmatory --workers 8",
            "python scripts/run_frozen_cell_results.py replay --workers 8",
            "python scripts/run_frozen_cell_results.py analyze",
            "python scripts/package_s11_evidence.py",
            "PYTHONPATH=. pytest -q",
        ],
    }
    write_json(OUTPUT / "commands.json", commands)
    release = {
        "researchStepId": "S11",
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "remote": git("remote", "get-url", "origin"),
        "publicationSnapshotClaimed": False,
    }
    write_json(OUTPUT / "repository_release.json", release)
    artifact_paths = sorted(
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    manifest = {
        "schemaVersion": "e01.s11.artifact_manifest.v1",
        "researchStepId": "S11",
        "artifactCount": len(artifact_paths),
        "artifacts": [file_record(path) for path in artifact_paths],
    }
    write_json(OUTPUT / "artifact_manifest.json", manifest)
    print(json.dumps({"success": validation["success"], "runs": len(frame), "stableClaimsAdded": len(additions)}, indent=2))
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
