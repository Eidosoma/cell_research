"""Post-freeze S11 audit of the positional inversion-distance calculation.

This module does not alter assignments or rerun continuations.  It regenerates
the frozen deterministic assignments, verifies their hashes, and computes the
intended i<j positional inversion metric after a permutation-invariant helper
was detected in the sealed S11 implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.composition_sweep import SweepCondition, materialize_sweep_scenario
from analysis.value_policy_shuffles import (
    CACHE_DIR,
    OUTPUT_DIR,
    PEAK_PROGRESS,
    _run_source,
    _source_labelled_copy,
    assert_frozen,
    canonical_hash,
    positional_sequences,
    select_policy_pair,
    select_value_pair,
    source_tasks,
    value_derangement_bank,
    write_json,
)


def positional_inversion_count(values: Sequence[int | float]) -> int:
    """Count pairs i<j for which value[i] > value[j]."""
    array = np.asarray(values)
    comparison = array[:, None] > array[None, :]
    return int(np.sum(np.triu(comparison, k=1)))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def run_audit(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict:
    freeze = assert_frozen(output)
    assignments = pd.read_parquet(output / "intervention_assignment_audit.parquet")
    indexed = assignments.set_index("scenario_id")
    rows = []
    for task in source_tasks():
        condition = SweepCondition.from_dict(task["condition"])
        source_raw, _ = materialize_sweep_scenario(condition, task["base"])
        source = _source_labelled_copy(source_raw)
        expected = task["expected_native"]
        target = math.floor(
            PEAK_PROGRESS[condition.condition_id]
            * int(expected["successful_swap_count"])
        )
        _, checkpoints = _run_source(source, {"eligible_peak": target})
        checkpoint = checkpoints["eligible_peak"]
        values, policies, strata = positional_sequences(source, checkpoint)
        policy_pair = select_policy_pair(strata, policies, source.scenario_id)
        primary_support = np.flatnonzero(
            policy_pair["primaryBinary"] != policy_pair["originalBinary"]
        )
        matched_support = np.flatnonzero(
            policy_pair["matchedBinary"] != policy_pair["originalBinary"]
        )
        value_pair = select_value_pair(
            values, primary_support, matched_support, source.scenario_id
        )
        primary_bank = value_derangement_bank(
            values, primary_support, source.scenario_id, "primary"
        )
        matched_bank = value_derangement_bank(
            values, matched_support, source.scenario_id, "matched"
        )
        primary_inversions = np.asarray(
            [positional_inversion_count(row) for row in primary_bank], dtype=np.int64
        )
        matched_inversions = np.asarray(
            [positional_inversion_count(row) for row in matched_bank], dtype=np.int64
        )
        original = np.asarray(values, dtype=np.int64)
        primary_l1 = np.asarray(
            [int(np.abs(row - original).sum()) for row in primary_bank], dtype=np.int64
        )
        matched_l1 = np.asarray(
            [int(np.abs(row - original).sum()) for row in matched_bank], dtype=np.int64
        )
        inversion_differences = np.abs(
            primary_inversions[:, None] - matched_inversions[None, :]
        )
        minimum_inversion_difference = int(inversion_differences.min())
        eligible = np.argwhere(inversion_differences == minimum_inversion_difference)
        minimum_l1_difference = int(
            min(abs(int(primary_l1[left]) - int(matched_l1[right])) for left, right in eligible)
        )
        left = int(value_pair["primaryCandidateIndex"])
        right = int(value_pair["matchedCandidateIndex"])
        chosen_inversion_difference = int(inversion_differences[left, right])
        chosen_l1_difference = abs(int(primary_l1[left]) - int(matched_l1[right]))
        expected_row = indexed.loc[source.scenario_id]
        hashes_match = bool(
            value_pair["primaryBankHash"] == expected_row.value_primary_bank_hash
            and value_pair["matchedBankHash"] == expected_row.value_matched_bank_hash
            and policy_pair["bankHash"] == expected_row.policy_bank_hash
        )
        chosen_attains_intended_lexicographic_target = bool(
            chosen_inversion_difference == minimum_inversion_difference
            and chosen_l1_difference == minimum_l1_difference
        )
        rows.append(
            {
                "condition_id": condition.condition_id,
                "input_profile": condition.input_profile,
                "policy_set_label": condition.policy_set_label,
                "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
                "scenario_id": source.scenario_id,
                "frozen_assignment_hashes_match": hashes_match,
                "positional_inversions_before": positional_inversion_count(values),
                "positional_inversions_primary": int(primary_inversions[left]),
                "positional_inversions_matched": int(matched_inversions[right]),
                "primary_inversion_change": int(primary_inversions[left])
                - positional_inversion_count(values),
                "matched_inversion_change": int(matched_inversions[right])
                - positional_inversion_count(values),
                "chosen_inversion_difference": chosen_inversion_difference,
                "minimum_available_inversion_difference": minimum_inversion_difference,
                "chosen_l1_difference": chosen_l1_difference,
                "minimum_l1_difference_at_best_inversion": minimum_l1_difference,
                "chosen_attains_intended_lexicographic_target": chosen_attains_intended_lexicographic_target,
                "continuation_or_assignment_changed": False,
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit) != 150 or not audit.frozen_assignment_hashes_match.all():
        raise AssertionError("post-freeze audit could not reproduce frozen assignments")
    _write_parquet(audit, output / "corrected_target_distance_audit.parquet")
    plotting = audit.merge(
        assignments[
            [
                "scenario_id",
                "reference_sortedness_before",
                "reference_sortedness_value_primary",
                "input_profile",
                "primary_policy_hamming",
                "immediate_policy_drop_primary",
            ]
        ],
        on=["scenario_id", "input_profile"],
        how="left",
        validate="one_to_one",
    )
    colors = np.where(
        plotting.input_profile.str.startswith("repeated"), "#e45756", "#4c78a8"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(
        plotting.primary_inversion_change,
        plotting.reference_sortedness_value_primary
        - plotting.reference_sortedness_before,
        c=colors,
        alpha=0.55,
        s=20,
    )
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("Corrected positional-inversion change")
    axes[0].set_ylabel("Reference-Sortedness change")
    axes[0].set_title("Value shuffle changes task progress")
    axes[1].scatter(
        plotting.chosen_inversion_difference,
        plotting.minimum_available_inversion_difference,
        c=colors,
        alpha=0.55,
        s=20,
    )
    limit = max(1, int(plotting.chosen_inversion_difference.max()))
    axes[1].plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Frozen chosen inversion gap")
    axes[1].set_ylabel("Minimum bank inversion gap")
    axes[1].set_title("Post-freeze matching failure")
    axes[2].scatter(
        plotting.immediate_policy_drop_primary,
        plotting.primary_policy_hamming / 100.0,
        c=colors,
        alpha=0.55,
        s=20,
    )
    axes[2].set_xlabel("Immediate corrected-adjacency drop")
    axes[2].set_ylabel("Changed-position fraction")
    axes[2].set_title("Support-qualified policy magnitude")
    fig.tight_layout()
    fig.savefig(output / "intervention_distance_and_progress.png", dpi=180)
    fig.savefig(output / "intervention_distance_and_progress.svg")
    plt.close(fig)
    intended_pass = bool(audit.chosen_attains_intended_lexicographic_target.all())
    amendment = {
        "schema": "e04.s11.validation_amendment.v1",
        "researchStepId": "S11",
        "recordedAt": datetime.now(timezone.utc).isoformat(),
        "amendmentClass": "post-freeze outcome-independent diagnostic correction; no assignment or continuation changed",
        "originalFreezeRecordSha256": hashlib.sha256(
            (output / "freeze_record.json").read_bytes()
        ).hexdigest(),
        "frozenImplementationSha256": freeze["implementationSha256"],
        "problem": "The sealed helper counted comparisons over all ordered value pairs and divided by two, making its result permutation-invariant instead of counting positional i<j inversions.",
        "correction": "Recompute true positional inversions on the frozen chosen assignments and all 128x128 candidate pairs; preserve assignments and every continuation exactly.",
        "scenarioRows": len(audit),
        "frozenAssignmentsReproduced": bool(audit.frozen_assignment_hashes_match.all()),
        "intendedLexicographicTargetPassed": intended_pass,
        "scenariosAttainingIntendedTarget": int(
            audit.chosen_attains_intended_lexicographic_target.sum()
        ),
        "medianChosenInversionDifference": float(
            audit.chosen_inversion_difference.median()
        ),
        "maximumChosenInversionDifference": int(
            audit.chosen_inversion_difference.max()
        ),
        "medianMinimumAvailableInversionDifference": float(
            audit.minimum_available_inversion_difference.median()
        ),
        "medianPrimaryInversionChange": float(audit.primary_inversion_change.median()),
        "effectiveTargetDistanceValidationPassed": intended_pass,
        "effectiveValidityPassed": False if not intended_pass else None,
        "outcomeImpact": "Factorial trajectories are unchanged and remain descriptive for the frozen assignments. Failure of the intended distance-selection rule is a constraining design deviation and prevents a fully validated separability claim.",
        "continuationOrAssignmentChanged": False,
        "auditSha256": canonical_hash(audit.to_dict(orient="records")),
    }
    write_json(output / "validation_amendment.json", amendment)
    return amendment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def finalize_artifacts(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict:
    freeze = assert_frozen(output)
    original = json.loads((output / "validation_summary.json").read_text())
    amendment = json.loads((output / "validation_amendment.json").read_text())
    effective = {
        "schema": "e04.s11.effective_validation_summary.v1",
        "researchStepId": "S11",
        "originalAnalyzerValidityPassed": bool(original["validityPassed"]),
        "sourceReplayPassed": bool(original["sourceReplay"]["passed"]),
        "statePreservationPassed": bool(original["statePreservation"]["passed"]),
        "policyEdgeHammingSupportPassed": bool(
            original["assignmentAndTargetDistance"]["passed"]
        ),
        "schedulerCouplingPassed": bool(original["schedulerCoupling"]["passed"]),
        "accountingPassed": bool(original["accounting"]["passed"]),
        "completionGatePassed": bool(original["completionGatePassed"]),
        "numericalSeparabilityGatePassed": bool(
            original["separableContributionGatePassed"]
        ),
        "correctedPositionalInversionTargetPassed": bool(
            amendment["effectiveTargetDistanceValidationPassed"]
        ),
        "effectiveValidityPassed": bool(
            original["validityPassed"]
            and amendment["effectiveTargetDistanceValidationPassed"]
        ),
        "effectiveStepSuccess": False,
        "outcomeClassification": "constraining/contradictory",
        "continuationOrAssignmentChangedByAmendment": False,
        "s12Absent": not Path("/artifacts/research_steps/S12").exists(),
    }
    write_json(output / "effective_validation_summary.json", effective)
    commands = {
        "schema": "e04.s11.commands.v2",
        "researchStepId": "S11",
        "commands": [
            "python -m pytest tests/test_e04_value_policy_shuffles.py tests/test_e04_s11_postfreeze_audit.py tests/test_e04_declustering_intervention.py tests/test_e04_policy_label_switches.py -q --junitxml=/artifacts/research_steps/S11/repository_tests.junit.xml",
            "python -m ruff check analysis/value_policy_shuffles.py analysis/s11_postfreeze_audit.py tests/test_e04_value_policy_shuffles.py tests/test_e04_s11_postfreeze_audit.py",
            "python -m py_compile analysis/value_policy_shuffles.py analysis/s11_postfreeze_audit.py",
            "python -m analysis.value_policy_shuffles freeze",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.value_policy_shuffles run --workers 8",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.value_policy_shuffles analyze",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.s11_postfreeze_audit audit",
            "python -m analysis.value_policy_shuffles report",
            "python -m analysis.value_policy_shuffles package",
            "python -m analysis.s11_postfreeze_audit finalize",
        ],
    }
    provenance = {
        "schema": "e04.s11.provenance.v2",
        "researchStepId": "S11",
        "repository": str(Path(__file__).resolve().parents[1]),
        "branch": _git_output("branch", "--show-current"),
        "frozenInterventionGitHead": freeze["gitHead"],
        "auditAndFinalizationGitHead": _git_output("rev-parse", "HEAD"),
        "contractPath": freeze["contractPath"],
        "contractSha256": freeze["contractSha256"],
        "frozenImplementationPath": freeze["implementationPath"],
        "frozenImplementationSha256": freeze["implementationSha256"],
        "auditImplementationPath": str(Path(__file__).resolve()),
        "auditImplementationSha256": _sha256_file(Path(__file__).resolve()),
        "freezeRecordSha256": _sha256_file(output / "freeze_record.json"),
        "validationAmendmentSha256": _sha256_file(
            output / "validation_amendment.json"
        ),
        "cachePath": str(cache / "value_policy_continuations.jsonl"),
        "cacheCollectible": False,
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(output / "commands.json", commands)
    write_json(output / "provenance.json", provenance)
    status = {
        "researchStepId": "S11",
        "stepNumber": 11,
        "success": False,
        "status": "complete",
        "artifactsWritten": sorted(
            path.name
            for path in output.iterdir()
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
        "validationResult": "effective validation failed: corrected positional-inversion target passed 1/150 assignments; completion gate also failed",
        "caveatsOrBlockers": [
            "Value reassignment changes task progress and is not a pure mediator intervention.",
            "Unique inputs cross S10's physical boundary through direct field reassignment.",
            "Only 1/150 frozen value pairs attained the intended corrected positional-inversion match; no assignment or continuation was changed.",
            "Value-only completion was 33.3% and joint completion was 43.3%.",
        ],
        "recommendedNextAction": "Return S11 for Chief Scientist review; do not start S12 automatically. If validated separation is required, authorize a newly frozen corrected intervention.",
    }
    write_json(output / "status.json", status)
    entries = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        entries.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema": "e04.s11.artifact_manifest.v2",
        "researchStepId": "S11",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "artifactCount": len(entries),
        "artifacts": entries,
        "originalAnalyzerValidationPassed": bool(original["validityPassed"]),
        "effectiveValidationPassed": bool(effective["effectiveValidityPassed"]),
        "outcomeClassification": "constraining/contradictory",
        "continuationOrAssignmentChangedByAmendment": False,
        "s12Started": False,
        "freeze": {
            "contractSha256": freeze["contractSha256"],
            "implementationSha256": freeze["implementationSha256"],
        },
    }
    write_json(output / "artifact_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "finalize"), nargs="?", default="audit")
    arguments = parser.parse_args()
    if arguments.command == "audit":
        run_audit()
    else:
        finalize_artifacts()
