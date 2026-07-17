"""Post-freeze S11 audit of the positional inversion-distance calculation.

This module does not alter assignments or rerun continuations.  It regenerates
the frozen deterministic assignments, verifies their hashes, and computes the
intended i<j positional inversion metric after a permutation-invariant helper
was detected in the sealed S11 implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


if __name__ == "__main__":
    run_audit()
