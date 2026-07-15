#!/usr/bin/env python3
"""Independent compact validation of the final E03 S02 replay artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/artifacts/research_steps/S02"))
    args = parser.parse_args()
    table = pq.read_table(args.input / "replayed_distances.parquet")
    frame = table.to_pandas()
    diagnostics = json.loads((args.input / "replay_diagnostics.json").read_text())
    coverage = json.loads((args.input / "trace_coverage_manifest.json").read_text())
    immutability = json.loads((args.input / "source_immutability.json").read_text())

    checks: dict[str, Any] = {}
    checks["schema_version"] = set(frame["schema_version"]) == {"e03.s02.replayed_distance.v1"}
    checks["research_step_id"] = set(frame["research_step_id"]) == {"S02"}
    checks["row_count"] = len(frame) == int(diagnostics["outputRows"])
    checks["logical_trace_count"] = frame["logical_trace_id"].nunique() == 85
    checks["primary_coverage_accounting"] = (
        coverage["primaryLogicalTraces"] == 90
        and coverage["alignedPrimaryLogicalTraces"] == 84
        and coverage["coverageGapLogicalTraces"] == 6
        and coverage["supplementalAlignedLogicalTraces"] == 1
    )
    checks["source_immutability"] = bool(immutability["success"])
    checks["diagnostic_success"] = diagnostics["success"] and all(
        row["success"] for row in diagnostics["diagnostics"]
    )
    required_metrics = [
        "adjacent_descents", "normalized_adjacent_descents", "paper_sortedness_strict",
        "paper_sortedness_distance", "inversion_count", "maximum_inversion_count",
        "normalized_kendall_distance", "spearman_footrule",
        "maximum_spearman_footrule", "normalized_spearman_footrule",
        "maximum_rank_error", "maximum_possible_rank_error",
        "normalized_maximum_rank_error", "duplicate_aware_earth_movers_distance",
        "maximum_duplicate_aware_earth_movers_distance",
        "normalized_duplicate_aware_earth_movers_distance",
    ]
    checks["metric_fields_complete"] = not frame[required_metrics].isna().any().any()
    normalized = [
        "normalized_adjacent_descents", "paper_sortedness_strict",
        "paper_sortedness_distance", "normalized_kendall_distance",
        "normalized_spearman_footrule", "normalized_maximum_rank_error",
        "normalized_duplicate_aware_earth_movers_distance",
    ]
    checks["normalized_bounds"] = all(
        frame[column].between(0.0, 1.0, inclusive="both").all() for column in normalized
    )
    checks["paper_complement_identity"] = (
        (frame["paper_sortedness_strict"] + frame["paper_sortedness_distance"] - 1.0)
        .abs().max() <= 1e-12
    )
    checks["footrule_emd_identity"] = (
        (frame["spearman_footrule"] / frame["n"] - frame["duplicate_aware_earth_movers_distance"])
        .abs().max() <= 1e-12
        and (frame["normalized_spearman_footrule"] - frame["normalized_duplicate_aware_earth_movers_distance"])
        .abs().max() <= 1e-12
    )
    duplicate_key = [
        "logical_trace_id", "direction", "activation_count", "checkpoint_kind",
        "accepted_swap_index",
    ]
    checks["checkpoint_key_unique"] = not frame.duplicated(duplicate_key).any()
    diagnostic_by_trace = {
        row["logicalTraceId"]: row for row in diagnostics["diagnostics"]
    }
    scheduler_ok = True
    swap_ok = True
    final_ok = True
    for (trace_id, direction), rows in frame.groupby(["logical_trace_id", "direction"]):
        rows = rows.sort_values(["activation_count", "checkpoint_kind"])
        diagnostic = diagnostic_by_trace[trace_id]
        if diagnostic["activationCount"] is not None:
            scheduler_ok &= int(rows["scheduler_checkpoint_count"].sum()) == int(
                diagnostic["activationCount"]
            )
        accepted = rows.loc[rows["checkpoint_kind"] == "accepted_action", "accepted_swap_index"]
        if diagnostic["acceptedSwaps"] is not None:
            swap_ok &= accepted.tolist() == list(range(1, int(diagnostic["acceptedSwaps"]) + 1))
        final_rows = rows.loc[rows["activation_count"] == rows["activation_count"].max()]
        final_ok &= len(final_rows) == 1
    checks["scheduler_checkpoint_completeness"] = bool(scheduler_ok)
    checks["accepted_action_completeness"] = bool(swap_ok)
    checks["one_final_metric_state_per_direction"] = bool(final_ok)
    checks["all_gap_reason_codes_explicit"] = (
        len(coverage["gaps"]) == 6
        and all(row.get("reasonCode") for row in coverage["gaps"])
    )
    checks = {key: bool(value) for key, value in checks.items()}
    success = all(checks.values())
    result = {
        "schemaVersion": "e03.s02.validation.v1",
        "researchStepId": "S02",
        "success": success,
        "rows": len(frame),
        "logicalTraces": int(frame["logical_trace_id"].nunique()),
        "checks": checks,
    }
    write_json(args.input / "validation_results.json", result)
    print(json.dumps(result, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
