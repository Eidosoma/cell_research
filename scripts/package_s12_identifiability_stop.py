#!/usr/bin/env python3
"""Package S12 descriptive results after the frozen model gate stops fitting."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from reference_simulator.model import canonical_json_bytes
from scripts.fit_s12_context_models import (
    OUTPUT,
    MODELS,
    add_model_fields,
    cluster_bootstrap_cells,
    evidence_layer_comparison,
    rare_event_diagnostics,
    segmentation_sensitivity,
    threshold_sensitivity,
    write_parquet,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main() -> None:
    frame = add_model_fields(
        pq.read_table(OUTPUT / "larger_n_metric_outcomes.parquet").to_pandas()
    )
    event_counts = (
        frame.groupby("metric", observed=True)
        .agg(runs=("run_id", "size"), events=("any_detour_episode", "sum"))
        .reset_index()
    )
    event_counts["nonevents"] = event_counts.runs - event_counts.events
    event_counts["identifiable_binary_occurrence"] = (
        event_counts.events.ge(25) & event_counts.nonevents.ge(25)
    )
    event_counts["gate_decision"] = np.where(
        event_counts.identifiable_binary_occurrence,
        "individually_identifiable_but_family_not_fit_after_gate",
        "not_identifiable_zero_or_rare_events",
    )
    write_parquet(
        MODELS / "identifiability_by_metric.parquet",
        event_counts,
        "e03.s12.identifiability_by_metric.v1",
    )

    model_rows = []
    for metric in event_counts.metric:
        identifiable = bool(
            event_counts.loc[event_counts.metric.eq(metric), "identifiable_binary_occurrence"].iloc[0]
        )
        for family in ("occurrence", "depth", "recovery"):
            model_rows.append(
                {
                    "model_id": f"{family}_{metric}",
                    "metric": metric,
                    "model_family": family,
                    "status": (
                        "not_fit_family_gate_enforced"
                        if identifiable
                        else "not_identifiable_zero_detour_events"
                    ),
                    "coefficients_written": False,
                    "calibration_written": False,
                    "held_out_prediction_written": False,
                    "residual_diagnostics_written": False,
                }
            )
    model_rows.append(
        {
            "model_id": "completion_all_metrics",
            "metric": "not_metric_specific",
            "model_family": "completion",
            "status": "not_fit_family_gate_enforced",
            "coefficients_written": False,
            "calibration_written": False,
            "held_out_prediction_written": False,
            "residual_diagnostics_written": False,
        }
    )
    model_status = pd.DataFrame(model_rows)
    write_parquet(
        MODELS / "model_status.parquet", model_status, "e03.s12.model_status.v1"
    )

    marginal = cluster_bootstrap_cells(frame)
    segmentation = segmentation_sensitivity(frame)
    thresholds = threshold_sensitivity(frame)
    rare = rare_event_diagnostics(frame)
    evidence = evidence_layer_comparison(frame)
    write_parquet(OUTPUT / "marginal_effects.parquet", marginal, "e03.s12.marginal_effects_descriptive.v1")
    write_parquet(OUTPUT / "episode_segmentation_sensitivity.parquet", segmentation, "e03.s12.episode_segmentation.v1")
    write_parquet(OUTPUT / "threshold_sensitivity.parquet", thresholds, "e03.s12.threshold_sensitivity.v1")
    write_parquet(OUTPUT / "rare_event_separation_diagnostics.parquet", rare, "e03.s12.rare_event_diagnostics.v1")
    write_parquet(OUTPUT / "evidence_layer_comparison.parquet", evidence, "e03.s12.evidence_layer_comparison.v1")

    blocked = {
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": False,
        "status": "predeclared_identifiability_gate_triggered",
        "artifactsWritten": [
            "context_models/design_diagnostics.json",
            "context_models/identifiability_failure.json",
            "context_models/identifiability_by_metric.parquet",
            "context_models/model_status.parquet",
            "marginal_effects.parquet",
            "episode_segmentation_sensitivity.parquet",
            "threshold_sensitivity.parquet",
            "rare_event_separation_diagnostics.parquet",
            "evidence_layer_comparison.parquet",
        ],
        "validationResult": "The 46-column supported-cell design is full rank, but three of four metric-specific occurrence outcomes have zero events; hierarchical model fitting stopped before any reduced model was fit.",
        "caveatsOrBlockers": [
            "Inversion count, Spearman footrule, and maximum-rank error have 0/96,768 running-minimum detour episodes and 0 worsening accepted proposals.",
            "Calibration, residual, and held-out prediction diagnostics are undefined for the predeclared three-metric binary family and were not fabricated from a reduced model."
        ],
        "recommendedNextAction": "Chief Scientist review: treat S12 as a constraining metric-specific result; do not start S13 automatically or redefine detours post hoc.",
    }
    write_json(MODELS / "modeling_status.json", blocked)
    write_json(
        MODELS / "held_out_prediction_status.json",
        {
            **blocked,
            "status": "not_run_due_predeclared_identifiability_gate",
            "artifactsWritten": ["context_models/held_out_prediction_status.json"],
            "validationResult": "Held-out prediction was correctly not attempted because the frozen multi-metric occurrence family is not identifiable.",
        },
    )
    write_json(
        MODELS / "calibration_residual_status.json",
        {
            **blocked,
            "status": "not_estimable_due_predeclared_identifiability_gate",
            "artifactsWritten": ["context_models/calibration_residual_status.json"],
            "validationResult": "Calibration and residual checks are not estimable without fitting a silently narrowed model; the stop is explicit.",
        },
    )
    write_json(
        OUTPUT / "model_build_summary.json",
        {
            "researchStepId": "S12",
            "success": False,
            "status": "predeclared_identifiability_gate_triggered",
            "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "modelsFit": 0,
            "modelRowsDeclared": len(model_status),
            "marginalEffectRows": len(marginal),
            "episodeOccurrence": event_counts.to_dict("records"),
            "designFullRank": True,
            "conditionNumber": json.loads((MODELS / "design_diagnostics.json").read_text())["conditionNumber"],
            "materialS11Conflict": False,
            "materialS11ConflictReason": "Barrier removal reverses overall completion direction (+36.62 points at empirical n=12-24 versus -12.61 points for the S11 open-loop bridge), but no shared exact-reachability support exists between n=4-5 and n=12-24. The reversal is retained as a cross-layer constraint, not pooled or used to change the model.",
        },
    )
    print(json.dumps(blocked, indent=2))


if __name__ == "__main__":
    main()
