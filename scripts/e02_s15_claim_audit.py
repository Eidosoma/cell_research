#!/usr/bin/env python3
"""Build the E02 S15 claim audit from E01 and E02 artifacts.

S15 is a synthesis-only step. It treats the E01 replication classification
table as the original-claim contract, assigns exactly one E02-supported verdict
to each claim, records which alternative explanations were tested across
S02-S14, packages report-bundle inputs, and validates the resulting evidence
links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "E02"
STEP_ID = "S15"
STEP_NUMBER = 15
STATUS = "completed"
OUTCOME_CLASSIFICATION = "constraining/contradictory"

ARTIFACTS_DIR_DEFAULT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
PREVIOUS_ARTIFACTS_DEFAULT = Path("/previous-artifacts/E01")
RESULTS_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "results"
REPORTS_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "reports"
FIGURES_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "figures" / "e02"
STEP_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "research_steps" / STEP_ID
PROVENANCE_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "provenance"
REPORT_BUNDLE_DIR_DEFAULT = ARTIFACTS_DIR_DEFAULT / "report_bundle_inputs"

ALLOWED_VERDICTS = {
    "robust",
    "robust but narrower",
    "explained by alternative mechanism",
    "not replicated",
}

MECHANISMS = [
    {
        "mechanismId": "S02_scheduler_regimes",
        "mechanismName": "scheduler regime sensitivity",
        "stepIds": "S02,S14",
        "artifactPaths": "/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_strong_statistics_fdr_families.parquet",
        "defaultRationale": "Scheduler regime is not the primary alternative for this claim family.",
    },
    {
        "mechanismId": "S03_activation_rates",
        "mechanismName": "activation-rate artifacts",
        "stepIds": "S03,S14",
        "artifactPaths": "/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_fdr_families.parquet",
        "defaultRationale": "Activation-rate perturbations are not the primary alternative for this claim family.",
    },
    {
        "mechanismId": "S04_label_shuffle",
        "mechanismName": "trajectory-preserving label-shuffle nulls",
        "stepIds": "S04,S14",
        "artifactPaths": "/artifacts/results/e02_label_shuffle_mixture_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Label-shuffle nulls apply mainly to same-goal Aggregation claims.",
    },
    {
        "mechanismId": "S05_dummy_labels",
        "mechanismName": "behavior-preserving dummy labels",
        "stepIds": "S05,S14",
        "artifactPaths": "/artifacts/results/e02_dummy_algotypes_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Dummy-label controls apply mainly to spurious Aggregation controls.",
    },
    {
        "mechanismId": "S06_speed_matching",
        "mechanismName": "speed-matched Algotype controls",
        "stepIds": "S06,S14",
        "artifactPaths": "/artifacts/results/e02_speed_matched_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Speed matching applies mainly to mixed-policy efficiency and Aggregation claims.",
    },
    {
        "mechanismId": "S07_local_move_nulls",
        "mechanismName": "matched local-move nulls",
        "stepIds": "S07,S14",
        "artifactPaths": "/artifacts/results/e02_local_move_null_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Local-move nulls apply mainly to policy competence and Frozen Cell robustness claims.",
    },
    {
        "mechanismId": "S08_dg_matched_nulls",
        "mechanismName": "matched-trajectory Delayed Gratification nulls",
        "stepIds": "S08,S14",
        "artifactPaths": "/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Matched DG nulls apply mainly to Delayed Gratification claims.",
    },
    {
        "mechanismId": "S09_alternative_metrics",
        "mechanismName": "alternative distance metrics",
        "stepIds": "S09,S14",
        "artifactPaths": "/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_alternative_metric_sensitivity.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Alternative metrics are a secondary check unless the claim depends on Frozen residual distance or duplicate ties.",
    },
    {
        "mechanismId": "S10_input_distributions",
        "mechanismName": "input-distribution stress tests",
        "stepIds": "S10,S14",
        "artifactPaths": "/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Input-distribution stress tests apply mainly to generality beyond the original random arrays.",
    },
    {
        "mechanismId": "S11_frozen_placement",
        "mechanismName": "Frozen Cell placement",
        "stepIds": "S11,S14",
        "artifactPaths": "/artifacts/results/e02_frozen_placement_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Frozen placement applies mainly to Frozen Cell and DG claims.",
    },
    {
        "mechanismId": "S12_frozen_behavior",
        "mechanismName": "Frozen Cell behavior variants",
        "stepIds": "S12,S14",
        "artifactPaths": "/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Frozen behavior applies mainly to Frozen Cell and DG claims.",
    },
    {
        "mechanismId": "S13_stop_conditions",
        "mechanismName": "stop-condition sensitivity",
        "stepIds": "S13,S14",
        "artifactPaths": "/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_stop_condition_policy_effects.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "defaultRationale": "Stop-condition sensitivity is most relevant for capped, equilibrium, and Frozen-condition claims.",
    },
    {
        "mechanismId": "S14_strong_statistics",
        "mechanismName": "FDR and clustered statistical audit",
        "stepIds": "S14",
        "artifactPaths": "/artifacts/results/e02_strong_statistics.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet;/artifacts/results/e02_strong_statistics_fdr_families.parquet",
        "defaultRationale": "S14 provides the cross-step multiplicity and effect-size audit for the claim family.",
    },
]

CLAIM_VERDICTS: dict[str, dict[str, str]] = {
    "FIG01_METHOD_ENTITIES": {
        "verdict": "robust",
        "rationale": "The S01 simulator and E01 code mapping keep Position, Value, Algotype, local behavior, and metric entities explicit; no E02 stress test challenged this conceptual framing.",
        "evidence": "S01,S14",
        "artifacts": "/artifacts/results/e02_s01_replicate_summary.parquet;/artifacts/research_steps/S01/validation_report.md;/artifacts/results/e02_strong_statistics_source_validation.parquet",
        "caveats": "This is a method-framing claim, not a quantitative biological claim.",
    },
    "FIG02_CELL_VIEW_POLICIES": {
        "verdict": "robust",
        "rationale": "The deterministic S01 simulator executes Bubble, Insertion, and Selection cell-view policies with validated policy identity across later matched runs.",
        "evidence": "S01,S03,S05,S14",
        "artifacts": "/artifacts/results/e02_s01_replicate_summary.parquet;/artifacts/results/e02_activation_rates.parquet;/artifacts/results/e02_dummy_algotypes_diagnostics.parquet;/artifacts/results/e02_strong_statistics_source_validation.parquet",
        "caveats": "Traditional wrappers remain reconstructed controls rather than byte-for-byte paper archive entry points.",
    },
    "FIG03_COMPLETION": {
        "verdict": "robust",
        "rationale": "No-Frozen pure-policy completion survives deterministic replication, scheduler variation, activation-rate perturbation, alternative metrics, and input-distribution stress tests.",
        "evidence": "S01,S02,S03,S09,S10,S14",
        "artifacts": "/artifacts/results/e02_s01_replicate_summary.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Completion support is for the bounded n=30 deterministic E02 controls and the E01 n=100 baseline, not arbitrary array sizes.",
    },
    "FIG03_TRAJECTORY_STYLE": {
        "verdict": "robust but narrower",
        "rationale": "Endpoint sorting is stable, but S02/S03/S13 show activation counts, peak Aggregation, DG drops, and stop definitions can alter trajectory-level summaries.",
        "evidence": "S01,S02,S03,S13,S14",
        "artifacts": "/artifacts/results/e02_s01_replicate_summary.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "No pixel-level comparison to the original figure was possible.",
    },
    "FIG04_SWAP_BUBBLE": {
        "verdict": "robust but narrower",
        "rationale": "Bubble swap-only similarity is preserved in the E01 replication and sorting remains stable, but efficiency metrics are scheduler/counting-context proxies rather than scheduler-invariant laws.",
        "evidence": "E01,S02,S03,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The verdict supports the relation under the audited regimes, not every possible scheduler implementation.",
    },
    "FIG04_SWAP_INSERTION": {
        "verdict": "robust but narrower",
        "rationale": "Insertion swap-only similarity is preserved in the E01 replication, while S02/S03 identify regime and rate sensitivity in nearby count metrics.",
        "evidence": "E01,S02,S03,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The claim is retained as a bounded count comparison.",
    },
    "FIG04_SWAP_SELECTION": {
        "verdict": "robust but narrower",
        "rationale": "Selection's larger cell-view swap burden is reproduced, but E02 scheduler and activation-rate tests show count magnitudes depend on execution regime.",
        "evidence": "E01,S02,S03,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The direction is stronger than the exact paper factor.",
    },
    "FIG04_TOTAL_BUBBLE": {
        "verdict": "robust but narrower",
        "rationale": "The comparison-inclusive Bubble direction is reproduced, but E02 confirms that total-step quantities are counting-convention and scheduler sensitive.",
        "evidence": "E01,S02,S03,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_fdr_families.parquet",
        "caveats": "The audited factor differs from the exact paper magnitude.",
    },
    "FIG04_TOTAL_INSERTION": {
        "verdict": "robust but narrower",
        "rationale": "The E01 run only directionally supports the reported Insertion total-step advantage; S02/S03 keep it as a bounded direction rather than the paper's exact significant 2.03x claim.",
        "evidence": "E01,S02,S03,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_fdr_families.parquet",
        "caveats": "The paper's exact magnitude and independent z-test significance were not reproduced.",
    },
    "FIG04_TOTAL_SELECTION": {
        "verdict": "robust but narrower",
        "rationale": "The comparison-inclusive Selection disadvantage is reproduced, while execution-regime tests constrain the exact magnitude.",
        "evidence": "E01,S02,S03,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_strong_statistics_fdr_families.parquet",
        "caveats": "The relation is a computational count proxy, not a biological cost measurement.",
    },
    "FIG05_CELL_VIEW_ERROR_TOLERANCE": {
        "verdict": "robust but narrower",
        "rationale": "Cell-view policies remain competent relative to several local-move nulls, but alternative metrics, input profiles, Frozen placement, Frozen behavior, and stop rules all narrow broad error-tolerance claims.",
        "evidence": "E01,S07,S09,S10,S11,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_local_move_null_summary.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_frozen_placement_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Frozen robustness should be stated conditional on passive/stuck definitions, placement, input profile, metric, and stop policy.",
    },
    "FIG05_PASSIVE_CELL_VIEW_RANKING": {
        "verdict": "robust but narrower",
        "rationale": "Bubble's low passive-Frozen error is supported, but Selection is not strictly highest in every audited passive setting and placement/profile confounds remain.",
        "evidence": "E01,S09,S10,S11,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_frozen_placement_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Ranking claims are sensitive to tied or near-tied algorithms.",
    },
    "FIG05_STUCK_CELL_VIEW_RANKING": {
        "verdict": "robust but narrower",
        "rationale": "Selection remains the strongest stuck-Frozen cell-view policy in the audited E01 comparison, but Bubble and Insertion tie in some regenerated settings.",
        "evidence": "E01,S09,S10,S11,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_frozen_placement_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The verdict does not support a unique Bubble-worst ordering in every variant.",
    },
    "FIG06_DG_FORMULA_AND_UNIT_TESTS": {
        "verdict": "robust but narrower",
        "rationale": "The E01-identical DG function is implemented and unit-tested, but S08/S12/S13 show DG values are proxy trajectory summaries sensitive to null model, Frozen behavior, and stopping.",
        "evidence": "E01,S08,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The extracted paper text did not specify every formula edge case.",
    },
    "FIG07_DG_ALGORITHM_ORDERING": {
        "verdict": "explained by alternative mechanism",
        "rationale": "The canonical ordering is visible in the E01 regeneration, but S08 matched-delta nulls remove broad evidence that DG exceeds trajectory-budget artifacts after correction.",
        "evidence": "E01,S08,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "DG should not be interpreted as direct evidence of barrier-navigation intent.",
    },
    "FIG07_DG_BUBBLE_CELL_GT_TRAD": {
        "verdict": "explained by alternative mechanism",
        "rationale": "The paper-style Bubble DG direction was reproduced, but S08 matched-trajectory nulls indicate DG magnitude is largely explainable by the start/end/swap trajectory budget.",
        "evidence": "E01,S08,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The direction may remain descriptive, but the mechanistic interpretation is not supported.",
    },
    "FIG07_DG_INSERTION_SIMILAR": {
        "verdict": "explained by alternative mechanism",
        "rationale": "Insertion's cell/traditional DG similarity is reproduced, but S08 shows similar DG can arise under matched trajectory nulls without policy-specific barrier handling.",
        "evidence": "E01,S08,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Similarity is descriptive only under the audited stuck-Frozen setting.",
    },
    "FIG07_DG_SELECTION_CELL_LT_TRAD": {
        "verdict": "explained by alternative mechanism",
        "rationale": "Selection's lower cell-view DG is reproduced, but the S08 null audit constrains DG as a trajectory-budget artifact rather than a unique delayed-gratification behavior.",
        "evidence": "E01,S08,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Interpret as a bounded proxy metric, not a direct behavioral proof.",
    },
    "FIG07_DG_FROZEN_COUNT_TRENDS": {
        "verdict": "explained by alternative mechanism",
        "rationale": "The canonical trend pattern is directionally reproduced, but S08/S12/S13 show DG depends on trajectory-matched null structure, Frozen behavior gates, and stop policy.",
        "evidence": "E01,S08,S12,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_dg_null_summary.parquet;/artifacts/results/e02_frozen_behavior_variant_summary.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Count trends should not be generalized outside the tested Frozen definitions.",
    },
    "FIG08_SAME_GOAL_SORT_COMPLETION": {
        "verdict": "robust",
        "rationale": "Same-goal unique-value chimeras completed under E01, S02 scheduler controls, S03 activation-rate interventions, S09 tie-aware metrics, and S10 input profiles.",
        "evidence": "E01,S02,S03,S09,S10,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "This supports numeric sorting completion, not all social/aggregation interpretations.",
    },
    "FIG08_SAME_GOAL_EFFICIENCY_INTERPOLATION": {
        "verdict": "robust but narrower",
        "rationale": "Mixed same-goal swap counts remain inside pure-policy envelopes, but S02/S03/S06 show speed and activation schedules affect count-based interpretations.",
        "evidence": "E01,S02,S03,S06,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_scheduler_regime_summary.parquet;/artifacts/results/e02_activation_rate_summary.parquet;/artifacts/results/e02_speed_matched_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Rough-average language is safer than exact interpolation.",
    },
    "FIG08_UNIQUE_AGGREGATION_SIGNIFICANCE": {
        "verdict": "robust but narrower",
        "rationale": "Aggregation above label nulls survives in selected same-goal mixtures, especially the three-policy mixture, but S04/S06/S14 do not support every pairwise mixture after family correction.",
        "evidence": "E01,S04,S05,S06,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_label_shuffle_mixture_summary.parquet;/artifacts/results/e02_dummy_algotypes_summary.parquet;/artifacts/results/e02_speed_matched_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The broad claim should be limited to subsets that beat trajectory-preserving and speed-matched controls.",
    },
    "FIG08_UNIQUE_AGGREGATION_PEAK_NUMBERS": {
        "verdict": "robust but narrower",
        "rationale": "All regenerated unique-value same-goal mixtures peak above 0.5, but several peak heights and timings are lower or shifted relative to the paper and family-corrected null support is subset-specific.",
        "evidence": "E01,S04,S06,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_label_shuffle_mixture_summary.parquet;/artifacts/results/e02_speed_matched_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Exact peak numbers should not be reused as stable constants.",
    },
    "FIG08_SAME_CODE_NEGATIVE_CONTROL": {
        "verdict": "robust",
        "rationale": "S05 explicitly tests behavior-preserving dummy Algotypes with identical code paths, activation rates, and tie histories; labels alone do not create broad FDR-significant Aggregation.",
        "evidence": "S05,S14",
        "artifacts": "/artifacts/results/e02_dummy_algotypes_summary.parquet;/artifacts/results/e02_dummy_algotypes_diagnostics.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The control validates spurious-label artifacts for audited dummy-label designs, not every possible labeling scheme.",
    },
    "FIG08_DUPLICATE_SORT_COMPLETION": {
        "verdict": "robust",
        "rationale": "Duplicate-value same-goal chimeras sort numerically in E01, and S09/S10 tie-aware duplicate-heavy checks preserve numeric completion under audited profiles.",
        "evidence": "E01,S09,S10,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Tie-aware metrics are required for duplicate-heavy arrays.",
    },
    "FIG08_DUPLICATE_AGGREGATION_NUMBERS": {
        "verdict": "robust but narrower",
        "rationale": "Two duplicate pairings closely match high final/peak Aggregation, while Bubble-Insertion returns near 0.5 final Aggregation and the exact numeric peaks are not uniformly stable.",
        "evidence": "E01,S09,S10,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Use duplicate-specific, pairwise-qualified language.",
    },
    "FIG08_DUPLICATE_FINAL_VS_UNIQUE": {
        "verdict": "robust but narrower",
        "rationale": "Relaxed duplicate-value pressure raises final Aggregation for Bubble-Selection and Insertion-Selection but not uniformly for Bubble-Insertion.",
        "evidence": "E01,S09,S10,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_alternative_metric_summary.parquet;/artifacts/results/e02_input_distribution_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The release-of-pressure explanation is pair-specific.",
    },
    "FIG09_UNIQUE_OPPOSITE_AGGREGATION_DOMINANCE": {
        "verdict": "robust but narrower",
        "rationale": "Unique opposite-direction dominance ordering is directionally reproduced, but S13 shows conflict-chimera final states and completion labels depend on stop/cap definitions.",
        "evidence": "E01,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_stop_condition_policy_effects.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Dominance is supported for audited pairings, not as a scheduler-independent law.",
    },
    "FIG09_UNIQUE_OPPOSITE_DYNAMICS_EQUILIBRIUM": {
        "verdict": "robust but narrower",
        "rationale": "Distinct flattening trajectories are reproduced, but many runs are cap-truncated and S13 shows stable-state definitions change final labels.",
        "evidence": "E01,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_stop_condition_policy_effects.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Call these bounded intermediate states rather than exact global equilibria.",
    },
    "FIG09_UNIQUE_OPPOSITE_FINAL_SORTEDNESS": {
        "verdict": "robust but narrower",
        "rationale": "All unique opposite-direction pairings stop below 100% and two final Sortedness values are close to paper values, but one pairing differs and S13 stop definitions move final metrics.",
        "evidence": "E01,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_stop_condition_policy_effects.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "The below-100% result is safer than exact paper percentages.",
    },
    "FIG10_REPEATED_OPPOSITE_DOMINANCE": {
        "verdict": "not replicated",
        "rationale": "E01 failed the Selection-Insertion repeated-value dominance ordering, and no E02 control restored that repeated-value Figure 10 claim.",
        "evidence": "E01,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "S13 documents stop-condition sensitivity but does not rescue the original repeated-value dominance claim.",
    },
    "FIG10_REPEATED_OPPOSITE_SIMILARITY": {
        "verdict": "not replicated",
        "rationale": "E01 repeated-value opposite-direction final states are not similar to the unique-value pattern for all pairings, and E02 stop-condition checks do not resolve that mismatch.",
        "evidence": "E01,S13,S14",
        "artifacts": "/previous-artifacts/E01/results/e01_replication_classification.parquet;/artifacts/results/e02_stop_condition_summary.parquet;/artifacts/results/e02_strong_statistics_claim_summary.parquet",
        "caveats": "Aggregation above 0.5 still occurs, but the broader similarity claim does not survive.",
    },
}

CLAIM_GROUPS = {
    "method": {"FIG01_METHOD_ENTITIES", "FIG02_CELL_VIEW_POLICIES"},
    "completion": {"FIG03_COMPLETION", "FIG08_SAME_GOAL_SORT_COMPLETION", "FIG08_DUPLICATE_SORT_COMPLETION"},
    "trajectory": {"FIG03_TRAJECTORY_STYLE"},
    "efficiency": {
        "FIG04_SWAP_BUBBLE",
        "FIG04_SWAP_INSERTION",
        "FIG04_SWAP_SELECTION",
        "FIG04_TOTAL_BUBBLE",
        "FIG04_TOTAL_INSERTION",
        "FIG04_TOTAL_SELECTION",
        "FIG08_SAME_GOAL_EFFICIENCY_INTERPOLATION",
    },
    "frozen": {
        "FIG05_CELL_VIEW_ERROR_TOLERANCE",
        "FIG05_PASSIVE_CELL_VIEW_RANKING",
        "FIG05_STUCK_CELL_VIEW_RANKING",
    },
    "dg": {
        "FIG06_DG_FORMULA_AND_UNIT_TESTS",
        "FIG07_DG_ALGORITHM_ORDERING",
        "FIG07_DG_BUBBLE_CELL_GT_TRAD",
        "FIG07_DG_INSERTION_SIMILAR",
        "FIG07_DG_SELECTION_CELL_LT_TRAD",
        "FIG07_DG_FROZEN_COUNT_TRENDS",
    },
    "same_goal_aggregation": {
        "FIG08_UNIQUE_AGGREGATION_SIGNIFICANCE",
        "FIG08_UNIQUE_AGGREGATION_PEAK_NUMBERS",
        "FIG08_SAME_CODE_NEGATIVE_CONTROL",
        "FIG08_DUPLICATE_AGGREGATION_NUMBERS",
        "FIG08_DUPLICATE_FINAL_VS_UNIQUE",
    },
    "opposite_direction": {
        "FIG09_UNIQUE_OPPOSITE_AGGREGATION_DOMINANCE",
        "FIG09_UNIQUE_OPPOSITE_DYNAMICS_EQUILIBRIUM",
        "FIG09_UNIQUE_OPPOSITE_FINAL_SORTEDNESS",
        "FIG10_REPEATED_OPPOSITE_DOMINANCE",
        "FIG10_REPEATED_OPPOSITE_SIMILARITY",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "ok": proc.returncode == 0,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc), "ok": False}


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"] if commit["ok"] else "unknown",
        "branch": branch["stdout"] if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"],
        "remote": remote["stdout"],
    }


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            if abs(value) >= 1000 or (0 < abs(value) < 0.001):
                return f"{value:.3g}"
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(out)


def split_paths(value: str) -> list[str]:
    return [part for part in str(value).split(";") if part]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table extension: {path}")


def write_table_pair(df: pd.DataFrame, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = stem.with_suffix(".csv")
    parquet_path = stem.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return [str(csv_path), str(parquet_path)]


def claim_group(claim_id: str) -> str:
    for group, ids in CLAIM_GROUPS.items():
        if claim_id in ids:
            return group
    return "uncategorized"


def build_claim_audit_rows(e01_claims: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(e01_claims["claimId"]) - set(CLAIM_VERDICTS))
    extra = sorted(set(CLAIM_VERDICTS) - set(e01_claims["claimId"]))
    if missing or extra:
        raise ValueError(f"Claim verdict map mismatch. missing={missing}; extra={extra}")

    rows: list[dict[str, Any]] = []
    for row in e01_claims.sort_values(["paperFigure", "claimId"]).to_dict(orient="records"):
        claim_id = str(row["claimId"])
        spec = CLAIM_VERDICTS[claim_id]
        verdict = spec["verdict"]
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Invalid verdict for {claim_id}: {verdict}")
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "stepNumber": STEP_NUMBER,
                "claimId": claim_id,
                "claimGroup": claim_group(claim_id),
                "paperFigure": row.get("paperFigure", ""),
                "claimType": row.get("claimType", ""),
                "paperClaim": row.get("paperClaim", ""),
                "paperExpectedValue": row.get("paperExpectedValue", ""),
                "e01Classification": row.get("classification", ""),
                "e01ObservedEvidence": row.get("observedEvidence", ""),
                "e01Justification": row.get("justification", ""),
                "e01ReviewPriority": row.get("reviewPriority", ""),
                "e02Verdict": verdict,
                "verdictRationale": spec["rationale"],
                "primaryEvidenceStepIds": spec["evidence"],
                "primaryEvidenceArtifacts": spec["artifacts"],
                "alternativeMechanismsAddressed": ";".join(m["mechanismId"] for m in MECHANISMS),
                "residualCaveats": spec["caveats"],
                "outcomeClassification": OUTCOME_CLASSIFICATION,
            }
        )
    return pd.DataFrame(rows)


def mechanism_assessment(claim_id: str, mechanism_id: str) -> tuple[str, str]:
    group = claim_group(claim_id)
    if mechanism_id == "S14_strong_statistics":
        return (
            "integrated",
            "FDR, bootstrap, and model diagnostics are used to temper the final claim verdict.",
        )
    if group == "completion":
        if mechanism_id in {"S02_scheduler_regimes", "S03_activation_rates", "S09_alternative_metrics", "S10_input_distributions"}:
            return ("tested and survived", "Numeric completion remains stable under this alternative-explanation check.")
        if mechanism_id == "S13_stop_conditions":
            return ("bounded", "Stop policies matter for conflict/Frozen cases, but no-Frozen and same-goal completion remains stable in audited runs.")
    if group == "trajectory":
        if mechanism_id in {"S02_scheduler_regimes", "S03_activation_rates", "S13_stop_conditions"}:
            return ("narrows claim", "Trajectory-level summaries vary under this execution or stopping alternative.")
    if group == "efficiency":
        if mechanism_id in {"S02_scheduler_regimes", "S03_activation_rates", "S06_speed_matching", "S13_stop_conditions"}:
            return ("narrows claim", "Count-based efficiency summaries are sensitive to execution, rate, speed, or stopping definitions.")
    if group == "frozen":
        if mechanism_id in {
            "S07_local_move_nulls",
            "S09_alternative_metrics",
            "S10_input_distributions",
            "S11_frozen_placement",
            "S12_frozen_behavior",
            "S13_stop_conditions",
        }:
            return ("narrows claim", "Frozen Cell outcomes depend on this mechanism or remain only boundedly above null.")
    if group == "dg":
        if mechanism_id == "S08_dg_matched_nulls":
            return ("explains or challenges", "Matched-delta trajectory nulls explain much of the DG signal after correction.")
        if mechanism_id in {"S11_frozen_placement", "S12_frozen_behavior", "S13_stop_conditions"}:
            return ("narrows claim", "DG values are sensitive to Frozen setup or stop policy.")
    if group == "same_goal_aggregation":
        if mechanism_id == "S04_label_shuffle":
            return ("narrows claim", "Trajectory-preserving label shuffles retain support only for selected same-goal Aggregation cases.")
        if mechanism_id == "S05_dummy_labels":
            return ("supports control", "Behavior-preserving dummy labels do not create broad spurious Aggregation.")
        if mechanism_id == "S06_speed_matching":
            return ("narrows claim", "Speed-matched controls reduce broad pairwise support and retain stronger three-policy evidence.")
        if mechanism_id in {"S02_scheduler_regimes", "S03_activation_rates"}:
            return ("narrows claim", "Execution and activation-rate conditions can change Aggregation trajectory summaries.")
        if mechanism_id in {"S09_alternative_metrics", "S10_input_distributions"}:
            return ("bounded", "Tie-aware and distribution checks support sorting but do not make all Aggregation numbers invariant.")
    if group == "opposite_direction":
        if mechanism_id == "S13_stop_conditions":
            return ("narrows or fails claim", "Conflict-chimera endpoints and stable-state labels are stop-condition sensitive.")
        if mechanism_id in {"S09_alternative_metrics", "S10_input_distributions"}:
            return ("bounded", "Metric and distribution checks do not rescue failed repeated-value opposite-direction claims.")
    if group == "method" and mechanism_id in {"S02_scheduler_regimes", "S03_activation_rates", "S14_strong_statistics"}:
        return ("context", "The mechanism does not challenge the existence of the method entities or policies.")
    mechanism = next(m for m in MECHANISMS if m["mechanismId"] == mechanism_id)
    return ("not primary", mechanism["defaultRationale"])


def build_alternative_matrix(claim_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for claim in claim_table.to_dict(orient="records"):
        for mechanism in MECHANISMS:
            assessment, rationale = mechanism_assessment(str(claim["claimId"]), mechanism["mechanismId"])
            rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "researchStepId": STEP_ID,
                    "claimId": claim["claimId"],
                    "claimGroup": claim["claimGroup"],
                    "e02Verdict": claim["e02Verdict"],
                    "mechanismId": mechanism["mechanismId"],
                    "mechanismName": mechanism["mechanismName"],
                    "assessment": assessment,
                    "supportingStepIds": mechanism["stepIds"],
                    "supportingArtifacts": mechanism["artifactPaths"],
                    "rationale": rationale,
                }
            )
    return pd.DataFrame(rows)


def collect_status_index(artifacts_dir: Path) -> pd.DataFrame:
    rows = []
    for step_num in range(1, 15):
        step_id = f"S{step_num:02d}"
        status_path = artifacts_dir / "research_steps" / step_id / "status.json"
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "sourceResearchStepId": step_id,
                    "statusPath": str(status_path),
                    "success": bool(payload.get("success")),
                    "status": payload.get("status", ""),
                    "validationResult": payload.get("validationResult", ""),
                    "recommendedNextAction": payload.get("recommendedNextAction", ""),
                }
            )
        else:
            rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "sourceResearchStepId": step_id,
                    "statusPath": str(status_path),
                    "success": False,
                    "status": "missing",
                    "validationResult": "missing",
                    "recommendedNextAction": "",
                }
            )
    return pd.DataFrame(rows)


def collect_file_index(root: Path, glob_pattern: str, kind: str) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob(glob_pattern)):
        if path.is_file():
            rows.append(
                {
                    "experimentId": EXPERIMENT_ID,
                    "researchStepId": STEP_ID,
                    "artifactKind": kind,
                    "artifactPath": str(path),
                    "fileName": path.name,
                    "suffix": path.suffix,
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_path(path),
                }
            )
    return pd.DataFrame(rows)


def build_tables_index(
    results_dir: Path,
    report_bundle_dir: Path,
    previous_artifacts_dir: Path,
    required_outputs: list[str],
) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("e02_*.csv")) + sorted(results_dir.glob("e02_*.parquet")):
        rows.append(path)
    e01_claims = previous_artifacts_dir / "results" / "e01_replication_classification.parquet"
    if e01_claims.exists():
        rows.append(e01_claims)
    for out in required_outputs:
        p = Path(out)
        if p.suffix in {".csv", ".parquet"} and p.exists():
            rows.append(p)
    seen: set[str] = set()
    indexed = []
    for path in rows:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        row_count = None
        columns = ""
        if path.suffix in {".csv", ".parquet"}:
            try:
                table = read_table(path)
                row_count = len(table)
                columns = ",".join(map(str, table.columns))
            except Exception:
                row_count = None
                columns = "unreadable"
        indexed.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "artifactKind": "table",
                "artifactPath": str(path),
                "fileName": path.name,
                "suffix": path.suffix,
                "rowCount": row_count,
                "columns": columns,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return pd.DataFrame(indexed)


def validate_outputs(
    e01_claims: pd.DataFrame,
    claim_table: pd.DataFrame,
    matrix: pd.DataFrame,
    status_index: pd.DataFrame,
    required_artifacts: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "checkId": check_id,
                "validationPassed": bool(passed),
                "detail": detail,
            }
        )

    add("e01_claim_row_count", len(e01_claims) == len(claim_table), f"e01={len(e01_claims)}; audit={len(claim_table)}")
    add(
        "all_claims_have_one_verdict",
        claim_table["claimId"].is_unique
        and claim_table["e02Verdict"].notna().all()
        and set(claim_table["e02Verdict"]).issubset(ALLOWED_VERDICTS),
        f"uniqueClaims={claim_table['claimId'].nunique()}; verdicts={sorted(claim_table['e02Verdict'].unique())}",
    )
    missing_claims = sorted(set(e01_claims["claimId"]) - set(claim_table["claimId"]))
    extra_claims = sorted(set(claim_table["claimId"]) - set(e01_claims["claimId"]))
    add("claim_set_matches_e01", not missing_claims and not extra_claims, f"missing={missing_claims}; extra={extra_claims}")
    expected_matrix_rows = len(claim_table) * len(MECHANISMS)
    add(
        "alternative_matrix_complete",
        len(matrix) == expected_matrix_rows
        and matrix.groupby("claimId")["mechanismId"].nunique().eq(len(MECHANISMS)).all(),
        f"rows={len(matrix)}; expectedRows={expected_matrix_rows}; mechanisms={len(MECHANISMS)}",
    )
    add(
        "source_statuses_success",
        len(status_index) == 14
        and status_index["success"].all()
        and status_index["validationResult"].astype(str).str.lower().str.startswith("passed").all(),
        f"statuses={len(status_index)}; successes={int(status_index['success'].sum())}",
    )

    referenced_paths: set[str] = set()
    for value in claim_table["primaryEvidenceArtifacts"].tolist():
        referenced_paths.update(split_paths(value))
    for value in matrix["supportingArtifacts"].tolist():
        referenced_paths.update(split_paths(value))
    missing_paths = sorted(path for path in referenced_paths if not Path(path).exists())
    add("referenced_artifacts_exist", not missing_paths, f"missingCount={len(missing_paths)}; missing={missing_paths[:8]}")

    missing_required = sorted(path for path in required_artifacts if not Path(path).exists())
    add("required_s15_artifacts_exist", not missing_required, f"missingCount={len(missing_required)}; missing={missing_required}")

    verdict_counts = claim_table["e02Verdict"].value_counts().to_dict()
    add("verdict_count_is_32", int(sum(verdict_counts.values())) == 32, f"verdictCounts={verdict_counts}")
    return pd.DataFrame(rows)


def render_claim_audit(
    claim_table: pd.DataFrame,
    matrix: pd.DataFrame,
    s14_claim_summary: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    verdict_counts = claim_table["e02Verdict"].value_counts().reindex(sorted(ALLOWED_VERDICTS), fill_value=0)
    group_counts = pd.crosstab(claim_table["claimGroup"], claim_table["e02Verdict"])
    matrix_counts = matrix["assessment"].value_counts().sort_index()
    s14_rows = s14_claim_summary[["claimFamily", "classification", "testCount", "fdrSignificantCount", "minQValue"]].fillna("")
    validations_ok = bool(validation["validationPassed"].all())

    lines = [
        "# E02 S15 Claim Audit",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Research Step",
        "",
        f"- Research step ID: {STEP_ID}",
        f"- Completion status: {STATUS}",
        f"- Validation result: {'passed' if validations_ok else 'failed'}",
        "- Artifacts written: `/artifacts/reports/claim_audit.md`, `/artifacts/results/e02_claim_survival_table.csv`, `/artifacts/results/e02_alternative_explanation_matrix.csv`, `/artifacts/results/e02_claim_audit_validation.csv`, `/artifacts/report_bundle_inputs/`, and `/artifacts/research_steps/S15/status.json`.",
        "- Caveats or blockers: This is a computational synthesis of existing E01 and E02 artifacts. It does not add new simulator trajectories, wet-lab validation, or causal evidence.",
        "- Recommended next action: Hand back for Chief Scientist review; after approval, downstream E03 may consume the E02 report bundle inputs.",
        "",
        "## Executive Summary",
        "",
        (
            "S15 assigns exactly one verdict to each of the 32 original E01 claim rows. "
            "No-Frozen completion, same-goal sorting completion, duplicate numeric sorting, method entities, policy translations, and the dummy-label negative control are robust. "
            "Efficiency, trajectory, Frozen Cell, Aggregation, and opposite-direction equilibrium claims mostly survive only in narrower forms. "
            "Delayed Gratification is best treated as explained by matched trajectory mechanics rather than direct evidence of barrier-navigation intent. "
            "The two repeated-value opposite-direction Figure 10 claims remain not replicated."
        ),
        "",
        "## Verdict Counts",
        "",
        markdown_table(["Verdict", "Claim count"], [[idx, int(val)] for idx, val in verdict_counts.items()]),
        "",
        "## Verdicts By Claim Group",
        "",
        markdown_table(["Claim group"] + list(group_counts.columns), [[idx] + [int(v) for v in row] for idx, row in group_counts.iterrows()]),
        "",
        "## Claim Survival Table",
        "",
        markdown_table(
            ["Claim ID", "Figure", "E01 class", "E02 verdict", "Rationale"],
            [
                [
                    row["claimId"],
                    row["paperFigure"],
                    row["e01Classification"],
                    row["e02Verdict"],
                    row["verdictRationale"],
                ]
                for row in claim_table.to_dict(orient="records")
            ],
        ),
        "",
        "## Alternative-Explanation Matrix Summary",
        "",
        markdown_table(["Assessment", "Rows"], [[idx, int(val)] for idx, val in matrix_counts.items()]),
        "",
        "The complete claim-by-mechanism matrix is written to `e02_alternative_explanation_matrix.csv` and `.parquet`.",
        "",
        "## Strong Statistics Context",
        "",
        markdown_table(
            ["Claim family", "S14 classification", "Tests", "FDR significant", "Minimum q"],
            s14_rows.values.tolist(),
        ),
        "",
        "## Validation",
        "",
        markdown_table(
            ["Check", "Passed", "Detail"],
            [[row["checkId"], row["validationPassed"], row["detail"]] for row in validation.to_dict(orient="records")],
        ),
        "",
        "## Claim Boundaries",
        "",
        "- These verdicts are bounded computational proxy evidence.",
        "- Aggregation is a trajectory/label metric, not direct biological self-recognition.",
        "- Delayed Gratification is a path-shape metric and should not be treated as direct evidence of intent.",
        "- Frozen Cell conclusions are conditional on defect placement, defect behavior, distance metric, input distribution, and stop rule.",
    ]
    return "\n".join(lines) + "\n"


def render_summary(claim_table: pd.DataFrame, validation: pd.DataFrame, artifacts_written: list[str]) -> str:
    verdict_counts = claim_table["e02Verdict"].value_counts().reindex(sorted(ALLOWED_VERDICTS), fill_value=0)
    return "\n".join(
        [
            "# E02 S15 Summary",
            "",
            f"- Research step ID: {STEP_ID}",
            f"- Completion status: {STATUS}",
            "- Artifacts written: `/artifacts/reports/claim_audit.md`, `/artifacts/results/e02_claim_survival_table.csv`, `/artifacts/results/e02_alternative_explanation_matrix.csv`, `/artifacts/results/e02_claim_audit_validation.csv`, `/artifacts/report_bundle_inputs/`, `/artifacts/research_steps/S15/status.json`, `/artifacts/research_steps/S15/validation_report.md`, and `/artifacts/research_steps/S15/artifact_manifest.json`.",
            f"- Validation result: {'passed' if validation['validationPassed'].all() else 'failed'}",
            "- Caveats or blockers: No blockers. The audit is a synthesis over completed computational artifacts and does not create new experimental evidence.",
            "- Lay summary: The original claim set mostly survives as a narrower computational result. Basic sorting and same-goal completion are robust; DG and broad Frozen Cell/generalization claims need qualified language; repeated-value opposite-direction Figure 10 claims remain not replicated.",
            "- Recommended next action: Hand back for Chief Scientist review; after approval, use the report bundle as the E02 input package for downstream E03 planning.",
            "",
            "## Verdict Counts",
            "",
            markdown_table(["Verdict", "Claim count"], [[idx, int(val)] for idx, val in verdict_counts.items()]),
        ]
    ) + "\n"


def render_validation_report(validation: pd.DataFrame, status_index: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# E02 S15 Validation Report",
            "",
            f"- Research step ID: {STEP_ID}",
            f"- Completion status: {STATUS}",
            "- Artifacts written: validation table, claim audit report, claim survival table, alternative-explanation matrix, report bundle inputs, provenance manifest, and copied reproducible code.",
            f"- Validation result: {'passed' if validation['validationPassed'].all() else 'failed'}",
            "- Caveats or blockers: Validation checks file presence, row-count contracts, source status success, and one-verdict-per-claim coverage; it does not independently rerun S01-S14 simulations.",
            "- Recommended next action: Chief Scientist review before any downstream E03 use.",
            "",
            "## Checks",
            "",
            markdown_table(
                ["Check", "Passed", "Detail"],
                [[row["checkId"], row["validationPassed"], row["detail"]] for row in validation.to_dict(orient="records")],
            ),
            "",
            "## Source Step Statuses",
            "",
            markdown_table(
                ["Step", "Success", "Validation", "Status path"],
                [
                    [row["sourceResearchStepId"], row["success"], row["validationResult"], row["statusPath"]]
                    for row in status_index.to_dict(orient="records")
                ],
            ),
        ]
    ) + "\n"


def write_report_bundle(
    report_bundle_dir: Path,
    claim_table: pd.DataFrame,
    matrix: pd.DataFrame,
    validation: pd.DataFrame,
    tables_index: pd.DataFrame,
    figures_index: pd.DataFrame,
    reports_index: pd.DataFrame,
    status_index: pd.DataFrame,
    claim_audit_text: str,
    artifacts_written: list[str],
) -> list[str]:
    report_bundle_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    bundle_claim_audit = report_bundle_dir / "claim_audit.md"
    bundle_claim_audit.write_text(claim_audit_text, encoding="utf-8")
    written.append(str(bundle_claim_audit))

    for df, stem in [
        (claim_table, report_bundle_dir / "claim_survival_table"),
        (matrix, report_bundle_dir / "alternative_explanation_matrix"),
        (validation, report_bundle_dir / "claim_audit_validation"),
        (tables_index, report_bundle_dir / "tables_index"),
        (figures_index, report_bundle_dir / "figures_index"),
        (reports_index, report_bundle_dir / "reports_index"),
        (status_index, report_bundle_dir / "step_status_index"),
    ]:
        written.extend(write_table_pair(df, stem))

    readme = report_bundle_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# E02 Report Bundle Inputs",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {STATUS}",
                "- Contents: S15 claim audit, claim survival table, alternative-explanation matrix, validation table, and indices of source E02 tables, figures, reports, and S01-S14 statuses.",
                "- Validation result: see `claim_audit_validation.csv`.",
                "- Caveats or blockers: Source artifacts remain in their original locations; this directory indexes and packages synthesis inputs rather than duplicating all upstream artifacts.",
                "- Recommended next action: Chief Scientist review and report drafting.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(str(readme))

    files_for_checksum = sorted(path for path in report_bundle_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    checksums = report_bundle_dir / "checksums.sha256"
    checksums.write_text(
        "".join(f"{sha256_path(path)}  {path.name}\n" for path in files_for_checksum),
        encoding="utf-8",
    )
    written.append(str(checksums))

    manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "bundleDirectory": str(report_bundle_dir),
        "files": [
            {
                "path": str(path),
                "name": path.name,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
            for path in sorted(report_bundle_dir.iterdir())
            if path.is_file()
        ],
        "sourceArtifactCount": len(artifacts_written),
    }
    bundle_manifest = report_bundle_dir / "bundle_manifest.json"
    write_json(bundle_manifest, manifest)
    written.append(str(bundle_manifest))
    return written


def copy_code(step_dir: Path) -> list[str]:
    code_dir = step_dir / "code"
    script_dst = code_dir / "scripts" / Path(__file__).name
    test_src = REPO_ROOT / "tests" / "test_e02_claim_audit.py"
    test_dst = code_dir / "tests" / test_src.name
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    test_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_dst)
    copied = [str(script_dst)]
    if test_src.exists():
        shutil.copy2(test_src, test_dst)
        copied.append(str(test_dst))
    return copied


def update_run_manifest(provenance_dir: Path, artifacts_written: list[str], validation_passed: bool) -> str:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = provenance_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    payload.setdefault("experimentId", EXPERIMENT_ID)
    payload["lastUpdatedAt"] = utc_now()
    payload.setdefault("researchSteps", {})
    payload["researchSteps"][STEP_ID] = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "status": STATUS,
        "success": validation_passed,
        "artifactsWritten": artifacts_written,
        "git": get_git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(manifest_path, payload)
    return str(manifest_path)


def build_and_write(args: argparse.Namespace) -> dict[str, Any]:
    artifacts_dir = args.artifacts_dir
    previous_artifacts_dir = args.previous_artifacts_dir
    results_dir = args.results_dir
    reports_dir = args.reports_dir
    figures_dir = args.figures_dir
    step_dir = args.step_dir
    report_bundle_dir = args.report_bundle_dir
    provenance_dir = args.provenance_dir

    for path in [results_dir, reports_dir, figures_dir, step_dir, report_bundle_dir, provenance_dir]:
        path.mkdir(parents=True, exist_ok=True)

    e01_claims_path = previous_artifacts_dir / "results" / "e01_replication_classification.parquet"
    s14_claim_summary_path = results_dir / "e02_strong_statistics_claim_summary.parquet"
    if not e01_claims_path.exists():
        raise FileNotFoundError(e01_claims_path)
    if not s14_claim_summary_path.exists():
        raise FileNotFoundError(s14_claim_summary_path)

    e01_claims = pd.read_parquet(e01_claims_path)
    s14_claim_summary = pd.read_parquet(s14_claim_summary_path)
    claim_table = build_claim_audit_rows(e01_claims)
    matrix = build_alternative_matrix(claim_table)
    status_index = collect_status_index(artifacts_dir)

    artifacts_written: list[str] = []
    artifacts_written.extend(write_table_pair(claim_table, results_dir / "e02_claim_survival_table"))
    artifacts_written.extend(write_table_pair(matrix, results_dir / "e02_alternative_explanation_matrix"))

    tables_index = build_tables_index(results_dir, report_bundle_dir, previous_artifacts_dir, artifacts_written)
    figures_index = collect_file_index(figures_dir, "*", "figure")
    reports_index = collect_file_index(artifacts_dir / "research_steps", "S*/summary.md", "summary")
    extra_reports = collect_file_index(reports_dir, "*.md", "report")
    if not extra_reports.empty:
        reports_index = pd.concat([reports_index, extra_reports], ignore_index=True)

    required_for_validation = [
        str(results_dir / "e02_claim_survival_table.csv"),
        str(results_dir / "e02_claim_survival_table.parquet"),
        str(results_dir / "e02_alternative_explanation_matrix.csv"),
        str(results_dir / "e02_alternative_explanation_matrix.parquet"),
        str(report_bundle_dir),
    ]
    validation = validate_outputs(e01_claims, claim_table, matrix, status_index, required_for_validation)
    artifacts_written.extend(write_table_pair(validation, results_dir / "e02_claim_audit_validation"))

    claim_audit_text = render_claim_audit(claim_table, matrix, s14_claim_summary, validation)
    claim_audit_path = reports_dir / "claim_audit.md"
    claim_audit_path.write_text(claim_audit_text, encoding="utf-8")
    artifacts_written.append(str(claim_audit_path))

    # Rebuild indices after claim_audit.md exists.
    reports_index = collect_file_index(artifacts_dir / "research_steps", "S*/summary.md", "summary")
    extra_reports = collect_file_index(reports_dir, "*.md", "report")
    if not extra_reports.empty:
        reports_index = pd.concat([reports_index, extra_reports], ignore_index=True)

    bundle_written = write_report_bundle(
        report_bundle_dir=report_bundle_dir,
        claim_table=claim_table,
        matrix=matrix,
        validation=validation,
        tables_index=tables_index,
        figures_index=figures_index,
        reports_index=reports_index,
        status_index=status_index,
        claim_audit_text=claim_audit_text,
        artifacts_written=artifacts_written,
    )
    artifacts_written.extend(bundle_written)

    code_written = copy_code(step_dir)
    artifacts_written.extend(code_written)

    validation_passed = bool(validation["validationPassed"].all())
    run_manifest_path = update_run_manifest(provenance_dir, artifacts_written, validation_passed)
    artifacts_written.append(run_manifest_path)

    artifact_manifest = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            {
                "path": path,
                "exists": Path(path).exists(),
                "sizeBytes": Path(path).stat().st_size if Path(path).exists() and Path(path).is_file() else None,
                "sha256": sha256_path(Path(path)) if Path(path).exists() and Path(path).is_file() else None,
            }
            for path in sorted(set(artifacts_written))
        ],
    }
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    write_json(artifact_manifest_path, artifact_manifest)
    artifacts_written.append(str(artifact_manifest_path))

    summary_text = render_summary(claim_table, validation, artifacts_written)
    summary_path = step_dir / "summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")
    artifacts_written.append(str(summary_path))

    validation_report_path = step_dir / "validation_report.md"
    validation_report_path.write_text(render_validation_report(validation, status_index), encoding="utf-8")
    artifacts_written.append(str(validation_report_path))

    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": STATUS if validation_passed else "completed_with_validation_failures",
        "outcomeClassification": OUTCOME_CLASSIFICATION,
        "startedAt": None,
        "completedAt": utc_now(),
        "artifactsWritten": sorted(set(artifacts_written)),
        "validationResult": "passed" if validation_passed else "failed",
        "validationChecks": validation.to_dict(orient="records"),
        "claimCount": int(len(claim_table)),
        "verdictCounts": {str(k): int(v) for k, v in claim_table["e02Verdict"].value_counts().sort_index().items()},
        "alternativeMatrixRows": int(len(matrix)),
        "caveatsOrBlockers": "No blockers. S15 is a synthesis over completed S01-S14 and E01 artifacts; conclusions remain bounded computational proxy evidence, not new empirical validation.",
        "recommendedNextAction": "Hand back for Chief Scientist review; after approval, downstream E03 may consume the E02 report bundle inputs.",
        "git": get_git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    status_path = step_dir / "status.json"
    write_json(status_path, status_payload)
    artifacts_written.append(str(status_path))

    # Refresh manifests now that status/summary/validation paths exist.
    artifact_manifest["artifacts"] = [
        {
            "path": path,
            "exists": Path(path).exists(),
            "sizeBytes": Path(path).stat().st_size if Path(path).exists() and Path(path).is_file() else None,
            "sha256": sha256_path(Path(path)) if Path(path).exists() and Path(path).is_file() else None,
        }
        for path in sorted(set(artifacts_written))
    ]
    write_json(artifact_manifest_path, artifact_manifest)

    return {
        "success": validation_passed,
        "claimCount": int(len(claim_table)),
        "verdictCounts": {str(k): int(v) for k, v in claim_table["e02Verdict"].value_counts().sort_index().items()},
        "alternativeMatrixRows": int(len(matrix)),
        "artifactsWritten": sorted(set(artifacts_written)),
        "validation": validation.to_dict(orient="records"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR_DEFAULT)
    parser.add_argument("--previous-artifacts-dir", type=Path, default=PREVIOUS_ARTIFACTS_DEFAULT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR_DEFAULT)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR_DEFAULT)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR_DEFAULT)
    parser.add_argument("--provenance-dir", type=Path, default=PROVENANCE_DIR_DEFAULT)
    parser.add_argument("--report-bundle-dir", type=Path, default=REPORT_BUNDLE_DIR_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_and_write(args)
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
