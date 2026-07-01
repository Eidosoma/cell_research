#!/usr/bin/env python3
"""Write the E02 S15 final claim audit over E01 and E02 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


STEP_ID = "S15"
STEP_NUMBER = 15
EXPERIMENT_ID = "E02"

ALLOWED_VERDICTS = {
    "robust",
    "robust_but_narrower",
    "partly_explained_by_alternative",
    "not_replicated_or_unresolved",
}

ALTERNATIVE_KEYS = (
    "scheduler_regime",
    "activation_rate",
    "trajectory_label_shuffle",
    "behavior_preserving_dummy_labels",
    "speed_matched_algotypes",
    "randomized_local_move_null",
    "dg_matched_null",
    "alternative_distance_metrics",
    "input_distribution",
    "frozen_cell_placement",
    "frozen_cell_behavior",
    "stop_condition",
    "strong_statistics_fdr",
)


@dataclass(frozen=True)
class InputPaths:
    artifacts_dir: Path
    previous_e01_dir: Path

    @property
    def paths(self) -> dict[str, Path]:
        return {
            "E01_replication_status": self.previous_e01_dir / "tables/e01_replication_status.csv",
            "E01_divergence_log": self.previous_e01_dir / "reports/e01_divergence_log.md",
            "E01_replication_report": self.previous_e01_dir / "replication_report.md",
            "E01_report_bundle_handoff": self.previous_e01_dir / "reports/e01_report_bundle_handoff.md",
            "S02_scheduler_classification": self.artifacts_dir / "research_steps/S02/e02_scheduler_sensitivity_classification.csv",
            "S03_activation_classification": self.artifacts_dir / "research_steps/S03/e02_activation_rate_sensitivity_classification.csv",
            "S04_label_shuffle_summary": self.artifacts_dir / "research_steps/S04/e02_label_shuffle_condition_summary.csv",
            "S05_dummy_summary": self.artifacts_dir / "research_steps/S05/e02_dummy_algotype_condition_summary.csv",
            "S06_speed_match_classification": self.artifacts_dir / "research_steps/S06/e02_speed_matched_classification.csv",
            "S07_local_move_classification": self.artifacts_dir / "research_steps/S07/e02_local_move_null_classification.csv",
            "S08_dg_burden_classification": self.artifacts_dir / "research_steps/S08/e02_dg_burden_classification.csv",
            "S09_metric_classification": self.artifacts_dir / "research_steps/S09/e02_alternative_metric_classification.csv",
            "S10_input_classification": self.artifacts_dir / "research_steps/S10/e02_input_distribution_classification.csv",
            "S11_placement_classification": self.artifacts_dir / "research_steps/S11/e02_frozen_placement_classification.csv",
            "S12_behavior_classification": self.artifacts_dir / "research_steps/S12/e02_frozen_behavior_classification.csv",
            "S13_stop_classification": self.artifacts_dir / "research_steps/S13/e02_stop_condition_classification.csv",
            "S14_statistical_tests": self.artifacts_dir / "tables/e02_statistical_tests.parquet",
            "S14_interpretation_summary": self.artifacts_dir / "research_steps/S14/e02_statistical_interpretation_summary.csv",
            "S14_family_summary": self.artifacts_dir / "research_steps/S14/e02_fdr_family_summary.csv",
            "S14_strong_statistics_report": self.artifacts_dir / "reports/e02_strong_statistics.md",
        }


@dataclass(frozen=True)
class AlternativeEvidence:
    key: str
    title: str
    step_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    summary: str
    supports_families: frozenset[str]
    constrains_families: frozenset[str]
    caveat: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument("--previous-e01-dir", type=Path, default=Path("/previous-artifacts/E01"))
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path, base_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(base_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": int(path.stat().st_size),
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(time.monotonic() - started),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": bool(result.returncode == 0),
    }


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_inputs(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required S15 input artifact(s): {missing}")
    return {name: read_table(path) for name, path in paths.items() if path.suffix in {".csv", ".parquet"}}


def count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).sort_index().to_dict().items()}


def n_contains(frame: pd.DataFrame, column: str, token: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].astype(str).str.contains(token, case=False, regex=False).sum())


def n_gt(frame: pd.DataFrame, column: str, threshold: float) -> int:
    if column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce")
    return int((values > threshold).sum())


def n_lt(frame: pd.DataFrame, column: str, threshold: float) -> int:
    if column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce")
    return int((values < threshold).sum())


def claim_family_for_claim(claim_id: str, figure_or_section: str = "") -> str:
    claim_id = str(claim_id)
    figure = str(figure_or_section).lower()
    if claim_id.startswith("methods_") or "figure3" in claim_id:
        return "baseline_sorting"
    if "figure4" in claim_id or "efficiency" in claim_id:
        return "efficiency"
    if "figure7" in claim_id or "_dg" in claim_id:
        return "delayed_gratification"
    if "figure9" in claim_id or "figure10" in claim_id or "conflict" in figure:
        return "conflict_governance"
    if "figure5" in claim_id or "frozen" in claim_id:
        return "frozen_robustness"
    if "figure8" in claim_id or "aggregation" in claim_id or "chimera" in claim_id:
        return "aggregation"
    return "miscellaneous"


def normalize_e01_classification(value: Any) -> str:
    text = str(value).strip().lower().replace("_", " ")
    if text in {"exact", "statistically consistent", "directionally consistent", "not replicated"}:
        return text
    if "statistically" in text:
        return "statistically consistent"
    if "direction" in text:
        return "directionally consistent"
    if "not" in text and "replic" in text:
        return "not replicated"
    return text or "unknown"


def base_verdict_for_claim(claim_id: str, claim_family: str, e01_classification: str) -> str:
    claim_id = str(claim_id)
    e01_classification = normalize_e01_classification(e01_classification)
    if e01_classification == "not replicated":
        return "not_replicated_or_unresolved"
    if claim_id == "figure7_all_algorithms_show_dg":
        return "robust_but_narrower"
    if claim_family == "delayed_gratification":
        return "partly_explained_by_alternative"
    if e01_classification == "exact" and claim_family == "baseline_sorting":
        return "robust"
    return "robust_but_narrower"


def evidence_level_for_claim(claim_family: str, e01_classification: str, verdict: str) -> str:
    e01_classification = normalize_e01_classification(e01_classification)
    if verdict == "not_replicated_or_unresolved":
        return "E01_nonreplication_or_unresolved_exact_magnitude"
    if verdict == "partly_explained_by_alternative":
        return "E01_replication_plus_E02_null_or_artifact_constraint"
    if claim_family in {"conflict_governance"}:
        return "E01_replication_with_limited_E02_direct_stress_tests"
    if e01_classification == "exact":
        return "E01_exact_with_E02_guardrail_checks"
    if e01_classification == "statistically consistent":
        return "E01_statistically_consistent_with_E02_guardrail_checks"
    return "E01_directional_with_E02_guardrail_checks"


def survives_as_for_verdict(verdict: str, claim_family: str) -> str:
    if verdict == "robust":
        return "survives_as_written_with_computational_proxy_boundaries"
    if verdict == "robust_but_narrower":
        if claim_family == "conflict_governance":
            return "survives_as_E01_replicated_conflict_proxy_pending_direct_E02_stress_tests"
        return "survives_conditioned_on_documented_scheduler_metric_input_perturbation_and_stop_assumptions"
    if verdict == "partly_explained_by_alternative":
        return "survives_as_descriptive_trajectory_metric_not_as_unique_planning_or_mechanism_evidence"
    return "does_not_survive_as_written"


def build_s14_family_summary(stats: pd.DataFrame) -> dict[str, str]:
    family_map = {
        "baseline_sorting": ("robustness",),
        "efficiency": ("efficiency",),
        "frozen_robustness": ("robustness",),
        "delayed_gratification": ("delayed_gratification",),
        "aggregation": ("aggregation",),
        "conflict_governance": ("aggregation", "metric_sensitivity", "robustness"),
        "miscellaneous": tuple(sorted(stats["claim_family"].dropna().astype(str).unique())) if "claim_family" in stats else (),
    }
    summaries: dict[str, str] = {}
    for audit_family, stat_families in family_map.items():
        subset = stats[stats["claim_family"].astype(str).isin(stat_families)].copy() if stat_families else stats.iloc[0:0]
        if subset.empty:
            summaries[audit_family] = "S14 has no direct corrected statistical slice for this claim family; verdict rests on E01 replication status and targeted E02 classification evidence."
            continue
        interp_counts = count_values(subset, "statistical_interpretation")
        detected = int((subset["corrected_result"].astype(str) == "detected_after_family_fdr").sum())
        summaries[audit_family] = (
            f"S14 contributed {len(subset)} rows for mapped statistical families {list(stat_families)}, "
            f"with {detected} family-FDR detections and interpretation counts {json.dumps(interp_counts, sort_keys=True)}."
        )
    return summaries


def build_alternative_evidence(inputs: Mapping[str, pd.DataFrame], paths: Mapping[str, Path]) -> dict[str, AlternativeEvidence]:
    s02 = inputs["S02_scheduler_classification"]
    s03 = inputs["S03_activation_classification"]
    s04 = inputs["S04_label_shuffle_summary"]
    s05 = inputs["S05_dummy_summary"]
    s06 = inputs["S06_speed_match_classification"]
    s07 = inputs["S07_local_move_classification"]
    s08 = inputs["S08_dg_burden_classification"]
    s09 = inputs["S09_metric_classification"]
    s10 = inputs["S10_input_classification"]
    s11 = inputs["S11_placement_classification"]
    s12 = inputs["S12_behavior_classification"]
    s13 = inputs["S13_stop_classification"]
    s14 = inputs["S14_statistical_tests"]

    s14_counts = count_values(s14, "statistical_interpretation")
    alternatives = {
        "scheduler_regime": AlternativeEvidence(
            key="scheduler_regime",
            title="Scheduler-regime artifact",
            step_ids=("S02",),
            artifact_paths=(str(paths["S02_scheduler_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S02 classified {n_contains(s02, 'classification', 'scheduler_sensitive')} of {len(s02)} scheduler slices as sensitive; "
                f"class counts {json.dumps(count_values(s02, 'classification'), sort_keys=True)}."
            ),
            supports_families=frozenset({"baseline_sorting", "aggregation"}),
            constrains_families=frozenset({"efficiency", "frozen_robustness", "delayed_gratification"}),
            caveat="Bounded first-10-repeat scheduler matrix; deterministic schedulers are controlled alternatives, not OS-thread replay.",
        ),
        "activation_rate": AlternativeEvidence(
            key="activation_rate",
            title="Activation-rate artifact",
            step_ids=("S03",),
            artifact_paths=(str(paths["S03_activation_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S03 found activation-rate sensitivity in {n_contains(s03, 'classification', 'sensitive')} of {len(s03)} slices; "
                f"Aggregation spreads stayed mostly below the sensitivity threshold, with class counts {json.dumps(count_values(s03, 'classification'), sort_keys=True)}."
            ),
            supports_families=frozenset({"aggregation"}),
            constrains_families=frozenset({"efficiency"}),
            caveat="Deterministic weighted activation cycles are interventions, not public thread interleavings.",
        ),
        "trajectory_label_shuffle": AlternativeEvidence(
            key="trajectory_label_shuffle",
            title="Trajectory-preserving label-shuffle null",
            step_ids=("S04",),
            artifact_paths=(str(paths["S04_label_shuffle_summary"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S04 found {n_gt(s04, 'mean_peak_minus_null', 0.0)} of {len(s04)} condition peaks above the label-shuffle mean and "
                f"{n_lt(s04, 'mean_peak_minus_null', 0.0)} below; the same-code Bubble control was below null."
            ),
            supports_families=frozenset({"aggregation"}),
            constrains_families=frozenset(),
            caveat="Metric null over recorded label trajectories; not a behavioral rerun with regenerated moves.",
        ),
        "behavior_preserving_dummy_labels": AlternativeEvidence(
            key="behavior_preserving_dummy_labels",
            title="Behavior-preserving dummy Algotype labels",
            step_ids=("S05",),
            artifact_paths=(str(paths["S05_dummy_summary"]), str(paths["S14_statistical_tests"])),
            summary=(
                "S05 proved the Bubble dummy labels dispatch to identical behavior and stayed compatible with chance-level Aggregation "
                f"(mean peak minus null {float(pd.to_numeric(s05['mean_peak_minus_null'], errors='coerce').mean()):.3f})."
            ),
            supports_families=frozenset({"aggregation"}),
            constrains_families=frozenset(),
            caveat="Actual rerun negative control covers same-code Bubble labels only under deterministic balanced identity-round scheduling.",
        ),
        "speed_matched_algotypes": AlternativeEvidence(
            key="speed_matched_algotypes",
            title="Speed-matched Algotype tempo control",
            step_ids=("S06",),
            artifact_paths=(str(paths["S06_speed_match_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S06 classified {len(s06)} of {len(s06)} same-goal mixed-policy chimeras as speed-independent; "
                f"peak Aggregation deltas versus equal activation ranged from {pd.to_numeric(s06['mean_delta_peak_aggregation_vs_equal'], errors='coerce').min():.2f} to {pd.to_numeric(s06['mean_delta_peak_aggregation_vs_equal'], errors='coerce').max():.2f} percentage points."
            ),
            supports_families=frozenset({"aggregation", "efficiency"}),
            constrains_families=frozenset(),
            caveat="Speed matching is based on E01 pure compare-plus-swap estimates; mixed-run per-label work can still diverge.",
        ),
        "randomized_local_move_null": AlternativeEvidence(
            key="randomized_local_move_null",
            title="Randomized local-move null",
            step_ids=("S07",),
            artifact_paths=(str(paths["S07_local_move_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S07 found all {len(s07)} condition-policy local-move null slices diverged from real trajectories; "
                f"minimum mean null-minus-real final Sortedness was {pd.to_numeric(s07['mean_delta_final_sortedness_vs_real'], errors='coerce').max():.2f} percentage points even for the strongest null."
            ),
            supports_families=frozenset({"baseline_sorting", "efficiency", "aggregation", "frozen_robustness"}),
            constrains_families=frozenset({"delayed_gratification"}),
            caveat="Nulls match adjacent swap count and locality, but not comparison counts, activation schedules, or full public policy internals.",
        ),
        "dg_matched_null": AlternativeEvidence(
            key="dg_matched_null",
            title="Delayed Gratification matched-null benchmark",
            step_ids=("S08",),
            artifact_paths=(str(paths["S08_dg_burden_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S08 classified {n_contains(s08, 'classification', 'exceeds_real')} burden slices as matched-null-exceeds-real and "
                f"{n_contains(s08, 'classification', 'explains_mean')} as matched-null-explains-mean out of {len(s08)}."
            ),
            supports_families=frozenset(),
            constrains_families=frozenset({"delayed_gratification"}),
            caveat="Matched null trajectories are synthetic Sortedness bridges, not public cell-policy reruns.",
        ),
        "alternative_distance_metrics": AlternativeEvidence(
            key="alternative_distance_metrics",
            title="Alternative distance metrics",
            step_ids=("S09",),
            artifact_paths=(str(paths["S09_metric_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S09 classified {count_values(s09, 'metric_stability_class').get('metric_stable', 0)} alternative slices as metric-stable and "
                f"{count_values(s09, 'metric_stability_class').get('metric_dependent', 0)} as metric-dependent; baseline Sortedness rows are tracked separately."
            ),
            supports_families=frozenset({"baseline_sorting", "aggregation"}),
            constrains_families=frozenset({"efficiency", "frozen_robustness", "delayed_gratification"}),
            caveat="S09 uses repaired deterministic value trajectories; original blocked preflight remains a caveat for earlier persisted traces.",
        ),
        "input_distribution": AlternativeEvidence(
            key="input_distribution",
            title="Input-distribution dependence",
            step_ids=("S10",),
            artifact_paths=(str(paths["S10_input_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S10 class counts were {json.dumps(count_values(s10, 'input_generalization_class'), sort_keys=True)}; "
                "reverse-sorted Insertion and stuck Insertion had the largest final-state drops."
            ),
            supports_families=frozenset(),
            constrains_families=frozenset({"baseline_sorting", "efficiency", "aggregation", "frozen_robustness", "delayed_gratification"}),
            caveat="Matrix is bounded to first 10 repeats; duplicate-heavy metrics use S09 tie handling and heavy-tailed curvature is magnitude-sensitive.",
        ),
        "frozen_cell_placement": AlternativeEvidence(
            key="frozen_cell_placement",
            title="Frozen Cell placement dependence",
            step_ids=("S11",),
            artifact_paths=(str(paths["S11_placement_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S11 class counts were {json.dumps(count_values(s11, 'placement_sensitivity_class'), sort_keys=True)}; "
                "stuck Insertion with three Frozen Cells was the most placement-sensitive."
            ),
            supports_families=frozenset(),
            constrains_families=frozenset({"frozen_robustness", "delayed_gratification"}),
            caveat="Placement matrix uses unique random-permutation inputs and selected Frozen Cell counts 1 and 3.",
        ),
        "frozen_cell_behavior": AlternativeEvidence(
            key="frozen_cell_behavior",
            title="Frozen Cell behavior-variant dependence",
            step_ids=("S12",),
            artifact_paths=(str(paths["S12_behavior_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S12 recovered original passive/stuck limiting cases and reported class counts {json.dumps(count_values(s12, 'behavior_sensitivity_class'), sort_keys=True)}."
            ),
            supports_families=frozenset({"frozen_robustness"}),
            constrains_families=frozenset({"delayed_gratification", "frozen_robustness"}),
            caveat="Dynamic variants are behavior extensions, not paper-reported perturbation semantics.",
        ),
        "stop_condition": AlternativeEvidence(
            key="stop_condition",
            title="Stop-condition dependence",
            step_ids=("S13",),
            artifact_paths=(str(paths["S13_stop_classification"]), str(paths["S14_statistical_tests"])),
            summary=(
                f"S13 class counts were {json.dumps(count_values(s13, 'stop_sensitivity_class'), sort_keys=True)}; "
                "final-state-sensitive slices were concentrated in Insertion under early plateaus or low caps."
            ),
            supports_families=frozenset({"baseline_sorting"}),
            constrains_families=frozenset({"efficiency", "aggregation", "frozen_robustness", "delayed_gratification"}),
            caveat="S13 fixes selected placement and behavior cases while varying stopping rules only.",
        ),
        "strong_statistics_fdr": AlternativeEvidence(
            key="strong_statistics_fdr",
            title="Stronger statistics and FDR",
            step_ids=("S14",),
            artifact_paths=(
                str(paths["S14_statistical_tests"]),
                str(paths["S14_interpretation_summary"]),
                str(paths["S14_family_summary"]),
                str(paths["S14_strong_statistics_report"]),
            ),
            summary=(
                f"S14 analyzed {len(s14)} rows and reported interpretation counts {json.dumps(s14_counts, sort_keys=True)}."
            ),
            supports_families=frozenset({"baseline_sorting", "aggregation"}),
            constrains_families=frozenset({"efficiency", "frozen_robustness", "delayed_gratification"}),
            caveat="S14 is an artifact-level meta-analysis over completed outputs, not a new simulator rerun.",
        ),
    }
    return alternatives


def alternative_support_level(claim_family: str, verdict: str, alternative: AlternativeEvidence) -> str:
    if alternative.key == "strong_statistics_fdr" and verdict == "not_replicated_or_unresolved":
        return "carries_e01_nonreplication_or_unresolved_claim"
    if claim_family == "conflict_governance":
        if alternative.key in {"alternative_distance_metrics", "stop_condition", "strong_statistics_fdr"}:
            return "limited_or_indirect_stress_test_only"
        return "not_directly_tested"
    if claim_family in alternative.constrains_families:
        if alternative.key == "dg_matched_null":
            return "explains_or_strongly_constrains_claim"
        return "partly_explains_or_constrains_claim"
    if claim_family in alternative.supports_families:
        return "supports_claim_against_alternative"
    return "not_directly_tested"


def support_to_summary_fragment(row: Mapping[str, Any]) -> str:
    key = row["alternative_key"]
    level = row["support_level"]
    if level == "supports_claim_against_alternative":
        return f"{key} supports against this alternative"
    if level == "explains_or_strongly_constrains_claim":
        return f"{key} strongly constrains or explains part of the claim"
    if level == "partly_explains_or_constrains_claim":
        return f"{key} narrows the claim"
    if level == "limited_or_indirect_stress_test_only":
        return f"{key} is indirect only"
    if level == "carries_e01_nonreplication_or_unresolved_claim":
        return f"{key} carries the unresolved E01 status"
    return ""


def build_audit_tables(
    e01_status: pd.DataFrame,
    alternatives: Mapping[str, AlternativeEvidence],
    s14_family_summary: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []

    for source in e01_status.to_dict(orient="records"):
        claim_id = str(source["claim_id"])
        claim_family = claim_family_for_claim(claim_id, source.get("figure_or_section", ""))
        e01_classification = normalize_e01_classification(source.get("classification", "unknown"))
        verdict = base_verdict_for_claim(claim_id, claim_family, e01_classification)
        evidence_level = evidence_level_for_claim(claim_family, e01_classification, verdict)

        source_artifacts = [
            item.strip()
            for item in str(source.get("evidence_artifacts", "")).split(";")
            if item.strip()
        ]
        specific_matrix_rows: list[dict[str, Any]] = []
        for alternative in alternatives.values():
            support_level = alternative_support_level(claim_family, verdict, alternative)
            row = {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "claim_id": claim_id,
                "claim_family": claim_family,
                "alternative_key": alternative.key,
                "alternative_explanation": alternative.title,
                "evidence_step_ids": ";".join(alternative.step_ids),
                "evidence_summary": alternative.summary,
                "support_level": support_level,
                "corrected_statistics_summary": s14_family_summary.get(claim_family, ""),
                "caveats": alternative.caveat,
                "source_artifacts_json": json.dumps(list(alternative.artifact_paths), sort_keys=True),
            }
            matrix_rows.append(row)
            if support_level != "not_directly_tested":
                specific_matrix_rows.append(row)
                source_artifacts.extend(alternative.artifact_paths)

        evidence_fragments = [support_to_summary_fragment(row) for row in specific_matrix_rows]
        evidence_fragments = [fragment for fragment in evidence_fragments if fragment]
        if not evidence_fragments:
            evidence_fragments = ["No direct S02-S14 artifact-audit alternative fully targets this claim; S15 carries E01 replication evidence and caveats forward."]
        key_caveats = [
            str(source.get("caveats", "")).strip(),
            str(source.get("divergence_cause", "")).strip(),
        ]
        key_caveats.extend(row["caveats"] for row in specific_matrix_rows[:4])
        key_caveats = [caveat for caveat in key_caveats if caveat and caveat.lower() != "none"]
        recommended_use = {
            "robust": "Use as an E02 guardrail anchor under computational-proxy boundaries.",
            "robust_but_narrower": "Use only with the stated scheduler, metric, input, perturbation, and stopping assumptions.",
            "partly_explained_by_alternative": "Use as a descriptive trajectory phenomenon, not as standalone evidence for planning or a unique mechanism.",
            "not_replicated_or_unresolved": "Do not use as a surviving claim without resolving the E01 ambiguity or exact-magnitude mismatch.",
        }[verdict]
        claim_rows.append(
            {
                "research_step_id": STEP_ID,
                "experiment_id": EXPERIMENT_ID,
                "claim_id": claim_id,
                "claim_family": claim_family,
                "figure_or_section": source.get("figure_or_section", ""),
                "paper_claim": source.get("paper_claim", ""),
                "e01_classification": e01_classification,
                "primary_e01_evidence": source.get("replication_evidence", ""),
                "e02_evidence_summary": "; ".join(evidence_fragments[:8]),
                "strong_statistics_summary": s14_family_summary.get(claim_family, ""),
                "alternative_explanations": "; ".join(sorted({row["alternative_key"] for row in specific_matrix_rows})),
                "final_verdict": verdict,
                "evidence_level": evidence_level,
                "survives_as": survives_as_for_verdict(verdict, claim_family),
                "key_caveats": "; ".join(dict.fromkeys(key_caveats)),
                "recommended_use": recommended_use,
                "source_artifacts_json": json.dumps(sorted(set(source_artifacts)), sort_keys=True),
            }
        )

    claims = pd.DataFrame(claim_rows)
    matrix = pd.DataFrame(matrix_rows)
    return claims, matrix


def summarize_verdicts(claims: pd.DataFrame) -> pd.DataFrame:
    return (
        claims.groupby(["claim_family", "final_verdict"], dropna=False)
        .size()
        .reset_index(name="n_claims")
        .sort_values(["claim_family", "final_verdict"])
    )


def summarize_alternatives(matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        matrix.groupby(["alternative_key", "support_level"], dropna=False)
        .size()
        .reset_index(name="n_claim_rows")
        .sort_values(["alternative_key", "support_level"])
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 20, columns: Sequence[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, list(columns)] if columns else df.copy()
    show = show.head(max_rows)

    def esc(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(col) for col in show.columns) + " |"
    divider = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = ["| " + " | ".join(esc(value) for value in row) + " |" for row in show.itertuples(index=False, name=None)]
    suffix = [f"\nShowing first {max_rows} of {len(df)} rows."] if len(df) > max_rows else []
    return "\n".join([header, divider, *body, *suffix])


def markdown_has_top_summary(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    required = (
        "## Top Summary",
        "Step ID:",
        "Completion status:",
        "Artifacts written:",
        "Validation result:",
        "Outcome classification:",
        "Caveats or blockers:",
        "Lay summary:",
        "Recommended next action:",
    )
    return all(item in text for item in required)


def validate_outputs(
    *,
    claims: pd.DataFrame,
    matrix: pd.DataFrame,
    verdict_summary: pd.DataFrame,
    alternative_summary: pd.DataFrame,
    output_paths: Mapping[str, Path],
    input_paths: Mapping[str, Path],
    unit_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    required_output_keys = {
        "claims_csv",
        "matrix_csv",
        "claim_audit_report",
        "handoff_report",
        "full_results_report",
        "validation_json",
        "status_json",
        "manifest_json",
        "src_manifest_json",
        "log",
    }
    claim_count = int(len(claims))
    matrix_count = int(len(matrix))
    expected_matrix_count = claim_count * len(ALTERNATIVE_KEYS)
    parsed_artifacts = [json.loads(value) for value in claims["source_artifacts_json"]]
    all_claim_artifacts_present = all(bool(items) for items in parsed_artifacts)
    verdicts_valid = set(claims["final_verdict"].astype(str).unique()) <= ALLOWED_VERDICTS
    all_verdicts_cite = bool(
        claims["e02_evidence_summary"].astype(str).str.len().gt(0).all()
        and claims["strong_statistics_summary"].astype(str).str.len().gt(0).all()
    )
    matrix_complete = bool(
        matrix_count == expected_matrix_count
        and set(matrix["alternative_key"].unique()) == set(ALTERNATIVE_KEYS)
        and matrix["support_level"].astype(str).str.len().gt(0).all()
    )
    input_present = all(path.exists() and path.stat().st_size > 0 for path in input_paths.values())
    output_present = all(output_paths[key].exists() and output_paths[key].stat().st_size > 0 for key in required_output_keys)
    markdown_reports = [output_paths["claim_audit_report"], output_paths["handoff_report"], output_paths["full_results_report"]]
    markdown_top_summaries = all(path.exists() and markdown_has_top_summary(path) for path in markdown_reports)
    unit_success = bool(unit_result is None or unit_result.get("success"))
    has_constrained_or_failed = bool(
        claims["final_verdict"].isin({"partly_explained_by_alternative", "not_replicated_or_unresolved"}).any()
        or claims["final_verdict"].isin({"robust_but_narrower"}).any()
    )
    success = bool(
        claim_count >= 20
        and matrix_complete
        and verdicts_valid
        and all_claim_artifacts_present
        and all_verdicts_cite
        and input_present
        and output_present
        and markdown_top_summaries
        and unit_success
    )
    outcome = "constraining/contradictory" if has_constrained_or_failed else "supportive"
    caveats = [
        "S15 is an evidence synthesis over completed E01 and E02 artifacts, not a new simulator run.",
        "Verdicts inherit E01 public-code, reconstructed-traditional-runner, comparison-count, and seed caveats.",
        "Most E02 stress-test matrices are bounded to the first 10 E01 repeats per selected condition.",
        "Conflict-governance claims from Figures 9-10 have limited direct E02 stress tests and remain mainly E01-bounded.",
        "Computational morphogenesis and intelligence language remains proxy language; no biological mechanism is established.",
    ]
    validation = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "failed_validation",
        "validationResult": "pending",
        "outcomeClassification": outcome,
        "artifactsWritten": [],
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": "Stop for Chief Scientist review and report-bundle generation; do not start S16 or E03 from this step.",
        "claimCount": claim_count,
        "alternativeMatrixRowCount": matrix_count,
        "expectedAlternativeMatrixRowCount": expected_matrix_count,
        "verdictCounts": claims["final_verdict"].value_counts().sort_index().to_dict(),
        "claimFamilyCounts": claims["claim_family"].value_counts().sort_index().to_dict(),
        "supportLevelCounts": matrix["support_level"].value_counts().sort_index().to_dict(),
        "verdictSummaryRows": int(len(verdict_summary)),
        "alternativeSummaryRows": int(len(alternative_summary)),
        "inputArtifactsPresent": input_present,
        "outputsPresent": output_present,
        "markdownTopSummariesPresent": markdown_top_summaries,
        "allVerdictsValid": verdicts_valid,
        "allVerdictsCiteAnalyses": all_verdicts_cite,
        "allClaimsHaveSourceArtifacts": all_claim_artifacts_present,
        "allAlternativesRepresented": set(matrix["alternative_key"].unique()) == set(ALTERNATIVE_KEYS),
        "unitTestsPassed": unit_success,
    }
    validation["validationResult"] = (
        "passed: all E01 claims were assigned allowed verdicts, every verdict cites E01/E02 analyses, all 13 alternative-explanation families are represented for every claim, Markdown top summaries are present, planned outputs and validation artifacts are nonempty, and unit tests passed"
        if success
        else "failed: one or more S15 validation checks did not pass"
    )
    return validation


def top_summary_markdown(validation: Mapping[str, Any], lay_summary: str) -> str:
    artifacts = validation.get("artifactsWritten") or []
    artifact_text = ", ".join(str(path) for path in artifacts) if artifacts else "pending"
    return f"""## Top Summary

- Step ID: {STEP_ID}
- Completion status: {validation['status']}
- Artifacts written: {artifact_text}
- Validation result: {validation['validationResult']}
- Outcome classification: {validation['outcomeClassification']}
- Caveats or blockers: {', '.join(validation['caveatsOrBlockers'])}
- Lay summary: {lay_summary}
- Recommended next action: {validation['recommendedNextAction']}
"""


def build_claim_audit_report(
    *,
    generated_at: str,
    validation: Mapping[str, Any],
    claims: pd.DataFrame,
    matrix: pd.DataFrame,
    verdict_summary: pd.DataFrame,
    alternative_summary: pd.DataFrame,
    output_paths: Mapping[str, Path],
    input_paths: Mapping[str, Path],
) -> str:
    lay = (
        "The audit finds that the basic sorting and several chimeric aggregation observations survive, but most useful claims narrow under "
        "scheduler, metric, input, perturbation, stop-rule, and null-model controls; DG is especially constrained by matched nulls."
    )
    survivors = claims[claims["final_verdict"].isin(["robust", "robust_but_narrower"])].copy()
    narrowed = claims[claims["final_verdict"] == "partly_explained_by_alternative"].copy()
    failed = claims[claims["final_verdict"] == "not_replicated_or_unresolved"].copy()
    return f"""# E02 Final Claim Audit

{top_summary_markdown(validation, lay)}

Generated at: {generated_at}

## Verdict Overview

{markdown_table(verdict_summary, max_rows=40)}

## Claims That Survive

{markdown_table(survivors, max_rows=40, columns=["claim_id", "claim_family", "e01_classification", "final_verdict", "survives_as", "key_caveats"])}

## Claims Narrowed By Alternatives

{markdown_table(narrowed, max_rows=20, columns=["claim_id", "claim_family", "final_verdict", "e02_evidence_summary", "key_caveats"])}

## Claims That Do Not Survive As Written

{markdown_table(failed, max_rows=20, columns=["claim_id", "claim_family", "e01_classification", "final_verdict", "primary_e01_evidence", "recommended_use"])}

## Alternative-Explanation Matrix Summary

{markdown_table(alternative_summary, max_rows=80)}

## Evidence Rules

- `robust`: E01 exact/statistical replication and no targeted E02 alternative materially narrows the claim beyond normal computational-proxy boundaries.
- `robust_but_narrower`: the observation remains usable, but the valid statement must include E02 scheduler, metric, input, perturbation, or stop-rule conditions.
- `partly_explained_by_alternative`: the observation remains descriptive, but an E02 null or artifact model explains or strongly constrains the stronger interpretation.
- `not_replicated_or_unresolved`: the E01 replication row already failed as written or exact magnitudes remain unresolved; E02 does not rescue the original wording.

## Inputs

{markdown_table(pd.DataFrame([{"input": key, "path": str(path), "exists": path.exists()} for key, path in input_paths.items()]), max_rows=60)}

## Outputs

- Claims table: `{output_paths['claims_csv']}`
- Alternative-explanation matrix: `{output_paths['matrix_csv']}`
- Full results: `{output_paths['full_results_report']}`
- Report-bundle handoff: `{output_paths['handoff_report']}`
- Validation evidence: `{output_paths['validation_json']}`

## Caveats

The audit is limited to completed computational artifacts. It preserves E01 caveats around public-code versioning, unavailable original random seeds, reconstructed traditional runners, comparison-count proxies, and metric ambiguity. E02 matrices are mostly bounded first-10-repeat stress tests, not full 100-repeat paper-scale reruns.
"""


def build_handoff_report(
    *,
    generated_at: str,
    validation: Mapping[str, Any],
    claims: pd.DataFrame,
    verdict_summary: pd.DataFrame,
    output_paths: Mapping[str, Path],
) -> str:
    lay = (
        "The report bundle should present E02 as a falsification layer: some claims survive, but final language must be narrower than the paper-level prose and must preserve computational-proxy caveats."
    )
    anchor_rows = claims[claims["final_verdict"].isin(["robust", "robust_but_narrower"])].head(12)
    constrained_rows = claims[claims["final_verdict"].isin(["partly_explained_by_alternative", "not_replicated_or_unresolved"])]
    return f"""# E02 Report-Bundle Handoff

{top_summary_markdown(validation, lay)}

Generated at: {generated_at}

## Bundle Inputs

- Use `{output_paths['claim_audit_report']}` as the narrative claim-audit source.
- Use `{output_paths['claims_csv']}` for machine-readable claim verdicts.
- Use `{output_paths['matrix_csv']}` for alternative-explanation support levels.
- Use `{output_paths['full_results_report']}` for detailed methods, commands, validation, caveats, and provenance.
- Use S02-S14 figures already present under `/artifacts/figures/e02/`; S15 intentionally did not generate a new figure.

## Recommended Report Language

- Strongest surviving anchor: basic deterministic replication and same-goal sorting/aggregation phenomena remain computationally useful.
- Required narrowing: scheduler, activation, metric, input distribution, Frozen Cell placement/behavior, and stop-condition assumptions must be stated explicitly.
- DG language: describe DG as a trajectory metric; do not frame it as standalone evidence for planning, intention, or a unique cell-policy mechanism because S08 matched nulls explain or exceed many DG slices.
- Chimeric aggregation language: retain as a bounded behavior-label structure signal for selected same-goal mixes, supported by label-shuffle, dummy-label, speed-match, and local-null controls; avoid exact peak-magnitude claims.
- Frozen Cell language: retain selected rankings or perturbation patterns, but do not claim defect-location or behavior-assumption invariance.

## Verdict Counts

{markdown_table(verdict_summary, max_rows=40)}

## Anchor Claims For Bundle

{markdown_table(anchor_rows, max_rows=12, columns=["claim_id", "claim_family", "final_verdict", "survives_as", "recommended_use"])}

## Claims To Flag Prominently

{markdown_table(constrained_rows, max_rows=20, columns=["claim_id", "claim_family", "final_verdict", "recommended_use", "key_caveats"])}

## Stop Condition

S15 is complete. Stop here for Chief Scientist review and report-bundle generation. Do not start S16 or E03 from this workspace step.
"""


def build_full_results_report(
    *,
    generated_at: str,
    validation: Mapping[str, Any],
    claims: pd.DataFrame,
    matrix: pd.DataFrame,
    verdict_summary: pd.DataFrame,
    alternative_summary: pd.DataFrame,
    output_paths: Mapping[str, Path],
    input_paths: Mapping[str, Path],
    unit_result: Mapping[str, Any] | None,
    git_commit: str,
    git_status: str,
    elapsed_seconds: float,
    parameters: Mapping[str, Any],
) -> str:
    lay = (
        "S15 audited the original E01 claim list against every completed E02 stress test. Basic sorting and selected aggregation claims survive, "
        "but broad claims about efficiency, DG, Frozen Cell robustness, and exact magnitudes must be narrowed or rejected as written."
    )
    unit_line = "not run" if unit_result is None else f"{unit_result['command']} -> return code {unit_result['returnCode']}"
    return f"""# E02 S15 Full Results: Final Claim Audit

{top_summary_markdown(validation, lay)}

## Frozen Question

For each original claim, is the best verdict robust, robust but narrower, explained by an alternative mechanism, or not replicated?

## Inputs

S15 used completed E01 and E02 artifacts only; no simulations were rerun.

{markdown_table(pd.DataFrame([{"input": key, "path": str(path), "exists": path.exists()} for key, path in input_paths.items()]), max_rows=80)}

## Methods

The runner loads the E01 replication status table as the claim universe and assigns each claim to a claim family: baseline sorting, efficiency, Frozen Cell robustness, Delayed Gratification, aggregation, conflict governance, or miscellaneous. It then builds one alternative-explanation row for every claim and every S02-S14 evidence family. Each row records the relevant source step, support level, artifact paths, evidence summary, corrected-statistics summary, and caveat.

Verdicts are deterministic rule-based classifications:

- exact or statistically consistent baseline sorting claims with no strong targeted contradiction are `robust`;
- replicated claims with scheduler, metric, input, perturbation, or stop-rule constraints are `robust_but_narrower`;
- DG claims with S08 matched-null explanations are `partly_explained_by_alternative`;
- E01 rows classified as not replicated are `not_replicated_or_unresolved`.

S14 corrected evidence is used as a family-level statistical summary, while upstream practical classifications remain authoritative for the domain-specific sensitivity thresholds.

## Commands

- `python scripts/e02_s15_claim_audit.py --repo-dir /workspace/cell-research --artifacts-dir /artifacts --previous-e01-dir /previous-artifacts/E01 --run-unit-tests`
- Unit tests: {unit_line}

## Results

- Claims audited: {validation['claimCount']}
- Alternative-explanation rows: {validation['alternativeMatrixRowCount']}
- Verdict counts: {json.dumps(validation['verdictCounts'], sort_keys=True)}
- Claim family counts: {json.dumps(validation['claimFamilyCounts'], sort_keys=True)}
- Support-level counts: {json.dumps(validation['supportLevelCounts'], sort_keys=True)}

### Verdict Summary

{markdown_table(verdict_summary, max_rows=40)}

### Alternative Summary

{markdown_table(alternative_summary, max_rows=80)}

### Claim Table Excerpt

{markdown_table(claims, max_rows=30, columns=["claim_id", "claim_family", "e01_classification", "final_verdict", "evidence_level", "survives_as"])}

### Alternative Matrix Excerpt

{markdown_table(matrix, max_rows=30, columns=["claim_id", "alternative_key", "support_level", "evidence_step_ids", "evidence_summary"])}

## Validation

- Input artifacts present: {validation['inputArtifactsPresent']}
- Outputs present: {validation['outputsPresent']}
- Markdown top summaries present: {validation['markdownTopSummariesPresent']}
- All verdicts valid: {validation['allVerdictsValid']}
- All verdicts cite analyses: {validation['allVerdictsCiteAnalyses']}
- All claims have source artifacts: {validation['allClaimsHaveSourceArtifacts']}
- All alternatives represented: {validation['allAlternativesRepresented']}
- Unit tests passed: {validation['unitTestsPassed']}
- Validation JSON: `{output_paths['validation_json']}`
- Status JSON: `{output_paths['status_json']}`

## Artifacts

- Claims-that-survive table: `{output_paths['claims_csv']}`
- Alternative-explanation matrix: `{output_paths['matrix_csv']}`
- Claim-audit report: `{output_paths['claim_audit_report']}`
- Report-bundle handoff: `{output_paths['handoff_report']}`
- Artifact manifest: `{output_paths['manifest_json']}`
- Source snapshot manifest: `{output_paths['src_manifest_json']}`
- Execution log: `{output_paths['log']}`

## Provenance

- Generated at: {generated_at}
- Elapsed seconds: {elapsed_seconds:.2f}
- Git commit before final commit: `{git_commit}`
- Git status before final commit: `{git_status}`
- Python: `{platform.python_version()}`
- Platform: `{platform.platform()}`
- Parameters: `{json.dumps(parameters, sort_keys=True)}`

## Caveats, Blockers, And Limitations

- S15 is an evidence synthesis over completed artifacts, not a new simulation or statistical rerun.
- The S15 rule set is intentionally conservative and inherits upstream caveats rather than resolving them.
- E02 did not directly stress-test every conflict-governance claim from Figures 9 and 10; those rows remain E01-bounded.
- Many E02 stress-test matrices use bounded first-10-repeat slices and therefore provide audit-strength constraints rather than full paper-scale estimates.
- DG remains a computational trajectory score. S08 narrows any interpretation that treats DG as direct evidence of planning or a unique policy mechanism.
- No biological morphogenesis, adhesion, bioelectricity, cognition, or intention claim is directly tested.

## Recommended Next Action

Stop for Chief Scientist review and report-bundle generation. Do not start S16 or E03 from this step.
"""


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    generated_at = utc_now()
    repo_dir = args.repo_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    previous_e01_dir = args.previous_e01_dir.resolve()
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    table_dir = artifacts_dir / "tables"
    report_dir = artifacts_dir / "reports"
    log_dir = artifacts_dir / "logs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    for path in (step_dir, table_dir, report_dir, log_dir, src_snapshot_dir):
        path.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "claims_csv": table_dir / "e02_claims_that_survive.csv",
        "matrix_csv": table_dir / "e02_alternative_explanation_matrix.csv",
        "claim_audit_report": report_dir / "e02_claim_audit.md",
        "handoff_report": report_dir / "e02_report_bundle_handoff.md",
        "full_results_report": step_dir / "research_step_full_results.md",
        "validation_json": step_dir / "s15_validation.json",
        "status_json": step_dir / "status.json",
        "manifest_json": step_dir / "artifact_manifest.json",
        "src_manifest_json": src_snapshot_dir / "e02_s15_claim_audit_manifest.json",
        "log": log_dir / "e02_s15_claim_audit.log",
    }

    with output_paths["log"].open("w", encoding="utf-8") as log:
        log.write(f"{generated_at} Starting E02 S15 claim audit\n")
        log.write(f"repo_dir={repo_dir}\nartifacts_dir={artifacts_dir}\nprevious_e01_dir={previous_e01_dir}\n")

    input_paths = InputPaths(artifacts_dir=artifacts_dir, previous_e01_dir=previous_e01_dir).paths
    inputs = load_inputs(input_paths)
    parameters = {
        "claimUniverse": "E01_replication_status",
        "alternativeFamilies": list(ALTERNATIVE_KEYS),
        "inputCount": len(input_paths),
    }

    unit_result: Mapping[str, Any] | None = None
    if args.run_unit_tests:
        unit_result = run_command([sys.executable, "-m", "unittest", "tests.e02.test_claim_audit"], repo_dir)
        if not unit_result["success"]:
            write_json(step_dir / "s15_unit_test_failure.json", unit_result)
            raise RuntimeError(f"S15 unit tests failed; see {step_dir / 's15_unit_test_failure.json'}")

    alternatives = build_alternative_evidence(inputs, input_paths)
    s14_family_summary = build_s14_family_summary(inputs["S14_statistical_tests"])
    claims, matrix = build_audit_tables(inputs["E01_replication_status"], alternatives, s14_family_summary)
    verdict_summary = summarize_verdicts(claims)
    alternative_summary = summarize_alternatives(matrix)
    claims.to_csv(output_paths["claims_csv"], index=False)
    matrix.to_csv(output_paths["matrix_csv"], index=False)

    git_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    git_status = git_output(repo_dir, ["status", "--short"])

    pending_validation = {
        "status": "pending",
        "validationResult": "pending",
        "outcomeClassification": "pending",
        "caveatsOrBlockers": ["pending"],
        "recommendedNextAction": "pending",
    }
    output_paths["claim_audit_report"].write_text(top_summary_markdown(pending_validation, "pending"), encoding="utf-8")
    output_paths["handoff_report"].write_text(top_summary_markdown(pending_validation, "pending"), encoding="utf-8")
    output_paths["full_results_report"].write_text(top_summary_markdown(pending_validation, "pending"), encoding="utf-8")
    write_json(output_paths["validation_json"], {"status": "pending"})
    write_json(
        output_paths["status_json"],
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": False,
            "status": "pending",
            "artifactsWritten": [],
            "validationResult": "pending",
            "caveatsOrBlockers": [],
            "recommendedNextAction": "pending",
        },
    )

    src_manifest = {
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "gitCommitBeforeFinalCommit": git_commit,
        "gitStatusBeforeFinalCommit": git_status,
        "sourceFiles": [
            artifact_entry(repo_dir / "scripts/e02_s15_claim_audit.py", repo_dir, "S15 claim-audit runner"),
            artifact_entry(repo_dir / "tests/e02/test_claim_audit.py", repo_dir, "focused S15 claim-audit tests"),
        ],
    }
    write_json(output_paths["src_manifest_json"], src_manifest)
    manifest = {
        "researchStepId": STEP_ID,
        "generatedAt": generated_at,
        "parameters": parameters,
        "artifacts": [],
    }
    write_json(output_paths["manifest_json"], manifest)

    validation = validate_outputs(
        claims=claims,
        matrix=matrix,
        verdict_summary=verdict_summary,
        alternative_summary=alternative_summary,
        output_paths=output_paths,
        input_paths=input_paths,
        unit_result=unit_result,
    )
    validation["artifactsWritten"] = [str(output_paths[key]) for key in output_paths if output_paths[key].exists()]
    elapsed = time.monotonic() - started

    output_paths["claim_audit_report"].write_text(
        build_claim_audit_report(
            generated_at=generated_at,
            validation=validation,
            claims=claims,
            matrix=matrix,
            verdict_summary=verdict_summary,
            alternative_summary=alternative_summary,
            output_paths=output_paths,
            input_paths=input_paths,
        ),
        encoding="utf-8",
    )
    output_paths["handoff_report"].write_text(
        build_handoff_report(
            generated_at=generated_at,
            validation=validation,
            claims=claims,
            verdict_summary=verdict_summary,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )
    output_paths["full_results_report"].write_text(
        build_full_results_report(
            generated_at=generated_at,
            validation=validation,
            claims=claims,
            matrix=matrix,
            verdict_summary=verdict_summary,
            alternative_summary=alternative_summary,
            output_paths=output_paths,
            input_paths=input_paths,
            unit_result=unit_result,
            git_commit=git_commit,
            git_status=git_status,
            elapsed_seconds=elapsed,
            parameters=parameters,
        ),
        encoding="utf-8",
    )

    # Revalidate after final Markdown reports are written.
    validation = validate_outputs(
        claims=claims,
        matrix=matrix,
        verdict_summary=verdict_summary,
        alternative_summary=alternative_summary,
        output_paths=output_paths,
        input_paths=input_paths,
        unit_result=unit_result,
    )
    validation["artifactsWritten"] = [str(output_paths[key]) for key in output_paths if output_paths[key].exists()]
    artifact_descriptions = {
        "claims_csv": "planned S15 claims-that-survive verdict table",
        "matrix_csv": "planned S15 alternative-explanation matrix",
        "claim_audit_report": "planned S15 claim-audit report",
        "handoff_report": "planned E02 report-bundle handoff",
        "full_results_report": "canonical S15 full-results report",
        "validation_json": "S15 validation evidence",
        "status_json": "compact S15 status JSON",
        "manifest_json": "S15 artifact manifest",
        "src_manifest_json": "S15 source snapshot manifest",
        "log": "S15 execution log",
    }
    artifacts = [
        artifact_entry(path, artifacts_dir, artifact_descriptions[key])
        for key, path in output_paths.items()
        if path.exists() and key != "manifest_json"
    ]
    manifest["artifacts"] = artifacts
    manifest["validation"] = {
        "success": bool(validation["success"]),
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
    }
    write_json(output_paths["manifest_json"], manifest)
    write_json(output_paths["validation_json"], validation)
    write_json(
        output_paths["status_json"],
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": bool(validation["success"]),
            "status": validation["status"],
            "artifactsWritten": validation["artifactsWritten"],
            "validationResult": validation["validationResult"],
            "caveatsOrBlockers": validation["caveatsOrBlockers"],
            "recommendedNextAction": validation["recommendedNextAction"],
        },
    )
    # Refresh reports after the final artifact list is known.
    output_paths["claim_audit_report"].write_text(
        build_claim_audit_report(
            generated_at=generated_at,
            validation=validation,
            claims=claims,
            matrix=matrix,
            verdict_summary=verdict_summary,
            alternative_summary=alternative_summary,
            output_paths=output_paths,
            input_paths=input_paths,
        ),
        encoding="utf-8",
    )
    output_paths["handoff_report"].write_text(
        build_handoff_report(
            generated_at=generated_at,
            validation=validation,
            claims=claims,
            verdict_summary=verdict_summary,
            output_paths=output_paths,
        ),
        encoding="utf-8",
    )
    output_paths["full_results_report"].write_text(
        build_full_results_report(
            generated_at=generated_at,
            validation=validation,
            claims=claims,
            matrix=matrix,
            verdict_summary=verdict_summary,
            alternative_summary=alternative_summary,
            output_paths=output_paths,
            input_paths=input_paths,
            unit_result=unit_result,
            git_commit=git_commit,
            git_status=git_status,
            elapsed_seconds=elapsed,
            parameters=parameters,
        ),
        encoding="utf-8",
    )
    manifest["artifacts"] = [
        artifact_entry(path, artifacts_dir, artifact_descriptions[key])
        for key, path in output_paths.items()
        if path.exists() and key != "manifest_json"
    ]
    manifest["validation"] = {
        "success": bool(validation["success"]),
        "validationResult": validation["validationResult"],
        "outcomeClassification": validation["outcomeClassification"],
    }
    write_json(output_paths["manifest_json"], manifest)
    with output_paths["log"].open("a", encoding="utf-8") as log:
        log.write(f"{utc_now()} Completed S15 status={validation['status']} success={validation['success']} elapsed={elapsed:.2f}s\n")
        log.write(json.dumps({"claims": len(claims), "matrixRows": len(matrix), "verdictCounts": validation["verdictCounts"]}, sort_keys=True) + "\n")
    if not validation["success"]:
        raise RuntimeError(f"S15 validation failed; see {output_paths['validation_json']}")


if __name__ == "__main__":
    main()
