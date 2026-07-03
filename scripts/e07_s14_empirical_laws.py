#!/usr/bin/env python3
"""Run E07 S14 empirical-law synthesis from S09-S13 evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.empirical_law_schema import (  # noqa: E402
    EMPIRICAL_LAW_SCHEMA_VERSION,
    dataframe_from_law_records,
    validate_empirical_law_artifacts,
    validation_summary,
)


STEP_ID = "S14"
STEP_NUMBER = 14
EXPERIMENT_ID = "E07"
RANDOM_SEED = 2026070314


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--s09-candidate-invariants", type=Path, default=artifacts_dir / "results" / "e07_candidate_invariants.parquet")
    parser.add_argument("--s09-model-metrics", type=Path, default=artifacts_dir / "results" / "e07_invariant_model_metrics.parquet")
    parser.add_argument("--s09-counterexamples", type=Path, default=artifacts_dir / "results" / "e07_invariant_counterexamples.parquet")
    parser.add_argument("--s09-validation-checks", type=Path, default=artifacts_dir / "research_steps" / "S09" / "e07_s09_validation_checks.csv")
    parser.add_argument("--s10-classes", type=Path, default=artifacts_dir / "results" / "e07_universality_classes.parquet")
    parser.add_argument("--s10-assignments", type=Path, default=artifacts_dir / "results" / "e07_universality_assignments.parquet")
    parser.add_argument("--s10-exemplars", type=Path, default=artifacts_dir / "tables" / "e07_universality_exemplars.csv")
    parser.add_argument("--s10-counterexamples", type=Path, default=artifacts_dir / "tables" / "e07_universality_counterexamples.csv")
    parser.add_argument("--s10-validation-checks", type=Path, default=artifacts_dir / "research_steps" / "S10" / "e07_s10_validation_checks.csv")
    parser.add_argument("--s11-predictions", type=Path, default=artifacts_dir / "results" / "e07_counterfactual_predictions.parquet")
    parser.add_argument("--s11-validations", type=Path, default=artifacts_dir / "results" / "e07_counterfactual_validations.parquet")
    parser.add_argument("--s11-accuracy-summary", type=Path, default=artifacts_dir / "tables" / "e07_counterfactual_accuracy_summary.csv")
    parser.add_argument("--s11-validation-checks", type=Path, default=artifacts_dir / "research_steps" / "S11" / "e07_s11_validation_checks.csv")
    parser.add_argument("--s12-validation", type=Path, default=artifacts_dir / "results" / "e07_inverse_design_validation.parquet")
    parser.add_argument("--s12-validation-summary", type=Path, default=artifacts_dir / "tables" / "e07_inverse_design_validation_summary.csv")
    parser.add_argument("--s12-validation-checks", type=Path, default=artifacts_dir / "research_steps" / "S12" / "e07_s12_validation_checks.csv")
    parser.add_argument("--s13-transfer", type=Path, default=artifacts_dir / "results" / "e07_substrate_transfer.parquet")
    parser.add_argument("--s13-transfer-coverage", type=Path, default=artifacts_dir / "tables" / "e07_substrate_transfer_coverage.csv")
    parser.add_argument("--s13-validation-checks", type=Path, default=artifacts_dir / "research_steps" / "S13" / "e07_s13_validation_checks.csv")
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def maybe_sha256(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run_command(command: list[str], repo_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=repo_dir, capture_output=True, text=True)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "command": " ".join(command),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": result.stdout[-6000:],
        "stderr": result.stderr[-6000:],
        "success": result.returncode == 0,
    }


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path) if path.is_file() else None,
        "sizeBytes": path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "artifactType": "directory" if path.is_dir() else "file",
    }


def self_referential_artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "artifactType": "file",
        "note": "Checksum omitted because this report or manifest contains the artifact list.",
    }


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        view[column] = view[column].map(format_cell)
    header = "| " + " | ".join(map(str, view.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.to_numpy()]
    if len(frame) > max_rows:
        rows.append(f"| ... | {len(frame) - max_rows} more rows omitted |" + " |" * max(0, len(view.columns) - 2))
    return "\n".join([header, separator, *rows])


def validation_counts(path: Path) -> dict[str, Any]:
    checks = pd.read_csv(path)
    return {"passed": int(checks["success"].sum()), "total": int(len(checks)), "allPassed": bool(checks["success"].all())}


def build_source_manifest(args: argparse.Namespace, artifacts_dir: Path) -> pd.DataFrame:
    specs = [
        ("S09", "candidate_invariants", args.s09_candidate_invariants, "Stable and unstable invariant candidates."),
        ("S09", "invariant_model_metrics", args.s09_model_metrics, "Held-out invariant model checks."),
        ("S09", "invariant_counterexamples", args.s09_counterexamples, "Counterexamples for invariant candidates."),
        ("S09", "validation_checks", args.s09_validation_checks, "S09 validation outcome."),
        ("S10", "universality_classes", args.s10_classes, "Conservative universality-class rows."),
        ("S10", "universality_assignments", args.s10_assignments, "Entity assignments to S10 classes."),
        ("S10", "universality_exemplars", args.s10_exemplars, "S10 class exemplars."),
        ("S10", "universality_counterexamples", args.s10_counterexamples, "S10 class counterexamples."),
        ("S10", "validation_checks", args.s10_validation_checks, "S10 validation outcome."),
        ("S11", "counterfactual_predictions", args.s11_predictions, "Frozen counterfactual predictions."),
        ("S11", "counterfactual_validations", args.s11_validations, "Counterfactual validation and blocker rows."),
        ("S11", "counterfactual_accuracy_summary", args.s11_accuracy_summary, "Counterfactual baseline comparison summary."),
        ("S11", "validation_checks", args.s11_validation_checks, "S11 validation outcome."),
        ("S12", "inverse_design_validation", args.s12_validation, "Inverse-design heldout validation and blocker rows."),
        ("S12", "inverse_design_validation_summary", args.s12_validation_summary, "Inverse-design summary."),
        ("S12", "validation_checks", args.s12_validation_checks, "S12 validation outcome."),
        ("S13", "substrate_transfer", args.s13_transfer, "S13 transfer result corpus and blocker rows."),
        ("S13", "substrate_transfer_coverage", args.s13_transfer_coverage, "S13 transfer coverage summary."),
        ("S13", "validation_checks", args.s13_validation_checks, "S13 validation outcome."),
    ]
    rows: list[dict[str, Any]] = []
    for step_id, name, path, role in specs:
        row_count = 0
        if path.exists() and path.is_file():
            try:
                row_count = len(read_table(path))
            except Exception:
                row_count = 1
        rows.append(
            {
                "source_step_id": step_id,
                "input_name": name,
                "path": str(path),
                "relative_path": str(path.relative_to(artifacts_dir)) if str(path).startswith(str(artifacts_dir)) else str(path),
                "sha256": maybe_sha256(path),
                "row_count": int(row_count),
                "role": role,
            }
        )
    return pd.DataFrame(rows)


def best_invariant_model_deltas(model_metrics: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sub = model_metrics[~model_metrics["model_name"].astype(str).eq("source_metric_median")].copy()
    for split, group in sub.groupby("split_name", sort=True):
        best = group.sort_values("mae_delta_vs_source_metric_median", kind="mergesort").iloc[0]
        out[str(split)] = {
            "bestModel": str(best["model_name"]),
            "bestMaeDeltaVsSourceMetricMedian": float(best["mae_delta_vs_source_metric_median"]),
            "bestMae": float(best["mae"]),
        }
    return out


def build_law_records(
    *,
    invariants: pd.DataFrame,
    model_metrics: pd.DataFrame,
    invariant_counterexamples: pd.DataFrame,
    classes: pd.DataFrame,
    class_counterexamples: pd.DataFrame,
    s11_accuracy: pd.DataFrame,
    s11_validations: pd.DataFrame,
    s12_validation: pd.DataFrame,
    s13_transfer: pd.DataFrame,
    source_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stable = invariants[invariants["candidate_status"].astype(str).eq("stable_supported_candidate")].copy()
    stable_features = stable["candidate_feature"].astype(str).tolist()
    stable_summary = stable[
        [
            "candidate_feature",
            "effect_full",
            "effect_heldout_policy",
            "effect_heldout_world",
            "source_dominance_rate",
            "active_source_count",
            "stability_score",
        ]
    ].to_dict(orient="records")
    invariant_deltas = best_invariant_model_deltas(model_metrics)
    stable_counterexample_counts = (
        invariant_counterexamples[invariant_counterexamples["candidate_feature"].astype(str).isin(stable_features)]
        .groupby("candidate_feature")
        .size()
        .to_dict()
    )

    class_status_counts = classes["class_status"].astype(str).value_counts().to_dict()
    bounded_count = int((classes["class_status"].astype(str) == "bounded_interpretable_class").sum())
    constraining_count = int(classes["class_status"].astype(str).str.contains("constraint", regex=False).sum())
    primary_bounded = (
        classes[(classes["branch_role"].astype(str) == "primary") & (classes["class_status"].astype(str) == "bounded_interpretable_class")]
        .groupby("entity_type")
        .size()
        .to_dict()
    )
    class_counterexample_counts = class_counterexamples.groupby(["entity_type", "class_status"]).size().reset_index(name="rows").to_dict(orient="records")

    s11_accuracy_rows = s11_accuracy.to_dict(orient="records")
    s11_validation_counts = s11_validations.groupby(["candidate_role", "capability_target", "validation_kind", "validation_status"]).size().reset_index(name="rows")
    s11_fresh_proxy_rows = int((s11_validations["validation_kind"].astype(str) == "fresh_simulator_proxy").sum())
    s11_blocker_rows = int((s11_validations["validation_kind"].astype(str) == "fresh_simulator_blocker").sum())

    heldout = s12_validation[s12_validation["validation_kind"].astype(str) == "heldout_e03_simulation"].copy()
    s12_summary = (
        heldout.groupby(["control_family", "design_role"], dropna=False)
        .agg(
            rows=("design_id", "count"),
            mean_target_match_score=("target_match_score", "mean"),
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
            mean_work_per_item=("work_per_item", "mean"),
            mean_dg_recovery_proxy=("dg_recovery_proxy", "mean"),
        )
        .reset_index()
    )
    s12_blocker_rows = int((s12_validation["validation_kind"].astype(str) == "simulator_blocker").sum())
    inverse_mean = float(s12_summary.loc[s12_summary["control_family"].eq("inverse_design"), "mean_target_match_score"].iloc[0])
    best_control_mean = float(s12_summary.loc[~s12_summary["control_family"].eq("inverse_design"), "mean_target_match_score"].max())

    fresh_transfer = s13_transfer[
        (s13_transfer["evaluation_kind"].astype(str) == "fresh_existing_path_evaluation")
        & (s13_transfer["control_family"].astype(str) == "cross_substrate_transfer")
    ]
    s13_success = int(fresh_transfer["transfer_success"].fillna(False).astype(bool).sum())
    s13_fresh_total = int(len(fresh_transfer))
    s13_blockers = s13_transfer[s13_transfer["evaluation_kind"].astype(str) == "transfer_blocker"]

    source_by_step = {
        step: sorted(source_manifest.loc[source_manifest["source_step_id"].eq(step), "path"].astype(str).tolist())
        for step in ["S09", "S10", "S11", "S12", "S13"]
    }

    records = [
        {
            "law_id": "S14-LAW-001",
            "law_slug": "weak_stable_proxy_modifiers",
            "law_title": "Stable feature proxies are weak modifiers, not standalone laws",
            "claim_status": "provisional_constraining",
            "outcome_classification": "constraining/contradictory",
            "unsupported_speculation": False,
            "s13_scope_limited": False,
            "evidence_steps_json": ["S09"],
            "evidence_summary": (
                f"S09 found {len(stable)} stable supported feature candidates ({', '.join(stable_features)}), "
                "but invariant-only and invariant-plus-proxy held-out models did not beat source-metric medians."
            ),
            "quantitative_support_json": {
                "stableFeatureCount": int(len(stable)),
                "stableFeatures": stable_summary,
                "bestInvariantModelMaeDeltaVsSourceMetricMedian": invariant_deltas,
                "stableCounterexampleCounts": stable_counterexample_counts,
            },
            "scope": "Residualized, source-balanced S09 metric-observation analysis over the S04-derived balanced frame.",
            "counterexamples": "S09 recorded counterexamples for each stable feature, including memory-depth low-residual rows and stochastic/damage high-residual rows.",
            "falsification_tests": (
                "Repeat on new held-out policies/worlds and require invariant-only or invariant-plus-supported-proxy models to improve over "
                "source-metric medians with low counterexample rates."
            ),
            "caveats": "Effects are proxy associations, not causal mechanisms; S08 source-dominance and goal-conflict failures remain active.",
            "recommended_use": "Use stable features as cautious descriptors or stratification variables, not as predictive universal laws.",
            "source_artifacts_json": [{"step": "S09", "paths": source_by_step["S09"]}],
        },
        {
            "law_id": "S14-LAW-002",
            "law_slug": "support_filtering_precondition",
            "law_title": "Support filtering is a precondition for interpretable universality classes",
            "claim_status": "provisional_constraining",
            "outcome_classification": "constraining/contradictory",
            "unsupported_speculation": False,
            "s13_scope_limited": False,
            "evidence_steps_json": ["S10"],
            "evidence_summary": (
                f"S10 produced {len(classes)} class rows, but only {bounded_count} were bounded-interpretable while "
                f"{constraining_count} were source-dominated, missingness-driven, small, or otherwise constraining."
            ),
            "quantitative_support_json": {
                "classStatusCounts": class_status_counts,
                "boundedInterpretableRows": bounded_count,
                "constrainingRows": constraining_count,
                "primaryBoundedByEntityType": primary_bounded,
                "classCounterexampleCounts": class_counterexample_counts,
            },
            "scope": "S10 supported-only and uncertainty-weighted class taxonomies over S06-S09 bounded proxy inputs.",
            "counterexamples": "S10 recorded 53 class counterexample rows, and many classes are source-dominated or missingness-driven.",
            "falsification_tests": "Rerun taxonomy on source-balanced future data and require stable classes with low source dominance and low control alignment.",
            "caveats": "Classes are descriptive bins from computational proxies; they should not be read as natural kinds or biological taxa.",
            "recommended_use": "Use bounded-interpretable S10 classes as report labels only when class_status is checked.",
            "source_artifacts_json": [{"step": "S10", "paths": source_by_step["S10"]}],
        },
        {
            "law_id": "S14-LAW-003",
            "law_slug": "baseline_dominance_guardrail",
            "law_title": "Simple baselines remain the default comparator for predictions and designs",
            "claim_status": "provisional_constraining",
            "outcome_classification": "constraining/contradictory",
            "unsupported_speculation": False,
            "s13_scope_limited": False,
            "evidence_steps_json": ["S09", "S11", "S12"],
            "evidence_summary": (
                "Across S09, S11, and S12, source/metric/class-status baselines often matched or exceeded synthesized predictors or designs; "
                "S12 inverse-design mean target score stayed below the best non-inverse control mean."
            ),
            "quantitative_support_json": {
                "s09BestInvariantDeltas": invariant_deltas,
                "s11AccuracySummary": s11_accuracy_rows,
                "s11FreshProxyRows": s11_fresh_proxy_rows,
                "s11FreshBlockerRows": s11_blocker_rows,
                "s12HeldoutSummary": s12_summary.to_dict(orient="records"),
                "s12InverseMeanTargetScore": inverse_mean,
                "s12BestNonInverseControlMeanTargetScore": best_control_mean,
                "s12SimulatorBlockerRows": s12_blocker_rows,
            },
            "scope": "S09 held-out proxy models, S11 frozen counterfactual validations, and S12 E03-only inverse-design proxy validation.",
            "counterexamples": "S11 positive high-aggregation retrospective rows beat baselines in the accuracy summary, but they were not fresh executable simulations.",
            "falsification_tests": "Freeze future predictions/designs, then require fresh held-out simulator validations to beat source/metric/missingness/class-status baselines.",
            "caveats": "This is a guardrail, not a law of nature; baselines may be weaker on new evidence layers or better-balanced datasets.",
            "recommended_use": "In S15, display model/design claims only with their strongest simple baseline comparator.",
            "source_artifacts_json": [
                {"step": "S09", "paths": source_by_step["S09"]},
                {"step": "S11", "paths": source_by_step["S11"]},
                {"step": "S12", "paths": source_by_step["S12"]},
            ],
        },
        {
            "law_id": "S14-LAW-004",
            "law_slug": "embedded_row_transfer_continuity",
            "law_title": "Embedded-row transfer preserves adjacent-swap row behavior exactly, and only in that narrow path",
            "claim_status": "supported_narrow",
            "outcome_classification": "supportive within a constraining scope",
            "unsupported_speculation": False,
            "s13_scope_limited": True,
            "evidence_steps_json": ["S13"],
            "evidence_summary": (
                f"S13 fresh E03-to-E05 embedded-row transfer succeeded exactly in {s13_success}/{s13_fresh_total} bubble/insertion rows, "
                f"while {len(s13_blockers)} transfer mappings were blocked."
            ),
            "quantitative_support_json": {
                "freshCrossSubstrateSuccessRows": s13_success,
                "freshCrossSubstrateRows": s13_fresh_total,
                "transferBlockerRows": int(len(s13_blockers)),
                "blockerKinds": s13_blockers["mapping_kind"].astype(str).value_counts().to_dict(),
            },
            "scope": "Only the existing E05 embedded-row bubble/insertion transfer path with row-restricted adjacent swaps and explicit blocked mappings.",
            "counterexamples": "S13 blocked all arbitrary S12 DSL-to-E05 mappings plus graph transfer, E05-to-E03 reduction, and E06 governance transfer.",
            "falsification_tests": "Pre-freeze graph, E05 repair/regeneration, or E06 governance adapters and validate transfer on held-out seeds/worlds.",
            "caveats": "Do not generalize S13 beyond the embedded-row path; E05 context rows are reference controls, not fresh transfer simulations.",
            "recommended_use": "Use as narrow continuity evidence for S15; keep broader substrate-transfer cells marked blocked.",
            "source_artifacts_json": [{"step": "S13", "paths": source_by_step["S13"]}],
        },
        {
            "law_id": "S14-LAW-005",
            "law_slug": "damage_repair_context_not_repair_law",
            "law_title": "Damage, repair, and frozen perturbation proxies define stress-test contexts, not a proven repair law",
            "claim_status": "provisional_constraining",
            "outcome_classification": "constraining/contradictory",
            "unsupported_speculation": False,
            "s13_scope_limited": True,
            "evidence_steps_json": ["S09", "S10", "S11", "S12", "S13"],
            "evidence_summary": (
                "S09 found stable world damage/repair and frozen-count proxies, and S10 created bounded world groups around damage/repair, "
                "but S11-S13 left repair/regeneration transfer and validation blocked or only proxy-level."
            ),
            "quantitative_support_json": {
                "s09DamageRepairFeature": stable[stable["candidate_feature"].eq("world_has_damage_or_repair")].to_dict(orient="records"),
                "s09FrozenCountFeature": stable[stable["candidate_feature"].eq("world_frozen_count")].to_dict(orient="records"),
                "s10PrimaryBoundedWorldRows": int(primary_bounded.get("world", 0)),
                "s11HighRepairRows": s11_accuracy[s11_accuracy["capability_target"].astype(str).eq("high_repair_robustness")].to_dict(orient="records"),
                "s12RepairRelatedBlockerRows": int(
                    s12_validation["simulator_blocker"].fillna("").astype(str).str.contains("repair|regeneration|damage", case=False, regex=True).sum()
                ),
                "s13RepairOrGovernanceBlockerRows": int(
                    s13_blockers["mapping_kind"].astype(str).str.contains("e06|e05_2d|s12_dsl", case=False, regex=True).sum()
                ),
            },
            "scope": "Computational stress-test descriptors across S09-S13; S13 scope is limited to blocked transfer records for non-row repair/governance cases.",
            "counterexamples": "S09 records damage/repair counterexamples, and S11 high-repair robustness rows did not beat best baselines.",
            "falsification_tests": "Run held-out E05/E06 repair and regeneration simulations with explicit damage mechanisms and compare to source/metric/class baselines.",
            "caveats": "Repair, regeneration, and governance claims remain simulator-blocked in S12/S13 and cannot be promoted to biological or broad substrate-transfer laws.",
            "recommended_use": "Use as a caveated context marker for stress-test rows, not as evidence that repair has been solved.",
            "source_artifacts_json": [
                {"step": "S09", "paths": source_by_step["S09"]},
                {"step": "S10", "paths": source_by_step["S10"]},
                {"step": "S11", "paths": source_by_step["S11"]},
                {"step": "S12", "paths": source_by_step["S12"]},
                {"step": "S13", "paths": source_by_step["S13"]},
            ],
        },
        {
            "law_id": "S14-SPEC-001",
            "law_slug": "unsupported_memory_depth_threshold_for_repair",
            "law_title": "Unsupported speculation: memory-depth threshold for repair",
            "claim_status": "unsupported_speculation",
            "outcome_classification": "unsupported",
            "unsupported_speculation": True,
            "s13_scope_limited": False,
            "evidence_steps_json": ["S09", "S10", "S11", "S12"],
            "evidence_summary": (
                "S09 memory-depth proxy is a stable supported candidate and S10 has memory-proxy policy classes, "
                "but no S09-S13 artifact identifies a threshold or validates repair improvement from memory depth."
            ),
            "quantitative_support_json": {
                "s09MemoryFeature": stable[stable["candidate_feature"].eq("policy_memory_depth_proxy")].to_dict(orient="records"),
                "memoryCounterexampleRows": int(stable_counterexample_counts.get("policy_memory_depth_proxy", 0)),
                "s10BoundedMemoryPolicyClasses": classes[
                    (classes["entity_type"].astype(str).eq("policy"))
                    & (classes["class_status"].astype(str).eq("bounded_interpretable_class"))
                    & (pd.to_numeric(classes["mean_policy_memory_depth_proxy"], errors="coerce").fillna(0.0) >= 1.0)
                ][["class_id", "entity_count", "cautious_label", "mean_policy_memory_depth_proxy"]].to_dict(orient="records"),
                "s12RepairBlockerRows": s12_blocker_rows,
            },
            "scope": "No current law scope; this remains a proposed follow-up hypothesis outside validated S09-S13 repair evidence.",
            "counterexamples": "S09 memory-depth counterexamples exist, and S11/S12 repair validation is absent or blocked.",
            "falsification_tests": "Design repair-world memory-depth ablations, freeze thresholds, and validate on held-out E05/E06 repair tasks against no-memory baselines.",
            "caveats": "Unsupported speculation; do not use in S15 as an empirical law.",
            "recommended_use": "List only as a parked follow-up hypothesis.",
            "source_artifacts_json": [
                {"step": "S09", "paths": source_by_step["S09"]},
                {"step": "S10", "paths": source_by_step["S10"]},
                {"step": "S11", "paths": source_by_step["S11"]},
                {"step": "S12", "paths": source_by_step["S12"]},
            ],
        },
        {
            "law_id": "S14-SPEC-002",
            "law_slug": "unsupported_aggregation_target_acquisition_tempo",
            "law_title": "Unsupported speculation: aggregation from target-acquisition tempo differences",
            "claim_status": "unsupported_speculation",
            "outcome_classification": "unsupported",
            "unsupported_speculation": True,
            "s13_scope_limited": False,
            "evidence_steps_json": ["S11", "S12"],
            "evidence_summary": (
                "S11 includes high-aggregation retrospective predictions, but S12 explicitly records aggregation as blocked in the E03-only inverse-design branch; "
                "no S09-S13 artifact validates target-acquisition tempo as an aggregation mechanism."
            ),
            "quantitative_support_json": {
                "s11HighAggregationAccuracy": s11_accuracy[s11_accuracy["capability_target"].astype(str).eq("high_aggregation")].to_dict(orient="records"),
                "s11HighAggregationValidationRows": int((s11_validations["capability_target"].astype(str) == "high_aggregation").sum()),
                "s12AggregationBlockerRows": int(
                    s12_validation["simulator_blocker"].fillna("").astype(str).str.contains("aggregation", case=False, regex=False).sum()
                ),
            },
            "scope": "No current law scope; aggregation-tempo remains a speculative mechanism not validated by S09-S13.",
            "counterexamples": "S11 high-aggregation evidence is retrospective or blocked, and S12 does not instantiate chimeric algotype aggregation.",
            "falsification_tests": "Freeze tempo-difference predictions and validate aggregation trajectories in executable chimeric simulators with label-shuffle and speed-matched controls.",
            "caveats": "Unsupported speculation; do not present as an empirical law or substrate-independent mechanism.",
            "recommended_use": "List only as an unsupported follow-up question.",
            "source_artifacts_json": [
                {"step": "S11", "paths": source_by_step["S11"]},
                {"step": "S12", "paths": source_by_step["S12"]},
            ],
        },
    ]

    context = {
        "stableFeatures": stable_features,
        "invariantDeltas": invariant_deltas,
        "classStatusCounts": class_status_counts,
        "s11AccuracyRows": s11_accuracy_rows,
        "s12Summary": s12_summary.to_dict(orient="records"),
        "s13FreshTransferSuccessRows": s13_success,
        "s13FreshTransferRows": s13_fresh_total,
        "s13BlockerRows": int(len(s13_blockers)),
    }
    return dataframe_from_law_records(records), context


def law_report_text(laws: pd.DataFrame, validation_checks: pd.DataFrame, context: Mapping[str, Any]) -> str:
    checks = validation_summary(validation_checks)
    table = laws[
        [
            "law_id",
            "law_title",
            "claim_status",
            "outcome_classification",
            "evidence_steps_json",
            "scope",
            "recommended_use",
        ]
    ]
    sections: list[str] = []
    for row in laws.to_dict(orient="records"):
        sections.append(
            f"""### {row['law_id']}: {row['law_title']}

- Claim status: {row['claim_status']}
- Evidence steps: {', '.join(json.loads(row['evidence_steps_json']))}
- Evidence summary: {row['evidence_summary']}
- Scope: {row['scope']}
- Counterexamples: {row['counterexamples']}
- Falsification tests: {row['falsification_tests']}
- Caveats: {row['caveats']}
- Recommended use: {row['recommended_use']}
"""
        )
    return f"""# E07 Empirical Laws Report

## Top Summary

- Research step ID: S14
- Completion status: Completed; stopped before S15.
- Artifacts written: `/artifacts/reports/e07_empirical_laws.md`, `/artifacts/tables/e07_law_evidence_matrix.csv`, `/artifacts/results/e07_empirical_laws.parquet`, `/artifacts/research_steps/S14/research_step_full_results.md`.
- Validation result: {checks['passed']}/{checks['total']} checks passed.
- Outcome classification: constraining/contradictory synthesis.
- Caveats or blockers: Candidate laws are bounded computational summaries from S09-S13 only. S13 is not generalized beyond the existing embedded-row bubble/insertion transfer path. Two speculative ideas are explicitly labeled unsupported.
- Lay summary: The evidence supports a cautious atlas narrative: weak feature proxies and bounded classes can organize results, simple baselines remain essential, one narrow embedded-row transfer path works exactly, and broader repair, aggregation, governance, and transfer laws remain unvalidated.
- Recommended next action: Chief Scientist review before S15. Use this report as the law/caveat layer for the periodic table, with unsupported speculation separated from empirical candidates.

## Law Candidate Summary

{markdown_table(table, max_rows=20)}

## Candidate Laws And Unsupported Speculation

{''.join(sections)}
## Cross-Step Evidence Snapshot

```json
{json.dumps(context, indent=2, sort_keys=True, default=str)}
```

## Boundary Statement

These are not causal biological laws. They are compact computational summaries over S09-S13 artifacts. Any S15 presentation should keep claim status, evidence scope, counterexamples, and falsification tests visible next to each law candidate.
"""


def full_results_report(
    *,
    laws: pd.DataFrame,
    validation_checks: pd.DataFrame,
    source_manifest: pd.DataFrame,
    context: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, Path],
) -> str:
    checks = validation_summary(validation_checks)
    command_rows = pd.DataFrame(
        [
            {
                "command": command.get("command"),
                "returnCode": command.get("returnCode"),
                "elapsedSeconds": command.get("elapsedSeconds"),
                "success": command.get("success"),
            }
            for command in commands
        ]
    )
    artifacts_text = "\n".join(f"- `{path}`" for path in artifacts.values())
    status_counts = laws["claim_status"].value_counts().reset_index()
    status_counts.columns = ["claim_status", "law_rows"]
    return f"""# E07 S14 Full Results: Derive Empirical Laws

## Top Summary

- Research step ID: S14
- Completion status: Completed; stopped before S15.
- Artifacts written:
{artifacts_text}
- Validation result: {checks['passed']}/{checks['total']} checks passed; unit tests {'passed' if all(command.get('success') for command in commands) else 'had failures recorded'}.
- Outcome classification: constraining/contradictory.
- Caveats or blockers: S14 synthesizes only S09-S13 evidence. Candidate laws are provisional computational summaries, not biological or causal laws. S13 is limited to the narrow embedded-row transfer path, and unsupported speculation is explicitly labeled.
- Lay summary: The S09-S13 record supports cautious organizing rules, not broad universal laws. Feature proxies and classes help label the evidence, but simple baselines and blockers still dominate many tests. One existing substrate-transfer path works exactly; repair, aggregation, graph, and governance transfer remain open.
- Recommended next action: Chief Scientist review before S15. Use the empirical-laws report as a bounded evidence layer for the final periodic table.

## Frozen Question

Can the corpus support candidate principles such as memory-depth thresholds for repair or aggregation emerging from target-acquisition tempo differences?

## Inputs

{markdown_table(source_manifest, max_rows=30)}

## Methods

S14 loaded only S09-S13 artifacts. It summarized S09 stable invariant candidates, held-out invariant model controls, and counterexamples; S10 class status, exemplars, and counterexamples; S11 frozen counterfactual accuracy and simulator blockers; S12 E03-only inverse-design validation and blockers; and S13 existing-path substrate-transfer successes and blockers.

Each candidate law record requires evidence steps, a quantitative support JSON object, an explicit scope, counterexamples, falsification tests, caveats, and recommended use. Unsupported ideas are retained only as `unsupported_speculation` rows. S13-linked rows must explicitly state the embedded-row/blocker scope and cannot generalize to graph, repair/regeneration, or E06 governance transfer.

No new simulator runs, model training, or adapter work were performed in S14.

## Commands

{markdown_table(command_rows)}

## Results

S14 wrote {len(laws)} law/speculation rows: {', '.join(f"{row.claim_status}={row.law_rows}" for row in status_counts.itertuples())}.

{markdown_table(laws[['law_id', 'law_title', 'claim_status', 'outcome_classification', 'evidence_steps_json', 'recommended_use']], max_rows=20)}

## Validation

{markdown_table(validation_checks)}

## Detailed Evidence Context

```json
{json.dumps(context, indent=2, sort_keys=True, default=str)}
```

## Artifacts

- Empirical-laws report: `{artifacts['empirical_report']}`
- Evidence matrix CSV: `{artifacts['evidence_csv']}`
- Evidence matrix Parquet: `{artifacts['evidence_parquet']}`
- Validation checks: `{artifacts['validation_checks']}`
- Source manifest: `{artifacts['source_manifest']}`
- Artifact manifest: `{artifacts['artifact_manifest']}`

## Caveats And Limitations

All claims are computational proxy claims from already-produced artifacts. S09 stable candidates did not yield held-out predictive models that beat source-metric medians. S10 classes are often source-dominated, missingness-driven, or small. S11/S12 predictions and designs were constrained by retrospective validation, E03-only proxies, and simulator blockers. S13 supports only the existing E05 embedded-row bubble/insertion path and does not validate arbitrary S12 DSL transfer, graph transfer, E05 repair/regeneration transfer, or E06 governance transfer.

## Provenance

- Repository: `{git_output(REPO_ROOT, ['rev-parse', '--show-toplevel'])}`
- Branch: `{git_output(REPO_ROOT, ['branch', '--show-current'])}`
- Commit before S14 run: `{git_output(REPO_ROOT, ['rev-parse', 'HEAD'])}`
- Python: `{sys.version.split()[0]}`
- Platform: `{platform.platform()}`
- NumPy: `{np.__version__}`
- pandas: `{pd.__version__}`
- Random seed: `{RANDOM_SEED}`

## Recommended Next Action

Stop before S15 for Chief Scientist review. If S15 proceeds, keep each periodic-table cell linked to law claim status, evidence scope, counterexamples, and falsification tests.
"""


def main() -> int:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    reports_dir = artifacts_dir / "reports"
    logs_dir = step_dir / "logs"
    for directory in (step_dir, results_dir, tables_dir, reports_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evidence_csv = tables_dir / "e07_law_evidence_matrix.csv"
    evidence_parquet = results_dir / "e07_empirical_laws.parquet"
    empirical_report = reports_dir / "e07_empirical_laws.md"
    validation_checks_path = step_dir / "e07_s14_validation_checks.csv"
    source_manifest_path = step_dir / "s14_source_manifest.csv"
    config_path = step_dir / "s14_config.json"
    command_log_path = logs_dir / "s14_command_log.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    full_report_path = step_dir / "research_step_full_results.md"

    source_manifest = build_source_manifest(args, artifacts_dir)
    source_manifest.to_csv(source_manifest_path, index=False)

    invariants = pd.read_parquet(args.s09_candidate_invariants)
    model_metrics = pd.read_parquet(args.s09_model_metrics)
    invariant_counterexamples = pd.read_parquet(args.s09_counterexamples)
    classes = pd.read_parquet(args.s10_classes)
    class_counterexamples = pd.read_csv(args.s10_counterexamples)
    s11_accuracy = pd.read_csv(args.s11_accuracy_summary)
    s11_validations = pd.read_parquet(args.s11_validations)
    s12_validation = pd.read_parquet(args.s12_validation)
    s13_transfer = pd.read_parquet(args.s13_transfer)

    laws, context = build_law_records(
        invariants=invariants,
        model_metrics=model_metrics,
        invariant_counterexamples=invariant_counterexamples,
        classes=classes,
        class_counterexamples=class_counterexamples,
        s11_accuracy=s11_accuracy,
        s11_validations=s11_validations,
        s12_validation=s12_validation,
        s13_transfer=s13_transfer,
        source_manifest=source_manifest,
    )

    validation_checks = validate_empirical_law_artifacts(laws, source_manifest, expected_report_law_count=len(laws))
    step_counts = {
        f"{step}Validation": validation_counts(getattr(args, f"s{int(step[1:]):02d}_validation_checks")) for step in ["S09", "S10", "S11", "S12", "S13"]
    }
    validation_checks = pd.concat(
        [
            validation_checks,
            pd.DataFrame(
                [
                    {
                        "validation_case": "only_s09_s13_artifacts_used",
                        "success": set(source_manifest["source_step_id"].astype(str)) <= {"S09", "S10", "S11", "S12", "S13"},
                        "detail": f"steps={sorted(set(source_manifest['source_step_id'].astype(str)))}",
                    },
                    {
                        "validation_case": "upstream_validation_context_recorded",
                        "success": all(value["total"] > 0 for value in step_counts.values()),
                        "detail": json.dumps(step_counts, sort_keys=True),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    commands: list[dict[str, Any]] = []
    if args.run_unit_tests:
        commands.append(run_command([sys.executable, "-m", "unittest", "tests.e07.test_empirical_law_schema"], args.repo_dir))
    commands.append(run_command([sys.executable, "-m", "py_compile", str(Path(__file__).relative_to(args.repo_dir))], args.repo_dir))
    commands_ok = all(command["success"] for command in commands)
    validation_checks = pd.concat(
        [
            validation_checks,
            pd.DataFrame(
                [
                    {
                        "validation_case": "s14_reproducibility_commands_succeeded",
                        "success": commands_ok,
                        "detail": "Unit test and py_compile commands succeeded." if commands_ok else "One or more commands failed; see command log.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    laws.to_csv(evidence_csv, index=False)
    laws.to_parquet(evidence_parquet, index=False)
    validation_checks.to_csv(validation_checks_path, index=False)
    write_json(command_log_path, commands)

    empirical_text = law_report_text(laws, validation_checks, context)
    write_text(empirical_report, empirical_text)

    artifacts = {
        "full_report": full_report_path,
        "empirical_report": empirical_report,
        "evidence_csv": evidence_csv,
        "evidence_parquet": evidence_parquet,
        "validation_checks": validation_checks_path,
        "source_manifest": source_manifest_path,
        "config": config_path,
        "command_log": command_log_path,
        "artifact_manifest": artifact_manifest_path,
    }

    config = {
        "schemaVersion": EMPIRICAL_LAW_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "randomSeed": RANDOM_SEED,
        "evidenceBoundary": "S09-S13 only",
        "lawRows": int(len(laws)),
        "claimStatusCounts": laws["claim_status"].value_counts().to_dict(),
        "validationResult": validation_summary(validation_checks),
        "upstreamValidationContext": step_counts,
    }
    write_json(config_path, config)

    full_text = full_results_report(
        laws=laws,
        validation_checks=validation_checks,
        source_manifest=source_manifest,
        context=context,
        commands=commands,
        artifacts=artifacts,
    )
    write_text(full_report_path, full_text)

    artifact_manifest = {
        "schemaVersion": "eidosoma.e07.s14.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "outcomeClassification": "constraining/contradictory",
        "validationResult": validation_summary(validation_checks),
        "artifacts": [
            self_referential_artifact_entry(full_report_path, artifacts_dir, "S14 full-results report."),
            self_referential_artifact_entry(empirical_report, artifacts_dir, "S14 empirical-laws report."),
            artifact_entry(evidence_csv, artifacts_dir, "S14 law evidence matrix CSV."),
            artifact_entry(evidence_parquet, artifacts_dir, "S14 law evidence matrix Parquet."),
            artifact_entry(validation_checks_path, artifacts_dir, "S14 validation checks."),
            artifact_entry(source_manifest_path, artifacts_dir, "S14 source manifest."),
            artifact_entry(config_path, artifacts_dir, "S14 run configuration."),
            artifact_entry(command_log_path, artifacts_dir, "S14 command log."),
            self_referential_artifact_entry(artifact_manifest_path, artifacts_dir, "S14 artifact manifest."),
        ],
        "repository": {
            "path": str(args.repo_dir),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        },
    }
    write_json(artifact_manifest_path, artifact_manifest)

    print(f"[S14] wrote {len(laws)} law/speculation rows")
    print(f"[S14] validation checks: {validation_summary(validation_checks)}")
    print(f"[S14] report: {full_report_path}")
    return 0 if bool(validation_checks["success"].all()) and commands_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
