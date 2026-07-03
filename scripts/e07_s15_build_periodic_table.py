#!/usr/bin/env python3
"""Build the E07 S15 periodic-table atlas and final report bundle."""

from __future__ import annotations

import argparse
import html
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
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.e07.corpus_schema import sha256_file  # noqa: E402
from src.e07.periodic_table_schema import (  # noqa: E402
    PERIODIC_TABLE_SCHEMA_VERSION,
    report_relative_href,
    sanitize_json,
    stable_hash,
    validate_periodic_table_artifacts,
    validation_summary,
)


STEP_ID = "S15"
STEP_NUMBER = 15
EXPERIMENT_ID = "E07"
RANDOM_SEED = 2026070315


STEP_TITLES = {
    "S01": "Formalize each world",
    "S02": "Represent policies abstractly",
    "S03": "Represent goals abstractly",
    "S04": "Build a unified dataset",
    "S05": "Train a behavior predictor",
    "S06": "Learn policy embeddings",
    "S07": "Learn goal embeddings",
    "S08": "Define Platonic distance",
    "S09": "Search for invariants",
    "S10": "Find universality classes",
    "S11": "Run counterfactual prediction tests",
    "S12": "Do inverse design",
    "S13": "Test substrate transfer",
    "S14": "Derive empirical laws",
    "S15": "Publish the periodic table",
}


def parse_args() -> argparse.Namespace:
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--workspace-dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_dir)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def row_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix == ".parquet":
            return int(pq.ParquetFile(path).metadata.num_rows)
        if path.suffix == ".csv":
            return int(sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return len(payload)
            if isinstance(payload, Mapping):
                return 1
        if path.suffix == ".jsonl":
            return int(sum(1 for _ in path.open("r", encoding="utf-8")))
    except Exception:
        return None
    return None


def path_sha(path: Path) -> str:
    return sha256_file(path) if path.exists() and path.is_file() else ""


def git_output(repo_dir: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc!r}"
    return result.stdout.strip()


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


def html_table(frame: pd.DataFrame, columns: Sequence[str] | None = None, *, max_rows: int = 50) -> str:
    if frame.empty:
        return "<p class=\"empty\">No rows.</p>"
    view = frame.loc[:, list(columns)] if columns else frame.copy()
    view = view.head(max_rows)
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in view.columns)
    body_rows = []
    for row in view.to_dict(orient="records"):
        cells = []
        for column in view.columns:
            value = row.get(column)
            text = format_cell(value)
            css_class = ""
            if column in {"claim_status", "class_status", "candidate_status", "validation_status"}:
                css_class = f" class=\"tag {html.escape(text.replace('_', '-'))}\""
            cells.append(f"<td{css_class}>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    more = ""
    if len(frame) > max_rows:
        more = f"<caption>{len(frame) - max_rows} additional rows omitted from this preview.</caption>"
    return f"<table>{more}<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def validation_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"passed": 0, "total": 0, "allPassed": False}
    checks = pd.read_csv(path)
    return {"passed": int(checks["success"].sum()), "total": int(len(checks)), "allPassed": bool(checks["success"].all())}


def source_specs(artifacts_dir: Path) -> list[dict[str, Any]]:
    return [
        {"sourceStepId": "S01", "artifactKey": "world_inventory_csv", "path": artifacts_dir / "tables" / "e07_world_inventory.csv", "role": "World/task inventory."},
        {"sourceStepId": "S01", "artifactKey": "world_inventory_parquet", "path": artifacts_dir / "results" / "e07_world_inventory.parquet", "role": "Machine-readable world/task inventory."},
        {"sourceStepId": "S02", "artifactKey": "policy_representations", "path": artifacts_dir / "tables" / "e07_policy_representations.parquet", "role": "Abstract policy records."},
        {"sourceStepId": "S03", "artifactKey": "goal_representations", "path": artifacts_dir / "tables" / "e07_goal_representations.parquet", "role": "Abstract goal records."},
        {"sourceStepId": "S03", "artifactKey": "goal_conflicts", "path": artifacts_dir / "tables" / "e07_goal_conflicts.csv", "role": "Goal conflict encodings."},
        {"sourceStepId": "S04", "artifactKey": "unified_behavior_corpus", "path": artifacts_dir / "results" / "e07_unified_behavior_corpus.parquet", "role": "Unified metric-observation corpus."},
        {"sourceStepId": "S04", "artifactKey": "corpus_coverage_summary", "path": artifacts_dir / "tables" / "e07_corpus_coverage_summary.csv", "role": "Coverage and missingness summary."},
        {"sourceStepId": "S04", "artifactKey": "corpus_source_manifest", "path": artifacts_dir / "tables" / "e07_corpus_source_manifest.csv", "role": "Upstream source hashes."},
        {"sourceStepId": "S05", "artifactKey": "behavior_predictor_metrics", "path": artifacts_dir / "results" / "e07_behavior_predictor_metrics.parquet", "role": "Surrogate and baseline metrics."},
        {"sourceStepId": "S05", "artifactKey": "behavior_predictor_calibration", "path": artifacts_dir / "results" / "e07_behavior_predictor_calibration.parquet", "role": "Calibration checks."},
        {"sourceStepId": "S05", "artifactKey": "model_manifest", "path": artifacts_dir / "models" / "e07_behavior_predictor" / "model_manifest.json", "role": "Model provenance manifest."},
        {"sourceStepId": "S06", "artifactKey": "policy_embeddings", "path": artifacts_dir / "results" / "e07_policy_embeddings.parquet", "role": "Policy behavior embeddings."},
        {"sourceStepId": "S06", "artifactKey": "policy_embedding_stability", "path": artifacts_dir / "results" / "e07_policy_embedding_stability.parquet", "role": "Policy embedding stability."},
        {"sourceStepId": "S07", "artifactKey": "goal_embeddings", "path": artifacts_dir / "results" / "e07_goal_embeddings.parquet", "role": "Goal embeddings."},
        {"sourceStepId": "S07", "artifactKey": "goal_conflict_distances", "path": artifacts_dir / "results" / "e07_goal_embedding_conflict_distances.parquet", "role": "Goal conflict stress test."},
        {"sourceStepId": "S08", "artifactKey": "distance_benchmarks", "path": artifacts_dir / "results" / "e07_distance_benchmarks.parquet", "role": "Distance baseline comparisons."},
        {"sourceStepId": "S08", "artifactKey": "platonic_neighbors", "path": artifacts_dir / "results" / "e07_platonic_neighbors.parquet", "role": "Nearest-neighbor tables."},
        {"sourceStepId": "S09", "artifactKey": "candidate_invariants", "path": artifacts_dir / "results" / "e07_candidate_invariants.parquet", "role": "Candidate invariant table."},
        {"sourceStepId": "S09", "artifactKey": "invariant_counterexamples", "path": artifacts_dir / "results" / "e07_invariant_counterexamples.parquet", "role": "Invariant counterexamples."},
        {"sourceStepId": "S10", "artifactKey": "universality_classes", "path": artifacts_dir / "results" / "e07_universality_classes.parquet", "role": "Universality class taxonomy."},
        {"sourceStepId": "S10", "artifactKey": "universality_assignments", "path": artifacts_dir / "results" / "e07_universality_assignments.parquet", "role": "Entity-class assignments."},
        {"sourceStepId": "S10", "artifactKey": "universality_exemplars", "path": artifacts_dir / "tables" / "e07_universality_exemplars.csv", "role": "Class exemplars."},
        {"sourceStepId": "S11", "artifactKey": "counterfactual_predictions", "path": artifacts_dir / "results" / "e07_counterfactual_predictions.parquet", "role": "Frozen counterfactual predictions."},
        {"sourceStepId": "S11", "artifactKey": "counterfactual_validations", "path": artifacts_dir / "results" / "e07_counterfactual_validations.parquet", "role": "Counterfactual validations and blockers."},
        {"sourceStepId": "S12", "artifactKey": "inverse_design_policies", "path": artifacts_dir / "policies" / "e07_inverse_designed_policies.jsonl", "role": "Frozen inverse-designed E03 DSL policies."},
        {"sourceStepId": "S12", "artifactKey": "inverse_design_validation", "path": artifacts_dir / "results" / "e07_inverse_design_validation.parquet", "role": "E03-only inverse-design validation."},
        {"sourceStepId": "S13", "artifactKey": "substrate_transfer", "path": artifacts_dir / "results" / "e07_substrate_transfer.parquet", "role": "Existing-path substrate transfer rows and blockers."},
        {"sourceStepId": "S13", "artifactKey": "transfer_mapping_freeze", "path": artifacts_dir / "research_steps" / "S13" / "transfer_mapping_freeze_manifest.json", "role": "Frozen S13 mapping manifest."},
        {"sourceStepId": "S14", "artifactKey": "empirical_laws_parquet", "path": artifacts_dir / "results" / "e07_empirical_laws.parquet", "role": "S14 law/speculation records."},
        {"sourceStepId": "S14", "artifactKey": "law_evidence_matrix", "path": artifacts_dir / "tables" / "e07_law_evidence_matrix.csv", "role": "S14 evidence matrix."},
        {"sourceStepId": "S14", "artifactKey": "empirical_laws_report", "path": artifacts_dir / "reports" / "e07_empirical_laws.md", "role": "Human-readable S14 law synthesis."},
    ]


def build_source_manifest(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in source_specs(artifacts_dir):
        path = Path(spec["path"])
        rows.append(
            {
                "source_step_id": spec["sourceStepId"],
                "artifact_key": spec["artifactKey"],
                "path": str(path),
                "relative_path": str(path.relative_to(artifacts_dir)) if path.exists() and str(path).startswith(str(artifacts_dir)) else str(path),
                "sha256": path_sha(path),
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
                "row_count": row_count(path),
                "role": spec["role"],
                "exists": bool(path.exists()),
            }
        )
    return pd.DataFrame(rows)


def artifact_entry(
    *,
    artifact_key: str,
    path: Path,
    artifacts_dir: Path,
    artifact_kind: str,
    source_step_id: str,
    role: str,
    checksum_omitted_reason: str | None = None,
) -> dict[str, Any]:
    entry = {
        "artifactKey": artifact_key,
        "artifactKind": artifact_kind,
        "sourceStepId": source_step_id,
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)) if str(path).startswith(str(artifacts_dir)) else str(path),
        "role": role,
        "exists": bool(path.exists()),
        "sizeBytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
        "rowCount": row_count(path),
    }
    if checksum_omitted_reason:
        entry["sha256"] = None
        entry["checksumOmittedReason"] = checksum_omitted_reason
    else:
        entry["sha256"] = path_sha(path)
    return entry


def build_step_cards(artifacts_dir: Path) -> list[dict[str, Any]]:
    cards = []
    outcome_by_step = {
        "S01": "supportive metadata inventory",
        "S02": "supportive representation audit",
        "S03": "supportive representation audit",
        "S04": "supportive corpus construction",
        "S05": "supportive proxy modeling with caveats",
        "S06": "constraining/contradictory",
        "S07": "constraining/contradictory",
        "S08": "constraining/contradictory",
        "S09": "constraining/contradictory",
        "S10": "constraining/contradictory",
        "S11": "constraining/contradictory",
        "S12": "constraining/contradictory",
        "S13": "constraining/contradictory",
        "S14": "constraining/contradictory",
        "S15": "supportive packaging; scientific claims remain bounded",
    }
    for idx in range(1, 15):
        step_id = f"S{idx:02d}"
        validation_path = artifacts_dir / "research_steps" / step_id / f"e07_s{idx:02d}_validation_checks.csv"
        report_path = artifacts_dir / "research_steps" / step_id / "research_step_full_results.md"
        counts = validation_counts(validation_path)
        cards.append(
            {
                "step_id": step_id,
                "title": STEP_TITLES[step_id],
                "status": "Completed",
                "outcome": outcome_by_step[step_id],
                "validation": f"{counts['passed']}/{counts['total']}",
                "report_path": str(report_path),
                "report_href": report_relative_href(report_path, artifacts_dir / "reports"),
            }
        )
    cards.append(
        {
            "step_id": "S15",
            "title": STEP_TITLES["S15"],
            "status": "Completed",
            "outcome": outcome_by_step["S15"],
            "validation": "pending in this run",
            "report_path": str(artifacts_dir / "research_steps" / "S15" / "research_step_full_results.md"),
            "report_href": report_relative_href(artifacts_dir / "research_steps" / "S15" / "research_step_full_results.md", artifacts_dir / "reports"),
        }
    )
    return cards


def build_context(artifacts_dir: Path) -> dict[str, Any]:
    laws = pd.read_parquet(artifacts_dir / "results" / "e07_empirical_laws.parquet")
    transfer = pd.read_parquet(artifacts_dir / "results" / "e07_substrate_transfer.parquet")
    classes = pd.read_parquet(artifacts_dir / "results" / "e07_universality_classes.parquet")
    assignments = pd.read_parquet(artifacts_dir / "results" / "e07_universality_assignments.parquet")
    invariants = pd.read_parquet(artifacts_dir / "results" / "e07_candidate_invariants.parquet")
    s11_validations = pd.read_parquet(artifacts_dir / "results" / "e07_counterfactual_validations.parquet")
    s12_validations = pd.read_parquet(artifacts_dir / "results" / "e07_inverse_design_validation.parquet")
    distance_benchmarks = pd.read_parquet(artifacts_dir / "results" / "e07_distance_benchmarks.parquet")
    predictor_metrics = pd.read_parquet(artifacts_dir / "results" / "e07_behavior_predictor_metrics.parquet")

    worlds_count = row_count(artifacts_dir / "tables" / "e07_world_inventory.csv")
    policies_count = row_count(artifacts_dir / "tables" / "e07_policy_representations.parquet")
    goals_count = row_count(artifacts_dir / "tables" / "e07_goal_representations.parquet")
    corpus_count = row_count(artifacts_dir / "results" / "e07_unified_behavior_corpus.parquet")

    fresh = transfer[
        (transfer["evaluation_kind"].astype(str) == "fresh_existing_path_evaluation")
        & (transfer["control_family"].astype(str) == "cross_substrate_transfer")
    ]
    blockers = transfer[transfer["evaluation_kind"].astype(str) == "transfer_blocker"]
    stable = invariants[invariants["candidate_status"].astype(str).eq("stable_supported_candidate")]
    best_predictor = (
        predictor_metrics[predictor_metrics["model_name"].astype(str) != "global_median"]
        .sort_values(["split_name", "mae"], kind="mergesort")
        .groupby("split_name", as_index=False)
        .first()[["split_name", "model_name", "mae", "calibration_ece"]]
    )

    return {
        "laws": laws,
        "transfer": transfer,
        "classes": classes,
        "assignments": assignments,
        "invariants": invariants,
        "s11_validations": s11_validations,
        "s12_validations": s12_validations,
        "distance_benchmarks": distance_benchmarks,
        "best_predictor": best_predictor,
        "counts": {
            "worlds": int(worlds_count or 0),
            "policy_records": int(policies_count or 0),
            "goals": int(goals_count or 0),
            "metric_observations": int(corpus_count or 0),
            "universality_classes": int(len(classes)),
            "class_assignments": int(len(assignments)),
            "law_rows": int(len(laws)),
            "stable_invariant_candidates": int(len(stable)),
        },
        "law_status_counts": {str(k): int(v) for k, v in laws["claim_status"].astype(str).value_counts().sort_index().items()},
        "unsupported_law_ids": laws.loc[laws["unsupported_speculation"].astype(bool), "law_id"].astype(str).tolist(),
        "class_status_counts": {str(k): int(v) for k, v in classes["class_status"].astype(str).value_counts().sort_index().items()},
        "primary_bounded_classes": int(
            (
                (classes["branch_role"].astype(str) == "primary")
                & (classes["class_status"].astype(str) == "bounded_interpretable_class")
            ).sum()
        ),
        "transfer_summary": {
            "freshExistingPathRows": int(len(fresh)),
            "freshCrossSubstrateSuccessRows": int(fresh["transfer_success"].fillna(False).astype(bool).sum()) if not fresh.empty else 0,
            "transferBlockerRows": int(len(blockers)),
            "blockedMappingKinds": {str(k): int(v) for k, v in blockers["mapping_kind"].astype(str).value_counts().sort_index().items()},
            "scopeStatement": "S13 supports exact continuity only for the existing E03/E05 embedded-row bubble/insertion adjacent-swap path; do not generalize to graph, repair/regeneration, arbitrary DSL, E05-to-E03, or E06 governance transfer.",
        },
    }


def status_badge(status: str) -> str:
    return f"<span class=\"badge {html.escape(status.replace('_', '-'))}\">{html.escape(status)}</span>"


def atlas_html(
    *,
    context: Mapping[str, Any],
    source_manifest: pd.DataFrame,
    step_cards: Sequence[Mapping[str, Any]],
    artifacts_dir: Path,
    generated_at: str,
    repo_commit: str,
) -> str:
    laws: pd.DataFrame = context["laws"]
    classes: pd.DataFrame = context["classes"]
    invariants: pd.DataFrame = context["invariants"]
    transfer: pd.DataFrame = context["transfer"]
    distance_benchmarks: pd.DataFrame = context["distance_benchmarks"]
    best_predictor: pd.DataFrame = context["best_predictor"]

    reports_dir = artifacts_dir / "reports"
    links = {
        "empirical_laws": report_relative_href(artifacts_dir / "reports" / "e07_empirical_laws.md", reports_dir),
        "law_matrix": report_relative_href(artifacts_dir / "tables" / "e07_law_evidence_matrix.csv", reports_dir),
        "final_manifest": report_relative_href(artifacts_dir / "results" / "e07_final_corpus_manifest.json", reports_dir),
        "handoff": report_relative_href(artifacts_dir / "reports" / "e07_report_bundle_handoff.md", reports_dir),
        "class_map": report_relative_href(artifacts_dir / "figures" / "e07" / "universality_class_map.png", reports_dir),
        "policy_map": report_relative_href(artifacts_dir / "figures" / "e07" / "policy_embedding_map.png", reports_dir),
        "goal_map": report_relative_href(artifacts_dir / "figures" / "e07" / "goal_embedding_map.png", reports_dir),
        "distance_benchmark": report_relative_href(artifacts_dir / "figures" / "e07" / "platonic_distance_benchmark.png", reports_dir),
        "transfer_matrix": report_relative_href(artifacts_dir / "figures" / "e07" / "substrate_transfer_matrix.png", reports_dir),
        "counterfactual": report_relative_href(artifacts_dir / "figures" / "e07" / "counterfactual_validation.png", reports_dir),
    }

    stat_cards = "".join(
        f"<article><b>{value:,}</b><span>{html.escape(label.replace('_', ' '))}</span></article>"
        for label, value in context["counts"].items()
    )
    step_tiles = "".join(
        f"""
        <article class="step">
          <a href="{html.escape(str(card['report_href']))}">{html.escape(str(card['step_id']))}</a>
          <h3>{html.escape(str(card['title']))}</h3>
          <p>{html.escape(str(card['outcome']))}</p>
          <small>Validation: {html.escape(str(card['validation']))}</small>
        </article>
        """
        for card in step_cards
    )

    law_rows = []
    for row in laws.to_dict(orient="records"):
        evidence = ", ".join(json.loads(str(row["evidence_steps_json"])))
        law_rows.append(
            {
                "law_id": row["law_id"],
                "claim_status": row["claim_status"],
                "unsupported_speculation": str(bool(row["unsupported_speculation"])),
                "law_title": row["law_title"],
                "evidence_steps": evidence,
                "scope": row["scope"],
                "recommended_use": row["recommended_use"],
            }
        )
    law_frame = pd.DataFrame(law_rows)

    unsupported = law_frame[law_frame["claim_status"].astype(str).eq("unsupported_speculation")]
    atlas_classes = classes[
        [
            "entity_type",
            "branch_name",
            "class_id",
            "entity_count",
            "class_status",
            "cautious_label",
            "source_dominance_rate",
            "mean_support_weight",
            "stability_mean_ari",
        ]
    ].sort_values(["entity_type", "branch_name", "class_status", "class_id"], kind="mergesort")
    invariant_view = invariants[
        [
            "candidate_feature",
            "feature_group",
            "effect_type",
            "effect_heldout_policy",
            "effect_heldout_world",
            "source_dominance_rate",
            "candidate_status",
            "stability_score",
        ]
    ].sort_values(["candidate_status", "candidate_feature"], kind="mergesort")
    transfer_view = (
        transfer[
            [
                "mapping_kind",
                "control_family",
                "evaluation_kind",
                "execution_status",
                "transfer_success",
                "exact_trajectory_match",
                "blocker_reason",
                "claim_boundary",
            ]
        ]
        .drop_duplicates()
        .sort_values(["evaluation_kind", "mapping_kind", "control_family"], kind="mergesort")
    )
    source_view = source_manifest[
        ["source_step_id", "artifact_key", "relative_path", "row_count", "sha256", "role"]
    ].copy()
    source_view["sha256"] = source_view["sha256"].astype(str).str.slice(0, 16)

    law_status_html = "".join(
        f"<li>{status_badge(status)} <b>{count}</b> rows</li>" for status, count in context["law_status_counts"].items()
    )
    class_status_html = "".join(
        f"<li>{status_badge(status)} <b>{count}</b> classes</li>" for status, count in context["class_status_counts"].items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>E07 Periodic Table of Computational Morphogenesis Forms</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d7dce2;
      --paper: #ffffff;
      --band: #f5f7f9;
      --accent: #0f766e;
      --warn: #9a3412;
      --bad: #991b1b;
      --ok: #166534;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--paper); }}
    header {{ padding: 36px clamp(20px, 4vw, 64px) 28px; background: linear-gradient(180deg, #eef7f5, #ffffff); border-bottom: 1px solid var(--line); }}
    main {{ padding: 0 clamp(20px, 4vw, 64px) 64px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4rem); line-height: 1; max-width: 1050px; margin: 0 0 14px; letter-spacing: 0; }}
    h2 {{ margin: 48px 0 14px; font-size: 1.45rem; }}
    h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    p {{ max-width: 960px; color: var(--muted); line-height: 1.55; }}
    a {{ color: #0b5cad; text-decoration-thickness: 1px; }}
    .boundary {{ margin-top: 22px; padding: 16px 18px; border-left: 5px solid var(--warn); background: #fff7ed; max-width: 1120px; }}
    .stats, .steps, .figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }}
    .stats article, .step, .figure-tile {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fff; }}
    .stats b {{ display: block; font-size: 1.65rem; }}
    .stats span, small {{ color: var(--muted); }}
    .step a {{ font-weight: 800; }}
    .step p {{ margin: 0 0 10px; font-size: .9rem; }}
    .status-list {{ display: flex; gap: 10px; flex-wrap: wrap; list-style: none; padding: 0; margin: 14px 0; }}
    .status-list li {{ border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; background: #fff; }}
    .badge, .tag {{ display: inline-block; border-radius: 999px; padding: 2px 7px; border: 1px solid var(--line); background: #f8fafc; font-size: .76rem; white-space: nowrap; }}
    .supported-narrow, .bounded-interpretable-class {{ background: #ecfdf5; color: var(--ok); border-color: #bbf7d0; }}
    .provisional-constraining, .source-dominated-constraint, .small-or-uninterpretable-constraint, .missingness-driven-constraint {{ background: #fff7ed; color: var(--warn); border-color: #fed7aa; }}
    .unsupported-speculation {{ background: #fef2f2; color: var(--bad); border-color: #fecaca; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; background: #fff; }}
    caption {{ caption-side: bottom; text-align: left; color: var(--muted); padding: 8px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; vertical-align: top; text-align: left; }}
    th {{ background: var(--band); font-size: .78rem; text-transform: uppercase; letter-spacing: 0; color: #3c4043; }}
    .figure-tile img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: contain; border: 1px solid var(--line); background: #fff; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
    .note {{ color: var(--muted); font-size: .92rem; }}
    footer {{ padding: 24px clamp(20px, 4vw, 64px); border-top: 1px solid var(--line); background: var(--band); color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>E07 Periodic Table of Computational Morphogenesis Forms</h1>
    <p>Browsable atlas generated for S15 at {html.escape(generated_at)} from commit <code>{html.escape(repo_commit)}</code>. It links worlds, policies, goals, metric corpus, embeddings, distances, classes, predictions, inverse-design rows, substrate-transfer rows, and S14 empirical-law records.</p>
    <div class="boundary"><b>Claim boundary.</b> This is a bounded computational atlas, not causal evidence and not biological validation. S14 claim statuses are preserved. Rows labeled <code>unsupported_speculation</code> remain parked follow-up ideas. S13 transfer support is exact only for the existing E03/E05 embedded-row adjacent-swap path; do not generalize it to graph, repair/regeneration, arbitrary DSL, E05-to-E03, or E06 governance transfer.</div>
    <section class="stats">{stat_cards}</section>
  </header>
  <main>
    <section>
      <h2>Report Bundle</h2>
      <p>Primary bundle links: <a href="{links['final_manifest']}">final corpus manifest</a>, <a href="{links['handoff']}">report-bundle handoff</a>, <a href="{links['empirical_laws']}">S14 empirical laws report</a>, and <a href="{links['law_matrix']}">law evidence matrix</a>.</p>
    </section>

    <section>
      <h2>Step Anchors</h2>
      <div class="steps">{step_tiles}</div>
    </section>

    <section>
      <h2>Claim Status Matrix</h2>
      <ul class="status-list">{law_status_html}</ul>
      <div class="table-wrap">{html_table(law_frame, max_rows=20)}</div>
    </section>

    <section>
      <h2>Unsupported Speculation Kept Separate</h2>
      <p>The following rows are preserved as <code>unsupported_speculation</code>, not empirical laws.</p>
      <div class="table-wrap">{html_table(unsupported, max_rows=10)}</div>
    </section>

    <section>
      <h2>Universality Class Cells</h2>
      <ul class="status-list">{class_status_html}</ul>
      <p class="note">Bounded-interpretable classes are report labels. Source-dominated, missingness-driven, or small/uninterpretable classes remain constraining cells rather than tuned-away failures.</p>
      <div class="table-wrap">{html_table(atlas_classes, max_rows=50)}</div>
    </section>

    <section>
      <h2>Feature And Baseline Guardrails</h2>
      <div class="grid-2">
        <div>
          <h3>Stable Invariant Candidates</h3>
          <div class="table-wrap">{html_table(invariant_view, max_rows=20)}</div>
        </div>
        <div>
          <h3>Best Predictor Rows By Split</h3>
          <div class="table-wrap">{html_table(best_predictor, max_rows=12)}</div>
        </div>
      </div>
    </section>

    <section>
      <h2>Distance And Neighbor Controls</h2>
      <p class="note">S08 source dominance and goal-conflict separation failures remain active caveats. Supported-only and uncertainty-weighted distances are atlas coordinates, not substrate-independent proof.</p>
      <div class="table-wrap">{html_table(distance_benchmarks[['entity_type', 'distance_variant', 'baseline_family', 'eligible_entity_count', 'same_source_rate_at_k', 'same_family_rate_at_k', 'low_support_neighbor_rate_at_k']], max_rows=20)}</div>
    </section>

    <section>
      <h2>S13 Transfer Scope</h2>
      <p>{html.escape(context['transfer_summary']['scopeStatement'])}</p>
      <div class="table-wrap">{html_table(transfer_view, max_rows=40)}</div>
    </section>

    <section>
      <h2>Maps And Figures</h2>
      <div class="figures">
        <a class="figure-tile" href="{links['class_map']}"><h3>Universality Classes</h3><img src="{links['class_map']}" alt="Universality class map"></a>
        <a class="figure-tile" href="{links['policy_map']}"><h3>Policy Embeddings</h3><img src="{links['policy_map']}" alt="Policy embedding map"></a>
        <a class="figure-tile" href="{links['goal_map']}"><h3>Goal Embeddings</h3><img src="{links['goal_map']}" alt="Goal embedding map"></a>
        <a class="figure-tile" href="{links['distance_benchmark']}"><h3>Distance Benchmarks</h3><img src="{links['distance_benchmark']}" alt="Platonic distance benchmark"></a>
        <a class="figure-tile" href="{links['transfer_matrix']}"><h3>Substrate Transfer</h3><img src="{links['transfer_matrix']}" alt="Substrate transfer matrix"></a>
        <a class="figure-tile" href="{links['counterfactual']}"><h3>Counterfactual Tests</h3><img src="{links['counterfactual']}" alt="Counterfactual validation"></a>
      </div>
    </section>

    <section>
      <h2>Source Provenance Preview</h2>
      <p class="note">The final manifest contains full paths, hashes, row counts, and roles. SHA-256 values in this preview are truncated.</p>
      <div class="table-wrap">{html_table(source_view, max_rows=50)}</div>
    </section>
  </main>
  <footer>
    S15 completed for Chief review. The atlas is a report index and synthesis surface; it does not relax S06-S14 caveats.
  </footer>
</body>
</html>
"""


def top_summary_block(
    *,
    validation: Mapping[str, Any],
    artifacts: Sequence[Path],
    outcome: str,
    caveats: str,
    lay_summary: str,
    next_action: str,
) -> str:
    artifact_lines = "\n".join(f"  - `{path}`" for path in artifacts)
    return f"""## Top Summary

- Research step ID: S15
- Completion status: Completed; stopped after S15 for Chief Scientist review.
- Artifacts written:
{artifact_lines}
- Validation result: {validation['passed']}/{validation['total']} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {caveats}
- Lay summary: {lay_summary}
- Recommended next action: {next_action}
"""


def summary_report(
    *,
    validation: Mapping[str, Any],
    artifacts: Sequence[Path],
    context: Mapping[str, Any],
    manifest_path: Path,
    atlas_path: Path,
    handoff_path: Path,
) -> str:
    caveats = "This is a bounded computational atlas, not causal evidence and not biological validation. Unsupported S14 speculation remains explicitly separated. S13 transfer support is embedded-row only; do not generalize."
    lay = "S15 packages the E07 evidence into a linked atlas and manifest. The table helps navigate forms and caveats, but the strongest conclusion remains that most claimed universality is constrained by source balance, missingness, sparse support, baseline strength, and simulator blockers."
    top = top_summary_block(
        validation=validation,
        artifacts=artifacts,
        outcome="supportive packaging; scientific outcome remains bounded and constraining",
        caveats=caveats,
        lay_summary=lay,
        next_action="Chief Scientist review and report-bundle generation; no further E07 research step remains queued.",
    )
    counts_frame = pd.DataFrame([context["counts"]]).T.reset_index()
    counts_frame.columns = ["quantity", "value"]
    law_counts = pd.DataFrame(
        [{"claim_status": key, "rows": value} for key, value in context["law_status_counts"].items()]
    )
    return f"""# E07 Periodic Table Summary

{top}

## Atlas Outputs

- Atlas HTML: `{atlas_path}`
- Final corpus manifest: `{manifest_path}`
- Report-bundle handoff: `{handoff_path}`

## Lay Summary

The final E07 atlas is a navigation layer over already-produced computational evidence. It does not add new biological or causal claims. It keeps the S14 claim labels attached to every law/speculation row, explicitly marks unsupported speculation, and keeps S13 transfer scoped to the narrow embedded-row path.

## Corpus Snapshot

{markdown_table(counts_frame)}

## S14 Claim Statuses Preserved

{markdown_table(law_counts)}

## S13 Scope Reminder

{context['transfer_summary']['scopeStatement']}

## Validation

Validation checked that atlas links resolve, the corpus version and provenance manifest are present, S14 claim-status counts and unsupported-speculation IDs are preserved, markdown handoff reports have top summaries, and S13 embedded-row-only scope plus computational/not-causal/not-biological claim language remain visible.
"""


def handoff_report(
    *,
    validation: Mapping[str, Any],
    artifacts: Sequence[Path],
    context: Mapping[str, Any],
    source_manifest: pd.DataFrame,
) -> str:
    caveats = "S15 is a publication bundle, not new evidence. S14 unsupported speculation is not promoted. S13 remains embedded-row-only; do not generalize. S06-S14 support, source, conflict, missingness, baseline, and simulator caveats remain active."
    lay = "The bundle is ready for Chief review as a bounded computational atlas. It is useful for navigation and synthesis, but broad substrate-independent laws remain unsupported or constraining."
    top = top_summary_block(
        validation=validation,
        artifacts=artifacts,
        outcome="supportive packaging with bounded scientific claims",
        caveats=caveats,
        lay_summary=lay,
        next_action="Chief Scientist review; generate any external-facing report only with claim-status labels, caveats, and S13 scope intact.",
    )
    key_artifacts = source_manifest[["source_step_id", "artifact_key", "relative_path", "row_count", "role"]]
    return f"""# E07 Report-Bundle Handoff

{top}

## Chief Review Notes

Use the HTML atlas as the entry point and the final corpus manifest as the provenance checklist. The evidence supports a careful periodic-table presentation of computational forms, not a claim that the forms are causal, biological, or substrate-independent in the broad sense.

## Required Claim Boundaries

- Preserve every S14 `claim_status` value.
- Keep `unsupported_speculation` rows separate from empirical laws.
- State that S13 supports exact transfer only for the existing E03/E05 embedded-row adjacent-swap path; do not generalize to graph, repair/regeneration, arbitrary DSL, E05-to-E03, or E06 governance transfer.
- Retain S08 source-dominance and goal-conflict failures.
- Retain S09-S12 baseline, counterexample, and simulator-blocker constraints.

## Bundle Contents

{markdown_table(key_artifacts, max_rows=80)}

## Recommended External-Framing Sentence

E07 produced a linked computational atlas that organizes policies, goals, worlds, classes, distances, and candidate laws, while showing that most universality claims remain constrained by source imbalance, missing support, baselines, and simulator availability.

## No Further Queued Research Step

The E07 S01-S15 queue is complete. Any next work should be a new Chief-scoped follow-up experiment, such as new repair/regeneration adapters, source-balanced replication, or prospective substrate-transfer validation.
"""


def full_results_report(
    *,
    validation_checks: pd.DataFrame,
    artifacts: Sequence[Path],
    context: Mapping[str, Any],
    source_manifest: pd.DataFrame,
    command_log: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    generated_at: str,
    repo_dir: Path,
) -> str:
    validation = validation_summary(validation_checks)
    caveats = "Atlas publication does not add new simulator evidence. S14 unsupported speculation stays separate. S13 exact transfer is embedded-row only; do not generalize. S06-S14 missingness, source-dominance, conflict-separation, baseline, and simulator-blocker caveats remain active."
    lay = "S15 turned the E07 evidence package into a browsable atlas, final manifest, summary, and handoff. The packaging is complete, but the scientific story stays cautious: the atlas is a map of computational evidence and limitations, not proof of broad biological laws."
    top = top_summary_block(
        validation=validation,
        artifacts=artifacts,
        outcome="supportive for report-bundle readiness; bounded/constraining for broad scientific claims",
        caveats=caveats,
        lay_summary=lay,
        next_action="Chief Scientist review; no S16 or further E07 step is queued.",
    )
    command_rows = pd.DataFrame(
        [
            {
                "command": item.get("command"),
                "returnCode": item.get("returnCode"),
                "elapsedSeconds": item.get("elapsedSeconds"),
                "success": item.get("success"),
            }
            for item in command_log
        ]
    )
    law_counts = pd.DataFrame(
        [{"claim_status": key, "rows": value} for key, value in context["law_status_counts"].items()]
    )
    class_counts = pd.DataFrame(
        [{"class_status": key, "rows": value} for key, value in context["class_status_counts"].items()]
    )
    return f"""# E07 S15 Full Results: Publish The Periodic Table

{top}

## Frozen Question

Can algorithms, goals, worlds, competencies, failure modes, and nearest neighbors be organized into a browsable atlas of substrate-independent behavioral forms?

## Inputs

S15 used only existing E07 outputs from S01-S14 and did not run new simulations, train new models, or build new adapters. Inputs included the world inventory, policy and goal representation tables, unified behavior corpus and coverage reports, S05 surrogate outputs, S06/S07 embeddings, S08 distances, S09 invariants, S10 classes, S11 counterfactual tests, S12 E03-only inverse-design rows, S13 existing-path transfer rows, and S14 law/speculation records.

{markdown_table(source_manifest[['source_step_id', 'artifact_key', 'relative_path', 'row_count', 'sha256', 'role']], max_rows=80)}

## Methods

The S15 builder loaded compact CSV/Parquet/JSON artifacts from `/artifacts`, extracted corpus counts, class-status counts, law-status counts, unsupported-speculation law IDs, and S13 transfer/blocker summaries, then wrote a static HTML atlas. The atlas links reports, manifests, law matrices, figures, step reports, and source artifacts using relative paths from `/artifacts/reports`.

The final manifest records corpus version, repository commit, source-step coverage, row counts, file paths, SHA-256 hashes where stable, and explicit checksum omissions for mutually referential S15 Markdown/manifest files. S15 deliberately keeps unsupported speculation as a separate status and repeats the embedded-row-only S13 transfer scope in the atlas, summary, handoff, and full-results report.

## Commands

{markdown_table(command_rows)}

## Results

S15 wrote the atlas, summary, final corpus manifest, report-bundle handoff, validation table, source manifest, command log, config, artifact manifest, and this full-results report.

Corpus snapshot:

```json
{json.dumps(context['counts'], indent=2, sort_keys=True)}
```

S14 claim statuses:

{markdown_table(law_counts)}

S10 class statuses:

{markdown_table(class_counts)}

S13 transfer scope:

```json
{json.dumps(context['transfer_summary'], indent=2, sort_keys=True, default=str)}
```

## Validation

{markdown_table(validation_checks)}

Validation passed only if required outputs existed, atlas links resolved, the corpus version was recorded, S14 claim-status counts matched the S14 law matrix, unsupported-speculation IDs appeared in both manifest and atlas, S13 embedded-row-only scope remained visible, computational/not-causal/not-biological claim boundaries appeared in reports, Markdown reports had top summaries, source steps S01-S15 were represented, source hashes were present, and unit/compile checks passed.

## Artifacts

{chr(10).join(f'- `{path}`' for path in artifacts)}

## Caveats And Limitations

S15 is a packaging and publication step. It does not repair upstream missingness, add behavior support to sparse policies/goals, resolve S08 source dominance or goal-conflict separation failures, make S09 invariants predictive, make S10 classes causal or biological, make S11/S12 candidates executable beyond E03 proxies, or extend S13 beyond the embedded-row path. The final atlas should be read as a bounded computational evidence map.

## Provenance

- Generated at UTC: `{generated_at}`
- Schema version: `{PERIODIC_TABLE_SCHEMA_VERSION}`
- Corpus version: `{manifest.get('corpusVersion')}`
- Repository: `{git_output(repo_dir, ['rev-parse', '--show-toplevel'])}`
- Branch: `{git_output(repo_dir, ['branch', '--show-current'])}`
- Commit before S15 commit: `{git_output(repo_dir, ['rev-parse', 'HEAD'])}`
- Python: `{sys.version.split()[0]}`
- Platform: `{platform.platform()}`
- NumPy: `{np.__version__}`
- pandas: `{pd.__version__}`
- pyarrow: `{pq.__version__ if hasattr(pq, '__version__') else 'available'}`
- Random seed: `{RANDOM_SEED}`

## Recommended Next Action

Stop for Chief Scientist review. If a user-facing report is generated, preserve S14 claim statuses, unsupported-speculation labels, S13 embedded-row-only transfer scope, and all caveats about computational proxy evidence.
"""


def build_manifest(
    *,
    artifacts_dir: Path,
    source_manifest: pd.DataFrame,
    output_paths: Mapping[str, Path],
    context: Mapping[str, Any],
    generated_at: str,
    repo_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    source_entries = []
    for row in source_manifest.to_dict(orient="records"):
        source_entries.append(
            {
                "artifactKey": f"source:{row['source_step_id']}:{row['artifact_key']}",
                "artifactKind": "source",
                "sourceStepId": row["source_step_id"],
                "path": row["path"],
                "relativePath": row["relative_path"],
                "role": row["role"],
                "sha256": row["sha256"],
                "sizeBytes": row["size_bytes"],
                "rowCount": row["row_count"],
                "exists": row["exists"],
            }
        )

    mutable_note = "Checksum omitted because this S15 report-bundle file is mutually referential with the final manifest or validation table."
    output_entries = [
        artifact_entry(
            artifact_key="periodic_table_html",
            path=output_paths["atlas_html"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="Browsable atlas.",
        ),
        artifact_entry(
            artifact_key="periodic_table_summary",
            path=output_paths["summary"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="Concise atlas summary.",
            checksum_omitted_reason=mutable_note,
        ),
        artifact_entry(
            artifact_key="final_corpus_manifest",
            path=output_paths["final_manifest"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="Final corpus and report-bundle manifest.",
            checksum_omitted_reason="Self checksum omitted because this file contains the artifact list.",
        ),
        artifact_entry(
            artifact_key="report_bundle_handoff",
            path=output_paths["handoff"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="Chief Scientist handoff report.",
            checksum_omitted_reason=mutable_note,
        ),
        artifact_entry(
            artifact_key="full_results",
            path=output_paths["full_results"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="S15 full-results report.",
            checksum_omitted_reason=mutable_note,
        ),
        artifact_entry(
            artifact_key="validation_checks",
            path=output_paths["validation_checks"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="S15 validation checks.",
            checksum_omitted_reason=mutable_note,
        ),
        artifact_entry(
            artifact_key="source_manifest",
            path=output_paths["source_manifest"],
            artifacts_dir=artifacts_dir,
            artifact_kind="output",
            source_step_id=STEP_ID,
            role="S15 source artifact manifest.",
        ),
    ]
    source_steps = sorted({str(row["source_step_id"]) for row in source_manifest.to_dict(orient="records")} | {STEP_ID})
    repo_commit = git_output(repo_dir, ["rev-parse", "HEAD"])
    corpus_version = f"E07-S15-20260703-{repo_commit[:12] if repo_commit else 'unknown'}"
    atlas_id = stable_hash(
        {
            "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
            "corpusVersion": corpus_version,
            "lawStatusCounts": context["law_status_counts"],
            "classStatusCounts": context["class_status_counts"],
            "transferSummary": context["transfer_summary"],
            "sourceHashes": sorted(
                [str(row["sha256"]) for row in source_manifest.to_dict(orient="records") if row.get("sha256")]
            ),
        }
    )[:24]
    return {
        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "generatedAtUtc": generated_at,
        "corpusVersion": corpus_version,
        "atlasId": f"e07atlas:{atlas_id}",
        "manifestPath": str(output_paths["final_manifest"]),
        "repository": {
            "path": str(repo_dir),
            "branch": git_output(repo_dir, ["branch", "--show-current"]),
            "commitBeforeS15Commit": repo_commit,
            "statusShortAtRun": git_output(repo_dir, ["status", "--short"]),
        },
        "sourceStepsRepresented": source_steps,
        "corpusCounts": context["counts"],
        "s14LawRowCount": int(context["counts"]["law_rows"]),
        "s14ClaimStatusCounts": context["law_status_counts"],
        "unsupportedSpeculationLawIds": context["unsupported_law_ids"],
        "s10ClassStatusCounts": context["class_status_counts"],
        "s13TransferScope": context["transfer_summary"],
        "claimBoundary": "Bounded computational atlas; not causal evidence and not biological validation. S14 unsupported speculation remains separate. S13 transfer is embedded-row only; do not generalize.",
        "validationSummary": validation,
        "artifacts": [*source_entries, *output_entries],
    }


def append_command_checks(validation_checks: pd.DataFrame, commands: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in commands:
        rows.append(
            {
                "validation_case": "command_" + str(item.get("command", "")).replace(" ", "_")[:80],
                "success": bool(item.get("success")),
                "detail": f"returnCode={item.get('returnCode')} elapsedSeconds={item.get('elapsedSeconds')}",
            }
        )
    if not rows:
        return validation_checks
    return pd.concat([validation_checks, pd.DataFrame(rows)], ignore_index=True)


def main() -> int:
    args = parse_args()
    np.random.seed(RANDOM_SEED)

    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    logs_dir = step_dir / "logs"
    for directory in (step_dir, reports_dir, results_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "atlas_html": reports_dir / "e07_periodic_table.html",
        "summary": reports_dir / "e07_periodic_table_summary.md",
        "final_manifest": results_dir / "e07_final_corpus_manifest.json",
        "handoff": reports_dir / "e07_report_bundle_handoff.md",
        "full_results": step_dir / "research_step_full_results.md",
        "validation_checks": step_dir / "e07_s15_validation_checks.csv",
        "source_manifest": step_dir / "s15_source_manifest.csv",
        "config": step_dir / "s15_config.json",
        "command_log": logs_dir / "s15_command_log.json",
        "artifact_manifest": step_dir / "artifact_manifest.json",
    }

    generated_at = utc_now()
    context = build_context(artifacts_dir)
    source_manifest = build_source_manifest(artifacts_dir)
    source_manifest.to_csv(output_paths["source_manifest"], index=False)
    step_cards = build_step_cards(artifacts_dir)

    command_log: list[dict[str, Any]] = []
    if args.run_unit_tests:
        command_log.append(run_command([sys.executable, "-m", "unittest", "tests.e07.test_periodic_table_schema"], args.repo_dir))
    command_log.append(
        run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                "scripts/e07_s15_build_periodic_table.py",
                "src/e07/periodic_table_schema.py",
            ],
            args.repo_dir,
        )
    )
    write_json(output_paths["command_log"], command_log)

    atlas = atlas_html(
        context=context,
        source_manifest=source_manifest,
        step_cards=step_cards,
        artifacts_dir=artifacts_dir,
        generated_at=generated_at,
        repo_commit=git_output(args.repo_dir, ["rev-parse", "HEAD"]),
    )
    write_text(output_paths["atlas_html"], atlas)

    empty_validation = {"passed": 0, "total": 0, "allPassed": False}
    s15_artifacts = [
        output_paths["full_results"],
        output_paths["atlas_html"],
        output_paths["summary"],
        output_paths["final_manifest"],
        output_paths["handoff"],
    ]
    write_text(
        output_paths["summary"],
        summary_report(
            validation=empty_validation,
            artifacts=s15_artifacts,
            context=context,
            manifest_path=output_paths["final_manifest"],
            atlas_path=output_paths["atlas_html"],
            handoff_path=output_paths["handoff"],
        ),
    )
    write_text(
        output_paths["handoff"],
        handoff_report(
            validation=empty_validation,
            artifacts=s15_artifacts,
            context=context,
            source_manifest=source_manifest,
        ),
    )
    provisional_manifest = build_manifest(
        artifacts_dir=artifacts_dir,
        source_manifest=source_manifest,
        output_paths=output_paths,
        context=context,
        generated_at=generated_at,
        repo_dir=args.repo_dir,
        validation=empty_validation,
    )
    write_json(output_paths["final_manifest"], provisional_manifest)
    write_text(
        output_paths["full_results"],
        full_results_report(
            validation_checks=pd.DataFrame(columns=["validation_case", "success", "detail"]),
            artifacts=s15_artifacts,
            context=context,
            source_manifest=source_manifest,
            command_log=command_log,
            manifest=provisional_manifest,
            generated_at=generated_at,
            repo_dir=args.repo_dir,
        ),
    )

    validation_checks = validate_periodic_table_artifacts(
        manifest=provisional_manifest,
        html_path=output_paths["atlas_html"],
        summary_path=output_paths["summary"],
        handoff_path=output_paths["handoff"],
        full_results_path=output_paths["full_results"],
        law_matrix=context["laws"],
        source_manifest=source_manifest,
        expected_claim_status_counts=context["law_status_counts"],
        expected_unsupported_ids=context["unsupported_law_ids"],
    )
    validation_checks = append_command_checks(validation_checks, command_log)
    validation = validation_summary(validation_checks)
    validation_checks.to_csv(output_paths["validation_checks"], index=False)

    write_text(
        output_paths["summary"],
        summary_report(
            validation=validation,
            artifacts=s15_artifacts,
            context=context,
            manifest_path=output_paths["final_manifest"],
            atlas_path=output_paths["atlas_html"],
            handoff_path=output_paths["handoff"],
        ),
    )
    write_text(
        output_paths["handoff"],
        handoff_report(
            validation=validation,
            artifacts=s15_artifacts,
            context=context,
            source_manifest=source_manifest,
        ),
    )
    final_manifest = build_manifest(
        artifacts_dir=artifacts_dir,
        source_manifest=source_manifest,
        output_paths=output_paths,
        context=context,
        generated_at=generated_at,
        repo_dir=args.repo_dir,
        validation=validation,
    )
    write_json(output_paths["final_manifest"], final_manifest)
    write_text(
        output_paths["full_results"],
        full_results_report(
            validation_checks=validation_checks,
            artifacts=s15_artifacts,
            context=context,
            source_manifest=source_manifest,
            command_log=command_log,
            manifest=final_manifest,
            generated_at=generated_at,
            repo_dir=args.repo_dir,
        ),
    )

    final_validation = validate_periodic_table_artifacts(
        manifest=final_manifest,
        html_path=output_paths["atlas_html"],
        summary_path=output_paths["summary"],
        handoff_path=output_paths["handoff"],
        full_results_path=output_paths["full_results"],
        law_matrix=context["laws"],
        source_manifest=source_manifest,
        expected_claim_status_counts=context["law_status_counts"],
        expected_unsupported_ids=context["unsupported_law_ids"],
    )
    final_validation = append_command_checks(final_validation, command_log)
    final_validation.to_csv(output_paths["validation_checks"], index=False)
    final_validation_summary = validation_summary(final_validation)

    config = {
        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "experimentId": EXPERIMENT_ID,
        "randomSeed": RANDOM_SEED,
        "generatedAtUtc": generated_at,
        "repoDir": str(args.repo_dir),
        "artifactsDir": str(artifacts_dir),
        "validationSummary": final_validation_summary,
    }
    write_json(output_paths["config"], config)

    artifact_manifest = {
        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
        "researchStepId": STEP_ID,
        "artifacts": [
            artifact_entry(
                artifact_key=key,
                path=path,
                artifacts_dir=artifacts_dir,
                artifact_kind="s15_artifact",
                source_step_id=STEP_ID,
                role="S15 output",
                checksum_omitted_reason=(
                    "Checksum omitted for mutually referential S15 report-bundle text."
                    if key in {"final_manifest", "summary", "handoff", "full_results", "validation_checks", "artifact_manifest"}
                    else None
                ),
            )
            for key, path in output_paths.items()
            if key != "artifact_manifest"
        ],
    }
    artifact_manifest["artifacts"].append(
        artifact_entry(
            artifact_key="artifact_manifest",
            path=output_paths["artifact_manifest"],
            artifacts_dir=artifacts_dir,
            artifact_kind="s15_artifact",
            source_step_id=STEP_ID,
            role="S15 artifact manifest.",
            checksum_omitted_reason="Self checksum omitted because this file contains the artifact list.",
        )
    )
    write_json(output_paths["artifact_manifest"], artifact_manifest)

    print(json.dumps({"validation": final_validation_summary, "artifacts": {k: str(v) for k, v in output_paths.items()}}, indent=2))
    return 0 if final_validation_summary["allPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
