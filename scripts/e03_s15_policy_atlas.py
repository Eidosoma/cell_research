#!/usr/bin/env python3
"""Execute E03 S15: produce the policy morphospace atlas.

S15 packages the completed S01-S14 evidence into a static browsable atlas,
machine-readable policy catalog, report-bundle inputs, and validation records.
It does not start any downstream E04/E06/E07 work.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morphospace import (  # noqa: E402
    POLICY_ATLAS_VERSION,
    atlas_class_summary,
    build_artifact_link_table,
    build_policy_atlas_catalog,
    extract_html_links,
    render_policy_atlas_html,
    select_atlas_highlights,
    validate_policy_atlas_outputs,
    validate_policy_id_references,
)
from morphospace.competence import canonical_json  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S15"
STEP_NUMBER = 15
TITLE = "Produce an atlas"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
REPORT_BUNDLE_DIR = ARTIFACTS_DIR / "report_bundle_inputs"
FIGURES_DIR = STEP_DIR / "figures"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_policy_atlas"

CATALOG_PATH = RESULTS_DIR / "e03_policy_atlas_catalog.parquet"
CATALOG_CSV_PATH = RESULTS_DIR / "e03_policy_atlas_catalog.csv"
CATALOG_JSONL_PATH = RESULTS_DIR / "e03_policy_atlas_catalog.jsonl"
ATLAS_HTML_PATH = REPORTS_DIR / "e03_policy_atlas.html"
ATLAS_SUMMARY_PATH = REPORTS_DIR / "e03_atlas_summary.md"
REPORT_FIGURE_PATH = REPORTS_DIR / "e03_policy_atlas_embedding.png"

INPUT_PATHS = {
    "policySummary": RESULTS_DIR / "e03_s13_policy_comparison_summary.parquet",
    "corpus": ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet",
    "dslIndex": ARTIFACTS_DIR / "data" / "e03_policy_corpus_dsl_file_index.parquet",
    "classAssignments": RESULTS_DIR / "e03_s11_policy_class_assignments.parquet",
    "classExemplars": RESULTS_DIR / "e03_s11_class_exemplars.parquet",
    "embeddings": RESULTS_DIR / "e03_policy_embeddings.parquet",
    "neighbors": RESULTS_DIR / "e03_s10_embedding_neighbors.parquet",
    "qdElites": RESULTS_DIR / "e03_qd_elites.parquet",
    "phaseBoundaries": RESULTS_DIR / "e03_phase_boundaries.parquet",
    "paretoFronts": RESULTS_DIR / "e03_s13_pareto_fronts.parquet",
    "frontierCandidates": RESULTS_DIR / "e03_frontier_candidates.parquet",
    "frontierValidationSummary": RESULTS_DIR / "e03_s14_frontier_validation_summary.parquet",
    "frontierFileIndex": ARTIFACTS_DIR / "research_steps" / "S14" / "frontier_policy_file_index.parquet",
    "missingComparisonRecords": RESULTS_DIR / "e03_s13_missing_comparison_records.parquet",
    "s13Runs": RESULTS_DIR / "e03_s13_same_seed_runs.parquet",
    "s14Runs": RESULTS_DIR / "e03_s14_frontier_validation_runs.parquet",
    "s12FeatureEffects": RESULTS_DIR / "e03_s12_feature_effect_summary.parquet",
    "s09Boundaries": RESULTS_DIR / "e03_phase_boundaries.parquet",
    "s08Elites": RESULTS_DIR / "e03_qd_elites.parquet",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_ready(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        if prepared[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            prepared[column] = prepared[column].map(
                lambda value: canonical_json(sorted(value) if isinstance(value, set) else value)
                if isinstance(value, (dict, list, tuple, set))
                else value
            )
    return prepared


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path, jsonl_path: Path | None = None) -> list[Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out = table_ready(df)
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    paths = [csv_path, parquet_path]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_json(jsonl_path, orient="records", lines=True)
        paths.append(jsonl_path)
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen or not path.exists() or not path.is_file():
            continue
        seen.add(key)
        rows.append({"path": key, "sizeBytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return rows


def run_command(command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 900) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returnCode": int(completed.returncode),
        "success": completed.returncode == 0,
        "runtimeSeconds": time.perf_counter() - started,
        "output": completed.stdout,
    }


def get_git_metadata() -> dict[str, Any]:
    def read(command: list[str]) -> str:
        result = subprocess.run(command, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return result.stdout.strip()

    return {
        "branch": read(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": read(["git", "rev-parse", "HEAD"]),
        "statusShort": read(["git", "status", "--short"]),
        "remote": read(["git", "remote", "get-url", "origin"]),
    }


def load_status(step_id: str) -> dict[str, Any]:
    path = ARTIFACTS_DIR / "research_steps" / step_id / "status.json"
    if not path.exists():
        return {"success": False, "missing": True, "path": str(path), "status": "missing"}
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in INPUT_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"S15 required inputs are missing: {missing}")
    return {name: pd.read_parquet(path) for name, path in INPUT_PATHS.items()}


def plot_embedding(catalog_df: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = catalog_df.copy()
    x_col = "pca_embeddingDim1"
    y_col = "pca_embeddingDim2"
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df[y_col] = pd.to_numeric(df.get(y_col, np.nan), errors="coerce")
    plot_df = df[df[x_col].notna() & df[y_col].notna()].copy()
    role_order = ["generated", "qd_elite", "phase_boundary", "pathological", "null", "classic", "frontier"]
    colors = {
        "generated": "#7c8a96",
        "qd_elite": "#4f7f52",
        "phase_boundary": "#8566a6",
        "pathological": "#9b4b46",
        "null": "#4c6475",
        "classic": "#2f5f8f",
        "frontier": "#b36b2c",
    }
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    for role in role_order:
        subset = plot_df[plot_df["atlasPrimaryRole"].astype(str).eq(role)]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=38 if role in {"frontier", "classic"} else 18,
            alpha=0.88 if role in {"frontier", "classic"} else 0.55,
            linewidths=0.3,
            edgecolors="#ffffff",
            color=colors.get(role, "#7c8a96"),
            label=role,
        )
    frontier = plot_df[plot_df["isFrontierCandidate"]].sort_values("frontierRank", kind="mergesort").head(16)
    for _, row in frontier.iterrows():
        ax.annotate(
            str(int(row["frontierRank"])),
            (float(row[x_col]), float(row[y_col])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="#5a3416",
        )
    ax.set_title("E03 policy atlas: S10 PCA embedding seed 0")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    step_path = FIGURES_DIR / "s15_policy_embedding_atlas.png"
    fig.savefig(step_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIGURE_PATH)
    plt.close(fig)
    return [step_path, REPORT_FIGURE_PATH]


def role_counts(catalog_df: pd.DataFrame) -> dict[str, int]:
    counts = catalog_df["atlasPrimaryRole"].astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.to_dict().items()}


def atlas_summary_metrics(catalog_df: pd.DataFrame, highlight_df: pd.DataFrame, class_summary_df: pd.DataFrame) -> dict[str, Any]:
    frontier = catalog_df[catalog_df["isFrontierCandidate"]].copy()
    return {
        "catalogRows": int(len(catalog_df)),
        "catalogColumns": int(len(catalog_df.columns)),
        "uniquePolicyIds": int(catalog_df["policyId"].nunique()),
        "frontierCandidateCount": int(catalog_df["isFrontierCandidate"].sum()),
        "classicPolicyCount": int(catalog_df["isClassicPolicy"].sum()),
        "nullPolicyCount": int(catalog_df["isNullPolicy"].sum()),
        "qdEliteCount": int(catalog_df["isS08Elite"].sum()),
        "pathologicalPolicyCount": int(catalog_df["isPathologicalPolicy"].sum()),
        "phaseBoundaryPolicyCount": int(catalog_df["isPhaseBoundaryPolicy"].sum()),
        "roleCounts": role_counts(catalog_df),
        "highlightRows": int(len(highlight_df)),
        "highlightSections": sorted(highlight_df["highlightSection"].astype(str).unique().tolist()),
        "classRows": int(len(class_summary_df)),
        "meanS13CompositeScore": float(pd.to_numeric(catalog_df["s13CompositeScore"], errors="coerce").mean()),
        "frontierHoldoutCompositeMean": float(pd.to_numeric(frontier["holdoutCompositeScore"], errors="coerce").mean()) if not frontier.empty else None,
        "topFrontierPolicyIds": frontier.sort_values("frontierRank", kind="mergesort")["policyId"].astype(str).head(5).tolist(),
    }


def render_atlas_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    metrics: dict[str, Any],
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    return "\n".join(
        [
            "# E03 Policy Atlas Summary",
            "",
            "- Research step ID: S15",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{ATLAS_HTML_PATH}`, `{CATALOG_PATH}`, and `{REPORT_BUNDLE_DIR}/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            "- Lay summary: S15 turns the completed policy morphospace evidence into a browsable atlas and reusable catalog so downstream experiments can select policies with traceable context.",
            "",
            "## Key Counts",
            "",
            f"- Catalog policies: {metrics['catalogRows']} rows and {metrics['catalogColumns']} columns",
            f"- Frontier candidates: {metrics['frontierCandidateCount']}",
            f"- Classic policies: {metrics['classicPolicyCount']}",
            f"- Null/context controls: {metrics['nullPolicyCount']}",
            f"- QD elites: {metrics['qdEliteCount']}",
            f"- Pathological/failure-mode examples: {metrics['pathologicalPolicyCount']}",
            f"- Universality-class summary rows: {metrics['classRows']}",
            f"- Highlight sections: `{canonical_json(metrics['highlightSections'])}`",
            f"- Mean S13 composite score: {metrics['meanS13CompositeScore']:.4f}",
            f"- Mean S14 holdout composite among frontier candidates: {metrics['frontierHoldoutCompositeMean']:.4f}",
            f"- First five frontier policy IDs: `{canonical_json(metrics['topFrontierPolicyIds'])}`",
            "",
            "## Interpretation Boundary",
            "",
            "Atlas labels, competence vectors, universality classes, phase boundaries, and frontier status are empirical computational summaries over bounded local-rule simulations. They are not biological validation, causal proof, or optimality guarantees.",
        ]
    ) + "\n"


def render_validation_report(
    *,
    validation_df: pd.DataFrame,
    artifacts_written: list[Path],
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    result = f"{int(validation_df['success'].sum())}/{len(validation_df)} checks passed"
    rows = [
        "# S15 Validation Report",
        "",
        "- Research step ID: S15",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {result}",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "- Lay summary: S15 validation checks that upstream steps succeeded, policy IDs and links resolve, required atlas roles are present, report-bundle inputs exist, and repository tests pass.",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for _, row in validation_df.iterrows():
        rows.append(f"| `{row['checkId']}` | {'pass' if row['success'] else 'fail'} | {str(row['detail']).replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def write_report_bundle_inputs(
    *,
    catalog_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    artifact_link_df: pd.DataFrame,
    html_link_df: pd.DataFrame,
    policy_reference_df: pd.DataFrame,
    validation_df: pd.DataFrame | None,
    metrics: dict[str, Any],
    caveats: list[str],
    recommended_next_action: str,
    validation_result: str,
    artifacts_written: list[Path],
) -> list[Path]:
    REPORT_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(
        write_table(
            catalog_df,
            REPORT_BUNDLE_DIR / "e03_policy_atlas_catalog.csv",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_catalog.parquet",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_catalog.jsonl",
        )
    )
    paths.extend(
        write_table(
            highlight_df,
            REPORT_BUNDLE_DIR / "e03_policy_atlas_highlights.csv",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_highlights.parquet",
        )
    )
    paths.extend(
        write_table(
            class_summary_df,
            REPORT_BUNDLE_DIR / "e03_policy_atlas_class_summary.csv",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_class_summary.parquet",
        )
    )
    paths.extend(
        write_table(
            artifact_link_df,
            REPORT_BUNDLE_DIR / "e03_policy_atlas_artifact_links.csv",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_artifact_links.parquet",
        )
    )
    paths.extend(
        write_table(
            html_link_df,
            REPORT_BUNDLE_DIR / "e03_policy_atlas_html_links.csv",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_html_links.parquet",
        )
    )
    paths.extend(
        write_table(
            policy_reference_df,
            REPORT_BUNDLE_DIR / "e03_policy_atlas_policy_references.csv",
            REPORT_BUNDLE_DIR / "e03_policy_atlas_policy_references.parquet",
        )
    )
    if validation_df is not None:
        paths.extend(
            write_table(
                validation_df,
                REPORT_BUNDLE_DIR / "e03_policy_atlas_validation.csv",
                REPORT_BUNDLE_DIR / "e03_policy_atlas_validation.parquet",
            )
        )
    narrative_path = REPORT_BUNDLE_DIR / "e03_atlas_report_narrative.md"
    narrative_path.write_text(
        "\n".join(
            [
                "# E03 Atlas Report Narrative Input",
                "",
                "- Research step ID: S15",
                f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
                f"- Artifacts written: {len(artifacts_written)} files plus report-bundle tables in `{REPORT_BUNDLE_DIR}`",
                f"- Validation result: {validation_result}",
                f"- Caveats or blockers: {'; '.join(caveats)}",
                f"- Recommended next action: {recommended_next_action}",
                "- Lay summary: The atlas gives downstream users a single policy catalog with DSL/source links, competence summaries, class placements, frontier status, and trace links.",
                "",
                "## Reusable Evidence",
                "",
                f"- Policy catalog rows: {metrics['catalogRows']}",
                f"- Highlight sections: `{canonical_json(metrics['highlightSections'])}`",
                f"- Required policy groups represented: classic, null/context, QD elite, S14 frontier, and pathological examples.",
                "- Use the machine-readable catalog for exact policy IDs and the HTML atlas for human review.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(narrative_path)
    caveat_path = REPORT_BUNDLE_DIR / "e03_claim_boundaries_and_caveats.md"
    caveat_path.write_text(
        "\n".join(
            [
                "# E03 Atlas Claim Boundaries",
                "",
                "- Research step ID: S15",
                f"- Completion status: {STATUS}",
                f"- Artifacts written: {len(artifacts_written)} files plus `{REPORT_BUNDLE_DIR}`",
                f"- Validation result: {validation_result}",
                f"- Caveats or blockers: {'; '.join(caveats)}",
                f"- Recommended next action: {recommended_next_action}",
                "- Lay summary: The atlas is a computational guide to local-rule policies, not a biological or mathematical proof.",
                "",
                "## Boundaries",
                "",
                *[f"- {caveat}" for caveat in caveats],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(caveat_path)
    column_path = REPORT_BUNDLE_DIR / "e03_policy_atlas_catalog_columns.json"
    write_json(
        column_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": validation_df is None or bool(validation_df["success"].all()),
            "status": STATUS,
            "artifactsWritten": [str(path) for path in paths],
            "validationResult": validation_result,
            "caveatsOrBlockers": caveats,
            "recommendedNextAction": recommended_next_action,
            "policyAtlasVersion": POLICY_ATLAS_VERSION,
            "columns": list(catalog_df.columns),
            "roleTagSemantics": {
                "frontier": "S14 curated frontier candidate",
                "classic": "Bubble, Insertion, or Selection classic-family policy",
                "null": "Null or context-control policy",
                "qd_elite": "S08 MAP-Elites archive member",
                "phase_boundary": "S09 phase-boundary participant",
                "pathological": "S11/S13 failure-mode or pathological label",
            },
        },
    )
    paths.append(column_path)
    artifact_index_path = REPORT_BUNDLE_DIR / "e03_atlas_artifact_index.json"
    write_json(
        artifact_index_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": validation_df is None or bool(validation_df["success"].all()),
            "status": STATUS,
            "artifactsWritten": [str(path) for path in paths],
            "validationResult": validation_result,
            "caveatsOrBlockers": caveats,
            "recommendedNextAction": recommended_next_action,
            "primaryArtifacts": {
                "atlasHtml": str(ATLAS_HTML_PATH),
                "atlasSummary": str(ATLAS_SUMMARY_PATH),
                "policyCatalog": str(CATALOG_PATH),
                "policyCatalogCsv": str(CATALOG_CSV_PATH),
                "policyCatalogJsonl": str(CATALOG_JSONL_PATH),
            },
            "upstreamInputs": {name: str(path) for name, path in INPUT_PATHS.items()},
        },
    )
    paths.append(artifact_index_path)
    manifest_path = REPORT_BUNDLE_DIR / "e03_s15_report_bundle_manifest.json"
    write_json(
        manifest_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "success": validation_df is not None and bool(validation_df["success"].all()),
            "status": STATUS if validation_df is not None and bool(validation_df["success"].all()) else "pending_validation",
            "artifactsWritten": [str(path) for path in paths] + [str(manifest_path)],
            "validationResult": validation_result,
            "caveatsOrBlockers": caveats,
            "recommendedNextAction": recommended_next_action,
            "policyAtlasVersion": POLICY_ATLAS_VERSION,
            "metrics": metrics,
            "manifestSelfReference": {"path": str(manifest_path), "sha256": "omitted_self_referential_manifest"},
        },
    )
    paths.append(manifest_path)
    return paths


def global_artifacts() -> list[tuple[str, str, str]]:
    return [
        ("S05 policy corpus", str(INPUT_PATHS["corpus"]), "input_table"),
        ("S05 DSL file index", str(INPUT_PATHS["dslIndex"]), "input_table"),
        ("S07/S13 policy summary", str(INPUT_PATHS["policySummary"]), "input_table"),
        ("S08 QD elites", str(INPUT_PATHS["qdElites"]), "input_table"),
        ("S09 phase boundaries", str(INPUT_PATHS["phaseBoundaries"]), "input_table"),
        ("S10 embeddings", str(INPUT_PATHS["embeddings"]), "input_table"),
        ("S10 nearest neighbors", str(INPUT_PATHS["neighbors"]), "input_table"),
        ("S11 class assignments", str(INPUT_PATHS["classAssignments"]), "input_table"),
        ("S11 class exemplars", str(INPUT_PATHS["classExemplars"]), "input_table"),
        ("S12 feature effects", str(INPUT_PATHS["s12FeatureEffects"]), "input_table"),
        ("S13 Pareto fronts", str(INPUT_PATHS["paretoFronts"]), "input_table"),
        ("S13 same-seed runs", str(INPUT_PATHS["s13Runs"]), "input_table"),
        ("S14 frontier candidates", str(INPUT_PATHS["frontierCandidates"]), "input_table"),
        ("S14 frontier validation runs", str(INPUT_PATHS["s14Runs"]), "input_table"),
        ("S14 frontier validation summary", str(INPUT_PATHS["frontierValidationSummary"]), "input_table"),
        ("S15 atlas HTML", str(ATLAS_HTML_PATH), "report"),
        ("S15 atlas summary", str(ATLAS_SUMMARY_PATH), "report"),
        ("S15 policy catalog", str(CATALOG_PATH), "result_table"),
        ("S15 policy catalog CSV", str(CATALOG_CSV_PATH), "result_table"),
    ]


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s15_policy_atlas.py", root / "scripts" / "e03_s15_policy_atlas.py"),
            (REPO_ROOT / "morphospace" / "policy_atlas.py", root / "morphospace" / "policy_atlas.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "morphospace" / "__init__.py", root / "morphospace" / "__init__.py"),
            (REPO_ROOT / "tests" / "test_e03_policy_atlas.py", root / "tests" / "test_e03_policy_atlas.py"),
        ]
        for source, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            paths.append(dest)
    return paths


def additional_validation_rows(
    *,
    catalog_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    artifact_link_df: pd.DataFrame,
    html_link_df: pd.DataFrame,
    policy_summary_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    report_bundle_paths: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append({"checkId": check_id, "success": bool(success), "detail": str(detail)})

    catalog_ids = set(catalog_df["policyId"].astype(str))
    summary_ids = set(policy_summary_df["policyId"].astype(str))
    frontier_ids = set(frontier_df["policyId"].astype(str))
    catalog_frontier_ids = set(catalog_df.loc[catalog_df["isFrontierCandidate"], "policyId"].astype(str))
    add("catalog_covers_s13_policy_universe", catalog_ids == summary_ids, f"catalog={len(catalog_ids)} summary={len(summary_ids)}")
    add("frontier_ids_match_s14", catalog_frontier_ids == frontier_ids, f"catalogFrontier={len(catalog_frontier_ids)} s14Frontier={len(frontier_ids)}")
    add(
        "frontier_entries_have_dsl_links",
        bool(catalog_df.loc[catalog_df["isFrontierCandidate"], "policyArtifactLinkCount"].min() >= 2),
        "frontier rows link to S05/S14 DSL or S14 evidence",
    )
    add(
        "catalog_core_columns_present",
        all(
            column in catalog_df.columns
            for column in [
                "policyId",
                "policyAtlasVersion",
                "roleTagsJson",
                "atlasPrimaryRole",
                "nearestNeighborsJson",
                "representativeTrajectoriesJson",
                "policyArtifactLinksJson",
            ]
        ),
        "core atlas columns checked",
    )
    add(
        "report_bundle_inputs_written",
        bool(report_bundle_paths) and all(path.exists() for path in report_bundle_paths),
        f"bundleFiles={len(report_bundle_paths)} missing={sum(not path.exists() for path in report_bundle_paths)}",
    )
    add(
        "machine_readable_catalog_formats_exist",
        CATALOG_PATH.exists() and CATALOG_CSV_PATH.exists() and CATALOG_JSONL_PATH.exists(),
        f"parquet={CATALOG_PATH.exists()} csv={CATALOG_CSV_PATH.exists()} jsonl={CATALOG_JSONL_PATH.exists()}",
    )
    add(
        "html_static_load_smoke",
        ATLAS_HTML_PATH.exists() and ATLAS_HTML_PATH.stat().st_size > 100_000 and len(html_link_df) > 0,
        f"htmlSize={ATLAS_HTML_PATH.stat().st_size if ATLAS_HTML_PATH.exists() else 0} links={len(html_link_df)}",
    )
    add(
        "artifact_link_table_has_policy_and_global_links",
        {"policy", "global"}.issubset(set(artifact_link_df["linkScope"].astype(str))),
        f"linkScopes={sorted(set(artifact_link_df['linkScope'].astype(str)))}",
    )
    add(
        "highlight_ids_resolve",
        set(highlight_df["policyId"].astype(str)).issubset(catalog_ids),
        f"highlightRows={len(highlight_df)}",
    )
    return pd.DataFrame(rows)


def main() -> None:
    started_at = utc_now()
    for directory in [STEP_DIR, RESULTS_DIR, REPORTS_DIR, REPORT_BUNDLE_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    artifacts_written: list[Path] = []

    inputs = load_inputs()
    config = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "policyAtlasVersion": POLICY_ATLAS_VERSION,
        "inputs": {name: str(path) for name, path in INPUT_PATHS.items()},
        "outputs": {
            "atlasHtml": str(ATLAS_HTML_PATH),
            "atlasSummary": str(ATLAS_SUMMARY_PATH),
            "policyCatalog": str(CATALOG_PATH),
            "reportBundleInputs": str(REPORT_BUNDLE_DIR),
        },
    }
    config_path = STEP_DIR / "policy_atlas_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    catalog_df = build_policy_atlas_catalog(
        policy_summary_df=inputs["policySummary"],
        corpus_df=inputs["corpus"],
        dsl_index_df=inputs["dslIndex"],
        class_assignments_df=inputs["classAssignments"],
        embeddings_df=inputs["embeddings"],
        neighbors_df=inputs["neighbors"],
        qd_elites_df=inputs["qdElites"],
        pareto_df=inputs["paretoFronts"],
        frontier_df=inputs["frontierCandidates"],
        frontier_validation_summary_df=inputs["frontierValidationSummary"],
        frontier_file_index_df=inputs["frontierFileIndex"],
        s13_missing_df=inputs["missingComparisonRecords"],
        s13_runs_df=inputs["s13Runs"],
        s14_runs_df=inputs["s14Runs"],
    )
    class_summary_df = atlas_class_summary(catalog_df)
    highlight_df = select_atlas_highlights(catalog_df, inputs["classExemplars"])

    artifacts_written.extend(
        write_table(
            catalog_df,
            STEP_DIR / "policy_atlas_catalog.csv",
            STEP_DIR / "policy_atlas_catalog.parquet",
            STEP_DIR / "policy_atlas_catalog.jsonl",
        )
    )
    artifacts_written.extend(write_table(catalog_df, CATALOG_CSV_PATH, CATALOG_PATH, CATALOG_JSONL_PATH))
    artifacts_written.extend(write_table(highlight_df, STEP_DIR / "policy_atlas_highlights.csv", STEP_DIR / "policy_atlas_highlights.parquet"))
    artifacts_written.extend(write_table(highlight_df, RESULTS_DIR / "e03_policy_atlas_highlights.csv", RESULTS_DIR / "e03_policy_atlas_highlights.parquet"))
    artifacts_written.extend(
        write_table(class_summary_df, STEP_DIR / "policy_atlas_class_summary.csv", STEP_DIR / "policy_atlas_class_summary.parquet")
    )
    artifacts_written.extend(
        write_table(class_summary_df, RESULTS_DIR / "e03_policy_atlas_class_summary.csv", RESULTS_DIR / "e03_policy_atlas_class_summary.parquet")
    )
    figure_paths = plot_embedding(catalog_df)
    artifacts_written.extend(figure_paths)

    caveats = [
        "Atlas labels and competence values are bounded computational proxy evidence, not causal proof or biological validation.",
        "The static HTML intentionally degrades gracefully; interactive behavior is limited to local filtering and anchor links.",
        "Frontier-candidate holdout evidence is independent-seed and larger-array relative to S13, but still a small CPU-reference validation panel.",
        "Some non-corpus classic/null/context rows do not have S05 DSL files; the catalog records missing-evidence notes rather than fabricating sources.",
        "Near-duplicate behavior is mitigated with structure and neighbor metadata but not exhaustively ruled out.",
    ]
    recommended_next_action = "Stop here for Chief Scientist review; E03 is ready to hand the atlas, catalog, and frontier candidates to E04/E06/E07 planning."
    metrics = atlas_summary_metrics(catalog_df, highlight_df, class_summary_df)
    pending_validation = "pending until S15 validation table is written"
    summary_text = render_atlas_summary(
        validation_result=pending_validation,
        artifacts_written=artifacts_written,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )
    ATLAS_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    (STEP_DIR / "summary.md").write_text(summary_text, encoding="utf-8")
    artifacts_written.extend([ATLAS_SUMMARY_PATH, STEP_DIR / "summary.md"])

    html = render_policy_atlas_html(
        catalog_df=catalog_df,
        highlight_df=highlight_df,
        class_summary_df=class_summary_df,
        atlas_summary_href="e03_atlas_summary.md",
        catalog_csv_href="../results/e03_policy_atlas_catalog.csv",
        catalog_parquet_href="../results/e03_policy_atlas_catalog.parquet",
        embedding_figure_href="e03_policy_atlas_embedding.png",
    )
    ATLAS_HTML_PATH.write_text(html, encoding="utf-8")
    artifacts_written.append(ATLAS_HTML_PATH)

    artifact_link_df = build_artifact_link_table(catalog_df=catalog_df, global_artifacts=global_artifacts())
    html_link_df = extract_html_links(ATLAS_HTML_PATH)
    policy_reference_df = validate_policy_id_references(catalog_df)
    artifacts_written.extend(write_table(artifact_link_df, STEP_DIR / "policy_atlas_artifact_links.csv", STEP_DIR / "policy_atlas_artifact_links.parquet"))
    artifacts_written.extend(
        write_table(artifact_link_df, RESULTS_DIR / "e03_policy_atlas_artifact_links.csv", RESULTS_DIR / "e03_policy_atlas_artifact_links.parquet")
    )
    artifacts_written.extend(write_table(html_link_df, STEP_DIR / "policy_atlas_html_links.csv", STEP_DIR / "policy_atlas_html_links.parquet"))
    artifacts_written.extend(write_table(html_link_df, RESULTS_DIR / "e03_policy_atlas_html_links.csv", RESULTS_DIR / "e03_policy_atlas_html_links.parquet"))
    artifacts_written.extend(
        write_table(policy_reference_df, STEP_DIR / "policy_atlas_policy_references.csv", STEP_DIR / "policy_atlas_policy_references.parquet")
    )
    artifacts_written.extend(
        write_table(policy_reference_df, RESULTS_DIR / "e03_policy_atlas_policy_references.csv", RESULTS_DIR / "e03_policy_atlas_policy_references.parquet")
    )

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_policy_atlas",
            "tests.test_e03_frontier_candidates",
            "tests.test_e03_universality_classes",
            "tests.test_e03_behavior_embeddings",
        ],
        timeout=900,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    preliminary_bundle_paths = write_report_bundle_inputs(
        catalog_df=catalog_df,
        highlight_df=highlight_df,
        class_summary_df=class_summary_df,
        artifact_link_df=artifact_link_df,
        html_link_df=html_link_df,
        policy_reference_df=policy_reference_df,
        validation_df=None,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
        validation_result=pending_validation,
        artifacts_written=artifacts_written,
    )
    artifacts_written.extend(preliminary_bundle_paths)

    upstream_statuses = {f"S{index:02d}": load_status(f"S{index:02d}") for index in range(1, 15)}
    validation_df = validate_policy_atlas_outputs(
        catalog_df=catalog_df,
        highlight_df=highlight_df,
        class_summary_df=class_summary_df,
        artifact_link_df=artifact_link_df,
        html_link_df=html_link_df,
        policy_reference_df=policy_reference_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload={key: value for key, value in repo_test_payload.items() if key != "output"},
        atlas_html_exists=ATLAS_HTML_PATH.exists(),
        atlas_summary_exists=ATLAS_SUMMARY_PATH.exists(),
        report_bundle_manifest_exists=(REPORT_BUNDLE_DIR / "e03_s15_report_bundle_manifest.json").exists(),
    )
    validation_df = pd.concat(
        [
            validation_df,
            additional_validation_rows(
                catalog_df=catalog_df,
                highlight_df=highlight_df,
                artifact_link_df=artifact_link_df,
                html_link_df=html_link_df,
                policy_summary_df=inputs["policySummary"],
                frontier_df=inputs["frontierCandidates"],
                report_bundle_paths=preliminary_bundle_paths,
            ),
        ],
        ignore_index=True,
    )
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "policy_atlas_validation.csv", STEP_DIR / "policy_atlas_validation.parquet"))
    artifacts_written.extend(
        write_table(validation_df, RESULTS_DIR / "e03_policy_atlas_validation.csv", RESULTS_DIR / "e03_policy_atlas_validation.parquet")
    )

    report_bundle_paths = write_report_bundle_inputs(
        catalog_df=catalog_df,
        highlight_df=highlight_df,
        class_summary_df=class_summary_df,
        artifact_link_df=artifact_link_df,
        html_link_df=html_link_df,
        policy_reference_df=policy_reference_df,
        validation_df=validation_df,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
        validation_result=validation_result,
        artifacts_written=artifacts_written,
    )
    artifacts_written.extend([path for path in report_bundle_paths if path not in artifacts_written])

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)
    summary_path = STEP_DIR / "summary.md"
    validation_report_path = STEP_DIR / "validation_report.md"
    status_path = STEP_DIR / "status.json"
    manifest_path = STEP_DIR / "artifact_manifest.json"
    final_artifact_paths = artifacts_written + [status_path, manifest_path]
    summary_text = render_atlas_summary(
        validation_result=validation_result,
        artifacts_written=final_artifact_paths,
        metrics=metrics,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )
    ATLAS_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")
    summary_path.write_text(summary_text, encoding="utf-8")
    validation_report_path.write_text(
        render_validation_report(
            validation_df=validation_df,
            artifacts_written=final_artifact_paths,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    artifacts_written.extend([validation_report_path])

    git_metadata = get_git_metadata()
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": TITLE,
        "success": validation_passed,
        "status": STATUS if validation_passed else "completed_with_validation_failures",
        "outcomeClassification": OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory",
        "artifactsWritten": [str(path) for path in artifacts_written] + [str(status_path), str(manifest_path)],
        "validationResult": validation_result,
        "validationChecksPassed": int(validation_df["success"].sum()),
        "validationChecksTotal": int(len(validation_df)),
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "laySummary": (
            "S15 packaged the E03 local-rule policy morphospace into a browsable atlas, complete machine-readable catalog, "
            "link validation tables, and report-bundle inputs."
        ),
        "atlasSummary": metrics,
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {"policyAtlasVersion": POLICY_ATLAS_VERSION},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
        },
        "git": git_metadata,
        "startedAt": started_at,
        "completedAt": utc_now(),
    }
    write_json(status_path, status_payload)
    artifacts_written.append(status_path)

    manifest_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": validation_passed,
        "status": status_payload["status"],
        "artifactsWritten": [str(path) for path in artifacts_written] + [str(manifest_path)],
        "validationResult": validation_result,
        "caveatsOrBlockers": caveats,
        "recommendedNextAction": recommended_next_action,
        "artifactCount": len(artifacts_written) + 1,
        "checksummedArtifactCount": len(artifact_rows(artifacts_written)),
        "artifacts": artifact_rows(artifacts_written),
        "primaryArtifacts": {
            "atlasHtml": str(ATLAS_HTML_PATH),
            "atlasSummary": str(ATLAS_SUMMARY_PATH),
            "policyCatalog": str(CATALOG_PATH),
            "reportBundleManifest": str(REPORT_BUNDLE_DIR / "e03_s15_report_bundle_manifest.json"),
        },
        "manifestSelfReference": {"path": str(manifest_path), "sha256": "omitted_self_referential_manifest"},
        "git": git_metadata,
        "versions": status_payload["versions"],
    }
    write_json(manifest_path, manifest_payload)
    artifacts_written.append(manifest_path)

    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation_passed,
                "validationResult": validation_result,
                "catalogRows": metrics["catalogRows"],
                "frontierCandidateCount": metrics["frontierCandidateCount"],
                "statusPath": str(status_path),
                "atlasHtmlPath": str(ATLAS_HTML_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
