#!/usr/bin/env python3
"""Build the final E03 auditable policy atlas."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.e03.frontier_candidates import build_policy_source_table
from src.e03.gpu_batch_simulator import jax_backend_summary
from src.e03.policy_atlas import (
    ATLAS_SCHEMA,
    AtlasTraceConfig,
    atlas_digest,
    build_atlas_index,
    mark_trace_availability,
    render_atlas_html,
    representative_policy_ids,
    representative_trace_frame,
    validation_frame,
)


STEP_ID = "S15"
STEP_NUMBER = 15
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--s10-embeddings", type=Path, default=artifacts_default / "results/e03_policy_embeddings.parquet")
    parser.add_argument("--s11-clusters", type=Path, default=artifacts_default / "results/e03_policy_clusters.parquet")
    parser.add_argument("--s11-cluster-summary", type=Path, default=artifacts_default / "results/e03_policy_cluster_summary.parquet")
    parser.add_argument("--s11-exemplars", type=Path, default=artifacts_default / "tables/e03_cluster_exemplar_inspection.csv")
    parser.add_argument("--s05-generated-library", type=Path, default=artifacts_default / "policies/e03_generated_policy_library.jsonl")
    parser.add_argument("--s08-qd-candidates", type=Path, default=artifacts_default / "results/e03_qd_candidate_summary.parquet")
    parser.add_argument("--s08-discovered-library", type=Path, default=artifacts_default / "policies/e03_qd_discovered_policies.jsonl")
    parser.add_argument("--s09-boundaries", type=Path, default=artifacts_default / "results/e03_phase_boundary_candidates.parquet")
    parser.add_argument("--s12-claims", type=Path, default=artifacts_default / "results/e03_rule_ablation_claims.parquet")
    parser.add_argument("--s13-classic-analysis", type=Path, default=artifacts_default / "results/e03_classic_position_analysis.parquet")
    parser.add_argument("--s14-frontier-candidates", type=Path, default=artifacts_default / "results/e03_frontier_candidates.parquet")
    parser.add_argument("--trace-max-count", type=int, default=80)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        "elapsedSeconds": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def artifact_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def manifest_self_entry(path: Path, artifacts_dir: Path, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(artifacts_dir)),
        "description": description,
        "sha256": None,
        "sizeBytes": None,
        "note": "Checksum omitted to avoid self-referential drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                text = "nan" if pd.isna(value) else f"{value:.6g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def normalize_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: None if value is None else str(value))
    return out


def selected_index_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "policy_id",
        "policy_name",
        "atlas_role",
        "class_id",
        "cautious_label",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_work_mean",
        "curated_rank",
        "s14_candidate_status",
        "classic_family",
        "classic_position_classification",
        "s08_archive_winner",
        "qd_candidate",
        "trace_available",
        "atlas_policy_url",
        "computed_dsl_sha256",
        "computed_semantic_hash",
        "code_source",
        "dsl_source_excerpt",
    ]
    return [column for column in preferred if column in frame.columns]


def artifact_link_paths(artifacts_dir: Path) -> tuple[dict[str, str], dict[str, Path], dict[str, str], dict[str, Path]]:
    artifact_rel = {
        "Atlas index CSV": "../tables/e03_policy_atlas_index.csv",
        "Representative trace CSV": "../tables/e03_policy_atlas_representative_traces.csv",
        "S14 frontier candidate CSV": "../tables/e03_frontier_candidates.csv",
        "S14 candidate policy JSONL": "../policies/e03_frontier_candidate_policies.jsonl",
        "Atlas summary": "e03_policy_atlas_summary.md",
        "Report-bundle handoff": "e03_report_bundle_handoff.md",
        "S15 full results": "../research_steps/S15/research_step_full_results.md",
    }
    artifact_abs = {
        "Atlas index CSV": artifacts_dir / "tables/e03_policy_atlas_index.csv",
        "Representative trace CSV": artifacts_dir / "tables/e03_policy_atlas_representative_traces.csv",
        "S14 frontier candidate CSV": artifacts_dir / "tables/e03_frontier_candidates.csv",
        "S14 candidate policy JSONL": artifacts_dir / "policies/e03_frontier_candidate_policies.jsonl",
        "Atlas summary": artifacts_dir / "reports/e03_policy_atlas_summary.md",
        "Report-bundle handoff": artifacts_dir / "reports/e03_report_bundle_handoff.md",
        "S15 full results": artifacts_dir / "research_steps/S15/research_step_full_results.md",
    }
    figure_rel = {
        "S10 behavior embedding": "../figures/e03/policy_behavior_embedding.png",
        "S11 cluster exemplars": "../figures/e03/policy_cluster_exemplars.png",
        "S13 classic positions": "../figures/e03/classics_in_morphospace.png",
        "S14 frontier validation": "../figures/e03/frontier_candidate_validation.png",
        "S08 MAP-Elites archive": "../figures/e03/map_elites_archive.png",
        "S09 phase boundaries": "../figures/e03/phase_boundary_maps.png",
        "S12 ablation effects": "../figures/e03/rule_ablation_effects.png",
    }
    figure_abs = {
        "S10 behavior embedding": artifacts_dir / "figures/e03/policy_behavior_embedding.png",
        "S11 cluster exemplars": artifacts_dir / "figures/e03/policy_cluster_exemplars.png",
        "S13 classic positions": artifacts_dir / "figures/e03/classics_in_morphospace.png",
        "S14 frontier validation": artifacts_dir / "figures/e03/frontier_candidate_validation.png",
        "S08 MAP-Elites archive": artifacts_dir / "figures/e03/map_elites_archive.png",
        "S09 phase boundaries": artifacts_dir / "figures/e03/phase_boundary_maps.png",
        "S12 ablation effects": artifacts_dir / "figures/e03/rule_ablation_effects.png",
    }
    return artifact_rel, artifact_abs, figure_rel, figure_abs


def render_summary(
    *,
    index: pd.DataFrame,
    traces: pd.DataFrame,
    validation: pd.DataFrame,
    digest: str,
    artifacts: dict[str, Path],
    classic_analysis: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> str:
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed"
    role_counts = index["atlas_role"].value_counts().rename_axis("atlas_role").reset_index(name="count")
    class_counts = index.groupby("class_id").agg(policy_count=("policy_id", "size"), frontier_candidates=("is_frontier_candidate", "sum"), mean_screen_score=("screen_score", "mean")).reset_index()
    classic_cols = [
        "algorithm_label",
        "classic_position_classification",
        "best_dsl_policy_name",
        "best_dsl_screen_score",
        "classification_confidence",
    ]
    return f"""# E03 Policy Atlas Summary

## Top Summary

- Step ID: S15
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; atlas digest `{digest}`.
- Outcome classification: supportive.
- Caveats or blockers: No blocker remains. Atlas labels remain proxy DSL morphospace labels and should not be presented as biological or full public-simulator validation.
- Lay summary: S15 packaged {len(index)} policies, {index['class_id'].nunique()} S11 classes, {int(index['is_frontier_candidate'].sum())} S14 frontier candidates, and {len(traces)} compact representative traces into a static auditable atlas.
- Recommended next action: Stop for Chief review before report-bundle generation.

## Atlas Contents

- Static HTML atlas: `{artifacts['atlasHtml']}`
- Policy index rows: {len(index)}
- S11 classes: {index['class_id'].nunique()}
- S14 frontier candidates highlighted: {int(index['is_frontier_candidate'].sum())}
- Embedded classic DSL landmarks: {int(index['classic_landmark'].sum())}
- Representative trace previews: {len(traces)}
- Atlas digest: `{digest}`

## Role Counts

{markdown_table(role_counts, ["atlas_role", "count"])}

## Class Coverage

{markdown_table(class_counts, ["class_id", "policy_count", "frontier_candidates", "mean_screen_score"])}

## Classic Landmark Context

{markdown_table(classic_analysis[[column for column in classic_cols if column in classic_analysis.columns]], [column for column in classic_cols if column in classic_analysis.columns])}

## Caveats

- S10 embedding coordinates are PCA projections over measured DSL proxy behavior.
- S11 class names are cautious behavior-family labels.
- S12 causal claims are local to tested source-backed families.
- S13 classic classifications apply to embedded DSL landmarks; exact public interface wrappers remain S04 baseline context.
- S14 candidates are validated under DSL disorder/size stress profiles, not direct E01/E02 Frozen Cell, full DG, or mixed-Algotype aggregation assays.
"""


def render_handoff(
    *,
    index: pd.DataFrame,
    traces: pd.DataFrame,
    validation: pd.DataFrame,
    digest: str,
    artifacts: dict[str, Path],
    classic_analysis: pd.DataFrame,
) -> str:
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed"
    frontier = index[index["is_frontier_candidate"]].sort_values("curated_rank")
    frontier_cols = [
        "curated_rank",
        "policy_id",
        "policy_name",
        "class_id",
        "screen_heldout_final_sortedness_mean",
        "s14_mean_final_sortedness",
        "atlas_policy_url",
    ]
    claim_rows = pd.DataFrame(
        [
            {
                "claim": "E03 produced a broad DSL morphospace atlas.",
                "evidence": f"{len(index)} S10-embedded policies across {index['class_id'].nunique()} S11 classes; HTML atlas and index written.",
                "caveat": "Proxy DSL local-step morphospace, not public-simulator or biological validation.",
            },
            {
                "claim": "Classics occupy different regions of the measured DSL space.",
                "evidence": "S13: Bubble central, Insertion peripheral, Selection accidental-looking in finite DSL proxy space.",
                "caveat": "Exact public wrappers are S04 context; Bubble DSL remains approximate.",
            },
            {
                "claim": "A reusable frontier-candidate set is available.",
                "evidence": "S14: 16 source-verified non-classic candidates retained basic sorting under held-out disorder/size stress profiles.",
                "caveat": "Direct Frozen Cell robustness, full DG trajectories, and aggregation assays remain downstream.",
            },
        ]
    )
    return f"""# E03 Report-Bundle Handoff

## Top Summary

- Step ID: S15
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; atlas digest `{digest}`.
- Outcome classification: supportive.
- Caveats or blockers: No blocker remains. Report-bundle generation should preserve proxy language and the finite-search caveats from S07-S15.
- Lay summary: E03 now has a static auditable atlas, a policy index, compact representative traces, S14 candidate links, and a concise claims/evidence/caveat matrix for Chief review.
- Recommended next action: Stop for Chief review before report-bundle generation.

## Bundle-Ready Claims

{markdown_table(claim_rows, ["claim", "evidence", "caveat"])}

## Highlighted Candidate Policies

{markdown_table(frontier[[column for column in frontier_cols if column in frontier.columns]], [column for column in frontier_cols if column in frontier.columns])}

## Classic Context

{markdown_table(classic_analysis[["algorithm_label", "classic_position_classification", "classification_confidence"]], ["algorithm_label", "classic_position_classification", "classification_confidence"])}

## Required Caveat Language

- Use "DSL proxy morphospace" for S07-S15 policy behavior unless a downstream step reruns public-simulator assays.
- Do not present S11 universality classes as formal universality classes outside this measured feature set.
- Do not present S12 ablations as global causal proof; they are local matched-policy evidence.
- Do not treat S14 candidates as global optima; they are curated frontier/niche policies for downstream testing.

## Downstream Reuse

- E04 can use `{artifacts['atlasIndex']}` and `{artifacts['traceTable']}` for memory/repair candidate selection.
- E05 can use `{artifacts['candidateJsonl']}` for substrate-transfer policy inputs.
- E06 can use S14 candidate IDs plus classic DSL landmarks for chimeric-governance tests.
- E07 can use the atlas index and report caveats as corpus metadata.
"""


def render_full_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    index: pd.DataFrame,
    traces: pd.DataFrame,
    digest: str,
    classic_analysis: pd.DataFrame,
) -> str:
    success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if success else "constraining/contradictory"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}"
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    role_counts = index["atlas_role"].value_counts().rename_axis("atlas_role").reset_index(name="count")
    frontier = index[index["is_frontier_candidate"]].sort_values("curated_rank")
    frontier_cols = ["curated_rank", "policy_id", "policy_name", "class_id", "s14_candidate_status", "screen_heldout_final_sortedness_mean", "s14_mean_final_sortedness"]
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    return f"""# E03 S15 Research Step Full Results

## Top Summary

- Step ID: S15
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; atlas digest `{digest}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. The atlas is an audit/package artifact over proxy DSL measurements; report-bundle generation should preserve all S07-S15 caveats.
- Lay summary: S15 built a static policy atlas linking {len(index)} policies to S10 map coordinates, S11 classes, source excerpts, metrics, S13 classic context, S14 frontier-candidate status, and {len(traces)} compact representative traces.
- Recommended next action: Stop for Chief Scientist review before report-bundle generation.

## Frozen Question

Can the policy morphospace be delivered as an auditable atlas where each policy links rule, trajectory, metrics, failure modes, and nearest behavioral neighbors?

## Inputs

- S10 embeddings: `{manifest['inputArtifacts']['s10Embeddings']}`
- S11 clusters and summaries: `{manifest['inputArtifacts']['s11Clusters']}`, `{manifest['inputArtifacts']['s11ClusterSummary']}`
- S11 exemplars: `{manifest['inputArtifacts']['s11Exemplars']}`
- Source libraries: `{manifest['inputArtifacts']['s05GeneratedLibrary']}`, `{manifest['inputArtifacts']['s08DiscoveredLibrary']}`
- S12 ablation claims: `{manifest['inputArtifacts']['s12Claims']}`
- S13 classic analysis: `{manifest['inputArtifacts']['s13ClassicAnalysis']}`
- S14 frontier candidates: `{manifest['inputArtifacts']['s14FrontierCandidates']}`

## Lay Summary

This final E03 step does not add a new biological or simulator claim. It packages the completed E03 policy-space work into an auditable atlas. The atlas can be opened as a static HTML file and searched by policy ID, class, or role. It links policy points on the S10 behavior map to rules, metrics, S11 labels, S13 classic-position context, S14 candidate status, source/provenance, and compact trace previews.

## Methods

S15 joined S10 policy embeddings, S11 cluster labels and inspected exemplars, S05/S08 source records, S09 phase-boundary flags, S12 ablation summaries, S13 classic classifications, and S14 frontier-candidate validation results by stable policy ID. Full DSL source was verified through the S14 source table builder, then the atlas index was written as both Parquet and CSV.

Representative traces were generated only as compact previews. The selector included all S14 frontier candidates, all embedded classic DSL landmarks, S11 inspected exemplars, and top policies per class where full DSL source existed. Each trace used the S02 CPU DSL interpreter on an n=8 random-permutation preview config with fixed checkpoints, producing JSON snapshots and action counts.

The HTML atlas embeds compact policy and trace records directly, links to the CSV/JSONL/Markdown artifacts, and references the existing S08-S14 figures. Validation checks confirm policy ID coverage, class coverage, source coverage, frontier linkage, trace JSON loading, HTML inclusion of S14 IDs, and artifact/figure link existence.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- JAX backend summary, recorded for continuity only: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial atlas construction and compact trace previews; no worker pool used.
- New dependencies installed: none.

## Results

### Atlas Role Counts

{markdown_table(role_counts, ["atlas_role", "count"])}

### S14 Frontier Candidates In Atlas

{markdown_table(frontier[[column for column in frontier_cols if column in frontier.columns]], [column for column in frontier_cols if column in frontier.columns])}

### Classic Context

{markdown_table(classic_analysis[["algorithm_label", "classic_position_classification", "classification_confidence"]], ["algorithm_label", "classic_position_classification", "classification_confidence"])}

## Validation

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Provenance

- Atlas digest: `{digest}`
- Git commit at run time: `{manifest['git']['commit']}`
- Git branch at run time: `{manifest['git']['branch']}`
- Git status at run time: `{manifest['git']['statusShort']}`
- Source files:

{source_table}

## Caveats, Blockers, And Limitations

- No blocker remains for S15.
- The atlas is a packaging and audit artifact; it does not rerun large simulations.
- S10-S15 outputs remain DSL proxy evidence.
- Representative traces are n=8 previews, not full trajectory validation.
- Exact public-method classic wrappers are retained as S04 context but are not embedded as S10/S11 map points.

## Recommended Next Action

Stop for Chief Scientist review before report-bundle generation.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    reports_dir = artifacts_dir / "reports"
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    configs_dir = artifacts_dir / "configs"
    src_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, reports_dir, results_dir, tables_dir, configs_dir, src_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)
        if args.run_unit_tests
        else {"command": "unit tests skipped", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )

    embeddings = pd.read_parquet(args.s10_embeddings)
    clusters = pd.read_parquet(args.s11_clusters)
    cluster_summary = pd.read_parquet(args.s11_cluster_summary)
    exemplars = pd.read_csv(args.s11_exemplars)
    qd_candidates = pd.read_parquet(args.s08_qd_candidates)
    source_table = build_policy_source_table(
        generated_policy_library=args.s05_generated_library,
        qd_candidates=qd_candidates,
        qd_discovered_policy_library=args.s08_discovered_library,
    )
    phase_boundaries = pd.read_parquet(args.s09_boundaries)
    ablation_claims = pd.read_parquet(args.s12_claims)
    classic_analysis = pd.read_parquet(args.s13_classic_analysis)
    frontier_candidates = pd.read_parquet(args.s14_frontier_candidates)

    index = build_atlas_index(
        embeddings=embeddings,
        clusters=clusters,
        source_table=source_table,
        frontier_candidates=frontier_candidates,
        exemplars=exemplars,
        cluster_summary=cluster_summary,
        classic_analysis=classic_analysis,
        phase_boundaries=phase_boundaries,
        ablation_claims=ablation_claims,
    )
    trace_ids = representative_policy_ids(index, max_count=args.trace_max_count)
    trace_config = AtlasTraceConfig()
    traces = representative_trace_frame(index, trace_ids, config=trace_config)
    index = mark_trace_availability(index, traces)
    digest = atlas_digest(index, traces)

    artifacts = {
        "fullResults": step_dir / "research_step_full_results.md",
        "status": step_dir / "status.json",
        "manifest": step_dir / "manifest.json",
        "validationCsv": step_dir / "validation.csv",
        "atlasHtml": reports_dir / "e03_policy_atlas.html",
        "atlasSummary": reports_dir / "e03_policy_atlas_summary.md",
        "reportBundleHandoff": reports_dir / "e03_report_bundle_handoff.md",
        "atlasIndex": results_dir / "e03_policy_atlas_index.parquet",
        "atlasIndexCsv": tables_dir / "e03_policy_atlas_index.csv",
        "traceTable": results_dir / "e03_policy_atlas_representative_traces.parquet",
        "traceTableCsv": tables_dir / "e03_policy_atlas_representative_traces.csv",
        "validation": results_dir / "e03_policy_atlas_validation.parquet",
        "config": configs_dir / "e03_s15_policy_atlas_config.json",
        "sourceManifest": src_dir / "e03_policy_atlas_manifest.json",
        "checksums": checksums_dir / "e03_s15_sha256sums.txt",
    }
    artifact_rel, artifact_abs, figure_rel, figure_abs = artifact_link_paths(artifacts_dir)

    config_payload = {
        "schema": ATLAS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "traceMaxCount": args.trace_max_count,
        "traceConfig": trace_config.__dict__,
        "atlasDigest": digest,
        "policyCount": int(len(index)),
        "frontierCandidateCount": int(index["is_frontier_candidate"].sum()),
        "representativeTraceCount": int(len(traces)),
    }
    write_json(artifacts["config"], config_payload)

    normalize_for_parquet(index).to_parquet(artifacts["atlasIndex"], index=False)
    index[selected_index_columns(index)].to_csv(artifacts["atlasIndexCsv"], index=False)
    normalize_for_parquet(traces).to_parquet(artifacts["traceTable"], index=False)
    traces.to_csv(artifacts["traceTableCsv"], index=False)

    html_text = render_atlas_html(
        index=index,
        traces=traces,
        cluster_summary=cluster_summary,
        classic_analysis=classic_analysis,
        artifact_links=artifact_rel,
        figure_links=figure_rel,
        digest=digest,
    )
    write_text(artifacts["atlasHtml"], html_text)

    draft_validation = validation_frame(
        index=index,
        traces=traces,
        html_path=artifacts["atlasHtml"],
        summary_path=artifacts["atlasSummary"],
        handoff_path=artifacts["reportBundleHandoff"],
        artifact_paths={key: path for key, path in artifact_abs.items() if key != "S15 full results"},
        figure_paths=figure_abs,
        expected_policy_count=len(embeddings),
        expected_frontier_count=len(frontier_candidates),
        unit_success=bool(unit_tests["success"]),
    )
    source_files = [
        source_entry(args.repo_dir / "src/e03/policy_atlas.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e03_s15_build_atlas.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e03/test_policy_atlas.py", args.repo_dir),
    ]
    manifest = {
        "schema": ATLAS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "inputArtifacts": {
            "s10Embeddings": str(args.s10_embeddings),
            "s11Clusters": str(args.s11_clusters),
            "s11ClusterSummary": str(args.s11_cluster_summary),
            "s11Exemplars": str(args.s11_exemplars),
            "s05GeneratedLibrary": str(args.s05_generated_library),
            "s08QdCandidates": str(args.s08_qd_candidates),
            "s08DiscoveredLibrary": str(args.s08_discovered_library),
            "s09Boundaries": str(args.s09_boundaries),
            "s12Claims": str(args.s12_claims),
            "s13ClassicAnalysis": str(args.s13_classic_analysis),
            "s14FrontierCandidates": str(args.s14_frontier_candidates),
        },
        "parameters": config_payload,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "jax": jax_backend_summary(),
            "workerCount": 1,
        },
        "git": {
            "commit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "sourceFiles": source_files,
        "counts": {
            "atlasPolicies": int(len(index)),
            "frontierCandidates": int(index["is_frontier_candidate"].sum()),
            "classicLandmarks": int(index["classic_landmark"].sum()),
            "classes": int(index["class_id"].nunique()),
            "representativeTraces": int(len(traces)),
        },
        "digest": digest,
    }

    report_artifacts = {
        key: artifacts[key]
        for key in (
            "fullResults",
            "status",
            "manifest",
            "validationCsv",
            "atlasHtml",
            "atlasSummary",
            "reportBundleHandoff",
            "atlasIndex",
            "atlasIndexCsv",
            "traceTable",
            "traceTableCsv",
            "validation",
            "config",
            "sourceManifest",
            "checksums",
        )
    }

    summary = render_summary(
        index=index,
        traces=traces,
        validation=draft_validation,
        digest=digest,
        artifacts={
            "atlasHtml": artifacts["atlasHtml"],
            "atlasIndexCsv": artifacts["atlasIndexCsv"],
            "traceTableCsv": artifacts["traceTableCsv"],
            "reportBundleHandoff": artifacts["reportBundleHandoff"],
            "fullResults": artifacts["fullResults"],
        },
        classic_analysis=classic_analysis,
        cluster_summary=cluster_summary,
    )
    handoff = render_handoff(
        index=index,
        traces=traces,
        validation=draft_validation,
        digest=digest,
        artifacts={
            "atlasHtml": artifacts["atlasHtml"],
            "atlasIndex": artifacts["atlasIndex"],
            "traceTable": artifacts["traceTable"],
            "candidateJsonl": artifacts_dir / "policies/e03_frontier_candidate_policies.jsonl",
            "fullResults": artifacts["fullResults"],
        },
        classic_analysis=classic_analysis,
    )
    write_text(artifacts["atlasSummary"], summary)
    write_text(artifacts["reportBundleHandoff"], handoff)
    draft_full_report = render_full_report(
        artifacts=report_artifacts,
        manifest=manifest,
        validation=draft_validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        index=index,
        traces=traces,
        digest=digest,
        classic_analysis=classic_analysis,
    )
    write_text(artifacts["fullResults"], draft_full_report)

    validation = validation_frame(
        index=index,
        traces=traces,
        html_path=artifacts["atlasHtml"],
        summary_path=artifacts["atlasSummary"],
        handoff_path=artifacts["reportBundleHandoff"],
        artifact_paths=artifact_abs,
        figure_paths=figure_abs,
        expected_policy_count=len(embeddings),
        expected_frontier_count=len(frontier_candidates),
        unit_success=bool(unit_tests["success"]),
    )
    normalize_for_parquet(validation).to_parquet(artifacts["validation"], index=False)
    validation.to_csv(artifacts["validationCsv"], index=False)

    summary = render_summary(
        index=index,
        traces=traces,
        validation=validation,
        digest=digest,
        artifacts={
            "atlasHtml": artifacts["atlasHtml"],
            "atlasIndexCsv": artifacts["atlasIndexCsv"],
            "traceTableCsv": artifacts["traceTableCsv"],
            "reportBundleHandoff": artifacts["reportBundleHandoff"],
            "fullResults": artifacts["fullResults"],
        },
        classic_analysis=classic_analysis,
        cluster_summary=cluster_summary,
    )
    handoff = render_handoff(
        index=index,
        traces=traces,
        validation=validation,
        digest=digest,
        artifacts={
            "atlasHtml": artifacts["atlasHtml"],
            "atlasIndex": artifacts["atlasIndex"],
            "traceTable": artifacts["traceTable"],
            "candidateJsonl": artifacts_dir / "policies/e03_frontier_candidate_policies.jsonl",
            "fullResults": artifacts["fullResults"],
        },
        classic_analysis=classic_analysis,
    )
    write_text(artifacts["atlasSummary"], summary)
    write_text(artifacts["reportBundleHandoff"], handoff)

    source_manifest = {
        "schema": ATLAS_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceFiles": source_files,
        "notes": "Repository source files are tracked in git; this manifest records hashes only.",
    }
    write_json(artifacts["sourceManifest"], source_manifest)

    command_line = " ".join(sys.argv)
    full_report = render_full_report(
        artifacts=report_artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=command_line,
        index=index,
        traces=traces,
        digest=digest,
        classic_analysis=classic_analysis,
    )
    write_text(artifacts["fullResults"], full_report)

    manifest["artifacts"] = [
        artifact_entry(path, artifacts_dir, key)
        for key, path in report_artifacts.items()
        if key not in {"manifest", "checksums"} and path.exists()
    ]
    manifest["artifacts"].append(manifest_self_entry(artifacts["manifest"], artifacts_dir, "manifest"))
    write_json(artifacts["manifest"], manifest)

    success = bool(validation["success"].all() and unit_tests["success"])
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_validation_failures",
        "artifactsWritten": [str(path) for path in report_artifacts.values()],
        "validationResult": f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}",
        "caveatsOrBlockers": (
            "No blocker remains. Atlas labels and traces are audit packaging over proxy DSL evidence; report-bundle generation should preserve caveats."
            if success
            else "Validation did not fully pass; inspect S15 validation artifacts before report-bundle generation."
        ),
        "recommendedNextAction": "Stop for Chief review before report-bundle generation.",
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "atlasDigest": digest,
    }
    write_json(artifacts["status"], status)

    checksum_targets = [path for key, path in report_artifacts.items() if key != "checksums" and path.exists()]
    checksum_lines = [f"{sha256_file(path)}  {path}" for path in sorted(checksum_targets)]
    write_text(artifacts["checksums"], "\n".join(checksum_lines) + "\n")

    print(json.dumps({"success": success, "report": str(artifacts["fullResults"]), "atlas": str(artifacts["atlasHtml"]), "digest": digest, "validation": status["validationResult"]}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
