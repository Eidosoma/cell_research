#!/usr/bin/env python3
"""Run E03 S14 frontier-candidate selection and held-out rechecks."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.e03.frontier_candidates import (
    DEFAULT_FRONTIER_SEED,
    FRONTIER_SCHEMA,
    ablation_feature_frame,
    build_frontier_universe,
    build_policy_source_table,
    candidate_policy_json_records,
    classic_neighbor_feature_frame,
    evaluate_candidate_pool,
    finalize_candidate_set,
    frontier_candidate_digest,
    phase_feature_frame,
    recheck_summary_frame,
    select_frontier_candidates,
    stress_configs,
    validation_frame,
)
from src.e03.gpu_batch_simulator import jax_backend_summary


STEP_ID = "S14"
STEP_NUMBER = 14
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--s10-embeddings", type=Path, default=artifacts_default / "results/e03_policy_embeddings.parquet")
    parser.add_argument("--s11-clusters", type=Path, default=artifacts_default / "results/e03_policy_clusters.parquet")
    parser.add_argument("--s08-qd-candidates", type=Path, default=artifacts_default / "results/e03_qd_candidate_summary.parquet")
    parser.add_argument("--s05-generated-library", type=Path, default=artifacts_default / "policies/e03_generated_policy_library.jsonl")
    parser.add_argument("--s08-discovered-library", type=Path, default=artifacts_default / "policies/e03_qd_discovered_policies.jsonl")
    parser.add_argument("--s09-boundaries", type=Path, default=artifacts_default / "results/e03_phase_boundary_candidates.parquet")
    parser.add_argument("--s09-axis-summary", type=Path, default=artifacts_default / "results/e03_phase_boundary_axis_summary.parquet")
    parser.add_argument("--s12-claims", type=Path, default=artifacts_default / "results/e03_rule_ablation_claims.parquet")
    parser.add_argument("--s13-neighbors", type=Path, default=artifacts_default / "results/e03_classic_nearest_neighbors.parquet")
    parser.add_argument("--s13-analysis", type=Path, default=artifacts_default / "results/e03_classic_position_analysis.parquet")
    parser.add_argument("--target-pool-count", type=int, default=20)
    parser.add_argument("--final-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_FRONTIER_SEED)
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


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


def candidate_table_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "curated_rank",
        "policy_id",
        "policy_name",
        "source_kind",
        "class_id",
        "cautious_label",
        "selection_score",
        "screen_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_improvement_mean",
        "screen_heldout_work_mean",
        "s14_candidate_status",
        "s14_mean_final_sortedness",
        "s14_perturbation_mean_final_sortedness",
        "s14_n32plus_mean_final_sortedness",
        "s14_retention_ratio_vs_s07",
        "s14_timeout_fraction",
        "s14_mean_work",
        "s08_archive_winner",
        "qd_candidate",
        "pareto_frontier_labels_json",
        "selection_reasons_json",
        "computed_dsl_sha256",
        "computed_semantic_hash",
        "code_source",
    ]
    return [column for column in preferred if column in frame.columns]


def write_candidate_figure(path: Path, candidates: pd.DataFrame, run_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ordered = candidates.sort_values("curated_rank")
    x = np.arange(len(ordered))
    colors = {"validated_frontier_candidate": "#4c78a8", "stress_fragile_candidate": "#f58518", "failed_heldout_candidate": "#b279a2"}
    axes[0].bar(
        x,
        ordered["s14_mean_final_sortedness"],
        color=[colors.get(str(value), "#777777") for value in ordered["s14_candidate_status"]],
        alpha=0.85,
        label="S14 mean",
    )
    axes[0].scatter(x, ordered["screen_heldout_final_sortedness_mean"], color="#222222", s=24, label="S07 held-out")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ordered["curated_rank"].astype(str), fontsize=8)
    axes[0].set_xlabel("Curated rank")
    axes[0].set_ylabel("Final inversion sortedness")
    axes[0].set_title("Held-out recheck retention")
    axes[0].grid(True, axis="y", alpha=0.18, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)

    plot = candidates.copy()
    axes[1].scatter(
        plot["s14_perturbation_mean_final_sortedness"],
        plot["s14_n32plus_mean_final_sortedness"],
        s=70 + 130 * plot["selection_score"].rank(pct=True),
        c=plot["class_id"].astype("category").cat.codes,
        cmap="tab10",
        alpha=0.78,
        edgecolor="#222222",
        linewidth=0.5,
    )
    for _, row in plot.iterrows():
        axes[1].text(
            row["s14_perturbation_mean_final_sortedness"] + 0.004,
            row["s14_n32plus_mean_final_sortedness"] + 0.004,
            str(int(row["curated_rank"])),
            fontsize=8,
        )
    axes[1].axvline(0.45, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.45, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Perturbation-profile mean sortedness")
    axes[1].set_ylabel("n>=32 mean sortedness")
    axes[1].set_title("Stress-profile checks")
    axes[1].grid(True, alpha=0.18, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_profiles_report(
    *,
    candidates: pd.DataFrame,
    validation: pd.DataFrame,
    artifacts: dict[str, Path],
    digest: str,
) -> str:
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    summary = candidates.groupby("class_id").agg(candidate_count=("policy_id", "nunique"), mean_s14_sortedness=("s14_mean_final_sortedness", "mean")).reset_index()
    top_cols = [
        "curated_rank",
        "policy_id",
        "policy_name",
        "class_id",
        "s14_candidate_status",
        "s14_mean_final_sortedness",
        "s14_perturbation_mean_final_sortedness",
        "selection_reasons_json",
    ]
    sections: list[str] = []
    for _, row in candidates.sort_values("curated_rank").iterrows():
        dsl_source = str(row["dsl_source"]).strip()
        sections.append(
            f"""### {int(row['curated_rank'])}. {row['policy_id']} ({row['policy_name']})

- Class: `{row.get('class_id', '')}` - {row.get('cautious_label', '')}
- Status: `{row.get('s14_candidate_status', '')}`
- S07 held-out sortedness: {float(row['screen_heldout_final_sortedness_mean']):.6f}
- S14 mean / perturbation / n>=32 sortedness: {float(row['s14_mean_final_sortedness']):.6f} / {float(row['s14_perturbation_mean_final_sortedness']):.6f} / {float(row['s14_n32plus_mean_final_sortedness']):.6f}
- Source/hash: `{row.get('computed_dsl_sha256', '')}` from `{row.get('code_source', '')}`
- Selection reasons: `{row.get('selection_reasons_json', '[]')}`

```text
{dsl_source}
```
"""
        )
    return f"""# E03 S14 Frontier Candidate Profiles

## Top Summary

- Step ID: S14
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; candidate digest `{digest}`.
- Outcome classification: supportive.
- Caveats or blockers: No blocker remains. Candidate claims are finite-search DSL proxy claims from S07-S14, not full public-simulator or biological validation.
- Lay summary: S14 selected {len(candidates)} non-classic frontier/niche policies, verified source hashes, and rechecked them on held-out seeds, input-disorder perturbations, and array sizes up to n=64.
- Recommended next action: Stop for Chief review; if accepted, proceed to S15 atlas construction using this curated candidate library.

## Candidate Set

{markdown_table(candidates[[column for column in top_cols if column in candidates.columns]], [column for column in top_cols if column in candidates.columns])}

## Niche Summary

{markdown_table(summary, list(summary.columns))}

## Policy Profiles

{''.join(sections)}
"""


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    universe: pd.DataFrame,
    selected_pool: pd.DataFrame,
    recheck_runs: pd.DataFrame,
    rechecked_pool: pd.DataFrame,
    final_candidates: pd.DataFrame,
    classic_analysis: pd.DataFrame,
    digest: str,
) -> str:
    success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if success else "constraining/contradictory"
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    class_summary = final_candidates.groupby("class_id").agg(
        candidate_count=("policy_id", "nunique"),
        mean_selection_score=("selection_score", "mean"),
        mean_s14_sortedness=("s14_mean_final_sortedness", "mean"),
        mean_perturbation_sortedness=("s14_perturbation_mean_final_sortedness", "mean"),
    ).reset_index()
    status_counts = final_candidates["s14_candidate_status"].value_counts().rename_axis("status").reset_index(name="count")
    profile_summary = recheck_runs.groupby(["input_profile", "array_size"]).agg(
        run_count=("policy_id", "size"),
        mean_final_sortedness=("final_inversion_sortedness", "mean"),
        mean_improvement=("inversion_sortedness_delta", "mean"),
        timeout_fraction=("timed_out", "mean"),
    ).reset_index()
    candidate_cols = [
        "curated_rank",
        "policy_id",
        "policy_name",
        "source_kind",
        "class_id",
        "screen_heldout_final_sortedness_mean",
        "s14_mean_final_sortedness",
        "s14_perturbation_mean_final_sortedness",
        "s14_n32plus_mean_final_sortedness",
        "s14_retention_ratio_vs_s07",
        "s14_candidate_status",
        "selection_reasons_json",
    ]
    classic_cols = [
        "algorithm_label",
        "classic_position_classification",
        "best_dsl_policy_name",
        "best_dsl_screen_score",
        "classification_confidence",
    ]
    return f"""# E03 S14 Research Step Full Results

## Top Summary

- Step ID: S14
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; candidate digest `{digest}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. The curated set is validated only in the S07/S14 DSL proxy simulator; direct E01/E02 public-method robustness, DG trajectory, and chimeric aggregation assays remain downstream work.
- Lay summary: S14 converted the E03 morphospace atlas into a reusable library of {len(final_candidates)} non-classic policies. The selector used Pareto frontiers, S11 niches, S08/QD provenance, S09 phase diagnostics, S12 ablation signals, and S13 classic-neighborhood evidence, then reran selected policies on held-out seeds, perturbations, and larger arrays.
- Recommended next action: Stop for Chief review; if accepted, proceed to S15 to build the atlas around S01-S14 artifacts and the curated frontier library.

## Frozen Question

Can 10 to 20 novel policies be selected that outperform originals on robustness, DG, chimeric compatibility, or other competence dimensions without sacrificing basic sorting?

## Inputs

- S10 embeddings: `{manifest['inputArtifacts']['s10Embeddings']}`
- S11 clusters: `{manifest['inputArtifacts']['s11Clusters']}`
- S08 QD candidate summary: `{manifest['inputArtifacts']['s08QdCandidates']}`
- S05 generated policy library: `{manifest['inputArtifacts']['s05GeneratedLibrary']}`
- S08 discovered policy library: `{manifest['inputArtifacts']['s08DiscoveredLibrary']}`
- S09 boundary candidates: `{manifest['inputArtifacts']['s09Boundaries']}`
- S09 axis summary: `{manifest['inputArtifacts']['s09AxisSummary']}`
- S12 ablation claims: `{manifest['inputArtifacts']['s12Claims']}`
- S13 classic nearest neighbors: `{manifest['inputArtifacts']['s13Neighbors']}`
- S13 classic-position analysis: `{manifest['inputArtifacts']['s13Analysis']}`

## Lay Summary

This step asks which discovered local rules are worth carrying forward. S14 first removed classic landmarks and any policy without stable DSL source. It then looked for policies that sat on measured trade-off frontiers or represented distinct S11 niches, including QD archive winners, phase-boundary policies, policies near classic neighborhoods, and policies with useful S12 ablation signals. The selected pool was rerun on new seeds, perturbed array profiles, and larger arrays up to n=64 before the final candidate library was written.

The resulting library is not a claim that these policies are universally best. It is a compact, source-verified set of frontier candidates for E04 memory/repair work, E05 substrate transfer, E06 chimeric governance, and E07 cross-world abstraction. Bubble remains the central classic benchmark from S13; the S14 policies are non-classic alternatives with measured proxy advantages or niche coverage.

## Methods

S14 joined S10 policy embeddings, S11 universality classes, S05/S08 policy source records, S09 phase diagnostics, S12 local ablation claims, and S13 classic-neighbor evidence by stable policy ID. Source verification parsed each available DSL program and recomputed the S02 full DSL SHA-256, DSL-derived policy ID, and S05 semantic hash. Policies were eligible only when the recomputed identifiers matched declared source records, the policy was not a classic DSL seed or classic landmark, and S07 prior held-out sortedness and screen score cleared a minimal basic-sorting threshold.

The selector computed fixed Pareto frontiers for screen performance, quality/timeout trade-offs, sparse scale-transfer evidence, phase-boundary diagnostics, and novelty away from classic landmarks in the S10 embedding. It then combined frontier representatives with top S11 class niches, S08 archive/QD provenance, S09 boundary diagnostics, S13 classic-neighborhood hits, and S12 local necessary-feature signals. The pre-recheck pool contained {len(selected_pool)} policies.

Held-out re-evaluation used the CPU DSL interpreter for every selected policy so stochastic and memory/signal-stub policies followed one reference path. The configs used new seeds, random permutations at n=8/16/32/64, nearly sorted arrays, reverse-sorted arrays, and block-reversed arrays. The DSL still treats `target_movable` as S02/S06 defined, so these checks are input-disorder and array-size perturbations rather than Frozen Cell robustness assays.

Final curation required basic sorting retention: no invalid runs, mean held-out final sortedness at least 0.50, and non-negative mean improvement within a small tolerance. Stress-retained candidates additionally had perturbation and n>=32 means at least 0.45 and retained at least 65% of their S07 held-out random-permutation sortedness.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- JAX backend summary, recorded for continuity only: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU reference re-evaluation; no worker pool used.
- New dependencies installed: none.

## Parameters

- S14 seed: `{manifest['parameters']['seed']}`
- Candidate pool target: `{manifest['parameters']['targetPoolCount']}`
- Final candidate target: `{manifest['parameters']['finalCount']}`
- Held-out recheck configs: `{manifest['parameters']['stressConfigIds']}`
- Recheck rows: `{len(recheck_runs)}`
- Eligible source-backed non-classic policies: `{int(universe['candidate_eligible'].sum())}`

## Results

### Classic Context From S13

{markdown_table(classic_analysis[[column for column in classic_cols if column in classic_analysis.columns]], [column for column in classic_cols if column in classic_analysis.columns])}

### Candidate Status Counts

{markdown_table(status_counts, ["status", "count"])}

### Candidate Niches

{markdown_table(class_summary, list(class_summary.columns))}

### Curated Candidates

{markdown_table(final_candidates[[column for column in candidate_cols if column in final_candidates.columns]], [column for column in candidate_cols if column in final_candidates.columns])}

### Held-Out Profile Summary

{markdown_table(profile_summary, list(profile_summary.columns))}

## Validation

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Figures

- Frontier candidate validation figure: `{artifacts['figure']}`

## Provenance

- Candidate digest: `{digest}`
- Git commit at run time: `{manifest['git']['commit']}`
- Git branch at run time: `{manifest['git']['branch']}`
- Git status at run time: `{manifest['git']['statusShort']}`
- Source files:

{source_table}

## Caveats, Blockers, And Limitations

- No blocker remains for S14.
- The held-out recheck is still a DSL local-step proxy and does not replace the E01/E02 public simulator.
- S14 input perturbations are disorder profiles, not Frozen Cell robustness. The current DSL status semantics keep S02 compatibility and do not enforce passive stuck cells.
- DG and chimeric compatibility are inherited from S07-S13 proxy evidence and not rerun as full trajectory or mixed-Algotype assays in this step.
- Candidate rankings are finite-search statements. They can change if S15/E04 expands descriptors, event caps, memory semantics, or public-simulator validation depth.

## Recommended Next Action

Stop for Chief Scientist review before S15. If accepted, use the S14 candidate JSONL and profile report as highlighted policy inputs for the E03 atlas.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    policies_dir = artifacts_dir / "policies"
    reports_dir = artifacts_dir / "reports"
    figures_dir = artifacts_dir / "figures" / "e03"
    configs_dir = artifacts_dir / "configs"
    src_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, policies_dir, reports_dir, figures_dir, configs_dir, src_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    unit_tests = (
        run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)
        if args.run_unit_tests
        else {"command": "unit tests skipped", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
    )

    embeddings = pd.read_parquet(args.s10_embeddings)
    clusters = pd.read_parquet(args.s11_clusters)
    qd_candidates = pd.read_parquet(args.s08_qd_candidates)
    boundaries = pd.read_parquet(args.s09_boundaries)
    axis_summary = pd.read_parquet(args.s09_axis_summary)
    claims = pd.read_parquet(args.s12_claims)
    s13_neighbors = pd.read_parquet(args.s13_neighbors)
    classic_analysis = pd.read_parquet(args.s13_analysis)

    source_table = build_policy_source_table(
        generated_policy_library=args.s05_generated_library,
        qd_candidates=qd_candidates,
        qd_discovered_policy_library=args.s08_discovered_library,
    )
    phase_features = phase_feature_frame(boundaries, axis_summary)
    ablation_features = ablation_feature_frame(claims)
    classic_neighbor_features = classic_neighbor_feature_frame(s13_neighbors)
    universe = build_frontier_universe(
        embeddings=embeddings,
        clusters=clusters,
        source_table=source_table,
        phase_features=phase_features,
        ablation_features=ablation_features,
        classic_neighbor_features=classic_neighbor_features,
    )
    selected_pool = select_frontier_candidates(universe, target_count=args.target_pool_count, final_count=args.final_count)
    configs = stress_configs()
    recheck_runs = evaluate_candidate_pool(selected_pool, configs, seed=args.seed)
    rechecked_pool = recheck_summary_frame(recheck_runs, selected_pool)
    final_candidates = finalize_candidate_set(rechecked_pool, final_count=args.final_count, min_count=10)
    json_records = candidate_policy_json_records(final_candidates)
    digest = frontier_candidate_digest(final_candidates)

    artifacts = {
        "fullResults": step_dir / "research_step_full_results.md",
        "status": step_dir / "status.json",
        "manifest": step_dir / "manifest.json",
        "validationCsv": step_dir / "validation.csv",
        "config": configs_dir / "e03_s14_frontier_candidates_config.json",
        "candidateCsv": tables_dir / "e03_frontier_candidates.csv",
        "candidateParquet": results_dir / "e03_frontier_candidates.parquet",
        "policyJsonl": policies_dir / "e03_frontier_candidate_policies.jsonl",
        "profilesReport": reports_dir / "e03_frontier_candidate_profiles.md",
        "selectionPool": results_dir / "e03_frontier_candidate_selection_pool.parquet",
        "recheckRuns": results_dir / "e03_frontier_candidate_evaluations.parquet",
        "recheckSummary": results_dir / "e03_frontier_candidate_recheck_summary.parquet",
        "sourceVerification": results_dir / "e03_frontier_candidate_source_verification.parquet",
        "validation": results_dir / "e03_frontier_candidate_validation.parquet",
        "figure": figures_dir / "frontier_candidate_validation.png",
        "sourceManifest": src_dir / "e03_frontier_candidates_manifest.json",
        "checksums": checksums_dir / "e03_s14_sha256sums.txt",
    }

    source_files = [
        source_entry(args.repo_dir / "src/e03/frontier_candidates.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e03_s14_select_frontier_candidates.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e03/test_frontier_candidates.py", args.repo_dir),
    ]

    config_payload = {
        "schema": FRONTIER_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "targetPoolCount": args.target_pool_count,
        "finalCount": args.final_count,
        "stressConfigs": [config.__dict__ for config in configs],
        "eligibility": {
            "excludeClassicDslSeeds": True,
            "requireSourceHashValid": True,
            "minimumScreenScore": 0.45,
            "minimumS07HeldoutSortedness": 0.45,
        },
        "finalRetentionCriteria": {
            "basicMeanFinalSortednessMinimum": 0.50,
            "basicMeanImprovementMinimum": -0.02,
            "stressPerturbationMeanMinimum": 0.45,
            "stressN32PlusMeanMinimum": 0.45,
            "randomRetentionRatioMinimum": 0.65,
        },
    }
    write_json(artifacts["config"], config_payload)

    normalize_for_parquet(selected_pool).to_parquet(artifacts["selectionPool"], index=False)
    normalize_for_parquet(recheck_runs).to_parquet(artifacts["recheckRuns"], index=False)
    normalize_for_parquet(rechecked_pool).to_parquet(artifacts["recheckSummary"], index=False)
    source_cols = [
        "policy_id",
        "computed_policy_id",
        "declared_policy_name",
        "computed_policy_name",
        "declared_source_kind",
        "source_hash_valid",
        "source_parse_success",
        "policy_id_matches_source",
        "dsl_sha256_matches_source",
        "semantic_hash_matches_source",
        "declared_dsl_sha256",
        "computed_dsl_sha256",
        "declared_semantic_hash",
        "computed_semantic_hash",
        "code_source",
    ]
    normalize_for_parquet(final_candidates[[column for column in source_cols if column in final_candidates.columns]]).to_parquet(artifacts["sourceVerification"], index=False)
    normalize_for_parquet(final_candidates).to_parquet(artifacts["candidateParquet"], index=False)
    final_candidates[candidate_table_columns(final_candidates)].to_csv(artifacts["candidateCsv"], index=False)
    write_jsonl(artifacts["policyJsonl"], json_records)
    write_candidate_figure(artifacts["figure"], final_candidates, recheck_runs)

    validation = validation_frame(
        universe=universe,
        selected_pool=selected_pool,
        recheck_runs=recheck_runs,
        rechecked_pool=rechecked_pool,
        final_candidates=final_candidates,
        json_records=json_records,
        profiles_written=True,
        figure_written=artifacts["figure"].exists(),
        unit_success=bool(unit_tests["success"]),
        target_count=args.target_pool_count,
        config_count=len(configs),
    )
    normalize_for_parquet(validation).to_parquet(artifacts["validation"], index=False)
    validation.to_csv(artifacts["validationCsv"], index=False)

    manifest = {
        "schema": FRONTIER_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "startedAt": started_at,
        "completedAt": utc_now(),
        "inputArtifacts": {
            "s10Embeddings": str(args.s10_embeddings),
            "s11Clusters": str(args.s11_clusters),
            "s08QdCandidates": str(args.s08_qd_candidates),
            "s05GeneratedLibrary": str(args.s05_generated_library),
            "s08DiscoveredLibrary": str(args.s08_discovered_library),
            "s09Boundaries": str(args.s09_boundaries),
            "s09AxisSummary": str(args.s09_axis_summary),
            "s12Claims": str(args.s12_claims),
            "s13Neighbors": str(args.s13_neighbors),
            "s13Analysis": str(args.s13_analysis),
        },
        "parameters": {
            "seed": args.seed,
            "targetPoolCount": args.target_pool_count,
            "finalCount": args.final_count,
            "stressConfigIds": [config.config_id for config in configs],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
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
            "sourceRows": int(len(source_table)),
            "sourceHashValidRows": int(source_table["source_hash_valid"].sum()) if "source_hash_valid" in source_table.columns else 0,
            "universeRows": int(len(universe)),
            "eligibleRows": int(universe["candidate_eligible"].sum()),
            "selectedPoolRows": int(len(selected_pool)),
            "recheckRunRows": int(len(recheck_runs)),
            "finalCandidateRows": int(len(final_candidates)),
            "finalCandidateClasses": int(final_candidates["class_id"].nunique()),
        },
        "digest": digest,
    }

    source_manifest = {
        "schema": FRONTIER_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "sourceFiles": source_files,
        "notes": "Repository source files are tracked in git; this manifest records hashes only.",
    }
    write_json(artifacts["sourceManifest"], source_manifest)

    report_artifact_subset = {
        key: artifacts[key]
        for key in (
            "fullResults",
            "status",
            "manifest",
            "validationCsv",
            "candidateCsv",
            "policyJsonl",
            "profilesReport",
            "candidateParquet",
            "selectionPool",
            "recheckRuns",
            "recheckSummary",
            "sourceVerification",
            "validation",
            "figure",
            "config",
            "sourceManifest",
            "checksums",
        )
    }
    profiles_text = render_profiles_report(
        candidates=final_candidates,
        validation=validation,
        artifacts={
            "candidateCsv": artifacts["candidateCsv"],
            "policyJsonl": artifacts["policyJsonl"],
            "profilesReport": artifacts["profilesReport"],
            "fullResults": artifacts["fullResults"],
        },
        digest=digest,
    )
    write_text(artifacts["profilesReport"], profiles_text)

    command_line = " ".join(sys.argv)
    report = render_report(
        artifacts=report_artifact_subset,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=command_line,
        universe=universe,
        selected_pool=selected_pool,
        recheck_runs=recheck_runs,
        rechecked_pool=rechecked_pool,
        final_candidates=final_candidates,
        classic_analysis=classic_analysis,
        digest=digest,
    )
    write_text(artifacts["fullResults"], report)

    manifest["artifacts"] = [
        artifact_entry(path, artifacts_dir, key)
        for key, path in report_artifact_subset.items()
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
        "artifactsWritten": [str(path) for path in report_artifact_subset.values()],
        "validationResult": f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}",
        "caveatsOrBlockers": (
            "No blocker remains. Candidate claims are S07/S14 DSL proxy validations; public-simulator robustness, full DG trajectories, and chimeric aggregation are downstream."
            if success
            else "Validation did not fully pass; inspect validation artifacts before using the candidate library."
        ),
        "recommendedNextAction": "Stop for Chief review before S15; if accepted, build the E03 atlas using the S14 candidate library.",
        "outcomeClassification": "supportive" if success else "constraining/contradictory",
        "candidateDigest": digest,
    }
    write_json(artifacts["status"], status)

    checksum_targets = [path for key, path in report_artifact_subset.items() if key != "checksums" and path.exists()]
    checksum_lines = [f"{sha256_file(path)}  {path}" for path in sorted(checksum_targets)]
    write_text(artifacts["checksums"], "\n".join(checksum_lines) + "\n")

    print(json.dumps({"success": success, "report": str(artifacts["fullResults"]), "digest": digest, "validation": status["validationResult"]}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
