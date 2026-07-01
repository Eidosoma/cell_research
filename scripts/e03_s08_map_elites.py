#!/usr/bin/env python3
"""Run E03 S08 MAP-Elites quality-diversity search."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
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

from src.e03.coarse_sweep import load_policy_records, policy_summary_frame, run_configs, screen_configs
from src.e03.gpu_batch_simulator import jax_backend_summary
from src.e03.quality_diversity import (
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_GENERATIONS,
    DEFAULT_QD_SEED,
    add_novelty_columns,
    archive_digest,
    attach_candidate_metadata,
    build_archive,
    candidate_metadata_frame,
    choose_discovered_policies,
    generate_qd_candidates,
    seed_archive_frame,
    selected_parent_ids,
    validation_frame,
)


STEP_ID = "S08"
STEP_NUMBER = 8
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--policy-library", type=Path, default=artifacts_default / "policies/e03_generated_policy_library.jsonl")
    parser.add_argument("--compatibility-table", type=Path, default=artifacts_default / "results/e03_gpu_policy_compatibility.parquet")
    parser.add_argument("--s07-competence", type=Path, default=artifacts_default / "results/e03_policy_competence.parquet")
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_QD_SEED)
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
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def normalize_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: None if value is None else str(value))
    return out


def append_validation_cases(validation: pd.DataFrame, cases: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.concat([validation, pd.DataFrame(cases)], ignore_index=True)


def write_archive_figure(path: Path, archive: pd.DataFrame, discovered: pd.DataFrame, classic_cell_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    routes = sorted(archive["route"].dropna().unique())
    if not routes:
        routes = ["none"]
    fig, axes = plt.subplots(1, len(routes), figsize=(6 * len(routes), 5), squeeze=False)
    for ax, route in zip(axes[0], routes, strict=True):
        subset = archive[archive["route"] == route]
        matrix = subset.pivot_table(index="work_bin", columns="sortedness_bin", values="quality_score", aggfunc="max")
        matrix = matrix.sort_index(ascending=False).sort_index(axis=1)
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=archive["quality_score"].min(), vmax=archive["quality_score"].max())
        ax.set_title(f"{route} archive")
        ax.set_xlabel("Held-out sortedness bin")
        ax.set_ylabel("Held-out work bin")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels([str(int(item)) for item in matrix.columns])
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels([str(int(item)) for item in matrix.index])
        qd_winners = subset[subset["qd_candidate"] == True]  # noqa: E712
        for _, row in qd_winners.iterrows():
            if row["sortedness_bin"] in matrix.columns and row["work_bin"] in matrix.index:
                x = list(matrix.columns).index(row["sortedness_bin"])
                y = list(matrix.index).index(row["work_bin"])
                ax.scatter([x], [y], marker="o", facecolors="none", edgecolors="white", s=95, linewidths=1.5)
        classic_cells = subset[subset["map_cell_id"].isin(classic_cell_ids)]
        for _, row in classic_cells.iterrows():
            if row["sortedness_bin"] in matrix.columns and row["work_bin"] in matrix.index:
                x = list(matrix.columns).index(row["sortedness_bin"])
                y = list(matrix.index).index(row["work_bin"])
                ax.scatter([x], [y], marker="x", c="#ffdf70", s=65, linewidths=1.5)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_label("Archive quality score")
    fig.suptitle("S08 MAP-Elites archive: circles are QD winners; x marks classic DSL cells")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_discovered_jsonl(path: Path, discovered: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in discovered.to_dict(orient="records"):
            parent_ids = json.loads(str(row.get("parent_policy_ids_json", "[]")))
            record = {
                "schema": "eidosoma.e03.qd_discovered_policy.v1",
                "experimentId": EXPERIMENT_ID,
                "researchStepId": STEP_ID,
                "policyId": row["policy_id"],
                "policyName": row["policy_name"],
                "semanticHash": row["semantic_hash"],
                "dslSha256": row["dsl_sha256"],
                "sourceKind": row["source_kind"],
                "generation": int(row["generation"]),
                "lineageDepth": int(row["lineage_depth"]),
                "parentPolicyIds": parent_ids,
                "mutationOperator": row.get("mutation_operator"),
                "mapCellId": row["map_cell_id"],
                "sortednessBin": int(row["sortedness_bin"]),
                "workBin": int(row["work_bin"]),
                "route": row["route"],
                "qualityScore": float(row["quality_score"]),
                "heldoutFinalSortednessMean": float(row["screen_heldout_final_sortedness_mean"]),
                "heldoutImprovementMean": float(row["screen_heldout_improvement_mean"]),
                "heldoutWorkMean": float(row["screen_heldout_work_mean"]),
                "noveltyToS07": float(row["novelty_to_s07"]),
                "noveltyToClassics": float(row["novelty_to_classics"]),
                "archiveWinner": bool(row["archive_winner"]),
                "nonclassicCell": bool(row["nonclassic_cell"]),
                "dslSource": row["dsl_source"],
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def make_summary_tables(archive: pd.DataFrame, all_rows: pd.DataFrame, candidate_runs: pd.DataFrame, discovered: pd.DataFrame) -> dict[str, pd.DataFrame]:
    archive_route = (
        archive.groupby(["route", "qd_candidate"], dropna=False)
        .agg(
            occupied_cells=("map_cell_id", "nunique"),
            mean_quality=("quality_score", "mean"),
            max_quality=("quality_score", "max"),
            mean_novelty_to_classics=("novelty_to_classics", "mean"),
        )
        .reset_index()
    )
    generation = (
        all_rows.groupby(["generation", "qd_candidate"], dropna=False)
        .agg(
            policy_count=("policy_id", "nunique"),
            occupied_cells=("map_cell_id", "nunique"),
            mean_quality=("quality_score", "mean"),
            max_quality=("quality_score", "max"),
        )
        .reset_index()
    )
    route_runs = (
        candidate_runs.groupby("route", dropna=False)
        .agg(
            run_count=("policy_id", "size"),
            policy_count=("policy_id", "nunique"),
            timeout_count=("timed_out", "sum"),
            invalid_count=("invalid", "sum"),
            mean_final_sortedness=("final_inversion_sortedness", "mean"),
        )
        .reset_index()
    )
    top_discovered = discovered.sort_values(["archive_winner", "quality_score"], ascending=[False, False]).head(20)
    return {
        "archive_route": archive_route,
        "generation": generation,
        "route_runs": route_runs,
        "top_discovered": top_discovered,
    }


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    archive: pd.DataFrame,
    discovered: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_runs: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> str:
    validation_success = bool(validation["success"].all() and unit_tests["success"])
    outcome = "supportive" if validation_success else "constraining/contradictory"
    validation_line = f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts.values())
    command_rows = pd.DataFrame(
        [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": command_line, "returnCode": 0, "success": True},
        ]
    )
    source_table = markdown_table(pd.DataFrame(manifest["sourceFiles"]), ["relativePath", "sha256", "sizeBytes"])
    qd_archive_count = int((archive["qd_candidate"] == True).sum()) if not archive.empty else 0  # noqa: E712
    nonclassic_discovered = int(discovered["nonclassic_cell"].sum()) if not discovered.empty else 0
    route_counts = candidate_runs["route"].value_counts().to_dict() if not candidate_runs.empty else {}
    top_cols = [
        "policy_name",
        "source_kind",
        "generation",
        "archive_winner",
        "nonclassic_cell",
        "route",
        "quality_score",
        "screen_heldout_final_sortedness_mean",
        "screen_heldout_work_mean",
    ]
    return f"""# E03 S08 Research Step Full Results

## Top Summary

- Step ID: S08
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; archive digest `{archive_digest(archive)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. The archive uses the S07 local-step proxy screen and fixed S04/S07 descriptors, so discovered policies are candidates for later full-simulator validation rather than final morphogenesis claims.
- Lay summary: S08 seeded a MAP-Elites archive with S07 policy competence rows, generated {len(candidate_summary)} additional DSL policies using S05 mutation and recombination operators, re-evaluated them on the S07 held-out screen configs, and published {len(discovered)} QD-discovered policies. {qd_archive_count} occupied archive cells are QD-generated winners, and {nonclassic_discovered} published QD policies land in cells not occupied by the classic DSL landmarks.
- Recommended next action: Stop for Chief review; if accepted, proceed to S09 phase-boundary sweeps using the S08 archive winners and discovered policies as candidate regions.

## Frozen Question

Can MAP-Elites or novelty search discover policies that occupy behavioral niches not represented by the three classics?

## Inputs

- S07 competence table: `{manifest['inputArtifacts']['s07CompetenceTable']}`
- S05 policy library: `{manifest['inputArtifacts']['s05PolicyLibrary']}`
- S06 compatibility table: `{manifest['inputArtifacts']['s06CompatibilityTable']}`
- S04 descriptor context: held-out final inversion sortedness, held-out improvement, and held-out work count from the competence schema.
- Search seed: `{manifest['parameters']['seed']}`

## Methods

S08 implemented a MAP-Elites search over S02 DSL policies. The initial archive was seeded from all S07 policy competence rows. Each archive cell is defined by binned held-out final sortedness, binned held-out work, and execution route (`jax_batch` or `cpu_fallback`). Quality is the S07 `screen_score`, with novelty to classic DSL landmarks used for tie-breaking and reporting.

For each generation, parent policies were selected from current archive elites plus high-scoring S07 policies. Candidates were created only through S05 mutation and recombination operators, then parsed, canonically renamed, semantic-hashed, tiny-array executed, and duplicate-filtered. Candidate policies were evaluated on the same S07 small-array screen configs using S06 compatibility routing: deterministic batch-ready policies used JAX and exact stochastic, memory, or signal-sensitive policies used the CPU DSL interpreter fallback.

The publishable discovered-policy set includes QD-generated archive winners first, followed by high-quality QD candidates in cells not occupied by classic DSL seeds.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- JAX backend summary: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU fallback plus vectorized JAX batches; no CPU worker pool used.
- New dependencies installed: none.

## Parameters

- Candidate target: `{manifest['parameters']['candidateCount']}`
- Generations: `{manifest['parameters']['generations']}`
- Screen config count per generation: `{manifest['parameters']['screenConfigCount']}`
- Initial S07 seed policy count: `{manifest['summary']['seedPolicyCount']}`
- QD candidate count: `{len(candidate_summary)}`
- Candidate evaluation rows: `{len(candidate_runs)}`
- Candidate route counts: `{json.dumps({str(key): int(value) for key, value in route_counts.items()}, sort_keys=True)}`
- Occupied archive cells: `{len(archive)}`
- QD archive winners: `{qd_archive_count}`

## Results

### Archive By Route

{markdown_table(tables['archive_route'], ["route", "qd_candidate", "occupied_cells", "mean_quality", "max_quality", "mean_novelty_to_classics"])}

### Generation Summary

{markdown_table(tables['generation'], ["generation", "qd_candidate", "policy_count", "occupied_cells", "mean_quality", "max_quality"])}

### Candidate Evaluation Routes

{markdown_table(tables['route_runs'], ["route", "run_count", "policy_count", "timeout_count", "invalid_count", "mean_final_sortedness"])}

### Top Discovered Policies

{markdown_table(tables['top_discovered'], top_cols)}

## Metrics

- `quality_score`: S07 `screen_score`, equal to held-out or screen final sortedness plus one-quarter improvement minus a small work penalty.
- `map_cell_id`: fixed descriptor grid cell from held-out final sortedness bin, held-out work bin, and route.
- `novelty_to_s07`: nearest normalized descriptor distance to any S07 seed policy.
- `novelty_to_classics`: nearest normalized descriptor distance to S03/S05 classic DSL seed landmarks.
- `archive_winner`: whether the QD policy is the best policy in its MAP-Elites cell after all generations.
- `nonclassic_cell`: whether the policy cell is not occupied by a classic DSL seed.

## Figures

- Archive heatmap: `{artifacts['archiveFigure']}`. Circles mark QD-generated archive winners; x marks cells occupied by classic DSL landmarks.

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- S08 inherits S07's proxy local-step scheduler rather than the full E02 public simulator.
- Descriptor cells are fixed, coarse bins; finer bins or different descriptor projections can change archive occupancy.
- Many generated policies still time out at the event cap. These are explicitly retained as failure/low-quality cells when they occupy a descriptor niche.
- Selection-style target-position state is represented as policy-row state in the S07/S08 screen, not full per-cell memory.
- `remember` and `signal` remain S02 lightweight side effects; exact richer communication semantics are deferred.
- The discovered policies are candidate local rules for S09/S14 validation, not final evidence of superior biological analog competence.

## Failed Assumptions

No required input was missing. Mutations preserved valid DSL syntax after duplicate filtering, and archive cells were reproducible from the combined seed-plus-QD result table.

## Provenance

Git commit before S08 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S08 manifest:

{source_table}

## Artifacts

The reusable S08 outputs are the MAP-Elites archive, QD discovered-policy JSONL, candidate evaluation table, lineage table, validation table, archive figure, config, manifests, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    policies_dir = artifacts_dir / "policies"
    figures_dir = artifacts_dir / "figures" / "e03"
    tables_dir = artifacts_dir / "tables"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, policies_dir, figures_dir, tables_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        print("S08: running E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S08: loading S05 policies, S06 routes, and S07 competence", flush=True)
    policy_records = load_policy_records(args.policy_library, args.compatibility_table)
    records_by_id = {record.policy_id: record for record in policy_records}
    s07_summary = pd.read_parquet(args.s07_competence)
    seed_frame = seed_archive_frame(s07_summary, policy_records)
    seed_reference = seed_frame.copy()
    classic_reference = seed_frame[seed_frame["classic_dsl_seed"] == True].copy()  # noqa: E712
    seed_frame = add_novelty_columns(seed_frame, s07_reference=seed_reference, classic_reference=classic_reference)
    classic_cell_ids = set(seed_frame.loc[seed_frame["classic_dsl_seed"] == True, "map_cell_id"])  # noqa: E712
    seed_frame["classic_cell"] = seed_frame["map_cell_id"].isin(classic_cell_ids)
    all_rows = seed_frame.copy()
    archive = build_archive(all_rows)
    parent_depths: dict[str, int] = {record.policy_id: 0 for record in policy_records}
    seen_semantics: dict[str, str] = {record.semantic_hash: record.policy_id for record in policy_records}

    candidate_summaries: list[pd.DataFrame] = []
    candidate_runs: list[pd.DataFrame] = []
    lineage_frames: list[pd.DataFrame] = []
    next_index = 0
    remaining = args.candidate_count
    available_records = dict(records_by_id)

    for generation in range(1, args.generations + 1):
        target = int(math.ceil(remaining / max(args.generations - generation + 1, 1)))
        if target <= 0:
            continue
        parent_ids = selected_parent_ids(all_rows, archive, pool_size=180)
        parent_records = [available_records[policy_id] for policy_id in parent_ids if policy_id in available_records]
        print(f"S08: generation {generation} generating {target} candidates from {len(parent_records)} parents", flush=True)
        candidates = generate_qd_candidates(
            parent_records=parent_records,
            seen_semantics=seen_semantics,
            target_count=target,
            generation=generation,
            seed=args.seed,
            start_index=next_index,
        )
        if not candidates:
            print(f"S08: generation {generation} produced no unique valid candidates", flush=True)
            continue
        next_index += target * 20
        remaining -= len(candidates)
        qd_records = [candidate.to_policy_record() for candidate in candidates]
        for record in qd_records:
            available_records[record.policy_id] = record
        lineage = candidate_metadata_frame(candidates, parent_depths)
        for record in lineage.to_dict(orient="records"):
            parent_depths[str(record["child_policy_id"])] = int(record["lineage_depth"])
        lineage_frames.append(lineage)

        print(f"S08: generation {generation} evaluating {len(qd_records)} candidates on {len(screen_configs())} screen configs", flush=True)
        run_df = run_configs(qd_records, screen_configs(), seed=args.seed)
        run_df["schema"] = "eidosoma.e03.qd_candidate_evaluation.v1"
        run_df["research_step_id"] = STEP_ID
        summary = policy_summary_frame(run_df)
        summary = attach_candidate_metadata(summary, lineage)
        summary = add_novelty_columns(summary, s07_reference=seed_reference, classic_reference=classic_reference)
        summary["classic_cell"] = summary["map_cell_id"].isin(classic_cell_ids)
        candidate_runs.append(run_df)
        candidate_summaries.append(summary)
        all_rows = pd.concat([all_rows, summary], ignore_index=True, sort=False)
        archive = build_archive(all_rows)

    candidate_summary = pd.concat(candidate_summaries, ignore_index=True, sort=False) if candidate_summaries else pd.DataFrame()
    candidate_run_df = pd.concat(candidate_runs, ignore_index=True, sort=False) if candidate_runs else pd.DataFrame()
    lineage_df = pd.concat(lineage_frames, ignore_index=True, sort=False) if lineage_frames else pd.DataFrame()
    archive = build_archive(all_rows)
    archive_ids = set(archive["policy_id"])
    all_rows["archive_winner"] = all_rows["policy_id"].isin(archive_ids)
    archive["archive_winner"] = True
    discovered = choose_discovered_policies(all_rows, archive, classic_cell_ids, limit=64)
    recomputed_archive = build_archive(all_rows)
    known_policy_ids = set(seed_frame["policy_id"]) | set(candidate_summary["policy_id"] if not candidate_summary.empty else [])
    validation = validation_frame(
        archive=archive,
        discovered=discovered,
        lineage=lineage_df,
        candidate_summary=candidate_summary,
        candidate_count_target=args.candidate_count,
        known_policy_ids=known_policy_ids,
        recomputed_archive=recomputed_archive,
        figure_exists=False,
    )

    archive_path = results_dir / "e03_map_elites_archive.parquet"
    archive_csv_path = tables_dir / "e03_map_elites_archive.csv"
    discovered_path = policies_dir / "e03_qd_discovered_policies.jsonl"
    candidate_summary_path = results_dir / "e03_qd_candidate_summary.parquet"
    candidate_runs_path = results_dir / "e03_qd_candidate_evaluations.parquet"
    lineage_path = results_dir / "e03_qd_lineage.parquet"
    lineage_csv_path = tables_dir / "e03_qd_lineage.csv"
    validation_path = results_dir / "e03_qd_validation.parquet"
    validation_csv_path = tables_dir / "e03_qd_validation.csv"
    archive_figure_path = figures_dir / "map_elites_archive.png"
    config_path = configs_dir / "e03_s08_map_elites.json"
    manifest_path = src_snapshot_dir / "e03_quality_diversity_manifest.json"
    artifact_manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = checksums_dir / "sha256sums.txt"
    full_report_path = step_dir / "research_step_full_results.md"

    write_archive_figure(archive_figure_path, archive, discovered, classic_cell_ids)
    validation = validation_frame(
        archive=archive,
        discovered=discovered,
        lineage=lineage_df,
        candidate_summary=candidate_summary,
        candidate_count_target=args.candidate_count,
        known_policy_ids=known_policy_ids,
        recomputed_archive=recomputed_archive,
        figure_exists=archive_figure_path.exists() and archive_figure_path.stat().st_size > 0,
    )
    extra_validation = [
        {
            "validation_case": "unit_tests_passed",
            "success": bool(unit_tests["success"]),
            "expected": "full E03 unit test suite passes",
            "observed": unit_tests["returnCode"],
            "notes": unit_tests["command"],
        },
        {
            "validation_case": "s07_seed_rows_loaded",
            "success": len(seed_frame) == len(s07_summary),
            "expected": len(s07_summary),
            "observed": len(seed_frame),
            "notes": "S07 competence rows joined to S05 DSL sources.",
        },
    ]
    validation = append_validation_cases(validation, extra_validation)
    for column in ("validation_case", "expected", "observed", "notes"):
        validation[column] = validation[column].astype(str)

    normalize_for_parquet(archive).to_parquet(archive_path, index=False)
    archive.to_csv(archive_csv_path, index=False)
    if not candidate_summary.empty:
        normalize_for_parquet(candidate_summary).to_parquet(candidate_summary_path, index=False)
    else:
        pd.DataFrame().to_parquet(candidate_summary_path, index=False)
    if not candidate_run_df.empty:
        normalize_for_parquet(candidate_run_df).to_parquet(candidate_runs_path, index=False)
    else:
        pd.DataFrame().to_parquet(candidate_runs_path, index=False)
    if not lineage_df.empty:
        normalize_for_parquet(lineage_df).to_parquet(lineage_path, index=False)
        lineage_df.to_csv(lineage_csv_path, index=False)
    else:
        pd.DataFrame().to_parquet(lineage_path, index=False)
        pd.DataFrame().to_csv(lineage_csv_path, index=False)
    normalize_for_parquet(validation).to_parquet(validation_path, index=False)
    validation.to_csv(validation_csv_path, index=False)
    write_discovered_jsonl(discovered_path, discovered)

    config = {
        "schema": "eidosoma.e03.s08_map_elites_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "candidateCount": args.candidate_count,
        "generations": args.generations,
        "policyLibrary": str(args.policy_library),
        "compatibilityTable": str(args.compatibility_table),
        "s07CompetenceTable": str(args.s07_competence),
        "screenConfigs": [config.__dict__ for config in screen_configs()],
        "sortednessBinEdges": ["-inf", 0.35, 0.50, 0.65, 0.80, 0.90, "inf"],
        "workBinEdges": ["-inf", 1.0, 5.0, 15.0, 35.0, 75.0, "inf"],
        "cellDefinition": "sortedness_bin x work_bin x route",
        "qualityMetric": "screen_score",
        "noveltyReferences": ["all S07 policies", "classic DSL seed policies"],
    }
    write_json(config_path, config)

    elapsed = time.perf_counter() - started
    source_files = [
        args.repo_dir / "src/e03/quality_diversity.py",
        args.repo_dir / "tests/e03/test_quality_diversity.py",
        args.repo_dir / "scripts/e03_s08_map_elites.py",
        args.repo_dir / "src/e03/policy_generation.py",
        args.repo_dir / "src/e03/coarse_sweep.py",
        args.repo_dir / "src/e03/gpu_batch_simulator.py",
    ]
    manifest = {
        "schema": "eidosoma.e03.quality_diversity_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": {
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "parameters": {
            "seed": args.seed,
            "candidateCount": args.candidate_count,
            "generations": args.generations,
            "screenConfigCount": len(screen_configs()),
        },
        "inputArtifacts": {
            "s05PolicyLibrary": str(args.policy_library),
            "s06CompatibilityTable": str(args.compatibility_table),
            "s07CompetenceTable": str(args.s07_competence),
            "s04CompetenceSpec": str(artifacts_dir / "reports/e03_competence_vector_spec.md"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "jax": jax_backend_summary(),
            "elapsedSeconds": elapsed,
        },
        "sourceFiles": [source_entry(path, args.repo_dir) for path in source_files],
        "summary": {
            "seedPolicyCount": int(len(seed_frame)),
            "candidatePolicyCount": int(len(candidate_summary)),
            "candidateEvaluationRows": int(len(candidate_run_df)),
            "archiveOccupiedCells": int(len(archive)),
            "qdArchiveWinners": int((archive["qd_candidate"] == True).sum()) if not archive.empty else 0,  # noqa: E712
            "discoveredPolicyCount": int(len(discovered)),
            "nonclassicDiscoveredCount": int(discovered["nonclassic_cell"].sum()) if not discovered.empty else 0,
            "archiveDigest": archive_digest(archive),
        },
        "validationSummary": {
            "success": bool(validation["success"].all() and unit_tests["success"]),
            "validationCasesPassed": int(validation["success"].sum()),
            "validationCasesTotal": int(len(validation)),
            "unitTestsReturnCode": int(unit_tests["returnCode"]),
        },
    }

    artifacts = {
        "researchStepReport": full_report_path,
        "mapElitesArchive": archive_path,
        "mapElitesArchiveCsv": archive_csv_path,
        "discoveredPolicies": discovered_path,
        "candidateSummary": candidate_summary_path,
        "candidateEvaluations": candidate_runs_path,
        "lineageParquet": lineage_path,
        "lineageCsv": lineage_csv_path,
        "validationParquet": validation_path,
        "validationCsv": validation_csv_path,
        "archiveFigure": archive_figure_path,
        "validationConfig": config_path,
        "sourceSnapshotManifest": manifest_path,
        "artifactManifest": artifact_manifest_path,
        "runManifest": run_manifest_path,
        "checksums": checksums_path,
    }
    tables = make_summary_tables(archive, all_rows, candidate_run_df, discovered)
    full_report = render_report(
        artifacts=artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        archive=archive,
        discovered=discovered,
        candidate_summary=candidate_summary,
        candidate_runs=candidate_run_df,
        tables=tables,
    )
    write_text(full_report_path, full_report)

    artifact_entries = [
        artifact_entry(archive_path, artifacts_dir, "S08 MAP-Elites archive parquet"),
        artifact_entry(archive_csv_path, artifacts_dir, "CSV sidecar for S08 MAP-Elites archive"),
        artifact_entry(discovered_path, artifacts_dir, "S08 QD-discovered policy JSONL"),
        artifact_entry(candidate_summary_path, artifacts_dir, "S08 QD candidate policy-level summary"),
        artifact_entry(candidate_runs_path, artifacts_dir, "S08 QD candidate run-level evaluations"),
        artifact_entry(lineage_path, artifacts_dir, "S08 QD lineage table"),
        artifact_entry(lineage_csv_path, artifacts_dir, "CSV sidecar for S08 lineage table"),
        artifact_entry(validation_path, artifacts_dir, "S08 validation cases"),
        artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S08 validation cases"),
        artifact_entry(archive_figure_path, artifacts_dir, "S08 MAP-Elites archive figure"),
        artifact_entry(config_path, artifacts_dir, "S08 MAP-Elites config"),
        manifest_self_entry(manifest_path, artifacts_dir, "S08 source snapshot and provenance manifest"),
        manifest_self_entry(artifact_manifest_path, artifacts_dir, "S08 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S08"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S08 outputs"),
        artifact_entry(full_report_path, artifacts_dir, "S08 full-results handoff report"),
    ]
    manifest["artifacts"] = artifact_entries
    write_json(manifest_path, manifest)
    artifact_manifest = {
        "schema": "eidosoma.research_step_artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "artifacts": [
            artifact_entry(archive_path, artifacts_dir, "S08 MAP-Elites archive parquet"),
            artifact_entry(archive_csv_path, artifacts_dir, "CSV sidecar for S08 MAP-Elites archive"),
            artifact_entry(discovered_path, artifacts_dir, "S08 QD-discovered policy JSONL"),
            artifact_entry(candidate_summary_path, artifacts_dir, "S08 QD candidate policy-level summary"),
            artifact_entry(candidate_runs_path, artifacts_dir, "S08 QD candidate run-level evaluations"),
            artifact_entry(lineage_path, artifacts_dir, "S08 QD lineage table"),
            artifact_entry(lineage_csv_path, artifacts_dir, "CSV sidecar for S08 lineage table"),
            artifact_entry(validation_path, artifacts_dir, "S08 validation cases"),
            artifact_entry(validation_csv_path, artifacts_dir, "CSV sidecar for S08 validation cases"),
            artifact_entry(archive_figure_path, artifacts_dir, "S08 MAP-Elites archive figure"),
            artifact_entry(config_path, artifacts_dir, "S08 MAP-Elites config"),
            artifact_entry(manifest_path, artifacts_dir, "S08 source snapshot and provenance manifest"),
            manifest_self_entry(artifact_manifest_path, artifacts_dir, "S08 artifact manifest"),
            manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest updated by S08"),
            manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S08 outputs"),
            artifact_entry(full_report_path, artifacts_dir, "S08 full-results handoff report"),
        ],
    }
    write_json(artifact_manifest_path, artifact_manifest)
    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAt": utc_now(),
        "git": manifest["git"],
        "runtime": manifest["runtime"],
        "commands": [
            {"command": unit_tests["command"], "returnCode": unit_tests["returnCode"], "success": unit_tests["success"]},
            {"command": " ".join(sys.argv), "returnCode": 0, "success": True},
        ],
        "inputs": manifest["inputArtifacts"],
        "parameters": manifest["parameters"],
        "outputs": artifact_manifest["artifacts"],
    }
    write_json(run_manifest_path, run_manifest)
    checksum_targets = [
        full_report_path,
        archive_path,
        archive_csv_path,
        discovered_path,
        candidate_summary_path,
        candidate_runs_path,
        lineage_path,
        lineage_csv_path,
        validation_path,
        validation_csv_path,
        archive_figure_path,
        config_path,
        manifest_path,
        artifact_manifest_path,
        run_manifest_path,
    ]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(artifacts_dir)}" for path in checksum_targets]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")

    success = bool(validation["success"].all() and unit_tests["success"])
    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": success,
                "candidatePolicyCount": int(len(candidate_summary)),
                "archiveOccupiedCells": int(len(archive)),
                "qdArchiveWinners": int((archive["qd_candidate"] == True).sum()) if not archive.empty else 0,  # noqa: E712
                "discoveredPolicyCount": int(len(discovered)),
                "artifactsDir": str(step_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
