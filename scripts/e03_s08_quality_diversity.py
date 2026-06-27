#!/usr/bin/env python3
"""Execute E03 S08: bounded quality-diversity search.

S08 consumes the completed S07 competence and missing-reason artifacts, builds
a deterministic MAP-Elites-style archive, and CPU-validates the selected elites
on small holdout tasks before stopping for Chief Scientist review.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morphospace import (  # noqa: E402
    DSLPolicy,
    QD_SEARCH_VERSION,
    PolicyEventSimulator,
    aggregate_s07_policy_metrics,
    build_map_elites_archive,
    compute_competence_vector,
    parse_rule_program,
    qd_config_dict,
    run_qd_smoke_check,
    validate_qd_archive,
)
from morphospace.competence import COMPETENCE_VECTOR_VERSION, canonical_json, parse_int_array  # noqa: E402
from morphospace.rule_dsl import DSL_VERSION  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S08"
STEP_NUMBER = 8
TITLE = "Use quality-diversity search"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
MAX_ELITES = 32
SMOKE_SAMPLE_SIZE = 16
SMOKE_MAX_ELITES = 5

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_quality_diversity"
ELITE_POLICY_DIR = STEP_DIR / "elite_policies"

S07_METRICS_PATH = RESULTS_DIR / "e03_morphospace_metrics.parquet"
S07_RUNS_PATH = RESULTS_DIR / "e03_coarse_sweep.parquet"
S07_MISSING_PATH = RESULTS_DIR / "e03_s07_missing_reason_records.parquet"
S07_STATUS_PATH = ARTIFACTS_DIR / "research_steps" / "S07" / "status.json"
CORPUS_PATH = ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"


@dataclass(frozen=True)
class CpuValidationTask:
    task_id: str
    task_family: str
    task_panel: str
    input_profile: str
    initial_values: tuple[int, ...]
    scheduler_seed_base: int
    tie_seed_base: int
    frozen_variant: str = "none"
    frozen_positions: tuple[int, ...] = ()
    max_activations: int = 384
    max_swaps: int = 384
    max_comparisons: int = 1536


CPU_VALIDATION_TASKS = (
    CpuValidationTask(
        task_id="s08_cpu_validate_unique_n4_holdout",
        task_family="sorting",
        task_panel="s08_cpu_elite_holdout",
        input_profile="unique_n4_holdout",
        initial_values=(4, 1, 3, 2),
        scheduler_seed_base=21000,
        tie_seed_base=31000,
        max_activations=384,
        max_swaps=384,
        max_comparisons=1536,
    ),
    CpuValidationTask(
        task_id="s08_cpu_validate_duplicate_n5_holdout",
        task_family="sorting_duplicate_values",
        task_panel="s08_cpu_elite_holdout",
        input_profile="duplicate_n5_holdout",
        initial_values=(2, 3, 1, 2, 1),
        scheduler_seed_base=22000,
        tie_seed_base=32000,
        max_activations=512,
        max_swaps=512,
        max_comparisons=2048,
    ),
    CpuValidationTask(
        task_id="s08_cpu_validate_stuck_frozen_n5_holdout",
        task_family="frozen",
        task_panel="s08_cpu_elite_holdout",
        input_profile="unique_n5_stuck_frozen_holdout",
        initial_values=(5, 1, 4, 2, 3),
        scheduler_seed_base=23000,
        tie_seed_base=33000,
        frozen_variant="stuck",
        frozen_positions=(2,),
        max_activations=512,
        max_swaps=512,
        max_comparisons=2048,
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_ready(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        series = prepared[column]
        if series.map(lambda value: isinstance(value, (dict, list, tuple))).any():
            prepared[column] = series.map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple))
                else value
            )
    return prepared


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> list[Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out = table_ready(df)
    out.to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append({"path": str(path), "sizeBytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return rows


def run_command(command: list[str], *, cwd: Path = REPO_ROOT, timeout: int = 600) -> dict[str, Any]:
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


def load_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"success": False, "missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def value_counts_conserved(initial_values: tuple[int, ...], final_values: list[int]) -> bool:
    return sorted(map(int, initial_values)) == sorted(map(int, final_values))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_paths = [S07_METRICS_PATH, S07_RUNS_PATH, S07_MISSING_PATH, CORPUS_PATH]
    absent = [str(path) for path in missing_paths if not path.exists()]
    if absent:
        raise FileNotFoundError(f"S08 requires completed S07/S05 inputs; missing: {absent}")
    return (
        pd.read_parquet(S07_METRICS_PATH),
        pd.read_parquet(S07_RUNS_PATH),
        pd.read_parquet(S07_MISSING_PATH),
        pd.read_parquet(CORPUS_PATH),
        pd.DataFrame([load_status(S07_STATUS_PATH)]),
    )


def write_elite_policy_files(elites_df: pd.DataFrame) -> pd.DataFrame:
    ELITE_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, elite in elites_df.sort_values("eliteRank", kind="mergesort").iterrows():
        policy_id = str(elite["policyId"])
        rank = int(elite["eliteRank"])
        program = parse_rule_program(elite["dslProgramJson"])
        path = ELITE_POLICY_DIR / f"rank_{rank:02d}_{policy_id}.json"
        path.write_text(json.dumps(program.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "eliteRank": rank,
                "policyId": policy_id,
                "descriptorCellId": elite["descriptorCellId"],
                "qualityScore": float(elite["qualityScore"]),
                "noveltyScore": float(elite["noveltyScore"]),
                "elitePolicyPath": str(path),
                "elitePolicySha256": sha256_file(path),
                "dslVersion": program.version,
            }
        )
    return pd.DataFrame(rows)


def elite_lineage_table(elites_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "eliteRank",
        "policyId",
        "family",
        "generationMethod",
        "lineageId",
        "lineageDepth",
        "parentPolicyIdsJson",
        "mutationOperatorsJson",
        "recombinationParentsJson",
        "structureHash",
        "generationIndex",
        "descriptorCellId",
        "descriptorCellLabel",
        "qualityScore",
        "noveltyScore",
    ]
    lineage = elites_df[[column for column in columns if column in elites_df.columns]].copy()
    lineage["lineageSource"] = "S05_policy_corpus_plus_S07_competence_archive"
    return lineage


def run_cpu_elite_validation(elites_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for _, elite in elites_df.sort_values("eliteRank", kind="mergesort").iterrows():
        policy_id = str(elite["policyId"])
        program = parse_rule_program(elite["dslProgramJson"])
        for task_index, task in enumerate(CPU_VALIDATION_TASKS):
            seed = int(task.scheduler_seed_base + int(elite["eliteRank"]))
            tie_seed = int(task.tie_seed_base + int(elite["eliteRank"]))
            condition_id = f"S08::{task.task_id}::{policy_id}::seed{seed}"
            started = time.perf_counter()
            result = PolicyEventSimulator(
                list(task.initial_values),
                DSLPolicy(program),
                frozen_positions=task.frozen_positions,
                frozen_variant=task.frozen_variant,
                scheduler_seed=seed,
                tie_breaker_seed=tie_seed,
                condition_id=condition_id,
                implementation="e03_s08_cpu_elite_validation",
                research_step_id=STEP_ID,
            ).run(
                max_activations=task.max_activations,
                max_swaps=task.max_swaps,
                max_comparisons=task.max_comparisons,
            )
            runtime = time.perf_counter() - started
            conserved = value_counts_conserved(task.initial_values, result.final_values)
            validation_rows.append(
                {
                    "qdSearchVersion": QD_SEARCH_VERSION,
                    "researchStepId": STEP_ID,
                    "eliteRank": int(elite["eliteRank"]),
                    "policyId": policy_id,
                    "descriptorCellId": elite["descriptorCellId"],
                    "validationTaskId": task.task_id,
                    "taskFamily": task.task_family,
                    "taskPanel": task.task_panel,
                    "inputProfile": task.input_profile,
                    "initialValuesJson": canonical_json(list(task.initial_values)),
                    "finalValuesJson": canonical_json(result.final_values),
                    "schedulerSeed": seed,
                    "tieBreakerSeed": tie_seed,
                    "frozenVariant": task.frozen_variant,
                    "frozenPositionsJson": canonical_json(list(task.frozen_positions)),
                    "completed": bool(result.completed),
                    "stopReason": result.stop_reason,
                    "swapCount": int(result.swap_count),
                    "comparisonCount": int(result.comparison_count),
                    "activationCount": int(result.activation_count),
                    "eventCount": int(result.event_count),
                    "blockedMoveAttempts": int(result.blocked_move_attempts),
                    "frozenSwapAttempts": int(result.frozen_swap_attempts),
                    "finalSortednessPercent": float(result.final_sortedness_percent),
                    "finalMonotonicityError": int(result.final_monotonicity_error),
                    "finalAggregation": float(result.final_aggregation),
                    "traceHashSequenceJson": canonical_json([row["state_hash"] for row in result.trace_rows]),
                    "runtimeSeconds": float(runtime),
                    "valueCountsConserved": bool(conserved),
                }
            )
            vector = compute_competence_vector(
                result,
                policy_id=policy_id,
                policy_family=str(elite.get("family", "unknown")),
                task_id=task.task_id,
                task_family=task.task_family,
                task_panel=task.task_panel,
                input_profile=task.input_profile,
                frozen_variant=task.frozen_variant,
                frozen_count=len(task.frozen_positions),
                replicate_index=task_index,
                source_metric_source="s08_cpu_elite_validation",
                source_artifact_path=str(STEP_DIR / "cpu_elite_validation.parquet"),
            )
            vector["eliteRank"] = int(elite["eliteRank"])
            vector["descriptorCellId"] = elite["descriptorCellId"]
            vector["backend"] = "cpu_reference"
            vector["qdSearchVersion"] = QD_SEARCH_VERSION
            vector_rows.append(vector)
    return pd.DataFrame(validation_rows), pd.DataFrame(vector_rows)


def validate_s08(
    *,
    policy_summary: pd.DataFrame,
    archive_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    smoke_df: pd.DataFrame,
    qd_checks_df: pd.DataFrame,
    elite_index_df: pd.DataFrame,
    cpu_validation_df: pd.DataFrame,
    validation_vectors_df: pd.DataFrame,
    upstream_statuses: dict[str, dict[str, Any]],
    repo_test_payload: dict[str, Any],
) -> pd.DataFrame:
    expected_validation_rows = len(elites_df) * len(CPU_VALIDATION_TASKS)
    final_sortedness = pd.to_numeric(cpu_validation_df["finalSortednessPercent"], errors="coerce")
    final_error = pd.to_numeric(cpu_validation_df["finalMonotonicityError"], errors="coerce")
    checks = [
        {
            "checkId": "upstream_s01_s07_success",
            "success": all(bool(status.get("success")) for status in upstream_statuses.values()),
            "detail": canonical_json({step: status.get("success") for step, status in upstream_statuses.items()}),
        },
        {
            "checkId": "s07_inputs_loaded",
            "success": len(policy_summary) >= 100,
            "detail": f"{len(policy_summary)} S07-evaluated base policies summarized",
        },
        {
            "checkId": "smoke_checks_passed",
            "success": bool(smoke_df["success"].all()),
            "detail": f"{int(smoke_df['success'].sum())}/{len(smoke_df)} smoke checks passed before full archive",
        },
        *qd_checks_df.to_dict("records"),
        {
            "checkId": "elite_policy_files_written",
            "success": len(elite_index_df) == len(elites_df) and elite_index_df["elitePolicyPath"].map(lambda p: Path(str(p)).exists()).all(),
            "detail": f"{len(elite_index_df)} elite DSL files written",
        },
        {
            "checkId": "cpu_validation_rows_complete",
            "success": len(cpu_validation_df) == expected_validation_rows,
            "detail": f"{len(cpu_validation_df)} CPU validation rows; expected {expected_validation_rows}",
        },
        {
            "checkId": "cpu_validation_value_counts_conserved",
            "success": bool(cpu_validation_df["valueCountsConserved"].all()),
            "detail": "all CPU elite validation rows preserved input value multisets",
        },
        {
            "checkId": "cpu_validation_metric_ranges",
            "success": bool(final_sortedness.between(0.0, 100.0).all() and (final_error >= 0).all()),
            "detail": "CPU validation final Sortedness is in [0,100] and monotonicity error is nonnegative",
        },
        {
            "checkId": "validation_vectors_match_cpu_rows",
            "success": len(validation_vectors_df) == len(cpu_validation_df) and validation_vectors_df["vectorId"].is_unique,
            "detail": f"{len(validation_vectors_df)} validation competence vectors for {len(cpu_validation_df)} CPU rows",
        },
        {
            "checkId": "repo_unit_tests",
            "success": bool(repo_test_payload["success"]),
            "detail": f"{repo_test_payload['command']} returned {repo_test_payload['returnCode']}",
        },
        {
            "checkId": "s09_not_started",
            "success": not (ARTIFACTS_DIR / "research_steps" / "S09").exists(),
            "detail": "S09 artifact directory is absent",
        },
    ]
    return pd.DataFrame(checks)


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s08_quality_diversity.py", root / "scripts" / "e03_s08_quality_diversity.py"),
            (REPO_ROOT / "morphospace" / "quality_diversity.py", root / "morphospace" / "quality_diversity.py"),
            (REPO_ROOT / "morphospace" / "coarse_sweep.py", root / "morphospace" / "coarse_sweep.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "morphospace" / "policies.py", root / "morphospace" / "policies.py"),
            (REPO_ROOT / "morphospace" / "rule_dsl.py", root / "morphospace" / "rule_dsl.py"),
            (REPO_ROOT / "tests" / "test_e03_quality_diversity.py", root / "tests" / "test_e03_quality_diversity.py"),
        ]
        for source, dest in targets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            paths.append(dest)
    return paths


def render_summary(
    *,
    validation_result: str,
    artifacts_written: list[Path],
    policy_summary: pd.DataFrame,
    archive_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    cpu_validation_df: pd.DataFrame,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    completed = int(cpu_validation_df["completed"].sum()) if not cpu_validation_df.empty else 0
    return "\n".join(
        [
            "# S08 Quality-Diversity Summary",
            "",
            "- Research step ID: S08",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{STEP_DIR}/qd_elites.parquet`, `{STEP_DIR}/cpu_elite_validation.parquet`, and `{STEP_DIR}/elite_policies/`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            (
                "- Lay summary: S08 converted S07's coarse competence table into a bounded MAP-Elites archive. "
                "It selected diverse policy elites by behavior bins, wrote their DSL records, and replayed them on CPU holdout tasks before stopping."
            ),
            "",
            "## Key Counts",
            "",
            f"- Candidate base DSL policies summarized: {len(policy_summary)}",
            f"- Archive cells with one elite each: {len(archive_df)}",
            f"- Selected bounded elites: {len(elites_df)}",
            f"- CPU elite validation rows: {len(cpu_validation_df)}",
            f"- CPU validation rows that reached sorted stop condition: {completed}",
        ]
    ) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S08 Validation Report",
        "",
        "- Research step ID: S08",
        f"- Completion status: {STATUS}",
        f"- Artifacts written: {len(artifacts_written)} files",
        f"- Validation result: {'passed' if validation_df['success'].all() else 'failed'}; {int(validation_df['success'].sum())}/{len(validation_df)} checks passed",
        f"- Caveats or blockers: {'; '.join(caveats)}",
        f"- Recommended next action: {recommended_next_action}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for _, row in validation_df.iterrows():
        rows.append(f"| `{row['checkId']}` | {'pass' if row['success'] else 'fail'} | {str(row['detail']).replace('|', '/')} |")
    return "\n".join(rows) + "\n"


def main() -> None:
    started_at = utc_now()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts_written: list[Path] = []

    competence_df, run_df, missing_df, corpus_df, _ = load_inputs()
    config = qd_config_dict(max_elites=MAX_ELITES, smoke_sample_size=SMOKE_SAMPLE_SIZE, smoke_max_elites=SMOKE_MAX_ELITES)
    config_path = STEP_DIR / "quality_diversity_config.json"
    write_json(config_path, config)
    artifacts_written.append(config_path)

    policy_summary = aggregate_s07_policy_metrics(competence_df, run_df, corpus_df, missing_df)
    smoke_df = run_qd_smoke_check(policy_summary, sample_size=SMOKE_SAMPLE_SIZE, max_elites=SMOKE_MAX_ELITES)
    archive_df, elites_df = build_map_elites_archive(policy_summary, max_elites=MAX_ELITES)
    qd_checks_df = validate_qd_archive(policy_summary, archive_df, elites_df)
    elite_index_df = write_elite_policy_files(elites_df)
    lineage_df = elite_lineage_table(elites_df)
    cpu_validation_df, validation_vectors_df = run_cpu_elite_validation(elites_df)

    artifacts_written.extend(write_table(policy_summary, STEP_DIR / "qd_policy_summary.csv", STEP_DIR / "qd_policy_summary.parquet"))
    artifacts_written.extend(write_table(archive_df, STEP_DIR / "qd_archive_cells.csv", STEP_DIR / "qd_archive_cells.parquet"))
    artifacts_written.extend(write_table(archive_df, RESULTS_DIR / "e03_s08_qd_archive_cells.csv", RESULTS_DIR / "e03_s08_qd_archive_cells.parquet"))
    artifacts_written.extend(write_table(elites_df, STEP_DIR / "qd_elites.csv", STEP_DIR / "qd_elites.parquet"))
    artifacts_written.extend(write_table(elites_df, RESULTS_DIR / "e03_qd_elites.csv", RESULTS_DIR / "e03_qd_elites.parquet"))
    artifacts_written.extend(write_table(elite_index_df, STEP_DIR / "elite_policy_index.csv", STEP_DIR / "elite_policy_index.parquet"))
    artifacts_written.extend(write_table(lineage_df, STEP_DIR / "elite_lineage.csv", STEP_DIR / "elite_lineage.parquet"))
    artifacts_written.extend(write_table(smoke_df, STEP_DIR / "qd_smoke_checks.csv", STEP_DIR / "qd_smoke_checks.parquet"))
    artifacts_written.extend(write_table(cpu_validation_df, STEP_DIR / "cpu_elite_validation.csv", STEP_DIR / "cpu_elite_validation.parquet"))
    artifacts_written.extend(write_table(cpu_validation_df, RESULTS_DIR / "e03_s08_cpu_elite_validation.csv", RESULTS_DIR / "e03_s08_cpu_elite_validation.parquet"))
    artifacts_written.extend(
        write_table(
            validation_vectors_df,
            STEP_DIR / "elite_validation_competence_vectors.csv",
            STEP_DIR / "elite_validation_competence_vectors.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            validation_vectors_df,
            RESULTS_DIR / "e03_s08_elite_validation_competence_vectors.csv",
            RESULTS_DIR / "e03_s08_elite_validation_competence_vectors.parquet",
        )
    )
    artifacts_written.extend(write_table(elite_index_df, RESULTS_DIR / "e03_s08_elite_policy_index.csv", RESULTS_DIR / "e03_s08_elite_policy_index.parquet"))
    artifacts_written.extend([Path(path) for path in elite_index_df["elitePolicyPath"].tolist()])

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_e03_quality_diversity",
            "tests.test_e03_coarse_sweep",
            "tests.test_e03_batch_simulator",
            "tests.test_e03_competence",
        ],
        timeout=600,
    )
    repo_log = STEP_DIR / "repo_unit_test_log.txt"
    repo_log.write_text(repo_test_payload["output"], encoding="utf-8")
    artifacts_written.append(repo_log)

    upstream_statuses = {
        step: load_status(ARTIFACTS_DIR / "research_steps" / step / "status.json")
        for step in ["S01", "S02", "S03", "S04", "S05", "S06", "S07"]
    }
    validation_df = validate_s08(
        policy_summary=policy_summary,
        archive_df=archive_df,
        elites_df=elites_df,
        smoke_df=smoke_df,
        qd_checks_df=qd_checks_df,
        elite_index_df=elite_index_df,
        cpu_validation_df=cpu_validation_df,
        validation_vectors_df=validation_vectors_df,
        upstream_statuses=upstream_statuses,
        repo_test_payload=repo_test_payload,
    )
    artifacts_written.extend(write_table(validation_df, STEP_DIR / "quality_diversity_validation.csv", STEP_DIR / "quality_diversity_validation.parquet"))
    artifacts_written.extend(
        write_table(
            validation_df,
            RESULTS_DIR / "e03_s08_quality_diversity_validation.csv",
            RESULTS_DIR / "e03_s08_quality_diversity_validation.parquet",
        )
    )

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)

    caveats = [
        "S08 is a bounded archive over S07-observed small-array competence, not a new large-array evolutionary run.",
        "The search excludes synthetic chimera pair IDs from elite DSL output, but carries their simple aggregation signal back to base policy descriptors when available.",
        "Robustness and aggregation descriptors are sparse because S07 Frozen Cell and chimera panels were CPU spot checks.",
        "CPU elite validation uses deterministic holdout seeds and small arrays; n=100 and n=1,000 transfer remains deferred.",
    ]
    recommended_next_action = (
        "Stop before S09 for Chief Scientist review; if approved, use S08 elite set and validation table to probe phase boundaries, "
        "with extra CPU/JAX work focused on descriptor cells whose competence changes under small parameter differences."
    )
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )

    summary_path = STEP_DIR / "summary.md"
    validation_report_path = STEP_DIR / "validation_report.md"
    summary_path.write_text(
        render_summary(
            validation_result=validation_result,
            artifacts_written=artifacts_written,
            policy_summary=policy_summary,
            archive_df=archive_df,
            elites_df=elites_df,
            cpu_validation_df=cpu_validation_df,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    validation_report_path.write_text(
        render_validation_report(validation_df, artifacts_written, caveats, recommended_next_action),
        encoding="utf-8",
    )
    artifacts_written.extend([summary_path, validation_report_path])

    status_path = STEP_DIR / "status.json"
    manifest_path = STEP_DIR / "artifact_manifest.json"
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
            "S08 built a bounded quality-diversity archive from S07 competence vectors, wrote selected elite DSL files, "
            "and CPU-validated the elites on small holdout tasks before S09."
        ),
        "searchSummary": {
            "candidatePolicyCount": int(len(policy_summary)),
            "archiveCellCount": int(len(archive_df)),
            "selectedEliteCount": int(len(elites_df)),
            "maxElites": int(MAX_ELITES),
            "smokeSampleSize": int(SMOKE_SAMPLE_SIZE),
        },
        "cpuValidationSummary": {
            "taskCount": int(len(CPU_VALIDATION_TASKS)),
            "rowCount": int(len(cpu_validation_df)),
            "completedRows": int(cpu_validation_df["completed"].sum()),
            "valueCountsConserved": bool(cpu_validation_df["valueCountsConserved"].all()),
        },
        "repoUnitTests": {key: value for key, value in repo_test_payload.items() if key != "output"},
        "versions": {
            "qdSearchVersion": QD_SEARCH_VERSION,
            "competenceVectorVersion": COMPETENCE_VECTOR_VERSION,
            "dslVersion": DSL_VERSION,
        },
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
        "manifestSelfReference": {
            "path": str(manifest_path),
            "sha256": "omitted_self_referential_manifest",
        },
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
                "candidatePolicyCount": int(len(policy_summary)),
                "archiveCellCount": int(len(archive_df)),
                "selectedEliteCount": int(len(elites_df)),
                "cpuValidationRows": int(len(cpu_validation_df)),
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
