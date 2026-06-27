#!/usr/bin/env python3
"""Execute E03 S07: coarse morphospace sweep.

S07 starts with the S06-supported JAX subset, keeps CPU reference checks as the
correctness boundary, and records explicit missing reasons for unsupported
features and task panels rather than silently dropping them.
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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from morphospace import (  # noqa: E402
    COARSE_SWEEP_VERSION,
    DSLPolicy,
    NullPolicy,
    PolicyEventSimulator,
    SweepTask,
    batch_result_to_run_records,
    compare_batch_to_cpu,
    make_missing_record,
    parse_rule_program,
    run_batch_simulator,
    run_cpu_reference_case,
    run_record_validation,
    run_records_to_competence_vectors,
    select_stratified_policy_ids,
    simulation_result_to_run_record,
    summarize_s07_competence,
    support_reason_lookup,
    task_coverage_table,
)
from morphospace.batch_simulator import BATCH_SIMULATOR_VERSION, available_jax_devices  # noqa: E402
from morphospace.competence import COMPETENCE_VECTOR_VERSION, canonical_json  # noqa: E402
from morphospace.rule_dsl import DSL_VERSION  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S07"
STEP_NUMBER = 7
TITLE = "Run a coarse morphospace sweep"
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
STEP_DIR = ARTIFACTS_DIR / "research_steps" / STEP_ID
RESULTS_DIR = ARTIFACTS_DIR / "results"
SHARED_CODE_DIR = ARTIFACTS_DIR / "code" / "e03_coarse_sweep"
DATA_DIR = ARTIFACTS_DIR / "data"
CORPUS_PATH = DATA_DIR / "e03_policy_corpus.parquet"
S06_SUPPORT_PATH = ARTIFACTS_DIR / "research_steps" / "S06" / "batch_dsl_subset_support.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_ready(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in prepared.columns:
        if prepared[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            prepared[column] = prepared[column].map(
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
    rows = []
    for path in paths:
        if path.exists() and path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "sizeBytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
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


def jax_tasks() -> list[SweepTask]:
    return [
        SweepTask(
            task_id="s07_jax_unique_n3_a_seed0",
            task_panel="jax_no_frozen_unique_n3",
            task_family="sorting",
            input_profile="unique_n3_perm_a",
            initial_values=(3, 1, 2),
            scheduler_seed_base=7000,
            replicate_index=0,
            backend="jax_batch",
        ),
        SweepTask(
            task_id="s07_jax_unique_n3_a_seed1",
            task_panel="jax_no_frozen_unique_n3",
            task_family="sorting",
            input_profile="unique_n3_perm_a",
            initial_values=(3, 1, 2),
            scheduler_seed_base=7100,
            replicate_index=1,
            backend="jax_batch",
        ),
        SweepTask(
            task_id="s07_jax_unique_n3_b_seed0",
            task_panel="jax_no_frozen_unique_n3",
            task_family="sorting",
            input_profile="unique_n3_perm_b",
            initial_values=(2, 3, 1),
            scheduler_seed_base=7200,
            replicate_index=0,
            backend="jax_batch",
        ),
        SweepTask(
            task_id="s07_jax_unique_n3_b_seed1",
            task_panel="jax_no_frozen_unique_n3",
            task_family="sorting",
            input_profile="unique_n3_perm_b",
            initial_values=(2, 3, 1),
            scheduler_seed_base=7300,
            replicate_index=1,
            backend="jax_batch",
        ),
        SweepTask(
            task_id="s07_jax_unique_n5_a_seed0",
            task_panel="jax_no_frozen_unique_n5",
            task_family="sorting",
            input_profile="unique_n5_perm_a",
            initial_values=(5, 1, 4, 2, 3),
            scheduler_seed_base=7400,
            replicate_index=0,
            backend="jax_batch",
            max_activations=192,
            max_swaps=192,
        ),
        SweepTask(
            task_id="s07_jax_unique_n5_b_seed0",
            task_panel="jax_no_frozen_unique_n5",
            task_family="sorting",
            input_profile="unique_n5_perm_b",
            initial_values=(4, 2, 5, 1, 3),
            scheduler_seed_base=7500,
            replicate_index=0,
            backend="jax_batch",
            max_activations=192,
            max_swaps=192,
        ),
        SweepTask(
            task_id="s07_jax_duplicate_n5_a_seed0",
            task_panel="jax_duplicate_values_n5",
            task_family="sorting_duplicate_values",
            input_profile="duplicate_n5_a",
            initial_values=(3, 1, 2, 2, 1),
            scheduler_seed_base=7600,
            replicate_index=0,
            backend="jax_batch",
            max_activations=192,
            max_swaps=192,
        ),
        SweepTask(
            task_id="s07_jax_duplicate_n5_b_seed0",
            task_panel="jax_duplicate_values_n5",
            task_family="sorting_duplicate_values",
            input_profile="duplicate_n5_b",
            initial_values=(2, 1, 2, 3, 1),
            scheduler_seed_base=7700,
            replicate_index=0,
            backend="jax_batch",
            max_activations=192,
            max_swaps=192,
        ),
    ]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str], list[str]]:
    corpus_df = pd.read_parquet(CORPUS_PATH)
    support_df = pd.read_parquet(S06_SUPPORT_PATH)
    metadata_by_policy = corpus_df.set_index("policyId").to_dict("index")
    supported_ids = support_df.loc[support_df["batchSupported"], "policyId"].astype(str).tolist()
    unsupported_ids = support_df.loc[~support_df["batchSupported"], "policyId"].astype(str).tolist()
    return corpus_df, support_df, metadata_by_policy, supported_ids, unsupported_ids


def program_map(corpus_df: pd.DataFrame) -> dict[str, Any]:
    return {
        str(row["policyId"]): parse_rule_program(row["dslProgramJson"])
        for _, row in corpus_df.iterrows()
    }


def run_jax_panels(
    supported_ids: list[str],
    programs_by_policy: dict[str, Any],
    metadata_by_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in jax_tasks():
        programs = [programs_by_policy[policy_id] for policy_id in supported_ids]
        initial_values = [list(task.initial_values) for _ in supported_ids]
        seeds = [task.scheduler_seed(index) for index, _ in enumerate(supported_ids)]
        result = run_batch_simulator(
            programs,
            initial_values,
            seeds,
            max_activations=task.max_activations,
            max_swaps=task.max_swaps,
            max_comparisons=task.max_comparisons,
        )
        records.extend(batch_result_to_run_records(result, task, metadata_by_policy))
    return records


def run_cpu_single_policy(
    program: Any,
    task: SweepTask,
    policy_id: str,
    policy_index: int,
    metadata_by_policy: dict[str, Any],
    *,
    tie_breaker_seed: int,
) -> tuple[dict[str, Any], Any]:
    result = PolicyEventSimulator(
        list(task.initial_values),
        DSLPolicy(program),
        frozen_positions=task.frozen_positions,
        frozen_variant=task.frozen_variant,
        scheduler_seed=task.scheduler_seed(policy_index),
        tie_breaker_seed=tie_breaker_seed,
        condition_id=f"S07::{task.task_id}::{policy_id}::seed{task.scheduler_seed(policy_index)}",
        implementation="e03_s07_cpu_reference",
        research_step_id=STEP_ID,
    ).run(
        max_activations=task.max_activations,
        max_swaps=task.max_swaps,
        max_comparisons=task.max_comparisons,
    )
    record = simulation_result_to_run_record(result, task, policy_id, policy_index, metadata_by_policy)
    return record, result


def run_cpu_spot_panels(
    corpus_df: pd.DataFrame,
    supported_ids: list[str],
    unsupported_ids: list[str],
    programs_by_policy: dict[str, Any],
    metadata_by_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    records: list[dict[str, Any]] = []

    unsupported_sample_ids = select_stratified_policy_ids(corpus_df, unsupported_ids, sample_size=64)
    unsupported_task = SweepTask(
        task_id="s07_cpu_unsupported_no_frozen_n3_spot",
        task_panel="cpu_unsupported_no_frozen_sample",
        task_family="sorting",
        input_profile="unique_n3_perm_a",
        initial_values=(3, 1, 2),
        scheduler_seed_base=9000,
        replicate_index=0,
        backend="cpu_reference",
        max_activations=160,
        max_swaps=160,
        max_comparisons=512,
    )
    for policy_index, policy_id in enumerate(unsupported_sample_ids):
        record, _ = run_cpu_single_policy(
            programs_by_policy[policy_id],
            unsupported_task,
            policy_id,
            policy_index,
            metadata_by_policy,
            tie_breaker_seed=13000 + policy_index,
        )
        records.append(record)

    frozen_sample_ids = supported_ids[:24]
    frozen_tasks = [
        SweepTask(
            task_id="s07_cpu_frozen_passive_n4_spot",
            task_panel="cpu_frozen_spot",
            task_family="frozen",
            input_profile="unique_n4_with_frozen_at_1",
            initial_values=(4, 1, 3, 2),
            scheduler_seed_base=10000,
            replicate_index=0,
            backend="cpu_reference",
            frozen_variant="passive",
            frozen_positions=(1,),
            max_activations=256,
            max_swaps=256,
            max_comparisons=768,
        ),
        SweepTask(
            task_id="s07_cpu_frozen_stuck_n4_spot",
            task_panel="cpu_frozen_spot",
            task_family="frozen",
            input_profile="unique_n4_with_frozen_at_1",
            initial_values=(4, 1, 3, 2),
            scheduler_seed_base=10100,
            replicate_index=0,
            backend="cpu_reference",
            frozen_variant="stuck",
            frozen_positions=(1,),
            max_activations=256,
            max_swaps=256,
            max_comparisons=768,
        ),
    ]
    for task in frozen_tasks:
        for policy_index, policy_id in enumerate(frozen_sample_ids):
            record, _ = run_cpu_single_policy(
                programs_by_policy[policy_id],
                task,
                policy_id,
                policy_index,
                metadata_by_policy,
                tie_breaker_seed=14000 + policy_index,
            )
            records.append(record)

    chimera_sample_ids = supported_ids[:12]
    chimera_task = SweepTask(
        task_id="s07_cpu_chimera_dsl_null_alt_n4_spot",
        task_panel="cpu_chimera_spot",
        task_family="chimera",
        input_profile="unique_n4_alternating_dsl_null",
        initial_values=(4, 1, 3, 2),
        scheduler_seed_base=11000,
        replicate_index=0,
        backend="cpu_reference",
        policy_mode="dsl_plus_null_alternating",
        max_activations=256,
        max_swaps=256,
        max_comparisons=768,
    )
    for policy_index, base_policy_id in enumerate(chimera_sample_ids):
        chimera_policy_id = f"chimera_{base_policy_id}_null_alt"
        metadata_by_policy[chimera_policy_id] = {
            **metadata_by_policy.get(base_policy_id, {}),
            "family": "chimera",
            "generationMethod": "s07_spot_pairing",
            "lineageId": base_policy_id,
        }
        program_policy = DSLPolicy(programs_by_policy[base_policy_id])
        policies = [program_policy if position % 2 == 0 else NullPolicy() for position in range(chimera_task.n)]
        result = PolicyEventSimulator(
            list(chimera_task.initial_values),
            policies,
            labels=[position % 2 for position in range(chimera_task.n)],
            scheduler_seed=chimera_task.scheduler_seed(policy_index),
            tie_breaker_seed=15000 + policy_index,
            condition_id=f"S07::{chimera_task.task_id}::{chimera_policy_id}::seed{chimera_task.scheduler_seed(policy_index)}",
            implementation="e03_s07_cpu_reference",
            research_step_id=STEP_ID,
        ).run(
            max_activations=chimera_task.max_activations,
            max_swaps=chimera_task.max_swaps,
            max_comparisons=chimera_task.max_comparisons,
        )
        records.append(
            simulation_result_to_run_record(
                result,
                chimera_task,
                chimera_policy_id,
                policy_index,
                metadata_by_policy,
            )
        )

    sample_sets = {
        "cpuUnsupportedNoFrozenSamplePolicyIds": unsupported_sample_ids,
        "cpuFrozenSpotPolicyIds": frozen_sample_ids,
        "cpuChimeraSpotBasePolicyIds": chimera_sample_ids,
    }
    return records, sample_sets


def run_cpu_jax_spot_checks(
    supported_ids: list[str],
    programs_by_policy: dict[str, Any],
    jax_records: list[dict[str, Any]],
) -> pd.DataFrame:
    lookup = {(row["taskId"], row["policyId"]): row for row in jax_records}
    cpu_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    selected_tasks = jax_tasks()[:2]
    selected_ids = supported_ids[:24]
    for task in selected_tasks:
        for policy_index, policy_id in enumerate(selected_ids):
            cpu_row = run_cpu_reference_case(
                programs_by_policy[policy_id],
                list(task.initial_values),
                task.scheduler_seed(policy_index),
                max_activations=task.max_activations,
                max_swaps=task.max_swaps,
                max_comparisons=task.max_comparisons,
            )
            batch_row = lookup[(task.task_id, policy_id)]
            cpu_rows.append(cpu_row)
            batch_rows.append(batch_row)
            metadata_rows.append(
                {
                    "taskId": task.task_id,
                    "taskPanel": task.task_panel,
                    "policyId": policy_id,
                    "schedulerSeed": task.scheduler_seed(policy_index),
                }
            )
    agreement = pd.DataFrame(compare_batch_to_cpu(cpu_rows, batch_rows))
    metadata_df = pd.DataFrame(metadata_rows)[["taskId", "taskPanel"]]
    return pd.concat([metadata_df, agreement], axis=1)


def build_missing_records(
    corpus_df: pd.DataFrame,
    support_df: pd.DataFrame,
    supported_ids: list[str],
    unsupported_ids: list[str],
    sample_sets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    support_reasons = support_reason_lookup(support_df)
    by_policy = {
        str(policy_id): {**row, "policyId": str(policy_id)}
        for policy_id, row in corpus_df.set_index("policyId").to_dict("index").items()
    }
    records: list[dict[str, Any]] = []

    for panel, task_family in [
        ("jax_no_frozen_unique_n3", "sorting"),
        ("jax_no_frozen_unique_n5", "sorting"),
        ("jax_duplicate_values_n5", "sorting_duplicate_values"),
    ]:
        for policy_id in unsupported_ids:
            records.append(
                make_missing_record(
                    policy_row=by_policy[policy_id],
                    task_id=f"s07_missing_{panel}_unsupported_policy",
                    task_panel=panel,
                    task_family=task_family,
                    backend="jax_batch",
                    missing_reason="policy_outside_s06_jax_subset",
                    support_reasons_json=support_reasons.get(policy_id, "[]"),
                    replacement_handling="eligible for CPU reference or future expanded kernel; 64 unsupported policies sampled on CPU in S07",
                )
            )

    unsupported_sample = set(sample_sets["cpuUnsupportedNoFrozenSamplePolicyIds"])
    for policy_id in unsupported_ids:
        if policy_id not in unsupported_sample:
            records.append(
                make_missing_record(
                    policy_row=by_policy[policy_id],
                    task_id="s07_missing_cpu_unsupported_no_frozen_full_corpus",
                    task_panel="cpu_unsupported_no_frozen_sample",
                    task_family="sorting",
                    backend="cpu_reference",
                    missing_reason="cpu_reference_budgeted_spot_sample_only",
                    support_reasons_json=support_reasons.get(policy_id, "[]"),
                    replacement_handling="not evaluated in S07 CPU sample; retain policy for later CPU or kernel-expanded sweeps",
                )
            )

    frozen_sample = set(sample_sets["cpuFrozenSpotPolicyIds"])
    chimera_sample_base = set(sample_sets["cpuChimeraSpotBasePolicyIds"])
    for policy_id in supported_ids + unsupported_ids:
        if policy_id not in frozen_sample:
            for variant in ("passive", "stuck"):
                records.append(
                    make_missing_record(
                        policy_row=by_policy[policy_id],
                        task_id=f"s07_missing_cpu_frozen_{variant}_full_corpus",
                        task_panel="cpu_frozen_spot",
                        task_family="frozen",
                        backend="cpu_reference",
                        missing_reason="frozen_full_corpus_budgeted_spot_sample_only",
                        support_reasons_json=support_reasons.get(policy_id, "[]"),
                        replacement_handling="S07 ran only a supported-policy frozen CPU spot panel; full frozen sweep deferred",
                    )
                )
        if policy_id not in chimera_sample_base:
            records.append(
                make_missing_record(
                    policy_row=by_policy[policy_id],
                    task_id="s07_missing_cpu_chimera_full_corpus",
                    task_panel="cpu_chimera_spot",
                    task_family="chimera",
                    backend="cpu_reference",
                    missing_reason="chimera_full_corpus_budgeted_spot_sample_only",
                    support_reasons_json=support_reasons.get(policy_id, "[]"),
                    replacement_handling="S07 ran only a simple DSL-plus-null chimeric spot panel; full mixed-policy sweep deferred",
                )
            )
    return records


def validation_table(
    *,
    corpus_df: pd.DataFrame,
    support_df: pd.DataFrame,
    jax_records: list[dict[str, Any]],
    cpu_records: list[dict[str, Any]],
    missing_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    vectors_df: pd.DataFrame,
    cpu_jax_df: pd.DataFrame,
    repo_test_payload: dict[str, Any],
    upstream_statuses: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    run_df = pd.DataFrame(jax_records + cpu_records)
    supported_count = int(support_df["batchSupported"].sum())
    checks = [
        {
            "checkId": "upstream_s01_s06_success",
            "success": all(bool(status.get("success")) for status in upstream_statuses.values()),
            "detail": canonical_json({step: status.get("success") for step, status in upstream_statuses.items()}),
        },
        {
            "checkId": "corpus_rows_and_unique_ids",
            "success": len(corpus_df) == 2048 and corpus_df["policyId"].is_unique,
            "detail": f"{len(corpus_df)} corpus rows; unique IDs={corpus_df['policyId'].is_unique}",
        },
        {
            "checkId": "s06_supported_policy_count",
            "success": supported_count >= 50,
            "detail": f"{supported_count} S06-supported policies available for JAX sweep",
        },
        {
            "checkId": "jax_panel_rows_complete",
            "success": len(jax_records) == supported_count * len(jax_tasks()),
            "detail": f"{len(jax_records)} JAX rows across {len(jax_tasks())} task/seed records",
        },
        {
            "checkId": "duplicate_panel_present",
            "success": bool((pd.DataFrame(jax_records)["taskPanel"] == "jax_duplicate_values_n5").any()),
            "detail": "duplicate-value JAX task panel has evaluated rows",
        },
        {
            "checkId": "cpu_unsupported_sample_present",
            "success": int((pd.DataFrame(cpu_records)["taskPanel"] == "cpu_unsupported_no_frozen_sample").sum()) == 64,
            "detail": "64 unsupported S05 policies evaluated through CPU reference",
        },
        {
            "checkId": "cpu_frozen_spot_present",
            "success": int((pd.DataFrame(cpu_records)["taskPanel"] == "cpu_frozen_spot").sum()) == 48,
            "detail": "24 supported policies evaluated under passive and stuck frozen spot tasks",
        },
        {
            "checkId": "cpu_chimera_spot_present",
            "success": int((pd.DataFrame(cpu_records)["taskPanel"] == "cpu_chimera_spot").sum()) == 12,
            "detail": "12 supported DSL policies paired with null policies in simple chimeras",
        },
        {
            "checkId": "cpu_jax_spot_agreement",
            "success": bool(cpu_jax_df["agreement"].all()),
            "detail": f"{int(cpu_jax_df['agreement'].sum())}/{len(cpu_jax_df)} CPU/JAX spot cases matched exactly",
        },
        {
            "checkId": "value_counts_conserved",
            "success": bool(run_df["valueCountsConserved"].all()),
            "detail": "all evaluated rows preserve input value multisets",
        },
        {
            "checkId": "metric_ranges_valid",
            "success": bool(run_record_validation(run_df)["metricRangesValid"]),
            "detail": "final Sortedness is in [0,100] and monotonicity error is nonnegative",
        },
        {
            "checkId": "competence_vectors_match_runs",
            "success": len(vectors_df) == len(run_df) and vectors_df["vectorId"].is_unique,
            "detail": f"{len(vectors_df)} vectors for {len(run_df)} runs; unique vectors={vectors_df['vectorId'].is_unique}",
        },
        {
            "checkId": "missing_reasons_recorded",
            "success": (
                len(missing_df) > 0
                and missing_df["missingReason"].nunique() >= 3
                and "unknown" not in set(missing_df["policyId"].astype(str))
                and missing_df["policyId"].nunique() >= int((~support_df["batchSupported"]).sum())
            ),
            "detail": (
                f"{len(missing_df)} missing-reason rows; {missing_df['missingReason'].nunique()} reasons; "
                f"{missing_df['policyId'].nunique()} policies with explicit IDs"
            ),
        },
        {
            "checkId": "task_seed_coverage_recorded",
            "success": len(coverage_df) >= 8 and {"evaluated", "missing"}.issubset(set(coverage_df["recordStatus"])),
            "detail": f"{len(coverage_df)} coverage rows with evaluated and missing records",
        },
        {
            "checkId": "repo_unit_tests",
            "success": bool(repo_test_payload["success"]),
            "detail": f"{repo_test_payload['command']} returned {repo_test_payload['returnCode']}",
        },
        {
            "checkId": "s08_not_started",
            "success": not (ARTIFACTS_DIR / "research_steps" / "S08").exists(),
            "detail": "S08 artifact directory is absent",
        },
    ]
    return pd.DataFrame(checks)


def copy_code_artifacts() -> list[Path]:
    paths: list[Path] = []
    for root in [STEP_DIR / "code", SHARED_CODE_DIR]:
        targets = [
            (REPO_ROOT / "scripts" / "e03_s07_coarse_sweep.py", root / "scripts" / "e03_s07_coarse_sweep.py"),
            (REPO_ROOT / "morphospace" / "coarse_sweep.py", root / "morphospace" / "coarse_sweep.py"),
            (REPO_ROOT / "morphospace" / "competence.py", root / "morphospace" / "competence.py"),
            (REPO_ROOT / "morphospace" / "batch_simulator.py", root / "morphospace" / "batch_simulator.py"),
            (REPO_ROOT / "morphospace" / "policies.py", root / "morphospace" / "policies.py"),
            (REPO_ROOT / "morphospace" / "rule_dsl.py", root / "morphospace" / "rule_dsl.py"),
            (REPO_ROOT / "tests" / "test_e03_coarse_sweep.py", root / "tests" / "test_e03_coarse_sweep.py"),
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
    support_summary: dict[str, Any],
    run_summary: dict[str, Any],
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    return "\n".join(
        [
            "# S07 Coarse Morphospace Sweep Summary",
            "",
            "- Research step ID: S07",
            f"- Completion status: {STATUS}; {OUTCOME_CLASSIFICATION}",
            f"- Artifacts written: {len(artifacts_written)} files, including `{STEP_DIR}/coarse_sweep_runs.parquet`, `{STEP_DIR}/competence_vectors.parquet`, `{STEP_DIR}/missing_reason_records.parquet`, and `{STEP_DIR}/task_seed_coverage.parquet`",
            f"- Validation result: {validation_result}",
            f"- Caveats or blockers: {'; '.join(caveats)}",
            f"- Recommended next action: {recommended_next_action}",
            (
                "- Lay summary: S07 evaluated the first trusted JAX subset across no-Frozen and duplicate-value small arrays, "
                "then used CPU reference spot panels for unsupported DSL features, Frozen Cells, and simple DSL-plus-null chimeras. "
                "The output is a coarse competence table for search, not a full morphospace atlas."
            ),
            "",
            "## Key Counts",
            "",
            f"- S06-supported JAX policies: {support_summary['supportedPolicyCount']}",
            f"- S05 policies outside the S06 JAX subset: {support_summary['unsupportedPolicyCount']}",
            f"- Evaluated run rows: {run_summary['rowCount']}",
            f"- Evaluated policy IDs, including chimeric pair IDs: {run_summary['policyCount']}",
            f"- Evaluated task IDs: {run_summary['taskCount']}",
            f"- Completed rows: {run_summary['completedRows']}",
        ]
    ) + "\n"


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[Path], caveats: list[str], recommended_next_action: str) -> str:
    rows = [
        "# S07 Validation Report",
        "",
        "- Research step ID: S07",
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

    corpus_df, support_df, metadata_by_policy, supported_ids, unsupported_ids = load_inputs()
    programs_by_policy = program_map(corpus_df)
    jax_devices = available_jax_devices()

    jax_records = run_jax_panels(supported_ids, programs_by_policy, metadata_by_policy)
    cpu_records, sample_sets = run_cpu_spot_panels(
        corpus_df,
        supported_ids,
        unsupported_ids,
        programs_by_policy,
        metadata_by_policy,
    )
    cpu_jax_df = run_cpu_jax_spot_checks(supported_ids, programs_by_policy, jax_records)
    missing_records = build_missing_records(corpus_df, support_df, supported_ids, unsupported_ids, sample_sets)

    run_records_df = pd.DataFrame(jax_records + cpu_records)
    missing_df = pd.DataFrame(missing_records)
    coverage_df = task_coverage_table(run_records_df, missing_df)

    artifacts_written.extend(
        write_table(
            run_records_df,
            STEP_DIR / "coarse_sweep_runs.csv",
            STEP_DIR / "coarse_sweep_runs.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            run_records_df,
            RESULTS_DIR / "e03_coarse_sweep.csv",
            RESULTS_DIR / "e03_coarse_sweep.parquet",
        )
    )

    vectors_df = run_records_to_competence_vectors(
        run_records_df,
        source_artifact_path=str(STEP_DIR / "coarse_sweep_runs.parquet"),
    )
    summary_df = summarize_s07_competence(vectors_df)
    artifacts_written.extend(
        write_table(
            vectors_df,
            STEP_DIR / "competence_vectors.csv",
            STEP_DIR / "competence_vectors.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            vectors_df,
            RESULTS_DIR / "e03_morphospace_metrics.csv",
            RESULTS_DIR / "e03_morphospace_metrics.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            summary_df,
            STEP_DIR / "competence_summary.csv",
            STEP_DIR / "competence_summary.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            summary_df,
            RESULTS_DIR / "e03_s07_competence_summary.csv",
            RESULTS_DIR / "e03_s07_competence_summary.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            missing_df,
            STEP_DIR / "missing_reason_records.csv",
            STEP_DIR / "missing_reason_records.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            missing_df,
            RESULTS_DIR / "e03_s07_missing_reason_records.csv",
            RESULTS_DIR / "e03_s07_missing_reason_records.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            coverage_df,
            STEP_DIR / "task_seed_coverage.csv",
            STEP_DIR / "task_seed_coverage.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            coverage_df,
            RESULTS_DIR / "e03_s07_task_seed_coverage.csv",
            RESULTS_DIR / "e03_s07_task_seed_coverage.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            cpu_jax_df,
            STEP_DIR / "cpu_spot_checks.csv",
            STEP_DIR / "cpu_spot_checks.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            cpu_jax_df,
            RESULTS_DIR / "e03_s07_cpu_spot_checks.csv",
            RESULTS_DIR / "e03_s07_cpu_spot_checks.parquet",
        )
    )

    repo_test_payload = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
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
        for step in ["S01", "S02", "S03", "S04", "S05", "S06"]
    }
    validation_df = validation_table(
        corpus_df=corpus_df,
        support_df=support_df,
        jax_records=jax_records,
        cpu_records=cpu_records,
        missing_df=missing_df,
        coverage_df=coverage_df,
        vectors_df=vectors_df,
        cpu_jax_df=cpu_jax_df,
        repo_test_payload=repo_test_payload,
        upstream_statuses=upstream_statuses,
    )
    artifacts_written.extend(
        write_table(
            validation_df,
            STEP_DIR / "coarse_sweep_validation.csv",
            STEP_DIR / "coarse_sweep_validation.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            validation_df,
            RESULTS_DIR / "e03_s07_coarse_sweep_validation.csv",
            RESULTS_DIR / "e03_s07_coarse_sweep_validation.parquet",
        )
    )

    code_paths = copy_code_artifacts()
    artifacts_written.extend(code_paths)

    support_summary = {
        "supportedPolicyCount": int(len(supported_ids)),
        "unsupportedPolicyCount": int(len(unsupported_ids)),
        "jaxDeviceSummary": list(jax_devices),
    }
    run_summary = run_record_validation(run_records_df)
    caveats = [
        "JAX sweep is limited to the S06 stateless deterministic adjacent-rule subset.",
        "Frozen Cell and chimeric panels are CPU spot checks only; full-corpus task rows are represented by explicit missing-reason records.",
        "Unsupported stochastic, stateful, target-position, and signal policies are sampled on CPU but not exhaustively swept in S07.",
        "Small-array competence gradients are screening evidence and should not be treated as n=100 or n=1,000 performance claims.",
    ]
    recommended_next_action = (
        "Stop before S08 for Chief Scientist review; if approved, use S07 competence vectors and missing-reason records "
        "to seed quality-diversity search while deciding which unsupported DSL features merit CPU expansion or a new JAX kernel."
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
            support_summary=support_summary,
            run_summary=run_summary,
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
            "S07 produced a coarse competence table for the S06-supported JAX subset, plus CPU spot checks and explicit "
            "missing-reason records for unsupported policy/task combinations before S08."
        ),
        "supportSummary": support_summary,
        "runSummary": run_summary,
        "taskCoverageSummary": {
            "coverageRows": int(len(coverage_df)),
            "evaluatedRows": int((coverage_df["recordStatus"] == "evaluated").sum()),
            "missingRows": int((coverage_df["recordStatus"] == "missing").sum()),
        },
        "sampleSets": sample_sets,
        "cpuJaxSpotAgreement": {
            "caseCount": int(len(cpu_jax_df)),
            "agreementCount": int(cpu_jax_df["agreement"].sum()),
            "allAgreement": bool(cpu_jax_df["agreement"].all()),
        },
        "repoUnitTests": {
            key: value
            for key, value in repo_test_payload.items()
            if key != "output"
        },
        "versions": {
            "coarseSweepVersion": COARSE_SWEEP_VERSION,
            "batchSimulatorVersion": BATCH_SIMULATOR_VERSION,
            "competenceVectorVersion": COMPETENCE_VECTOR_VERSION,
            "dslVersion": DSL_VERSION,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "jaxDevices": list(jax_devices),
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
                "runRows": int(len(run_records_df)),
                "competenceRows": int(len(vectors_df)),
                "missingReasonRows": int(len(missing_df)),
                "statusPath": str(status_path),
                "summaryPath": str(summary_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
