#!/usr/bin/env python3
"""Execute E03 S06 batch-simulator validation and artifact packaging.

S06 introduces a GPU-friendly JAX simulator for a deliberately narrow,
stateless, deterministic adjacent-swap DSL subset. The script first validates
CPU reference cases, then runs the JAX batch simulator only if that CPU gate
passes, records exact CPU/GPU agreement, documents subset limitations, and
stops before S07.
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
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from morphospace import (  # noqa: E402
    BATCH_SIMULATOR_VERSION,
    batch_support_report,
    compare_batch_to_cpu,
    run_batch_simulator,
    run_cpu_reference_case,
    summarize_support,
)
from morphospace.rule_dsl import DSL_VERSION, parse_rule_program  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S06"
STEP_NUMBER = 6
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_S05_CORPUS = DEFAULT_ARTIFACTS_DIR / "data" / "e03_policy_corpus.parquet"
VALIDATION_LIMITS = {"maxActivations": 128, "maxSwaps": 128, "maxComparisons": 512}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "args": args,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return {
            "args": args,
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": repr(exc),
        }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "-v"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["ok"] else "unknown",
        "branch": branch["stdout"].strip() if branch["ok"] else "unknown",
        "dirtyStatus": status["stdout"].strip(),
        "remote": remote["stdout"].strip(),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> list[Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return [csv_path, parquet_path]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not math.isfinite(value):
                return ""
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def artifact_preview(artifacts_written: list[str], limit: int = 24) -> str:
    preview = "\n".join(f"- `{path}`" for path in artifacts_written[:limit])
    if len(artifacts_written) > limit:
        preview += f"\n- ... {len(artifacts_written) - limit} additional artifact path(s) in `status.json` and `artifact_manifest.json`"
    return preview


def load_s05_programs(corpus_path: Path) -> tuple[pd.DataFrame, list[Any]]:
    corpus_df = pd.read_parquet(corpus_path)
    programs = [parse_rule_program(payload) for payload in corpus_df["dslProgramJson"]]
    return corpus_df, programs


def support_table(corpus_df: pd.DataFrame, programs: list[Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    reports = [batch_support_report(program) for program in programs]
    rows = []
    for row, report in zip(corpus_df.to_dict("records"), reports):
        report_row = report.to_row()
        report_row.update(
            {
                "family": row["family"],
                "generationMethod": row["generationMethod"],
                "structureHash": row["structureHash"],
                "complexityScore": int(row["complexityScore"]),
            }
        )
        rows.append(report_row)
    return pd.DataFrame(rows), summarize_support(reports)


def validation_case_rows(corpus_df: pd.DataFrame, programs: list[Any], support_df: pd.DataFrame) -> tuple[list[Any], list[list[int]], list[int], pd.DataFrame]:
    supported_ids = set(support_df.loc[support_df["batchSupported"], "policyId"])
    selected_indices = [index for index, program in enumerate(programs) if program.policy_id in supported_ids]
    selected_programs = [programs[index] for index in selected_indices]
    selected_rows = corpus_df.iloc[selected_indices].reset_index(drop=True)

    case_programs: list[Any] = []
    initial_values: list[list[int]] = []
    scheduler_seeds: list[int] = []
    case_rows: list[dict[str, Any]] = []
    profiles = [([3, 1, 2], 1000), ([2, 3, 1], 2000)]
    for profile_index, (values, seed_base) in enumerate(profiles):
        for policy_index, program in enumerate(selected_programs):
            seed = seed_base + policy_index
            case_programs.append(program)
            initial_values.append(list(values))
            scheduler_seeds.append(seed)
            case_rows.append(
                {
                    "caseId": f"S06_n3_p{profile_index}_{policy_index:03d}",
                    "policyId": program.policy_id,
                    "arrayLength": len(values),
                    "profileIndex": profile_index,
                    "schedulerSeed": seed,
                    "initialValuesJson": json.dumps(values, separators=(",", ":")),
                    "family": selected_rows.iloc[policy_index]["family"],
                    "generationMethod": selected_rows.iloc[policy_index]["generationMethod"],
                }
            )

    five_programs = selected_programs[:24]
    five_rows = selected_rows.iloc[:24].reset_index(drop=True)
    for policy_index, program in enumerate(five_programs):
        values = [5, 1, 4, 2, 3]
        seed = 3000 + policy_index
        case_programs.append(program)
        initial_values.append(values)
        scheduler_seeds.append(seed)
        case_rows.append(
            {
                "caseId": f"S06_n5_0_{policy_index:03d}",
                "policyId": program.policy_id,
                "arrayLength": len(values),
                "profileIndex": 2,
                "schedulerSeed": seed,
                "initialValuesJson": json.dumps(values, separators=(",", ":")),
                "family": five_rows.iloc[policy_index]["family"],
                "generationMethod": five_rows.iloc[policy_index]["generationMethod"],
            }
        )
    return case_programs, initial_values, scheduler_seeds, pd.DataFrame(case_rows)


def run_cpu_gate(case_programs: list[Any], initial_values: list[list[int]], scheduler_seeds: list[int]) -> pd.DataFrame:
    rows = []
    for program, values, seed in zip(case_programs, initial_values, scheduler_seeds):
        rows.append(
            run_cpu_reference_case(
                program,
                values,
                seed,
                max_activations=VALIDATION_LIMITS["maxActivations"],
                max_swaps=VALIDATION_LIMITS["maxSwaps"],
                max_comparisons=VALIDATION_LIMITS["maxComparisons"],
            )
        )
    df = pd.DataFrame(rows)
    df["valueCountsConserved"] = [
        Counter(json.loads(row.initialValuesJson)) == Counter(json.loads(row.finalValuesJson)) for row in df.itertuples()
    ]
    return df


def run_batch_by_length(case_programs: list[Any], initial_values: list[list[int]], scheduler_seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_length: dict[int, list[int]] = {}
    for index, values in enumerate(initial_values):
        by_length.setdefault(len(values), []).append(index)
    for _, indices in sorted(by_length.items()):
        batch_result = run_batch_simulator(
            [case_programs[index] for index in indices],
            [initial_values[index] for index in indices],
            [scheduler_seeds[index] for index in indices],
            max_activations=VALIDATION_LIMITS["maxActivations"],
            max_swaps=VALIDATION_LIMITS["maxSwaps"],
            max_comparisons=VALIDATION_LIMITS["maxComparisons"],
        )
        rows.extend(batch_result.to_rows())
    return pd.DataFrame(rows)


def run_benchmark(case_programs: list[Any], initial_values: list[list[int]], scheduler_seeds: list[int]) -> pd.DataFrame:
    indices = [index for index, values in enumerate(initial_values) if len(values) == 3][:184]
    bench_programs = [case_programs[index] for index in indices]
    bench_values = [initial_values[index] for index in indices]
    bench_seeds = [scheduler_seeds[index] for index in indices]

    cpu_started = time.perf_counter()
    _ = [
        run_cpu_reference_case(
            program,
            values,
            seed,
            max_activations=VALIDATION_LIMITS["maxActivations"],
            max_swaps=VALIDATION_LIMITS["maxSwaps"],
            max_comparisons=VALIDATION_LIMITS["maxComparisons"],
        )
        for program, values, seed in zip(bench_programs, bench_values, bench_seeds)
    ]
    cpu_seconds = time.perf_counter() - cpu_started

    warm_started = time.perf_counter()
    warm_result = run_batch_simulator(
        bench_programs,
        bench_values,
        bench_seeds,
        max_activations=VALIDATION_LIMITS["maxActivations"],
        max_swaps=VALIDATION_LIMITS["maxSwaps"],
        max_comparisons=VALIDATION_LIMITS["maxComparisons"],
    )
    warm_seconds = time.perf_counter() - warm_started

    timed_started = time.perf_counter()
    timed_result = run_batch_simulator(
        bench_programs,
        bench_values,
        bench_seeds,
        max_activations=VALIDATION_LIMITS["maxActivations"],
        max_swaps=VALIDATION_LIMITS["maxSwaps"],
        max_comparisons=VALIDATION_LIMITS["maxComparisons"],
    )
    timed_seconds = time.perf_counter() - timed_started

    return pd.DataFrame(
        [
            {
                "benchmarkId": "cpu_reference_serial",
                "backend": "python_cpu_reference",
                "caseCount": len(bench_programs),
                "arrayLength": 3,
                "wallTimeSeconds": cpu_seconds,
                "casesPerSecond": len(bench_programs) / cpu_seconds if cpu_seconds else None,
                "device": "cpu",
                "notes": "Serial S01/S02 PolicyEventSimulator references.",
            },
            {
                "benchmarkId": "jax_batch_compile_and_run",
                "backend": "jax_batch",
                "caseCount": len(bench_programs),
                "arrayLength": 3,
                "wallTimeSeconds": warm_seconds,
                "casesPerSecond": len(bench_programs) / warm_seconds if warm_seconds else None,
                "device": warm_result.device,
                "notes": "Includes first JIT compilation for this shape.",
            },
            {
                "benchmarkId": "jax_batch_cached_run",
                "backend": "jax_batch",
                "caseCount": len(bench_programs),
                "arrayLength": 3,
                "wallTimeSeconds": timed_seconds,
                "casesPerSecond": len(bench_programs) / timed_seconds if timed_seconds else None,
                "device": timed_result.device,
                "notes": "Same shape after JIT cache warmup.",
            },
        ]
    )


def run_repo_unit_tests(step_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    result = run_command(cmd, cwd=REPO_ROOT)
    log_path = step_dir / "repo_unit_test_log.txt"
    log_path.write_text(
        "$ " + " ".join(cmd) + "\n\nSTDOUT\n" + result["stdout"] + "\n\nSTDERR\n" + result["stderr"],
        encoding="utf-8",
    )
    row = {
        "validationFamily": "repo_unit_tests",
        "conditionId": "repo_unit_tests",
        "success": result["ok"],
        "validationDetail": f"Repository unittest discovery return code {result['returncode']}.",
        "sourcePath": str(log_path),
    }
    payload = {
        "command": cmd,
        "returnCode": result["returncode"],
        "success": result["ok"],
        "logPath": str(log_path),
    }
    return row, payload, log_path


def validate_s06(
    *,
    support_df: pd.DataFrame,
    support_summary: dict[str, Any],
    cpu_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    upstream_status_paths: dict[str, Path],
    repo_test_row: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "validationFamily": "upstream_boundary",
            "conditionId": "s05_policy_corpus_present",
            "success": bool(support_summary["policyCount"] >= 2000),
            "validationDetail": f"S05 corpus rows assessed for S06 support: {support_summary['policyCount']}.",
        }
    )
    rows.append(
        {
            "validationFamily": "subset_contract",
            "conditionId": "supported_subset_nontrivial",
            "success": bool(support_summary["supportedPolicyCount"] >= 50),
            "validationDetail": f"{support_summary['supportedPolicyCount']} policies support the S06 stateless adjacent subset.",
        }
    )
    rows.append(
        {
            "validationFamily": "subset_contract",
            "conditionId": "unsupported_reasons_documented",
            "success": bool(support_summary["unsupportedReasonCounts"]),
            "validationDetail": f"Unsupported reason classes: {len(support_summary['unsupportedReasonCounts'])}.",
        }
    )
    rows.append(
        {
            "validationFamily": "cpu_reference_gate",
            "conditionId": "cpu_reference_cases_complete",
            "success": bool(len(cpu_df) > 0 and cpu_df["valueCountsConserved"].all()),
            "validationDetail": f"{len(cpu_df)} CPU reference cases ran before JAX batch validation.",
        }
    )
    rows.append(
        {
            "validationFamily": "jax_batch",
            "conditionId": "batch_rows_match_cpu_case_count",
            "success": bool(len(batch_df) == len(cpu_df)),
            "validationDetail": f"Batch rows={len(batch_df)}; CPU rows={len(cpu_df)}.",
        }
    )
    rows.append(
        {
            "validationFamily": "cpu_gpu_agreement",
            "conditionId": "exact_metrics_and_trace_hashes",
            "success": bool(len(agreement_df) == len(cpu_df) and agreement_df["agreement"].all()),
            "validationDetail": f"{int(agreement_df['agreement'].sum())} of {len(agreement_df)} CPU/JAX cases matched exactly.",
        }
    )
    rows.append(
        {
            "validationFamily": "jax_device",
            "conditionId": "cuda_device_used",
            "success": bool(batch_df["backendDevice"].astype(str).str.contains("cuda|gpu", case=False, regex=True).any()),
            "validationDetail": f"Observed batch devices: {sorted(batch_df['backendDevice'].astype(str).unique())}.",
        }
    )
    rows.append(
        {
            "validationFamily": "benchmark",
            "conditionId": "benchmark_rows_present",
            "success": bool({"cpu_reference_serial", "jax_batch_cached_run"}.issubset(set(benchmark_df["benchmarkId"]))),
            "validationDetail": f"Benchmark rows written: {len(benchmark_df)}.",
        }
    )
    for step_id, path in upstream_status_paths.items():
        status = read_json(path) if path.exists() else {}
        rows.append(
            {
                "validationFamily": "upstream_boundary",
                "conditionId": f"{step_id.lower()}_completed_anchor",
                "success": bool(status.get("success")),
                "validationDetail": f"{step_id} status confirms required upstream boundary completed successfully.",
                "sourcePath": str(path),
            }
        )
    rows.append(repo_test_row)
    return pd.DataFrame(rows)


def validation_counts(validation_df: pd.DataFrame) -> pd.DataFrame:
    return (
        validation_df.groupby("validationFamily", dropna=False)["success"]
        .agg(total="count", passed="sum")
        .reset_index()
        .assign(failed=lambda df: df["total"] - df["passed"])
    )


def render_limitations_md(
    *,
    support_summary: dict[str, Any],
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    reason_rows = [[key, value] for key, value in support_summary["unsupportedReasonCounts"].items()]
    return f"""# E03 S06 Batch Simulator Limitations

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: {recommended_next_action}

## Supported DSL Subset

S06 supports only stateless deterministic adjacent local policies:

- Predicates: `always`, `left_exists`, `right_exists`, `compare_left`, `compare_right`, `position_compare`
- Actions: `wait`, `swap_left`, `swap_right`
- Execution mode: one fixed array length per JAX batch, no frozen cells, no per-cell mixed policies, increasing-order sortedness metrics
- Scheduler: precomputed NumPy actor-cell schedules passed into JAX so the CPU reference and GPU batch use identical actor cell IDs

## Excluded For Now

Stochastic action probabilities, tie-breaker RNG, internal state, state updates, target-position estimates, `swap_target`, target predicates, signal placeholders, frozen-cell behavior, mixed-policy chimeras, and variable-length arrays inside a single compiled batch are outside this first exact-match kernel.

## S05 Coverage

- S05 policies assessed: {support_summary['policyCount']}
- Supported by S06 subset: {support_summary['supportedPolicyCount']}
- Unsupported by S06 subset: {support_summary['unsupportedPolicyCount']}

{markdown_table(["Unsupported reason", "Policy count"], reason_rows)}
"""


def render_validation_report(validation_df: pd.DataFrame, artifacts_written: list[str], caveats: list[str]) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    failed = validation_df[~validation_df["success"].astype(bool)]
    failure_text = "None"
    if not failed.empty:
        failure_text = markdown_table(
            ["Family", "Condition", "Detail"],
            failed[["validationFamily", "conditionId", "validationDetail"]].head(30).values.tolist(),
        )
    rows = validation_counts(validation_df)[["validationFamily", "passed", "total", "failed"]].values.tolist()
    return f"""# S06 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: stop before S07 for Chief Scientist review; if approved, run the coarse morphospace sweep using S06-supported policies first and CPU-reference unsupported policies or later kernels as needed.

## Validation Counts

{markdown_table(["Validation family", "Passed", "Total", "Failed"], rows)}

## Failed Checks

{failure_text}
"""


def render_summary(
    *,
    validation_df: pd.DataFrame,
    artifacts_written: list[str],
    caveats: list[str],
    support_summary: dict[str, Any],
    agreement_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    outcome = OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory"
    cached = benchmark_df[benchmark_df["benchmarkId"] == "jax_batch_cached_run"].iloc[0]
    return f"""# S06 Summary

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written: `{artifacts_written[0]}` and {len(artifacts_written) - 1} additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {"; ".join(caveats)}
- Lay summary: S06 built a JAX batch simulator for the stateless deterministic adjacent-swap DSL subset. It first ran CPU references, then validated the GPU-friendly batch path against those references. {support_summary['supportedPolicyCount']} of {support_summary['policyCount']} S05 policies fit the first subset, and {int(agreement_df['agreement'].sum())} of {len(agreement_df)} CPU/JAX validation cases matched final arrays, stop reasons, metrics, and swap-trace hashes exactly. The cached JAX batch benchmark processed {cached['caseCount']} validation-like cases at {cached['casesPerSecond']:.2f} cases/s on `{cached['device']}`.
- Recommended next action: stop before S07 for Chief Scientist review. If approved, run the coarse morphospace sweep with this validated subset and keep unsupported S05 policies on the CPU reference path or extend the kernel deliberately.
"""


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path, shared_code_dir: Path) -> list[Path]:
    copied: list[Path] = []
    mapping = [
        (REPO_ROOT / "morphospace" / "__init__.py", "morphospace/__init__.py"),
        (REPO_ROOT / "morphospace" / "batch_simulator.py", "morphospace/batch_simulator.py"),
        (REPO_ROOT / "morphospace" / "policies.py", "morphospace/policies.py"),
        (REPO_ROOT / "morphospace" / "rule_dsl.py", "morphospace/rule_dsl.py"),
        (REPO_ROOT / "morphospace" / "policy_corpus.py", "morphospace/policy_corpus.py"),
        (REPO_ROOT / "scripts" / "e03_s06_batch_simulator.py", "scripts/e03_s06_batch_simulator.py"),
        (REPO_ROOT / "tests" / "test_e03_batch_simulator.py", "tests/test_e03_batch_simulator.py"),
    ]
    for source, rel in mapping:
        for root in [step_dir / "code", shared_code_dir]:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(dest)
    return copied


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--s05-corpus", type=Path, default=DEFAULT_S05_CORPUS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started_at = utc_now()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    shared_code_dir = artifacts_dir / "code" / "e03_gpu_batch_simulator"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    shared_code_dir.mkdir(parents=True, exist_ok=True)
    artifacts_written: list[Path] = []

    corpus_df, programs = load_s05_programs(args.s05_corpus)
    support_df, support_summary = support_table(corpus_df, programs)
    artifacts_written.extend(
        write_table(
            support_df,
            step_dir / "batch_dsl_subset_support.csv",
            step_dir / "batch_dsl_subset_support.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            support_df,
            results_dir / "e03_s06_batch_dsl_subset_support.csv",
            results_dir / "e03_s06_batch_dsl_subset_support.parquet",
        )
    )
    support_summary_path = step_dir / "batch_dsl_subset_summary.json"
    config_path = step_dir / "batch_simulator_config.json"
    write_json(support_summary_path, support_summary)
    write_json(
        config_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "batchSimulatorVersion": BATCH_SIMULATOR_VERSION,
            "dslVersion": DSL_VERSION,
            "s05CorpusPath": str(args.s05_corpus),
            "validationLimits": VALIDATION_LIMITS,
            "workerCount": 1,
            "threading": "serial_cpu_reference_then_single_jax_batch_per_shape",
            "cpuReferenceGate": "JAX batch validation runs after CPU reference rows complete and conserve value counts.",
            "startedAt": started_at,
        },
    )
    artifacts_written.extend([support_summary_path, config_path])

    case_programs, initial_values, scheduler_seeds, case_df = validation_case_rows(corpus_df, programs, support_df)
    artifacts_written.extend(
        write_table(case_df, step_dir / "validation_cases.csv", step_dir / "validation_cases.parquet")
    )

    cpu_df = run_cpu_gate(case_programs, initial_values, scheduler_seeds)
    cpu_gate_passed = bool(len(cpu_df) > 0 and cpu_df["valueCountsConserved"].all())
    artifacts_written.extend(
        write_table(
            cpu_df,
            step_dir / "cpu_reference_validation.csv",
            step_dir / "cpu_reference_validation.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            cpu_df,
            results_dir / "e03_s06_cpu_reference_validation.csv",
            results_dir / "e03_s06_cpu_reference_validation.parquet",
        )
    )
    if not cpu_gate_passed:
        raise RuntimeError("CPU reference gate failed; refusing to run JAX batch validation")

    batch_df = run_batch_by_length(case_programs, initial_values, scheduler_seeds)
    artifacts_written.extend(
        write_table(
            batch_df,
            step_dir / "batch_simulation_validation.csv",
            step_dir / "batch_simulation_validation.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            batch_df,
            results_dir / "e03_s06_batch_simulation_validation.csv",
            results_dir / "e03_s06_batch_simulation_validation.parquet",
        )
    )
    agreement_df = pd.DataFrame(compare_batch_to_cpu(cpu_df.to_dict("records"), batch_df.to_dict("records")))
    artifacts_written.extend(
        write_table(
            agreement_df,
            step_dir / "cpu_gpu_agreement.csv",
            step_dir / "cpu_gpu_agreement.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            agreement_df,
            results_dir / "e03_s06_cpu_gpu_agreement.csv",
            results_dir / "e03_s06_cpu_gpu_agreement.parquet",
        )
    )

    benchmark_df = run_benchmark(case_programs, initial_values, scheduler_seeds)
    artifacts_written.extend(
        write_table(
            benchmark_df,
            step_dir / "batch_benchmark.csv",
            step_dir / "batch_benchmark.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            benchmark_df,
            results_dir / "e03_s06_batch_benchmark.csv",
            results_dir / "e03_s06_batch_benchmark.parquet",
        )
    )

    repo_test_row, repo_test_payload, repo_test_log = run_repo_unit_tests(step_dir)
    artifacts_written.append(repo_test_log)
    upstream_status_paths = {
        "S01": artifacts_dir / "research_steps" / "S01" / "status.json",
        "S02": artifacts_dir / "research_steps" / "S02" / "status.json",
        "S03": artifacts_dir / "research_steps" / "S03" / "status.json",
        "S04": artifacts_dir / "research_steps" / "S04" / "status.json",
        "S05": artifacts_dir / "research_steps" / "S05" / "status.json",
    }
    validation_df = validate_s06(
        support_df=support_df,
        support_summary=support_summary,
        cpu_df=cpu_df,
        batch_df=batch_df,
        agreement_df=agreement_df,
        benchmark_df=benchmark_df,
        upstream_status_paths=upstream_status_paths,
        repo_test_row=repo_test_row,
    )
    artifacts_written.extend(
        write_table(
            validation_df,
            step_dir / "batch_simulator_validation.csv",
            step_dir / "batch_simulator_validation.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            validation_df,
            results_dir / "e03_s06_batch_simulator_validation.csv",
            results_dir / "e03_s06_batch_simulator_validation.parquet",
        )
    )

    caveats = [
        "The S06 JAX path supports only stateless deterministic adjacent-swap/wait DSL policies.",
        "Stochastic actions, internal state, target-position rules, signals, frozen-cell behavior, and mixed-policy chimeras remain on the CPU reference path or require later kernels.",
        "Each compiled JAX batch has one fixed array length; variable-length sweeps require separate batches by length.",
        "S06 validates simulator agreement, not policy competence on the full S04 task panels.",
    ]
    recommended_next_action = (
        "Stop before S07 for Chief Scientist review; if approved, run coarse morphospace sweeps with S06-supported "
        "policies first and route unsupported policies through CPU reference or explicit later kernel extensions."
    )
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )
    code_paths = copy_code_artifacts(step_dir, shared_code_dir)
    artifacts_written.extend(code_paths)

    artifact_strings_so_far = [str(path) for path in artifacts_written]
    limitations_path = step_dir / "batch_simulator_limitations.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_md_path = step_dir / "summary.md"
    limitations_path.write_text(
        render_limitations_md(
            support_summary=support_summary,
            artifacts_written=artifact_strings_so_far,
            validation_result=validation_result,
            caveats=caveats,
            recommended_next_action=recommended_next_action,
        ),
        encoding="utf-8",
    )
    validation_report_path.write_text(
        render_validation_report(validation_df, artifact_strings_so_far, caveats),
        encoding="utf-8",
    )
    summary_md_path.write_text(
        render_summary(
            validation_df=validation_df,
            artifacts_written=artifact_strings_so_far,
            caveats=caveats,
            support_summary=support_summary,
            agreement_df=agreement_df,
            benchmark_df=benchmark_df,
        ),
        encoding="utf-8",
    )
    artifacts_written.extend([limitations_path, validation_report_path, summary_md_path])

    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    git_metadata = get_git_metadata()
    cached_benchmark = benchmark_df[benchmark_df["benchmarkId"] == "jax_batch_cached_run"].iloc[0].to_dict()
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": "Vectorize the simulator",
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
            f"S06 built a JAX batch simulator for {support_summary['supportedPolicyCount']} S05 policies in the "
            "stateless deterministic adjacent DSL subset and validated exact CPU/JAX agreement before S07."
        ),
        "supportSummary": support_summary,
        "cpuReferenceGatePassed": cpu_gate_passed,
        "agreementSummary": {
            "caseCount": int(len(agreement_df)),
            "agreementCount": int(agreement_df["agreement"].sum()),
            "allAgreement": bool(agreement_df["agreement"].all()),
        },
        "benchmarkSummary": cached_benchmark,
        "repoUnitTests": repo_test_payload,
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

    checksummed_artifacts = collect_artifacts(artifacts_written)
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
        "checksummedArtifactCount": len(checksummed_artifacts),
        "artifacts": checksummed_artifacts,
        "manifestSelfReference": {
            "path": str(manifest_path),
            "sha256": "omitted_self_referential_manifest",
        },
        "git": git_metadata,
        "batchSimulatorVersion": BATCH_SIMULATOR_VERSION,
        "dslVersion": DSL_VERSION,
    }
    write_json(manifest_path, manifest_payload)
    artifacts_written.append(manifest_path)

    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation_passed,
                "supportedPolicyCount": support_summary["supportedPolicyCount"],
                "agreementCases": len(agreement_df),
                "validationResult": validation_result,
                "statusPath": str(status_path),
                "manifestPath": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
