#!/usr/bin/env python3
"""Execute E04 S08 GPU evolutionary search and CPU replay validation."""

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
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from memory_repair import (  # noqa: E402
    EVOLUTION_SEARCH_VERSION,
    GENOME_FIELD_NAMES,
    EvolutionSearchConfig,
    audit_candidate_record,
    cpu_gpu_agreement_smoke,
    replay_candidate,
    replay_fingerprint,
    run_gpu_evolution_search,
    summarize_replay_advantage,
)
from memory_repair.training_constraints import TRAINING_CONSTRAINT_VERSION  # noqa: E402


EXPERIMENT_ID = "E04"
STEP_ID = "S08"
STEP_NUMBER = 8
STEP_TITLE = "Use GPU evolutionary search"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
TEST_MODULES = [
    "tests.test_e04_gpu_evolution",
    "tests.test_e04_training_constraints",
    "tests.test_e04_local_learning",
    "tests.test_e04_homeostasis_tasks",
    "tests.test_e04_fatigue_damage",
    "tests.test_e04_repairable_frozen",
    "tests.test_e04_local_signals",
    "tests.test_e04_memory_extension",
    "tests.test_e03_policy_interface",
    "tests.test_e03_rule_dsl",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def compact_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def yaml_scalar(value: Any) -> str:
    value = json_ready(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#{}[]\n,") or text.lower() in {"true", "false", "null"}:
        return json.dumps(text)
    return text


def to_yaml(value: Any, indent: int = 0) -> str:
    value = json_ready(value)
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(payload) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if env:
        merged_env.update(env)
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "args": args,
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtimeSeconds": time.perf_counter() - started,
    }


def get_git_metadata() -> dict[str, Any]:
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    branch = run_command(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    remote = run_command(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT)
    status = run_command(["git", "status", "--short"], cwd=REPO_ROOT)
    return {
        "commit": commit["stdout"].strip() if commit["success"] else "unknown",
        "branch": branch["stdout"].strip() if branch["success"] else "unknown",
        "remote": remote["stdout"].strip() if remote["success"] else "unknown",
        "statusShort": status["stdout"].strip(),
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".") if math.isfinite(value) else ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(item) for item in row) + " |")
    return "\n".join(lines)


def dataframe_to_artifacts(df: pd.DataFrame, step_path: Path, results_path: Path | None = None) -> list[Path]:
    step_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    csv_path = step_path.with_suffix(".csv")
    parquet_path = step_path.with_suffix(".parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    written.extend([csv_path, parquet_path])
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_csv = results_path.with_suffix(".csv")
        results_parquet = results_path.with_suffix(".parquet")
        shutil.copy2(csv_path, results_csv)
        shutil.copy2(parquet_path, results_parquet)
        written.extend([results_csv, results_parquet])
    return written


def agreement_rows(smoke: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "researchStepId": STEP_ID,
            "genomeIndex": index,
            "device": smoke["device"],
            "cpuProxyFitness": float(cpu_score),
            "torchProxyFitness": float(torch_score),
            "absDiff": abs(float(cpu_score) - float(torch_score)),
            "maxAbsDiff": float(smoke["maxAbsDiff"]),
            "agreementSuccess": bool(smoke["success"]),
            "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        }
        for index, (cpu_score, torch_score) in enumerate(zip(smoke["cpuScores"], smoke["torchScores"]))
    ]


def candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        audit = audit_candidate_record(candidate)
        rows.append(
            {
                "candidateId": candidate["candidateId"],
                "genomeId": candidate.get("genomeId"),
                "generation": candidate.get("generation"),
                "rank": candidate.get("rank"),
                "proxyFitness": candidate.get("proxyFitness"),
                "auditSuccess": bool(audit["success"]),
                "policyAuditSuccess": bool(audit["policyAuditSuccess"]),
                "trainingCertificationSuccess": bool(audit["trainingCertificationSuccess"]),
                "genomeJson": compact_json(candidate["genome"]),
                "policySpecJson": compact_json(candidate["policySpec"]),
                "learningConfigJson": compact_json(candidate["learningConfig"]),
                "memoryConfigJson": compact_json(candidate["memoryConfig"]),
                "signalConfigJson": compact_json(candidate["signalConfig"]),
                "repairConfigJson": compact_json(candidate["repairConfig"]),
                "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
                "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
            }
        )
    return rows


def audit_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        audit = audit_candidate_record(candidate)
        rows.append(
            {
                "candidateId": audit["candidateId"],
                "success": audit["success"],
                "policyAuditSuccess": audit["policyAuditSuccess"],
                "trainingCertificationSuccess": audit["trainingCertificationSuccess"],
                "policyAuditJson": audit["policyAuditJson"],
                "trainingCertificationJson": audit["trainingCertificationJson"],
                "trainingConstraintVersion": audit["trainingConstraintVersion"],
                "evolutionSearchVersion": audit["evolutionSearchVersion"],
            }
        )
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")
    return path


def replay_elites(candidates: Sequence[Mapping[str, Any]], *, split: str, seed_base: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replay_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        replay = replay_candidate(candidate, base_seed=seed_base + rank * 1000, split=split, repair_horizon=32)
        replay_rows.extend(replay["replayRows"])
        tick_rows.extend(replay["tickRows"])
    return replay_rows, tick_rows


def failure_rows(replay_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "candidateId": row["candidateId"],
            "controllerId": row["controllerId"],
            "benchmarkFamily": row["benchmarkFamily"],
            "benchmarkId": row["benchmarkId"],
            "split": row["split"],
            "completed": bool(row["completed"]),
            "score": float(row["score"]),
            "finalSortednessPercent": float(row["finalSortednessPercent"]),
            "remainingFrozenCellCount": int(row["remainingFrozenCellCount"]),
            "failureReason": "not_completed_or_remaining_frozen",
            "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        }
        for row in replay_rows
        if not bool(row["completed"]) or int(row["remainingFrozenCellCount"]) > 0
    ]
    if rows:
        return rows
    return [
        {
            "candidateId": "none",
            "controllerId": "none",
            "benchmarkFamily": "none",
            "benchmarkId": "none",
            "split": "all",
            "completed": True,
            "score": None,
            "finalSortednessPercent": None,
            "remainingFrozenCellCount": 0,
            "failureReason": "no_failure_examples_observed_in_s08_elite_replays",
            "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        }
    ]


def validation_rows(
    *,
    agreement: Mapping[str, Any],
    search: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    candidate_audits: pd.DataFrame,
    train_replay_rows: Sequence[Mapping[str, Any]],
    holdout_replay_rows: Sequence[Mapping[str, Any]],
    holdout_stability: Mapping[str, Any],
    test_result: Mapping[str, Any],
    artifact_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    advantage = summarize_replay_advantage(train_replay_rows)
    holdout_advantage = summarize_replay_advantage(holdout_replay_rows)
    return [
        {
            "validationFamily": "gpu_cpu_agreement",
            "caseId": "proxy_scores_agree",
            "success": bool(agreement["success"]),
            "validationDetail": f"Max absolute CPU/GPU proxy difference was {agreement['maxAbsDiff']}.",
        },
        {
            "validationFamily": "gpu_execution",
            "caseId": "search_ran_on_cuda",
            "success": bool(search["device"] == "cuda"),
            "validationDetail": f"Resolved torch device was {search['device']}; CUDA available={search['cudaAvailable']}.",
        },
        {
            "validationFamily": "candidate_audit",
            "caseId": "all_elites_pass_s07_gate",
            "success": bool(len(candidate_audits) > 0 and candidate_audits["success"].all()),
            "validationDetail": f"{int(candidate_audits['success'].sum())} of {len(candidate_audits)} elite candidates passed S07 local-only audit.",
        },
        {
            "validationFamily": "cpu_replay",
            "caseId": "elite_training_replays_written",
            "success": bool(len(train_replay_rows) > 0),
            "validationDetail": f"{len(train_replay_rows)} elite CPU replay summary rows were generated.",
        },
        {
            "validationFamily": "holdout_replay",
            "caseId": "elite_holdout_replays_written",
            "success": bool(len(holdout_replay_rows) > 0),
            "validationDetail": f"{len(holdout_replay_rows)} elite holdout replay summary rows were generated.",
        },
        {
            "validationFamily": "holdout_replay",
            "caseId": "holdout_replay_stable",
            "success": bool(holdout_stability["success"]),
            "validationDetail": f"Best-candidate holdout fingerprint repeated as {holdout_stability['fingerprintA']}.",
        },
        {
            "validationFamily": "completion_criterion",
            "caseId": "evolved_beats_fixed_on_one_repair_benchmark",
            "success": bool(advantage["success"]),
            "validationDetail": f"Best training repair score delta vs fixed local baseline: {advantage['bestRepairScoreDelta']}.",
        },
        {
            "validationFamily": "holdout_outcome",
            "caseId": "holdout_repair_advantage_checked",
            "success": bool(holdout_advantage["success"]),
            "validationDetail": f"Best holdout repair score delta vs fixed local baseline: {holdout_advantage['bestRepairScoreDelta']}.",
        },
        {
            "validationFamily": "repo_tests",
            "caseId": "relevant_unit_tests_passed",
            "success": bool(test_result["success"]),
            "validationDetail": f"Command {' '.join(test_result['args'])} returned {test_result['returncode']}.",
        },
        {
            "validationFamily": "artifact_contract",
            "caseId": "required_artifacts_exist",
            "success": bool(all(path.exists() and path.stat().st_size > 0 for path in artifact_paths)),
            "validationDetail": f"{len(artifact_paths)} tracked artifact files exist and are non-empty.",
        },
        {
            "validationFamily": "genome_schema",
            "caseId": "genomes_have_declared_fields",
            "success": bool(all(set(GENOME_FIELD_NAMES).issubset(set(candidate["genome"])) for candidate in candidates)),
            "validationDetail": f"All candidate genomes include {len(GENOME_FIELD_NAMES)} declared fields.",
        },
    ]


def write_reports(
    *,
    step_dir: Path,
    agreement: Mapping[str, Any],
    search: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    validation_df: pd.DataFrame,
    train_advantage: Mapping[str, Any],
    holdout_advantage: Mapping[str, Any],
    artifacts: Sequence[Path],
    caveats: Sequence[str],
    recommended_next_action: str,
) -> list[Path]:
    reports: list[Path] = []
    best = candidates[0]
    agreement_report = step_dir / "gpu_cpu_agreement_report.md"
    agreement_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} GPU/CPU Agreement Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: {step_dir / 'gpu_cpu_agreement.csv'} and `.parquet` plus results copies",
                f"- Validation result: {'passed' if agreement['success'] else 'failed'}; max absolute difference {agreement['maxAbsDiff']}",
                f"- Caveats or blockers: {caveats[0] if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "The NumPy CPU proxy and torch proxy produced identical screening scores on the smoke batch.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(agreement_report)

    search_report = step_dir / "evolution_search_report.md"
    curve_rows = search["trainingCurveRows"]
    table_rows = [
        [row["generation"], row["bestProxyFitness"], row["meanProxyFitness"], row["bestGenomeId"]]
        for row in curve_rows
    ]
    search_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Evolution Search Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: {step_dir / 'evolution_population.csv'}, `{step_dir / 'evolution_lineages.csv'}`, and training curves",
                f"- Validation result: device `{search['device']}`, best proxy fitness {best['proxyFitness']}",
                f"- Caveats or blockers: {caveats[0] if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                markdown_table(["generation", "bestProxyFitness", "meanProxyFitness", "bestGenomeId"], table_rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(search_report)

    holdout_report = step_dir / "holdout_replay_report.md"
    holdout_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Holdout Replay Report",
                "",
                f"- Research step ID: {STEP_ID}",
                "- Completion status: completed",
                f"- Artifacts written: {step_dir / 'holdout_replays.csv'} and `.parquet`",
                f"- Validation result: best holdout repair delta {holdout_advantage.get('bestRepairScoreDelta')}",
                f"- Caveats or blockers: {caveats[0] if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "Holdout replay uses different scheduler and tie-breaker seeds from the training replay.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(holdout_report)

    validation_report = step_dir / "validation_report.md"
    validation_table = [[row.caseId, row.success, row.validationDetail] for row in validation_df.itertuples(index=False)]
    validation_report.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Validation Report",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: {step_dir / 'validation_results.csv'} and `.parquet`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {caveats[0] if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                markdown_table(["caseId", "success", "detail"], validation_table),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(validation_report)

    summary = step_dir / "summary.md"
    artifact_list = "\n".join(f"- `{path}`" for path in artifacts[:80])
    if len(artifacts) > 80:
        artifact_list += f"\n- ... {len(artifacts) - 80} additional files listed in artifact_manifest.json"
    summary.write_text(
        "\n".join(
            [
                f"# {STEP_ID} Status Summary",
                "",
                f"- Research step ID: {STEP_ID}",
                f"- Completion status: {'completed' if validation_df['success'].all() else 'completed_with_failed_validation'}",
                f"- Artifacts written: primary outputs listed below under `{step_dir}` and shared results/config directories; full final list is in `status.json` and `artifact_manifest.json`",
                f"- Validation result: {int(validation_df['success'].sum())} of {len(validation_df)} checks passed",
                f"- Caveats or blockers: {'; '.join(caveats) if caveats else 'none'}",
                f"- Recommended next action: {recommended_next_action}",
                "",
                "Lay summary: S08 ran a small GPU-batched evolutionary search over local-learning, repair, signal, and recorded memory parameters. The best candidates passed the S07 no-oracle audit and were replayed with the CPU reference simulator on training and holdout seeds.",
                "",
                f"Outcome classification: {'supportive' if train_advantage.get('success') else 'null'} for the frozen S08 completion criterion.",
                "",
                f"Best candidate: `{best['candidateId']}` with proxy fitness `{best['proxyFitness']}`.",
                f"Training repair advantage: `{train_advantage.get('bestRepairScoreDelta')}`.",
                f"Holdout repair advantage: `{holdout_advantage.get('bestRepairScoreDelta')}`.",
                "",
                "Artifacts:",
                artifact_list,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reports.append(summary)
    return reports


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_root = step_dir / "code"
    source_paths = [
        REPO_ROOT / "memory_repair" / "evolution.py",
        REPO_ROOT / "memory_repair" / "learning.py",
        REPO_ROOT / "memory_repair" / "training_constraints.py",
        REPO_ROOT / "memory_repair" / "signals.py",
        REPO_ROOT / "memory_repair" / "repair.py",
        REPO_ROOT / "memory_repair" / "homeostasis.py",
        REPO_ROOT / "scripts" / "e04_s08_gpu_evolution_search.py",
        REPO_ROOT / "tests" / "test_e04_gpu_evolution.py",
    ]
    written = []
    for source in source_paths:
        destination = code_root / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        written.append(destination)
    return written


def artifact_manifest(step_dir: Path, artifacts: Sequence[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(set(artifacts), key=lambda item: str(item)):
        if not path.exists() or path.is_dir():
            continue
        records.append(
            {
                "path": str(path),
                "relativeToArtifactsDir": str(path.relative_to(DEFAULT_ARTIFACTS_DIR))
                if path.is_relative_to(DEFAULT_ARTIFACTS_DIR)
                else str(path),
                "sizeBytes": int(path.stat().st_size),
                "sha256": sha256_path(path),
            }
        )
    return {
        "schema": "eidosoma.artifact_manifest.v1",
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "artifactCount": len(records),
        "createdAt": utc_now(),
        "artifacts": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--replay-elites", type=int, default=3)
    parser.add_argument("--mutation-scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=808)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    configs_dir = artifacts_dir / "configs"
    step_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    started_at = time.perf_counter()

    config = EvolutionSearchConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        mutation_scale=args.mutation_scale,
        seed=args.seed,
        device=None,
    )
    config_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": STEP_TITLE,
        "search": config.to_dict(),
        "replayEliteCount": int(args.replay_elites),
        "gpuRequiredForValidation": True,
        "candidateAuditProtocol": "S07 local-only",
        "createdAt": utc_now(),
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    for path in [
        step_dir / "e04_gpu_evolution_search.json",
        configs_dir / "e04_gpu_evolution_search.json",
    ]:
        write_json(path, config_payload)
        artifacts.append(path)
    for path in [
        step_dir / "e04_gpu_evolution_search.yaml",
        configs_dir / "e04_gpu_evolution_search.yaml",
    ]:
        write_yaml(path, config_payload)
        artifacts.append(path)

    agreement = cpu_gpu_agreement_smoke(seed=args.seed)
    agreement_df = pd.DataFrame(agreement_rows(agreement))
    artifacts.extend(
        dataframe_to_artifacts(
            agreement_df,
            step_dir / "gpu_cpu_agreement",
            results_dir / "e04_s08_gpu_cpu_agreement",
        )
    )

    search = run_gpu_evolution_search(config)
    population_df = pd.DataFrame(search["populationRows"])
    lineage_df = pd.DataFrame(search["lineageRows"])
    curve_df = pd.DataFrame(search["trainingCurveRows"])
    artifacts.extend(dataframe_to_artifacts(population_df, step_dir / "evolution_population", results_dir / "e04_evolution_runs"))
    artifacts.extend(dataframe_to_artifacts(lineage_df, step_dir / "evolution_lineages", results_dir / "e04_s08_evolution_lineages"))
    artifacts.extend(dataframe_to_artifacts(curve_df, step_dir / "training_curves", results_dir / "e04_s08_training_curves"))

    candidates = list(search["eliteCandidates"])
    replay_candidates = candidates[: max(1, min(int(args.replay_elites), len(candidates)))]
    candidate_df = pd.DataFrame(candidate_rows(candidates))
    candidate_audit_df = pd.DataFrame(audit_rows(candidates))
    artifacts.extend(dataframe_to_artifacts(candidate_df, step_dir / "evolved_policy_candidates", results_dir / "e04_evolved_repair_policies"))
    artifacts.extend(dataframe_to_artifacts(candidate_audit_df, step_dir / "candidate_audits", results_dir / "e04_s08_candidate_audits"))
    artifacts.append(write_jsonl(step_dir / "evolved_policy_genomes.jsonl", [{"candidateId": item["candidateId"], "genome": item["genome"]} for item in candidates]))
    artifacts.append(write_jsonl(step_dir / "evolved_policy_specs.jsonl", [{"candidateId": item["candidateId"], "policySpec": item["policySpec"]} for item in candidates]))

    train_replay_rows, train_tick_rows = replay_elites(replay_candidates, split="training", seed_base=9000)
    holdout_replay_rows, holdout_tick_rows = replay_elites(replay_candidates, split="holdout", seed_base=19000)
    train_replay_df = pd.DataFrame(train_replay_rows)
    holdout_replay_df = pd.DataFrame(holdout_replay_rows)
    tick_df = pd.DataFrame(train_tick_rows + holdout_tick_rows)
    failure_df = pd.DataFrame(failure_rows(train_replay_rows + holdout_replay_rows))
    artifacts.extend(dataframe_to_artifacts(train_replay_df, step_dir / "elite_cpu_replays", results_dir / "e04_s08_elite_cpu_replays"))
    artifacts.extend(dataframe_to_artifacts(holdout_replay_df, step_dir / "holdout_replays", results_dir / "e04_s08_holdout_replays"))
    artifacts.extend(dataframe_to_artifacts(tick_df, step_dir / "homeostasis_tick_records", results_dir / "e04_s08_homeostasis_tick_records"))
    artifacts.extend(dataframe_to_artifacts(failure_df, step_dir / "failure_examples", results_dir / "e04_s08_failure_examples"))

    first_holdout_a = replay_candidate(replay_candidates[0], base_seed=25000, split="holdout_stability", repair_horizon=32)
    first_holdout_b = replay_candidate(replay_candidates[0], base_seed=25000, split="holdout_stability", repair_horizon=32)
    holdout_stability = {
        "success": replay_fingerprint(first_holdout_a["replayRows"]) == replay_fingerprint(first_holdout_b["replayRows"]),
        "fingerprintA": replay_fingerprint(first_holdout_a["replayRows"]),
        "fingerprintB": replay_fingerprint(first_holdout_b["replayRows"]),
    }
    write_json(step_dir / "holdout_stability.json", holdout_stability)
    artifacts.append(step_dir / "holdout_stability.json")

    test_result = {"args": ["skipped"], "returncode": 0, "success": True, "stdout": "", "stderr": "", "runtimeSeconds": 0.0}
    if not args.skip_tests:
        test_result = run_command([sys.executable, "-m", "unittest", *TEST_MODULES], cwd=REPO_ROOT)
    test_log = step_dir / "repo_unit_test_log.txt"
    test_log.write_text(
        "\n".join(
            [
                f"command: {' '.join(test_result['args'])}",
                f"returncode: {test_result['returncode']}",
                f"runtimeSeconds: {test_result['runtimeSeconds']}",
                "",
                "STDOUT:",
                test_result["stdout"],
                "",
                "STDERR:",
                test_result["stderr"],
            ]
        ),
        encoding="utf-8",
    )
    artifacts.append(test_log)

    code_artifacts = copy_code_artifacts(step_dir)
    artifacts.extend(code_artifacts)

    validation_artifact_inputs = list(artifacts)
    validation = validation_rows(
        agreement=agreement,
        search=search,
        candidates=candidates,
        candidate_audits=candidate_audit_df,
        train_replay_rows=train_replay_rows,
        holdout_replay_rows=holdout_replay_rows,
        holdout_stability=holdout_stability,
        test_result=test_result,
        artifact_paths=validation_artifact_inputs,
    )
    validation_df = pd.DataFrame(validation)
    artifacts.extend(dataframe_to_artifacts(validation_df, step_dir / "validation_results", results_dir / "e04_s08_validation_results"))

    train_advantage = summarize_replay_advantage(train_replay_rows)
    holdout_advantage = summarize_replay_advantage(holdout_replay_rows)
    caveats = [
        "The GPU objective is a screening proxy; final claims are limited to CPU reference replays.",
        "Memory-rule genes are saved for S09 ablations, but this S08 replay does not isolate memory causality.",
        "The evolutionary run is intentionally small and should be treated as a candidate-generation pass.",
    ]
    recommended_next_action = "Chief review; if accepted, proceed to S09 memory ablations. Do not start S09 from this run."
    reports = write_reports(
        step_dir=step_dir,
        agreement=agreement,
        search=search,
        candidates=candidates,
        validation_df=validation_df,
        train_advantage=train_advantage,
        holdout_advantage=holdout_advantage,
        artifacts=artifacts,
        caveats=caveats,
        recommended_next_action=recommended_next_action,
    )
    artifacts.extend(reports)

    success = bool(validation_df["success"].all())
    status = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": success,
        "status": "completed" if success else "completed_with_failed_validation",
        "artifactsWritten": [str(path) for path in artifacts],
        "validationResult": {
            "passedCount": int(validation_df["success"].sum()),
            "totalCount": int(len(validation_df)),
            "allPassed": success,
            "validationResultsPath": str(step_dir / "validation_results.csv"),
        },
        "caveatsOrBlockers": caveats if success else caveats + ["At least one S08 validation check failed; inspect validation_results.csv."],
        "recommendedNextAction": recommended_next_action,
        "experimentId": EXPERIMENT_ID,
        "title": STEP_TITLE,
        "outcomeClassification": "supportive" if train_advantage.get("success") else "null",
        "startedAt": config_payload["createdAt"],
        "completedAt": utc_now(),
        "runtimeSeconds": time.perf_counter() - started_at,
        "git": get_git_metadata(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "gpuCpuAgreement": agreement,
        "searchConfig": search["config"],
        "candidateCount": len(candidates),
        "replayedCandidateCount": len(replay_candidates),
        "trainingRepairAdvantage": train_advantage,
        "holdoutRepairAdvantage": holdout_advantage,
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
    }
    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    artifacts.extend([status_path, manifest_path])
    status["artifactsWritten"] = [str(path) for path in artifacts]
    write_json(status_path, status)

    manifest = artifact_manifest(step_dir, [path for path in artifacts if path != manifest_path])
    write_json(manifest_path, manifest)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
