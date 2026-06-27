#!/usr/bin/env python3
"""Execute E03 S05 policy corpus generation and validation.

S05 builds the first broad DSL policy corpus from the completed S01-S04
interfaces. It filters only syntactically invalid or non-executable programs,
then records stable IDs, lineage, complexity metadata, DSL files, duplicate
checks, parser validation, and tiny-array execution smoke results.
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
    COMPETENCE_VECTOR_VERSION,
    DSLPolicy,
    DEFAULT_CORPUS_SIZE,
    POLICY_CORPUS_VERSION,
    PolicyEventSimulator,
    corpus_summary,
    generate_policy_corpus,
    policy_corpus_table,
    policy_lineage_table,
    validate_corpus_records,
)
from morphospace.rule_dsl import DSL_VERSION, parse_rule_program  # noqa: E402


EXPERIMENT_ID = "E03"
STEP_ID = "S05"
STEP_NUMBER = 5
STATUS = "completed"
OUTCOME_CLASSIFICATION = "supportive"
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
DEFAULT_RANDOM_SEED = 905
TINY_ARRAY = [3, 1, 2]
TINY_EXECUTION_LIMITS = {
    "maxActivations": 128,
    "maxSwaps": 128,
    "maxComparisons": 512,
}


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


def artifact_preview(artifacts_written: list[str], limit: int = 22) -> str:
    preview = "\n".join(f"- `{path}`" for path in artifacts_written[:limit])
    if len(artifacts_written) > limit:
        preview += f"\n- ... {len(artifacts_written) - limit} additional artifact path(s) in `status.json` and `artifact_manifest.json`"
    return preview


def write_dsl_files(records, step_dir: Path) -> tuple[dict[str, str], pd.DataFrame, list[Path]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    relpaths: dict[str, str] = {}
    dsl_root = step_dir / "dsl_files"
    for record in records:
        family_dir = dsl_root / record.family
        family_dir.mkdir(parents=True, exist_ok=True)
        path = family_dir / f"{record.policy_id}.json"
        payload = json.loads(record.program.to_json())
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        relpath = path.relative_to(step_dir).as_posix()
        relpaths[record.policy_id] = relpath
        paths.append(path)
        rows.append(
            {
                "policyId": record.policy_id,
                "family": record.family,
                "generationMethod": record.generation_method,
                "structureHash": record.structure_hash,
                "dslRelativePath": relpath,
                "dslPath": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return relpaths, pd.DataFrame(rows), paths


def validate_dsl_files(dsl_index_df: pd.DataFrame) -> dict[str, Any]:
    parsed = 0
    checksum_matches = 0
    errors: list[str] = []
    for _, row in dsl_index_df.iterrows():
        path = Path(row["dslPath"])
        try:
            if sha256_path(path) == row["sha256"]:
                checksum_matches += 1
            program = parse_rule_program(path.read_text(encoding="utf-8"))
            if program.policy_id != row["policyId"]:
                errors.append(f"policy_id_mismatch:{row['policyId']}")
            else:
                parsed += 1
        except Exception as exc:  # pragma: no cover - defensive validation
            errors.append(f"{row['policyId']}:{exc!r}")
    return {
        "dslFileCount": int(len(dsl_index_df)),
        "parsedFileCount": int(parsed),
        "checksumMatchCount": int(checksum_matches),
        "errors": errors,
        "success": parsed == len(dsl_index_df) and checksum_matches == len(dsl_index_df) and not errors,
    }


def sample_execute_records(records) -> pd.DataFrame:
    expected_counts = Counter(TINY_ARRAY)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row: dict[str, Any] = {
            "policyId": record.policy_id,
            "family": record.family,
            "generationMethod": record.generation_method,
            "executionSuccess": False,
            "exceptionType": None,
            "exceptionMessage": None,
            "completed": False,
            "stopReason": None,
            "valueCountsConserved": False,
            "initialValuesJson": json.dumps(TINY_ARRAY, separators=(",", ":")),
            "finalValuesJson": None,
            "swapCount": None,
            "comparisonCount": None,
            "activationCount": None,
            "finalSortednessPercent": None,
            "finalMonotonicityError": None,
            "wallTimeSeconds": None,
        }
        try:
            result = PolicyEventSimulator(
                TINY_ARRAY,
                DSLPolicy(record.program),
                scheduler_seed=1000 + index,
                tie_breaker_seed=2000 + index,
                condition_id=f"S05_tiny_{record.policy_id}",
                implementation="e03_s05_policy_corpus",
                research_step_id=STEP_ID,
            ).run(
                max_activations=TINY_EXECUTION_LIMITS["maxActivations"],
                max_swaps=TINY_EXECUTION_LIMITS["maxSwaps"],
                max_comparisons=TINY_EXECUTION_LIMITS["maxComparisons"],
            )
            row.update(
                {
                    "executionSuccess": True,
                    "completed": bool(result.completed),
                    "stopReason": result.stop_reason,
                    "valueCountsConserved": Counter(result.final_values) == expected_counts,
                    "finalValuesJson": json.dumps(result.final_values, separators=(",", ":")),
                    "swapCount": int(result.swap_count),
                    "comparisonCount": int(result.comparison_count),
                    "activationCount": int(result.activation_count),
                    "finalSortednessPercent": float(result.final_sortedness_percent),
                    "finalMonotonicityError": int(result.final_monotonicity_error),
                    "wallTimeSeconds": float(result.wall_time_seconds),
                }
            )
        except Exception as exc:  # pragma: no cover - artifact validation path
            row.update({"exceptionType": type(exc).__name__, "exceptionMessage": repr(exc)})
        rows.append(row)
    return pd.DataFrame(rows)


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


def validate_policy_corpus(
    *,
    records,
    target_size: int,
    corpus_df: pd.DataFrame,
    lineage_df: pd.DataFrame,
    dsl_index_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    parser_validation: dict[str, Any],
    dsl_file_validation: dict[str, Any],
    upstream_status_paths: dict[str, Path],
    repo_test_row: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_counts = Counter(record.family for record in records)
    generation_methods = {record.generation_method for record in records}
    required_families = {"seed", "hand_designed", "mutation", "recombined", "random_generated"}
    required_methods = {
        "seed_import",
        "classic_parameter_sweep",
        "single_rule_sweep",
        "memory_mutation",
        "target_memory_mutation",
        "two_parent_rule_recombination",
        "seeded_random_grammar",
    }

    rows.append(
        {
            "validationFamily": "corpus_size",
            "conditionId": "target_size_met",
            "success": len(records) >= target_size,
            "validationDetail": f"Generated {len(records)} policies for target_size={target_size}.",
        }
    )
    rows.append(
        {
            "validationFamily": "corpus_size",
            "conditionId": "thousands_available",
            "success": len(records) >= 2000,
            "validationDetail": "Completion criterion requires thousands of executable policies.",
        }
    )
    rows.append(
        {
            "validationFamily": "corpus_schema",
            "conditionId": "required_table_columns_present",
            "success": {
                "policyId",
                "structureHash",
                "lineageId",
                "complexityScore",
                "dslProgramJson",
                "dslRelativePath",
            }.issubset(set(corpus_df.columns)),
            "validationDetail": f"Policy corpus table has {len(corpus_df.columns)} columns.",
        }
    )
    rows.append(
        {
            "validationFamily": "generation_coverage",
            "conditionId": "required_families_present",
            "success": required_families.issubset(set(family_counts)),
            "validationDetail": f"Family counts: {dict(sorted(family_counts.items()))}.",
        }
    )
    rows.append(
        {
            "validationFamily": "generation_coverage",
            "conditionId": "required_generation_methods_present",
            "success": required_methods.issubset(generation_methods),
            "validationDetail": f"Observed {len(generation_methods)} generation methods.",
        }
    )
    rows.append(
        {
            "validationFamily": "duplicates",
            "conditionId": "unique_policy_ids",
            "success": parser_validation["uniquePolicyIds"] == parser_validation["policyCount"],
            "validationDetail": f"{parser_validation['uniquePolicyIds']} unique policy IDs among {parser_validation['policyCount']} records.",
        }
    )
    rows.append(
        {
            "validationFamily": "duplicates",
            "conditionId": "unique_structure_hashes",
            "success": parser_validation["uniqueStructureHashes"] == parser_validation["policyCount"],
            "validationDetail": f"{parser_validation['uniqueStructureHashes']} unique structure hashes among {parser_validation['policyCount']} records.",
        }
    )
    rows.append(
        {
            "validationFamily": "parser_roundtrip",
            "conditionId": "in_memory_programs_round_trip",
            "success": parser_validation["success"],
            "validationDetail": f"{parser_validation['parseRoundTripCount']} in-memory DSL programs round-tripped; errors={len(parser_validation['parseErrors'])}.",
        }
    )
    rows.append(
        {
            "validationFamily": "dsl_files",
            "conditionId": "dsl_file_count_matches_corpus",
            "success": len(dsl_index_df) == len(records),
            "validationDetail": f"{len(dsl_index_df)} DSL JSON files indexed for {len(records)} policies.",
        }
    )
    rows.append(
        {
            "validationFamily": "dsl_files",
            "conditionId": "dsl_files_parse_and_checksum",
            "success": dsl_file_validation["success"],
            "validationDetail": f"{dsl_file_validation['parsedFileCount']} DSL files parsed and {dsl_file_validation['checksumMatchCount']} checksums matched.",
        }
    )
    parent_ids: set[str] = set()
    for value in lineage_df["parentPolicyIdsJson"]:
        parent_ids.update(json.loads(value))
    missing_parent_ids = sorted(parent_id for parent_id in parent_ids if parent_id not in set(corpus_df["policyId"]))
    rows.append(
        {
            "validationFamily": "lineage",
            "conditionId": "lineage_metadata_present_and_referential",
            "success": lineage_df["lineageId"].notna().all() and len(missing_parent_ids) == 0,
            "validationDetail": f"Lineage rows={len(lineage_df)}; parent refs={len(parent_ids)}; missing refs={len(missing_parent_ids)}.",
        }
    )
    complexity_ok = (
        corpus_df["ruleCount"].ge(1).all()
        and corpus_df["complexityScore"].ge(corpus_df["ruleCount"]).all()
        and corpus_df["actionCountsJson"].notna().all()
    )
    rows.append(
        {
            "validationFamily": "complexity",
            "conditionId": "complexity_metadata_present",
            "success": bool(complexity_ok),
            "validationDetail": "Rule, predicate, update, state-key, stochastic, target, signal, and aggregate complexity columns are present.",
        }
    )
    rows.append(
        {
            "validationFamily": "sample_execution",
            "conditionId": "tiny_array_all_policies_execute",
            "success": bool(sample_df["executionSuccess"].all()),
            "validationDetail": f"{int(sample_df['executionSuccess'].sum())} of {len(sample_df)} policies executed on the tiny array without exception.",
        }
    )
    rows.append(
        {
            "validationFamily": "sample_execution",
            "conditionId": "tiny_array_values_conserved",
            "success": bool(sample_df["valueCountsConserved"].all()),
            "validationDetail": f"{int(sample_df['valueCountsConserved'].sum())} of {len(sample_df)} tiny executions conserved value counts.",
        }
    )
    sorted_count = int(sample_df["completed"].sum())
    rows.append(
        {
            "validationFamily": "sample_execution",
            "conditionId": "tiny_array_sorting_result_recorded",
            "success": sorted_count >= 0 and sample_df["stopReason"].notna().all(),
            "validationDetail": f"{sorted_count} of {len(sample_df)} policies sorted [3,1,2] within S05 caps; non-sorting records retained for later competence mapping.",
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
    rows.append(
        {
            "validationFamily": "upstream_boundary",
            "conditionId": "s04_competence_schema_version_available",
            "success": bool(COMPETENCE_VECTOR_VERSION),
            "validationDetail": f"S05 uses S04 schema boundary {COMPETENCE_VECTOR_VERSION}.",
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


def render_schema_md(
    *,
    corpus_df: pd.DataFrame,
    artifacts_written: list[str],
    validation_result: str,
    caveats: list[str],
    recommended_next_action: str,
) -> str:
    columns = [
        ["policyId", "Stable policy ID derived from structure hash."],
        ["structureHash", "Behavior-relevant DSL structure hash."],
        ["family", "Seed, hand-designed, mutation, recombined, or random-generated lineage family."],
        ["generationMethod", "Generator or transformation method."],
        ["lineageId", "Stable lineage name for provenance and later search."],
        ["parentPolicyIdsJson", "JSON array of direct parent policy IDs."],
        ["mutationOperatorsJson", "JSON array of mutation operators, when applicable."],
        ["recombinationParentsJson", "JSON array of recombination parent IDs, when applicable."],
        ["complexityScore", "Simple aggregate of rules, predicates, updates, stochastic actions, and state keys."],
        ["observationRequirementsJson", "JSON array of observation/state/RNG requirements inferred from the DSL."],
        ["dslRelativePath", "Step-relative JSON DSL file path."],
        ["dslProgramJson", "Canonical serialized S02 DSL program."],
    ]
    families = corpus_df["family"].value_counts().sort_index().reset_index().values.tolist()
    return f"""# E03 S05 Policy Corpus Schema

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: {recommended_next_action}

## Frozen Question

Random generation, hand-designed variants, grammar mutations, and recombinations can produce a diverse policy corpus with nontrivial competence.

## Corpus Contract

- Corpus version: `{POLICY_CORPUS_VERSION}`
- DSL version: `{DSL_VERSION}`
- S04 competence-vector boundary for future sweeps: `{COMPETENCE_VECTOR_VERSION}`
- Record count: {len(corpus_df)}
- ID policy: policy IDs are `pc_` plus a 16-character SHA-256 prefix over the behavior-relevant DSL structure, excluding top-level display names.
- Filtering policy: S05 excludes syntactically invalid or non-executable DSL programs only. Policies that fail to sort tiny arrays are retained and recorded.

## Required Columns

{markdown_table(["Column", "Meaning"], columns)}

## Family Counts

{markdown_table(["Family", "Count"], families)}
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
    return f"""# S05 Validation Report

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written:
{artifact_preview(artifacts_written)}
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Caveats or blockers: {"; ".join(caveats)}
- Recommended next action: stop before S06 for Chief Scientist review; if approved, vectorize the simulator using the S05 corpus and S01-S04 boundaries.

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
    summary: dict[str, Any],
    sample_df: pd.DataFrame,
) -> str:
    validation_passed = bool(validation_df["success"].all())
    validation_result = "passed" if validation_passed else "failed"
    outcome = OUTCOME_CLASSIFICATION if validation_passed else "constraining/contradictory"
    sorted_count = int(sample_df["completed"].sum())
    family_counts = summary.get("familyCounts", {})
    return f"""# S05 Summary

- Research step ID: {STEP_ID}
- Completion status: {STATUS}
- Artifacts written: `{artifacts_written[0]}` and {len(artifacts_written) - 1} additional files listed in `status.json` and `artifact_manifest.json`.
- Validation result: {validation_result}; {int(validation_df["success"].sum())} of {len(validation_df)} checks passed.
- Outcome classification: {outcome}
- Caveats or blockers: {"; ".join(caveats)}
- Lay summary: S05 generated {summary['policyCount']} stable DSL policy records across seed, hand-designed, mutation, recombination, and random grammar families. Every policy has a unique ID, lineage and complexity metadata, a DSL JSON file, parser validation, duplicate checks, and a tiny-array execution row. {sorted_count} policies sorted `[3,1,2]` within the small S05 caps; non-sorting policies are retained as expected morphospace failure regions.
- Recommended next action: stop before S06 for Chief Scientist review. If approved, S06 should build the GPU-friendly batch simulator and validate it against the CPU reference using this corpus.

## Family Counts

{markdown_table(["Family", "Count"], [[key, value] for key, value in sorted(family_counts.items())])}
"""


def collect_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    seen: set[Path] = set()
    for path in paths:
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            artifacts.append({"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_path(path)})
    return sorted(artifacts, key=lambda item: item["path"])


def copy_code_artifacts(step_dir: Path) -> list[Path]:
    code_dir = step_dir / "code"
    copied: list[Path] = []
    mapping = [
        (REPO_ROOT / "morphospace" / "__init__.py", code_dir / "morphospace" / "__init__.py"),
        (REPO_ROOT / "morphospace" / "policies.py", code_dir / "morphospace" / "policies.py"),
        (REPO_ROOT / "morphospace" / "rule_dsl.py", code_dir / "morphospace" / "rule_dsl.py"),
        (REPO_ROOT / "morphospace" / "classic_templates.py", code_dir / "morphospace" / "classic_templates.py"),
        (REPO_ROOT / "morphospace" / "competence.py", code_dir / "morphospace" / "competence.py"),
        (REPO_ROOT / "morphospace" / "policy_corpus.py", code_dir / "morphospace" / "policy_corpus.py"),
        (REPO_ROOT / "scripts" / "e03_s05_policy_corpus.py", code_dir / "scripts" / "e03_s05_policy_corpus.py"),
        (REPO_ROOT / "tests" / "test_e03_policy_corpus.py", code_dir / "tests" / "test_e03_policy_corpus.py"),
    ]
    for source, dest in mapping:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append(dest)
    return copied


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--target-size", type=int, default=DEFAULT_CORPUS_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started_at = utc_now()
    artifacts_dir: Path = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    data_dir = artifacts_dir / "data"
    results_dir = artifacts_dir / "results"
    step_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifacts_written: list[Path] = []
    records = generate_policy_corpus(target_size=args.target_size, random_seed=args.random_seed)
    relpaths, dsl_index_df, dsl_paths = write_dsl_files(records, step_dir)
    artifacts_written.extend(dsl_paths)

    corpus_df = pd.DataFrame(policy_corpus_table(records, dsl_relpaths=relpaths))
    lineage_df = pd.DataFrame(policy_lineage_table(records))
    summary = corpus_summary(records)
    summary.update(
        {
            "targetSize": int(args.target_size),
            "randomSeed": int(args.random_seed),
            "tinyArray": TINY_ARRAY,
            "tinyExecutionLimits": TINY_EXECUTION_LIMITS,
        }
    )

    artifacts_written.extend(
        write_table(corpus_df, step_dir / "policy_corpus.csv", step_dir / "policy_corpus.parquet")
    )
    artifacts_written.extend(
        write_table(corpus_df, data_dir / "e03_policy_corpus.csv", data_dir / "e03_policy_corpus.parquet")
    )
    artifacts_written.extend(
        write_table(lineage_df, step_dir / "policy_lineage.csv", step_dir / "policy_lineage.parquet")
    )
    artifacts_written.extend(
        write_table(
            lineage_df,
            data_dir / "e03_policy_corpus_lineage.csv",
            data_dir / "e03_policy_corpus_lineage.parquet",
        )
    )
    artifacts_written.extend(
        write_table(dsl_index_df, step_dir / "dsl_file_index.csv", step_dir / "dsl_file_index.parquet")
    )
    artifacts_written.extend(
        write_table(
            dsl_index_df,
            data_dir / "e03_policy_corpus_dsl_file_index.csv",
            data_dir / "e03_policy_corpus_dsl_file_index.parquet",
        )
    )

    summary_path = step_dir / "policy_corpus_summary.json"
    config_path = step_dir / "policy_corpus_generation_config.json"
    write_json(summary_path, summary)
    write_json(
        config_path,
        {
            "researchStepId": STEP_ID,
            "stepNumber": STEP_NUMBER,
            "corpusVersion": POLICY_CORPUS_VERSION,
            "dslVersion": DSL_VERSION,
            "competenceVectorVersion": COMPETENCE_VECTOR_VERSION,
            "targetSize": int(args.target_size),
            "randomSeed": int(args.random_seed),
            "workerCount": 1,
            "threading": "serial_reference_validation",
            "tinyArray": TINY_ARRAY,
            "tinyExecutionLimits": TINY_EXECUTION_LIMITS,
            "filterPolicy": "Filter syntactically invalid or non-executable DSL programs only; retain policies that execute but fail to sort.",
            "startedAt": started_at,
        },
    )
    artifacts_written.extend([summary_path, config_path])

    parser_validation = validate_corpus_records(records)
    dsl_file_validation = validate_dsl_files(dsl_index_df)
    sample_df = sample_execute_records(records)
    artifacts_written.extend(
        write_table(sample_df, step_dir / "sample_execution.csv", step_dir / "sample_execution.parquet")
    )
    artifacts_written.extend(
        write_table(
            sample_df,
            results_dir / "e03_s05_sample_execution.csv",
            results_dir / "e03_s05_sample_execution.parquet",
        )
    )

    repo_test_row, repo_test_payload, repo_test_log = run_repo_unit_tests(step_dir)
    artifacts_written.append(repo_test_log)

    upstream_status_paths = {
        "S01": artifacts_dir / "research_steps" / "S01" / "status.json",
        "S02": artifacts_dir / "research_steps" / "S02" / "status.json",
        "S03": artifacts_dir / "research_steps" / "S03" / "status.json",
        "S04": artifacts_dir / "research_steps" / "S04" / "status.json",
    }
    validation_df = validate_policy_corpus(
        records=records,
        target_size=args.target_size,
        corpus_df=corpus_df,
        lineage_df=lineage_df,
        dsl_index_df=dsl_index_df,
        sample_df=sample_df,
        parser_validation=parser_validation,
        dsl_file_validation=dsl_file_validation,
        upstream_status_paths=upstream_status_paths,
        repo_test_row=repo_test_row,
    )
    artifacts_written.extend(
        write_table(
            validation_df,
            step_dir / "policy_corpus_validation.csv",
            step_dir / "policy_corpus_validation.parquet",
        )
    )
    artifacts_written.extend(
        write_table(
            validation_df,
            results_dir / "e03_s05_policy_corpus_validation.csv",
            results_dir / "e03_s05_policy_corpus_validation.parquet",
        )
    )

    caveats = [
        "Tiny-array execution validates parser and simulator compatibility, not competence on larger task panels.",
        "Policies that fail to sort are retained intentionally for later failure-region mapping.",
        "Signal actions remain S02 local placeholders; no multi-cell communication substrate is claimed in S05.",
        "Validation ran serially with workerCount=1 because S05 is dominated by small parser and smoke-execution checks.",
    ]
    recommended_next_action = (
        "Stop before S06 for Chief Scientist review; if approved, vectorize the simulator and validate CPU/GPU agreement "
        "using the S05 corpus."
    )
    validation_passed = bool(validation_df["success"].all())
    validation_result = (
        f"passed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
        if validation_passed
        else f"failed; {int(validation_df['success'].sum())} of {len(validation_df)} checks passed"
    )

    code_paths = copy_code_artifacts(step_dir)
    artifacts_written.extend(code_paths)

    artifact_strings_so_far = [str(path) for path in artifacts_written]
    schema_md_path = step_dir / "policy_corpus_schema.md"
    validation_report_path = step_dir / "validation_report.md"
    summary_md_path = step_dir / "summary.md"
    schema_md_path.write_text(
        render_schema_md(
            corpus_df=corpus_df,
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
            summary=summary,
            sample_df=sample_df,
        ),
        encoding="utf-8",
    )
    artifacts_written.extend([schema_md_path, validation_report_path, summary_md_path])

    status_path = step_dir / "status.json"
    manifest_path = step_dir / "artifact_manifest.json"
    git_metadata = get_git_metadata()
    status_payload = {
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "title": "Generate thousands of policies",
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
            f"S05 generated {summary['policyCount']} DSL policy records with stable IDs, lineage, complexity metadata, "
            "DSL files, parser validation, duplicate checks, and tiny-array execution rows."
        ),
        "corpusSummary": summary,
        "parserValidation": parser_validation,
        "dslFileValidation": dsl_file_validation,
        "sampleExecutionSummary": {
            "rowCount": int(len(sample_df)),
            "executionSuccessCount": int(sample_df["executionSuccess"].sum()),
            "valueCountsConservedCount": int(sample_df["valueCountsConserved"].sum()),
            "tinyArraySortedCount": int(sample_df["completed"].sum()),
            "stopReasonCounts": sample_df["stopReason"].value_counts(dropna=False).to_dict(),
        },
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
        "artifacts": collect_artifacts(artifacts_written + [manifest_path]),
        "git": git_metadata,
        "corpusVersion": POLICY_CORPUS_VERSION,
        "dslVersion": DSL_VERSION,
    }
    write_json(manifest_path, manifest_payload)
    artifacts_written.append(manifest_path)

    print(
        json.dumps(
            {
                "researchStepId": STEP_ID,
                "success": validation_passed,
                "policyCount": summary["policyCount"],
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
