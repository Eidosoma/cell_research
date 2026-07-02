#!/usr/bin/env python3
"""Run E04 S13 overfitting and transfer tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.e04.memory_ablations import load_selected_s08_policies
from src.e04.no_oracle_protocol import ALLOWED_TRAINING_SIGNAL_FIELDS, EXCLUDED_TRAINING_SIGNAL_FIELDS, LOCAL_ONLY_PROTOCOL_ID
from src.e04.overfitting_transfer import (
    S13_GENERALIZATION_AXES,
    S13_MEMORY_ABLATIONS,
    S13_PROTOCOL_ID,
    assert_s13_design,
    default_s13_transfer_configs,
    infer_s13_outcome,
    run_s13_transfer_matrix,
    s13_config_hash,
    s13_config_records,
    s13_memory_specs,
    summarize_transfer_gaps,
    summarize_transfer_results,
    transfer_gaps,
)


STEP_ID = "S13"
STEP_NUMBER = 13
EXPERIMENT_ID = "E04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")))
    parser.add_argument(
        "--selected-policies",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")) / "policies" / "e04_evolved_repair_policies.jsonl",
    )
    parser.add_argument("--max-events", type=int, default=120)
    parser.add_argument("--stress-max-events", type=int, default=180)
    parser.add_argument("--policy-limit", type=int, default=None)
    parser.add_argument("--run-unit-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
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
        "note": "Checksum omitted to avoid self-referential checksum drift.",
    }


def source_entry(path: Path, repo_dir: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relativePath": str(path.relative_to(repo_dir)),
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "(no rows)"
    present = [column for column in columns if column in df.columns]
    view = df[present].head(max_rows)
    header = "| " + " | ".join(present) + " |"
    separator = "| " + " | ".join("---" for _ in present) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]).replace("|", "\\|") for column in present) + " |")
    return "\n".join([header, separator, *rows])


def _parse_json(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str) and value:
        return json.loads(value)
    if isinstance(value, Mapping):
        return value
    return {}


def selected_policy_summary(path: Path) -> dict[str, Any]:
    policy_ids: list[str] = []
    generations: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            policy_ids.append(str(record["policy_id"]))
            generations.append(int(record["generation"]))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "policyIds": policy_ids,
        "policyCount": len(policy_ids),
        "generationRange": [min(generations), max(generations)] if generations else [],
    }


def write_figure(summary_df: pd.DataFrame, gap_df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    neighbor = gap_df[gap_df["memory_ablation"] == "neighbor_memory"].copy()
    pivot = neighbor.pivot_table(
        index="s08_source_policy_id",
        columns="generalization_axis",
        values="fitness_retention_fraction",
        aggfunc="mean",
    ).reindex(columns=[axis for axis in S13_GENERALIZATION_AXES if axis != "s08_train_reference"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    image = axes[0].imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.1)
    axes[0].set_title("Neighbor-memory held-out fitness retention")
    axes[0].set_xticks(range(len(pivot.columns)), [item.replace("_", " ") for item in pivot.columns], rotation=35, ha="right")
    axes[0].set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axis_delta = (
        gap_df[gap_df["memory_ablation"] == "neighbor_memory"]
        .groupby("generalization_axis", as_index=False)["fitness_delta_vs_no_memory"]
        .mean()
        .sort_values("generalization_axis")
    )
    axes[1].bar(axis_delta["generalization_axis"].str.replace("_", " "), axis_delta["fitness_delta_vs_no_memory"])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Neighbor-memory held-out delta vs no-memory")
    axes[1].set_ylabel("Mean fitness delta")
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)


def validate_results(
    *,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    gap_summary_df: pd.DataFrame,
    config_records: Sequence[Mapping[str, Any]],
    frozen_config_hash: str,
    post_eval_config_hash: str,
    selected_policy_ids: Sequence[str],
    configs: Sequence[Any],
    specs: Sequence[Any],
    design_audit: Mapping[str, Any],
    unit_tests: Mapping[str, Any],
    e04_s13_tests: Mapping[str, Any],
    e04_s09_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_rows = len(selected_policy_ids) * len(configs) * len(specs)
    rows.append(
        {
            "validation_case": "transfer_matrix_complete",
            "success": bool(
                len(df) == expected_rows
                and set(df["memory_ablation"]) == set(S13_MEMORY_ABLATIONS)
                and set(df["s08_source_policy_id"]) == set(selected_policy_ids)
            ),
            "detail": f"rows={len(df)}; expected={expected_rows}; policies={sorted(df['s08_source_policy_id'].unique())}",
        }
    )
    rows.append(
        {
            "validation_case": "heldout_configs_predefined_before_evaluation",
            "success": bool(
                frozen_config_hash == post_eval_config_hash
                and all(bool(record.get("predefined_before_evaluation")) for record in config_records)
                and bool(design_audit.get("success", False))
            ),
            "detail": json.dumps(
                {
                    "frozenConfigHash": frozen_config_hash,
                    "postEvalConfigHash": post_eval_config_hash,
                    "designAudit": dict(design_audit),
                },
                sort_keys=True,
                default=str,
            )[:1800],
        }
    )
    rows.append(
        {
            "validation_case": "no_policy_retuning_or_test_result_selection",
            "success": bool(
                (df["selected_policy_was_retuned"] == False).all()
                and (df["policy_parameter_update_count"] == 0).all()
                and set(df["s08_source_policy_id"]) == set(selected_policy_ids)
            ),
            "detail": f"retuned rows={int(df['selected_policy_was_retuned'].sum())}; parameter update total={int(df['policy_parameter_update_count'].sum())}",
        }
    )
    group_cols = ["s08_source_policy_id", "s13_config_id"]
    grouped = df.groupby(group_cols)
    complete_specs = grouped["memory_ablation"].nunique().eq(len(S13_MEMORY_ABLATIONS)).all()
    matched_schedules = grouped["schedule_sha256"].nunique().eq(1).all()
    rows.append(
        {
            "validation_case": "matched_policy_config_schedule_groups",
            "success": bool(complete_specs and matched_schedules),
            "detail": f"groups={grouped.ngroups}; complete specs={bool(complete_specs)}; matched schedules={bool(matched_schedules)}",
        }
    )
    feature_records = [_parse_json(value) for value in df["feature_audit_summary_json"]]
    signal_records = [_parse_json(value) for value in df["signal_audit_json"]]
    capacity_records = [_parse_json(value) for value in df["memory_capacity_observed_json"]]
    forbidden_hits = [
        record
        for record in feature_records
        if record.get("forbiddenTokenHits") or record.get("excludedSignalHits") or record.get("listLikeFeatureValueHits")
    ]
    target_nonzero = [record for record in signal_records if not record.get("targetDerivedSignalFieldsZeroed", False)]
    capacity_violations = [record for record in capacity_records if record.get("capacityViolations")]
    rows.append(
        {
            "validation_case": "s07_projected_local_only_policy_inputs",
            "success": bool(
                (df["uses_global_oracle"] == False).all()
                and all(not record.get("usesGlobalOracle", True) for record in feature_records)
                and df["policy_input_contract"].str.startswith("S07 LocalTrainingObservation").all()
            ),
            "detail": f"oracle rows={int(df['uses_global_oracle'].sum())}; protocol={LOCAL_ONLY_PROTOCOL_ID}; feature records={len(feature_records)}",
        }
    )
    rows.append(
        {
            "validation_case": "forbidden_target_derived_fields_excluded_and_capacity_limited",
            "success": bool(not forbidden_hits and not target_nonzero and not capacity_violations),
            "detail": (
                f"feature forbidden/excluded hits={len(forbidden_hits)}; target-derived nonzero signal rows={len(target_nonzero)}; "
                f"capacity violation rows={len(capacity_violations)}; allowed={list(ALLOWED_TRAINING_SIGNAL_FIELDS)}; excluded={list(EXCLUDED_TRAINING_SIGNAL_FIELDS)}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "train_reference_and_heldout_profiles_present",
            "success": bool(
                set(df["split"]) == {"train_reference", "heldout"}
                and set(df["generalization_axis"]) == set(S13_GENERALIZATION_AXES)
                and df[df["split"] == "heldout"]["array_size"].max() > df[df["split"] == "train_reference"]["array_size"].max()
                and not summary_df.empty
                and not gap_df.empty
                and not gap_summary_df.empty
            ),
            "detail": (
                f"splits={sorted(df['split'].unique())}; axes={sorted(df['generalization_axis'].unique())}; "
                f"array sizes={sorted(int(value) for value in df['array_size'].unique())}"
            ),
        }
    )
    rows.append(
        {
            "validation_case": "no_centralized_baseline_mixed_into_local_claims",
            "success": bool((df["centralized_baseline"] == False).all() and (df["eligible_for_local_only_claims"] == True).all()),
            "detail": f"centralized rows={int(df['centralized_baseline'].sum())}",
        }
    )
    rows.append(
        {
            "validation_case": "unit_and_regression_tests_passed",
            "success": bool(
                unit_tests["success"]
                and e04_s13_tests["success"]
                and e04_s09_tests["success"]
                and e03_policy_tests["success"]
                and e02_tests["success"]
            ),
            "detail": (
                f"E04 discover={unit_tests['returnCode']}; E04 S13={e04_s13_tests['returnCode']}; "
                f"E04 S09={e04_s09_tests['returnCode']}; E03 policy={e03_policy_tests['returnCode']}; "
                f"E02 deterministic={e02_tests['returnCode']}"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_report(
    *,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    gap_summary_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    outcome: Mapping[str, Any],
    selected_policy_info: Mapping[str, Any],
    design_audit: Mapping[str, Any],
    frozen_config_hash: str,
    validation_line: str,
    validation_success: bool,
    artifact_paths: Sequence[Path],
    result_path: Path,
    figure_path: Path,
    predefined_config_path: Path,
    config_path: Path,
    manifest_path: Path,
    run_manifest_path: Path,
    checksums_path: Path,
    args: argparse.Namespace,
    unit_tests: Mapping[str, Any],
    e04_s13_tests: Mapping[str, Any],
    e04_s09_tests: Mapping[str, Any],
    e03_policy_tests: Mapping[str, Any],
    e02_tests: Mapping[str, Any],
) -> str:
    classification = "constraining/contradictory" if not validation_success else str(outcome.get("outcome", "null"))
    artifact_md = "\n".join(f"- `{path}`" for path in artifact_paths)
    commands = "\n".join(
        [
            f"- `{e04_s13_tests['command']}` -> return code {e04_s13_tests['returnCode']}",
            f"- `{unit_tests['command']}` -> return code {unit_tests['returnCode']}",
            f"- `{e04_s09_tests['command']}` -> return code {e04_s09_tests['returnCode']}",
            f"- `{e03_policy_tests['command']}` -> return code {e03_policy_tests['returnCode']}",
            f"- `{e02_tests['command']}` -> return code {e02_tests['returnCode']}",
            (
                "- `python scripts/e04_s13_overfitting_tests.py --repo-dir /workspace/cell-research "
                "--artifacts-dir $ARTIFACTS_DIR`"
            ),
        ]
    )
    summary_table = markdown_table(
        summary_df,
        [
            "s08_source_policy_id",
            "memory_ablation",
            "split",
            "generalization_axis",
            "runs",
            "mean_fitness",
            "mean_time_in_target_fraction",
            "mean_final_sortedness_percent",
            "oracle_hit_rows",
        ],
        max_rows=36,
    )
    gap_table = markdown_table(
        gap_df,
        [
            "s08_source_policy_id",
            "memory_ablation",
            "generalization_axis",
            "mean_fitness",
            "train_reference_mean_fitness",
            "fitness_transfer_gap",
            "fitness_retention_fraction",
            "fitness_delta_vs_no_memory",
        ],
        max_rows=32,
    )
    gap_summary_table = markdown_table(
        gap_summary_df,
        [
            "memory_ablation",
            "heldout_axis_rows",
            "mean_train_reference_fitness",
            "mean_heldout_fitness",
            "mean_fitness_transfer_gap",
            "mean_fitness_retention_fraction",
            "mean_fitness_delta_vs_no_memory",
            "positive_axis_fraction_vs_no_memory",
        ],
        max_rows=10,
    )
    validation_table = markdown_table(validation_df, ["validation_case", "success", "detail"], max_rows=20)
    return f"""# E04 S13 Research Step Full Results

## Top Summary

- Research step ID: S13
- Completion status: {'Completed' if validation_success else 'Completed with validation failure'} on {utc_now()}
- Artifacts written:
{artifact_md}
- Validation result: {validation_line}
- Outcome classification: {classification}
- Caveats or blockers: S13 evaluates selected S08 policies without retuning. Held-out configs were written and hashed before evaluation. Results are computational transfer proxies in the 1D sorting-array substrate.
- Lay summary: S13 compared selected policies under no-memory and S09-supported neighbor-memory masks on S08 train-reference conditions and predefined held-out transfer conditions covering larger arrays, denser perturbations, edge/cluster Frozen Cell placement, and combined stress.
- Recommended next action: Hand control back to the Chief Scientist. If accepted, proceed to S14 centralized repair baseline comparison.

## Frozen Question

Do evolved or learned repair policies generalize to unseen array sizes, perturbation rates, and Frozen Cell types?

S13 outcome rule: `{json.dumps(dict(outcome), sort_keys=True)}`.

## Inputs

- Active plan: `/workspace/RESEARCH_PLAN.md`, Experiment E04, step S13.
- Selected S08 policies: `{args.selected_policies}`.
- Selected policy summary: `{json.dumps(dict(selected_policy_info), sort_keys=True)}`.
- Predefined config hash: `{frozen_config_hash}`.
- Predefined held-out config artifact: `{predefined_config_path}`.
- Design audit: `{json.dumps(dict(design_audit), sort_keys=True, default=str)}`.
- Datasets: none required.

## Methods

Implemented `src/e04/overfitting_transfer.py`, `tests/e04/test_overfitting_transfer.py`, and `scripts/e04_s13_overfitting_tests.py`.

S13 first materializes a JSON config artifact containing all train-reference and held-out transfer configs, schedule hashes, seeds, array sizes, and schedule variants. That hash is checked again after evaluation. The selected S08 policies are loaded from JSONL and evaluated without parameter updates or selection on S13 outcomes.

Policy-visible local inputs are the S07 `LocalTrainingObservation` projection masked by S09 memory-ablation specs. S13 includes `no_memory` and S09-supported `neighbor_memory` only. Forbidden and target-derived signal fields remain excluded.

## Commands

{commands}

## Dependencies And Runtime

- Python: {platform.python_version()}
- pandas: {pd.__version__}
- matplotlib backend: Agg
- New dependencies installed: none.
- CPU use: serial stateful trajectory evaluation with worker count `1`; host logical CPUs recorded in run manifest.
- Platform: {platform.platform()}

## Parameters

- Protocol ID: `{S13_PROTOCOL_ID}`
- Selected policies: `{selected_policy_info.get('policyCount')}`
- Memory ablations: `{list(S13_MEMORY_ABLATIONS)}`
- Generalization axes: `{list(S13_GENERALIZATION_AXES)}`
- Max events: `{args.max_events}`
- Stress max events: `{args.stress_max_events}`
- Result table: `{result_path}`
- Figure: `{figure_path}`
- Config: `{config_path}`

## Results

- Total result rows: `{len(df)}`
- Policies evaluated: `{df['s08_source_policy_id'].nunique()}`
- Memory ablations: `{df['memory_ablation'].nunique()}`
- Generalization axes: `{sorted(df['generalization_axis'].unique())}`
- Oracle-hit rows: `{int(df['uses_global_oracle'].sum())}`
- Outcome: `{classification}`

### Transfer Profiles

{summary_table}

### Held-Out Transfer Gaps

{gap_table}

### Gap Summary

{gap_summary_table}

## Validation Checks

{validation_table}

## Caveats, Blockers, Failed Assumptions, And Limitations

- No blocker remains for S13 artifact generation if validation passed.
- S13 does not retune policies or claim S13 held-out performance as search feedback.
- S13 tests two selected S08 adjacent-action policies under no-memory and neighbor-memory masks; it does not exhaust all E03 policy families.
- Held-out Frozen Cell "types" are represented by predefined placement/schedule regimes in a 1D adjacent-swap model, not biological damage categories.
- S13 does not revise S10/S12 communication conclusions because communication modes are not treated as supported mechanisms here.

## Provenance

- Git commit at validation time: `{git_output(args.repo_dir, ['rev-parse', 'HEAD'])}`
- Git branch at validation time: `{git_output(args.repo_dir, ['branch', '--show-current'])}`
- Git status at validation time: `{git_output(args.repo_dir, ['status', '--short']) or 'clean'}`
- Predefined config hash: `{frozen_config_hash}`
- Config: `{config_path}`
- Artifact manifest: `{manifest_path}`
- Run manifest: `{run_manifest_path}`
- Checksums: `{checksums_path}`
- Created at UTC: `{utc_now()}`

## Artifact Manifest

- Result parquet: `{result_path}`
- Figure: `{figure_path}`
- Predefined held-out configs: `{predefined_config_path}`

## Recommended Next Action

Hand control back to the Chief Scientist. If S13 is accepted, execute S14 centralized repair baseline comparison with centralized rows clearly labeled and excluded from local-only claims.
"""


def main() -> int:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    result_path = artifacts_dir / "results" / "e04_overfitting_and_transfer.parquet"
    result_csv_path = artifacts_dir / "tables" / "e04_overfitting_and_transfer.csv"
    summary_path = artifacts_dir / "tables" / "e04_overfitting_and_transfer_summary.csv"
    gap_path = artifacts_dir / "tables" / "e04_overfitting_and_transfer_gaps.csv"
    gap_summary_path = artifacts_dir / "tables" / "e04_overfitting_and_transfer_gap_summary.csv"
    validation_path = artifacts_dir / "tables" / "e04_overfitting_and_transfer_validation.csv"
    figure_path = artifacts_dir / "figures" / "e04" / "transfer_gap_summary.png"
    predefined_config_path = artifacts_dir / "configs" / "e04_s13_predefined_heldout_configs.json"
    config_path = artifacts_dir / "configs" / "e04_s13_overfitting_transfer.json"
    source_manifest_path = artifacts_dir / "src_snapshot" / "e04_overfitting_transfer_manifest.json"
    report_path = step_dir / "research_step_full_results.md"
    manifest_path = step_dir / "artifact_manifest.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    checksums_path = artifacts_dir / "checksums" / "sha256sums.txt"

    configs = default_s13_transfer_configs(max_events=args.max_events, stress_max_events=args.stress_max_events)
    specs = s13_memory_specs()
    design_audit = assert_s13_design(configs, specs)
    config_records = s13_config_records(configs)
    frozen_config_hash = s13_config_hash(configs)
    predefined_config = {
        "schema": "eidosoma.e04.s13.predefined_heldout_configs.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "createdAtUtc": utc_now(),
        "createdBeforeEvaluation": True,
        "configHashSha256": frozen_config_hash,
        "designAudit": design_audit,
        "configs": config_records,
    }
    write_json(predefined_config_path, predefined_config)

    selected_policies = load_selected_s08_policies(args.selected_policies, limit=args.policy_limit)
    selected_policy_info = selected_policy_summary(args.selected_policies)
    if args.policy_limit is not None:
        selected_policy_info["policyLimitApplied"] = int(args.policy_limit)
        selected_policy_info["policyIds"] = [policy.policy_id for policy in selected_policies]
        selected_policy_info["policyCount"] = len(selected_policies)

    config_record = {
        "schema": "eidosoma.e04.s13.overfitting_transfer_config.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "protocolId": S13_PROTOCOL_ID,
        "selectedPoliciesPath": str(args.selected_policies),
        "selectedPoliciesSha256": selected_policy_info["sha256"],
        "selectedPolicyIds": selected_policy_info["policyIds"],
        "maxEvents": args.max_events,
        "stressMaxEvents": args.stress_max_events,
        "workerCount": 1,
        "allowedTrainingSignalFields": list(ALLOWED_TRAINING_SIGNAL_FIELDS),
        "excludedTrainingSignalFields": list(EXCLUDED_TRAINING_SIGNAL_FIELDS),
        "memoryAblations": [spec.to_dict() for spec in specs],
        "predefinedConfigPath": str(predefined_config_path),
        "predefinedConfigHashSha256": frozen_config_hash,
        "designAudit": design_audit,
        "noRetuningPolicy": {
            "selectedPolicyWasRetuned": False,
            "policyParameterUpdateCount": 0,
            "testResultsUsedForSelection": False,
        },
    }
    write_json(config_path, config_record)

    rows = run_s13_transfer_matrix(
        genomes=selected_policies,
        configs=configs,
        specs=specs,
        frozen_config_hash=frozen_config_hash,
    )
    post_eval_config_hash = s13_config_hash(configs)
    df = pd.DataFrame(rows).sort_values(
        ["s08_source_policy_id", "memory_ablation_order", "split", "generalization_axis", "scenario_name", "activation_seed"]
    ).reset_index(drop=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_path, index=False)
    df.to_csv(result_csv_path, index=False)
    summary_df = summarize_transfer_results(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    gap_df = transfer_gaps(df)
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    gap_df.to_csv(gap_path, index=False)
    gap_summary_df = summarize_transfer_gaps(gap_df)
    gap_summary_path.parent.mkdir(parents=True, exist_ok=True)
    gap_summary_df.to_csv(gap_summary_path, index=False)
    outcome = infer_s13_outcome(gap_summary_df)
    write_figure(summary_df, gap_df, figure_path)

    if args.run_unit_tests:
        e04_s13_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_overfitting_transfer"], args.repo_dir)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e04", "-p", "test_*.py"], args.repo_dir)
        e04_s09_tests = run_command([sys.executable, "-m", "unittest", "tests.e04.test_memory_ablations"], args.repo_dir)
        e03_policy_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03", "-p", "test_policy_interface.py"], args.repo_dir)
        e02_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e02", "-p", "test_deterministic_simulator.py"], args.repo_dir)
    else:
        skipped = {"command": "skipped by --no-run-unit-tests", "returnCode": 0, "elapsedSeconds": 0.0, "stdout": "", "stderr": "", "success": True}
        e04_s13_tests = unit_tests = e04_s09_tests = e03_policy_tests = e02_tests = skipped

    validation_df = validate_results(
        df=df,
        summary_df=summary_df,
        gap_df=gap_df,
        gap_summary_df=gap_summary_df,
        config_records=config_records,
        frozen_config_hash=frozen_config_hash,
        post_eval_config_hash=post_eval_config_hash,
        selected_policy_ids=[policy.policy_id for policy in selected_policies],
        configs=configs,
        specs=specs,
        design_audit=design_audit,
        unit_tests=unit_tests,
        e04_s13_tests=e04_s13_tests,
        e04_s09_tests=e04_s09_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(validation_path, index=False)
    validation_success = bool(validation_df["success"].all())
    validation_line = (
        f"{int(validation_df['success'].sum())}/{len(validation_df)} validation cases passed; "
        f"E04 S13 tests return code {e04_s13_tests['returnCode']}; E04 tests return code {unit_tests['returnCode']}; "
        f"E04 S09 tests return code {e04_s09_tests['returnCode']}; E03 policy tests return code {e03_policy_tests['returnCode']}; "
        f"E02 simulator tests return code {e02_tests['returnCode']}; {len(df)} transfer rows; "
        f"heldout config hash stable={frozen_config_hash == post_eval_config_hash}; "
        f"oracle-hit rows={int(df['uses_global_oracle'].sum())}; outcome={outcome.get('outcome')}"
    )

    source_files = [
        args.repo_dir / "src/e04/overfitting_transfer.py",
        args.repo_dir / "scripts/e04_s13_overfitting_tests.py",
        args.repo_dir / "tests/e04/test_overfitting_transfer.py",
        args.repo_dir / "src/e04/memory_ablations.py",
        args.repo_dir / "src/e04/evolutionary_search.py",
        args.repo_dir / "src/e04/no_oracle_protocol.py",
        args.repo_dir / "src/e04/homeostasis.py",
        args.repo_dir / "src/e04/memory_policies.py",
        args.repo_dir / "src/e04/signaling.py",
    ]
    source_manifest = {
        "schema": "eidosoma.src_snapshot.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "sources": [source_entry(path, args.repo_dir) for path in source_files if path.exists()],
    }
    write_json(source_manifest_path, source_manifest)

    artifact_paths = [
        report_path,
        result_path,
        result_csv_path,
        summary_path,
        gap_path,
        gap_summary_path,
        validation_path,
        figure_path,
        predefined_config_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
        checksums_path,
    ]
    report_text = render_report(
        df=df,
        summary_df=summary_df,
        gap_df=gap_df,
        gap_summary_df=gap_summary_df,
        validation_df=validation_df,
        outcome=outcome,
        selected_policy_info=selected_policy_info,
        design_audit=design_audit,
        frozen_config_hash=frozen_config_hash,
        validation_line=validation_line,
        validation_success=validation_success,
        artifact_paths=artifact_paths,
        result_path=result_path,
        figure_path=figure_path,
        predefined_config_path=predefined_config_path,
        config_path=config_path,
        manifest_path=manifest_path,
        run_manifest_path=run_manifest_path,
        checksums_path=checksums_path,
        args=args,
        unit_tests=unit_tests,
        e04_s13_tests=e04_s13_tests,
        e04_s09_tests=e04_s09_tests,
        e03_policy_tests=e03_policy_tests,
        e02_tests=e02_tests,
    )
    write_text(report_path, report_text)

    manifest_artifacts = [
        artifact_entry(report_path, artifacts_dir, "S13 full-results handoff report"),
        artifact_entry(result_path, artifacts_dir, "S13 overfitting and transfer result rows"),
        artifact_entry(result_csv_path, artifacts_dir, "CSV sidecar for S13 transfer rows"),
        artifact_entry(summary_path, artifacts_dir, "S13 train/heldout profile summary"),
        artifact_entry(gap_path, artifacts_dir, "S13 held-out transfer gaps"),
        artifact_entry(gap_summary_path, artifacts_dir, "S13 transfer gap summary"),
        artifact_entry(validation_path, artifacts_dir, "S13 validation table"),
        artifact_entry(figure_path, artifacts_dir, "S13 transfer gap figure"),
        artifact_entry(predefined_config_path, artifacts_dir, "S13 predefined held-out config artifact"),
        artifact_entry(config_path, artifacts_dir, "S13 run configuration"),
        artifact_entry(source_manifest_path, artifacts_dir, "S13 source/provenance manifest"),
        manifest_self_entry(manifest_path, artifacts_dir, "S13 artifact manifest"),
        manifest_self_entry(run_manifest_path, artifacts_dir, "Experiment run manifest"),
        manifest_self_entry(checksums_path, artifacts_dir, "SHA-256 checksums for key S13 outputs"),
    ]
    artifact_manifest = {
        "schema": "eidosoma.artifact_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "artifacts": manifest_artifacts,
    }
    write_json(manifest_path, artifact_manifest)

    run_manifest = {
        "schema": "eidosoma.run_manifest.v1",
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "createdAtUtc": utc_now(),
        "gitCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
        "gitBranch": git_output(args.repo_dir, ["branch", "--show-current"]),
        "gitStatusShort": git_output(args.repo_dir, ["status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "workerCount": 1,
        "cpuCount": os.cpu_count(),
        "configPath": str(config_path),
        "predefinedConfigPath": str(predefined_config_path),
        "predefinedConfigHashSha256": frozen_config_hash,
        "transferRows": int(len(df)),
        "outcome": dict(outcome),
        "validation": validation_df.to_dict(orient="records"),
        "artifacts": manifest_artifacts,
    }
    write_json(run_manifest_path, run_manifest)

    checksum_lines = []
    for path in [
        report_path,
        result_path,
        result_csv_path,
        summary_path,
        gap_path,
        gap_summary_path,
        validation_path,
        figure_path,
        predefined_config_path,
        config_path,
        source_manifest_path,
        manifest_path,
        run_manifest_path,
    ]:
        checksum_lines.append(f"{sha256_file(path)}  {path}\n")
    checksums_path.parent.mkdir(parents=True, exist_ok=True)
    checksums_path.write_text("".join(checksum_lines), encoding="utf-8")

    print(validation_line)
    print(f"report={report_path}")
    print(f"results={result_path}")
    print(f"figure={figure_path}")
    print(f"predefined_configs={predefined_config_path}")
    return 0 if validation_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
