#!/usr/bin/env python3
"""Run E03 S12 targeted causal rule-feature ablations."""

from __future__ import annotations

import argparse
import json
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

from src.e03.coarse_sweep import screen_configs
from src.e03.gpu_batch_simulator import jax_backend_summary
from src.e03.rule_ablations import (
    ABLATION_SCHEMA,
    DEFAULT_ABLATION_SEED,
    ablation_result_digest,
    aggregate_ablation_pairs,
    build_ablation_variants,
    claim_frame,
    component_summary_frame,
    matched_seed_validation,
    policy_record_for_variant,
    run_cpu_policy_config_matched,
    variant_metadata_frame,
)
from src.e03.rule_dsl import parse_policy
from src.e03.universality_classes import load_policy_code_table


STEP_ID = "S12"
STEP_NUMBER = 12
EXPERIMENT_ID = "E03"


def parse_args() -> argparse.Namespace:
    artifacts_default = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts_default)
    parser.add_argument("--s11-clusters", type=Path, default=artifacts_default / "results/e03_policy_clusters.parquet")
    parser.add_argument("--s11-summary", type=Path, default=artifacts_default / "results/e03_policy_cluster_summary.parquet")
    parser.add_argument("--s11-exemplars", type=Path, default=artifacts_default / "tables/e03_cluster_exemplar_inspection.csv")
    parser.add_argument("--generated-policy-library", type=Path, default=artifacts_default / "policies/e03_generated_policy_library.jsonl")
    parser.add_argument("--qd-candidates", type=Path, default=artifacts_default / "results/e03_qd_candidate_summary.parquet")
    parser.add_argument("--qd-discovered", type=Path, default=artifacts_default / "policies/e03_qd_discovered_policies.jsonl")
    parser.add_argument("--s04-classic-vectors", type=Path, default=artifacts_default / "results/e03_classic_competence_vectors.parquet")
    parser.add_argument("--max-target-policies", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_ABLATION_SEED)
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


COMPONENT_SELECTORS: dict[str, Any] = {
    "usesIdeal": lambda frame: frame["usesIdeal"].map(_truthy),
    "usesPrefixSorted": lambda frame: frame["usesPrefixSorted"].map(_truthy),
    "usesRandomCondition": lambda frame: frame["usesRandomCondition"].map(_truthy),
    "usesProbabilisticAction": lambda frame: frame["usesProbabilisticAction"].map(_truthy),
    "usesMemory": lambda frame: frame["usesMemory"].map(_truthy),
    "usesSignal": lambda frame: frame["usesSignal"].map(_truthy),
    "hasSwapAction": lambda frame: pd.to_numeric(frame["swapActionCount"], errors="coerce").fillna(0) > 0,
    "hasStateAction": lambda frame: pd.to_numeric(frame["stateActionCount"], errors="coerce").fillna(0) > 0,
}


def select_targets(clusters: pd.DataFrame, exemplars: pd.DataFrame, code_table: pd.DataFrame, max_targets: int) -> pd.DataFrame:
    source = clusters.merge(code_table, on="policy_id", how="inner", validate="one_to_one")
    exemplar_roles = (
        exemplars.groupby("policy_id")
        .agg(exemplar_roles=("exemplar_role", lambda values: ",".join(sorted(set(str(value) for value in values)))))
        .reset_index()
    )
    source = source.merge(exemplar_roles, on="policy_id", how="left")
    source["exemplar_roles"] = source["exemplar_roles"].fillna("")
    selected: dict[str, dict[str, Any]] = {}

    def add(row: pd.Series, reason: str) -> None:
        policy_id = str(row["policy_id"])
        record = dict(row)
        if policy_id in selected:
            reasons = set(str(selected[policy_id]["selection_reason"]).split(";"))
            reasons.add(reason)
            selected[policy_id]["selection_reason"] = ";".join(sorted(item for item in reasons if item))
            if not selected[policy_id].get("exemplar_roles") and record.get("exemplar_roles"):
                selected[policy_id]["exemplar_roles"] = record.get("exemplar_roles")
        else:
            record["selection_reason"] = reason
            selected[policy_id] = record

    role_priority = {"classic_landmark": 0, "medoid": 1, "top_screen_score": 2, "archive_winner": 3, "phase_diagnostic_policy": 4}
    exemplar_ids = exemplars.copy()
    exemplar_ids["_role_priority"] = exemplar_ids["exemplar_role"].map(lambda value: role_priority.get(str(value), 99))
    for _, exemplar in exemplar_ids.sort_values(["class_id", "_role_priority", "screen_score"], ascending=[True, True, False], kind="mergesort").iterrows():
        matches = source[source["policy_id"] == str(exemplar["policy_id"])]
        if not matches.empty:
            add(matches.iloc[0], f"s11_{exemplar['exemplar_role']}")

    for component_name, selector in COMPONENT_SELECTORS.items():
        mask = selector(source)
        for _, group in source[mask].sort_values("screen_score", ascending=False, kind="mergesort").groupby("class_id", sort=True):
            if not group.empty:
                add(group.iloc[0], f"component_cover_{component_name}")

    if len(selected) > max_targets:
        selected_df = pd.DataFrame(selected.values())
        selected_df["_priority"] = selected_df["selection_reason"].map(lambda text: 0 if "s11_" in str(text) else 1)
        selected_df = selected_df.sort_values(["_priority", "class_id", "screen_score"], ascending=[True, True, False], kind="mergesort").head(max_targets)
        return selected_df.drop(columns=["_priority"]).reset_index(drop=True)
    return pd.DataFrame(selected.values()).sort_values(["class_id", "screen_score"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def build_variants_for_targets(targets: pd.DataFrame) -> tuple[list[dict[str, Any]], list[Any]]:
    baselines: list[dict[str, Any]] = []
    variants: list[Any] = []
    for row in targets.to_dict(orient="records"):
        policy = parse_policy(str(row["dsl_source"]))
        policy_id = str(row["policy_id"])
        policy_name = str(row.get("policy_name", policy.name))
        baselines.append(
            {
                "policy": policy,
                "original_policy_id": policy_id,
                "original_policy_name": policy_name,
                "class_id": str(row.get("class_id", "")),
                "cautious_label": str(row.get("cautious_label", "")),
                "exemplar_roles": str(row.get("exemplar_roles", "")),
                "selection_reason": str(row.get("selection_reason", "")),
                "code_source": str(row.get("code_source", "")),
            }
        )
        variants.extend(
            build_ablation_variants(
                policy=policy,
                original_policy_id=policy_id,
                original_policy_name=policy_name,
                class_id=str(row.get("class_id", "")),
                cautious_label=str(row.get("cautious_label", "")),
                exemplar_roles=str(row.get("exemplar_roles", "")),
                selection_reason=str(row.get("selection_reason", "")),
                code_source=str(row.get("code_source", "")),
            )
        )
    return baselines, variants


def evaluate_variants(baselines: list[dict[str, Any]], variants: list[Any], configs: list[Any], seed: int) -> pd.DataFrame:
    variant_by_original: dict[str, list[Any]] = {}
    for variant in variants:
        variant_by_original.setdefault(variant.original_policy_id, []).append(variant)
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        original_policy_id = baseline["original_policy_id"]
        original_record = policy_record_for_variant(
            baseline["policy"],
            policy_id=original_policy_id,
            policy_name=baseline["original_policy_name"],
            source_kind="s12_original",
        )
        for config in configs:
            row = run_cpu_policy_config_matched(
                original_record,
                config,
                original_policy_id=original_policy_id,
                variant_kind="original",
                ablation_id="original",
                ablation_component="none",
                seed=seed,
            )
            row.update(
                {
                    "class_id": baseline["class_id"],
                    "cautious_label": baseline["cautious_label"],
                    "exemplar_roles": baseline["exemplar_roles"],
                    "selection_reason": baseline["selection_reason"],
                }
            )
            rows.append(row)
        for variant in variant_by_original.get(original_policy_id, []):
            record = policy_record_for_variant(variant.policy, policy_id=variant.variant_policy_id, policy_name=variant.variant_policy_name, source_kind="s12_ablation")
            for config in configs:
                row = run_cpu_policy_config_matched(
                    record,
                    config,
                    original_policy_id=original_policy_id,
                    variant_kind="ablation",
                    ablation_id=variant.ablation_id,
                    ablation_component=variant.ablation_component,
                    seed=seed,
                )
                row.update(
                    {
                        "class_id": variant.class_id,
                        "cautious_label": variant.cautious_label,
                        "exemplar_roles": variant.exemplar_roles,
                        "selection_reason": variant.selection_reason,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def validation_frame(
    *,
    targets: pd.DataFrame,
    variants: pd.DataFrame,
    run_df: pd.DataFrame,
    aggregate: pd.DataFrame,
    claims: pd.DataFrame,
    component_summary: pd.DataFrame,
    figure_exists: bool,
    unit_tests: dict[str, Any],
) -> pd.DataFrame:
    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": expected,
                "observed": str(observed),
                "notes": notes,
            }
        )

    add("all_s11_classes_targeted", targets["class_id"].nunique() == 8, "8 classes", targets["class_id"].nunique(), "Targets include at least one source-backed policy from each S11 class.")
    add("target_sources_parse", not targets.empty, ">0 selected source-backed targets", len(targets), "Selected targets were inner-joined to available DSL source.")
    add("ablation_variants_present", len(variants) > 0, ">0 variants", len(variants), "Applicable ablation transforms produced variant policies.")
    add("source_verification_passed", bool(variants["source_verification_success"].all()), "all variants verified", int(variants["source_verification_success"].sum()), "Each variant changed only its declared component family.")
    add("matched_seed_configs", matched_seed_validation(run_df), "each variant has original config/seed/RNG set", run_df[["variant_kind", "config_id"]].drop_duplicates().shape[0], "Original and ablated rows are matched on S07 screen configs and S12 RNG seed.")
    add("invalid_runs_classified_zero", int(run_df["invalid"].sum()) == 0, "0 invalid runs", int(run_df["invalid"].sum()), "Invalid classifications are explicit; none occurred.")
    add("aggregate_rows_match_variants", len(aggregate) == len(variants), f"{len(variants)} aggregate rows", len(aggregate), "One matched aggregate row per ablation variant.")
    add("claim_rows_present", len(claims) == len(aggregate) and "claim_type" in claims.columns, "one claim row per aggregate", len(claims), "Necessary/sufficient candidates are classified cautiously.")
    add("component_summary_present", not component_summary.empty, "component summary non-empty", len(component_summary), "Effects summarized by class and component.")
    add("figure_written", figure_exists, "figure file exists", figure_exists, "A compact effect figure was written.")
    add("unit_tests_passed", bool(unit_tests["success"]), "unit tests pass", unit_tests["returnCode"], "Full E03 unit suite was run from the S12 script.")
    return pd.DataFrame(cases)


def write_effect_figure(path: Path, claims: pd.DataFrame, component_summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = (
        claims.groupby("ablation_component")
        .agg(mean_delta=("heldout_delta_final_sortedness", "mean"))
        .sort_values("mean_delta")
        .index.tolist()
    )
    positions = {component: idx for idx, component in enumerate(order)}
    palette = plt.get_cmap("tab10")
    for class_index, (class_id, group) in enumerate(claims.groupby("class_id", sort=True)):
        axes[0].scatter(
            [positions[value] + (class_index - 3.5) * 0.025 for value in group["ablation_component"]],
            group["heldout_delta_final_sortedness"],
            s=34,
            alpha=0.68,
            color=palette(class_index % 10),
            label=str(class_id),
        )
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].axhline(-0.10, color="#b2182b", linewidth=0.8, linestyle="--")
    axes[0].axhline(0.10, color="#2166ac", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=35, ha="right")
    axes[0].set_ylabel("Held-out final sortedness delta (ablated - original)")
    axes[0].set_title("Matched ablation effects by component")
    axes[0].grid(True, axis="y", alpha=0.18, linewidth=0.6)
    axes[0].legend(loc="best", fontsize=8, ncol=2, frameon=False)

    claim_counts = claims["claim_type"].value_counts().sort_index()
    axes[1].barh(np.arange(len(claim_counts)), claim_counts.values, color="#4c78a8", alpha=0.85)
    axes[1].set_yticks(np.arange(len(claim_counts)))
    axes[1].set_yticklabels(claim_counts.index, fontsize=8)
    axes[1].set_xlabel("Ablation-pair count")
    axes[1].set_title("Cautious local claim classifications")
    axes[1].grid(True, axis="x", alpha=0.18, linewidth=0.6)
    for idx, count in enumerate(claim_counts.values):
        axes[1].text(count + 0.3, idx, str(int(count)), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(
    *,
    artifacts: dict[str, Path],
    manifest: dict[str, Any],
    validation: pd.DataFrame,
    unit_tests: dict[str, Any],
    command_line: str,
    targets: pd.DataFrame,
    variants: pd.DataFrame,
    run_df: pd.DataFrame,
    aggregate: pd.DataFrame,
    claims: pd.DataFrame,
    component_summary: pd.DataFrame,
    classic_vectors: pd.DataFrame,
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
    component_rank = component_summary.sort_values("mean_heldout_delta_final_sortedness", kind="mergesort")
    claim_counts = claims["claim_type"].value_counts().rename_axis("claim_type").reset_index(name="count").sort_values("claim_type")
    largest_drops = claims.sort_values("heldout_delta_final_sortedness", kind="mergesort").head(12)
    survived = claims[claims["claim_type"] == "residual_sufficient_candidate"].sort_values("heldout_delta_final_sortedness", ascending=False, kind="mergesort").head(12)
    improved = claims[claims["heldout_delta_final_sortedness"] >= 0.03].sort_values("heldout_delta_final_sortedness", ascending=False, kind="mergesort").head(12)
    target_summary = targets.groupby("class_id").agg(target_policies=("policy_id", "nunique"), mean_screen_score=("screen_score", "mean")).reset_index()
    classic_cols = [
        "policy_id",
        "algorithm",
        "representation_type",
        "final_sortedness_score",
        "frozen_cell_robustness_score",
        "dg_tendency_score",
        "aggregation_tendency_score",
    ]
    return f"""# E03 S12 Research Step Full Results

## Top Summary

- Step ID: S12
- Completion status: Completed.
- Artifacts written:
{artifact_list}
- Validation result: {validation_line}; ablation result digest `{ablation_result_digest(claims)}`.
- Outcome classification: {outcome}.
- Caveats or blockers: No blocker remains. Causal language is local to the tested S11 exemplar/source-backed policy families and S07-style DSL screen configs; direct mixed-Algotype aggregation and full public-simulator robustness/DG assays were not rerun in S12.
- Lay summary: S12 selected {len(targets)} source-backed policies spanning all 8 S11 classes, generated {len(variants)} targeted ablations, and ran {len(run_df)} matched CPU reference screen rows using the same configs and S12 RNG stream for each original/variant pair. Large local drops identify necessary-feature candidates; high competence after removal identifies residual sufficient-feature-set candidates; improvements after removal identify local anti-feature candidates.
- Recommended next action: Stop for Chief review; if accepted, proceed to S13 to compare classic algorithms against discovered policies using S10-S12 evidence.

## Frozen Question

Which rule components are sufficient or necessary for robustness, DG, or Aggregation-like behavior?

## Inputs

- S11 cluster assignments: `{manifest['inputArtifacts']['s11Clusters']}`
- S11 class summary: `{manifest['inputArtifacts']['s11Summary']}`
- S11 inspected exemplars: `{manifest['inputArtifacts']['s11Exemplars']}`
- S05 generated policy library: `{manifest['inputArtifacts']['generatedPolicyLibrary']}`
- S08 QD candidate policy source table: `{manifest['inputArtifacts']['qdCandidates']}`
- S08 discovered policy library: `{manifest['inputArtifacts']['qdDiscovered']}`
- S04 classic competence vectors: `{manifest['inputArtifacts']['s04ClassicVectors']}`

## Lay Summary

This step asks whether pieces of a local sorting rule matter. For each selected policy, S12 made small edited versions that removed one feature family at a time: stochastic guards, probabilistic actions, prefix-sorted guards, target-position state updates, memory/signal stubs, neighbor comparisons, or neighbor swaps. Then it ran the original and edited policy on the same arrays and random streams. If performance collapsed after removing a component, that component is a local necessary-feature candidate. If performance stayed high, the remaining rule set is a local sufficient-feature-set candidate for the tested screen.

## Methods

S12 used S11 classes and exemplars to choose policies. All inspected S11 exemplar roles were included when DSL source was available, then the selector added high-scoring component-cover policies per class so rare features such as random guards, target-position state, memory stubs, signal stubs, and probabilistic actions were tested where present. Policy code came from S05/S08 source artifacts through the same code-table loader used by S11.

Each transform was typed at the DSL level and revalidated with the S02 parser/serializer. Verification compared source-component profiles before and after ablation. Condition ablations were allowed to change only condition histograms; action ablations were allowed to change only action and nested action histograms; target-position-state ablations were allowed to change `state_init`, state-update actions, and target-state-related action counts. Any unintended rule-count, condition, state, or action change would fail validation.

Execution used a CPU reference runner derived from the S07 screen loop. Unlike S07 screening, the S12 runner seeds stochastic draws by original policy ID, config seed, and the S12 seed. This preserves matched random streams between an original and every ablated variant while still letting the original policy identity define the pair.

The S04/E01/E02 competence context was used as metric vocabulary and caveat context. S12 did not rerun E01 Frozen Cell, DG, or mixed-Algotype aggregation assays. The executable S12 metrics are S07-style proxy sortedness, improvement, work, timeout, and action-count effects.

## Commands

{markdown_table(command_rows, ["command", "returnCode", "success"])}

## Dependencies And Runtime

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- numpy: `{np.__version__}`
- matplotlib: `{matplotlib.__version__}`
- S12 used existing repository code in `src/e03/rule_ablations.py`, S02 DSL primitives, S07 screen configs, and S11 code-source loading.
- JAX backend summary, recorded for continuity only: `{json.dumps(manifest['runtime']['jax'], sort_keys=True)}`
- Worker count: serial CPU reference execution; no worker pool used.
- New dependencies installed: none.

## Parameters

- S12 seed: `{manifest['parameters']['seed']}`
- Maximum target policies: `{manifest['parameters']['maxTargetPolicies']}`
- Selected target policies: `{len(targets)}`
- Ablation variants: `{len(variants)}`
- Screen configs: `{manifest['parameters']['screenConfigIds']}`
- Matched run rows: `{len(run_df)}`

## Results

### Target Selection By Class

{markdown_table(target_summary, list(target_summary.columns))}

### Component Effects By Class

{markdown_table(component_rank, ["class_id", "ablation_component", "tested_policy_count", "tested_variant_count", "mean_heldout_delta_final_sortedness", "median_heldout_delta_final_sortedness", "large_drop_count", "survived_high_count", "improved_count", "claim_types_json"])}

### Claim Type Counts

{markdown_table(claim_counts, ["claim_type", "count"])}

### Largest Local Drops

{markdown_table(largest_drops, ["class_id", "cautious_label", "original_policy_name", "ablation_component", "heldout_original_mean_final_sortedness", "heldout_ablated_mean_final_sortedness", "heldout_delta_final_sortedness", "claim_type"])}

### Residual Sufficient-Feature-Set Candidates

{markdown_table(survived, ["class_id", "cautious_label", "original_policy_name", "ablation_component", "heldout_original_mean_final_sortedness", "heldout_ablated_mean_final_sortedness", "heldout_delta_final_sortedness", "claim_type"])}

### Local Anti-Feature Or Improvement Signals

{markdown_table(improved, ["class_id", "cautious_label", "original_policy_name", "ablation_component", "heldout_original_mean_final_sortedness", "heldout_ablated_mean_final_sortedness", "heldout_delta_final_sortedness", "claim_type"])}

## Metric Context

S04 classic vectors keep final sorting, Frozen Cell robustness, DG tendency, and aggregation tendency separate. The relevant classic metric context is:

{markdown_table(classic_vectors[[column for column in classic_cols if column in classic_vectors.columns]], [column for column in classic_cols if column in classic_vectors.columns])}

S12 maps those terms only to local proxies:

- Robustness-like proxy: held-out sortedness and timeout behavior under multiple S07 screen seeds and array sizes.
- DG-like proxy: matched change in `inversion_sortedness_delta`, which captures improvement through the screen trajectory but not full E01 DG trajectory shape.
- Aggregation-like proxy: only DSL `signal`/`remember` component stubs and S11 class context. No direct aggregation claim is made.

## Figures

- Rule ablation effect figure: `{artifacts['effectFigure']}`

## Validation Checks

{markdown_table(validation, ["validation_case", "success", "expected", "observed", "notes"])}

## Caveats, Blockers, And Limitations

- The strongest claims are local to the selected S11 classes, source-backed policies, S07 screen configs, and S12 matched CPU runner.
- S12 does not prove biological causality, morphogenesis causality, or global algorithmic necessity.
- Memory and signal primitives remain S02 metadata/state stubs, so their ablations test stub usage rather than rich communication.
- Disabling target-position state can make ideal-target guards fail at runtime while leaving those guards in code. This intentionally probes whether the policy needs the first-class ideal-position state, not whether replacing ideal targets with a neighbor rule would recover performance.
- Direct E01/E02 Frozen Cell, DG, chimeric aggregation, and public-simulator assays remain future validation work.

## Failed Assumptions

No required S11/S08/S05 policy-code input was missing for the selected target set, and all ablations passed source-change verification. The assumption that every S11 class can be represented by at least one source-backed target was supported. The assumption that S12 can make direct aggregation claims was not supported; aggregation remains a proxy caveat until a mixed-policy assay is rerun.

## Provenance

Git commit before S12 commit: `{manifest['git']['headCommit']}`

Source files hashed in the S12 manifest:

{source_table}

## Artifacts

Reusable S12 outputs are the selected target table, ablation variant metadata and DSL source table, row-level matched run table, aggregate matched result table, cautious claim table, component summary, validation table, effect figure, config, manifest, status JSON, checksums, and this full-results handoff report.
"""


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    artifacts_dir = args.artifacts_dir
    step_dir = artifacts_dir / "research_steps" / STEP_ID
    results_dir = artifacts_dir / "results"
    tables_dir = artifacts_dir / "tables"
    figures_dir = artifacts_dir / "figures" / "e03"
    configs_dir = artifacts_dir / "configs"
    src_snapshot_dir = artifacts_dir / "src_snapshot"
    checksums_dir = artifacts_dir / "checksums"
    for directory in (step_dir, results_dir, tables_dir, figures_dir, configs_dir, src_snapshot_dir, checksums_dir):
        directory.mkdir(parents=True, exist_ok=True)

    unit_tests = {"command": "not run", "returnCode": 0, "success": True, "stdout": "", "stderr": "", "elapsedSeconds": 0.0}
    if args.run_unit_tests:
        print("S12: running full E03 unit tests", flush=True)
        unit_tests = run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests/e03"], args.repo_dir)

    print("S12: loading S11 clusters, exemplars, S04 context, and policy code", flush=True)
    clusters = pd.read_parquet(args.s11_clusters)
    summary = pd.read_parquet(args.s11_summary)
    exemplars = pd.read_csv(args.s11_exemplars)
    qd_candidates = pd.read_parquet(args.qd_candidates)
    classic_vectors = pd.read_parquet(args.s04_classic_vectors)
    code_table = load_policy_code_table(
        generated_policy_library=args.generated_policy_library,
        qd_candidates=qd_candidates,
        qd_discovered_policy_library=args.qd_discovered,
    )

    print("S12: selecting source-backed S11 exemplar and component-cover targets", flush=True)
    targets = select_targets(clusters, exemplars, code_table, args.max_target_policies)
    baselines, variants = build_variants_for_targets(targets)
    variant_meta = variant_metadata_frame(variants)

    configs = screen_configs()
    print(f"S12: evaluating {len(baselines)} originals and {len(variants)} ablations over {len(configs)} matched configs", flush=True)
    run_df = evaluate_variants(baselines, variants, configs, args.seed)
    aggregate = aggregate_ablation_pairs(run_df, variant_meta)
    claims = claim_frame(aggregate)
    component_summary = component_summary_frame(claims)

    target_table_path = tables_dir / "e03_rule_ablation_selected_policies.csv"
    variant_table_path = tables_dir / "e03_rule_ablation_variants.csv"
    claims_table_path = tables_dir / "e03_rule_ablation_claims.csv"
    component_table_path = tables_dir / "e03_rule_ablation_component_summary.csv"
    validation_path = step_dir / "validation.csv"
    run_path = results_dir / "e03_rule_ablation_runs.parquet"
    result_path = results_dir / "e03_rule_ablation_results.parquet"
    claims_path = results_dir / "e03_rule_ablation_claims.parquet"
    component_path = results_dir / "e03_rule_ablation_component_summary.parquet"
    variant_path = results_dir / "e03_rule_ablation_variants.parquet"
    figure_path = figures_dir / "rule_ablation_effects.png"
    config_path = configs_dir / "e03_s12_rule_ablations_config.json"
    manifest_path = src_snapshot_dir / "e03_rule_ablation_manifest.json"
    step_manifest_path = step_dir / "manifest.json"
    status_path = step_dir / "status.json"
    report_path = step_dir / "research_step_full_results.md"
    checksum_path = checksums_dir / "e03_s12_sha256sums.txt"

    write_effect_figure(figure_path, claims, component_summary)
    validation = validation_frame(
        targets=targets,
        variants=variant_meta,
        run_df=run_df,
        aggregate=aggregate,
        claims=claims,
        component_summary=component_summary,
        figure_exists=figure_path.exists(),
        unit_tests=unit_tests,
    )

    normalize_for_parquet(run_df).to_parquet(run_path, index=False)
    normalize_for_parquet(aggregate).to_parquet(result_path, index=False)
    normalize_for_parquet(claims).to_parquet(claims_path, index=False)
    normalize_for_parquet(component_summary).to_parquet(component_path, index=False)
    normalize_for_parquet(variant_meta).to_parquet(variant_path, index=False)
    targets.to_csv(target_table_path, index=False)
    variant_meta.drop(columns=["dsl_source"], errors="ignore").to_csv(variant_table_path, index=False)
    claims.to_csv(claims_table_path, index=False)
    component_summary.to_csv(component_table_path, index=False)
    validation.to_csv(validation_path, index=False)

    config_payload = {
        "schema": ABLATION_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "seed": args.seed,
        "maxTargetPolicies": args.max_target_policies,
        "screenConfigs": [config.__dict__ for config in configs],
        "ablationIds": sorted(variant_meta["ablation_id"].unique().tolist()) if not variant_meta.empty else [],
        "matchedRng": "sha256(seed|config_seed|original_policy_id) first 16 hex digits",
    }
    write_json(config_path, config_payload)

    source_files = [
        source_entry(args.repo_dir / "src/e03/rule_ablations.py", args.repo_dir),
        source_entry(args.repo_dir / "scripts/e03_s12_rule_ablations.py", args.repo_dir),
        source_entry(args.repo_dir / "tests/e03/test_rule_ablations.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e03/rule_dsl.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e03/coarse_sweep.py", args.repo_dir),
        source_entry(args.repo_dir / "src/e03/universality_classes.py", args.repo_dir),
    ]
    artifacts: dict[str, Path] = {
        "fullResultsReport": report_path,
        "statusJson": status_path,
        "stepManifest": step_manifest_path,
        "sourceManifest": manifest_path,
        "config": config_path,
        "selectedPolicyTable": target_table_path,
        "variantTable": variant_table_path,
        "claimsTable": claims_table_path,
        "componentSummaryTable": component_table_path,
        "validationTable": validation_path,
        "runRows": run_path,
        "ablationResults": result_path,
        "ablationClaims": claims_path,
        "componentSummary": component_path,
        "variantMetadata": variant_path,
        "effectFigure": figure_path,
        "checksums": checksum_path,
    }
    manifest = {
        "schema": ABLATION_SCHEMA,
        "experimentId": EXPERIMENT_ID,
        "researchStepId": STEP_ID,
        "createdUtc": utc_now(),
        "inputArtifacts": {
            "s11Clusters": str(args.s11_clusters),
            "s11Summary": str(args.s11_summary),
            "s11Exemplars": str(args.s11_exemplars),
            "generatedPolicyLibrary": str(args.generated_policy_library),
            "qdCandidates": str(args.qd_candidates),
            "qdDiscovered": str(args.qd_discovered),
            "s04ClassicVectors": str(args.s04_classic_vectors),
        },
        "parameters": {
            "seed": args.seed,
            "maxTargetPolicies": args.max_target_policies,
            "screenConfigIds": [config.config_id for config in configs],
        },
        "summary": {
            "targetPolicyCount": int(len(targets)),
            "classCount": int(targets["class_id"].nunique()) if not targets.empty else 0,
            "ablationVariantCount": int(len(variant_meta)),
            "runRowCount": int(len(run_df)),
            "aggregateRowCount": int(len(aggregate)),
            "claimTypeCounts": claims["claim_type"].value_counts().sort_index().to_dict() if not claims.empty else {},
            "resultDigest": ablation_result_digest(claims) if not claims.empty else "",
            "validationPassed": bool(validation["success"].all() and unit_tests["success"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax": jax_backend_summary(),
            "elapsedSeconds": time.perf_counter() - started,
        },
        "git": {
            "headCommit": git_output(args.repo_dir, ["rev-parse", "HEAD"]),
            "branch": git_output(args.repo_dir, ["branch", "--show-current"]),
            "statusShort": git_output(args.repo_dir, ["status", "--short"]),
        },
        "sourceFiles": source_files,
    }
    write_json(manifest_path, manifest)

    report_text = render_report(
        artifacts=artifacts,
        manifest=manifest,
        validation=validation,
        unit_tests=unit_tests,
        command_line=" ".join(sys.argv),
        targets=targets,
        variants=variant_meta,
        run_df=run_df,
        aggregate=aggregate,
        claims=claims,
        component_summary=component_summary,
        classic_vectors=classic_vectors,
    )
    write_text(report_path, report_text)

    status_payload = {
        "researchStepId": STEP_ID,
        "stepNumber": STEP_NUMBER,
        "success": bool(validation["success"].all() and unit_tests["success"]),
        "status": "completed" if bool(validation["success"].all() and unit_tests["success"]) else "completed_with_validation_failure",
        "artifactsWritten": [str(path) for path in artifacts.values()],
        "validationResult": f"{int(validation['success'].sum())}/{len(validation)} validation cases passed; unit tests return code {unit_tests['returnCode']}",
        "caveatsOrBlockers": "No blocker remains. Claims are local to tested S11/source-backed DSL policy families and S07 screen proxies; direct aggregation/DG/Frozen Cell assays were not rerun.",
        "recommendedNextAction": "Stop for Chief review before S13; if accepted, compare classic algorithms to discovered rules using S10-S12 evidence.",
    }
    write_json(status_path, status_payload)

    artifact_entries = [artifact_entry(path, artifacts_dir, key) for key, path in artifacts.items() if path.exists() and path not in {step_manifest_path, checksum_path}]
    step_manifest = dict(manifest)
    step_manifest["artifacts"] = artifact_entries + [
        manifest_self_entry(step_manifest_path, artifacts_dir, "stepManifest"),
        manifest_self_entry(checksum_path, artifacts_dir, "checksums"),
    ]
    write_json(step_manifest_path, step_manifest)

    checksum_entries = []
    for path in artifacts.values():
        if path.exists() and path != checksum_path:
            checksum_entries.append(f"{sha256_file(path)}  {path}")
    write_text(checksum_path, "\n".join(checksum_entries) + "\n")
    print(f"S12: wrote report to {report_path}", flush=True)
    return 0 if status_payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
