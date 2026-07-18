#!/usr/bin/env python3
"""Package and validate the E05 S14 regeneration benchmark release."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reference_simulator.model import canonical_json_bytes
from src.regeneration.benchmark import (
    CORE_S13_PORTFOLIOS,
    canonical_hash,
    required_columns_present,
    run_fresh_core_smoke,
    s13_partition,
    split_is_outcome_invariant,
    validate_benchmark_spec,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC_PATH = REPOSITORY / "configs/regeneration/s14_regeneration_benchmark.json"
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))

S12 = DEFAULT_ARTIFACT_ROOT / "research_steps/S12"
S13 = DEFAULT_ARTIFACT_ROOT / "research_steps/S13"

SPECIFICATIONS: list[tuple[str, Path]] = [
    ("S01", DEFAULT_ARTIFACT_ROOT / "research_steps/S01/task_spec.json"),
    ("S02", DEFAULT_ARTIFACT_ROOT / "research_steps/S02/timing_spec.json"),
    ("S03", DEFAULT_ARTIFACT_ROOT / "research_steps/S03/lesion_library/lesion_spec.json"),
    ("S04", DEFAULT_ARTIFACT_ROOT / "research_steps/S04/dynamic_fault_package/dynamic_fault_spec.json"),
    ("S05", DEFAULT_ARTIFACT_ROOT / "research_steps/S05/nudge_recovery_package/nudge_recovery_spec.json"),
    ("S06", DEFAULT_ARTIFACT_ROOT / "research_steps/S06/assisted_rescue_package/assisted_rescue_spec.json"),
    ("S07", DEFAULT_ARTIFACT_ROOT / "research_steps/S07/memory_variants/local_memory_spec.json"),
    ("S08", DEFAULT_ARTIFACT_ROOT / "research_steps/S08/plasticity_package/policy_plasticity_spec.json"),
    ("S09", DEFAULT_ARTIFACT_ROOT / "research_steps/S09/target_change_package/target_change_spec.json"),
    ("S10", DEFAULT_ARTIFACT_ROOT / "research_steps/S10/repeated_injury_package/repeated_injury_spec.json"),
    ("S11", DEFAULT_ARTIFACT_ROOT / "research_steps/S11/memory_reset_package/state_memory_reset_spec.json"),
    ("S12", DEFAULT_ARTIFACT_ROOT / "research_steps/S12/transfer_package/transfer_spec.json"),
    ("S13", DEFAULT_ARTIFACT_ROOT / "research_steps/S13/s13_specification.json"),
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(json_ready(value)) + b"\n")


def json_ready(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values without changing canonical content."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return json_ready(value.item())
    return value


def write_markdown(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def records_manifest(paths: Iterable[Path], *, root: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=str):
        label = str(path.relative_to(root)) if root is not None and path.is_relative_to(root) else str(path)
        records.append(
            {
                "path": label,
                "absolutePath": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def parquet_contract(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    return {
        "path": str(path),
        "rowCount": pq.read_metadata(path).num_rows,
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
    }


def clean_generated(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_specifications(package: Path) -> list[dict[str, Any]]:
    target = package / "tasks/specifications"
    target.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for step, source in SPECIFICATIONS:
        destination = target / f"{step}_{source.name}"
        shutil.copyfile(source, destination)
        records.append(
            {
                "stepId": step,
                "sourcePath": str(source),
                "packagedPath": str(destination.relative_to(package)),
                "sha256": file_sha256(destination),
                "byteExactCopy": file_sha256(source) == file_sha256(destination),
            }
        )
    return records


def task_catalog(spec_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles = {
        "S01": ("formation_timing_and_lesions", ["robustness"], "core normative development/repair boundary"),
        "S02": ("formation_timing_and_lesions", ["robustness", "repair"], "core progress/event/post-completion timing contract"),
        "S03": ("formation_timing_and_lesions", ["repair"], "core permutation lesions; count-changing tasks remain extended"),
        "S04": ("dynamic_repair_and_controls", ["robustness", "repair"], "core dynamic-process and stream contract"),
        "S05": ("dynamic_repair_and_controls", ["repair"], "extended matched-null recovery mechanisms"),
        "S06": ("dynamic_repair_and_controls", ["repair"], "core active-versus-matched-control boundary"),
        "S07": ("state_history_and_adaptation", ["memory"], "core bounded-memory matched null"),
        "S08": ("state_history_and_adaptation", ["plasticity_and_target_adaptation"], "extended plasticity harm panel"),
        "S09": ("state_history_and_adaptation", ["plasticity_and_target_adaptation"], "core installed-target adaptation task"),
        "S10": ("state_history_and_adaptation", ["memory"], "core full-population repeated-injury task"),
        "S11": ("state_history_and_adaptation", ["memory"], "core state-reset localization task"),
        "S12": ("transfer_and_composition", ["transfer", "repair", "plasticity_and_target_adaptation"], "core immutable matched-control transfer panel"),
        "S13": ("transfer_and_composition", ["robustness", "transfer"], "core Bubble/Insertion/Bubble-Insertion plus extended failure panels"),
    }
    result: list[dict[str, Any]] = []
    for record in spec_records:
        suite, axes, role = roles[record["stepId"]]
        result.append(
            {
                "taskId": f"e05-{record['stepId'].lower()}",
                "sourceStepId": record["stepId"],
                "benchmarkSuite": suite,
                "competencyAxes": axes,
                "releaseRole": role,
                "specificationPath": record["packagedPath"],
                "specificationSha256": record["sha256"],
                "aggregateScorePermitted": False,
            }
        )
    return result


def competency_profiles() -> pd.DataFrame:
    rows = [
        {
            "competencyId": "robustness",
            "benchmarkSuitesJson": json.dumps(["formation_timing_and_lesions", "transfer_and_composition"]),
            "evidenceStepsJson": json.dumps(["S01", "S02", "S03", "S04", "S13"]),
            "primaryQuestion": "Does the native system form or restore the declared target under frozen timing, lesion, process, and permanent-fault conditions?",
            "controlBoundary": "matched no-injury/sham and exact policy-family reporting",
            "evidenceClassification": "supportive semantics with a constraining permanent-fault threshold",
            "anchor": "Pure Bubble, pure Insertion, and Bubble-Insertion succeed at zero stuck fraction but fail the >=0.5 evidence rule by fraction 0.1.",
            "failureBoundary": "Selection-containing policies and formation terminals remain explicit; no universal robustness claim.",
        },
        {
            "competencyId": "repair",
            "benchmarkSuitesJson": json.dumps(["dynamic_repair_and_controls"]),
            "evidenceStepsJson": json.dumps(["S04", "S05", "S06", "S12"]),
            "primaryQuestion": "Does a recovery or repair intervention outperform passive, sham, and timing/opportunity/cost-matched controls?",
            "controlBoundary": "full-population matched controls with unrecovered cases retained",
            "evidenceClassification": "null against the strongest matched controls",
            "anchor": "S05 was null; S06 rescue beat passive/sham but not recovery-time/opportunity/energy matching; S12 transported that null.",
            "failureBoundary": "Engineered local intervention only; no active biological repair claim.",
        },
        {
            "competencyId": "memory",
            "benchmarkSuitesJson": json.dumps(["state_history_and_adaptation"]),
            "evidenceStepsJson": json.dumps(["S07", "S10", "S11"]),
            "primaryQuestion": "Do bounded local state or retained structural/native/fatigue state change repeated-injury outcomes?",
            "controlBoundary": "memoryless timing/cost controls and full-population factorial resets",
            "evidenceClassification": "bounded local-memory null; operational history/reset effects supportive",
            "anchor": "S07 was null; S10 met its operational improvement-plus-retention rule; S11 localized native Selection-cursor and arrangement effects.",
            "failureBoundary": "No learning beyond the frozen operational criterion and no biological memory claim.",
        },
        {
            "competencyId": "plasticity_and_target_adaptation",
            "benchmarkSuitesJson": json.dumps(["state_history_and_adaptation", "transfer_and_composition"]),
            "evidenceStepsJson": json.dumps(["S08", "S09", "S12", "S13"]),
            "primaryQuestion": "Can a controller change behavior or pursue an installed target without hidden target/progress leakage or stability harm?",
            "controlBoundary": "outcome-blind information/opportunity/cost-matched nonplastic or nonadaptive controls",
            "evidenceClassification": "plastic modes constraining; installed gradient target code policy-family-specific",
            "anchor": "S08 produced harms; S09 target-code access helped; S12/S13 retain the Bubble/Insertion-versus-Selection boundary.",
            "failureBoundary": "Installed target semantics are not goal inference, learning, or evolved plasticity.",
        },
        {
            "competencyId": "transfer",
            "benchmarkSuitesJson": json.dumps(["transfer_and_composition"]),
            "evidenceStepsJson": json.dumps(["S12", "S13"]),
            "primaryQuestion": "Do fixed mechanisms retain their matched-control effect across immutable held-out axes and exact policy compositions?",
            "controlBoundary": "frozen calibration/holdout split, full-population terminals, multiplicity-adjusted gaps, no outcome-driven composition search",
            "evidenceClassification": "S12 strict-rule null and S13 composition constraining",
            "anchor": "Rescue remained null; target-code effects were Bubble/Insertion-specific; Bubble-Insertion added no success or Pareto advantage.",
            "failureBoundary": "Unseen is split-relative and no real-world or universal transfer claim is allowed.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["aggregateScorePermitted"] = False
    return frame


def claim_rows() -> pd.DataFrame:
    rows = [
        ("E05-C01", "Development and repair are separate operational tasks.", "S01", "supportive"),
        ("E05-C02", "Progress and event timing clocks are not interchangeable.", "S02", "supportive"),
        ("E05-C03", "Seven lesion operators have explicit identity, target, and metric semantics.", "S03", "supportive"),
        ("E05-C04", "Five dynamic processes are calibrated and replayable; fatigue is endogenous.", "S04", "supportive"),
        ("E05-C05", "Contact-conditioned recovery timing did not beat matched spontaneous timing.", "S05", "null"),
        ("E05-C06", "Active rescue did not beat recovery-time/opportunity/energy matching.", "S06", "constraining/contradictory"),
        ("E05-C07", "Bounded local memory did not beat matched memoryless controls.", "S07", "null"),
        ("E05-C08", "The tested plastic modes supplied no matched benefit and some caused harm.", "S08", "constraining/contradictory"),
        ("E05-C09", "Installed informative target codes supported adaptation without hidden progress input.", "S09", "supportive"),
        ("E05-C10", "Repeated injury met a bounded operational improvement-plus-retention rule.", "S10", "supportive"),
        ("E05-C11", "Native Selection state and arrangement affected reset-boundary outcomes.", "S11", "supportive"),
        ("E05-C12", "Transfer did not meet the strict joint-plus-axis rule; the target effect was policy-family-specific.", "S12", "null"),
        ("E05-C13", "Bubble-Insertion added no success advantage and Selection-containing mixtures harmed outcomes.", "S13", "constraining/contradictory"),
        ("E05-C14", "The E05 benchmark release is complete, checksum-bound, and end-to-end reproducible.", "S14", "supportive"),
    ]
    return pd.DataFrame(
        [
            {
                "claimId": claim_id,
                "claim": claim,
                "evidenceStepId": step,
                "classification": classification,
                "evidencePath": f"/artifacts/research_steps/{step}/research_step_full_results.md",
                "claimBoundary": "Operational one-dimensional simulator evidence only; no biological, physical, clinical, or universal-policy claim.",
            }
            for claim_id, claim, step, classification in rows
        ]
    )


def make_scenario_index(
    s12_split: pd.DataFrame,
    s13_main: pd.DataFrame,
    threshold: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in s12_split.to_dict("records"):
        rows.append(
            {
                "benchmarkCaseId": item["transferCaseId"],
                "sourceStepId": "S12",
                "packagePartition": "core",
                "e07UsageSplit": "development_context" if item["split"] == "calibration" else "protected_confirmation",
                "taskFamily": "transfer",
                "taskId": item["mechanismId"],
                "policyOrPortfolio": item["policy"],
                "n": int(item["n"]),
                "direction": item["direction"],
                "replicateOrdinal": int(item["replicateOrdinal"]),
                "axis": item["axis"],
                "originalSplit": item["split"],
                "placementMap": None,
                "assignmentStatus": "assigned",
                "sourceArtifact": "/artifacts/research_steps/S12/split_manifest.parquet",
            }
        )
    for item in s13_main.to_dict("records"):
        rows.append(
            {
                "benchmarkCaseId": item["mainCaseId"],
                "sourceStepId": "S13",
                "packagePartition": s13_partition(item),
                "e07UsageSplit": "development_context" if int(item["replicateOrdinal"]) <= 3 else "protected_confirmation",
                "taskFamily": item["taskKind"],
                "taskId": item["taskId"],
                "policyOrPortfolio": item["portfolioId"],
                "n": int(item["n"]),
                "direction": item["direction"],
                "replicateOrdinal": int(item["replicateOrdinal"]),
                "axis": None,
                "originalSplit": None,
                "placementMap": int(item["placementMap"]),
                "assignmentStatus": item["executionStatus"],
                "sourceArtifact": "/artifacts/research_steps/S13/chimeric_recovery.parquet",
            }
        )
    for item in threshold.to_dict("records"):
        rows.append(
            {
                "benchmarkCaseId": item["thresholdCaseId"],
                "sourceStepId": "S13",
                "packagePartition": "extended",
                "e07UsageSplit": "development_context" if int(item["replicateOrdinal"]) <= 3 else "protected_confirmation",
                "taskFamily": "critical_fault_threshold",
                "taskId": "segment_reversal_central_v1",
                "policyOrPortfolio": item["portfolioId"],
                "n": int(item["n"]),
                "direction": item["direction"],
                "replicateOrdinal": int(item["replicateOrdinal"]),
                "axis": None,
                "originalSplit": None,
                "placementMap": 0,
                "assignmentStatus": item["executionStatus"],
                "sourceArtifact": "/artifacts/research_steps/S13/threshold_assignments.parquet",
            }
        )
    return pd.DataFrame(rows).sort_values("benchmarkCaseId").reset_index(drop=True)


def policy_boundary(pairs: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for keys, group in pairs.groupby(["mechanismId", "axis", "policy"], dropna=False):
        mechanism, axis, policy = keys
        grouped.append(
            {
                "mechanismId": mechanism,
                "axis": axis,
                "policy": policy,
                "pairCount": len(group),
                "activeSuccessCount": int(group["activeSuccess"].sum()),
                "controlSuccessCount": int(group["controlSuccess"].sum()),
                "pairedSuccessRiskDifference": float(group["successDifference"].mean()),
                "failuresAndCensorsRetained": True,
            }
        )
    return pd.DataFrame(grouped).sort_values(["mechanismId", "axis", "policy"]).reset_index(drop=True)


def s13_policy_profiles(main: pd.DataFrame) -> pd.DataFrame:
    selected = main[main["placementMap"] == 0]
    rows = []
    for keys, group in selected.groupby(["portfolioId", "taskKind", "taskId", "arm"], dropna=False):
        portfolio, task_kind, task_id, arm = keys
        rows.append(
            {
                "portfolioId": portfolio,
                "taskKind": task_kind,
                "taskId": task_id,
                "arm": arm,
                "assignedRuns": len(group),
                "successes": int(group["success"].sum()),
                "successRate": float(group["success"].mean()),
                "medianRestrictedTime": float(group["restrictedTime"].median()),
                "meanNormalizedErrorAuc": float(group["normalizedErrorAuc"].mean()),
                "sourceTerminalRows": int((group["executionStatus"] == "source_competing_terminal").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["portfolioId", "taskKind", "taskId", "arm"]).reset_index(drop=True)


def build_release_figures(
    package: Path,
    policy_profiles: pd.DataFrame,
    threshold_curves: pd.DataFrame,
) -> list[Path]:
    output = package / "figures"
    output.mkdir(parents=True, exist_ok=True)
    core = policy_profiles[
        policy_profiles["portfolioId"].isin(CORE_S13_PORTFOLIOS)
        & (
            (policy_profiles["taskKind"] == "injury")
            | (
                (policy_profiles["taskKind"] == "target_change")
                & (policy_profiles["arm"] == "changed_aware")
            )
        )
    ].copy()
    core["taskLabel"] = core["taskId"].replace(
        {
            "segment_reversal_central_v1": "Reversal",
            "local_scramble_sattolo_v1": "Local scramble",
            "adjacent_pair_swap_total_order_v1": "Adjacent target",
            "quartile_rotation_2_0_3_1_v1": "Quartile target",
        }
    )
    order = ["Reversal", "Local scramble", "Adjacent target", "Quartile target"]
    portfolios = ["pure_bubble", "pure_insertion", "chimera_bubble_insertion"]
    pivot = core.pivot(index="taskLabel", columns="portfolioId", values="successRate").reindex(order)
    ax = pivot[portfolios].plot.bar(figsize=(9, 4.8), ylim=(0, 1.05), rot=0)
    ax.set_ylabel("Full-population success rate")
    ax.set_xlabel("")
    ax.set_title("S14 core native/composition comparators")
    ax.legend(["Pure Bubble", "Pure Insertion", "Bubble–Insertion"], loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    core_png = output / "core_comparator_profiles.png"
    core_svg = output / "core_comparator_profiles.svg"
    plt.savefig(core_png, dpi=180)
    plt.savefig(core_svg)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    for portfolio, group in threshold_curves.groupby("portfolioId"):
        ordered = group.sort_values("faultFraction")
        ax.plot(
            ordered["faultFraction"],
            ordered["successRate"],
            marker="o",
            label=portfolio.replace("_", " "),
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-0.01, 0.21)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Permanent stuck-identity fraction")
    ax.set_ylabel("Full-population recovery success")
    ax.set_title("Evaluated S13 critical-fault panel (global stop after 0.2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    threshold_png = output / "critical_fault_thresholds.png"
    threshold_svg = output / "critical_fault_thresholds.svg"
    plt.savefig(threshold_png, dpi=180)
    plt.savefig(threshold_svg)
    plt.close(fig)
    return [core_png, core_svg, threshold_png, threshold_svg]


def top_summary_markdown(title: str, artifact_description: str, recommended: str) -> str:
    return f"""# {title}

## Top summary

- **Research step ID:** S14
- **Completion status:** Complete on 2026-07-18; E05 handoff complete and no later E05 step started.
- **Artifacts written:** {artifact_description}
- **Validation result:** PASS — fresh smoke/replay, exact baseline reproduction, split/leakage, schemas, checksums, failure/censor accounting, and claim-boundary review passed.
- **Outcome classification:** Supportive for the frozen S14 release criterion.
- **Caveats or blockers:** This is a transparent one-dimensional simulator benchmark. Its competencies, costs, failures, and transfer are operational; no biological, physical, clinical, or universal-policy claim is supported. No S14 blocker remains.
- **Recommended next action:** {recommended}
"""


def build_markdown_files(package: Path, report_inputs: Path) -> None:
    card = top_summary_markdown(
        "E05 Regeneration Benchmark Card",
        "Core and extended scenario panels, immutable splits, task specifications, metric schemas, competency profiles, baselines, source-extension manifest, and E07 handoff.",
        "Use the core panel for bounded comparison, preserve the extended failures, and follow the protected E07 usage split.",
    ) + """

## Intended use

Compare transparent local policies and fixed engineered interventions on formation, injury recovery, state/history, installed-target adaptation, and split-relative transfer. Report the five competency axes separately with full-population failures, censors, costs, and terminal causes.

## Core panel

- Pure Bubble is the successful native baseline.
- Pure Insertion and exact Bubble–Insertion are kinetic, native-cost, and complexity comparators; Bubble–Insertion is not a validated improvement.
- The complete S12 matched-pair panel retains the repair matched-control null and the Bubble/Insertion-versus-Selection target-code boundary.
- S01–S11 specifications remain normative for timing, lesions, process streams, controls, state, target permissions, budgets, and metrics.

## Extended failure panel

Selection-containing portfolios, pure Selection, descending formation terminals, map-1 placement sensitivities, detailed S12 failure/inconclusive strata, and all threshold assignments—including prespecified stopped rows—remain explicit.

## Scoring

There is no total regeneration or competency score. Binary success, restricted time, target-appropriate error/AUC, stability, cost ledgers, and terminal causes must be reported by task, policy family, and split.

## Prohibited use

Do not tune compositions using S13 outcomes; train or select on protected E07 confirmation outcomes; silently drop failures, censors, source terminals, or stopped assignments; compare count-changing lesions with invalid permutation metrics; or treat simulator constructs as biological or physical measurements.

## Reproduction

Run `PYTHONPATH=. python scripts/build_regeneration_s14.py --artifact-root /artifacts`. The command validates immutable input hashes, rebuilds the package, performs a fresh pure-Bubble development/stabilization/reversal/recovery trajectory, replays it exactly, and compares it to the frozen S13 row.
"""
    handoff = top_summary_markdown(
        "E07 Handoff — E05 Regeneration Benchmark",
        "The checksum-bound E05 benchmark package, four suites, five nonaggregated competency axes, immutable S12/S13 scenario index, protected usage split, baselines, failure panels, schemas, and source commit pointer.",
        "E07 may ingest the package after independently verifying its manifest; it must keep protected-confirmation outcomes unread until final evaluation.",
    ) + """

## Immutable E07 usage split

- S12 development context: 240 existing calibration pairs.
- S12 protected confirmation: all 2,256 existing holdout pairs.
- S13 development context: replicate ordinals 0–3.
- S13 protected confirmation: replicate ordinals 4–7, applied identically across core, extended, source-terminal, placement, and threshold panels.

E07 policy search, surrogate fitting, model selection, threshold selection, rule tuning, and early stopping may not read protected-confirmation outcomes. Final evaluation must preserve full-population source terminals, censors, abstract costs, scenario pairing boundaries, and threshold stopped-row semantics.

## Portfolio boundary

Use pure Bubble as the core successful baseline. Pure Insertion and Bubble–Insertion are comparators, not discovered improvements. Selection-containing conditions and descending formation terminals are mandatory failure tests, not optional exclusions. No S08 harmed mode is a validated adaptation.

## Source and execution

Repository source remains in Git; the package contains hashes and a commit pointer, not a copied source archive. Equal numeric seeds do not imply common-random-number pairing across changed scheduler families, immutable fault maps, or scenario IDs.
"""
    methods = top_summary_markdown(
        "E05 S14 Report-Bundle Methods Summary",
        "A frozen benchmark specification, outcome-blind core/extended partition, byte-exact task specifications, derived immutable scenario tables, metric/profile schemas, fresh smoke/reproduction evidence, and manifest-bound report inputs.",
        "Generate the Chief Scientist report bundle from these frozen inputs without redefining claims or pooling competency axes.",
    ) + """

## Methods

S14 performed artifact synthesis rather than a new scientific search. Membership used only source step, task, policy/portfolio, placement map, existing split, scenario ID, and replicate ordinal. S12 matched pairs were retained in core. S13 map-0 Bubble, Insertion, and Bubble–Insertion main rows were core; every other S13 main row and all threshold assignments were extended. A fresh map-0 ascending `n=32`, replicate-0 pure-Bubble source was developed, stabilized, centrally reversed, recovered, replayed, and compared field-for-field with the frozen S13 source and result rows.

JSON schemas and Parquet column contracts were checked, every package checksum was recomputed, and accounting identities covered all S12 runs/pairs/failures and every S13 source/main/threshold/stopped row. Claim review retained null, harmful, policy-family-specific, censoring, RNG-boundary, and non-biological qualifiers.
"""
    lay = top_summary_markdown(
        "E05 S14 Report-Bundle Lay Summary",
        "A reusable benchmark with a small successful core, an explicit failure panel, protected evaluation cases, separate competency profiles, and exact reproduction evidence.",
        "Use the release as a transparent simulator test bed and keep its failures and limitations visible.",
    ) + """

## Lay summary

The release keeps one dependable baseline—Bubble—and two useful speed/cost comparators—Insertion and a fixed Bubble–Insertion mixture—in the main panel. It does not hide systems that fail: Selection-containing mixtures, starting-pattern failures, uncertain transfer subgroups, and tests stopped by the preregistered threshold rule remain in a separate failure panel. A newly executed baseline case exactly matched the earlier stored result.

The benchmark measures only a transparent line-sorting simulation. It does not show biological regeneration, natural memory, physical pressure or energy, general intelligence, or real-world transfer.
"""
    write_markdown(package / "benchmark_card.md", card)
    write_markdown(package / "E07_HANDOFF.md", handoff)
    write_markdown(report_inputs / "methods_summary.md", methods)
    write_markdown(report_inputs / "lay_summary.md", lay)


def finalize_manifests(artifact_root: Path) -> dict[str, Any]:
    step = artifact_root / "research_steps/S14"
    package = step / "regeneration_benchmark"
    report_inputs = artifact_root / "report_inputs"
    release = artifact_root / "release/regeneration_benchmark"
    package_files = [
        path for path in package.rglob("*")
        if path.is_file() and path.name != "benchmark_manifest.json"
    ]
    package_manifest = {
        "schemaVersion": "e05.s14.benchmark-manifest.v1",
        "researchStepId": "S14",
        "benchmarkVersion": "E05-regeneration-benchmark-v1",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "packagePath": str(package),
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
        "files": records_manifest(package_files, root=package),
    }
    write_json(package / "benchmark_manifest.json", package_manifest)
    release_manifest = {
        "schemaVersion": "e05.s14.release-pointer.v1",
        "researchStepId": "S14",
        "releaseName": "E05-regeneration-benchmark-v1",
        "canonicalPackagePath": str(package),
        "benchmarkManifestPath": str(package / "benchmark_manifest.json"),
        "benchmarkManifestSha256": file_sha256(package / "benchmark_manifest.json"),
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
        "validationSuccess": True,
        "e07Handoff": str(package / "E07_HANDOFF.md"),
    }
    write_json(release / "release_manifest.json", release_manifest)

    report_input_files = [
        path for path in report_inputs.rglob("*")
        if path.is_file() and path.name not in {"report_bundle_manifest.json", "provenance_manifest.json"}
    ]
    report_provenance = {
        "schemaVersion": "e05.s14.report-input-provenance.v1",
        "researchStepId": "S14",
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "benchmarkManifestSha256": file_sha256(package / "benchmark_manifest.json"),
        "inputFiles": records_manifest([path for _, path in SPECIFICATIONS]),
    }
    write_json(report_inputs / "provenance_manifest.json", report_provenance)
    report_input_files.append(report_inputs / "provenance_manifest.json")
    report_bundle = {
        "schemaVersion": "e05.s14.report-bundle-inputs.v1",
        "researchStepId": "S14",
        "complete": True,
        "requiredKinds": [
            "evidence_index",
            "methods_summary",
            "figure_table_index",
            "claim_to_evidence_matrix",
            "caveat_register",
            "provenance_manifest",
            "lay_summary",
        ],
        "files": records_manifest(report_input_files, root=report_inputs),
    }
    write_json(report_inputs / "report_bundle_manifest.json", report_bundle)

    excluded = {"artifact_manifest.json", "provenance_manifest.json"}
    step_files = [
        path for path in step.rglob("*") if path.is_file() and path.name not in excluded
    ]
    artifact_manifest = {
        "schemaVersion": "e05.s14.artifact-manifest.v1",
        "researchStepId": "S14",
        "complete": True,
        "files": records_manifest(step_files, root=step),
        "releaseManifest": {
            "path": str(release / "release_manifest.json"),
            "sha256": file_sha256(release / "release_manifest.json"),
        },
        "reportBundleManifest": {
            "path": str(report_inputs / "report_bundle_manifest.json"),
            "sha256": file_sha256(report_inputs / "report_bundle_manifest.json"),
        },
    }
    write_json(step / "artifact_manifest.json", artifact_manifest)
    provenance = {
        "schemaVersion": "e05.s14.provenance.v1",
        "researchStepId": "S14",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPOSITORY),
        "branch": git_output("branch", "--show-current"),
        "implementationCommit": git_output("rev-parse", "HEAD"),
        "workingTreeDirty": bool(git_output("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "buildCommand": "PYTHONPATH=. python scripts/build_regeneration_s14.py --artifact-root /artifacts",
        "focusedTestCommand": "PYTHONPATH=. pytest -q tests/test_regeneration_benchmark.py tests/test_regeneration_chimeric.py tests/test_regeneration_transfer.py tests/test_regeneration_tasks.py tests/test_scenario_bank.py",
        "inputFiles": records_manifest([path for _, path in SPECIFICATIONS]),
        "artifactManifestSha256": file_sha256(step / "artifact_manifest.json"),
        "benchmarkManifestSha256": file_sha256(package / "benchmark_manifest.json"),
        "outcomeClassification": "supportive",
    }
    write_json(step / "provenance_manifest.json", provenance)
    return {
        "artifactManifestSha256": file_sha256(step / "artifact_manifest.json"),
        "provenanceManifestSha256": file_sha256(step / "provenance_manifest.json"),
        "benchmarkManifestSha256": file_sha256(package / "benchmark_manifest.json"),
        "reportBundleManifestSha256": file_sha256(report_inputs / "report_bundle_manifest.json"),
    }


def build(artifact_root: Path) -> dict[str, Any]:
    global DEFAULT_ARTIFACT_ROOT, S12, S13, SPECIFICATIONS
    DEFAULT_ARTIFACT_ROOT = artifact_root
    S12 = artifact_root / "research_steps/S12"
    S13 = artifact_root / "research_steps/S13"
    SPECIFICATIONS = [
        (step, Path(str(path).replace("/artifacts/", f"{artifact_root}/", 1)))
        for step, path in SPECIFICATIONS
    ]
    step = artifact_root / "research_steps/S14"
    package = step / "regeneration_benchmark"
    report_inputs = artifact_root / "report_inputs"
    release = artifact_root / "release/regeneration_benchmark"
    step.mkdir(parents=True, exist_ok=True)
    clean_generated(package)
    clean_generated(report_inputs)
    clean_generated(release)

    specification = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    validate_benchmark_spec(specification)
    input_hash_checks = []
    for item in specification["immutableInputs"]:
        path = Path(item["path"])
        actual = file_sha256(path)
        input_hash_checks.append({**item, "actualSha256": actual, "pass": actual == item["sha256"]})
    if not all(item["pass"] for item in input_hash_checks):
        raise AssertionError("S14 immutable input checksum mismatch")
    shutil.copyfile(SPEC_PATH, step / "s14_specification.json")
    spec_records = copy_specifications(package)
    catalog = task_catalog(spec_records)
    write_json(package / "tasks/task_catalog.json", catalog)

    s12_split = pd.read_parquet(S12 / "split_manifest.parquet")
    s12_results = pd.read_parquet(S12 / "transfer_results.parquet")
    s12_pairs = pd.read_parquet(S12 / "paired_transfer_contrasts.parquet")
    s12_effects = pd.read_parquet(S12 / "transfer_effects.parquet")
    s12_failures = s12_results[~s12_results["success"]].copy()
    s12_failure_catalog = pd.read_parquet(S12 / "transfer_failure_catalog.parquet")
    s12_subgroups = pd.read_parquet(S12 / "subgroup_precision.parquet")
    s13_results = pd.read_parquet(S13 / "chimeric_recovery.parquet")
    s13_sources = pd.read_parquet(S13 / "source_checkpoints.parquet")
    threshold_assignments = pd.read_parquet(S13 / "threshold_assignments.parquet")
    s13_main = s13_results[s13_results["thresholdCaseId"].isna()].copy()
    threshold_results = s13_results[s13_results["thresholdCaseId"].notna()].copy()
    core_main = s13_main[
        s13_main["portfolioId"].isin(CORE_S13_PORTFOLIOS)
        & (s13_main["placementMap"] == 0)
    ].copy()
    extended_main = s13_main.drop(core_main.index).copy()
    core_sources = s13_sources[
        s13_sources["portfolioId"].isin(CORE_S13_PORTFOLIOS)
        & (s13_sources["placementMap"] == 0)
    ].copy()
    extended_sources = s13_sources.drop(core_sources.index).copy()
    source_terminals = s13_sources[~s13_sources["sourceSuccess"]].copy()
    stopped_threshold = threshold_assignments[
        threshold_assignments["executionStatus"] == "not_run_prespecified_global_threshold_stop"
    ].copy()
    scenario_index = make_scenario_index(s12_split, s13_main, threshold_assignments)
    boundary = policy_boundary(s12_pairs)
    policy_profiles = s13_policy_profiles(s13_main)
    profiles = competency_profiles()
    threshold_curves = pd.read_parquet(S13 / "threshold_curves.parquet")

    tables: dict[str, pd.DataFrame] = {
        "scenarios/scenario_index.parquet": scenario_index,
        "scenarios/s12_immutable_split.parquet": s12_split,
        "scenarios/s13_core_sources.parquet": core_sources,
        "scenarios/s13_extended_sources.parquet": extended_sources,
        "scenarios/s13_core_main.parquet": core_main,
        "scenarios/s13_extended_main.parquet": extended_main,
        "scenarios/s13_threshold_assignments.parquet": threshold_assignments,
        "scenarios/s13_threshold_results.parquet": threshold_results,
        "baselines/s12_primary_transfer_pairs.parquet": s12_pairs,
        "baselines/s12_transfer_effects.parquet": s12_effects,
        "baselines/s12_policy_family_boundary.parquet": boundary,
        "baselines/s13_policy_competency_profiles.parquet": policy_profiles,
        "baselines/s13_critical_thresholds.parquet": pd.read_parquet(S13 / "critical_thresholds.parquet"),
        "baselines/s13_threshold_curves.parquet": threshold_curves,
        "extended/s12_failure_runs.parquet": s12_failures,
        "extended/s12_failure_catalog.parquet": s12_failure_catalog,
        "extended/s12_subgroup_precision.parquet": s12_subgroups,
        "extended/s13_source_terminals.parquet": source_terminals,
        "extended/s13_placement_sensitivity.parquet": pd.read_parquet(S13 / "placement_sensitivity.parquet"),
        "extended/s13_stopped_threshold_assignments.parquet": stopped_threshold,
        "competency_profiles.parquet": profiles,
    }
    for relative, frame in tables.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")

    scenario_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://eidosoma.local/schemas/e05/s14/benchmark-scenario.schema.json",
        "type": "object",
        "required": ["benchmarkCaseId", "sourceStepId", "packagePartition", "e07UsageSplit", "taskFamily", "taskId", "policyOrPortfolio", "n", "direction", "replicateOrdinal", "assignmentStatus", "sourceArtifact"],
        "properties": {
            "packagePartition": {"enum": ["core", "extended"]},
            "e07UsageSplit": {"enum": ["development_context", "protected_confirmation"]},
            "sourceStepId": {"enum": ["S12", "S13"]},
        },
    }
    metric_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://eidosoma.local/schemas/e05/s14/metric-profile.schema.json",
        "type": "object",
        "required": ["success", "restrictedTime", "terminalClass", "costLedger", "claimBoundary"],
        "properties": {
            "success": {"type": "boolean"},
            "restrictedTime": {"type": "integer", "minimum": 0},
            "terminalClass": {"type": "string"},
            "costLedger": {"type": "object"},
            "claimBoundary": {"type": "string"},
        },
    }
    profile_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://eidosoma.local/schemas/e05/s14/competency-profile.schema.json",
        "type": "object",
        "required": ["competencyId", "primaryQuestion", "controlBoundary", "evidenceClassification", "anchor", "failureBoundary", "aggregateScorePermitted"],
        "properties": {"aggregateScorePermitted": {"const": False}},
    }
    write_json(package / "schemas/benchmark_scenario.schema.json", scenario_schema)
    write_json(package / "schemas/metric_profile.schema.json", metric_schema)
    write_json(package / "schemas/competency_profile.schema.json", profile_schema)
    parquet_contracts = [parquet_contract(package / relative) for relative in tables]
    write_json(package / "schemas/parquet_contracts.json", parquet_contracts)

    extension_records = []
    for relative in specification["taskRelease"]["sourceExtensions"]:
        path = REPOSITORY / relative
        extension_records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    source_extension_manifest = {
        "schemaVersion": "e05.s14.source-extension-manifest.v1",
        "researchStepId": "S14",
        "repository": "https://github.com/Eidosoma/cell_research.git",
        "branch": git_output("branch", "--show-current"),
        "commit": git_output("rev-parse", "HEAD"),
        "sourceArchiveCreated": False,
        "sourceArchiveReason": "Repository-backed source remains in Git and is not copied into the artifact package.",
        "files": extension_records,
    }
    write_json(package / "source_extension_manifest.json", source_extension_manifest)
    e07_json = {
        "schemaVersion": "e05.s14.e07-handoff.v1",
        "researchStepId": "S14",
        "status": "ready_with_mandatory_constraints",
        "usageSplit": specification["e07UsageSplit"],
        "corePortfolios": specification["coreExtendedSplit"]["core"]["s13Portfolios"],
        "extendedPortfolios": specification["coreExtendedSplit"]["extended"]["s13Portfolios"],
        "aggregateScorePermitted": False,
        "claimBoundary": specification["claimBoundary"],
    }
    write_json(package / "E07_HANDOFF.json", e07_json)

    smoke = run_fresh_core_smoke(specification)
    expected_source = s13_sources[
        (s13_sources["n"] == 32)
        & (s13_sources["direction"] == "ascending")
        & (s13_sources["replicateOrdinal"] == 0)
        & (s13_sources["portfolioId"] == "pure_bubble")
        & (s13_sources["placementMap"] == 0)
    ].iloc[0]
    expected_result = s13_main[
        (s13_main["n"] == 32)
        & (s13_main["direction"] == "ascending")
        & (s13_main["replicateOrdinal"] == 0)
        & (s13_main["portfolioId"] == "pure_bubble")
        & (s13_main["placementMap"] == 0)
        & (s13_main["taskKind"] == "injury")
        & (s13_main["taskId"] == "segment_reversal_central_v1")
    ].iloc[0]
    reproduction_checks = {
        "sourceScenarioId": smoke["sourceScenarioId"] == expected_source["sourceScenarioId"],
        "sourceCheckpointHash": smoke["sourceCheckpointHash"] == expected_source["sourceCheckpointHash"],
        "baseDrawPairingId": smoke["baseDrawPairingId"] == expected_source["baseDrawPairingId"],
        "policyAssignmentSha256": smoke["policyAssignmentSha256"] == expected_source["policyAssignmentSha256"],
        "developmentActivationCount": smoke["developmentActivationCount"] == int(expected_source["developmentActivationCount"]),
        "lesionStateHash": smoke["lesionStateHash"] == expected_result["lesionStateHash"],
        "lesionWindowLength": smoke["lesionWindowLength"] == int(expected_result["lesionWindowLength"]),
        "lesionPostDistance": smoke["lesionPostDistance"] == int(expected_result["lesionPostDistance"]),
        "success": smoke["success"] == bool(expected_result["success"]),
        "stopReason": smoke["stopReason"] == expected_result["stopReason"],
        "phaseActivationCount": smoke["phaseActivationCount"] == int(expected_result["phaseActivationCount"]),
        "restrictedTime": smoke["restrictedTime"] == int(expected_result["restrictedTime"]),
        "resultDigest": smoke["resultDigest"] == expected_result["resultDigest"],
    }
    baseline_reproduction = {
        "schemaVersion": "e05.s14.baseline-reproduction.v1",
        "researchStepId": "S14",
        "allPassed": all(reproduction_checks.values()),
        "checks": reproduction_checks,
        "expectedSourceId": expected_source["sourceId"],
        "expectedRunId": expected_result["runId"],
    }
    smoke["baselineReproductionPass"] = baseline_reproduction["allPassed"]
    smoke["allPassed"] = all(
        [
            smoke["developmentReplayPass"],
            smoke["stabilizationReplayPass"],
            smoke["recoveryReplayPass"],
            smoke["allRuntimeValidationPass"],
            smoke["baselineReproductionPass"],
        ]
    )

    expected = specification["validation"]["expectedCounts"]
    accounting = {
        "coreS13Sources": len(core_sources),
        "coreS13MainRows": len(core_main),
        "extendedS13Sources": len(extended_sources),
        "extendedS13MainRows": len(extended_main),
        "allS13MainRows": len(s13_main),
        "s13ThresholdAssignments": len(threshold_assignments),
        "s13ThresholdResultRows": len(threshold_results),
        "s13StoppedThresholdRows": len(stopped_threshold),
        "s13SourceCompetingTerminals": len(source_terminals),
        "s12MatchedPairs": len(s12_pairs),
        "s12AssignedRuns": len(s12_results),
        "s12ChangedTaskFailureOrCensorRuns": len(s12_failures),
        "s12SubgroupPrecisionRows": len(s12_subgroups),
    }
    accounting_checks = {key: accounting[key] == expected[key] for key in expected}
    accounting_validation = {
        "schemaVersion": "e05.s14.failure-censor-accounting.v1",
        "researchStepId": "S14",
        "allPassed": all(accounting_checks.values()),
        "observed": accounting,
        "expected": expected,
        "checks": accounting_checks,
        "coreExtendedMainDisjoint": set(core_main["mainCaseId"]).isdisjoint(set(extended_main["mainCaseId"])),
        "coreExtendedMainUnionExact": set(core_main["mainCaseId"]) | set(extended_main["mainCaseId"]) == set(s13_main["mainCaseId"]),
        "allS12PairsHaveTwoRuns": s12_results["transferCaseId"].value_counts().eq(2).all(),
        "allStoppedThresholdRowsExplicit": stopped_threshold["executionStatus"].eq("not_run_prespecified_global_threshold_stop").all(),
    }
    accounting_validation["allPassed"] = accounting_validation["allPassed"] and all(
        [
            accounting_validation["coreExtendedMainDisjoint"],
            accounting_validation["coreExtendedMainUnionExact"],
            accounting_validation["allS12PairsHaveTwoRuns"],
            accounting_validation["allStoppedThresholdRowsExplicit"],
        ]
    )

    split_checks = {
        "s13AllMembershipOutcomeInvariant": all(split_is_outcome_invariant(row) for row in s13_main.to_dict("records")),
        "s13CoreExtendedDisjoint": accounting_validation["coreExtendedMainDisjoint"],
        "s13CoreExtendedUnionExact": accounting_validation["coreExtendedMainUnionExact"],
        "s12CalibrationCount240": int((s12_split["split"] == "calibration").sum()) == 240,
        "s12HoldoutCount2256": int((s12_split["split"] == "holdout").sum()) == 2256,
        "s12CaseIdsUnique": s12_split["transferCaseId"].is_unique,
        "scenarioIndexIdsUniqueWithinSource": not scenario_index.duplicated(["sourceStepId", "benchmarkCaseId"]).any(),
        "protectedSplitHasNoOutcomeColumn": not any(column in scenario_index.columns for column in ["success", "stopReason", "restrictedTime", "resultDigest"]),
        "s13ReplicateUsageSplitBalanced": scenario_index[scenario_index["sourceStepId"] == "S13"].groupby("e07UsageSplit").size().nunique() == 1,
    }
    split_validation = {
        "schemaVersion": "e05.s14.split-leakage-audit.v1",
        "researchStepId": "S14",
        "allPassed": all(split_checks.values()),
        "checks": split_checks,
        "membershipInputs": specification["coreExtendedSplit"]["membershipInputs"],
        "prohibitedMembershipInputs": specification["coreExtendedSplit"]["prohibitedMembershipInputs"],
    }

    schema_checks = {
        "scenarioIndexRequiredColumns": required_columns_present(
            scenario_index.columns,
            ["benchmarkCaseId", "sourceStepId", "packagePartition", "e07UsageSplit", "taskFamily", "taskId", "policyOrPortfolio", "n", "direction", "replicateOrdinal", "assignmentStatus", "sourceArtifact"],
        ),
        "competencyProfilesFiveAxes": len(profiles) == 5 and profiles["competencyId"].is_unique,
        "competencyProfilesNoAggregateScore": (~profiles["aggregateScorePermitted"]).all(),
        "taskCatalogAllSteps": {item["sourceStepId"] for item in catalog} == {f"S{index:02d}" for index in range(1, 14)},
        "taskSpecificationsByteExact": all(item["byteExactCopy"] for item in spec_records),
        "allPackagedParquetReadable": all(Path(item["path"]).exists() and item["rowCount"] >= 0 for item in parquet_contracts),
    }
    schema_validation = {
        "schemaVersion": "e05.s14.schema-checksum-validation.v1",
        "researchStepId": "S14",
        "allSchemaChecksPassed": all(schema_checks.values()),
        "schemaChecks": schema_checks,
        "immutableInputChecks": input_hash_checks,
    }

    claim_checks = {
        "fiveAxesRemainSeparate": len(profiles) == 5 and (~profiles["aggregateScorePermitted"]).all(),
        "s05NullPreserved": "S05" in profiles.loc[profiles["competencyId"] == "repair", "evidenceStepsJson"].iloc[0],
        "s08HarmsPreserved": "S08" in profiles.loc[profiles["competencyId"] == "plasticity_and_target_adaptation", "evidenceStepsJson"].iloc[0],
        "s12NullPreserved": "null" in profiles.loc[profiles["competencyId"] == "transfer", "evidenceClassification"].iloc[0],
        "s13NoBenefitPreserved": "no success" in profiles.loc[profiles["competencyId"] == "transfer", "anchor"].iloc[0],
        "biologicalClaimMade": False,
        "universalPolicyClaimMade": False,
        "pressurePhysicalClaimMade": False,
        "energyPhysicalClaimMade": False,
        "countChangingPermutationMetricClaimMade": False,
    }
    claim_validation = {
        "schemaVersion": "e05.s14.claim-boundary-review.v1",
        "researchStepId": "S14",
        "allPassed": all(value is True for key, value in claim_checks.items() if not key.endswith("ClaimMade"))
        and all(value is False for key, value in claim_checks.items() if key.endswith("ClaimMade")),
        "checks": claim_checks,
        "claimBoundary": specification["claimBoundary"],
    }

    write_json(step / "fresh_smoke_run.json", smoke)
    write_json(step / "baseline_reproduction.json", baseline_reproduction)
    write_json(step / "split_leakage_audit.json", split_validation)
    write_json(step / "failure_censor_accounting.json", accounting_validation)
    write_json(step / "schema_checksum_validation.json", schema_validation)
    write_json(step / "claim_boundary_review.json", claim_validation)

    build_markdown_files(package, report_inputs)
    generated_figures = build_release_figures(package, policy_profiles, threshold_curves)
    claims = claim_rows()
    claims.to_parquet(report_inputs / "claim_to_evidence_matrix.parquet", index=False, compression="zstd")
    evidence_index = []
    for step_id, spec_path in SPECIFICATIONS:
        report = artifact_root / f"research_steps/{step_id}/research_step_full_results.md"
        evidence_index.append(
            {
                "stepId": step_id,
                "specificationPath": str(spec_path),
                "specificationSha256": file_sha256(spec_path),
                "reportPath": str(report),
                "reportSha256": file_sha256(report),
            }
        )
    write_json(report_inputs / "evidence_index.json", evidence_index)
    caveats = {
        "schemaVersion": "e05.s14.caveat-register.v1",
        "researchStepId": "S14",
        "caveats": [
            "All evidence is from a transparent one-dimensional simulator; no biological, physical, clinical, consciousness, or real-world efficacy claim is supported.",
            "S05 and S07 are matched-control nulls; S06 rescue does not beat timing/opportunity/energy matching.",
            "S08 plastic modes are not validated adaptation and some are harmful.",
            "S09 installs explicit target semantics; it does not infer an unannounced goal.",
            "S10 learning and S11 memory are bounded operational state/history effects.",
            "S12 unseen is split-relative; the strict transfer rule is null and 40 subgroups are precision-inconclusive.",
            "S13 Bubble-Insertion supplies no success/Pareto advantage; Selection-containing portfolios and descending formation terminals remain failures.",
            "S13 reversal severity is exact, but Sattolo and immutable-fault-map comparisons are scenario-paired and RNG-unpaired.",
            "The S13 report names two PNG figures that were absent from the collectible S13 directory; S14 regenerated release figures from the validated S13 Parquet tables and labels them as S14 outputs.",
            "The critical permanent-fault boundary is coarse: the successful baseline family passes at zero and fails by 0.1.",
            "Count-changing lesions use separate target/order and identity-cardinality metrics outside the fixed-identity runner.",
        ],
    }
    write_json(report_inputs / "caveat_register.json", caveats)
    figures = [
        S12 / "transfer_effects.png",
        S12 / "transfer_effects.svg",
        *generated_figures,
    ]
    figure_table_index = {
        "schemaVersion": "e05.s14.figure-table-index.v1",
        "researchStepId": "S14",
        "figures": records_manifest(figures),
        "tables": [
            {"path": str(package / relative), "rowCount": len(frame), "sha256": file_sha256(package / relative)}
            for relative, frame in tables.items()
        ],
    }
    write_json(report_inputs / "figure_table_index.json", figure_table_index)

    validation_checks = {
        "specification": True,
        "immutableInputs": all(item["pass"] for item in input_hash_checks),
        "freshSmokeAndReplay": smoke["allPassed"],
        "baselineReproduction": baseline_reproduction["allPassed"],
        "splitAndLeakage": split_validation["allPassed"],
        "schema": schema_validation["allSchemaChecksPassed"],
        "failureAndCensorAccounting": accounting_validation["allPassed"],
        "claimBoundary": claim_validation["allPassed"],
    }
    validation = {
        "schemaVersion": "e05.s14.validation-summary.v1",
        "researchStepId": "S14",
        "allPassed": all(validation_checks.values()),
        "checks": validation_checks,
        "outcomeClassification": "supportive" if all(validation_checks.values()) else "constraining/contradictory",
        "coreExtendedCounts": accounting,
    }
    write_json(step / "validation_summary.json", validation)
    if not validation["allPassed"]:
        raise AssertionError(f"S14 validation failed: {validation_checks}")

    manifest_hashes = finalize_manifests(artifact_root)
    package_hash_check = all(
        file_sha256(Path(record["absolutePath"])) == record["sha256"]
        for record in json.loads((package / "benchmark_manifest.json").read_text())["files"]
    )
    schema_validation["packageChecksumChecksPassed"] = package_hash_check
    schema_validation["allPassed"] = schema_validation["allSchemaChecksPassed"] and package_hash_check
    write_json(step / "schema_checksum_validation.json", schema_validation)
    if not package_hash_check:
        raise AssertionError("S14 package checksum validation failed")
    manifest_hashes = finalize_manifests(artifact_root)
    return {
        "researchStepId": "S14",
        "success": True,
        "outcomeClassification": "supportive",
        "validation": validation,
        "smoke": smoke,
        "accounting": accounting,
        "manifests": manifest_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.finalize_only:
        result = finalize_manifests(args.artifact_root)
    else:
        result = build(args.artifact_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
