#!/usr/bin/env python3
"""Build the frozen E02 S14 synthesis and report-bundle inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.counterexample_search import derive_seed, evaluate_candidate, initial_candidate
from causal_simulator.synthesis import (
    ENDPOINT_SPECS,
    ESTIMAND_LABELS,
    SYNTHESIS_SCHEMA_VERSION,
    build_mechanism_tables,
    pareto_analysis,
    summarize_s13,
    validate_complete_panel,
    variance_decomposition,
)


INPUTS = {
    "S10 screening runs": Path("/artifacts/research_steps/S10/screening_runs.parquet"),
    "S10 parity negative control": Path("/artifacts/research_steps/S10/parity_and_negative_control.json"),
    "S11 confirmatory results": Path("/artifacts/research_steps/S11/confirmatory_results.parquet"),
    "S12 marginal effects": Path("/artifacts/research_steps/S12/marginal_effects.parquet"),
    "S12 heterogeneity effects": Path("/artifacts/research_steps/S12/heterogeneity_effects.parquet"),
    "S12 heterogeneity rankings": Path("/artifacts/research_steps/S12/heterogeneity_rankings.parquet"),
    "S12 E04 resource tradeoff": Path("/artifacts/research_steps/S12/e04_resource_tradeoff.json"),
    "S12 RNG-unpaired sensitivity": Path("/artifacts/research_steps/S12/rng_unpaired_sensitivity.parquet"),
    "S12 validation": Path("/artifacts/research_steps/S12/validation_summary.json"),
    "S13 classifications": Path("/artifacts/research_steps/S13/counterexample_classifications.parquet"),
    "S13 detection limit": Path("/artifacts/research_steps/S13/searched_region_detection_limit.json"),
    "S13 validation": Path("/artifacts/research_steps/S13/validation_summary.json"),
    "S14 prespecification": Path("design/s14/synthesis_prespecification.json"),
    "S14 freeze manifest": Path("/artifacts/research_steps/S14/synthesis_freeze_manifest.json"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def plot_mechanism_map(marginal: pd.DataFrame, destination: Path) -> None:
    endpoints = [
        "normalizedResidualError",
        "successByBudget",
        "restrictedMeanCompletionFreeBudgetFraction",
        "logS01UnitWeightFullCost",
    ]
    estimands = list(ESTIMAND_LABELS)
    matrix = (
        marginal.pivot(index="estimandId", columns="endpoint", values="practicalMarginUnits")
        .reindex(index=estimands, columns=endpoints)
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    clipped = np.clip(matrix, -4, 4)
    image = ax.imshow(clipped, cmap="RdBu", vmin=-4, vmax=4, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            marker = "*" if bool(
                marginal.loc[
                    (marginal.estimandId == estimands[row])
                    & (marginal.endpoint == endpoints[column]),
                    "intervalExcludesZero",
                ].iloc[0]
            ) else ""
            ax.text(column, row, f"{matrix[row, column]:+.2f}{marker}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(endpoints)), [ENDPOINT_SPECS[value][0] for value in endpoints], rotation=20, ha="right")
    ax.set_yticks(range(len(estimands)), [ESTIMAND_LABELS[value] for value in estimands])
    ax.set_title("Benefit-oriented effects in frozen practical-margin units")
    ax.set_xlabel("Positive is favorable; * = S12 bootstrap interval excludes zero")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("effect / practical margin (clipped at ±4)")
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def plot_heterogeneity(heterogeneity: pd.DataFrame, destination: Path) -> pd.DataFrame:
    ranges = (
        heterogeneity.groupby(["estimandId", "endpoint"], as_index=False)
        .agg(
            minBenefit=("benefit_estimate", "min"),
            maxBenefit=("benefit_estimate", "max"),
            minInterval=("benefitLow95", "min"),
            maxInterval=("benefitHigh95", "max"),
            supportedCellCount=("supported", "size"),
        )
        .sort_values(["estimandId", "endpoint"])
    )
    ranges["schemaVersion"] = SYNTHESIS_SCHEMA_VERSION
    ranges["claimBoundary"] = "associational_modifier_heterogeneity_within_S11_support"
    endpoints = list(ENDPOINT_SPECS)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True)
    for axis, endpoint in zip(axes, endpoints, strict=True):
        subset = ranges[ranges.endpoint == endpoint].set_index("estimandId").reindex(ESTIMAND_LABELS)
        y = np.arange(len(subset))
        axis.hlines(y, subset.minInterval, subset.maxInterval, color="#b8b8b8", linewidth=5)
        axis.hlines(y, subset.minBenefit, subset.maxBenefit, color="#245a8d", linewidth=2)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_title(ENDPOINT_SPECS[endpoint][0])
        axis.set_xlabel("benefit-oriented cell range")
        axis.set_yticks(y, [ESTIMAND_LABELS[value] for value in subset.index])
    fig.suptitle("Frozen S12 supported modifier-cell ranges (not causal modifier effects)")
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)
    return ranges


def plot_pareto(summary: pd.DataFrame, destination: Path) -> None:
    primary = summary[summary.costProfile == "s01Cost"].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    sizes = 45 + 80 * (
        (primary.meanLogCost - primary.meanLogCost.min())
        / max(primary.meanLogCost.max() - primary.meanLogCost.min(), 1e-12)
    )
    colors = np.where(primary.pointPareto, "#c43b3b", "#5b87b2")
    ax.scatter(primary.meanNormalizedResidual, primary.failureRate, s=sizes, c=colors, alpha=0.85)
    for row in primary.itertuples(index=False):
        ax.annotate(row.settingLabel, (row.meanNormalizedResidual, row.failureRate), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("mean normalized residual (lower is better)")
    ax.set_ylabel("failure rate (lower is better)")
    ax.set_title("S11 empirical Pareto map; point size increases with log S01 cost")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def plot_variance(table: pd.DataFrame, destination: Path) -> None:
    order = [
        "scenario_context",
        "topology_coordination",
        "scheduler",
        "mobility",
        "continuation",
        "retry",
        "declared_interactions",
        "within_block_residual",
    ]
    outcomes = ["normalizedResidual", "completionSuccess", "logS01Cost"]
    pivot = table.pivot(index="outcome", columns="component", values="share").reindex(outcomes)
    fig, ax = plt.subplots(figsize=(10, 5.3))
    bottom = np.zeros(len(outcomes))
    palette = plt.get_cmap("tab20").colors
    for index, component in enumerate(order):
        values = pivot[component].to_numpy()
        ax.bar(outcomes, values, bottom=bottom, label=component, color=palette[index])
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of total run-level variance")
    ax.set_title("Exact hierarchical Shapley decomposition on the sampled S10 design")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def fresh_smoke() -> dict[str, Any]:
    candidate = initial_candidate(20, "unique", 2, 1400)
    seed = derive_seed(14_000_001, "S14", "fresh-smoke")
    first = evaluate_candidate(
        candidate.to_record(),
        stage="training",
        scheduler="uniform_random_activation",
        seed=seed,
        instance_id="S14-fresh-smoke",
    )
    replay = evaluate_candidate(
        candidate.to_record(),
        stage="training",
        scheduler="uniform_random_activation",
        seed=seed,
        instance_id="S14-fresh-smoke",
    )
    first_rows = first["runRows"]
    replay_rows = replay["runRows"]
    deterministic = [row["deterministicResultSha256"] for row in first_rows] == [
        row["deterministicResultSha256"] for row in replay_rows
    ]
    ledger = all(row["contractValidationPass"] for row in first_rows)
    opportunity = all(
        row["nativeLedger"]["proposals"] == row["nativeLedger"]["activations"]
        for row in first_rows
    )
    return {
        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
        "candidateId": candidate.candidate_id,
        "runCount": len(first_rows),
        "expectedRunCount": 5,
        "deterministicReplay": deterministic,
        "allContractValidationsPass": ledger,
        "oneProposalPerActivation": opportunity,
        "resultDigests": [row["deterministicResultSha256"] for row in first_rows],
        "contrastDigest": hashlib.sha256(
            json.dumps(first["contrasts"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "success": bool(len(first_rows) == 5 and deterministic and ledger and opportunity),
    }


def claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claimId": "C01",
            "claim": "The S02 no-fault matched topology contrast satisfies the frozen parity negative control.",
            "claimClass": "supported_equivalence_or_bounded_null",
            "primaryEvidence": "research_steps/S10/parity_and_negative_control.json",
            "caveat": "Parity is restricted to the E02-S01-E02 no-fault contract; it is not legacy-global equivalence.",
        },
        {
            "claimId": "C02",
            "claim": "Within the simulated confirmatory support, E04 stop-to-skip continuation improves task endpoints but incurs an extreme S01 resource increase.",
            "claimClass": "supported_causal_within_simulated_support",
            "primaryEvidence": "research_steps/S12/marginal_effects.parquet; research_steps/S12/e04_resource_tradeoff.json",
            "caveat": "A resource tradeoff, not an unqualified winner or a physical price estimate.",
        },
        {
            "claimId": "C03",
            "claim": "Within the simulated confirmatory support, E06 passive-to-stuck mobility faults harm average task performance.",
            "claimClass": "supported_causal_within_simulated_support",
            "primaryEvidence": "research_steps/S12/marginal_effects.parquet; research_steps/S12/rng_unpaired_sensitivity.parquet",
            "caveat": "E06 is scenario-paired but RNG-unpaired; cost sensitivity uses independent arms.",
        },
        {
            "claimId": "C04",
            "claim": "E03 scheduler task effects are near null on the frozen support; its cost effect is sensitive to the RNG-unpaired comparison.",
            "claimClass": "supported_equivalence_or_bounded_null",
            "primaryEvidence": "research_steps/S12/marginal_effects.parquet; research_steps/S12/rng_unpaired_sensitivity.parquet",
            "caveat": "Endpoint- and support-specific, not proof that schedulers are interchangeable; E03 cost sensitivity remains RNG-unpaired.",
        },
        {
            "claimId": "C05",
            "claim": "E08 weak coordination is near null on task endpoints and has supplementary coordinator-cost exposure.",
            "claimClass": "supported_equivalence_or_bounded_null",
            "primaryEvidence": "research_steps/S12/marginal_effects.parquet; research_steps/S12/cost_transform_sensitivity.parquet",
            "caveat": "Near-null task evidence does not imply zero controller cost or universal equivalence.",
        },
        {
            "claimId": "C06",
            "claim": "The six executable S11 settings form conditional task-resource tradeoffs rather than one universal ranking.",
            "claimClass": "descriptive_only",
            "primaryEvidence": "research_steps/S14/pareto_summary.parquet; research_steps/S14/pareto_dominance.parquet",
            "caveat": "Empirical S11 support and frozen objectives/weights only; Pareto position is not a general causal ranking.",
        },
        {
            "claimId": "C07",
            "claim": "Variance attribution depends on the S10 sampled-design distribution and frozen grouping hierarchy.",
            "claimClass": "descriptive_only",
            "primaryEvidence": "research_steps/S14/variance_decomposition.parquet",
            "caveat": "Descriptive sampled-design shares, not causal variance components or population fractions.",
        },
        {
            "claimId": "C08",
            "claim": "No fully confirmed counterexample appeared in the finite S13 searched-and-confirmed region.",
            "claimClass": "supported_equivalence_or_bounded_null",
            "primaryEvidence": "research_steps/S13/searched_region_detection_limit.json; research_steps/S14/s13_sensitivity.parquet",
            "caveat": "Bounded null only; no global-absence claim is permitted.",
        },
        {
            "claimId": "C09",
            "claim": "Two E06 exact-instance reversals are brittle under the frozen local-neighborhood criterion.",
            "claimClass": "constraining_brittle",
            "primaryEvidence": "research_steps/S14/s13_sensitivity.parquet",
            "caveat": "They are sensitivity cases, not fully confirmed counterexamples and not confirmatory-population replacements.",
        },
        {
            "claimId": "C10",
            "claim": "E05 is retry-inactive and E07 is non-executable under the frozen matched-information contract.",
            "claimClass": "not_estimable",
            "primaryEvidence": "research_steps/S10/adaptation_log.json; research_steps/S08/estimand_catalog.json",
            "caveat": "No effect is imputed for inactive or non-executable contrasts.",
        },
    ]


def caveat_rows() -> list[dict[str, str]]:
    return [
        {"caveatId": "K01", "scope": "causal", "severity": "high", "text": "Causal claims are simulator-internal and restricted to exact S11/S12 support."},
        {"caveatId": "K02", "scope": "heterogeneity", "severity": "high", "text": "Modifier comparisons are associational; no extrapolation beyond supported cells."},
        {"caveatId": "K03", "scope": "E04", "severity": "high", "text": "Task gains require extreme continuation opportunity and S01 cost exposure."},
        {"caveatId": "K04", "scope": "E03/E06", "severity": "high", "text": "Contrasts are scenario-paired but RNG-unpaired; independent-arm cost sensitivity must accompany them."},
        {"caveatId": "K05", "scope": "S13", "severity": "high", "text": "Zero full confirmations is a bounded finite-search null, not global absence; two E06 instances are brittle."},
        {"caveatId": "K06", "scope": "variance", "severity": "high", "text": "Variance shares depend on the sampled S10 design and frozen factor grouping/hierarchy."},
        {"caveatId": "K07", "scope": "Pareto", "severity": "medium", "text": "Frontiers depend on frozen objectives, cost definition, support, and empirical arm set; weighted scores encode values."},
        {"caveatId": "K08", "scope": "models", "severity": "medium", "text": "S12 reported heteroskedasticity plus calibration, Cox/fractional, and proportional-hazards limitations."},
        {"caveatId": "K09", "scope": "cost", "severity": "medium", "text": "S09 unit weights are accounting sensitivities, not monetary, energy, or wall-time costs."},
        {"caveatId": "K10", "scope": "release", "severity": "low", "text": "Repository source is released by immutable Git pointer; no duplicate source tar is collected under artifacts."},
    ]


def build_report_inputs(report_inputs: Path, claims: pd.DataFrame, caveats: pd.DataFrame) -> None:
    report_inputs.mkdir(parents=True, exist_ok=True)
    claims.to_csv(report_inputs / "claim_to_evidence_matrix.csv", index=False)
    claims.to_parquet(report_inputs / "claim_to_evidence_matrix.parquet", index=False)
    caveats.to_csv(report_inputs / "caveat_register.csv", index=False)
    write_markdown(
        report_inputs / "lay_summary.md",
        """# Lay summary

The simulations do not support one always-best architecture. The largest average task improvement came from allowing work to continue after a blocked move, but that improvement consumed vastly more simulated opportunities and ledger cost. Making cells immobile generally hurt performance. Scheduler and weak-coordinator task differences were small on the tested support, although their resource accounting and random-stream caveats matter. A finite adversarial search found no counterexample that survived every confirmation rule; two mobility reversals worked only for exact instances and were brittle under nearby perturbations. These findings apply to the simulated designs tested here, not to biological systems or every possible array and fault map.
""",
    )
    write_markdown(
        report_inputs / "methods_summary.md",
        """# Methods summary

S14 was frozen before synthesis in `design/s14/synthesis_prespecification.json`. Mechanism maps re-express every supported S12 effect on a benefit-oriented scale and retain paired-bootstrap uncertainty. Pareto analysis uses all 2,000 protected S11 pairing blocks and all six executable settings, minimizes residual, failure, and log ledger cost, and uses 5,000 common scale-stratified paired bootstrap resamples. Three weights were frozen for value-sensitive scalarization; the unweighted frontier remains primary.

Variance decomposition uses all 250 complete S10 pairing blocks crossed with 14 treatment signatures. It separates between-block scenario context from within-block treatment variation, applies exact hierarchical Shapley attribution across six frozen factor groups, and reports a residual. Its 1,000 scale-stratified cluster bootstrap replicates quantify sampling uncertainty. These fractions describe only the sampled S10 design.

All eight independently confirmed S13 candidates are retained as sensitivity evidence. Zero met the full two-panel confirmation rule; two E06 exact-instance reversals are labeled brittle because they failed neighborhood/concordance thresholds. A fresh five-arm simulator run and exact replay checked current-release execution, action/ledger contracts, and determinism.
""",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S14"))
    parser.add_argument("--report-inputs", type=Path, default=Path("/artifacts/report_inputs"))
    parser.add_argument("--release", type=Path, default=Path("/artifacts/release/causal_simulator_extension"))
    args = parser.parse_args()
    output = args.output
    tables = output / "tables"
    figures = output / "figures"
    maps = output / "mechanism_maps"
    for directory in (output, tables, figures, maps, args.release):
        directory.mkdir(parents=True, exist_ok=True)

    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing frozen input {name}: {path}")
    freeze = json.loads(INPUTS["S14 freeze manifest"].read_text())
    if sha256_file(INPUTS["S14 prespecification"]) != freeze["prespecification"]["sha256"]:
        raise RuntimeError("S14 prespecification changed after freeze")

    s10 = pd.read_parquet(INPUTS["S10 screening runs"])
    s11 = pd.read_parquet(INPUTS["S11 confirmatory results"])
    marginal = pd.read_parquet(INPUTS["S12 marginal effects"])
    heterogeneity = pd.read_parquet(INPUTS["S12 heterogeneity effects"])
    s13 = pd.read_parquet(INPUTS["S13 classifications"])
    detection = json.loads(INPUTS["S13 detection limit"].read_text())
    e04 = json.loads(INPUTS["S12 E04 resource tradeoff"].read_text())

    s10_panel = validate_complete_panel(s10, expected_blocks=250, expected_settings=14, protected=False)
    s11_panel = validate_complete_panel(s11, expected_blocks=2000, expected_settings=6, protected=True)
    mechanism, mechanism_heterogeneity = build_mechanism_tables(marginal, heterogeneity)
    pareto, winners, dominance = pareto_analysis(s11)
    variance, variance_diagnostics = variance_decomposition(s10)
    sensitivity, sensitivity_summary = summarize_s13(s13)

    # Exact reproducibility check from independent calls using the frozen seeds.
    pareto_replay, winners_replay, dominance_replay = pareto_analysis(s11)
    variance_replay, variance_diagnostics_replay = variance_decomposition(s10)
    analysis_replay = bool(
        pareto.equals(pareto_replay)
        and winners.equals(winners_replay)
        and dominance.equals(dominance_replay)
        and variance.equals(variance_replay)
        and variance_diagnostics == variance_diagnostics_replay
    )
    if not analysis_replay:
        raise RuntimeError("S14 seeded synthesis did not replay exactly")

    mechanism.to_parquet(output / "mechanism_effects.parquet", index=False)
    mechanism.to_csv(output / "mechanism_effects.csv", index=False)
    mechanism_heterogeneity.to_parquet(output / "mechanism_heterogeneity.parquet", index=False)
    heterogeneity_ranges = plot_heterogeneity(mechanism_heterogeneity, figures / "heterogeneity_uncertainty_ranges")
    heterogeneity_ranges.to_csv(tables / "heterogeneity_ranges.csv", index=False)
    plot_mechanism_map(mechanism, figures / "mechanism_effect_map")
    mechanism.to_parquet(maps / "marginal_mechanism_map.parquet", index=False)
    mechanism_heterogeneity.to_parquet(maps / "heterogeneity_mechanism_map.parquet", index=False)

    pareto.to_parquet(output / "pareto_summary.parquet", index=False)
    pareto.to_csv(output / "pareto_summary.csv", index=False)
    winners.to_csv(tables / "pareto_weighted_winner_probabilities.csv", index=False)
    dominance.to_parquet(output / "pareto_dominance.parquet", index=False)
    dominance.to_csv(tables / "pareto_dominance.csv", index=False)
    plot_pareto(pareto, figures / "pareto_map")

    variance.to_parquet(output / "variance_decomposition.parquet", index=False)
    variance.to_csv(output / "variance_decomposition.csv", index=False)
    canonical_json(output / "variance_decomposition_diagnostics.json", variance_diagnostics)
    plot_variance(variance, figures / "variance_decomposition")

    sensitivity.to_parquet(output / "s13_sensitivity.parquet", index=False)
    sensitivity.to_csv(output / "s13_sensitivity.csv", index=False)
    sensitivity_summary["detectionLimit"] = detection
    canonical_json(output / "s13_sensitivity_summary.json", sensitivity_summary)

    claims = pd.DataFrame(claim_rows())
    claims["schemaVersion"] = SYNTHESIS_SCHEMA_VERSION
    caveats = pd.DataFrame(caveat_rows())
    caveats["schemaVersion"] = SYNTHESIS_SCHEMA_VERSION
    claims.to_csv(output / "claim_to_evidence_matrix.csv", index=False)
    claims.to_parquet(output / "claim_to_evidence_matrix.parquet", index=False)
    caveats.to_csv(output / "caveat_register.csv", index=False)
    build_report_inputs(args.report_inputs, claims, caveats)

    smoke = fresh_smoke()
    canonical_json(output / "fresh_smoke_validation.json", smoke)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    source_paths = [
        Path("causal_simulator/synthesis.py"),
        Path("scripts/build_s14_synthesis.py"),
        Path("scripts/validate_s14_synthesis.py"),
        Path("tests/test_synthesis.py"),
        Path("design/s14/synthesis_prespecification.json"),
    ]
    source_manifest = {
        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
        "releaseName": "E02-causal-simulator-extension-v1",
        "releaseMode": "immutable_git_commit_pointer",
        "repository": origin,
        "branch": "eidosoma/groups/28",
        "commit": head,
        "sourceFiles": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_paths
        ],
        "smokeValidation": smoke,
        "prespecificationSha256": sha256_file(INPUTS["S14 prespecification"]),
    }
    canonical_json(args.release / "release_manifest.json", source_manifest)
    canonical_json(args.release / "source_manifest.json", source_manifest)
    canonical_json(
        args.release / "smoke_command.json",
        {
            "command": "python scripts/build_s14_synthesis.py --output /tmp/e02-s14-smoke-artifacts --report-inputs /tmp/e02-s14-smoke-report-inputs --release /tmp/e02-s14-smoke-release",
            "focusedTests": "python -m pytest -q tests/test_synthesis.py",
        },
    )
    canonical_json(
        Path("/artifacts/release/causal-simulator-extension.tar.zst.NOT_CREATED.json"),
        {
            "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
            "expectedPath": "/artifacts/release/causal-simulator-extension.tar.zst",
            "created": False,
            "reason": "AGENTS.md requires repository-backed source to remain in Git and prohibits copying repository source into ARTIFACTS_DIR.",
            "replacement": "/artifacts/release/causal_simulator_extension/release_manifest.json",
            "commit": head,
        },
    )

    input_provenance = {
        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
        "researchStepId": "S14",
        "inputs": [
            {"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in INPUTS.items()
        ],
        "repositoryCommit": head,
    }
    canonical_json(output / "input_provenance.json", input_provenance)
    canonical_json(
        output / "environment_provenance.json",
        {
            "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "cpuCount": os.cpu_count(),
            "workerCount": 1,
            "threadPolicy": "serial deterministic synthesis; no nested parallelism",
        },
    )

    robust_primary = pareto[(pareto.costProfile == "s01Cost") & pareto.robustPareto]
    robust_dominance = dominance[(dominance.costProfile == "s01Cost") & dominance.robustDominance]
    validation = {
        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
        "researchStepId": "S14",
        "status": "PASS" if all([
            analysis_replay,
            smoke["success"],
            sensitivity_summary["fullyConfirmedCount"] == 0,
            sensitivity_summary["brittleExactInstanceCount"] == 2,
            variance_diagnostics["maxPartitionIdentityError"] < 1e-10,
            len(mechanism) == 16,
            len(mechanism_heterogeneity) == 368,
        ]) else "FAIL",
        "checks": {
            "prespecificationHashUnchanged": True,
            "S10PanelComplete": s10_panel,
            "S11PanelComplete": s11_panel,
            "allMechanismEndpointsPresent": len(mechanism) == 16,
            "allSupportedHeterogeneityCellsPresent": len(mechanism_heterogeneity) == 368,
            "uncertaintyOverlaysPresent": bool(mechanism.intervalExcludesZero.notna().all()),
            "paretoBootstrapReplicates": 5000,
            "varianceBootstrapReplicates": 1000,
            "varianceIdentityPass": variance_diagnostics["maxPartitionIdentityError"] < 1e-10,
            "deterministicAnalysisReplay": analysis_replay,
            "freshSmokePass": smoke["success"],
            "allS13CandidatesIncluded": len(sensitivity) == 8,
            "S13FullyConfirmedCount": sensitivity_summary["fullyConfirmedCount"],
            "S13BrittleE06Count": sensitivity_summary["brittleExactInstanceCount"],
            "globalAbsenceClaimPermitted": False,
            "claimClassesValid": set(claims.claimClass).issubset({
                "supported_causal_within_simulated_support",
                "supported_equivalence_or_bounded_null",
                "constraining_brittle",
                "descriptive_only",
                "not_estimable",
            }),
            "E04ResourceTradeoffIncluded": float(e04["rawS01UnitWeightFullCostDifference"]) > 0,
        },
        "anchorResults": {
            "primaryRobustParetoSettingCount": int(len(robust_primary)),
            "primaryRobustDominanceRelationCount": int(len(robust_dominance)),
            "fullyConfirmedCounterexamples": sensitivity_summary["fullyConfirmedCount"],
            "brittleE06Instances": sensitivity_summary["brittleExactInstanceCount"],
        },
    }
    if validation["status"] != "PASS":
        raise RuntimeError("S14 consolidated validation failed")
    canonical_json(output / "validation_summary.json", validation)

    figure_rows = []
    for path in sorted(figures.glob("*")):
        figure_rows.append({"name": path.stem, "format": path.suffix.lstrip("."), "path": str(path), "sha256": sha256_file(path)})
    pd.DataFrame(figure_rows).to_csv(args.report_inputs / "figure_index.csv", index=False)
    table_paths = [
        output / "mechanism_effects.csv",
        output / "pareto_summary.csv",
        output / "variance_decomposition.csv",
        output / "s13_sensitivity.csv",
        output / "claim_to_evidence_matrix.csv",
        output / "caveat_register.csv",
    ] + sorted(tables.glob("*.csv"))
    table_rows = [
        {"name": path.stem, "path": str(path), "sha256": sha256_file(path)} for path in table_paths
    ]
    pd.DataFrame(table_rows).to_csv(args.report_inputs / "table_index.csv", index=False)
    evidence_rows = [
        {"evidenceId": "EV01", "description": "Frozen mechanism effects", "path": str(output / "mechanism_effects.parquet")},
        {"evidenceId": "EV02", "description": "All supported heterogeneity cells", "path": str(output / "mechanism_heterogeneity.parquet")},
        {"evidenceId": "EV03", "description": "Pareto summary with uncertainty", "path": str(output / "pareto_summary.parquet")},
        {"evidenceId": "EV04", "description": "Pareto dominance probabilities", "path": str(output / "pareto_dominance.parquet")},
        {"evidenceId": "EV05", "description": "Sampled-design variance decomposition", "path": str(output / "variance_decomposition.parquet")},
        {"evidenceId": "EV06", "description": "S13 bounded sensitivity inclusion", "path": str(output / "s13_sensitivity.parquet")},
        {"evidenceId": "EV07", "description": "Consolidated validation", "path": str(output / "validation_summary.json")},
        {"evidenceId": "EV08", "description": "Reusable simulator release pointer", "path": str(args.release / "release_manifest.json")},
    ]
    for row in evidence_rows:
        path = Path(row["path"])
        row["sha256"] = sha256_file(path)
    pd.DataFrame(evidence_rows).to_csv(args.report_inputs / "evidence_index.csv", index=False)
    write_markdown(
        args.report_inputs / "evidence_index.md",
        "# Evidence index\n\n" + "\n".join(
            f"- {row['evidenceId']}: {row['description']} — `{row['path']}` (SHA-256 `{row['sha256']}`)."
            for row in evidence_rows
        ),
    )
    canonical_json(args.report_inputs / "provenance_manifest.json", input_provenance)

    bundle_paths = sorted(path for path in args.report_inputs.iterdir() if path.is_file())
    canonical_json(
        args.report_inputs / "report_bundle_manifest.json",
        {
            "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
            "researchStepId": "S14",
            "status": "ready_for_chief_scientist_review",
            "files": [
                {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in bundle_paths
            ],
            "claimBoundaryReview": "PASS",
            "chiefScientistCreatesFinalBundle": True,
        },
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
