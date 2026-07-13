"""S10 efficiency replication and loss-explicit cost accounting.

The S09 run population is immutable input.  This module does not rerun the
simulators: it projects the complete clean-room ledger and the deliberately
lossy frozen-public-commit counters into explicitly named cost conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from reference_simulator import create_scenario
from reference_simulator.engine import initial_state
from reference_simulator.policies import cell_view_proposal, traditional_proposal


REPOSITORY = Path(__file__).resolve().parents[1]
S09_RUNS = Path("/artifacts/research_steps/S09/no_fault_runs.parquet")
PREREGISTRATION = REPOSITORY / "analysis" / "s10_efficiency_preregistration.json"
OUTPUT_SCHEMA = "e01.s10.cost_ledger.v1"
R_PROFILE = "R-clean-room-reference-E01-v1"
C_PROFILE = "C-frozen-public-commit"
POLICIES = ("Bubble", "Insertion", "Selection")
ALLOWED_SPLITS = ("paper_scale", "confirmatory_holdout")
PRIMARY_METRICS = (
    "swap_only_cost",
    "publication_swap_plus_comparison_cost",
)
FULL_REFERENCE_METRICS = (
    "swap_only_cost",
    "publication_swap_plus_comparison_cost",
    "observation_reads",
    "value_comparisons",
    "target_calculations",
    "proposals",
    "rejected_actions",
    "activations",
    "accepted_swaps",
    "displaced_cells",
    "unit_weight_full_ledger",
)
RANKING_METRICS = FULL_REFERENCE_METRICS + ("wall_time_seconds",)
SUMMARY_METRICS = (
    "activations",
    "observation_reads",
    "value_comparisons",
    "target_calculations",
    "proposals",
    "no_ops",
    "rejections",
    "rejected_actions",
    "memory_updates",
    "accepted_swaps",
    "displaced_cells",
    "conflict_losses",
    "swap_only_cost",
    "publication_swap_plus_comparison_cost",
    "unit_weight_full_ledger",
    "historical_compare_and_swap_probe",
    "wall_time_seconds",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def _int_or_nan(value: Any) -> float | int:
    return np.nan if pd.isna(value) else int(value)


def build_cost_ledger(runs: pd.DataFrame) -> pd.DataFrame:
    """Create one loss-explicit S10 cost row for every S09 run."""
    required = {
        "run_id", "backend_profile", "split", "architecture", "policy",
        "condition_id", "scenario_id", "base_draw_id", "pairing_block_id",
        "replicate_ordinal", "completed", "stop_reason", "timed_out",
        "protected", "successful_swap_count", "ledger_valueComparisons",
        "ledger_displacedCells", "elapsed_seconds", "historical_swap_probe",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"S09 input is missing columns: {missing}")

    rows: list[dict[str, Any]] = []
    identity_fields = (
        "run_id", "backend_profile", "publication_snapshot_claimed", "split",
        "protected", "condition_id", "scenario_id", "base_draw_id",
        "pairing_block_id", "replicate_ordinal", "architecture", "policy",
        "scheduler", "sequence_basis", "completed", "timed_out", "stop_reason",
        "event_budget", "runtime_seed", "generation_key", "elapsed_seconds",
    )
    for source in runs.to_dict(orient="records"):
        row = {name: source.get(name) for name in identity_fields}
        row.update(
            {
                "schema_version": OUTPUT_SCHEMA,
                "research_step_id": "S10",
                "source_s09_schema_version": source.get("schema_version"),
                "source_s09_run_id": source["run_id"],
                "wall_time_seconds": float(source["elapsed_seconds"]),
                "wall_time_in_algorithmic_cost": False,
            }
        )
        swaps = int(source["successful_swap_count"])
        if source["backend_profile"] == R_PROFILE:
            ledger = {
                "activations": int(source["ledger_activations"]),
                "observation_reads": int(source["ledger_observationReads"]),
                "value_comparisons": int(source["ledger_valueComparisons"]),
                "proposals": int(source["ledger_proposals"]),
                "no_ops": int(source["ledger_noOps"]),
                "rejections": int(source["ledger_rejections"]),
                "memory_updates": int(source["ledger_memoryUpdates"]),
                "accepted_swaps": int(source["ledger_acceptedSwaps"]),
                "displaced_cells": int(source["ledger_displacedCells"]),
                "conflict_losses": int(source["ledger_conflictLosses"]),
            }
            target_calculations = ledger["proposals"]
            rejected_actions = ledger["rejections"] + ledger["conflict_losses"]
            publication = ledger["accepted_swaps"] + ledger["value_comparisons"]
            unit_weight = (
                ledger["activations"]
                + ledger["observation_reads"]
                + ledger["value_comparisons"]
                + target_calculations
                + ledger["proposals"]
                + ledger["no_ops"]
                + ledger["rejections"]
                + ledger["memory_updates"]
                + ledger["accepted_swaps"]
                + ledger["displaced_cells"]
                + ledger["conflict_losses"]
            )
            row.update(ledger)
            row.update(
                {
                    "target_calculations": target_calculations,
                    "rejected_actions": rejected_actions,
                    "swap_only_cost": ledger["accepted_swaps"],
                    "publication_swap_plus_comparison_cost": publication,
                    "historical_compare_and_swap_probe": np.nan,
                    "unit_weight_full_ledger": unit_weight,
                    "ledger_profile": "complete_reference_s03_plus_s10_target_projection",
                    "publication_cost_interpretation": "R_S03_logical_value_comparisons",
                }
            )
        elif source["backend_profile"] == C_PROFILE:
            probe = int(source["ledger_valueComparisons"])
            displaced = int(source["ledger_displacedCells"])
            row.update(
                {
                    "activations": np.nan,
                    "observation_reads": np.nan,
                    "value_comparisons": np.nan,
                    "target_calculations": np.nan,
                    "proposals": np.nan,
                    "no_ops": np.nan,
                    "rejections": np.nan,
                    "rejected_actions": np.nan,
                    "memory_updates": np.nan,
                    "accepted_swaps": swaps,
                    "displaced_cells": displaced,
                    "conflict_losses": np.nan,
                    "swap_only_cost": swaps,
                    "historical_compare_and_swap_probe": probe,
                    "publication_swap_plus_comparison_cost": swaps + probe,
                    "unit_weight_full_ledger": np.nan,
                    "ledger_profile": "lossy_frozen_C_recorded_swap_and_source_probe",
                    "publication_cost_interpretation": "C_source_specific_nonparity_projection",
                }
            )
        else:
            raise ValueError(f"unexpected backend profile: {source['backend_profile']}")
        rows.append(row)

    result = pd.DataFrame(rows)
    result["wall_time_seconds"] = result.pop("wall_time_seconds")
    return result.sort_values(
        ["split", "backend_profile", "architecture", "policy", "replicate_ordinal"]
    ).reset_index(drop=True)


def availability_matrix() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = {
        "accepted_swaps": ("observed", "derived_exact", True, "C successful recorded swaps"),
        "swap_only_cost": ("derived_exact", "derived_exact", True, "same successful-swap projection"),
        "observation_reads": ("observed", "unavailable", False, "C activation observations not recorded"),
        "value_comparisons": ("observed", "unavailable", False, "C probe is not an individual comparison count"),
        "historical_compare_and_swap_probe": ("not_applicable", "observed", False, "source-specific move-eligible probe"),
        "publication_swap_plus_comparison_cost": ("derived_exact", "derived_source_specific", False, "C projection uses non-parity source probe"),
        "target_calculations": ("derived_exact", "unavailable", False, "C activations/proposals not recorded"),
        "proposals": ("observed", "unavailable", False, "C proposals not recorded"),
        "no_ops": ("observed", "unavailable", False, "C non-swap activations not recorded"),
        "rejections": ("observed", "unavailable", False, "C validation outcomes not recorded"),
        "rejected_actions": ("derived_exact", "unavailable", False, "C rejections/conflict losses not recorded"),
        "memory_updates": ("observed", "unavailable", False, "C Selection cursor updates not recorded"),
        "activations": ("observed", "unavailable", False, "C activation sequence not recorded"),
        "displaced_cells": ("observed", "derived_exact", True, "two displacements per successful recorded swap for unique values"),
        "conflict_losses": ("observed", "unavailable", False, "C lock contention is not serialized as proposals"),
        "unit_weight_full_ledger": ("derived_exact", "unavailable", False, "C complete operation vector unavailable"),
        "wall_time_seconds": ("observed_descriptive", "observed_descriptive", False, "backend/runtime-specific and nonportable"),
    }
    for field, (r_status, c_status, comparable, reason) in fields.items():
        rows.append(
            {
                "field": field,
                "reference_status": r_status,
                "historical_status": c_status,
                "directly_comparable": comparable,
                "historical_reason_or_boundary": reason,
            }
        )
    return pd.DataFrame(rows)


def validate_ledger(ledger: pd.DataFrame, source: pd.DataFrame) -> dict[str, Any]:
    r = ledger[ledger.backend_profile.eq(R_PROFILE)].copy()
    c = ledger[ledger.backend_profile.eq(C_PROFILE)].copy()
    checks: dict[str, Any] = {}

    def add(name: str, values: pd.Series | np.ndarray | bool) -> None:
        if isinstance(values, bool):
            passed, count = values, 1
        else:
            array = np.asarray(values, dtype=bool)
            passed, count = bool(array.all()), int(array.size)
        checks[name] = {"passed": passed, "checked": count}

    add("row_count_preserved", len(ledger) == len(source))
    add("run_ids_unique", ledger.run_id.nunique() == len(ledger))
    add("allowed_splits_only", ledger.split.isin(ALLOWED_SPLITS))
    add("all_source_runs_complete", ledger.completed & ~ledger.timed_out)
    add("R_activations_equal_proposals", r.activations.eq(r.proposals))
    add("R_target_calculations_equal_proposals", r.target_calculations.eq(r.proposals))
    add("R_accepted_swaps_match_swap_cost", r.accepted_swaps.eq(r.swap_only_cost))
    add("R_displacement_identity", r.displaced_cells.eq(2 * r.accepted_swaps))
    add(
        "R_proposal_partition",
        r.proposals.eq(
            r.no_ops + r.rejections + r.memory_updates + r.accepted_swaps + r.conflict_losses
        ),
    )
    add("R_rejected_action_identity", r.rejected_actions.eq(r.rejections + r.conflict_losses))
    add(
        "R_publication_projection_identity",
        r.publication_swap_plus_comparison_cost.eq(r.accepted_swaps + r.value_comparisons),
    )
    recomputed = (
        r.activations + r.observation_reads + r.value_comparisons
        + r.target_calculations + r.proposals + r.no_ops + r.rejections
        + r.memory_updates + r.accepted_swaps + r.displaced_cells + r.conflict_losses
    )
    add("R_unit_weight_projection_identity", r.unit_weight_full_ledger.eq(recomputed))
    add("R_serial_no_fault_has_no_rejection_or_conflict", (r.rejections + r.conflict_losses).eq(0))
    add("C_swap_probe_identity", c.accepted_swaps.eq(c.swap_only_cost))
    add("C_displacement_identity", c.displaced_cells.eq(2 * c.accepted_swaps))
    add(
        "C_source_publication_projection_identity",
        c.publication_swap_plus_comparison_cost.eq(
            c.accepted_swaps + c.historical_compare_and_swap_probe
        ),
    )
    unavailable = (
        "activations", "observation_reads", "value_comparisons", "target_calculations",
        "proposals", "no_ops", "rejections", "rejected_actions", "memory_updates",
        "conflict_losses", "unit_weight_full_ledger",
    )
    add("C_unavailable_fields_are_null", c[list(unavailable)].isna().all(axis=1))
    add("wall_time_positive", ledger.wall_time_seconds.gt(0))
    add("wall_time_never_in_algorithmic_cost", ~ledger.wall_time_in_algorithmic_cost)
    checks["success"] = all(item["passed"] for item in checks.values())
    checks["referenceRows"] = len(r)
    checks["historicalRows"] = len(c)
    checks["totalRows"] = len(ledger)
    return checks


def hand_counted_toys() -> dict[str, Any]:
    """Validate proposal costs against independently written tiny expectations."""
    cases = [
        {
            "id": "T01_cv_bubble_swap", "values": [2, 1], "policy": "Bubble",
            "architecture": "cell_view", "actor_index": 0, "side": "right",
            "expected": {"kind": "Swap", "reads": 2, "comparisons": 1, "targetCalculations": 1, "proposals": 1, "swaps": 1, "displacement": 2},
        },
        {
            "id": "T02_cv_bubble_boundary", "values": [2, 1], "policy": "Bubble",
            "architecture": "cell_view", "actor_index": 1, "side": "right",
            "expected": {"kind": "NoOp", "reads": 1, "comparisons": 0, "targetCalculations": 1, "proposals": 1, "swaps": 0, "displacement": 0},
        },
        {
            "id": "T03_cv_insertion_swap", "values": [2, 1], "policy": "Insertion",
            "architecture": "cell_view", "actor_index": 1,
            "expected": {"kind": "Swap", "reads": 3, "comparisons": 1, "targetCalculations": 1, "proposals": 1, "swaps": 1, "displacement": 2},
        },
        {
            "id": "T04_cv_selection_swap", "values": [2, 1], "policy": "Selection",
            "architecture": "cell_view", "actor_index": 1,
            "expected": {"kind": "Swap", "reads": 2, "comparisons": 1, "targetCalculations": 1, "proposals": 1, "swaps": 1, "displacement": 2},
        },
        {
            "id": "T05_traditional_bubble_swap", "values": [2, 1], "policy": "Bubble",
            "architecture": "traditional",
            "expected": {"kind": "Swap", "reads": 2, "comparisons": 1, "targetCalculations": 1, "proposals": 1, "swaps": 1, "displacement": 2},
        },
        {
            "id": "T06_traditional_selection_scan", "values": [3, 1, 2], "policy": "Selection",
            "architecture": "traditional",
            "expected": {"kind": "Swap", "reads": 4, "comparisons": 2, "targetCalculations": 1, "proposals": 1, "swaps": 1, "displacement": 2},
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        scenario = create_scenario(
            case["values"], policy=case["policy"], architecture=case["architecture"],
            generation_key=f"S10/toy/{case['id']}", permute=False, seed=1,
            max_activations=100,
        )
        state = initial_state(scenario)
        if case["architecture"] == "cell_view":
            actor = state.occupancy[case["actor_index"]]
            kwargs = {"side": case["side"]} if "side" in case else {}
            proposal = cell_view_proposal(scenario, state, actor, **kwargs)
        else:
            proposal = traditional_proposal(scenario, state)
        observed = {
            "kind": proposal.kind.value,
            "reads": proposal.observation_reads,
            "comparisons": proposal.value_comparisons,
            "targetCalculations": 1,
            "proposals": 1,
            "swaps": int(proposal.kind.value == "Swap"),
            "displacement": 2 * int(proposal.kind.value == "Swap"),
        }
        passed = observed == case["expected"]
        results.append(
            {"id": case["id"], "expected": case["expected"], "observed": observed, "passed": passed}
        )
    return {
        "schema": "e01.s10.hand_counted_toys.v1",
        "researchStepId": "S10",
        "targetCalculationConvention": "one per no-fault proposal/activation",
        "total": len(results),
        "passed": sum(item["passed"] for item in results),
        "success": all(item["passed"] for item in results),
        "cases": results,
    }


def _paired_arrays(
    ledger: pd.DataFrame, policy: str, metric: str, split: str = "confirmatory_holdout"
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    subset = ledger[
        ledger.backend_profile.eq(R_PROFILE)
        & ledger.split.eq(split)
        & ledger.policy.eq(policy)
    ][["pairing_block_id", "architecture", metric]].copy()
    wide = subset.pivot(index="pairing_block_id", columns="architecture", values=metric)
    if list(sorted(wide.columns)) != ["cell_view", "traditional"]:
        raise AssertionError(f"incomplete architecture pairs for {policy}/{metric}/{split}")
    if wide.isna().any(axis=None):
        raise AssertionError(f"null paired values for {policy}/{metric}/{split}")
    wide = wide.sort_index()
    return (
        wide["cell_view"].to_numpy(dtype=np.float64),
        wide["traditional"].to_numpy(dtype=np.float64),
        wide,
    )


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: np.ndarray
    ratio: np.ndarray


def paired_bootstrap(
    cell: np.ndarray,
    traditional: np.ndarray,
    *,
    seed: int,
    repetitions: int = 10_000,
    batch_size: int = 250,
) -> BootstrapResult:
    if len(cell) != len(traditional) or not len(cell):
        raise ValueError("paired bootstrap requires nonempty equal arrays")
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    ratios = np.empty(repetitions, dtype=np.float64)
    cursor = 0
    while cursor < repetitions:
        count = min(batch_size, repetitions - cursor)
        indexes = rng.integers(0, len(cell), size=(count, len(cell)))
        cell_means = cell[indexes].mean(axis=1)
        traditional_means = traditional[indexes].mean(axis=1)
        differences[cursor:cursor + count] = cell_means - traditional_means
        ratios[cursor:cursor + count] = np.divide(
            cell_means,
            traditional_means,
            out=np.full(count, np.nan, dtype=np.float64),
            where=traditional_means != 0,
        )
        cursor += count
    return BootstrapResult(differences, ratios)


def _interval(values: np.ndarray, coverage: float) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    tail = (1.0 - coverage) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail], method="linear")
    return float(low), float(high)


def paired_effects(
    ledger: pd.DataFrame,
    preregistration: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    claims = preregistration["equivalenceTests"]["claims"]

    for policy_index, policy in enumerate(POLICIES):
        for metric_index, metric in enumerate(FULL_REFERENCE_METRICS):
            cell, traditional, wide = _paired_arrays(ledger, policy, metric)
            seed_material = f"S10|{policy}|{metric}|paired-bootstrap-v1".encode()
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            boot = paired_bootstrap(cell, traditional, seed=seed)
            diff = cell - traditional
            diff_sd = float(np.std(diff, ddof=1))
            point_ratio = (
                float(cell.mean() / traditional.mean())
                if float(traditional.mean()) != 0
                else np.nan
            )
            d95 = _interval(boot.mean_diff, 0.95)
            d_adj = _interval(boot.mean_diff, 1.0 - 0.05 / 6.0)
            r95 = _interval(boot.ratio, 0.95)
            r_adj = _interval(boot.ratio, 1.0 - 0.05 / 6.0)
            row = {
                "schema_version": "e01.s10.paired_effect.v1",
                "research_step_id": "S10",
                "split": "confirmatory_holdout",
                "policy": policy,
                "metric": metric,
                "pairs": len(wide),
                "mean_cell_view": float(cell.mean()),
                "mean_traditional": float(traditional.mean()),
                "paired_mean_difference": float(diff.mean()),
                "paired_median_difference": float(np.median(diff)),
                "paired_sd_difference": diff_sd,
                "paired_dz": float(diff.mean() / diff_sd) if diff_sd > 0 else np.nan,
                "ratio_of_means": point_ratio,
                "mean_difference_ci95_low": d95[0],
                "mean_difference_ci95_high": d95[1],
                "mean_difference_ci99_1667_low": d_adj[0],
                "mean_difference_ci99_1667_high": d_adj[1],
                "ratio_ci95_low": r95[0],
                "ratio_ci95_high": r95[1],
                "ratio_ci99_1667_low": r_adj[0],
                "ratio_ci99_1667_high": r_adj[1],
                "bootstrap_repetitions": 10_000,
                "bootstrap_seed_uint64": seed,
            }
            rows.append(row)

            claim_key = f"{policy}.{metric}"
            if claim_key in claims:
                target = float(claims[claim_key]["paperTargetRatio"])
                margin = float(claims[claim_key]["relativeMargin"])
                lower, upper = target * (1.0 - margin), target * (1.0 + margin)
                nominal = _interval(boot.ratio, 0.90)
                adjusted = _interval(boot.ratio, 1.0 - 0.10 / 6.0)
                equivalent_nominal = nominal[0] > lower and nominal[1] < upper
                equivalent_adjusted = adjusted[0] > lower and adjusted[1] < upper
                expected_direction = "lower" if target < 1 else ("higher" if target > 1 else "equal")
                direction_reproduced = (
                    point_ratio < 1 if target < 1 else point_ratio > 1 if target > 1 else lower < point_ratio < upper
                )
                classification = (
                    "supportive" if equivalent_adjusted
                    else "approximately_reproduced" if direction_reproduced
                    else "not_reproduced"
                )
                equivalence_rows.append(
                    {
                        "schema_version": "e01.s10.equivalence.v1",
                        "research_step_id": "S10",
                        "claim_id": _figure4_claim_id(policy, metric),
                        "policy": policy,
                        "metric": metric,
                        "pairs": len(wide),
                        "paper_target_ratio_cell_over_traditional": target,
                        "relative_margin": margin,
                        "equivalence_lower_bound": lower,
                        "equivalence_upper_bound": upper,
                        "observed_ratio_of_means": point_ratio,
                        "ratio_ci90_low": nominal[0],
                        "ratio_ci90_high": nominal[1],
                        "ratio_ci98_3333_low": adjusted[0],
                        "ratio_ci98_3333_high": adjusted[1],
                        "equivalent_nominal": equivalent_nominal,
                        "equivalent_multiplicity_adjusted": equivalent_adjusted,
                        "expected_direction": expected_direction,
                        "direction_reproduced": direction_reproduced,
                        "replication_classification": classification,
                    }
                )
                half90 = _interval(boot.ratio[:5000], 0.90)
                stability_rows.append(
                    {
                        "claim": claim_key,
                        "full_repetitions": 10_000,
                        "prefix_repetitions": 5_000,
                        "full_ci90": list(nominal),
                        "prefix_ci90": list(half90),
                        "max_endpoint_absolute_difference": max(
                            abs(nominal[0] - half90[0]), abs(nominal[1] - half90[1])
                        ),
                    }
                )
    stability = {
        "schema": "e01.s10.uncertainty_diagnostics.v1",
        "researchStepId": "S10",
        "method": "Compare the first 5,000 deterministic resamples with all 10,000 for each primary ratio CI.",
        "primaryClaims": stability_rows,
        "maximumCiEndpointAbsoluteDifference": max(
            item["max_endpoint_absolute_difference"] for item in stability_rows
        ),
        "deterministicReplayRequired": True,
    }
    return pd.DataFrame(rows), pd.DataFrame(equivalence_rows), stability


def _figure4_claim_id(policy: str, metric: str) -> str:
    panel = "A" if metric == "swap_only_cost" else "B"
    return f"F04-{panel}-{policy.upper()}"


def historical_z_reconstruction(
    ledger: pd.DataFrame, preregistration: Mapping[str, Any]
) -> pd.DataFrame:
    targets = preregistration["historicalZReconstruction"]["paperTargets"]
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for metric in PRIMARY_METRICS:
            cell, traditional, wide = _paired_arrays(ledger, policy, metric, split="paper_scale")
            n = len(cell)
            difference = float(cell.mean() - traditional.mean())
            unpaired_se = math.sqrt(float(cell.var(ddof=0) / n + traditional.var(ddof=0) / n))
            unpaired_z = difference / unpaired_se if unpaired_se > 0 else np.nan
            paired_diff = cell - traditional
            paired_se = float(paired_diff.std(ddof=1) / math.sqrt(n))
            paired_z = difference / paired_se if paired_se > 0 else np.nan
            key = f"{policy}.{metric}"
            target = float(targets[key])
            rows.append(
                {
                    "schema_version": "e01.s10.historical_z_reconstruction.v1",
                    "research_step_id": "S10",
                    "claim_id": _figure4_claim_id(policy, metric),
                    "policy": policy,
                    "metric": metric,
                    "split": "paper_scale",
                    "n_per_architecture": n,
                    "paper_reported_z": target,
                    "paper_test_specification_status": "incomplete_raw_data_unavailable",
                    "R_mean_cell_view": float(cell.mean()),
                    "R_mean_traditional": float(traditional.mean()),
                    "R_mean_difference": difference,
                    "R_unpaired_population_z_candidate": unpaired_z,
                    "R_unpaired_population_two_sided_p": 2 * stats.norm.sf(abs(unpaired_z)) if np.isfinite(unpaired_z) else np.nan,
                    "R_paired_z_candidate": paired_z,
                    "R_paired_two_sided_p": 2 * stats.norm.sf(abs(paired_z)) if np.isfinite(paired_z) else np.nan,
                    "absolute_discrepancy_unpaired_vs_paper_z": abs(unpaired_z - target) if np.isfinite(unpaired_z) else np.nan,
                    "interpretation": "historical_formula_sensitivity_only_not_recovered_paper_test",
                }
            )
    return pd.DataFrame(rows)


def cost_summaries(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    for keys, group in ledger.groupby(
        ["backend_profile", "split", "architecture", "policy"], sort=True
    ):
        base = dict(zip(("backend_profile", "split", "architecture", "policy"), keys))
        for metric in SUMMARY_METRICS:
            values = group[metric].dropna().to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    **base,
                    "metric": metric,
                    "runs": len(group),
                    "available_runs": len(values),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "population_sd": float(values.std(ddof=0)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "minimum": float(values.min()) if len(values) else np.nan,
                    "maximum": float(values.max()) if len(values) else np.nan,
                    "sem": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)

    available = summary[summary.available_runs.gt(0)].copy()
    available["rank_within_backend_split_architecture"] = available.groupby(
        ["backend_profile", "split", "architecture", "metric"]
    )["mean"].rank(method="average", ascending=True)
    rankings = available.sort_values(
        ["backend_profile", "split", "architecture", "metric", "rank_within_backend_split_architecture", "policy"]
    ).reset_index(drop=True)

    sensitivity_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for metric in RANKING_METRICS:
            cell, traditional, _ = _paired_arrays(ledger, policy, metric)
            cv_mean, trad_mean = float(cell.mean()), float(traditional.mean())
            if math.isclose(cv_mean, trad_mean, rel_tol=0.0, abs_tol=1e-12):
                winner = "tie"
            else:
                winner = "cell_view" if cv_mean < trad_mean else "traditional"
            sensitivity_rows.append(
                {
                    "backend_profile": R_PROFILE,
                    "split": "confirmatory_holdout",
                    "policy": policy,
                    "metric": metric,
                    "mean_cell_view": cv_mean,
                    "mean_traditional": trad_mean,
                    "cell_over_traditional_ratio": cv_mean / trad_mean if trad_mean != 0 else np.nan,
                    "lower_cost_architecture": winner,
                    "wall_time_algorithmic": False if metric == "wall_time_seconds" else None,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    return summary, rankings, sensitivity


def run_accounting(ledger: pd.DataFrame) -> dict[str, Any]:
    expected = {
        (R_PROFILE, "paper_scale"): 600,
        (C_PROFILE, "paper_scale"): 300,
        (R_PROFILE, "confirmatory_holdout"): 6000,
        (C_PROFILE, "confirmatory_holdout"): 3000,
    }
    groups = ledger.groupby(["backend_profile", "split"], sort=True)
    observed = {key: len(group) for key, group in groups}
    pairing: list[dict[str, Any]] = []
    for split, expected_pairs in (("paper_scale", 100), ("confirmatory_holdout", 1000)):
        for policy in POLICIES:
            group = ledger[
                ledger.backend_profile.eq(R_PROFILE)
                & ledger.split.eq(split)
                & ledger.policy.eq(policy)
            ]
            counts = group.groupby("pairing_block_id").architecture.nunique()
            pairing.append(
                {
                    "split": split,
                    "policy": policy,
                    "expectedPairs": expected_pairs,
                    "observedPairs": len(counts),
                    "allPairsHaveTwoArchitectures": bool(counts.eq(2).all()),
                }
            )
    forbidden = sorted(set(ledger.split) - set(ALLOWED_SPLITS))
    result = {
        "schema": "e01.s10.run_accounting.v1",
        "researchStepId": "S10",
        "intendedRuns": 9900,
        "observedRuns": len(ledger),
        "completedRuns": int(ledger.completed.sum()),
        "timedOutRuns": int(ledger.timed_out.sum()),
        "failedOrExcludedRuns": int((~ledger.completed).sum()),
        "backendSplitAccounting": [
            {
                "backendProfile": backend,
                "split": split,
                "expected": expected[(backend, split)],
                "observed": observed.get((backend, split), 0),
            }
            for backend, split in expected
        ],
        "pairing": pairing,
        "forbiddenSplitsPresent": forbidden,
    }
    result["success"] = (
        len(ledger) == 9900
        and all(observed.get(key) == count for key, count in expected.items())
        and all(item["observedPairs"] == item["expectedPairs"] and item["allPairsHaveTwoArchitectures"] for item in pairing)
        and not forbidden
        and result["completedRuns"] == 9900
        and result["timedOutRuns"] == 0
    )
    return result


def figure4(
    ledger: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
) -> None:
    paper = ledger[ledger.split.eq("paper_scale")]
    colors = {"traditional": "#3A86B8", "cell_view": "#E98FA3"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    metrics = (
        ("swap_only_cost", "(a) Successful swaps only"),
        ("publication_swap_plus_comparison_cost", "(b) Swaps + comparison projection"),
    )
    x = np.arange(len(POLICIES), dtype=float)
    width = 0.34
    for axis, (metric, title) in zip(axes, metrics):
        for offset, architecture in ((-width / 2, "traditional"), (width / 2, "cell_view")):
            means: list[float] = []
            sds: list[float] = []
            for policy in POLICIES:
                values = paper[
                    paper.backend_profile.eq(R_PROFILE)
                    & paper.architecture.eq(architecture)
                    & paper.policy.eq(policy)
                ][metric].to_numpy(dtype=float)
                means.append(float(values.mean()))
                sds.append(float(values.std(ddof=0)))
            axis.bar(
                x + offset, means, width, yerr=sds, capsize=3,
                color=colors[architecture], edgecolor="black", linewidth=0.55,
                label=f"R {architecture.replace('_', ' ')}",
            )
        c_means = [
            float(
                paper[
                    paper.backend_profile.eq(C_PROFILE) & paper.policy.eq(policy)
                ][metric].mean()
            )
            for policy in POLICIES
        ]
        axis.scatter(
            x + width / 2, c_means, marker="D", s=36, facecolors="white",
            edgecolors="#7C3AED", linewidths=1.3,
            label="C cell-view source projection",
            zorder=5,
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(x, POLICIES)
        axis.set_ylabel("Operation count (mean; population SD whiskers)")
        axis.grid(axis="y", color="#D5D5D5", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(4, 4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Figure 4 reconstruction — n=100, N=100 paper-scale inputs\n"
        "C diamonds are non-parity source projections; public HEAD is not the publication snapshot",
        fontsize=13,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def update_replication_summary(equivalence: pd.DataFrame) -> Path:
    """Append the six S10 Figure 4 decisions to the stable evidence table."""
    path = Path("/artifacts/results/replication_summary.parquet")
    prior = pd.read_parquet(path)
    s10_rows = []
    for row in equivalence.to_dict(orient="records"):
        s10_rows.append(
            {
                "research_step_id": "S10",
                "figure": 4,
                "claim_id": row["claim_id"],
                "endpoint_classification": "not_applicable_cost_claim",
                "qualitative_classification": (
                    "supportive" if row["direction_reproduced"] else "contradictory"
                ),
                "outcome_classification": row["replication_classification"],
                "estimate": row["observed_ratio_of_means"],
                "adjusted_ci_low": row["ratio_ci98_3333_low"],
                "adjusted_ci_high": row["ratio_ci98_3333_high"],
                "analysis_split": "confirmatory_holdout",
            }
        )
    combined = pd.concat(
        [prior[~prior.research_step_id.eq("S10")], pd.DataFrame(s10_rows)],
        ignore_index=True,
    ).sort_values(["figure", "claim_id"]).reset_index(drop=True)
    write_parquet(path, combined)
    return path


def input_provenance() -> dict[str, Any]:
    paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md"),
        Path("/workspace/input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md"),
        Path("/artifacts/research_steps/S01/claim_registry.parquet"),
        Path("/artifacts/research_steps/S01/paper_metric_definitions.md"),
        Path("/artifacts/research_steps/S02/historical_code_findings.csv"),
        Path("/artifacts/research_steps/S02/lawful_source_manifest.json"),
        Path("/artifacts/research_steps/S03/transition_spec.md"),
        Path("/artifacts/research_steps/S03/transition_contract.json"),
        Path("/artifacts/research_steps/S03/toy_fixtures.json"),
        Path("/artifacts/research_steps/S04/missing_evidence.json"),
        Path("/artifacts/research_steps/S04/patch_ledger.json"),
        Path("/artifacts/research_steps/S05/semantic_decisions.json"),
        Path("/artifacts/research_steps/S05/toy_fixture_results.json"),
        Path("/artifacts/research_steps/S06/event_schema.json"),
        Path("/artifacts/research_steps/S06/field_availability_matrix.csv"),
        Path("/artifacts/research_steps/S07/invariant_coverage_matrix.csv"),
        Path("/artifacts/research_steps/S07/validation_summary.json"),
        Path("/artifacts/research_steps/S08/paired_scenario_bank.parquet"),
        Path("/artifacts/research_steps/S08/split_manifest.json"),
        Path("/artifacts/research_steps/S08/pairing_validation.json"),
        Path("/artifacts/research_steps/S09/confirmatory_preregistration.json"),
        S09_RUNS,
        Path("/artifacts/research_steps/S09/run_accounting.json"),
        Path("/artifacts/research_steps/S09/validation_summary.json"),
        PREREGISTRATION,
    ]
    for step in range(1, 10):
        paths.append(Path(f"/artifacts/research_steps/S{step:02d}/research_step_full_results.md"))
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        records.append(
            {"path": str(path), "sizeBytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "schema": "e01.s10.input_provenance.v1",
        "researchStepId": "S10",
        "inputs": sorted(records, key=lambda item: item["path"]),
        "historicalSourceCommit": "1fd2bd5921c1f6b423a71f691d5189106a8a1020",
        "publicationSnapshotClaimed": False,
        "historicalRawArraysRecovered": False,
    }


def source_manifest() -> dict[str, Any]:
    paths = [
        REPOSITORY / "analysis" / "efficiency_costs.py",
        REPOSITORY / "analysis" / "s10_efficiency_preregistration.json",
        REPOSITORY / "scripts" / "replicate_efficiency.py",
        REPOSITORY / "tests" / "test_efficiency_costs.py",
    ]
    return {
        "schema": "e01.s10.source_manifest.v1",
        "researchStepId": "S10",
        "repository": "https://github.com/Eidosoma/cell_research",
        "branch": git_output("branch", "--show-current"),
        "parentCommitBeforeS10": "d7c1d18b3f8d21096a6521d3163193b39272f357",
        "repositoryCommitAtGeneration": git_output("rev-parse", "HEAD"),
        "files": [
            {"path": str(path.relative_to(REPOSITORY)), "sha256": sha256_file(path), "sizeBytes": path.stat().st_size}
            for path in paths
        ],
    }


def environment_provenance() -> dict[str, Any]:
    return {
        "schema": "e01.s10.environment.v1",
        "researchStepId": "S10",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "workersUsed": 1,
        "parallelism": "serial vectorized analysis; no simulator run",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "newDependenciesInstalled": [],
    }


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", version="2.6")


def execute(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    preregistration = json.loads(PREREGISTRATION.read_text())
    source_runs = pd.read_parquet(S09_RUNS)
    ledger = build_cost_ledger(source_runs)
    ledger_validation = validate_ledger(ledger, source_runs)
    toys = hand_counted_toys()
    accounting = run_accounting(ledger)
    effects, equivalence, uncertainty = paired_effects(ledger, preregistration)
    ztests = historical_z_reconstruction(ledger, preregistration)
    summary, rankings, sensitivity = cost_summaries(ledger)
    availability = availability_matrix()

    write_parquet(output / "cost_ledger.parquet", ledger)
    write_parquet(output / "paired_effects.parquet", effects)
    summary.to_csv(output / "cost_summary.csv", index=False, float_format="%.12g")
    rankings.to_csv(output / "backend_rankings.csv", index=False, float_format="%.12g")
    sensitivity.to_csv(output / "cost_definition_sensitivity.csv", index=False, float_format="%.12g")
    equivalence.to_csv(output / "equivalence_analysis.csv", index=False, float_format="%.12g")
    ztests.to_csv(output / "historical_z_reconstruction.csv", index=False, float_format="%.12g")
    availability.to_csv(output / "historical_field_availability.csv", index=False)
    equivalence.to_csv(output / "figure4_claim_classifications.csv", index=False, float_format="%.12g")
    write_json(output / "ledger_validation.json", ledger_validation)
    write_json(output / "hand_counted_toy_validation.json", toys)
    write_json(output / "run_accounting.json", accounting)
    write_json(output / "uncertainty_diagnostics.json", uncertainty)
    write_json(output / "input_provenance.json", input_provenance())
    write_json(output / "environment_provenance.json", environment_provenance())
    figure4(ledger, output / "figure4_reconstruction.png", output / "figure4_reconstruction.svg")
    shutil.copyfile(PREREGISTRATION, output / "confirmatory_preregistration.json")
    stable_summary = update_replication_summary(equivalence)
    write_json(output / "source_package_manifest.json", source_manifest())

    # Determinism check for the analysis core: recompute all table-valued results.
    effects2, equivalence2, uncertainty2 = paired_effects(ledger, preregistration)
    deterministic = (
        effects.to_csv(index=False) == effects2.to_csv(index=False)
        and equivalence.to_csv(index=False) == equivalence2.to_csv(index=False)
        and canonical_json_bytes(uncertainty) == canonical_json_bytes(uncertainty2)
    )
    validation = {
        "schema": "e01.s10.validation_summary.v1",
        "researchStepId": "S10",
        "success": bool(
            ledger_validation["success"]
            and toys["success"]
            and accounting["success"]
            and deterministic
            and len(equivalence) == 6
            and len(effects) == 33
        ),
        "ledgerValidation": ledger_validation,
        "handCountedToys": {"passed": toys["passed"], "total": toys["total"]},
        "runAccounting": {"success": accounting["success"], "runs": accounting["observedRuns"]},
        "pairedEffects": {"rows": len(effects), "pairsPerPolicy": 1000},
        "equivalenceClaims": len(equivalence),
        "stableReplicationSummary": {"path": str(stable_summary), "rows": 10},
        "deterministicAnalysisReplay": deterministic,
        "historicalUnavailableFieldsRemainNull": ledger_validation["C_unavailable_fields_are_null"]["passed"],
        "wallTimeSeparate": ledger_validation["wall_time_never_in_algorithmic_cost"]["passed"],
        "zeroUnexplainedFailures": bool(ledger_validation["success"] and toys["success"] and accounting["success"]),
    }
    write_json(output / "validation_summary.json", validation)
    if not validation["success"]:
        raise AssertionError(json.dumps(validation, indent=2))
    return validation
