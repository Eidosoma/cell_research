"""Frozen S14 mechanism, Pareto, and sampled-design synthesis helpers.

The functions in this module operate only on already-frozen S10--S13 tables.
They do not run adaptive searches or alter any simulator contract.
"""

from __future__ import annotations

from itertools import combinations
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SYNTHESIS_SCHEMA_VERSION = "e02.s14.synthesis.v1"

ENDPOINT_SPECS: dict[str, tuple[str, float, float]] = {
    "normalizedResidualError": ("normalized_residual", -1.0, 0.02),
    "successByBudget": ("completion_success", 1.0, 0.02),
    "restrictedMeanCompletionFreeBudgetFraction": ("restricted_mean_completion_free_fraction", -1.0, 0.02),
    "logS01UnitWeightFullCost": ("log_s01_cost", -1.0, math.log(1.10)),
}

ESTIMAND_LABELS = {
    "E02-S01-E03": "E03 scheduler",
    "E02-S01-E04": "E04 continuation",
    "E02-S01-E06": "E06 mobility",
    "E02-S01-E08": "E08 coordination",
}

PARETO_COST_COLUMNS = {
    "s01Cost": "projection_s01UnitWeightFullCost",
    "controllerExpandedCost": "projection_controllerExpandedSensitivity",
    "mechanismExpandedCost": "projection_mechanismExpandedSensitivity",
}

SCALARIZATION_WEIGHTS = {
    "task_priority": np.asarray([0.45, 0.45, 0.10]),
    "balanced": np.asarray([0.35, 0.35, 0.30]),
    "resource_priority": np.asarray([0.20, 0.20, 0.60]),
}

VARIANCE_GROUPS = (
    "topology_coordination",
    "scheduler",
    "mobility",
    "continuation",
    "retry",
    "declared_interactions",
)


def _setting_label(record: Mapping[str, Any]) -> str:
    architecture = {
        "distributed_local": "distributed",
        "distributed_weak_coordinator": "weak",
        "central_local_proposal_k1": "central",
    }.get(str(record["architecture"]), str(record["architecture"]))
    coordinator = str(record["coordinatorProfile"])
    scheduler = {
        "uniform_random_activation": "uniform",
        "random_permutation_sweep": "permutation",
    }.get(str(record["scheduler"]), str(record["scheduler"]))
    mobility = str(record["mobility"])
    continuation = {
        "skip_and_continue": "skip",
        "stop_on_first_blocking_failure": "stop",
    }.get(str(record["continuation"]), str(record["continuation"]))
    parts = [architecture]
    if coordinator != "none":
        parts.append("coordinated")
    parts.extend([scheduler, mobility, continuation])
    return " / ".join(parts)


def validate_complete_panel(
    data: pd.DataFrame,
    *,
    expected_blocks: int,
    expected_settings: int,
    protected: bool | None,
) -> dict[str, Any]:
    required = {
        "pairingBlockId",
        "treatmentSignature",
        "n",
        "normalizedResidualError",
        "successByBudget",
        "projection_s01UnitWeightFullCost",
        "contractValidationPass",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"missing panel columns: {missing}")
    blocks = data["pairingBlockId"].nunique()
    settings = data["treatmentSignature"].nunique()
    duplicate_cells = int(data.duplicated(["pairingBlockId", "treatmentSignature"]).sum())
    counts = data.groupby("pairingBlockId")["treatmentSignature"].nunique()
    complete = bool(
        blocks == expected_blocks
        and settings == expected_settings
        and duplicate_cells == 0
        and len(data) == expected_blocks * expected_settings
        and (counts == expected_settings).all()
    )
    if not complete:
        raise ValueError("panel is not complete at the frozen block-by-setting boundary")
    if protected is not None:
        if "protected" not in data or not (data["protected"].astype(bool) == protected).all():
            raise ValueError("protected split flag violates frozen boundary")
    if not data["contractValidationPass"].astype(bool).all():
        raise ValueError("one or more source runs failed the frozen contract validation")
    signatures_by_block = data.groupby("pairingBlockId")["treatmentSignature"].apply(
        lambda values: tuple(sorted(values))
    )
    if signatures_by_block.nunique() != 1:
        raise ValueError("treatment marginals differ across pairing blocks")
    return {
        "rowCount": int(len(data)),
        "pairingBlockCount": int(blocks),
        "treatmentSettingCount": int(settings),
        "duplicateCells": duplicate_cells,
        "complete": complete,
        "contractValidationAllPass": True,
        "protectedFlagExpected": protected,
    }


def build_mechanism_tables(
    marginal: pd.DataFrame,
    heterogeneity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orient all frozen S12 effects toward benefit without selecting outcomes."""
    expected_estimands = set(ESTIMAND_LABELS)
    if set(marginal["estimandId"]) != expected_estimands:
        raise ValueError("marginal estimands differ from the four frozen executable contrasts")
    missing_endpoints = set(ENDPOINT_SPECS).difference(marginal["endpoint"])
    if missing_endpoints:
        raise ValueError(f"missing frozen mechanism endpoints: {sorted(missing_endpoints)}")

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[frame["endpoint"].isin(ENDPOINT_SPECS)].copy()
        result["estimandLabel"] = result["estimandId"].map(ESTIMAND_LABELS)
        result["endpointLabel"] = result["endpoint"].map(lambda value: ENDPOINT_SPECS[value][0])
        result["benefitOrientation"] = result["endpoint"].map(
            lambda value: ENDPOINT_SPECS[value][1]
        )
        result["practicalMargin"] = result["endpoint"].map(
            lambda value: ENDPOINT_SPECS[value][2]
        )
        for column in ("estimate", "bootstrapLow95", "bootstrapHigh95"):
            result[f"benefit_{column}"] = result[column] * result["benefitOrientation"]
        low = result[["benefit_bootstrapLow95", "benefit_bootstrapHigh95"]].min(axis=1)
        high = result[["benefit_bootstrapLow95", "benefit_bootstrapHigh95"]].max(axis=1)
        result["benefitLow95"] = low
        result["benefitHigh95"] = high
        result["practicalMarginUnits"] = result["benefit_estimate"] / result["practicalMargin"]
        result["intervalExcludesZero"] = (low > 0) | (high < 0)
        result["materialPointEstimate"] = result["benefit_estimate"].abs() >= result["practicalMargin"]
        result["schemaVersion"] = SYNTHESIS_SCHEMA_VERSION
        return result.drop(columns=["benefit_bootstrapLow95", "benefit_bootstrapHigh95"])

    marginal_map = transform(marginal)
    heterogeneity_map = transform(heterogeneity)
    if not heterogeneity_map["supported"].astype(bool).all():
        raise ValueError("unsupported cells entered the frozen heterogeneity map")
    expected_rows = len(ESTIMAND_LABELS) * len(ENDPOINT_SPECS)
    if len(marginal_map) != expected_rows:
        raise ValueError(f"expected {expected_rows} marginal mechanism rows")
    if len(heterogeneity_map) != 368:
        raise ValueError("expected all 368 supported S12 heterogeneity cells for four endpoints")
    return (
        marginal_map.sort_values(["estimandId", "endpoint"]).reset_index(drop=True),
        heterogeneity_map.sort_values(
            ["estimandId", "endpoint", "modifier", "level"]
        ).reset_index(drop=True),
    )


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Return the non-dominated mask for a minimization objective matrix."""
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2 or not np.isfinite(objectives).all():
        raise ValueError("Pareto objectives must be a finite 2D matrix")
    keep = np.ones(objectives.shape[0], dtype=bool)
    for index in range(objectives.shape[0]):
        others = np.arange(objectives.shape[0]) != index
        dominated = np.any(
            np.all(objectives[others] <= objectives[index], axis=1)
            & np.any(objectives[others] < objectives[index], axis=1)
        )
        keep[index] = not dominated
    return keep


def _pareto_arrays(data: pd.DataFrame, cost_column: str) -> tuple[np.ndarray, list[str], list[int]]:
    blocks = (
        data[["pairingBlockId", "n"]]
        .drop_duplicates()
        .sort_values(["n", "pairingBlockId"])
    )
    signatures = sorted(data["treatmentSignature"].unique())
    scales = sorted(map(int, blocks["n"].unique()))
    n_per_scale = blocks.groupby("n").size()
    if n_per_scale.nunique() != 1:
        raise ValueError("S11 scale strata are not balanced")
    count = int(n_per_scale.iloc[0])
    arrays: list[np.ndarray] = []
    for scale in scales:
        block_ids = sorted(blocks.loc[blocks["n"] == scale, "pairingBlockId"])
        subset = data[data["pairingBlockId"].isin(block_ids)].copy()
        lookup = {
            (row.pairingBlockId, row.treatmentSignature): (
                float(row.normalizedResidualError),
                1.0 - float(row.successByBudget),
                math.log(float(getattr(row, cost_column)) + 0.5),
            )
            for row in subset.itertuples(index=False)
        }
        arrays.append(
            np.asarray(
                [[lookup[(block, signature)] for signature in signatures] for block in block_ids],
                dtype=float,
            )
        )
    panel = np.stack(arrays, axis=0)
    if panel.shape != (len(scales), count, len(signatures), 3):
        raise ValueError("unexpected scale-stratified Pareto panel shape")
    return panel, signatures, scales


def _dominance_matrix(objectives: np.ndarray) -> np.ndarray:
    count = objectives.shape[0]
    result = np.zeros((count, count), dtype=bool)
    for left in range(count):
        for right in range(count):
            if left != right:
                result[left, right] = bool(
                    np.all(objectives[left] <= objectives[right])
                    and np.any(objectives[left] < objectives[right])
                )
    return result


def pareto_analysis(
    data: pd.DataFrame,
    *,
    replicates: int = 5000,
    seed: int = 14_005_001,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute frozen empirical Pareto summaries and paired uncertainty."""
    metadata_columns = [
        "treatmentSignature",
        "architecture",
        "coordinatorProfile",
        "scheduler",
        "mobility",
        "continuation",
        "retry",
    ]
    metadata = (
        data[metadata_columns]
        .drop_duplicates("treatmentSignature")
        .sort_values("treatmentSignature")
        .reset_index(drop=True)
    )
    metadata["settingLabel"] = [
        _setting_label(record) for record in metadata.to_dict("records")
    ]
    rng = np.random.default_rng(seed)
    # Reuse exactly the same bootstrap indices for every cost sensitivity.
    primary_panel, signatures, scales = _pareto_arrays(
        data, PARETO_COST_COLUMNS["s01Cost"]
    )
    strata, per_scale, settings, _ = primary_panel.shape
    bootstrap_indices = rng.integers(0, per_scale, size=(replicates, strata, per_scale))

    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    dominance_rows: list[dict[str, Any]] = []

    for cost_profile, cost_column in PARETO_COST_COLUMNS.items():
        panel, panel_signatures, panel_scales = _pareto_arrays(data, cost_column)
        if panel_signatures != signatures or panel_scales != scales:
            raise ValueError("cost sensitivity changed the paired panel")
        point = panel.mean(axis=(0, 1))
        point_frontier = pareto_mask(point)
        boot = np.empty((replicates, settings, 3), dtype=float)
        frontier = np.empty((replicates, settings), dtype=bool)
        dominance_counts = np.zeros((settings, settings), dtype=int)
        winner_counts = {name: np.zeros(settings, dtype=int) for name in SCALARIZATION_WEIGHTS}
        for replicate in range(replicates):
            selected = [
                panel[stratum, bootstrap_indices[replicate, stratum]]
                for stratum in range(strata)
            ]
            means = np.concatenate(selected, axis=0).mean(axis=0)
            boot[replicate] = means
            frontier[replicate] = pareto_mask(means)
            dominance_counts += _dominance_matrix(means)
            if cost_profile == "s01Cost":
                ranges = np.ptp(means, axis=0)
                normalized = np.divide(
                    means - means.min(axis=0),
                    ranges,
                    out=np.zeros_like(means),
                    where=ranges > 0,
                )
                for name, weights in SCALARIZATION_WEIGHTS.items():
                    scores = normalized @ weights
                    winners = np.flatnonzero(np.isclose(scores, scores.min(), atol=1e-12))
                    winner_counts[name][winners] += 1

        lows = np.quantile(boot, 0.025, axis=0)
        highs = np.quantile(boot, 0.975, axis=0)
        for index, signature in enumerate(signatures):
            row = metadata.iloc[index].to_dict()
            summary_rows.append(
                {
                    "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
                    **row,
                    "costProfile": cost_profile,
                    "meanNormalizedResidual": point[index, 0],
                    "residualLow95": lows[index, 0],
                    "residualHigh95": highs[index, 0],
                    "failureRate": point[index, 1],
                    "failureLow95": lows[index, 1],
                    "failureHigh95": highs[index, 1],
                    "meanLogCost": point[index, 2],
                    "logCostLow95": lows[index, 2],
                    "logCostHigh95": highs[index, 2],
                    "pointPareto": bool(point_frontier[index]),
                    "paretoInclusionProbability": float(frontier[:, index].mean()),
                    "robustPareto": bool(frontier[:, index].mean() >= 0.90),
                }
            )
            if cost_profile == "s01Cost":
                for name in SCALARIZATION_WEIGHTS:
                    bootstrap_rows.append(
                        {
                            "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
                            "scalarization": name,
                            "treatmentSignature": signature,
                            "settingLabel": row["settingLabel"],
                            "winnerProbability": float(winner_counts[name][index] / replicates),
                            "bootstrapReplicates": replicates,
                        }
                    )
        for left, right in combinations(range(settings), 2):
            for source, target in ((left, right), (right, left)):
                probability = dominance_counts[source, target] / replicates
                dominance_rows.append(
                    {
                        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
                        "costProfile": cost_profile,
                        "dominatingSignature": signatures[source],
                        "dominatingLabel": metadata.iloc[source]["settingLabel"],
                        "dominatedSignature": signatures[target],
                        "dominatedLabel": metadata.iloc[target]["settingLabel"],
                        "dominanceProbability": float(probability),
                        "robustDominance": bool(probability >= 0.95),
                        "bootstrapReplicates": replicates,
                    }
                )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(bootstrap_rows),
        pd.DataFrame(dominance_rows),
    )


def _centered_dummies(values: Sequence[str]) -> np.ndarray:
    matrix = pd.get_dummies(pd.Series(values, dtype="string"), dtype=float).to_numpy(copy=True)
    matrix -= matrix.mean(axis=0, keepdims=True)
    keep = np.linalg.norm(matrix, axis=0) > 1e-12
    return matrix[:, keep]


def _interaction_dummies(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    values = [f"{a}__BY__{b}" for a, b in zip(left, right, strict=True)]
    return _centered_dummies(values)


def _treatment_design(settings: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, set[str]]]:
    architecture = (
        settings["architecture"].astype(str)
        + "__"
        + settings["coordinatorProfile"].astype(str)
    )
    scheduler = settings["scheduler"].astype(str)
    mobility = settings["mobility"].astype(str)
    continuation = settings["continuation"].astype(str)
    retry = settings["retry"].astype(str)
    blocks = {
        "topology_coordination": _centered_dummies(architecture),
        "scheduler": _centered_dummies(scheduler),
        "mobility": _centered_dummies(mobility),
        "continuation": _centered_dummies(continuation),
        "retry": _centered_dummies(retry),
        "interaction_topology_scheduler": _interaction_dummies(architecture, scheduler),
        "interaction_topology_mobility": _interaction_dummies(architecture, mobility),
        "interaction_topology_continuation": _interaction_dummies(architecture, continuation),
        "interaction_scheduler_mobility": _interaction_dummies(scheduler, mobility),
    }
    parents = {
        "interaction_topology_scheduler": {"topology_coordination", "scheduler"},
        "interaction_topology_mobility": {"topology_coordination", "mobility"},
        "interaction_topology_continuation": {"topology_coordination", "continuation"},
        "interaction_scheduler_mobility": {"scheduler", "mobility"},
    }
    return blocks, parents


def _coalition_matrix(
    coalition: frozenset[str],
    blocks: Mapping[str, np.ndarray],
    parents: Mapping[str, set[str]],
) -> np.ndarray | None:
    selected: list[np.ndarray] = [blocks[group] for group in VARIANCE_GROUPS[:-1] if group in coalition]
    if "declared_interactions" in coalition:
        for interaction, required in parents.items():
            if required.issubset(coalition):
                selected.append(blocks[interaction])
    selected = [matrix for matrix in selected if matrix.shape[1] > 0]
    return np.concatenate(selected, axis=1) if selected else None


def _coalition_values(
    mean_outcome_by_setting: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    parents: Mapping[str, set[str]],
    treatment_ss_multiplier: float,
    total_ss: float,
) -> dict[frozenset[str], float]:
    values: dict[frozenset[str], float] = {frozenset(): 0.0}
    groups = list(VARIANCE_GROUPS)
    for size in range(1, len(groups) + 1):
        for subset in combinations(groups, size):
            coalition = frozenset(subset)
            matrix = _coalition_matrix(coalition, blocks, parents)
            if matrix is None:
                values[coalition] = 0.0
                continue
            fitted = matrix @ np.linalg.lstsq(matrix, mean_outcome_by_setting, rcond=None)[0]
            values[coalition] = float(
                treatment_ss_multiplier * np.sum(fitted**2) / total_ss
            )
    return values


def _shapley(values: Mapping[frozenset[str], float]) -> dict[str, float]:
    groups = list(VARIANCE_GROUPS)
    factorial = math.factorial
    count = len(groups)
    result: dict[str, float] = {}
    for group in groups:
        contribution = 0.0
        others = [item for item in groups if item != group]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            for subset in combinations(others, size):
                coalition = frozenset(subset)
                contribution += weight * (
                    values[coalition | {group}] - values[coalition]
                )
        result[group] = contribution
    return result


def _variance_components_for_blocks(
    outcome: np.ndarray,
    sampled_block_indices: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    parents: Mapping[str, set[str]],
) -> dict[str, float]:
    sampled = outcome[sampled_block_indices]
    grand = float(sampled.mean())
    total_ss = float(np.sum((sampled - grand) ** 2))
    if total_ss <= 0:
        raise ValueError("variance decomposition outcome has zero total variance")
    block_means = sampled.mean(axis=1, keepdims=True)
    between_ss = float(sampled.shape[1] * np.sum((block_means - grand) ** 2))
    centered = sampled - block_means
    setting_means = centered.mean(axis=0)
    values = _coalition_values(
        setting_means,
        blocks,
        parents,
        float(sampled.shape[0]),
        total_ss,
    )
    contributions = _shapley(values)
    full = values[frozenset(VARIANCE_GROUPS)]
    result = {"scenario_context": between_ss / total_ss, **contributions}
    result["within_block_residual"] = 1.0 - result["scenario_context"] - full
    result["full_treatment_explained"] = full
    result["sum_partition"] = (
        result["scenario_context"]
        + sum(result[group] for group in VARIANCE_GROUPS)
        + result["within_block_residual"]
    )
    return result


def variance_decomposition(
    data: pd.DataFrame,
    *,
    replicates: int = 1000,
    seed: int = 14_001_001,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact hierarchical Shapley decomposition on the frozen S10 panel."""
    settings_columns = [
        "treatmentSignature",
        "architecture",
        "coordinatorProfile",
        "scheduler",
        "mobility",
        "continuation",
        "retry",
    ]
    settings = (
        data[settings_columns]
        .drop_duplicates("treatmentSignature")
        .sort_values("treatmentSignature")
        .reset_index(drop=True)
    )
    signatures = settings["treatmentSignature"].tolist()
    blocks_table = (
        data[["pairingBlockId", "n"]]
        .drop_duplicates()
        .sort_values(["n", "pairingBlockId"])
        .reset_index(drop=True)
    )
    block_ids = blocks_table["pairingBlockId"].tolist()
    outcomes = {
        "normalizedResidual": "normalizedResidualError",
        "completionSuccess": "successByBudget",
        "logS01Cost": "projection_s01UnitWeightFullCost",
    }
    arrays: dict[str, np.ndarray] = {}
    indexed = data.set_index(["pairingBlockId", "treatmentSignature"])
    for name, column in outcomes.items():
        values = np.asarray(
            [[indexed.loc[(block, signature), column] for signature in signatures] for block in block_ids],
            dtype=float,
        )
        if name == "logS01Cost":
            values = np.log(values + 0.5)
        arrays[name] = values
    design_blocks, parents = _treatment_design(settings)
    all_indices = np.arange(len(block_ids))
    point = {
        name: _variance_components_for_blocks(values, all_indices, design_blocks, parents)
        for name, values in arrays.items()
    }
    strata = {
        int(scale): np.flatnonzero(blocks_table["n"].to_numpy() == scale)
        for scale in sorted(blocks_table["n"].unique())
    }
    rng = np.random.default_rng(seed)
    component_names = ["scenario_context", *VARIANCE_GROUPS, "within_block_residual"]
    bootstrap: dict[str, dict[str, list[float]]] = {
        outcome: {component: [] for component in component_names} for outcome in outcomes
    }
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in strata.values()]
        )
        for name, values in arrays.items():
            components = _variance_components_for_blocks(
                values, sampled, design_blocks, parents
            )
            for component in component_names:
                bootstrap[name][component].append(components[component])
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for component in component_names:
            samples = np.asarray(bootstrap[outcome][component])
            rows.append(
                {
                    "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
                    "outcome": outcome,
                    "component": component,
                    "share": point[outcome][component],
                    "bootstrapLow95": float(np.quantile(samples, 0.025)),
                    "bootstrapHigh95": float(np.quantile(samples, 0.975)),
                    "bootstrapReplicates": replicates,
                    "interpretation": "descriptive_sampled_design_only",
                }
            )
    diagnostics = {
        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
        "pairingBlockCount": len(block_ids),
        "treatmentSettingCount": len(signatures),
        "runCount": len(block_ids) * len(signatures),
        "scales": sorted(map(int, strata)),
        "blocksPerScale": {str(scale): len(indices) for scale, indices in strata.items()},
        "hierarchicalInteractionRuleApplied": True,
        "exactCoalitionCount": 2 ** len(VARIANCE_GROUPS),
        "bootstrapReplicates": replicates,
        "maxPartitionIdentityError": float(
            max(abs(point[name]["sum_partition"] - 1.0) for name in outcomes)
        ),
        "sampledDesignBoundary": "S10 250 complete pairing blocks x 14 frozen treatment signatures",
    }
    return pd.DataFrame(rows), diagnostics


def summarize_s13(classifications: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve every finite S13 confirmation classification and brittle label."""
    if len(classifications) != 8 or classifications["candidateId"].nunique() != 8:
        raise ValueError("S13 sensitivity must include all eight unique confirmed candidates")
    records: list[dict[str, Any]] = []
    for row in classifications.sort_values("selectionOrder").itertuples(index=False):
        exact = json.loads(row.exactPanelJson)
        neighborhood = json.loads(row.neighborhoodPanelJson)
        records.append(
            {
                "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
                "selectionOrder": int(row.selectionOrder),
                "candidateId": row.candidateId,
                "objectiveId": row.objectiveId,
                "classification": row.classification,
                "fullyConfirmed": bool(row.fullyConfirmed),
                "exactDirectionalResidualMean": float(exact["directionalResidualMean"]),
                "exactDirectionalLow95": float(exact["directionalResidualBootstrapLow95"]),
                "exactDirectionalHigh95": float(exact["directionalResidualBootstrapHigh95"]),
                "exactDirectionalConcordance": float(exact["directionalConcordanceRate"]),
                "neighborhoodDirectionalResidualMean": float(neighborhood["directionalResidualMean"]),
                "neighborhoodDirectionalLow95": float(neighborhood["directionalResidualBootstrapLow95"]),
                "neighborhoodDirectionalHigh95": float(neighborhood["directionalResidualBootstrapHigh95"]),
                "neighborhoodDirectionalConcordance": float(neighborhood["directionalConcordanceRate"]),
                "synthesisLabel": (
                    "brittle_exact_instance_reversal"
                    if row.classification == "stream_confirmed_brittle_instance"
                    else "not_confirmed"
                ),
            }
        )
    summary = {
        "schemaVersion": SYNTHESIS_SCHEMA_VERSION,
        "candidateCount": 8,
        "fullyConfirmedCount": int(classifications["fullyConfirmed"].sum()),
        "brittleExactInstanceCount": int(
            (classifications["classification"] == "stream_confirmed_brittle_instance").sum()
        ),
        "globalAbsenceClaimPermitted": False,
        "interpretation": "bounded finite-search sensitivity only",
    }
    return pd.DataFrame(records), summary
