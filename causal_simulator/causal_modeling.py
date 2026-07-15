"""Frozen S12 causal-modeling data and statistical utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ESTIMANDS = (
    "E02-S01-E04",
    "E02-S01-E06",
    "E02-S01-E03",
    "E02-S01-E08",
)
SCALES = (20, 50, 100, 200, 500)
MODIFIERS = (
    "n",
    "valueProfile",
    "orderStructure",
    "policyProfile",
    "direction",
    "placementClass",
    "faultCount",
)
TWO_WAY_MODIFIERS = (
    ("n", "policyProfile"),
    ("n", "placementClass"),
    ("policyProfile", "direction"),
)
ENDPOINTS = (
    "normalizedResidualError",
    "successByBudget",
    "completionCumulativeIncidenceAtBudget",
    "restrictedMeanCompletionFreeBudgetFraction",
    "logS01UnitWeightFullCost",
)
COUPLING = {
    "E02-S01-E03": "scenario_paired_rng_unpaired_different_stream_consumption",
    "E02-S01-E04": "shared_prefix_until_continuation_stop",
    "E02-S01-E06": "scenario_paired_rng_unpaired_different_scenario_roots",
    "E02-S01-E08": "shared_prefix_same_scheduler_root",
}
ADDITIVE_S01_COSTS = (
    "activations",
    "observationRecordReads",
    "valueComparisons",
    "targetCalculations",
    "proposals",
    "noOps",
    "rejections",
    "memoryUpdates",
    "acceptedSwaps",
    "displacedCells",
    "conflictLosses",
    "coordinatorMessages",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    size = len(p)
    for rank, index in enumerate(order):
        running = max(running, (size - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def bh_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    size = len(p)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = size - reverse_rank + 1
        running = min(running, p[index] * size / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def sign_tail_p(draws: np.ndarray) -> float:
    values = np.asarray(draws, dtype=float)
    return float(min(1.0, 2 * min((np.sum(values <= 0) + 1) / (len(values) + 1),
                                  (np.sum(values >= 0) + 1) / (len(values) + 1))))


def hash_fold(freeze_sha: str, pairing_block_id: str, folds: int = 5) -> int:
    digest = hashlib.sha256(
        f"{freeze_sha}/calibration/{pairing_block_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % folds


def build_bootstrap_weights(
    blocks: pd.DataFrame,
    freeze_sha: str,
    replicates: int,
    *,
    stream: str = "paired",
) -> np.ndarray:
    """Return scale-stratified deterministic bootstrap multiplicities."""

    ordered = blocks[["pairingBlockId", "n"]].drop_duplicates().sort_values(
        ["n", "pairingBlockId"]
    ).reset_index(drop=True)
    if len(ordered) != 2000 or set(ordered.n) != set(SCALES):
        raise AssertionError("S12 bootstrap requires the exact 2,000-block support")
    weights = np.zeros((replicates, len(ordered)), dtype=np.int16)
    for n in SCALES:
        indices = np.flatnonzero(ordered.n.to_numpy() == n)
        if len(indices) != 400:
            raise AssertionError(f"scale {n} does not contain 400 terminal blocks")
        size = len(indices)
        for replicate in range(replicates):
            counts = np.zeros(size, dtype=np.int16)
            for draw in range(size):
                address = f"{freeze_sha}/{stream}/{replicate}/{n}/{draw}"
                value = int.from_bytes(hashlib.sha256(address.encode()).digest()[:8], "big")
                counts[value % size] += 1
            weights[replicate, indices] = counts
    return weights


def load_arm_data(
    results_path: Path,
    contrast_map_path: Path,
    pairing_path: Path,
) -> pd.DataFrame:
    results = pd.read_parquet(results_path)
    contrast = pd.read_parquet(contrast_map_path)
    contrast = contrast[
        (contrast.confirmatoryLook <= 2) & contrast.estimandId.isin(ESTIMANDS)
    ].copy()
    pairing = pd.read_parquet(pairing_path)
    pairing = pairing[pairing.confirmatoryLook <= 2][
        ["pairingBlockId", "eventBudgetOpportunities"]
    ].drop_duplicates()
    results_for_merge = results.drop(columns=[
        "pairingBlockId", "confirmatoryLook"
    ])
    arm = contrast.merge(results_for_merge, on="runDesignId", validate="many_to_one")
    arm = arm.merge(pairing, on="pairingBlockId", validate="many_to_one")
    arm["treatment"] = (arm.contrastRole == "active").astype(int)
    arm["logS01UnitWeightFullCost"] = np.log(
        arm.projection_s01UnitWeightFullCost.astype(float) + 0.5
    )
    arm["log1pS01UnitWeightFullCost"] = np.log1p(
        arm.projection_s01UnitWeightFullCost.astype(float)
    )
    arm["logControllerExpandedCost"] = np.log(
        arm.projection_controllerExpandedSensitivity.astype(float) + 0.5
    )
    arm["logMechanismExpandedCost"] = np.log(
        arm.projection_mechanismExpandedSensitivity.astype(float) + 0.5
    )
    arm["logCompletionOpportunity"] = np.log(
        arm.completionOpportunity.astype(float) + 0.5
    )
    arm["normalizedCompletionTime"] = np.minimum(
        arm.completionOpportunity.astype(float) / arm.eventBudgetOpportunities.astype(float),
        1.0,
    )
    arm["competingFailure"] = arm.stopReason.isin(
        ["quiescent", "blocking_failure"]
    ).astype(int)
    arm["completionCumulativeIncidenceAtBudget"] = arm.completionObserved.astype(float)
    arm["restrictedMeanCompletionFreeBudgetFraction"] = np.where(
        arm.completionObserved.astype(bool), arm.normalizedCompletionTime, 1.0
    )
    arm["s01CostPerN2"] = (
        arm.projection_s01UnitWeightFullCost.astype(float) / arm.n.astype(float) ** 2
    )
    arm["couplingClassification"] = arm.estimandId.map(COUPLING)
    arm["analysisPopulation"] = "P_CONFIRMATORY_ITS"
    arm = arm.sort_values(
        ["estimandId", "pairingBlockId", "treatment"]
    ).reset_index(drop=True)
    validate_arm_data(arm)
    return arm


def validate_arm_data(arm: pd.DataFrame) -> None:
    if len(arm) != 16000:
        raise AssertionError(f"expected 16,000 conceptual arm rows, found {len(arm)}")
    counts = arm.groupby(["estimandId", "contrastRole"]).size()
    if set(counts.index.get_level_values(0)) != set(ESTIMANDS):
        raise AssertionError("frozen estimand set changed")
    if set(counts.to_numpy()) != {2000}:
        raise AssertionError("active/reference pair accounting changed")
    if arm.duplicated(["estimandId", "pairingBlockId", "contrastRole"]).any():
        raise AssertionError("duplicate conceptual arm")
    if not arm.contractValidationPass.all():
        raise AssertionError("an S11 run contract failed")
    if set(arm.n) != set(SCALES):
        raise AssertionError("scale support changed")
    if not arm.protected.all() or set(arm.split) != {"confirmatory_holdout"}:
        raise AssertionError("nonprotected outcome reached S12")
    if not (arm.costSchemaVersion == "E02.complete-cost-ledger.v1").all():
        raise AssertionError("S09 cost schema changed")
    if not np.allclose(
        arm.successByBudget.to_numpy(), arm.completionObserved.to_numpy()
    ):
        raise AssertionError("success and completion-event indicators diverged")


def build_paired_data(arm: pd.DataFrame) -> pd.DataFrame:
    endpoints = [
        "normalizedResidualError",
        "successByBudget",
        "completionCumulativeIncidenceAtBudget",
        "restrictedMeanCompletionFreeBudgetFraction",
        "logS01UnitWeightFullCost",
        "log1pS01UnitWeightFullCost",
        "logControllerExpandedCost",
        "logMechanismExpandedCost",
        "logCompletionOpportunity",
        "s01CostPerN2",
        "projection_s01UnitWeightFullCost",
    ]
    records: list[pd.DataFrame] = []
    for estimand in ESTIMANDS:
        part = arm[arm.estimandId == estimand]
        active = part[part.treatment == 1].set_index("pairingBlockId")
        reference = part[part.treatment == 0].set_index("pairingBlockId")
        if set(active.index) != set(reference.index):
            raise AssertionError(f"incomplete pairs for {estimand}")
        active = active.sort_index()
        reference = reference.loc[active.index]
        frame = active[[*MODIFIERS]].copy()
        for modifier in MODIFIERS:
            if not (active[modifier].to_numpy() == reference[modifier].to_numpy()).all():
                raise AssertionError(f"baseline modifier {modifier} differs within pair")
        for endpoint in endpoints:
            frame[endpoint] = (
                active[endpoint].to_numpy(dtype=float)
                - reference[endpoint].to_numpy(dtype=float)
            )
        frame["estimandId"] = estimand
        frame["couplingClassification"] = COUPLING[estimand]
        frame["pairingBlockId"] = active.index
        records.append(frame.reset_index(drop=True))
    paired = pd.concat(records, ignore_index=True)
    paired = paired.sort_values(["estimandId", "pairingBlockId"]).reset_index(drop=True)
    if len(paired) != 8000 or paired.duplicated(
        ["estimandId", "pairingBlockId"]
    ).any():
        raise AssertionError("paired-effect accounting failed")
    return paired


@dataclass(frozen=True)
class AalenJohansenResult:
    times: np.ndarray
    cif_completion: np.ndarray
    survival_any_terminal: np.ndarray
    cif_at_horizon: float
    restricted_mean_completion_free: float


def aalen_johansen(
    times: Sequence[float],
    states: Sequence[int],
    *,
    horizon: float = 1.0,
) -> AalenJohansenResult:
    """Aalen–Johansen completion CIF with states 0=censor, 1=event, 2=competing."""

    time = np.asarray(times, dtype=float)
    state = np.asarray(states, dtype=int)
    if len(time) == 0 or len(time) != len(state):
        raise ValueError("nonempty equal-length time/state arrays required")
    if np.any(time < 0) or not set(np.unique(state)).issubset({0, 1, 2}):
        raise ValueError("invalid time or state")
    order = np.argsort(time, kind="stable")
    time = time[order]
    state = state[order]
    event_times = np.unique(time[(state != 0) & (time <= horizon)])
    survival = 1.0
    cif = 0.0
    out_times = [0.0]
    out_cif = [0.0]
    out_survival = [1.0]
    rmcf = 0.0
    previous = 0.0
    for current in event_times:
        rmcf += (current - previous) * (1.0 - cif)
        at_risk = int(np.sum(time >= current))
        complete = int(np.sum((time == current) & (state == 1)))
        competing = int(np.sum((time == current) & (state == 2)))
        if at_risk <= 0:
            raise AssertionError("empty risk set at event time")
        cif += survival * complete / at_risk
        survival *= 1.0 - (complete + competing) / at_risk
        out_times.append(float(current))
        out_cif.append(float(cif))
        out_survival.append(float(survival))
        previous = float(current)
    rmcf += (horizon - previous) * (1.0 - cif)
    return AalenJohansenResult(
        times=np.asarray(out_times),
        cif_completion=np.asarray(out_cif),
        survival_any_terminal=np.asarray(out_survival),
        cif_at_horizon=float(cif),
        restricted_mean_completion_free=float(rmcf),
    )


def standardized_aalen_johansen(
    arm: pd.DataFrame,
    *,
    grid: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    scale_results = []
    curves = []
    for n in SCALES:
        part = arm[arm.n == n]
        state = np.where(
            part.completionObserved.to_numpy() == 1,
            1,
            np.where(part.competingFailure.to_numpy() == 1, 2, 0),
        )
        result = aalen_johansen(part.normalizedCompletionTime, state)
        cif_grid = np.zeros(len(grid), dtype=float)
        for index, value in enumerate(grid):
            position = np.searchsorted(result.times, value, side="right") - 1
            cif_grid[index] = result.cif_completion[max(position, 0)]
        scale_results.append((result.cif_at_horizon, result.restricted_mean_completion_free))
        curves.append(pd.DataFrame({"n": n, "normalizedTime": grid, "cifCompletion": cif_grid}))
    metrics = {
        "completionCumulativeIncidenceAtBudget": float(np.mean([item[0] for item in scale_results])),
        "restrictedMeanCompletionFreeBudgetFraction": float(np.mean([item[1] for item in scale_results])),
    }
    curve = pd.concat(curves, ignore_index=True).groupby(
        "normalizedTime", as_index=False
    ).cifCompletion.mean()
    return metrics, curve


def supported_cells(
    paired: pd.DataFrame,
    modifier: str | tuple[str, str],
    *,
    minimum: int,
) -> pd.DataFrame:
    columns = [modifier] if isinstance(modifier, str) else list(modifier)
    counts = paired.groupby(columns, observed=True).size().rename("pairCount").reset_index()
    return counts[counts.pairCount >= minimum].copy()


def interval_specific_completion_rates(
    arm: pd.DataFrame,
    intervals: Sequence[tuple[float, float]] = ((0.0, 0.25), (0.25, 0.75), (0.75, 1.0)),
) -> pd.DataFrame:
    rows = []
    for lower, upper in intervals:
        exposure = np.maximum(
            0.0, np.minimum(arm.normalizedCompletionTime.to_numpy(), upper) - lower
        )
        event = (
            (arm.completionObserved.to_numpy() == 1)
            & (arm.normalizedCompletionTime.to_numpy() > lower)
            & (arm.normalizedCompletionTime.to_numpy() <= upper)
        )
        rows.append({
            "intervalLower": lower,
            "intervalUpper": upper,
            "completionEvents": int(event.sum()),
            "normalizedExposure": float(exposure.sum()),
            "completionRate": float(event.sum() / exposure.sum()) if exposure.sum() else np.nan,
        })
    return pd.DataFrame(rows)
