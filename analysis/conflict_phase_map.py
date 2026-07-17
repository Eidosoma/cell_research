"""E04 S13 opposing-goal phase map.

The design is frozen in :mod:`analysis/s13_conflict_phase_contract.json`.  The
module implements a counter-addressed paired scheduler, an independently
validated Numba projection of the E01 cell-view transitions, staged boundary
resampling, longer-budget replay, time-resolved state/flux diagnostics, and a
prespecified regime classifier.  It intentionally does not execute S14.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from reference_simulator.engine import run as ordinary_run
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.scheduler import ScheduledOpportunity


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
OUTPUT = Path("/artifacts/research_steps/S13")
CACHE = Path("/cache/e04_s13")
E01_S08 = Path("/previous-artifacts/E01/research_steps/S08")
CONTRACT_PATH = REPOSITORY / "analysis/s13_conflict_phase_contract.json"
STEP_SCHEMA = "e04.s13.conflict_phase.v1"
MASTER_SEED = 0xE0413000000000000000000000000001
PRIMARY_BUDGET = 200_000
LONG_BUDGET = 1_000_000
BASE_REPLICATES = (0, 1, 2, 3)
SEQUENTIAL_REPLICATES = tuple(range(4, 12))
INPUTS = ("unique_1_100", "repeated_1_10_x10")
POLICY_PAIRS = (
    ("Bubble", "Insertion"),
    ("Bubble", "Selection"),
    ("Insertion", "Selection"),
)
FIRST_COUNTS = (20, 35, 50, 65, 80)
ACTIVATION_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
FAULT_COUNTS = (0, 5, 10)
STATE_PROFILES = (
    "ascending_blocks_absent",
    "random_absent",
    "descending_blocks_absent",
    "random_positive",
    "random_negative",
)
POLICY_CODE = {"Bubble": 0, "Insertion": 1, "Selection": 2}
INPUT_CODE = {"unique_1_100": "UNQ", "repeated_1_10_x10": "REP"}
SHORT_POLICY = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}
TRACE_COLUMNS = (
    "order_score",
    "corrected_adjacency",
    "policy_position_score",
    "p1_value_position_correlation",
    "p2_value_position_correlation",
    "p1_goal_satisfaction",
    "p2_goal_satisfaction",
    "swap_rate",
    "moved_fraction",
    "occupancy_hamming",
    "p1_absolute_flux_rate",
    "p2_absolute_flux_rate",
    "p1_signed_flux_rate",
    "p2_signed_flux_rate",
    "cross_policy_swap_rate",
    "center_crossing_flux_rate",
    "p1_activation_share",
    "cumulative_swaps",
    "cumulative_memory_updates",
    "cumulative_stuck_rejections",
    "nominal_activation",
    "quiescent_state",
)
REGIMES = (
    "fixed_quiescence",
    "active_dominance",
    "oscillation",
    "metastability",
    "dynamic_equilibrium",
    "unresolved_transient",
)


def _native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_native(value)) + b"\n")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_native(value))).hexdigest()


def derive_seed(stream: str, *address: Any, bits: int = 128) -> int:
    payload = {
        "version": "E04/S13/SHA256_COUNTER/v1",
        "stream": stream,
        "address": _native(address),
    }
    digest = hashlib.sha256(
        b"E04/S13/seed/v1\x00"
        + MASTER_SEED.to_bytes(16, "big")
        + b"\x00"
        + canonical_json_bytes(payload)
    ).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


@dataclass(frozen=True, slots=True)
class PhaseCondition:
    condition_id: str
    input_profile: str
    first_policy: str
    second_policy: str
    first_direction: str
    second_direction: str
    first_count: int
    activation_ratio: float
    fault_count: int
    state_profile: str

    @property
    def policy_pair(self) -> str:
        return f"{self.first_policy}+{self.second_policy}"

    @property
    def direction_assignment(self) -> str:
        return (
            f"{self.first_policy}:{self.first_direction}|"
            f"{self.second_policy}:{self.second_direction}"
        )

    @property
    def association_profile(self) -> str:
        if self.state_profile.endswith("positive"):
            return "positive"
        if self.state_profile.endswith("negative"):
            return "negative"
        return "absent"

    @property
    def disorder_profile(self) -> str:
        if self.state_profile.startswith("ascending"):
            return "ascending_blocks"
        if self.state_profile.startswith("descending"):
            return "descending_blocks"
        return "random"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            policy_pair=self.policy_pair,
            direction_assignment=self.direction_assignment,
            association_profile=self.association_profile,
            disorder_profile=self.disorder_profile,
            second_count=100 - self.first_count,
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhaseCondition":
        return cls(
            condition_id=str(value["condition_id"]),
            input_profile=str(value["input_profile"]),
            first_policy=str(value["first_policy"]),
            second_policy=str(value["second_policy"]),
            first_direction=str(value["first_direction"]),
            second_direction=str(value["second_direction"]),
            first_count=int(value["first_count"]),
            activation_ratio=float(value["activation_ratio"]),
            fault_count=int(value["fault_count"]),
            state_profile=str(value["state_profile"]),
        )


def build_conditions() -> tuple[PhaseCondition, ...]:
    conditions: list[PhaseCondition] = []
    for input_profile in INPUTS:
        for first_policy, second_policy in POLICY_PAIRS:
            for first_direction, second_direction, orientation in (
                ("ascending", "descending", "AD"),
                ("descending", "ascending", "DA"),
            ):
                for first_count in FIRST_COUNTS:
                    for ratio in ACTIVATION_RATIOS:
                        for fault_count in FAULT_COUNTS:
                            for state_profile in STATE_PROFILES:
                                condition_id = (
                                    f"S13-{INPUT_CODE[input_profile]}-"
                                    f"{SHORT_POLICY[first_policy]}-{SHORT_POLICY[second_policy]}-"
                                    f"{orientation}-P{first_count:02d}-R{ratio:g}-"
                                    f"F{fault_count:02d}-{state_profile.upper()}"
                                )
                                conditions.append(
                                    PhaseCondition(
                                        condition_id,
                                        input_profile,
                                        first_policy,
                                        second_policy,
                                        first_direction,
                                        second_direction,
                                        first_count,
                                        ratio,
                                        fault_count,
                                        state_profile,
                                    )
                                )
    result = tuple(sorted(conditions, key=lambda item: item.condition_id))
    if len(result) != 4500 or len({item.condition_id for item in result}) != 4500:
        raise AssertionError(f"S13 factorial must contain 4500 conditions, got {len(result)}")
    return result


_BASE_CACHE: dict[tuple[str, int], dict[str, Any]] | None = None


def load_bases() -> dict[tuple[str, int], dict[str, Any]]:
    global _BASE_CACHE
    if _BASE_CACHE is None:
        rows = pq.read_table(
            E01_S08 / "base_draw_bank.parquet",
            filters=[("split", "=", "exploratory")],
        ).to_pylist()
        _BASE_CACHE = {
            (str(row["inputProfile"]), int(row["replicateOrdinal"])): row
            for row in rows
            if row["inputProfile"] in INPUTS and int(row["replicateOrdinal"]) < 12
        }
        if len(_BASE_CACHE) != 24:
            raise ValueError(f"expected 24 paired E01 base draws, found {len(_BASE_CACHE)}")
        if any(bool(row["protected"]) for row in _BASE_CACHE.values()):
            raise PermissionError("protected E01 base draw entered S13")
    return _BASE_CACHE


def _rng_permutation(items: Sequence[int], stream: str, *address: Any) -> list[int]:
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed(stream, *address)))
    return [int(item) for item in rng.permutation(np.asarray(items, dtype=np.int16))]


def _policy_assignment(
    condition: PhaseCondition, values: np.ndarray, replicate: int
) -> np.ndarray:
    n = len(values)
    ids = list(range(n))
    profile = condition.association_profile
    if profile == "absent":
        ordered = _rng_permutation(
            ids,
            "policy_assignment",
            condition.input_profile,
            condition.first_count,
            replicate,
            profile,
        )
    else:
        ties = _rng_permutation(
            ids,
            "policy_value_ties",
            condition.input_profile,
            condition.first_count,
            replicate,
        )
        tie_rank = {cell_id: rank for rank, cell_id in enumerate(ties)}
        sign = 1 if profile == "positive" else -1
        ordered = sorted(ids, key=lambda cell_id: (sign * values[cell_id], tie_rank[cell_id]))
    policy = np.ones(n, dtype=np.int8)
    policy[np.asarray(ordered[: condition.first_count], dtype=int)] = 0
    if int((policy == 0).sum()) != condition.first_count:
        raise AssertionError("policy assignment count drift")
    return policy


def _initial_occupancy(
    condition: PhaseCondition,
    values: np.ndarray,
    base: Mapping[str, Any],
    replicate: int,
) -> np.ndarray:
    if condition.disorder_profile == "random":
        occupancy = np.asarray(base["initialOccupancyIndices"], dtype=np.int16)
        return occupancy.copy()
    ties = _rng_permutation(
        list(range(len(values))),
        "occupancy_value_ties",
        condition.input_profile,
        replicate,
        condition.disorder_profile,
    )
    tie_rank = {cell_id: rank for rank, cell_id in enumerate(ties)}
    reverse = condition.disorder_profile == "descending_blocks"
    occupancy = sorted(
        range(len(values)),
        key=lambda cell_id: ((-1 if reverse else 1) * values[cell_id], tie_rank[cell_id]),
    )
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            derive_seed(
                "occupancy_block_shuffle",
                condition.input_profile,
                replicate,
                condition.disorder_profile,
            )
        )
    )
    for start in range(0, len(occupancy), 10):
        block = np.asarray(occupancy[start : start + 10], dtype=np.int16)
        occupancy[start : start + 10] = [int(item) for item in rng.permutation(block)]
    return np.asarray(occupancy, dtype=np.int16)


def _fault_assignment(
    condition: PhaseCondition, policy: np.ndarray, replicate: int
) -> np.ndarray:
    faults = np.zeros(len(policy), dtype=np.int8)
    total = condition.fault_count
    if total == 0:
        return faults
    raw_first = total * condition.first_count / len(policy)
    first_faults = math.floor(raw_first)
    if total - first_faults > (100 - condition.first_count):
        first_faults += 1
    elif raw_first - first_faults >= 0.5 and first_faults < condition.first_count:
        first_faults += 1
    second_faults = total - first_faults
    for group, count in ((0, first_faults), (1, second_faults)):
        ids = np.flatnonzero(policy == group).tolist()
        selected = _rng_permutation(
            ids,
            "fault_placement",
            condition.input_profile,
            condition.first_count,
            condition.association_profile,
            replicate,
            total,
            group,
        )[:count]
        faults[np.asarray(selected, dtype=int)] = 1
    if int(faults.sum()) != total:
        raise AssertionError("fault assignment is not exact")
    return faults


def _association_metrics(values: np.ndarray, policy: np.ndarray) -> tuple[float, float]:
    rho = float(stats.spearmanr(values, policy).statistic)
    overall = float(values.mean())
    total = float(np.square(values - overall).sum())
    between = 0.0
    for group in (0, 1):
        subset = values[policy == group]
        between += len(subset) * float((subset.mean() - overall) ** 2)
    return rho, between / total if total else 0.0


def materialize_arrays(
    condition: PhaseCondition, replicate: int
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    base = load_bases()[(condition.input_profile, replicate)]
    values = np.asarray(base["valuesById"], dtype=np.float64)
    policy = _policy_assignment(condition, values, replicate)
    directions = np.where(
        policy == 0,
        1 if condition.first_direction == "ascending" else -1,
        1 if condition.second_direction == "ascending" else -1,
    ).astype(np.int8)
    faults = _fault_assignment(condition, policy, replicate)
    occupancy = _initial_occupancy(condition, values, base, replicate)
    p1_ids = np.flatnonzero(policy == 0).astype(np.int16)
    p2_ids = np.flatnonzero(policy == 1).astype(np.int16)
    cursors = np.full(len(values), -32768, dtype=np.int16)
    for cell_id in range(len(values)):
        actual_policy = condition.first_policy if policy[cell_id] == 0 else condition.second_policy
        if actual_policy == "Selection":
            cursors[cell_id] = 0 if directions[cell_id] == 1 else len(values) - 1
    policy_kind = np.asarray(
        [
            POLICY_CODE[condition.first_policy if policy[cell_id] == 0 else condition.second_policy]
            for cell_id in range(len(values))
        ],
        dtype=np.int8,
    )
    rho, eta = _association_metrics(values, policy)
    direction_rho = float(stats.spearmanr(values, directions).statistic)
    schedule_seed = derive_seed(
        "paired_schedule", condition.input_profile, replicate, bits=64
    )
    scenario_content = {
        "condition": condition.to_dict(),
        "replicate": replicate,
        "values": values.tolist(),
        "policy": policy.tolist(),
        "direction": directions.tolist(),
        "fault": faults.tolist(),
        "occupancy": occupancy.tolist(),
        "selectionCursors": cursors.tolist(),
        "scheduleSeed": str(schedule_seed),
    }
    static = {
        **condition.to_dict(),
        "replicate": replicate,
        "run_pairing_id": f"S13-{condition.input_profile}-R{replicate:02d}",
        "scenario_id": "s13:" + canonical_hash(scenario_content),
        "base_draw_id": str(base["baseDrawId"]),
        "schedule_seed": str(schedule_seed),
        "policy_assignment_sha256": canonical_hash(policy.tolist()),
        "direction_assignment_sha256": canonical_hash(directions.tolist()),
        "fault_assignment_sha256": canonical_hash(faults.tolist()),
        "initial_occupancy_sha256": canonical_hash(occupancy.tolist()),
        "first_fault_count": int(faults[policy == 0].sum()),
        "second_fault_count": int(faults[policy == 1].sum()),
        "policy_value_spearman_rho": rho,
        "policy_value_eta_squared": eta,
        "direction_value_spearman_rho": direction_rho,
        "expected_first_activation_share": (
            condition.activation_ratio * condition.first_count
            / (
                condition.activation_ratio * condition.first_count
                + (100 - condition.first_count)
            )
        ),
    }
    arrays = (
        values,
        policy,
        policy_kind,
        directions,
        faults,
        occupancy,
        cursors,
        p1_ids,
        p2_ids,
    )
    return static, arrays


@njit(cache=True)
def _splitmix64(value: np.uint64) -> np.uint64:
    z = value + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _counter_draw(seed: np.uint64, event: int, stream: int) -> np.uint64:
    return _splitmix64(
        seed
        ^ (np.uint64(event) * np.uint64(0xD2B74407B1CE6E93))
        ^ (np.uint64(stream) * np.uint64(0xCA5A826395121157))
    )


def schedule_choice(
    seed: int,
    event: int,
    p1_ids: np.ndarray,
    p2_ids: np.ndarray,
    ratio: float,
) -> tuple[int, int, int]:
    draw_group = int(_counter_draw(np.uint64(seed), event, 1))
    u = (draw_group >> 11) * (1.0 / (1 << 53))
    threshold = ratio * len(p1_ids) / (ratio * len(p1_ids) + len(p2_ids))
    group = 0 if u < threshold else 1
    ids = p1_ids if group == 0 else p2_ids
    draw_actor = int(_counter_draw(np.uint64(seed), event, 2))
    actor = int(ids[draw_actor % len(ids)])
    side_draw = int(_counter_draw(np.uint64(seed), event, 3))
    side = -1 if side_draw & 1 == 0 else 1
    return actor, side, side_draw


@njit(cache=True)
def _has_change(
    occupancy: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    policy_kind: np.ndarray,
    directions: np.ndarray,
    faults: np.ndarray,
    cursors: np.ndarray,
) -> bool:
    n = len(occupancy)
    for pos in range(n):
        actor_id = occupancy[pos]
        if faults[actor_id] != 0:
            continue
        kind = policy_kind[actor_id]
        direction = directions[actor_id]
        if kind == 0:
            for target_pos in (pos - 1, pos + 1):
                if target_pos < 0 or target_pos >= n:
                    continue
                target_id = occupancy[target_pos]
                if faults[target_id] != 0:
                    continue
                if target_pos < pos:
                    inversion = values[actor_id] < values[target_id] if direction == 1 else values[actor_id] > values[target_id]
                else:
                    inversion = values[actor_id] > values[target_id] if direction == 1 else values[actor_id] < values[target_id]
                if inversion:
                    return True
        elif kind == 1 and pos > 0:
            valid = True
            have_prior = False
            prior = 0.0
            for index in range(pos):
                cell_id = occupancy[index]
                if faults[cell_id] != 0:
                    have_prior = False
                    continue
                if have_prior:
                    if (direction == 1 and prior > values[cell_id]) or (direction == -1 and prior < values[cell_id]):
                        valid = False
                        break
                have_prior = True
                prior = values[cell_id]
            if valid:
                target_id = occupancy[pos - 1]
                if faults[target_id] == 0:
                    inversion = values[actor_id] < values[target_id] if direction == 1 else values[actor_id] > values[target_id]
                    if inversion:
                        return True
        elif kind == 2:
            cursor = cursors[actor_id]
            if cursor >= 0 and cursor < n and cursor != positions[actor_id]:
                return True
    return False


@njit(cache=True)
def _correlation_positions(
    occupancy: np.ndarray, values: np.ndarray, policy: np.ndarray, group: int
) -> float:
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for pos in range(len(occupancy)):
        cell_id = occupancy[pos]
        if policy[cell_id] != group:
            continue
        x = float(pos)
        y = values[cell_id]
        n += 1
        sx += x
        sy += y
        sxx += x * x
        syy += y * y
        sxy += x * y
    if n < 2:
        return 0.0
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return (sxy - sx * sy / n) / math.sqrt(vx * vy)


@njit(cache=True)
def _sample_state(
    out: np.ndarray,
    row: int,
    nominal_activation: int,
    interval: int,
    occupancy: np.ndarray,
    previous_occupancy: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
    directions: np.ndarray,
    moved: np.ndarray,
    window_swaps: int,
    abs_flux: np.ndarray,
    signed_flux: np.ndarray,
    cross_swaps: int,
    center_crossings: int,
    activations: np.ndarray,
    cumulative_swaps: int,
    cumulative_memory: int,
    cumulative_stuck_rejections: int,
    quiescent: bool,
) -> None:
    n = len(occupancy)
    comparable = discordant = 0
    for left in range(n - 1):
        vl = values[occupancy[left]]
        for right in range(left + 1, n):
            vr = values[occupancy[right]]
            if vl == vr:
                continue
            comparable += 1
            if vl > vr:
                discordant += 1
    out[row, 0] = 1.0 - 2.0 * discordant / comparable if comparable else 0.0
    same_edges = 0
    for index in range(n - 1):
        same_edges += int(policy[occupancy[index]] == policy[occupancy[index + 1]])
    n1 = 0
    sum1 = sum2 = 0.0
    for pos in range(n):
        if policy[occupancy[pos]] == 0:
            n1 += 1
            sum1 += pos
        else:
            sum2 += pos
    n2 = n - n1
    expected_paper = (n1 * (n1 - 1) + n2 * (n2 - 1)) / (n * n)
    out[row, 1] = same_edges / n - expected_paper
    out[row, 2] = ((sum1 / n1) - (sum2 / n2)) / (n - 1)
    out[row, 3] = _correlation_positions(occupancy, values, policy, 0)
    out[row, 4] = _correlation_positions(occupancy, values, policy, 1)
    satisfied = np.zeros(2, dtype=np.float64)
    opportunities = np.zeros(2, dtype=np.float64)
    for left in range(n - 1):
        left_id = occupancy[left]
        right_id = occupancy[left + 1]
        if values[left_id] == values[right_id]:
            continue
        for cell_id in (left_id, right_id):
            group = policy[cell_id]
            opportunities[group] += 1.0
            ordered = values[left_id] < values[right_id] if directions[cell_id] == 1 else values[left_id] > values[right_id]
            if ordered:
                satisfied[group] += 1.0
    out[row, 5] = satisfied[0] / opportunities[0] if opportunities[0] else 0.0
    out[row, 6] = satisfied[1] / opportunities[1] if opportunities[1] else 0.0
    denom = float(interval) if interval > 0 else 1.0
    out[row, 7] = window_swaps / denom
    moved_count = 0
    hamming = 0
    for index in range(n):
        moved_count += int(moved[index])
        hamming += int(occupancy[index] != previous_occupancy[index])
    out[row, 8] = moved_count / n
    out[row, 9] = hamming / n
    out[row, 10] = abs_flux[0] / denom
    out[row, 11] = abs_flux[1] / denom
    out[row, 12] = signed_flux[0] / denom
    out[row, 13] = signed_flux[1] / denom
    out[row, 14] = cross_swaps / denom
    out[row, 15] = center_crossings / denom
    total_activations = activations[0] + activations[1]
    out[row, 16] = activations[0] / total_activations if total_activations else 0.0
    out[row, 17] = cumulative_swaps
    out[row, 18] = cumulative_memory
    out[row, 19] = cumulative_stuck_rejections
    out[row, 20] = nominal_activation
    out[row, 21] = 1.0 if quiescent else 0.0


@njit(cache=True)
def simulate_phase_kernel(
    values: np.ndarray,
    policy: np.ndarray,
    policy_kind: np.ndarray,
    directions: np.ndarray,
    faults: np.ndarray,
    initial_occupancy: np.ndarray,
    initial_cursors: np.ndarray,
    p1_ids: np.ndarray,
    p2_ids: np.ndarray,
    ratio: float,
    schedule_seed: np.uint64,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    occupancy = initial_occupancy.copy()
    cursors = initial_cursors.copy()
    positions = np.empty(n, dtype=np.int16)
    for pos in range(n):
        positions[occupancy[pos]] = pos
    out = np.zeros((101, len(TRACE_COLUMNS)), dtype=np.float64)
    previous_occupancy = occupancy.copy()
    moved = np.zeros(n, dtype=np.uint8)
    abs_flux = np.zeros(2, dtype=np.float64)
    signed_flux = np.zeros(2, dtype=np.float64)
    activations = np.zeros(2, dtype=np.int64)
    window_swaps = cross_swaps = center_crossings = 0
    cumulative_swaps = cumulative_memory = cumulative_stuck_rejections = 0
    interval = budget // 100
    initial_quiescent = not _has_change(
        occupancy, positions, values, policy_kind, directions, faults, cursors
    )
    _sample_state(
        out, 0, 0, 0, occupancy, previous_occupancy, values, policy, directions,
        moved, 0, abs_flux, signed_flux, 0, 0, activations, 0, 0, 0,
        initial_quiescent,
    )
    if initial_quiescent:
        for row in range(1, 101):
            _sample_state(
                out, row, row * interval, interval, occupancy, occupancy, values,
                policy, directions, moved, 0, abs_flux, signed_flux, 0, 0,
                activations * 0, 0, 0, 0, True,
            )
        terminal = np.asarray((1, 0, 0, 0, 0, 0), dtype=np.int64)
        return out, occupancy, cursors, terminal
    threshold = ratio * len(p1_ids) / (ratio * len(p1_ids) + len(p2_ids))
    quiescent = False
    detected_activation = budget
    event = 0
    next_row = 1
    while event < budget:
        group_draw = _counter_draw(schedule_seed, event, 1)
        u = float(group_draw >> np.uint64(11)) * (1.0 / (1 << 53))
        group = 0 if u < threshold else 1
        actor_draw = _counter_draw(schedule_seed, event, 2)
        if group == 0:
            actor_id = p1_ids[int(actor_draw % np.uint64(len(p1_ids)))]
        else:
            actor_id = p2_ids[int(actor_draw % np.uint64(len(p2_ids)))]
        activations[group] += 1
        actor_pos = positions[actor_id]
        kind = policy_kind[actor_id]
        direction = directions[actor_id]
        target_pos = -1
        outcome = 0  # 0 noop, 1 swap, 2 memory, 3 stuck rejection
        if faults[actor_id] == 0:
            if kind == 0:
                side_draw = _counter_draw(schedule_seed, event, 3)
                target_pos = actor_pos - 1 if side_draw & np.uint64(1) == 0 else actor_pos + 1
                if target_pos >= 0 and target_pos < n:
                    target_id = occupancy[target_pos]
                    if target_pos < actor_pos:
                        inversion = values[actor_id] < values[target_id] if direction == 1 else values[actor_id] > values[target_id]
                    else:
                        inversion = values[actor_id] > values[target_id] if direction == 1 else values[actor_id] < values[target_id]
                    if inversion:
                        outcome = 3 if faults[target_id] != 0 else 1
            elif kind == 1 and actor_pos > 0:
                valid = True
                have_prior = False
                prior = 0.0
                for index in range(actor_pos):
                    prefix_id = occupancy[index]
                    if faults[prefix_id] != 0:
                        have_prior = False
                        continue
                    if have_prior:
                        if (direction == 1 and prior > values[prefix_id]) or (direction == -1 and prior < values[prefix_id]):
                            valid = False
                            break
                    have_prior = True
                    prior = values[prefix_id]
                if valid:
                    target_pos = actor_pos - 1
                    target_id = occupancy[target_pos]
                    inversion = values[actor_id] < values[target_id] if direction == 1 else values[actor_id] > values[target_id]
                    if inversion:
                        outcome = 3 if faults[target_id] != 0 else 1
            elif kind == 2:
                cursor = cursors[actor_id]
                if cursor >= 0 and cursor < n and cursor != actor_pos:
                    target_pos = cursor
                    target_id = occupancy[target_pos]
                    if faults[target_id] != 0 or values[target_id] <= values[actor_id]:
                        outcome = 2
                    else:
                        outcome = 1
        if outcome == 1:
            target_id = occupancy[target_pos]
            delta = target_pos - actor_pos
            occupancy[actor_pos] = target_id
            occupancy[target_pos] = actor_id
            positions[actor_id] = target_pos
            positions[target_id] = actor_pos
            moved[actor_id] = 1
            moved[target_id] = 1
            actor_group = policy[actor_id]
            target_group = policy[target_id]
            abs_flux[actor_group] += abs(delta)
            abs_flux[target_group] += abs(delta)
            signed_flux[actor_group] += delta
            signed_flux[target_group] -= delta
            if actor_group != target_group:
                cross_swaps += 1
            middle = n // 2
            if (actor_pos < middle <= target_pos) or (target_pos < middle <= actor_pos):
                center_crossings += 2
            window_swaps += 1
            cumulative_swaps += 1
        elif outcome == 2:
            cursors[actor_id] += 1 if direction == 1 else -1
            cumulative_memory += 1
        elif outcome == 3:
            cumulative_stuck_rejections += 1
        event += 1
        if event == next_row * interval:
            quiescent = not _has_change(
                occupancy, positions, values, policy_kind, directions, faults, cursors
            )
            _sample_state(
                out, next_row, event, interval, occupancy, previous_occupancy,
                values, policy, directions, moved, window_swaps, abs_flux,
                signed_flux, cross_swaps, center_crossings, activations,
                cumulative_swaps, cumulative_memory, cumulative_stuck_rejections,
                quiescent,
            )
            previous_occupancy[:] = occupancy
            moved[:] = 0
            abs_flux[:] = 0.0
            signed_flux[:] = 0.0
            activations[:] = 0
            window_swaps = cross_swaps = center_crossings = 0
            next_row += 1
            if quiescent:
                detected_activation = event
                break
    if quiescent:
        zero_moved = np.zeros(n, dtype=np.uint8)
        zeros2 = np.zeros(2, dtype=np.float64)
        zeros_i = np.zeros(2, dtype=np.int64)
        while next_row <= 100:
            _sample_state(
                out, next_row, next_row * interval, interval, occupancy, occupancy,
                values, policy, directions, zero_moved, 0, zeros2, zeros2, 0,
                0, zeros_i, cumulative_swaps, cumulative_memory,
                cumulative_stuck_rejections, True,
            )
            next_row += 1
    terminal = np.asarray(
        (
            1 if quiescent else 0,
            detected_activation,
            cumulative_swaps,
            cumulative_memory,
            cumulative_stuck_rejections,
            event,
        ),
        dtype=np.int64,
    )
    return out, occupancy, cursors, terminal


def _line_slope(series: np.ndarray) -> float:
    x = np.linspace(0.0, 1.0, len(series))
    return float(np.polyfit(x, series, 1)[0]) if len(series) > 1 else 0.0


def _oscillation_diagnostic(series: np.ndarray) -> dict[str, float | int | bool]:
    x = np.arange(len(series), dtype=float)
    residual = series - np.polyval(np.polyfit(x, series, 1), x)
    amplitude = float(np.quantile(residual, 0.95) - np.quantile(residual, 0.05))
    spectrum = np.square(np.abs(np.fft.rfft(residual)))
    if len(spectrum) <= 1 or float(spectrum[1:].sum()) <= 0:
        return {
            "oscillation_amplitude": amplitude,
            "spectral_power_fraction": 0.0,
            "oscillation_period": math.inf,
            "mean_crossings": 0,
            "oscillation_pass": False,
        }
    valid = np.arange(1, len(spectrum))
    periods = len(series) / valid
    allowed = valid[(periods >= 4.0) & (periods <= 25.0)]
    if len(allowed) == 0:
        peak = 1
    else:
        peak = int(allowed[np.argmax(spectrum[allowed])])
    power = float(spectrum[peak] / spectrum[1:].sum())
    centered = residual - residual.mean()
    crossings = int(np.sum(centered[:-1] * centered[1:] < 0))
    period = float(len(series) / peak)
    passed = amplitude >= 0.20 and power >= 0.35 and 4 <= period <= 25 and crossings >= 3
    return {
        "oscillation_amplitude": amplitude,
        "spectral_power_fraction": power,
        "oscillation_period": period,
        "mean_crossings": crossings,
        "oscillation_pass": passed,
    }


def _metastability_diagnostic(trace: np.ndarray) -> dict[str, Any]:
    order = trace[:, 0]
    position = trace[:, 2]
    swaps = trace[:, 7]
    passed = False
    plateau_start = plateau_end = -1
    escape_size = 0.0
    for start in range(20, 86):
        end = start + 15
        if end >= len(order):
            break
        if (
            np.ptp(order[start:end]) <= 0.08
            and np.ptp(position[start:end]) <= 0.08
            and float(np.median(swaps[start:end])) <= 0.001
        ):
            displacement = np.maximum(
                np.abs(order[end:] - np.median(order[start:end])),
                np.abs(position[end:] - np.median(position[start:end])),
            )
            candidate = float(displacement.max()) if len(displacement) else 0.0
            if candidate >= 0.20 and float(swaps[end:].max(initial=0.0)) >= 0.002:
                passed = True
                plateau_start, plateau_end, escape_size = start, end - 1, candidate
                break
    return {
        "metastability_pass": passed,
        "plateau_start_index": plateau_start,
        "plateau_end_index": plateau_end,
        "escape_size": escape_size,
    }


def classify_trace(trace: np.ndarray, quiescent: bool) -> dict[str, Any]:
    order = trace[:, 0]
    aggregation = trace[:, 1]
    position = trace[:, 2]
    swaps = trace[:, 7]
    last_quarter = slice(75, 101)
    post_burn = slice(51, 101)
    sustained_fraction = float(np.mean(swaps[post_burn] >= 0.0001))
    median_final_order = float(np.median(order[-10:]))
    order_range = float(np.ptp(order[last_quarter]))
    position_range = float(np.ptp(position[last_quarter]))
    aggregation_range = float(np.ptp(aggregation[last_quarter]))
    order_slope = abs(_line_slope(order[last_quarter]))
    position_slope = abs(_line_slope(position[last_quarter]))
    stable_state = (
        order_range <= 0.12
        and position_range <= 0.12
        and aggregation_range <= 0.10
        and order_slope <= 0.10
        and position_slope <= 0.10
    )
    dominance = (
        not quiescent
        and abs(median_final_order) >= 0.70
        and order_range <= 0.12
        and order_slope <= 0.10
    )
    oscillation_order = _oscillation_diagnostic(order[51:])
    oscillation_position = _oscillation_diagnostic(position[51:])
    oscillation = (
        not quiescent
        and sustained_fraction >= 0.70
        and (
            bool(oscillation_order["oscillation_pass"])
            or bool(oscillation_position["oscillation_pass"])
        )
    )
    metastability = _metastability_diagnostic(trace)
    if quiescent:
        regime = "fixed_quiescence"
    elif dominance:
        regime = "active_dominance"
    elif oscillation:
        regime = "oscillation"
    elif bool(metastability["metastability_pass"]):
        regime = "metastability"
    elif sustained_fraction >= 0.70 and stable_state:
        regime = "dynamic_equilibrium"
    else:
        regime = "unresolved_transient"
    best_oscillation = max(
        (oscillation_order, oscillation_position),
        key=lambda item: float(item["spectral_power_fraction"]),
    )
    return {
        "regime": regime,
        "sustained_turnover_fraction": sustained_fraction,
        "median_final_order_score": median_final_order,
        "dominance_direction": (
            "ascending" if median_final_order >= 0.70
            else "descending" if median_final_order <= -0.70
            else "none"
        ),
        "last_quarter_order_range": order_range,
        "last_quarter_policy_position_range": position_range,
        "last_quarter_corrected_adjacency_range": aggregation_range,
        "last_quarter_order_abs_slope": order_slope,
        "last_quarter_policy_position_abs_slope": position_slope,
        "stable_state_pass": stable_state,
        **best_oscillation,
        **metastability,
    }


def run_one(condition: PhaseCondition, replicate: int, budget: int, stage: str) -> tuple[dict[str, Any], pd.DataFrame]:
    static, arrays = materialize_arrays(condition, replicate)
    values, policy, policy_kind, directions, faults, occupancy, cursors, p1_ids, p2_ids = arrays
    trace, final_occupancy, final_cursors, terminal = simulate_phase_kernel(
        values,
        policy,
        policy_kind,
        directions,
        faults,
        occupancy,
        cursors,
        p1_ids,
        p2_ids,
        condition.activation_ratio,
        np.uint64(int(static["schedule_seed"])),
        budget,
    )
    classification = classify_trace(trace, bool(terminal[0]))
    run_id = f"{static['scenario_id']}:{stage}:B{budget}"
    initial_positions = np.empty(len(occupancy), dtype=int)
    final_positions = np.empty(len(occupancy), dtype=int)
    for pos, cell_id in enumerate(occupancy):
        initial_positions[cell_id] = pos
    for pos, cell_id in enumerate(final_occupancy):
        final_positions[cell_id] = pos
    stuck_preserved = bool(np.array_equal(initial_positions[faults == 1], final_positions[faults == 1]))
    total_abs_flux = float(trace[:, 10:12].sum() * (budget // 100))
    total_signed_flux = float(trace[:, 12:14].sum() * (budget // 100))
    expected_abs_flux = 2.0 * float(
        # Bubble and Insertion ranges are one; Selection is represented in the
        # experienced flux itself.  The exact identity below is independently
        # checked using signed conservation and per-event validation fixtures.
        total_abs_flux / 2.0
    )
    summary = {
        "schema_version": STEP_SCHEMA,
        "research_step_id": "S13",
        "run_id": run_id,
        "stage": stage,
        "activation_budget": budget,
        **static,
        "terminal": "quiescent" if terminal[0] else "event_budget",
        "quiescence_detection_activation": int(terminal[1]),
        "accepted_swaps": int(terminal[2]),
        "memory_updates": int(terminal[3]),
        "stuck_rejections": int(terminal[4]),
        "executed_activations": int(terminal[5]),
        "final_occupancy_sha256": canonical_hash(final_occupancy.tolist()),
        "final_cursor_sha256": canonical_hash(final_cursors.tolist()),
        "occupancy_bijection": bool(
            np.array_equal(np.sort(final_occupancy), np.arange(len(final_occupancy)))
        ),
        "stuck_positions_preserved": stuck_preserved,
        "signed_flux_conservation_error": abs(total_signed_flux),
        "absolute_flux_accounted": expected_abs_flux >= 0.0,
        "final_order_score": float(trace[-1, 0]),
        "final_corrected_adjacency": float(trace[-1, 1]),
        "final_policy_position_score": float(trace[-1, 2]),
        "postburn_median_swap_rate": float(np.median(trace[51:, 7])),
        "postburn_median_total_flux_rate": float(np.median(trace[51:, 10] + trace[51:, 11])),
        "achieved_first_activation_share": float(
            np.average(trace[1:, 16], weights=np.ones(100))
        ),
        **classification,
    }
    frame = pd.DataFrame(trace, columns=TRACE_COLUMNS)
    frame.insert(0, "checkpoint", np.arange(101, dtype=np.int16))
    frame.insert(0, "run_id", run_id)
    frame.insert(0, "condition_id", condition.condition_id)
    frame.insert(0, "replicate", replicate)
    frame.insert(0, "stage", stage)
    return summary, frame


def _run_batch(payload: tuple[str, int, int, list[dict[str, Any]]]) -> dict[str, Any]:
    stage, budget, chunk_index, tasks = payload
    stage_dir = CACHE / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    traces: list[pd.DataFrame] = []
    for task in tasks:
        condition = PhaseCondition.from_dict(task["condition"])
        summary, trace = run_one(condition, int(task["replicate"]), budget, stage)
        summaries.append(summary)
        traces.append(trace)
    summary_path = stage_dir / f"summary_{chunk_index:05d}.parquet"
    trace_path = stage_dir / f"trace_{chunk_index:05d}.parquet"
    write_parquet(pd.DataFrame(summaries), summary_path)
    write_parquet(pd.concat(traces, ignore_index=True), trace_path)
    return {
        "chunk": chunk_index,
        "runs": len(tasks),
        "summary": str(summary_path),
        "trace": str(trace_path),
    }


def _combine_stage(stage: str) -> tuple[Path, Path]:
    stage_dir = CACHE / stage
    summary_files = sorted(stage_dir.glob("summary_*.parquet"))
    trace_files = sorted(stage_dir.glob("trace_*.parquet"))
    if not summary_files or len(summary_files) != len(trace_files):
        raise ValueError(f"stage {stage} has incomplete chunks")
    summaries = pd.concat((pd.read_parquet(path) for path in summary_files), ignore_index=True)
    traces = pd.concat((pd.read_parquet(path) for path in trace_files), ignore_index=True)
    summary_path = CACHE / f"{stage}_runs.parquet"
    trace_path = CACHE / f"{stage}_traces.parquet"
    write_parquet(summaries.sort_values("run_id"), summary_path)
    write_parquet(traces.sort_values(["run_id", "checkpoint"]), trace_path)
    return summary_path, trace_path


def run_stage(stage: str, tasks: list[dict[str, Any]], budget: int, workers: int) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1,8]")
    stage_dir = CACHE / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 12
    chunks = [tasks[index : index + chunk_size] for index in range(0, len(tasks), chunk_size)]
    payloads = [(stage, budget, index, chunk) for index, chunk in enumerate(chunks)]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_run_batch, payloads, chunksize=1):
            results.append(result)
    if sum(item["runs"] for item in results) != len(tasks):
        raise AssertionError("stage run accounting failed")
    _combine_stage(stage)
    write_json(
        CACHE / f"{stage}_accounting.json",
        {
            "schema": "e04.s13.stage_accounting.v1",
            "stage": stage,
            "budget": budget,
            "workers": workers,
            "runCount": len(tasks),
            "chunkCount": len(chunks),
        },
    )


def verify_manifest(step: str) -> dict[str, Any]:
    directory = Path("/artifacts/research_steps") / step
    manifest_path = directory / "artifact_manifest.json"
    document = json.loads(manifest_path.read_text())
    artifacts = document.get("artifacts", document.get("files", []))
    checks: list[dict[str, Any]] = []
    for item in artifacts:
        path = directory / item["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "observed": observed,
                "passed": observed == item["sha256"],
            }
        )
    return {
        "manifestSha256": sha256_file(manifest_path),
        "artifactCount": len(artifacts),
        "checks": checks,
        "allPassed": all(item["passed"] for item in checks),
    }


def freeze() -> None:
    if (Path("/artifacts/research_steps/S14")).exists():
        raise PermissionError("S14 artifact directory exists before S13")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions()
    upstream = {
        step: verify_manifest(step)
        for step in [f"S{index:02d}" for index in range(1, 12)] + ["S11R", "S12"]
    }
    if not all(item["allPassed"] for item in upstream.values()):
        raise ValueError("an upstream artifact manifest failed before freeze")
    contract = json.loads(CONTRACT_PATH.read_text())
    write_json(OUTPUT / "preregistration.json", contract)
    condition_frame = pd.DataFrame([condition.to_dict() for condition in conditions])
    write_parquet(condition_frame, OUTPUT / "condition_catalog.parquet")
    manifest_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for replicate in BASE_REPLICATES:
            static, arrays = materialize_arrays(condition, replicate)
            manifest_rows.append(
                {
                    **static,
                    "initial_order_score": float(
                        _initial_order_score(arrays[0], arrays[5])
                    ),
                }
            )
    manifest = pd.DataFrame(manifest_rows).sort_values(["condition_id", "replicate"])
    write_parquet(manifest, OUTPUT / "scenario_manifest.parquet")
    write_json(
        OUTPUT / "upstream_immutability_audit.json",
        {
            "schema": "e04.s13.upstream_immutability.v1",
            "researchStepId": "S13",
            "phase": "freeze",
            "steps": upstream,
            "allPassed": True,
            "s14Absent": True,
        },
    )
    write_json(
        OUTPUT / "freeze_record.json",
        {
            "schema": "e04.s13.freeze_record.v1",
            "researchStepId": "S13",
            "frozenAt": datetime.now(timezone.utc).isoformat(),
            "contractPath": str(CONTRACT_PATH.relative_to(REPOSITORY)),
            "contractSha256": sha256_file(CONTRACT_PATH),
            "preregistrationSha256": sha256_file(OUTPUT / "preregistration.json"),
            "repositoryHead": _git("rev-parse", "HEAD"),
            "conditionCount": len(conditions),
            "baseRunCount": len(manifest),
            "primaryActivationBudget": PRIMARY_BUDGET,
            "longActivationBudget": LONG_BUDGET,
            "baseReplicates": list(BASE_REPLICATES),
            "sequentialReplicates": list(SEQUENTIAL_REPLICATES),
            "s14Absent": True,
        },
    )


def _initial_order_score(values: np.ndarray, occupancy: np.ndarray) -> float:
    ordered = values[occupancy]
    comparable = discordant = 0
    for left in range(len(ordered) - 1):
        for right in range(left + 1, len(ordered)):
            if ordered[left] == ordered[right]:
                continue
            comparable += 1
            discordant += int(ordered[left] > ordered[right])
    return 1 - 2 * discordant / comparable if comparable else 0.0


def base_tasks() -> list[dict[str, Any]]:
    return [
        {"condition": condition.to_dict(), "replicate": replicate}
        for condition in build_conditions()
        for replicate in BASE_REPLICATES
    ]


def _modal_table(runs: pd.DataFrame) -> pd.DataFrame:
    counts = (
        runs.groupby(["condition_id", "regime"]).size().rename("n").reset_index()
    )
    totals = runs.groupby("condition_id").size().rename("replicates")
    rows: list[dict[str, Any]] = []
    for condition_id, group in counts.groupby("condition_id"):
        group = group.sort_values(["n", "regime"], ascending=[False, True])
        total = int(totals.loc[condition_id])
        modal = str(group.iloc[0].regime)
        modal_n = int(group.iloc[0].n)
        probabilities = {regime: 0.0 for regime in REGIMES}
        for row in group.itertuples(index=False):
            probabilities[str(row.regime)] = int(row.n) / total
        entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
        rows.append(
            {
                "condition_id": condition_id,
                "replicates": total,
                "modal_regime": modal,
                "modal_count": modal_n,
                "modal_fraction": modal_n / total,
                "normalized_entropy": entropy / math.log(len(REGIMES)),
                **{f"prob_{key}": value for key, value in probabilities.items()},
            }
        )
    return pd.DataFrame(rows)


def select_boundary() -> None:
    runs = pd.read_parquet(CACHE / "base_runs.parquet")
    if len(runs) != 18_000 or runs.run_id.nunique() != 18_000:
        raise ValueError("base outcome accounting is incomplete")
    modal = _modal_table(runs)
    catalog = pd.read_parquet(OUTPUT / "condition_catalog.parquet")
    table = catalog.merge(modal, on="condition_id", validate="one_to_one")
    selected: set[str] = set(
        table.loc[table.modal_fraction < 1.0, "condition_id"].astype(str)
    )
    fixed = [
        "input_profile",
        "first_policy",
        "second_policy",
        "first_direction",
        "second_direction",
        "fault_count",
        "state_profile",
    ]
    for varying, other in (
        ("first_count", "activation_ratio"),
        ("activation_ratio", "first_count"),
    ):
        group_keys = fixed + [other]
        for _, group in table.groupby(group_keys, sort=False):
            group = group.sort_values(varying)
            rows = list(group.itertuples(index=False))
            for left, right in zip(rows, rows[1:]):
                if left.modal_regime != right.modal_regime:
                    selected.add(str(left.condition_id))
                    selected.add(str(right.condition_id))
    selection = table[["condition_id", "modal_regime", "modal_fraction", "normalized_entropy"]].copy()
    selection["nonunanimous_trigger"] = selection.modal_fraction.lt(1.0)
    selection["selected_for_sequential"] = selection.condition_id.isin(selected)
    selection["adjacent_boundary_trigger"] = (
        selection.selected_for_sequential & ~selection.nonunanimous_trigger
    )
    write_parquet(selection.sort_values("condition_id"), OUTPUT / "boundary_selection.parquet")
    write_json(
        OUTPUT / "boundary_selection_summary.json",
        {
            "schema": "e04.s13.boundary_selection.v1",
            "baseConditions": 4500,
            "baseRuns": 18000,
            "nonunanimousConditions": int(selection.nonunanimous_trigger.sum()),
            "adjacentOnlyConditions": int(selection.adjacent_boundary_trigger.sum()),
            "selectedConditions": int(selection.selected_for_sequential.sum()),
            "addedReplicatesPerCondition": len(SEQUENTIAL_REPLICATES),
            "expectedAddedRuns": int(selection.selected_for_sequential.sum()) * len(SEQUENTIAL_REPLICATES),
        },
    )


def sequential_tasks() -> list[dict[str, Any]]:
    selection = pd.read_parquet(OUTPUT / "boundary_selection.parquet")
    selected = set(
        selection.loc[selection.selected_for_sequential, "condition_id"].astype(str)
    )
    return [
        {"condition": condition.to_dict(), "replicate": replicate}
        for condition in build_conditions()
        if condition.condition_id in selected
        for replicate in SEQUENTIAL_REPLICATES
    ]


def _paper_anchor_ids(catalog: pd.DataFrame) -> set[str]:
    paper_orientations = {
        "Bubble:descending|Selection:ascending",
        "Bubble:ascending|Insertion:descending",
        "Insertion:ascending|Selection:descending",
    }
    return set(
        catalog.loc[
            catalog.direction_assignment.isin(paper_orientations)
            & catalog.first_count.eq(50)
            & catalog.activation_ratio.eq(1.0)
            & catalog.fault_count.eq(0)
            & catalog.state_profile.eq("random_absent"),
            "condition_id",
        ].astype(str)
    )


def select_long() -> None:
    base = pd.read_parquet(CACHE / "base_runs.parquet")
    sequential_path = CACHE / "sequential_runs.parquet"
    sequential = pd.read_parquet(sequential_path) if sequential_path.exists() else base.iloc[0:0]
    runs = pd.concat([base, sequential], ignore_index=True)
    modal = _modal_table(runs)
    catalog = pd.read_parquet(OUTPUT / "condition_catalog.parquet")
    boundary = pd.read_parquet(OUTPUT / "boundary_selection.parquet")
    table = catalog.merge(modal, on="condition_id", validate="one_to_one").merge(
        boundary[["condition_id", "selected_for_sequential", "adjacent_boundary_trigger"]],
        on="condition_id",
        validate="one_to_one",
    )
    anchors = _paper_anchor_ids(catalog)
    if len(anchors) != 6:
        raise AssertionError(f"expected six paper anchors, got {len(anchors)}")
    candidates = table[
        table.selected_for_sequential & table.adjacent_boundary_trigger
    ].copy()
    strata = [
        "input_profile",
        "first_policy",
        "second_policy",
        "first_direction",
        "second_direction",
        "fault_count",
        "state_profile",
    ]
    candidates = candidates.sort_values(
        [*strata, "normalized_entropy", "condition_id"],
        ascending=[True] * len(strata) + [False, True],
    )
    stratified = set(candidates.groupby(strata, sort=False).head(1).condition_id.astype(str))
    nonanchors = table[table.condition_id.isin(stratified - anchors)].sort_values(
        ["normalized_entropy", "condition_id"], ascending=[False, True]
    )
    selected = anchors | set(nonanchors.head(186 - len(anchors)).condition_id.astype(str))
    selection = table[
        ["condition_id", "modal_regime", "modal_fraction", "normalized_entropy"]
    ].copy()
    selection["paper_anchor"] = selection.condition_id.isin(anchors)
    selection["selected_long"] = selection.condition_id.isin(selected)
    write_parquet(selection.sort_values("condition_id"), OUTPUT / "long_budget_selection.parquet")
    available = runs.groupby("condition_id").replicate.apply(lambda x: sorted(set(map(int, x))))
    expected_runs = sum(len(available.loc[condition_id]) for condition_id in selected)
    write_json(
        OUTPUT / "long_budget_selection_summary.json",
        {
            "schema": "e04.s13.long_selection.v1",
            "paperAnchors": len(anchors),
            "stratifiedBoundaryCandidates": len(stratified),
            "selectedConditions": len(selected),
            "expectedRuns": int(expected_runs),
            "cap": 186,
        },
    )


def long_tasks() -> list[dict[str, Any]]:
    selection = pd.read_parquet(OUTPUT / "long_budget_selection.parquet")
    selected = set(selection.loc[selection.selected_long, "condition_id"].astype(str))
    base = pd.read_parquet(CACHE / "base_runs.parquet")
    sequential_path = CACHE / "sequential_runs.parquet"
    sequential = pd.read_parquet(sequential_path) if sequential_path.exists() else base.iloc[0:0]
    available = pd.concat([base, sequential]).groupby("condition_id").replicate.apply(
        lambda values: sorted(set(map(int, values)))
    )
    return [
        {"condition": condition.to_dict(), "replicate": replicate}
        for condition in build_conditions()
        if condition.condition_id in selected
        for replicate in available.loc[condition.condition_id]
    ]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def _condition_phase_map(runs: pd.DataFrame) -> pd.DataFrame:
    modal = _modal_table(runs)
    catalog = pd.read_parquet(OUTPUT / "condition_catalog.parquet")
    aggregate = runs.groupby("condition_id").agg(
        mean_final_order_score=("final_order_score", "mean"),
        sd_final_order_score=("final_order_score", "std"),
        mean_final_corrected_adjacency=("final_corrected_adjacency", "mean"),
        mean_postburn_swap_rate=("postburn_median_swap_rate", "mean"),
        mean_postburn_flux_rate=("postburn_median_total_flux_rate", "mean"),
        quiescent_fraction=("terminal", lambda values: float(np.mean(values == "quiescent"))),
        mean_activation_share=("achieved_first_activation_share", "mean"),
    ).reset_index()
    table = catalog.merge(modal, on="condition_id", validate="one_to_one").merge(
        aggregate, on="condition_id", validate="one_to_one"
    )
    intervals = [wilson_interval(int(row.modal_count), int(row.replicates)) for row in table.itertuples()]
    table["modal_wilson_lower"] = [item[0] for item in intervals]
    table["modal_wilson_upper"] = [item[1] for item in intervals]
    table["boundary_uncertain"] = table.modal_fraction.lt(2 / 3) | table.modal_wilson_lower.lt(0.40)
    return table


def _long_stability(primary: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition_id", "replicate"]
    columns = keys + [
        "run_id",
        "regime",
        "dominance_direction",
        "final_order_score",
        "final_corrected_adjacency",
        "terminal",
        "accepted_swaps",
    ]
    merged = primary[columns].merge(
        long[columns], on=keys, suffixes=("_primary", "_long"), validate="one_to_one"
    )
    merged["regime_agreement"] = merged.regime_primary.eq(merged.regime_long)
    merged["dominance_direction_agreement"] = merged.dominance_direction_primary.eq(
        merged.dominance_direction_long
    )
    merged["endpoint_order_drift"] = merged.final_order_score_long - merged.final_order_score_primary
    merged["endpoint_aggregation_drift"] = (
        merged.final_corrected_adjacency_long - merged.final_corrected_adjacency_primary
    )
    return merged


def _time_means(traces: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    factors = runs[
        [
            "run_id",
            "input_profile",
            "policy_pair",
            "direction_assignment",
            "first_count",
            "activation_ratio",
            "fault_count",
            "state_profile",
            "regime",
        ]
    ]
    merged = traces.merge(factors, on="run_id", validate="many_to_one")
    value_cols = [column for column in TRACE_COLUMNS if column != "nominal_activation"]
    result = (
        merged.groupby(
            [
                "condition_id",
                "checkpoint",
                "input_profile",
                "policy_pair",
                "direction_assignment",
                "first_count",
                "activation_ratio",
                "fault_count",
                "state_profile",
            ],
            observed=True,
        )[value_cols]
        .mean()
        .reset_index()
    )
    return result


def _plot_phase_map(phase: pd.DataFrame) -> None:
    colors = {
        "fixed_quiescence": "#4d4d4d",
        "active_dominance": "#b2182b",
        "oscillation": "#7b3294",
        "metastability": "#ef8a62",
        "dynamic_equilibrium": "#2166ac",
        "unresolved_transient": "#bdbdbd",
    }
    anchor = phase[
        phase.fault_count.eq(0)
        & phase.state_profile.eq("random_absent")
        & phase.input_profile.eq("unique_1_100")
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, (assignment, group) in zip(axes.ravel(), anchor.groupby("direction_assignment")):
        for regime in REGIMES:
            subset = group[group.modal_regime.eq(regime)]
            axis.scatter(
                subset.first_count,
                subset.activation_ratio,
                s=50 + 100 * subset.modal_fraction,
                color=colors[regime],
                edgecolor="white",
                linewidth=0.4,
                label=regime,
            )
        axis.set_yscale("log", base=2)
        axis.set_yticks(ACTIVATION_RATIOS, [str(item) for item in ACTIVATION_RATIOS])
        axis.set_xticks(FIRST_COUNTS)
        axis.set_title(assignment.replace("|", "\n"), fontsize=10)
        axis.set_xlabel("First-policy count")
        axis.set_ylabel("Per-identity activation multiplier")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[key], label=key)
        for key in REGIMES
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=3)
    figure.suptitle("S13 opposing-goal phase map: unique, no-fault, random-absent slice")
    for suffix in ("png", "svg"):
        figure.savefig(OUTPUT / f"conflict_phase_map.{suffix}", dpi=180)
    plt.close(figure)


def _plot_flux(time_means: pd.DataFrame, phase: pd.DataFrame) -> None:
    anchor_ids = _paper_anchor_ids(phase)
    subset = time_means[time_means.condition_id.isin(anchor_ids)]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (condition_id, group) in zip(axes.ravel(), subset.groupby("condition_id")):
        x = group.checkpoint / 100
        axis.plot(x, group.order_score, color="#b2182b", label="value order")
        axis.plot(x, group.policy_position_score, color="#2166ac", label="policy position")
        flux = group.p1_absolute_flux_rate + group.p2_absolute_flux_rate
        axis2 = axis.twinx()
        axis2.plot(x, flux, color="#4daf4a", alpha=0.65, label="absolute flux")
        axis.set_title(condition_id.replace("S13-", ""), fontsize=8)
        axis.set_xlabel("Activation-budget fraction")
        axis.set_ylabel("Macrostate")
        axis2.set_ylabel("Flux / activation")
    handles = [
        plt.Line2D([0], [0], color="#b2182b", label="value order"),
        plt.Line2D([0], [0], color="#2166ac", label="policy position"),
        plt.Line2D([0], [0], color="#4daf4a", label="absolute flux"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=3)
    figure.suptitle("S13 paper-oriented anchors: state and time-resolved flux")
    for suffix in ("png", "svg"):
        figure.savefig(OUTPUT / f"state_flux_diagnostics.{suffix}", dpi=180)
    plt.close(figure)


def analyze() -> None:
    base_runs = pd.read_parquet(CACHE / "base_runs.parquet")
    sequential_path = CACHE / "sequential_runs.parquet"
    sequential_runs = pd.read_parquet(sequential_path) if sequential_path.exists() else base_runs.iloc[0:0]
    primary = pd.concat([base_runs, sequential_runs], ignore_index=True)
    if primary.run_id.duplicated().any():
        raise ValueError("duplicate primary run IDs")
    phase = _condition_phase_map(primary)
    write_parquet(phase, OUTPUT / "conflict_phase_map.parquet")
    run_columns = [column for column in primary.columns if column not in {"schedule_seed"}]
    write_parquet(primary[run_columns].sort_values("run_id"), OUTPUT / "phase_run_summaries.parquet")
    base_traces = pd.read_parquet(CACHE / "base_traces.parquet")
    sequential_trace_path = CACHE / "sequential_traces.parquet"
    sequential_traces = pd.read_parquet(sequential_trace_path) if sequential_trace_path.exists() else base_traces.iloc[0:0]
    time_means = _time_means(
        pd.concat([base_traces, sequential_traces], ignore_index=True), primary
    )
    write_parquet(time_means, OUTPUT / "time_resolved_transport.parquet")
    long_runs = pd.read_parquet(CACHE / "long_runs.parquet")
    stability = _long_stability(primary, long_runs)
    write_parquet(stability, OUTPUT / "long_budget_stability.parquet")
    condition_stability = stability.groupby("condition_id").agg(
        runs=("replicate", "size"),
        regime_agreement=("regime_agreement", "mean"),
        direction_agreement=("dominance_direction_agreement", "mean"),
        mean_abs_order_drift=("endpoint_order_drift", lambda x: float(np.mean(np.abs(x)))),
        mean_abs_aggregation_drift=("endpoint_aggregation_drift", lambda x: float(np.mean(np.abs(x)))),
    ).reset_index()
    primary_modal = _modal_table(primary).rename(columns={"modal_regime": "primary_modal"})
    long_modal = _modal_table(long_runs).rename(columns={"modal_regime": "long_modal"})
    condition_stability = condition_stability.merge(
        primary_modal[["condition_id", "primary_modal"]], on="condition_id"
    ).merge(long_modal[["condition_id", "long_modal"]], on="condition_id")
    condition_stability["modal_agreement"] = condition_stability.primary_modal.eq(
        condition_stability.long_modal
    )
    write_parquet(condition_stability, OUTPUT / "condition_stability.parquet")
    long_traces = pd.read_parquet(CACHE / "long_traces.parquet")
    long_lookup = long_runs.set_index(["condition_id", "replicate"])
    prefix_checks: list[dict[str, Any]] = []
    # Long checkpoints 0,10,...,100 coincide with primary checkpoints 0,1,...,10.
    base_lookup = pd.concat([base_traces, sequential_traces]).set_index(
        ["condition_id", "replicate", "checkpoint"]
    )
    for (condition_id, replicate), _ in long_lookup.iterrows():
        long_part = long_traces[
            long_traces.condition_id.eq(condition_id)
            & long_traces.replicate.eq(replicate)
            & long_traces.checkpoint.isin(range(0, 21, 2))
        ].sort_values("checkpoint")
        primary_part = base_lookup.loc[
            [(condition_id, replicate, point) for point in range(11)]
        ].reset_index()
        compared = long_part[list(TRACE_COLUMNS[:20])].to_numpy(float) - primary_part[list(TRACE_COLUMNS[:20])].to_numpy(float)
        prefix_checks.append(
            {
                "condition_id": condition_id,
                "replicate": int(replicate),
                "max_abs_metric_error": float(np.max(np.abs(compared))),
                "passed": bool(np.max(np.abs(compared)) <= 1e-12),
            }
        )
    prefix_frame = pd.DataFrame(prefix_checks)
    write_parquet(prefix_frame, OUTPUT / "long_budget_prefix_replay.parquet")
    _plot_phase_map(phase)
    _plot_flux(time_means, phase)
    prevalence = primary.regime.value_counts().reindex(REGIMES, fill_value=0)
    stable_major = stability.groupby("regime_primary").regime_agreement.mean()
    anchors = _paper_anchor_ids(phase)
    anchor_stability = condition_stability[condition_stability.condition_id.isin(anchors)]
    paper_dynamic = int(
        (
            anchor_stability.primary_modal.eq("dynamic_equilibrium")
            & anchor_stability.long_modal.eq("dynamic_equilibrium")
        ).sum()
    )
    summary = {
        "schema": "e04.s13.analysis_summary.v1",
        "researchStepId": "S13",
        "conditionCount": len(phase),
        "primaryRunCount": len(primary),
        "baseRunCount": len(base_runs),
        "sequentialRunCount": len(sequential_runs),
        "longRunCount": len(long_runs),
        "regimeRunCounts": prevalence.to_dict(),
        "conditionModalCounts": phase.modal_regime.value_counts().to_dict(),
        "uncertainBoundaryConditions": int(phase.boundary_uncertain.sum()),
        "longConditionCount": int(condition_stability.condition_id.nunique()),
        "longRunRegimeAgreement": float(stability.regime_agreement.mean()),
        "longConditionModalAgreement": float(condition_stability.modal_agreement.mean()),
        "majorRegimeStability": stable_major.to_dict(),
        "prefixReplayAllPassed": bool(prefix_frame.passed.all()),
        "paperAnchorDynamicEquilibriumStableCount": paper_dynamic,
        "paperAnchorCount": len(anchors),
    }
    write_json(OUTPUT / "analysis_summary.json", summary)


def _ordinary_scenario(condition: PhaseCondition, replicate: int, budget: int) -> tuple[Scenario, tuple[np.ndarray, ...], dict[str, Any]]:
    static, arrays = materialize_arrays(condition, replicate)
    values, policy, policy_kind, directions, faults, occupancy, cursors, p1_ids, p2_ids = arrays
    cells = []
    for cell_id in range(len(values)):
        actual_policy = condition.first_policy if policy[cell_id] == 0 else condition.second_policy
        cells.append(
            Cell(
                f"cell-{cell_id:04d}",
                float(values[cell_id]),
                Policy(actual_policy),
                Direction.ASCENDING if directions[cell_id] == 1 else Direction.DESCENDING,
                FaultMode.STUCK if faults[cell_id] else FaultMode.NORMAL,
            )
        )
    cursor_map = {
        f"cell-{cell_id:04d}": int(cursors[cell_id])
        for cell_id in range(len(values))
        if cursors[cell_id] > -32768
    }
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(f"cell-{int(cell_id):04d}" for cell_id in occupancy),
        initial_selection_cursors=cursor_map,
        seed=0,
        max_activations=budget,
        architecture=Architecture.CELL_VIEW,
        scheduler="s13_counter_weighted",
        generation_key=f"E04/S13/validation/{condition.condition_id}/R{replicate}",
        requested_fault_count=condition.fault_count,
    )
    return scenario, arrays, static


def validate_kernel() -> None:
    conditions = build_conditions()
    selected = [conditions[index] for index in (0, 173, 911, 1822, 2711, 4499)]
    validations: list[dict[str, Any]] = []
    for ordinal, condition in enumerate(selected):
        budget = 400 + ordinal * 37
        replicate = ordinal % 4
        scenario, arrays, static = _ordinary_scenario(condition, replicate, budget)
        values, policy, policy_kind, directions, faults, occupancy, cursors, p1_ids, p2_ids = arrays

        def scheduler(event_index: int, remaining: int) -> tuple[ScheduledOpportunity, ...]:
            actor, side, side_draw = schedule_choice(
                int(static["schedule_seed"]),
                event_index,
                p1_ids,
                p2_ids,
                condition.activation_ratio,
            )
            actor_id = f"cell-{actor:04d}"
            if policy_kind[actor] == POLICY_CODE["Bubble"] and faults[actor] == 0:
                return (
                    ScheduledOpportunity(
                        actor_id,
                        random_draws=(("bubble_side", event_index, 0, side_draw),),
                        stream_consumption=(("bubble_side", 1),),
                        bubble_side="left" if side == -1 else "right",
                    ),
                )
            return (ScheduledOpportunity(actor_id),)

        ordinary = ordinary_run(scenario, trace_mode="none", schedule_factory=scheduler)
        fast_trace, fast_occupancy, fast_cursors, terminal = simulate_phase_kernel(
            values,
            policy,
            policy_kind,
            directions,
            faults,
            occupancy,
            cursors,
            p1_ids,
            p2_ids,
            condition.activation_ratio,
            np.uint64(int(static["schedule_seed"])),
            budget,
        )
        ordinary_occupancy = np.asarray(
            [int(cell_id.split("-")[1]) for cell_id in ordinary.summary["finalOccupancy"]],
            dtype=np.int16,
        )
        ordinary_cursors = np.full(len(values), -32768, dtype=np.int16)
        for cell_id, cursor in ordinary.final_state["selectionCursors"].items():
            ordinary_cursors[int(cell_id.split("-")[1])] = int(cursor)
        same_state = bool(
            np.array_equal(ordinary_occupancy, fast_occupancy)
            and np.array_equal(ordinary_cursors, fast_cursors)
        )
        swap_match = int(ordinary.summary["ledger"]["acceptedSwaps"]) == int(terminal[2])
        memory_match = int(ordinary.summary["ledger"]["memoryUpdates"]) == int(terminal[3])
        validations.append(
            {
                "conditionId": condition.condition_id,
                "replicate": replicate,
                "budget": budget,
                "ordinaryTerminal": ordinary.summary["stopReason"],
                "fastTerminal": "quiescent" if terminal[0] else "event_budget",
                "sameFinalState": same_state,
                "swapCountMatch": swap_match,
                "memoryCountMatch": memory_match,
                "passed": same_state and swap_match and memory_match,
                "fastFinalOrder": float(fast_trace[-1, 0]),
            }
        )
    # Scheduler frequencies and deterministic-prefix identity.
    frequency_checks: list[dict[str, Any]] = []
    condition = next(item for item in conditions if item.first_count == 35 and item.activation_ratio == 2.0)
    static, arrays = materialize_arrays(condition, 0)
    p1_ids, p2_ids = arrays[-2], arrays[-1]
    actors_short = [
        schedule_choice(int(static["schedule_seed"]), event, p1_ids, p2_ids, 2.0)[0]
        for event in range(10_000)
    ]
    actors_long_prefix = [
        schedule_choice(int(static["schedule_seed"]), event, p1_ids, p2_ids, 2.0)[0]
        for event in range(10_000)
    ]
    achieved = float(np.mean(np.isin(actors_short, p1_ids)))
    expected = float(static["expected_first_activation_share"])
    frequency_checks.append(
        {
            "events": 10000,
            "expected": expected,
            "achieved": achieved,
            "absoluteError": abs(achieved - expected),
            "prefixIdentical": actors_short == actors_long_prefix,
            "passed": abs(achieved - expected) <= 0.02 and actors_short == actors_long_prefix,
        }
    )
    # Outcome-independent classifier fixtures.
    fixtures: dict[str, np.ndarray] = {}
    base = np.zeros((101, len(TRACE_COLUMNS)), dtype=float)
    fixed = base.copy()
    fixed[:, 21] = 1
    fixtures["fixed_quiescence"] = fixed
    dominance = base.copy()
    dominance[:71, 0] = np.linspace(0, 0.9, 71)
    dominance[71:, 0] = 0.9
    dominance[:, 7] = 0.002
    fixtures["active_dominance"] = dominance
    equilibrium = base.copy()
    equilibrium[:, 0] = 0.1
    equilibrium[:, 2] = -0.1
    equilibrium[:, 7] = 0.002
    fixtures["dynamic_equilibrium"] = equilibrium
    oscillation = base.copy()
    oscillation[:, 0] = 0.25 * np.sin(2 * np.pi * np.arange(101) / 10)
    oscillation[:, 7] = 0.002
    fixtures["oscillation"] = oscillation
    metastable = base.copy()
    metastable[:, 7] = 0.003
    metastable[25:45, 7] = 0.0002
    metastable[45:, 0] = 0.35
    fixtures["metastability"] = metastable
    fixture_results = {
        expected: classify_trace(trace, expected == "fixed_quiescence")["regime"]
        for expected, trace in fixtures.items()
    }
    fixture_pass = all(expected == observed for expected, observed in fixture_results.items())
    result = {
        "schema": "e04.s13.kernel_validation.v1",
        "researchStepId": "S13",
        "ordinaryComparisons": validations,
        "scheduleCalibration": frequency_checks,
        "classifierFixtures": fixture_results,
        "allPassed": all(item["passed"] for item in validations)
        and all(item["passed"] for item in frequency_checks)
        and fixture_pass,
    }
    write_json(OUTPUT / "kernel_validation.json", result)
    if not result["allPassed"]:
        raise AssertionError("S13 kernel validation failed")


def finalize() -> None:
    required = [
        "preregistration.json",
        "freeze_record.json",
        "condition_catalog.parquet",
        "scenario_manifest.parquet",
        "kernel_validation.json",
        "boundary_selection.parquet",
        "long_budget_selection.parquet",
        "conflict_phase_map.parquet",
        "phase_run_summaries.parquet",
        "time_resolved_transport.parquet",
        "long_budget_stability.parquet",
        "condition_stability.parquet",
        "long_budget_prefix_replay.parquet",
        "analysis_summary.json",
        "conflict_phase_map.png",
        "conflict_phase_map.svg",
        "state_flux_diagnostics.png",
        "state_flux_diagnostics.svg",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing S13 outputs: {missing}")
    upstream = {
        step: verify_manifest(step)
        for step in [f"S{index:02d}" for index in range(1, 12)] + ["S11R", "S12"]
    }
    s14_absent = not Path("/artifacts/research_steps/S14").exists()
    all_upstream = all(item["allPassed"] for item in upstream.values())
    audit = {
        "schema": "e04.s13.upstream_immutability.v1",
        "researchStepId": "S13",
        "phase": "final",
        "steps": upstream,
        "allPassed": all_upstream,
        "s14Absent": s14_absent,
    }
    write_json(OUTPUT / "upstream_immutability_audit.json", audit)
    summary = json.loads((OUTPUT / "analysis_summary.json").read_text())
    validation = {
        "schema": "e04.s13.validation_summary.v1",
        "researchStepId": "S13",
        "upstreamImmutability": all_upstream,
        "s14Absent": s14_absent,
        "kernelValidation": json.loads((OUTPUT / "kernel_validation.json").read_text())["allPassed"],
        "primaryConditionCount": summary["conditionCount"] == 4500,
        "baseRunCount": summary["baseRunCount"] == 18000,
        "longPrefixReplay": summary["prefixReplayAllPassed"],
        "occupancyBijection": bool(pd.read_parquet(OUTPUT / "phase_run_summaries.parquet").occupancy_bijection.all()),
        "stuckPreservation": bool(pd.read_parquet(OUTPUT / "phase_run_summaries.parquet").stuck_positions_preserved.all()),
        "fluxConservation": bool(
            pd.read_parquet(OUTPUT / "phase_run_summaries.parquet").signed_flux_conservation_error.le(1e-12).all()
        ),
    }
    validation["allPassed"] = all(bool(value) for key, value in validation.items() if key not in {"schema", "researchStepId"})
    write_json(OUTPUT / "validation_summary.json", validation)
    if not validation["allPassed"]:
        raise AssertionError("S13 final validation failed")
    write_json(
        OUTPUT / "environment.json",
        {
            "schema": "e04.s13.environment.v1",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "numbaWorkers": 8,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
            },
            "gpuUsed": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("validate-kernel")
    for name in ("run-base", "run-sequential", "run-long"):
        child = subparsers.add_parser(name)
        child.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("select-boundary")
    subparsers.add_parser("select-long")
    subparsers.add_parser("analyze")
    subparsers.add_parser("finalize")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze()
    elif args.command == "validate-kernel":
        validate_kernel()
    elif args.command == "run-base":
        run_stage("base", base_tasks(), PRIMARY_BUDGET, args.workers)
    elif args.command == "select-boundary":
        select_boundary()
    elif args.command == "run-sequential":
        run_stage("sequential", sequential_tasks(), PRIMARY_BUDGET, args.workers)
    elif args.command == "select-long":
        select_long()
    elif args.command == "run-long":
        run_stage("long", long_tasks(), LONG_BUDGET, args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
