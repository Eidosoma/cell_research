#!/usr/bin/env python3
"""Enumerate the frozen E03 S04 finite structural state families."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from reference_simulator.model import (
    Architecture,
    Direction,
    FaultMode,
    Policy,
    canonical_json_bytes,
)
from reference_simulator.policies import (
    cell_view_proposal,
    has_admissible_change,
    traditional_proposal,
)
from reference_simulator.transition_primitives import commit_proposal, validate_proposal
from src.detours.state_space import (
    FAMILY_SCHEMA,
    STATE_DOMAIN,
    STATE_SCHEMA,
    FamilySpec,
    StructuralState,
    all_policy_assignments,
    all_policy_state_count,
    cell_id,
    family_lookup_key,
    fault_maps,
    nonzero_fault_maps,
    pure_policy,
)


CONTRACT = Path(__file__).resolve().parents[1] / "analysis/s04_enumeration_contract.json"
OUTPUT = Path("/artifacts/research_steps/S04")
SCRATCH = Path("/cache/e03_s04_enumeration")
REPOSITORY = Path(__file__).resolve().parents[1]
E01 = Path("/previous-artifacts/E01")
BATCH_ROWS = 65_536
STARTING_COMMIT = "ed583cc98d335b4156f1019c638bc29700a5975d"


STATE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("family_ordinal", pa.int32(), nullable=False),
        pa.field("state_digest", pa.binary(32), nullable=False),
        pa.field("occupancy_rank", pa.uint32(), nullable=False),
        pa.field("selection_cursor_code", pa.uint64(), nullable=False),
        pa.field("collision_ordinal", pa.uint16(), nullable=False),
    ],
    metadata={
        b"schemaVersion": STATE_SCHEMA.encode(),
        b"researchStepId": b"S04",
        b"familyJoin": b"family_ordinal -> state_family_inventory.parquet",
        b"authoritativeKey": b"family_ordinal,occupancy_rank,selection_cursor_code",
    },
)


@dataclass(frozen=True, slots=True)
class PlannedFamily:
    ordinal: int
    tier: str
    family: FamilySpec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[dict[str, Any]] | pd.DataFrame) -> None:
    table = (
        pa.Table.from_pandas(rows, preserve_index=False)
        if isinstance(rows, pd.DataFrame)
        else pa.Table.from_pylist(list(rows))
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def _family_sort_key(item: tuple[str, FamilySpec]) -> tuple[object, ...]:
    tier, family = item
    return (
        family.n,
        family.architecture.value,
        family.direction.value,
        family.policy_code,
        family.fault_code,
        tier,
    )


def build_family_plan() -> list[PlannedFamily]:
    """Expand the exact materialization tiers frozen in the S04 contract."""

    candidates: dict[tuple[object, ...], tuple[str, FamilySpec]] = {}

    def add(tier: str, family: FamilySpec) -> None:
        key = family_lookup_key(family)
        prior = candidates.get(key)
        if prior is not None and prior[1] != family:
            raise AssertionError("family lookup key collision")
        candidates.setdefault(key, (tier, family))

    # Every identity-owned policy assignment and all declared fault maps at n=4.
    for direction, policies, faults in itertools.product(
        (Direction.ASCENDING, Direction.DESCENDING),
        all_policy_assignments(4),
        fault_maps(4),
    ):
        add(
            "cell_all_policies_fault_complete_n4",
            FamilySpec(4, Architecture.CELL_VIEW, direction, policies, faults),
        )

    # Every policy/memory assignment at n=5 in the no-fault system.
    for direction, policies in itertools.product(
        (Direction.ASCENDING, Direction.DESCENDING), all_policy_assignments(5)
    ):
        add(
            "cell_all_policies_no_fault_n5",
            FamilySpec(
                5,
                Architecture.CELL_VIEW,
                direction,
                policies,
                (FaultMode.NORMAL,) * 5,
            ),
        )

    # Fault-complete pure memoryless cell-view policies through n=7.
    for n in (5, 6, 7):
        maps: Iterable[tuple[FaultMode, ...]] = (
            nonzero_fault_maps(n) if n == 5 else fault_maps(n)
        )
        fault_values = tuple(maps)
        for direction, policy, faults in itertools.product(
            (Direction.ASCENDING, Direction.DESCENDING),
            (Policy.BUBBLE, Policy.INSERTION),
            fault_values,
        ):
            add(
                "cell_memoryless_fault_complete_n5_n7",
                FamilySpec(
                    n,
                    Architecture.CELL_VIEW,
                    direction,
                    pure_policy(policy, n),
                    faults,
                ),
            )

    # No-fault memoryless cell-view reach through n=9.
    for n, direction, policy in itertools.product(
        (8, 9),
        (Direction.ASCENDING, Direction.DESCENDING),
        (Policy.BUBBLE, Policy.INSERTION),
    ):
        add(
            "cell_memoryless_no_fault_n8_n9",
            FamilySpec(
                n,
                Architecture.CELL_VIEW,
                direction,
                pure_policy(policy, n),
                (FaultMode.NORMAL,) * n,
            ),
        )

    # Conventional controllers are distinct graph families but have no cursor memory.
    for n in (4, 5, 6, 7):
        maps = tuple(fault_maps(n))
        for direction, policy, faults in itertools.product(
            (Direction.ASCENDING, Direction.DESCENDING),
            (Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION),
            maps,
        ):
            add(
                "traditional_fault_complete_n4_n7",
                FamilySpec(
                    n,
                    Architecture.TRADITIONAL,
                    direction,
                    pure_policy(policy, n),
                    faults,
                ),
            )
    for n, direction, policy in itertools.product(
        (8, 9),
        (Direction.ASCENDING, Direction.DESCENDING),
        (Policy.BUBBLE, Policy.INSERTION, Policy.SELECTION),
    ):
        add(
            "traditional_no_fault_n8_n9",
            FamilySpec(
                n,
                Architecture.TRADITIONAL,
                direction,
                pure_policy(policy, n),
                (FaultMode.NORMAL,) * n,
            ),
        )

    ordered = sorted(candidates.values(), key=_family_sort_key)
    return [
        PlannedFamily(ordinal=index, tier=tier, family=family)
        for index, (tier, family) in enumerate(ordered)
    ]


def family_rows(plan: Sequence[PlannedFamily]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan:
        family = item.family
        rows.append(
            {
                "schema_version": FAMILY_SCHEMA,
                "research_step_id": "S04",
                "family_ordinal": item.ordinal,
                "family_id": family.family_id,
                "family_digest": family.family_digest,
                "materialization_tier": item.tier,
                "n": family.n,
                "architecture": family.architecture.value,
                "direction": family.direction.value,
                "source_scheduler": (
                    "traditional_controller"
                    if family.architecture == Architecture.TRADITIONAL
                    else "serial_counter_addressed"
                ),
                "scheduler_projection": family.scheduler_projection,
                "policy_code": family.policy_code,
                "policy_profile": family.policy_profile,
                "policies_by_identity": [value.value for value in family.policies],
                "selection_owner_ids": [cell_id(value) for value in family.selection_owners],
                "selection_owner_count": family.selection_owner_count,
                "fault_code": family.fault_code,
                "fault_mode": family.fault_mode,
                "fault_count": family.fault_count,
                "faults_by_identity": [value.value for value in family.faults],
                "fault_identity_ids": [
                    cell_id(index)
                    for index, value in enumerate(family.faults)
                    if value != FaultMode.NORMAL
                ],
                "occupancy_state_count": family.occupancy_state_count,
                "cursor_state_count": family.cursor_state_count,
                "state_count": family.state_count,
                "identity_profile": "unique_value_identity_c0_to_c(n-1)",
                "value_profile": "unique_integer_0_to_n_minus_1",
                "direction_scope": "homogeneous_native_goal",
                "reachability_scope": "syntactically_valid_not_reachability_filtered",
                "canonical_family_json": canonical_json_bytes(family.canonical_dict()).decode(),
            }
        )
    return rows


def _state_table(
    family_ordinal: int,
    family_digest: bytes,
    occupancy_state_count: int,
    start_index: int,
    stop_index: int,
) -> pa.Table:
    linear = np.arange(start_index, stop_index, dtype=np.uint64)
    ranks = (linear % occupancy_state_count).astype(np.uint32)
    cursor_codes = linear // occupancy_state_count
    base = hashlib.sha256(STATE_DOMAIN + family_digest)
    digests: list[bytes] = []
    pack = struct.Struct(">QQ").pack
    for rank, cursor_code in zip(ranks, cursor_codes):
        digest = base.copy()
        digest.update(pack(int(rank), int(cursor_code)))
        digests.append(digest.digest())
    count = stop_index - start_index
    arrays = [
        pa.array(np.full(count, family_ordinal, dtype=np.int32), type=pa.int32()),
        pa.array(digests, type=pa.binary(32)),
        pa.array(ranks, type=pa.uint32()),
        pa.array(cursor_codes, type=pa.uint64()),
        pa.array(np.zeros(count, dtype=np.uint16), type=pa.uint16()),
    ]
    return pa.Table.from_arrays(arrays, schema=STATE_ARROW_SCHEMA)


def _worker_write_part(args: tuple[int, list[PlannedFamily], str]) -> dict[str, Any]:
    worker_index, families, scratch_string = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    scratch = Path(scratch_string)
    path = scratch / f"state_inventory.worker-{worker_index:02d}.parquet"
    started = time.perf_counter()
    rows = 0
    family_count = 0
    with pq.ParquetWriter(
        path,
        STATE_ARROW_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=["family_ordinal"],
        write_statistics=True,
        version="2.6",
    ) as writer:
        for item in families:
            family = item.family
            for start in range(0, family.state_count, BATCH_ROWS):
                stop = min(start + BATCH_ROWS, family.state_count)
                writer.write_table(
                    _state_table(
                        item.ordinal,
                        family.family_digest,
                        family.occupancy_state_count,
                        start,
                        stop,
                    ),
                    row_group_size=BATCH_ROWS,
                )
                rows += stop - start
            family_count += 1
    return {
        "worker": worker_index,
        "path": str(path),
        "families": family_count,
        "rows": rows,
        "bytes": path.stat().st_size,
        "elapsedSeconds": time.perf_counter() - started,
    }


def merge_parts(parts: Sequence[dict[str, Any]], output: Path) -> None:
    with pq.ParquetWriter(
        output,
        STATE_ARROW_SCHEMA,
        compression="zstd",
        compression_level=6,
        use_dictionary=["family_ordinal"],
        write_statistics=True,
        version="2.6",
    ) as writer:
        for part in sorted(parts, key=lambda value: value["worker"]):
            parquet = pq.ParquetFile(part["path"])
            for index in range(parquet.metadata.num_row_groups):
                writer.write_table(parquet.read_row_group(index), row_group_size=BATCH_ROWS)


def _scope_projection(scope: str, n: int) -> tuple[int, int, int]:
    factorial = math.factorial(n)
    maps = len(tuple(fault_maps(n)))
    if scope == "cell_all_policy_fault_complete":
        aggregate = 2 * maps * all_policy_state_count(n)
        largest = factorial * (n + 1) ** n
        opportunities = 2 * n
    elif scope == "cell_all_policy_no_fault":
        aggregate = 2 * all_policy_state_count(n)
        largest = factorial * (n + 1) ** n
        opportunities = 2 * n
    elif scope == "cell_selection_no_fault":
        largest = factorial * (n + 1) ** n
        aggregate = 2 * largest
        opportunities = n
    elif scope == "cell_memoryless_fault_complete":
        largest = factorial
        aggregate = 2 * 2 * maps * factorial
        opportunities = 2 * n
    elif scope == "cell_memoryless_no_fault":
        largest = factorial
        aggregate = 2 * 2 * factorial
        opportunities = 2 * n
    elif scope == "traditional_fault_complete":
        largest = factorial
        aggregate = 2 * 3 * maps * factorial
        opportunities = 1
    elif scope == "traditional_no_fault":
        largest = factorial
        aggregate = 2 * 3 * factorial
        opportunities = 1
    else:
        raise AssertionError(scope)
    return aggregate, largest, aggregate * opportunities


def tractability_rows() -> list[dict[str, Any]]:
    materialized_max = {
        "cell_all_policy_fault_complete": 4,
        "cell_all_policy_no_fault": 5,
        "cell_selection_no_fault": 5,
        "cell_memoryless_fault_complete": 7,
        "cell_memoryless_no_fault": 9,
        "traditional_fault_complete": 7,
        "traditional_no_fault": 9,
    }
    rows: list[dict[str, Any]] = []
    for scope, maximum in materialized_max.items():
        for n in range(4, min(maximum + 2, 11)):
            aggregate, largest, edges = _scope_projection(scope, n)
            estimated_bytes = 48 * aggregate
            gates: list[str] = []
            if aggregate > 25_000_000:
                gates.append("scope_rows_gt_25m")
            if largest > 5_000_000:
                gates.append("single_family_nodes_gt_5m")
            if estimated_bytes > 1_610_612_736:
                gates.append("estimated_parquet_gt_1.5GiB")
            if edges > 50_000_000:
                gates.append("downstream_edge_warning_gt_50m")
            rows.append(
                {
                    "scope": scope,
                    "n": n,
                    "status": "materialized" if n <= maximum else "first_rejected_n",
                    "projected_state_rows": aggregate,
                    "largest_single_family_nodes": largest,
                    "projected_legal_opportunity_slots_upper": edges,
                    "estimated_uncompressed_core_bytes": estimated_bytes,
                    "triggered_gates": gates,
                    "stopping_note": (
                        "within frozen tier"
                        if n <= maximum
                        else "first n outside the frozen tier; gate list and cumulative 25m step budget justify stop"
                    ),
                }
            )
    return rows


def sample_states(plan: Sequence[PlannedFamily], maximum_families: int = 384) -> list[dict[str, Any]]:
    groups: dict[tuple[object, ...], PlannedFamily] = {}
    for item in plan:
        family = item.family
        key = (
            item.tier,
            family.n,
            family.architecture.value,
            family.direction.value,
            family.policy_profile,
            family.fault_mode,
            family.fault_count,
        )
        groups.setdefault(key, item)
    selected = list(groups.values())
    if len(selected) > maximum_families:
        indices = np.linspace(0, len(selected) - 1, maximum_families, dtype=int)
        selected = [selected[index] for index in sorted(set(indices.tolist()))]
    rows: list[dict[str, Any]] = []
    for item in selected:
        family = item.family
        coordinates = {
            (0, 0),
            ((family.occupancy_state_count - 1) // 2, (family.cursor_state_count - 1) // 2),
            (family.occupancy_state_count - 1, family.cursor_state_count - 1),
        }
        for rank, cursor in sorted(coordinates):
            state = StructuralState(family, rank, cursor)
            rows.append(
                {
                    "family_ordinal": item.ordinal,
                    "family_id": family.family_id,
                    "state_id": state.state_id,
                    "n": family.n,
                    "architecture": family.architecture.value,
                    "direction": family.direction.value,
                    "policy_profile": family.policy_profile,
                    "fault_mode": family.fault_mode,
                    "fault_count": family.fault_count,
                    "occupancy_rank": rank,
                    "occupancy_ids": [cell_id(value) for value in state.occupancy],
                    "occupancy_values": list(state.occupancy),
                    "selection_cursor_code": cursor,
                    "selection_cursors_json": json.dumps(
                        {cell_id(key): value for key, value in state.selection_cursors.items()},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
    return rows


def symmetry_rows(plan: Sequence[PlannedFamily]) -> list[dict[str, Any]]:
    lookup = {family_lookup_key(item.family): item for item in plan}
    rows: list[dict[str, Any]] = []
    for item in plan:
        family = item.family
        dual_item = lookup.get(family_lookup_key(family.dual()))
        coordinates = {
            (0, 0),
            ((family.occupancy_state_count - 1) // 2, (family.cursor_state_count - 1) // 2),
            (family.occupancy_state_count - 1, family.cursor_state_count - 1),
        }
        for rank, cursor in sorted(coordinates):
            state = StructuralState(family, rank, cursor)
            dual = state.dual()
            passed = (
                dual_item is not None
                and dual.family == dual_item.family
                and dual.dual() == state
                and dual.family.state_count == family.state_count
            )
            rows.append(
                {
                    "family_ordinal": item.ordinal,
                    "family_id": family.family_id,
                    "dual_family_ordinal": dual_item.ordinal if dual_item else -1,
                    "dual_family_id": dual.family.family_id,
                    "occupancy_rank": rank,
                    "selection_cursor_code": cursor,
                    "dual_state_id": dual.state_id,
                    "involution_passed": passed,
                }
            )
    return rows


def simulator_consistency_rows(
    plan: Sequence[PlannedFamily], maximum_families: int = 768
) -> list[dict[str, Any]]:
    groups: dict[tuple[object, ...], PlannedFamily] = {}
    for item in plan:
        family = item.family
        key = (
            item.tier,
            family.n,
            family.architecture.value,
            family.direction.value,
            family.policy_profile,
            family.fault_mode,
            family.fault_count,
        )
        groups.setdefault(key, item)
    selected = list(groups.values())
    if len(selected) > maximum_families:
        indices = np.linspace(0, len(selected) - 1, maximum_families, dtype=int)
        selected = [selected[index] for index in sorted(set(indices.tolist()))]
    rows: list[dict[str, Any]] = []
    for item in selected:
        family = item.family
        rank = int.from_bytes(family.family_digest[:4], "big") % family.occupancy_state_count
        cursor = int.from_bytes(family.family_digest[4:8], "big") % family.cursor_state_count
        structural = StructuralState(family, rank, cursor)
        scenario = family.scenario()
        baseline = structural.to_run_state()
        terminal_before = (
            "complete"
            if all(
                left <= right
                for left, right in zip(structural.occupancy, structural.occupancy[1:])
            )
            and family.direction == Direction.ASCENDING
            else "not_directly_classified"
        )
        if family.architecture == Architecture.TRADITIONAL:
            opportunities: list[tuple[str, str | None]] = [("__controller__", None)]
        else:
            opportunities = []
            for actor_id in baseline.occupancy:
                policy = scenario.cell_map[actor_id].policy
                sides: tuple[str | None, ...] = (
                    ("left", "right") if policy == Policy.BUBBLE else (None,)
                )
                opportunities.extend((actor_id, side) for side in sides)
        for actor_id, side in opportunities:
            state = structural.to_run_state()
            proposal = (
                traditional_proposal(scenario, state)
                if family.architecture == Architecture.TRADITIONAL
                else cell_view_proposal(scenario, state, actor_id, side=side)
            )
            validation = validate_proposal(scenario, state, proposal)
            changed = False
            successor = structural
            if validation.eligible_for_commit:
                snapshot = state.clone()
                changed = commit_proposal(state, snapshot, proposal, "accepted")
                successor = StructuralState.from_run_state(family, state)
            closure = (
                0 <= successor.occupancy_rank < family.occupancy_state_count
                and 0 <= successor.selection_cursor_code < family.cursor_state_count
            )
            rows.append(
                {
                    "family_ordinal": item.ordinal,
                    "family_id": family.family_id,
                    "architecture": family.architecture.value,
                    "n": family.n,
                    "direction": family.direction.value,
                    "policy_profile": family.policy_profile,
                    "fault_mode": family.fault_mode,
                    "fault_count": family.fault_count,
                    "predecessor_state_id": structural.state_id,
                    "actor_id": actor_id,
                    "side": side or "not_applicable",
                    "proposal_kind": proposal.kind.value,
                    "proposal_reason": proposal.reason,
                    "validation_decision": validation.decision,
                    "changed": changed,
                    "successor_state_id": successor.state_id,
                    "successor_in_family_domain": closure,
                    "has_admissible_change": has_admissible_change(scenario, baseline),
                    "terminal_fixture_label": terminal_before,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")
    plan = build_family_plan()
    total_rows = sum(item.family.state_count for item in plan)
    if total_rows > 25_000_000:
        raise AssertionError(f"frozen S04 total row gate exceeded: {total_rows}")
    tier_counts = Counter(item.tier for item in plan)
    tier_rows: Counter[str] = Counter()
    for item in plan:
        tier_rows[item.tier] += item.family.state_count
    preflight = {
        "researchStepId": "S04",
        "familyCount": len(plan),
        "stateRows": total_rows,
        "tierFamilyCounts": dict(sorted(tier_counts.items())),
        "tierStateRows": dict(sorted(tier_rows.items())),
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    if args.scratch.exists():
        shutil.rmtree(args.scratch)
    args.scratch.mkdir(parents=True)
    input_paths = [
        CONTRACT,
        REPOSITORY / "reference_simulator/model.py",
        REPOSITORY / "reference_simulator/policies.py",
        REPOSITORY / "reference_simulator/scheduler.py",
        REPOSITORY / "reference_simulator/engine.py",
        REPOSITORY / "reference_simulator/transition_primitives.py",
        E01 / "research_steps/S03/state_diagrams.md",
        E01 / "research_steps/S05/semantic_decisions.json",
        E01 / "release/reference_simulator/release_manifest.json",
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
    ]
    pre_hashes = {str(path): sha256_file(path) for path in input_paths}
    started = time.perf_counter()
    family_output = args.output / "state_family_inventory.parquet"
    write_parquet(family_output, family_rows(plan))
    worker_families = [plan[index:: args.workers] for index in range(args.workers)]
    worker_args = [
        (index, worker_families[index], str(args.scratch))
        for index in range(args.workers)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        parts = list(executor.map(_worker_write_part, worker_args))
    if sum(part["rows"] for part in parts) != total_rows:
        raise AssertionError("worker row counts do not match the frozen plan")
    merge_started = time.perf_counter()
    state_output = args.output / "state_inventory.parquet"
    merge_parts(parts, state_output)
    merge_seconds = time.perf_counter() - merge_started

    samples = sample_states(plan)
    symmetry = symmetry_rows(plan)
    consistency = simulator_consistency_rows(plan)
    projections = tractability_rows()
    write_parquet(args.output / "state_samples.parquet", samples)
    write_parquet(args.output / "symmetry_validation.parquet", symmetry)
    write_parquet(args.output / "simulator_consistency.parquet", consistency)
    write_parquet(args.output / "tractability_projection.parquet", projections)

    post_hashes = {str(path): sha256_file(path) for path in input_paths}
    immutable = pre_hashes == post_hashes
    if not immutable:
        raise AssertionError("an S04 input changed during enumeration")
    elapsed = time.perf_counter() - started
    by_n: Counter[int] = Counter()
    by_architecture: Counter[str] = Counter()
    for item in plan:
        by_n[item.family.n] += item.family.state_count
        by_architecture[item.family.architecture.value] += item.family.state_count
    summary = {
        "schemaVersion": "e03.s04.enumeration_summary.v1",
        "researchStepId": "S04",
        "success": True,
        "familyCount": len(plan),
        "stateRows": total_rows,
        "stateRowsByN": {str(key): value for key, value in sorted(by_n.items())},
        "stateRowsByArchitecture": dict(sorted(by_architecture.items())),
        "tierFamilyCounts": dict(sorted(tier_counts.items())),
        "tierStateRows": dict(sorted(tier_rows.items())),
        "maximumSingleFamilyNodes": max(item.family.state_count for item in plan),
        "maximumN": max(item.family.n for item in plan),
        "allPolicyFaultCompleteMaximumN": 4,
        "allPolicyNoFaultMaximumN": 5,
        "faultCompleteMemorylessMaximumN": 7,
        "noFaultMemorylessMaximumN": 9,
        "workerCount": args.workers,
        "workerParts": parts,
        "mergeSeconds": merge_seconds,
        "elapsedSeconds": elapsed,
        "stateInventoryBytes": state_output.stat().st_size,
        "stateInventoryRowGroups": pq.ParquetFile(state_output).metadata.num_row_groups,
        "stateSampleRows": len(samples),
        "symmetryRows": len(symmetry),
        "simulatorConsistencyRows": len(consistency),
        "inputsUnchanged": immutable,
        "startingCommit": STARTING_COMMIT,
    }
    write_json(args.output / "enumeration_summary.json", summary)
    write_json(
        args.output / "input_immutability.json",
        {
            "schemaVersion": "e03.s04.input_immutability.v1",
            "researchStepId": "S04",
            "success": immutable,
            "preRunSha256": pre_hashes,
            "postRunSha256": post_hashes,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
