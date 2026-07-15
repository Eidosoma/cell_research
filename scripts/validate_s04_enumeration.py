#!/usr/bin/env python3
"""Independent integrity checks for the E03 S04 exact state inventory."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.detours.state_space import STATE_DOMAIN, FamilySpec, StructuralState


OUTPUT = Path("/artifacts/research_steps/S04")
HASH_SCRATCH = Path("/cache/e03_s04_validation_hashes.bin")
EXPECTED_FAMILIES = 7_984
EXPECTED_STATES = 22_301_808


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def family_from_json(value: str) -> FamilySpec:
    return FamilySpec.from_canonical_dict(json.loads(value))


def main() -> None:
    started = time.perf_counter()
    family_path = OUTPUT / "state_family_inventory.parquet"
    state_path = OUTPUT / "state_inventory.parquet"
    summary_path = OUTPUT / "enumeration_summary.json"
    required = [
        family_path,
        state_path,
        summary_path,
        OUTPUT / "state_samples.parquet",
        OUTPUT / "symmetry_validation.parquet",
        OUTPUT / "simulator_consistency.parquet",
        OUTPUT / "tractability_projection.parquet",
        OUTPUT / "input_immutability.json",
        OUTPUT / "canonical_state_encoder.md",
        OUTPUT / "tractability_report.md",
        OUTPUT / "research_step_full_results.md",
        OUTPUT / "environment.json",
        OUTPUT / "commands.log",
        OUTPUT / "provenance_manifest.json",
        OUTPUT / "artifact_manifest.json",
    ]
    for path in required:
        require(path.is_file() and path.stat().st_size > 0, f"missing output: {path}")

    family_table = pq.read_table(family_path)
    family_rows = family_table.to_pylist()
    require(len(family_rows) == EXPECTED_FAMILIES, "unexpected family count")
    require(
        sum(column.null_count for column in family_table.columns) == 0,
        "family inventory contains null metadata",
    )
    ordinals = np.asarray(family_table["family_ordinal"], dtype=np.int64)
    require(np.array_equal(ordinals, np.arange(EXPECTED_FAMILIES)), "family ordinals")

    families: list[FamilySpec] = []
    family_digests: list[bytes] = []
    expected_counts = np.empty(EXPECTED_FAMILIES, dtype=np.int64)
    occupancy_counts = np.empty(EXPECTED_FAMILIES, dtype=np.int64)
    cursor_counts = np.empty(EXPECTED_FAMILIES, dtype=np.int64)
    for index, row in enumerate(family_rows):
        family = family_from_json(row["canonical_family_json"])
        families.append(family)
        family_digests.append(family.family_digest)
        expected_counts[index] = family.state_count
        occupancy_counts[index] = family.occupancy_state_count
        cursor_counts[index] = family.cursor_state_count
        require(row["family_id"] == family.family_id, f"family id mismatch {index}")
        require(row["family_digest"] == family.family_digest, f"family digest mismatch {index}")
        require(row["n"] in range(4, 10), f"n outside materialized domain {index}")
        require(row["occupancy_state_count"] == math.factorial(row["n"]), f"factorial {index}")
        require(row["cursor_state_count"] == family.cursor_state_count, f"cursor count {index}")
        require(row["state_count"] == family.state_count, f"state count {index}")
        require(len(row["policies_by_identity"]) == row["n"], f"policies metadata {index}")
        require(len(row["faults_by_identity"]) == row["n"], f"fault metadata {index}")
        require(row["scheduler_projection"] == family.scheduler_projection, f"scheduler metadata {index}")

    require(int(expected_counts.sum()) == EXPECTED_STATES, "family state-count sum")
    require(len(set(family_digests)) == EXPECTED_FAMILIES, "family digest collision")

    parquet = pq.ParquetFile(state_path)
    require(parquet.metadata.num_rows == EXPECTED_STATES, "state Parquet row count")
    schema_metadata = parquet.schema_arrow.metadata or {}
    require(schema_metadata.get(b"schemaVersion") == b"e03.s04.structural_state.v1", "state schema")
    require(
        schema_metadata.get(b"authoritativeKey")
        == b"family_ordinal,occupancy_rank,selection_cursor_code",
        "authoritative key metadata",
    )

    seen_counts = np.zeros(EXPECTED_FAMILIES, dtype=np.int64)
    expected_next = np.zeros(EXPECTED_FAMILIES, dtype=np.int64)
    sample_hash_checks = 0
    hash_map = np.memmap(HASH_SCRATCH, dtype="V32", mode="w+", shape=(EXPECTED_STATES,))
    hash_offset = 0
    pack = struct.Struct(">QQ").pack
    for row_group_index in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(
            row_group_index,
            columns=[
                "family_ordinal",
                "state_digest",
                "occupancy_rank",
                "selection_cursor_code",
                "collision_ordinal",
            ],
        )
        family_ordinal = np.asarray(table["family_ordinal"], dtype=np.int64)
        ranks = np.asarray(table["occupancy_rank"], dtype=np.int64)
        cursors = np.asarray(table["selection_cursor_code"], dtype=np.int64)
        collisions = np.asarray(table["collision_ordinal"], dtype=np.int64)
        require(np.all((0 <= family_ordinal) & (family_ordinal < EXPECTED_FAMILIES)), "family range")
        require(np.all(collisions == 0), "nonzero collision ordinal before collision audit")

        for ordinal in np.unique(family_ordinal):
            mask = family_ordinal == ordinal
            linear = cursors[mask] * occupancy_counts[ordinal] + ranks[mask]
            start = expected_next[ordinal]
            require(
                np.array_equal(linear, np.arange(start, start + len(linear))),
                f"authoritative tuple coverage/order family {ordinal}",
            )
            expected_next[ordinal] += len(linear)
            seen_counts[ordinal] += len(linear)

        digest_values = table["state_digest"].to_pylist()
        digest_block = np.frombuffer(b"".join(digest_values), dtype="V32")
        hash_map[hash_offset : hash_offset + len(digest_block)] = digest_block
        hash_offset += len(digest_block)

        for sample_index in sorted({0, len(table) // 2, len(table) - 1}):
            ordinal = int(family_ordinal[sample_index])
            digest = hashlib.sha256(STATE_DOMAIN + family_digests[ordinal])
            digest.update(pack(int(ranks[sample_index]), int(cursors[sample_index])))
            require(digest.digest() == digest_values[sample_index], "sample state hash mismatch")
            sample_hash_checks += 1

    require(hash_offset == EXPECTED_STATES, "hash scratch row count")
    require(np.array_equal(seen_counts, expected_counts), "per-family tuple count")
    require(np.array_equal(expected_next, expected_counts), "complete tuple envelope")
    hash_map.flush()
    hash_map.sort(kind="quicksort")
    duplicate_hash_count = int(np.count_nonzero(hash_map[1:] == hash_map[:-1]))
    del hash_map
    require(duplicate_hash_count == 0, "SHA-256 state digest collision")

    samples = pq.read_table(OUTPUT / "state_samples.parquet").to_pylist()
    for row in samples:
        family = families[row["family_ordinal"]]
        state = StructuralState(family, row["occupancy_rank"], row["selection_cursor_code"])
        require(state.state_id == row["state_id"], "sample encoder roundtrip")

    symmetry = pq.read_table(OUTPUT / "symmetry_validation.parquet")
    require(symmetry.num_rows == 3 * EXPECTED_FAMILIES, "symmetry fixture count")
    require(bool(pc.all(symmetry["involution_passed"]).as_py()), "direction symmetry failure")

    simulator = pq.read_table(OUTPUT / "simulator_consistency.parquet")
    require(simulator.num_rows > 0, "empty simulator semantic fixtures")
    require(bool(pc.all(simulator["successor_in_family_domain"]).as_py()), "simulator closure")
    simulator_rows = simulator.to_pylist()
    covered_architectures = {row["architecture"] for row in simulator_rows}
    covered_directions = {row["direction"] for row in simulator_rows}
    covered_fault_modes = {row["fault_mode"] for row in simulator_rows}
    covered_policy_profiles = {row["policy_profile"] for row in simulator_rows}
    require(covered_architectures == {"cell_view", "traditional"}, "architecture coverage")
    require(covered_directions == {"ascending", "descending"}, "direction coverage")
    require(covered_fault_modes == {"none", "passive", "stuck"}, "fault-mode coverage")
    require(
        any(not profile.startswith("pure_") for profile in covered_policy_profiles),
        "mixed policy fixtures absent",
    )

    projections = pq.read_table(OUTPUT / "tractability_projection.parquet").to_pylist()
    scopes = {row["scope"] for row in projections}
    rejected = [row for row in projections if row["status"] == "first_rejected_n"]
    require(len(scopes) == 7 and len(rejected) == 7, "tractability scope/rejection coverage")
    require(all(row["n"] >= 5 for row in rejected), "invalid stopping n")

    summary = json.loads(summary_path.read_text())
    immutability = json.loads((OUTPUT / "input_immutability.json").read_text())
    require(summary["success"] and summary["inputsUnchanged"], "enumeration summary failure")
    require(immutability["success"], "input mutation detected")

    collision_audit = {
        "schemaVersion": "e03.s04.hash_collision_audit.v1",
        "researchStepId": "S04",
        "success": True,
        "digestAlgorithm": "sha256",
        "stateDigestsChecked": EXPECTED_STATES,
        "duplicateDigestCount": duplicate_hash_count,
        "familyDigestsChecked": EXPECTED_FAMILIES,
        "duplicateFamilyDigestCount": 0,
        "collisionHandling": (
            "The authoritative key is (family_ordinal, occupancy_rank, selection_cursor_code). "
            "collision_ordinal is retained in the schema; a digest collision would fail validation "
            "and require disambiguation by the authoritative tuple rather than hash-only identity."
        ),
        "method": "exact in-place sort of all 32-byte state digests plus exact family-digest set audit",
    }
    write_json(OUTPUT / "hash_collision_audit.json", collision_audit)

    checks = {
        "required_outputs_present": True,
        "complete_family_metadata": True,
        "expected_factorial_counts": True,
        "exact_authoritative_tuple_coverage": True,
        "family_hash_uniqueness": True,
        "state_hash_uniqueness": True,
        "sample_hash_recomputation": True,
        "sample_encoder_roundtrip": True,
        "direction_duality_involution": True,
        "simulator_successor_closure": True,
        "simulator_stratum_coverage": True,
        "tractability_stopping_rows": True,
        "inputs_unchanged": True,
    }
    results = {
        "schemaVersion": "e03.s04.validation.v1",
        "researchStepId": "S04",
        "success": True,
        "checks": checks,
        "familyCount": EXPECTED_FAMILIES,
        "stateCount": EXPECTED_STATES,
        "stateRowGroups": parquet.metadata.num_row_groups,
        "sampleHashRecomputations": sample_hash_checks,
        "sampleRoundtrips": len(samples),
        "symmetryFixtures": symmetry.num_rows,
        "simulatorFixtures": simulator.num_rows,
        "simulatorCoverage": {
            "architectures": sorted(covered_architectures),
            "directions": sorted(covered_directions),
            "faultModes": sorted(covered_fault_modes),
            "policyProfiles": sorted(covered_policy_profiles),
        },
        "duplicateStateDigestCount": duplicate_hash_count,
        "elapsedSeconds": time.perf_counter() - started,
        "hashScratchPath": str(HASH_SCRATCH),
    }
    write_json(OUTPUT / "validation_results.json", results)
    (OUTPUT / "validation.log").write_text(
        "S04 VALIDATION PASS\n"
        + "\n".join(f"{key}=PASS" for key in checks)
        + f"\nfamily_count={EXPECTED_FAMILIES}\nstate_count={EXPECTED_STATES}\n"
        + f"duplicate_state_digest_count={duplicate_hash_count}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
