#!/usr/bin/env python3
"""Build and validate the frozen S08 paired-comparison design."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.faults import (
    ActionFailureProfile,
    FaultExecutionContract,
    MobilityProfile,
    exact_replay_fault,
    run_faulted_architecture,
)
from causal_simulator.pairing import (
    DIRECTIONS,
    FAULT_PROFILES,
    FORBIDDEN_PLACEMENTS,
    MASTER_SEED,
    PAIRING_INTERFACE_VERSION,
    PAIRING_PRESPECIFICATION_SHA256,
    PAIRING_SCHEMA_VERSION,
    POLICIES,
    STREAM_SPECIFICATION_SHA256,
    algotype_assignment_id,
    assign_fault_profiles,
    content_id,
    contrast_catalog,
    coupling_matrix_rows,
    executable_arms,
    pairing_block_content,
    pairing_block_id,
    scenario_variant_id,
    sha256_file,
)
from causal_simulator.placements import (
    AnalysisRole,
    OutcomeAccess,
    PlacementClass,
    PlacementContext,
    generate_structural_placement,
    materialize_fault_scenario,
)
from causal_simulator.scenario_extensions import scenario_from_record
from causal_simulator.schedulers import (
    FrozenSchedulerController,
    SchedulerExecutionContract,
    SchedulerFamily,
)
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.rng import u64


S07_BANK = Path("/artifacts/research_steps/S07/scenario_extension.parquet")
S07_BANK_SHA256 = "1ee7075f0367b989acc31f66dca4dadd5321f5962dd6a318b1cde0c1a7bbf339"
S01_ESTIMANDS = Path("/artifacts/research_steps/S01/estimand_registry.yaml")
S06_STRUCTURAL_LOCK = Path("/artifacts/research_steps/S06/structural_lock.json")
S06_BANK = Path("/artifacts/research_steps/S06/fault_placement_bank.parquet")
PAIRING_SPEC = REPOSITORY / "design/s08/pairing_prespecification.json"
STREAM_SPEC = REPOSITORY / "design/s08/semantic_random_stream_specification.json"
E01_SEED_SPEC = Path("/previous-artifacts/E01/research_steps/S08/seed_specification.json")
CORE_FILES = (
    "pairing_prespecification.json",
    "semantic_random_stream_specification.json",
    "contrast_arm_catalog.json",
    "pairing_manifest.parquet",
    "fault_map_catalog.parquet",
    "coupling_matrix.csv",
    "treatment_arm_marginals.parquet",
    "fault_profile_distribution.csv",
    "pair_completeness.json",
    "scenario_id_validation.json",
    "split_enforcement_validation.json",
    "stream_name_isolation_validation.json",
    "coupling_fixture_validation.json",
    "treatment_marginal_validation.json",
    "validation_summary.json",
)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pq.write_table(
        pa.Table.from_pylist(list(rows)),
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=16384,
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPOSITORY, text=True).strip()


def load_inputs() -> list[dict[str, Any]]:
    if sha256_file(S07_BANK) != S07_BANK_SHA256:
        raise AssertionError("S07 scenario bank hash changed")
    records = pq.read_table(S07_BANK).to_pylist()
    records.sort(
        key=lambda item: (
            item["split"],
            item["n"],
            item["valueProfile"],
            item["orderStructure"],
            item["replicateOrdinal"],
            item["inputScenarioId"],
        )
    )
    if len(records) != 56_430:
        raise AssertionError("S08 requires the complete frozen S07 bank")
    if len({item["inputScenarioId"] for item in records}) != len(records):
        raise AssertionError("S07 input scenario IDs are not unique")
    if any(item["outcomeAccess"] != "none" for item in records):
        raise AssertionError("input scenario bank unexpectedly carries outcome access")
    if any(
        item["forbiddenConfirmatoryPlacementClass"]
        != "search_derived_exploratory"
        for item in records
    ):
        raise AssertionError("S07 forbidden-placement marker changed")
    return records


def placement_context(record: Mapping[str, Any]) -> PlacementContext:
    identities = tuple(
        f"cell-{int(index):04d}" for index in record["initialOccupancyIndices"]
    )
    values = tuple(record["initialValues"])
    return PlacementContext(
        context_id=str(record["inputScenarioId"]),
        identity_ids=identities,
        values=values,
    )


def build_fault_maps(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    profiles = assign_fault_profiles(records)
    rows: list[dict[str, Any]] = []
    by_input: dict[str, dict[str, Any]] = {}
    for record in records:
        input_id = str(record["inputScenarioId"])
        profile = profiles[input_id]
        context = placement_context(record)
        placement = generate_structural_placement(
            context,
            PlacementClass(profile.placement_class),
            profile.fault_count,
            seed=MASTER_SEED,
            replicate_ordinal=int(record["replicateOrdinal"]),
        )
        placement.validate_against(context)
        if (
            placement.outcome_access != OutcomeAccess.NONE
            or placement.analysis_role
            != AnalysisRole.OUTCOME_BLIND_STRUCTURAL_CANDIDATE
            or not placement.confirmatory_eligible
        ):
            raise AssertionError("an ineligible placement reached the S08 map catalog")
        row = {
            "schemaVersion": "e02.s08.fault_map_catalog.v1",
            "inputScenarioId": input_id,
            "split": record["split"],
            "protected": record["protected"],
            "n": record["n"],
            "valueProfile": record["valueProfile"],
            "orderStructure": record["orderStructure"],
            "replicateOrdinal": record["replicateOrdinal"],
            "faultProfileId": profile.profile_id,
            "faultMapId": placement.placement_id,
            "contextSha256": placement.context_sha256,
            "placementClass": placement.placement_class.value,
            "faultCount": placement.fault_count,
            "positions": list(placement.positions),
            "identityIds": list(placement.identity_ids),
            "valueRanks": list(placement.value_ranks),
            "seed": str(placement.seed),
            "rngStream": placement.rng_stream,
            "generationAddress": placement.generation_address,
            "outcomeAccess": placement.outcome_access.value,
            "analysisRole": placement.analysis_role.value,
            "confirmatoryEligible": placement.confirmatory_eligible,
            "searchDerived": False,
            "searchTail": None,
            "searchScore": None,
            "pairingPrespecificationSha256": PAIRING_PRESPECIFICATION_SHA256,
        }
        rows.append(row)
        by_input[input_id] = row
    if len(by_input) != len(records):
        raise AssertionError("fault map catalog did not preserve one map per input")
    return rows, by_input


def build_pairing_rows(
    records: Sequence[Mapping[str, Any]],
    maps: Mapping[str, Mapping[str, Any]],
    arm_catalog_sha256: str,
    executable_arm_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        fault_map = maps[str(record["inputScenarioId"])]
        initial_permutation_id = content_id(
            "ip1",
            {
                "inputScenarioId": record["inputScenarioId"],
                "initialValuesSha256": record["initialValuesSha256"],
                "initialOccupancySha256": record["initialOccupancySha256"],
            },
        )
        for policy in POLICIES:
            assignment_id = algotype_assignment_id(int(record["n"]), policy)
            for direction in DIRECTIONS:
                content = pairing_block_content(
                    record,
                    policy=policy,
                    direction=direction,
                    algotype_id=assignment_id,
                    fault_map_id=str(fault_map["faultMapId"]),
                )
                block_id = pairing_block_id(content)
                rows.append(
                    {
                        "schemaVersion": PAIRING_SCHEMA_VERSION,
                        "pairingInterfaceVersion": PAIRING_INTERFACE_VERSION,
                        "pairingBlockId": block_id,
                        "pairingBlockContentSha256": block_id.split(":", 1)[1],
                        "inputScenarioId": record["inputScenarioId"],
                        "split": record["split"],
                        "protected": record["protected"],
                        "allowedUse": record["allowedUse"],
                        "n": record["n"],
                        "valueProfile": record["valueProfile"],
                        "orderStructure": record["orderStructure"],
                        "replicateOrdinal": record["replicateOrdinal"],
                        "initialPermutationId": initial_permutation_id,
                        "initialValuesSha256": record["initialValuesSha256"],
                        "initialOccupancySha256": record["initialOccupancySha256"],
                        "permutationCoupling": "exact_shared_immutable_s07",
                        "policyProfile": policy,
                        "algotypeAssignmentProfile": "homogeneous_exact",
                        "algotypeAssignmentId": assignment_id,
                        "algotypeCoupling": "exact_shared_within_pairing_block",
                        "direction": direction,
                        "directionCoupling": "exact_shared_within_pairing_block",
                        "faultProfileId": fault_map["faultProfileId"],
                        "faultMapId": fault_map["faultMapId"],
                        "placementClass": fault_map["placementClass"],
                        "faultCount": fault_map["faultCount"],
                        "faultPositions": fault_map["positions"],
                        "faultIdentityIds": fault_map["identityIds"],
                        "faultMapOutcomeAccess": fault_map["outcomeAccess"],
                        "faultMapConfirmatoryEligible": fault_map[
                            "confirmatoryEligible"
                        ],
                        "faultMapCoupling": "exact_shared_across_valid_faulted_arms",
                        "normalScenarioVariantId": scenario_variant_id(
                            block_id, "normal", str(fault_map["faultMapId"])
                        ),
                        "passiveScenarioVariantId": scenario_variant_id(
                            block_id, "passive", str(fault_map["faultMapId"])
                        ),
                        "stuckScenarioVariantId": scenario_variant_id(
                            block_id, "stuck", str(fault_map["faultMapId"])
                        ),
                        "runtimeSeed": record["scenarioSeed"],
                        "runtimeRootRule": "seed_plus_executable_scenario_id",
                        "eventBudgetProfile": record["eventBudgetProfile"],
                        "eventBudgetOpportunities": record[
                            "eventBudgetOpportunities"
                        ],
                        "contrastArmCatalogSha256": arm_catalog_sha256,
                        "executableArmCount": executable_arm_count,
                        "primaryEstimandCount": 12,
                        "outcomeAccess": "none",
                        "protectedOutcomeOpened": False,
                        "searchDerivedPlacementAssigned": False,
                        "traceSelected": record["traceSelected"],
                        "traceSelectionReason": record["traceSelectionReason"],
                        "pairingPrespecificationSha256": PAIRING_PRESPECIFICATION_SHA256,
                        "streamSpecificationSha256": STREAM_SPECIFICATION_SHA256,
                    }
                )
    return rows


def validate_scenario_ids(
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    maps: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reconstructed_ids = 0
    for record in records:
        content = {
            "schemaVersion": "E02-scale-input-suite-v1",
            "split": record["split"],
            "protected": record["protected"],
            "n": record["n"],
            "valueProfile": record["valueProfile"],
            "orderStructure": record["orderStructure"],
            "replicateOrdinal": record["replicateOrdinal"],
            "generationAttempt": record["generationAttempt"],
            "generationAddress": record["generationAddress"],
            "levelValues": record["levelValues"],
            "levelCounts": record["levelCounts"],
            "initialOccupancyIndices": record["initialOccupancyIndices"],
            "scenarioSeed": record["scenarioSeed"],
            "eventBudgetOpportunities": record["eventBudgetOpportunities"],
        }
        if content_id("i1", content) != record["inputScenarioId"]:
            raise AssertionError("S07 input scenario ID does not match canonical content")
        reconstructed_ids += 1
    full_reconstruction_keys: set[tuple[Any, ...]] = set()
    full_reconstructions = 0
    for record in records:
        key = (
            record["split"],
            record["n"],
            record["valueProfile"],
            record["orderStructure"],
        )
        if key in full_reconstruction_keys:
            continue
        rebuilt = scenario_from_record(record)
        if rebuilt.input_scenario_id != record["inputScenarioId"]:
            raise AssertionError("sampled full S07 reconstruction changed its ID")
        full_reconstruction_keys.add(key)
        full_reconstructions += 1
    seen_blocks: set[str] = set()
    variants: set[str] = set()
    blocks_per_input: Counter[str] = Counter()
    for row in pair_rows:
        record = {
            key: row[key]
            for key in (
                "inputScenarioId",
                "split",
                "n",
                "valueProfile",
                "orderStructure",
                "replicateOrdinal",
                "initialValuesSha256",
                "initialOccupancySha256",
                "eventBudgetProfile",
                "eventBudgetOpportunities",
            )
        }
        record["scenarioSeed"] = row["runtimeSeed"]
        expected_content = pairing_block_content(
            record,
            policy=str(row["policyProfile"]),
            direction=str(row["direction"]),
            algotype_id=str(row["algotypeAssignmentId"]),
            fault_map_id=str(row["faultMapId"]),
        )
        expected = pairing_block_id(expected_content)
        if expected != row["pairingBlockId"]:
            raise AssertionError("pairing block ID does not match semantic content")
        if expected in seen_blocks:
            raise AssertionError("duplicate pairing block ID")
        seen_blocks.add(expected)
        blocks_per_input[str(row["inputScenarioId"])] += 1
        for mobility, field in (
            ("normal", "normalScenarioVariantId"),
            ("passive", "passiveScenarioVariantId"),
            ("stuck", "stuckScenarioVariantId"),
        ):
            expected_variant = scenario_variant_id(
                expected, mobility, str(row["faultMapId"])
            )
            if expected_variant != row[field] or expected_variant in variants:
                raise AssertionError("scenario variant ID mismatch or collision")
            variants.add(expected_variant)
    return {
        "schemaVersion": "e02.s08.scenario_id_validation.v1",
        "researchStepId": "S08",
        "success": True,
        "inputScenarioRows": len(records),
        "uniqueInputScenarioIds": len({item["inputScenarioId"] for item in records}),
        "inputScenarioIdsRecomputedFromCanonicalContent": reconstructed_ids,
        "fullInputScenariosReconstructed": full_reconstructions,
        "fullReconstructionCoverage": "one row per split-by-n-by-valueProfile-by-orderStructure stratum",
        "pairingRows": len(pair_rows),
        "uniquePairingBlockIds": len(seen_blocks),
        "blocksPerInputMinimum": min(blocks_per_input.values()),
        "blocksPerInputMaximum": max(blocks_per_input.values()),
        "uniqueScenarioVariantIds": len(variants),
        "scenarioVariantIdSemantics": "S08 design identity, not executable r1 identity",
    }


def validate_splits(
    records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    fault_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    input_counts = Counter(item["split"] for item in records)
    pair_counts = Counter(item["split"] for item in pair_rows)
    protected_inputs = sum(item["protected"] for item in records)
    protected_pairs = sum(item["protected"] for item in pair_rows)
    if input_counts != {
        "screening_pool": 11_250,
        "confirmatory_holdout": 45_000,
        "runtime_validation": 180,
    }:
        raise AssertionError("S07 split counts changed")
    if pair_counts != {
        "screening_pool": 67_500,
        "confirmatory_holdout": 270_000,
        "runtime_validation": 1_080,
    }:
        raise AssertionError("S08 pairing split counts are incorrect")
    if protected_inputs != 45_000 or protected_pairs != 270_000:
        raise AssertionError("protected flags are not preserved")
    if any(
        item["protected"] != (item["split"] == "confirmatory_holdout")
        for item in pair_rows
    ):
        raise AssertionError("protected flag leaked across split")
    search_assignments = sum(
        item["placementClass"] in FORBIDDEN_PLACEMENTS or item["searchDerived"]
        for item in fault_rows
    )
    if search_assignments:
        raise AssertionError("forbidden search-derived placement was assigned")
    if any(item["outcomeAccess"] != "none" for item in fault_rows):
        raise AssertionError("fault assignment used an outcome-accessed placement")
    return {
        "schemaVersion": "e02.s08.split_enforcement_validation.v1",
        "researchStepId": "S08",
        "success": True,
        "inputRowsBySplit": dict(sorted(input_counts.items())),
        "pairingRowsBySplit": dict(sorted(pair_counts.items())),
        "protectedInputRows": protected_inputs,
        "protectedPairingRows": protected_pairs,
        "protectedOutcomeFieldsRead": [],
        "protectedOutcomeReads": 0,
        "searchOutcomeFieldsRead": [],
        "searchOutcomeReads": 0,
        "searchDerivedAssignments": search_assignments,
        "confirmatoryAssignmentInputs": "input-only S07 state and S08 frozen addresses",
        "confirmatoryOutcomeStatus": "unopened",
    }


def marginal_artifacts(
    pair_rows: Sequence[Mapping[str, Any]],
    fault_rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_counts: Counter[tuple[Any, ...]] = Counter(
        (
            item["split"],
            item["n"],
            item["valueProfile"],
            item["orderStructure"],
            item["policyProfile"],
            item["direction"],
        )
        for item in pair_rows
    )
    arms = executable_arms(catalog)
    marginal_rows: list[dict[str, Any]] = []
    for arm in arms:
        settings_hash = hashlib.sha256(
            canonical_json_bytes(arm["settings"])
        ).hexdigest()
        for key, count in sorted(base_counts.items()):
            marginal_rows.append(
                {
                    "schemaVersion": "e02.s08.treatment_arm_marginals.v1",
                    "estimandId": arm["estimandId"],
                    "armId": arm["armId"],
                    "armLabel": arm["armLabel"],
                    "armSettingsSha256": settings_hash,
                    "split": key[0],
                    "n": key[1],
                    "valueProfile": key[2],
                    "orderStructure": key[3],
                    "policyProfile": key[4],
                    "direction": key[5],
                    "pairingBlockCount": count,
                    "assignmentWeight": 1.0,
                }
            )
    by_estimand_stratum: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in marginal_rows:
        key = (
            row["estimandId"],
            row["split"],
            row["n"],
            row["valueProfile"],
            row["orderStructure"],
            row["policyProfile"],
            row["direction"],
        )
        by_estimand_stratum[key].append(int(row["pairingBlockCount"]))
    arm_spreads = [max(values) - min(values) for values in by_estimand_stratum.values()]

    profile_counts: Counter[tuple[Any, ...]] = Counter(
        (
            item["split"],
            item["n"],
            item["valueProfile"],
            item["orderStructure"],
            item["placementClass"],
            item["faultCount"],
        )
        for item in fault_rows
    )
    profile_rows = [
        {
            "split": key[0],
            "n": key[1],
            "valueProfile": key[2],
            "orderStructure": key[3],
            "placementClass": key[4],
            "faultCount": key[5],
            "inputScenarioCount": count,
        }
        for key, count in sorted(profile_counts.items())
    ]
    grouped_profiles: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in profile_rows:
        grouped_profiles[
            (
                row["split"],
                row["n"],
                row["valueProfile"],
                row["orderStructure"],
            )
        ].append(row["inputScenarioCount"])
    scientific_spreads = [
        max(values) - min(values)
        for key, values in grouped_profiles.items()
        if key[0] in {"screening_pool", "confirmatory_holdout"}
    ]
    if max(arm_spreads, default=0) != 0:
        raise AssertionError("treatment-arm marginals differ within an estimand")
    if max(scientific_spreads, default=0) > 1:
        raise AssertionError("fault profile allocation exceeded near-balance threshold")
    validation = {
        "schemaVersion": "e02.s08.treatment_marginal_validation.v1",
        "researchStepId": "S08",
        "success": True,
        "executableArms": len(arms),
        "baselineMarginalStrata": len(base_counts),
        "treatmentArmMarginalRows": len(marginal_rows),
        "maximumWithinEstimandArmCountSpread": max(arm_spreads, default=0),
        "faultProfiles": len(FAULT_PROFILES),
        "faultProfileDistributionRows": len(profile_rows),
        "maximumScientificStratumFaultProfileCountSpread": max(
            scientific_spreads, default=0
        ),
        "runtimeValidationFaultProfileCoverageCaveat": (
            "four rows per factorial cell cannot cover all 12 profiles; runtime_validation "
            "is diagnostic and excluded from scientific estimation"
        ),
        "marginalTreatmentDistributionsChanged": False,
    }
    return marginal_rows, profile_rows, validation


def pair_completeness(
    pair_rows: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    split_counts = Counter(item["split"] for item in pair_rows)
    contrasts = []
    total_assignments = 0
    for contrast in catalog["contrasts"]:
        arm_count = sum(arm["executable"] for arm in contrast["arms"])
        status = contrast["status"]
        if status == "paired_executable" and arm_count not in {2, 4}:
            raise AssertionError("executable contrast has incomplete arm support")
        if status.startswith("unpaired_non_executable") and arm_count != 0:
            raise AssertionError("non-executable contrast leaked an executable arm")
        assignments = len(pair_rows) * arm_count
        total_assignments += assignments
        contrasts.append(
            {
                "estimandId": contrast["estimandId"],
                "status": status,
                "rngPairingStatus": contrast["rngPairingStatus"],
                "armsPerPairingBlock": arm_count,
                "pairingBlocks": len(pair_rows) if arm_count else 0,
                "logicalArmAssignments": assignments,
                "complete": arm_count in {2, 4} if status == "paired_executable" else True,
                "caveat": contrast["caveat"],
            }
        )
    if len(contrasts) != 12:
        raise AssertionError("all 12 S01 primary estimands must be accounted")
    return {
        "schemaVersion": "e02.s08.pair_completeness.v1",
        "researchStepId": "S08",
        "success": True,
        "normalizedDesign": True,
        "pairingBlockRows": len(pair_rows),
        "pairingBlocksBySplit": dict(sorted(split_counts.items())),
        "primaryEstimands": len(contrasts),
        "pairedExecutableEstimands": sum(
            item["status"] == "paired_executable" for item in contrasts
        ),
        "declaredUnpairedNonExecutableEstimands": sum(
            item["status"].startswith("unpaired_non_executable")
            for item in contrasts
        ),
        "executableArmCount": catalog["executableArmCount"],
        "logicalBlockByArmAssignments": total_assignments,
        "cartesianManifestMaterialized": False,
        "contrasts": contrasts,
    }


def stream_isolation(catalog: Mapping[str, Any]) -> dict[str, Any]:
    streams = {
        "fault_profile_assignment_s08_v1": "S08 construction",
        "placement_uniform_exact_s06_v1": "S06 structural placement",
        "placement_clustered_exact_s06_v1": "S06 structural placement",
        "placement_boundary_exact_s06_v1": "S06 structural placement",
        "placement_median_rank_exact_s06_v1": "S06 structural placement",
        "actor_activation": "uniform scheduler",
        "scheduler_permutation_s04_v1": "permutation scheduler",
        "bubble_side": "Bubble policy",
        "conflict_priority": "synchronous conflict",
        "action_failure_bernoulli_s05_v1": "Bernoulli action failure",
        "action_failure_transient_s05_v1": "transient action failure",
        "sensing_value_error_s05_v1": "value sensing error",
        "sensing_status_error_s05_v1": "status sensing error",
    }
    if len(streams) != 13:
        raise AssertionError("semantic stream namespace collision")
    by_id = {item["estimandId"]: item for item in catalog["contrasts"]}
    required_unpaired = {
        "E02-S01-E03",
        "E02-S01-E06",
        "E02-S01-E09",
        "E02-S01-E10",
        "E02-S01-E12",
    }
    naive_claims = [
        estimand
        for estimand in required_unpaired
        if "rng_unpaired" not in by_id[estimand]["rngPairingStatus"]
    ]
    if naive_claims:
        raise AssertionError("an incompatible RNG contrast was labeled coupled")
    return {
        "schemaVersion": "e02.s08.stream_name_isolation_validation.v1",
        "researchStepId": "S08",
        "success": True,
        "semanticStreamCount": len(streams),
        "uniqueSemanticStreamNames": len(set(streams)),
        "streams": [
            {"name": name, "owner": owner} for name, owner in sorted(streams.items())
        ],
        "requiredRngUnpairedEstimands": sorted(required_unpaired),
        "naiveSharedStreamClaims": naive_claims,
        "uniformPermutationNamesDistinct": (
            "actor_activation" != "scheduler_permutation_s04_v1"
        ),
        "workerOrderInfluence": "none",
    }


def normal_validation_scenario(
    record: Mapping[str, Any], *, max_activations: int
) -> Scenario:
    source = scenario_from_record(record)
    cells = tuple(
        Cell(
            cell_id=identity,
            value=value,
            policy=Policy.BUBBLE,
            direction=Direction.ASCENDING,
            fault=FaultMode.NORMAL,
        )
        for identity, value in zip(source.identity_ids, source.values_by_identity)
    )
    return Scenario.create(
        cells,
        initial_occupancy=source.initial_occupancy,
        seed=source.scenario_seed,
        max_activations=max_activations,
        generation_key=f"S08/coupling-fixture/{source.input_scenario_id}/normal",
        fault_placement="explicit",
    )


def coupling_fixture(
    records: Sequence[Mapping[str, Any]], maps: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    record = next(
        item
        for item in records
        if item["split"] == "runtime_validation"
        and item["n"] == 20
        and item["valueProfile"] == "unique"
        and item["orderStructure"] == "block_scrambled"
        and item["replicateOrdinal"] == 0
    )
    source = scenario_from_record(record)
    context = placement_context(record)
    map_row = maps[source.input_scenario_id]
    placement = generate_structural_placement(
        context,
        PlacementClass(map_row["placementClass"]),
        int(map_row["faultCount"]),
        seed=MASTER_SEED,
        replicate_ordinal=int(record["replicateOrdinal"]),
    )
    normal = normal_validation_scenario(record, max_activations=64)
    passive = materialize_fault_scenario(
        context,
        placement,
        fault_mode=FaultMode.PASSIVE,
        policy=Policy.BUBBLE,
        direction=Direction.ASCENDING,
        seed=source.scenario_seed,
        max_activations=64,
        generation_key=f"S08/coupling-fixture/{source.input_scenario_id}/passive",
    )
    stuck = materialize_fault_scenario(
        context,
        placement,
        fault_mode=FaultMode.STUCK,
        policy=Policy.BUBBLE,
        direction=Direction.ASCENDING,
        seed=source.scenario_seed,
        max_activations=64,
        generation_key=f"S08/coupling-fixture/{source.input_scenario_id}/stuck",
    )
    passive_ids = sorted(
        cell.cell_id for cell in passive.cells if cell.fault == FaultMode.PASSIVE
    )
    stuck_ids = sorted(
        cell.cell_id for cell in stuck.cells if cell.fault == FaultMode.STUCK
    )
    if passive_ids != stuck_ids or passive_ids != sorted(placement.identity_ids):
        raise AssertionError("passive/stuck pairing did not retain the same fault map")
    same_root_a = u64(
        passive.seed, passive.scenario_id, "action_failure_bernoulli_s05_v1", 0
    )
    same_root_b = u64(
        passive.seed, passive.scenario_id, "action_failure_bernoulli_s05_v1", 0
    )
    different_mobility_root = u64(
        stuck.seed, stuck.scenario_id, "action_failure_bernoulli_s05_v1", 0
    )
    uniform_value = u64(
        passive.seed, passive.scenario_id, "actor_activation", 0
    )
    permutation_value = u64(
        passive.seed, passive.scenario_id, "scheduler_permutation_s04_v1", 0
    )

    uniform_controller = FrozenSchedulerController(
        SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION),
        seed=passive.seed,
        scenario_id=passive.scenario_id,
        actor_ids=tuple(cell.cell_id for cell in passive.cells),
    )
    permutation_controller = FrozenSchedulerController(
        SchedulerExecutionContract(SchedulerFamily.RANDOM_PERMUTATION_SWEEP),
        seed=passive.seed,
        scenario_id=passive.scenario_id,
        actor_ids=tuple(cell.cell_id for cell in passive.cells),
    )
    uniform_names: set[str] = set()
    permutation_names: set[str] = set()
    for event_index in range(40):
        for opportunity in uniform_controller(event_index, 40 - event_index):
            uniform_names.update(item[0] for item in opportunity.random_draws)
        for opportunity in permutation_controller(event_index, 40 - event_index):
            permutation_names.update(item[0] for item in opportunity.random_draws)
    if uniform_names != {"actor_activation"}:
        raise AssertionError("uniform scheduler stream identity changed")
    if permutation_names != {"scheduler_permutation_s04_v1"}:
        raise AssertionError("permutation scheduler stream identity changed")

    fault_contract = FaultExecutionContract(
        mobility=MobilityProfile.PASSIVE,
        action_failure=ActionFailureProfile.BERNOULLI_P,
    )
    scheduler = SchedulerExecutionContract(SchedulerFamily.UNIFORM_RANDOM_ACTIVATION)
    central = run_faulted_architecture(
        passive,
        ArchitectureExecutionContract.central_local_k1(),
        scheduler,
        fault_contract,
        trace_mode="digest",
    )
    distributed = run_faulted_architecture(
        passive,
        ArchitectureExecutionContract.distributed_local(),
        scheduler,
        fault_contract,
        trace_mode="digest",
    )
    central_draws = [
        (item.event_index, item.stream, item.raw_uint64)
        for item in central.action_failure_audit
    ]
    distributed_draws = [
        (item.event_index, item.stream, item.raw_uint64)
        for item in distributed.action_failure_audit
    ]
    central_replay = exact_replay_fault(central).to_json_bytes() == central.to_json_bytes()
    distributed_replay = (
        exact_replay_fault(distributed).to_json_bytes()
        == distributed.to_json_bytes()
    )
    if central_draws != distributed_draws:
        raise AssertionError("matched architectures did not share exogenous failures")
    if central.result.to_json_bytes() != distributed.result.to_json_bytes():
        raise AssertionError("S02 no-fault/matched-route differential parity changed")
    if not all(central.opportunity_validation().values()) or not all(
        distributed.opportunity_validation().values()
    ):
        raise AssertionError("coupling fixture failed opportunity/ledger validation")
    return {
        "schemaVersion": "e02.s08.coupling_fixture_validation.v1",
        "researchStepId": "S08",
        "success": True,
        "inputScenarioId": source.input_scenario_id,
        "inputSplit": source.split.value,
        "protected": source.protected,
        "scientificOutcomeUse": False,
        "normalScenarioId": normal.scenario_id,
        "passiveScenarioId": passive.scenario_id,
        "stuckScenarioId": stuck.scenario_id,
        "passiveStuckScenarioIdsDiffer": passive.scenario_id != stuck.scenario_id,
        "faultMapSharedPassiveStuck": passive_ids == stuck_ids,
        "faultCount": len(passive_ids),
        "sameScenarioSameStreamDrawExact": same_root_a == same_root_b,
        "passiveStuckSameSeedDifferentRuntimeRoot": (
            same_root_a != different_mobility_root
        ),
        "uniformPermutationStreamNames": {
            "uniform": sorted(uniform_names),
            "permutation": sorted(permutation_names),
        },
        "uniformPermutationSampleValuesDiffer": uniform_value != permutation_value,
        "uniformPermutationClaim": "scenario_paired_rng_unpaired",
        "architectureFailureDrawsExact": central_draws == distributed_draws,
        "architectureFailureDrawCount": len(central_draws),
        "architectureNativeDifferentialParity": (
            central.result.to_json_bytes() == distributed.result.to_json_bytes()
        ),
        "centralExactReplay": central_replay,
        "distributedExactReplay": distributed_replay,
        "centralOpportunityLedgerChecksPassed": all(
            central.opportunity_validation().values()
        ),
        "distributedOpportunityLedgerChecksPassed": all(
            distributed.opportunity_validation().values()
        ),
    }


def validate_prespecification() -> None:
    if sha256_file(PAIRING_SPEC) != PAIRING_PRESPECIFICATION_SHA256:
        raise AssertionError("frozen S08 pairing prespecification changed")
    if sha256_file(STREAM_SPEC) != STREAM_SPECIFICATION_SHA256:
        raise AssertionError("frozen S08 stream specification changed")
    pairing = json.loads(PAIRING_SPEC.read_text(encoding="utf-8"))
    streams = json.loads(STREAM_SPEC.read_text(encoding="utf-8"))
    if not pairing["frozenBeforeImplementation"] or not streams[
        "frozenBeforeImplementation"
    ]:
        raise AssertionError("S08 contracts were not marked frozen before implementation")


def compare_core(output: Path, compare_to: Path | None) -> dict[str, Any]:
    if compare_to is None:
        return {
            "schemaVersion": "e02.s08.deterministic_regeneration.v1",
            "researchStepId": "S08",
            "comparisonPerformed": False,
            "success": True,
            "coreFiles": len(CORE_FILES),
            "mismatches": [],
        }
    comparisons = []
    mismatches = []
    for name in CORE_FILES:
        current = sha256_file(output / name)
        prior = sha256_file(compare_to / name)
        equal = current == prior
        comparisons.append(
            {"path": name, "currentSha256": current, "priorSha256": prior, "equal": equal}
        )
        if not equal:
            mismatches.append(name)
    if mismatches:
        raise AssertionError(f"S08 deterministic regeneration mismatch: {mismatches}")
    return {
        "schemaVersion": "e02.s08.deterministic_regeneration.v1",
        "researchStepId": "S08",
        "comparisonPerformed": True,
        "success": True,
        "coreFiles": len(CORE_FILES),
        "byteIdenticalCoreFiles": len(CORE_FILES),
        "mismatches": mismatches,
        "comparisons": comparisons,
    }


def build(output: Path, compare_to: Path | None) -> None:
    validate_prespecification()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairing_prespecification.json").write_bytes(PAIRING_SPEC.read_bytes())
    (output / "semantic_random_stream_specification.json").write_bytes(
        STREAM_SPEC.read_bytes()
    )
    records = load_inputs()
    catalog = contrast_catalog()
    catalog_bytes = canonical_json_bytes(catalog) + b"\n"
    (output / "contrast_arm_catalog.json").write_bytes(catalog_bytes)
    catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()

    fault_rows, maps = build_fault_maps(records)
    pair_rows = build_pairing_rows(
        records,
        maps,
        catalog_sha,
        int(catalog["executableArmCount"]),
    )
    if len(pair_rows) != 338_580:
        raise AssertionError("pairing bank row count does not match the frozen contract")
    write_parquet(output / "fault_map_catalog.parquet", fault_rows)
    write_parquet(output / "pairing_manifest.parquet", pair_rows)

    coupling_rows = coupling_matrix_rows(catalog)
    write_csv(
        output / "coupling_matrix.csv",
        coupling_rows,
        ("estimandId", "componentOrStream", "coupling", "reason"),
    )
    marginal_rows, profile_rows, marginal_validation = marginal_artifacts(
        pair_rows, fault_rows, catalog
    )
    write_parquet(output / "treatment_arm_marginals.parquet", marginal_rows)
    write_csv(
        output / "fault_profile_distribution.csv",
        profile_rows,
        (
            "split",
            "n",
            "valueProfile",
            "orderStructure",
            "placementClass",
            "faultCount",
            "inputScenarioCount",
        ),
    )
    write_json(output / "treatment_marginal_validation.json", marginal_validation)

    scenario_validation = validate_scenario_ids(records, pair_rows, maps)
    split_validation = validate_splits(records, pair_rows, fault_rows)
    completeness = pair_completeness(pair_rows, catalog)
    stream_validation = stream_isolation(catalog)
    fixture_validation = coupling_fixture(records, maps)
    write_json(output / "scenario_id_validation.json", scenario_validation)
    write_json(output / "split_enforcement_validation.json", split_validation)
    write_json(output / "pair_completeness.json", completeness)
    write_json(output / "stream_name_isolation_validation.json", stream_validation)
    write_json(output / "coupling_fixture_validation.json", fixture_validation)

    validation = {
        "schemaVersion": "e02.s08.validation_summary.v1",
        "researchStepId": "S08",
        "success": True,
        "validationResult": "PASS",
        "componentGatesPassed": 7,
        "componentGatesTotal": 7,
        "inputRows": len(records),
        "pairingRows": len(pair_rows),
        "faultMaps": len(fault_rows),
        "primaryEstimands": catalog["primaryEstimandCount"],
        "pairedExecutableEstimands": catalog["executableEstimandCount"],
        "declaredUnpairedNonExecutableEstimands": catalog[
            "declaredUnpairedNonExecutableCount"
        ],
        "executableArms": catalog["executableArmCount"],
        "logicalBlockByArmAssignments": completeness[
            "logicalBlockByArmAssignments"
        ],
        "scenarioIdValidation": scenario_validation["success"],
        "splitEnforcement": split_validation["success"],
        "streamIsolation": stream_validation["success"],
        "pairCompleteness": completeness["success"],
        "treatmentMarginals": marginal_validation["success"],
        "couplingFixtures": fixture_validation["success"],
        "protectedOutcomeReads": split_validation["protectedOutcomeReads"],
        "searchDerivedAssignments": split_validation["searchDerivedAssignments"],
    }
    write_json(output / "validation_summary.json", validation)

    provenance = {
        "schemaVersion": "e02.s08.provenance.v1",
        "researchStepId": "S08",
        "repository": str(REPOSITORY),
        "branch": git_value("branch", "--show-current"),
        "sourceCommitBeforeS08": git_value("rev-parse", "HEAD"),
        "generator": "scripts/build_pairing_manifest.py",
        "pairingModule": "causal_simulator/pairing.py",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyarrow": pa.__version__,
        "workers": 1,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "masterSeedHex": f"0x{MASTER_SEED:032x}",
        "inputs": {
            str(S07_BANK): sha256_file(S07_BANK),
            str(S01_ESTIMANDS): sha256_file(S01_ESTIMANDS),
            str(S06_STRUCTURAL_LOCK): sha256_file(S06_STRUCTURAL_LOCK),
            str(S06_BANK): sha256_file(S06_BANK),
            str(E01_SEED_SPEC): sha256_file(E01_SEED_SPEC),
            str(PAIRING_SPEC): sha256_file(PAIRING_SPEC),
            str(STREAM_SPEC): sha256_file(STREAM_SPEC),
        },
        "outcomeFilesRead": [],
        "networkAccessUsed": False,
        "newDependenciesInstalled": [],
    }
    write_json(output / "provenance_manifest.json", provenance)

    deterministic = compare_core(output, compare_to)
    write_json(output / "deterministic_regeneration.json", deterministic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    build(args.output, args.compare_to)


if __name__ == "__main__":
    main()
