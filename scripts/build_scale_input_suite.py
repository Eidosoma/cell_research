#!/usr/bin/env python3
"""Build and validate the frozen E02 S07 scale/input scenario extension."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from causal_simulator.architectures import ArchitectureExecutionContract
from causal_simulator.faults import FaultExecutionContract, run_faulted_architecture
from causal_simulator.scenario_extensions import (
    FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS,
    MASTER_SEED,
    PRESPECIFICATION_SHA256,
    SCENARIO_EXTENSION_VERSION,
    SIZES,
    SPLIT_REPLICATES,
    STRUCTURAL_PLACEMENT_CLASSES,
    TRACE_RATES,
    InputScenario,
    OrderStructure,
    ScenarioSplit,
    ValueProfile,
    assign_trace_selection,
    brute_force_duplicate_aware_footrule,
    generate_unique_input_scenario,
    maximum_strict_unequal_inversions,
    minimum_duplicate_aware_footrule,
    scenario_from_record,
    strict_unequal_inversions,
    trace_selection_count,
    value_counts,
)
from causal_simulator.schedulers import SchedulerExecutionContract, SchedulerFamily
from reference_simulator.model import (
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)


S06_BANK = Path("/artifacts/research_steps/S06/fault_placement_bank.parquet")
S06_LOCK = Path("/artifacts/research_steps/S06/structural_lock.json")
DESCRIPTOR_COLUMNS = (
    "inputScenarioId",
    "split",
    "protected",
    "n",
    "valueProfile",
    "orderStructure",
    "replicateOrdinal",
    "generationAttempt",
    "distinctValueCount",
    "duplicateIdentityFraction",
    "strictUnequalInversionsAscending",
    "maximumStrictUnequalInversions",
    "normalizedStrictUnequalInversionsAscending",
    "strictAdjacentViolationsAscending",
    "strictAdjacentViolationsDescending",
    "minimumDuplicateAwareFootruleAscending",
    "minimumDuplicateAwareFootruleDescending",
    "traceSelected",
    "traceSelectionRateNominal",
    "eventBudgetOpportunities",
    "outcomeAccess",
)
CORE_REGENERATION_FILES = (
    "scenario_extension.parquet",
    "disorder_descriptors.parquet",
    "scenario_extension_schema.json",
    "balance_summary.csv",
    "disorder_summary.csv",
    "split_manifest.json",
    "holdout_integrity.json",
    "target_feasibility_validation.json",
    "metric_validation.json",
    "outcome_access_validation.json",
    "reconstruction_validation.json",
    "validation_summary.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def _factor_cell_key(
    n: int, value_profile: ValueProfile, order_structure: OrderStructure
) -> str:
    return f"n{n:03d}_{value_profile.value}_{order_structure.value}"


def _generate_factor_cell(args: tuple[int, str, str, str]) -> dict[str, Any]:
    n, value_name, order_name, scratch_text = args
    value_profile = ValueProfile(value_name)
    order_structure = OrderStructure(order_name)
    scenarios: list[InputScenario] = []
    seen: set[str] = set()
    for split in ScenarioSplit:
        for replicate in range(SPLIT_REPLICATES[split]):
            scenarios.append(
                generate_unique_input_scenario(
                    split,
                    n,
                    value_profile,
                    order_structure,
                    replicate,
                    seen_initial_value_hashes=seen,
                )
            )
    trace_flags = assign_trace_selection(scenarios)
    records: list[dict[str, Any]] = []
    descriptor_records: list[dict[str, Any]] = []
    for scenario in scenarios:
        record = scenario.core_record()
        selected, reason = trace_flags[scenario.input_scenario_id]
        record.update(
            {
                "traceSelected": selected,
                "traceSelectionReason": reason,
                "traceSelectionRateNominal": (
                    0.25
                    if scenario.split == ScenarioSplit.RUNTIME_VALIDATION
                    else TRACE_RATES[n]
                ),
                "traceFailuresAndAnomaliesAlways": True,
            }
        )
        records.append(record)
        descriptor_records.append({key: record[key] for key in DESCRIPTOR_COLUMNS})
    key = _factor_cell_key(n, value_profile, order_structure)
    scratch = Path(scratch_text)
    write_parquet(scratch / f"{key}.scenarios.parquet", records)
    write_parquet(scratch / f"{key}.descriptors.parquet", descriptor_records)
    return {
        "key": key,
        "n": n,
        "valueProfile": value_profile.value,
        "orderStructure": order_structure.value,
        "rows": len(records),
        "maximumGenerationAttempt": max(item.generation_attempt for item in scenarios),
        "collisionOrRangeRejections": sum(item.generation_attempt for item in scenarios),
        "sequenceHashes": len(seen),
    }


def merge_parts(
    scratch: Path, output: Path, suffix: str, ordered_keys: Sequence[str]
) -> None:
    writer: pq.ParquetWriter | None = None
    try:
        for key in ordered_keys:
            table = pq.read_table(scratch / f"{key}.{suffix}.parquet")
            if writer is None:
                writer = pq.ParquetWriter(
                    output,
                    table.schema,
                    compression="zstd",
                    compression_level=9,
                    use_dictionary=True,
                    write_statistics=True,
                    version="2.6",
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_scenario_tables(output: Path, scratch: Path, workers: int) -> list[dict[str, Any]]:
    cells = [
        (n, value_profile.value, order_structure.value, str(scratch))
        for n in SIZES
        for value_profile in ValueProfile
        for order_structure in OrderStructure
    ]
    context = multiprocessing.get_context("spawn")
    if workers == 1:
        summaries = [_generate_factor_cell(cell) for cell in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            summaries = list(executor.map(_generate_factor_cell, cells))
    ordered_keys = [item[0] for item in [
        (_factor_cell_key(n, value_profile, order_structure),)
        for n in SIZES
        for value_profile in ValueProfile
        for order_structure in OrderStructure
    ]]
    merge_parts(
        scratch, output / "scenario_extension.parquet", "scenarios", ordered_keys
    )
    merge_parts(
        scratch,
        output / "disorder_descriptors.parquet",
        "descriptors",
        ordered_keys,
    )
    return sorted(summaries, key=lambda item: item["key"])


def generation_summaries_from_output(output: Path) -> list[dict[str, Any]]:
    """Recover deterministic cell summaries when resuming after table creation."""

    parquet = pq.ParquetFile(output / "scenario_extension.parquet")
    summaries: list[dict[str, Any]] = []
    for row_group in range(parquet.num_row_groups):
        rows = parquet.read_row_group(
            row_group,
            columns=[
                "n",
                "valueProfile",
                "orderStructure",
                "generationAttempt",
                "initialValuesSha256",
            ],
        ).to_pylist()
        first = rows[0]
        summaries.append(
            {
                "key": _factor_cell_key(
                    first["n"],
                    ValueProfile(first["valueProfile"]),
                    OrderStructure(first["orderStructure"]),
                ),
                "n": first["n"],
                "valueProfile": first["valueProfile"],
                "orderStructure": first["orderStructure"],
                "rows": len(rows),
                "maximumGenerationAttempt": max(
                    item["generationAttempt"] for item in rows
                ),
                "collisionOrRangeRejections": sum(
                    item["generationAttempt"] for item in rows
                ),
                "sequenceHashes": len(
                    {item["initialValuesSha256"] for item in rows}
                ),
            }
        )
    return sorted(summaries, key=lambda item: item["key"])


def schema_record(schema: pa.Schema) -> dict[str, Any]:
    fields = [
        {
            "name": field.name,
            "arrowType": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return {
        "schemaVersion": "e02.s07.input_scenario_schema.v1",
        "researchStepId": "S07",
        "rowSchemaVersion": "e02.s07.input_scenario.v1",
        "additionalColumnsAllowed": False,
        "fieldCount": len(fields),
        "fields": fields,
        "requiredColumns": [item["name"] for item in fields],
        "listSemantics": {
            "levelValues": "ordered increasing value levels",
            "levelCounts": "exact multiplicities aligned to levelValues",
            "initialOccupancyIndices": "position-ordered identity indices",
            "initialValues": "position-ordered values",
            "confirmatoryEligiblePlacementClasses": "future S08 rule-class allowlist only; no map assignment",
        },
    }


def balance_and_disorder(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    columns = [
        "split",
        "n",
        "valueProfile",
        "orderStructure",
        "traceSelected",
        "normalizedStrictUnequalInversionsAscending",
        "strictAdjacentViolationsAscending",
        "minimumDuplicateAwareFootruleAscending",
    ]
    frame = pq.read_table(output / "scenario_extension.parquet", columns=columns).to_pandas()
    balance_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["split", "n", "valueProfile", "orderStructure"], sort=True
    ):
        split, n, value_profile, order_structure = keys
        expected = SPLIT_REPLICATES[ScenarioSplit(split)]
        balance_rows.append(
            {
                "split": split,
                "n": int(n),
                "valueProfile": value_profile,
                "orderStructure": order_structure,
                "scenarioCount": len(group),
                "expectedScenarioCount": expected,
                "balanced": len(group) == expected,
                "traceSelectedCount": int(group["traceSelected"].sum()),
                "expectedTraceSelectedCount": trace_selection_count(
                    ScenarioSplit(split), int(n)
                ),
                "traceQuotaMatched": int(group["traceSelected"].sum())
                == trace_selection_count(ScenarioSplit(split), int(n)),
            }
        )
    disorder_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["n", "valueProfile", "orderStructure"], sort=True
    ):
        n, value_profile, order_structure = keys
        disorder_rows.append(
            {
                "n": int(n),
                "valueProfile": value_profile,
                "orderStructure": order_structure,
                "scenarioCount": len(group),
                "normalizedInversionsMinimum": float(
                    group["normalizedStrictUnequalInversionsAscending"].min()
                ),
                "normalizedInversionsMedian": float(
                    group["normalizedStrictUnequalInversionsAscending"].median()
                ),
                "normalizedInversionsMaximum": float(
                    group["normalizedStrictUnequalInversionsAscending"].max()
                ),
                "adjacentViolationsMedian": float(
                    group["strictAdjacentViolationsAscending"].median()
                ),
                "duplicateAwareFootruleMedian": float(
                    group["minimumDuplicateAwareFootruleAscending"].median()
                ),
            }
        )
    write_csv(output / "balance_summary.csv", balance_rows)
    write_csv(output / "disorder_summary.csv", disorder_rows)
    return balance_rows, disorder_rows


def split_and_holdout_audit(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = [
        "inputScenarioId",
        "split",
        "protected",
        "n",
        "valueProfile",
        "orderStructure",
        "initialValuesSha256",
        "outcomeAccess",
    ]
    rows = pq.read_table(output / "scenario_extension.parquet", columns=columns).to_pylist()
    split_rows = []
    for split in ScenarioSplit:
        selected = [item for item in rows if item["split"] == split.value]
        ordered_ids = [item["inputScenarioId"] for item in selected]
        split_rows.append(
            {
                "name": split.value,
                "protected": split == ScenarioSplit.CONFIRMATORY_HOLDOUT,
                "allowedUse": {
                    ScenarioSplit.SCREENING_POOL: "S10 screening and method diagnostics only",
                    ScenarioSplit.CONFIRMATORY_HOLDOUT: "locked until estimands, exclusions, and analysis are preregistered",
                    ScenarioSplit.RUNTIME_VALIDATION: "S07 implementation runtime and replay only",
                }[split],
                "replicatesPerFactorialCell": SPLIT_REPLICATES[split],
                "scenarioCount": len(selected),
                "orderedInputScenarioIdSha256": hashlib.sha256(
                    canonical_json_bytes(ordered_ids)
                ).hexdigest(),
            }
        )
    protected = {
        item["initialValuesSha256"]
        for item in rows
        if item["split"] == ScenarioSplit.CONFIRMATORY_HOLDOUT.value
    }
    nonprotected = {
        item["initialValuesSha256"]
        for item in rows
        if item["split"] != ScenarioSplit.CONFIRMATORY_HOLDOUT.value
    }
    cell_hashes: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for item in rows:
        cell_hashes[
            (item["n"], item["valueProfile"], item["orderStructure"])
        ].append(item["initialValuesSha256"])
    holdout = {
        "schemaVersion": "e02.s07.holdout_integrity.v1",
        "researchStepId": "S07",
        "success": (
            not protected & nonprotected
            and all(len(values) == len(set(values)) for values in cell_hashes.values())
            and all(item["outcomeAccess"] == "none" for item in rows)
        ),
        "protectedScenarioCount": sum(
            item["split"] == ScenarioSplit.CONFIRMATORY_HOLDOUT.value for item in rows
        ),
        "protectedNonprotectedInitialSequenceOverlap": len(protected & nonprotected),
        "factorialCellsWithWithinCellSequenceCollisions": sum(
            len(values) != len(set(values)) for values in cell_hashes.values()
        ),
        "uniqueInputScenarioIds": len({item["inputScenarioId"] for item in rows}),
        "scenarioRows": len(rows),
        "allGenerationOutcomeAccessNone": all(
            item["outcomeAccess"] == "none" for item in rows
        ),
        "proceduralProtectionNote": "protected is a workflow-use contract, not filesystem access control",
    }
    manifest = {
        "schemaVersion": "e02.s07.split_manifest.v1",
        "researchStepId": "S07",
        "assignment": "semantic split plus local replicate ordinal; no post-generation random split",
        "crossSplitInitialSequenceOverlapAllowed": False,
        "splits": split_rows,
    }
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "holdout_integrity.json", holdout)
    return manifest, holdout


def target_and_metric_validation(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    columns = [
        "inputScenarioId",
        "n",
        "valueProfile",
        "orderStructure",
        "levelCounts",
        "strictUnequalInversionsAscending",
        "maximumStrictUnequalInversions",
        "normalizedStrictUnequalInversionsAscending",
        "minimumDuplicateAwareFootruleAscending",
        "minimumDuplicateAwareFootruleDescending",
        "ascendingMultisetPreserved",
        "ascendingNonstrictOrdered",
        "descendingMultisetPreserved",
        "descendingNonstrictOrdered",
    ]
    rows = pq.read_table(output / "scenario_extension.parquet", columns=columns).to_pylist()
    target_fields = (
        "ascendingMultisetPreserved",
        "ascendingNonstrictOrdered",
        "descendingMultisetPreserved",
        "descendingNonstrictOrdered",
    )
    reverse_valid = all(
        item["strictUnequalInversionsAscending"]
        == item["maximumStrictUnequalInversions"]
        and item["normalizedStrictUnequalInversionsAscending"] == 1.0
        for item in rows
        if item["orderStructure"] == OrderStructure.REVERSE.value
    )
    nearly_valid = all(
        0 < item["normalizedStrictUnequalInversionsAscending"] <= 0.15
        for item in rows
        if item["orderStructure"] == OrderStructure.NEARLY_SORTED.value
    )
    block_valid = all(
        0.15 <= item["normalizedStrictUnequalInversionsAscending"] <= 0.85
        for item in rows
        if item["orderStructure"] == OrderStructure.BLOCK_SCRAMBLED.value
    )
    exact_counts = all(
        tuple(item["levelCounts"])
        == value_counts(item["n"], ValueProfile(item["valueProfile"]))
        for item in rows
    )
    target = {
        "schemaVersion": "e02.s07.target_feasibility_validation.v1",
        "researchStepId": "S07",
        "scenarioRowsChecked": len(rows),
        "targetPredicatesChecked": len(rows) * len(target_fields),
        "allTargetPredicatesPass": all(
            item[field] for item in rows for field in target_fields
        ),
        "exactMultiplicityVectorsPass": exact_counts,
        "reverseMaximumDistancePass": reverse_valid,
        "nearlySortedDistanceRangePass": nearly_valid,
        "blockScrambledDistanceRangePass": block_valid,
    }
    target["success"] = all(
        target[key]
        for key in (
            "allTargetPredicatesPass",
            "exactMultiplicityVectorsPass",
            "reverseMaximumDistancePass",
            "nearlySortedDistanceRangePass",
            "blockScrambledDistanceRangePass",
        )
    )

    fixtures = (
        (2, 1, 1),
        (2, 2, 1, 1),
        (1, 2, 1, 2),
        (2, 1, 2, 1),
        (3, 1, 2, 1, 3),
        (1, 1, 2, 3, 2, 1),
        (4, 2, 4, 2, 1, 1),
    )
    fixture_rows = []
    for values in fixtures:
        fixture_rows.append(
            {
                "values": list(values),
                "strictUnequalInversions": strict_unequal_inversions(values),
                "maximumStrictUnequalInversions": maximum_strict_unequal_inversions(values),
                "minimumFootruleAscending": minimum_duplicate_aware_footrule(values),
                "bruteForceFootruleAscending": brute_force_duplicate_aware_footrule(values),
                "minimumFootruleDescending": minimum_duplicate_aware_footrule(
                    values, descending=True
                ),
                "bruteForceFootruleDescending": brute_force_duplicate_aware_footrule(
                    values, descending=True
                ),
            }
        )
    sample_columns = [
        "initialValues",
        "strictUnequalInversionsAscending",
        "maximumStrictUnequalInversions",
        "minimumDuplicateAwareFootruleAscending",
        "minimumDuplicateAwareFootruleDescending",
    ]
    parquet = pq.ParquetFile(output / "scenario_extension.parquet")
    sampled_rows: list[dict[str, Any]] = []
    for row_group in range(parquet.num_row_groups):
        group = parquet.read_row_group(row_group, columns=sample_columns).to_pylist()
        sampled_rows.extend((group[0], group[len(group) // 2], group[-1]))
    recomputation_pass = all(
        strict_unequal_inversions(item["initialValues"])
        == item["strictUnequalInversionsAscending"]
        and maximum_strict_unequal_inversions(item["initialValues"])
        == item["maximumStrictUnequalInversions"]
        and minimum_duplicate_aware_footrule(item["initialValues"])
        == item["minimumDuplicateAwareFootruleAscending"]
        and minimum_duplicate_aware_footrule(item["initialValues"], descending=True)
        == item["minimumDuplicateAwareFootruleDescending"]
        for item in sampled_rows
    )
    metric = {
        "schemaVersion": "e02.s07.duplicate_distance_validation.v1",
        "researchStepId": "S07",
        "semantics": {
            "strictUnequalInversions": "pairs i<j with value[i]>value[j]; equality excluded",
            "normalizer": "choose(n,2)-sum choose(multiplicity,2)",
            "minimumDuplicateAwareFootrule": "monotone matching of current and target positions within each equal-value class",
        },
        "fixtureCount": len(fixture_rows),
        "fixtureRows": fixture_rows,
        "bruteForceFixtureAgreement": all(
            item["minimumFootruleAscending"] == item["bruteForceFootruleAscending"]
            and item["minimumFootruleDescending"] == item["bruteForceFootruleDescending"]
            for item in fixture_rows
        ),
        "sampledFullBankRowsRecomputed": len(sampled_rows),
        "sampledFullBankRecomputationPass": recomputation_pass,
    }
    metric["success"] = metric["bruteForceFixtureAgreement"] and recomputation_pass
    write_json(output / "target_feasibility_validation.json", target)
    write_json(output / "metric_validation.json", metric)
    return target, metric


def outcome_access_validation(output: Path) -> dict[str, Any]:
    if not S06_BANK.exists() or not S06_LOCK.exists():
        raise FileNotFoundError("S06 placement evidence is required")
    bank = pq.read_table(
        S06_BANK,
        columns=[
            "placementClass",
            "confirmatoryEligible",
            "outcomeAccess",
            "analysisRole",
        ],
    ).to_pylist()
    classes = Counter(item["placementClass"] for item in bank)
    structural = [
        item for item in bank if item["placementClass"] in STRUCTURAL_PLACEMENT_CLASSES
    ]
    search = [
        item
        for item in bank
        if item["placementClass"] == FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS
    ]
    scenario_columns = [
        "faultMapAssignmentStatus",
        "confirmatoryEligiblePlacementClasses",
        "forbiddenConfirmatoryPlacementClass",
        "outcomeAccess",
    ]
    scenarios = pq.read_table(
        output / "scenario_extension.parquet", columns=scenario_columns
    ).to_pylist()
    record = {
        "schemaVersion": "e02.s07.outcome_access_validation.v1",
        "researchStepId": "S07",
        "s06BankSha256": sha256_file(S06_BANK),
        "s06StructuralLockSha256": sha256_file(S06_LOCK),
        "s06PlacementClassCounts": dict(sorted(classes.items())),
        "s06StructuralRows": len(structural),
        "s06SearchDerivedRows": len(search),
        "structuralRowsOutcomeBlindAndEligible": all(
            item["confirmatoryEligible"] and item["outcomeAccess"] == "none"
            for item in structural
        ),
        "searchRowsExploratoryAndIneligible": all(
            not item["confirmatoryEligible"]
            and item["outcomeAccess"] == "exploratory_training_only"
            and item["analysisRole"] == "exploratory_search_derived"
            for item in search
        ),
        "scenarioRows": len(scenarios),
        "scenarioRowsHaveNoMapAssignment": all(
            item["faultMapAssignmentStatus"]
            == "unassigned_S08_outcome_blind_only"
            for item in scenarios
        ),
        "scenarioRowsAllowOnlyStructuralClasses": all(
            tuple(item["confirmatoryEligiblePlacementClasses"])
            == STRUCTURAL_PLACEMENT_CLASSES
            and item["forbiddenConfirmatoryPlacementClass"]
            == FORBIDDEN_CONFIRMATORY_PLACEMENT_CLASS
            and item["outcomeAccess"] == "none"
            for item in scenarios
        ),
        "faultMapAssignmentOwner": "S08",
    }
    record["success"] = all(
        record[key]
        for key in (
            "structuralRowsOutcomeBlindAndEligible",
            "searchRowsExploratoryAndIneligible",
            "scenarioRowsHaveNoMapAssignment",
            "scenarioRowsAllowOnlyStructuralClasses",
        )
    )
    write_json(output / "outcome_access_validation.json", record)
    return record


def reconstruction_validation(output: Path) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    parquet = pq.ParquetFile(output / "scenario_extension.parquet")
    for row_group in range(parquet.num_row_groups):
        group = parquet.read_row_group(row_group).to_pylist()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            grouped[row["split"]].append(row)
        for split in sorted(grouped):
            cell = sorted(grouped[split], key=lambda item: item["replicateOrdinal"])
            selected.extend((cell[0], cell[len(cell) // 2], cell[-1]))
    mismatches = []
    for row in selected:
        try:
            scenario = scenario_from_record(row)
            if scenario.input_scenario_id != row["inputScenarioId"]:
                mismatches.append({"inputScenarioId": row["inputScenarioId"], "error": "ID"})
        except Exception as error:  # validation record retains exact reason
            mismatches.append(
                {"inputScenarioId": row["inputScenarioId"], "error": repr(error)}
            )
    record = {
        "schemaVersion": "e02.s07.reconstruction_validation.v1",
        "researchStepId": "S07",
        "sampling": "first, middle, and last replicate in every split/scale/value/order cell",
        "rowsReconstructed": len(selected),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "success": not mismatches,
    }
    write_json(output / "reconstruction_validation.json", record)
    return record


def _runtime_scenario(input_scenario: InputScenario, max_activations: int) -> Scenario:
    cells = tuple(
        Cell(
            f"cell-{index:04d}",
            value,
            Policy.BUBBLE,
            Direction.ASCENDING,
            FaultMode.NORMAL,
        )
        for index, value in enumerate(input_scenario.values_by_identity)
    )
    return Scenario.create(
        cells,
        initial_occupancy=input_scenario.initial_occupancy,
        seed=input_scenario.scenario_seed,
        max_activations=max_activations,
        generation_key=f"S07/runtime/{input_scenario.input_scenario_id}",
        fault_placement="explicit",
    )


def runtime_forecast(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = pq.read_table(
        output / "scenario_extension.parquet",
        filters=[
            ("split", "=", ScenarioSplit.RUNTIME_VALIDATION.value),
            ("valueProfile", "=", ValueProfile.UNIQUE.value),
            ("orderStructure", "=", OrderStructure.REVERSE.value),
            ("replicateOrdinal", "=", 0),
        ],
    ).to_pylist()
    by_n = {item["n"]: scenario_from_record(item) for item in records}
    rows: list[dict[str, Any]] = []
    replay_failures: list[dict[str, Any]] = []
    for n in SIZES:
        for family in SchedulerFamily:
            calibration_budget = (
                min(n, 64)
                if family == SchedulerFamily.FAIR_ADVERSARIAL
                else min(4 * n, 512)
            )
            scenario = _runtime_scenario(by_n[n], calibration_budget)
            architecture = ArchitectureExecutionContract.distributed_local()
            scheduler = SchedulerExecutionContract(family)
            fault = FaultExecutionContract()
            timings = []
            outputs = []
            summaries = []
            for _ in range(2):
                started = time.perf_counter()
                run = run_faulted_architecture(
                    scenario,
                    architecture,
                    scheduler,
                    fault,
                    trace_mode="none",
                )
                timings.append(time.perf_counter() - started)
                outputs.append(run.to_json_bytes())
                summaries.append(run.result.summary)
            if outputs[0] != outputs[1]:
                replay_failures.append({"n": n, "scheduler": family.value})
            activations = int(summaries[0]["ledger"]["activations"])
            seconds = statistics.median(timings)
            seconds_per_opportunity = seconds / max(activations, 1)
            frozen_budget = 100 * n * n
            projected = seconds_per_opportunity * frozen_budget
            rows.append(
                {
                    "n": n,
                    "scheduler": family.value,
                    "calibrationInput": "runtime_validation_unique_reverse_replicate0",
                    "calibrationOpportunitiesRequested": calibration_budget,
                    "calibrationOpportunitiesExecuted": activations,
                    "calibrationStopReason": summaries[0]["stopReason"],
                    "timedRepetitions": 2,
                    "medianCalibrationSeconds": seconds,
                    "secondsPerOpportunity": seconds_per_opportunity,
                    "frozenBudgetOpportunities": frozen_budget,
                    "projectedWorstCaseSecondsPerRun": projected,
                    "projectedWorstCaseMinutesPerRun": projected / 60,
                    "projected250PairCellCpuHours": projected * 500 / 3600,
                    "complexityCaveat": (
                        "clear reference fair-adversarial scheduler performs O(n^2) history/lookahead work per opportunity"
                        if family == SchedulerFamily.FAIR_ADVERSARIAL
                        else "linear extrapolation at fixed n; completion may occur before the ceiling"
                    ),
                    "byteExactReplay": outputs[0] == outputs[1],
                    "opportunityValidationPass": all(run.opportunity_validation().values()),
                }
            )
    write_csv(output / "runtime_forecast.csv", rows)
    summary = {
        "schemaVersion": "e02.s07.runtime_forecast.v1",
        "researchStepId": "S07",
        "scope": "implementation forecast, not scientific outcome evidence or a scheduler comparison",
        "host": platform.node(),
        "python": platform.python_version(),
        "cpuWorkersForScenarioGeneration": 8,
        "runtimeCalibrationExecution": "serial to avoid host-load and nested parallelism confounding",
        "calibrationRows": len(rows),
        "exactReplayRows": sum(item["byteExactReplay"] for item in rows),
        "opportunityValidationRows": sum(
            item["opportunityValidationPass"] for item in rows
        ),
        "replayFailures": replay_failures,
        "maximumProjectedWorstCaseMinutesPerRun": max(
            item["projectedWorstCaseMinutesPerRun"] for item in rows
        ),
        "n500TracePolicy": {
            "nominalRate": TRACE_RATES[500],
            "confirmatoryTracesPerFactorialCell": trace_selection_count(
                ScenarioSplit.CONFIRMATORY_HOLDOUT, 500
            ),
            "screeningTracesPerFactorialCell": trace_selection_count(
                ScenarioSplit.SCREENING_POOL, 500
            ),
            "summaryRowsRequired": "all runs",
            "failureAndAnomalyTraceOverride": True,
        },
        "interpretation": "The ceiling forecast is deliberately conservative. S08 may pair all candidates, but S10 must benchmark selected cells and cannot assume that the current clear fair-adversarial implementation is affordable at n=500 without a transition-equivalent optimization.",
        "success": not replay_failures
        and all(item["opportunityValidationPass"] for item in rows),
    }
    write_json(output / "runtime_forecast.json", summary)
    return rows, summary


def validation_summary(
    output: Path,
    generation: Sequence[Mapping[str, Any]],
    balance: Sequence[Mapping[str, Any]],
    holdout: Mapping[str, Any],
    target: Mapping[str, Any],
    metric: Mapping[str, Any],
    access: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = pq.read_metadata(output / "scenario_extension.parquet")
    table = pq.read_table(
        output / "scenario_extension.parquet",
        columns=["inputScenarioId", "traceSelected", "n", "valueProfile", "orderStructure"],
    )
    rows = table.to_pylist()
    checks = {
        "scenarioCount": metadata.num_rows == 56_430,
        "factorialCellCount": len(generation) == 45,
        "allPlannedSizes": {item["n"] for item in rows} == set(SIZES),
        "allValueProfiles": {item["valueProfile"] for item in rows}
        == {item.value for item in ValueProfile},
        "allOrderStructures": {item["orderStructure"] for item in rows}
        == {item.value for item in OrderStructure},
        "balance": all(item["balanced"] and item["traceQuotaMatched"] for item in balance),
        "uniqueInputScenarioIds": len({item["inputScenarioId"] for item in rows})
        == len(rows),
        "holdoutIntegrity": bool(holdout["success"]),
        "targetFeasibility": bool(target["success"]),
        "duplicateDistance": bool(metric["success"]),
        "outcomeAccess": bool(access["success"]),
        "reconstruction": bool(reconstruction["success"]),
        "runtime": runtime is None or bool(runtime["success"]),
    }
    return {
        "schemaVersion": "e02.s07.validation_summary.v1",
        "researchStepId": "S07",
        "scenarioRows": len(rows),
        "factorialCells": len(generation),
        "splitCounts": dict(
            sorted(
                Counter(
                    item["split"]
                    for item in pq.read_table(
                        output / "scenario_extension.parquet", columns=["split"]
                    ).to_pylist()
                ).items()
            )
        ),
        "sizeCounts": dict(sorted(Counter(item["n"] for item in rows).items())),
        "traceSelectedRows": sum(item["traceSelected"] for item in rows),
        "maximumGenerationAttempt": max(
            item["maximumGenerationAttempt"] for item in generation
        ),
        "generationRejections": sum(
            item["collisionOrRangeRejections"] for item in generation
        ),
        "checks": checks,
        "validationResult": "PASS" if all(checks.values()) else "FAIL",
        "success": all(checks.values()),
    }


def provenance(output: Path, prespecification: Path, workers: int) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schemaVersion": "e02.s07.provenance.v1",
        "researchStepId": "S07",
        "repository": str(REPOSITORY),
        "sourceCommitBeforeS07": git_commit,
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "generator": "scripts/build_scale_input_suite.py",
        "scenarioModule": "causal_simulator/scenario_extensions.py",
        "prespecificationPath": str(prespecification),
        "prespecificationSha256": sha256_file(prespecification),
        "masterSeedDecimal": str(MASTER_SEED),
        "masterSeedHex": hex(MASTER_SEED),
        "workers": workers,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "previousArtifact": {
            "experimentId": "39214707-79fc-48a6-b102-5c706bc1fed3",
            "mount": "/previous-artifacts/E01",
            "releaseClassification": "validated_clean_room_baseline_with_constraining_historical_replication_evidence",
        },
        "s06StructuralLockFileSha256": sha256_file(S06_LOCK),
        "s06PlacementBankFileSha256": sha256_file(S06_BANK),
        "outputDirectory": str(output),
    }


def compare_regeneration(output: Path, compare_to: Path) -> dict[str, Any]:
    rows = []
    for name in CORE_REGENERATION_FILES:
        current = sha256_file(output / name)
        prior = sha256_file(compare_to / name)
        rows.append(
            {
                "path": name,
                "currentSha256": current,
                "priorSha256": prior,
                "byteIdentical": current == prior,
            }
        )
    return {
        "schemaVersion": "e02.s07.deterministic_regeneration.v1",
        "researchStepId": "S07",
        "filesCompared": len(rows),
        "filesByteIdentical": sum(item["byteIdentical"] for item in rows),
        "comparisons": rows,
        "success": all(item["byteIdentical"] for item in rows),
    }


def build(
    output: Path,
    scratch: Path,
    prespecification: Path,
    workers: int,
    compare_to: Path | None,
    skip_runtime: bool,
    reuse_scenario_tables: bool,
) -> None:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    if sha256_file(prespecification) != PRESPECIFICATION_SHA256:
        raise ValueError("S07 prespecification hash mismatch")
    output.mkdir(parents=True, exist_ok=True)
    if reuse_scenario_tables:
        generation = generation_summaries_from_output(output)
    else:
        scratch.mkdir(parents=True, exist_ok=False)
        generation = build_scenario_tables(output, scratch, workers)
    scenario_schema = pq.ParquetFile(
        output / "scenario_extension.parquet"
    ).schema_arrow
    write_json(output / "scenario_extension_schema.json", schema_record(scenario_schema))
    balance, _ = balance_and_disorder(output)
    _, holdout = split_and_holdout_audit(output)
    target, metric = target_and_metric_validation(output)
    access = outcome_access_validation(output)
    reconstruction = reconstruction_validation(output)
    runtime_summary = None
    if not skip_runtime:
        _, runtime_summary = runtime_forecast(output)
    summary = validation_summary(
        output,
        generation,
        balance,
        holdout,
        target,
        metric,
        access,
        reconstruction,
        runtime_summary,
    )
    write_json(output / "validation_summary.json", summary)
    write_json(output / "provenance_manifest.json", provenance(output, prespecification, workers))
    if compare_to is not None:
        comparison = compare_regeneration(output, compare_to)
        write_json(output / "deterministic_regeneration.json", comparison)
        if not comparison["success"]:
            raise AssertionError("independent scenario-suite build was not byte-identical")
    if not summary["success"]:
        raise AssertionError("S07 validation gate failed")
    print(
        json.dumps(
            {
                "researchStepId": "S07",
                "scenarioRows": summary["scenarioRows"],
                "factorialCells": summary["factorialCells"],
                "traceSelectedRows": summary["traceSelectedRows"],
                "validationResult": summary["validationResult"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--prespecification",
        type=Path,
        default=Path(
            "/artifacts/research_steps/S07/scenario_suite_prespecification.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--reuse-scenario-tables", action="store_true")
    args = parser.parse_args()
    build(
        args.output,
        args.scratch,
        args.prespecification,
        args.workers,
        args.compare_to,
        args.skip_runtime,
        args.reuse_scenario_tables,
    )


if __name__ == "__main__":
    main()
