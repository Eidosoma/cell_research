"""Build and validate the normalized, immutable E01 S08 scenario bank."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import jsonschema
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .core import (
    BANK_SCHEMA_VERSION,
    BASE_DRAW_JSON_SCHEMA,
    BASE_DRAW_SCHEMA_VERSION,
    CONDITION_JSON_SCHEMA,
    CONDITION_SCHEMA_VERSION,
    FROZEN_PUBLIC_COMMIT,
    SCENARIO_ROW_JSON_SCHEMA,
    SEED_DERIVATION_VERSION,
    SPLITS,
    ConditionSpec,
    build_condition_catalog,
    iter_base_draws,
    materialize_scenario,
    seed_specification,
)


CORE_FILES = (
    "paired_scenario_bank.parquet",
    "base_draw_bank.parquet",
    "condition_catalog.parquet",
    "condition_catalog.csv",
    "claim_scenario_coverage.parquet",
    "claim_scenario_coverage.csv",
    "base_draw_schema.json",
    "condition_schema.json",
    "scenario_bank_schema.json",
    "seed_specification.json",
    "split_manifest.json",
    "distribution_summary.csv",
    "distribution_summary.json",
    "pairing_validation.json",
    "holdout_integrity.json",
    "validation_summary.json",
)


SCENARIO_ARROW_SCHEMA = pa.schema(
    [
        ("schemaVersion", pa.string()),
        ("scenarioId", pa.string()),
        ("conditionId", pa.string()),
        ("conditionFamily", pa.string()),
        ("profileRole", pa.string()),
        ("conditionConfigSha256", pa.string()),
        ("baseDrawId", pa.string()),
        ("pairingBlockId", pa.string()),
        ("inputProfile", pa.string()),
        ("n", pa.int16()),
        ("split", pa.string()),
        ("protected", pa.bool_()),
        ("replicateOrdinal", pa.int32()),
        ("architecture", pa.string()),
        ("policySet", pa.list_(pa.string())),
        ("assignmentProfile", pa.string()),
        ("directionProfile", pa.string()),
        ("faultMode", pa.string()),
        ("requestedFaultCount", pa.int8()),
        ("realizedFaultCount", pa.int8()),
        ("placementProfile", pa.string()),
        ("faultDrawIndices", pa.list_(pa.int16())),
        ("faultDistinctIndices", pa.list_(pa.int16())),
        ("faultMapId", pa.string()),
        ("runtimeSeed", pa.string()),
        ("assignmentSeed", pa.string()),
        ("faultSeed", pa.string()),
        ("generationKey", pa.string()),
        ("maxActivations", pa.int32()),
        ("scenarioContentSha256", pa.string()),
        ("scenarioJsonSha256", pa.string()),
        ("initialValuesSha256", pa.string()),
        ("initialOccupancySha256", pa.string()),
        ("policyAssignmentSha256", pa.string()),
        ("directionAssignmentSha256", pa.string()),
        ("analysisLabelAssignmentSha256", pa.string()),
        ("faultAssignmentSha256", pa.string()),
        ("compositionCountsJson", pa.string()),
        ("directionCountsJson", pa.string()),
        ("backendEligibility", pa.string()),
        ("historicalRandomStreamStatus", pa.string()),
        ("runtimeCouplingProfile", pa.string()),
        ("referenceRngProfile", pa.string()),
        ("claimIds", pa.list_(pa.string())),
        ("configSchemaVersion", pa.string()),
    ],
    metadata={b"schemaVersion": BANK_SCHEMA_VERSION.encode("ascii")},
)


BASE_ARROW_SCHEMA = pa.schema(
    [
        ("schemaVersion", pa.string()),
        ("baseDrawId", pa.string()),
        ("pairingBlockId", pa.string()),
        ("inputProfile", pa.string()),
        ("split", pa.string()),
        ("protected", pa.bool_()),
        ("replicateOrdinal", pa.int32()),
        ("n", pa.int16()),
        ("initialOccupancySeed", pa.string()),
        ("valuesById", pa.list_(pa.int16())),
        ("initialOccupancyIndices", pa.list_(pa.int16())),
        ("valuesByIdSha256", pa.string()),
        ("initialOccupancySha256", pa.string()),
        ("initialValueSequenceSha256", pa.string()),
    ],
    metadata={b"schemaVersion": BASE_DRAW_SCHEMA_VERSION.encode("ascii")},
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema | None = None) -> None:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table, path, compression="zstd", compression_level=9,
        row_group_size=8192, version="2.6", use_dictionary=True,
        write_statistics=True,
    )


def _components(value: str | None) -> set[str]:
    if not value:
        return set()
    return {name for name in ("Bubble", "Insertion", "Selection") if name in value}


def _claim_condition_mapping(
    claims: Sequence[Mapping[str, Any]], conditions: Sequence[ConditionSpec]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def select(**criteria: Any) -> list[ConditionSpec]:
        selected = []
        for condition in conditions:
            if all(
                (value(condition) if callable(value) else getattr(condition, key) == value)
                for key, value in criteria.items()
            ):
                selected.append(condition)
        return selected

    for claim in claims:
        claim_id = str(claim["claim_id"])
        figure = int(claim["figure"])
        group = str(claim["claim_group"])
        kind = str(claim["claim_kind"])
        algorithm = claim.get("algorithm")
        policies = _components(algorithm)
        selected: list[ConditionSpec] = []
        disposition = "direct_condition_coverage"
        note = "Condition dimensions are explicit; paper outcome remains a later replication target."

        if claim_id in {"F06-A-CONCEPTUAL-CONTEXT", "F06-D-DG-DEFINITION", "F08-C-AGGREGATION-DEFINITION"}:
            disposition = "no_scenario_required"
            note = "Conceptual or metric-definition row; no simulator initial condition is implied."
        elif group == "fault_semantics":
            mode = str(claim.get("fault_type"))
            selected = [item for item in conditions if item.family == "unique_fault" and item.fault_mode == mode]
            disposition = "definition_covered_by_fault_family"
        elif figure in {3, 4}:
            selected = [
                item for item in conditions
                if item.family == "unique_pure" and set(item.policies) == policies
            ]
        elif figure == 5:
            mode = str(claim.get("fault_type"))
            count = claim.get("fault_count")
            architecture = str(claim.get("architecture"))
            selected = [
                item for item in conditions
                if item.family == "unique_fault"
                and item.fault_mode == mode
                and (count is None or item.requested_fault_count == int(count))
                and (architecture not in {"cell_view", "traditional"} or item.architecture == architecture)
                and (not policies or bool(set(item.policies) & policies))
            ]
            note = "Paper placement is unreported; corrected primary and exact-rule legacy sensitivity conditions are both retained."
        elif claim_id == "F06-B-FROZEN-CELL-EXAMPLE":
            selected = [item for item in conditions if item.family == "worked_example_ambiguity_envelope"]
            disposition = "bounded_ambiguity_envelope"
            note = "Paper algorithm and exact input order are unreported; all 720 orders for each of three policies are preserved without choosing one."
        elif claim_id == "F06-C-MULTIPLE-LOCAL-DROPS":
            selected = [
                item for item in conditions
                if item.family == "worked_example_ambiguity_envelope"
                or (item.family == "unique_fault" and item.architecture == "cell_view" and item.fault_mode == "stuck")
            ]
            disposition = "bounded_condition_envelope"
            note = "Caption omits algorithm, size, count, and placement; the declared worked and n=100 stuck envelopes preserve those uncertainties."
        elif figure == 7:
            count = claim.get("fault_count")
            architecture = str(claim.get("architecture"))
            if count is not None and int(count) == 0:
                selected = [
                    item for item in conditions if item.family == "unique_pure"
                    and (not policies or bool(set(item.policies) & policies))
                    and (architecture not in {"cell_view", "traditional"} or item.architecture == architecture)
                ]
            else:
                selected = [
                    item for item in conditions if item.family == "unique_fault"
                    and item.fault_mode == "stuck"
                    and (count is None or item.requested_fault_count == int(count))
                    and (not policies or bool(set(item.policies) & policies))
                    and (architecture not in {"cell_view", "traditional"} or item.architecture == architecture)
                ]
            note = "f=0 reuses the immutable no-fault pair; f>0 retains both placement profiles."
        elif group == "aggregation_control":
            code_map = {
                "F08-A-CONTROL-BUBBLE-INSERTION": {"Bubble", "Insertion"},
                "F08-A-CONTROL-BUBBLE-SELECTION": {"Bubble", "Selection"},
                "F08-A-CONTROL-INSERTION-SELECTION": {"Insertion", "Selection"},
            }
            wanted = code_map[claim_id]
            selected = [
                item for item in conditions
                if item.family == "identical_policy_label_control"
                and all(name in item.analysis_label_profile for name in wanted)
            ]
            disposition = "clean_room_negative_control"
            note = "All policies are Bubble; immutable 50:50 ghost labels resolve the paper control contradiction without attribution."
        elif group in {"chimeric_completion", "aggregation", "chimeric_efficiency"} and figure == 8:
            if kind == "metric_definition":
                disposition = "no_scenario_required"
                note = "Metric definition applies downstream; no unique initial condition is implied."
            elif policies and len(policies) == 1:
                selected = [
                    item for item in conditions
                    if item.family == "unique_pure" and item.architecture == "cell_view"
                    and set(item.policies) == policies
                ]
            elif claim_id == "F08-A-UNIQUE-START-END-BASELINE":
                selected = [
                    item for item in conditions
                    if item.family == "same_direction_unique"
                ]
                disposition = "summary_over_all_unique_chimeras"
            elif claim_id == "F08-B-LINEAR-EFFICIENCY":
                selected = [
                    item for item in conditions
                    if (
                        item.family == "same_direction_unique"
                        and len(item.policies) == 2
                    )
                    or (
                        item.family == "unique_pure"
                        and item.architecture == "cell_view"
                    )
                ]
                disposition = "summary_over_pairwise_and_pure_conditions"
            else:
                selected = [
                    item for item in conditions
                    if item.family == "same_direction_unique"
                    and (not policies or set(item.policies) == policies)
                ]
            note = "Balanced exact assignment is primary; independent random assignment is retained as a sensitivity profile."
        elif group == "duplicate_value_aggregation":
            selected = [
                item for item in conditions
                if item.family == "same_direction_repeated"
                and (not policies or set(item.policies) == policies)
            ]
            note = "Exactly ten copies of values 1..10; exact and independent-random Algotype profiles remain separate."
        elif group == "opposite_direction_chimeras":
            if claim_id in {
                "F09-ABC-AGGREGATION-RISE",
                "F09-ABC-STARTING-SORTEDNESS",
                "F09-ABC-WINNER-ORDER",
            }:
                selected = [
                    item for item in conditions
                    if item.family == "opposite_unique"
                ]
                disposition = "summary_over_all_opposing_pairwise_conditions"
            else:
                selected = [
                    item for item in conditions
                    if item.family == "opposite_unique"
                    and (not policies or set(item.policies) == policies)
                ]
            note = "N=100 in paper_scale is a clean-room declaration because the historical sample size is unreported."
        elif group == "opposite_direction_duplicate_chimeras":
            selected = [
                item for item in conditions
                if item.family == "opposite_repeated"
                and (not policies or set(item.policies) == policies)
            ]
            note = "Figure 10 numeric targets remain unavailable; scenarios do not substitute regenerated numbers for missing evidence."
        else:
            raise AssertionError(f"no mapping rule for {claim_id}")

        if not selected and disposition != "no_scenario_required":
            raise AssertionError(f"claim {claim_id} has no scenario condition")
        if disposition == "no_scenario_required":
            rows.append({
                "claimId": claim_id, "figure": figure, "panel": str(claim["panel"]),
                "claimGroup": group, "claimKind": kind,
                "paperArraySize": claim.get("array_size"),
                "paperRepetitionCount": claim.get("repetition_count"),
                "coverageDisposition": disposition, "conditionId": None,
                "conditionFamily": None, "profileRole": "not_applicable",
                "scenarioN": None, "historicalSampleSizeStatus": "not_applicable",
                "note": note,
            })
            continue
        for condition in sorted(set(selected), key=lambda item: item.condition_id):
            n = 6 if condition.input_profile == "worked_1_6_exhaustive" else 100
            rows.append({
                "claimId": claim_id, "figure": figure, "panel": str(claim["panel"]),
                "claimGroup": group, "claimKind": kind,
                "paperArraySize": claim.get("array_size"),
                "paperRepetitionCount": claim.get("repetition_count"),
                "coverageDisposition": disposition, "conditionId": condition.condition_id,
                "conditionFamily": condition.family, "profileRole": condition.profile_role,
                "scenarioN": n,
                "historicalSampleSizeStatus": (
                    "reported_N_100" if claim.get("repetition_count") == 100
                    else "unreported_not_inferred"
                ),
                "note": note,
            })
    return rows


def _scenario_row(
    condition: ConditionSpec,
    base: Mapping[str, Any],
    claim_ids: Sequence[str],
    condition_hash: str,
) -> dict[str, Any]:
    scenario, metadata = materialize_scenario(condition, base)
    fault_map_id = "fm1:" + _hash_json({
        "inputProfile": condition.input_profile,
        "split": base["split"],
        "replicateOrdinal": int(base["replicateOrdinal"]),
        "placementProfile": condition.placement_profile,
        "requestedFaultCount": condition.requested_fault_count,
        "drawIndices": metadata["faultDrawIndices"],
        "distinctIdentityIndices": metadata["faultDistinctIndices"],
    })
    return {
        "schemaVersion": BANK_SCHEMA_VERSION,
        "scenarioId": scenario.scenario_id,
        "conditionId": condition.condition_id,
        "conditionFamily": condition.family,
        "profileRole": condition.profile_role,
        "conditionConfigSha256": condition_hash,
        "baseDrawId": base["baseDrawId"],
        "pairingBlockId": base["pairingBlockId"],
        "inputProfile": condition.input_profile,
        "n": int(base["n"]),
        "split": base["split"],
        "protected": bool(base["protected"]),
        "replicateOrdinal": int(base["replicateOrdinal"]),
        "architecture": condition.architecture,
        "policySet": list(condition.policies),
        "assignmentProfile": condition.assignment_profile,
        "directionProfile": condition.direction_profile,
        "faultMode": condition.fault_mode,
        "requestedFaultCount": condition.requested_fault_count,
        "realizedFaultCount": scenario.realized_fault_count,
        "placementProfile": condition.placement_profile,
        "faultDrawIndices": metadata["faultDrawIndices"],
        "faultDistinctIndices": metadata["faultDistinctIndices"],
        "faultMapId": fault_map_id,
        "runtimeSeed": metadata["runtimeSeed"],
        "assignmentSeed": metadata["assignmentSeed"],
        "faultSeed": metadata["faultSeed"],
        "generationKey": scenario.generation_key,
        "maxActivations": scenario.max_activations,
        "scenarioContentSha256": scenario.scenario_id.removeprefix("r1:"),
        "scenarioJsonSha256": metadata["scenarioJsonSha256"],
        "initialValuesSha256": base["valuesByIdSha256"],
        "initialOccupancySha256": base["initialOccupancySha256"],
        "policyAssignmentSha256": metadata["policyAssignmentSha256"],
        "directionAssignmentSha256": metadata["directionAssignmentSha256"],
        "analysisLabelAssignmentSha256": metadata["analysisLabelAssignmentSha256"],
        "faultAssignmentSha256": metadata["faultAssignmentSha256"],
        "compositionCountsJson": json.dumps(metadata["compositionCounts"], sort_keys=True, separators=(",", ":")),
        "directionCountsJson": json.dumps(metadata["directionCounts"], sort_keys=True, separators=(",", ":")),
        "backendEligibility": condition.historical_eligibility,
        "historicalRandomStreamStatus": "unavailable_not_invented",
        "runtimeCouplingProfile": "shared_numeric_seed_by_base; runtime_draws_scenario_id_separated",
        "referenceRngProfile": scenario.rng_profile,
        "claimIds": list(claim_ids),
        "configSchemaVersion": CONDITION_SCHEMA_VERSION,
    }


def _distribution_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = (
        "split", "inputProfile", "conditionFamily", "profileRole", "architecture",
        "assignmentProfile", "directionProfile", "faultMode", "placementProfile",
        "requestedFaultCount", "realizedFaultCount", "backendEligibility",
    )
    for dimension in dimensions:
        counts = frame[dimension].astype(str).value_counts().sort_index()
        for level, count in counts.items():
            rows.append({"dimension": dimension, "level": level, "scenarioCount": int(count)})
    return rows


def validate_bank(
    output: Path, claims: Sequence[Mapping[str, Any]], conditions: Sequence[ConditionSpec]
) -> dict[str, Any]:
    scenario = pq.read_table(output / "paired_scenario_bank.parquet").to_pandas()
    base = pq.read_table(output / "base_draw_bank.parquet").to_pandas()
    coverage = pq.read_table(output / "claim_scenario_coverage.parquet").to_pandas()
    expected_main = sum(item.count for item in SPLITS) * 107
    expected_total = expected_main + 720 * 3
    errors: list[str] = []

    def check(value: bool, message: str) -> None:
        if not value:
            errors.append(message)

    check(len(scenario) == expected_total, f"scenario row count != {expected_total}")
    check(scenario.scenarioId.nunique() == len(scenario), "scenario IDs are not unique")
    check(scenario.conditionId.nunique() == 110, "condition coverage is not 110")
    check(len(base) == 3920 and base.baseDrawId.nunique() == 3920, "base draw count/IDs invalid")
    check(set(scenario.historicalRandomStreamStatus) == {"unavailable_not_invented"}, "historical streams were invented")
    check((scenario.scenarioId.str.slice(3) == scenario.scenarioContentSha256).all(), "scenario ID/content hashes differ")

    for split in SPLITS:
        subset = scenario[scenario.split == split.name]
        check(len(subset) == split.count * 107, f"split {split.name} count invalid")
        check(set(subset.protected) == {split.protected}, f"split {split.name} protection invalid")
        per_condition = subset.groupby("conditionId").size()
        check(len(per_condition) == 107 and (per_condition == split.count).all(), f"split {split.name} incomplete conditions")
    envelope = scenario[scenario.split == "ambiguity_envelope"]
    check(len(envelope) == 2160 and envelope.conditionId.nunique() == 3, "worked envelope invalid")

    for profile in ("unique_1_100", "repeated_1_10_x10"):
        subset = base[base.inputProfile == profile]
        check(len(subset) == 1600, f"{profile} base count invalid")
        check(subset.initialValueSequenceSha256.nunique() == 1600, f"{profile} initial permutations repeat")
        check(subset.initialOccupancySha256.nunique() == 1600, f"{profile} occupancy permutations repeat")
        expected_values = (
            list(range(1, 101))
            if profile == "unique_1_100"
            else [value for value in range(1, 11) for _ in range(10)]
        )
        check(
            all(list(row.valuesById) == expected_values for row in subset.itertuples()),
            f"{profile} value distribution invalid",
        )
        check(
            all(sorted(int(value) for value in row.initialOccupancyIndices) == list(range(100)) for row in subset.itertuples()),
            f"{profile} occupancy is not a permutation",
        )
    worked = base[base.inputProfile == "worked_1_6_exhaustive"]
    check(len(worked) == 720 and worked.initialOccupancySha256.nunique() == 720, "worked permutations are not exhaustive")
    check(
        all(list(row.valuesById) == list(range(1, 7)) for row in worked.itertuples()),
        "worked value distribution invalid",
    )
    check(
        all(sorted(int(value) for value in row.initialOccupancyIndices) == list(range(6)) for row in worked.itertuples()),
        "worked occupancy is not a permutation",
    )

    corrected = scenario[scenario.placementProfile == "reference_without_replacement"]
    legacy = scenario[scenario.placementProfile == "legacy_with_replacement"]
    check((corrected.requestedFaultCount == corrected.realizedFaultCount).all(), "corrected placement does not realize exact f")
    check((legacy.realizedFaultCount <= legacy.requestedFaultCount).all(), "legacy placement realizes more than requested")
    check(
        all(len(draws) == requested for draws, requested in zip(legacy.faultDrawIndices, legacy.requestedFaultCount)),
        "legacy placement does not preserve the requested draw count",
    )
    underrealized = int((legacy.realizedFaultCount < legacy.requestedFaultCount).sum())
    check(underrealized > 0, "legacy placement did not exercise under-realization")
    faulted = scenario[scenario.requestedFaultCount > 0].copy()
    faulted["drawsCanonical"] = faulted.faultDrawIndices.map(
        lambda values: json.dumps([int(value) for value in values], separators=(",", ":"))
    )
    shared_map_groups = faulted.groupby(
        ["inputProfile", "split", "replicateOrdinal", "placementProfile", "requestedFaultCount"],
        dropna=False,
    )
    check(
        (shared_map_groups.faultMapId.nunique() == 1).all()
        and (shared_map_groups.drawsCanonical.nunique() == 1).all(),
        "fixed fault maps are not shared across paired policy/architecture/mode conditions",
    )

    main_base = base[base.inputProfile != "worked_1_6_exhaustive"]
    expected_per_base = {"unique_1_100": 95, "repeated_1_10_x10": 12}
    counts = scenario.groupby("baseDrawId").size().to_dict()
    check(all(counts[row.baseDrawId] == expected_per_base[row.inputProfile] for row in main_base.itertuples()), "pairing block completeness failed")
    check(all(counts[row.baseDrawId] == 3 for row in worked.itertuples()), "worked pairing completeness failed")

    exact = scenario[scenario.assignmentProfile == "balanced_exact"]
    exact_counts = exact.compositionCountsJson.map(json.loads)
    check(all(max(item.values()) - min(item.values()) <= 1 for item in exact_counts), "exact compositions are imbalanced")
    random_rows = scenario[scenario.assignmentProfile == "independent_random"]
    random_counts = random_rows.compositionCountsJson.map(json.loads)
    check(all(len(item) == len(policies) for item, policies in zip(random_counts, random_rows.policySet)), "random assignment lost a policy")
    opposed = scenario[scenario.directionProfile == "opposed_by_algotype"]
    check(all(len(json.loads(item)) == 2 for item in opposed.directionCountsJson), "opposed directions are not both represented")

    claim_ids = {str(item["claim_id"]) for item in claims}
    mapped_ids = set(coverage.claimId)
    check(mapped_ids == claim_ids and len(claim_ids) == 118, "claim coverage is not exactly 118 IDs")
    no_scenario = set(coverage.loc[coverage.coverageDisposition == "no_scenario_required", "claimId"])
    check(no_scenario == {"F06-A-CONCEPTUAL-CONTEXT", "F06-D-DG-DEFINITION", "F08-C-AGGREGATION-DEFINITION"}, "no-scenario claim set changed")
    empirical = coverage[coverage.coverageDisposition != "no_scenario_required"]
    check(empirical.conditionId.notna().all(), "empirical claim lacks condition")
    mapped_conditions = set(empirical.conditionId)
    check(mapped_conditions == {item.condition_id for item in conditions}, "condition catalog has unmapped profiles")

    split_base_ids = {
        split.name: set(base.loc[base.split == split.name, "baseDrawId"]) for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            check(not split_base_ids[left.name] & split_base_ids[right.name], f"base leakage {left.name}/{right.name}")
            left_pairs = set(base.loc[base.split == left.name, "pairingBlockId"])
            right_pairs = set(base.loc[base.split == right.name, "pairingBlockId"])
            check(not left_pairs & right_pairs, f"pairing-block leakage {left.name}/{right.name}")
    initial_hashes = {
        (profile, split.name): set(base.loc[(base.inputProfile == profile) & (base.split == split.name), "initialValueSequenceSha256"])
        for profile in ("unique_1_100", "repeated_1_10_x10") for split in SPLITS
    }
    for profile in ("unique_1_100", "repeated_1_10_x10"):
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                check(not initial_hashes[(profile, left.name)] & initial_hashes[(profile, right.name)], f"initial-state leakage {profile} {left.name}/{right.name}")

    # JSON Schema validates all condition records and a deterministic scenario sample;
    # Arrow enforces the full table's column types.
    condition_rows = pq.read_table(output / "condition_catalog.parquet").to_pylist()
    for row in condition_rows:
        jsonschema.Draft202012Validator(CONDITION_JSON_SCHEMA).validate(json.loads(row["conditionJson"]))
    base_validator = jsonschema.Draft202012Validator(BASE_DRAW_JSON_SCHEMA)
    base_source_rows = pq.read_table(output / "base_draw_bank.parquet").to_pylist()
    for row in base_source_rows:
        base_validator.validate(row)
    sample_indices = sorted(set([0, len(scenario) - 1] + list(range(0, len(scenario), 173))))
    validator = jsonschema.Draft202012Validator(SCENARIO_ROW_JSON_SCHEMA)
    source_rows = pq.read_table(output / "paired_scenario_bank.parquet").take(pa.array(sample_indices)).to_pylist()
    for row in source_rows:
        validator.validate(row)
    condition_by_id = {item.condition_id: item for item in conditions}
    base_by_id = {row["baseDrawId"]: row for row in pq.read_table(output / "base_draw_bank.parquet").to_pylist()}
    for row in source_rows:
        replayed = _scenario_row(
            condition_by_id[row["conditionId"]],
            base_by_id[row["baseDrawId"]],
            row["claimIds"],
            row["conditionConfigSha256"],
        )
        check(replayed == row, f"normalized scenario replay differs for {row['scenarioId']}")

    pairing = {
        "success": not errors,
        "pairingBlockCount": int(base.pairingBlockId.nunique()),
        "uniqueInputScenariosPerBlock": 95,
        "repeatedInputScenariosPerBlock": 12,
        "workedEnvelopeScenariosPerBlock": 3,
        "faultMapSharedAcross": ["architecture", "policy", "fault_mode"],
        "initialStateSharedAcross": ["architecture", "policy", "composition", "direction", "fault_profile"],
        "runtimeCoupling": "same numeric seed per base draw; S05 runtime stream still incorporates scenario ID, so activation draws are not claimed coupled across condition IDs",
        "historicalCoupling": "none; historical random stream is unavailable",
        "errors": errors,
    }
    _write_json(output / "pairing_validation.json", pairing)
    holdout = {
        "success": not errors,
        "protectedSplits": [item.name for item in SPLITS if item.protected],
        "disjointBaseDrawIds": True,
        "disjointInitialValueSequencesWithinInputProfile": True,
        "protectedScenarioCount": int(scenario.loc[scenario.protected].shape[0]),
        "policySearchHoldoutUse": next(item.allowed_use for item in SPLITS if item.name == "policy_search_holdout"),
        "confirmatoryUse": next(item.allowed_use for item in SPLITS if item.name == "confirmatory_holdout"),
        "leakageChecks": len(SPLITS) * (len(SPLITS) - 1) // 2 * 4,
        "errors": errors,
    }
    _write_json(output / "holdout_integrity.json", holdout)
    summary = {
        "researchStepId": "S08",
        "success": not errors,
        "validationResult": "PASS" if not errors else "FAIL",
        "scenarioCount": len(scenario),
        "conditionCount": scenario.conditionId.nunique(),
        "baseDrawCount": len(base),
        "claimCount": len(claim_ids),
        "claimCoverageRowCount": len(coverage),
        "unmappedClaimCount": len(claim_ids - mapped_ids),
        "splitCounts": {name: int(count) for name, count in scenario.split.value_counts().sort_index().items()},
        "inputProfileCounts": {name: int(count) for name, count in scenario.inputProfile.value_counts().sort_index().items()},
        "correctedFaultScenarioCount": len(corrected),
        "legacyFaultScenarioCount": len(legacy),
        "legacyUnderRealizedScenarioCount": underrealized,
        "exactCompositionScenarioCount": len(exact),
        "independentRandomCompositionScenarioCount": len(random_rows),
        "opposingDirectionScenarioCount": len(opposed),
        "protectedScenarioCount": int(scenario.protected.sum()),
        "jsonSchemaScenarioSamplesValidated": len(sample_indices),
        "jsonSchemaConditionsValidated": len(condition_rows),
        "jsonSchemaBaseDrawsValidated": len(base_source_rows),
        "normalizedScenarioSamplesReplayed": len(source_rows),
        "historicalStreamsInvented": 0,
        "errors": errors,
    }
    _write_json(output / "validation_summary.json", summary)
    distribution = _distribution_rows(scenario)
    _write_csv(output / "distribution_summary.csv", distribution, ("dimension", "level", "scenarioCount"))
    _write_json(output / "distribution_summary.json", {"researchStepId": "S08", "rows": distribution})
    if errors:
        raise AssertionError("; ".join(errors))
    return summary


def _source_manifest(repository: Path) -> dict[str, Any]:
    paths = (
        "reference_simulator/model.py",
        "scenario_bank/__init__.py",
        "scenario_bank/core.py",
        "scenario_bank/builder.py",
        "scripts/build_scenario_bank.py",
        "tests/test_scenario_bank.py",
        "tests/test_reference_simulator.py",
    )
    return {
        "researchStepId": "S08",
        "repositoryRoot": str(repository),
        "repositoryCommit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "repositoryBranch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip(),
        "repositoryDirty": bool(subprocess.check_output(["git", "status", "--short"], cwd=repository, text=True).strip()),
        "files": [
            {"path": path, "bytes": (repository / path).stat().st_size, "sha256": _sha256_file(repository / path)}
            for path in paths
        ],
        "sourceStorage": "Git repository; source is not duplicated into $ARTIFACTS_DIR.",
    }


def compare_core_outputs(output: Path, prior: Path) -> dict[str, Any]:
    rows = []
    for name in CORE_FILES:
        current_hash = _sha256_file(output / name)
        prior_hash = _sha256_file(prior / name)
        rows.append({
            "path": name,
            "currentSha256": current_hash,
            "priorSha256": prior_hash,
            "identical": current_hash == prior_hash,
        })
    return {
        "researchStepId": "S08",
        "success": all(item["identical"] for item in rows),
        "comparisonDirectory": str(prior),
        "filesCompared": len(rows),
        "rows": rows,
    }


def build_bank(
    claim_registry: Path,
    output: Path,
    *,
    repository: Path,
    stable_output: Path | None = None,
    compare_to: Path | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    claims = pq.read_table(claim_registry).to_pylist()
    conditions = build_condition_catalog()
    for condition in conditions:
        jsonschema.Draft202012Validator(CONDITION_JSON_SCHEMA).validate(condition.to_dict())
    mapping = _claim_condition_mapping(claims, conditions)
    claims_by_condition: dict[str, set[str]] = defaultdict(set)
    for row in mapping:
        if row["conditionId"] is not None:
            claims_by_condition[row["conditionId"]].add(row["claimId"])

    base_rows = list(iter_base_draws())
    base_rows.sort(key=lambda item: (item["inputProfile"], item["split"], item["replicateOrdinal"]))
    _write_table(output / "base_draw_bank.parquet", base_rows, BASE_ARROW_SCHEMA)
    base_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        base_by_profile[row["inputProfile"]].append(row)

    condition_rows = []
    condition_hashes: dict[str, str] = {}
    for condition in conditions:
        condition_dict = condition.to_dict()
        condition_hash = _hash_json(condition_dict)
        condition_hashes[condition.condition_id] = condition_hash
        condition_rows.append({
            **condition_dict,
            "conditionJson": _json_bytes(condition_dict).decode("utf-8"),
            "conditionJsonSha256": condition_hash,
            "claimIds": sorted(claims_by_condition[condition.condition_id]),
            "claimCount": len(claims_by_condition[condition.condition_id]),
        })
    _write_table(output / "condition_catalog.parquet", condition_rows)
    csv_condition_rows = [
        {
            "conditionId": row["conditionId"], "family": row["family"],
            "inputProfile": row["inputProfile"], "architecture": row["architecture"],
            "policies": "|".join(row["policies"]), "assignmentProfile": row["assignmentProfile"],
            "directionProfile": row["directionProfile"], "faultMode": row["faultMode"],
            "requestedFaultCount": row["requestedFaultCount"], "placementProfile": row["placementProfile"],
            "profileRole": row["profileRole"], "maxActivations": row["maxActivations"],
            "historicalEligibility": row["historicalEligibility"], "claimCount": row["claimCount"],
            "conditionJsonSha256": row["conditionJsonSha256"],
        }
        for row in condition_rows
    ]
    _write_csv(output / "condition_catalog.csv", csv_condition_rows, tuple(csv_condition_rows[0]))
    _write_table(output / "claim_scenario_coverage.parquet", mapping)
    csv_mapping = [
        {key: ("" if value is None else value) for key, value in row.items()} for row in mapping
    ]
    _write_csv(output / "claim_scenario_coverage.csv", csv_mapping, tuple(csv_mapping[0]))

    scenario_path = output / "paired_scenario_bank.parquet"
    writer = pq.ParquetWriter(
        scenario_path, SCENARIO_ARROW_SCHEMA, compression="zstd", compression_level=9,
        version="2.6", use_dictionary=True, write_statistics=True,
    )
    buffer: list[dict[str, Any]] = []
    split_counts = Counter()
    split_digests: dict[str, Any] = defaultdict(hashlib.sha256)
    try:
        for condition in conditions:
            for base in base_by_profile[condition.input_profile]:
                row = _scenario_row(
                    condition, base, sorted(claims_by_condition[condition.condition_id]),
                    condition_hashes[condition.condition_id],
                )
                buffer.append(row)
                split_counts[row["split"]] += 1
                split_digests[row["split"]].update((row["scenarioId"] + "\n").encode("ascii"))
                if len(buffer) == 8192:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=SCENARIO_ARROW_SCHEMA), row_group_size=8192)
                    buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCENARIO_ARROW_SCHEMA), row_group_size=8192)
    finally:
        writer.close()

    _write_json(output / "base_draw_schema.json", BASE_DRAW_JSON_SCHEMA)
    _write_json(output / "condition_schema.json", CONDITION_JSON_SCHEMA)
    _write_json(output / "scenario_bank_schema.json", SCENARIO_ROW_JSON_SCHEMA)
    _write_json(output / "seed_specification.json", seed_specification())
    _write_json(output / "split_manifest.json", {
        "schemaVersion": "e01.s08.split_manifest.v1",
        "researchStepId": "S08",
        "splits": [
            {
                "name": item.name,
                "replicateCountPerCondition": item.count,
                "conditionCount": 107,
                "scenarioCount": split_counts[item.name],
                "protected": item.protected,
                "allowedUse": item.allowed_use,
                "historicalSampleSizeBasis": item.historical_sample_size_basis,
                "orderedScenarioIdSha256": split_digests[item.name].hexdigest(),
            }
            for item in SPLITS
        ] + [{
            "name": "ambiguity_envelope", "replicateCountPerCondition": 720,
            "conditionCount": 3, "scenarioCount": split_counts["ambiguity_envelope"],
            "protected": False, "allowedUse": "coverage of the underspecified Figure 6 worked condition only; excluded from population inference",
            "historicalSampleSizeBasis": "exhaustive clean-room envelope, not a recovered paper sample",
            "orderedScenarioIdSha256": split_digests["ambiguity_envelope"].hexdigest(),
        }],
        "splitAssignment": "semantic split name plus zero-based local replicate ordinal; no random post-generation split",
        "crossSplitOverlapAllowed": False,
    })

    summary = validate_bank(output, claims, conditions)
    _write_json(output / "environment_provenance.json", {
        "researchStepId": "S08",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpuCountVisible": os.cpu_count(),
        "workerCount": 1,
        "parallelism": "intentional serial deterministic generation; no simulation was run",
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "jsonschema": package_version("jsonschema"),
        "seedDerivationVersion": SEED_DERIVATION_VERSION,
        "newDependenciesInstalled": [],
    })
    _write_json(output / "source_package_manifest.json", _source_manifest(repository))
    if compare_to is not None:
        comparison = compare_core_outputs(output, compare_to)
        _write_json(output / "deterministic_regeneration.json", comparison)
        if not comparison["success"]:
            raise AssertionError("deterministic regeneration differs")
    if stable_output is not None:
        stable_output.parent.mkdir(parents=True, exist_ok=True)
        if stable_output.exists() or stable_output.is_symlink():
            stable_output.unlink()
        try:
            os.link(scenario_path, stable_output)
            stable_mode = "hardlink_same_inode"
        except OSError:
            shutil.copy2(scenario_path, stable_output)
            stable_mode = "byte_identical_copy_cross_filesystem"
        _write_json(stable_output.parent / "scenario_bank_index.json", {
            "schemaVersion": BANK_SCHEMA_VERSION,
            "canonicalPath": str(scenario_path),
            "stablePath": str(stable_output),
            "linkMode": stable_mode,
            "bytes": stable_output.stat().st_size,
            "sha256": _sha256_file(stable_output),
            "supportingDirectory": str(output),
        })
    return summary
