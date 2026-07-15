#!/usr/bin/env python3
"""Freeze S11's protected, outcome-blind nested confirmatory design."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes, sha256_json


STEP = "S11"
SCALES = (20, 50, 100, 200, 500)
VALUE_PROFILES = ("unique", "balanced_duplicate", "uneven_duplicate")
ORDER_STRUCTURES = ("nearly_sorted", "reverse", "block_scrambled")
POLICY_DIRECTIONS = tuple(
    (policy, direction)
    for policy in ("Bubble", "Insertion", "Selection")
    for direction in ("ascending", "descending")
)
FAULT_PROFILES = tuple(
    f"{placement}:f{count}"
    for placement in (
        "uniform_exact",
        "clustered_exact",
        "boundary_exact",
        "median_rank_exact",
    )
    for count in (2, 4, 6)
)
EXPECTED_SELECTIONS = (
    "E02-S01-E04",
    "E02-S01-E06",
    "E02-S01-E03",
    "E02-S01-E08",
)
SETTING_FIELDS = (
    "architecture",
    "coordinatorProfile",
    "scheduler",
    "mobility",
    "continuation",
    "retry",
    "actionFailure",
    "sensing",
    "informationPermission",
    "legalPrimitives",
    "proposalCandidatesPerOpportunity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    )


def _hash_rank(*parts: object) -> str:
    return hashlib.sha256("/".join(map(str, parts)).encode("utf-8")).hexdigest()


def _balanced_permutation(labels: Sequence[Any], total: int, *, address: str) -> list[Any]:
    base, extra = divmod(total, len(labels))
    extra_order = sorted(labels, key=lambda item: _hash_rank(address, "extra", item))
    extra_set = set(extra_order[:extra])
    instances: list[tuple[str, Any]] = []
    for label in labels:
        for ordinal in range(base + int(label in extra_set)):
            instances.append((_hash_rank(address, "instance", label, ordinal), label))
    return [label for _, label in sorted(instances)]


def _spread(values: Iterable[int]) -> int:
    frozen = list(values)
    return max(frozen) - min(frozen)


def select_pairing_blocks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose the full nested 5,000-block design without outcome fields."""

    forbidden_outcome_names = {
        "successByBudget", "normalizedResidualError", "completionOpportunity",
        "projection_s01UnitWeightFullCost", "stopReason", "finalValues",
    }
    if any(forbidden_outcome_names.intersection(row) for row in rows[:1]):
        raise AssertionError("pairing input unexpectedly contains protected outcome fields")
    eligible = [
        dict(row)
        for row in rows
        if row["split"] == "confirmatory_holdout"
        and bool(row["protected"])
        and not bool(row["protectedOutcomeOpened"])
        and bool(row["faultMapConfirmatoryEligible"])
        and not bool(row["searchDerivedPlacementAssigned"])
        and row["outcomeAccess"] == "none"
    ]
    by_input: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    input_meta: dict[str, dict[str, Any]] = {}
    for row in eligible:
        input_id = str(row["inputScenarioId"])
        by_input[input_id][(str(row["policyProfile"]), str(row["direction"]))] = row
        input_meta.setdefault(input_id, row)
    if any(set(items) != set(POLICY_DIRECTIONS) for items in by_input.values()):
        raise AssertionError("an eligible input lacks complete policy-by-direction support")

    candidates: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in input_meta.values():
        candidates[(
            int(row["n"]), str(row["valueProfile"]),
            str(row["orderStructure"]), str(row["faultProfileId"]),
        )].append(row)
    for key in candidates:
        candidates[key].sort(key=lambda row: (
            _hash_rank("S11", "protected-input", row["inputScenarioId"]),
            row["inputScenarioId"],
        ))

    selected: list[dict[str, Any]] = []
    used_inputs: set[str] = set()
    structural_labels = tuple(
        (value, order) for value in VALUE_PROFILES for order in ORDER_STRUCTURES
    )
    for look in range(1, 6):
        for n in SCALES:
            structures = _balanced_permutation(
                structural_labels, 200, address=f"S11/look-{look}/structure/{n}"
            )
            faults = _balanced_permutation(
                FAULT_PROFILES, 200, address=f"S11/look-{look}/fault/{n}"
            )
            policy_directions = _balanced_permutation(
                POLICY_DIRECTIONS, 200, address=f"S11/look-{look}/policy-direction/{n}"
            )
            for scale_ordinal, ((value, order), fault, policy_direction) in enumerate(
                zip(structures, faults, policy_directions, strict=True)
            ):
                pool = candidates[(n, value, order, fault)]
                chosen = next(
                    (row for row in pool if str(row["inputScenarioId"]) not in used_inputs),
                    None,
                )
                if chosen is None:
                    raise AssertionError(f"no unused protected input for {(look, n, value, order, fault)}")
                input_id = str(chosen["inputScenarioId"])
                used_inputs.add(input_id)
                pairing = dict(by_input[input_id][policy_direction])
                pairing["confirmatoryLook"] = look
                pairing["confirmatoryCumulativePairCount"] = look * 1000
                pairing["confirmatoryScaleOrdinal"] = scale_ordinal
                selected.append(pairing)
    return sorted(selected, key=lambda row: (
        row["confirmatoryLook"], row["n"], row["pairingBlockId"]
    ))


def treatment_catalog(
    catalog: Mapping[str, Any], selections: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_signature: dict[str, dict[str, Any]] = {}
    mapping: list[dict[str, Any]] = []
    contrasts = {item["estimandId"]: item for item in catalog["contrasts"]}
    for estimand in selections:
        contrast = contrasts[estimand]
        if contrast["status"] != "paired_executable":
            raise AssertionError(f"selected contrast is not executable: {estimand}")
        for arm in contrast["arms"]:
            if not arm["executable"]:
                raise AssertionError(f"selected arm is not executable: {estimand}")
            settings = dict(arm["settings"])
            signature = sha256_json(settings)
            by_signature.setdefault(signature, {
                "treatmentSignature": signature,
                "settingsJson": canonical_json_bytes(settings).decode("utf-8"),
                **settings,
            })
            mapping.append({
                "estimandId": estimand,
                "estimandTitle": contrast["title"],
                "rngPairingStatus": contrast["rngPairingStatus"],
                "armId": arm["armId"],
                "armLabel": arm["armLabel"],
                "contrastRole": arm["contrastRole"],
                "treatmentSignature": signature,
            })
    return [by_signature[key] for key in sorted(by_signature)], mapping


def build_design(
    selected: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    treatments, arm_map = treatment_catalog(catalog, EXPECTED_SELECTIONS)
    if len(treatments) != 6 or len(arm_map) != 8:
        raise AssertionError("the four frozen contrasts must map to six signatures/eight arms")
    linked: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in arm_map:
        linked[item["treatmentSignature"]].append({
            "estimandId": item["estimandId"],
            "armId": item["armId"],
            "contrastRole": item["contrastRole"],
        })
    design: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for block in selected:
        for treatment in treatments:
            content = {
                "schemaVersion": "e02.s11.run_design.v1",
                "pairingBlockId": block["pairingBlockId"],
                "treatmentSignature": treatment["treatmentSignature"],
            }
            run_id = f"s11r1:{sha256_json(content)}"
            design.append({
                "schemaVersion": "e02.s11.confirmatory_design.v1",
                "researchStepId": STEP,
                "runDesignId": run_id,
                "pairingBlockId": block["pairingBlockId"],
                "inputScenarioId": block["inputScenarioId"],
                "split": block["split"],
                "protected": block["protected"],
                "n": block["n"],
                "valueProfile": block["valueProfile"],
                "orderStructure": block["orderStructure"],
                "policyProfile": block["policyProfile"],
                "direction": block["direction"],
                "faultProfileId": block["faultProfileId"],
                "faultMapId": block["faultMapId"],
                "placementClass": block["placementClass"],
                "faultCount": block["faultCount"],
                "confirmatoryLook": block["confirmatoryLook"],
                "confirmatoryCumulativePairCount": block["confirmatoryCumulativePairCount"],
                "treatmentSignature": treatment["treatmentSignature"],
                "settingsJson": treatment["settingsJson"],
                **{field: treatment[field] for field in SETTING_FIELDS},
                "linkedEstimandArmsJson": canonical_json_bytes(sorted(
                    linked[treatment["treatmentSignature"]],
                    key=lambda item: (item["estimandId"], item["armId"]),
                )).decode("utf-8"),
                "protectedOutcomeAccessAuthorizedAfterDesignFreeze": True,
            })
        for item in arm_map:
            content = {
                "schemaVersion": "e02.s11.run_design.v1",
                "pairingBlockId": block["pairingBlockId"],
                "treatmentSignature": item["treatmentSignature"],
            }
            contrasts.append({
                "schemaVersion": "e02.s11.confirmatory_contrast_map.v1",
                "pairingBlockId": block["pairingBlockId"],
                "confirmatoryLook": block["confirmatoryLook"],
                **item,
                "runDesignId": f"s11r1:{sha256_json(content)}",
            })
    return design, contrasts


def validate(
    selected: Sequence[Mapping[str, Any]],
    design: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["selectedPairingBlocks"] = len(selected) == 5000
    checks["uniquePairingBlocks"] = len({row["pairingBlockId"] for row in selected}) == 5000
    checks["uniqueInputScenarios"] = len({row["inputScenarioId"] for row in selected}) == 5000
    checks["protectedHoldoutOnly"] = all(
        row["split"] == "confirmatory_holdout" and row["protected"] for row in selected
    )
    checks["priorProtectedOutcomesUnopened"] = all(
        not row["protectedOutcomeOpened"] and row["outcomeAccess"] == "none" for row in selected
    )
    checks["confirmatoryFaultMapsOnly"] = all(
        row["faultMapConfirmatoryEligible"] and not row["searchDerivedPlacementAssigned"]
        for row in selected
    )
    scale_look = Counter((row["confirmatoryLook"], int(row["n"])) for row in selected)
    checks["nestedScaleBalance"] = len(scale_look) == 25 and set(scale_look.values()) == {200}
    structure = Counter((
        row["confirmatoryLook"], int(row["n"]), row["valueProfile"], row["orderStructure"]
    ) for row in selected)
    checks["nestedStructureBalance"] = len(structure) == 225 and max(
        _spread(structure[(look, n, value, order)] for value in VALUE_PROFILES for order in ORDER_STRUCTURES)
        for look in range(1, 6) for n in SCALES
    ) <= 1
    policy = Counter((
        row["confirmatoryLook"], int(row["n"]), row["policyProfile"], row["direction"]
    ) for row in selected)
    checks["nestedPolicyDirectionBalance"] = len(policy) == 150 and max(
        _spread(policy[(look, n, *level)] for level in POLICY_DIRECTIONS)
        for look in range(1, 6) for n in SCALES
    ) <= 1
    faults = Counter((
        row["confirmatoryLook"], int(row["n"]), row["faultProfileId"]
    ) for row in selected)
    checks["nestedFaultProfileBalance"] = len(faults) == 300 and max(
        _spread(faults[(look, n, profile)] for profile in FAULT_PROFILES)
        for look in range(1, 6) for n in SCALES
    ) <= 1
    run_counts = Counter(row["treatmentSignature"] for row in design)
    checks["runDesignAccounting"] = (
        len(design) == 30000 and len({row["runDesignId"] for row in design}) == 30000
        and len(run_counts) == 6 and set(run_counts.values()) == {5000}
    )
    arm_counts = Counter((row["estimandId"], row["contrastRole"]) for row in contrasts)
    checks["contrastArmMarginals"] = (
        len(contrasts) == 40000 and len(arm_counts) == 8
        and set(arm_counts.values()) == {5000}
    )
    checks["selectedEstimandsPreserved"] = set(row["estimandId"] for row in contrasts) == set(EXPECTED_SELECTIONS)
    coupling = {row["estimandId"]: row["rngPairingStatus"] for row in contrasts}
    checks["couplingClassificationsPreserved"] = coupling == {
        "E02-S01-E03": "scenario_paired_rng_unpaired_different_stream_consumption",
        "E02-S01-E04": "shared_prefix_until_continuation_stop",
        "E02-S01-E06": "scenario_paired_rng_unpaired_different_scenario_roots",
        "E02-S01-E08": "shared_prefix_same_scheduler_root",
    }
    return {
        "schemaVersion": "e02.s11.design_validation.v1",
        "researchStepId": STEP,
        "checks": checks,
        "selectedPairingBlocks": len(selected),
        "runDesignRows": len(design),
        "contrastMapRows": len(contrasts),
        "success": all(checks.values()),
    }


def freeze(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    specification = json.loads(args.specification.read_text())
    selections = json.loads(args.selections.read_text())
    if specification["selectedEstimandsInFrozenPriorityOrder"] != list(EXPECTED_SELECTIONS):
        raise AssertionError("S11 prespecification selection order changed")
    if selections["selectedEstimands"] != list(EXPECTED_SELECTIONS):
        raise AssertionError("S10 frozen selections changed")
    if selections["selectionSha256"] != specification["inputContracts"]["requiredSelectionSha256"]:
        raise AssertionError("S10 selection content hash changed")
    if selections["confirmatoryOutcomesOpened"] or selections["protectedOutcomeReads"] != 0:
        raise AssertionError("S10 reports prior protected outcome access")

    pairing_table = pq.read_table(args.pairing)
    selected = select_pairing_blocks(pairing_table.to_pylist())
    catalog = json.loads(args.catalog.read_text())
    design, contrasts = build_design(selected, catalog)
    validation = validate(selected, design, contrasts)
    if not validation["success"]:
        raise AssertionError(f"S11 design validation failed: {validation}")

    spec_copy = args.output / "confirmatory_prespecification.json"
    shutil.copy2(args.specification, spec_copy)
    pairing_path = args.output / "confirmatory_pairing_blocks.parquet"
    design_path = args.output / "confirmatory_design.parquet"
    contrast_path = args.output / "confirmatory_contrast_map.parquet"
    validation_path = args.output / "design_validation.json"
    write_parquet(pairing_path, selected)
    write_parquet(design_path, design)
    write_parquet(contrast_path, contrasts)
    write_json(validation_path, validation)

    hashes = {
        "confirmatoryPrespecificationSha256": sha256_file(spec_copy),
        "s10FrozenSelectionsFileSha256": sha256_file(args.selections),
        "s10SelectionContentSha256": selections["selectionSha256"],
        "s08PairingManifestSha256": sha256_file(args.pairing),
        "s08ArmCatalogSha256": sha256_file(args.catalog),
        "confirmatoryPairingBlocksSha256": sha256_file(pairing_path),
        "confirmatoryDesignSha256": sha256_file(design_path),
        "confirmatoryContrastMapSha256": sha256_file(contrast_path),
        "designValidationSha256": sha256_file(validation_path),
        "maximumPairingBlockIdListSha256": sha256_json([
            row["pairingBlockId"] for row in selected
        ]),
        "maximumRunDesignIdListSha256": sha256_json(sorted(
            row["runDesignId"] for row in design
        )),
    }
    manifest = {
        "schemaVersion": "e02.s11.confirmatory_freeze_manifest.v1",
        "researchStepId": STEP,
        "frozenAtUtc": specification["frozenAtUtc"],
        "frozenBeforeProtectedOutcomeAccess": True,
        "protectedOutcomeReadsBeforeFreeze": 0,
        "selectedEstimands": list(EXPECTED_SELECTIONS),
        "primaryEndpoints": specification["primaryEndpoints"],
        "nestedCumulativeLooks": specification["sequentialRule"]["looks"],
        "maximumPairingBlocks": len(selected),
        "maximumRunDesignRows": len(design),
        "hashes": hashes,
        "validationSuccess": True,
        "protectedOutcomeAccessAuthorizedOnlyAfterThisFreeze": True,
    }
    manifest["designFreezeSha256"] = sha256_json(manifest)
    write_json(args.output / "confirmatory_freeze_manifest.json", manifest)
    print(json.dumps({
        "status": "frozen_before_protected_outcome_access",
        "designFreezeSha256": manifest["designFreezeSha256"],
        "selectedBlocks": len(selected),
        "maximumRuns": len(design),
        "validation": validation["success"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=Path, default=Path("design/s11/confirmatory_prespecification.json"))
    parser.add_argument("--selections", type=Path, default=Path("/artifacts/research_steps/S10/s11_frozen_selections.json"))
    parser.add_argument("--pairing", type=Path, default=Path("/artifacts/research_steps/S08/pairing_manifest.parquet"))
    parser.add_argument("--catalog", type=Path, default=Path("/artifacts/research_steps/S08/contrast_arm_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S11"))
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
