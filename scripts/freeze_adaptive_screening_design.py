#!/usr/bin/env python3
"""Freeze the S10 outcome-blind fractional/sequential screening design."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from reference_simulator.model import canonical_json_bytes, sha256_json


STEP = "S10"
PRESPEC_FILES = (
    "adaptive_screening_prespecification.json",
    "sparse_effect_model.json",
    "s11_selection_rule.json",
)
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
CANDIDATE_ESTIMANDS = (
    "E02-S01-E01",
    "E02-S01-E03",
    "E02-S01-E04",
    "E02-S01-E06",
    "E02-S01-E08",
    "E02-S01-E09",
    "E02-S01-E10",
    "E02-S01-E11",
    "E02-S01-E12",
)
MODIFIERS = (
    "log2n",
    "balanced_duplicate",
    "uneven_duplicate",
    "reverse",
    "block_scrambled",
    "Insertion",
    "Selection",
    "descending",
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


def _balanced_permutation(
    labels: Sequence[Any], total: int, *, address: str
) -> list[Any]:
    base, extra = divmod(total, len(labels))
    instances: list[tuple[str, Any]] = []
    extra_order = sorted(labels, key=lambda item: _hash_rank(address, "extra", item))
    extra_set = set(extra_order[:extra])
    for label in labels:
        count = base + int(label in extra_set)
        for ordinal in range(count):
            instances.append(
                (_hash_rank(address, "instance", label, ordinal), label)
            )
    return [label for _, label in sorted(instances)]


def select_pairing_blocks(pairing_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select 250 screening blocks without reading any outcome column."""

    rows = [
        dict(row)
        for row in pairing_rows
        if row["split"] == "screening_pool"
        and not row["protected"]
        and not row["protectedOutcomeOpened"]
        and not row["searchDerivedPlacementAssigned"]
    ]
    by_input: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    input_meta: dict[str, dict[str, Any]] = {}
    candidates: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        input_id = str(row["inputScenarioId"])
        pd = (str(row["policyProfile"]), str(row["direction"]))
        by_input[input_id][pd] = row
        input_meta.setdefault(input_id, row)
    for row in input_meta.values():
        key = (
            int(row["n"]),
            str(row["valueProfile"]),
            str(row["orderStructure"]),
            str(row["faultProfileId"]),
        )
        candidates[key].append(row)
    for key in candidates:
        candidates[key].sort(
            key=lambda row: (
                _hash_rank("S10", "input", row["inputScenarioId"]),
                row["inputScenarioId"],
            )
        )

    selected: list[dict[str, Any]] = []
    used_inputs: set[str] = set()
    for n in SCALES:
        cells = [(value, order) for value in VALUE_PROFILES for order in ORDER_STRUCTURES]
        extra_cells = sorted(
            cells, key=lambda cell: _hash_rank("S10", "extra-cell", n, *cell)
        )[:5]
        slot_cells: list[tuple[int, str, str]] = []
        for stage in range(1, 6):
            rotation = int(_hash_rank("S10", "cell-rotation", n, stage)[:8], 16) % 9
            rotated = cells[rotation:] + cells[:rotation]
            slot_cells.extend((stage, value, order) for value, order in rotated)
            slot_cells.append((stage, *extra_cells[stage - 1]))
        faults = _balanced_permutation(
            FAULT_PROFILES, 50, address=f"S10/fault/{n}"
        )
        policy_directions = _balanced_permutation(
            POLICY_DIRECTIONS, 50, address=f"S10/policy-direction/{n}"
        )
        for local_index, ((stage, value, order), fault, pd) in enumerate(
            zip(slot_cells, faults, policy_directions, strict=True)
        ):
            pool = candidates[(n, value, order, fault)]
            chosen = next(
                (row for row in pool if str(row["inputScenarioId"]) not in used_inputs),
                None,
            )
            if chosen is None:
                raise AssertionError("outcome-blind slot has no unused eligible input")
            input_id = str(chosen["inputScenarioId"])
            used_inputs.add(input_id)
            pairing = dict(by_input[input_id][pd])
            pairing["screeningStage"] = stage
            pairing["screeningCumulativePairCount"] = stage * 50
            pairing["screeningScaleOrdinal"] = local_index
            selected.append(pairing)

    # Frozen five-fold CV assignment, exactly ten blocks per scale and fold.
    by_scale: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_scale[int(row["n"])].append(row)
    for n, items in by_scale.items():
        ranked = sorted(
            items,
            key=lambda row: (
                _hash_rank("S10", "cv-fold", n, row["pairingBlockId"]),
                row["pairingBlockId"],
            ),
        )
        for index, row in enumerate(ranked):
            row["screeningCvFold"] = index % 5
    return sorted(selected, key=lambda row: (row["screeningStage"], row["n"], row["pairingBlockId"]))


def _treatment_catalog(catalog: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_signature: dict[str, dict[str, Any]] = {}
    mapping: list[dict[str, Any]] = []
    for contrast in catalog["contrasts"]:
        if contrast["status"] != "paired_executable":
            continue
        for arm in contrast["arms"]:
            if not arm["executable"]:
                continue
            settings = dict(arm["settings"])
            signature = sha256_json(settings)
            by_signature.setdefault(
                signature,
                {
                    "treatmentSignature": signature,
                    "settingsJson": canonical_json_bytes(settings).decode("utf-8"),
                    **settings,
                },
            )
            mapping.append(
                {
                    "estimandId": contrast["estimandId"],
                    "estimandTitle": contrast["title"],
                    "rngPairingStatus": contrast["rngPairingStatus"],
                    "armId": arm["armId"],
                    "armLabel": arm["armLabel"],
                    "contrastRole": arm["contrastRole"],
                    "treatmentSignature": signature,
                }
            )
    return [by_signature[key] for key in sorted(by_signature)], mapping


def build_design(
    selected: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    treatments, catalog_map = _treatment_catalog(catalog)
    linked: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in catalog_map:
        linked[item["treatmentSignature"]].append(
            {
                "estimandId": item["estimandId"],
                "armId": item["armId"],
                "contrastRole": item["contrastRole"],
            }
        )
    design_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for block in selected:
        for treatment in treatments:
            run_content = {
                "schemaVersion": "e02.s10.run_design.v1",
                "pairingBlockId": block["pairingBlockId"],
                "treatmentSignature": treatment["treatmentSignature"],
            }
            run_id = f"s10r1:{sha256_json(run_content)}"
            design_rows.append(
                {
                    "schemaVersion": "e02.s10.screening_design.v1",
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
                    "screeningStage": block["screeningStage"],
                    "screeningCumulativePairCount": block["screeningCumulativePairCount"],
                    "screeningCvFold": block["screeningCvFold"],
                    "treatmentSignature": treatment["treatmentSignature"],
                    "settingsJson": treatment["settingsJson"],
                    "architecture": treatment["architecture"],
                    "coordinatorProfile": treatment["coordinatorProfile"],
                    "scheduler": treatment["scheduler"],
                    "mobility": treatment["mobility"],
                    "continuation": treatment["continuation"],
                    "retry": treatment["retry"],
                    "actionFailure": treatment["actionFailure"],
                    "sensing": treatment["sensing"],
                    "informationPermission": treatment["informationPermission"],
                    "legalPrimitives": treatment["legalPrimitives"],
                    "proposalCandidatesPerOpportunity": treatment["proposalCandidatesPerOpportunity"],
                    "linkedEstimandArmsJson": canonical_json_bytes(
                        sorted(linked[treatment["treatmentSignature"]], key=lambda item: (item["estimandId"], item["armId"]))
                    ).decode("utf-8"),
                    "outcomeAccessBeforeFreeze": "none",
                }
            )
        for item in catalog_map:
            run_id = f"s10r1:{sha256_json({'schemaVersion': 'e02.s10.run_design.v1', 'pairingBlockId': block['pairingBlockId'], 'treatmentSignature': item['treatmentSignature']})}"
            contrast_rows.append(
                {
                    "schemaVersion": "e02.s10.screening_contrast_map.v1",
                    "pairingBlockId": block["pairingBlockId"],
                    "screeningStage": block["screeningStage"],
                    **item,
                    "runDesignId": run_id,
                }
            )
    return design_rows, contrast_rows


def build_sparse_design_matrix(selected: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int, int, float]:
    raw: list[list[float]] = []
    for row in selected:
        raw.append(
            [
                math.log2(int(row["n"])),
                float(row["valueProfile"] == "balanced_duplicate"),
                float(row["valueProfile"] == "uneven_duplicate"),
                float(row["orderStructure"] == "reverse"),
                float(row["orderStructure"] == "block_scrambled"),
                float(row["policyProfile"] == "Insertion"),
                float(row["policyProfile"] == "Selection"),
                float(row["direction"] == "descending"),
            ]
        )
    array = np.asarray(raw, dtype=float)
    means = array.mean(axis=0)
    scales = array.std(axis=0, ddof=0)
    if np.any(scales == 0):
        raise AssertionError("frozen design has a constant modifier")
    standardized = (array - means) / scales
    columns = [
        f"{estimand}__{term}"
        for estimand in CANDIDATE_ESTIMANDS
        for term in ("main", *MODIFIERS)
    ]
    output: list[dict[str, Any]] = []
    matrix_rows: list[list[float]] = []
    for estimand_index, estimand in enumerate(CANDIDATE_ESTIMANDS):
        for block_index, block in enumerate(selected):
            values = np.zeros(len(columns), dtype=float)
            start = estimand_index * (len(MODIFIERS) + 1)
            values[start] = 1.0
            values[start + 1 : start + 1 + len(MODIFIERS)] = standardized[block_index]
            row = {
                "schemaVersion": "e02.s10.sparse_design_matrix.v1",
                "estimandId": estimand,
                "pairingBlockId": block["pairingBlockId"],
                "screeningStage": block["screeningStage"],
                "screeningCvFold": block["screeningCvFold"],
                "n": block["n"],
                **{name: float(value) for name, value in zip(columns, values, strict=True)},
            }
            output.append(row)
            matrix_rows.append(values.tolist())
    matrix = np.asarray(matrix_rows, dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    singular = np.linalg.svd(matrix, compute_uv=False)
    condition = float(singular[0] / singular[-1])
    return output, matrix.shape[1], rank, condition


def _spread(values: Iterable[int]) -> int:
    frozen = list(values)
    return max(frozen) - min(frozen)


def validate_design(
    selected: Sequence[Mapping[str, Any]],
    design: Sequence[Mapping[str, Any]],
    contrast_map: Sequence[Mapping[str, Any]],
    *,
    matrix_columns: int,
    matrix_rank: int,
    condition_number: float,
) -> dict[str, Any]:
    scale_counts = Counter(int(row["n"]) for row in selected)
    structural = Counter(
        (int(row["n"]), row["valueProfile"], row["orderStructure"])
        for row in selected
    )
    policy_direction = Counter(
        (int(row["n"]), row["policyProfile"], row["direction"])
        for row in selected
    )
    faults = Counter((int(row["n"]), row["faultProfileId"]) for row in selected)
    stage_scale = Counter((int(row["screeningStage"]), int(row["n"])) for row in selected)
    stage_structural = Counter(
        (int(row["screeningStage"]), int(row["n"]), row["valueProfile"], row["orderStructure"])
        for row in selected
    )
    estimand_role_counts = Counter(
        (row["estimandId"], row["contrastRole"]) for row in contrast_map
    )
    treatment_counts = Counter(row["treatmentSignature"] for row in design)
    checks = {
        "selectedPairingBlockCount": len(selected) == 250,
        "selectedPairingBlocksUnique": len({row["pairingBlockId"] for row in selected}) == 250,
        "selectedInputScenariosUnique": len({row["inputScenarioId"] for row in selected}) == 250,
        "screeningOnly": all(row["split"] == "screening_pool" and not row["protected"] for row in selected),
        "zeroProtectedOutcomeAccess": all(not row["protectedOutcomeOpened"] for row in selected),
        "zeroSearchDerivedAssignments": all(not row["searchDerivedPlacementAssigned"] for row in selected),
        "scaleBalance": set(scale_counts) == set(SCALES) and _spread(scale_counts.values()) == 0,
        "structuralBalance": len(structural) == 45 and _spread(structural.values()) <= 1,
        "policyDirectionBalance": len(policy_direction) == 30 and max(
            _spread([policy_direction[(n, *pd)] for pd in POLICY_DIRECTIONS]) for n in SCALES
        ) <= 1,
        "faultProfileBalance": len(faults) == 60 and max(
            _spread([faults[(n, profile)] for profile in FAULT_PROFILES]) for n in SCALES
        ) <= 1,
        "everyStageEveryScale": len(stage_scale) == 25 and set(stage_scale.values()) == {10},
        "everyStageEveryStructuralCell": len(stage_structural) == 225 and min(stage_structural.values()) >= 1,
        "uniqueRunDesignRows": len(design) == 3500 and len({row["runDesignId"] for row in design}) == 3500,
        "fourteenTreatmentSignatures": len(treatment_counts) == 14 and set(treatment_counts.values()) == {250},
        "contrastMapAccounting": len(contrast_map) == 7500,
        "pairCompleteness": all(count == 250 for count in estimand_role_counts.values()) and len(estimand_role_counts) == 30,
        "e05Explicit": any(row["estimandId"] == "E02-S01-E05" for row in contrast_map),
        "e07Unexecuted": not any(row["estimandId"] == "E02-S01-E07" for row in contrast_map),
        "designMatrixFullRank": matrix_columns == matrix_rank,
        "finiteConditionNumber": math.isfinite(condition_number),
    }
    return {
        "schemaVersion": "e02.s10.design_validation.v1",
        "researchStepId": STEP,
        "success": all(checks.values()),
        "validationResult": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "selectedPairingBlocks": len(selected),
            "runDesignRows": len(design),
            "contrastMapRows": len(contrast_map),
            "treatmentSignatures": len(treatment_counts),
            "sparseDesignRows": len(selected) * len(CANDIDATE_ESTIMANDS),
            "sparseDesignColumns": matrix_columns,
            "sparseDesignRank": matrix_rank,
        },
        "designConditionNumber": condition_number,
        "scaleCounts": {str(key): value for key, value in sorted(scale_counts.items())},
        "structuralCellCountRange": [min(structural.values()), max(structural.values())],
        "policyDirectionCountRangeWithinScale": [
            min(policy_direction.values()), max(policy_direction.values())
        ],
        "faultProfileCountRangeWithinScale": [min(faults.values()), max(faults.values())],
        "protectedOutcomeReads": 0,
        "searchDerivedAssignments": 0,
    }


def freeze(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    prespec_hashes: dict[str, str] = {}
    for name in PRESPEC_FILES:
        source = args.design_dir / name
        target = args.output / name
        shutil.copyfile(source, target)
        prespec_hashes[name] = sha256_file(target)

    pairing_rows = pq.read_table(
        args.pairing_manifest,
        filters=[("split", "=", "screening_pool")],
    ).to_pylist()
    selected = select_pairing_blocks(pairing_rows)
    catalog = json.loads(args.arm_catalog.read_text())
    design, contrast_map = build_design(selected, catalog)
    matrix, columns, rank, condition = build_sparse_design_matrix(selected)
    validation = validate_design(
        selected,
        design,
        contrast_map,
        matrix_columns=columns,
        matrix_rank=rank,
        condition_number=condition,
    )
    if not validation["success"]:
        raise AssertionError(json.dumps(validation, indent=2, sort_keys=True))

    write_parquet(args.output / "screening_pairing_blocks.parquet", selected)
    write_parquet(args.output / "screening_design.parquet", design)
    write_parquet(args.output / "screening_contrast_map.parquet", contrast_map)
    write_parquet(args.output / "design_matrix.parquet", matrix)
    write_json(args.output / "design_validation.json", validation)
    core = [
        *PRESPEC_FILES,
        "screening_pairing_blocks.parquet",
        "screening_design.parquet",
        "screening_contrast_map.parquet",
        "design_matrix.parquet",
        "design_validation.json",
    ]
    manifest = {
        "schemaVersion": "e02.s10.design_freeze_manifest.v1",
        "researchStepId": STEP,
        "frozenBeforeOutcomeAccess": True,
        "outcomeRowsReadBeforeFreeze": 0,
        "protectedOutcomeReads": 0,
        "sourceHashes": {
            "pairingManifest": sha256_file(args.pairing_manifest),
            "armCatalog": sha256_file(args.arm_catalog),
            "scenarioBankMetadataOnlyNotOpened": sha256_file(args.scenario_bank),
            **prespec_hashes,
        },
        "files": [
            {
                "path": name,
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256_file(args.output / name),
            }
            for name in core
        ],
    }
    manifest["designFreezeSha256"] = sha256_json(manifest)
    write_json(args.output / "design_freeze_manifest.json", manifest)
    print(json.dumps({"validation": validation, "designFreezeSha256": manifest["designFreezeSha256"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairing-manifest", type=Path, default=Path("/artifacts/research_steps/S08/pairing_manifest.parquet"))
    parser.add_argument("--scenario-bank", type=Path, default=Path("/artifacts/research_steps/S07/scenario_extension.parquet"))
    parser.add_argument("--arm-catalog", type=Path, default=Path("/artifacts/research_steps/S08/contrast_arm_catalog.json"))
    parser.add_argument("--design-dir", type=Path, default=REPOSITORY / "design" / "s10")
    parser.add_argument("--output", type=Path, default=Path("/artifacts/research_steps/S10"))
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
