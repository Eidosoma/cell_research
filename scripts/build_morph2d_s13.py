#!/usr/bin/env python3
"""Freeze, calibrate, and apply the E06 S13 morphometric panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from morph2d.grammar import load_grammar_catalog
from morph2d.morphometrics import (
    ALL_COST_SOURCE_COLUMNS,
    COST_SOURCE_COLUMNS,
    block_upscale,
    categorical_transport_distance,
    component_counts,
    heterotypic_boundary_edges,
    load_morphometric_catalog,
    occupied_region_purity,
    score_morphology,
    standardize_cost_components,
    standardize_endpoint_time,
    symmetry_mismatch_fraction,
)
from morph2d.targets import (
    Grid,
    exact_equivalence_orbit,
    load_target_catalog,
    vacancy_hole_count,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S13")
CATALOG_PATH = ROOT / "configs/morphologies/morphometric_catalog.yaml"
TARGET_PATH = ROOT / "configs/morphologies/target_catalog.yaml"
GRAMMAR_PATH = ROOT / "configs/morphologies/grammar_catalog.yaml"
ADVERSARIAL_PATH = ROOT / "configs/morphologies/grammar_adversarial_fixtures.yaml"
SOURCE_TABLES = {
    "S09": Path("/artifacts/research_steps/S09/baseline_results.parquet"),
    "S10": Path("/artifacts/research_steps/S10/perturbation_results.parquet"),
    "S11": Path("/artifacts/research_steps/S11/chimera_results.parquet"),
    "S12": Path("/artifacts/research_steps/S12/hybrid_control_results.parquet"),
}
EXPECTED_ROWS = {"S09": 26000, "S10": 39000, "S11": 18000, "S12": 20000}
PRIOR_CLASSIFICATIONS = {
    "S09": "supportive_with_constraining_broad_formation_subfinding",
    "S10": "null",
    "S11": "constraining_contradictory",
    "S12": "constraining_contradictory",
}
CONTEXT_COLUMNS = (
    "phase",
    "split",
    "scenarioId",
    "pairingBlockId",
    "runId",
    "targetId",
    "grammarId",
    "replicate",
    "policyId",
    "startFamily",
    "arm",
    "armId",
    "challengeId",
    "lesionId",
    "severity",
    "mixtureId",
    "interventionArm",
    "absoluteEndpointFamily",
    "eventBudgetTransitions",
    "censored",
    "failed",
    "initialConjunctiveCompletion",
    "formationEligible",
    "repairEligible",
    "lesionRepairByBudget",
    "firstRepairTransition",
    "conjunctiveCompletionByBudget",
    "firstCompletionTransition",
    "terminalConjunctiveCompletion",
    "terminalS01GlobalSuccess",
    "terminalS01GeometryMatch",
    "terminalS01ComponentMatch",
    "terminalS01TopologyMatch",
    "terminalS01MismatchCount",
    "terminalS01MismatchFraction",
    "terminalVacancyHoles",
    "terminalS02GrammarAccepted",
    "terminalS02SoftScore",
    "terminalS02RelationalScore",
    "terminalS02HardViolationCount",
    "finalGridRowsJson",
)

_TARGETS: dict[str, Any] = {}
_GRAMMARS: dict[str, Any] = {}
_ORBITS: dict[str, tuple[Grid, ...]] = {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def swap(grid: Grid, first: Sequence[int], second: Sequence[int]) -> Grid:
    mutable = [list(row) for row in grid]
    first_coordinate = tuple(int(value) for value in first)
    second_coordinate = tuple(int(value) for value in second)
    mutable[first_coordinate[0]][first_coordinate[1]], mutable[second_coordinate[0]][second_coordinate[1]] = (
        mutable[second_coordinate[0]][second_coordinate[1]],
        mutable[first_coordinate[0]][first_coordinate[1]],
    )
    return tuple(tuple(row) for row in mutable)


def metric_specification_text(catalog: Mapping[str, Any]) -> str:
    sections = [
        "# S13 frozen morphometric specification",
        "",
        "This specification was frozen before opening S09–S12 outcome values. It adds diagnostic views and does not redefine the conjunctive S02-local/S01-global endpoint.",
        "",
        "## Equivalence",
        "",
        f"- Image distance: {catalog['equivalenceHandling']['imageDistance']}.",
        f"- Reference selection: {catalog['equivalenceHandling']['referenceSelection']}.",
        f"- Transport anchor: {catalog['equivalenceHandling']['transportAnchor']}.",
        f"- Caveat: {catalog['equivalenceHandling']['caveat']}",
        "",
        "## Metrics",
        "",
    ]
    for name, definition in catalog["metricDefinitions"].items():
        sections.extend(
            [
                f"### {name}",
                "",
                str(definition.get("definition", definition.get("fields", ""))),
                "",
            ]
        )
    sections.extend(
        [
            "## Decision boundary",
            "",
            f"The completion contract remains `{catalog['decisionRules']['completionContract']}`. Prior classifications are locked. Resolution tests above canonical size are diagnostics only; S01/S02 completion is not extrapolated beyond its bounded-square calibration.",
            "",
        ]
    )
    return "\n".join(sections)


def _assets() -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[Grid, ...]]]:
    _, targets = load_target_catalog(TARGET_PATH)
    _, grammars = load_grammar_catalog(GRAMMAR_PATH)
    target_map = {target.target_id: target for target in targets}
    grammar_map = {grammar.target_id: grammar for grammar in grammars}
    orbits = {
        target_id: exact_equivalence_orbit(target)
        for target_id, target in target_map.items()
    }
    return target_map, grammar_map, orbits


def calibration_tables(catalog: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    targets, grammars, orbits = _assets()
    rows: list[dict[str, Any]] = []
    for target_id, target in targets.items():
        for index, grid in enumerate(orbits[target_id]):
            metrics = score_morphology(
                grid, target, grammars[target_id], equivalence_orbit=orbits[target_id]
            )
            rows.append(
                {
                    "fixtureClass": "exact_equivalence",
                    "fixtureId": f"{target_id}__orbit_{index:03d}",
                    **metrics,
                    "expectedS01": True,
                    "expectedS02": True,
                    "expectedMatched": bool(
                        metrics["s01GlobalSuccess"] and metrics["s02GrammarAccepted"]
                    ),
                }
            )
        declared = target.validation.get("acceptedBoundarySwap")
        if declared:
            grid = swap(target.grid, declared[0], declared[1])
            metrics = score_morphology(
                grid, target, grammars[target_id], equivalence_orbit=orbits[target_id]
            )
            rows.append(
                {
                    "fixtureClass": "s01_tolerated_variant",
                    "fixtureId": f"{target_id}__tolerated_interface_exchange",
                    **metrics,
                    "expectedS01": True,
                    "expectedS02": True,
                    "expectedMatched": bool(
                        metrics["s01GlobalSuccess"] and metrics["s02GrammarAccepted"]
                    ),
                }
            )
    adversarial_raw = yaml.safe_load(ADVERSARIAL_PATH.read_text(encoding="utf-8"))
    adversarial_rows = []
    for fixture in adversarial_raw["fixtures"]:
        target_id = str(fixture["targetId"])
        metrics = score_morphology(
            fixture["rows"],
            targets[target_id],
            grammars[target_id],
            equivalence_orbit=orbits[target_id],
        )
        record = {
            "fixtureClass": "s02_grammar_false_positive",
            "fixtureId": fixture["fixtureId"],
            "challengedRelation": fixture["challengedRelation"],
            **metrics,
            "expectedS01": False,
            "expectedS02": True,
            "expectedMatched": bool(
                not metrics["s01GlobalSuccess"] and metrics["s02GrammarAccepted"]
            ),
            "globalFlagCount": int(not metrics["s01GeometryMatch"])
            + int(metrics["componentExcess"] > 0)
            + int(metrics["holeCountError"] > 0)
            + int(metrics["s01BoundaryViolationCount"] > 0),
            "rowsJson": json.dumps(fixture["rows"], separators=(",", ":")),
        }
        rows.append(record)
        adversarial_rows.append(record)
    hand_rows = []
    for fixture in catalog["calibrationFixtures"]["handLabeled"]:
        fixture_grid = tuple(tuple(row) for row in fixture["rows"])
        counts = component_counts(fixture_grid, ["T", fixture["vacancyLabel"]])
        hand_rows.append(
            {
                "fixtureClass": "hand_labeled",
                "fixtureId": fixture["fixtureId"],
                "targetId": None,
                "componentCountsJson": json.dumps(counts, sort_keys=True, separators=(",", ":")),
                "vacancyHoles": vacancy_hole_count(fixture_grid, fixture["vacancyLabel"]),
                "heterotypicBoundaryEdges": heterotypic_boundary_edges(fixture_grid),
                "expectedMatched": bool(
                    counts["T"] == int(fixture["expected"]["TComponents"])
                    and counts[fixture["vacancyLabel"]]
                    == int(fixture["expected"]["vacancyComponents"])
                    and vacancy_hole_count(fixture_grid, fixture["vacancyLabel"])
                    == int(fixture["expected"]["holes"])
                    and heterotypic_boundary_edges(fixture_grid)
                    == int(fixture["expected"]["heterotypicBoundaryEdges"])
                ),
                "rowsJson": json.dumps(fixture["rows"], separators=(",", ":")),
            }
        )
    combined = pd.concat([pd.DataFrame(rows), pd.DataFrame(hand_rows)], ignore_index=True)

    reference = tuple(
        tuple(row)
        for row in ["TTTTT", "T...T", "T...T", "T...T", "TTTTT"]
    )
    candidate = swap(reference, (0, 2), (1, 2))
    resolution_rows = []
    baseline: dict[str, Any] | None = None
    for scale in catalog["resolutionRobustness"]["scales"]:
        scaled_reference = block_upscale(reference, int(scale))
        scaled_candidate = block_upscale(candidate, int(scale))
        site_count = len(scaled_candidate) * len(scaled_candidate[0])
        image = sum(
            scaled_candidate[row][col] != scaled_reference[row][col]
            for row in range(len(scaled_candidate))
            for col in range(len(scaled_candidate[0]))
        ) / site_count
        transport, raw_transport, _ = categorical_transport_distance(
            scaled_candidate, scaled_reference
        )
        symmetry, symmetry_count = symmetry_mismatch_fraction(
            scaled_candidate,
            scaled_reference,
            vacancy_label=".",
            translation_mode="none",
        )
        current = {
            "scale": int(scale),
            "siteCount": site_count,
            "imageMismatchFraction": image,
            "transportDistance": transport,
            "transportRawCost": raw_transport,
            "componentCountsJson": json.dumps(
                component_counts(scaled_candidate, ["T", "."]),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "boundaryLengthNormalized": heterotypic_boundary_edges(scaled_candidate)
            / math.sqrt(site_count),
            "occupiedRegionPurity": occupied_region_purity(
                scaled_candidate, scaled_reference, "."
            ),
            "symmetryMismatchFraction": symmetry,
            "symmetryTransformCount": symmetry_count,
            "vacancyHoles": vacancy_hole_count(scaled_candidate, "."),
        }
        if baseline is None:
            baseline = dict(current)
        current["imageAbsoluteDrift"] = abs(
            current["imageMismatchFraction"] - baseline["imageMismatchFraction"]
        )
        current["transportAbsoluteDrift"] = abs(
            float(current["transportDistance"]) - float(baseline["transportDistance"])
        )
        current["boundaryAbsoluteDrift"] = abs(
            current["boundaryLengthNormalized"] - baseline["boundaryLengthNormalized"]
        )
        current["purityAbsoluteDrift"] = abs(
            float(current["occupiedRegionPurity"])
            - float(baseline["occupiedRegionPurity"])
        )
        current["symmetryAbsoluteDrift"] = abs(
            float(current["symmetryMismatchFraction"])
            - float(baseline["symmetryMismatchFraction"])
        )
        current["componentInvariant"] = (
            current["componentCountsJson"] == baseline["componentCountsJson"]
        )
        current["holeInvariant"] = current["vacancyHoles"] == baseline["vacancyHoles"]
        current["resolutionGatePassed"] = bool(
            current["imageAbsoluteDrift"] == 0
            and current["boundaryAbsoluteDrift"] <= 1e-12
            and current["purityAbsoluteDrift"] == 0
            and current["symmetryAbsoluteDrift"] <= 1e-12
            and current["transportAbsoluteDrift"]
            <= float(catalog["resolutionRobustness"]["transportMaximumAbsoluteDrift"])
            and current["componentInvariant"]
            and current["holeInvariant"]
        )
        resolution_rows.append(current)
    resolution = pd.DataFrame(resolution_rows)
    exact = combined[combined["fixtureClass"] == "exact_equivalence"]
    tolerated = combined[combined["fixtureClass"] == "s01_tolerated_variant"]
    adversarial = combined[
        combined["fixtureClass"] == "s02_grammar_false_positive"
    ]
    hand = combined[combined["fixtureClass"] == "hand_labeled"]
    validation = {
        "schemaVersion": "e06.s13.pre-outcome-validation.v1",
        "researchStepId": "S13",
        "exactFixtureCount": len(exact),
        "exactFixturesPassed": int(exact["expectedMatched"].sum()),
        "toleratedFixtureCount": len(tolerated),
        "toleratedFixturesPassed": int(tolerated["expectedMatched"].sum()),
        "adversarialFixtureCount": len(adversarial),
        "adversarialFixturesPassed": int(adversarial["expectedMatched"].sum()),
        "adversarialFixturesWithGlobalFlag": int((adversarial["globalFlagCount"] > 0).sum()),
        "handFixtureCount": len(hand),
        "handFixturesPassed": int(hand["expectedMatched"].sum()),
        "resolutionScaleCount": len(resolution),
        "resolutionScalesPassed": int(resolution["resolutionGatePassed"].sum()),
    }
    validation["success"] = bool(
        validation["exactFixtureCount"] == 148
        and validation["exactFixturesPassed"] == 148
        and validation["toleratedFixtureCount"] == 3
        and validation["toleratedFixturesPassed"] == 3
        and validation["adversarialFixtureCount"] == 7
        and validation["adversarialFixturesPassed"] == 7
        and validation["adversarialFixturesWithGlobalFlag"] == 7
        and validation["handFixtureCount"] == 4
        and validation["handFixturesPassed"] == 4
        and validation["resolutionScalesPassed"] == 3
    )
    return combined, pd.DataFrame(adversarial_rows), resolution, validation


def source_snapshot() -> list[dict[str, Any]]:
    records = []
    for step_id, path in SOURCE_TABLES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        metadata = pq.ParquetFile(path).metadata
        records.append(
            {
                "stepId": step_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rowCountFromMetadata": metadata.num_rows,
                "schemaSha256": hashlib.sha256(str(pq.read_schema(path)).encode()).hexdigest(),
            }
        )
    return records


def freeze_design(output: Path, *, replace: bool) -> dict[str, Any]:
    if replace and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    catalog = load_morphometric_catalog(CATALOG_PATH)
    frozen_path = output / "frozen_morphometric_design.yaml"
    frozen_path.write_bytes(CATALOG_PATH.read_bytes())
    combined, adversarial, resolution, prevalidation = calibration_tables(catalog)
    combined.to_parquet(output / "calibration_fixtures.parquet", index=False, compression="zstd")
    adversarial.to_csv(output / "adversarial_metric_atlas.csv", index=False)
    resolution.to_csv(output / "resolution_robustness.csv", index=False)
    write_json(output / "pre_outcome_validation.json", prevalidation)
    (output / "metric_specification.md").write_text(
        metric_specification_text(catalog), encoding="utf-8"
    )
    snapshot = source_snapshot()
    freeze = {
        "schemaVersion": "e06.s13.design-freeze.v1",
        "researchStepId": "S13",
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "repositoryBranch": git_output("branch", "--show-current"),
        "catalogPath": str(CATALOG_PATH),
        "catalogSha256": file_sha256(CATALOG_PATH),
        "frozenArtifactSha256": file_sha256(frozen_path),
        "targetCatalogSha256": file_sha256(TARGET_PATH),
        "grammarCatalogSha256": file_sha256(GRAMMAR_PATH),
        "adversarialCatalogSha256": file_sha256(ADVERSARIAL_PATH),
        "sourceTableSnapshot": snapshot,
        "frozenBeforeOutcomeValueRead": True,
        "outcomeValuesReadDuringFreeze": False,
        "priorOutcomeClassificationsLocked": PRIOR_CLASSIFICATIONS,
        "completionContractChanged": False,
        "s12TerminalMaintenanceReinterpretedAsUninterrupted": False,
        "preOutcomeValidationSuccess": bool(prevalidation["success"]),
    }
    write_json(output / "design_freeze.json", freeze)
    return freeze


def _worker_init() -> None:
    global _TARGETS, _GRAMMARS, _ORBITS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _TARGETS, _GRAMMARS, _ORBITS = _assets()


def _analysis_arm(step_id: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if step_id == "S09":
        return str(row.get("policyId")), str(row.get("startFamily"))
    if step_id == "S10":
        return str(row.get("arm")), f"{row.get('lesionId')}:{row.get('severity')}"
    if step_id == "S11":
        return f"{row.get('mixtureId')}:{row.get('interventionArm')}", str(
            row.get("startFamily")
        )
    if step_id == "S12":
        return str(row.get("armId")), str(row.get("challengeId"))
    raise ValueError(step_id)


def _score_record(step_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    target_id = str(row["targetId"])
    grid = tuple(tuple(item) for item in json.loads(str(row["finalGridRowsJson"])))
    panel = score_morphology(
        grid,
        _TARGETS[target_id],
        _GRAMMARS[target_id],
        equivalence_orbit=_ORBITS[target_id],
    )
    arm, challenge = _analysis_arm(step_id, row)
    retained = {
        column: row.get(column)
        for column in CONTEXT_COLUMNS
        if column in row and column != "finalGridRowsJson"
    }
    retained_cost_sources = {
        column: row.get(column) for column in COST_SOURCE_COLUMNS[step_id]
    }
    return {
        "sourceStepId": step_id,
        "analysisArm": arm,
        "analysisChallenge": challenge,
        **retained,
        **retained_cost_sources,
        **panel,
        **standardize_endpoint_time(step_id, row),
        **standardize_cost_components(step_id, row),
    }


def _score_chunk(task: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    step_id, records = task
    return [_score_record(step_id, record) for record in records]


def _chunks(values: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def load_and_score_outcomes(workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = []
    input_counts = {}
    started = time.perf_counter()
    for step_id, path in SOURCE_TABLES.items():
        frame = pd.read_parquet(path)
        input_counts[step_id] = len(frame)
        if len(frame) != EXPECTED_ROWS[step_id]:
            raise ValueError(f"{step_id} row count {len(frame)} != {EXPECTED_ROWS[step_id]}")
        selected = sorted(
            set(CONTEXT_COLUMNS)
            .union(COST_SOURCE_COLUMNS[step_id])
            .intersection(frame.columns)
        )
        records = frame[selected].to_dict(orient="records")
        tasks.extend((step_id, chunk) for chunk in _chunks(records, 100))
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(_score_chunk, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"S13 scored chunks {index}/{len(futures)}", flush=True)
    output = pd.DataFrame(rows).sort_values(["sourceStepId", "runId"]).reset_index(drop=True)
    return output, {
        "sourceRows": input_counts,
        "scoredRows": len(output),
        "workers": workers,
        "chunkSize": 100,
        "wallSeconds": time.perf_counter() - started,
    }


def validate_outcomes(frame: pd.DataFrame) -> dict[str, Any]:
    float_pairs = {
        "s02SoftScore": "terminalS02SoftScore",
        "s02RelationalScore": "terminalS02RelationalScore",
        "imageDistanceModuloEquivalence": "terminalS01MismatchFraction",
    }
    exact_pairs = {
        "s01GlobalSuccess": "terminalS01GlobalSuccess",
        "s01GeometryMatch": "terminalS01GeometryMatch",
        "s01ComponentMatch": "terminalS01ComponentMatch",
        "s01TopologyMatch": "terminalS01TopologyMatch",
        "s01MismatchCount": "terminalS01MismatchCount",
        "vacancyHoles": "terminalVacancyHoles",
        "s02GrammarAccepted": "terminalS02GrammarAccepted",
        "s02HardViolationCount": "terminalS02HardViolationCount",
        "conjunctiveCompletion": "terminalConjunctiveCompletion",
    }
    agreement: dict[str, int] = {}
    for calculated, recorded in exact_pairs.items():
        agreement[f"{calculated}Agreement"] = int(
            (frame[calculated] == frame[recorded]).sum()
        )
    for calculated, recorded in float_pairs.items():
        agreement[f"{calculated}Agreement"] = int(
            np.isclose(
                frame[calculated].astype(float),
                frame[recorded].astype(float),
                rtol=0,
                atol=1e-12,
            ).sum()
        )
    eligible = frame[frame["timeEligible"]]
    raw_censor_match = int(
        (eligible["timeRightCensored"] == eligible["censored"]).sum()
    )
    structural_missing = {}
    cost_agreements = {}
    for step_id in SOURCE_TABLES:
        group = frame[frame["sourceStepId"] == step_id]
        applicable = set(COST_SOURCE_COLUMNS[step_id])
        structural_columns = [
            f"cost_{column}"
            for column in ALL_COST_SOURCE_COLUMNS
            if column not in applicable
        ]
        structural_missing[step_id] = int(group[structural_columns].isna().all(axis=1).sum())
        source_matches = []
        for column in applicable:
            cost_column = f"cost_{column}"
            source_matches.append(group[cost_column] == group[column])
        cost_agreements[step_id] = int(pd.concat(source_matches, axis=1).all(axis=1).sum())
    central_exact = frame[
        (frame["sourceStepId"] == "S12")
        & (frame["phase"] == "confirmation")
        & (frame["analysisChallenge"] == "exact_maintenance")
        & (frame["analysisArm"] == "central_only")
    ]
    central_exact_terminal = int(central_exact["conjunctiveCompletion"].sum())
    central_time_ineligible = int((~central_exact["timeEligible"]).sum())
    row_counts = frame.groupby("sourceStepId").size().astype(int).to_dict()
    validation = {
        "schemaVersion": "e06.s13.outcome-validation.v1",
        "researchStepId": "S13",
        "rowCount": len(frame),
        "expectedRowCount": sum(EXPECTED_ROWS.values()),
        "rowCountsByStep": row_counts,
        "duplicateSourceRunIds": int(frame.duplicated(["sourceStepId", "runId"]).sum()),
        "failedSourceRuns": int(frame["failed"].fillna(False).sum()),
        "transportCompositionValidRows": int(frame["transportCompositionValid"].sum()),
        "rawCensorAgreementEligibleRows": raw_censor_match,
        "timeEligibleRows": len(eligible),
        "s12ConfirmationCentralExactRows": len(central_exact),
        "s12ConfirmationCentralExactTerminalConjunctions": central_exact_terminal,
        "s12ConfirmationCentralExactTimeIneligibleRows": central_time_ineligible,
        "classificationChanges": 0,
        "completionContractChanges": 0,
        "metricAgreements": agreement,
        "structuralCostMissingnessRows": structural_missing,
        "componentCostAgreementRows": cost_agreements,
    }
    validation["success"] = bool(
        validation["rowCount"] == validation["expectedRowCount"]
        and row_counts == EXPECTED_ROWS
        and validation["duplicateSourceRunIds"] == 0
        and validation["failedSourceRuns"] == 0
        and validation["transportCompositionValidRows"] == len(frame)
        and raw_censor_match == len(eligible)
        and all(value == len(frame) for value in agreement.values())
        and all(structural_missing[step] == EXPECTED_ROWS[step] for step in SOURCE_TABLES)
        and all(cost_agreements[step] == EXPECTED_ROWS[step] for step in SOURCE_TABLES)
        and len(central_exact) == 1000
        and central_exact_terminal == 1000
        and central_time_ineligible == 1000
    )
    return validation


def outcome_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["impurity"] = 1 - working["occupiedRegionPurity"]
    grouping = [
        "sourceStepId",
        "phase",
        "targetId",
        "analysisChallenge",
        "analysisArm",
    ]
    return (
        working.groupby(grouping, dropna=False)
        .agg(
            runs=("runId", "size"),
            s02GrammarAcceptanceRate=("s02GrammarAccepted", "mean"),
            s01GlobalSuccessRate=("s01GlobalSuccess", "mean"),
            conjunctiveCompletionRate=("conjunctiveCompletion", "mean"),
            meanImageDistance=("imageDistanceModuloEquivalence", "mean"),
            meanTransportDistance=("transportDistance", "mean"),
            meanComponentExcess=("componentExcess", "mean"),
            meanBoundaryRelativeError=("targetBoundaryRelativeError", "mean"),
            meanOccupiedRegionPurity=("occupiedRegionPurity", "mean"),
            meanSymmetryMismatch=("symmetryMismatchFraction", "mean"),
            meanHoleCountError=("holeCountError", "mean"),
        )
        .reset_index()
    )


def disagreement_tables(frame: pd.DataFrame, catalog: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = catalog["disagreementRules"]["thresholds"]
    rows = []
    for (step_id, phase), group in frame.groupby(["sourceStepId", "phase"]):
        global_reject = ~group["s01GlobalSuccess"]
        symmetry_valid = group["symmetryMismatchFraction"].notna()
        records = {
            "s02AcceptsS01Rejects": group["s02GrammarAccepted"] & global_reject,
            "componentMatchesS01Rejects": group["s01ComponentMatch"] & global_reject,
            "topologyMatchesS01Rejects": group["s01TopologyMatch"] & global_reject,
            "highPurityS01Rejects": (
                group["occupiedRegionPurity"]
                >= float(thresholds["occupiedRegionPurityHigh"])
            )
            & global_reject,
            "symmetryNearS01Rejects": symmetry_valid
            & (
                group["symmetryMismatchFraction"]
                <= float(thresholds["symmetryNear"])
            )
            & global_reject,
            "boundaryNearS01Rejects": (
                group["targetBoundaryRelativeError"]
                <= float(thresholds["boundaryNear"])
            )
            & global_reject,
            "transportNearS01Rejects": (
                group["transportDistance"] <= float(thresholds["transportNear"])
            )
            & global_reject,
        }
        for disagreement, mask in records.items():
            rows.append(
                {
                    "sourceStepId": step_id,
                    "phase": phase,
                    "disagreement": disagreement,
                    "count": int(mask.sum()),
                    "fractionOfRuns": float(mask.mean()),
                    "s01RejectedRuns": int(global_reject.sum()),
                }
            )
    disagreement = pd.DataFrame(rows)
    measure_frame = frame.assign(
        s02LocalDeficit=1 - frame["s02RelationalScore"],
        occupiedRegionImpurity=1 - frame["occupiedRegionPurity"],
    )
    measures = [
        "s02LocalDeficit",
        "imageDistanceModuloEquivalence",
        "transportDistance",
        "componentExcess",
        "targetBoundaryRelativeError",
        "occupiedRegionImpurity",
        "symmetryMismatchFraction",
        "holeCountError",
    ]
    correlations = []
    for step_id, group in measure_frame.groupby("sourceStepId"):
        for first_index, first in enumerate(measures):
            for second in measures[first_index + 1 :]:
                complete = group[[first, second]].dropna()
                rho = complete[first].corr(complete[second], method="spearman")
                correlations.append(
                    {
                        "sourceStepId": step_id,
                        "metricA": first,
                        "metricB": second,
                        "completeRows": len(complete),
                        "uniqueA": int(complete[first].nunique()),
                        "uniqueB": int(complete[second].nunique()),
                        "spearmanRho": rho,
                    }
                )
    return disagreement, pd.DataFrame(correlations)


def km_summary(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame["timeEligible"]].copy()
    grouping = [
        "sourceStepId",
        "phase",
        "timeEndpointType",
        "targetId",
        "analysisChallenge",
        "analysisArm",
    ]
    rows = []
    for keys, group in eligible.groupby(grouping, dropna=False):
        times = group["timeAnalysisTransition"].astype(float).to_numpy()
        events = group["timeEventObserved"].astype(bool).to_numpy()
        tau = float(group["eventBudgetTransitions"].max())
        survival = 1.0
        previous = 0.0
        rmst = 0.0
        median = None
        for event_time in sorted(set(times[events])):
            rmst += survival * (event_time - previous)
            at_risk = int((times >= event_time).sum())
            event_count = int(((times == event_time) & events).sum())
            survival *= 1 - event_count / at_risk
            previous = event_time
            if median is None and survival <= 0.5:
                median = event_time
        rmst += survival * (tau - previous)
        event_times = times[events]
        rows.append(
            {
                **dict(zip(grouping, keys)),
                "eligibleRuns": len(group),
                "events": int(events.sum()),
                "rightCensored": int((~events).sum()),
                "eventFraction": float(events.mean()),
                "medianObservedEventTime": float(np.median(event_times))
                if len(event_times)
                else None,
                "kaplanMeierMedian": median,
                "restrictedMeanTimeToEndpoint": rmst,
                "restrictionTransition": tau,
            }
        )
    return pd.DataFrame(rows)


def cost_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = ["sourceStepId", "phase", "analysisChallenge", "analysisArm"]
    rows = []
    for keys, group in frame.groupby(grouping, dropna=False):
        step_id = str(keys[0])
        for column in COST_SOURCE_COLUMNS[step_id]:
            values = group[f"cost_{column}"].dropna().astype(float)
            rows.append(
                {
                    **dict(zip(grouping, keys)),
                    "costComponent": column,
                    "runs": len(values),
                    "sum": float(values.sum()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "maximum": float(values.max()),
                    "structurallyApplicable": True,
                }
            )
    return pd.DataFrame(rows)


def classification_lock() -> dict[str, Any]:
    sources = []
    for step_id in SOURCE_TABLES:
        directory = Path("/artifacts/research_steps") / step_id
        for filename in (
            "research_step_full_results.md",
            "artifact_manifest.json",
            "outcome_decision.json",
        ):
            path = directory / filename
            if path.exists():
                sources.append(
                    {
                        "stepId": step_id,
                        "path": str(path),
                        "sha256": file_sha256(path),
                    }
                )
    return {
        "schemaVersion": "e06.s13.classification-lock.v1",
        "researchStepId": "S13",
        "classificationsBeforeS13": PRIOR_CLASSIFICATIONS,
        "classificationsAfterS13": PRIOR_CLASSIFICATIONS,
        "classificationsChanged": False,
        "completionContractChanged": False,
        "s12CentralOnlyClaim": "terminal_exact_start_restoration_or_maintenance_not_uninterrupted_maintenance",
        "sourceLocks": sources,
    }


def upstream_hashes() -> list[dict[str, Any]]:
    records = []
    for step in range(1, 13):
        directory = Path(f"/artifacts/research_steps/S{step:02d}")
        for name in ("research_step_full_results.md", "artifact_manifest.json"):
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(path)
            records.append(
                {
                    "stepId": f"S{step:02d}",
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def create_figures(adversarial: pd.DataFrame, disagreement: pd.DataFrame, output: Path) -> None:
    color_map = {
        ".": "#f5f5f5",
        "A": "#4575b4",
        "B": "#d73027",
        "C": "#66bd63",
        "R": "#fdae61",
        "M": "#984ea3",
        "T": "#636363",
    }
    figure, axes = plt.subplots(1, len(adversarial), figsize=(14, 2.8))
    for axis, row in zip(axes, adversarial.to_dict(orient="records")):
        grid = json.loads(row["rowsJson"])
        rgba = np.asarray(
            [
                [matplotlib.colors.to_rgba(color_map[token]) for token in grid_row]
                for grid_row in grid
            ]
        )
        axis.imshow(rgba)
        axis.set_title(
            f"{row['targetId'].split('_')[0]}\nH={row['imageDistanceModuloEquivalence']:.2f}, "
            f"W={row['transportDistance']:.3f}",
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Frozen S02-accepted / S01-rejected adversarial fixtures")
    figure.tight_layout()
    figure.savefig(output / "adversarial_examples.png", dpi=180)
    plt.close(figure)

    pivot = disagreement.groupby(["sourceStepId", "disagreement"])["fractionOfRuns"].mean().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(11, 4))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right", fontsize=8)
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Prespecified diagnostic disagreement rates")
    figure.colorbar(image, ax=axis, label="fraction of runs")
    figure.tight_layout()
    figure.savefig(output / "metric_disagreement.png", dpi=180)
    plt.close(figure)


def artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schemaVersion": "e06.s13.artifact-manifest.v1",
        "researchStepId": "S13",
        "fileCount": len(files),
        "files": files,
    }


def report_text(
    frame: pd.DataFrame,
    validation: Mapping[str, Any],
    prevalidation: Mapping[str, Any],
    disagreement: pd.DataFrame,
    correlations: pd.DataFrame,
    time_summary: pd.DataFrame,
    execution: Mapping[str, Any],
    commit: str,
) -> str:
    local_global = int(
        (frame["s02GrammarAccepted"] & ~frame["s01GlobalSuccess"]).sum()
    )
    events = int(frame["timeEventObserved"].sum())
    eligible = int(frame["timeEligible"].sum())
    censored = int(frame["timeRightCensored"].sum())
    finite_rho = correlations["spearmanRho"].dropna()
    minimum_rho = float(finite_rho.min()) if len(finite_rho) else float("nan")
    maximum_rho = float(finite_rho.max()) if len(finite_rho) else float("nan")
    artifacts = [
        "frozen_morphometric_design.yaml and design_freeze.json",
        "metric_specification.md",
        "calibration_fixtures.parquet and adversarial_metric_atlas.csv",
        "resolution_robustness.csv",
        "morphology_metrics.parquet",
        "outcome_metric_summary.csv",
        "metric_disagreement_summary.csv and metric_correlations.csv",
        "time_to_endpoint_summary.csv and cost_component_summary.csv",
        "classification_lock.json",
        "validation_results.json, provenance_manifest.json, execution_manifest.json, and artifact_manifest.json",
        "test_results.json",
        "adversarial_examples.png and metric_disagreement.png",
    ]
    return f"""# S13 — Use morphological and topological metrics

## Top summary

- **Research step:** S13 — Use morphological and topological metrics
- **Completion status:** Complete on 2026-07-21; S13 only was executed and S14 was not started.
- **Artifacts written:** {'; '.join(artifacts)}.
- **Validation result:** Passed fixture-only pre-outcome calibration and all {validation['rowCount']:,} outcome rescores. Exact S02/S01 agreement, run accounting, composition, censoring, component-cost, classification-lock, and S12 terminal-only interpretation gates passed.
- **Outcome classification:** Supportive for the frozen metric-calibration hypothesis, with a constraining noninterchangeability and bounded-resolution subfinding.
- **Caveats or blockers:** The panel is calibrated only on unobstructed bounded square four-neighbor fixtures. Transport is anchored to the S01-selected nearest image rather than separately minimized over the orbit. Purity is label-blind, boundary length is scalar, fixed-budget nulls remain censored, and no metric can establish uninterrupted maintenance from terminal checkpoints.
- **Lay summary:** The new measurements agree on exact target images but deliberately disagree on many malformed or simulated images. That is useful: a locally plausible grammar score, clean components, high purity, or symmetry can each look good while the full target still fails. The panel therefore exposes why a result looks close without replacing the original pass/fail rule.
- **Recommended next action:** Return control to the Chief Scientist. Review the calibrated panel and its blind spots; if S14 is separately authorized, freeze its metric directions and component-wise budgets before optimization. Do not treat S12 central-only terminal restoration as uninterrupted maintenance.

## Frozen question and decision rule

The frozen question was whether a prespecified panel of local, geometric, transport, component, boundary, purity, symmetry, topology, time, and cost metrics prevents interpretation from depending on one favorable score. Support required 148/148 exact equivalence states, 3/3 tolerated variants, all seven known S02 false positives, four hand-labeled fixtures, three resolution scales, and every S09–S12 run to validate without changing completion or prior classifications. The endpoint remained `S02 grammar accepted AND independent S01 global success`.

S13 did not select metrics after outcomes, redefine completion, rerun simulations, or reclassify S09–S12. Metric adequacy—not formation, repair, chimera, or hybrid-control success—was the primary S13 outcome.

## Inputs and provenance

- Repository commit used for outcome analysis: `{commit}` on `eidosoma/groups/28`.
- Canonical S01 target catalog and S02 grammar catalog, including all 148 equivalence states and seven retained grammar false positives.
- Completed S01–S12 reports and manifests; outcome tables were S09 {EXPECTED_ROWS['S09']:,}, S10 {EXPECTED_ROWS['S10']:,}, S11 {EXPECTED_ROWS['S11']:,}, and S12 {EXPECTED_ROWS['S12']:,} rows.
- Relevant E01 identity/schedule/pairing contracts and E04 metric-disagreement, composition, state/flux, and intervention guidance.
- Attachment manifest and sidecar; the supplied paper provided one-dimensional background but no alternative calibrated two-dimensional completion panel.
- No dataset or new dependency was required.

The catalog, source-table schemas/hashes, prior classifications, definitions, fixtures, thresholds, and decision gates were written to `design_freeze.json` before outcome values were opened. Repository code remains in Git; artifacts contain compact evidence only.

## Methods

### Spatial metrics

For each terminal grid, S02 `score_grid` supplied local acceptance, soft score, relational score, and hard violations. S01 `evaluate_success` independently supplied count, minimum Hamming image distance over the exact equivalence orbit, component, hole, and global-membership fields.

Categorical Earth Mover distance was the sum of exact minimum one-to-one Manhattan assignments for every declared token, including vacancy, to the S01-selected reference. It was normalized by site count times bounded-grid diameter. Connected components and holes used four-neighbor bounded topology. Boundary length counted undirected heterotypic internal edges and was normalized by the square root of area. Region purity was the size-weighted modal-token fraction within non-vacancy target regions and was explicitly label-blind. Symmetry mismatch averaged self-mismatch under nonidentity D4 stabilizers of the selected target reference, using foreground frames only for translatable targets.

### Time and costs

Formation time applied only to initially incomplete formation starts. Repair time applied only to explicitly repair-eligible damaged starts. Non-events were retained as right-censored at the fixed budget. Exact-start S11/S12 runs were not assigned a time of zero: loss/restoration instrumentation is insufficient for uninterrupted-maintenance time. The S12 central-only 1,000/1,000 terminal result remained terminal restoration or maintenance only.

Every source ledger component was retained separately. Unavailable components were null, not zero; no information/action/intervention/source-work/computation/opportunity scalar was formed.

### Disagreement and resolution analyses

The analysis applied all prespecified thresholds, reported every pairwise Spearman association among the eight spatial deficit views, and did not choose a favorable subset. Resolution robustness used nearest-neighbor block replication at 1×, 2×, and 3×. Image fraction, purity, normalized boundary, symmetry, components, and holes had exact invariance gates; transport allowed at most 0.03 absolute drift. S01/S02 membership was not evaluated at noncanonical resolution.

## Commands and runtime

```bash
PYTHONPATH=src python scripts/build_morph2d_s13.py --freeze-only --replace
PYTHONPATH=src:. pytest -q tests/test_morph2d_morphometrics.py tests/test_morph2d_targets.py tests/test_morph2d_grammar.py
PYTHONPATH=src python scripts/build_morph2d_s13.py --workers 8
PYTHONPATH=src:. pytest -q tests/test_morph2d_*.py
```

Outcome rescoring used {execution['workers']} worker processes with one BLAS/OpenMP thread each and took {execution['wallSeconds']:.1f} seconds. The fixture-focused suite passed 30/30 tests and the complete morphology suite passed 157/157. An initial broad-suite invocation with only `PYTHONPATH=.` produced three import-collection errors in older builders; the canonical `PYTHONPATH=src:.` invocation above resolved the environment path and passed without code changes. GPU acceleration was not used: this step was small-grid CPU metric analysis, not a transition-data-plane workload.

## Calibration results

- Exact equivalence: {prevalidation['exactFixturesPassed']}/{prevalidation['exactFixtureCount']} passed with zero image and transport distance, target component/hole/boundary agreement, purity 1, and zero declared-stabilizer mismatch.
- S01 tolerated interfaces: {prevalidation['toleratedFixturesPassed']}/{prevalidation['toleratedFixtureCount']} retained both S01 and S02 acceptance while recording nonzero image/transport distance.
- Grammar adversaries: {prevalidation['adversarialFixturesPassed']}/{prevalidation['adversarialFixtureCount']} remained S02-accepted/S01-rejected and {prevalidation['adversarialFixturesWithGlobalFlag']}/{prevalidation['adversarialFixtureCount']} were flagged by an independent S01 geometric, component, topology, or boundary gate.
- Hand labels: {prevalidation['handFixturesPassed']}/{prevalidation['handFixtureCount']} matched declared component, boundary, and hole counts.
- Resolution: {prevalidation['resolutionScalesPassed']}/{prevalidation['resolutionScaleCount']} scales passed. This is diagnostic robustness, not topology or resolution transfer of completion.

## Outcome-panel results

All {validation['rowCount']:,} terminal grids had valid exact composition transport and reproduced every stored S01 and S02 field to exact/1e-12 tolerance. There were {local_global:,} runs where S02 accepted but S01 rejected, directly preserving the local/global distinction. Across step-level metric pairs, finite Spearman coefficients ranged from {minimum_rho:.3f} to {maximum_rho:.3f}; several component, hole, purity, symmetry, and boundary views agreed on states that the full S01 gate rejected. These are prespecified disagreements, not alternate successes.

The endpoint-time risk set contained {eligible:,} runs: {events:,} observed first completions/repairs and {censored:,} right-censored non-events. The event table retains endpoint type and fixed restriction time. No activity or quiescence criterion was used.

S09 remained supportive only for its frozen near-target contrast with broad-formation constraints; S10 remained null; S11 and S12 remained constraining/contradictory. S12 central-only confirmation retained exactly 1,000/1,000 terminal conjunctions in the exact-maintenance block, and all 1,000 were explicitly ineligible for a repair/maintenance-time claim.

## Validation

- Complete accounting: {validation['rowCount']:,}/{validation['expectedRowCount']:,}; zero duplicate source/run keys and zero failed source runs.
- Target rescoring: every stored S01 global, geometry, component, topology, mismatch, and hole field agreed.
- Grammar rescoring: every stored S02 acceptance, soft, relational, and hard-violation field agreed.
- Conjunction: every terminal conjunctive field agreed; classifications changed = 0 and completion-contract changes = 0.
- Censoring: {validation['rawCensorAgreementEligibleRows']:,}/{validation['timeEligibleRows']:,} eligible rows matched source censoring.
- Costs: all applicable components agreed for every row; every structurally unavailable component remained null.
- S12 claim boundary: {validation['s12ConfirmationCentralExactTerminalConjunctions']:,}/{validation['s12ConfirmationCentralExactRows']:,} central-only exact rows were terminally conjunctive, and {validation['s12ConfirmationCentralExactTimeIneligibleRows']:,}/{validation['s12ConfirmationCentralExactRows']:,} were excluded from uninterrupted-maintenance time.
- Artifact hashes cover all compact outputs; upstream report/manifest and source-table hashes are recorded.
- Tests: 30/30 fixture-focused and 157/157 complete morphology tests passed; Ruff and byte-compilation checks passed.

## Caveats, blockers, and failed assumptions

The S02 sufficiency assumption remains contradicted. Metrics are not interchangeable: purity is label-blind, component and hole counts ignore placement, boundary length ignores boundary location, scalar symmetry can miss asymmetric relational errors, and image distance can penalize unenumerated relational equivalents. Conversely, this benchmark only treats the enumerated S01 orbit as equivalent; S13 did not expand it.

Transport is exact for the selected anchor, but the anchor is chosen by S01 Hamming/boundary rank; S13 did not add a post-outcome transport minimization over orbit members. Resolution replication does not authorize S01/S02 completion claims on larger, periodic, hexagonal, irregular, obstacle, or fixed-boundary environments. Fixed-budget noncompletion remains censoring, not impossibility. Count-changing repair remains unsupported. Formed sources remain operational exact fixtures. No causal, biological, wet-lab, attractor, or convergence claim follows.

## Artifact guide

`morphology_metrics.parquet` is the run-level panel. `outcome_metric_summary.csv`, `metric_disagreement_summary.csv`, and `metric_correlations.csv` provide aggregate views. `time_to_endpoint_summary.csv` preserves eligibility and censoring; `cost_component_summary.csv` preserves ledgers without scalar collapse. Fixture and resolution evidence is separate from outcomes. `classification_lock.json` documents the unchanged S09–S12 narrative.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("S13 workers must be in 1..8")
    output = arguments.output_dir
    if arguments.freeze_only:
        freeze = freeze_design(output, replace=arguments.replace)
        print(json.dumps(freeze, indent=2))
        return
    freeze_path = output / "design_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError("run --freeze-only before opening outcome values")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if file_sha256(CATALOG_PATH) != freeze["catalogSha256"]:
        raise ValueError("metric catalog changed after pre-outcome freeze")
    if not freeze["frozenBeforeOutcomeValueRead"] or not freeze["preOutcomeValidationSuccess"]:
        raise ValueError("invalid S13 pre-outcome freeze")
    current_snapshot = source_snapshot()
    frozen_hashes = {item["stepId"]: item["sha256"] for item in freeze["sourceTableSnapshot"]}
    if any(item["sha256"] != frozen_hashes[item["stepId"]] for item in current_snapshot):
        raise ValueError("source outcome table changed after S13 freeze")
    catalog = load_morphometric_catalog(CATALOG_PATH)
    frame, execution = load_and_score_outcomes(arguments.workers)
    validation = validate_outcomes(frame)
    if not validation["success"]:
        raise ValueError(f"S13 outcome validation failed: {validation}")
    prevalidation = json.loads((output / "pre_outcome_validation.json").read_text(encoding="utf-8"))
    disagreement, correlations = disagreement_tables(frame, catalog)
    summary = outcome_summary(frame)
    time_summary = km_summary(frame)
    costs = cost_summary(frame)
    lock = classification_lock()
    frame.to_parquet(output / "morphology_metrics.parquet", index=False, compression="zstd")
    summary.to_csv(output / "outcome_metric_summary.csv", index=False)
    disagreement.to_csv(output / "metric_disagreement_summary.csv", index=False)
    correlations.to_csv(output / "metric_correlations.csv", index=False)
    time_summary.to_csv(output / "time_to_endpoint_summary.csv", index=False)
    costs.to_csv(output / "cost_component_summary.csv", index=False)
    write_json(output / "classification_lock.json", lock)
    write_json(output / "validation_results.json", validation)
    adversarial = pd.read_csv(output / "adversarial_metric_atlas.csv")
    create_figures(adversarial, disagreement, output)
    execution.update(
        {
            "schemaVersion": "e06.s13.execution-manifest.v1",
            "researchStepId": "S13",
            "threadEnvironment": {
                "OMP_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
            },
            "gpuUsed": False,
            "outcomeValuesOpenedOnlyAfterFreeze": True,
        }
    )
    write_json(output / "execution_manifest.json", execution)
    commit = git_output("rev-parse", "HEAD")
    write_json(
        output / "provenance_manifest.json",
        {
            "schemaVersion": "e06.s13.provenance-manifest.v1",
            "researchStepId": "S13",
            "repositoryCommit": commit,
            "branch": git_output("branch", "--show-current"),
            "catalogSha256": freeze["catalogSha256"],
            "sourceTables": current_snapshot,
            "upstreamArtifacts": upstream_hashes(),
            "previousArtifacts": [
                "/previous-artifacts/E01/research_steps/S03/transition_spec.md",
                "/previous-artifacts/E01/research_steps/S08/seed_specification.json",
                "/previous-artifacts/E01/research_steps/S08/pairing_validation.json",
                "/previous-artifacts/E04/report_inputs/e06_e07_handoff.md",
                "/previous-artifacts/E04/report_inputs/aggregation_classification.md",
            ],
            "inputAttachmentManifest": "/workspace/input-attachments/MANIFEST.json",
            "datasetRequired": False,
            "newDependencies": [],
        },
    )
    decision = {
        "schemaVersion": "e06.s13.metric-decision.v1",
        "researchStepId": "S13",
        "preOutcomeCalibrationPassed": bool(prevalidation["success"]),
        "outcomeValidationPassed": bool(validation["success"]),
        "priorClassificationsChanged": False,
        "completionContractChanged": False,
        "panelCalibrationSupport": bool(prevalidation["success"] and validation["success"]),
        "outcomeClassification": "supportive_with_constraining_noninterchangeability_and_resolution_subfinding",
        "recommendedNextAction": "Return control to the Chief Scientist; review S13 before separately authorizing S14.",
    }
    write_json(output / "metric_decision.json", decision)
    report = report_text(
        frame,
        validation,
        prevalidation,
        disagreement,
        correlations,
        time_summary,
        execution,
        commit,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    write_json(output / "artifact_manifest.json", artifact_manifest(output))
    print(json.dumps({"validation": validation, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
