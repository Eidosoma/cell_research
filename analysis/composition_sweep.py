"""E04 S03 paired composition and policy-value correlation factorial.

The design is frozen in ``analysis/s03_composition_factorial.json``.  This
module consumes only unprotected E01 exploratory base draws, preserves their
values and initial occupancy across all paired conditions, executes the
validated clean-room reference simulator, applies the S02 exact-composition
correction, and packages compact response surfaces and uncertainty records.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
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
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from analysis.aggregation_baselines import expected_paper_aggregation
from analysis.chimeric_replication import GRID, canonical_hash, run_reference_summary
from reference_simulator.model import (
    Architecture,
    Cell,
    Direction,
    FaultMode,
    Policy,
    Scenario,
    canonical_json_bytes,
)
from reference_simulator.rng import permutation


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
UPSTREAM = Path("/previous-artifacts/E01")
S08_DIR = UPSTREAM / "research_steps/S08"
S01_DIR = Path("/artifacts/research_steps/S01")
S02_DIR = Path("/artifacts/research_steps/S02")
CONTRACT_PATH = REPOSITORY / "analysis/s03_composition_factorial.json"
OUTPUT_SCHEMA = "e04.s03.composition_sweep.v1"
SEED_VERSION = "E04/S03/SHA256_COUNTER/v1"
MASTER_SEED = 0xE0403000000000000000000000000001
EXPECTED_CONDITIONS = 186
REPLICATES = 250
EXPECTED_RUNS = EXPECTED_CONDITIONS * REPLICATES
BOOTSTRAP_DRAWS = 10_000
VARIANCE_TOLERANCE = 1e-24
S01_MANIFEST_SHA256 = "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079"
S02_MANIFEST_SHA256 = "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54"
POLICY_CODES = {"Bubble": "BUB", "Insertion": "INS", "Selection": "SEL"}
INPUT_CODES = {"unique_1_100": "UNQ", "repeated_1_10_x10": "REP"}
CORRELATION_CODES = {"positive": "POS", "negative": "NEG", "absent": "ABS"}
POLICY_PAIRS = (
    ("Bubble", "Insertion"),
    ("Bubble", "Selection"),
    ("Insertion", "Selection"),
)
THREE_POLICIES = ("Bubble", "Insertion", "Selection")
CORRELATIONS = ("positive", "negative", "absent")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_native(value)) + b"\n")


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_s03_seed(stream: str, *address: Any) -> int:
    if not stream or not stream.isascii():
        raise ValueError("stream must be nonempty ASCII")
    material = {
        "version": SEED_VERSION,
        "stream": stream,
        "address": list(address),
    }
    digest = hashlib.sha256(
        b"E04/S03/seed/v1\x00"
        + MASTER_SEED.to_bytes(16, "big")
        + b"\x00"
        + canonical_json_bytes(material)
    ).digest()
    return int.from_bytes(digest[:16], "big")


@dataclass(frozen=True, slots=True)
class SweepCondition:
    condition_id: str
    input_profile: str
    policies: tuple[str, ...]
    composition_profile: str
    correlation_profile: str
    first_policy_count: int | None = None
    rare_policy: str | None = None

    @property
    def policy_set_label(self) -> str:
        return "+".join(self.policies)

    @property
    def composition_class(self) -> str:
        return "pairwise" if len(self.policies) == 2 else "three_way"

    def counts(self, replicate: int) -> tuple[int, ...]:
        if len(self.policies) == 2:
            if self.first_policy_count is None:
                raise ValueError("pairwise condition lacks first-policy count")
            return (self.first_policy_count, 100 - self.first_policy_count)
        if self.composition_profile == "balanced_rotating":
            values = [33, 33, 33]
            values[replicate % 3] += 1
            return tuple(values)
        if self.rare_policy is None:
            raise ValueError("rare three-way condition lacks rare policy")
        return tuple(
            10 if policy == self.rare_policy else 45 for policy in self.policies
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditionId": self.condition_id,
            "inputProfile": self.input_profile,
            "policies": list(self.policies),
            "policySetLabel": self.policy_set_label,
            "compositionClass": self.composition_class,
            "compositionProfile": self.composition_profile,
            "correlationProfile": self.correlation_profile,
            "firstPolicyCount": self.first_policy_count,
            "rarePolicy": self.rare_policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepCondition":
        return cls(
            condition_id=str(value["conditionId"]),
            input_profile=str(value["inputProfile"]),
            policies=tuple(str(item) for item in value["policies"]),
            composition_profile=str(value["compositionProfile"]),
            correlation_profile=str(value["correlationProfile"]),
            first_policy_count=(
                int(value["firstPolicyCount"])
                if value.get("firstPolicyCount") is not None
                else None
            ),
            rare_policy=(
                str(value["rarePolicy"])
                if value.get("rarePolicy") is not None
                else None
            ),
        )


def build_conditions() -> tuple[SweepCondition, ...]:
    conditions: list[SweepCondition] = []
    for input_profile in INPUT_CODES:
        input_code = INPUT_CODES[input_profile]
        for policies in POLICY_PAIRS:
            policy_code = "-".join(POLICY_CODES[item] for item in policies)
            for first_count in range(10, 100, 10):
                composition = f"p{first_count:02d}_{100 - first_count:02d}"
                for correlation in CORRELATIONS:
                    conditions.append(
                        SweepCondition(
                            condition_id=(
                                f"S03-{input_code}-{policy_code}-P{first_count:02d}-"
                                f"{CORRELATION_CODES[correlation]}"
                            ),
                            input_profile=input_profile,
                            policies=policies,
                            composition_profile=composition,
                            correlation_profile=correlation,
                            first_policy_count=first_count,
                        )
                    )
        three_code = "-".join(POLICY_CODES[item] for item in THREE_POLICIES)
        for correlation in CORRELATIONS:
            conditions.append(
                SweepCondition(
                    condition_id=(
                        f"S03-{input_code}-{three_code}-BAL-"
                        f"{CORRELATION_CODES[correlation]}"
                    ),
                    input_profile=input_profile,
                    policies=THREE_POLICIES,
                    composition_profile="balanced_rotating",
                    correlation_profile=correlation,
                )
            )
        for rare_policy in THREE_POLICIES:
            for correlation in CORRELATIONS:
                conditions.append(
                    SweepCondition(
                        condition_id=(
                            f"S03-{input_code}-{three_code}-RARE-"
                            f"{POLICY_CODES[rare_policy]}10-"
                            f"{CORRELATION_CODES[correlation]}"
                        ),
                        input_profile=input_profile,
                        policies=THREE_POLICIES,
                        composition_profile=f"rare_{rare_policy.lower()}_10",
                        correlation_profile=correlation,
                        rare_policy=rare_policy,
                    )
                )
    result = tuple(sorted(conditions, key=lambda item: item.condition_id))
    if len(result) != EXPECTED_CONDITIONS:
        raise AssertionError(
            f"expected {EXPECTED_CONDITIONS} conditions, got {len(result)}"
        )
    if len({item.condition_id for item in result}) != len(result):
        raise AssertionError("S03 condition IDs are not unique")
    return result


def load_base_draws() -> dict[tuple[str, int], dict[str, Any]]:
    rows = pq.read_table(
        S08_DIR / "base_draw_bank.parquet",
        filters=[("split", "=", "exploratory")],
    ).to_pylist()
    selected = {
        (str(row["inputProfile"]), int(row["replicateOrdinal"])): row
        for row in rows
        if row["inputProfile"] in INPUT_CODES
    }
    if len(selected) != len(INPUT_CODES) * REPLICATES:
        raise ValueError(f"expected 500 exploratory base draws, found {len(selected)}")
    if any(bool(row["protected"]) for row in selected.values()):
        raise PermissionError("protected E01 base draw entered S03")
    if any(row["split"] != "exploratory" for row in selected.values()):
        raise PermissionError("S03 base draw escaped exploratory split")
    return selected


def _association_metrics(
    values: Sequence[int], policies: Sequence[str], policy_order: Sequence[str]
) -> tuple[float, float]:
    ordinal = np.asarray(
        [policy_order.index(policy) for policy in policies], dtype=float
    )
    value_array = np.asarray(values, dtype=float)
    rho = float(stats.spearmanr(value_array, ordinal).statistic)
    overall = float(value_array.mean())
    total = float(np.square(value_array - overall).sum())
    between = 0.0
    for policy in policy_order:
        group = value_array[np.asarray(policies) == policy]
        between += len(group) * float((group.mean() - overall) ** 2)
    eta_squared = between / total if total else 0.0
    return rho, eta_squared


def _policy_assignment(
    condition: SweepCondition, base: Mapping[str, Any]
) -> tuple[list[str], int, int, tuple[int, ...]]:
    replicate = int(base["replicateOrdinal"])
    counts = condition.counts(replicate)
    values = [int(value) for value in base["valuesById"]]
    ids = tuple(f"cell-{index:04d}" for index in range(len(values)))
    assignment_address = (
        condition.input_profile,
        str(base["split"]),
        replicate,
        list(counts),
        condition.correlation_profile,
    )
    assignment_seed = derive_s03_seed("policy_assignment", *assignment_address)
    tie_seed = derive_s03_seed(
        "value_tie_break",
        condition.input_profile,
        str(base["split"]),
        replicate,
        list(counts),
    )
    assignment_key = "E04/S03/assignment/" + canonical_hash(assignment_address)
    tie_order = permutation(ids, tie_seed, assignment_key + "/ties")
    tie_rank = {cell_id: rank for rank, cell_id in enumerate(tie_order)}
    if condition.correlation_profile == "positive":
        ordered = sorted(
            ids,
            key=lambda cell_id: (
                values[int(cell_id.split("-")[1])],
                tie_rank[cell_id],
            ),
        )
    elif condition.correlation_profile == "negative":
        ordered = sorted(
            ids,
            key=lambda cell_id: (
                -values[int(cell_id.split("-")[1])],
                tie_rank[cell_id],
            ),
        )
    elif condition.correlation_profile == "absent":
        ordered = permutation(ids, assignment_seed, assignment_key + "/absent")
    else:
        raise ValueError(f"unknown correlation profile {condition.correlation_profile}")
    assigned: dict[str, str] = {}
    cursor = 0
    for policy, count in zip(condition.policies, counts):
        for cell_id in ordered[cursor : cursor + count]:
            assigned[cell_id] = policy
        cursor += count
    if len(assigned) != len(ids):
        raise AssertionError("policy assignment did not cover all identities")
    return [assigned[cell_id] for cell_id in ids], assignment_seed, tie_seed, counts


def materialize_sweep_scenario(
    condition: SweepCondition, base: Mapping[str, Any]
) -> tuple[Scenario, dict[str, Any]]:
    values = [int(value) for value in base["valuesById"]]
    occupancy_indices = [int(value) for value in base["initialOccupancyIndices"]]
    policies, assignment_seed, tie_seed, counts = _policy_assignment(condition, base)
    cells = tuple(
        Cell(
            cell_id=f"cell-{index:04d}",
            value=values[index],
            policy=Policy(policies[index]),
            direction=Direction.ASCENDING,
            fault=FaultMode.NORMAL,
        )
        for index in range(len(values))
    )
    runtime_seed = derive_s03_seed(
        "reference_runtime",
        condition.input_profile,
        str(base["split"]),
        int(base["replicateOrdinal"]),
    )
    generation_key = f"E04/S03/{condition.condition_id}/{base['baseDrawId']}"
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(f"cell-{index:04d}" for index in occupancy_indices),
        seed=runtime_seed,
        max_activations=1_000_000,
        architecture=Architecture.CELL_VIEW,
        generation_key=generation_key,
        fault_placement="explicit",
        requested_fault_count=0,
    )
    scenario.validate()
    rho, eta_squared = _association_metrics(values, policies, condition.policies)
    counts_by_policy = {
        policy: int(count) for policy, count in zip(condition.policies, counts)
    }
    minimum = min(counts)
    minority_indices = [index for index, count in enumerate(counts) if count == minimum]
    minority_policy = (
        condition.policies[minority_indices[0]] if len(minority_indices) == 1 else None
    )
    metadata = {
        "runtimeSeed": str(runtime_seed),
        "assignmentSeed": str(assignment_seed),
        "tieBreakSeed": str(tie_seed),
        "policyCounts": counts_by_policy,
        "countsVector": list(counts),
        "policyAssignmentSha256": canonical_hash(policies),
        "scenarioJsonSha256": hashlib.sha256(scenario.to_json_bytes()).hexdigest(),
        "achievedSpearmanRho": rho,
        "achievedEtaSquared": eta_squared,
        "minorityPolicy": minority_policy,
        "minorityCount": minimum if minority_policy is not None else None,
    }
    return scenario, metadata


def build_tasks() -> list[dict[str, Any]]:
    conditions = build_conditions()
    bases = load_base_draws()
    tasks: list[dict[str, Any]] = []
    for condition in conditions:
        for replicate in range(REPLICATES):
            base = bases[(condition.input_profile, replicate)]
            scenario, metadata = materialize_sweep_scenario(condition, base)
            tasks.append(
                {
                    "condition": condition.to_dict(),
                    "base": dict(base),
                    "scenarioId": scenario.scenario_id,
                    "scenarioJsonSha256": metadata["scenarioJsonSha256"],
                    "metadata": metadata,
                }
            )
    tasks.sort(
        key=lambda task: (
            task["condition"]["conditionId"],
            int(task["base"]["replicateOrdinal"]),
        )
    )
    if len(tasks) != EXPECTED_RUNS:
        raise AssertionError(f"expected {EXPECTED_RUNS} tasks, found {len(tasks)}")
    if len({task["scenarioId"] for task in tasks}) != len(tasks):
        raise AssertionError("S03 scenario IDs are not unique")
    return tasks


def _scenario_manifest_frame(tasks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for task in tasks:
        condition = task["condition"]
        base = task["base"]
        metadata = task["metadata"]
        rows.append(
            {
                "schema_version": "e04.s03.scenario_manifest.v1",
                "research_step_id": "S03",
                "scenario_id": task["scenarioId"],
                "scenario_json_sha256": task["scenarioJsonSha256"],
                "condition_id": condition["conditionId"],
                "input_profile": condition["inputProfile"],
                "policy_set_label": condition["policySetLabel"],
                "policies_json": json.dumps(
                    condition["policies"], separators=(",", ":")
                ),
                "composition_class": condition["compositionClass"],
                "composition_profile": condition["compositionProfile"],
                "first_policy_count": condition["firstPolicyCount"],
                "rare_policy": condition["rarePolicy"],
                "correlation_profile": condition["correlationProfile"],
                "base_draw_id": base["baseDrawId"],
                "pairing_block_id": base["pairingBlockId"],
                "split": base["split"],
                "protected": bool(base["protected"]),
                "replicate_ordinal": int(base["replicateOrdinal"]),
                "n": int(base["n"]),
                "values_by_id_sha256": base["valuesByIdSha256"],
                "initial_occupancy_sha256": base["initialOccupancySha256"],
                "initial_value_sequence_sha256": base["initialValueSequenceSha256"],
                "runtime_seed": metadata["runtimeSeed"],
                "assignment_seed": metadata["assignmentSeed"],
                "tie_break_seed": metadata["tieBreakSeed"],
                "policy_counts_json": json.dumps(
                    metadata["policyCounts"], sort_keys=True, separators=(",", ":")
                ),
                "counts_vector_json": json.dumps(
                    metadata["countsVector"], separators=(",", ":")
                ),
                "policy_assignment_sha256": metadata["policyAssignmentSha256"],
                "achieved_spearman_rho": metadata["achievedSpearmanRho"],
                "achieved_eta_squared": metadata["achievedEtaSquared"],
                "minority_policy": metadata["minorityPolicy"],
                "minority_count": metadata["minorityCount"],
                "event_budget": 1_000_000,
                "architecture": "cell_view",
                "direction_profile": "consensus_ascending",
                "fault_mode": "none",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["condition_id", "replicate_ordinal"])
        .reset_index(drop=True)
    )


def _composition_factorial_markdown(contract_hash: str) -> str:
    return f"""# S03 frozen composition factorial and scenario-generation rules

## Top summary

| Field | Frozen result |
| --- | --- |
| Research step ID | S03 |
| Completion status | Design frozen at 2026-07-17T00:26:26Z before any S03 simulation; execution was pending at freeze time |
| Artifacts written | `preregistration.json`, `scenario_manifest.parquet`, `freeze_record.json`, and this design record; implementation is repository-backed |
| Validation result | Pre-simulation structural validation: 186 conditions, 46,500 unique scenarios, 250 paired unprotected exploratory replicates per condition, exact count rules, and immutable base-value/occupancy hashes |
| Outcome classification | Not yet classified at freeze time; classification rules are fixed below |
| Caveats or blockers | Policy is nominal; positive/negative direction uses an explicitly frozen policy ordinal. Peaks are secondary and not max-null corrected. No design blocker. |
| Recommended next action | Execute only the frozen S03 population, validate and report it, then return control without starting S04. |

Contract SHA-256: `{contract_hash}`.

## Frozen factorial

- Inputs: unique values 1–100 and repeated values 1–10 × 10.
- Pairwise policy sets: Bubble+Insertion, Bubble+Selection, and Insertion+Selection.
- Pairwise compositions: first policy at 10%, 20%, …, 90%; second policy receives the complement.
- Three-way compositions: rotating 34/33/33 balance and three 10/45/45 profiles, one for each rare policy.
- Association profiles: positive, negative, and absent.
- Replication: 250 paired E01 exploratory base draws per input profile.
- Total: 186 conditions and 46,500 simulations.

## Frozen scenario construction

Every condition sharing an input profile and replicate ordinal reuses the same immutable E01 `valuesById`, initial occupancy, and pairing block. Exact counts are assigned by deterministic SHA-256-addressed streams. Positive association assigns increasing value ranks to increasing policy ordinal; negative reverses value rank; absent uniformly permutes identities independently of value. Repeated-value ties use a separate deterministic random tie-break stream. All cells sort ascending, have no faults, use the cell-view reference architecture, and receive a one-million-activation budget.

The pairwise policy ordinal is the printed tuple order. The three-way order is Bubble, Insertion, Selection. Spearman rho checks direction; eta-squared records order-invariant association strength.

## Frozen estimands and inference

Primary magnitude outcomes are corrected final publication aggregation and corrected AUC over accepted-swap progress. Primary kinetic outcomes are accepted swaps and activations. Composition anchors compare 10/90 and 90/10 with 50/50 under absent association. Correlation anchors compare positive and negative with absent at 50/50 pairs and balanced three-way composition. Paired 10,000-draw bootstrap intervals use Bonferroni 95% family coverage.

Run-level peaks, peak time, fraction above the static composition null, and minority homotypy are secondary. Condition scalar intervals use 10,000 whole-run bootstrap draws. Trajectory bands are pointwise. Neither is a dynamic maximum-null result.

## Frozen success rules

Support requires complete validated execution plus: (1) at least one adjusted composition contrast with corrected-AUC difference ≥0.03; (2) at least one adjusted correlation contrast with corrected-final difference ≥0.05; and (3) at least one adjusted accepted-swap contrast differing by ≥5% of the reference mean. A partial pass is constraining/contradictory, and no usable signal is null.

## Extreme-minority audit

Every 10-cell minority has a minority-homotypy lattice resolution of 0.1 and exact random expectation 0.09. S03 will report SD, IQR, unique values, quantiles, and extreme-to-balanced variance ratios rather than smoothing away this discreteness.

## Claim boundaries

This is a clean-room simulator factorial. It does not establish biological effects, affinity, intention, causality outside the constructed contrasts, a static null distribution, a dynamic peak null, or an operational attractor. Alternative clustering metrics remain S04 work and are not part of S03.
"""


def _verify_upstream_manifest(directory: Path, expected_hash: str) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    observed_manifest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for artifact in manifest["artifacts"]:
        path = directory / artifact["path"]
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": artifact["path"],
                "expectedSha256": artifact["sha256"],
                "observedSha256": observed,
                "passed": observed == artifact["sha256"],
            }
        )
    return {
        "manifestPath": str(manifest_path),
        "expectedManifestSha256": expected_hash,
        "observedManifestSha256": observed_manifest,
        "manifestPassed": observed_manifest == expected_hash,
        "artifactChecks": checks,
        "allPassed": observed_manifest == expected_hash
        and all(item["passed"] for item in checks),
    }


def freeze_design(output: Path, cache: Path) -> dict[str, Any]:
    if (cache / "composition_sweep.jsonl").exists():
        raise FileExistsError("cannot freeze S03 after a simulation checkpoint exists")
    output.mkdir(parents=True, exist_ok=True)
    contract_bytes = CONTRACT_PATH.read_bytes()
    contract = json.loads(contract_bytes)
    contract_hash = sha256_file(CONTRACT_PATH)
    upstream = {
        "S01": _verify_upstream_manifest(S01_DIR, S01_MANIFEST_SHA256),
        "S02": _verify_upstream_manifest(S02_DIR, S02_MANIFEST_SHA256),
    }
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream immutability failed before S03 freeze")
    tasks = build_tasks()
    manifest = _scenario_manifest_frame(tasks)
    _write_parquet(manifest, output / "scenario_manifest.parquet")
    write_json(output / "preregistration.json", contract)
    (output / "composition_factorial.md").write_text(
        _composition_factorial_markdown(contract_hash), encoding="utf-8"
    )
    record = {
        "schema": "e04.s03.freeze_record.v1",
        "researchStepId": "S03",
        "frozenAtUtc": contract["frozenAtUtc"],
        "frozenBeforeSimulation": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "scenarioManifestPath": str(output / "scenario_manifest.parquet"),
        "scenarioManifestSha256": sha256_file(output / "scenario_manifest.parquet"),
        "conditionCount": int(manifest.condition_id.nunique()),
        "scenarioCount": len(manifest),
        "simulationCheckpointAbsentAtFreeze": True,
        "upstreamImmutability": upstream,
    }
    write_json(output / "freeze_record.json", record)
    return record


def assert_frozen(output: Path) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text())
    artifact = json.loads((output / "preregistration.json").read_text())
    record = json.loads((output / "freeze_record.json").read_text())
    if contract != artifact:
        raise AssertionError(
            "artifact preregistration differs from repository contract"
        )
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("frozen S03 contract hash changed")
    if record["scenarioManifestSha256"] != sha256_file(
        output / "scenario_manifest.parquet"
    ):
        raise AssertionError("frozen scenario manifest hash changed")
    if not record["frozenBeforeSimulation"]:
        raise AssertionError("S03 freeze record is not pre-simulation")
    return record


def _worker(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    scenario, metadata = materialize_sweep_scenario(condition, task["base"])
    if scenario.scenario_id != task["scenarioId"]:
        raise ValueError("S03 scenario identity failed rematerialization")
    if metadata["scenarioJsonSha256"] != task["scenarioJsonSha256"]:
        raise ValueError("S03 scenario JSON hash failed rematerialization")
    result = run_reference_summary(
        scenario,
        family="s03_composition_factorial",
        retain_raw_trace=False,
    )
    corrected = np.asarray(
        [
            float(point["publication_aggregation"])
            - float(result["publication_aggregation_null"])
            for point in result["curve_grid"]
        ],
        dtype=np.float64,
    )
    peak_index = int(np.argmax(corrected))
    final_edges = json.loads(result["final_policy_edge_counts_json"])
    minority_policy = metadata["minorityPolicy"]
    minority_count = metadata["minorityCount"]
    minority_rate: float | None = None
    minority_excess: float | None = None
    minority_resolution: float | None = None
    if minority_policy is not None and minority_count is not None:
        minority_rate = final_edges[minority_policy] / minority_count
        minority_excess = minority_rate - (minority_count - 1) / len(scenario.cells)
        minority_resolution = 1 / minority_count
    result.update(
        {
            "schema_version": OUTPUT_SCHEMA,
            "research_step_id": "S03",
            "condition_id": condition.condition_id,
            "input_profile": condition.input_profile,
            "policy_set_label": condition.policy_set_label,
            "policies_json": json.dumps(condition.policies, separators=(",", ":")),
            "composition_class": condition.composition_class,
            "composition_profile": condition.composition_profile,
            "first_policy_count": condition.first_policy_count,
            "rare_policy": condition.rare_policy,
            "correlation_profile": condition.correlation_profile,
            "base_draw_id": task["base"]["baseDrawId"],
            "pairing_block_id": task["base"]["pairingBlockId"],
            "split": task["base"]["split"],
            "protected": bool(task["base"]["protected"]),
            "replicate_ordinal": int(task["base"]["replicateOrdinal"]),
            "scenario_json_sha256": metadata["scenarioJsonSha256"],
            "policy_assignment_sha256": metadata["policyAssignmentSha256"],
            "achieved_spearman_rho": metadata["achievedSpearmanRho"],
            "achieved_eta_squared": metadata["achievedEtaSquared"],
            "minority_policy": minority_policy,
            "minority_count": minority_count,
            "corrected_initial_publication_aggregation": corrected[0],
            "corrected_final_publication_aggregation": corrected[-1],
            "corrected_peak_of_run": corrected[peak_index],
            "corrected_peak_progress_of_run": float(GRID[peak_index]),
            "corrected_auc_over_progress": float(np.trapezoid(corrected, GRID)),
            "fraction_grid_above_composition_null": float(np.mean(corrected > 0)),
            "final_minority_homotypy_rate": minority_rate,
            "final_minority_homotypy_excess": minority_excess,
            "minority_homotypy_resolution": minority_resolution,
            "run_id": "e04s03:"
            + hashlib.sha256(
                f"E04-S03-clean-room|{scenario.scenario_id}".encode()
            ).hexdigest(),
        }
    )
    result["raw_trace"] = None
    return result


def _read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["scenario_id"]] = row
    return rows


def run_population(
    tasks: Sequence[Mapping[str, Any]], checkpoint: Path, *, workers: int = 8
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_checkpoint(checkpoint)
    expected = {str(task["scenarioId"]) for task in tasks}
    if not set(completed) <= expected:
        raise ValueError("checkpoint contains scenarios outside frozen S03")
    pending = [task for task in tasks if task["scenarioId"] not in completed]
    pending_iterator = iter(pending)
    max_in_flight = workers * 4
    with checkpoint.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            active: dict[Any, Mapping[str, Any]] = {}
            for _ in range(min(max_in_flight, len(pending))):
                task = next(pending_iterator)
                active[executor.submit(_worker, task)] = task
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    result = future.result()
                    handle.write(
                        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    handle.flush()
                    completed[result["scenario_id"]] = result
                    try:
                        next_task = next(pending_iterator)
                    except StopIteration:
                        next_task = None
                    if next_task is not None:
                        active[executor.submit(_worker, next_task)] = next_task
                    count = len(completed)
                    if count % 250 == 0 or count == len(tasks):
                        print(
                            f"E04 S03 progress {count}/{len(tasks)}",
                            file=sys.stderr,
                            flush=True,
                        )
    if set(completed) != expected:
        raise ValueError("S03 run accounting mismatch")
    return {
        "schema": "e04.s03.run_accounting.v1",
        "researchStepId": "S03",
        "intendedRuns": len(tasks),
        "completedCheckpointRows": len(completed),
        "pendingAtStart": len(pending),
        "workers": workers,
    }


def _selected_replay_tasks(
    tasks: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected = []
    for input_profile in INPUT_CODES:
        for correlation in CORRELATIONS:
            wanted = (
                ("Bubble+Insertion", "p10_90"),
                ("Bubble+Selection", "p50_50"),
                ("Bubble+Insertion+Selection", "rare_selection_10"),
            )
            for policy_set, composition in wanted:
                match = next(
                    task
                    for task in tasks
                    if task["condition"]["inputProfile"] == input_profile
                    and task["condition"]["correlationProfile"] == correlation
                    and task["condition"]["policySetLabel"] == policy_set
                    and task["condition"]["compositionProfile"] == composition
                    and int(task["base"]["replicateOrdinal"]) == 0
                )
                selected.append(match)
    if len(selected) != 18:
        raise AssertionError("S03 replay selection must contain 18 scenarios")
    return selected


def replay_selected(
    tasks: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    output: Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    canonical = _read_checkpoint(checkpoint)
    selected = _selected_replay_tasks(tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        reruns = list(executor.map(_worker, selected))
    fields = (
        "scenario_id",
        "stop_reason",
        "activation_count",
        "successful_swap_count",
        "initial_state_hash",
        "final_state_hash",
        "trajectory_sha256",
        "policy_counts_json",
        "publication_aggregation_null",
        "corrected_final_publication_aggregation",
        "corrected_peak_of_run",
        "corrected_auc_over_progress",
        "final_policy_edge_counts_json",
    )
    rows = []
    for rerun in reruns:
        original = canonical[rerun["scenario_id"]]
        mismatches = [field for field in fields if original[field] != rerun[field]]
        rows.append(
            {
                "scenarioId": rerun["scenario_id"],
                "conditionId": rerun["condition_id"],
                "replicateOrdinal": rerun["replicate_ordinal"],
                "fieldsChecked": len(fields),
                "mismatches": mismatches,
                "passed": not mismatches,
            }
        )
    value = {
        "schema": "e04.s03.deterministic_replay.v1",
        "researchStepId": "S03",
        "selectedScenarios": len(rows),
        "allPassed": all(row["passed"] for row in rows),
        "rows": rows,
    }
    write_json(output / "deterministic_replay.json", value)
    return value


def _checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} checkpoint rows, found {len(rows)}")
    if len({row["scenario_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate scenario in S03 checkpoint")
    return rows


def _aligned_event_budgets(runs: pd.DataFrame, manifest: pd.DataFrame) -> pd.Series:
    """Align frozen scenario budgets to compact run rows with strict identity checks."""
    if manifest.scenario_id.duplicated().any():
        raise ValueError("scenario manifest has duplicate IDs")
    budget_by_scenario = manifest.set_index("scenario_id").event_budget
    budgets = runs.scenario_id.map(budget_by_scenario)
    if budgets.isna().any():
        raise ValueError("run table contains scenarios absent from frozen manifest")
    return budgets.astype(np.int64)


def _run_frame(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"curve_grid", "raw_trace"}
        }
        for result in results
    ]
    frame = (
        pd.DataFrame(rows)
        .sort_values(["condition_id", "replicate_ordinal"])
        .reset_index(drop=True)
    )
    if len(frame) != EXPECTED_RUNS or frame.run_id.duplicated().any():
        raise ValueError("S03 run frame accounting mismatch")
    return frame


def _bootstrap_scalar_means(
    values: np.ndarray, address: str, *, draws: int = BOOTSTRAP_DRAWS
) -> np.ndarray:
    if values.ndim == 1:
        values = values[:, None]
    n = values.shape[0]
    seed = derive_s03_seed("bootstrap", address)
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    weights = generator.multinomial(n, [1 / n] * n, size=draws)
    return weights @ values / n


def _response_and_condition_summaries(
    results: Sequence[Mapping[str, Any]], run_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        by_condition.setdefault(str(result["condition_id"]), []).append(result)
    response_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    scalar_columns = (
        "corrected_final_publication_aggregation",
        "corrected_auc_over_progress",
        "corrected_peak_of_run",
        "corrected_peak_progress_of_run",
        "fraction_grid_above_composition_null",
        "successful_swap_count",
        "activation_count",
    )
    t_critical = float(stats.t.ppf(0.975, REPLICATES - 1))
    for condition_id, condition_results in sorted(by_condition.items()):
        condition_results = sorted(
            condition_results, key=lambda row: int(row["replicate_ordinal"])
        )
        meta = condition_results[0]
        raw = np.asarray(
            [
                [float(point["publication_aggregation"]) for point in row["curve_grid"]]
                for row in condition_results
            ]
        )
        sortedness = np.asarray(
            [
                [
                    float(point["reference_sortedness_percent"])
                    for point in row["curve_grid"]
                ]
                for row in condition_results
            ]
        )
        baselines = np.asarray(
            [float(row["publication_aggregation_null"]) for row in condition_results]
        )
        corrected = raw - baselines[:, None]
        for grid_index, progress in enumerate(GRID):
            raw_values = raw[:, grid_index]
            corrected_values = corrected[:, grid_index]
            sorted_values = sortedness[:, grid_index]
            response_rows.append(
                {
                    "schema_version": "e04.s03.response_surface.v1",
                    "research_step_id": "S03",
                    "condition_id": condition_id,
                    "input_profile": meta["input_profile"],
                    "policy_set_label": meta["policy_set_label"],
                    "composition_class": meta["composition_class"],
                    "composition_profile": meta["composition_profile"],
                    "first_policy_count": meta["first_policy_count"],
                    "rare_policy": meta["rare_policy"],
                    "correlation_profile": meta["correlation_profile"],
                    "grid_index": grid_index,
                    "accepted_swap_progress": progress,
                    "runs": len(condition_results),
                    "mean_publication_aggregation": float(raw_values.mean()),
                    "sd_publication_aggregation": float(raw_values.std(ddof=1)),
                    "mean_composition_baseline": float(baselines.mean()),
                    "mean_corrected_aggregation": float(corrected_values.mean()),
                    "sd_corrected_aggregation": float(corrected_values.std(ddof=1)),
                    "corrected_pointwise_ci95_low": float(
                        corrected_values.mean()
                        - t_critical
                        * corrected_values.std(ddof=1)
                        / math.sqrt(REPLICATES)
                    ),
                    "corrected_pointwise_ci95_high": float(
                        corrected_values.mean()
                        + t_critical
                        * corrected_values.std(ddof=1)
                        / math.sqrt(REPLICATES)
                    ),
                    "mean_reference_sortedness_percent": float(sorted_values.mean()),
                }
            )
        scalar_frame = run_frame[run_frame.condition_id.eq(condition_id)].sort_values(
            "replicate_ordinal"
        )
        scalar_values = scalar_frame[list(scalar_columns)].to_numpy(dtype=np.float64)
        boot = _bootstrap_scalar_means(scalar_values, f"condition/{condition_id}")
        means = scalar_values.mean(axis=0)
        sds = scalar_values.std(axis=0, ddof=1)
        mean_curve = corrected.mean(axis=0)
        mean_peak_index = int(np.argmax(mean_curve))
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "input_profile": meta["input_profile"],
            "policy_set_label": meta["policy_set_label"],
            "composition_class": meta["composition_class"],
            "composition_profile": meta["composition_profile"],
            "first_policy_count": meta["first_policy_count"],
            "rare_policy": meta["rare_policy"],
            "correlation_profile": meta["correlation_profile"],
            "runs": len(condition_results),
            "mean_achieved_spearman_rho": float(
                scalar_frame.achieved_spearman_rho.mean()
            ),
            "mean_achieved_eta_squared": float(
                scalar_frame.achieved_eta_squared.mean()
            ),
            "mean_composition_baseline": float(baselines.mean()),
            "peak_of_mean_corrected_curve": float(mean_curve[mean_peak_index]),
            "peak_of_mean_progress": float(GRID[mean_peak_index]),
        }
        for index, column in enumerate(scalar_columns):
            row[f"mean_{column}"] = float(means[index])
            row[f"sd_{column}"] = float(sds[index])
            row[f"{column}_ci95_low"] = float(np.quantile(boot[:, index], 0.025))
            row[f"{column}_ci95_high"] = float(np.quantile(boot[:, index], 0.975))
        summary_rows.append(row)
    response = (
        pd.DataFrame(response_rows)
        .sort_values(["condition_id", "grid_index"])
        .reset_index(drop=True)
    )
    summary = (
        pd.DataFrame(summary_rows).sort_values("condition_id").reset_index(drop=True)
    )
    return response, summary


def _condition_id(
    conditions: Sequence[SweepCondition],
    *,
    input_profile: str,
    policy_set_label: str,
    composition_profile: str,
    correlation_profile: str,
) -> str:
    matches = [
        item.condition_id
        for item in conditions
        if item.input_profile == input_profile
        and item.policy_set_label == policy_set_label
        and item.composition_profile == composition_profile
        and item.correlation_profile == correlation_profile
    ]
    if len(matches) != 1:
        raise ValueError(f"condition lookup returned {matches}")
    return matches[0]


def _paired_contrast(
    run_frame: pd.DataFrame,
    *,
    family: str,
    label: str,
    alternative_id: str,
    reference_id: str,
    metric: str,
    confidence_level: float,
    practical_threshold: float | None,
    relative_threshold: float | None,
) -> dict[str, Any]:
    alternative = run_frame[run_frame.condition_id.eq(alternative_id)][
        ["pairing_block_id", metric]
    ].rename(columns={metric: "alternative"})
    reference = run_frame[run_frame.condition_id.eq(reference_id)][
        ["pairing_block_id", metric]
    ].rename(columns={metric: "reference"})
    paired = alternative.merge(reference, on="pairing_block_id", validate="one_to_one")
    if len(paired) != REPLICATES:
        raise ValueError(f"paired contrast {label}/{metric} has {len(paired)} rows")
    differences = paired.alternative.to_numpy(float) - paired.reference.to_numpy(float)
    boot = _bootstrap_scalar_means(
        differences,
        f"primary/{family}/{label}/{metric}",
    )[:, 0]
    alpha = 1 - confidence_level
    difference = float(differences.mean())
    reference_mean = float(paired.reference.mean())
    relative = difference / reference_mean if reference_mean != 0 else None
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    interval_excludes_zero = low > 0 or high < 0
    practical_pass = True
    if practical_threshold is not None:
        practical_pass = abs(difference) >= practical_threshold
    if relative_threshold is not None:
        practical_pass = relative is not None and abs(relative) >= relative_threshold
    return {
        "contrast_family": family,
        "contrast_label": label,
        "alternative_condition_id": alternative_id,
        "reference_condition_id": reference_id,
        "metric": metric,
        "pairs": len(paired),
        "confidence_level": confidence_level,
        "alternative_mean": float(paired.alternative.mean()),
        "reference_mean": reference_mean,
        "mean_difference": difference,
        "relative_difference": relative,
        "ci_low": low,
        "ci_high": high,
        "interval_excludes_zero": interval_excludes_zero,
        "practical_threshold_absolute": practical_threshold,
        "practical_threshold_relative": relative_threshold,
        "practical_pass": practical_pass,
        "criterion_pass": interval_excludes_zero and practical_pass,
    }


def _primary_contrasts(run_frame: pd.DataFrame) -> pd.DataFrame:
    conditions = build_conditions()
    rows: list[dict[str, Any]] = []
    composition_confidence = 1 - 0.05 / 12
    correlation_confidence = 1 - 0.05 / 16
    metrics = (
        "corrected_auc_over_progress",
        "corrected_final_publication_aggregation",
        "successful_swap_count",
        "activation_count",
    )
    for input_profile in INPUT_CODES:
        for policies in POLICY_PAIRS:
            policy_set = "+".join(policies)
            reference = _condition_id(
                conditions,
                input_profile=input_profile,
                policy_set_label=policy_set,
                composition_profile="p50_50",
                correlation_profile="absent",
            )
            for count in (10, 90):
                alternative = _condition_id(
                    conditions,
                    input_profile=input_profile,
                    policy_set_label=policy_set,
                    composition_profile=f"p{count:02d}_{100 - count:02d}",
                    correlation_profile="absent",
                )
                label = f"{input_profile}/{policy_set}/p{count:02d}_vs_p50/absent"
                for metric in metrics:
                    rows.append(
                        _paired_contrast(
                            run_frame,
                            family="composition_anchor",
                            label=label,
                            alternative_id=alternative,
                            reference_id=reference,
                            metric=metric,
                            confidence_level=composition_confidence,
                            practical_threshold=(
                                0.03
                                if metric == "corrected_auc_over_progress"
                                else None
                            ),
                            relative_threshold=(
                                0.05 if metric == "successful_swap_count" else None
                            ),
                        )
                    )
    structures = [tuple(pair) for pair in POLICY_PAIRS] + [THREE_POLICIES]
    for input_profile in INPUT_CODES:
        for policies in structures:
            policy_set = "+".join(policies)
            composition = "p50_50" if len(policies) == 2 else "balanced_rotating"
            reference = _condition_id(
                conditions,
                input_profile=input_profile,
                policy_set_label=policy_set,
                composition_profile=composition,
                correlation_profile="absent",
            )
            for correlation in ("positive", "negative"):
                alternative = _condition_id(
                    conditions,
                    input_profile=input_profile,
                    policy_set_label=policy_set,
                    composition_profile=composition,
                    correlation_profile=correlation,
                )
                label = f"{input_profile}/{policy_set}/{composition}/{correlation}_vs_absent"
                for metric in metrics:
                    rows.append(
                        _paired_contrast(
                            run_frame,
                            family="correlation_anchor",
                            label=label,
                            alternative_id=alternative,
                            reference_id=reference,
                            metric=metric,
                            confidence_level=correlation_confidence,
                            practical_threshold=(
                                0.05
                                if metric == "corrected_final_publication_aggregation"
                                else None
                            ),
                            relative_threshold=(
                                0.05 if metric == "successful_swap_count" else None
                            ),
                        )
                    )
    frame = pd.DataFrame(rows)
    if len(frame) != 112:
        raise AssertionError(f"expected 112 primary contrast rows, found {len(frame)}")
    return frame


def _correlation_fidelity(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition_id, group in manifest.groupby("condition_id", sort=True):
        rho = group.achieved_spearman_rho.to_numpy(float)
        mean = float(rho.mean())
        sem = float(rho.std(ddof=1) / math.sqrt(len(rho)))
        critical = float(stats.t.ppf(0.975, len(rho) - 1))
        rows.append(
            {
                "condition_id": condition_id,
                "input_profile": group.input_profile.iloc[0],
                "policy_set_label": group.policy_set_label.iloc[0],
                "composition_profile": group.composition_profile.iloc[0],
                "correlation_profile": group.correlation_profile.iloc[0],
                "runs": len(group),
                "mean_spearman_rho": mean,
                "sd_spearman_rho": float(rho.std(ddof=1)),
                "min_spearman_rho": float(rho.min()),
                "max_spearman_rho": float(rho.max()),
                "rho_mean_ci95_low": mean - critical * sem,
                "rho_mean_ci95_high": mean + critical * sem,
                "rho_mean_ci95_contains_zero": mean - critical * sem
                <= 0
                <= mean + critical * sem,
                "mean_eta_squared": float(group.achieved_eta_squared.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("condition_id").reset_index(drop=True)


def _interpretation_changes(
    response: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    """Quantify how the exact composition baseline changes zero-excess claims."""
    final = response[response.grid_index.eq(len(GRID) - 1)][
        [
            "condition_id",
            "mean_publication_aggregation",
            "mean_composition_baseline",
            "mean_corrected_aggregation",
        ]
    ].rename(
        columns={
            "mean_publication_aggregation": "mean_raw_final",
            "mean_composition_baseline": "mean_exact_composition_baseline",
            "mean_corrected_aggregation": "mean_corrected_final",
        }
    )
    columns = [
        "condition_id",
        "input_profile",
        "policy_set_label",
        "composition_class",
        "composition_profile",
        "first_policy_count",
        "rare_policy",
        "correlation_profile",
        "runs",
        "mean_corrected_auc_over_progress",
        "corrected_final_publication_aggregation_ci95_low",
        "corrected_final_publication_aggregation_ci95_high",
    ]
    frame = summary[columns].merge(final, on="condition_id", validate="one_to_one")
    frame["mean_raw_auc_over_progress"] = (
        frame.mean_corrected_auc_over_progress + frame.mean_exact_composition_baseline
    )
    frame["universal_half_final_excess"] = frame.mean_raw_final - 0.5
    frame["exact_composition_final_excess"] = frame.mean_corrected_final
    frame["final_excess_change_after_correction"] = (
        frame.exact_composition_final_excess - frame.universal_half_final_excess
    )
    frame["universal_half_auc_excess"] = frame.mean_raw_auc_over_progress - 0.5
    frame["exact_composition_auc_excess"] = frame.mean_corrected_auc_over_progress
    frame["auc_excess_change_after_correction"] = (
        frame.exact_composition_auc_excess - frame.universal_half_auc_excess
    )
    frame["universal_half_final_positive"] = frame.universal_half_final_excess.gt(0)
    frame["exact_composition_final_positive"] = frame.exact_composition_final_excess.gt(
        0
    )
    frame["final_interpretation_changed"] = (
        frame.universal_half_final_positive != frame.exact_composition_final_positive
    )
    frame["corrected_final_ci95_excludes_zero"] = (
        frame.corrected_final_publication_aggregation_ci95_low.gt(0)
        | frame.corrected_final_publication_aggregation_ci95_high.lt(0)
    )
    frame["final_interpretation_category"] = np.select(
        [
            frame.universal_half_final_positive
            & frame.exact_composition_final_positive,
            ~frame.universal_half_final_positive
            & frame.exact_composition_final_positive,
            frame.universal_half_final_positive
            & ~frame.exact_composition_final_positive,
        ],
        [
            "both_positive",
            "composition_positive_universal_nonpositive",
            "universal_positive_composition_nonpositive",
        ],
        default="both_nonpositive",
    )
    if len(frame) != EXPECTED_CONDITIONS:
        raise AssertionError("interpretation table must contain every S03 condition")
    return frame.sort_values("condition_id").reset_index(drop=True)


def _variance_ratio_bootstrap(
    extreme: np.ndarray, balanced: np.ndarray, address: str
) -> tuple[float, float, float]:
    if len(extreme) != len(balanced):
        raise ValueError("variance ratio requires paired populations")
    n = len(extreme)
    observed_denominator = float(np.var(balanced, ddof=1))
    if observed_denominator <= VARIANCE_TOLERANCE:
        return math.nan, math.nan, math.nan
    seed = derive_s03_seed("variance_bootstrap", address)
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    ratios = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        size = min(500, BOOTSTRAP_DRAWS - start)
        indices = generator.integers(0, n, size=(size, n))
        numerator = np.var(extreme[indices], axis=1, ddof=1)
        denominator = np.var(balanced[indices], axis=1, ddof=1)
        ratios[start : start + size] = np.divide(
            numerator,
            denominator,
            out=np.full(size, np.nan),
            where=denominator > VARIANCE_TOLERANCE,
        )
    observed = float(np.var(extreme, ddof=1)) / observed_denominator
    finite = ratios[np.isfinite(ratios)]
    if not len(finite):
        return observed, math.nan, math.nan
    return (
        observed,
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def _minority_discreteness(run_frame: pd.DataFrame) -> pd.DataFrame:
    rare = run_frame[run_frame.minority_count.eq(10)].copy()
    rows = []
    conditions = build_conditions()
    for condition_id, group in rare.groupby("condition_id", sort=True):
        meta = group.iloc[0]
        balanced_profile = (
            "p50_50" if meta.composition_class == "pairwise" else "balanced_rotating"
        )
        balanced_id = _condition_id(
            conditions,
            input_profile=meta.input_profile,
            policy_set_label=meta.policy_set_label,
            composition_profile=balanced_profile,
            correlation_profile=meta.correlation_profile,
        )
        balanced = run_frame[run_frame.condition_id.eq(balanced_id)].sort_values(
            "replicate_ordinal"
        )
        extreme = group.sort_values("replicate_ordinal")
        peak_ratio = _variance_ratio_bootstrap(
            extreme.corrected_peak_of_run.to_numpy(float),
            balanced.corrected_peak_of_run.to_numpy(float),
            f"{condition_id}/peak",
        )
        final_ratio = _variance_ratio_bootstrap(
            extreme.corrected_final_publication_aggregation.to_numpy(float),
            balanced.corrected_final_publication_aggregation.to_numpy(float),
            f"{condition_id}/final",
        )
        minority = extreme.final_minority_homotypy_rate.to_numpy(float)
        unique = np.unique(minority)
        observed_steps = np.diff(unique)
        rows.append(
            {
                "condition_id": condition_id,
                "balanced_reference_condition_id": balanced_id,
                "input_profile": meta.input_profile,
                "policy_set_label": meta.policy_set_label,
                "composition_profile": meta.composition_profile,
                "correlation_profile": meta.correlation_profile,
                "minority_policy": meta.minority_policy,
                "minority_count": 10,
                "runs": len(group),
                "theoretical_homotypy_resolution": 0.1,
                "theoretical_random_homotypy_expectation": 0.09,
                "mean_final_minority_homotypy_rate": float(minority.mean()),
                "sd_final_minority_homotypy_rate": float(minority.std(ddof=1)),
                "iqr_final_minority_homotypy_rate": float(
                    np.quantile(minority, 0.75) - np.quantile(minority, 0.25)
                ),
                "q025_final_minority_homotypy_rate": float(
                    np.quantile(minority, 0.025)
                ),
                "q975_final_minority_homotypy_rate": float(
                    np.quantile(minority, 0.975)
                ),
                "unique_final_minority_homotypy_values": len(unique),
                "minimum_observed_nonzero_step": (
                    float(observed_steps[observed_steps > 1e-12].min())
                    if np.any(observed_steps > 1e-12)
                    else None
                ),
                "mean_final_minority_homotypy_excess": float(
                    extreme.final_minority_homotypy_excess.mean()
                ),
                "sd_corrected_peak": float(extreme.corrected_peak_of_run.std(ddof=1)),
                "sd_corrected_final": float(
                    extreme.corrected_final_publication_aggregation.std(ddof=1)
                ),
                "peak_variance_ratio_vs_balanced": peak_ratio[0],
                "peak_variance_ratio_estimable": math.isfinite(peak_ratio[0]),
                "peak_variance_ratio_ci95_low": peak_ratio[1],
                "peak_variance_ratio_ci95_high": peak_ratio[2],
                "final_variance_ratio_vs_balanced": final_ratio[0],
                "final_variance_ratio_estimable": math.isfinite(final_ratio[0]),
                "final_variance_ratio_ci95_low": final_ratio[1],
                "final_variance_ratio_ci95_high": final_ratio[2],
            }
        )
    frame = pd.DataFrame(rows).sort_values("condition_id").reset_index(drop=True)
    if len(frame) != 54:
        raise AssertionError(f"expected 54 minority profiles, found {len(frame)}")
    return frame


def _plot_pairwise(summary: pd.DataFrame, output: Path, input_profile: str) -> None:
    source = summary[
        summary.input_profile.eq(input_profile)
        & summary.composition_class.eq("pairwise")
    ]
    colors = {"positive": "#b23a48", "negative": "#3c5aa6", "absent": "#3b7d5b"}
    fig, axes = plt.subplots(
        3, 3, figsize=(15, 11), sharex=True, constrained_layout=True
    )
    for row, policy_set in enumerate("+".join(pair) for pair in POLICY_PAIRS):
        for column, correlation in enumerate(CORRELATIONS):
            group = source[
                source.policy_set_label.eq(policy_set)
                & source.correlation_profile.eq(correlation)
            ].sort_values("first_policy_count")
            axis = axes[row, column]
            x = group.first_policy_count.to_numpy(float)
            color = colors[correlation]
            auc = group.mean_corrected_auc_over_progress.to_numpy(float)
            auc_low = group.corrected_auc_over_progress_ci95_low.to_numpy(float)
            auc_high = group.corrected_auc_over_progress_ci95_high.to_numpy(float)
            final = group.mean_corrected_final_publication_aggregation.to_numpy(float)
            final_low = group.corrected_final_publication_aggregation_ci95_low.to_numpy(
                float
            )
            final_high = (
                group.corrected_final_publication_aggregation_ci95_high.to_numpy(float)
            )
            axis.plot(x, auc, color=color, marker="o", label="Corrected AUC")
            axis.fill_between(x, auc_low, auc_high, color=color, alpha=0.12)
            axis.plot(
                x,
                final,
                color="#6a4c93",
                marker="s",
                linestyle="--",
                label="Corrected final",
            )
            axis.fill_between(x, final_low, final_high, color="#6a4c93", alpha=0.10)
            axis.axhline(0, color="#333333", linewidth=0.8, linestyle=":")
            axis.grid(alpha=0.2)
            axis.set_title(f"{policy_set} — {correlation}")
            if row == 2:
                axis.set_xlabel(f"{policy_set.split('+')[0]} count (%)")
            if column == 0:
                axis.set_ylabel("Composition-corrected score")
    axes[0, 0].legend(fontsize=8)
    title = (
        "Unique values 1–100"
        if input_profile == "unique_1_100"
        else "Repeated values 1–10 × 10"
    )
    fig.suptitle(f"E04 S03 pairwise composition response — {title}", fontsize=15)
    stem = (
        "pairwise_response_unique"
        if input_profile == "unique_1_100"
        else "pairwise_response_repeated"
    )
    fig.savefig(output / f"{stem}.png", dpi=180)
    fig.savefig(output / f"{stem}.svg")
    plt.close(fig)


def _plot_three_way(summary: pd.DataFrame, output: Path) -> None:
    source = summary[summary.composition_class.eq("three_way")].copy()
    profiles = (
        "balanced_rotating",
        "rare_bubble_10",
        "rare_insertion_10",
        "rare_selection_10",
    )
    labels = ("34/33/33", "B10/I45/S45", "B45/I10/S45", "B45/I45/S10")
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8), sharey="row", constrained_layout=True
    )
    for row, input_profile in enumerate(INPUT_CODES):
        for column, correlation in enumerate(CORRELATIONS):
            group = (
                source[
                    source.input_profile.eq(input_profile)
                    & source.correlation_profile.eq(correlation)
                ]
                .set_index("composition_profile")
                .loc[list(profiles)]
            )
            x = np.arange(len(profiles))
            axis = axes[row, column]
            axis.bar(
                x - 0.18,
                group.mean_corrected_auc_over_progress,
                width=0.36,
                color="#3b7d5b",
                label="Corrected AUC",
            )
            axis.bar(
                x + 0.18,
                group.mean_corrected_final_publication_aggregation,
                width=0.36,
                color="#6a4c93",
                label="Corrected final",
            )
            axis.axhline(0, color="#333333", linewidth=0.8)
            axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
            axis.set_title(correlation)
            axis.grid(axis="y", alpha=0.2)
            if column == 0:
                text_label = "Unique" if input_profile == "unique_1_100" else "Repeated"
                axis.set_ylabel(f"{text_label}\nCorrected score")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("E04 S03 balanced and rare-minority three-way responses", fontsize=15)
    fig.savefig(output / "three_way_response.png", dpi=180)
    fig.savefig(output / "three_way_response.svg")
    plt.close(fig)


def _plot_correlation_fidelity(manifest: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    order = ("negative", "absent", "positive")
    colors = ("#3c5aa6", "#3b7d5b", "#b23a48")
    for axis, input_profile in zip(axes, INPUT_CODES):
        data = [
            manifest[
                manifest.input_profile.eq(input_profile)
                & manifest.correlation_profile.eq(profile)
            ].achieved_spearman_rho.to_numpy(float)
            for profile in order
        ]
        plot = axis.boxplot(data, labels=order, showfliers=False, patch_artist=True)
        for patch, color in zip(plot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        axis.axhline(0, color="#222222", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylabel("Achieved policy-ordinal/value Spearman ρ")
        title = (
            "Unique values" if input_profile == "unique_1_100" else "Repeated values"
        )
        axis.set_title(title)
    fig.suptitle("E04 S03 policy–value association fidelity", fontsize=14)
    fig.savefig(output / "correlation_fidelity.png", dpi=180)
    fig.savefig(output / "correlation_fidelity.svg")
    plt.close(fig)


def _plot_minority_variance(minority: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    correlation_x = {"negative": 0, "absent": 1, "positive": 2}
    input_colors = {"unique_1_100": "#bf5b17", "repeated_1_10_x10": "#386cb0"}
    for input_profile, group in minority.groupby("input_profile"):
        x = group.correlation_profile.map(correlation_x).to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(group))
        color = input_colors[input_profile]
        label = "Unique" if input_profile == "unique_1_100" else "Repeated"
        axes[0, 0].scatter(
            x + jitter,
            group.sd_final_minority_homotypy_rate,
            alpha=0.65,
            s=25,
            color=color,
            label=label,
        )
        axes[0, 1].scatter(
            x + jitter,
            group.unique_final_minority_homotypy_values,
            alpha=0.65,
            s=25,
            color=color,
            label=label,
        )
        axes[1, 0].scatter(
            x + jitter,
            group.peak_variance_ratio_vs_balanced,
            alpha=0.65,
            s=25,
            color=color,
            label=label,
        )
        axes[1, 1].scatter(
            x + jitter,
            group.final_variance_ratio_vs_balanced,
            alpha=0.65,
            s=25,
            color=color,
            label=label,
        )
    titles = (
        "Minority homotypy SD (10 cells; 0.1 lattice)",
        "Observed unique minority-homotypy values",
        "Corrected-peak variance ratio vs balanced",
        "Corrected-final variance ratio vs balanced",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_xticks([0, 1, 2], ["negative", "absent", "positive"])
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1, 0].axhline(1, color="#222222", linestyle=":")
    axes[1, 1].axhline(1, color="#222222", linestyle=":")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("E04 S03 extreme-minority discreteness and variance", fontsize=14)
    fig.savefig(output / "extreme_minority_variance.png", dpi=180)
    fig.savefig(output / "extreme_minority_variance.svg")
    plt.close(fig)


def build_artifacts(checkpoint: Path, output: Path) -> dict[str, Any]:
    assert_frozen(output)
    output.mkdir(parents=True, exist_ok=True)
    results = _checkpoint_rows(checkpoint)
    run_frame = _run_frame(results)
    manifest = pd.read_parquet(output / "scenario_manifest.parquet")
    response, summary = _response_and_condition_summaries(results, run_frame)
    contrasts = _primary_contrasts(run_frame)
    fidelity = _correlation_fidelity(manifest)
    interpretation = _interpretation_changes(response, summary)
    minority = _minority_discreteness(run_frame)
    _write_parquet(run_frame, output / "composition_sweep.parquet")
    _write_parquet(response, output / "response_surface.parquet")
    summary.to_csv(output / "condition_summary.csv", index=False)
    contrasts.to_csv(output / "factorial_effects.csv", index=False)
    fidelity.to_csv(output / "correlation_fidelity.csv", index=False)
    interpretation.to_csv(output / "interpretation_changes.csv", index=False)
    minority.to_csv(output / "extreme_minority_variance.csv", index=False)

    composition_pass = bool(
        contrasts[
            contrasts.contrast_family.eq("composition_anchor")
            & contrasts.metric.eq("corrected_auc_over_progress")
        ].criterion_pass.any()
    )
    correlation_pass = bool(
        contrasts[
            contrasts.contrast_family.eq("correlation_anchor")
            & contrasts.metric.eq("corrected_final_publication_aggregation")
        ].criterion_pass.any()
    )
    kinetics_pass = bool(
        contrasts[contrasts.metric.eq("successful_swap_count")].criterion_pass.any()
    )
    execution_pass = bool(
        len(run_frame) == EXPECTED_RUNS
        and run_frame.completed.all()
        and not run_frame.censored.any()
        and run_frame.value_multiset_conserved.all()
        and not run_frame.protected.any()
    )
    supportive = (
        execution_pass and composition_pass and correlation_pass and kinetics_pass
    )
    outcome = "supportive" if supportive else "constraining/contradictory"
    criteria = {
        "schema": "e04.s03.success_criteria.v1",
        "researchStepId": "S03",
        "executionPassed": execution_pass,
        "compositionMagnitudePassed": composition_pass,
        "correlationMagnitudePassed": correlation_pass,
        "kineticsPassed": kinetics_pass,
        "success": supportive,
        "outcomeClassification": outcome,
    }
    write_json(output / "success_criteria.json", criteria)
    accounting = {
        "schema": "e04.s03.run_accounting.v1",
        "researchStepId": "S03",
        "intendedRuns": EXPECTED_RUNS,
        "resultRows": len(run_frame),
        "conditions": int(run_frame.condition_id.nunique()),
        "runsPerConditionMin": int(run_frame.groupby("condition_id").size().min()),
        "runsPerConditionMax": int(run_frame.groupby("condition_id").size().max()),
        "completed": int(run_frame.completed.sum()),
        "censored": int(run_frame.censored.sum()),
        "stopReasons": run_frame.stop_reason.value_counts().sort_index().to_dict(),
        "responseSurfaceRows": len(response),
        "primaryContrastRows": len(contrasts),
        "minorityProfiles": len(minority),
    }
    write_json(output / "run_accounting.json", accounting)
    _plot_pairwise(summary, output, "unique_1_100")
    _plot_pairwise(summary, output, "repeated_1_10_x10")
    _plot_three_way(summary, output)
    _plot_correlation_fidelity(manifest, output)
    _plot_minority_variance(minority, output)
    return {**accounting, **criteria}


def validate_artifacts(output: Path) -> dict[str, Any]:
    freeze = assert_frozen(output)
    manifest = pd.read_parquet(output / "scenario_manifest.parquet")
    runs = pd.read_parquet(output / "composition_sweep.parquet")
    response = pd.read_parquet(output / "response_surface.parquet")
    summary = pd.read_csv(output / "condition_summary.csv")
    contrasts = pd.read_csv(output / "factorial_effects.csv")
    fidelity = pd.read_csv(output / "correlation_fidelity.csv")
    interpretation = pd.read_csv(output / "interpretation_changes.csv")
    minority = pd.read_csv(output / "extreme_minority_variance.csv")
    replay = json.loads((output / "deterministic_replay.json").read_text())
    criteria = json.loads((output / "success_criteria.json").read_text())
    accounting = json.loads((output / "run_accounting.json").read_text())
    upstream = {
        "S01": _verify_upstream_manifest(S01_DIR, S01_MANIFEST_SHA256),
        "S02": _verify_upstream_manifest(S02_DIR, S02_MANIFEST_SHA256),
    }
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": _json_native(detail)}

    check(
        "freeze_contract",
        freeze["frozenBeforeSimulation"]
        and freeze["conditionCount"] == EXPECTED_CONDITIONS
        and freeze["scenarioCount"] == EXPECTED_RUNS,
        [freeze["contractSha256"], freeze["conditionCount"], freeze["scenarioCount"]],
    )
    check(
        "upstream_immutability",
        all(item["allPassed"] for item in upstream.values()),
        {key: item["observedManifestSha256"] for key, item in upstream.items()},
    )
    check(
        "scenario_accounting",
        len(manifest) == EXPECTED_RUNS
        and manifest.condition_id.nunique() == EXPECTED_CONDITIONS
        and manifest.groupby("condition_id").size().eq(REPLICATES).all(),
        [len(manifest), manifest.condition_id.nunique()],
    )
    check(
        "scenario_unique_ids",
        not manifest.scenario_id.duplicated().any()
        and not manifest.scenario_json_sha256.duplicated().any(),
        int(manifest.scenario_id.duplicated().sum()),
    )
    check(
        "unprotected_exploratory_only",
        manifest.split.eq("exploratory").all() and not manifest.protected.any(),
        manifest.split.value_counts().to_dict(),
    )
    pairing = manifest.groupby("pairing_block_id").agg(
        rows=("scenario_id", "size"),
        value_hashes=("initial_value_sequence_sha256", "nunique"),
        occupancy_hashes=("initial_occupancy_sha256", "nunique"),
        input_profiles=("input_profile", "nunique"),
    )
    check(
        "paired_value_construction",
        len(pairing) == 500
        and pairing.rows.eq(93).all()
        and pairing.value_hashes.eq(1).all()
        and pairing.occupancy_hashes.eq(1).all()
        and pairing.input_profiles.eq(1).all(),
        [len(pairing), int(pairing.rows.min()), int(pairing.rows.max())],
    )
    conditions = {item.condition_id: item for item in build_conditions()}
    counts_match = []
    for row in manifest.itertuples(index=False):
        observed = tuple(json.loads(row.counts_vector_json))
        expected = conditions[row.condition_id].counts(int(row.replicate_ordinal))
        counts_match.append(observed == expected and sum(observed) == 100)
    check("exact_counts", all(counts_match), len(counts_match) - sum(counts_match))
    positive = manifest[manifest.correlation_profile.eq("positive")]
    negative = manifest[manifest.correlation_profile.eq("negative")]
    absent_fidelity = fidelity[fidelity.correlation_profile.eq("absent")]
    check(
        "association_direction",
        positive.achieved_spearman_rho.gt(0).all()
        and negative.achieved_spearman_rho.lt(0).all(),
        [
            float(positive.achieved_spearman_rho.min()),
            float(negative.achieved_spearman_rho.max()),
        ],
    )
    check(
        "absent_association_centered",
        absent_fidelity.mean_spearman_rho.abs().le(0.03).all(),
        float(absent_fidelity.mean_spearman_rho.abs().max()),
    )
    check(
        "absent_interval_coverage",
        absent_fidelity.rho_mean_ci95_contains_zero.mean() >= 0.90,
        float(absent_fidelity.rho_mean_ci95_contains_zero.mean()),
    )
    check(
        "run_accounting",
        len(runs) == EXPECTED_RUNS
        and runs.condition_id.nunique() == EXPECTED_CONDITIONS
        and runs.groupby("condition_id").size().eq(REPLICATES).all(),
        [len(runs), runs.condition_id.nunique()],
    )
    check(
        "run_ids_unique",
        not runs.run_id.duplicated().any(),
        int(runs.run_id.duplicated().sum()),
    )
    check(
        "all_runs_complete",
        runs.completed.all()
        and not runs.censored.any()
        and runs.stop_reason.eq("complete").all(),
        runs.stop_reason.value_counts().to_dict(),
    )
    check(
        "value_conservation",
        runs.value_multiset_conserved.all() and runs.final_consensus_ordered.all(),
        [
            int((~runs.value_multiset_conserved).sum()),
            int((~runs.final_consensus_ordered).sum()),
        ],
    )
    check(
        "event_budget_respected",
        runs.activation_count.le(_aligned_event_budgets(runs, manifest)).all(),
        int(runs.activation_count.max()),
    )
    expected_nulls = runs.policy_counts_json.map(
        lambda value: expected_paper_aggregation(tuple(json.loads(value).values()))
    )
    check(
        "validated_composition_correction",
        np.allclose(runs.publication_aggregation_null, expected_nulls, atol=1e-15),
        float((runs.publication_aggregation_null - expected_nulls).abs().max()),
    )
    check(
        "correction_identities",
        np.allclose(
            runs.corrected_initial_publication_aggregation,
            runs.initial_publication_aggregation - runs.publication_aggregation_null,
            atol=1e-15,
        )
        and np.allclose(
            runs.corrected_final_publication_aggregation,
            runs.final_publication_aggregation - runs.publication_aggregation_null,
            atol=1e-15,
        ),
        None,
    )
    check(
        "response_surface_accounting",
        len(response) == EXPECTED_CONDITIONS * len(GRID)
        and response.groupby("condition_id").size().eq(len(GRID)).all(),
        len(response),
    )
    check(
        "response_grid_complete",
        response.groupby("condition_id")
        .grid_index.apply(lambda values: set(values) == set(range(101)))
        .all(),
        None,
    )
    check(
        "condition_summary_accounting",
        len(summary) == EXPECTED_CONDITIONS and summary.runs.eq(REPLICATES).all(),
        len(summary),
    )
    scalar_interval_columns = [
        column.removesuffix("_ci95_low")
        for column in summary.columns
        if column.endswith("_ci95_low")
    ]
    interval_order = all(
        (summary[f"{base}_ci95_low"] <= summary[f"{base}_ci95_high"]).all()
        for base in scalar_interval_columns
    )
    check("condition_interval_order", interval_order, len(scalar_interval_columns))
    check(
        "interpretation_accounting",
        len(interpretation) == EXPECTED_CONDITIONS
        and not interpretation.condition_id.duplicated().any(),
        len(interpretation),
    )
    check(
        "interpretation_correction_identities",
        np.allclose(
            interpretation.mean_raw_final
            - interpretation.mean_exact_composition_baseline,
            interpretation.mean_corrected_final,
            atol=1e-15,
        )
        and np.allclose(
            interpretation.final_excess_change_after_correction,
            0.5 - interpretation.mean_exact_composition_baseline,
            atol=1e-15,
        ),
        int(interpretation.final_interpretation_changed.sum()),
    )
    check(
        "primary_contrast_accounting",
        len(contrasts) == 112
        and (contrasts.contrast_family.eq("composition_anchor").sum() == 48)
        and (contrasts.contrast_family.eq("correlation_anchor").sum() == 64),
        contrasts.contrast_family.value_counts().to_dict(),
    )
    check(
        "primary_contrast_intervals",
        (contrasts.ci_low <= contrasts.ci_high).all(),
        None,
    )
    check(
        "success_criteria_consistent",
        criteria["executionPassed"]
        and criteria["compositionMagnitudePassed"]
        and criteria["correlationMagnitudePassed"]
        and criteria["kineticsPassed"]
        and criteria["success"],
        criteria,
    )
    check(
        "minority_profile_accounting",
        len(minority) == 54 and minority.minority_count.eq(10).all(),
        len(minority),
    )
    check(
        "minority_lattice_resolution",
        np.allclose(minority.theoretical_homotypy_resolution, 0.1)
        and np.allclose(minority.theoretical_random_homotypy_expectation, 0.09)
        and minority.minimum_observed_nonzero_step.dropna()
        .sub(0.1)
        .abs()
        .le(1e-12)
        .all(),
        sorted(minority.minimum_observed_nonzero_step.dropna().unique()),
    )
    check(
        "minority_variance_estimability",
        minority.peak_variance_ratio_estimable.eq(
            minority.peak_variance_ratio_vs_balanced.notna()
        ).all()
        and minority.final_variance_ratio_estimable.eq(
            minority.final_variance_ratio_vs_balanced.notna()
        ).all()
        and (~minority.final_variance_ratio_estimable).any(),
        {
            "peakEstimable": int(minority.peak_variance_ratio_estimable.sum()),
            "finalEstimable": int(minority.final_variance_ratio_estimable.sum()),
        },
    )
    estimable_peak = minority[minority.peak_variance_ratio_estimable]
    estimable_final = minority[minority.final_variance_ratio_estimable]
    check(
        "minority_variance_interval_order",
        estimable_peak.peak_variance_ratio_ci95_low.le(
            estimable_peak.peak_variance_ratio_ci95_high
        ).all()
        and estimable_final.final_variance_ratio_ci95_low.le(
            estimable_final.final_variance_ratio_ci95_high
        ).all(),
        [len(estimable_peak), len(estimable_final)],
    )
    check(
        "deterministic_replay",
        replay["selectedScenarios"] == 18 and replay["allPassed"],
        [replay["selectedScenarios"], sum(not row["passed"] for row in replay["rows"])],
    )
    check(
        "run_accounting_record",
        accounting["resultRows"] == EXPECTED_RUNS
        and accounting["completed"] == EXPECTED_RUNS
        and accounting["censored"] == 0,
        accounting,
    )
    figure_paths = [
        output / f"{stem}.{suffix}"
        for stem in (
            "pairwise_response_unique",
            "pairwise_response_repeated",
            "three_way_response",
            "correlation_fidelity",
            "extreme_minority_variance",
        )
        for suffix in ("png", "svg")
    ]
    check(
        "figures_present",
        all(path.is_file() and path.stat().st_size > 0 for path in figure_paths),
        [path.name for path in figure_paths],
    )
    check(
        "s04_not_started",
        not Path("/artifacts/research_steps/S04").exists(),
        None,
    )
    failed = [name for name, value in checks.items() if not value["passed"]]
    value = {
        "schema": "e04.s03.validation_summary.v1",
        "researchStepId": "S03",
        "success": not failed,
        "checkCount": len(checks),
        "failedChecks": failed,
        "checks": checks,
    }
    write_json(output / "validation_summary.json", value)
    write_json(
        output / "upstream_immutability_audit.json",
        {
            "schema": "e04.s03.upstream_immutability.v1",
            "researchStepId": "S03",
            "upstream": upstream,
            "allPassed": all(item["allPassed"] for item in upstream.values()),
        },
    )
    if failed:
        raise AssertionError(f"E04 S03 validation failed: {failed}")
    return value


def write_provenance(output: Path) -> dict[str, Any]:
    inputs = {
        "agents": WORKSPACE / "AGENTS.md",
        "fullPlan": WORKSPACE / "FULL_PLAN.md",
        "researchPlan": WORKSPACE / "RESEARCH_PLAN.md",
        "attachmentManifest": WORKSPACE / "input-attachments/MANIFEST.json",
        "attachmentSidecar": WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/_metadata/ATTACHMENT.md",
        "paperMarkdown": WORKSPACE
        / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec/pdf-markdown.md",
        "previousArtifactsMarkdown": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
        "previousArtifactsJson": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
        "e01BaseDrawBank": S08_DIR / "base_draw_bank.parquet",
        "e01MetricDefinitions": UPSTREAM
        / "research_steps/S01/paper_metric_definitions.md",
        "e01TransitionSpecification": UPSTREAM
        / "research_steps/S03/transition_spec.md",
        "e01S13Report": UPSTREAM / "research_steps/S13/research_step_full_results.md",
        "e01ReleaseManifest": UPSTREAM
        / "release/reference_simulator/release_manifest.json",
        "s01ArtifactManifest": S01_DIR / "artifact_manifest.json",
        "s01FullReport": S01_DIR / "research_step_full_results.md",
        "s02ArtifactManifest": S02_DIR / "artifact_manifest.json",
        "s02FullReport": S02_DIR / "research_step_full_results.md",
        "s02BaselineDerivation": S02_DIR / "composition_baseline.md",
        "s03RepositoryContract": CONTRACT_PATH,
        "s03FrozenPreregistration": output / "preregistration.json",
        "s03FreezeRecord": output / "freeze_record.json",
        "s03ScenarioManifest": output / "scenario_manifest.parquet",
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"S03 provenance inputs missing: {missing}")
    value = {
        "schema": "e04.s03.provenance.v1",
        "researchStepId": "S03",
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryHeadAtPackaging": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "upstreamModified": False,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in inputs.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpuCount": os.cpu_count(),
            "workers": 8,
            "threadEnvironment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "matplotlib": matplotlib.__version__,
                "scipy": __import__("scipy").__version__,
            },
        },
    }
    write_json(output / "provenance.json", value)
    write_json(output / "environment.json", value["environment"])
    return value


def write_artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    value = {
        "schema": "e04.s03.artifact_manifest.v1",
        "researchStepId": "S03",
        "artifactCount": len(files),
        "artifacts": files,
    }
    write_json(output / "artifact_manifest.json", value)
    return value
