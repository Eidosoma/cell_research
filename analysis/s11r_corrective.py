"""E04 S11R corrective value-position intervention.

The design in ``s11r_corrective_contract.json`` is frozen before any S11R
continuation outcome.  Calibration may inspect terminal accounting only;
holdout trajectories are opened only after a candidate passes every frozen
calibration gate.  This module never starts S12.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.composition_sweep import (
    SweepCondition,
    build_conditions,
    load_base_draws,
    materialize_sweep_scenario,
)
from analysis.policy_label_switches import (
    _run_source,
    _source_labelled_copy,
)
from analysis.value_policy_shuffles import run_branch, sample_branch
from reference_simulator.engine import (
    evaluate_terminal,
    execute_serial_summary_activation,
    scheduled_actor,
)
from reference_simulator.model import (
    RunState,
    Scenario,
    canonical_json_bytes,
    state_hash,
)
from reference_simulator.scheduler import scheduled_side


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "analysis/s11r_corrective_contract.json"
OUTPUT_DIR = Path("/artifacts/research_steps/S11R")
CACHE_DIR = Path("/cache/e04_s11r")
S03_DIR = Path("/artifacts/research_steps/S03")
S12_DIR = Path("/artifacts/research_steps/S12")
UPSTREAM_DIRS = {
    **{f"S{index:02d}": Path(f"/artifacts/research_steps/S{index:02d}") for index in range(1, 11)},
    "S11": Path("/artifacts/research_steps/S11"),
}
EXPECTED_MANIFEST_HASHES = {
    "S01": "df9860285fc5b1ed9f7442fb68ea5e9ed9e407918d90988d0b3592f4b3ef8079",
    "S02": "9e632eedac59f6b08ff6b619ce2451a99b49c952319af2747076fd7bb3411b54",
    "S03": "85371d82a53096764b0de52029472a7978261ca5d4857f4d392f441db076f9e9",
    "S04": "90db3cb1afbb8e1ff53fb1d4fe0e04a4d21fb66ccd3a0b922738bade82d05641",
    "S05": "8a9528c5f14abb3598cc7e1f41a145978f7f7d47717c23497623f5c00579b4ce",
    "S06": "e8137e96571fdd281629bf6d0c32e6c8bbbc081ede2953f19ca9c11f957aa786",
    "S07": "294b89d1bd38762fada95b785aa2f82450fe26b4fc522c2dedaf0f44e360dd18",
    "S08": "1ac800f81ffb7ece006724a140120bf88d8924ab706b9fb8b4ef13ea144f5bd4",
    "S09": "00fffe60a61a4707d98b99a257a1df2a42a14d26af2ed4e20155c1bdd0103840",
    "S10": "5e9d15d43c74805c6edbf53d86691ba296315be11ccf16584a627d6bfa3a78f6",
    "S11": "77a419d9e367cf5746f444207d4bef94c06e8ab40c9a472a7f838f8ff568db37",
}
CONDITIONS = (
    "S03-REP-BUB-INS-P50-ABS",
    "S03-REP-BUB-SEL-P50-ABS",
    "S03-REP-INS-SEL-P50-ABS",
    "S03-UNQ-BUB-INS-P50-ABS",
    "S03-UNQ-BUB-SEL-P50-ABS",
    "S03-UNQ-INS-SEL-P50-ABS",
)
PEAK_PROGRESS = {
    "S03-REP-BUB-INS-P50-ABS": 0.16,
    "S03-REP-BUB-SEL-P50-ABS": 0.36,
    "S03-REP-INS-SEL-P50-ABS": 0.75,
    "S03-UNQ-BUB-INS-P50-ABS": 0.19,
    "S03-UNQ-BUB-SEL-P50-ABS": 0.33,
    "S03-UNQ-INS-SEL-P50-ABS": 0.27,
}
CANDIDATES = (
    ("M02_MIN", 2, "minimum"),
    ("M02_MED", 2, "median"),
    ("M04_MED", 4, "median"),
    ("M06_MED", 6, "median"),
    ("M08_MED", 8, "median"),
    ("M12_MED", 12, "median"),
    ("M16_MED", 16, "median"),
    ("M24_MED", 24, "median"),
)
CANDIDATE_BANK_SIZE = 4096
MINIMUM_TARGET_SUPPORT = 16
GRID = np.linspace(0.0, 1.0, 101)
BOOTSTRAP_DRAWS = 10_000
SEED_NAMESPACE = "E04/S11R/corrective/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(_json_native(value))).hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_json_native(value)) + b"\n")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd"
    )


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY, text=True, stderr=subprocess.STDOUT
    ).strip()


def derive_seed(stream: str, *address: Any) -> int:
    payload = {
        "namespace": SEED_NAMESPACE,
        "stream": stream,
        "address": list(address),
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json_bytes(payload)).digest()[:16], "big"
    )


def verify_manifest(step: str) -> dict[str, Any]:
    directory = UPSTREAM_DIRS[step]
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = []
    for item in manifest["artifacts"]:
        path = directory / item["path"]
        observed = sha256_file(path) if path.is_file() else None
        artifacts.append(
            {
                "path": item["path"],
                "expected": item["sha256"],
                "observed": observed,
                "passed": observed == item["sha256"],
            }
        )
    observed_manifest = sha256_file(manifest_path)
    expected_manifest = EXPECTED_MANIFEST_HASHES[step]
    return {
        "step": step,
        "manifestPath": str(manifest_path),
        "expectedManifestSha256": expected_manifest,
        "observedManifestSha256": observed_manifest,
        "manifestPassed": observed_manifest == expected_manifest,
        "artifactCount": len(artifacts),
        "artifactFailures": [item for item in artifacts if not item["passed"]],
        "allPassed": observed_manifest == expected_manifest
        and all(item["passed"] for item in artifacts),
    }


def positional_inversion_count(values: Sequence[int | float]) -> int:
    """Count strict positional inversions i<j; ties do not contribute."""
    array = np.asarray(values)
    return int(np.sum(np.triu(array[:, None] > array[None, :], k=1)))


def brute_positional_inversion_count(values: Sequence[int | float]) -> int:
    return sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def bank_inversion_counts(bank: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    result = np.empty(len(bank), dtype=np.int32)
    upper = np.triu(np.ones((bank.shape[1], bank.shape[1]), dtype=bool), k=1)
    for start in range(0, len(bank), chunk_size):
        chunk = bank[start : start + chunk_size]
        result[start : start + len(chunk)] = np.sum(
            (chunk[:, :, None] > chunk[:, None, :]) & upper, axis=(1, 2)
        )
    return result


def select_pair_at_exact_target(
    bank: np.ndarray,
    original: np.ndarray,
    inversion_counts: np.ndarray,
    target_inversion_count: int,
) -> dict[str, Any]:
    """Select two distinct assignments at exact inversion distance zero."""
    eligible = np.flatnonzero(inversion_counts == target_inversion_count)
    if len(eligible) < 2:
        raise ValueError("exact inversion target has fewer than two assignments")
    l1 = np.sum(np.abs(bank[eligible] - original), axis=1)
    hashes = [hashlib.sha256(bank[index].tobytes()).hexdigest() for index in eligible]
    choices: list[tuple[int, int, str, str, int, int]] = []
    for left_offset in range(len(eligible)):
        for right_offset in range(left_offset + 1, len(eligible)):
            left = int(eligible[left_offset])
            right = int(eligible[right_offset])
            choices.append(
                (
                    abs(int(l1[left_offset]) - int(l1[right_offset])),
                    -int(np.sum(bank[left] != bank[right])),
                    min(hashes[left_offset], hashes[right_offset]),
                    max(hashes[left_offset], hashes[right_offset]),
                    left,
                    right,
                )
            )
    choice = min(choices)
    left, right = int(choice[-2]), int(choice[-1])
    if hashes[list(eligible).index(left)] > hashes[list(eligible).index(right)]:
        left, right = right, left
    return {
        "primaryIndex": left,
        "matchedIndex": right,
        "primaryValues": bank[left].copy(),
        "matchedValues": bank[right].copy(),
        "primaryInversions": int(inversion_counts[left]),
        "matchedInversions": int(inversion_counts[right]),
        "inversionDistance": abs(
            int(inversion_counts[left]) - int(inversion_counts[right])
        ),
        "l1Difference": int(choice[0]),
        "assignmentHamming": int(-choice[1]),
        "targetSupport": len(eligible),
    }


def value_assignment_bank(
    values: Sequence[int], changed_positions: int, scenario_id: str
) -> tuple[np.ndarray, dict[str, Any]]:
    original = np.asarray(values, dtype=np.int16)
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            derive_seed("candidate_bank", scenario_id, changed_positions)
        )
    )
    seen: dict[bytes, np.ndarray] = {}
    attempts = 0
    maximum_attempts = CANDIDATE_BANK_SIZE * 80
    while len(seen) < CANDIDATE_BANK_SIZE and attempts < maximum_attempts:
        attempts += 1
        support = np.sort(
            rng.choice(len(original), size=changed_positions, replace=False)
        )
        old = original[support]
        replacement = rng.permutation(old)
        if np.any(replacement == old):
            continue
        candidate = original.copy()
        candidate[support] = replacement
        seen.setdefault(candidate.tobytes(), candidate)
    if len(seen) < CANDIDATE_BANK_SIZE:
        raise RuntimeError(
            f"only {len(seen)} distinct assignments after {attempts} attempts"
        )
    bank = np.stack(list(seen.values()))
    changed = np.sum(bank != original, axis=1)
    if not np.all(changed == changed_positions):
        raise AssertionError("candidate bank changed-position count drift")
    if any(sorted(row.tolist()) != sorted(original.tolist()) for row in bank):
        raise AssertionError("candidate bank failed value-multiset preservation")
    return bank, {
        "bankSize": len(bank),
        "attempts": attempts,
        "changedPositions": changed_positions,
        "bankSha256": hashlib.sha256(bank.tobytes()).hexdigest(),
    }


def select_value_pair(
    values: Sequence[int], candidate_id: str, changed_positions: int, target_mode: str, scenario_id: str
) -> dict[str, Any]:
    original = np.asarray(values, dtype=np.int16)
    before = positional_inversion_count(original)
    bank, bank_audit = value_assignment_bank(values, changed_positions, scenario_id)
    inversions = bank_inversion_counts(bank)
    counts = Counter(int(item) for item in inversions)
    supported_positive = sorted(
        target
        for target, count in counts.items()
        if target > before and count >= MINIMUM_TARGET_SUPPORT
    )
    if not supported_positive:
        raise RuntimeError("candidate bank has no support-qualified positive target")
    if target_mode == "minimum":
        target = supported_positive[0]
    elif target_mode == "median":
        target = supported_positive[(len(supported_positive) - 1) // 2]
    else:
        raise ValueError(f"unknown target mode {target_mode}")
    pair = select_pair_at_exact_target(bank, original, inversions, target)
    primary = pair["primaryValues"]
    matched = pair["matchedValues"]
    result = {
        "candidateId": candidate_id,
        "changedPositions": changed_positions,
        "targetMode": target_mode,
        "inversionsBefore": before,
        "targetInversionCount": target,
        "targetInversionChange": target - before,
        "supportedPositiveTargetCount": len(supported_positive),
        "minimumSupportedPositiveTarget": supported_positive[0],
        "maximumSupportedPositiveTarget": supported_positive[-1],
        **bank_audit,
        **pair,
        "primarySha256": hashlib.sha256(primary.tobytes()).hexdigest(),
        "matchedSha256": hashlib.sha256(matched.tobytes()).hexdigest(),
        "primaryChangedCount": int(np.sum(primary != original)),
        "matchedChangedCount": int(np.sum(matched != original)),
        "primaryValueMultisetPreserved": sorted(primary.tolist())
        == sorted(original.tolist()),
        "matchedValueMultisetPreserved": sorted(matched.tolist())
        == sorted(original.tolist()),
        "targetAttained": bool(
            pair["inversionDistance"] == 0
            and pair["primaryInversions"] == target
            and pair["matchedInversions"] == target
        ),
        "supportQualified": pair["targetSupport"] >= MINIMUM_TARGET_SUPPORT,
    }
    return result


def exhaustive_small_validation() -> dict[str, Any]:
    arrangement_checks = 0
    unique_arrangements = 0
    repeated_arrangements = 0
    for n in range(1, 8):
        for arrangement in itertools.permutations(range(n)):
            arrangement_checks += 1
            unique_arrangements += 1
            if positional_inversion_count(arrangement) != brute_positional_inversion_count(
                arrangement
            ):
                raise AssertionError(f"unique inversion mismatch: {arrangement}")
    for n in range(2, 9):
        base = tuple(index // 2 for index in range(n))
        for arrangement in sorted(set(itertools.permutations(base))):
            arrangement_checks += 1
            repeated_arrangements += 1
            if positional_inversion_count(arrangement) != brute_positional_inversion_count(
                arrangement
            ):
                raise AssertionError(f"repeated inversion mismatch: {arrangement}")

    matcher_cases = 0
    matcher_pair_comparisons = 0
    for n in range(3, 7):
        bank = np.asarray(list(itertools.permutations(range(n))), dtype=np.int16)
        original = np.arange(n, dtype=np.int16)
        inversions = bank_inversion_counts(bank)
        counts = Counter(int(item) for item in inversions)
        targets = [target for target, count in sorted(counts.items()) if count >= 2]
        for target in targets:
            selected = select_pair_at_exact_target(bank, original, inversions, target)
            eligible = np.flatnonzero(inversions == target)
            objectives = []
            for left_offset in range(len(eligible)):
                for right_offset in range(left_offset + 1, len(eligible)):
                    left, right = int(eligible[left_offset]), int(eligible[right_offset])
                    objectives.append(
                        (
                            abs(
                                int(np.abs(bank[left] - original).sum())
                                - int(np.abs(bank[right] - original).sum())
                            ),
                            -int(np.sum(bank[left] != bank[right])),
                        )
                    )
            chosen_objective = (
                selected["l1Difference"], -selected["assignmentHamming"]
            )
            if chosen_objective != min(objectives) or selected["inversionDistance"] != 0:
                raise AssertionError(f"matcher objective mismatch n={n}, target={target}")
            matcher_cases += 1
            matcher_pair_comparisons += len(objectives)
    return {
        "schema": "e04.s11r.exhaustive_matcher_validation.v1",
        "researchStepId": "S11R",
        "passed": True,
        "uniqueArrangementsChecked": unique_arrangements,
        "repeatedArrangementsChecked": repeated_arrangements,
        "totalArrangementsChecked": arrangement_checks,
        "matcherCasesChecked": matcher_cases,
        "matcherCandidatePairsCompared": matcher_pair_comparisons,
        "maximumUniqueN": 7,
        "maximumRepeatedN": 8,
        "maximumMatcherN": 6,
        "intendedInversionDistance": 0,
    }


def _split_rank(condition_id: str, replicate: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "namespace": SEED_NAMESPACE,
                "stream": "calibration_holdout_split",
                "conditionId": condition_id,
                "replicateOrdinal": replicate,
            }
        )
    ).hexdigest()


def source_tasks(split: str) -> list[dict[str, Any]]:
    if split not in {"calibration", "holdout", "reserve"}:
        raise ValueError("split must be calibration, holdout, or reserve")
    conditions = {
        condition.condition_id: condition
        for condition in build_conditions()
        if condition.condition_id in CONDITIONS
    }
    if set(conditions) != set(CONDITIONS):
        raise AssertionError("S11R source conditions changed")
    bases = load_base_draws()
    native = pd.read_parquet(S03_DIR / "composition_sweep.parquet")
    native = native[native.condition_id.isin(CONDITIONS)].set_index("scenario_id")
    tasks: list[dict[str, Any]] = []
    for condition_id in CONDITIONS:
        condition = conditions[condition_id]
        eligible = [replicate for replicate in range(250) if replicate % 10 not in {1, 5}]
        ranked = sorted(eligible, key=lambda replicate: _split_rank(condition_id, replicate))
        selected = {
            "calibration": ranked[:25],
            "holdout": ranked[25:50],
            "reserve": ranked[50:],
        }[split]
        for replicate in selected:
            base = bases[(condition.input_profile, replicate)]
            scenario, metadata = materialize_sweep_scenario(condition, base)
            expected = native.loc[scenario.scenario_id]
            tasks.append(
                {
                    "condition": condition.to_dict(),
                    "base": dict(base),
                    "metadata": {
                        "condition_id": condition_id,
                        "input_profile": condition.input_profile,
                        "policy_set_label": condition.policy_set_label,
                        "replicate_ordinal": replicate,
                        "scenario_id": scenario.scenario_id,
                        "scenario_json_sha256": metadata["scenarioJsonSha256"],
                        "split": split,
                    },
                    "expected_native": {
                        "activation_count": int(expected.activation_count),
                        "successful_swap_count": int(expected.successful_swap_count),
                        "stop_reason": str(expected.stop_reason),
                        "final_state_hash": str(expected.final_state_hash),
                        "event_budget": int(scenario.max_activations),
                    },
                }
            )
    tasks.sort(
        key=lambda task: (
            task["metadata"]["condition_id"],
            task["metadata"]["replicate_ordinal"],
        )
    )
    expected_count = 900 if split == "reserve" else 150
    if len(tasks) != expected_count:
        raise AssertionError(f"S11R {split} task count changed: {len(tasks)}")
    return tasks


def _split_frame() -> pd.DataFrame:
    rows = []
    for split in ("calibration", "holdout", "reserve"):
        for task in source_tasks(split):
            rows.append(task["metadata"])
    return pd.DataFrame(rows).sort_values(
        ["condition_id", "split", "replicate_ordinal"]
    )


def _specification_markdown(contract_hash: str) -> str:
    return f"""# S11R frozen corrective specification

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S11R |
| Completion status | Design frozen before any S11R continuation outcome |
| Artifacts written | `preregistration.json`, `freeze_record.json`, `scenario_split.parquet`, `exhaustive_matcher_validation.json`, and this specification |
| Validation result | True i<j positional inversions and the exact-distance matcher passed exhaustive unique/repeated small-case enumeration; six calibration and six sealed holdout strata contain 25 unused scenarios each |
| Outcome classification | Pending S11R calibration |
| Caveats or blockers | Value reassignment changes task progress and is not pure mediation; holdout stays sealed unless a candidate passes every completion and preservation gate |
| Recommended next action | Run terminal-only calibration, promote only a fully qualifying candidate, and do not start S12 |

Contract SHA-256: `{contract_hash}`.

## Corrected matcher and support

The metric counts pairs `i<j` for which the left value is strictly greater than
the right value. Candidate assignments preserve the exact global value multiset
and change exactly the requested number of identity-owned value fields. A target
is support-qualified only when at least 16 distinct assignments in the fixed
4,096-assignment bank have the exact same post-intervention inversion count.
The selected primary and matched assignments must both attain that count, hence
their intended true inversion-distance is exactly zero. Total absolute value
change and assignment diversity are outcome-blind secondary criteria.

## Outcome-blind calibration and sealed holdout

Calibration and holdout each contain 25 SHA-ranked, previously unused scenarios
per condition. S07 calibration and S07/S11 holdout ordinals are excluded.
Calibration retains only terminal/accounting and assignment diagnostics—no
aggregation or Sortedness trajectory. Each assignment must complete at least
24/25 in every condition and 147/150 overall, with no invariant, budget,
preservation, scheduler, pairing, or target-attainment failure. The strongest
passing ladder entry is promoted. If none passes, holdout is not run.

## Preservation and interpretation

Occupancy, policies, labels, direction, faults, identity set, activation and
stream counters, ledger, seed, runtime key, scheduler, maximum event budget,
and identity-owned Selection cursors remain exact. Only values on the declared
support change. This directly perturbs the sort task and possibly policy–value
association; it is not a natural mediator intervention.
"""


def freeze_design(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("cannot freeze S11R after S11R artifacts exist")
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("cannot freeze S11R after S11R cache exists")
    if S12_DIR.exists():
        raise AssertionError("S12 artifacts exist before S11R freeze")
    contract = json.loads(CONTRACT_PATH.read_text())
    if contract["researchStepId"] != "S11R":
        raise AssertionError("S11R contract identity drift")
    upstream = {step: verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    if not all(item["allPassed"] for item in upstream.values()):
        raise AssertionError("upstream artifact immutability failed")
    exhaustive = exhaustive_small_validation()
    split = _split_frame()
    calibration = set(split.loc[split.split == "calibration", "scenario_id"])
    holdout = set(split.loc[split.split == "holdout", "scenario_id"])
    if calibration & holdout:
        raise AssertionError("calibration and holdout overlap")
    if any(replicate % 10 in {1, 5} for replicate in split.replicate_ordinal):
        raise AssertionError("S07/S11 source ordinal leaked into S11R")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    contract_hash = sha256_file(CONTRACT_PATH)
    implementation_hash = sha256_file(Path(__file__).resolve())
    record = {
        "schema": "e04.s11r.freeze_record.v1",
        "researchStepId": "S11R",
        "frozenAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeOutcomes": True,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": contract_hash,
        "implementationPath": str(Path(__file__).resolve()),
        "implementationSha256": implementation_hash,
        "gitHead": git_output("rev-parse", "HEAD"),
        "conditions": list(CONDITIONS),
        "calibrationScenarioCount": 150,
        "holdoutScenarioCount": 150,
        "reserveScenarioCount": 900,
        "calibrationScenarioIdsSha256": canonical_hash(sorted(calibration)),
        "holdoutScenarioIdsSha256": canonical_hash(sorted(holdout)),
        "candidateIds": [item[0] for item in CANDIDATES],
        "peakProgressByCondition": PEAK_PROGRESS,
        "upstream": upstream,
        "s12Absent": not S12_DIR.exists(),
    }
    write_json(output / "preregistration.json", contract)
    write_json(output / "freeze_record.json", record)
    write_json(output / "exhaustive_matcher_validation.json", exhaustive)
    write_parquet(split, output / "scenario_split.parquet")
    write_json(
        output / "upstream_immutability_audit.json",
        {
            "schema": "e04.s11r.upstream_immutability_audit.v1",
            "researchStepId": "S11R",
            "stage": "freeze",
            "steps": upstream,
            "allPassed": all(item["allPassed"] for item in upstream.values()),
            "s12Absent": not S12_DIR.exists(),
        },
    )
    (output / "corrective_specification.md").write_text(
        _specification_markdown(contract_hash), encoding="utf-8"
    )
    return record


def assert_frozen(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    record = json.loads((output / "freeze_record.json").read_text())
    if not record["frozenBeforeOutcomes"]:
        raise AssertionError("S11R was not frozen before outcomes")
    if record["contractSha256"] != sha256_file(CONTRACT_PATH):
        raise AssertionError("S11R contract changed after freeze")
    if record["implementationSha256"] != sha256_file(Path(__file__).resolve()):
        raise AssertionError("S11R implementation changed after freeze")
    if S12_DIR.exists():
        raise AssertionError("S12 artifacts appeared during S11R")
    return record


def _prepare_source(task: Mapping[str, Any]) -> dict[str, Any]:
    condition = SweepCondition.from_dict(task["condition"])
    source_raw, metadata = materialize_sweep_scenario(condition, task["base"])
    if metadata["scenarioJsonSha256"] != task["metadata"]["scenario_json_sha256"]:
        raise AssertionError("source rematerialization changed")
    source = _source_labelled_copy(source_raw)
    expected = task["expected_native"]
    target = math.floor(
        PEAK_PROGRESS[condition.condition_id]
        * int(expected["successful_swap_count"])
    )
    final, checkpoints = _run_source(source, {"intervention": target})
    checkpoint = checkpoints["intervention"]
    source_checks = {
        "activation": final.activation_count == expected["activation_count"],
        "swaps": final.ledger["acceptedSwaps"]
        == expected["successful_swap_count"],
        "stopReason": final.terminal == expected["stop_reason"],
        "finalStateHash": state_hash(source.scenario_id, final)
        == expected["final_state_hash"],
        "budget": source.max_activations == expected["event_budget"] == 1_000_000,
    }
    values = [int(source.cell_map[cell_id].value) for cell_id in checkpoint.occupancy]
    return {
        "condition": condition,
        "source": source,
        "checkpoint": checkpoint,
        "final": final,
        "values": values,
        "source_row": {
            **task["metadata"],
            "checkpoint_progress": PEAK_PROGRESS[condition.condition_id],
            "checkpoint_target_swaps": target,
            "checkpoint_activation": checkpoint.activation_count,
            "checkpoint_successful_swaps": checkpoint.ledger["acceptedSwaps"],
            "checkpoint_state_hash": state_hash(source.scenario_id, checkpoint),
            "observed_activation_count": final.activation_count,
            "observed_successful_swap_count": final.ledger["acceptedSwaps"],
            "observed_stop_reason": final.terminal,
            "observed_final_state_hash": state_hash(source.scenario_id, final),
            **{f"check_{key}": value for key, value in source_checks.items()},
            "passed": all(source_checks.values()),
        },
    }


def _value_map(source: Scenario, checkpoint: RunState, positional_values: Sequence[int]) -> dict[str, int]:
    values = {cell.cell_id: int(cell.value) for cell in source.cells}
    for position, cell_id in enumerate(checkpoint.occupancy):
        values[cell_id] = int(positional_values[position])
    return values


def build_value_continuation(
    source: Scenario,
    checkpoint: RunState,
    positional_values: Sequence[int],
    arm: str,
) -> tuple[Scenario, RunState, dict[str, Any]]:
    new_values = _value_map(source, checkpoint, positional_values)
    old_values = {cell.cell_id: int(cell.value) for cell in source.cells}
    old_static = {
        cell.cell_id: (
            cell.policy.value,
            cell.direction.value,
            cell.fault.value,
            cell.analysis_label,
        )
        for cell in source.cells
    }
    cells = tuple(replace(cell, value=new_values[cell.cell_id]) for cell in source.cells)
    state = checkpoint.clone()
    state.terminal = None
    scenario = Scenario.create(
        cells,
        initial_occupancy=tuple(checkpoint.occupancy),
        seed=source.seed,
        max_activations=source.max_activations,
        architecture=source.architecture,
        scheduler=source.scheduler,
        batch_width=source.batch_width,
        traditional_policy=source.traditional_policy,
        generation_key=f"E04/S11R/{arm}/{source.scenario_id}",
        fault_placement=source.fault_placement,
        requested_fault_count=source.requested_fault_count,
        rng_profile=source.rng_profile,
        goal_profile=source.goal_profile,
        metric_profile=source.metric_profile,
    )
    scenario.validate()
    content_id = scenario.scenario_id
    object.__setattr__(scenario, "scenario_id", source.scenario_id)
    state.terminal = evaluate_terminal(scenario, state)
    observed_static = {
        cell.cell_id: (
            cell.policy.value,
            cell.direction.value,
            cell.fault.value,
            cell.analysis_label,
        )
        for cell in scenario.cells
    }
    changed = {cell_id for cell_id in old_values if old_values[cell_id] != new_values[cell_id]}
    audit = {
        "arm": arm,
        "scenario_id": source.scenario_id,
        "content_scenario_id": content_id,
        "runtime_key_preserved": scenario.scenario_id == source.scenario_id,
        "seed_preserved": scenario.seed == source.seed,
        "max_activations_preserved": scenario.max_activations == source.max_activations,
        "max_activations": scenario.max_activations,
        "scheduler_preserved": scenario.scheduler == source.scheduler,
        "occupancy_preserved": state.occupancy == checkpoint.occupancy,
        "identity_bijection_preserved": sorted(state.occupancy)
        == sorted(checkpoint.occupancy),
        "activation_preserved": state.activation_count == checkpoint.activation_count,
        "stream_counters_preserved": state.stream_counters == checkpoint.stream_counters,
        "ledger_preserved": state.ledger == checkpoint.ledger,
        "policy_direction_fault_label_preserved": old_static == observed_static,
        "global_value_multiset_preserved": sorted(old_values.values())
        == sorted(new_values.values()),
        "selection_cursors_preserved": state.selection_cursors
        == checkpoint.selection_cursors,
        "selection_cursor_hash_before": canonical_hash(checkpoint.selection_cursors),
        "selection_cursor_hash_after": canonical_hash(state.selection_cursors),
        "value_changed_count": len(changed),
        "value_changed_ids_sha256": canonical_hash(sorted(changed)),
        "terminal_immediately_after_intervention": state.terminal,
    }
    audit["passed"] = all(
        audit[key]
        for key in (
            "runtime_key_preserved",
            "seed_preserved",
            "max_activations_preserved",
            "scheduler_preserved",
            "occupancy_preserved",
            "identity_bijection_preserved",
            "activation_preserved",
            "stream_counters_preserved",
            "ledger_preserved",
            "policy_direction_fault_label_preserved",
            "global_value_multiset_preserved",
            "selection_cursors_preserved",
        )
    )
    return scenario, state, audit


def run_terminal(scenario: Scenario, state: RunState) -> dict[str, Any]:
    while state.terminal is None:
        changed = execute_serial_summary_activation(scenario, state)
        if changed:
            state.terminal = evaluate_terminal(scenario, state)
    return {
        "stop_reason": state.terminal,
        "final_activation_count": state.activation_count,
        "final_successful_swap_count": state.ledger["acceptedSwaps"],
        "post_checkpoint_activations": None,
        "final_state_hash": state_hash(scenario.scenario_id, state),
        "final_occupancy_hash": canonical_hash(state.occupancy),
        "final_cursor_hash": canonical_hash(state.selection_cursors),
    }


def scheduler_coupling_audit(
    reference: Scenario, comparison: Scenario, checkpoint_activation: int, event_count: int
) -> dict[str, Any]:
    offsets = (
        np.asarray([], dtype=np.int64)
        if event_count <= 0
        else np.unique(
            np.rint(np.linspace(0, event_count - 1, min(257, event_count))).astype(
                np.int64
            )
        )
    )
    records = []
    passed = True
    for offset in offsets:
        event = checkpoint_activation + int(offset)
        left = scheduled_actor(reference, event, include_draws=True)
        right = scheduled_actor(comparison, event, include_draws=True)
        actor_equal = left == right
        actor = left[0]
        side_equal = True
        if reference.cell_map[actor].policy.value == "Bubble":
            side_equal = scheduled_side(reference, event) == scheduled_side(
                comparison, event
            )
        passed = passed and actor_equal and side_equal
        records.append((event, actor, actor_equal, side_equal))
    return {
        "sampled_event_count": len(offsets),
        "actor_addresses_equal": passed,
        "policy_side_addresses_equal": passed,
        "sample_sha256": canonical_hash(records),
        "passed": passed,
    }


def _assignment_audit_row(
    prepared: Mapping[str, Any], pair: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = prepared["source_row"]
    return {
        "condition_id": metadata["condition_id"],
        "input_profile": metadata["input_profile"],
        "policy_set_label": metadata["policy_set_label"],
        "replicate_ordinal": metadata["replicate_ordinal"],
        "scenario_id": metadata["scenario_id"],
        **{
            key: value
            for key, value in pair.items()
            if key not in {"primaryValues", "matchedValues"}
        },
    }


def _calibration_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    prepared = _prepare_source(task)
    source = prepared["source"]
    checkpoint = prepared["checkpoint"]
    result = {
        "scenario_id": source.scenario_id,
        "source_row": prepared["source_row"],
        "assignment_rows": [],
        "run_rows": [],
        "preservation_rows": [],
        "scheduler_rows": [],
    }
    for candidate_id, changed_positions, target_mode in CANDIDATES:
        pair = select_value_pair(
            prepared["values"],
            candidate_id,
            changed_positions,
            target_mode,
            source.scenario_id,
        )
        result["assignment_rows"].append(_assignment_audit_row(prepared, pair))
        for arm, positional in (
            ("value_primary", pair["primaryValues"]),
            ("value_matched", pair["matchedValues"]),
        ):
            scenario, state, preservation = build_value_continuation(
                source, checkpoint, positional, arm
            )
            start_activation = state.activation_count
            terminal = run_terminal(scenario, state)
            terminal["post_checkpoint_activations"] = (
                terminal["final_activation_count"] - start_activation
            )
            coupling = scheduler_coupling_audit(
                source,
                scenario,
                checkpoint.activation_count,
                terminal["post_checkpoint_activations"],
            )
            identity = {
                "condition_id": task["metadata"]["condition_id"],
                "input_profile": task["metadata"]["input_profile"],
                "policy_set_label": task["metadata"]["policy_set_label"],
                "replicate_ordinal": task["metadata"]["replicate_ordinal"],
                "scenario_id": source.scenario_id,
                "candidate_id": candidate_id,
                "arm": arm,
            }
            result["run_rows"].append(
                {
                    **identity,
                    **terminal,
                    "complete": terminal["stop_reason"] == "complete",
                    "event_budget_respected": terminal["final_activation_count"]
                    <= scenario.max_activations,
                }
            )
            result["preservation_rows"].append({**identity, **preservation})
            result["scheduler_rows"].append({**identity, **coupling})
    return result


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {json.loads(line)["scenario_id"] for line in handle if line.strip()}


def _parallel_corpus(
    tasks: Sequence[Mapping[str, Any]], worker: Any, checkpoint: Path, workers: int
) -> dict[str, Any]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(checkpoint)
    pending = [task for task in tasks if task["metadata"]["scenario_id"] not in completed]
    written = 0
    started = time.perf_counter()
    with checkpoint.open("a", buffering=1) as handle, ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        iterator = iter(pending)
        futures: dict[Any, Mapping[str, Any]] = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(worker, task)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(json.dumps(_json_native(result), sort_keys=True) + "\n")
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(worker, task)] = task
    return {
        "preexisting": len(completed),
        "written": written,
        "observed": len(_completed_ids(checkpoint)),
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }


def run_calibration(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR, workers: int = 8
) -> dict[str, Any]:
    assert_frozen(output)
    if (output / "calibration_decision.json").exists():
        raise FileExistsError("calibration decision already exists")
    accounting = _parallel_corpus(
        source_tasks("calibration"),
        _calibration_worker,
        cache / "calibration.jsonl",
        workers,
    )
    if accounting["observed"] != 150:
        raise AssertionError("calibration corpus incomplete")
    result = {
        "schema": "e04.s11r.calibration_run_accounting.v1",
        "researchStepId": "S11R",
        "workers": workers,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "sourceScenariosExpected": 150,
        "candidateCount": len(CANDIDATES),
        "armsPerCandidate": 2,
        "continuationsExpected": 150 * len(CANDIDATES) * 2,
        **accounting,
    }
    write_json(output / "calibration_run_accounting.json", result)
    return result


def _read_jsonl(path: Path, expected: int) -> list[dict[str, Any]]:
    with path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != expected:
        raise AssertionError(f"expected {expected} records at {path}, found {len(records)}")
    return records


def _flatten(records: Sequence[Mapping[str, Any]], key: str) -> pd.DataFrame:
    return pd.DataFrame([row for record in records for row in record[key]])


def _completion_gate(
    runs: pd.DataFrame,
    assignments: pd.DataFrame,
    preservation: pd.DataFrame,
    scheduler: pd.DataFrame,
    candidate_id: str,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    subset = runs[runs.candidate_id == candidate_id]
    rows = []
    condition_passes = []
    for condition_id in CONDITIONS:
        counts = {}
        for arm in ("value_primary", "value_matched"):
            group = subset[
                (subset.condition_id == condition_id) & (subset.arm == arm)
            ]
            complete = int(group.complete.sum())
            counts[arm] = complete
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "arm": arm,
                    "runs": len(group),
                    "complete": complete,
                    "completion_rate": complete / len(group),
                    "quiescent": int((group.stop_reason == "quiescent").sum()),
                    "event_budget": int((group.stop_reason == "event_budget").sum()),
                    "invariant_error": int((group.stop_reason == "invariant_error").sum()),
                }
            )
        condition_passes.append(
            counts["value_primary"] >= 24
            and counts["value_matched"] >= 24
            and abs(counts["value_primary"] - counts["value_matched"]) <= 1
        )
    pooled = {
        arm: int(subset[subset.arm == arm].complete.sum())
        for arm in ("value_primary", "value_matched")
    }
    assignment_subset = assignments[assignments.candidateId == candidate_id]
    preservation_subset = preservation[preservation.candidate_id == candidate_id]
    scheduler_subset = scheduler[scheduler.candidate_id == candidate_id]
    diagnostics = {
        "allConditionArmGatesPass": all(condition_passes),
        "pooledPrimaryComplete": pooled["value_primary"],
        "pooledMatchedComplete": pooled["value_matched"],
        "pooledGatesPass": min(pooled.values()) >= 147,
        "invariantErrors": int((subset.stop_reason == "invariant_error").sum()),
        "eventBudgetStops": int((subset.stop_reason == "event_budget").sum()),
        "eventBudgetsRespected": bool(subset.event_budget_respected.all()),
        "assignmentRows": len(assignment_subset),
        "assignmentTargetFailures": int((~assignment_subset.targetAttained).sum()),
        "assignmentSupportFailures": int((~assignment_subset.supportQualified).sum()),
        "preservationFailures": int((~preservation_subset.passed).sum()),
        "schedulerFailures": int((~scheduler_subset.passed).sum()),
    }
    passed = bool(
        diagnostics["allConditionArmGatesPass"]
        and diagnostics["pooledGatesPass"]
        and diagnostics["invariantErrors"] == 0
        and diagnostics["eventBudgetStops"] == 0
        and diagnostics["eventBudgetsRespected"]
        and diagnostics["assignmentRows"] == 150
        and diagnostics["assignmentTargetFailures"] == 0
        and diagnostics["assignmentSupportFailures"] == 0
        and diagnostics["preservationFailures"] == 0
        and diagnostics["schedulerFailures"] == 0
    )
    diagnostics["passed"] = passed
    return passed, rows, diagnostics


def analyze_calibration(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    assert_frozen(output)
    records = _read_jsonl(cache / "calibration.jsonl", 150)
    sources = pd.DataFrame([record["source_row"] for record in records])
    assignments = _flatten(records, "assignment_rows")
    runs = _flatten(records, "run_rows")
    preservation = _flatten(records, "preservation_rows")
    scheduler = _flatten(records, "scheduler_rows")
    if not sources.passed.all() or len(sources) != 150:
        raise AssertionError("source replay failed")
    gate_rows: list[dict[str, Any]] = []
    decisions = []
    passing = []
    for candidate_id, changed_positions, target_mode in CANDIDATES:
        passed, rows, diagnostics = _completion_gate(
            runs, assignments, preservation, scheduler, candidate_id
        )
        gate_rows.extend(rows)
        decisions.append(
            {
                "candidateId": candidate_id,
                "changedPositions": changed_positions,
                "targetMode": target_mode,
                **diagnostics,
            }
        )
        if passed:
            passing.append(candidate_id)
    promoted = passing[-1] if passing else None
    decision = {
        "schema": "e04.s11r.calibration_decision.v1",
        "researchStepId": "S11R",
        "madeAt": datetime.now(timezone.utc).isoformat(),
        "outcomeRestrictionHonored": "Only terminal/completion, assignment, preservation, scheduler, and accounting fields were analyzed; no aggregation or Sortedness trajectory was retained.",
        "candidateOrderWeakToStrong": [item[0] for item in CANDIDATES],
        "candidateDiagnostics": decisions,
        "passingCandidates": passing,
        "promotedCandidateId": promoted,
        "holdoutAuthorized": promoted is not None,
        "classificationIfNoPromotion": "constraining/contradictory",
        "reason": (
            "The strongest candidate passing every frozen gate was promoted."
            if promoted
            else "No candidate passed completion comparability in every condition and arm; holdout remains sealed."
        ),
    }
    write_parquet(sources, output / "calibration_source_replay.parquet")
    write_parquet(assignments, output / "calibration_assignment_audit.parquet")
    write_parquet(runs, output / "calibration_terminal_outcomes.parquet")
    write_parquet(preservation, output / "calibration_preservation_audit.parquet")
    write_parquet(scheduler, output / "calibration_scheduler_audit.parquet")
    completion = pd.DataFrame(gate_rows)
    write_parquet(completion, output / "calibration_completion_by_condition.parquet")
    write_json(output / "calibration_decision.json", decision)

    pivot = completion.pivot_table(
        index=["candidate_id", "arm"], columns="condition_id", values="completion_rate"
    )
    fig, ax = plt.subplots(figsize=(12, 6.5))
    image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(pivot)), [f"{a} / {b}" for a, b in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), [item.replace("S03-", "") for item in pivot.columns], rotation=45, ha="right")
    ax.set_title("S11R outcome-blind calibration completion rates")
    fig.colorbar(image, ax=ax, label="Completion rate")
    fig.tight_layout()
    fig.savefig(output / "calibration_completion.png", dpi=180)
    fig.savefig(output / "calibration_completion.svg")
    plt.close(fig)
    return decision


def _holdout_worker(task: Mapping[str, Any], candidate: tuple[str, int, str]) -> dict[str, Any]:
    prepared = _prepare_source(task)
    source = prepared["source"]
    checkpoint = prepared["checkpoint"]
    candidate_id, changed_positions, target_mode = candidate
    pair = select_value_pair(
        prepared["values"], candidate_id, changed_positions, target_mode, source.scenario_id
    )
    arm_values = {
        "no_switch": np.asarray(prepared["values"], dtype=np.int16),
        "sham": np.asarray(prepared["values"], dtype=np.int16),
        "value_primary": pair["primaryValues"],
        "value_matched": pair["matchedValues"],
    }
    scenarios = {}
    branches = {}
    preservation_rows = []
    for arm, values in arm_values.items():
        scenario, state, audit = build_value_continuation(source, checkpoint, values, arm)
        scenarios[arm] = scenario
        branches[arm] = run_branch(scenario, state)
        preservation_rows.append(
            {
                "condition_id": task["metadata"]["condition_id"],
                "replicate_ordinal": task["metadata"]["replicate_ordinal"],
                "scenario_id": source.scenario_id,
                "candidate_id": candidate_id,
                **audit,
            }
        )
    common_horizon = max(
        branch["state"].activation_count - checkpoint.activation_count
        for branch in branches.values()
    )
    sampled = {
        arm: sample_branch(
            scenarios[arm], branches[arm], checkpoint.activation_count, common_horizon
        )
        for arm in arm_values
    }
    run_rows = []
    trace_rows = []
    scheduler_rows = []
    identity = {
        "condition_id": task["metadata"]["condition_id"],
        "input_profile": task["metadata"]["input_profile"],
        "policy_set_label": task["metadata"]["policy_set_label"],
        "replicate_ordinal": task["metadata"]["replicate_ordinal"],
        "scenario_id": source.scenario_id,
        "candidate_id": candidate_id,
    }
    for arm in arm_values:
        final = branches[arm]["state"]
        post_events = final.activation_count - checkpoint.activation_count
        run_rows.append(
            {
                **identity,
                "arm": arm,
                "common_activation_horizon": common_horizon,
                "post_checkpoint_activations": post_events,
                "post_checkpoint_successful_swaps": final.ledger["acceptedSwaps"]
                - checkpoint.ledger["acceptedSwaps"],
                "stop_reason": final.terminal,
                "complete": final.terminal == "complete",
                "event_budget_respected": final.activation_count
                <= scenarios[arm].max_activations,
                "final_activation_count": final.activation_count,
                "final_successful_swap_count": final.ledger["acceptedSwaps"],
                "final_state_hash": state_hash(scenarios[arm].scenario_id, final),
                "final_occupancy_hash": canonical_hash(final.occupancy),
                "final_cursor_hash": canonical_hash(final.selection_cursors),
            }
        )
        scheduler_rows.append(
            {
                **identity,
                "arm": arm,
                **scheduler_coupling_audit(
                    source, scenarios[arm], checkpoint.activation_count, post_events
                ),
            }
        )
        for grid_index, exposure in enumerate(GRID):
            trace_rows.append(
                {
                    **identity,
                    "arm": arm,
                    "grid_index": grid_index,
                    "common_activation_progress": float(exposure),
                    "corrected_aggregation": float(
                        sampled[arm]["corrected_aggregation"][grid_index]
                    ),
                    "reference_sortedness": float(
                        sampled[arm]["reference_sortedness"][grid_index]
                    ),
                    "paper_sortedness": float(sampled[arm]["paper_sortedness"][grid_index]),
                }
            )
    exact = {
        **identity,
        "sham_final_state_exact": run_rows[0]["final_state_hash"]
        == run_rows[1]["final_state_hash"],
        "sham_curve_exact": all(
            np.array_equal(sampled["no_switch"][key], sampled["sham"][key])
            for key in ("corrected_aggregation", "reference_sortedness", "paper_sortedness")
        ),
    }
    exact["passed"] = exact["sham_final_state_exact"] and exact["sham_curve_exact"]
    return {
        "scenario_id": source.scenario_id,
        "source_row": prepared["source_row"],
        "assignment_row": _assignment_audit_row(prepared, pair),
        "run_rows": run_rows,
        "trace_rows": trace_rows,
        "preservation_rows": preservation_rows,
        "scheduler_rows": scheduler_rows,
        "exact_row": exact,
    }


def run_holdout(
    output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR, workers: int = 8
) -> dict[str, Any]:
    assert_frozen(output)
    decision = json.loads((output / "calibration_decision.json").read_text())
    if not decision["holdoutAuthorized"]:
        status = {
            "schema": "e04.s11r.holdout_status.v1",
            "researchStepId": "S11R",
            "executed": False,
            "reason": "No calibration candidate passed; holdout remained sealed.",
            "holdoutScenarioCountOpened": 0,
        }
        write_json(output / "holdout_status.json", status)
        return status
    candidate_id = decision["promotedCandidateId"]
    candidate = next(item for item in CANDIDATES if item[0] == candidate_id)

    def worker(task: Mapping[str, Any]) -> dict[str, Any]:
        return _holdout_worker(task, candidate)

    # ProcessPool requires a module-level callable, so execute with explicit futures here.
    tasks = source_tasks("holdout")
    checkpoint = cache / "holdout.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(checkpoint)
    pending = [task for task in tasks if task["metadata"]["scenario_id"] not in completed]
    written = 0
    started = time.perf_counter()
    with checkpoint.open("a", buffering=1) as handle, ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = iter(pending)
        futures = {}
        for _ in range(min(workers * 2, len(pending))):
            task = next(iterator, None)
            if task is None:
                break
            futures[executor.submit(_holdout_worker, task, candidate)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                handle.write(json.dumps(_json_native(result), sort_keys=True) + "\n")
                written += 1
                task = next(iterator, None)
                if task is not None:
                    futures[executor.submit(_holdout_worker, task, candidate)] = task
    observed = len(_completed_ids(checkpoint))
    if observed != 150:
        raise AssertionError("holdout corpus incomplete")
    accounting = {
        "schema": "e04.s11r.holdout_run_accounting.v1",
        "researchStepId": "S11R",
        "executed": True,
        "promotedCandidateId": candidate_id,
        "workers": workers,
        "sourceScenariosExpected": 150,
        "continuationsPerScenario": 4,
        "continuationsExpected": 600,
        "preexisting": len(completed),
        "written": written,
        "observed": observed,
        "elapsedSeconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }
    write_json(output / "holdout_run_accounting.json", accounting)
    return accounting


def _bootstrap_mean_ci(values: np.ndarray, address: str) -> tuple[float, float, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(derive_seed("bootstrap", address)))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def analyze_holdout(output: Path = OUTPUT_DIR, cache: Path = CACHE_DIR) -> dict[str, Any]:
    assert_frozen(output)
    decision = json.loads((output / "calibration_decision.json").read_text())
    if not decision["holdoutAuthorized"]:
        return json.loads((output / "holdout_status.json").read_text())
    records = _read_jsonl(cache / "holdout.jsonl", 150)
    sources = pd.DataFrame([record["source_row"] for record in records])
    assignments = pd.DataFrame([record["assignment_row"] for record in records])
    runs = _flatten(records, "run_rows")
    traces = _flatten(records, "trace_rows")
    preservation = _flatten(records, "preservation_rows")
    scheduler = _flatten(records, "scheduler_rows")
    exact = pd.DataFrame([record["exact_row"] for record in records])
    candidate_id = decision["promotedCandidateId"]
    passed, gate_rows, diagnostics = _completion_gate(
        runs[runs.arm.isin(["value_primary", "value_matched"])],
        assignments,
        preservation[preservation.arm.isin(["value_primary", "value_matched"])],
        scheduler[scheduler.arm.isin(["value_primary", "value_matched"])],
        candidate_id,
    )
    validation = bool(
        passed
        and sources.passed.all()
        and preservation.passed.all()
        and scheduler.passed.all()
        and exact.passed.all()
        and runs.event_budget_respected.all()
    )
    status = {
        "schema": "e04.s11r.holdout_status.v1",
        "researchStepId": "S11R",
        "executed": True,
        "promotedCandidateId": candidate_id,
        "completionGatePassed": passed,
        "validationPassed": validation,
        "diagnostics": diagnostics,
        "effectEstimateAuthorized": validation,
    }
    write_parquet(sources, output / "holdout_source_replay.parquet")
    write_parquet(assignments, output / "holdout_assignment_audit.parquet")
    write_parquet(runs, output / "holdout_runs.parquet")
    write_parquet(traces, output / "holdout_trajectories.parquet")
    write_parquet(preservation, output / "holdout_preservation_audit.parquet")
    write_parquet(scheduler, output / "holdout_scheduler_audit.parquet")
    write_parquet(exact, output / "holdout_sham_audit.parquet")
    write_parquet(pd.DataFrame(gate_rows), output / "holdout_completion_by_condition.parquet")
    if validation:
        indexed = traces.set_index(
            [
                "condition_id",
                "input_profile",
                "policy_set_label",
                "replicate_ordinal",
                "scenario_id",
                "grid_index",
                "common_activation_progress",
                "arm",
            ]
        )
        effects = []
        scenario_effects = []
        for metric in ("corrected_aggregation", "reference_sortedness"):
            reference = indexed.xs("no_switch", level="arm")[metric]
            for arm in ("value_primary", "value_matched"):
                effect = (indexed.xs(arm, level="arm")[metric] - reference).rename("effect").reset_index()
                effect["metric"] = metric
                effect["arm"] = arm
                effects.append(effect)
                for keys, group in effect.groupby(
                    ["condition_id", "input_profile", "policy_set_label", "replicate_ordinal", "scenario_id"],
                    sort=True,
                ):
                    curve = group.sort_values("grid_index").effect.to_numpy()
                    scenario_effects.append(
                        {
                            **dict(
                                zip(
                                    ["condition_id", "input_profile", "policy_set_label", "replicate_ordinal", "scenario_id"],
                                    keys,
                                )
                            ),
                            "metric": metric,
                            "arm": arm,
                            "signed_area": float(np.trapezoid(curve, GRID)),
                            "positive_area": float(np.trapezoid(np.maximum(curve, 0), GRID)),
                            "negative_area": float(np.trapezoid(np.minimum(curve, 0), GRID)),
                            "final_effect": float(curve[-1]),
                            "peak_absolute_effect": float(np.max(np.abs(curve))),
                        }
                    )
        effects_frame = pd.concat(effects, ignore_index=True)
        scenario_frame = pd.DataFrame(scenario_effects)
        pooled = []
        for (metric, arm), group in scenario_frame.groupby(["metric", "arm"], sort=True):
            for endpoint in ("signed_area", "positive_area", "negative_area", "final_effect", "peak_absolute_effect"):
                mean, low, high = _bootstrap_mean_ci(group[endpoint].to_numpy(), f"{metric}/{arm}/{endpoint}")
                pooled.append({"metric": metric, "arm": arm, "endpoint": endpoint, "mean": mean, "ci95_low": low, "ci95_high": high, "runs": len(group)})
        write_parquet(effects_frame, output / "holdout_effect_trajectories.parquet")
        write_parquet(scenario_frame, output / "holdout_scenario_effects.parquet")
        pd.DataFrame(pooled).to_csv(output / "holdout_pooled_effects.csv", index=False)
        curves = effects_frame.groupby(["metric", "arm", "grid_index", "common_activation_progress"], as_index=False).effect.mean()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, metric in zip(axes, ("corrected_aggregation", "reference_sortedness")):
            for arm, group in curves[curves.metric == metric].groupby("arm"):
                ax.plot(group.common_activation_progress, group.effect, label=arm)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Common activation exposure")
            ax.set_ylabel("Paired effect vs no-switch")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / "holdout_effect_curves.png", dpi=180)
        fig.savefig(output / "holdout_effect_curves.svg")
        plt.close(fig)
    write_json(output / "holdout_status.json", status)
    return status


def final_validation(output: Path = OUTPUT_DIR) -> dict[str, Any]:
    freeze = assert_frozen(output)
    calibration = json.loads((output / "calibration_decision.json").read_text())
    holdout = json.loads((output / "holdout_status.json").read_text())
    current_upstream = {step: verify_manifest(step) for step in EXPECTED_MANIFEST_HASHES}
    immutable = all(item["allPassed"] for item in current_upstream.values())
    freeze_same = all(
        current_upstream[step]["observedManifestSha256"]
        == freeze["upstream"][step]["observedManifestSha256"]
        for step in current_upstream
    )
    assignments = pd.read_parquet(output / "calibration_assignment_audit.parquet")
    runs = pd.read_parquet(output / "calibration_terminal_outcomes.parquet")
    preservation = pd.read_parquet(output / "calibration_preservation_audit.parquet")
    scheduler = pd.read_parquet(output / "calibration_scheduler_audit.parquet")
    validation = {
        "schema": "e04.s11r.validation_summary.v1",
        "researchStepId": "S11R",
        "matcherExhaustivePassed": json.loads((output / "exhaustive_matcher_validation.json").read_text())["passed"],
        "calibrationScenarioCount": int(runs.scenario_id.nunique()),
        "calibrationRunCount": len(runs),
        "calibrationAssignmentCount": len(assignments),
        "allSelectedAssignmentsAttainTarget": bool(assignments.targetAttained.all()),
        "allAssignmentsSupportQualified": bool(assignments.supportQualified.all()),
        "allCalibrationPreservationPassed": bool(preservation.passed.all()),
        "allCalibrationSchedulerCouplingPassed": bool(scheduler.passed.all()),
        "allCalibrationBudgetsRespected": bool(runs.event_budget_respected.all()),
        "calibrationDecisionPresent": True,
        "holdoutAuthorized": calibration["holdoutAuthorized"],
        "holdoutExecuted": holdout["executed"],
        "upstreamArtifactsImmutable": immutable and freeze_same,
        "s12Absent": not S12_DIR.exists(),
    }
    if holdout["executed"]:
        validation["holdoutValidationPassed"] = holdout["validationPassed"]
    else:
        validation["holdoutValidationPassed"] = None
    validation["allApplicableChecksPassed"] = bool(
        validation["matcherExhaustivePassed"]
        and validation["calibrationScenarioCount"] == 150
        and validation["calibrationRunCount"] == 2400
        and validation["calibrationAssignmentCount"] == 1200
        and validation["allSelectedAssignmentsAttainTarget"]
        and validation["allAssignmentsSupportQualified"]
        and validation["allCalibrationPreservationPassed"]
        and validation["allCalibrationSchedulerCouplingPassed"]
        and validation["allCalibrationBudgetsRespected"]
        and validation["upstreamArtifactsImmutable"]
        and validation["s12Absent"]
        and (
            validation["holdoutValidationPassed"]
            if calibration["holdoutAuthorized"]
            else not holdout["executed"]
        )
    )
    write_json(output / "validation_summary.json", validation)
    write_json(
        output / "upstream_immutability_audit.json",
        {
            "schema": "e04.s11r.upstream_immutability_audit.v1",
            "researchStepId": "S11R",
            "stage": "final",
            "steps": current_upstream,
            "freezeHashesUnchanged": freeze_same,
            "allPassed": immutable and freeze_same,
            "s12Absent": not S12_DIR.exists(),
        },
    )
    return validation


def _format_completion(completion: pd.DataFrame) -> str:
    pooled = (
        completion.groupby(["candidate_id", "arm"], as_index=False)
        .agg(complete=("complete", "sum"), runs=("runs", "sum"))
    )
    pooled["rate"] = pooled.complete / pooled.runs
    return pooled.to_markdown(index=False, floatfmt=".3f")


def write_report(output: Path = OUTPUT_DIR) -> None:
    validation = final_validation(output)
    decision = json.loads((output / "calibration_decision.json").read_text())
    holdout = json.loads((output / "holdout_status.json").read_text())
    completion = pd.read_parquet(output / "calibration_completion_by_condition.parquet")
    assignments = pd.read_parquet(output / "calibration_assignment_audit.parquet")
    runs = pd.read_parquet(output / "calibration_terminal_outcomes.parquet")
    exhaustive = json.loads((output / "exhaustive_matcher_validation.json").read_text())
    promoted = decision["promotedCandidateId"]
    if not promoted:
        outcome = "constraining/contradictory"
        result_sentence = (
            "No frozen candidate met completion comparability in every condition and both assignment arms; the sealed holdout was not opened and no causal/component effect was estimated."
        )
        caveat = "The conclusion is bounded to the frozen support-qualified ladder and simulator/checkpoint semantics; it does not prove that every conceivable smaller value perturbation is infeasible."
        next_action = "Return S11R for Chief Scientist review; revise the scientific question or intervention class before any separately authorized S12."
    elif not holdout.get("validationPassed", False):
        outcome = "constraining/contradictory"
        result_sentence = "A calibration candidate was promoted but failed sealed-holdout completion or validation; no causal/component effect was estimated."
        caveat = "Calibration feasibility did not generalize to the sealed holdout."
        next_action = "Return S11R for review; do not start S12."
    else:
        pooled = pd.read_csv(output / "holdout_pooled_effects.csv")
        value = pooled[(pooled.metric == "corrected_aggregation") & (pooled.arm == "value_primary") & (pooled.endpoint == "signed_area")].iloc[0]
        if value.ci95_low <= 0 <= value.ci95_high:
            outcome = "null"
        else:
            outcome = "supportive"
        result_sentence = f"Candidate {promoted} passed calibration and holdout; primary corrected-adjacency signed area was {value['mean']:.4f} (95% paired bootstrap CI {value.ci95_low:.4f} to {value.ci95_high:.4f})."
        caveat = "This is an intervention-specific direct value-field reassignment effect, not natural mediation."
        next_action = "Return S11R for Chief Scientist review; do not start S12 without a new instruction."

    stop_counts = runs.groupby(["candidate_id", "arm", "stop_reason"]).size().reset_index(name="runs")
    candidate_summary = pd.DataFrame(decision["candidateDiagnostics"])
    report = f"""# S11R — Corrected positional-inversion matching and completion-feasible value perturbation: full results

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | S11R |
| Completion status | Complete; S11R only; completed S11 and upstream artifacts preserved; S12 not started |
| Artifacts written | Frozen contract/specification and split; exhaustive matcher validation; 1,200 assignment audits and 2,400 terminal-only calibration continuations; completion, preservation, scheduler, accounting, decision, validation, provenance, status, and manifest records; calibration PNG/SVG; canonical report{'; sealed-holdout corpus and effect outputs' if holdout['executed'] else ''} |
| Validation result | {'Passed all applicable matcher, target-attainment, preservation, pairing, scheduler, budget, accounting, upstream-immutability, and S12-absence checks' if validation['allApplicableChecksPassed'] else 'One or more applicable validation checks failed; see validation summary'} |
| Outcome classification | {outcome} |
| Caveats or blockers | {caveat} |
| Recommended next action | {next_action} |

## Lay summary

S11 had a valid simulation corpus but its frozen matching helper did not
actually measure where values appeared: it counted all ordered comparisons,
which is unchanged by rearrangement. S11R replaced that helper with the true
left-before-right inversion count and proved the matcher on exhaustive small
arrangements. It then tried a frozen ladder of direct value reassignments while
keeping every Selection condition, native event budget, scheduler stream,
policy, occupancy, and Selection cursor intact. {result_sentence}

## Frozen question and decision rule

The question was whether any support-qualified direct value-position
perturbation in the frozen ladder could be matched at exact true inversion
distance zero and remain completion-comparable in all six balanced policy-pair
conditions. Calibration and holdout were split before outcomes. Calibration
could read terminal accounting only. Each of primary and matched assignments
had to complete at least 24/25 in every condition and 147/150 overall; their
condition-specific completion counts could differ by at most one. Any
invariant, event-budget stop, preservation failure, scheduler mismatch, or
target miss disqualified the candidate. The strongest passing candidate would
be promoted. No condition—especially no Selection condition—could be dropped.

## Inputs

- Governance: `AGENTS.md`, `FULL_PLAN.md`, and `RESEARCH_PLAN.md`.
- Supplied paper: the attachment manifest/sidecar and extracted paper Markdown. Natural paper runs keep Value and Algotype fixed; repeated values only partially dissociate sorting pressure from clustering.
- E01 transition contract: identity-owned state, counter-addressed scheduler, and terminal/budget precedence.
- Validated S01–S11 artifacts; S10's unique-value physical-feasibility boundary, 76% primary completion, and policy-pair heterogeneity; S11's completion failures and corrected post-freeze diagnostic.
- Reproducible repository code: `analysis/s11r_corrective.py`, `analysis/s11r_corrective_contract.json`, and focused tests at Git commit `{git_output('rev-parse', 'HEAD')}`.

## Methods

### Outcome-blind split

For each of the six conditions, ordinals used by S07 calibration (`1 mod 10`)
or S07/S11 holdout (`5 mod 10`) were excluded. The remaining 200 were ranked by
a frozen SHA-256 address: 25 calibration, 25 sealed holdout, and 150 reserve.
Calibration and holdout therefore share no scenario and neither reuses an S11
continuation outcome.

### Correct matcher

For a positional sequence `v`, true inversion count is the number of pairs
`i<j` with `v[i]>v[j]`; ties contribute zero. Every candidate bank contained
4,096 distinct global-value-multiset-preserving assignments with exactly the
declared number of changed positions. A target required at least 16 distinct
assignments at exactly one post-intervention inversion count. Primary and
matched assignments were distinct, both attained that count (distance zero),
then minimized L1-change mismatch and maximized assignment diversity.

Exhaustive validation checked {exhaustive['uniqueArrangementsChecked']:,} unique
and {exhaustive['repeatedArrangementsChecked']:,} repeated-value arrangements
through n={exhaustive['maximumRepeatedN']}, plus
{exhaustive['matcherCandidatePairsCompared']:,} candidate-pair objectives across
{exhaustive['matcherCasesChecked']} exact-target matcher cases.

### Interventions and continuations

At each carried-forward S06 peak checkpoint, S11R directly reassigned the value
fields implied by the positional assignment. Occupancy and all non-value cell
fields stayed exact. Selection cursors remained with their identities. Seed,
runtime key, scheduler, activation count, random-stream counters, ledger, and
the one-million-activation budget were preserved. This changes explicit task
progress and may change policy–value association; it is not pure mediation.

Calibration did not retain aggregation or Sortedness trajectories. Only after
a candidate passed all gates could the sealed holdout run no-switch, sham,
primary, and matched continuations on a common activation exposure and estimate
paired corrected-adjacency and reference-Sortedness effects.

## Calibration results

### Pooled terminal accounting

{_format_completion(completion)}

### Candidate gates

{candidate_summary[['candidateId','changedPositions','targetMode','pooledPrimaryComplete','pooledMatchedComplete','allConditionArmGatesPass','passed']].to_markdown(index=False)}

### Stop reasons

{stop_counts.to_markdown(index=False)}

Selected candidate: `{promoted or 'none'}`. Holdout authorized: `{decision['holdoutAuthorized']}`.

{result_sentence}

## Validation

- Exhaustive small-case matcher: passed; intended distance zero was attained.
- Production target attainment: {int(assignments.targetAttained.sum())}/{len(assignments)} assignment pairs passed; support qualification: {int(assignments.supportQualified.sum())}/{len(assignments)}.
- Calibration preservation: `{validation['allCalibrationPreservationPassed']}`; scheduler coupling: `{validation['allCalibrationSchedulerCouplingPassed']}`; budgets respected: `{validation['allCalibrationBudgetsRespected']}`.
- Accounting: 150 calibration scenarios, 1,200 scenario-candidate assignment pairs, and 2,400 continuations.
- Upstream S01–S11 artifact manifests and contents stayed byte-valid: `{validation['upstreamArtifactsImmutable']}`.
- S12 artifact directory absent: `{validation['s12Absent']}`.
- Applicable validation result: `{validation['allApplicableChecksPassed']}`.

## Commands

```bash
python -m pytest tests/test_e04_s11r_corrective.py tests/test_e04_s11_postfreeze_audit.py tests/test_e04_value_policy_shuffles.py -q
python -m ruff check analysis/s11r_corrective.py tests/test_e04_s11r_corrective.py
python -m py_compile analysis/s11r_corrective.py
python -m analysis.s11r_corrective freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.s11r_corrective calibrate --workers 8
python -m analysis.s11r_corrective select
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.s11r_corrective holdout --workers 8
python -m analysis.s11r_corrective analyze-holdout
python -m analysis.s11r_corrective report
python -m analysis.s11r_corrective package
```

## Dependencies and runtime

Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__},
PyArrow {pa.__version__}, and Matplotlib {matplotlib.__version__}; no new
dependency was installed. Calibration used eight process workers with BLAS/OpenMP
threads capped at one per worker. Disposable restart state is under
`/cache/e04_s11r`; compact evidence is under `/artifacts/research_steps/S11R`.

## Caveats, failed assumptions, and claim boundary

- Direct value reassignment changes the sort task and possibly policy–value association. It is not a pure mediator intervention or natural indirect effect.
- Completion comparability is a validity condition, not an outcome to optimize after holdout inspection.
- The candidate ladder is broad but finite. Failure constrains this declared intervention family; it does not prove logical impossibility for every value perturbation.
- The corrected matcher fixes the specific S11 distance error; it does not retroactively change S11 assignments or trajectories.
- Simulator evidence is computational proxy evidence, not wet-lab or general causal proof.

## Provenance and artifact locations

The frozen contract, implementation and Git hashes are in `freeze_record.json`;
scenario IDs are in `scenario_split.parquet`; exact matcher evidence is in
`exhaustive_matcher_validation.json`; terminal, assignment, preservation, and
scheduler evidence are in the calibration Parquet files; the decision is in
`calibration_decision.json`; final checks are in `validation_summary.json` and
`upstream_immutability_audit.json`; checksums are in `artifact_manifest.json`.

## Recommended next action

{next_action} Completed S11 and all upstream artifacts remain intact. S12 has
not been started.
"""
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    status = {
        "researchStepId": "S11R",
        "stepNumber": 11,
        "success": bool(validation["allApplicableChecksPassed"]),
        "status": "complete",
        "artifactsWritten": [
            "research_step_full_results.md",
            "corrective_specification.md",
            "calibration_decision.json",
            "calibration_terminal_outcomes.parquet",
            "calibration_assignment_audit.parquet",
            "validation_summary.json",
            "artifact_manifest.json",
        ],
        "validationResult": "passed" if validation["allApplicableChecksPassed"] else "failed",
        "outcomeClassification": outcome,
        "caveatsOrBlockers": [caveat],
        "recommendedNextAction": next_action,
        "s12Started": False,
    }
    write_json(output / "status.json", status)


def package(output: Path = OUTPUT_DIR) -> None:
    assert_frozen(output)
    commands = {
        "schema": "e04.s11r.commands.v1",
        "researchStepId": "S11R",
        "commands": [
            "python -m pytest tests/test_e04_s11r_corrective.py tests/test_e04_s11_postfreeze_audit.py tests/test_e04_value_policy_shuffles.py -q",
            "python -m ruff check analysis/s11r_corrective.py tests/test_e04_s11r_corrective.py",
            "python -m py_compile analysis/s11r_corrective.py",
            "python -m analysis.s11r_corrective freeze",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.s11r_corrective calibrate --workers 8",
            "python -m analysis.s11r_corrective select",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m analysis.s11r_corrective holdout --workers 8",
            "python -m analysis.s11r_corrective analyze-holdout",
            "python -m analysis.s11r_corrective report",
            "python -m analysis.s11r_corrective package",
        ],
    }
    write_json(output / "commands.json", commands)
    write_json(
        output / "environment.json",
        {
            "schema": "e04.s11r.environment.v1",
            "researchStepId": "S11R",
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "matplotlib": matplotlib.__version__,
            "cpuCount": os.cpu_count(),
            "workersUsed": 8,
        },
    )
    write_json(
        output / "provenance.json",
        {
            "schema": "e04.s11r.provenance.v1",
            "researchStepId": "S11R",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "repository": str(REPOSITORY),
            "branch": git_output("branch", "--show-current"),
            "gitHead": git_output("rev-parse", "HEAD"),
            "contractPath": str(CONTRACT_PATH),
            "contractSha256": sha256_file(CONTRACT_PATH),
            "implementationPath": str(Path(__file__).resolve()),
            "implementationSha256": sha256_file(Path(__file__).resolve()),
            "cachePath": str(CACHE_DIR),
            "cacheIsDisposable": True,
            "s12Absent": not S12_DIR.exists(),
        },
    )
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        output / "artifact_manifest.json",
        {
            "schema": "e04.s11r.artifact_manifest.v1",
            "researchStepId": "S11R",
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("select")
    holdout = subparsers.add_parser("holdout")
    holdout.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("analyze-holdout")
    subparsers.add_parser("validate")
    subparsers.add_parser("report")
    subparsers.add_parser("package")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze_design(), indent=2))
    elif args.command == "calibrate":
        if not 1 <= args.workers <= 8:
            raise ValueError("workers must be in [1,8]")
        print(json.dumps(run_calibration(workers=args.workers), indent=2))
    elif args.command == "select":
        print(json.dumps(analyze_calibration(), indent=2))
    elif args.command == "holdout":
        if not 1 <= args.workers <= 8:
            raise ValueError("workers must be in [1,8]")
        print(json.dumps(run_holdout(workers=args.workers), indent=2))
    elif args.command == "analyze-holdout":
        print(json.dumps(analyze_holdout(), indent=2))
    elif args.command == "validate":
        print(json.dumps(final_validation(), indent=2))
    elif args.command == "report":
        write_report()
    elif args.command == "package":
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
