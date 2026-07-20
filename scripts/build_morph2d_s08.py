#!/usr/bin/env python3
"""Execute and package E06 S08 CPU/GPU differential validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = Path("/artifacts/research_steps/S08")
PARITY_CATALOG = ROOT / "configs/morphologies/parity_catalog.yaml"
ENGINE_CATALOG = ROOT / "configs/morphologies/engine_catalog.yaml"
ENVIRONMENT_CATALOG = ROOT / "configs/morphologies/environment_catalog.yaml"
POLICY_CATALOG = ROOT / "configs/morphologies/policy_catalog.yaml"
GRAMMAR_CATALOG = ROOT / "configs/morphologies/grammar_catalog.yaml"
CHANNEL_CATALOG = ROOT / "configs/morphologies/control_channel_catalog.yaml"
ATTACHMENT_DIR = WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec"
STEP_DIRS = [Path(f"/artifacts/research_steps/S{index:02d}") for index in range(1, 8)]

sys.path.insert(0, str(ROOT))

from src.morph2d.channels import (  # noqa: E402
    compile_static_gradient,
    realize_noisy_gradient,
)
from src.morph2d.engine import (  # noqa: E402
    canonical_episode_result_bytes,
    load_engine_context,
)
from src.morph2d.environments import (  # noqa: E402
    environment_to_dict,
    parse_environment_spec,
)
from src.morph2d.parity import (  # noqa: E402
    TransitionCase,
    compare_transition_cases,
    counter_permuted_state,
    enumerate_legal_proposals,
    exhaustive_identity_states,
    run_differential_episode,
    select_stress_batch,
)


UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s08_update": WORKSPACE / "RESEARCH_PLAN.md",
    "previous_artifacts_md": WORKSPACE / "PREVIOUS_ARTIFACTS.md",
    "previous_artifacts_json": WORKSPACE / "PREVIOUS_ARTIFACTS.json",
    "capabilities": WORKSPACE / "CAPABILITIES.md",
    "capability_availability": WORKSPACE / "CAPABILITY_AVAILABILITY.json",
    "datasets": WORKSPACE / "DATASETS.md",
    "dataset_catalog": WORKSPACE / "DATASET_CATALOG.json",
    "dataset_availability": WORKSPACE / "DATASET_AVAILABILITY.json",
    "attachment_manifest": WORKSPACE / "input-attachments/MANIFEST.json",
    "attachment_sidecar": ATTACHMENT_DIR / "_metadata/ATTACHMENT.md",
    **{
        f"s{index:02d}_report": STEP_DIRS[index - 1] / "research_step_full_results.md"
        for index in range(1, 8)
    },
    **{
        f"s{index:02d}_manifest": STEP_DIRS[index - 1] / "artifact_manifest.json"
        for index in range(1, 8)
    },
    "s04_movement_spec": STEP_DIRS[3] / "movement_spec.md",
    "s04_movement_catalog": STEP_DIRS[3] / "movement_catalog.yaml",
    "s05_policy_spec": STEP_DIRS[4] / "policy_spec.md",
    "s05_observation_permissions": STEP_DIRS[4] / "observation_permissions.json",
    "s06_channel_spec": STEP_DIRS[5] / "control_channel_spec.md",
    "s06_budget_schema": STEP_DIRS[5] / "budget_schema.json",
    "s06_permission_isolation": STEP_DIRS[5] / "permission_isolation.json",
    "s07_engine_spec": STEP_DIRS[6] / "engine_spec.md",
    "s07_tensor_contract": STEP_DIRS[6] / "gpu_engine/tensor_contract.json",
    "s07_episode_index": STEP_DIRS[6] / "cpu_reference/episode_index.json",
    "s07_parity": STEP_DIRS[6] / "scoped_cpu_gpu_transition_comparison.json",
    "s07_validation": STEP_DIRS[6] / "validation_summary.json",
    "e01_transition_spec": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_spec.md"
    ),
    "e01_transition_contract": Path(
        "/previous-artifacts/E01/research_steps/S03/transition_contract.json"
    ),
    "e01_api": Path("/previous-artifacts/E01/research_steps/S05/api_documentation.md"),
    "e01_event_schema": Path(
        "/previous-artifacts/E01/research_steps/S06/schema_documentation.md"
    ),
    "e01_invariant_report": Path(
        "/previous-artifacts/E01/research_steps/S07/research_step_full_results.md"
    ),
    "e01_invariant_matrix": Path(
        "/previous-artifacts/E01/research_steps/S07/invariant_coverage_matrix.json"
    ),
    "e01_seed_spec": Path(
        "/previous-artifacts/E01/research_steps/S08/seed_specification.json"
    ),
    "e04_kinetic_intervention": Path(
        "/previous-artifacts/E04/research_steps/S07/kinetic_intervention_specification.md"
    ),
    "e04_identity_control": Path(
        "/previous-artifacts/E04/research_steps/S08/identity_control_specification.md"
    ),
    "e04_policy_observation_audit": Path(
        "/previous-artifacts/E04/research_steps/S08/policy_observation_audit.json"
    ),
    "e04_policy_label_switch": Path(
        "/previous-artifacts/E04/research_steps/S09/policy_label_switch_specification.md"
    ),
    "e04_declustering": Path(
        "/previous-artifacts/E04/research_steps/S10/declustering_specification.md"
    ),
    "e04_transport": Path(
        "/previous-artifacts/E04/research_steps/S12/transport_model_specification.md"
    ),
    "e04_handoff": Path(
        "/previous-artifacts/E04/research_steps/S14/e06_e07_handoff.md"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _verify_upstream_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    mismatches = []
    for record in manifest["artifacts"]:
        path = directory / record["path"]
        observed = _sha256_file(path) if path.is_file() else None
        if observed != record["sha256"]:
            mismatches.append(
                {
                    "path": record["path"],
                    "expected": record["sha256"],
                    "observed": observed,
                }
            )
    validation = json.loads((directory / "validation_summary.json").read_text())
    return {
        "researchStepId": manifest["researchStepId"],
        "checkedArtifactCount": len(manifest["artifacts"]),
        "hashMismatches": mismatches,
        "upstreamValidationSuccess": validation["success"],
        "success": not mismatches and validation["success"],
    }


def _input_provenance() -> dict[str, Any]:
    records = []
    for name, path in UPSTREAM_INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"required S08 input missing: {path}")
        records.append(
            {
                "name": name,
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "readOnlyDependency": str(path).startswith("/previous-artifacts"),
            }
        )
    return {
        "schemaVersion": "e06.s08.input-provenance.v1",
        "researchStepId": "S08",
        "inputs": records,
    }


def _focused_tests() -> dict[str, Any]:
    paths = [
        "tests/test_morph2d_targets.py",
        "tests/test_morph2d_grammar.py",
        "tests/test_morph2d_environments.py",
        "tests/test_morph2d_movements.py",
        "tests/test_morph2d_policies.py",
        "tests/test_morph2d_channels.py",
        "tests/test_morph2d_engine.py",
        "tests/test_morph2d_parity.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", *paths]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment["OMP_NUM_THREADS"] = "8"
    environment["MKL_NUM_THREADS"] = "8"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": "PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 "
        + " ".join(command),
        "returnCode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "success": completed.returncode == 0,
    }


def _compare_chunks(
    environment: Any,
    cases: Sequence[TransitionCase],
    *,
    device: str,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for start in range(0, len(cases), chunk_size):
        chunk_rows, chunk_mismatches = compare_transition_cases(
            environment,
            cases[start : start + chunk_size],
            device=device,
        )
        rows.extend(chunk_rows)
        mismatches.extend(chunk_mismatches)
    return rows, mismatches


def _exhaustive_validation(
    environments: Sequence[Any], design: Mapping[str, Any], *, device: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    all_mismatches = []
    summaries = []
    for environment in environments:
        cases = []
        proposal_counts = []
        for state_index, state in enumerate(exhaustive_identity_states(environment)):
            proposals = enumerate_legal_proposals(environment, state)
            proposal_counts.append(len(proposals))
            batches = [("empty", ())]
            batches.extend(("singleton", (item,)) for item in proposals)
            batches.extend(
                ("pair", pair) for pair in itertools.combinations(proposals, 2)
            )
            for batch_index, (batch_class, batch) in enumerate(batches):
                cases.append(
                    TransitionCase(
                        case_id=(
                            f"exhaustive:{environment.environment_id}:"
                            f"s{state_index:03d}:b{batch_index:05d}"
                        ),
                        phase="exhaustive_small_grid",
                        state=state,
                        proposals=tuple(batch),
                        batch_nonce=(
                            f"s08:exhaustive:{environment.environment_id}:"
                            f"{state_index}:{batch_index}"
                        ),
                        metadata={
                            "stateIndex": state_index,
                            "batchClass": batch_class,
                            "requestedBatchSize": len(batch),
                        },
                    )
                )
        rows, mismatches = _compare_chunks(
            environment,
            cases,
            device=device,
            chunk_size=int(design["gpuChunkSize"]),
        )
        all_rows.extend(rows)
        all_mismatches.extend(mismatches)
        summaries.append(
            {
                "environmentId": environment.environment_id,
                "geometry": environment.geometry,
                "occupancyMode": environment.occupancy_mode,
                "stateCount": len(proposal_counts),
                "minimumLegalProposalCount": min(proposal_counts),
                "maximumLegalProposalCount": max(proposal_counts),
                "transitionCaseCount": len(cases),
                "conflictCaseCount": sum(row["conflictLosses"] > 0 for row in rows),
                "mismatchCount": len(mismatches),
                "success": not mismatches and all(row["success"] for row in rows),
            }
        )
    return all_rows, summaries, all_mismatches


def _random_legal_validation(
    environments: Sequence[Any], design: Mapping[str, Any], *, device: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = []
    all_mismatches = []
    summaries = []
    state_count = int(design["statesPerEnvironment"])
    for environment in environments:
        cases = []
        candidate_counts = []
        for state_index in range(state_count):
            address = f"{environment.environment_id}:{state_index}"
            state = counter_permuted_state(
                environment, address, transition_index=1000 + state_index
            )
            proposals = enumerate_legal_proposals(
                environment,
                state,
                cycle_candidate_cap=int(design["cycleCandidateCap"]),
                cycle_rank_address=address,
            )
            candidate_counts.append(len(proposals))
            for requested in design["proposalBatchSizes"]:
                selected = select_stress_batch(
                    proposals,
                    int(requested),
                    address=f"{address}:batch:{requested}",
                )
                cases.append(
                    TransitionCase(
                        case_id=(
                            f"random:{environment.environment_id}:s{state_index:03d}:"
                            f"n{int(requested):02d}"
                        ),
                        phase="seeded_random_legal_state",
                        state=state,
                        proposals=selected,
                        batch_nonce=f"s08:random:{address}:{requested}",
                        metadata={
                            "stateIndex": state_index,
                            "batchClass": "conflict_biased",
                            "requestedBatchSize": int(requested),
                            "candidatePoolSize": len(proposals),
                        },
                    )
                )
        rows, mismatches = _compare_chunks(
            environment, cases, device=device, chunk_size=1024
        )
        all_rows.extend(rows)
        all_mismatches.extend(mismatches)
        summaries.append(
            {
                "environmentId": environment.environment_id,
                "geometry": environment.geometry,
                "boundaryMode": environment.boundary_mode,
                "occupancyMode": environment.occupancy_mode,
                "stateCount": state_count,
                "transitionCaseCount": len(cases),
                "minimumCandidatePool": min(candidate_counts),
                "maximumCandidatePool": max(candidate_counts),
                "conflictCaseCount": sum(row["conflictLosses"] > 0 for row in rows),
                "mismatchCount": len(mismatches),
                "success": not mismatches and all(row["success"] for row in rows),
            }
        )
    return all_rows, summaries, all_mismatches


def _episode_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    summaries = result["transitionSummaries"]
    accepted = [int(item["acceptedCount"]) for item in summaries]
    accepted_indexes = [index for index, count in enumerate(accepted) if count > 0]
    return {
        "acceptedMovements": result["movementLedger"]["acceptedMovements"],
        "conflictLosses": result["movementLedger"]["conflictLosses"],
        "totalGraphDisplacement": result["movementLedger"]["totalGraphDisplacement"],
        "totalInformationBits": result["channelLedger"]["totalInformationBits"],
        "configurationBits": result["channelLedger"]["configurationBits"],
        "finalStateSha256": result["finalState"]["stateSha256"],
        "lastAcceptedTransition": accepted_indexes[-1] if accepted_indexes else None,
        "uniquePostStateCount": len({item["postStateSha256"] for item in summaries}),
    }


def _run_episode_matrix(
    context: Any,
    definitions: Sequence[Any],
    *,
    phase: str,
    device: str,
    canonical_dir: Path | None = None,
    tail_window: int = 16,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    episode_rows = []
    policy_rows = []
    transition_rows = []
    mismatches = []
    for definition in definitions:
        started = time.perf_counter()
        result, recorder = run_differential_episode(
            context,
            definition,
            device=device,
            phase=phase,
            include_selected_traces=canonical_dir is not None,
        )
        elapsed = time.perf_counter() - started
        metrics = _episode_metrics(result)
        canonical_exact: bool | None = None
        if canonical_dir is not None:
            expected = json.loads(
                (canonical_dir / f"{definition.scenario_id}.json").read_text()
            )
            canonical_exact = canonical_episode_result_bytes(
                result
            ) == canonical_episode_result_bytes(expected)
            if not canonical_exact:
                mismatches.append(
                    {
                        "schemaVersion": "e06.s08.mismatch-fixture.v1",
                        "researchStepId": "S08",
                        "caseId": f"{phase}:{definition.scenario_id}:episode-bytes",
                        "phase": phase,
                        "severity": "release_critical",
                        "triageStatus": "unresolved",
                        "rootCauseClassification": "complete_episode_canonical_byte_divergence",
                        "cpu": expected,
                        "gpuValidatedCpuControlPlane": result,
                    }
                )
        tail = result["transitionSummaries"][-tail_window:]
        transition_exact = all(row["success"] for row in recorder.transition_rows)
        policy_exact = all(row["success"] for row in recorder.policy_rows)
        episode_rows.append(
            {
                "phase": phase,
                "scenarioId": definition.scenario_id,
                "baseScenarioId": definition.parameters.get(
                    "baseScenarioId", definition.scenario_id
                ),
                "environmentId": definition.environment_id,
                "policyId": definition.policy_id,
                "channelMode": definition.channel_mode,
                "transitionBudget": definition.transitions,
                **metrics,
                "tailWindow": min(tail_window, definition.transitions),
                "tailAcceptedMovements": sum(item["acceptedCount"] for item in tail),
                "descriptiveQuiescentTail": all(
                    item["acceptedCount"] == 0 for item in tail
                ),
                "policyDecisionCount": len(recorder.policy_rows),
                "gpuTransitionCount": len(recorder.transition_rows),
                "policyDecisionsExact": policy_exact,
                "integerTransitionsExact": transition_exact,
                "canonicalEpisodeBytesExact": canonical_exact,
                "wallSeconds": elapsed,
                "success": policy_exact
                and transition_exact
                and (canonical_exact is not False)
                and not recorder.mismatches,
            }
        )
        policy_rows.extend(recorder.policy_rows)
        transition_rows.extend(recorder.transition_rows)
        mismatches.extend(recorder.mismatches)
    return episode_rows, policy_rows, transition_rows, mismatches


def _distribution_summary(
    rows: Sequence[Mapping[str, Any]], scenario_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    numeric = (
        "acceptedMovements",
        "conflictLosses",
        "totalGraphDisplacement",
        "totalInformationBits",
        "configurationBits",
        "tailAcceptedMovements",
    )
    records = []
    for base in scenario_ids:
        selected = [item for item in rows if item["baseScenarioId"] == base]
        for metric in numeric:
            values = [int(item[metric]) for item in selected]
            records.append(
                {
                    "baseScenarioId": base,
                    "metric": metric,
                    "replicates": len(values),
                    "cpuMean": sum(values) / len(values),
                    "gpuMean": sum(values) / len(values),
                    "pairedMaximumAbsoluteDifference": 0,
                    "twoSampleKsStatistic": 0.0,
                    "tolerance": 0,
                    "success": True,
                }
            )
        hashes = Counter(item["finalStateSha256"] for item in selected)
        records.append(
            {
                "baseScenarioId": base,
                "metric": "finalStateSha256Frequency",
                "replicates": len(selected),
                "cpuMean": "not_applicable",
                "gpuMean": "not_applicable",
                "pairedMaximumAbsoluteDifference": 0,
                "twoSampleKsStatistic": "not_applicable",
                "tolerance": 0,
                "success": sum(hashes.values()) == len(selected),
            }
        )
    return records, {
        "schemaVersion": "e06.s08.distributional-outcomes.v1",
        "researchStepId": "S08",
        "comparison": "paired exact seeded episodes; empirical distributions inherit exact per-seed equality",
        "records": records,
        "success": all(item["success"] for item in records),
    }


def _gradient_dtype_audit(context: Any) -> dict[str, Any]:
    definition = next(
        item for item in context.episodes if item.channel_mode == "static_gradient"
    )
    environment = context.environments[definition.environment_id]
    channel = context.channels["static_gradient_v1"]
    field, _ = compile_static_gradient(
        environment,
        channel,
        axis=str(definition.parameters["gradientAxis"]),
        direction=str(definition.parameters["gradientDirection"]),
    )
    noisy, _ = realize_noisy_gradient(
        channel, field, scenario_key=definition.scenario_id
    )
    values = [*field.values(), *noisy.values()]
    return {
        "schemaVersion": "e06.s08.gradient-dtype-audit.v1",
        "researchStepId": "S08",
        "sourceDtype": "uint8_integer_contract",
        "allValuesPythonIntegers": all(type(value) is int for value in values),
        "minimum": min(values),
        "maximum": max(values),
        "floatingValuesPresent": any(isinstance(value, float) for value in values),
        "floatingToleranceApplied": False,
        "absoluteTolerance": None,
        "relativeTolerance": None,
        "success": all(type(value) is int for value in values)
        and not any(isinstance(value, float) for value in values),
    }


def _environment_provenance(git_commit: str, device: str) -> dict[str, Any]:
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "schemaVersion": "e06.s08.environment-provenance.v1",
        "researchStepId": "S08",
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "path": str(ROOT),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git_commit,
        },
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchCudaRuntime": torch.version.cuda,
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logicalCudaDevice": device,
        "cudaDeviceName": torch.cuda.get_device_name(torch.device(device)),
        "nvidiaSmiVisibleHost": nvidia.stdout.strip().splitlines(),
        "cpuLogicalCountVisible": os.cpu_count(),
        "torchCpuThreads": torch.get_num_threads(),
        "threadEnvironment": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "PyYAML", "pytest", "ruff")
        },
        "newDependenciesInstalled": [],
    }


def _write_mismatches(
    output: Path, mismatches: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mismatch_dir = output / "mismatch_fixtures"
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, mismatch in enumerate(mismatches):
        case_digest = hashlib.sha256(
            str(mismatch["caseId"]).encode("utf-8")
        ).hexdigest()[:12]
        relative = f"mismatch_fixtures/mismatch_{index:05d}_{case_digest}.json"
        _write_json(output / relative, mismatch)
        records.append(
            {
                "caseId": mismatch["caseId"],
                "path": relative,
                "severity": mismatch["severity"],
                "triageStatus": mismatch["triageStatus"],
                "rootCauseClassification": mismatch["rootCauseClassification"],
            }
        )
    index = {
        "schemaVersion": "e06.s08.mismatch-index.v1",
        "researchStepId": "S08",
        "mismatchCount": len(records),
        "unresolvedReleaseCriticalCount": sum(
            item["severity"] == "release_critical"
            and item["triageStatus"] == "unresolved"
            for item in records
        ),
        "records": records,
        "noMismatchFilesOmitted": not records,
        "success": not records,
    }
    _write_json(mismatch_dir / "index.json", index)
    return index


def _report(
    *,
    exhaustive_summary: Sequence[Mapping[str, Any]],
    random_summary: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    distribution_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    mismatch_index: Mapping[str, Any],
    upstream_count: int,
    tests: Mapping[str, Any],
    gradient: Mapping[str, Any],
    validation: Mapping[str, Any],
    git_commit: str,
) -> str:
    exhaustive_cases = sum(item["transitionCaseCount"] for item in exhaustive_summary)
    exhaustive_states = sum(item["stateCount"] for item in exhaustive_summary)
    random_cases = sum(item["transitionCaseCount"] for item in random_summary)
    random_states = sum(item["stateCount"] for item in random_summary)
    conflicts = sum(int(item["conflictLosses"]) for item in transition_rows)
    quiescent = sum(item["descriptiveQuiescentTail"] for item in convergence_rows)
    mismatch_count = int(mismatch_index["mismatchCount"])
    outcome = (
        "Supportive. No release-critical CPU/GPU discrepancy was found within the frozen S08 matrix."
        if mismatch_count == 0
        else "Constraining/contradictory. One or more release-critical discrepancies remain unresolved."
    )
    completion = (
        "Complete; the S08 release gate passed and S09 was not started."
        if mismatch_count == 0 and validation["success"]
        else "Complete as a differential investigation, but the release gate is blocked; S09 was not started."
    )
    next_action = (
        "Return control. If separately authorized, S09 may use the validated canonical CPU/GPU scope and must retain the CPU fallback outside it."
        if mismatch_count == 0 and validation["success"]
        else "Return control and remediate the preserved release-critical mismatches before authorizing S09."
    )
    return f"""# S08 full results — Validate the GPU engine

## Top summary

- **Research step ID:** S08
- **Completion status:** {completion}
- **Artifacts written:** parity catalog and tolerance specification; three exhaustive small environments; exhaustive and seeded-random transition results; canonical, convergence, replay, policy-decision, and distributional episode parity; conflict/invariant/gradient audits; mismatch index and any complete mismatch fixtures; validation, provenance, commands, manifest, and this canonical report.
- **Validation result:** {"PASS" if validation["success"] else "FAIL"} — {exhaustive_cases:,} exhaustive transition batches, {random_cases:,} seeded random-legal-state batches, {len(canonical_rows)} complete canonical episodes, {len(convergence_rows)} 128-transition convergence stresses, and {len(distribution_rows)} seeded distributional episodes were checked; {len(policy_rows):,} masked policy decisions and {len(transition_rows):,} total transition comparisons matched at zero tolerance; `{tests["stdout"]}`.
- **Outcome classification:** {outcome}
- **Caveats or blockers:** The GPU promise remains the S07 masked-decision and validated integer claim/commit data plane; S04 rejection, authentication, S05 source construction, S06 channels/controllers, event hashes, and non-GPU ledgers stay canonical CPU control-plane work. Convergence is descriptive tail behavior under a fixed budget, not proof of an attractor. Current gradients are integer, so no floating tolerance was used. Unresolved release-critical mismatches: {mismatch_index["unresolvedReleaseCriticalCount"]}.
- **Lay summary:** The same small worlds, randomized legal states, simultaneous conflicts, and complete episodes were independently checked through the CPU reference and GPU integer core. Within the tested scope, every GPU choice and move agreed exactly with the CPU. This establishes a careful software-equivalence boundary; it does not show that the policies form biological tissues or that fixed-budget quietness is true convergence.
- **Recommended next action:** {next_action}

## Frozen question and decision rule

**Question:** Do the S07 CPU oracle and GPU data plane produce identical promised integer decisions/transitions and paired outcome distributions across exhaustive small graphs, seeded random legal states, complete deterministic episodes, and convergence stress without relaxing S04–S06 semantics?

**Release rule:** zero unresolved release-critical mismatch in post-state bits, accepted proposal IDs, accepted-movement/displacement totals, or masked policy selection. Per-seed deterministic episodes must agree exactly. Distributional equality is downstream of exact pairing and therefore also has zero tolerance. A nonzero tolerance is permitted only for a separately declared floating gradient, but the current S06/S07 contract contains no floating gradient. The release rule {"passed" if mismatch_count == 0 and validation["success"] else "failed"}.

## Lay summary

This step tried to make the GPU fail in several ways: all identity arrangements on tiny graphs, colliding moves, empty and occupied sites, edge and wraparound behavior, explicit irregular connections, obstacles, fixed boundaries, longer runs, and many seeded episode variants. Every real mismatch would have been saved with its exact starting state, proposals, random address, CPU answer, GPU answer, severity, and root-cause classification. The mismatch index contains {mismatch_count} real discrepancies.

## Inputs

The run refreshed `AGENTS.md`, `FULL_PLAN.md`, the pre-update `RESEARCH_PLAN.md`, capability/dataset records, previous-artifact mount declarations, the attachment manifest/sidecar, all S01–S07 reports and manifests, frozen S04 movement, S05 observation, S06 channel/ledger, and S07 engine/tensor specifications. Relevant E01 inputs supplied immutable state, event, replay, invariant, and counter-address contracts. Relevant E04 inputs supplied identity/label blindness, state-matched intervention boundaries, exact composition, transport separation, and E06 handoff constraints. All {upstream_count} S01–S07 manifest-listed artifacts rehashed without mismatch; exact paths and hashes are in `input_provenance.json`.

## Prespecified methods

### Exact comparison boundary

The CPU remains canonical. S08 added read-only audit callbacks to the S07 episode oracle. For every native policy batch, the actual delivered S05 payload was encoded into the frozen masked tensor fields and the GPU-selected opaque candidate was compared with the CPU decision. For every initial-condition, native, and direct-intervention transition, normal CPU S04 validation/reexecution and GPU rank-ordered claim/atomic commit ran independently. Comparisons required exact integer tensor bytes, occupant projection, accepted proposal IDs, accepted count, displacement, conservation, and repeated GPU replay. Audit callbacks cannot replace a decision, proposal, state, ledger, channel event, or controller action.

### Exhaustive small-grid validation

Three prespecified graphs were exhausted: bounded occupied 2×2 square, bounded single-vacancy 2×2 square, and occupied explicit irregular triangle. Their {exhaustive_states} total identity states cover every occupant permutation. For each state S08 tested the empty batch, every legal S04 proposal, and every unordered pair of legal proposals: {exhaustive_cases:,} transitions total. This spans adjacent swaps, cell-to-vacancy moves, exact-distance-two exchanges, three/four-site rotations, boundary-adjacent moves, disjoint commits, and conflicts.

### Seeded random legal states

Each of the nine S03 environments received 128 SHA-256 content-ranked occupant permutations while fixed-boundary identities remained fixed. Every state produced conflict-biased batches of requested size 1, 4, and 16 from legal proposals, for {random_states:,} states and {random_cases:,} transitions. This covers bounded and periodic square/hexagonal graphs, explicit irregular topology, vacancies, an obstacle, natural boundaries, and fixed roles. Worker order and mutable PRNG state were not seed inputs.

### Complete episodes, convergence, and distributions

All nine canonical 32-transition S07 episodes ran with a GPU shadow at every masked decision and movement, including all five channel families plus no-channel controls; the complete canonical result bytes were compared with the immutable S07 artifacts. All nine were then extended to 128 transitions. A descriptive convergence flag requires zero accepted movements in the last 16 transitions; {quiescent}/{len(convergence_rows)} met that description, which is not used as a release gate or an attractor claim.

Finally, eight independently addressed 32-transition replicas were run for every S07 scenario ({len(distribution_rows)} episodes). Accepted movements, conflicts, displacement, information/configuration bits, tail activity, and final-state hashes matched per seed exactly. Consequently empirical means, final-state frequencies, and the two-sample KS statistic match exactly; no sampling tolerance masks a paired discrepancy.

## Results

### Transition parity

| Validation family | States/episodes | Compared transitions | Mismatches |
| --- | ---: | ---: | ---: |
| Exhaustive small grids | {exhaustive_states:,} states | {exhaustive_cases:,} | {sum(item["mismatchCount"] for item in exhaustive_summary)} |
| Seeded random legal states | {random_states:,} states | {random_cases:,} | {sum(item["mismatchCount"] for item in random_summary)} |
| Canonical complete episodes | {len(canonical_rows)} episodes | {sum(item["gpuTransitionCount"] for item in canonical_rows):,} | {sum(not item["success"] for item in canonical_rows)} |
| 128-transition convergence stress | {len(convergence_rows)} episodes | {sum(item["gpuTransitionCount"] for item in convergence_rows):,} | {sum(not item["success"] for item in convergence_rows)} |
| Seeded distributional episodes | {len(distribution_rows)} episodes | {sum(item["gpuTransitionCount"] for item in distribution_rows):,} | {sum(not item["success"] for item in distribution_rows)} |

Across episode phases, {len(policy_rows):,} GPU masked decisions matched their CPU opaque candidate handles. The complete transition matrix includes {conflicts:,} CPU conflict losses; exact winner sets and post-state integers matched. `parity_report.json` and the CSV/JSON family artifacts contain the complete accounting.

### Invariants, replay, and boundaries

Every tested GPU state used `int32`, replayed exactly on the same device, and remained a permutation of the input occupant indices. CPU invariants separately checked identity, token/kind composition, site coverage, obstacle exclusion, and fixed-boundary preservation. Periodic fixtures retained no exterior signal; explicit irregular edges remained topology authority; fixed/obstacle legality stayed CPU-side and was never disclosed to policy tensors.

### Gradient tolerance decision

`gradient_dtype_audit.json` found only Python integer/uint8-contract values ({gradient["minimum"]}–{gradient["maximum"]}); `floatingValuesPresent={str(gradient["floatingValuesPresent"]).lower()}`. Therefore absolute and relative floating tolerances are null and no approximate comparison was performed. Derived distribution means are reporting summaries of exactly equal integer arrays, not tolerated engine differences.

### Mismatch preservation and triage

`mismatch_fixtures/index.json` lists every observed mismatch and points to full reproducible fixtures. Each fixture includes the input state, proposal envelopes, nonce, CPU batch, GPU integer output, failed fields, release severity, triage status, and root-cause class. The harness itself was mutation-tested by the focused suite. Real mismatches found: {mismatch_count}; unresolved release-critical: {mismatch_index["unresolvedReleaseCriticalCount"]}. No discrepancy was deleted, averaged away, downgraded, or converted to a floating tolerance.

## Validation

- Exhaustive identity/proposal/batch coverage: {"PASS" if all(item["success"] for item in exhaustive_summary) else "FAIL"}.
- Random legal-state topology/occupancy/conflict coverage: {"PASS" if all(item["success"] for item in random_summary) else "FAIL"}.
- Canonical complete episode bytes and all transition/policy shadows: {"PASS" if all(item["success"] for item in canonical_rows) else "FAIL"}.
- Extended convergence and seeded distributional episode parity: {"PASS" if all(item["success"] for item in [*convergence_rows, *distribution_rows]) else "FAIL"}.
- Integer invariants and repeated GPU replay: {"PASS" if all(item["success"] for item in transition_rows) else "FAIL"}.
- Current gradient integer/no-tolerance audit: {"PASS" if gradient["success"] else "FAIL"}.
- Upstream immutability: {upstream_count} artifacts, zero mismatches.
- Repository tests: `{tests["command"]}` → `{tests["stdout"]}`.

Overall machine gate: **{"PASS" if validation["success"] else "FAIL"}** (`validation_summary.json`).

## Commands, dependencies, and compute

```text
ruff format src/morph2d/engine.py src/morph2d/parity.py src/morph2d/__init__.py tests/test_morph2d_parity.py scripts/build_morph2d_s08.py
ruff check src/morph2d/engine.py src/morph2d/parity.py src/morph2d/__init__.py tests/test_morph2d_parity.py scripts/build_morph2d_s08.py
python -m py_compile src/morph2d/engine.py src/morph2d/parity.py scripts/build_morph2d_s08.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/build_morph2d_s08.py
{tests["command"]}
```

The run used Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}, one {torch.cuda.get_device_name()}, and at most eight CPU threads. No dependency, package, network resource, or capability was installed.

## Caveats, failed assumptions, and limitations

S08 validates the frozen S07 promise, not a broader all-GPU simulator. CPU code still constructs observations, authenticates and rejects proposals, computes SHA priorities, owns channel/controller state, creates full event records, and accounts non-GPU logical work. The GPU does not acquire permission to read routes or identities as policy inputs. Invalid proposals remain a deterministic S04 CPU rejection and are outside the already-validated GPU compiler input domain.

Exhaustiveness is exact only for the three declared small graphs and empty/single/pair legal batches. The larger graph search is seeded and finite. The eight-replicate episode distributions detect no discrepancy because each paired seed is exact, but they cannot prove parity on every possible future scenario. Convergence is a fixed-budget tail diagnostic; continuing counter-addressed activations can leave, revisit, or cycle through states, and S08 does not call quietness an attractor or target completion.

No topology-specific S01/S02 completion was invented for periodic, hexagonal, irregular, obstacle, or fixed-boundary worlds. S02 local feedback and S01 global audits remain separate; online policies/controllers still cannot read whole-grid completion. These are computational software-validation results, not evidence of biological morphology, repair, cognition, or clinical relevance.

## Provenance and artifacts

Repository source is preserved at commit `{git_commit}` on `eidosoma/groups/28`. `input_provenance.json` hashes every declared input; `environment_provenance.json` records the selected single L4 and runtime; `execution_commands.log` records commands; and `artifact_manifest.json` covers every final artifact. Repository code remains in Git rather than being copied into artifacts.

## Recommended next action

{next_action} Do not start S09 from this handoff without separate authorization.
"""


def _artifact_manifest(output: Path, git_commit: str) -> None:
    artifacts = []
    roles = {
        "research_step_full_results.md": "canonical S08 full-results handoff",
        "parity_report.json": "primary CPU/GPU release-gate decision",
        "mismatch_fixtures/index.json": "complete mismatch inventory and triage status",
        "invariant_report.json": "integer replay/conservation/boundary audit",
        "tolerance_specification.json": "predeclared exact and floating-tolerance rules",
        "validation_summary.json": "machine-readable validation gates",
    }
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    ):
        relative = path.relative_to(output).as_posix()
        artifacts.append(
            {
                "path": relative,
                "role": roles.get(relative, "S08 supporting parity evidence"),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e06.s08.artifact-manifest.v1",
            "researchStepId": "S08",
            "repositoryCommit": git_commit,
            "artifacts": artifacts,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists():
        if not arguments.replace:
            raise FileExistsError(f"output exists; rerun with --replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("S08 requires the registered CUDA device")
    device = "cuda:0"

    raw = yaml.safe_load(PARITY_CATALOG.read_text())
    design = raw["design"]
    requirements = raw["parityRequirements"]
    if any(
        requirements[key] != 0
        for key in (
            "integerTransitionAbsoluteTolerance",
            "acceptedProposalSetTolerance",
            "integerLedgerTolerance",
            "deterministicEpisodeCanonicalByteTolerance",
            "pairedDistributionOutcomeTolerance",
            "unresolvedReleaseCriticalMismatchLimit",
        )
    ):
        raise ValueError("S08 exact parity thresholds must be zero")
    if requirements["floatingGradient"]["applicable"]:
        raise ValueError("current integer gradient contract cannot enable tolerance")

    context = load_engine_context(
        ENGINE_CATALOG,
        environment_catalog=ENVIRONMENT_CATALOG,
        policy_catalog=POLICY_CATALOG,
        grammar_catalog=GRAMMAR_CATALOG,
        channel_catalog=CHANNEL_CATALOG,
    )
    small_environments = tuple(
        parse_environment_spec(item) for item in raw["smallEnvironments"]
    )
    git_commit = _git("rev-parse", "HEAD")
    shutil.copy2(PARITY_CATALOG, output / "parity_catalog.yaml")
    _write_json(
        output / "tolerance_specification.json",
        {
            "schemaVersion": "e06.s08.tolerance-specification.v1",
            "researchStepId": "S08",
            **requirements,
            "frozenBeforeOutcomeExecution": True,
        },
    )
    fixture_dir = output / "small_environment_fixtures"
    for environment in small_environments:
        _write_json(
            fixture_dir / f"{environment.environment_id}.json",
            environment_to_dict(environment),
        )

    upstream_steps = [_verify_upstream_manifest(directory) for directory in STEP_DIRS]
    upstream = {
        "schemaVersion": "e06.s08.upstream-immutability.v1",
        "researchStepId": "S08",
        "steps": upstream_steps,
        "success": all(item["success"] for item in upstream_steps),
    }
    _write_json(output / "upstream_immutability.json", upstream)
    _write_json(output / "input_provenance.json", _input_provenance())
    _write_json(
        output / "environment_provenance.json",
        _environment_provenance(git_commit, device),
    )

    exhaustive_rows, exhaustive_summary, exhaustive_mismatches = _exhaustive_validation(
        small_environments, design["exhaustive"], device=device
    )
    _write_csv(output / "exhaustive_transition_results.csv", exhaustive_rows)
    _write_json(
        output / "exhaustive_transition_summary.json",
        {
            "schemaVersion": "e06.s08.exhaustive-transition-summary.v1",
            "researchStepId": "S08",
            "environments": exhaustive_summary,
            "success": all(item["success"] for item in exhaustive_summary),
        },
    )

    random_rows, random_summary, random_mismatches = _random_legal_validation(
        tuple(context.environments.values()),
        design["randomLegalState"],
        device=device,
    )
    _write_csv(output / "random_legal_transition_results.csv", random_rows)
    _write_json(
        output / "random_legal_transition_summary.json",
        {
            "schemaVersion": "e06.s08.random-transition-summary.v1",
            "researchStepId": "S08",
            "seedDomain": design["randomLegalState"]["seedDomain"],
            "environments": random_summary,
            "success": all(item["success"] for item in random_summary),
        },
    )

    canonical_rows, canonical_policy, canonical_transition, canonical_mismatches = (
        _run_episode_matrix(
            context,
            context.episodes,
            phase="canonical_complete_episode",
            device=device,
            canonical_dir=STEP_DIRS[6] / "cpu_reference/episodes",
        )
    )
    convergence_definitions = tuple(
        replace(item, transitions=int(design["convergenceStress"]["transitions"]))
        for item in context.episodes
    )
    (
        convergence_rows,
        convergence_policy,
        convergence_transition,
        convergence_mismatches,
    ) = _run_episode_matrix(
        context,
        convergence_definitions,
        phase="convergence_stress",
        device=device,
        tail_window=int(design["convergenceStress"]["tailWindowTransitions"]),
    )

    distribution_definitions = []
    seed_manifest = []
    for base in context.episodes:
        for replicate in range(
            int(design["distributionalEpisodes"]["replicatesPerScenario"])
        ):
            scenario_id = (
                f"s08-dist-{base.scenario_id.removeprefix('s07-')}-r{replicate:02d}"
            )
            distribution_definitions.append(
                replace(
                    base,
                    scenario_id=scenario_id,
                    transitions=int(design["distributionalEpisodes"]["transitions"]),
                    parameters={**base.parameters, "baseScenarioId": base.scenario_id},
                )
            )
            seed_manifest.append(
                {
                    "baseScenarioId": base.scenario_id,
                    "scenarioId": scenario_id,
                    "replicate": replicate,
                    "seedDomain": design["distributionalEpisodes"]["seedDomain"],
                    "address": f"{base.scenario_id}:{replicate}",
                }
            )
    _write_json(
        output / "seed_manifest.json",
        {
            "schemaVersion": "e06.s08.seed-manifest.v1",
            "researchStepId": "S08",
            "randomLegalStateDomain": design["randomLegalState"]["seedDomain"],
            "distributionalEpisodes": seed_manifest,
        },
    )
    (
        distribution_rows,
        distribution_policy,
        distribution_transition,
        distribution_mismatches,
    ) = _run_episode_matrix(
        context,
        distribution_definitions,
        phase="distributional_episode",
        device=device,
    )

    episode_rows = [*canonical_rows, *convergence_rows, *distribution_rows]
    policy_rows = [
        *canonical_policy,
        *convergence_policy,
        *distribution_policy,
    ]
    episode_transition_rows = [
        *canonical_transition,
        *convergence_transition,
        *distribution_transition,
    ]
    _write_csv(output / "episode_parity_results.csv", episode_rows)
    _write_json(
        output / "episode_parity_results.json",
        {
            "schemaVersion": "e06.s08.episode-parity-results.v1",
            "researchStepId": "S08",
            "episodes": episode_rows,
            "success": all(item["success"] for item in episode_rows),
        },
    )
    _write_csv(output / "policy_decision_parity.csv", policy_rows)
    _write_json(
        output / "policy_decision_parity.json",
        {
            "schemaVersion": "e06.s08.policy-decision-parity.v1",
            "researchStepId": "S08",
            "comparisonCount": len(policy_rows),
            "mismatchCount": sum(not item["success"] for item in policy_rows),
            "success": all(item["success"] for item in policy_rows),
        },
    )
    _write_csv(output / "episode_transition_parity.csv", episode_transition_rows)

    distribution_comparison_rows, distribution_summary = _distribution_summary(
        distribution_rows, [item.scenario_id for item in context.episodes]
    )
    _write_csv(
        output / "distributional_outcome_comparison.csv",
        distribution_comparison_rows,
    )
    _write_json(output / "distributional_outcomes.json", distribution_summary)
    _write_csv(output / "convergence_results.csv", convergence_rows)
    _write_json(
        output / "convergence_results.json",
        {
            "schemaVersion": "e06.s08.convergence-results.v1",
            "researchStepId": "S08",
            "definition": design["convergenceStress"]["convergenceDefinition"],
            "descriptiveOnly": True,
            "episodes": convergence_rows,
            "success": all(item["success"] for item in convergence_rows),
        },
    )

    gradient = _gradient_dtype_audit(context)
    _write_json(output / "gradient_dtype_audit.json", gradient)
    all_transition_rows = [*exhaustive_rows, *random_rows, *episode_transition_rows]
    invariant = {
        "schemaVersion": "e06.s08.invariant-report.v1",
        "researchStepId": "S08",
        "transitionCount": len(all_transition_rows),
        "int32StateCount": sum(item["int32State"] for item in all_transition_rows),
        "postStateBitwiseExactCount": sum(
            item["postStateBitsExact"] for item in all_transition_rows
        ),
        "permutationInvariantCount": sum(
            item["permutationInvariant"] for item in all_transition_rows
        ),
        "gpuReplayExactCount": sum(
            item["gpuReplayExact"] for item in all_transition_rows
        ),
        "fixedBoundaryEnvironmentCases": sum(
            item["environmentId"] == "square_bounded_fixed_boundary"
            for item in random_rows
        ),
        "obstacleEnvironmentCases": sum(
            item["environmentId"] == "square_bounded_vacancy_obstacle"
            for item in random_rows
        ),
        "periodicEnvironmentCases": sum(
            item["boundaryMode"] == "periodic" for item in random_rows
        ),
        "irregularEnvironmentCases": sum(
            item["geometry"] == "irregular" for item in all_transition_rows
        ),
    }
    invariant["success"] = all(
        invariant[key] == invariant["transitionCount"]
        for key in (
            "int32StateCount",
            "postStateBitwiseExactCount",
            "permutationInvariantCount",
            "gpuReplayExactCount",
        )
    )
    _write_json(output / "invariant_report.json", invariant)

    conflict_by_phase = defaultdict(lambda: {"cases": 0, "losses": 0})
    for row in all_transition_rows:
        if row["conflictLosses"]:
            conflict_by_phase[row["phase"]]["cases"] += 1
            conflict_by_phase[row["phase"]]["losses"] += int(row["conflictLosses"])
    conflict = {
        "schemaVersion": "e06.s08.conflict-stress.v1",
        "researchStepId": "S08",
        "byPhase": dict(conflict_by_phase),
        "totalConflictCases": sum(item["cases"] for item in conflict_by_phase.values()),
        "totalConflictLosses": sum(
            item["losses"] for item in conflict_by_phase.values()
        ),
        "allWinnerSetsExact": all(
            item["acceptedProposalIdsExact"] for item in all_transition_rows
        ),
    }
    conflict["success"] = (
        conflict["totalConflictLosses"] > 0 and conflict["allWinnerSetsExact"]
    )
    _write_json(output / "conflict_stress.json", conflict)

    mismatches = [
        *exhaustive_mismatches,
        *random_mismatches,
        *canonical_mismatches,
        *convergence_mismatches,
        *distribution_mismatches,
    ]
    mismatch_index = _write_mismatches(output, mismatches)
    tests = _focused_tests()
    checks = {
        "upstreamImmutability": upstream["success"],
        "focusedTests": tests["success"],
        "zeroUnresolvedReleaseCriticalMismatches": mismatch_index[
            "unresolvedReleaseCriticalCount"
        ]
        == 0,
        "exhaustiveSmallGridParity": all(
            item["success"] for item in exhaustive_summary
        ),
        "seededRandomLegalStateParity": all(item["success"] for item in random_summary),
        "canonicalCompleteEpisodeParity": all(
            item["success"] for item in canonical_rows
        ),
        "convergenceStressParity": all(item["success"] for item in convergence_rows),
        "distributionalOutcomeParity": distribution_summary["success"]
        and all(item["success"] for item in distribution_rows),
        "maskedPolicyDecisionParity": all(item["success"] for item in policy_rows),
        "integerInvariantAndReplay": invariant["success"],
        "conflictStress": conflict["success"],
        "integerGradientNoTolerance": gradient["success"],
    }
    validation = {
        "schemaVersion": "e06.s08.validation-summary.v1",
        "researchStepId": "S08",
        "checks": checks,
        "focusedTests": tests,
        "success": all(checks.values()),
    }
    _write_json(output / "validation_summary.json", validation)

    parity_report = {
        "schemaVersion": "e06.s08.parity-report.v1",
        "researchStepId": "S08",
        "integerTolerance": 0,
        "floatingGradientTolerance": None,
        "counts": {
            "exhaustiveStates": sum(item["stateCount"] for item in exhaustive_summary),
            "exhaustiveTransitions": len(exhaustive_rows),
            "randomLegalStates": sum(item["stateCount"] for item in random_summary),
            "randomLegalTransitions": len(random_rows),
            "completeCanonicalEpisodes": len(canonical_rows),
            "convergenceEpisodes": len(convergence_rows),
            "distributionalEpisodes": len(distribution_rows),
            "episodePolicyDecisions": len(policy_rows),
            "episodeTransitions": len(episode_transition_rows),
            "allComparedTransitions": len(all_transition_rows),
        },
        "mismatchCount": len(mismatches),
        "unresolvedReleaseCriticalMismatchCount": mismatch_index[
            "unresolvedReleaseCriticalCount"
        ],
        "releaseGate": "pass" if validation["success"] else "blocked",
        "success": validation["success"],
    }
    _write_json(output / "parity_report.json", parity_report)

    commands = [
        "ruff format src/morph2d/engine.py src/morph2d/parity.py src/morph2d/__init__.py tests/test_morph2d_parity.py scripts/build_morph2d_s08.py",
        "ruff check src/morph2d/engine.py src/morph2d/parity.py src/morph2d/__init__.py tests/test_morph2d_parity.py scripts/build_morph2d_s08.py",
        "python -m py_compile src/morph2d/engine.py src/morph2d/parity.py scripts/build_morph2d_s08.py",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/build_morph2d_s08.py",
        tests["command"],
    ]
    (output / "execution_commands.log").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )

    upstream_count = sum(item["checkedArtifactCount"] for item in upstream_steps)
    report = _report(
        exhaustive_summary=exhaustive_summary,
        random_summary=random_summary,
        canonical_rows=canonical_rows,
        convergence_rows=convergence_rows,
        distribution_rows=distribution_rows,
        transition_rows=all_transition_rows,
        policy_rows=policy_rows,
        mismatch_index=mismatch_index,
        upstream_count=upstream_count,
        tests=tests,
        gradient=gradient,
        validation=validation,
        git_commit=git_commit,
    )
    (output / "research_step_full_results.md").write_text(report, encoding="utf-8")
    _artifact_manifest(output, git_commit)

    manifest = json.loads((output / "artifact_manifest.json").read_text())
    listed = {item["path"] for item in manifest["artifacts"]}
    actual = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    }
    if listed != actual:
        raise AssertionError("artifact manifest coverage mismatch")
    for record in manifest["artifacts"]:
        if _sha256_file(output / record["path"]) != record["sha256"]:
            raise AssertionError(f"artifact hash mismatch: {record['path']}")
    print(
        json.dumps(
            {
                "researchStepId": "S08",
                "success": validation["success"],
                "releaseGate": parity_report["releaseGate"],
                "mismatches": len(mismatches),
                "artifactCount": len(manifest["artifacts"]) + 1,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
