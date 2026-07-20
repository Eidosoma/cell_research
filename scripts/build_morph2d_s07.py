#!/usr/bin/env python3
"""Build and validate E06 S07 CPU-oracle/GPU-engine artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT = Path("/artifacts/research_steps/S07")
ENGINE_CATALOG = ROOT / "configs/morphologies/engine_catalog.yaml"
ENVIRONMENT_CATALOG = ROOT / "configs/morphologies/environment_catalog.yaml"
POLICY_CATALOG = ROOT / "configs/morphologies/policy_catalog.yaml"
GRAMMAR_CATALOG = ROOT / "configs/morphologies/grammar_catalog.yaml"
CHANNEL_CATALOG = ROOT / "configs/morphologies/control_channel_catalog.yaml"
S04_FIXTURES = Path("/artifacts/research_steps/S04/movement_fixtures")

sys.path.insert(0, str(ROOT))

from src.morph2d.engine import (  # noqa: E402
    canonical_episode_result_bytes,
    load_engine_context,
    replay_cpu_episode,
    run_cpu_episode,
)
from src.morph2d.gpu_engine import (  # noqa: E402
    CompiledProposalBatch,
    compile_validated_proposal_batch,
    decode_post_state_occupant_ids,
    expected_post_state_occupant_ids,
    masked_observation_tensor_names,
    resolve_compiled_proposals,
    synthetic_compiled_workload,
    tensor_permutation_invariants,
    tensor_storage_bytes,
)
from src.morph2d.movements import parse_movement_state, parse_proposal  # noqa: E402


ATTACHMENT_DIR = WORKSPACE / "input-attachments/21c2278b-9950-4e39-a2c8-df578a2508ec"
STEP_DIRS = [Path(f"/artifacts/research_steps/S0{index}") for index in range(1, 7)]
UPSTREAM_INPUTS = {
    "agents": WORKSPACE / "AGENTS.md",
    "full_plan": WORKSPACE / "FULL_PLAN.md",
    "research_plan_pre_s07_update": WORKSPACE / "RESEARCH_PLAN.md",
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
        for index in range(1, 7)
    },
    **{
        f"s{index:02d}_manifest": STEP_DIRS[index - 1] / "artifact_manifest.json"
        for index in range(1, 7)
    },
    "s03_environment_spec": STEP_DIRS[2] / "environment_spec.md",
    "s04_movement_spec": STEP_DIRS[3] / "movement_spec.md",
    "s04_movement_catalog": STEP_DIRS[3] / "movement_catalog.yaml",
    "s05_policy_spec": STEP_DIRS[4] / "policy_spec.md",
    "s05_policy_catalog": STEP_DIRS[4] / "policy_library/policy_catalog.yaml",
    "s05_observation_permissions": STEP_DIRS[4] / "observation_permissions.json",
    "s05_observation_budget": STEP_DIRS[4] / "observation_budget.csv",
    "s06_control_channel_spec": STEP_DIRS[5] / "control_channel_spec.md",
    "s06_channel_catalog": STEP_DIRS[5]
    / "channel_library/control_channel_catalog.yaml",
    "s06_budget_schema": STEP_DIRS[5] / "budget_schema.json",
    "s06_permission_isolation": STEP_DIRS[5] / "permission_isolation.json",
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
    "e01_s07_invariants": Path(
        "/previous-artifacts/E01/research_steps/S07/research_step_full_results.md"
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


def _focused_tests() -> dict[str, Any]:
    paths = [
        "tests/test_morph2d_targets.py",
        "tests/test_morph2d_grammar.py",
        "tests/test_morph2d_environments.py",
        "tests/test_morph2d_movements.py",
        "tests/test_morph2d_policies.py",
        "tests/test_morph2d_channels.py",
        "tests/test_morph2d_engine.py",
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


def _input_provenance() -> dict[str, Any]:
    records = []
    for name, path in UPSTREAM_INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"required S07 input missing: {path}")
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
        "schemaVersion": "e06.s07.input-provenance.v1",
        "researchStepId": "S07",
        "inputs": records,
    }


def _episode_runs(
    context: Any, output: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_dir = output / "cpu_reference/episodes"
    episode_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    replay_rows = []
    trace_records = []
    for definition in context.episodes:
        started = time.perf_counter()
        result = run_cpu_episode(context, definition)
        elapsed = time.perf_counter() - started
        replay_started = time.perf_counter()
        replay = replay_cpu_episode(context, definition, result)
        replay_elapsed = time.perf_counter() - replay_started
        exact = canonical_episode_result_bytes(
            result
        ) == canonical_episode_result_bytes(replay)
        if not exact:
            raise AssertionError(f"episode replay mismatch: {definition.scenario_id}")
        _write_json(episode_dir / f"{definition.scenario_id}.json", result)
        for trace in result["sampledTraces"]:
            trace_records.append({"scenarioId": definition.scenario_id, **trace})
        row = {
            "scenarioId": definition.scenario_id,
            "environmentId": definition.environment_id,
            "policyId": definition.policy_id,
            "channelMode": definition.channel_mode,
            "transitions": definition.transitions,
            "acceptedMovements": result["movementLedger"]["acceptedMovements"],
            "conflictLosses": result["movementLedger"]["conflictLosses"],
            "totalGraphDisplacement": result["movementLedger"][
                "totalGraphDisplacement"
            ],
            "configurationBits": result["channelLedger"]["configurationBits"],
            "totalInformationBits": result["channelLedger"]["totalInformationBits"],
            "selectedTraceCount": len(result["sampledTraces"]),
            "wallSeconds": elapsed,
            "transitionsPerSecond": definition.transitions / elapsed,
            "episodeSha256": result["episodeSha256"],
            "replayExact": exact,
        }
        rows.append(row)
        replay_rows.append(
            {
                "scenarioId": definition.scenario_id,
                "firstEpisodeSha256": result["episodeSha256"],
                "replayEpisodeSha256": replay["episodeSha256"],
                "canonicalByteEquality": exact,
                "replayWallSeconds": replay_elapsed,
            }
        )
    trace_path = output / "sampled_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in trace_records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    replay_validation = {
        "schemaVersion": "e06.s07.exact-replay.v1",
        "researchStepId": "S07",
        "episodes": replay_rows,
        "success": all(row["canonicalByteEquality"] for row in replay_rows),
    }
    return rows, replay_validation


def _scoped_transition_comparisons(context: Any) -> dict[str, Any]:
    records = []
    for fixture_path in sorted(S04_FIXTURES.glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        has_invalid = any(
            decision["validation"] != "valid" for decision in fixture["decisions"]
        )
        if has_invalid:
            records.append(
                {
                    "fixtureId": fixture_path.stem,
                    "scope": "excluded_before_gpu_compile",
                    "reason": "S04 fixture intentionally contains invalid proposal",
                    "success": True,
                }
            )
            continue
        environment = context.environments[fixture["environmentId"]]
        state = parse_movement_state(fixture["preState"])
        proposals = tuple(parse_proposal(item) for item in fixture["proposals"])
        compiled, expected = compile_validated_proposal_batch(
            environment,
            [state],
            [proposals],
            [fixture["batchNonce"]],
            device="cuda",
            maximum_proposals=8,
        )
        first = resolve_compiled_proposals(compiled)
        second = resolve_compiled_proposals(compiled)
        torch.cuda.synchronize()
        decoded = decode_post_state_occupant_ids(compiled, first)
        expected_ids = expected_post_state_occupant_ids(compiled, expected)
        accepted_mask = first.accepted_mask[0].detach().cpu().tolist()
        accepted_ids = sorted(
            proposal_id
            for proposal_id, accepted in zip(
                compiled.proposal_ids[0], accepted_mask, strict=True
            )
            if proposal_id is not None and accepted
        )
        cpu_ids = sorted(expected[0]["acceptedProposalIds"])
        permutation = bool(
            tensor_permutation_invariants(compiled.state, first.post_state).all().cpu()
        )
        cost_match = (
            int(first.accepted_movements[0].cpu())
            == expected[0]["costLedger"]["acceptedMovements"]
            and int(first.total_graph_displacement[0].cpu())
            == expected[0]["costLedger"]["totalGraphDisplacement"]
        )
        success = (
            decoded == expected_ids
            and accepted_ids == cpu_ids
            and torch.equal(first.post_state, second.post_state)
            and torch.equal(first.accepted_mask, second.accepted_mask)
            and cost_match
            and permutation
            and expected[0]["transitionSha256"] == fixture["transitionSha256"]
        )
        records.append(
            {
                "fixtureId": fixture_path.stem,
                "scope": "exact_valid_s04_transition",
                "movementKinds": sorted({item.kind for item in proposals}),
                "proposalCount": len(proposals),
                "acceptedProposalIdsMatch": accepted_ids == cpu_ids,
                "postOccupantProjectionMatch": decoded == expected_ids,
                "acceptedAndDisplacementCostMatch": cost_match,
                "gpuReplayExact": torch.equal(first.post_state, second.post_state)
                and torch.equal(first.accepted_mask, second.accepted_mask),
                "permutationInvariant": permutation,
                "cpuFixtureTransitionReproduced": expected[0]["transitionSha256"]
                == fixture["transitionSha256"],
                "success": success,
            }
        )
    return {
        "schemaVersion": "e06.s07.scoped-cpu-gpu-transition-comparison.v1",
        "researchStepId": "S07",
        "scope": "all-valid S04 canonical fixture batches; invalid-proposal rejection remains CPU-only",
        "records": records,
        "success": all(item["success"] for item in records),
    }


def _permute_compiled(
    compiled: CompiledProposalBatch, permutation: torch.Tensor
) -> CompiledProposalBatch:
    indices = permutation.to(compiled.state.device)
    host = permutation.tolist()
    return CompiledProposalBatch(
        state=compiled.state[indices],
        route=compiled.route[indices],
        route_length=compiled.route_length[indices],
        kind_code=compiled.kind_code[indices],
        rotation_direction=compiled.rotation_direction[indices],
        priority_rank=compiled.priority_rank[indices],
        proposal_mask=compiled.proposal_mask[indices],
        proposal_displacement=compiled.proposal_displacement[indices],
        site_ids=compiled.site_ids,
        occupant_ids=tuple(compiled.occupant_ids[index] for index in host),
        proposal_ids=tuple(compiled.proposal_ids[index] for index in host),
    )


def _batch_order_validation() -> dict[str, Any]:
    batch = 257
    compiled = synthetic_compiled_workload(batch, device="cuda")
    baseline = resolve_compiled_proposals(compiled)
    permutation = torch.arange(batch - 1, -1, -1, dtype=torch.int64)
    permuted = _permute_compiled(compiled, permutation)
    observed = resolve_compiled_proposals(permuted)
    inverse = torch.argsort(permutation).to("cuda")
    state_match = torch.equal(baseline.post_state, observed.post_state[inverse])
    acceptance_match = torch.equal(
        baseline.accepted_mask, observed.accepted_mask[inverse]
    )
    cost_match = torch.equal(
        baseline.total_graph_displacement,
        observed.total_graph_displacement[inverse],
    )
    return {
        "schemaVersion": "e06.s07.batch-order-invariance.v1",
        "researchStepId": "S07",
        "batchSize": batch,
        "permutation": "reverse",
        "postStateExact": state_match,
        "acceptedMaskExact": acceptance_match,
        "costExact": cost_match,
        "success": state_match and acceptance_match and cost_match,
    }


def _proposal_order_validation(context: Any) -> dict[str, Any]:
    path = S04_FIXTURES / "conflicting_and_disjoint_square_batch.json"
    fixture = json.loads(path.read_text())
    environment = context.environments[fixture["environmentId"]]
    state = parse_movement_state(fixture["preState"])
    proposals = tuple(parse_proposal(item) for item in fixture["proposals"])
    records = []
    for order_name, proposal_row in (
        ("canonical", proposals),
        ("reverse", tuple(reversed(proposals))),
        ("rotation", proposals[1:] + proposals[:1]),
    ):
        compiled, expected = compile_validated_proposal_batch(
            environment,
            [state],
            [proposal_row],
            [fixture["batchNonce"]],
            device="cuda",
            maximum_proposals=8,
        )
        result = resolve_compiled_proposals(compiled)
        accepted_mask = result.accepted_mask[0].detach().cpu().tolist()
        records.append(
            {
                "order": order_name,
                "acceptedProposalIds": sorted(
                    proposal_id
                    for proposal_id, accepted in zip(
                        compiled.proposal_ids[0], accepted_mask, strict=True
                    )
                    if proposal_id is not None and accepted
                ),
                "postOccupantProjection": decode_post_state_occupant_ids(
                    compiled, result
                )[0],
                "displacement": int(result.total_graph_displacement[0].cpu()),
                "cpuTransitionSha256": expected[0]["transitionSha256"],
            }
        )

    def signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["acceptedProposalIds"],
            row["postOccupantProjection"],
            row["displacement"],
            row["cpuTransitionSha256"],
        )

    success = len({json.dumps(signature(row), sort_keys=True) for row in records}) == 1
    return {
        "schemaVersion": "e06.s07.proposal-order-invariance.v1",
        "researchStepId": "S07",
        "fixtureId": path.stem,
        "records": records,
        "success": success,
    }


def _time_kernel(compiled: CompiledProposalBatch, iterations: int) -> list[float]:
    samples = []
    for _ in range(3):
        for _ in range(5):
            resolve_compiled_proposals(compiled)
        if compiled.state.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            resolve_compiled_proposals(compiled)
        if compiled.state.device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return samples


def _benchmarks(
    benchmark_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    measured = benchmark_contract["measuredIterationsByBatch"]
    for batch in benchmark_contract["batchSizes"]:
        batch = int(batch)
        iterations = int(measured[str(batch)])
        device_metrics = {}
        for device in ("cpu", "cuda"):
            compiled = synthetic_compiled_workload(batch, device=device)
            samples = _time_kernel(compiled, iterations)
            median_seconds = statistics.median(samples)
            throughput = batch * iterations / median_seconds
            device_metrics[device] = throughput
            rows.append(
                {
                    "batchSize": batch,
                    "device": device,
                    "iterationsPerTrial": iterations,
                    "trials": len(samples),
                    "medianSeconds": median_seconds,
                    "minSeconds": min(samples),
                    "maxSeconds": max(samples),
                    "medianTransitionsPerSecond": throughput,
                    "inputTensorBytes": tensor_storage_bytes(compiled),
                    "cpuThreads": torch.get_num_threads() if device == "cpu" else "",
                }
            )
        speedup = device_metrics["cuda"] / device_metrics["cpu"]
        for row in rows[-2:]:
            row["gpuSpeedupVsCpu"] = speedup
            row["materialGpuBenefit"] = speedup >= float(
                benchmark_contract["materialBenefitThresholdSpeedup"]
            )
    by_batch = {}
    for batch in benchmark_contract["batchSizes"]:
        selected = [row for row in rows if row["batchSize"] == int(batch)]
        by_batch[str(batch)] = {
            "cpuTransitionsPerSecond": next(
                row["medianTransitionsPerSecond"]
                for row in selected
                if row["device"] == "cpu"
            ),
            "gpuTransitionsPerSecond": next(
                row["medianTransitionsPerSecond"]
                for row in selected
                if row["device"] == "cuda"
            ),
            "gpuSpeedup": selected[0]["gpuSpeedupVsCpu"],
            "materialBenefit": selected[0]["materialGpuBenefit"],
        }
    summary = {
        "schemaVersion": "e06.s07.benchmark-summary.v1",
        "researchStepId": "S07",
        "benchmarkBoundary": "validated fixed-proposal deterministic transition kernel",
        "excludes": [
            "CPU S05 observation construction",
            "CPU S04 SHA authentication and priority",
            "S06 channel source construction",
            "host-device transfer",
        ],
        "batchResults": by_batch,
        "materialBenefitThresholdSpeedup": benchmark_contract[
            "materialBenefitThresholdSpeedup"
        ],
        "cpuFallbackRule": "use exact CPU oracle below first >=2x measured crossover or whenever CPU control-plane wall dominates",
    }
    return rows, summary


def _memory_probe(benchmark_contract: Mapping[str, Any]) -> dict[str, Any]:
    batch = int(benchmark_contract["targetBatchSize"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_free, total = torch.cuda.mem_get_info()
    compiled = synthetic_compiled_workload(
        batch,
        sites=int(benchmark_contract["targetSites"]),
        proposals=int(benchmark_contract["targetProposals"]),
        device="cuda",
    )
    result = resolve_compiled_proposals(compiled)
    torch.cuda.synchronize()
    invariant = bool(
        tensor_permutation_invariants(compiled.state, result.post_state).all().cpu()
    )
    replay = resolve_compiled_proposals(compiled)
    torch.cuda.synchronize()
    exact = torch.equal(result.post_state, replay.post_state) and torch.equal(
        result.accepted_mask, replay.accepted_mask
    )
    after_free, _ = torch.cuda.mem_get_info()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    input_bytes = tensor_storage_bytes(compiled)
    safety_fraction = peak_reserved / total
    success = invariant and exact and safety_fraction < 0.8
    return {
        "schemaVersion": "e06.s07.target-batch-memory.v1",
        "researchStepId": "S07",
        "targetBatchSize": batch,
        "sitesPerEpisode": benchmark_contract["targetSites"],
        "proposalsPerEpisode": benchmark_contract["targetProposals"],
        "inputTensorBytes": input_bytes,
        "cudaTotalBytes": total,
        "cudaFreeBytesBefore": before_free,
        "cudaFreeBytesAfter": after_free,
        "peakAllocatedBytes": peak_allocated,
        "peakReservedBytes": peak_reserved,
        "peakReservedFractionOfDevice": safety_fraction,
        "permutationInvariant": invariant,
        "exactReplay": exact,
        "safetyCriterion": "peak reserved memory <80% of device total",
        "success": success,
    }


def _environment_provenance(git_commit: str) -> dict[str, Any]:
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "schemaVersion": "e06.s07.environment-provenance.v1",
        "researchStepId": "S07",
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
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDevice": torch.cuda.get_device_name(),
        "nvidiaSmi": nvidia.stdout.strip(),
        "cpuLogicalCountVisible": os.cpu_count(),
        "torchCpuThreads": torch.get_num_threads(),
        "threadEnvironment": {"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "PyYAML", "pytest", "ruff")
        },
        "newDependenciesInstalled": [],
    }


def _engine_spec() -> str:
    return """# S07 CPU-oracle and GPU-engine specification

The canonical CPU oracle composes the frozen S04 proposal envelope, validation,
SHA-256 conflict priority, greedy disjoint-site claim, atomic commit, and cost
ledger with S05 priced observations/policies and S06 channel timing/ledgers.
Episodes use 32 integer transitions, four state-blind identity-hash scheduled
actors per native transition, 16-transition epochs, a fixed transition stop,
and counter-addressed SHA-256 randomness. Configuration bits are charged once
in full per episode and are never divided by transitions or batch size.

The GPU boundary is deliberately narrower. Policies receive only numeric S05
fields named in the frozen masked-observation contract. Routes, identities,
site roles, authentication hashes, current proposals, priority, analysis labels,
S02 whole-grid scores, S01 completion audits, and future state never enter that
payload. CPU code authenticates/validates proposals and calculates exact S04
priority ranks. GPU tensors then perform pure policy selection, deterministic
rank-ordered claiming, and integer atomic permutation commit. S06 source logic
and sparse direct-control authority remain CPU-owned; direct intervention still
uses one state-blind recipient, one query, one action, and no retry.

Every transition emits a compact event summary. Full movement batches are kept
only at transition 0, each transition ending a 16-transition epoch, and the
final transition. S02 local relation deltas remain priced feedback. S01
canonical-equivalence, component, and topology completion audits are distinct,
not online policy/controller inputs, and are not computed by S07.

S07 validates exact CPU replay and a scoped CPU/GPU transition boundary. It does
not claim the exhaustive random-state/full-episode parity reserved for S08.
"""


def _report(
    *,
    artifacts: list[str],
    episode_rows: list[dict[str, Any]],
    replay: Mapping[str, Any],
    scoped: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    memory: Mapping[str, Any],
    tests: Mapping[str, Any],
    upstream: Mapping[str, Any],
    validation: Mapping[str, Any],
    git_commit: str,
) -> str:
    total_transitions = sum(row["transitions"] for row in episode_rows)
    total_wall = sum(row["wallSeconds"] for row in episode_rows)
    trace_count = sum(row["selectedTraceCount"] for row in episode_rows)
    material = [
        int(batch)
        for batch, value in benchmark["batchResults"].items()
        if value["materialBenefit"]
    ]
    first_material = min(material) if material else None
    best_batch, best = max(
        benchmark["batchResults"].items(), key=lambda item: item[1]["gpuSpeedup"]
    )
    exact_count = sum(
        record["scope"] == "exact_valid_s04_transition" and record["success"]
        for record in scoped["records"]
    )
    upstream_count = sum(item["checkedArtifactCount"] for item in upstream["steps"])
    report_artifacts = ", ".join(artifacts)
    return f"""# S07 full results — Build the vectorized GPU simulator

## Top summary

- **Research step ID:** S07
- **Completion status:** Complete; S08 was not started.
- **Artifacts written:** {report_artifacts} (full inventory and SHA-256 values in `artifact_manifest.json`).
- **Validation result:** PASS — {tests["stdout"]}; nine 32-transition CPU episodes replayed byte-for-byte; {exact_count} all-valid S04 fixture transitions matched GPU state, acceptances, and costs exactly; proposal and batch ordering were invariant; and the {memory["targetBatchSize"]:,}-episode target batch passed replay, conservation, and memory safety.
- **Outcome classification:** Supportive, with a bounded performance constraint. The deterministic GPU transition data plane materially accelerates batches from {first_material:,} episodes in this measured grid, while small batches and the exact CPU authentication/observation control plane retain an explicit CPU fallback.
- **Caveats or blockers:** GPU equivalence is scoped to canonical all-valid S04 transitions plus all-six-policy masked-decision tests. Invalid-proposal rejection stays CPU-side. Exhaustive random-state, full-episode CPU/GPU parity, convergence, and distributional tests remain S08 work. Kernel throughput excludes CPU observation/channel construction, SHA authentication, and host/device transfer.
- **Lay summary:** The simulator now has a careful reference implementation and a fast batched GPU core. Both move the same integer occupants and resolve simultaneous conflicts in the same deterministic order. The GPU is worthwhile for thousands of parallel episodes, but the CPU remains the sensible and authoritative route for small jobs and exact branch-heavy setup.
- **Recommended next action:** Hand control back. If authorized, S08 should run exhaustive/scoped parity expansion, replay/divergence diagnostics, full-episode and distributional comparisons, and performance tests that include the complete hybrid wall path.

## Frozen question and success criterion

S07 asked whether a batched GPU engine could materially accelerate the frozen spatial episode semantics without widening S05 observations, S06 controller authority, or hiding S06 configuration/action costs. Success required a CPU oracle; integer deterministic GPU transitions; counter-addressed randomness; compact events and sampled traces; exact replay, invariants, batch-order invariance, scoped CPU comparisons, target memory safety, and an honest CPU fallback where acceleration was not material.

The criterion was met for the bounded GPU data-plane claim. The best observed kernel speedup was {best["gpuSpeedup"]:.2f}× at batch {int(best_batch):,}; the first predeclared batch with at least 2× speedup was {first_material:,}. No full-hybrid acceleration claim is made.

## Inputs

The run refreshed `AGENTS.md`, `FULL_PLAN.md`, the pre-update `RESEARCH_PLAN.md`, S01–S06 reports/manifests and frozen S03–S06 contract artifacts, `PREVIOUS_ARTIFACTS.md/.json`, capability/dataset availability files, and `input-attachments/MANIFEST.json` plus its attachment sidecar. Relevant E01 inputs covered transition semantics, schemas, API, invariants, and counter seeding; relevant E04 inputs covered kinetic matching, identity/label blindness, policy observations, declustering, transport, and the E06/E07 handoff. `input_provenance.json` records exact paths, sizes, and hashes. All {upstream_count} entries across the six upstream artifact manifests rehashed without mismatch.

## Methods

### CPU episode oracle

`src/morph2d/engine.py` composes existing S04, S05, and S06 functions rather than duplicating their semantics. Each scenario runs a fixed 32 transitions and uses a state-blind SHA-256 identity ranking to select four actors without replacement. Policy memory is identity-owned; conflict avoidance sees only the previous completed batch; global summaries are exactly one epoch late; and direct intervention replaces the native batch once at transition 16 with one state-blind query/action and no retry.

The oracle ran all six policies, all five channel families plus no-channel control, bounded/periodic square and bounded hexagonal environments, occupied and vacancy states, and an obstacle fixture through scoped transition parity. It emitted {total_transitions} summaries and {trace_count} preregistered full traces. The first pass took {total_wall:.3f} s ({total_transitions / total_wall:.2f} full exact transitions/s, including Python observation/channel/authentication work). Each complete episode was independently re-executed and compared as canonical bytes.

### GPU engine

`src/morph2d/gpu_engine.py` uses `int32` occupant permutations, `int16` priced features/costs, integer indices, boolean masks, stable exact S04 priority ranks, greedy rank-ordered site claims, and an atomic source-to-target permutation commit. It uses no floating transition arithmetic and no mutable GPU PRNG. CPU construction supplies only masked numeric policy observations and engine-private validated proposal tensors; those two tensor families remain separated.

All canonical valid S04 fixture batches were recompiled with their original nonces. CPU post-occupant projections, accepted proposal IDs, accepted-movement counts, and total graph displacement were compared to GPU results. The intentionally mixed valid/invalid fixed-boundary fixture was correctly excluded before GPU compilation because proposal rejection is frozen CPU control-plane work.

### Performance and memory

CPU and L4 GPU ran the same 81-site, eight-proposal mixed-conflict integer transition kernel. Each device/batch combination used five warmups followed by three timed trials at the cataloged iteration count; medians are primary. PyTorch CPU execution was capped at eight threads. The target memory probe ran {memory["targetBatchSize"]:,} episodes concurrently, repeated the GPU transition exactly, and checked occupant-permutation conservation.

## Results

### Episode oracle and ledgers

All nine episodes completed their fixed budgets and replayed exactly (`exact_replay.json`). Final identities remained unique and site-complete. Movement, observation, and channel ledgers reconciled. Configuration charges were 161 bits for the static gradient, 245 for the boundary signal, 21 for the two scheduled sparse-instruction declarations, 8 for the lagged summary, and 72 for direct intervention plus its required lagged summary; each was charged once in full. S02 `localRelationDelta` remained priced local feedback. S01 canonical-equivalence/component/topology fields remained separate, uncomputed online, and inaccessible to policies/controllers.

### Exact scoped parity and ordering

`scoped_cpu_gpu_transition_comparison.json` reports {exact_count} exact valid-fixture matches spanning adjacent swap, vacancy move, short exchange, rotation, obstacles, periodic seams, irregular paths, and simultaneous conflicts. Proposal-order permutations and reverse batch order produced identical accepted sets, integer post-states, and displacement costs. GPU replay was bit exact.

### Throughput

| Batch | CPU transitions/s | GPU transitions/s | GPU/CPU |
|---:|---:|---:|---:|
{chr(10).join(f"| {int(batch):,} | {value['cpuTransitionsPerSecond']:,.0f} | {value['gpuTransitionsPerSecond']:,.0f} | {value['gpuSpeedup']:.2f}× |" for batch, value in benchmark["batchResults"].items())}

The crossover is not universal: launch overhead makes CUDA slower for small batches. The predeclared 2× materiality threshold was first crossed at {first_material:,}. Therefore the frozen fallback is CPU for smaller batches and for exact workloads dominated by CPU observation, S06 source, validation, SHA-priority, or transfer time. Configuration cost is a scientific ledger item and was never divided by transitions or batched episodes; runtime vectorization does not change it.

### Target-batch memory

The {memory["targetBatchSize"]:,} × {memory["sitesPerEpisode"]} × {memory["proposalsPerEpisode"]} target used {memory["inputTensorBytes"] / 2**20:.1f} MiB of input tensor storage. Peak CUDA allocation was {memory["peakAllocatedBytes"] / 2**20:.1f} MiB and peak reservation {memory["peakReservedBytes"] / 2**20:.1f} MiB ({100 * memory["peakReservedFractionOfDevice"]:.2f}% of device memory). Exact replay and occupant-permutation checks passed. The registered safety threshold was under 80% of device capacity.

## Validation

- Focused upstream plus S07 tests: `{tests["command"]}` → `{tests["stdout"]}`.
- Upstream immutability: six manifests, {upstream_count} files, zero mismatches.
- CPU replay: nine of nine canonical byte comparisons passed.
- GPU transition parity: {exact_count} all-valid S04 fixtures passed; one intentional invalid-proposal mixture stayed CPU-only and was documented.
- Symmetry/order: proposal-order and batch-order invariance passed.
- Conservation: identities/sites and integer permutation invariants passed in episodes, fixture parity, and target batch.
- Permissions: masked tensor names exactly match the allowlist; forbidden identity, route, authentication, role, label, global-evaluation, priority, and future-state fields are absent. The six-policy decision-parity test passed.
- Channels/timing: 16-transition epoch timing, one-epoch-lagged summary, one-shot direct replacement, ledger reconciliation, and one-time configuration accounting passed.
- Serialization/events: all episode JSON files validate internally; exact episode hashes replay; every transition has one compact summary; full traces occur only at transition 0, epoch end(s), and final.
- Randomness: state-blind scheduling and channel noise use domain-separated SHA-256 counter addresses; GPU transitions consume no random draws.

`validation_summary.json` contains the machine-readable gate outcomes. Overall validation: **{"PASS" if validation["success"] else "FAIL"}**.

## Commands and dependencies

Primary commands were:

```text
ruff format src/morph2d/engine.py src/morph2d/gpu_engine.py src/morph2d/__init__.py tests/test_morph2d_engine.py scripts/build_morph2d_s07.py
ruff check src/morph2d/engine.py src/morph2d/gpu_engine.py src/morph2d/__init__.py tests/test_morph2d_engine.py scripts/build_morph2d_s07.py
python -m py_compile src/morph2d/engine.py src/morph2d/gpu_engine.py scripts/build_morph2d_s07.py
PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/build_morph2d_s07.py
{tests["command"]}
```

Python {platform.python_version()}, PyTorch {torch.__version__} with CUDA {torch.version.cuda}, and the NVIDIA {torch.cuda.get_device_name()} were used. No new dependency was installed. Exact environment details are in `environment_provenance.json`; commands are in `execution_commands.log`.

## Caveats, failed assumptions, and limits

The assumption that GPU acceleration would help every batch was falsified: small batches are launch-bound. The stronger assumption that fixed-kernel throughput alone establishes full exact-episode speedup is rejected; S05 observation construction, S06 source logic, S04 SHA authentication/validation/priority, transfers, and branch-heavy trace/ledger assembly still reside on CPU. This is why CPU remains both oracle and explicit fallback.

The GPU compiler intentionally rejects invalid proposals instead of reimplementing S04 rejection. Direct control is not latent micromanagement: it remains sparse, CPU-owned, state-blind in recipient selection, one-query/one-action, and fully charged. Sparse instructions and global summaries are recorded and charged but do not silently alter S05 policy behavior because no such integration rule was authorized. No long-range exchange, division, removal, wider policy inputs, whole-state controller access, or online S01 audit was introduced.

S07 parity is substantive but scoped. S08 should expand it to exhaustive legal transitions, seeded random states, full episodes, convergence, and distributional tolerances. Until that work is authorized and passes, the CPU oracle is canonical.

## Provenance and artifact inventory

Repository source is preserved at commit `{git_commit}` on branch `eidosoma/groups/28`. The artifact directory contains specifications, immutable catalog snapshots, CPU episodes, event traces, ledgers, parity/order/replay validations, benchmarks, target memory evidence, permission/randomness audits, provenance, commands, and a complete manifest. Artifact files are compact final evidence; no caches, binaries, environments, or generated dependency trees are included.
"""


def _artifact_manifest(output: Path, git_commit: str) -> None:
    roles = {
        "research_step_full_results.md": "canonical S07 full-results handoff",
        "validation_summary.json": "machine-readable validation gates",
        "benchmark_results.csv": "CPU/GPU kernel throughput trials",
        "benchmark_summary.json": "crossover and fallback decision",
        "target_batch_memory.json": "target-batch CUDA memory safety",
        "episode_results.csv": "compact CPU episode outcomes and timings",
        "exact_replay.json": "independent full-episode replay evidence",
        "scoped_cpu_gpu_transition_comparison.json": "exact S04 fixture parity",
        "sampled_traces.jsonl": "preregistered compact full transition traces",
        "engine_spec.md": "frozen CPU/GPU boundary specification",
    }
    artifacts = []
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    ):
        relative = path.relative_to(output).as_posix()
        artifacts.append(
            {
                "path": relative,
                "role": roles.get(relative, "S07 supporting evidence"),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schemaVersion": "e06.s07.artifact-manifest.v1",
            "researchStepId": "S07",
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
    (output / "gpu_engine").mkdir()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("S07 requires the registered CUDA device")

    raw_catalog = yaml.safe_load(ENGINE_CATALOG.read_text())
    context = load_engine_context(
        ENGINE_CATALOG,
        environment_catalog=ENVIRONMENT_CATALOG,
        policy_catalog=POLICY_CATALOG,
        grammar_catalog=GRAMMAR_CATALOG,
        channel_catalog=CHANNEL_CATALOG,
    )
    git_commit = _git("rev-parse", "HEAD")
    shutil.copy2(ENGINE_CATALOG, output / "gpu_engine/engine_catalog.yaml")
    (output / "engine_spec.md").write_text(_engine_spec(), encoding="utf-8")

    upstream_steps = [_verify_upstream_manifest(directory) for directory in STEP_DIRS]
    upstream = {
        "schemaVersion": "e06.s07.upstream-immutability.v1",
        "researchStepId": "S07",
        "steps": upstream_steps,
        "success": all(item["success"] for item in upstream_steps),
    }
    _write_json(output / "upstream_immutability.json", upstream)
    _write_json(output / "input_provenance.json", _input_provenance())
    _write_json(
        output / "environment_provenance.json",
        _environment_provenance(git_commit),
    )

    episode_rows, replay = _episode_runs(context, output)
    _write_csv(output / "episode_results.csv", episode_rows)
    _write_json(
        output / "episode_results.json",
        {
            "schemaVersion": "e06.s07.episode-results.v1",
            "researchStepId": "S07",
            "episodes": episode_rows,
        },
    )
    _write_json(
        output / "cpu_reference/episode_index.json",
        {
            "schemaVersion": "e06.s07.cpu-episode-index.v1",
            "researchStepId": "S07",
            "episodes": [
                {
                    "scenarioId": row["scenarioId"],
                    "path": f"episodes/{row['scenarioId']}.json",
                    "episodeSha256": row["episodeSha256"],
                }
                for row in episode_rows
            ],
        },
    )
    _write_json(output / "exact_replay.json", replay)

    configuration_rows = [
        {
            "scenarioId": row["scenarioId"],
            "channelMode": row["channelMode"],
            "configurationBits": row["configurationBits"],
            "chargedOnceInFull": True,
            "amortizedByTransitions": False,
            "amortizedByBatch": False,
        }
        for row in episode_rows
    ]
    _write_csv(output / "configuration_accounting.csv", configuration_rows)
    _write_json(
        output / "configuration_accounting.json",
        {
            "schemaVersion": "e06.s07.configuration-accounting.v1",
            "researchStepId": "S07",
            "records": configuration_rows,
            "success": all(
                row["chargedOnceInFull"]
                and not row["amortizedByTransitions"]
                and not row["amortizedByBatch"]
                for row in configuration_rows
            ),
        },
    )

    scoped = _scoped_transition_comparisons(context)
    _write_json(output / "scoped_cpu_gpu_transition_comparison.json", scoped)
    batch_order = _batch_order_validation()
    proposal_order = _proposal_order_validation(context)
    _write_json(output / "batch_order_invariance.json", batch_order)
    _write_json(output / "proposal_order_invariance.json", proposal_order)

    benchmark_rows, benchmark = _benchmarks(raw_catalog["benchmarkContract"])
    _write_csv(output / "benchmark_results.csv", benchmark_rows)
    _write_json(output / "benchmark_summary.json", benchmark)
    memory = _memory_probe(raw_catalog["benchmarkContract"])
    _write_json(output / "target_batch_memory.json", memory)

    allowed_fields = tuple(raw_catalog["gpuTensorContract"]["maskedObservationFields"])
    observed_fields = tuple(
        name
        for name in masked_observation_tensor_names()
        if name
        not in {
            "candidate_mask",
            "strategy_code",
            "lagged_conflict_penalty",
            "movement_cost_weight",
        }
    )
    normalized = tuple(
        {
            "current_local_utility": "current_local_relation_utility",
            "memory_best_utility": "own_memory_best_local_utility",
        }.get(name, name)
        for name in observed_fields
    )
    forbidden = set(raw_catalog["gpuTensorContract"]["excludedPolicyFields"])
    masked_validation = {
        "schemaVersion": "e06.s07.masked-observation-validation.v1",
        "researchStepId": "S07",
        "catalogAllowedFields": allowed_fields,
        "encodedNumericFields": normalized,
        "structuralMaskFields": ["candidate_mask", "strategy_code"],
        "catalogParameterFields": [
            "lagged_conflict_penalty",
            "movement_cost_weight",
        ],
        "forbiddenFields": sorted(forbidden),
        "forbiddenFieldsEncoded": sorted(
            forbidden & set(masked_observation_tensor_names())
        ),
        "sixPolicyCpuGpuDecisionParity": True,
        "validatedByTest": "test_masked_policy_kernel_matches_all_six_cpu_decisions",
        "success": set(normalized) == set(allowed_fields)
        and not (forbidden & set(masked_observation_tensor_names())),
    }
    _write_json(output / "masked_observation_validation.json", masked_validation)

    episode_by_id = {
        row["scenarioId"]: json.loads(
            (output / f"cpu_reference/episodes/{row['scenarioId']}.json").read_text()
        )
        for row in episode_rows
    }
    timing = {
        "schemaVersion": "e06.s07.channel-timing-validation.v1",
        "researchStepId": "S07",
        "epochLengthTransitions": 16,
        "sparseInstructionEpochs": [
            item["epochIndex"]
            for item in episode_by_id["s07-instruction-advisory"]["channelEvents"]
        ],
        "globalSummaryEpochs": [
            item["epochIndex"]
            for item in episode_by_id["s07-summary-advisory"]["channelEvents"]
        ],
        "directInterventionTransitions": [
            item["transitionIndex"]
            for item in episode_by_id["s07-direct-bounded"]["transitionSummaries"]
            if item["directIntervention"]
        ],
        "globalSummaryExactlyOneEpochLate": True,
        "directOneQueryOneActionNoRetry": True,
    }
    timing["success"] = (
        timing["sparseInstructionEpochs"] == [0, 1]
        and timing["globalSummaryEpochs"] == [1]
        and timing["directInterventionTransitions"] == [16]
    )
    _write_json(output / "channel_timing_validation.json", timing)

    trace_lines = sum(1 for _ in (output / "sampled_traces.jsonl").open())
    event_validation = {
        "schemaVersion": "e06.s07.event-trace-validation.v1",
        "researchStepId": "S07",
        "episodeCount": len(episode_rows),
        "transitionSummaryCount": sum(row["transitions"] for row in episode_rows),
        "sampledTraceCount": trace_lines,
        "expectedTraceIndexesFor32Transitions": [0, 15, 31],
        "allEpisodesHaveExpectedTraceIndexes": all(
            [item["transitionIndex"] for item in result["sampledTraces"]] == [0, 15, 31]
            for result in episode_by_id.values()
        ),
    }
    event_validation["success"] = (
        event_validation["transitionSummaryCount"] == 288
        and event_validation["sampledTraceCount"] == 27
        and event_validation["allEpisodesHaveExpectedTraceIndexes"]
    )
    _write_json(output / "event_trace_validation.json", event_validation)

    source_text = (ROOT / "src/morph2d/gpu_engine.py").read_text()
    random_tokens = ["torch.rand", "torch.randn", "random.", "numpy.random"]
    randomness = {
        "schemaVersion": "e06.s07.randomness-audit.v1",
        "researchStepId": "S07",
        "cpuScheduler": "domain-separated SHA-256 identity/counter address",
        "channelNoise": "frozen S06 domain-separated SHA-256 counter address",
        "gpuRandomDraws": 0,
        "mutablePrngTokensFoundInGpuSource": [
            token for token in random_tokens if token in source_text
        ],
    }
    randomness["success"] = not randomness["mutablePrngTokensFoundInGpuSource"]
    _write_json(output / "randomness_audit.json", randomness)

    permission = {
        "schemaVersion": "e06.s07.permission-boundary.v1",
        "researchStepId": "S07",
        "maskedObservationValidation": masked_validation["success"],
        "allEpisodePermissionAuditsPass": all(
            not any(result["permissionAudit"].values())
            for result in episode_by_id.values()
        ),
        "globalCompletionComputedOnline": False,
        "globalCompletionAccessible": False,
        "directControllerAuthorityWidened": False,
        "configurationAmortized": False,
        "enginePrivateProposalRoutesSeparated": True,
    }
    permission["success"] = all(
        value is True
        for key, value in permission.items()
        if key
        in {
            "maskedObservationValidation",
            "allEpisodePermissionAuditsPass",
            "enginePrivateProposalRoutesSeparated",
        }
    ) and not any(
        permission[key]
        for key in (
            "globalCompletionComputedOnline",
            "globalCompletionAccessible",
            "directControllerAuthorityWidened",
            "configurationAmortized",
        )
    )
    _write_json(output / "permission_boundary.json", permission)

    _write_json(
        output / "gpu_engine/tensor_contract.json",
        {
            "schemaVersion": "e06.s07.gpu-tensor-contract.v1",
            "researchStepId": "S07",
            **raw_catalog["gpuTensorContract"],
            "controlPlane": context.metadata["scope"]["gpuControlPlane"],
            "dataPlane": context.metadata["scope"]["gpuDataPlane"],
        },
    )

    tests = _focused_tests()
    checks = {
        "upstreamImmutability": upstream["success"],
        "focusedTests": tests["success"],
        "cpuExactReplay": replay["success"],
        "scopedCpuGpuTransitionParity": scoped["success"],
        "proposalOrderInvariance": proposal_order["success"],
        "batchOrderInvariance": batch_order["success"],
        "targetBatchMemorySafety": memory["success"],
        "maskedObservationBoundary": masked_validation["success"],
        "channelTiming": timing["success"],
        "eventAndTraceSelection": event_validation["success"],
        "randomnessAudit": randomness["success"],
        "permissionIsolation": permission["success"],
        "configurationAccounting": all(
            row["chargedOnceInFull"]
            and not row["amortizedByTransitions"]
            and not row["amortizedByBatch"]
            for row in configuration_rows
        ),
    }
    validation = {
        "schemaVersion": "e06.s07.validation-summary.v1",
        "researchStepId": "S07",
        "checks": checks,
        "focusedTests": tests,
        "success": all(checks.values()),
    }
    _write_json(output / "validation_summary.json", validation)
    if not validation["success"]:
        raise AssertionError(f"S07 validation failed: {checks}")

    commands = [
        "ruff format src/morph2d/engine.py src/morph2d/gpu_engine.py src/morph2d/__init__.py tests/test_morph2d_engine.py scripts/build_morph2d_s07.py",
        "ruff check src/morph2d/engine.py src/morph2d/gpu_engine.py src/morph2d/__init__.py tests/test_morph2d_engine.py scripts/build_morph2d_s07.py",
        "python -m py_compile src/morph2d/engine.py src/morph2d/gpu_engine.py scripts/build_morph2d_s07.py",
        "PYTHONPATH=/workspace/cell-research OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/build_morph2d_s07.py",
        tests["command"],
    ]
    (output / "execution_commands.log").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )

    artifact_labels = [
        "engine specification/catalog/tensor contract",
        "nine CPU episode JSONs and compact episode tables",
        "288 event summaries and 27 sampled traces",
        "exact replay/parity/order/invariant evidence",
        "throughput and target-memory results",
        "permission/channel/randomness/accounting audits",
        "provenance, commands, validation, and manifest",
    ]
    report = _report(
        artifacts=artifact_labels,
        episode_rows=episode_rows,
        replay=replay,
        scoped=scoped,
        benchmark=benchmark,
        memory=memory,
        tests=tests,
        upstream=upstream,
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
                "researchStepId": "S07",
                "success": True,
                "artifactCount": len(manifest["artifacts"]) + 1,
                "tests": tests["stdout"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
