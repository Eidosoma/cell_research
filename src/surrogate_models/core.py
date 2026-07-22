"""Core contracts and models for E07 S06.

The module deliberately works from the persisted S05 selected-row ledger.  It
never materializes S02 validation or confirmation outcomes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.policy_dsl import compile_policy
from src.quality_diversity.core import OBJECTIVES, TASK_IDS as TASK_IDS


S05_ROOT = Path("/artifacts/research_steps/S05")
S06_ROOT = Path("/artifacts/research_steps/S06")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_bytes(value)
    ).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(numeric_leaves(child, path))
    elif isinstance(value, bool):
        leaves[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        leaves[prefix] = float(value)
    return leaves


def get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[part]
    return current


def build_s05_freeze(protocol: Mapping[str, Any]) -> dict[str, Any]:
    frozen = protocol["s05Freeze"]
    expected_manifest = str(frozen["artifactManifestSha256"])
    manifest_path = Path(str(frozen["artifactManifest"]))
    actual_manifest = hash_file(manifest_path)
    if actual_manifest != expected_manifest:
        raise RuntimeError("S05 artifact manifest changed before S06 freeze")
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for item in manifest["artifacts"]:
        path = S05_ROOT / item["path"]
        actual = hash_file(path)
        checks.append(
            {
                "path": item["path"],
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "pass": actual == item["sha256"],
            }
        )
    if not all(item["pass"] for item in checks):
        raise RuntimeError("S05 artifact hash verification failed")

    explicit = {
        "search_protocol.yaml": frozen["searchProtocolSha256"],
        "generation_plan_ledger.jsonl": frozen["generationPlanLedgerSha256"],
        "lineage_graph.jsonl": frozen["lineageLedgerSha256"],
        "policy_catalog.jsonl": frozen["policyCatalogSha256"],
        "archive/archive_manifest.json": frozen["archiveManifestSha256"],
    }
    for relative, expected in explicit.items():
        if hash_file(S05_ROOT / relative) != expected:
            raise RuntimeError(f"S05 frozen input changed: {relative}")

    rows = read_jsonl(S05_ROOT / "evaluation_ledger.jsonl")
    selected = [row for row in rows if row["selectedForCanonicalSearch"]]
    orphans = [row for row in rows if not row["selectedForCanonicalSearch"]]
    auth = protocol["authorizedPopulation"]
    if len(selected) != int(auth["expectedPrimaryRows"]):
        raise RuntimeError("canonical selected-row population changed")
    if len(orphans) != int(auth["expectedRecoveryOrphanRows"]):
        raise RuntimeError("recovery-orphan population changed")
    if any(row["split"] != "train" for row in rows):
        raise RuntimeError("non-training row found in S05 evaluation ledger")
    plans = read_jsonl(S05_ROOT / "generation_plan_ledger.jsonl")
    if len(plans) != int(frozen["expectedGenerationPlans"]):
        raise RuntimeError("authoritative generation-plan count changed")
    if any(
        row["protocolSha256"] != frozen["searchProtocolSha256"] for row in plans
    ):
        raise RuntimeError("generation plan references a different protocol")

    return {
        "schemaVersion": "e07.s06.s05-hash-freeze.v1",
        "researchStepId": "S06",
        "success": True,
        "s05ArtifactManifestPath": str(manifest_path),
        "s05ArtifactManifestSha256": actual_manifest,
        "verifiedArtifactCount": len(checks),
        "artifactHashChecks": checks,
        "explicitInputHashes": explicit,
        "persistedGenerationPlanIsAuthoritative": True,
        "authoritativeGenerationPlanCount": len(plans),
        "selectedCanonicalRowCount": len(selected),
        "recoveryOrphanRowCount": len(orphans),
        "selectedStableEvaluationCommitmentSha256": canonical_hash(
            "E07/S06/selected-evaluation-set/v1",
            sorted(row["stableEvaluationSha256"] for row in selected),
        ),
        "recoveryOrphanStableEvaluationCommitmentSha256": canonical_hash(
            "E07/S06/recovery-orphan-set/v1",
            sorted(row["stableEvaluationSha256"] for row in orphans),
        ),
        "validationOutcomeRows": 0,
        "confirmationOutcomeRows": 0,
        "claimBoundary": (
            "Hash freeze of persisted S05 training evidence; no validation or "
            "confirmation outcome was opened."
        ),
    }


def verify_frozen_inputs(
    protocol: Mapping[str, Any], freeze: Mapping[str, Any]
) -> None:
    current = build_s05_freeze(protocol)
    stable_keys = (
        "s05ArtifactManifestSha256",
        "explicitInputHashes",
        "authoritativeGenerationPlanCount",
        "selectedCanonicalRowCount",
        "recoveryOrphanRowCount",
        "selectedStableEvaluationCommitmentSha256",
        "recoveryOrphanStableEvaluationCommitmentSha256",
    )
    if any(current[key] != freeze[key] for key in stable_keys):
        raise RuntimeError("S05 evidence changed after S06 preregistration")


def load_policy_catalog_with_plans() -> dict[str, dict[str, Any]]:
    catalog = {
        row["policySha256"]: row
        for row in read_jsonl(S05_ROOT / "policy_catalog.jsonl")
    }
    for plan in read_jsonl(S05_ROOT / "generation_plan_ledger.jsonl"):
        for offspring in plan["offspring"]:
            row = {
                **offspring,
                "origin": "s05_recovery_plan",
                "family": "mutated_dsl",
                "generation": int(plan["generation"]),
                "searchTaskId": plan["taskId"],
            }
            catalog.setdefault(row["policySha256"], row)
    return catalog


def load_lineage() -> dict[str, dict[str, Any]]:
    return {
        row["policySha256"]: row
        for row in read_jsonl(S05_ROOT / "lineage_graph.jsonl")
    }


def lineage_component(policy_hash: str, lineage: Mapping[str, Mapping[str, Any]]) -> str:
    frontier = [policy_hash]
    seen: set[str] = set()
    roots: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        row = lineage.get(current)
        if row is None or not row.get("parents"):
            roots.add(current)
            continue
        for parent in row["parents"]:
            if parent in seen:
                continue
            if parent not in lineage:
                roots.add(str(parent))
            else:
                frontier.append(str(parent))
    return canonical_hash("E07/S06/lineage-component/v1", sorted(roots))


def _balanced_component_assignment(
    task_id: str, component_sizes: Mapping[str, int]
) -> dict[str, str]:
    fractions = {"fit": 0.60, "calibration": 0.20, "test": 0.20}
    current = {key: 0 for key in fractions}
    target = {
        key: max(1.0, sum(component_sizes.values()) * value)
        for key, value in fractions.items()
    }
    ordered = sorted(
        component_sizes,
        key=lambda component: (
            -component_sizes[component],
            canonical_hash("E07/S06/lineage-order/v1", [task_id, component]),
        ),
    )
    assignment: dict[str, str] = {}
    for component in ordered:
        partition = min(
            fractions,
            key=lambda key: (
                current[key] / target[key],
                canonical_hash(
                    "E07/S06/lineage-partition-tie/v1", [task_id, component, key]
                ),
            ),
        )
        assignment[component] = partition
        current[partition] += component_sizes[component]
    return assignment


def assign_grouped_splits(
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["selectedForCanonicalSearch"]]
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_task[str(row["taskId"])].append(row)
    scenario_rules = protocol["splitDesign"]["scenarioFamilies"]
    assignments: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(by_task.items()):
        policy_components = {
            row["policySha256"]: lineage_component(row["policySha256"], lineage)
            for row in task_rows
        }
        component_sizes = Counter(policy_components.values())
        component_partition = _balanced_component_assignment(task_id, component_sizes)
        scenario = scenario_rules.get(task_id, scenario_rules["default"])
        scenario_lookup = {
            int(ordinal): partition
            for partition, ordinals in scenario.items()
            for ordinal in ordinals
        }
        for row in task_rows:
            ordinal = int(row["scenarioOrdinal"])
            if ordinal not in scenario_lookup:
                raise RuntimeError(f"unregistered scenario family {task_id}:{ordinal}")
            component = policy_components[row["policySha256"]]
            assignments.append(
                {
                    "stableEvaluationSha256": row["stableEvaluationSha256"],
                    "taskId": task_id,
                    "policySha256": row["policySha256"],
                    "lineageComponent": component,
                    "lineagePartition": component_partition[component],
                    "scenarioOrdinal": ordinal,
                    "scenarioPartition": scenario_lookup[ordinal],
                    "failed": bool(row["failed"]),
                    "censored": bool(row["censored"]),
                    "selectedForCanonicalSearch": True,
                }
            )
    return sorted(assignments, key=lambda row: row["stableEvaluationSha256"])


def policy_feature_dict(
    row: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]], *, scenario: bool
) -> dict[str, float]:
    policy = catalog[row["policySha256"]]
    document = policy["document"]
    compiled = compile_policy(document)
    features: dict[str, float] = {
        f"complexity::{key}": float(value)
        for key, value in compiled.complexity.to_dict().items()
    }
    features["meta::generation"] = float(policy.get("generation", 0))
    features[f"meta::origin::{policy.get('origin', 'unknown')}"] = 1.0
    features[f"meta::family::{policy.get('family', 'unknown')}"] = 1.0
    mutation = policy.get("mutation") or {}
    features[f"mutation::{mutation.get('operatorId', 'none')}"] = 1.0
    for permission in document.get("permissions", []):
        features[f"permission::{permission}"] = 1.0
    for key, value in document.get("limits", {}).items():
        features[f"limit::{key}"] = float(value)
    for item in document.get("memory", []):
        features["memory::registers"] = features.get("memory::registers", 0.0) + 1
        features["memory::bits"] = features.get("memory::bits", 0.0) + float(
            item.get("bits", 0)
        )
    signals = document.get("signals", {})
    features["signals::channels"] = float(signals.get("channels", 0))
    features["signals::bitsPerChannel"] = float(signals.get("bitsPerChannel", 0))

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if "kind" in value:
                key = f"action::{value['kind']}"
                features[key] = features.get(key, 0.0) + 1.0
            if "op" in value:
                key = f"operator::{value['op']}"
                features[key] = features.get(key, 0.0) + 1.0
            if "obs" in value:
                key = f"observation::{value['obs']}"
                features[key] = features.get(key, 0.0) + 1.0
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(document.get("rules", []))
    walk(document.get("default", {}))
    if scenario:
        features["scenario::ordinal"] = float(row["scenarioOrdinal"])
        derivation = row.get("scenarioDerivation", {})
        for key in ("replicate", "faultIndex"):
            value = derivation.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                features[f"scenario::{key}"] = float(value)
    return features


def build_target_registry(task_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    objectives = []
    binary_objective_ids = {
        "sorting_completion",
        "fault_completion",
        "detour_completion",
        "chimera_completion",
        "regeneration_success",
        "target_completion",
        "post_hit_departure",
        "conjunctive_completion",
        "s02_local_acceptance",
        "repair_success",
    }
    continuous = []
    binary = []
    for objective_id, path, direction in OBJECTIVES[task_id]:
        item = {
            "name": f"objective::{objective_id}",
            "sourcePath": path,
            "direction": direction,
            "group": "performance",
            "transform": "identity",
        }
        objectives.append(item)
        if objective_id in binary_objective_ids:
            binary.append({**item, "type": "binary"})
        else:
            continuous.append({**item, "type": "continuous"})
    cost_fields = sorted(
        {
            path
            for row in rows
            for path in numeric_leaves(row["nativeLedgerFamilies"])
            if not path.endswith("licensedLongRangeMaximumRequestedDistance")
        }
    )
    licensed_tokens = (
        "licensedPrefix",
        "engineCursor",
        "licensedLongRange",
    )
    costs = []
    for path in cost_fields:
        item = {
            "name": f"cost::{path}",
            "sourcePath": path,
            "direction": "minimize",
            "group": "cost",
            "transform": "signed_log1p",
            "type": "continuous",
            "licensedCapability": any(token in path for token in licensed_tokens),
        }
        costs.append(item)
        continuous.append(item)
        present = sum(
            path in numeric_leaves(row["nativeLedgerFamilies"]) for row in rows
        )
        if present != len(rows):
            binary.append(
                {
                    "name": f"cost_present::{path}",
                    "sourcePath": path,
                    "group": "binary",
                    "type": "binary_presence",
                    "transform": "identity",
                    "licensedCapability": any(token in path for token in licensed_tokens),
                }
            )
    binary.extend(
        [
            {
                "name": "status::failed",
                "sourcePath": "failed",
                "group": "binary",
                "type": "binary",
                "transform": "identity",
            },
            {
                "name": "status::censored",
                "sourcePath": "censored",
                "group": "binary",
                "type": "binary",
                "transform": "identity",
            },
        ]
    )
    operational = {
        "name": "operational::elapsedSeconds",
        "sourcePath": "elapsedSeconds",
        "direction": "minimize",
        "group": "operational",
        "transform": "log1p",
        "type": "continuous",
    }
    continuous.append(operational)
    return {
        "schemaVersion": "e07.s06.target-registry.v1",
        "taskId": task_id,
        "universalScore": False,
        "continuous": continuous,
        "binary": binary,
        "objectiveCount": len(objectives),
        "costCount": len(costs),
        "licensedCapabilityCostTargets": [
            item["name"] for item in costs if item["licensedCapability"]
        ],
        "claimBoundary": "Task-specific targets retain native names and are never pooled into a universal score.",
    }


def target_arrays(
    rows: Sequence[Mapping[str, Any]], registry: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    continuous = np.full((len(rows), len(registry["continuous"])), np.nan, dtype=np.float64)
    binary = np.full((len(rows), len(registry["binary"])), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        costs = numeric_leaves(row["nativeLedgerFamilies"])
        for index, item in enumerate(registry["continuous"]):
            name = item["name"]
            if name.startswith("objective::"):
                continuous[row_index, index] = float(
                    get_path(row["outcome"], item["sourcePath"])
                )
            elif name.startswith("cost::"):
                if item["sourcePath"] in costs:
                    continuous[row_index, index] = costs[item["sourcePath"]]
            else:
                continuous[row_index, index] = float(row["elapsedSeconds"])
        for index, item in enumerate(registry["binary"]):
            name = item["name"]
            if name.startswith("objective::"):
                binary[row_index, index] = float(
                    get_path(row["outcome"], item["sourcePath"])
                )
            elif name.startswith("cost_present::"):
                binary[row_index, index] = float(item["sourcePath"] in costs)
            elif name == "status::failed":
                binary[row_index, index] = float(row["failed"])
            elif name == "status::censored":
                binary[row_index, index] = float(row["censored"])
    return continuous, binary


@dataclass
class TargetTransform:
    centers: np.ndarray
    scales: np.ndarray
    transforms: tuple[str, ...]

    def _base(self, values: np.ndarray) -> np.ndarray:
        result = values.copy()
        for index, transform in enumerate(self.transforms):
            if transform == "signed_log1p":
                result[:, index] = np.sign(result[:, index]) * np.log1p(
                    np.abs(result[:, index])
                )
            elif transform == "log1p":
                result[:, index] = np.log1p(np.maximum(result[:, index], 0.0))
        return result

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (self._base(values) - self.centers) / self.scales

    def inverse(self, values: np.ndarray) -> np.ndarray:
        result = values * self.scales + self.centers
        for index, transform in enumerate(self.transforms):
            if transform == "signed_log1p":
                result[:, index] = np.sign(result[:, index]) * np.expm1(
                    np.abs(result[:, index])
                )
            elif transform == "log1p":
                result[:, index] = np.expm1(result[:, index])
        return result


def fit_target_transform(values: np.ndarray, registry: Mapping[str, Any]) -> TargetTransform:
    transforms = tuple(item["transform"] for item in registry["continuous"])
    temporary = TargetTransform(
        np.zeros(values.shape[1]), np.ones(values.shape[1]), transforms
    )._base(values)
    centers = np.nanmean(temporary, axis=0)
    centers = np.where(np.isfinite(centers), centers, 0.0)
    scales = np.nanstd(temporary, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    return TargetTransform(centers, scales, transforms)


class SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int, continuous_dim: int, binary_dim: int):
        super().__init__()
        widths = (192, 128, 64)
        layers: list[nn.Module] = []
        previous = input_dim
        for width in widths:
            layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(0.1)]
            )
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.continuous = nn.Linear(previous, continuous_dim)
        self.binary = nn.Linear(previous, binary_dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(value)
        return self.continuous(hidden), self.binary(hidden)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not torch.any(mask):
        return prediction.sum() * 0.0
    return torch.square(prediction[mask] - target[mask]).mean()


def _masked_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not torch.any(mask):
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask])


def _group_loss(
    continuous_prediction: torch.Tensor,
    binary_logits: torch.Tensor,
    continuous_target: torch.Tensor,
    binary_target: torch.Tensor,
    registry: Mapping[str, Any],
    loss_weights: Mapping[str, float],
) -> torch.Tensor:
    total = continuous_prediction.sum() * 0.0
    for group in ("performance", "cost", "operational"):
        indices = [
            index
            for index, item in enumerate(registry["continuous"])
            if item["group"] == group
        ]
        binary_indices = [
            index
            for index, item in enumerate(registry["binary"])
            if item["group"] == group
        ]
        parts = []
        if indices:
            parts.append(
                _masked_mse(
                    continuous_prediction[:, indices], continuous_target[:, indices]
                )
            )
        if binary_indices:
            parts.append(
                _masked_bce(binary_logits[:, binary_indices], binary_target[:, binary_indices])
            )
        if parts:
            total = total + float(loss_weights[group]) * torch.stack(parts).mean()
    status_indices = [
        index
        for index, item in enumerate(registry["binary"])
        if item["group"] == "binary"
    ]
    if status_indices:
        total = total + float(loss_weights["binary"]) * _masked_bce(
            binary_logits[:, status_indices], binary_target[:, status_indices]
        )
    return total


def train_surrogate_member(
    x_train: np.ndarray,
    continuous_train: np.ndarray,
    binary_train: np.ndarray,
    x_calibration: np.ndarray,
    continuous_calibration: np.ndarray,
    binary_calibration: np.ndarray,
    registry: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    set_deterministic(seed)
    model = SurrogateMLP(
        x_train.shape[1], continuous_train.shape[1], binary_train.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learningRate"]),
        weight_decay=float(settings["weightDecay"]),
    )
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    ct = torch.as_tensor(continuous_train, dtype=torch.float32, device=device)
    bt = torch.as_tensor(binary_train, dtype=torch.float32, device=device)
    xv = torch.as_tensor(x_calibration, dtype=torch.float32, device=device)
    cv = torch.as_tensor(continuous_calibration, dtype=torch.float32, device=device)
    bv = torch.as_tensor(binary_calibration, dtype=torch.float32, device=device)
    batch_size = int(settings["batchSize"])
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for epoch in range(int(settings["maximumEpochs"])):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size].to(device)
            prediction, logits = model(xt[index])
            loss = _group_loss(
                prediction,
                logits,
                ct[index],
                bt[index],
                registry,
                settings["lossWeights"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            prediction, logits = model(xv)
            validation_loss = float(
                _group_loss(
                    prediction,
                    logits,
                    cv,
                    bv,
                    registry,
                    settings["lossWeights"],
                ).cpu()
            )
        history.append(
            {
                "epoch": epoch + 1,
                "trainLoss": float(np.mean(losses)),
                "calibrationLoss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if (
            epoch + 1 >= int(settings["minimumEpochs"])
            and epoch + 1 - best_epoch >= int(settings["earlyStoppingPatience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("surrogate training produced no state")
    return best_state, history


def ensemble_predict(
    states: Sequence[Mapping[str, torch.Tensor]],
    x: np.ndarray,
    continuous_dim: int,
    binary_dim: int,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    continuous_members = []
    binary_members = []
    tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    for state in states:
        model = SurrogateMLP(x.shape[1], continuous_dim, binary_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            continuous, logits = model(tensor)
        continuous_members.append(continuous.cpu().numpy())
        binary_members.append(torch.sigmoid(logits).cpu().numpy())
    c = np.stack(continuous_members)
    b = np.stack(binary_members)
    return c.mean(0), c.std(0), b.mean(0), b.std(0)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * gradient, None


class EventSummaryAutoencoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, task_count: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 96),
            nn.SiLU(),
            nn.Linear(96, input_dim),
        )
        self.task_head = nn.Linear(embedding_dim, task_count)
        self.length_head = nn.Linear(embedding_dim, 1)

    def forward(
        self, value: torch.Tensor, adversarial: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(value)
        reversed_embedding = (
            GradientReverse.apply(embedding, 1.0) if adversarial else embedding
        )
        return (
            embedding,
            self.decoder(embedding),
            self.task_head(reversed_embedding),
            self.length_head(reversed_embedding).squeeze(-1),
        )


def event_length(row: Mapping[str, Any]) -> float:
    leaves = {
        **numeric_leaves(row.get("nativeEvent", {})),
        **{
            f"outcome.{key}": value
            for key, value in numeric_leaves(row.get("outcome", {})).items()
        },
        **{
            f"ledger.{key}": value
            for key, value in numeric_leaves(row.get("nativeLedgerFamilies", {})).items()
        },
    }
    preferred = (
        "activationCount",
        "transitionCount",
        "phaseActivationCount",
        "fullTraceMetricCount",
        "distanceProfileCount",
        "dslRuntimeLedger.dslActivations",
    )
    for suffix in preferred:
        values = [value for key, value in leaves.items() if key.endswith(suffix)]
        if values:
            return max(values)
    return 0.0


def _suffix_value(leaves: Mapping[str, float], suffix: str) -> float | None:
    values = [value for key, value in leaves.items() if key.endswith(suffix)]
    return float(np.mean(values)) if values else None


def shared_event_features(row: Mapping[str, Any]) -> dict[str, float]:
    event = numeric_leaves(row.get("nativeEvent", {}))
    ledger = numeric_leaves(row.get("nativeLedgerFamilies", {}))
    length = max(event_length(row), 1.0)
    result: dict[str, float] = {
        "status_failed": float(row["failed"]),
        "status_censored": float(row["censored"]),
    }
    rates = (
        "acceptedNativeActionFraction",
        "committedDisplacementFraction",
        "committedMovementKindEntropy",
        "stateTurnoverFraction",
    )
    for suffix in rates:
        value = _suffix_value(event, suffix)
        result[suffix] = np.nan if value is None else value
    counts = (
        "dslOperations",
        "dslMemoryWrites",
        "emittedSignalWrites",
        "recipientDeliveries",
        "rejectedNativeActions",
        "licensedPrefixPredicateEvaluations",
        "engineCursorAdvanceActions",
        "engineCursorSwapActions",
        "acceptedSwaps",
        "noOps",
    )
    for suffix in counts:
        value = _suffix_value(ledger, suffix)
        result[f"per_length::{suffix}"] = (
            np.nan if value is None else math.log1p(max(value, 0.0)) / math.log1p(length)
        )
    for key, value in list(result.items()):
        result[f"missing::{key}"] = float(not math.isfinite(float(value)))
    return result


def raw_event_features(row: Mapping[str, Any]) -> dict[str, float]:
    result = {
        f"event::{key}": value
        for key, value in numeric_leaves(row.get("nativeEvent", {})).items()
    }
    result.update(
        {
            f"ledger::{key}": math.copysign(math.log1p(abs(value)), value)
            for key, value in numeric_leaves(row.get("nativeLedgerFamilies", {})).items()
        }
    )
    result["status::failed"] = float(row["failed"])
    result["status::censored"] = float(row["censored"])
    return result


def state_dict_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"E07/S06/state-dict/v1\x00")
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(canonical_bytes(list(value.shape)))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_bundle_hash(bundle: Mapping[str, Any]) -> str:
    metadata = deepcopy(dict(bundle))
    states = metadata.pop("stateDicts")
    return canonical_hash(
        "E07/S06/model-bundle/v1",
        {**metadata, "stateDictHashes": [state_dict_hash(state) for state in states]},
    )
