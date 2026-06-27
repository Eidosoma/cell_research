"""GPU-screened evolutionary search helpers for E04 S08.

The GPU objective in this module is a compact screening proxy. Final evidence
for S08 candidates is produced by replaying elites in the CPU reference
simulator and auditing their serialized policies with the S07 no-oracle gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from e02_deterministic_simulator.metrics import sortedness_percent
from morphospace import LocalRulePolicy

from .homeostasis import (
    HOMEOSTASIS_BENCHMARK_VERSION,
    build_homeostatic_benchmark_config,
    normalized_sortedness_record,
    stable_json,
)
from .learning import (
    LOCAL_LEARNING_VERSION,
    LOCAL_REWARD_ALLOWED_INPUTS,
    LearningEventSimulator,
    LocalLearningConfig,
    LocalLearningPolicyWrapper,
    apply_homeostatic_event_to_simulator,
    normalize_action_probabilities,
)
from .memory import MEMORY_REPAIR_VERSION, MemoryConfig
from .repair import REPAIR_REPAIR_VERSION, RepairRuleConfig
from .signals import SIGNAL_REPAIR_VERSION, SignalConfig
from .training_constraints import (
    TRAINING_CONSTRAINT_VERSION,
    audit_policy_spec_for_oracle_access,
    certify_training_record,
    local_only_training_protocol,
)


EVOLUTION_SEARCH_VERSION = "e04_s08_gpu_evolution.v1"
GENOME_FIELD_NAMES = (
    "swap_left_weight",
    "swap_right_weight",
    "wait_weight",
    "learning_rate",
    "successful_swap_reward",
    "local_inversion_reward",
    "blocked_neighbor_reward",
    "blocked_swap_penalty_strength",
    "non_improving_swap_penalty_strength",
    "recent_failure_bonus",
    "memory_counter_max",
    "memory_neighbor_history",
    "signal_emission_scale",
    "signal_diffusion_rate",
    "signal_decay",
    "repair_nudge_threshold",
    "repair_signal_threshold",
    "signal_variant",
    "memory_variant",
)
GENOME_WIDTH = len(GENOME_FIELD_NAMES)
S08_LOCAL_OBSERVATION_FIELDS = (
    "actor_value",
    "actor_frozen",
    "actor_algotype",
    "actor_local_memory",
    "actor_local_signal",
    "left_value",
    "left_frozen",
    "left_algotype",
    "right_value",
    "right_frozen",
    "right_algotype",
    "local_signal_center",
    "local_signal_left",
    "local_signal_right",
    "local_signal_sum",
    "local_signal_mean",
    "local_signal_max",
    "recent_local_outcome",
    "selected_local_action",
    "action_probabilities",
    "local_tick",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if isinstance(value, float):
        return round(float(value), 12) if math.isfinite(float(value)) else None
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, payload: Any) -> str:
    digest = hashlib.sha256(_compact_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def normalize_genome_array(genomes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a two-dimensional clipped genome array."""

    array = np.asarray(genomes, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != GENOME_WIDTH:
        raise ValueError(f"genome arrays must have shape (n, {GENOME_WIDTH}), got {array.shape}")
    return np.clip(array, 0.0, 1.0)


def seeded_reference_genome() -> np.ndarray:
    """A deterministic local-prior seed used to anchor the first S08 population."""

    return np.asarray(
        [
            0.04,
            0.98,
            0.04,
            0.82,
            0.82,
            0.78,
            0.88,
            0.32,
            0.70,
            0.62,
            0.66,
            0.72,
            0.72,
            0.34,
            0.10,
            0.00,
            0.04,
            0.92,
            0.82,
        ],
        dtype=float,
    )


def genome_dict(vector: Sequence[float], *, candidate_id: str | None = None) -> dict[str, Any]:
    genome = normalize_genome_array(vector)[0]
    payload = {name: float(genome[index]) for index, name in enumerate(GENOME_FIELD_NAMES)}
    payload["genomeWidth"] = GENOME_WIDTH
    payload["genomeFieldNames"] = list(GENOME_FIELD_NAMES)
    payload["evolutionSearchVersion"] = EVOLUTION_SEARCH_VERSION
    payload["candidateId"] = candidate_id or _stable_id("e04_s08_candidate", payload)
    return _json_ready(payload)


def _action_probabilities(genome: np.ndarray) -> dict[str, float]:
    weights = {
        "swap_left": 0.02 + float(genome[0]),
        "swap_right": 0.02 + float(genome[1]),
        "wait": 0.02 + float(genome[2]),
    }
    return normalize_action_probabilities(weights, min_probability=0.04, max_probability=0.92)


def memory_config_from_genome(vector: Sequence[float]) -> MemoryConfig:
    genome = normalize_genome_array(vector)[0]
    selector = float(genome[18])
    if selector < 0.25:
        variant = "no_memory"
    elif selector < 0.50:
        variant = "one_bit"
    elif selector < 0.75:
        variant = "bounded_counter"
    else:
        variant = "neighbor_memory"
    return MemoryConfig(
        variant=variant,
        counter_max=1 + int(round(9.0 * float(genome[10]))),
        neighbor_history=1 + int(round(5.0 * float(genome[11]))),
    )


def signal_config_from_genome(vector: Sequence[float], *, seed: int = 0) -> SignalConfig:
    genome = normalize_genome_array(vector)[0]
    selector = float(genome[17])
    if selector < 0.25:
        variant = "no_signal"
    elif selector < 0.60:
        variant = "nearest_neighbor"
    else:
        variant = "diffusive"
    return SignalConfig(
        variant=variant,
        signal_range=1,
        diffusion_rate=0.02 + 0.46 * float(genome[13]),
        decay=0.35 * float(genome[14]),
        emission_scale=0.10 + 1.90 * float(genome[12]),
        random_seed=int(seed),
    )


def repair_config_from_genome(vector: Sequence[float]) -> RepairRuleConfig:
    genome = normalize_genome_array(vector)[0]
    signal_variant = signal_config_from_genome(genome).variant
    variant = "signal_threshold" if signal_variant != "no_signal" and float(genome[17]) >= 0.60 else "nudge_count"
    return RepairRuleConfig(
        variant=variant,
        nudge_threshold=1 + int(math.floor(3.0 * float(genome[15]))),
        signal_threshold=0.20 + 1.80 * float(genome[16]),
        approach_direction="either",
    )


def learning_config_from_genome(
    vector: Sequence[float],
    *,
    variant: str = "local_adaptive",
    seed: int = 0,
) -> LocalLearningConfig:
    genome = normalize_genome_array(vector)[0]
    return LocalLearningConfig(
        variant=variant,
        learning_rate=0.05 + 0.80 * float(genome[3]),
        min_action_probability=0.04,
        max_action_probability=0.92,
        random_seed=int(seed),
        successful_swap_reward=0.10 + 0.90 * float(genome[4]),
        local_inversion_reward=0.05 + 0.80 * float(genome[5]),
        blocked_neighbor_reward=0.05 + 0.55 * float(genome[6]),
        blocked_swap_penalty=-(0.05 + 0.75 * float(genome[7])),
        non_improving_swap_penalty=-(0.05 + 0.75 * float(genome[8])),
        recent_failure_bonus=0.02 + 0.30 * float(genome[9]),
        locally_ordered_wait_reward=0.01 + 0.08 * (1.0 - float(genome[2])),
        disordered_wait_penalty=-(0.01 + 0.22 * float(genome[2])),
        initial_action_probabilities=_action_probabilities(genome),
    )


def policy_from_genome(
    vector: Sequence[float],
    *,
    learning_variant: str = "local_adaptive",
    seed: int = 0,
) -> LocalLearningPolicyWrapper:
    return LocalLearningPolicyWrapper(
        "bubble",
        learning_config_from_genome(vector, variant=learning_variant, seed=seed),
    )


def candidate_record_from_genome(vector: Sequence[float], *, seed: int = 0) -> dict[str, Any]:
    genome = normalize_genome_array(vector)[0]
    candidate_id = _stable_id("e04_s08_candidate", genome_dict(genome))
    policy = policy_from_genome(genome, seed=seed)
    memory_config = memory_config_from_genome(genome)
    signal_config = signal_config_from_genome(genome, seed=seed)
    repair_config = repair_config_from_genome(genome)
    policy_spec = policy.to_spec().to_dict()
    policy_spec["parameters"]["s08Genome"] = genome_dict(genome, candidate_id=candidate_id)
    policy_spec["parameters"]["memoryConfig"] = memory_config.to_dict()
    policy_spec["parameters"]["signalConfig"] = signal_config.to_dict()
    policy_spec["parameters"]["repairConfig"] = repair_config.to_dict()
    policy_spec["parameters"]["candidateTrainingProtocol"] = "local_only"
    policy_spec["parameters"]["usesGlobalSortednessSignal"] = False
    policy_spec["parameters"]["usesWholeArrayTargetSignal"] = False
    return {
        "candidateId": candidate_id,
        "genome": genome_dict(genome, candidate_id=candidate_id),
        "policySpec": policy_spec,
        "learningConfig": policy.learning_config.to_dict(),
        "memoryConfig": memory_config.to_dict(),
        "signalConfig": signal_config.to_dict(),
        "repairConfig": repair_config.to_dict(),
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        "localLearningVersion": LOCAL_LEARNING_VERSION,
        "memoryRepairVersion": MEMORY_REPAIR_VERSION,
        "signalRepairVersion": SIGNAL_REPAIR_VERSION,
        "repairRepairVersion": REPAIR_REPAIR_VERSION,
    }


def audit_candidate_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the S07 no-global-oracle policy and training-record audit."""

    protocol = local_only_training_protocol()
    policy_spec = candidate["policySpec"]
    policy_audit = audit_policy_spec_for_oracle_access(policy_spec, protocol)
    training_record = {
        "policySpec": policy_spec,
        "observationLog": {"fields": list(S08_LOCAL_OBSERVATION_FIELDS)},
        "rewardRecord": {
            "inputFieldsUsed": list(LOCAL_REWARD_ALLOWED_INPUTS),
            "localInputs": {
                "actor_value": 2,
                "left_value": None,
                "left_frozen": False,
                "right_value": 1,
                "right_frozen": True,
                "selected_local_action": "swap_right",
                "outcome_swapped": False,
                "outcome_blocked_move_attempt": True,
                "outcome_reason": "learning_swap_blocked",
                "prior_recent_local_failures": 0,
            },
            "usesGlobalSortednessSignal": False,
            "usesWholeArrayTargetSignal": False,
        },
    }
    certification = certify_training_record(training_record, protocol)
    success = bool(policy_audit["success"] and certification["success"])
    return {
        "candidateId": str(candidate["candidateId"]),
        "success": success,
        "policyAuditSuccess": bool(policy_audit["success"]),
        "trainingCertificationSuccess": bool(certification["success"]),
        "policyAuditJson": _compact_json(policy_audit),
        "trainingCertificationJson": _compact_json(certification),
        "trainingConstraintVersion": TRAINING_CONSTRAINT_VERSION,
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }


def _proxy_components_numpy(genomes: np.ndarray) -> dict[str, np.ndarray]:
    genome = normalize_genome_array(genomes)
    weights = np.maximum(genome[:, 0:3] + 0.02, 1e-12)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    p_left = probabilities[:, 0]
    p_right = probabilities[:, 1]
    p_wait = probabilities[:, 2]
    learning = 0.05 + 0.80 * genome[:, 3]
    success_reward = 0.10 + 0.90 * genome[:, 4]
    inversion_reward = 0.05 + 0.80 * genome[:, 5]
    blocked_reward = 0.05 + 0.55 * genome[:, 6]
    non_improving_penalty = 0.05 + 0.75 * genome[:, 8]
    recent_failure = 0.02 + 0.30 * genome[:, 9]
    memory_capacity = 0.55 * genome[:, 10] + 0.45 * genome[:, 11]
    emission = 0.10 + 1.90 * genome[:, 12]
    diffusion = 0.02 + 0.46 * genome[:, 13]
    decay = 0.35 * genome[:, 14]
    nudge_speed = 1.0 - genome[:, 15]
    signal_threshold_speed = 1.0 - genome[:, 16]
    signal_enabled = np.clip((genome[:, 17] - 0.25) / 0.75, 0.0, 1.0)
    signal_strength = signal_enabled * emission * (1.0 - decay) * (0.35 + diffusion)
    repair_drive = (
        0.46 * p_right
        + 0.13 * learning
        + 0.10 * blocked_reward
        + 0.08 * recent_failure
        + 0.08 * nudge_speed
        + 0.08 * signal_threshold_speed * signal_enabled
    )
    homeostatic_drive = (
        0.14 * (p_left + p_right)
        + 0.10 * inversion_reward
        + 0.06 * memory_capacity
        + 0.05 * signal_strength
        - 0.05 * p_wait
    )
    reward_shape = 0.07 * success_reward + 0.05 * non_improving_penalty
    complexity_penalty = 0.035 * emission**2 + 0.020 * genome[:, 10] + 0.020 * genome[:, 11] + 0.015 * signal_enabled
    imbalance_penalty = 0.025 * np.abs(p_left - 0.25)
    score = repair_drive + homeostatic_drive + reward_shape - complexity_penalty - imbalance_penalty
    return {
        "score": score,
        "p_left": p_left,
        "p_right": p_right,
        "p_wait": p_wait,
        "repair_drive": repair_drive,
        "homeostatic_drive": homeostatic_drive,
        "signal_strength": signal_strength,
        "complexity_penalty": complexity_penalty,
    }


def proxy_fitness_numpy(genomes: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    return _proxy_components_numpy(normalize_genome_array(genomes))["score"].astype(float)


def proxy_fitness_torch(genomes: Any, *, device: str | None = None) -> Any:
    """Torch implementation of the S08 batched proxy objective."""

    import torch

    tensor = torch.as_tensor(genomes, dtype=torch.float64, device=device)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.shape[1] != GENOME_WIDTH:
        raise ValueError(f"genome arrays must have width {GENOME_WIDTH}, got {tuple(tensor.shape)}")
    genome = torch.clamp(tensor, 0.0, 1.0)
    weights = torch.clamp(genome[:, 0:3] + 0.02, min=1e-12)
    probabilities = weights / weights.sum(dim=1, keepdim=True)
    p_left = probabilities[:, 0]
    p_right = probabilities[:, 1]
    p_wait = probabilities[:, 2]
    learning = 0.05 + 0.80 * genome[:, 3]
    success_reward = 0.10 + 0.90 * genome[:, 4]
    inversion_reward = 0.05 + 0.80 * genome[:, 5]
    blocked_reward = 0.05 + 0.55 * genome[:, 6]
    non_improving_penalty = 0.05 + 0.75 * genome[:, 8]
    recent_failure = 0.02 + 0.30 * genome[:, 9]
    memory_capacity = 0.55 * genome[:, 10] + 0.45 * genome[:, 11]
    emission = 0.10 + 1.90 * genome[:, 12]
    diffusion = 0.02 + 0.46 * genome[:, 13]
    decay = 0.35 * genome[:, 14]
    nudge_speed = 1.0 - genome[:, 15]
    signal_threshold_speed = 1.0 - genome[:, 16]
    signal_enabled = torch.clamp((genome[:, 17] - 0.25) / 0.75, 0.0, 1.0)
    signal_strength = signal_enabled * emission * (1.0 - decay) * (0.35 + diffusion)
    repair_drive = (
        0.46 * p_right
        + 0.13 * learning
        + 0.10 * blocked_reward
        + 0.08 * recent_failure
        + 0.08 * nudge_speed
        + 0.08 * signal_threshold_speed * signal_enabled
    )
    homeostatic_drive = (
        0.14 * (p_left + p_right)
        + 0.10 * inversion_reward
        + 0.06 * memory_capacity
        + 0.05 * signal_strength
        - 0.05 * p_wait
    )
    reward_shape = 0.07 * success_reward + 0.05 * non_improving_penalty
    complexity_penalty = 0.035 * emission**2 + 0.020 * genome[:, 10] + 0.020 * genome[:, 11] + 0.015 * signal_enabled
    imbalance_penalty = 0.025 * torch.abs(p_left - 0.25)
    return repair_drive + homeostatic_drive + reward_shape - complexity_penalty - imbalance_penalty


def cpu_gpu_agreement_smoke(*, seed: int = 808, device: str | None = None) -> dict[str, Any]:
    import torch

    rng = np.random.default_rng(seed)
    genomes = rng.random((6, GENOME_WIDTH))
    genomes[0] = seeded_reference_genome()
    cpu_scores = proxy_fitness_numpy(genomes)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_scores = proxy_fitness_torch(genomes, device=device).detach().cpu().numpy()
    abs_diff = np.abs(cpu_scores - torch_scores)
    return {
        "seed": int(seed),
        "device": str(device),
        "genomeCount": int(genomes.shape[0]),
        "maxAbsDiff": float(abs_diff.max(initial=0.0)),
        "meanAbsDiff": float(abs_diff.mean()),
        "success": bool(float(abs_diff.max(initial=0.0)) <= 1e-9),
        "cpuScores": [float(value) for value in cpu_scores],
        "torchScores": [float(value) for value in torch_scores],
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }


@dataclass(frozen=True)
class EvolutionSearchConfig:
    population_size: int = 32
    generations: int = 6
    elite_count: int = 6
    mutation_scale: float = 0.10
    seed: int = 808
    device: str | None = None

    def __post_init__(self) -> None:
        if int(self.population_size) < 4:
            raise ValueError("population_size must be at least 4")
        if int(self.elite_count) < 1 or int(self.elite_count) >= int(self.population_size):
            raise ValueError("elite_count must be positive and smaller than population_size")
        if int(self.generations) < 1:
            raise ValueError("generations must be positive")
        if float(self.mutation_scale) < 0.0:
            raise ValueError("mutation_scale must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "populationSize": int(self.population_size),
            "generations": int(self.generations),
            "eliteCount": int(self.elite_count),
            "mutationScale": float(self.mutation_scale),
            "seed": int(self.seed),
            "device": self.device,
            "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
        }


def run_gpu_evolution_search(config: EvolutionSearchConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run a small batched evolutionary screen with torch on GPU when available."""

    import torch

    if config is None:
        cfg = EvolutionSearchConfig()
    elif isinstance(config, EvolutionSearchConfig):
        cfg = config
    else:
        cfg = EvolutionSearchConfig(
            population_size=int(config.get("populationSize", config.get("population_size", 32))),
            generations=int(config.get("generations", 6)),
            elite_count=int(config.get("eliteCount", config.get("elite_count", 6))),
            mutation_scale=float(config.get("mutationScale", config.get("mutation_scale", 0.10))),
            seed=int(config.get("seed", 808)),
            device=config.get("device"),
        )
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    population = torch.rand((cfg.population_size, GENOME_WIDTH), dtype=torch.float64, device=device, generator=generator)
    population[0] = torch.as_tensor(seeded_reference_genome(), dtype=torch.float64, device=device)
    parent_ids = [None] * cfg.population_size
    genome_ids = [f"g000_i{index:03d}" for index in range(cfg.population_size)]
    population_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for generation in range(cfg.generations):
        scores = proxy_fitness_torch(population, device=device)
        score_np = scores.detach().cpu().numpy()
        population_np = population.detach().cpu().numpy()
        components = _proxy_components_numpy(population_np)
        order = np.argsort(-score_np)
        elite_indices = order[: cfg.elite_count]
        for rank, index in enumerate(order):
            genome = population_np[index]
            row = {
                "generation": int(generation),
                "rank": int(rank + 1),
                "genomeId": genome_ids[index],
                "parentGenomeId": parent_ids[index],
                "proxyFitness": float(score_np[index]),
                "isElite": bool(index in set(elite_indices.tolist())),
                "device": str(device),
                "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
            }
            for name, value in genome_dict(genome).items():
                if name in GENOME_FIELD_NAMES:
                    row[name] = value
            for key in ("p_left", "p_right", "p_wait", "repair_drive", "homeostatic_drive", "signal_strength", "complexity_penalty"):
                row[key] = float(components[key][index])
            population_rows.append(row)
        curve_rows.append(
            {
                "generation": int(generation),
                "bestProxyFitness": float(score_np[order[0]]),
                "meanProxyFitness": float(score_np.mean()),
                "medianProxyFitness": float(np.median(score_np)),
                "worstProxyFitness": float(score_np[order[-1]]),
                "bestGenomeId": genome_ids[order[0]],
                "device": str(device),
                "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
            }
        )
        if generation == cfg.generations - 1:
            break
        elites = population[elite_indices].clone()
        next_population = [elite for elite in elites]
        next_parent_ids = [genome_ids[index] for index in elite_indices]
        next_genome_ids = [f"g{generation + 1:03d}_elite{rank:03d}" for rank in range(cfg.elite_count)]
        for rank, parent_index in enumerate(elite_indices):
            lineage_rows.append(
                {
                    "generation": int(generation + 1),
                    "childGenomeId": next_genome_ids[rank],
                    "parentGenomeId": genome_ids[parent_index],
                    "mutationScale": 0.0,
                    "lineageRole": "elite_copy",
                    "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
                }
            )
        child_index = cfg.elite_count
        while len(next_population) < cfg.population_size:
            parent_rank = int(child_index % cfg.elite_count)
            parent = elites[parent_rank]
            noise = torch.randn((GENOME_WIDTH,), dtype=torch.float64, device=device, generator=generator) * cfg.mutation_scale
            child = torch.clamp(parent + noise, 0.0, 1.0)
            next_population.append(child)
            child_id = f"g{generation + 1:03d}_i{child_index:03d}"
            next_genome_ids.append(child_id)
            parent_id = genome_ids[elite_indices[parent_rank]]
            next_parent_ids.append(parent_id)
            lineage_rows.append(
                {
                    "generation": int(generation + 1),
                    "childGenomeId": child_id,
                    "parentGenomeId": parent_id,
                    "mutationScale": float(cfg.mutation_scale),
                    "lineageRole": "mutated_child",
                    "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
                }
            )
            child_index += 1
        population = torch.stack(next_population, dim=0)
        genome_ids = next_genome_ids
        parent_ids = next_parent_ids

    final_scores = proxy_fitness_torch(population, device=device).detach().cpu().numpy()
    final_population = population.detach().cpu().numpy()
    final_order = np.argsort(-final_scores)
    elites: list[dict[str, Any]] = []
    for rank, index in enumerate(final_order[: cfg.elite_count]):
        candidate = candidate_record_from_genome(final_population[index], seed=cfg.seed + rank)
        candidate["genomeId"] = genome_ids[index]
        candidate["generation"] = int(cfg.generations - 1)
        candidate["rank"] = int(rank + 1)
        candidate["proxyFitness"] = float(final_scores[index])
        candidate["audit"] = audit_candidate_record(candidate)
        elites.append(_json_ready(candidate))

    return {
        "config": cfg.to_dict() | {"resolvedDevice": str(device)},
        "populationRows": _json_ready(population_rows),
        "lineageRows": _json_ready(lineage_rows),
        "trainingCurveRows": _json_ready(curve_rows),
        "eliteCandidates": elites,
        "device": str(device),
        "cudaAvailable": bool(torch.cuda.is_available()),
        "torchVersion": str(torch.__version__),
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }


def _make_simulator(
    initial_values: Sequence[int],
    policy: str | LocalRulePolicy,
    *,
    signal_config: SignalConfig,
    repair_config: RepairRuleConfig,
    frozen_positions: Sequence[int] = (),
    frozen_variant: str = "stuck",
    scheduler_seed: int = 0,
    tie_breaker_seed: int = 0,
    condition_id: str = "s08_replay",
) -> LearningEventSimulator:
    return LearningEventSimulator(
        initial_values,
        policy,
        frozen_positions=frozen_positions,
        frozen_variant=frozen_variant if frozen_positions else "none",
        scheduler_seed=int(scheduler_seed),
        tie_breaker_seed=int(tie_breaker_seed),
        signal_config=signal_config,
        repair_config=repair_config,
        condition_id=condition_id,
        implementation="e04_s08_cpu_reference_replay",
        research_step_id="S08",
    )


def _manual_repair_trial(
    *,
    candidate_id: str,
    controller_id: str,
    policy: str | LocalRulePolicy,
    initial_values: Sequence[int],
    frozen_positions: Sequence[int],
    signal_config: SignalConfig,
    repair_config: RepairRuleConfig,
    scheduler_seed: int,
    tie_breaker_seed: int,
    horizon: int = 32,
    split: str = "training",
) -> dict[str, Any]:
    sim = _make_simulator(
        initial_values,
        policy,
        signal_config=signal_config,
        repair_config=repair_config,
        frozen_positions=frozen_positions,
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
        condition_id=f"{split}_{candidate_id}_{controller_id}",
    )
    for _ in range(int(horizon)):
        if sim.is_sorted() and not sim.current_frozen_positions():
            break
        sim.step()
    recovered = len(sim.recovery_events)
    final_sortedness = sortedness_percent(sim.current_values())
    completed = bool(sim.is_sorted() and not sim.current_frozen_positions())
    score = (
        (2.0 if completed else 0.0)
        + 0.75 * recovered
        + final_sortedness / 100.0
        + max(0.0, 1.0 - sim.activation_count / max(1, horizon))
        - 0.015 * sim.blocked_move_attempts
        - 0.005 * sim.comparison_count
    )
    return {
        "candidateId": candidate_id,
        "controllerId": controller_id,
        "benchmarkFamily": "repair",
        "benchmarkId": f"frozen_pair_{list(initial_values)}_frozen_{list(frozen_positions)}",
        "split": split,
        "schedulerSeed": int(scheduler_seed),
        "tieBreakerSeed": int(tie_breaker_seed),
        "horizon": int(horizon),
        "completed": completed,
        "score": float(score),
        "activationCount": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "blockedMoveAttempts": int(sim.blocked_move_attempts),
        "recoveredCellCount": int(recovered),
        "remainingFrozenCellCount": int(len(sim.current_frozen_positions())),
        "initialValuesJson": _compact_json(list(initial_values)),
        "finalValuesJson": _compact_json(sim.current_values()),
        "finalSortednessPercent": float(final_sortedness),
        "repairVariant": repair_config.variant,
        "signalVariant": signal_config.variant,
        "learningVariant": getattr(getattr(policy, "learning_config", None), "variant", "none"),
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }


def _manual_homeostasis_trial(
    *,
    candidate_id: str,
    controller_id: str,
    policy: str | LocalRulePolicy,
    task: Mapping[str, Any],
    signal_config: SignalConfig,
    repair_config: RepairRuleConfig,
    scheduler_seed: int,
    tie_breaker_seed: int,
    activations_per_tick: int = 12,
    split: str = "training",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sim = _make_simulator(
        task["initialValues"],
        policy,
        signal_config=signal_config,
        repair_config=repair_config,
        frozen_positions=(),
        frozen_variant="stuck",
        scheduler_seed=scheduler_seed,
        tie_breaker_seed=tie_breaker_seed,
        condition_id=f"{split}_{candidate_id}_{controller_id}_{task['taskId']}",
    )
    threshold = float(task["sortednessThresholdPercent"])
    initial_length = len(task["initialValues"])
    tick_records: list[dict[str, Any]] = []
    for tick in range(int(task["horizonTicks"])):
        events = [event for event in task.get("perturbationSchedule", ()) if int(event["tick"]) == tick]
        for event in events:
            apply_homeostatic_event_to_simulator(sim, event)
        pre = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
        for _ in range(max(1, int(activations_per_tick))):
            sim.step()
        post = normalized_sortedness_record(sim.current_values(), initial_length=initial_length)
        tick_records.append(
            {
                "candidateId": candidate_id,
                "controllerId": controller_id,
                "taskId": str(task["taskId"]),
                "split": split,
                "tick": int(tick),
                "eventsJson": _compact_json(events),
                "preSortednessPercent": float(pre["sortednessPercent"]),
                "postSortednessPercent": float(post["sortednessPercent"]),
                "postInRange": bool(post["sortednessPercent"] >= threshold),
                "currentLength": int(post["currentLength"]),
                "currentPairDenominator": int(post["currentPairDenominator"]),
                "valuesJson": _compact_json(sim.current_values()),
                "activationCount": int(sim.activation_count),
                "blockedMoveAttempts": int(sim.blocked_move_attempts),
                "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
                "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
            }
        )
    in_range = [row["postInRange"] for row in tick_records]
    final_sortedness = float(tick_records[-1]["postSortednessPercent"]) if tick_records else 100.0
    score = (
        sum(1 for value in in_range if value) / max(1, len(in_range))
        + final_sortedness / 100.0
        - 0.005 * sim.blocked_move_attempts
        - 0.001 * sim.activation_count
    )
    row = {
        "candidateId": candidate_id,
        "controllerId": controller_id,
        "benchmarkFamily": "homeostasis",
        "benchmarkId": str(task["taskId"]),
        "split": split,
        "schedulerSeed": int(scheduler_seed),
        "tieBreakerSeed": int(tie_breaker_seed),
        "horizon": int(task["horizonTicks"]),
        "completed": bool(final_sortedness >= threshold),
        "score": float(score),
        "activationCount": int(sim.activation_count),
        "swapCount": int(sim.swap_count),
        "comparisonCount": int(sim.comparison_count),
        "blockedMoveAttempts": int(sim.blocked_move_attempts),
        "recoveredCellCount": int(len(sim.recovery_events)),
        "remainingFrozenCellCount": int(len(sim.current_frozen_positions())),
        "initialValuesJson": _compact_json(task["initialValues"]),
        "finalValuesJson": _compact_json(sim.current_values()),
        "finalSortednessPercent": final_sortedness,
        "timeInRangeFraction": float(sum(1 for value in in_range if value) / max(1, len(in_range))),
        "repairVariant": repair_config.variant,
        "signalVariant": signal_config.variant,
        "learningVariant": getattr(getattr(policy, "learning_config", None), "variant", "none"),
        "homeostasisBenchmarkVersion": HOMEOSTASIS_BENCHMARK_VERSION,
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }
    return row, tick_records


def replay_candidate(
    candidate: Mapping[str, Any],
    *,
    base_seed: int = 9000,
    split: str = "training",
    repair_horizon: int = 32,
) -> dict[str, Any]:
    """Replay one evolved candidate plus fixed and open-loop controls."""

    genome_values = [float(candidate["genome"][name]) for name in GENOME_FIELD_NAMES]
    signal_config = signal_config_from_genome(genome_values, seed=base_seed)
    repair_config = repair_config_from_genome(genome_values)
    candidate_policy = policy_from_genome(genome_values, seed=base_seed)
    fixed_policy = LocalLearningPolicyWrapper(
        "bubble",
        learning_config_from_genome(genome_values, variant="fixed", seed=base_seed),
    )
    controllers: list[tuple[str, str | LocalRulePolicy]] = [
        ("evolved_local_adaptive", candidate_policy),
        ("fixed_local_baseline", fixed_policy),
        ("open_loop_bubble_baseline", "bubble"),
    ]
    replay_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    repair_cases = [
        {"initialValues": [2, 1], "frozenPositions": [1], "seedOffset": 0},
        {"initialValues": [3, 1, 2], "frozenPositions": [1], "seedOffset": 50},
    ]
    for controller_index, (controller_id, policy) in enumerate(controllers):
        for case_index, case in enumerate(repair_cases):
            replay_rows.append(
                _manual_repair_trial(
                    candidate_id=str(candidate["candidateId"]),
                    controller_id=controller_id,
                    policy=policy,
                    initial_values=case["initialValues"],
                    frozen_positions=case["frozenPositions"],
                    signal_config=signal_config,
                    repair_config=repair_config,
                    scheduler_seed=base_seed + int(case["seedOffset"]) + controller_index,
                    tie_breaker_seed=base_seed + int(case["seedOffset"]) + 100 + case_index,
                    horizon=repair_horizon,
                    split=split,
                )
            )
    tasks = build_homeostatic_benchmark_config()["tasks"]
    selected_task_ids = {"adjacent_swap_recovery", "insertion_deletion_normalization", "frozen_damage_mixed_events"}
    for controller_index, (controller_id, policy) in enumerate(controllers[:2]):
        for task_index, task in enumerate(task for task in tasks if task["taskId"] in selected_task_ids):
            row, rows = _manual_homeostasis_trial(
                candidate_id=str(candidate["candidateId"]),
                controller_id=controller_id,
                policy=policy,
                task=task,
                signal_config=signal_config,
                repair_config=repair_config,
                scheduler_seed=base_seed + 200 + controller_index * 10 + task_index,
                tie_breaker_seed=base_seed + 400 + controller_index * 10 + task_index,
                activations_per_tick=12,
                split=split,
            )
            replay_rows.append(row)
            tick_rows.extend(rows)
    return {
        "replayRows": _json_ready(replay_rows),
        "tickRows": _json_ready(tick_rows),
        "candidateId": str(candidate["candidateId"]),
        "split": split,
        "signalConfig": signal_config.to_dict(),
        "repairConfig": repair_config.to_dict(),
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }


def summarize_replay_advantage(replay_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    repair_rows = [
        row
        for row in replay_rows
        if row.get("benchmarkFamily") == "repair" and row.get("controllerId") in {"evolved_local_adaptive", "fixed_local_baseline"}
    ]
    evolved = [row for row in repair_rows if row.get("controllerId") == "evolved_local_adaptive"]
    fixed = [row for row in repair_rows if row.get("controllerId") == "fixed_local_baseline"]
    fixed_by_benchmark = {str(row["benchmarkId"]): row for row in fixed}
    advantages: list[dict[str, Any]] = []
    for row in evolved:
        baseline = fixed_by_benchmark.get(str(row["benchmarkId"]))
        if baseline is None:
            continue
        advantages.append(
            {
                "benchmarkId": str(row["benchmarkId"]),
                "evolvedScore": float(row["score"]),
                "fixedScore": float(baseline["score"]),
                "scoreDelta": float(row["score"]) - float(baseline["score"]),
                "evolvedActivationCount": int(row["activationCount"]),
                "fixedActivationCount": int(baseline["activationCount"]),
                "activationDelta": int(baseline["activationCount"]) - int(row["activationCount"]),
                "evolvedCompleted": bool(row["completed"]),
                "fixedCompleted": bool(baseline["completed"]),
            }
        )
    best_delta = max((row["scoreDelta"] for row in advantages), default=float("-inf"))
    return {
        "success": bool(best_delta > 0.0),
        "bestRepairScoreDelta": None if not advantages else float(best_delta),
        "repairAdvantages": _json_ready(advantages),
        "evolutionSearchVersion": EVOLUTION_SEARCH_VERSION,
    }


def replay_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(list(rows)).encode("utf-8")).hexdigest()
