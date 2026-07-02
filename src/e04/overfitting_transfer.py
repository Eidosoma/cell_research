"""Overfitting and transfer tests for E04 S13.

S13 evaluates the selected S08 policies without retuning.  The policy-visible
decision path reuses the S09 memory-ablation runner so local policies consume
only the S07 projected local-only view after either the no-memory mask or the
S09-supported neighbor-memory mask.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from src.e02.deterministic_simulator import stable_json_sha256
from src.e04.fatigue_damage import RELIABILITY_MODES, FatigueDamageBenchmarkConfig
from src.e04.homeostasis import HOMEOSTATIC_TASKS, HomeostaticBenchmarkConfig, PerturbationSpec, build_homeostatic_schedule
from src.e04.memory_ablations import (
    MemoryAblationSpec,
    S09EvaluationConfig,
    default_memory_ablation_specs,
    simulate_memory_ablation,
)
from src.e04.memory_policies import CellMemoryConfig
from src.e04.no_oracle_protocol import ALLOWED_TRAINING_SIGNAL_FIELDS, EXCLUDED_TRAINING_SIGNAL_FIELDS, LOCAL_ONLY_PROTOCOL_ID
from src.e04.repairable_frozen import RepairBenchmarkConfig
from src.e04.evolutionary_search import EvolutionGenome


S13_PROTOCOL_ID = f"{LOCAL_ONLY_PROTOCOL_ID}:e04_s13_overfitting_transfer"
S13_POLICY_FAMILY = "s08_selected_policy_s13_overfitting_transfer"
S13_MEMORY_ABLATIONS = ("no_memory", "neighbor_memory")
S13_SPLITS = ("train_reference", "heldout")
S13_GENERALIZATION_AXES = (
    "s08_train_reference",
    "unseen_array_size",
    "unseen_perturbation_rate",
    "unseen_frozen_placement",
    "combined_stress",
)
S13_SCHEDULE_VARIANTS = (
    "s08_train_reference",
    "standard_homeostatic",
    "dense_perturbation_rate",
    "edge_cluster_frozen",
    "combined_dense_stress",
)
S08_TRAIN_REFERENCE_SEEDS = (18001, 18002)
S13_HELDOUT_SEEDS = (52001, 52002)
S13_PRIOR_SEED_MAX = 41405


@dataclass(frozen=True)
class S13TransferConfig:
    """One predefined S13 train-reference or held-out transfer condition."""

    split: str
    generalization_axis: str
    scenario_name: str
    values: tuple[int, ...]
    task_name: str
    reliability_mode: str
    activation_seed: int
    policy_seed: int
    schedule_seed: int
    max_events: int
    schedule_variant: str
    benchmark_family: str = "overfitting_transfer"
    algorithm: str = "bubble"
    repair_rule: str = "nudge_repair"
    activation_distribution: str = "uniform_active"
    target_sortedness_percent: float = 100.0
    nudge_threshold: int = 2
    fatigue_threshold: int = 3
    recovery_events: int = 4
    damage_threshold: int = 4
    sort_direction: str = "increasing"

    def __post_init__(self) -> None:
        if self.split not in S13_SPLITS:
            raise ValueError(f"Unsupported S13 split: {self.split}")
        if self.generalization_axis not in S13_GENERALIZATION_AXES:
            raise ValueError(f"Unsupported S13 generalization axis: {self.generalization_axis}")
        if self.schedule_variant not in S13_SCHEDULE_VARIANTS:
            raise ValueError(f"Unsupported S13 schedule variant: {self.schedule_variant}")
        if not self.scenario_name.startswith("s13_"):
            raise ValueError("S13 scenario names must be prefixed with s13_")
        if self.task_name not in HOMEOSTATIC_TASKS:
            raise ValueError(f"Unsupported task_name: {self.task_name}")
        if self.reliability_mode not in RELIABILITY_MODES:
            raise ValueError(f"Unsupported reliability_mode: {self.reliability_mode}")
        if len(self.values) < 4:
            raise ValueError("S13 transfer configs require at least four cells")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.split == "train_reference" and self.generalization_axis != "s08_train_reference":
            raise ValueError("train_reference configs must use s08_train_reference axis")
        if self.split == "heldout" and self.generalization_axis == "s08_train_reference":
            raise ValueError("heldout configs cannot use s08_train_reference axis")

    def to_homeostatic_config(self, *, algorithm: str | None = None, interface_mode: str = "full") -> HomeostaticBenchmarkConfig:
        return HomeostaticBenchmarkConfig(
            values=self.values,
            algorithm=algorithm or self.algorithm,
            task_name=self.task_name,
            repair_rule=self.repair_rule,
            interface_mode=interface_mode,
            reliability_mode=self.reliability_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            schedule_seed=self.schedule_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            target_sortedness_percent=self.target_sortedness_percent,
            nudge_threshold=self.nudge_threshold,
            fatigue_threshold=self.fatigue_threshold,
            recovery_events=self.recovery_events,
            damage_threshold=self.damage_threshold,
            sort_direction=self.sort_direction,
        )

    def to_repair_config(self, *, interface_mode: str = "full") -> RepairBenchmarkConfig:
        return RepairBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            frozen_indices=(),
            repair_rule=self.repair_rule,
            interface_mode=interface_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            nudge_threshold=self.nudge_threshold,
            sort_direction=self.sort_direction,
        )

    def to_fatigue_config(self, *, interface_mode: str = "full") -> FatigueDamageBenchmarkConfig:
        return FatigueDamageBenchmarkConfig(
            values=self.values,
            algorithm=self.algorithm,
            frozen_indices=(),
            repair_rule=self.repair_rule,
            interface_mode=interface_mode,
            reliability_mode=self.reliability_mode,
            activation_seed=self.activation_seed,
            policy_seed=self.policy_seed,
            max_events=self.max_events,
            activation_distribution=self.activation_distribution,
            nudge_threshold=self.nudge_threshold,
            fatigue_threshold=self.fatigue_threshold,
            recovery_events=self.recovery_events,
            damage_threshold=self.damage_threshold,
            sort_direction=self.sort_direction,
        )

    def config_key(self) -> tuple[Any, ...]:
        return (
            self.split,
            self.generalization_axis,
            self.scenario_name,
            self.values,
            self.task_name,
            self.reliability_mode,
            self.activation_seed,
            self.policy_seed,
            self.schedule_seed,
            self.max_events,
            self.schedule_variant,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "benchmark_family": self.benchmark_family,
            "generalization_axis": self.generalization_axis,
            "scenario_name": self.scenario_name,
            "values": list(self.values),
            "array_size": len(self.values),
            "algorithm": self.algorithm,
            "task_name": self.task_name,
            "repair_rule": self.repair_rule,
            "interface_mode": "s13_transfer",
            "reliability_mode": self.reliability_mode,
            "activation_seed": int(self.activation_seed),
            "policy_seed": int(self.policy_seed),
            "schedule_seed": int(self.schedule_seed),
            "max_events": int(self.max_events),
            "schedule_variant": self.schedule_variant,
            "activation_distribution": self.activation_distribution,
            "target_sortedness_percent": float(self.target_sortedness_percent),
            "nudge_threshold": int(self.nudge_threshold),
            "fatigue_threshold": int(self.fatigue_threshold),
            "recovery_events": int(self.recovery_events),
            "damage_threshold": int(self.damage_threshold),
            "sort_direction": self.sort_direction,
        }


def _swap_spec(event_step: int, left: int, right: int, length: int, variant: str) -> PerturbationSpec:
    left = max(0, min(length - 1, int(left)))
    right = max(0, min(length - 1, int(right)))
    if left == right:
        right = min(length - 1, left + 1) if left < length - 1 else max(0, left - 1)
    left, right = sorted((left, right))
    return PerturbationSpec(
        event_step=int(event_step),
        perturbation_type="swap",
        params={"left_position": left, "right_position": right},
        safe_representation=f"s13_predefined_{variant}_swap",
    )


def _freeze_spec(event_step: int, position: int, length: int, variant: str) -> PerturbationSpec:
    return PerturbationSpec(
        event_step=int(event_step),
        perturbation_type="freeze",
        params={"position": max(0, min(length - 1, int(position)))},
        safe_representation=f"s13_predefined_{variant}_freeze",
    )


def _turnover_spec(event_step: int, delete_position: int, insert_position: int, inserted_value: int, length: int, variant: str) -> PerturbationSpec:
    return PerturbationSpec(
        event_step=int(event_step),
        perturbation_type="delete_insert",
        params={
            "delete_position": max(0, min(length - 1, int(delete_position))),
            "insert_position": max(0, min(length - 1, int(insert_position))),
            "inserted_value": int(inserted_value),
        },
        safe_representation=f"s13_predefined_{variant}_fixed_size_turnover",
    )


def build_s13_transfer_schedule(config: S13TransferConfig) -> tuple[PerturbationSpec, ...]:
    """Build a deterministic S13 schedule without reading trajectory state."""

    if config.schedule_variant in {"s08_train_reference", "standard_homeostatic"}:
        return build_homeostatic_schedule(config.to_homeostatic_config())
    rng = random.Random(config.schedule_seed)
    length = len(config.values)
    high_value = max(config.values) + 11
    variant = config.schedule_variant
    if variant == "dense_perturbation_rate":
        steps = [12, 24, 36, 48, 60, 72, 84, 96, 108]
        specs: list[PerturbationSpec] = []
        for index, step in enumerate(steps):
            if index % 3 == 0:
                specs.append(_swap_spec(step, rng.randrange(length), rng.randrange(length), length, variant))
            elif index % 3 == 1:
                specs.append(_freeze_spec(step, rng.randrange(length), length, variant))
            else:
                specs.append(_turnover_spec(step, rng.randrange(length), rng.randrange(length), high_value + index + rng.randrange(4), length, variant))
    elif variant == "edge_cluster_frozen":
        mid = length // 2
        specs = [
            _freeze_spec(14, 0, length, variant),
            _freeze_spec(28, length - 1, length, variant),
            _freeze_spec(44, mid, length, variant),
            _freeze_spec(58, min(length - 1, mid + 1), length, variant),
            _swap_spec(76, max(0, mid - 2), min(length - 1, mid + 2), length, variant),
            _turnover_spec(104, rng.randrange(length), rng.randrange(length), high_value + rng.randrange(5), length, variant),
        ]
    elif variant == "combined_dense_stress":
        mid = length // 2
        specs = [
            _freeze_spec(15, 0, length, variant),
            _swap_spec(30, rng.randrange(length), rng.randrange(length), length, variant),
            _turnover_spec(45, rng.randrange(length), rng.randrange(length), high_value + rng.randrange(5), length, variant),
            _freeze_spec(60, mid, length, variant),
            _swap_spec(75, max(0, mid - 3), min(length - 1, mid + 3), length, variant),
            _freeze_spec(90, length - 1, length, variant),
            _turnover_spec(120, rng.randrange(length), rng.randrange(length), high_value + 6 + rng.randrange(5), length, variant),
            _swap_spec(150, rng.randrange(length), rng.randrange(length), length, variant),
        ]
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported S13 schedule variant: {variant}")
    clipped = [spec for spec in specs if spec.event_step <= config.max_events]
    return tuple(sorted(clipped, key=lambda item: (item.event_step, item.perturbation_type)))


def default_s13_transfer_configs(
    *,
    train_seeds: Sequence[int] = S08_TRAIN_REFERENCE_SEEDS,
    heldout_seeds: Sequence[int] = S13_HELDOUT_SEEDS,
    max_events: int = 120,
    stress_max_events: int = 180,
) -> tuple[S13TransferConfig, ...]:
    """Return predefined train-reference and held-out S13 configs."""

    configs: list[S13TransferConfig] = []
    for seed in train_seeds:
        for task_name in ("swap_shocks", "frozen_damage"):
            for reliability_mode in ("no_fatigue_control", "fatigue_recovery"):
                configs.append(
                    S13TransferConfig(
                        split="train_reference",
                        generalization_axis="s08_train_reference",
                        scenario_name=f"s13_train_reference_{task_name}_{reliability_mode}",
                        values=tuple(range(1, 7)),
                        task_name=task_name,
                        reliability_mode=reliability_mode,
                        activation_seed=seed,
                        policy_seed=seed + 10_000,
                        schedule_seed=seed + 20_000,
                        max_events=max_events,
                        schedule_variant="s08_train_reference",
                    )
                )
    for index, seed in enumerate(heldout_seeds):
        configs.extend(
            [
                S13TransferConfig(
                    split="heldout",
                    generalization_axis="unseen_array_size",
                    scenario_name="s13_unseen_array_size_mixed",
                    values=tuple(range(1, 11 + 4 * index)),
                    task_name="mixed_perturbations",
                    reliability_mode="cumulative_damage",
                    activation_seed=seed,
                    policy_seed=seed + 10_000,
                    schedule_seed=seed + 20_000,
                    max_events=stress_max_events,
                    schedule_variant="standard_homeostatic",
                ),
                S13TransferConfig(
                    split="heldout",
                    generalization_axis="unseen_perturbation_rate",
                    scenario_name="s13_dense_perturbation_rate",
                    values=tuple(range(1, 9 + 2 * index)),
                    task_name="mixed_perturbations",
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed + 100,
                    policy_seed=seed + 10_100,
                    schedule_seed=seed + 20_100,
                    max_events=max_events,
                    schedule_variant="dense_perturbation_rate",
                ),
                S13TransferConfig(
                    split="heldout",
                    generalization_axis="unseen_frozen_placement",
                    scenario_name="s13_edge_cluster_frozen_cells",
                    values=tuple(range(1, 9 + 2 * index)),
                    task_name="frozen_damage",
                    reliability_mode="fatigue_recovery",
                    activation_seed=seed + 200,
                    policy_seed=seed + 10_200,
                    schedule_seed=seed + 20_200,
                    max_events=max_events,
                    schedule_variant="edge_cluster_frozen",
                ),
                S13TransferConfig(
                    split="heldout",
                    generalization_axis="combined_stress",
                    scenario_name="s13_combined_transfer_stress",
                    values=tuple(range(1, 13 + 2 * index)),
                    task_name="mixed_perturbations",
                    reliability_mode="cumulative_damage",
                    activation_seed=seed + 300,
                    policy_seed=seed + 10_300,
                    schedule_seed=seed + 20_300,
                    max_events=stress_max_events,
                    schedule_variant="combined_dense_stress",
                ),
            ]
        )
    return tuple(configs)


def s13_memory_specs() -> tuple[MemoryAblationSpec, ...]:
    specs = {spec.name: spec for spec in default_memory_ablation_specs()}
    return tuple(specs[name] for name in S13_MEMORY_ABLATIONS)


def s13_config_records(configs: Sequence[S13TransferConfig]) -> list[dict[str, Any]]:
    records = []
    for config in configs:
        schedule = build_s13_transfer_schedule(config)
        records.append(
            {
                **config.to_dict(),
                "config_id": stable_json_sha256(config.to_dict())[:16],
                "schedule_json": json.dumps([item.to_dict() for item in schedule], sort_keys=True, separators=(",", ":"), default=str),
                "schedule_sha256": stable_json_sha256([item.to_dict() for item in schedule]),
                "schedule_event_count": len(schedule),
                "predefined_before_evaluation": True,
            }
        )
    return records


def s13_config_hash(configs: Sequence[S13TransferConfig]) -> str:
    return stable_json_sha256(s13_config_records(configs))


def assert_s13_design(configs: Sequence[S13TransferConfig], specs: Sequence[MemoryAblationSpec]) -> dict[str, Any]:
    keys = [config.config_key() for config in configs]
    train = [config for config in configs if config.split == "train_reference"]
    heldout = [config for config in configs if config.split == "heldout"]
    train_seeds = {config.activation_seed for config in train} | {config.policy_seed for config in train} | {config.schedule_seed for config in train}
    heldout_seeds = {config.activation_seed for config in heldout} | {config.policy_seed for config in heldout} | {config.schedule_seed for config in heldout}
    train_schedule_hashes = {stable_json_sha256([item.to_dict() for item in build_s13_transfer_schedule(config)]) for config in train}
    heldout_schedule_hashes = {stable_json_sha256([item.to_dict() for item in build_s13_transfer_schedule(config)]) for config in heldout}
    axis_counts = Counter(config.generalization_axis for config in configs)
    spec_names = tuple(spec.name for spec in specs)
    return {
        "success": bool(
            len(keys) == len(set(keys))
            and train
            and heldout
            and spec_names == S13_MEMORY_ABLATIONS
            and set(axis_counts) == set(S13_GENERALIZATION_AXES)
            and not (train_seeds & heldout_seeds)
            and min(heldout_seeds) > S13_PRIOR_SEED_MAX
            and not (train_schedule_hashes & heldout_schedule_hashes)
        ),
        "configCount": len(configs),
        "trainReferenceConfigCount": len(train),
        "heldoutConfigCount": len(heldout),
        "duplicateConfigKeys": len(keys) - len(set(keys)),
        "generalizationAxisCounts": dict(sorted(axis_counts.items())),
        "memoryAblations": list(spec_names),
        "trainSeeds": sorted(train_seeds),
        "heldoutSeeds": sorted(heldout_seeds),
        "trainHeldoutSeedOverlap": sorted(train_seeds & heldout_seeds),
        "trainHeldoutScheduleOverlapCount": len(train_schedule_hashes & heldout_schedule_hashes),
        "arraySizes": sorted({len(config.values) for config in configs}),
        "scheduleVariants": sorted({config.schedule_variant for config in configs}),
        "heldoutSeedFloor": S13_PRIOR_SEED_MAX + 1,
    }


def run_s13_transfer_matrix(
    *,
    genomes: Sequence[EvolutionGenome],
    configs: Sequence[S13TransferConfig],
    specs: Sequence[MemoryAblationSpec],
    frozen_config_hash: str,
) -> list[dict[str, Any]]:
    """Evaluate selected policies on predefined S13 configs with no retuning."""

    rows: list[dict[str, Any]] = []
    for genome in genomes:
        for config in configs:
            schedule = build_s13_transfer_schedule(config)
            config_id = stable_json_sha256(config.to_dict())[:16]
            for spec in specs:
                row = simulate_memory_ablation(config, genome, spec, schedule_override=schedule)  # type: ignore[arg-type]
                row.update(
                    {
                        "protocol_id": S13_PROTOCOL_ID,
                        "policy_family": S13_POLICY_FAMILY,
                        "candidate_type": "s13_selected_policy_transfer_test",
                        "benchmark_family": config.benchmark_family,
                        "generalization_axis": config.generalization_axis,
                        "scenario_name": config.scenario_name,
                        "schedule_variant": config.schedule_variant,
                        "s13_config_id": config_id,
                        "s13_config_sha256": frozen_config_hash,
                        "predefined_before_evaluation": True,
                        "selected_policy_was_retuned": False,
                        "policy_parameter_update_count": 0,
                        "config_generation_stage": "predefined_before_evaluation",
                    }
                )
                row["match_key_sha256"] = stable_json_sha256(
                    {
                        "policy_id": genome.policy_id,
                        "memory_ablation": spec.name,
                        "s13_config_id": config_id,
                        "schedule_sha256": row["schedule_sha256"],
                    }
                )
                row["fitness_score"] = float(row["fitness_score"])
                rows.append(row)
    return rows


def summarize_transfer_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["s08_source_policy_id", "memory_ablation_order", "memory_ablation", "split", "generalization_axis"], as_index=False)
        .agg(
            runs=("fitness_score", "size"),
            scenarios=("scenario_name", "nunique"),
            mean_fitness=("fitness_score", "mean"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
            mean_recovery_rate=("recovered_perturbation_count", "mean"),
            mean_unrecovered_perturbations=("unrecovered_perturbation_count", "mean"),
            mean_energy_total=("energy_total", "mean"),
            oracle_hit_rows=("uses_global_oracle", "sum"),
        )
        .sort_values(["s08_source_policy_id", "memory_ablation_order", "split", "generalization_axis"])
        .reset_index(drop=True)
    )
    for column in [
        "mean_fitness",
        "mean_time_in_target_fraction",
        "mean_final_sortedness_percent",
        "mean_recovery_rate",
        "mean_unrecovered_perturbations",
        "mean_energy_total",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def transfer_gaps(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["s08_source_policy_id", "memory_ablation", "generalization_axis", "split"], as_index=False)
        .agg(
            rows=("fitness_score", "size"),
            mean_fitness=("fitness_score", "mean"),
            mean_time_in_target_fraction=("time_in_target_fraction", "mean"),
            mean_final_sortedness_percent=("final_sortedness_percent", "mean"),
        )
    )
    train = grouped[grouped["split"] == "train_reference"][
        [
            "s08_source_policy_id",
            "memory_ablation",
            "mean_fitness",
            "mean_time_in_target_fraction",
            "mean_final_sortedness_percent",
        ]
    ].rename(
        columns={
            "mean_fitness": "train_reference_mean_fitness",
            "mean_time_in_target_fraction": "train_reference_time_in_target_fraction",
            "mean_final_sortedness_percent": "train_reference_final_sortedness_percent",
        }
    )
    heldout = grouped[grouped["split"] == "heldout"].copy()
    merged = heldout.merge(train, on=["s08_source_policy_id", "memory_ablation"], how="left", validate="many_to_one")
    merged["fitness_transfer_gap"] = merged["mean_fitness"] - merged["train_reference_mean_fitness"]
    merged["fitness_retention_fraction"] = merged["mean_fitness"] / merged["train_reference_mean_fitness"].replace(0, pd.NA)
    no_memory = merged[merged["memory_ablation"] == "no_memory"][
        ["s08_source_policy_id", "generalization_axis", "mean_fitness"]
    ].rename(columns={"mean_fitness": "no_memory_heldout_mean_fitness"})
    merged = merged.merge(no_memory, on=["s08_source_policy_id", "generalization_axis"], how="left", validate="many_to_one")
    merged["fitness_delta_vs_no_memory"] = merged["mean_fitness"] - merged["no_memory_heldout_mean_fitness"]
    for column in [
        "mean_fitness",
        "train_reference_mean_fitness",
        "fitness_transfer_gap",
        "fitness_retention_fraction",
        "no_memory_heldout_mean_fitness",
        "fitness_delta_vs_no_memory",
    ]:
        merged[column] = merged[column].astype(float).round(6)
    return merged.sort_values(["s08_source_policy_id", "memory_ablation", "generalization_axis"]).reset_index(drop=True)


def summarize_transfer_gaps(gap_df: pd.DataFrame) -> pd.DataFrame:
    if gap_df.empty:
        return pd.DataFrame()
    summary = (
        gap_df.groupby(["memory_ablation"], as_index=False)
        .agg(
            heldout_axis_rows=("fitness_transfer_gap", "size"),
            mean_train_reference_fitness=("train_reference_mean_fitness", "mean"),
            mean_heldout_fitness=("mean_fitness", "mean"),
            mean_fitness_transfer_gap=("fitness_transfer_gap", "mean"),
            mean_fitness_retention_fraction=("fitness_retention_fraction", "mean"),
            mean_fitness_delta_vs_no_memory=("fitness_delta_vs_no_memory", "mean"),
            positive_axis_fraction_vs_no_memory=("fitness_delta_vs_no_memory", lambda values: float((values > 0).mean())),
        )
        .sort_values("memory_ablation")
        .reset_index(drop=True)
    )
    for column in [
        "mean_train_reference_fitness",
        "mean_heldout_fitness",
        "mean_fitness_transfer_gap",
        "mean_fitness_retention_fraction",
        "mean_fitness_delta_vs_no_memory",
        "positive_axis_fraction_vs_no_memory",
    ]:
        summary[column] = summary[column].round(6)
    return summary


def infer_s13_outcome(gap_summary_df: pd.DataFrame) -> dict[str, Any]:
    criterion = (
        "neighbor_memory mean heldout/train fitness retention >= 0.85, "
        "mean heldout fitness delta vs no_memory > 0, and positive heldout-axis fraction >= 0.60"
    )
    if gap_summary_df.empty:
        return {"outcome": "null", "criterion": criterion, "reason": "no heldout transfer gaps were available"}
    neighbor = gap_summary_df[gap_summary_df["memory_ablation"] == "neighbor_memory"]
    if neighbor.empty:
        return {"outcome": "null", "criterion": criterion, "reason": "neighbor_memory gap summary unavailable"}
    row = neighbor.iloc[0].to_dict()
    retention = float(row["mean_fitness_retention_fraction"])
    delta = float(row["mean_fitness_delta_vs_no_memory"])
    positive_fraction = float(row["positive_axis_fraction_vs_no_memory"])
    if retention >= 0.85 and delta > 0.0 and positive_fraction >= 0.60:
        outcome = "supportive"
    elif retention < 0.75 or delta <= 0.0:
        outcome = "constraining/contradictory"
    else:
        outcome = "null"
    return {
        "outcome": outcome,
        "criterion": criterion,
        "neighborMeanFitnessRetentionFraction": retention,
        "neighborMeanHeldoutFitnessDeltaVsNoMemory": delta,
        "neighborPositiveHeldoutAxisFractionVsNoMemory": positive_fraction,
    }

