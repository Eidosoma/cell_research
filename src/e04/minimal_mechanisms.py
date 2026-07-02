"""Evidence synthesis helpers for E04 S15 minimal-mechanism conclusions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MINIMAL_MECHANISM_ID = "neighbor_memory_no_signal_local_adjacent_policy"
CENTRALIZED_MECHANISM_ID = "centralized_global_state_repair_ceiling"


@dataclass(frozen=True)
class S15EvidencePaths:
    """Inputs and outputs used by the S15 synthesis."""

    artifacts_dir: Path = Path("/artifacts")

    @property
    def s08_policies(self) -> Path:
        return self.artifacts_dir / "policies" / "e04_evolved_repair_policies.jsonl"

    @property
    def s09_results(self) -> Path:
        return self.artifacts_dir / "results" / "e04_memory_ablations.parquet"

    @property
    def s09_delta_summary(self) -> Path:
        return self.artifacts_dir / "tables" / "e04_memory_ablations_delta_summary.csv"

    @property
    def s10_results(self) -> Path:
        return self.artifacts_dir / "results" / "e04_communication_ablations.parquet"

    @property
    def s10_delta_summary(self) -> Path:
        return self.artifacts_dir / "tables" / "e04_communication_ablations_delta_summary.csv"

    @property
    def s11_results(self) -> Path:
        return self.artifacts_dir / "results" / "e04_intelligence_like_competencies.parquet"

    @property
    def s11_delta_summary(self) -> Path:
        return self.artifacts_dir / "tables" / "e04_intelligence_like_competencies_delta_summary.csv"

    @property
    def s12_predictors(self) -> Path:
        return self.artifacts_dir / "results" / "e04_tissue_field_predictors.parquet"

    @property
    def s13_results(self) -> Path:
        return self.artifacts_dir / "results" / "e04_overfitting_and_transfer.parquet"

    @property
    def s13_gap_summary(self) -> Path:
        return self.artifacts_dir / "tables" / "e04_overfitting_and_transfer_gap_summary.csv"

    @property
    def s14_results(self) -> Path:
        return self.artifacts_dir / "results" / "e04_centralized_vs_local_repair.parquet"

    @property
    def s14_matched_deltas(self) -> Path:
        return self.artifacts_dir / "results" / "e04_centralized_vs_local_repair_matched_deltas.parquet"

    @property
    def required_inputs(self) -> dict[str, Path]:
        return {
            "s08_policies": self.s08_policies,
            "s09_results": self.s09_results,
            "s09_delta_summary": self.s09_delta_summary,
            "s10_results": self.s10_results,
            "s10_delta_summary": self.s10_delta_summary,
            "s11_results": self.s11_results,
            "s11_delta_summary": self.s11_delta_summary,
            "s12_predictors": self.s12_predictors,
            "s13_results": self.s13_results,
            "s13_gap_summary": self.s13_gap_summary,
            "s14_results": self.s14_results,
            "s14_matched_deltas": self.s14_matched_deltas,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def load_s15_evidence(paths: S15EvidencePaths) -> dict[str, Any]:
    """Load all S09-S14 machine-readable evidence for synthesis."""

    missing = {name: str(path) for name, path in paths.required_inputs.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing S15 input artifacts: {missing}")
    return {
        "policies": load_jsonl(paths.s08_policies),
        "s09": pd.read_parquet(paths.s09_results),
        "s09_delta": pd.read_csv(paths.s09_delta_summary),
        "s10": pd.read_parquet(paths.s10_results),
        "s10_delta": pd.read_csv(paths.s10_delta_summary),
        "s11": pd.read_parquet(paths.s11_results),
        "s11_delta": pd.read_csv(paths.s11_delta_summary),
        "s12": pd.read_parquet(paths.s12_predictors),
        "s13": pd.read_parquet(paths.s13_results),
        "s13_gap": pd.read_csv(paths.s13_gap_summary),
        "s14": pd.read_parquet(paths.s14_results),
        "s14_delta": pd.read_parquet(paths.s14_matched_deltas),
    }


def collect_s15_anchors(evidence: dict[str, Any]) -> dict[str, Any]:
    """Compute compact anchor metrics used in S15 conclusions."""

    s09_delta = evidence["s09_delta"]
    neighbor_s09 = s09_delta.loc[s09_delta["memory_ablation"] == "neighbor_memory"].iloc[0]
    non_neighbor_s09 = s09_delta.loc[s09_delta["memory_ablation"] != "neighbor_memory"]

    s10_delta = evidence["s10_delta"]
    max_comm_delta_row = s10_delta.sort_values(
        ["mean_fitness_delta_vs_memory_only", "positive_fitness_pair_fraction"],
        ascending=False,
    ).iloc[0]

    s11_delta = evidence["s11_delta"]
    neighbor_s11 = s11_delta.loc[s11_delta["policy_mode"] == "neighbor_memory"].iloc[0]
    signal_s11 = s11_delta.loc[
        s11_delta["policy_mode"].isin(
            [
                "nearest_neighbor_signaling",
                "diffusive_signaling",
                "long_range_scalar_fields",
                "noisy_diffusive_signaling",
            ]
        )
    ]

    s12 = evidence["s12"]
    s12_combined = s12.loc[s12["feature_set"] == "local_order_plus_allowed_signal_fields"]
    s12_allowed = s12.loc[s12["feature_set"] == "allowed_signal_fields"]

    s13_gap = evidence["s13_gap"]
    s13_neighbor = s13_gap.loc[s13_gap["memory_ablation"] == "neighbor_memory"].iloc[0]
    s13_no_memory = s13_gap.loc[s13_gap["memory_ablation"] == "no_memory"].iloc[0]

    s14_delta = evidence["s14_delta"]
    s14 = evidence["s14"]
    central = s14.loc[_as_bool(s14["centralized_baseline"])]
    local_s14 = s14.loc[~_as_bool(s14["centralized_baseline"])]

    return {
        "s09_neighbor_mean_delta": float(neighbor_s09["mean_fitness_delta_vs_no_memory"]),
        "s09_neighbor_positive_fraction": float(neighbor_s09["positive_fitness_pair_fraction"]),
        "s09_neighbor_final_sortedness_delta": float(
            neighbor_s09["mean_final_sortedness_delta_vs_no_memory"]
        ),
        "s09_non_neighbor_max_delta": float(non_neighbor_s09["mean_fitness_delta_vs_no_memory"].max()),
        "s09_non_neighbor_max_positive_fraction": float(
            non_neighbor_s09["positive_fitness_pair_fraction"].max()
        ),
        "s10_best_mode": str(max_comm_delta_row["communication_ablation"]),
        "s10_best_mean_delta": float(max_comm_delta_row["mean_fitness_delta_vs_memory_only"]),
        "s10_best_positive_fraction": float(max_comm_delta_row["positive_fitness_pair_fraction"]),
        "s10_max_positive_fraction": float(s10_delta["positive_fitness_pair_fraction"].max()),
        "s11_neighbor_delta_vs_open_loop": float(
            neighbor_s11["mean_competency_delta_vs_open_loop"]
        ),
        "s11_neighbor_positive_fraction_vs_open_loop": float(
            neighbor_s11["positive_competency_fraction_vs_open_loop"]
        ),
        "s11_best_signal_delta_vs_neighbor": float(
            signal_s11["mean_competency_delta_vs_neighbor_memory"].max()
        ),
        "s11_best_signal_positive_fraction_vs_neighbor": float(
            signal_s11["positive_competency_fraction_vs_neighbor_memory"].max()
        ),
        "s12_combined_mean_local_order_auc_delta": float(
            s12_combined["local_order_auc_delta"].mean()
        ),
        "s12_combined_max_local_order_auc_delta": float(
            s12_combined["local_order_auc_delta"].max()
        ),
        "s12_allowed_mean_local_order_auc_delta": float(
            s12_allowed["local_order_auc_delta"].mean()
        ),
        "s13_neighbor_mean_train_fitness": float(s13_neighbor["mean_train_reference_fitness"]),
        "s13_neighbor_mean_heldout_fitness": float(s13_neighbor["mean_heldout_fitness"]),
        "s13_neighbor_transfer_retention": float(
            s13_neighbor["mean_fitness_retention_fraction"]
        ),
        "s13_neighbor_delta_vs_no_memory": float(
            s13_neighbor["mean_fitness_delta_vs_no_memory"]
        ),
        "s13_neighbor_positive_axis_fraction": float(
            s13_neighbor["positive_axis_fraction_vs_no_memory"]
        ),
        "s13_no_memory_mean_heldout_fitness": float(s13_no_memory["mean_heldout_fitness"]),
        "s14_local_rows": int(len(local_s14)),
        "s14_central_rows": int(len(central)),
        "s14_local_mean_comparison_repair_score": float(
            local_s14["comparison_repair_score"].mean()
        ),
        "s14_central_mean_comparison_repair_score": float(
            central["comparison_repair_score"].mean()
        ),
        "s14_mean_repair_score_retention": float(
            s14_delta["comparison_repair_score_retention_local_vs_centralized"].mean()
        ),
        "s14_mean_repair_score_gap": float(
            s14_delta["delta_comparison_repair_score_centralized_minus_local"].mean()
        ),
        "s14_mean_repair_quality_gap": float(
            s14_delta["delta_repair_quality_score_centralized_minus_local"].mean()
        ),
        "s14_mean_energy_gap": float(s14_delta["delta_energy_total_centralized_minus_local"].mean()),
    }


def validate_claim_boundaries(evidence: dict[str, Any], anchors: dict[str, Any]) -> dict[str, Any]:
    """Validate S15 conclusion gates and local-only claim boundaries."""

    checks: dict[str, Any] = {}
    for step in ("s09", "s10", "s11", "s13"):
        df = evidence[step]
        checks[f"{step}_local_rows_have_no_oracle_access"] = {
            "passed": bool(
                (~_as_bool(df["uses_global_oracle"])).all()
                and (~_as_bool(df["centralized_baseline"])).all()
                and (_as_bool(df["eligible_for_local_only_claims"])).all()
            ),
            "rows": int(len(df)),
        }

    s14 = evidence["s14"]
    central = s14.loc[_as_bool(s14["centralized_baseline"])]
    local = s14.loc[~_as_bool(s14["centralized_baseline"])]
    checks["s14_centralized_rows_excluded_from_local_claims"] = {
        "passed": bool(
            len(central) > 0
            and (_as_bool(central["uses_global_oracle"])).all()
            and (~_as_bool(central["eligible_for_local_only_claims"])).all()
            and (~_as_bool(local["uses_global_oracle"])).all()
            and (_as_bool(local["eligible_for_local_only_claims"])).all()
        ),
        "centralized_rows": int(len(central)),
        "local_rows": int(len(local)),
    }
    checks["s09_supports_neighbor_memory_minimality"] = {
        "passed": bool(
            anchors["s09_neighbor_mean_delta"] >= 0.05
            and anchors["s09_neighbor_positive_fraction"] >= 0.75
            and anchors["s09_non_neighbor_max_delta"] < 0.01
            and anchors["s09_non_neighbor_max_positive_fraction"] < 0.5
        ),
        "neighbor_delta": anchors["s09_neighbor_mean_delta"],
        "neighbor_positive_fraction": anchors["s09_neighbor_positive_fraction"],
    }
    checks["s10_does_not_support_required_communication"] = {
        "passed": bool(
            anchors["s10_best_mean_delta"] < 0.01 and anchors["s10_max_positive_fraction"] < 0.5
        ),
        "best_mode": anchors["s10_best_mode"],
        "best_mean_delta": anchors["s10_best_mean_delta"],
        "max_positive_fraction": anchors["s10_max_positive_fraction"],
    }
    checks["s11_constrains_broad_competency_claims"] = {
        "passed": bool(
            anchors["s11_neighbor_delta_vs_open_loop"] < 0.0
            and anchors["s11_neighbor_positive_fraction_vs_open_loop"] < 0.5
        ),
        "neighbor_delta_vs_open_loop": anchors["s11_neighbor_delta_vs_open_loop"],
        "positive_fraction": anchors["s11_neighbor_positive_fraction_vs_open_loop"],
    }
    checks["s12_fields_are_not_minimal_mechanism_evidence"] = {
        "passed": bool(
            anchors["s12_combined_mean_local_order_auc_delta"] < 0.03
            and anchors["s12_allowed_mean_local_order_auc_delta"] < 0.0
        ),
        "combined_mean_delta": anchors["s12_combined_mean_local_order_auc_delta"],
        "allowed_signal_only_mean_delta": anchors["s12_allowed_mean_local_order_auc_delta"],
    }
    checks["s13_supports_heldout_neighbor_memory_transfer"] = {
        "passed": bool(
            anchors["s13_neighbor_delta_vs_no_memory"] > 0.05
            and anchors["s13_neighbor_positive_axis_fraction"] >= 1.0
            and anchors["s13_neighbor_transfer_retention"] >= 0.85
        ),
        "heldout_delta": anchors["s13_neighbor_delta_vs_no_memory"],
        "transfer_retention": anchors["s13_neighbor_transfer_retention"],
    }
    checks["s14_used_only_as_nonlocal_ceiling"] = {
        "passed": bool(
            anchors["s14_mean_repair_score_retention"] >= 0.85
            and anchors["s14_central_rows"] > 0
        ),
        "local_retention": anchors["s14_mean_repair_score_retention"],
        "central_rows": anchors["s14_central_rows"],
    }
    checks["all_passed"] = all(
        bool(item.get("passed")) for item in checks.values() if isinstance(item, dict)
    )
    return checks


def build_minimal_mechanism_table(anchors: dict[str, Any]) -> pd.DataFrame:
    """Build the S15 minimal mechanism decision table."""

    rows = [
        {
            "mechanism_id": "no_memory_selected_s08_policy",
            "mechanism_label": "Selected S08 adjacent-action policy without memory",
            "included_in_minimal_package": False,
            "local_only_claim_allowed": True,
            "evidence_status": "insufficient",
            "primary_support": "S08 selected policies were local-only, but S09/S13 show no-memory variants underperform neighbor memory.",
            "primary_metric": "S13 heldout fitness delta versus no-memory baseline",
            "primary_value": 0.0,
            "chief_claim": "Use as a local baseline, not as the minimal robust mechanism.",
        },
        {
            "mechanism_id": "one_bit_or_bounded_counter_memory",
            "mechanism_label": "One-bit or bounded-counter memory",
            "included_in_minimal_package": False,
            "local_only_claim_allowed": True,
            "evidence_status": "null_or_negative",
            "primary_support": (
                f"S09 non-neighbor memory max mean fitness delta was "
                f"{anchors['s09_non_neighbor_max_delta']:.6f} with positive-pair fraction "
                f"{anchors['s09_non_neighbor_max_positive_fraction']:.2f}."
            ),
            "primary_metric": "S09 non-neighbor memory max delta",
            "primary_value": anchors["s09_non_neighbor_max_delta"],
            "chief_claim": "Not enough capacity or wrong state to support repair robustness in this matrix.",
        },
        {
            "mechanism_id": MINIMAL_MECHANISM_ID,
            "mechanism_label": "Bounded neighbor-identity memory, no required signal channel",
            "included_in_minimal_package": True,
            "local_only_claim_allowed": True,
            "evidence_status": "supportive_with_constraints",
            "primary_support": (
                f"S09 neighbor memory mean fitness delta {anchors['s09_neighbor_mean_delta']:.6f} "
                f"with {anchors['s09_neighbor_positive_fraction']:.2f} positive matched-pair fraction; "
                f"S13 heldout delta {anchors['s13_neighbor_delta_vs_no_memory']:.6f} with "
                f"{anchors['s13_neighbor_positive_axis_fraction']:.2f} positive axis fraction."
            ),
            "primary_metric": "S09/S13 matched local deltas",
            "primary_value": anchors["s13_neighbor_delta_vs_no_memory"],
            "chief_claim": (
                "Minimal supported local mechanism for the two selected S08 adjacent-action policies: "
                "bounded neighbor memory improves repair/transfer over no-memory controls."
            ),
        },
        {
            "mechanism_id": "neighbor_memory_plus_communication",
            "mechanism_label": "Neighbor memory plus tested signaling modes",
            "included_in_minimal_package": False,
            "local_only_claim_allowed": True,
            "evidence_status": "not_necessary",
            "primary_support": (
                f"S10 best communication mode was {anchors['s10_best_mode']} with mean fitness delta "
                f"{anchors['s10_best_mean_delta']:.6f} and max positive-pair fraction "
                f"{anchors['s10_max_positive_fraction']:.2f}; S11 best signal delta versus neighbor "
                f"memory was {anchors['s11_best_signal_delta_vs_neighbor']:.6f}."
            ),
            "primary_metric": "S10 best signal delta versus memory-only",
            "primary_value": anchors["s10_best_mean_delta"],
            "chief_claim": "Do not claim tested communication channels are necessary for the minimal package.",
        },
        {
            "mechanism_id": "aggregate_allowed_signal_fields",
            "mechanism_label": "Allowed blocked/frustrated aggregate fields",
            "included_in_minimal_package": False,
            "local_only_claim_allowed": True,
            "evidence_status": "null_predictive_proxy",
            "primary_support": (
                f"S12 combined allowed-field mean AUROC gain over local-order baselines was "
                f"{anchors['s12_combined_mean_local_order_auc_delta']:.6f}; signal-only mean gain was "
                f"{anchors['s12_allowed_mean_local_order_auc_delta']:.6f}."
            ),
            "primary_metric": "S12 mean AUROC gain over local-order baseline",
            "primary_value": anchors["s12_combined_mean_local_order_auc_delta"],
            "chief_claim": "Do not use aggregate field predictors as evidence for a required tissue field.",
        },
        {
            "mechanism_id": CENTRALIZED_MECHANISM_ID,
            "mechanism_label": "Centralized global-state repair controller",
            "included_in_minimal_package": False,
            "local_only_claim_allowed": False,
            "evidence_status": "nonlocal_ceiling_control",
            "primary_support": (
                f"S14 local policies retained {anchors['s14_mean_repair_score_retention']:.6f} of the "
                f"unpenalized centralized comparison repair score; centralized rows use global state, "
                "target order, and global reordering."
            ),
            "primary_metric": "S14 local-vs-central comparison repair retention",
            "primary_value": anchors["s14_mean_repair_score_retention"],
            "chief_claim": "Use only as a ceiling/control; exclude from local-only mechanism claims.",
        },
    ]
    return pd.DataFrame(rows)


def build_repair_capable_algotypes(evidence: dict[str, Any], anchors: dict[str, Any]) -> list[dict[str, Any]]:
    """Build repair-capable Algotypes records from selected S08 policies."""

    policies = evidence["policies"]
    s09 = evidence["s09"]
    s13 = evidence["s13"]
    s14 = evidence["s14"]
    records: list[dict[str, Any]] = []
    for policy in policies:
        policy_id = policy["policy_id"]
        s09_policy = s09.loc[s09["policy_id"] == policy_id]
        s13_policy = s13.loc[s13["policy_id"] == policy_id]
        s14_policy = s14.loc[
            (s14["policy_id"] == policy_id) & (~_as_bool(s14["centralized_baseline"]))
        ]

        s09_neighbor = s09_policy.loc[s09_policy["memory_ablation"] == "neighbor_memory"]
        s09_no_memory = s09_policy.loc[s09_policy["memory_ablation"] == "no_memory"]
        s13_neighbor_heldout = s13_policy.loc[
            (s13_policy["memory_ablation"] == "neighbor_memory")
            & (s13_policy["split"] == "heldout")
        ]
        s13_no_memory_heldout = s13_policy.loc[
            (s13_policy["memory_ablation"] == "no_memory")
            & (s13_policy["split"] == "heldout")
        ]
        records.append(
            {
                "policy_id": policy_id,
                "policy_family": policy["policy_family"],
                "mechanism_id": MINIMAL_MECHANISM_ID,
                "repair_capable_status": "supported_with_constraints",
                "eligible_for_local_only_claims": True,
                "centralized_or_global_rows_used_for_local_claim": False,
                "policy_input_contract": policy["policy_input_contract"],
                "protocol_id": policy["protocol_id"],
                "allowed_training_signal_fields": policy["allowed_training_signal_fields"],
                "excluded_training_signal_fields": policy["excluded_training_signal_fields"],
                "minimal_memory": "bounded_neighbor_identity_memory",
                "required_signal_mode": "none_detected; S10 communication ablations did not pass reliability criterion",
                "s08_train_mean_fitness": policy["train_mean_fitness"],
                "s08_validation_mean_fitness": policy["validation_mean_fitness"],
                "s08_heldout_mean_fitness": policy["heldout_mean_fitness"],
                "s09_neighbor_memory_mean_fitness": float(s09_neighbor["fitness_score"].mean()),
                "s09_no_memory_mean_fitness": float(s09_no_memory["fitness_score"].mean()),
                "s09_neighbor_minus_no_memory_mean_fitness": float(
                    s09_neighbor["fitness_score"].mean() - s09_no_memory["fitness_score"].mean()
                ),
                "s13_neighbor_heldout_mean_fitness": float(
                    s13_neighbor_heldout["fitness_score"].mean()
                ),
                "s13_no_memory_heldout_mean_fitness": float(
                    s13_no_memory_heldout["fitness_score"].mean()
                ),
                "s13_neighbor_minus_no_memory_heldout_fitness": float(
                    s13_neighbor_heldout["fitness_score"].mean()
                    - s13_no_memory_heldout["fitness_score"].mean()
                ),
                "s14_local_comparison_rows": int(len(s14_policy)),
                "s14_local_mean_comparison_repair_score": (
                    None
                    if s14_policy.empty
                    else float(s14_policy["comparison_repair_score"].mean())
                ),
                "s14_global_ceiling_used_as_local_evidence": False,
                "parameters": policy["parameters"],
                "claim_scope": (
                    "Two selected Bubble-style adjacent-action policies in compact 1D E04 repair, "
                    "homeostasis, competency, and transfer benchmarks."
                ),
                "limitations": [
                    "Does not establish broad superiority over original Bubble open-loop behavior in S11.",
                    "Does not support tested communication channels as necessary.",
                    "Does not use centralized/global S14 rows as local-policy evidence.",
                ],
                "source_artifacts": [
                    "/artifacts/policies/e04_evolved_repair_policies.jsonl",
                    "/artifacts/results/e04_memory_ablations.parquet",
                    "/artifacts/results/e04_overfitting_and_transfer.parquet",
                    "/artifacts/results/e04_centralized_vs_local_repair.parquet",
                ],
                "synthesis_anchor_metrics": anchors,
            }
        )
    return records


def artifact_records(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    ]
