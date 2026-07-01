"""Quality-diversity search utilities for E03 S08.

S08 uses a small MAP-Elites style archive over the S07 competence proxy.  The
archive is intentionally descriptor-driven: cells are defined by held-out
sortedness, held-out work, and S06 route so the search can expose policies that
land outside the classic DSL landmarks without claiming full simulator parity.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.e03.coarse_sweep import PolicyRecord, parse_fallback_reasons
from src.e03.gpu_batch_simulator import compatibility_record_for_policy
from src.e03.policy_generation import (
    CandidatePolicy,
    policy_feature_flags,
    policy_with_name,
    recombine_policies,
    semantic_hash,
    tiny_execution_signature,
    mutate_policy,
    parse_round_trip_policy,
)
from src.e03.rule_dsl import DSLPolicy, stable_json


QD_SCHEMA = "eidosoma.e03.quality_diversity.v1"
DEFAULT_QD_SEED = 2026070108
DEFAULT_CANDIDATE_COUNT = 384
DEFAULT_GENERATIONS = 4

SORTEDNESS_BIN_EDGES = (-math.inf, 0.35, 0.50, 0.65, 0.80, 0.90, math.inf)
WORK_BIN_EDGES = (-math.inf, 1.0, 5.0, 15.0, 35.0, 75.0, math.inf)
DESCRIPTOR_COLUMNS = (
    "screen_heldout_final_sortedness_mean",
    "screen_heldout_improvement_mean",
    "screen_heldout_work_mean",
)


@dataclass(frozen=True)
class QDCandidate:
    """A generated and validated S08 policy candidate."""

    policy: DSLPolicy
    semantic_hash: str
    source_kind: str
    generator_index: int
    generator_seed: int
    generation: int
    parent_policy_ids: tuple[str, ...]
    mutation_operator: str | None
    features: Mapping[str, Any]
    tiny_execution_signature: str
    tiny_execution_action_counts: Mapping[str, int]
    compiles_for_batch: bool
    requires_cpu_fallback: bool
    fallback_reasons: tuple[str, ...]
    route: str

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    def to_policy_record(self) -> PolicyRecord:
        return PolicyRecord(
            policy=self.policy,
            policy_id=self.policy.policy_id,
            policy_name=self.policy.name,
            source_kind=self.source_kind,
            semantic_hash=self.semantic_hash,
            dsl_sha256=self.policy.sha256,
            features=dict(self.features),
            compiles_for_batch=bool(self.compiles_for_batch),
            requires_cpu_fallback=bool(self.requires_cpu_fallback),
            fallback_reasons=tuple(self.fallback_reasons),
            route=self.route,
        )

    def lineage_record(self, parent_depths: Mapping[str, int]) -> dict[str, Any]:
        parent_depth = max((int(parent_depths.get(parent, 0)) for parent in self.parent_policy_ids), default=0)
        return {
            "schema": QD_SCHEMA,
            "experiment_id": "E03",
            "research_step_id": "S08",
            "child_policy_id": self.policy.policy_id,
            "child_policy_name": self.policy.name,
            "semantic_hash": self.semantic_hash,
            "dsl_sha256": self.policy.sha256,
            "source_kind": self.source_kind,
            "generation": int(self.generation),
            "lineage_depth": int(parent_depth + 1),
            "generator_index": int(self.generator_index),
            "generator_seed": int(self.generator_seed),
            "parent_policy_ids_json": json.dumps(list(self.parent_policy_ids), separators=(",", ":")),
            "mutation_operator": self.mutation_operator,
            "route": self.route,
            "compiles_for_batch": bool(self.compiles_for_batch),
            "requires_cpu_fallback": bool(self.requires_cpu_fallback),
            "fallback_reasons_json": json.dumps(list(self.fallback_reasons), separators=(",", ":")),
            "tiny_execution_signature": self.tiny_execution_signature,
            "tiny_execution_action_counts_json": json.dumps(dict(self.tiny_execution_action_counts), sort_keys=True),
            "dsl_source": self.policy.to_source(),
        }


def descriptor_bin(value: float, edges: Sequence[float]) -> int:
    """Return a stable zero-based bin index for one numeric descriptor."""

    if pd.isna(value):
        return -1
    numeric = float(value)
    for index in range(len(edges) - 1):
        if edges[index] <= numeric < edges[index + 1]:
            return index
    return len(edges) - 2


def descriptor_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "final_sortedness": float(row["screen_heldout_final_sortedness_mean"]),
        "improvement": float(row["screen_heldout_improvement_mean"]),
        "work": float(row["screen_heldout_work_mean"]),
        "sortedness_bin": int(row["sortedness_bin"]),
        "work_bin": int(row["work_bin"]),
        "route": str(row["route"]),
    }


def add_descriptor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add S04/S07-derived MAP-Elites descriptors and fixed grid cells."""

    required = {"policy_id", "route", "screen_score", *DESCRIPTOR_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing descriptor input columns: {missing}")
    out = frame.copy()
    out["quality_score"] = pd.to_numeric(out["screen_score"], errors="coerce")
    out["sortedness_bin"] = [
        descriptor_bin(value, SORTEDNESS_BIN_EDGES)
        for value in pd.to_numeric(out["screen_heldout_final_sortedness_mean"], errors="coerce")
    ]
    out["work_bin"] = [
        descriptor_bin(value, WORK_BIN_EDGES)
        for value in pd.to_numeric(out["screen_heldout_work_mean"], errors="coerce")
    ]
    out["map_cell_id"] = [
        f"s{int(sbin):02d}_w{int(wbin):02d}_r{route}"
        for sbin, wbin, route in zip(out["sortedness_bin"], out["work_bin"], out["route"], strict=True)
    ]
    out["descriptor_vector_json"] = [
        json.dumps(
            {
                "final_sortedness": float(final),
                "improvement": float(improvement),
                "work": float(work),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for final, improvement, work in zip(
            out["screen_heldout_final_sortedness_mean"],
            out["screen_heldout_improvement_mean"],
            out["screen_heldout_work_mean"],
            strict=True,
        )
    ]
    return out


def descriptor_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return normalized descriptor vectors for novelty calculations."""

    final = pd.to_numeric(frame["screen_heldout_final_sortedness_mean"], errors="coerce").fillna(0.0).to_numpy(float)
    improvement = pd.to_numeric(frame["screen_heldout_improvement_mean"], errors="coerce").fillna(0.0).to_numpy(float)
    work = pd.to_numeric(frame["screen_heldout_work_mean"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    work_norm = np.log1p(work) / np.log1p(100.0)
    return np.stack([final, improvement, work_norm], axis=1)


def nearest_descriptor_distance(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    """Compute Euclidean distance to the nearest reference descriptor."""

    if frame.empty:
        return np.asarray([], dtype=float)
    if reference.empty:
        return np.full(len(frame), np.nan, dtype=float)
    query = descriptor_matrix(frame)
    ref = descriptor_matrix(reference)
    distances = np.sqrt(((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def add_novelty_columns(frame: pd.DataFrame, *, s07_reference: pd.DataFrame, classic_reference: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["novelty_to_s07"] = nearest_descriptor_distance(out, s07_reference)
    out["novelty_to_classics"] = nearest_descriptor_distance(out, classic_reference)
    out["novelty_score"] = out["novelty_to_classics"]
    return out


def archive_digest(archive: pd.DataFrame) -> str:
    """Return a stable digest of archive cell winners."""

    cols = ["map_cell_id", "policy_id", "quality_score", "novelty_score", "source_kind", "generation"]
    records = archive[cols].sort_values("map_cell_id").to_dict(orient="records")
    rounded: list[dict[str, Any]] = []
    for record in records:
        rounded.append(
            {
                key: (round(float(value), 12) if isinstance(value, float) and not pd.isna(value) else value)
                for key, value in record.items()
            }
        )
    return hashlib.sha256(stable_json(rounded).encode("utf-8")).hexdigest()


def build_archive(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the best policy per MAP-Elites cell."""

    if frame.empty:
        return frame.copy()
    required = {"map_cell_id", "policy_id", "quality_score", "novelty_score", "generation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing archive columns: {missing}")
    sortable = frame.copy()
    sortable["quality_score"] = pd.to_numeric(sortable["quality_score"], errors="coerce")
    sortable["novelty_score"] = pd.to_numeric(sortable["novelty_score"], errors="coerce")
    sortable = sortable.sort_values(
        ["map_cell_id", "quality_score", "novelty_score", "generation", "policy_id"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    archive = sortable.drop_duplicates("map_cell_id", keep="first").copy()
    archive = archive.sort_values(["sortedness_bin", "work_bin", "route"], kind="mergesort").reset_index(drop=True)
    archive.insert(0, "archive_rank", np.arange(1, len(archive) + 1, dtype=int))
    archive["archive_digest"] = archive_digest(archive)
    return archive


def policy_record_frame(records: Sequence[PolicyRecord]) -> pd.DataFrame:
    """Convert policy records to a compact source/feature table."""

    return pd.DataFrame(
        [
            {
                "policy_id": record.policy_id,
                "semantic_hash": record.semantic_hash,
                "dsl_sha256": record.dsl_sha256,
                "dsl_source": record.policy.to_source(),
                "features_json": json.dumps(dict(record.features), sort_keys=True),
                "fallback_reasons_json": json.dumps(list(record.fallback_reasons), separators=(",", ":")),
            }
            for record in records
        ]
    )


def seed_archive_frame(summary: pd.DataFrame, records: Sequence[PolicyRecord]) -> pd.DataFrame:
    """Join S07 competence rows to S05 DSL sources for archive seeding."""

    record_frame = policy_record_frame(records)
    merged = summary.merge(record_frame, on="policy_id", how="left", validate="one_to_one")
    if merged["dsl_source"].isna().any():
        missing = merged.loc[merged["dsl_source"].isna(), "policy_id"].head(10).tolist()
        raise ValueError(f"S07 competence rows missing S05 DSL sources: {missing}")
    merged["schema"] = QD_SCHEMA
    merged["experiment_id"] = "E03"
    merged["research_step_id"] = "S08"
    merged["generation"] = 0
    merged["lineage_depth"] = 0
    merged["generator_index"] = -1
    merged["generator_seed"] = DEFAULT_QD_SEED
    merged["parent_policy_ids_json"] = "[]"
    merged["mutation_operator"] = None
    merged["qd_candidate"] = False
    merged["evaluation_source"] = "s07_coarse_competence"
    merged["archive_winner"] = False
    merged["classic_cell"] = False
    return add_descriptor_columns(merged)


def selected_parent_ids(seed_frame: pd.DataFrame, archive: pd.DataFrame, *, pool_size: int = 160) -> list[str]:
    """Select deterministic parents from top scores, classics, and archive cells."""

    selected: list[str] = []

    def add(ids: Iterable[str]) -> None:
        for policy_id in ids:
            if policy_id not in selected:
                selected.append(str(policy_id))

    ranked = seed_frame.sort_values(["quality_score", "novelty_score"], ascending=False, kind="mergesort")
    add(seed_frame.loc[seed_frame.get("classic_dsl_seed", False) == True, "policy_id"].tolist())  # noqa: E712
    add(archive["policy_id"].tolist())
    add(ranked["policy_id"].head(pool_size).tolist())
    for route in sorted(seed_frame["route"].dropna().unique()):
        add(ranked.loc[ranked["route"] == route, "policy_id"].head(24).tolist())
    for source_kind in sorted(seed_frame["source_kind"].dropna().unique()):
        add(ranked.loc[ranked["source_kind"] == source_kind, "policy_id"].head(12).tolist())
    return selected[:pool_size]


def _accepted_qd_candidate(
    candidate: CandidatePolicy,
    *,
    generation: int,
    seen_semantics: dict[str, str],
) -> QDCandidate | None:
    parsed = parse_round_trip_policy(candidate.source)
    sem_hash = semantic_hash(parsed)
    if sem_hash in seen_semantics:
        return None
    canonical = policy_with_name(parsed, f"qd_s08_{sem_hash[:16]}")
    canonical = parse_round_trip_policy(canonical.to_source())
    sem_hash = semantic_hash(canonical)
    if sem_hash in seen_semantics:
        return None
    signature, action_counts = tiny_execution_signature(canonical, candidate.generator_seed + generation)
    features = policy_feature_flags(canonical)
    compatibility = compatibility_record_for_policy(canonical)
    compiles_for_batch = bool(compatibility.get("compiles_for_batch", False))
    requires_cpu_fallback = bool(compatibility.get("requires_cpu_fallback", True))
    route = "jax_batch" if compiles_for_batch and not requires_cpu_fallback else "cpu_fallback"
    seen_semantics[sem_hash] = canonical.policy_id
    source_kind = "qd_mutation" if candidate.source_kind == "mutation" else "qd_recombination"
    return QDCandidate(
        policy=canonical,
        semantic_hash=sem_hash,
        source_kind=source_kind,
        generator_index=candidate.generator_index,
        generator_seed=candidate.generator_seed,
        generation=generation,
        parent_policy_ids=tuple(candidate.parent_policy_ids),
        mutation_operator=candidate.mutation_operator,
        features=features,
        tiny_execution_signature=signature,
        tiny_execution_action_counts=action_counts,
        compiles_for_batch=compiles_for_batch,
        requires_cpu_fallback=requires_cpu_fallback,
        fallback_reasons=parse_fallback_reasons(compatibility.get("fallback_reasons_json")),
        route=route,
    )


def generate_qd_candidates(
    *,
    parent_records: Sequence[PolicyRecord],
    seen_semantics: dict[str, str],
    target_count: int,
    generation: int,
    seed: int,
    start_index: int = 0,
    max_attempt_multiplier: int = 20,
) -> list[QDCandidate]:
    """Generate valid, semantically unique MAP-Elites candidates."""

    if not parent_records:
        raise ValueError("parent_records must not be empty")
    if target_count < 1:
        return []
    rng = random.Random(seed + 1009 * generation + start_index)
    accepted: list[QDCandidate] = []
    attempts = 0
    max_attempts = max(target_count * max_attempt_multiplier, target_count)
    while len(accepted) < target_count and attempts < max_attempts:
        index = start_index + attempts
        try:
            if len(parent_records) >= 2 and rng.random() < 0.30:
                left, right = rng.sample(list(parent_records), 2)
                raw = recombine_policies(left.policy, right.policy, rng, index, seed)
            else:
                parent = rng.choice(list(parent_records))
                raw = mutate_policy(parent.policy, rng, index, seed)
            qd = _accepted_qd_candidate(raw, generation=generation, seen_semantics=seen_semantics)
            if qd is not None:
                accepted.append(qd)
        except Exception:
            pass
        attempts += 1
    return accepted


def candidate_metadata_frame(candidates: Sequence[QDCandidate], parent_depths: Mapping[str, int]) -> pd.DataFrame:
    return pd.DataFrame([candidate.lineage_record(parent_depths) for candidate in candidates])


def attach_candidate_metadata(summary: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join policy-level evaluation summaries to QD lineage/source metadata."""

    if summary.empty:
        return summary.copy()
    meta = metadata.rename(columns={"child_policy_id": "policy_id", "child_policy_name": "policy_name_meta"})
    merged = summary.merge(meta, on="policy_id", how="left", suffixes=("", "_lineage"), validate="one_to_one")
    merged["schema"] = QD_SCHEMA
    merged["experiment_id"] = "E03"
    merged["research_step_id"] = "S08"
    merged["qd_candidate"] = True
    merged["evaluation_source"] = "s08_qd_screen"
    merged["archive_winner"] = False
    merged["classic_dsl_seed"] = False
    if "policy_name_meta" in merged.columns:
        merged["policy_name"] = merged["policy_name_meta"].fillna(merged["policy_name"])
        merged = merged.drop(columns=["policy_name_meta"])
    for column in ("semantic_hash", "dsl_sha256", "parent_policy_ids_json", "mutation_operator", "route"):
        lineage_col = f"{column}_lineage"
        if lineage_col in merged.columns:
            merged[column] = merged.get(column, pd.Series(index=merged.index, dtype=object)).fillna(merged[lineage_col])
            merged = merged.drop(columns=[lineage_col])
    return add_descriptor_columns(merged)


def choose_discovered_policies(
    all_rows: pd.DataFrame,
    archive: pd.DataFrame,
    classic_cell_ids: set[str],
    *,
    limit: int = 64,
) -> pd.DataFrame:
    """Pick QD-generated policies to publish as the discovered set."""

    qd = all_rows[all_rows["qd_candidate"] == True].copy()  # noqa: E712
    if qd.empty:
        return qd
    archive_ids = set(archive.loc[archive["qd_candidate"] == True, "policy_id"])  # noqa: E712
    qd["archive_winner"] = qd["policy_id"].isin(archive_ids)
    qd["nonclassic_cell"] = ~qd["map_cell_id"].isin(classic_cell_ids)
    qd = qd.sort_values(
        ["archive_winner", "nonclassic_cell", "quality_score", "novelty_to_classics", "policy_id"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    )
    return qd.head(limit).reset_index(drop=True)


def validation_frame(
    *,
    archive: pd.DataFrame,
    discovered: pd.DataFrame,
    lineage: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    candidate_count_target: int,
    known_policy_ids: set[str],
    recomputed_archive: pd.DataFrame,
    figure_exists: bool,
) -> pd.DataFrame:
    """Return S08 validation cases."""

    cases: list[dict[str, Any]] = []

    def add(name: str, success: bool, expected: str, observed: Any, notes: str) -> None:
        cases.append(
            {
                "validation_case": name,
                "success": bool(success),
                "expected": str(expected),
                "observed": str(observed),
                "notes": notes,
            }
        )

    parent_ids: set[str] = set()
    if not lineage.empty:
        for text in lineage["parent_policy_ids_json"].fillna("[]"):
            parent_ids.update(str(item) for item in json.loads(str(text)))

    add("candidate_count_met", len(candidate_summary) >= candidate_count_target, f">= {candidate_count_target}", len(candidate_summary), "Validated QD candidates reached the requested search budget.")
    add("candidate_policy_ids_unique", candidate_summary["policy_id"].nunique() == len(candidate_summary), "unique candidate policy IDs", candidate_summary["policy_id"].nunique(), "Semantic duplicate filtering should leave one row per QD policy.")
    add("candidate_hashes_unique", candidate_summary["semantic_hash"].nunique() == len(candidate_summary), "unique semantic hashes", candidate_summary["semantic_hash"].nunique(), "Policy names do not define uniqueness.")
    add("archive_nonempty", len(archive) > 0, "> 0 occupied cells", len(archive), "MAP-Elites archive contains cell elites.")
    add("archive_cell_unique", archive["map_cell_id"].nunique() == len(archive), "one winner per cell", archive["map_cell_id"].nunique(), "Archive selection must produce one elite per descriptor cell.")
    add("archive_reproducible", archive_digest(archive) == archive_digest(recomputed_archive), "stable archive digest", archive_digest(recomputed_archive), "Rebuilding the archive from combined rows gives identical winners.")
    add("lineage_parent_ids_valid", parent_ids.issubset(known_policy_ids), "all parents exist in baseline or QD policy set", sorted(parent_ids - known_policy_ids)[:10], "Lineage parent references should resolve.")
    add("discovered_policy_set_nonempty", len(discovered) > 0, "> 0 discovered policies", len(discovered), "The publishable QD candidate set is non-empty.")
    add("nonclassic_niche_candidate_present", bool((discovered["nonclassic_cell"] == True).any()) if not discovered.empty else False, "at least one non-classic cell candidate", int(discovered["nonclassic_cell"].sum()) if not discovered.empty else 0, "Directly addresses the frozen niche question against classic DSL seed cells.")
    add("archive_has_qd_candidate", bool((archive["qd_candidate"] == True).any()) if not archive.empty else False, "at least one QD candidate in archive", int(archive["qd_candidate"].sum()) if not archive.empty else 0, "MAP-Elites search should add or replace at least one cell elite.")
    finite = bool(np.isfinite(pd.to_numeric(archive["quality_score"], errors="coerce")).all()) if not archive.empty else False
    add("archive_quality_finite", finite, "all archive quality scores finite", int(np.isfinite(pd.to_numeric(archive["quality_score"], errors="coerce")).sum()) if not archive.empty else 0, "Archive cells have evaluable best-policy scores.")
    add("figure_written", figure_exists, "archive figure exists and is non-empty", figure_exists, "S08 overview figure artifact.")
    return pd.DataFrame(cases)
