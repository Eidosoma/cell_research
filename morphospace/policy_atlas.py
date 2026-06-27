"""Policy atlas helpers for E03 S15.

The atlas is a static, dependency-light packaging layer over S01-S14 evidence.
It keeps a complete machine-readable policy catalog separate from the HTML
view, then validates that policy IDs and artifact links resolve.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .competence import canonical_json


POLICY_ATLAS_VERSION = "e03_s15_policy_atlas.v1"
ATLAS_REQUIRED_ROLES = ("classic", "null", "qd_elite", "frontier", "pathological")
ATLAS_SCORE_COLUMNS = (
    "s13CompositeScore",
    "completionScore",
    "sortednessScore",
    "energyScore",
    "robustnessScore",
    "delayedGratificationScore",
    "aggregationScore",
    "oscillationStabilityScore",
    "holdoutCompositeScore",
)


def safe_json(value: Any, default: Any = None) -> Any:
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return default


def _numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _bool(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].map(lambda value: bool(value) if pd.notna(value) else False)


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value)
    return text if text != "nan" else default


def _clean_anchor(policy_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(policy_id))
    return f"policy-{safe}"


def _path_to_href(path: str | Path, *, base_path: str | Path) -> str:
    target = Path(path)
    base = Path(base_path)
    try:
        return Path("../" + str(target.relative_to(base.parent.parent))).as_posix() if target.is_absolute() else target.as_posix()
    except ValueError:
        try:
            return target.relative_to(base.parent).as_posix()
        except ValueError:
            return target.as_posix()


def _merge_first(left: pd.DataFrame, right: pd.DataFrame, *, on: str, columns: Sequence[str]) -> pd.DataFrame:
    out = left.merge(right[[on, *[column for column in columns if column in right.columns]]], on=on, how="left", suffixes=("", "_new"))
    for column in columns:
        new_col = f"{column}_new"
        if new_col in out.columns:
            if column in out.columns:
                out[column] = out[column].combine_first(out[new_col])
            else:
                out[column] = out[new_col]
            out = out.drop(columns=[new_col])
    return out


def embedding_seed0_wide(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    if embeddings_df.empty:
        return pd.DataFrame(columns=["policyId"])
    df = embeddings_df[embeddings_df["embeddingSeed"].eq(0)].copy()
    methods = ["pca", "spectral", "mds"]
    pieces: list[pd.DataFrame] = []
    for method in methods:
        group = df[df["embeddingMethod"].eq(method)].copy()
        if group.empty:
            continue
        columns = ["policyId", "embeddingDim1", "embeddingDim2", "embeddingDim3", "embeddingDim4", "embeddingDim5"]
        available = [column for column in columns if column in group.columns]
        renamed = group[available].drop_duplicates("policyId").rename(
            columns={column: f"{method}_{column}" for column in available if column != "policyId"}
        )
        pieces.append(renamed)
    if not pieces:
        return pd.DataFrame(columns=["policyId"])
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="policyId", how="outer")
    return out


def nearest_neighbor_json(neighbors_df: pd.DataFrame, *, max_neighbors: int = 5) -> pd.DataFrame:
    if neighbors_df.empty:
        return pd.DataFrame(columns=["policyId", "nearestNeighborsJson", "nearestNeighborCount"])
    rows: list[dict[str, Any]] = []
    df = neighbors_df.copy()
    df["policyId"] = df["policyId"].astype(str)
    df["neighborPolicyId"] = df["neighborPolicyId"].astype(str)
    df["neighborRank"] = pd.to_numeric(df["neighborRank"], errors="coerce").fillna(9999).astype(int)
    df["featureDistance"] = pd.to_numeric(df.get("featureDistance", np.nan), errors="coerce")
    for policy_id, group in df.sort_values(["policyId", "neighborRank"], kind="mergesort").groupby("policyId", sort=True):
        payload = []
        for _, row in group.head(max_neighbors).iterrows():
            payload.append(
                {
                    "neighborPolicyId": str(row["neighborPolicyId"]),
                    "neighborRank": int(row["neighborRank"]),
                    "featureDistance": float(row["featureDistance"]) if pd.notna(row["featureDistance"]) else None,
                    "neighborPrimaryRole": _as_text(row.get("neighborPrimaryRole")),
                }
            )
        rows.append({"policyId": policy_id, "nearestNeighborsJson": canonical_json(payload), "nearestNeighborCount": len(payload)})
    return pd.DataFrame(rows)


def representative_trajectory_json(
    run_tables: Sequence[tuple[pd.DataFrame, str, str]],
    *,
    max_rows_per_policy: int = 5,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for run_df, source_step, artifact_path in run_tables:
        if run_df.empty:
            continue
        df = run_df.copy()
        df["policyId"] = df["policyId"].astype(str)
        df["sourceStep"] = source_step
        df["sourceArtifactPath"] = artifact_path
        pieces.append(df)
    if not pieces:
        return pd.DataFrame(columns=["policyId", "representativeTrajectoriesJson", "representativeTrajectoryCount"])
    combined = pd.concat(pieces, ignore_index=True, sort=False)
    sort_cols = [column for column in ["policyId", "sourceStep", "taskFamily", "taskId", "seedIndex"] if column in combined.columns]
    combined = combined.sort_values(sort_cols, kind="mergesort")
    rows: list[dict[str, Any]] = []
    for policy_id, group in combined.groupby("policyId", sort=True):
        payload = []
        for _, row in group.head(max_rows_per_policy).iterrows():
            payload.append(
                {
                    "sourceStep": _as_text(row.get("sourceStep")),
                    "sourceArtifactPath": _as_text(row.get("sourceArtifactPath")),
                    "taskId": _as_text(row.get("taskId")),
                    "taskFamily": _as_text(row.get("taskFamily")),
                    "seedIndex": int(row.get("seedIndex", 0)) if pd.notna(row.get("seedIndex", np.nan)) else None,
                    "schedulerSeed": int(row.get("schedulerSeed", 0)) if pd.notna(row.get("schedulerSeed", np.nan)) else None,
                    "completed": bool(row.get("completed", False)) if pd.notna(row.get("completed", np.nan)) else None,
                    "stopReason": _as_text(row.get("stopReason")),
                    "finalSortednessScore": float(row.get("finalSortednessScore")) if pd.notna(row.get("finalSortednessScore", np.nan)) else None,
                    "traceStateCount": int(row.get("traceStateCount", 0)) if pd.notna(row.get("traceStateCount", np.nan)) else None,
                }
            )
        rows.append(
            {
                "policyId": policy_id,
                "representativeTrajectoriesJson": canonical_json(payload),
                "representativeTrajectoryCount": len(payload),
            }
        )
    return pd.DataFrame(rows)


def _role_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if bool(row.get("isFrontierCandidate", False)):
        tags.append("frontier")
    if bool(row.get("isClassicPolicy", False)):
        tags.append("classic")
    if bool(row.get("isNullPolicy", False)):
        tags.append("null")
    if bool(row.get("isS08Elite", False)):
        tags.append("qd_elite")
    if bool(row.get("isPhaseBoundaryPolicy", False)):
        tags.append("phase_boundary")
    if bool(row.get("isPathologicalPolicy", False)):
        tags.append("pathological")
    if bool(row.get("isParetoOptimal", False)):
        tags.append("pareto")
    if _as_text(row.get("comparisonRole")) == "insufficient_evidence":
        tags.append("missing_evidence")
    if not tags:
        tags.append(_as_text(row.get("primaryRole"), "generated") or "generated")
    ordered = ["frontier", "classic", "null", "qd_elite", "phase_boundary", "pathological", "pareto", "missing_evidence"]
    return sorted(set(tags), key=lambda tag: ordered.index(tag) if tag in ordered else len(ordered))


def _artifact_links_for_row(row: pd.Series) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for label, path_col, sha_col in [
        ("S05 DSL", "dslPath", "dslSha256"),
        ("S14 frontier DSL", "frontierDslPath", "frontierDslSha256"),
    ]:
        path = _as_text(row.get(path_col))
        if path:
            payload = {"label": label, "path": path}
            sha = _as_text(row.get(sha_col))
            if sha:
                payload["sha256"] = sha
            links.append(payload)
    if int(row.get("representativeTrajectoryCount", 0) or 0) > 0:
        seen: set[str] = set()
        for trajectory in safe_json(row.get("representativeTrajectoriesJson"), []):
            path = _as_text(trajectory.get("sourceArtifactPath") if isinstance(trajectory, Mapping) else "")
            if path and path not in seen:
                seen.add(path)
                links.append({"label": f"{trajectory.get('sourceStep', 'run')} trajectory table", "path": path})
    if bool(row.get("isS08Elite", False)):
        links.append({"label": "S08 QD elites", "path": "/artifacts/results/e03_qd_elites.parquet"})
    if bool(row.get("isPhaseBoundaryPolicy", False)):
        links.append({"label": "S09 phase boundaries", "path": "/artifacts/results/e03_phase_boundaries.parquet"})
    if bool(row.get("isFrontierCandidate", False)):
        links.append({"label": "S14 frontier candidates", "path": "/artifacts/results/e03_frontier_candidates.parquet"})
    return links


def build_policy_atlas_catalog(
    *,
    policy_summary_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    dsl_index_df: pd.DataFrame,
    class_assignments_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    qd_elites_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    frontier_validation_summary_df: pd.DataFrame,
    frontier_file_index_df: pd.DataFrame,
    s13_missing_df: pd.DataFrame,
    s13_runs_df: pd.DataFrame,
    s14_runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a complete policy-level atlas catalog."""

    base = policy_summary_df.copy()
    base["policyId"] = base["policyId"].astype(str)

    class_cols = [
        "universalityClassId",
        "className",
        "classLabel",
        "classInterpretation",
        "empiricalCaveat",
        "distanceToCentroid",
        "primaryRole",
        "isNullPolicy",
        "isClassicPolicy",
        "isS08Elite",
        "isPhaseBoundaryPolicy",
        "isPathologicalPolicy",
        "phaseBoundaryInvolvementCount",
        "s08Elite_qualityScore",
    ]
    assignments = class_assignments_df.copy()
    assignments["policyId"] = assignments["policyId"].astype(str)
    base = _merge_first(base, assignments, on="policyId", columns=class_cols)

    corpus_cols = [
        "policyId",
        "dslProgramJson",
        "structureHash",
        "parentPolicyIdsJson",
        "mutationOperatorsJson",
        "recombinationParentsJson",
        "tagsJson",
        "observationRequirementsJson",
        "actionCountsJson",
    ]
    corpus = corpus_df[[column for column in corpus_cols if column in corpus_df.columns]].copy()
    corpus["policyId"] = corpus["policyId"].astype(str)
    base = _merge_first(base, corpus, on="policyId", columns=[column for column in corpus_cols if column != "policyId"])

    dsl = dsl_index_df.copy()
    dsl["policyId"] = dsl["policyId"].astype(str)
    dsl = dsl.rename(columns={"sha256": "dslSha256", "sizeBytes": "dslSizeBytes"})
    base = base.merge(
        dsl[[column for column in ["policyId", "dslPath", "dslRelativePath", "dslSha256", "dslSizeBytes"] if column in dsl.columns]],
        on="policyId",
        how="left",
        suffixes=("", "_dsl"),
    )
    if "dslRelativePath_dsl" in base.columns:
        base["dslRelativePath"] = base["dslRelativePath"].combine_first(base["dslRelativePath_dsl"])
        base = base.drop(columns=["dslRelativePath_dsl"])

    qd = qd_elites_df.copy()
    if not qd.empty:
        qd["policyId"] = qd["policyId"].astype(str)
        qd = qd.rename(
            columns={
                "eliteRank": "s08EliteRank",
                "qualityScore": "s08QualityScore",
                "descriptorCellId": "s08DescriptorCellId",
                "descriptorCellLabel": "s08DescriptorCellLabel",
            }
        )
        base = base.merge(
            qd[[column for column in ["policyId", "s08EliteRank", "s08QualityScore", "s08DescriptorCellId", "s08DescriptorCellLabel"] if column in qd.columns]],
            on="policyId",
            how="left",
        )

    pareto = pareto_df.copy()
    if not pareto.empty:
        pareto["policyId"] = pareto["policyId"].astype(str)
        pareto = pareto.rename(
            columns={
                "paretoFrontRank": "s13ParetoFrontRank",
                "isParetoOptimal": "isParetoOptimal",
            }
        )
        base = base.merge(
            pareto[[column for column in ["policyId", "paretoEligible", "s13ParetoFrontRank", "isParetoOptimal"] if column in pareto.columns]],
            on="policyId",
            how="left",
        )

    frontier = frontier_df.copy()
    if not frontier.empty:
        frontier["policyId"] = frontier["policyId"].astype(str)
        frontier_cols = [
            "policyId",
            "frontierRank",
            "frontierCompositeScore",
            "frontierObjectiveTagsJson",
            "frontierSelectionStage",
            "selectionRationaleJson",
            "duplicateOrNearDuplicateJustification",
        ]
        base = base.merge(frontier[[column for column in frontier_cols if column in frontier.columns]], on="policyId", how="left")

    fv = frontier_validation_summary_df.copy()
    if not fv.empty:
        fv["policyId"] = fv["policyId"].astype(str)
        fv_cols = [
            "policyId",
            "holdoutCompositeScore",
            "holdout_completionSuccessMean",
            "holdout_finalSortednessScoreMean",
            "holdout_energyScoreMean",
            "holdout_robustnessScoreMean",
            "holdout_delayedGratificationScoreMean",
            "holdout_aggregationAucScoreMean",
            "holdout_compatibilityScoreMean",
            "holdout_transferScoreMean",
            "holdout_oscillationScoreMean",
            "holdoutStopReasons",
        ]
        base = base.merge(fv[[column for column in fv_cols if column in fv.columns]], on="policyId", how="left", suffixes=("", "_holdout"))

    frontier_files = frontier_file_index_df.copy()
    if not frontier_files.empty:
        frontier_files["policyId"] = frontier_files["policyId"].astype(str)
        frontier_files = frontier_files.rename(columns={"dslPath": "frontierDslPath", "dslSha256": "frontierDslSha256"})
        base = base.merge(
            frontier_files[[column for column in ["policyId", "frontierDslPath", "frontierDslSha256"] if column in frontier_files.columns]],
            on="policyId",
            how="left",
        )

    missing = s13_missing_df.copy()
    if not missing.empty:
        missing["policyId"] = missing["policyId"].astype(str)
        base = base.merge(missing[["policyId", "missingReason"]], on="policyId", how="left")

    base = base.merge(embedding_seed0_wide(embeddings_df), on="policyId", how="left")
    base = base.merge(nearest_neighbor_json(neighbors_df), on="policyId", how="left")
    trajectories = representative_trajectory_json(
        [
            (s13_runs_df, "S13", "/artifacts/results/e03_s13_same_seed_runs.parquet"),
            (s14_runs_df, "S14", "/artifacts/results/e03_s14_frontier_validation_runs.parquet"),
        ]
    )
    base = base.merge(trajectories, on="policyId", how="left")

    base["policyAtlasVersion"] = POLICY_ATLAS_VERSION
    base["atlasPolicyAnchor"] = base["policyId"].map(_clean_anchor)
    base["hasDslProgram"] = base.get("dslProgramJson", pd.Series(index=base.index)).notna()
    base["hasDslFile"] = base.get("dslPath", pd.Series(index=base.index)).notna()
    base["isFrontierCandidate"] = base.get("frontierRank", pd.Series(index=base.index)).notna()
    for column in ["isClassicPolicy", "isNullPolicy", "isS08Elite", "isPhaseBoundaryPolicy", "isPathologicalPolicy", "isParetoOptimal"]:
        base[column] = _bool(base, column)
    base["representativeTrajectoriesJson"] = base["representativeTrajectoriesJson"].fillna("[]")
    base["representativeTrajectoryCount"] = _numeric(base, "representativeTrajectoryCount", 0.0).fillna(0).astype(int)
    base["nearestNeighborsJson"] = base["nearestNeighborsJson"].fillna("[]")
    base["nearestNeighborCount"] = _numeric(base, "nearestNeighborCount", 0.0).fillna(0).astype(int)
    base["roleTags"] = base.apply(_role_tags, axis=1)
    base["roleTagsJson"] = base["roleTags"].map(canonical_json)
    base["atlasPrimaryRole"] = base["roleTags"].map(lambda tags: tags[0] if tags else "generated")
    base["policyArtifactLinksJson"] = base.apply(lambda row: canonical_json(_artifact_links_for_row(row)), axis=1)
    base["policyArtifactLinkCount"] = base["policyArtifactLinksJson"].map(lambda value: len(safe_json(value, []))).astype(int)
    base["atlasMissingEvidenceNotes"] = base.apply(
        lambda row: "; ".join(
            note
            for note in [
                "no DSL file for non-corpus synthetic context" if not bool(row.get("hasDslFile", False)) else "",
                _as_text(row.get("missingReason")),
                "no representative S13/S14 trajectory row" if int(row.get("representativeTrajectoryCount", 0) or 0) == 0 else "",
            ]
            if note
        ),
        axis=1,
    )
    for column in ATLAS_SCORE_COLUMNS:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")
    role_order = {"frontier": 0, "classic": 1, "qd_elite": 2, "pathological": 3, "null": 4}
    base["_roleOrder"] = base["atlasPrimaryRole"].map(lambda value: role_order.get(str(value), 5))
    base = base.sort_values(["_roleOrder", "frontierRank", "s13CompositeScore", "policyId"], ascending=[True, True, False, True], kind="mergesort")
    return base.drop(columns=["_roleOrder"]).reset_index(drop=True)


def atlas_class_summary(catalog_df: pd.DataFrame) -> pd.DataFrame:
    if catalog_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for class_name, group in catalog_df.groupby("className", dropna=False, sort=True):
        role_counts = group["atlasPrimaryRole"].astype(str).value_counts().sort_index().to_dict()
        rows.append(
            {
                "policyAtlasVersion": POLICY_ATLAS_VERSION,
                "className": _as_text(class_name, "unassigned"),
                "classLabel": _as_text(group["classLabel"].dropna().iloc[0]) if group["classLabel"].notna().any() else "",
                "policyCount": int(len(group)),
                "roleCountsJson": canonical_json({str(key): int(value) for key, value in role_counts.items()}),
                "frontierCount": int(group["isFrontierCandidate"].sum()),
                "classicCount": int(group["isClassicPolicy"].sum()),
                "nullCount": int(group["isNullPolicy"].sum()),
                "eliteCount": int(group["isS08Elite"].sum()),
                "pathologicalCount": int(group["isPathologicalPolicy"].sum()),
                "meanS13CompositeScore": float(group["s13CompositeScore"].mean()) if "s13CompositeScore" in group else np.nan,
                "meanHoldoutCompositeScore": float(group["holdoutCompositeScore"].mean()) if "holdoutCompositeScore" in group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["policyCount", "className"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def select_atlas_highlights(catalog_df: pd.DataFrame, class_exemplars_df: pd.DataFrame, *, max_generated: int = 40) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_section(section: str, df: pd.DataFrame, reason: str) -> None:
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            rows.append(
                {
                    "policyAtlasVersion": POLICY_ATLAS_VERSION,
                    "highlightSection": section,
                    "sectionRank": rank,
                    "policyId": str(row["policyId"]),
                    "className": _as_text(row.get("className")),
                    "atlasPrimaryRole": _as_text(row.get("atlasPrimaryRole")),
                    "roleTagsJson": _as_text(row.get("roleTagsJson"), "[]"),
                    "highlightReason": reason,
                    "atlasPolicyAnchor": _as_text(row.get("atlasPolicyAnchor")),
                    "frontierRank": int(row.get("frontierRank")) if pd.notna(row.get("frontierRank", np.nan)) else -1,
                    "s13CompositeScore": float(row.get("s13CompositeScore")) if pd.notna(row.get("s13CompositeScore", np.nan)) else np.nan,
                    "holdoutCompositeScore": float(row.get("holdoutCompositeScore")) if pd.notna(row.get("holdoutCompositeScore", np.nan)) else np.nan,
                }
            )

    add_section("frontier_candidates", catalog_df[catalog_df["isFrontierCandidate"]].sort_values("frontierRank", kind="mergesort"), "S14 curated frontier set")
    add_section("classic_policies", catalog_df[catalog_df["isClassicPolicy"]].sort_values(["classicFamily", "policyId"], kind="mergesort"), "classic Bubble/Insertion/Selection family")
    add_section("qd_elites", catalog_df[catalog_df["isS08Elite"]].sort_values(["s08EliteRank", "policyId"], kind="mergesort"), "S08 MAP-Elites archive member")
    add_section(
        "pathological_examples",
        catalog_df[catalog_df["isPathologicalPolicy"]].sort_values(["s13CompositeScore", "policyId"], ascending=[True, True], kind="mergesort").head(25),
        "S11/S13 pathological or failure-mode label",
    )
    add_section(
        "null_controls",
        catalog_df[catalog_df["isNullPolicy"]].sort_values(["s13CompositeScore", "policyId"], ascending=[False, True], kind="mergesort").head(25),
        "null or context-control policy",
    )
    if not class_exemplars_df.empty:
        exemplar_ids = class_exemplars_df.sort_values(["className", "exemplarRank"], kind="mergesort")["policyId"].astype(str).drop_duplicates()
        add_section(
            "class_exemplars",
            catalog_df[catalog_df["policyId"].isin(exemplar_ids)].sort_values(["className", "distanceToCentroid", "policyId"], kind="mergesort"),
            "S11 class exemplar",
        )
    generated = catalog_df[
        ~catalog_df["isClassicPolicy"] & ~catalog_df["isNullPolicy"] & ~catalog_df["isS08Elite"] & ~catalog_df["isFrontierCandidate"]
    ].copy()
    add_section(
        "generated_reference",
        generated.sort_values(["s13CompositeScore", "policyId"], ascending=[False, True], kind="mergesort").head(max_generated),
        "high-scoring generated reference policy",
    )
    return pd.DataFrame(rows).reset_index(drop=True)


def build_artifact_link_table(
    *,
    catalog_df: pd.DataFrame,
    global_artifacts: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, path, kind in global_artifacts:
        path_obj = Path(path)
        rows.append(
            {
                "policyAtlasVersion": POLICY_ATLAS_VERSION,
                "linkScope": "global",
                "policyId": "",
                "label": label,
                "artifactKind": kind,
                "path": path,
                "pathExists": path_obj.exists(),
            }
        )
    seen: set[tuple[str, str, str]] = set()
    for _, row in catalog_df.iterrows():
        for link in safe_json(row.get("policyArtifactLinksJson"), []):
            if not isinstance(link, Mapping):
                continue
            policy_id = str(row["policyId"])
            path = _as_text(link.get("path"))
            label = _as_text(link.get("label"), "artifact")
            key = (policy_id, label, path)
            if not path or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "policyAtlasVersion": POLICY_ATLAS_VERSION,
                    "linkScope": "policy",
                    "policyId": policy_id,
                    "label": label,
                    "artifactKind": "policy_link",
                    "path": path,
                    "pathExists": Path(path).exists(),
                }
            )
    return pd.DataFrame(rows)


def extract_html_links(html_path: str | Path) -> pd.DataFrame:
    path = Path(html_path)
    text = path.read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', text))
    rows: list[dict[str, Any]] = []
    for attr, link in re.findall(r'\b(href|src)="([^"]+)"', text):
        if link.startswith("http://") or link.startswith("https://") or link.startswith("mailto:"):
            exists = True
            target = link
            kind = "external"
        elif link.startswith("#"):
            target = link[1:]
            exists = target in ids
            kind = "anchor"
        else:
            target_path = Path(link)
            resolved = target_path if target_path.is_absolute() else (path.parent / target_path).resolve()
            target = str(resolved)
            exists = resolved.exists()
            kind = "file"
        rows.append(
            {
                "policyAtlasVersion": POLICY_ATLAS_VERSION,
                "htmlPath": str(path),
                "attribute": attr,
                "hrefOrSrc": link,
                "targetKind": kind,
                "resolvedTarget": target,
                "targetExists": bool(exists),
            }
        )
    return pd.DataFrame(rows)


def validate_policy_id_references(catalog_df: pd.DataFrame) -> pd.DataFrame:
    policy_ids = set(catalog_df["policyId"].astype(str))
    rows: list[dict[str, Any]] = []
    for _, row in catalog_df.iterrows():
        for neighbor in safe_json(row.get("nearestNeighborsJson"), []):
            if isinstance(neighbor, Mapping):
                neighbor_id = _as_text(neighbor.get("neighborPolicyId"))
                rows.append(
                    {
                        "policyAtlasVersion": POLICY_ATLAS_VERSION,
                        "sourcePolicyId": str(row["policyId"]),
                        "referencedPolicyId": neighbor_id,
                        "referenceKind": "nearest_neighbor",
                        "resolves": neighbor_id in policy_ids,
                    }
                )
    return pd.DataFrame(rows)


def validate_policy_atlas_outputs(
    *,
    catalog_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    artifact_link_df: pd.DataFrame,
    html_link_df: pd.DataFrame,
    policy_reference_df: pd.DataFrame,
    upstream_statuses: Mapping[str, Mapping[str, Any]],
    repo_test_payload: Mapping[str, Any],
    atlas_html_exists: bool,
    atlas_summary_exists: bool,
    report_bundle_manifest_exists: bool,
    minimum_catalog_rows: int = 2060,
    expected_frontier_count: int = 16,
    minimum_highlight_sections: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(check_id: str, success: bool, detail: str) -> None:
        rows.append({"checkId": check_id, "success": bool(success), "detail": str(detail)})

    for step in [f"S{index:02d}" for index in range(1, 15)]:
        status = upstream_statuses.get(step, {})
        add(f"upstream_{step.lower()}_success", bool(status.get("success")), f"{step} status={status.get('status')}")
    add("catalog_nonempty", len(catalog_df) >= minimum_catalog_rows, f"catalogRows={len(catalog_df)} minimum={minimum_catalog_rows}")
    add("catalog_unique_policy_ids", catalog_df["policyId"].astype(str).is_unique, f"uniquePolicyIds={catalog_df['policyId'].nunique()}")
    for role in ATLAS_REQUIRED_ROLES:
        count = int(catalog_df["roleTagsJson"].map(lambda value: role in safe_json(value, [])).sum())
        add(f"role_{role}_present", count > 0, f"{role}Count={count}")
    add(
        "frontier_count_matches_s14",
        int(catalog_df["isFrontierCandidate"].sum()) == expected_frontier_count,
        f"frontierCount={int(catalog_df['isFrontierCandidate'].sum())} expected={expected_frontier_count}",
    )
    add(
        "highlight_sections_present",
        highlight_df["highlightSection"].nunique() >= minimum_highlight_sections,
        f"sections={sorted(set(highlight_df['highlightSection']))} minimum={minimum_highlight_sections}",
    )
    add("class_summary_nonempty", not class_summary_df.empty and int(class_summary_df["policyCount"].sum()) == len(catalog_df), f"classRows={len(class_summary_df)}")
    add("artifact_links_resolve", not artifact_link_df.empty and bool(artifact_link_df["pathExists"].all()), f"links={len(artifact_link_df)} broken={int((~artifact_link_df['pathExists']).sum()) if not artifact_link_df.empty else 'NA'}")
    add("html_links_resolve", not html_link_df.empty and bool(html_link_df["targetExists"].all()), f"htmlLinks={len(html_link_df)} broken={int((~html_link_df['targetExists']).sum()) if not html_link_df.empty else 'NA'}")
    unresolved = int((~policy_reference_df["resolves"]).sum()) if not policy_reference_df.empty else 0
    add("policy_id_references_resolve", unresolved == 0 and not policy_reference_df.empty, f"references={len(policy_reference_df)} unresolved={unresolved}")
    add("atlas_html_exists", bool(atlas_html_exists), f"atlasHtmlExists={atlas_html_exists}")
    add("atlas_summary_exists", bool(atlas_summary_exists), f"atlasSummaryExists={atlas_summary_exists}")
    add("report_bundle_manifest_exists", bool(report_bundle_manifest_exists), f"reportBundleManifestExists={report_bundle_manifest_exists}")
    add("repo_unit_tests", bool(repo_test_payload.get("success")), f"returnCode={repo_test_payload.get('returnCode')} command={repo_test_payload.get('command')}")
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def render_policy_atlas_html(
    *,
    catalog_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    atlas_summary_href: str,
    catalog_csv_href: str,
    catalog_parquet_href: str,
    embedding_figure_href: str,
) -> str:
    catalog = catalog_df.copy()
    total = len(catalog)
    role_counts = {
        role: int(catalog["roleTagsJson"].map(lambda value: role in safe_json(value, [])).sum())
        for role in ["frontier", "classic", "null", "qd_elite", "pathological", "phase_boundary"]
    }
    table_rows: list[str] = []
    for _, row in catalog.iterrows():
        tags = " ".join(safe_json(row.get("roleTagsJson"), []))
        neighbors = safe_json(row.get("nearestNeighborsJson"), [])[:3]
        neighbor_text = ", ".join(str(item.get("neighborPolicyId", "")) for item in neighbors if isinstance(item, Mapping))
        dsl_path = _as_text(row.get("dslPath"))
        dsl_link = f'<a href="{html.escape(dsl_path)}">DSL</a>' if dsl_path else "NA"
        table_rows.append(
            "<tr "
            f'id="{html.escape(row["atlasPolicyAnchor"])}" '
            f'data-tags="{html.escape(tags)}" data-class="{html.escape(_as_text(row.get("className")))}">'
            f'<td><a href="#{html.escape(row["atlasPolicyAnchor"])}">{html.escape(str(row["policyId"]))}</a></td>'
            f"<td>{html.escape(tags)}</td>"
            f"<td>{html.escape(_as_text(row.get('className'), 'unassigned'))}</td>"
            f"<td>{html.escape(_as_text(row.get('lineageId'))[:80])}</td>"
            f"<td>{_fmt(row.get('s13CompositeScore'))}</td>"
            f"<td>{_fmt(row.get('holdoutCompositeScore'))}</td>"
            f"<td>{html.escape(neighbor_text)}</td>"
            f"<td>{dsl_link}</td>"
            "</tr>"
        )
    highlight_rows: list[str] = []
    for _, row in highlight_df.head(180).iterrows():
        highlight_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['highlightSection']))}</td>"
            f"<td>{int(row['sectionRank'])}</td>"
            f'<td><a href="#{html.escape(str(row["atlasPolicyAnchor"]))}">{html.escape(str(row["policyId"]))}</a></td>'
            f"<td>{html.escape(str(row.get('className', '')))}</td>"
            f"<td>{html.escape(str(row.get('roleTagsJson', '[]')))}</td>"
            f"<td>{_fmt(row.get('s13CompositeScore'))}</td>"
            f"<td>{_fmt(row.get('holdoutCompositeScore'))}</td>"
            f"<td>{html.escape(str(row.get('highlightReason', '')))}</td>"
            "</tr>"
        )
    class_rows: list[str] = []
    for _, row in class_summary_df.iterrows():
        class_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('className', 'unassigned')))}</td>"
            f"<td>{int(row.get('policyCount', 0))}</td>"
            f"<td>{int(row.get('frontierCount', 0))}</td>"
            f"<td>{int(row.get('classicCount', 0))}</td>"
            f"<td>{int(row.get('nullCount', 0))}</td>"
            f"<td>{_fmt(row.get('meanS13CompositeScore'))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>E03 Policy Morphospace Atlas</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #64748b;
      --line: #d7dde5;
      --panel: #f8fafc;
      --accent: #2f5f8f;
      --good: #687d45;
      --warn: #8f4d3a;
    }}
    body {{ margin: 0; color: var(--ink); font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: #ffffff; }}
    header {{ padding: 28px 32px 18px; border-bottom: 1px solid var(--line); background: #eef4f8; }}
    main {{ max-width: 1420px; margin: 0 auto; padding: 24px 28px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin-top: 32px; font-size: 20px; letter-spacing: 0; }}
    p {{ line-height: 1.5; }}
    .counts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 18px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; background: #fff; }}
    .metric strong {{ display: block; font-size: 22px; }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .links a {{ display: inline-block; margin: 0 10px 8px 0; color: var(--accent); }}
    .figure {{ max-width: 100%; border: 1px solid var(--line); border-radius: 6px; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 12px 0; }}
    input, select {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; font: inherit; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: var(--panel); position: sticky; top: 0; z-index: 1; }}
    tbody tr:target {{ outline: 2px solid var(--accent); background: #edf6ff; }}
    .table-wrap {{ max-height: 680px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; }}
    .note {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>E03 Policy Morphospace Atlas</h1>
    <p>Static atlas generated by S15. Scores, embeddings, classes, and frontier calls are bounded computational proxy evidence from S01-S14.</p>
    <div class="links">
      <a href="{html.escape(atlas_summary_href)}">Atlas summary</a>
      <a href="{html.escape(catalog_csv_href)}">Catalog CSV</a>
      <a href="{html.escape(catalog_parquet_href)}">Catalog Parquet</a>
    </div>
  </header>
  <main>
    <section class="counts">
      <div class="metric"><strong>{total}</strong><span>policy entries</span></div>
      <div class="metric"><strong>{role_counts['frontier']}</strong><span>S14 frontier candidates</span></div>
      <div class="metric"><strong>{role_counts['classic']}</strong><span>classic policies</span></div>
      <div class="metric"><strong>{role_counts['null']}</strong><span>null/context controls</span></div>
      <div class="metric"><strong>{role_counts['qd_elite']}</strong><span>QD elites</span></div>
      <div class="metric"><strong>{role_counts['pathological']}</strong><span>pathological/failure-mode examples</span></div>
    </section>
    <h2>Embedding Overview</h2>
    <img class="figure" src="{html.escape(embedding_figure_href)}" alt="Policy embedding colored by atlas role">
    <p class="note">The plot uses S10 PCA seed 0 coordinates when present; missing coordinates are omitted from the figure only.</p>
    <h2>Highlighted Entries</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Section</th><th>Rank</th><th>Policy</th><th>Class</th><th>Tags</th><th>S13 score</th><th>S14 holdout</th><th>Reason</th></tr></thead>
        <tbody>{''.join(highlight_rows)}</tbody>
      </table>
    </div>
    <h2>Class Summary</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Class</th><th>Policies</th><th>Frontier</th><th>Classic</th><th>Null</th><th>Mean S13 score</th></tr></thead>
        <tbody>{''.join(class_rows)}</tbody>
      </table>
    </div>
    <h2>Policy Catalog</h2>
    <div class="toolbar">
      <input id="q" type="search" placeholder="Filter policy, class, lineage, or tag" aria-label="Filter policies">
      <select id="role">
        <option value="">All roles</option>
        <option value="frontier">Frontier</option>
        <option value="classic">Classic</option>
        <option value="null">Null</option>
        <option value="qd_elite">QD elite</option>
        <option value="pathological">Pathological</option>
        <option value="phase_boundary">Phase boundary</option>
      </select>
      <span class="note" id="visible-count"></span>
    </div>
    <div class="table-wrap">
      <table id="catalog">
        <thead><tr><th>Policy ID</th><th>Tags</th><th>Class</th><th>Lineage</th><th>S13 score</th><th>S14 holdout</th><th>Nearest neighbors</th><th>DSL</th></tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
    <h2>Claim Boundary</h2>
    <p>Atlas labels, frontier ranks, universality classes, and competence scores are empirical computational summaries over bounded local-rule simulations. They are not biological validation, causal proof, or mathematical optimality claims.</p>
  </main>
  <script>
    const q = document.getElementById('q');
    const role = document.getElementById('role');
    const rows = Array.from(document.querySelectorAll('#catalog tbody tr'));
    const count = document.getElementById('visible-count');
    function applyFilter() {{
      const text = q.value.toLowerCase();
      const selected = role.value;
      let visible = 0;
      for (const row of rows) {{
        const hay = row.innerText.toLowerCase();
        const tags = row.dataset.tags || '';
        const keep = (!text || hay.includes(text)) && (!selected || tags.includes(selected));
        row.style.display = keep ? '' : 'none';
        if (keep) visible += 1;
      }}
      count.textContent = `${{visible}} / ${{rows.length}} visible`;
    }}
    q.addEventListener('input', applyFilter);
    role.addEventListener('change', applyFilter);
    applyFilter();
  </script>
</body>
</html>
"""
