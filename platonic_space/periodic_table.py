from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .world_schema import compact_json, sha256_path


PERIODIC_TABLE_SCHEMA_VERSION = "e07_s15_periodic_table.v1"
PERIODIC_TABLE_MODEL_VERSION = "e07_s15_static_atlas.v1"
PERIODIC_TABLE_CLAIM_BOUNDARY = (
    "Computational atlas over E07 simulator artifacts only. Entries, distances, classes, laws, and transfer "
    "summaries are empirical computational proxies, not biological validation, causal proof, clinical advice, "
    "cognition, agency, sentience, or evidence about living systems."
)
S13_S14_CAVEAT = (
    "S13 constrains a simple S08-distance transfer-success law: S08 distance is retained as a diagnostic "
    "covariate only unless executable mechanism mappings and direct replay support stronger transfer claims. "
    "S14 laws are bounded empirical computational regularities with explicit scope, counterexamples, "
    "uncertainty, and falsification conditions."
)
REQUIRED_STEPS = tuple(f"S{index:02d}" for index in range(1, 15))


@dataclass(frozen=True)
class PeriodicTableBundle:
    step_summary: pd.DataFrame
    policy_catalog: pd.DataFrame
    goal_catalog: pd.DataFrame
    world_catalog: pd.DataFrame
    law_catalog: pd.DataFrame
    combined_catalog: pd.DataFrame
    artifact_index: pd.DataFrame
    model_card_index: pd.DataFrame
    figure_index: pd.DataFrame
    claim_boundary_audit: pd.DataFrame
    report_bundle_payload: dict[str, Any]


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"href", "src"} and value:
                self.links.append({"tag": tag, "attribute": key, "target": value})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    return value


def dataframe_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
            out[column] = out[column].map(
                lambda value: compact_json(value) if isinstance(value, (dict, list, tuple, set)) else value
            )
        if out[column].dtype == "object":
            non_null = out[column].dropna()
            observed = {type(value) for value in non_null}
            if len(observed) > 1:
                out[column] = out[column].map(lambda value: None if value is None or pd.isna(value) else str(value))
    return out


def _num(value: Any, digits: int = 6) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, digits) if math.isfinite(out) else None


def _first_text(row: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value)
    return default


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or pd.isna(value):
        return []
    text = str(value)
    try:
        loaded = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [text] if text else []
    return loaded if isinstance(loaded, list) else [loaded]


def load_step_statuses(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        step_id: _read_json(artifacts_dir / "research_steps" / step_id / "status.json")
        for step_id in REQUIRED_STEPS
    }


def build_step_summary_table(artifacts_dir: Path) -> pd.DataFrame:
    statuses = load_step_statuses(artifacts_dir)
    rows: list[dict[str, Any]] = []
    for step_id in REQUIRED_STEPS:
        status = statuses.get(step_id, {})
        rows.append(
            {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "researchStepId": step_id,
                "stepNumber": status.get("stepNumber"),
                "title": status.get("title", ""),
                "success": bool(status.get("success")),
                "status": status.get("status", "missing"),
                "outcomeClassification": status.get("outcomeClassification", ""),
                "validationResult": status.get("validationResult", ""),
                "caveatsOrBlockers": status.get("caveatsOrBlockers", ""),
                "recommendedNextAction": status.get("recommendedNextAction", ""),
                "artifactCount": len(status.get("artifactsWritten", []) or []),
                "claimBoundary": status.get("claimBoundary", PERIODIC_TABLE_CLAIM_BOUNDARY),
            }
        )
    return pd.DataFrame(rows)


def build_policy_atlas_table(artifacts_dir: Path) -> pd.DataFrame:
    policies = _read_parquet(artifacts_dir / "research_steps" / "S06" / "policy_embeddings.parquet")
    classes = _read_parquet(artifacts_dir / "research_steps" / "S10" / "universality_classes.parquet")
    exemplars = _read_parquet(artifacts_dir / "research_steps" / "S10" / "class_exemplars.parquet")
    neighbors = _read_parquet(artifacts_dir / "research_steps" / "S06" / "nearest_neighbors.parquet")
    policy_catalog = _read_parquet(artifacts_dir / "data" / "e07_policy_abstract_catalog.parquet")

    if policies.empty:
        return pd.DataFrame()

    out = policies.copy()
    if not classes.empty:
        class_cols = [
            "abstractPolicyId",
            "universalityClassId",
            "className",
            "classLabel",
            "classAssignmentConfidence",
            "distanceToClassMedoidS08",
            "nearestOtherClassMedoidDistanceS08",
            "distanceUncertaintyLevel",
            "goalContextTopGoalFamily",
            "claimBoundary",
        ]
        out = out.merge(classes[[column for column in class_cols if column in classes.columns]], on="abstractPolicyId", how="left")

    exemplar_ids = set(exemplars.get("abstractPolicyId", pd.Series(dtype=str)).astype(str)) if not exemplars.empty else set()
    if not neighbors.empty:
        labels = out[["abstractPolicyId", "policyLabel"]].rename(
            columns={"abstractPolicyId": "neighborPolicyId", "policyLabel": "neighborPolicyLabel"}
        )
        neighbor_rows = (
            neighbors[neighbors["neighborRank"].astype(int).le(3)]
            .merge(labels, on="neighborPolicyId", how="left")
            .sort_values(["queryPolicyId", "neighborRank"])
        )
        neighbor_summary = neighbor_rows.groupby("queryPolicyId").apply(
            lambda group: [
                {
                    "rank": int(row["neighborRank"]),
                    "policyId": str(row["neighborPolicyId"]),
                    "label": str(row.get("neighborPolicyLabel", "")),
                    "cosineSimilarity": _num(row.get("cosineSimilarity")),
                }
                for _, row in group.iterrows()
            ],
            include_groups=False,
        )
        out["nearestNeighborsJson"] = out["abstractPolicyId"].map(lambda pid: neighbor_summary.get(pid, []))
    else:
        out["nearestNeighborsJson"] = [[] for _ in range(len(out))]

    if not policy_catalog.empty:
        caveats = policy_catalog[["abstractPolicyId", "caveatsOrBlockers", "replayability"]].drop_duplicates("abstractPolicyId")
        out = out.merge(caveats, on="abstractPolicyId", how="left")

    out["entityType"] = "policy"
    out["entityId"] = out["abstractPolicyId"].astype(str)
    out["displayName"] = out.get("policyLabel", out["entityId"]).astype(str)
    out["atlasRole"] = out["abstractPolicyId"].astype(str).map(lambda value: "class_exemplar" if value in exemplar_ids else "embedded_policy")
    out["artifactLinksJson"] = out["abstractPolicyId"].map(
        lambda _pid: [
            "/artifacts/research_steps/S06/policy_embeddings.parquet",
            "/artifacts/research_steps/S10/universality_classes.parquet",
        ]
    )
    out["claimBoundary"] = out.get("claimBoundary", pd.Series(PERIODIC_TABLE_CLAIM_BOUNDARY, index=out.index)).fillna(
        PERIODIC_TABLE_CLAIM_BOUNDARY
    )
    keep = [
        "schemaVersion",
        "entityType",
        "entityId",
        "displayName",
        "atlasRole",
        "sourceExperimentId",
        "policyFamily",
        "policyKind",
        "representationType",
        "stochasticity",
        "rowCount",
        "uniqueWorldCount",
        "uniqueGoalCount",
        "observedTargetCount",
        "embeddingStatus",
        "embeddingX",
        "embeddingY",
        "universalityClassId",
        "className",
        "classLabel",
        "classAssignmentConfidence",
        "distanceToClassMedoidS08",
        "distanceUncertaintyLevel",
        "goalContextTopGoalFamily",
        "nearestNeighborsJson",
        "replayability",
        "caveatsOrBlockers",
        "artifactLinksJson",
        "claimBoundary",
    ]
    out["schemaVersion"] = PERIODIC_TABLE_SCHEMA_VERSION
    return out[[column for column in keep if column in out.columns]].copy()


def build_goal_atlas_table(artifacts_dir: Path) -> pd.DataFrame:
    goals = _read_parquet(artifacts_dir / "research_steps" / "S07" / "goal_embeddings.parquet")
    goal_catalog = _read_parquet(artifacts_dir / "data" / "e07_goal_catalog.parquet")
    neighbors = _read_parquet(artifacts_dir / "research_steps" / "S08" / "nearest_neighbors.parquet")
    if goals.empty:
        return pd.DataFrame()

    out = goals.copy()
    if not goal_catalog.empty:
        caveats = goal_catalog[["abstractGoalId", "caveatsOrBlockers", "linkedWorldCount", "linkedPolicyCount"]].drop_duplicates(
            "abstractGoalId"
        )
        out = out.merge(caveats, on="abstractGoalId", how="left", suffixes=("", "_catalog"))

    goal_neighbors = neighbors[neighbors.get("entityType", pd.Series(dtype=str)).astype(str).eq("goal")].copy()
    if not goal_neighbors.empty:
        labels = goals[["abstractGoalId", "goalLabel"]].rename(
            columns={"abstractGoalId": "neighborEntityId", "goalLabel": "neighborGoalLabel"}
        )
        top = goal_neighbors[goal_neighbors["neighborRank"].astype(int).le(3)].merge(labels, on="neighborEntityId", how="left")
        summary = top.groupby("queryEntityId").apply(
            lambda group: [
                {
                    "rank": int(row["neighborRank"]),
                    "goalId": str(row["neighborEntityId"]),
                    "label": str(row.get("neighborGoalLabel", "")),
                    "platonicDistance": _num(row.get("platonicDistance")),
                }
                for _, row in group.sort_values("neighborRank").iterrows()
            ],
            include_groups=False,
        )
        out["nearestNeighborsJson"] = out["abstractGoalId"].map(lambda gid: summary.get(gid, []))
    else:
        out["nearestNeighborsJson"] = [[] for _ in range(len(out))]

    out["schemaVersion"] = PERIODIC_TABLE_SCHEMA_VERSION
    out["entityType"] = "goal"
    out["entityId"] = out["abstractGoalId"].astype(str)
    out["displayName"] = out["goalLabel"].astype(str)
    out["atlasRole"] = out["sparseUncertaintyLevel"].astype(str).map(
        lambda value: "goal_sparse_uncertainty" if value.lower() == "high" else "embedded_goal"
    )
    out["artifactLinksJson"] = out["abstractGoalId"].map(
        lambda _gid: [
            "/artifacts/research_steps/S07/goal_embeddings.parquet",
            "/artifacts/data/e07_goal_catalog.parquet",
        ]
    )
    out["claimBoundary"] = PERIODIC_TABLE_CLAIM_BOUNDARY
    keep = [
        "schemaVersion",
        "entityType",
        "entityId",
        "displayName",
        "atlasRole",
        "goalFamily",
        "goalKind",
        "representationType",
        "targetDimensionality",
        "localObservability",
        "rowCount",
        "uniquePolicyCount",
        "embeddedPolicyCount",
        "uniqueWorldCount",
        "observedTargetCount",
        "embeddingStatus",
        "sparseUncertaintyLevel",
        "embeddingX",
        "embeddingY",
        "linkedWorldCount",
        "linkedPolicyCount",
        "nearestNeighborsJson",
        "caveatsOrBlockers",
        "artifactLinksJson",
        "claimBoundary",
    ]
    return out[[column for column in keep if column in out.columns]].copy()


def build_world_atlas_table(artifacts_dir: Path) -> pd.DataFrame:
    worlds = _read_parquet(artifacts_dir / "data" / "e07_world_catalog.parquet")
    entities = _read_parquet(artifacts_dir / "research_steps" / "S08" / "distance_entities.parquet")
    corpus = _read_parquet(artifacts_dir / "data" / "e07_unified_behavior_corpus.parquet")
    if worlds.empty:
        return pd.DataFrame()

    out = worlds.copy()
    world_entities = entities[entities.get("entityType", pd.Series(dtype=str)).astype(str).eq("world")].copy()
    if not world_entities.empty:
        cols = [
            "entityId",
            "embeddingDimension",
            "embeddingModel",
            "sparseCoveragePenalty",
            "distanceUncertaintyLevel",
            "rowCount",
            "observedTargetCount",
        ]
        out = out.merge(
            world_entities[[column for column in cols if column in world_entities.columns]].rename(columns={"entityId": "worldId"}),
            on="worldId",
            how="left",
        )
    if not corpus.empty and "worldId" in corpus.columns:
        counts = corpus.groupby("worldId").size().rename("behaviorRecordCount").reset_index()
        out = out.merge(counts, on="worldId", how="left")
    out["behaviorRecordCount"] = out.get("behaviorRecordCount", pd.Series(0, index=out.index)).fillna(0).astype(int)
    out["schemaVersion"] = PERIODIC_TABLE_SCHEMA_VERSION
    out["entityType"] = "world"
    out["entityId"] = out["worldId"].astype(str)
    out["displayName"] = out["title"].astype(str)
    out["atlasRole"] = out["metadataCompleteness"].astype(str).map(
        lambda value: "world_exception_documented" if value == "exception_documented" else "world_catalog_entry"
    )
    out["artifactLinksJson"] = out["worldId"].map(
        lambda _wid: [
            "/artifacts/research_steps/S01/normalized_world_catalog.parquet",
            "/artifacts/data/e07_world_catalog.parquet",
        ]
    )
    out["claimBoundary"] = out.get("claimBoundary", pd.Series(PERIODIC_TABLE_CLAIM_BOUNDARY, index=out.index)).fillna(
        PERIODIC_TABLE_CLAIM_BOUNDARY
    )
    keep = [
        "schemaVersion",
        "entityType",
        "entityId",
        "displayName",
        "atlasRole",
        "experimentId",
        "worldFamily",
        "substrateClass",
        "replayability",
        "metadataCompleteness",
        "behaviorRecordCount",
        "embeddingDimension",
        "embeddingModel",
        "sparseCoveragePenalty",
        "distanceUncertaintyLevel",
        "observedTargetCount",
        "exceptionsOrGaps",
        "artifactLinksJson",
        "claimBoundary",
    ]
    return out[[column for column in keep if column in out.columns]].copy()


def build_law_atlas_table(artifacts_dir: Path) -> pd.DataFrame:
    laws = _read_parquet(artifacts_dir / "research_steps" / "S14" / "empirical_law_catalog.parquet")
    evidence = _read_parquet(artifacts_dir / "research_steps" / "S14" / "law_evidence_links.parquet")
    if laws.empty:
        return pd.DataFrame()

    out = laws.copy()
    counts = evidence.groupby("lawId").size().rename("evidenceLinkCount").reset_index() if not evidence.empty else pd.DataFrame()
    if not counts.empty:
        out = out.merge(counts, on="lawId", how="left")
    out["schemaVersion"] = PERIODIC_TABLE_SCHEMA_VERSION
    out["entityType"] = "law"
    out["entityId"] = out["lawId"].astype(str)
    out["displayName"] = out["title"].astype(str)
    out["atlasRole"] = out["supportLevel"].astype(str)
    out["artifactLinksJson"] = out["lawId"].map(
        lambda _lid: [
            "/artifacts/research_steps/S14/empirical_law_catalog.parquet",
            "/artifacts/research_steps/S14/law_evidence_links.parquet",
            "/artifacts/reports/e07_empirical_laws.md",
        ]
    )
    out["s13S14Caveat"] = S13_S14_CAVEAT
    keep = [
        "schemaVersion",
        "entityType",
        "entityId",
        "displayName",
        "atlasRole",
        "lawFamily",
        "supportLevel",
        "outcomeClassification",
        "lawStatement",
        "scope",
        "excludedScope",
        "s13ConstraintRole",
        "recommendedUse",
        "evidenceLinkCount",
        "counterexampleCount",
        "uncertaintyCount",
        "falsificationConditionCount",
        "artifactLinksJson",
        "s13S14Caveat",
        "claimBoundary",
    ]
    return out[[column for column in keep if column in out.columns]].copy()


def build_combined_catalog(
    policy_catalog: pd.DataFrame,
    goal_catalog: pd.DataFrame,
    world_catalog: pd.DataFrame,
    law_catalog: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in (policy_catalog, goal_catalog, world_catalog, law_catalog):
        if frame.empty:
            continue
        common = frame.copy()
        for column in (
            "sourceExperimentId",
            "policyFamily",
            "goalFamily",
            "worldFamily",
            "lawFamily",
            "className",
            "supportLevel",
            "outcomeClassification",
            "embeddingX",
            "embeddingY",
        ):
            if column not in common.columns:
                common[column] = None
        frames.append(
            common[
                [
                    "schemaVersion",
                    "entityType",
                    "entityId",
                    "displayName",
                    "atlasRole",
                    "sourceExperimentId",
                    "policyFamily",
                    "goalFamily",
                    "worldFamily",
                    "lawFamily",
                    "className",
                    "supportLevel",
                    "outcomeClassification",
                    "embeddingX",
                    "embeddingY",
                    "artifactLinksJson",
                    "claimBoundary",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat([frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True)


def build_artifact_index(artifacts_dir: Path) -> pd.DataFrame:
    statuses = load_step_statuses(artifacts_dir)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step_id, status in statuses.items():
        for artifact in status.get("artifactsWritten", []) or []:
            path_text = artifact.get("path") if isinstance(artifact, Mapping) else artifact
            if not path_text:
                continue
            path = Path(path_text)
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                    "researchStepId": step_id,
                    "artifactRole": "step_artifact",
                    "artifactPath": key,
                    "exists": path.exists() and path.is_file(),
                    "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
                    "sha256": sha256_path(path) if path.exists() and path.is_file() else None,
                    "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
                }
            )
    extra_paths = [
        artifacts_dir / "reports" / "e07_empirical_laws.md",
        artifacts_dir / "results" / "e07_empirical_laws.parquet",
        artifacts_dir / "results" / "e07_empirical_law_evidence.parquet",
    ]
    for path in extra_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "researchStepId": "S15",
                "artifactRole": "report_ready_input",
                "artifactPath": key,
                "exists": path.exists() and path.is_file(),
                "sizeBytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": sha256_path(path) if path.exists() and path.is_file() else None,
                "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def build_model_card_index(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((artifacts_dir / "models").glob("e07_*/*.json")):
        payload = _read_json(path)
        rows.append(
            {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "modelDirectory": str(path.parent),
                "modelCardPath": str(path),
                "modelOrCardName": path.name,
                "researchStepId": payload.get("researchStepId", ""),
                "modelVersion": payload.get("modelVersion", ""),
                "title": payload.get("title", payload.get("modelName", "")),
                "exists": path.exists(),
                "sizeBytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_path(path) if path.exists() else None,
                "claimBoundary": payload.get("claimBoundary", PERIODIC_TABLE_CLAIM_BOUNDARY),
            }
        )
    return pd.DataFrame(rows)


def build_figure_index(artifacts_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((artifacts_dir / "figures").glob("e07_*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "figurePath": str(path),
                "figureName": path.name,
                "suffix": path.suffix.lower(),
                "exists": True,
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def build_claim_boundary_audit(
    step_summary: pd.DataFrame,
    laws: pd.DataFrame,
    reports: Sequence[Path] = (),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
            "checkId": "s13_caveat_preserved",
            "severity": "error",
            "success": S13_S14_CAVEAT.lower().find("simple s08-distance transfer-success law") >= 0,
            "observed": S13_S14_CAVEAT,
            "expected": "S13 caveat against a simple S08-distance transfer-success law is explicit",
        }
    )
    rows.append(
        {
            "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
            "checkId": "s14_claim_boundary_preserved",
            "severity": "error",
            "success": bool(
                not laws.empty
                and laws.get("claimBoundary", pd.Series(dtype=str)).astype(str).str.contains("computational", case=False).all()
            ),
            "observed": str(len(laws)),
            "expected": "every S14 law carries computational claim-boundary text",
        }
    )
    rows.append(
        {
            "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
            "checkId": "all_upstream_steps_successful",
            "severity": "error",
            "success": bool(step_summary.get("success", pd.Series(dtype=bool)).astype(bool).all()),
            "observed": f"{int(step_summary.get('success', pd.Series(dtype=bool)).astype(bool).sum())}/{len(step_summary)}",
            "expected": "S01-S14 status success before final atlas packaging",
        }
    )
    disallowed = re.compile(
        r"\b(proves biology|biological proof of|wet[- ]lab validated|clinical recommendation|sentient system)\b",
        re.I,
    )
    for report in reports:
        text = report.read_text(encoding="utf-8") if report.exists() else ""
        rows.append(
            {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "checkId": f"claim_scan::{report.name}",
                "severity": "error",
                "success": bool(report.exists() and not disallowed.search(text)),
                "observed": f"exists={report.exists()}; disallowedMatch={bool(disallowed.search(text))}",
                "expected": "report exists and avoids unbounded biological/causal claim wording",
            }
        )
    return pd.DataFrame(rows)


def build_periodic_table_bundle(artifacts_dir: Path) -> PeriodicTableBundle:
    step_summary = build_step_summary_table(artifacts_dir)
    policy_catalog = build_policy_atlas_table(artifacts_dir)
    goal_catalog = build_goal_atlas_table(artifacts_dir)
    world_catalog = build_world_atlas_table(artifacts_dir)
    law_catalog = build_law_atlas_table(artifacts_dir)
    combined_catalog = build_combined_catalog(policy_catalog, goal_catalog, world_catalog, law_catalog)
    artifact_index = build_artifact_index(artifacts_dir)
    model_card_index = build_model_card_index(artifacts_dir)
    figure_index = build_figure_index(artifacts_dir)
    claim_boundary_audit = build_claim_boundary_audit(step_summary, law_catalog)
    report_bundle_payload = {
        "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
        "researchStepId": "S15",
        "title": "E07 periodic table atlas and final synthesis",
        "claimBoundary": PERIODIC_TABLE_CLAIM_BOUNDARY,
        "s13S14Caveat": S13_S14_CAVEAT,
        "entityCounts": {
            "policies": int(len(policy_catalog)),
            "goals": int(len(goal_catalog)),
            "worlds": int(len(world_catalog)),
            "laws": int(len(law_catalog)),
            "combined": int(len(combined_catalog)),
        },
        "primaryReports": {
            "atlasHtml": str(artifacts_dir / "reports" / "e07_periodic_table.html"),
            "synthesisReport": str(artifacts_dir / "reports" / "e07_platonic_space_report.md"),
            "empiricalLawsReport": str(artifacts_dir / "reports" / "e07_empirical_laws.md"),
        },
        "staticExports": {
            "combinedCatalog": str(artifacts_dir / "results" / "e07_periodic_table_catalog.parquet"),
            "artifactIndex": str(artifacts_dir / "results" / "e07_final_artifact_index.parquet"),
        },
        "modelCards": sorted(model_card_index.get("modelCardPath", pd.Series(dtype=str)).astype(str).tolist()),
    }
    return PeriodicTableBundle(
        step_summary=step_summary,
        policy_catalog=policy_catalog,
        goal_catalog=goal_catalog,
        world_catalog=world_catalog,
        law_catalog=law_catalog,
        combined_catalog=combined_catalog,
        artifact_index=artifact_index,
        model_card_index=model_card_index,
        figure_index=figure_index,
        claim_boundary_audit=claim_boundary_audit,
        report_bundle_payload=report_bundle_payload,
    )


def _records(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return [
        {str(key): _json_ready(value) for key, value in row.items()}
        for row in dataframe_json_columns(df.head(limit)).to_dict(orient="records")
    ]


def _link(path: str, label: str | None = None) -> str:
    escaped_path = html.escape(path)
    escaped_label = html.escape(label or Path(path).name)
    return f'<a href="{escaped_path}">{escaped_label}</a>'


def _summary_metric(label: str, value: Any, sublabel: str) -> str:
    return (
        '<div class="metric">'
        f'<span class="metric-value">{html.escape(str(value))}</span>'
        f'<span class="metric-label">{html.escape(label)}</span>'
        f'<span class="metric-sub">{html.escape(sublabel)}</span>'
        "</div>"
    )


def render_periodic_table_html(
    *,
    bundle: PeriodicTableBundle,
    created_utc: str,
) -> str:
    counts = bundle.report_bundle_payload["entityCounts"]
    policy_rows = _records(bundle.policy_catalog, 120)
    goal_rows = _records(bundle.goal_catalog, 80)
    world_rows = _records(bundle.world_catalog, 100)
    law_rows = _records(bundle.law_catalog, 30)
    class_rows = _records(_readable_class_summary(bundle.policy_catalog), 20)
    step_rows = _records(bundle.step_summary, 20)
    model_rows = _records(bundle.model_card_index, 30)
    figure_paths = bundle.figure_index[bundle.figure_index["suffix"].eq(".png")]["figurePath"].head(8).tolist() if not bundle.figure_index.empty else []
    payload = {
        "policies": policy_rows,
        "goals": goal_rows,
        "worlds": world_rows,
        "laws": law_rows,
        "steps": step_rows,
        "models": model_rows,
    }
    payload_json = json.dumps(_json_ready(payload), sort_keys=True)
    figure_html = "\n".join(
        f'<figure><img src="{html.escape(path)}" alt="{html.escape(Path(path).stem)}"><figcaption>{html.escape(Path(path).stem)}</figcaption></figure>'
        for path in figure_paths
    )
    class_table = _html_table(class_rows, ["className", "policyCount", "topFamilies", "meanConfidence"])
    law_table = _html_table(law_rows, ["entityId", "displayName", "supportLevel", "outcomeClassification", "lawStatement"])
    step_table = _html_table(step_rows, ["researchStepId", "title", "outcomeClassification", "validationResult"])
    model_table = _html_table(model_rows, ["researchStepId", "title", "modelCardPath"])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>E07 Periodic Table Atlas</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #24302f;
      --muted: #5d6d6a;
      --line: #d7dfdc;
      --panel: #f7faf9;
      --panel-2: #eef5f3;
      --accent: #1f7a6b;
      --accent-2: #a45d20;
      --warn: #8a3d2a;
      --paper: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
      letter-spacing: 0;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 8px; font-size: 2rem; line-height: 1.1; letter-spacing: 0; }}
    h2 {{ font-size: 1.25rem; margin: 0 0 12px; letter-spacing: 0; }}
    h3 {{ font-size: 1rem; margin: 0 0 8px; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; }}
    .subtle {{ color: var(--muted); max-width: 980px; }}
    .claim {{
      border-left: 4px solid var(--warn);
      background: #fff7f4;
      padding: 12px 14px;
      margin-top: 16px;
      font-size: 0.95rem;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 8px;
      padding: 12px;
      min-height: 96px;
    }}
    .metric-value {{ display: block; font-size: 1.65rem; font-weight: 700; color: var(--accent); }}
    .metric-label {{ display: block; font-weight: 650; }}
    .metric-sub {{ display: block; color: var(--muted); font-size: 0.82rem; margin-top: 4px; }}
    nav {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    button.tab {{
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }}
    button.tab.active {{ border-color: var(--accent); background: var(--panel-2); color: var(--accent); }}
    section.band {{ border-bottom: 1px solid var(--line); }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin: 8px 0 12px;
    }}
    input[type="search"], select {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
      min-width: 220px;
      background: white;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }}
    .entry {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--paper);
      min-height: 190px;
    }}
    .entry h3 {{ overflow-wrap: anywhere; }}
    .tag {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      margin: 0 4px 5px 0;
      font-size: 0.78rem;
      color: var(--muted);
      background: var(--panel);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid var(--line);
      padding: 8px;
      overflow-wrap: anywhere;
    }}
    th {{ background: var(--panel); position: sticky; top: 0; z-index: 1; }}
    .table-wrap {{ overflow: auto; max-height: 460px; border: 1px solid var(--line); border-radius: 8px; }}
    .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: var(--paper); }}
    img {{ width: 100%; height: 210px; object-fit: contain; display: block; background: white; }}
    figcaption {{ color: var(--muted); font-size: 0.82rem; margin-top: 6px; overflow-wrap: anywhere; }}
    a {{ color: var(--accent); }}
    .hidden {{ display: none; }}
    footer {{ background: var(--panel); color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>E07 Periodic Table Atlas</h1>
      <p class="subtle">A final, static atlas of computational local-policy worlds, goals, behavior embeddings, empirical neighborhoods, validation results, transfer constraints, and bounded empirical laws. Created {html.escape(created_utc)}.</p>
      <div class="claim"><strong>Claim boundary.</strong> {html.escape(PERIODIC_TABLE_CLAIM_BOUNDARY)}<br><strong>S13/S14 caveat.</strong> {html.escape(S13_S14_CAVEAT)}</div>
      <div class="metrics">
        {_summary_metric("Policies", counts["policies"], "S06 behavior-embedded policy entries")}
        {_summary_metric("Goals", counts["goals"], "S07 behavior-embedded goal entries")}
        {_summary_metric("Worlds", counts["worlds"], "S01 normalized world entries")}
        {_summary_metric("Laws", counts["laws"], "S14 bounded empirical laws")}
        {_summary_metric("Steps", len(bundle.step_summary), "S01-S14 completed inputs")}
      </div>
      <nav aria-label="Atlas sections">
        <button class="tab active" data-tab="policies">Policies</button>
        <button class="tab" data-tab="goals">Goals</button>
        <button class="tab" data-tab="worlds">Worlds</button>
        <button class="tab" data-tab="laws">Laws</button>
        <button class="tab" data-tab="evidence">Evidence</button>
      </nav>
    </div>
  </header>

  <main>
    <section class="band" id="policies">
      <div class="wrap">
        <h2>Policy Table</h2>
        <p class="subtle">Rows are behavior-embedded policies from S06, annotated with S10 empirical classes and nearest-neighbor diagnostics.</p>
        <div class="toolbar">
          <input id="policySearch" type="search" placeholder="Search policies">
          <select id="policyClass"><option value="">All classes</option></select>
        </div>
        <div id="policyGrid" class="grid"></div>
      </div>
    </section>

    <section class="band hidden" id="goals">
      <div class="wrap">
        <h2>Goal Table</h2>
        <p class="subtle">Rows are S07 goals with sparse-coverage uncertainty retained.</p>
        <div class="toolbar"><input id="goalSearch" type="search" placeholder="Search goals"></div>
        <div id="goalGrid" class="grid"></div>
      </div>
    </section>

    <section class="band hidden" id="worlds">
      <div class="wrap">
        <h2>World Table</h2>
        <p class="subtle">Rows are S01 normalized computational worlds, including partial and summary-only caveats.</p>
        <div class="toolbar"><input id="worldSearch" type="search" placeholder="Search worlds"></div>
        <div id="worldGrid" class="grid"></div>
      </div>
    </section>

    <section class="band hidden" id="laws">
      <div class="wrap">
        <h2>Empirical Computational Laws</h2>
        <p class="subtle">Every law is bounded by evidence links, scope, counterexamples, uncertainty, and falsification conditions.</p>
        <div class="table-wrap">{law_table}</div>
      </div>
    </section>

    <section class="band hidden" id="evidence">
      <div class="wrap">
        <h2>Evidence And Model Cards</h2>
        <h3>Step Outcomes</h3>
        <div class="table-wrap">{step_table}</div>
        <h3 style="margin-top:18px">Model And Card Links</h3>
        <div class="table-wrap">{model_table}</div>
        <h3 style="margin-top:18px">Class Snapshot</h3>
        <div class="table-wrap">{class_table}</div>
        <h3 style="margin-top:18px">Figures</h3>
        <div class="figures">{figure_html}</div>
        <h3 style="margin-top:18px">Durable Exports</h3>
        <p>{_link("/artifacts/results/e07_periodic_table_catalog.parquet", "combined catalog parquet")} | {_link("/artifacts/results/e07_final_artifact_index.parquet", "artifact index parquet")} | {_link("/artifacts/reports/e07_platonic_space_report.md", "final synthesis report")} | {_link("/artifacts/report_bundle_inputs/e07_s15_periodic_table.json", "report bundle JSON")}</p>
      </div>
    </section>
  </main>

  <footer>
    <div class="wrap">Standalone file atlas. No external network assets. Data shown here are compact static exports; trace-level details remain in linked artifacts.</div>
  </footer>

  <script id="atlas-data" type="application/json">{html.escape(payload_json)}</script>
  <script>
    const data = JSON.parse(document.getElementById('atlas-data').textContent);
    const tabs = [...document.querySelectorAll('button.tab')];
    tabs.forEach(button => button.addEventListener('click', () => {{
      tabs.forEach(item => item.classList.remove('active'));
      button.classList.add('active');
      ['policies','goals','worlds','laws','evidence'].forEach(id => {{
        document.getElementById(id).classList.toggle('hidden', id !== button.dataset.tab);
      }});
    }}));
    function tag(value) {{
      return value ? `<span class="tag">${{String(value)}}</span>` : '';
    }}
    function entry(row, kind) {{
      const title = row.displayName || row.entityId;
      const tags = [
        row.className, row.policyFamily, row.goalFamily, row.worldFamily, row.lawFamily,
        row.sourceExperimentId, row.atlasRole, row.distanceUncertaintyLevel, row.sparseUncertaintyLevel
      ].filter(Boolean).map(tag).join('');
      const detail = kind === 'policy'
        ? `Targets: ${{row.observedTargetCount ?? ''}}; rows: ${{row.rowCount ?? ''}}; class confidence: ${{row.classAssignmentConfidence ?? ''}}`
        : kind === 'goal'
          ? `Policies: ${{row.uniquePolicyCount ?? ''}}; worlds: ${{row.uniqueWorldCount ?? ''}}; sparse uncertainty: ${{row.sparseUncertaintyLevel ?? ''}}`
          : `Substrate: ${{row.substrateClass ?? ''}}; behavior records: ${{row.behaviorRecordCount ?? ''}}; replayability: ${{row.replayability ?? ''}}`;
      return `<article class="entry"><h3>${{title}}</h3><div>${{tags}}</div><p>${{detail}}</p><p class="subtle">${{row.caveatsOrBlockers || row.claimBoundary || ''}}</p></article>`;
    }}
    function renderGrid(kind, searchId, gridId, classId) {{
      const source = data[kind];
      const input = document.getElementById(searchId);
      const grid = document.getElementById(gridId);
      const classSelect = classId ? document.getElementById(classId) : null;
      if (classSelect) {{
        [...new Set(source.map(row => row.className).filter(Boolean))].sort().forEach(value => {{
          const opt = document.createElement('option'); opt.value = value; opt.textContent = value; classSelect.appendChild(opt);
        }});
      }}
      function update() {{
        const q = (input.value || '').toLowerCase();
        const cls = classSelect ? classSelect.value : '';
        const filtered = source.filter(row => {{
          const hay = JSON.stringify(row).toLowerCase();
          return hay.includes(q) && (!cls || row.className === cls);
        }}).slice(0, 60);
        grid.innerHTML = filtered.map(row => entry(row, kind.slice(0, -1))).join('');
      }}
      input.addEventListener('input', update);
      if (classSelect) classSelect.addEventListener('change', update);
      update();
    }}
    renderGrid('policies', 'policySearch', 'policyGrid', 'policyClass');
    renderGrid('goals', 'goalSearch', 'goalGrid');
    renderGrid('worlds', 'worldSearch', 'worldGrid');
  </script>
</body>
</html>
"""


def _readable_class_summary(policy_catalog: pd.DataFrame) -> pd.DataFrame:
    if policy_catalog.empty or "className" not in policy_catalog.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for class_name, group in policy_catalog.groupby("className", dropna=False):
        top_families = group.get("policyFamily", pd.Series(dtype=str)).astype(str).value_counts().head(3).to_dict()
        rows.append(
            {
                "className": class_name,
                "policyCount": int(len(group)),
                "topFamilies": compact_json(top_families),
                "meanConfidence": _num(pd.to_numeric(group.get("classAssignmentConfidence", pd.Series(dtype=float)), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("policyCount", ascending=False)


def _html_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, str) and value.startswith("/artifacts/"):
                cells.append(f"<td>{_link(value)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value)[:900])}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + header + "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"


def extract_html_links(html_text: str) -> pd.DataFrame:
    parser = LinkExtractor()
    parser.feed(html_text)
    return pd.DataFrame(parser.links)


def local_link_exists(target: str) -> bool:
    if target.startswith("#"):
        return True
    if target.startswith(("http://", "https://", "mailto:")):
        return False
    if target.startswith("/"):
        return Path(target).exists()
    return True


def validate_periodic_table_outputs(
    *,
    artifacts_dir: Path,
    bundle: PeriodicTableBundle,
    html_path: Path,
    report_path: Path,
    bundle_path: Path,
    static_export_paths: Sequence[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    html_text = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    links = extract_html_links(html_text) if html_text else pd.DataFrame(columns=["tag", "attribute", "target"])
    local_links = links[~links["target"].astype(str).str.startswith("#", na=False)] if not links.empty else links
    missing_links = [
        target for target in local_links.get("target", pd.Series(dtype=str)).astype(str).tolist() if not local_link_exists(target)
    ]
    external_links = [
        target
        for target in links.get("target", pd.Series(dtype=str)).astype(str).tolist()
        if target.startswith(("http://", "https://"))
    ]

    def add(check_id: str, severity: str, success: bool, observed: Any, expected: str) -> None:
        rows.append(
            {
                "schemaVersion": PERIODIC_TABLE_SCHEMA_VERSION,
                "checkId": check_id,
                "severity": severity,
                "success": bool(success),
                "observed": str(observed),
                "expected": expected,
            }
        )

    add(
        "upstream_s01_s14_success",
        "error",
        bool(bundle.step_summary["success"].astype(bool).all() and len(bundle.step_summary) == 14),
        f"{int(bundle.step_summary['success'].astype(bool).sum())}/{len(bundle.step_summary)}",
        "all S01-S14 statuses successful",
    )
    add(
        "atlas_html_exists_and_loads",
        "error",
        bool(html_path.exists() and len(html_text) > 5000 and "E07 Periodic Table Atlas" in html_text),
        f"exists={html_path.exists()}; bytes={html_path.stat().st_size if html_path.exists() else 0}",
        "standalone atlas HTML exists and contains expected title",
    )
    add(
        "atlas_has_no_external_dependencies",
        "error",
        len(external_links) == 0,
        compact_json(external_links),
        "no http or https dependencies in standalone atlas",
    )
    add(
        "atlas_artifact_links_resolve",
        "error",
        len(missing_links) == 0,
        compact_json(missing_links[:20]),
        "all local href/src links in atlas resolve",
    )
    add(
        "static_exports_exist",
        "error",
        all(path.exists() and path.is_file() for path in static_export_paths),
        compact_json([str(path) for path in static_export_paths]),
        "CSV/Parquet/JSON durable exports exist",
    )
    add(
        "model_card_links_present",
        "error",
        len(bundle.model_card_index) >= 8 and bundle.model_card_index["exists"].astype(bool).all(),
        f"{len(bundle.model_card_index)} model/card rows",
        "model/card index covers E07 model cards and all paths exist",
    )
    add(
        "report_bundle_input_exists",
        "error",
        bool(bundle_path.exists() and "s13S14Caveat" in _read_json(bundle_path)),
        str(bundle_path),
        "report bundle JSON exists and preserves S13/S14 caveat",
    )
    add(
        "claim_boundary_present_in_reports",
        "error",
        bool(
            report_path.exists()
            and PERIODIC_TABLE_CLAIM_BOUNDARY in report_path.read_text(encoding="utf-8")
            and PERIODIC_TABLE_CLAIM_BOUNDARY in html_text
            and S13_S14_CAVEAT in html_text
        ),
        f"report={report_path.exists()}; html={html_path.exists()}",
        "HTML atlas and synthesis report contain strict computational claim boundary and S13/S14 caveat",
    )
    add(
        "combined_catalog_complete",
        "error",
        bool(len(bundle.policy_catalog) >= 500 and len(bundle.goal_catalog) >= 30 and len(bundle.world_catalog) >= 50 and len(bundle.law_catalog) == 8),
        bundle.report_bundle_payload.get("entityCounts", {}),
        "combined catalog includes policy, goal, world, and law entries",
    )
    return pd.concat([pd.DataFrame(rows), bundle.claim_boundary_audit], ignore_index=True)


def periodic_table_outcome_classification(checks: pd.DataFrame) -> str:
    hard_failures = checks[(checks["severity"].eq("error")) & (~checks["success"])]
    return "supportive" if hard_failures.empty else "constraining/contradictory"
