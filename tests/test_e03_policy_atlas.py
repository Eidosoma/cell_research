from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from morphospace import (
    POLICY_ATLAS_VERSION,
    atlas_class_summary,
    build_artifact_link_table,
    build_policy_atlas_catalog,
    extract_html_links,
    render_policy_atlas_html,
    select_atlas_highlights,
    validate_policy_atlas_outputs,
    validate_policy_id_references,
)


POLICY_IDS = [
    "frontier_policy",
    "classic_bubble",
    "null_idle",
    "elite_policy",
    "pathological_policy",
    "generated_policy",
]


def policy_summary_fixture() -> pd.DataFrame:
    rows = []
    for index, policy_id in enumerate(POLICY_IDS):
        rows.append(
            {
                "policyId": policy_id,
                "family": "hand_designed" if policy_id == "classic_bubble" else "generated",
                "generationMethod": "fixture",
                "lineageId": f"lineage_{index}",
                "description": "atlas fixture policy",
                "comparisonRole": "classic" if policy_id == "classic_bubble" else "discovered",
                "selectedForS13": True,
                "primaryRole": "generated",
                "isClassicPolicy": policy_id == "classic_bubble",
                "isNullPolicy": policy_id == "null_idle",
                "isS08Elite": policy_id == "elite_policy",
                "isPhaseBoundaryPolicy": policy_id == "frontier_policy",
                "isPathologicalPolicy": policy_id == "pathological_policy",
                "classicFamily": "bubble" if policy_id == "classic_bubble" else "none",
                "className": f"UC{index % 3:02d}",
                "classLabel": "fixture",
                "s13CompositeScore": 0.2 + 0.1 * index,
                "completionScore": 0.4 + 0.05 * index,
                "sortednessScore": 0.5 + 0.04 * index,
                "energyScore": 0.8 - 0.03 * index,
                "robustnessScore": 0.2 + 0.02 * index,
                "delayedGratificationScore": 0.1 + 0.03 * index,
                "aggregationScore": 0.3 + 0.02 * index,
                "oscillationStabilityScore": 0.7,
            }
        )
    return pd.DataFrame(rows)


def class_assignments_fixture() -> pd.DataFrame:
    rows = []
    for index, policy_id in enumerate(POLICY_IDS):
        rows.append(
            {
                "policyId": policy_id,
                "universalityClassId": f"UC{index % 3:02d}",
                "className": f"UC{index % 3:02d}",
                "classLabel": "fixture",
                "classInterpretation": "fixture-only class",
                "empiricalCaveat": "unit test fixture",
                "distanceToCentroid": 0.05 + 0.01 * index,
                "primaryRole": "generated",
                "isNullPolicy": policy_id == "null_idle",
                "isClassicPolicy": policy_id == "classic_bubble",
                "isS08Elite": policy_id == "elite_policy",
                "isPhaseBoundaryPolicy": policy_id == "frontier_policy",
                "isPathologicalPolicy": policy_id == "pathological_policy",
                "phaseBoundaryInvolvementCount": 1 if policy_id == "frontier_policy" else 0,
                "s08Elite_qualityScore": 0.9 if policy_id == "elite_policy" else None,
            }
        )
    return pd.DataFrame(rows)


def corpus_and_dsl_index_fixture(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus_rows = []
    index_rows = []
    for policy_id in POLICY_IDS:
        dsl_path = tmp_path / f"{policy_id}.json"
        dsl_path.write_text(json.dumps({"name": policy_id, "rules": [], "default": {"action": "wait"}}), encoding="utf-8")
        corpus_rows.append(
            {
                "policyId": policy_id,
                "dslProgramJson": dsl_path.read_text(encoding="utf-8"),
                "structureHash": f"hash_{policy_id}",
                "parentPolicyIdsJson": "[]",
                "mutationOperatorsJson": "[]",
                "recombinationParentsJson": "[]",
                "tagsJson": "[]",
                "observationRequirementsJson": "[]",
                "actionCountsJson": '{"wait": 1}',
            }
        )
        index_rows.append(
            {
                "policyId": policy_id,
                "dslPath": str(dsl_path),
                "dslRelativePath": dsl_path.name,
                "sha256": f"sha_{policy_id}",
                "sizeBytes": dsl_path.stat().st_size,
            }
        )
    return pd.DataFrame(corpus_rows), pd.DataFrame(index_rows)


def embedding_fixture() -> pd.DataFrame:
    rows = []
    for index, policy_id in enumerate(POLICY_IDS):
        rows.append(
            {
                "policyId": policy_id,
                "embeddingMethod": "pca",
                "embeddingSeed": 0,
                "embeddingDim1": float(index),
                "embeddingDim2": float(index % 2),
                "embeddingDim3": 0.0,
                "embeddingDim4": 0.0,
                "embeddingDim5": 0.0,
            }
        )
    return pd.DataFrame(rows)


def neighbors_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policyId": policy_id,
                "neighborPolicyId": POLICY_IDS[(index + 1) % len(POLICY_IDS)],
                "neighborRank": 1,
                "featureDistance": 0.1 + 0.01 * index,
                "neighborPrimaryRole": "generated",
            }
            for index, policy_id in enumerate(POLICY_IDS)
        ]
    )


def build_fixture_catalog(tmp_path: Path) -> pd.DataFrame:
    corpus, dsl_index = corpus_and_dsl_index_fixture(tmp_path)
    frontier_files = pd.DataFrame(
        [
            {
                "policyId": "frontier_policy",
                "dslPath": str(tmp_path / "frontier_policy.json"),
                "dslSha256": "sha_frontier_policy",
            }
        ]
    )
    return build_policy_atlas_catalog(
        policy_summary_df=policy_summary_fixture(),
        corpus_df=corpus,
        dsl_index_df=dsl_index,
        class_assignments_df=class_assignments_fixture(),
        embeddings_df=embedding_fixture(),
        neighbors_df=neighbors_fixture(),
        qd_elites_df=pd.DataFrame(
            [
                {
                    "policyId": "elite_policy",
                    "eliteRank": 1,
                    "qualityScore": 0.91,
                    "descriptorCellId": "cell_1",
                    "descriptorCellLabel": "fixture",
                }
            ]
        ),
        pareto_df=pd.DataFrame([{"policyId": "frontier_policy", "paretoEligible": True, "paretoFrontRank": 1, "isParetoOptimal": True}]),
        frontier_df=pd.DataFrame(
            [
                {
                    "policyId": "frontier_policy",
                    "frontierRank": 1,
                    "frontierCompositeScore": 0.92,
                    "frontierObjectiveTagsJson": '["fixture"]',
                    "frontierSelectionStage": "fixture",
                    "selectionRationaleJson": "{}",
                    "duplicateOrNearDuplicateJustification": "fixture",
                }
            ]
        ),
        frontier_validation_summary_df=pd.DataFrame([{"policyId": "frontier_policy", "holdoutCompositeScore": 0.88}]),
        frontier_file_index_df=frontier_files,
        s13_missing_df=pd.DataFrame(columns=["policyId", "missingReason"]),
        s13_runs_df=pd.DataFrame(columns=["policyId"]),
        s14_runs_df=pd.DataFrame(columns=["policyId"]),
    )


class PolicyAtlasTests(unittest.TestCase):
    def test_catalog_highlights_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            catalog = build_fixture_catalog(tmp_path)
            class_summary = atlas_class_summary(catalog)
            highlights = select_atlas_highlights(
                catalog,
                pd.DataFrame(
                    [
                        {"policyId": "generated_policy", "className": "UC02", "exemplarRank": 1},
                        {"policyId": "classic_bubble", "className": "UC01", "exemplarRank": 1},
                    ]
                ),
            )
            references = validate_policy_id_references(catalog)

            self.assertEqual(POLICY_ATLAS_VERSION, set(catalog["policyAtlasVersion"]).pop())
            self.assertEqual(len(catalog), len(POLICY_IDS))
            self.assertEqual(int(catalog["isFrontierCandidate"].sum()), 1)
            self.assertEqual(int(class_summary["policyCount"].sum()), len(catalog))
            self.assertIn("frontier_candidates", set(highlights["highlightSection"]))
            self.assertFalse(references.empty)
            self.assertTrue(bool(references["resolves"].all()))

    def test_html_link_and_output_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            catalog = build_fixture_catalog(tmp_path)
            class_summary = atlas_class_summary(catalog)
            highlights = select_atlas_highlights(catalog, pd.DataFrame(columns=["policyId", "className", "exemplarRank"]))
            catalog_csv = tmp_path / "catalog.csv"
            catalog_parquet = tmp_path / "catalog.parquet"
            summary_md = tmp_path / "summary.md"
            figure_png = tmp_path / "figure.png"
            catalog.to_csv(catalog_csv, index=False)
            catalog.to_parquet(catalog_parquet, index=False)
            summary_md.write_text("# Summary\n", encoding="utf-8")
            figure_png.write_bytes(b"fixture image")
            html_path = tmp_path / "atlas.html"
            html_path.write_text(
                render_policy_atlas_html(
                    catalog_df=catalog,
                    highlight_df=highlights,
                    class_summary_df=class_summary,
                    atlas_summary_href=summary_md.name,
                    catalog_csv_href=catalog_csv.name,
                    catalog_parquet_href=catalog_parquet.name,
                    embedding_figure_href=figure_png.name,
                ),
                encoding="utf-8",
            )
            html_links = extract_html_links(html_path)
            policy_refs = validate_policy_id_references(catalog)
            global_artifact = tmp_path / "global.parquet"
            global_artifact.write_bytes(b"fixture")
            link_table = build_artifact_link_table(catalog_df=catalog, global_artifacts=[("fixture global", str(global_artifact), "fixture")])

            self.assertTrue(bool(html_links["targetExists"].all()))
            self.assertIn("fixture global", set(link_table["label"]))

            validation = validate_policy_atlas_outputs(
                catalog_df=catalog,
                highlight_df=highlights,
                class_summary_df=class_summary,
                artifact_link_df=pd.DataFrame(
                    [
                        {
                            "policyAtlasVersion": POLICY_ATLAS_VERSION,
                            "linkScope": "global",
                            "policyId": "",
                            "label": "fixture",
                            "artifactKind": "fixture",
                            "path": str(global_artifact),
                            "pathExists": True,
                        }
                    ]
                ),
                html_link_df=html_links,
                policy_reference_df=policy_refs,
                upstream_statuses={f"S{index:02d}": {"success": True, "status": "completed"} for index in range(1, 15)},
                repo_test_payload={"success": True, "returnCode": 0, "command": ["fixture"]},
                atlas_html_exists=html_path.exists(),
                atlas_summary_exists=summary_md.exists(),
                report_bundle_manifest_exists=True,
                minimum_catalog_rows=len(POLICY_IDS),
                expected_frontier_count=1,
                minimum_highlight_sections=5,
            )
            self.assertTrue(bool(validation["success"].all()), validation.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
