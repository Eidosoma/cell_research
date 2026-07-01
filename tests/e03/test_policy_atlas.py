"""S15 policy atlas tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.e03.frontier_candidates import build_policy_source_table
from src.e03.policy_atlas import (
    AtlasTraceConfig,
    atlas_digest,
    build_atlas_index,
    mark_trace_availability,
    render_atlas_html,
    representative_policy_ids,
    representative_trace_frame,
    validation_frame,
)
from src.e03.policy_generation import semantic_hash
from src.e03.rule_dsl import SIMPLE_SWAP_LEFT_POLICY, SIMPLE_SWAP_RIGHT_POLICY, parse_policy


def policy_record(source: str, source_kind: str = "unit") -> dict[str, object]:
    policy = parse_policy(source)
    return {
        "policyId": policy.policy_id,
        "policyName": policy.name,
        "sourceKind": source_kind,
        "semanticHash": semantic_hash(policy),
        "dslSha256": policy.sha256,
        "dslSource": policy.to_source(),
    }


def source_table_for_unit() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policies.jsonl"
        path.write_text(
            __import__("json").dumps(policy_record(SIMPLE_SWAP_LEFT_POLICY)) + "\n"
            + __import__("json").dumps(policy_record(SIMPLE_SWAP_RIGHT_POLICY)) + "\n",
            encoding="utf-8",
        )
        return build_policy_source_table(generated_policy_library=path, qd_candidates=pd.DataFrame())


def unit_inputs() -> dict[str, pd.DataFrame]:
    left = parse_policy(SIMPLE_SWAP_LEFT_POLICY)
    right = parse_policy(SIMPLE_SWAP_RIGHT_POLICY)
    embeddings = pd.DataFrame(
        [
            {
                "policy_id": left.policy_id,
                "policy_name": left.name,
                "source_kind": "unit",
                "route": "cpu_fallback",
                "screen_score": 0.9,
                "screen_heldout_final_sortedness_mean": 0.85,
                "screen_heldout_improvement_mean": 0.2,
                "screen_heldout_work_mean": 5.0,
                "qd_candidate": False,
                "s08_archive_winner": False,
                "classic_landmark": True,
                "classic_family": "Bubble",
                "embedding_x": 0.0,
                "embedding_y": 1.0,
                "embedding_z": 0.0,
            },
            {
                "policy_id": right.policy_id,
                "policy_name": right.name,
                "source_kind": "unit",
                "route": "cpu_fallback",
                "screen_score": 0.8,
                "screen_heldout_final_sortedness_mean": 0.75,
                "screen_heldout_improvement_mean": 0.1,
                "screen_heldout_work_mean": 4.0,
                "qd_candidate": True,
                "s08_archive_winner": True,
                "classic_landmark": False,
                "classic_family": "",
                "embedding_x": 1.0,
                "embedding_y": 0.0,
                "embedding_z": 0.0,
            },
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "policy_id": left.policy_id,
                "class_id": "UC01",
                "cautious_label": "unit left",
                "cluster_distance": 0.1,
                "dsl_excerpt": SIMPLE_SWAP_LEFT_POLICY,
                "rule_count": 2,
            },
            {
                "policy_id": right.policy_id,
                "class_id": "UC02",
                "cautious_label": "unit right",
                "cluster_distance": 0.2,
                "dsl_excerpt": SIMPLE_SWAP_RIGHT_POLICY,
                "rule_count": 2,
            },
        ]
    )
    frontier = pd.DataFrame(
        [
            {
                "policy_id": right.policy_id,
                "curated_rank": 1,
                "s14_candidate_status": "validated_frontier_candidate",
                "selection_score": 0.75,
                "selection_reasons_json": "[\"unit\"]",
                "pareto_frontier_labels_json": "[\"screen_pareto\"]",
                "s14_mean_final_sortedness": 0.7,
                "s14_perturbation_mean_final_sortedness": 0.65,
            }
        ]
    )
    exemplars = pd.DataFrame(
        [
            {
                "policy_id": left.policy_id,
                "exemplar_role": "classic_landmark",
                "manual_inspection_status": "unit",
            }
        ]
    )
    cluster_summary = pd.DataFrame(
        [
            {"class_id": "UC01", "policy_count": 1, "mean_screen_score": 0.9, "cautious_label": "unit left", "label_basis": "unit"},
            {"class_id": "UC02", "policy_count": 1, "mean_screen_score": 0.8, "cautious_label": "unit right", "label_basis": "unit"},
        ]
    )
    classic_analysis = pd.DataFrame(
        [
            {
                "algorithm_label": "Bubble",
                "classic_position_classification": "central",
                "classification_confidence": "unit",
            }
        ]
    )
    return {
        "embeddings": embeddings,
        "clusters": clusters,
        "source_table": source_table_for_unit(),
        "frontier_candidates": frontier,
        "exemplars": exemplars,
        "cluster_summary": cluster_summary,
        "classic_analysis": classic_analysis,
        "phase_boundaries": pd.DataFrame(),
        "ablation_claims": pd.DataFrame(),
    }


class PolicyAtlasTests(unittest.TestCase):
    def test_atlas_index_links_source_class_frontier_and_classic_context(self) -> None:
        index = build_atlas_index(**unit_inputs())
        self.assertEqual(len(index), 2)
        self.assertTrue(index["has_full_dsl_source"].all())
        self.assertEqual(int(index["is_frontier_candidate"].sum()), 1)
        classic = index[index["classic_landmark"]].iloc[0]
        self.assertEqual(classic["classic_position_classification"], "central")
        self.assertIn("policy-dsl-", classic["atlas_anchor"])

    def test_representative_trace_frame_is_deterministic_and_json_loads(self) -> None:
        index = build_atlas_index(**unit_inputs())
        ids = representative_policy_ids(index, max_count=4)
        traces = representative_trace_frame(index, ids, config=AtlasTraceConfig(event_cap=8, checkpoints=(0, 1, 2, 4, 8)))
        marked = mark_trace_availability(index, traces)
        self.assertGreaterEqual(len(traces), 2)
        self.assertTrue(marked["trace_available"].all())
        snapshots = __import__("json").loads(traces.iloc[0]["snapshots_json"])
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertIn("trace_digest", traces.columns)

    def test_rendered_html_contains_policy_data_and_validation_rows(self) -> None:
        inputs = unit_inputs()
        index = build_atlas_index(**inputs)
        traces = representative_trace_frame(index, representative_policy_ids(index, max_count=4), config=AtlasTraceConfig(event_cap=8, checkpoints=(0, 1, 2, 4, 8)))
        index = mark_trace_availability(index, traces)
        digest = atlas_digest(index, traces)
        html_text = render_atlas_html(
            index=index,
            traces=traces,
            cluster_summary=inputs["cluster_summary"],
            classic_analysis=inputs["classic_analysis"],
            artifact_links={"index": "unit_index.csv"},
            figure_links={},
            digest=digest,
        )
        self.assertIn("E03 Policy Morphospace Atlas", html_text)
        self.assertIn(index.iloc[0]["policy_id"], html_text)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            html_path = tmp_path / "atlas.html"
            summary_path = tmp_path / "summary.md"
            handoff_path = tmp_path / "handoff.md"
            artifact_path = tmp_path / "unit_index.csv"
            figure_path = tmp_path / "figure.png"
            html_path.write_text(html_text, encoding="utf-8")
            summary_path.write_text("summary" * 200, encoding="utf-8")
            handoff_path.write_text("handoff" * 200, encoding="utf-8")
            artifact_path.write_text("x\n", encoding="utf-8")
            figure_path.write_bytes(b"png")
            validation = validation_frame(
                index=index,
                traces=traces,
                html_path=html_path,
                summary_path=summary_path,
                handoff_path=handoff_path,
                artifact_paths={"index": artifact_path},
                figure_paths={"figure": figure_path},
                expected_policy_count=2,
                expected_frontier_count=1,
                unit_success=True,
            )
        cases = dict(zip(validation["validation_case"], validation["success"], strict=True))
        self.assertTrue(cases["all_policy_ids_indexed"])
        self.assertTrue(cases["trace_snapshots_load"])
        self.assertTrue(cases["atlas_html_contains_frontier_ids"])


if __name__ == "__main__":
    unittest.main()
