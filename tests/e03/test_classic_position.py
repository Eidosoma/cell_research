"""S13 classic-position analysis tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.e03.classic_position import (
    CLASSIC_ALGORITHMS,
    algorithm_position_frame,
    classic_alignment_frame,
    classic_policy_position_frame,
    finite_feature_matrix,
    nearest_neighbors,
    neighbor_robustness_frame,
    validation_frame,
)


def s03_frame() -> pd.DataFrame:
    rows = []
    for algorithm in CLASSIC_ALGORITHMS:
        rows.append(
            {
                "policy_id": f"iface:{algorithm}",
                "algorithm": algorithm,
                "algorithm_label": algorithm.title(),
                "representation_type": "interface_wrapper",
                "exactness": "exact_public_method",
                "direction": "parameterized",
            }
        )
        for direction in ("increasing", "decreasing"):
            rows.append(
                {
                    "policy_id": f"dsl:{algorithm}_{direction}",
                    "algorithm": algorithm,
                    "algorithm_label": algorithm.title(),
                    "representation_type": "dsl",
                    "exactness": "exact_subset",
                    "direction": direction,
                }
            )
    return pd.DataFrame(rows)


def morphospace_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s03 = s03_frame()
    s04 = s03[["policy_id", "algorithm", "representation_type", "exactness", "direction"]].copy()
    s04["canonical_classic_baseline"] = s04["representation_type"] == "interface_wrapper"
    s04["final_sortedness_score"] = 1.0
    s04["dg_tendency_score"] = 0.5
    s04["aggregation_tendency_score"] = 0.1
    s04["competence_vector_hash"] = "unit"

    scores = {
        "bubble_increasing": (0.95, "UC01"),
        "bubble_decreasing": (0.05, "UC08"),
        "insertion_increasing": (0.70, "UC04"),
        "insertion_decreasing": (0.45, "UC06"),
        "selection_increasing": (0.25, "UC07"),
        "selection_decreasing": (0.35, "UC06"),
    }
    rows = []
    for name, (score, class_id) in scores.items():
        rows.append(
            {
                "policy_id": f"dsl:{name}",
                "policy_name": f"classic_{name}",
                "classic_landmark": True,
                "classic_family": name.split("_")[0].title(),
                "source_kind": "classic_dsl_seed",
                "screen_score": score,
                "screen_heldout_final_sortedness_mean": score,
                "screen_heldout_improvement_mean": score - 0.4,
                "screen_heldout_work_mean": 10.0,
                "embedding_x": score,
                "embedding_y": float(int(class_id.replace("UC", ""))),
                "embedding_z": 0.0,
                "class_id": class_id,
                "cautious_label": class_id,
                "label_confidence": "unit",
                "cluster_distance": 0.1,
                "qd_candidate": False,
                "s08_archive_winner": False,
                "f1": score,
                "f2": float(int(class_id.replace("UC", ""))),
            }
        )
    for idx in range(8):
        score = 0.92 - idx * 0.02
        rows.append(
            {
                "policy_id": f"p_core_{idx}",
                "policy_name": f"core_{idx}",
                "classic_landmark": False,
                "classic_family": "",
                "source_kind": "hand",
                "screen_score": score,
                "screen_heldout_final_sortedness_mean": score,
                "screen_heldout_improvement_mean": score - 0.4,
                "screen_heldout_work_mean": 10.0,
                "embedding_x": score,
                "embedding_y": 1.0,
                "embedding_z": 0.0,
                "class_id": "UC01",
                "cautious_label": "UC01",
                "label_confidence": "unit",
                "cluster_distance": 0.1,
                "qd_candidate": idx % 2 == 0,
                "s08_archive_winner": idx == 0,
                "f1": score,
                "f2": 1.0,
            }
        )
    for idx in range(8):
        score = 0.68 - idx * 0.01
        rows.append(
            {
                "policy_id": f"p_mid_{idx}",
                "policy_name": f"mid_{idx}",
                "classic_landmark": False,
                "classic_family": "",
                "source_kind": "random",
                "screen_score": score,
                "screen_heldout_final_sortedness_mean": score,
                "screen_heldout_improvement_mean": score - 0.4,
                "screen_heldout_work_mean": 10.0,
                "embedding_x": score,
                "embedding_y": 4.0,
                "embedding_z": 0.0,
                "class_id": "UC04",
                "cautious_label": "UC04",
                "label_confidence": "unit",
                "cluster_distance": 0.1,
                "qd_candidate": False,
                "s08_archive_winner": False,
                "f1": score,
                "f2": 4.0,
            }
        )
    s10 = pd.DataFrame(rows)
    s11 = s10[
        [
            "policy_id",
            "policy_name",
            "class_id",
            "cautious_label",
            "label_confidence",
            "cluster_distance",
            "classic_landmark",
            "classic_family",
        ]
    ].copy()
    s07 = s10[["policy_id", "policy_name", "source_kind", "screen_score"]].copy()
    s12 = pd.DataFrame(
        [
            {"original_policy_id": "dsl:bubble_increasing", "claim_type": "local_necessary_candidate", "heldout_delta_final_sortedness": -0.3},
            {"original_policy_id": "dsl:selection_increasing", "claim_type": "anti_feature_candidate", "heldout_delta_final_sortedness": 0.2},
        ]
    )
    return s04, s07, s10, s11, s12


class ClassicPositionTests(unittest.TestCase):
    def test_alignment_marks_interfaces_as_context_and_dsl_as_embedded(self) -> None:
        s04, s07, s10, s11, s12 = morphospace_frames()
        alignment = classic_alignment_frame(s03_frame(), s04, s07, s10, s11, s12)
        self.assertEqual(len(alignment), 9)
        self.assertTrue(alignment["s04_present"].all())
        self.assertEqual(int((alignment["representation_type"] == "dsl").sum()), 6)
        self.assertFalse(alignment[alignment["representation_type"] == "interface_wrapper"]["s10_present"].any())
        self.assertTrue(alignment[alignment["representation_type"] == "dsl"]["s10_present"].all())

    def test_nearest_neighbors_and_robustness_are_computed(self) -> None:
        _, _, s10, s11, _ = morphospace_frames()
        policy_table = s10.merge(s11[["policy_id", "class_id", "cautious_label"]], on="policy_id", how="left", suffixes=("", "_s11"))
        matrix = finite_feature_matrix(policy_table, ("f1", "f2")).matrix
        classic_ids = policy_table[policy_table["classic_landmark"]]["policy_id"].tolist()
        neighbors = pd.concat(
            [
                nearest_neighbors(policy_table, matrix, classic_ids, metric_space="normalized", k=20),
                nearest_neighbors(policy_table, matrix, classic_ids, metric_space="raw", k=20),
                nearest_neighbors(policy_table, matrix, classic_ids, metric_space="embedding", k=20),
            ],
            ignore_index=True,
        )
        robustness = neighbor_robustness_frame(neighbors)
        self.assertEqual(len(robustness), 6)
        self.assertTrue(np.isfinite(robustness["normalized_raw_top10_jaccard"]).all())
        first = neighbors[(neighbors["target_policy_id"] == "dsl:bubble_increasing") & (neighbors["metric_space"] == "normalized")].iloc[0]
        self.assertEqual(first["neighbor_class_id"], "UC01")

    def test_algorithm_classification_uses_requested_vocabulary(self) -> None:
        s04, s07, s10, s11, s12 = morphospace_frames()
        alignment = classic_alignment_frame(s03_frame(), s04, s07, s10, s11, s12)
        policy_table = s10.merge(s11[["policy_id", "class_id", "cautious_label"]], on="policy_id", how="left", suffixes=("", "_s11"))
        matrix = finite_feature_matrix(policy_table, ("f1", "f2")).matrix
        classic_ids = alignment[(alignment["representation_type"] == "dsl") & alignment["s10_present"]]["policy_id"].tolist()
        neighbors = pd.concat(
            [
                nearest_neighbors(policy_table, matrix, classic_ids, metric_space="normalized", k=20),
                nearest_neighbors(policy_table, matrix, classic_ids, metric_space="raw", k=20),
                nearest_neighbors(policy_table, matrix, classic_ids, metric_space="embedding", k=20),
            ],
            ignore_index=True,
        )
        robustness = neighbor_robustness_frame(neighbors)
        positions = classic_policy_position_frame(alignment, robustness, s12)
        algorithms = algorithm_position_frame(positions)
        labels = dict(zip(algorithms["algorithm"], algorithms["classic_position_classification"], strict=True))
        self.assertEqual(labels["bubble"], "central")
        self.assertEqual(labels["insertion"], "peripheral")
        self.assertEqual(labels["selection"], "accidental")
        validation = validation_frame(
            alignment=alignment,
            classic_positions=positions,
            algorithm_positions=algorithms,
            neighbors=neighbors,
            robustness=robustness,
            figure_exists=True,
            unit_success=True,
        )
        self.assertTrue(validation["success"].all(), validation.to_string(index=False))


if __name__ == "__main__":
    unittest.main()
