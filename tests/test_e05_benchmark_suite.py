from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from morphospace2d import (
    BENCHMARK_SUITE_SCHEMA_VERSION,
    REQUIRED_BENCHMARK_IDS,
    BenchmarkConfig,
    artifact_link_rows,
    assign_benchmark_membership,
    benchmark_suite_validation_rows,
    dg_caveat_preservation_audit,
    dry_run_benchmark_configs,
    information_flag_preservation_audit,
    standard_benchmark_configs,
)
from scripts.e05_s15_benchmark_suite import build_s06_stable_run_ids


def _canonical_rows() -> pd.DataFrame:
    base = {
        "run_id": "run",
        "trajectory_id": "trajectory",
        "target_id": "gradient_x_5x4",
        "task_id": "",
        "motif": "gradient",
        "policy_id": "classic",
        "policy_family": "classic_derived",
        "final_error": 0.2,
        "relative_error_reduction": 0.4,
        "information_scope": "local_target_map_free",
        "control_class": "local_only",
        "is_local_only_policy": True,
        "is_global_information_baseline": False,
        "uses_target_map": False,
        "uses_global_gradient": False,
        "uses_organizer": False,
        "dg_event_count": 1,
        "productive_dg_event_count": 1,
        "any_dg_event": True,
        "any_productive_dg_event": True,
        "normalization_artifact_flag": False,
        "productive_normalization_artifact_flag": False,
        "dg_caveat": "DG caveat preserved.",
        "normalization_warning": "Normalization warning preserved.",
        "total_productive_dg_index": 0.5,
        "exact_success": False,
        "perturbation_id": "",
        "perturbation_family": "",
    }
    rows: list[dict[str, object]] = []
    specs = [
        ("S06", "sorted_row", "", "", "", 300),
        ("S08", "gradient", "", "remove_center_chunk", "contiguous_chunk_removal", 90),
        ("S14", "gradient", "", "", "", 1935),
    ]
    for source, motif, task, perturbation, family, count in specs:
        for idx in range(count):
            row = dict(base)
            row.update(
                {
                    "source_step_id": source,
                    "run_id": f"{source}::run::{idx}",
                    "trajectory_id": f"{source}::trajectory::{idx}",
                    "motif": motif,
                    "task_id": task,
                    "perturbation_id": perturbation,
                    "perturbation_family": family,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


class TestE05BenchmarkSuite(unittest.TestCase):
    def test_standard_configs_cover_required_benchmarks(self) -> None:
        configs = standard_benchmark_configs("/tmp/artifacts")

        self.assertEqual(set(config.benchmark_id for config in configs), set(REQUIRED_BENCHMARK_IDS))
        self.assertTrue(all(config.expected_min_rows > 0 for config in configs))
        self.assertTrue(all(config.required_artifact_paths for config in configs))
        self.assertEqual({config.to_dict()["schemaVersion"] for config in configs}, {BENCHMARK_SUITE_SCHEMA_VERSION})

    def test_dry_run_and_audits_preserve_ids_flags_and_dg_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "source.parquet"
            artifact.write_text("placeholder", encoding="utf-8")
            configs = [
                BenchmarkConfig(
                    benchmark_id="sort_row",
                    title="Sort Row",
                    benchmark_family="baseline",
                    description="test",
                    primary_goal="test",
                    selectors=({"source_step_id": ("S06",)},),
                    expected_min_rows=300,
                    required_columns=(
                        "benchmark_id",
                        "source_step_id",
                        "run_id",
                        "trajectory_id",
                        "motif",
                        "policy_id",
                        "policy_family",
                        "information_scope",
                        "control_class",
                        "is_local_only_policy",
                        "is_global_information_baseline",
                        "uses_target_map",
                        "uses_global_gradient",
                        "uses_organizer",
                        "dg_event_count",
                        "productive_dg_event_count",
                        "any_dg_event",
                        "any_productive_dg_event",
                        "normalization_artifact_flag",
                        "productive_normalization_artifact_flag",
                        "dg_caveat",
                        "normalization_warning",
                    ),
                    required_artifact_paths=(str(artifact),),
                    metrics=("target_error",),
                    caveats=("caveat",),
                    validation_notes=("note",),
                    report_tags=("tag",),
                )
            ]
            packaged = assign_benchmark_membership(_canonical_rows(), configs)
            dry = dry_run_benchmark_configs(configs, packaged)
            links = artifact_link_rows(configs)
            info = information_flag_preservation_audit(packaged)
            dg = dg_caveat_preservation_audit(packaged)
            validation = benchmark_suite_validation_rows(configs, packaged, dry, links, info, dg, [artifact])

        self.assertTrue(dry["dry_run_success"].all(), dry.to_string(index=False))
        self.assertTrue(links["link_validation_success"].all())
        self.assertTrue(info["audit_success"].all(), info.to_string(index=False))
        self.assertTrue(dg["audit_success"].all(), dg.to_string(index=False))
        selected = validation[~validation["check_id"].eq("required_benchmark_configs_present")]
        self.assertTrue(selected["success"].all(), validation.to_string(index=False))

    def test_s06_stable_run_ids_do_not_collapse_when_primary_ids_missing(self) -> None:
        s06 = pd.DataFrame(
            {
                "condition_id": ["", ""],
                "source_condition_id": ["", ""],
                "algorithm": ["bubble", "odd_even"],
                "replicate_index": [0, 1],
                "input_permutation_seed": [11, 12],
                "scheduler_seed": [21, 22],
                "tie_breaker_seed": [31, 32],
            }
        )

        run_ids = build_s06_stable_run_ids(s06)

        self.assertEqual(run_ids.nunique(), 2)
        self.assertTrue(run_ids.str.len().gt(0).all())
        self.assertIn("bubble::rep0::input11::sched21::tie31", set(run_ids))


if __name__ == "__main__":
    unittest.main(verbosity=2)
