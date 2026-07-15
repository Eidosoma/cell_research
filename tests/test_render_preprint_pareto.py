from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.render_preprint_pareto import render_pareto, source_setting_label


def test_source_setting_label_preserves_continuation_terms() -> None:
    base = {
        "architecture": "distributed_local",
        "coordinatorProfile": "none",
        "scheduler": "random_permutation_sweep",
        "mobility": "passive",
    }
    continued = source_setting_label({**base, "continuation": "skip_and_continue"})
    stopped = source_setting_label(
        {**base, "continuation": "stop_on_first_blocking_failure"}
    )
    assert continued.endswith("skip-and-continue")
    assert stopped.endswith("stop-on-first-blocking-failure")


def test_renderer_labels_only_robustly_included_settings(tmp_path: Path) -> None:
    rows = []
    for index, robust in enumerate((True, True, False)):
        rows.append(
            {
                "treatmentSignature": f"t{index}",
                "architecture": "distributed_local",
                "coordinatorProfile": "none",
                "scheduler": (
                    "random_permutation_sweep"
                    if index == 0
                    else "uniform_random_activation"
                ),
                "mobility": "passive" if index == 0 else "stuck",
                "continuation": (
                    "skip_and_continue"
                    if index < 2
                    else "stop_on_first_blocking_failure"
                ),
                "costProfile": "s01Cost",
                "meanNormalizedResidual": 0.02 + index * 0.03,
                "failureRate": 0.60 + index * 0.10,
                "meanLogCost": 10.0 - index,
                "robustPareto": robust,
            }
        )
    source = tmp_path / "pareto.parquet"
    destination = tmp_path / "pareto.png"
    pd.DataFrame(rows).to_parquet(source, index=False)
    result = render_pareto(source, destination)
    assert destination.is_file() and destination.stat().st_size > 0
    assert result["primarySettingCount"] == 3
    assert result["robustlyIncludedCount"] == 2
    assert len(result["labeledSettings"]) == 2
    assert all("skip-and-continue" in label for label in result["labeledSettings"])
