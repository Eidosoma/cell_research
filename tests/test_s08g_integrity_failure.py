from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_s08g_integrity_failure as forensics


def test_cache_commitment_is_filename_order_independent(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"row":1}\n', encoding="utf-8")
    second.write_text('{"row":2}\n', encoding="utf-8")
    assert forensics._cache_commitment([first, second]) == forensics._cache_commitment(
        [second, first]
    )


def test_descriptor_forensics_fails_closed_on_complete_out_of_bounds_panel(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        forensics,
        "E05_REGENERATION_DESCRIPTOR_SPECS",
        {"phenotype:test": {"fields": ("coordinate",), "censorFields": (), "kind": "test"}},
    )
    monkeypatch.setattr(
        forensics,
        "_descriptor_sets",
        lambda _task, outcome: [
            {
                "archiveId": "phenotype:test",
                "kind": "test",
                "values": {"coordinate": outcome["coordinate"]},
            }
        ],
    )
    monkeypatch.setattr(
        forensics,
        "_edges_for",
        lambda _kind, _fields: {"coordinate": (0.0, 1.0)},
    )
    rows = [
        {
            "taskId": "e07_s02_regeneration_1d",
            "configurationId": "configuration",
            "scenarioFamilyOrdinal": family,
            "scenarioId": f"scenario-{family}",
            "stopReason": "phase_event_budget",
            "failed": False,
            "censored": True,
            "outcome": {"coordinate": -2.0},
        }
        for family in range(300, 304)
    ]
    result = forensics._descriptor_forensics(rows)
    assert result["success"] is False
    assert result["supportStateCounts"] == {"complete": 1}
    assert result["boundViolationCount"] == 1
    assert result["violations"][0]["mean"] == -2.0
    assert result["archiveConstructionAuthorized"] is False
    assert result["imputationApplied"] is False


def test_write_json_is_canonical_readable(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    forensics._write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").endswith("\n")
