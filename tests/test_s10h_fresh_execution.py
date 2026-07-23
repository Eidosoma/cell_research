from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from scripts import run_native_event_discovery_s10h as s10h
from src.phenotype_discovery import publication


def _slot(
    slot_id: str, family: str, state: str = "feasible_but_unused_or_noncandidate"
) -> dict:
    evidentiary = state == "evidentiary_candidate"
    return {
        "slotId": slot_id,
        "methodFamily": family,
        "structurallyAdmissible": state != "non_evidentiary_infeasible",
        "slotState": state,
        "rawPValue": 0.001 if evidentiary else 1.0,
        "inMultiplicityFamily": True,
        "holmAdjustedPValue": 0.01 if evidentiary else 1.0,
        "multiplicityPass": evidentiary,
    }


def _projection_fixture():
    task, status = "spatial_2d_local", "completed"
    candidates = [f"candidate-{index:02d}" for index in range(14)]
    slots = [
        _slot(
            f"{task}::{status}::clustering::{index}",
            "clustering",
            "evidentiary_candidate" if index == 0 else (
                "non_evidentiary_infeasible"
                if index == 5
                else "feasible_but_unused_or_noncandidate"
            ),
        )
        for index in range(6)
    ]
    slots.append(
        _slot(
            f"{task}::{status}::anomaly::0",
            "anomaly",
            "evidentiary_candidate",
        )
    )
    slots.extend(
        _slot(
            f"{task}::{status}::change_point::{candidate}",
            "change_point",
            "evidentiary_candidate"
            if index == 0
            else "feasible_but_unused_or_noncandidate",
        )
        for index, candidate in enumerate(candidates)
    )
    machine = [
        {
            "holmSlotId": slots[0]["slotId"],
            "memberCandidateIds": candidates[:2],
        },
        {
            "holmSlotId": slots[6]["slotId"],
            "memberCandidateIds": [candidates[2]],
        },
        {
            "holmSlotId": slots[7]["slotId"],
            "memberCandidateIds": [candidates[0]],
        },
    ]
    catalog = {
        "machineCandidates": machine,
        "results": {
            "clustering": [
                {
                    "taskId": task,
                    "statusStratum": status,
                    "candidateIds": candidates,
                    "selectedWardK": 2,
                    "selectedWardLabels": [0, 0] + [1] * 12,
                    "kResults": {
                        "2": {"meanBootstrapARI": 0.91},
                    },
                    "nullMaxAri95": 0.4,
                }
            ],
            "anomaly": [
                {
                    "taskId": task,
                    "statusStratum": status,
                    "selectedConfigurationId": candidates[2],
                    "scores": {
                        candidate: float(index) / 14
                        for index, candidate in enumerate(candidates)
                    },
                    "medianScore": 0.45,
                }
            ],
            "changePoint": [
                {
                    "taskId": task,
                    "statusStratum": status,
                    "configurationId": candidate,
                    "lockedCenterTransition": 8,
                    "discoveryPrevalence": 0.75,
                }
                for candidate in candidates
            ],
        },
    }
    discovery = {
        "holmSlots": slots,
        "assessmentCount": 1,
        "fixedFamilyShrunk": False,
    }
    reproduction = {
        "fixedHolmSlots": [
            {
                **row,
                "slotState": "feasible_but_unused_or_noncandidate",
                "rawPValue": 1.0,
                "holmAdjustedPValue": 1.0,
                "multiplicityPass": False,
            }
            for row in slots
        ],
        "fixedFamilyShrunk": False,
    }
    return catalog, discovery, reproduction


def test_prospective_plan_and_config_are_frozen() -> None:
    config = yaml.safe_load(s10h.CONFIG.read_text(encoding="utf-8"))
    plan = Path(config["immutableInputs"]["researchPlanAfterProspectiveRegistration"]["path"])
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == config[
        "immutableInputs"
    ]["researchPlanAfterProspectiveRegistration"]["sha256"]
    assert config["researchStepId"] == "S10H"
    assert config["execution"]["cacheNamespace"] == "/cache/e07-s10h"
    assert config["publication"]["artifactClassCount"] == 20
    assert config["publication"]["completeSetOnly"] is True


def test_registry_is_exact_s10g_twenty_class_set() -> None:
    specs = s10h.publication_specs()
    assert len(specs) == 20
    assert len({spec.class_id for spec in specs}) == 20
    assert {spec.media_type for spec in specs} == {
        "json",
        "markdown",
        "parquet",
        "method_records_parquet",
    }


def test_generic_parquet_has_explicit_nonnullable_schema() -> None:
    payload = s10h._generic_parquet_bytes(
        [{"available": 1, "missing": None}, {"available": 2, "missing": None}],
        domain="E07/S10H/test/v1",
    )
    table = pq.read_table(s10h.BytesIO(payload))
    assert table.schema == s10h.GENERIC_SCHEMA
    assert table.num_rows == 2
    assert all(column.null_count == 0 for column in table.columns)
    restored = [
        publication.strict_json_loads(row["record_json"])
        for row in table.to_pylist()
    ]
    assert restored == [
        {"available": 1, "missing": None},
        {"available": 2, "missing": None},
    ]


def test_cache_only_mixed_projection_is_explicit() -> None:
    frame = pd.DataFrame(
        [
            {"taskId": "a", "scores": {"x": 0.2}},
            {"taskId": "b"},
        ]
    )
    projected = s10h.safe_intermediate_json_columns(frame, ["scores"])
    assert json.loads(projected.loc[0, "scores"]) == {"x": 0.2}
    assert json.loads(projected.loc[1, "scores"]) == {
        "reasonCodes": ["METHOD_FIELD_UNAVAILABLE"],
        "state": "unavailable",
        "value": None,
    }
    assert not projected["scores"].isna().any()


def test_method_projection_conserves_slots_and_round_trips() -> None:
    catalog, discovery, reproduction = _projection_fixture()
    projected = s10h.method_record_projection(
        catalog, discovery, reproduction
    )
    assert {key: len(value) for key, value in projected.items()} == {
        "clustering": 6,
        "anomaly": 1,
        "change_point": 14,
        "fixed_holm": 42,
    }
    all_ids = [
        row["recordId"] for records in projected.values() for row in records
    ]
    assert len(all_ids) == len(set(all_ids))
    for records in projected.values():
        payload = publication.records_to_parquet_bytes(records)
        assert publication.records_from_parquet_bytes(payload) == sorted(
            records, key=lambda row: row["recordId"]
        )


def test_ambiguous_or_nonfinite_values_fail_closed() -> None:
    for value in (np.nan, np.inf, pd.NA):
        try:
            s10h._generic_parquet_bytes(
                [{"bad": value}], domain="E07/S10H/adversary/v1"
            )
        except publication.SerializationContractError:
            pass
        else:  # pragma: no cover - safety assertion
            raise AssertionError(f"adversary was accepted: {value!r}")
