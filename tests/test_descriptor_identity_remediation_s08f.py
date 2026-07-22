from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import yaml

from src.environment_suite.contracts import canonical_sha256
from src.portfolio_preregistration.core import canonical_hash, read_jsonl
from src.portfolio_search.execution import (
    S08P,
    _expand_logical_result,
    _physical_key,
)
from src.quality_diversity.core import (
    E05_DESCRIPTOR_AVAILABILITY_VERSION,
    E05_REGENERATION_DESCRIPTOR_SPECS,
    _aggregate_descriptors,
)


E05 = "e07_s02_regeneration_1d"
KNOWN_CHIMERA_IDS = {
    "611e01a8a56e5ac1398c382177e8619fa16709b40917a9b1a315c7b66b34f203",
    "ffe9d0be2c0e470f942238621b1a619d28f9d7e0fd51ac3f75f4f2af9073afd2",
}


def _movement(value: float = 0.25) -> dict[str, float]:
    return {
        "acceptedNativeActionFraction": value,
        "committedDisplacementFraction": value / 2,
    }


def _complete_outcome() -> dict:
    return {
        "initialDistance": 4,
        "finalDistance": 2,
        "sourceTerminal": False,
        "nativeMovementDescriptorsByPhase": {
            phase: _movement(0.2 + index * 0.01)
            for index, phase in enumerate(
                (
                    "development",
                    "stabilization",
                    "recovery",
                    "memoryResetRecovery",
                    "robustnessFault",
                    "transfer",
                )
            )
        },
        "descriptorsByAxis": {
            "robustness": {
                "pairedCompletionDelta": 0.0,
                "pairedResidualDelta": 0.1,
            },
            "repair": {
                "distanceRestorationFraction": 0.5,
                "restrictedRecoveryTimeFraction": 0.6,
                "recoveryCensored": True,
            },
            "memory": {
                "historyInterventionEffect": 0.0,
                "resetInterventionEffect": 0.0,
            },
            "transfer": {
                "frozenStratumSuccessFraction": 1.0,
                "frozenStratumResidual": 0.0,
            },
        },
    }


def _terminal_outcome(branch: str) -> dict:
    phases = {"development": _movement()}
    if branch == "stabilization_source_terminal":
        phases["stabilization"] = _movement(0.3)
    return {
        "sourceTerminal": True,
        "nativeMovementDescriptorsByPhase": phases,
        "descriptorsByAxis": {},
    }


def _row(family: int, outcome: dict, branch: str) -> dict:
    terminal = branch != "completed_panel"
    return {
        "taskId": E05,
        "scenarioFamilyOrdinal": family,
        "scenarioId": f"qualification:{family}",
        "stopReason": "source_terminal" if terminal else "completed_panel",
        "failed": terminal,
        "censored": branch in {"development_source_terminal", "completed_panel"},
        "outcome": outcome,
    }


def _logical(config: dict, slot: str) -> dict:
    return {
        "stage": "qualification",
        "generation": 0,
        "logicalSlotId": slot,
        "reservedConfigurationSlotId": config["configurationId"],
        "configurationRole": config["mode"],
        "scenarioFamilyOrdinal": 300,
        "split": "train",
        "configuration": config,
    }


def test_protocol_freezes_outcome_free_fail_closed_scope() -> None:
    protocol = yaml.safe_load(
        Path("configs/portfolio/s08f_descriptor_identity_remediation.yaml").read_text()
    )
    assert protocol["schemaVersion"] == "e07.s08f.descriptor-identity-remediation.v2"
    assert protocol["scope"]["qualificationOnly"] is True
    assert protocol["scope"]["freshFrozenSmokeRows"] == 0
    assert protocol["scope"]["efficacyRows"] == 0
    assert protocol["scope"]["archiveConstructionCalls"] == 0
    assert protocol["scope"]["validationOutcomeAccesses"] == 0
    assert protocol["scope"]["confirmationOutcomeAccesses"] == 0
    assert set(protocol["e05DescriptorAvailability"]["expectedArchiveIds"]) == set(
        E05_REGENERATION_DESCRIPTOR_SPECS
    )


def test_complete_e05_panel_yields_only_comparable_task_local_cells() -> None:
    rows = [_row(family, _complete_outcome(), "completed_panel") for family in range(4)]
    descriptors = _aggregate_descriptors(E05, rows)
    assert len(descriptors) == 10
    assert all(item["supportState"] == "complete" for item in descriptors)
    assert all(item["cellEligible"] for item in descriptors)
    assert all(
        item["values"] is not None and item["cell"] is not None for item in descriptors
    )
    assert all(
        item["descriptorAvailabilityVersion"] == E05_DESCRIPTOR_AVAILABILITY_VERSION
        for item in descriptors
    )
    assert all(item["comparabilityKeySha256"] for item in descriptors)
    assert not any(item["availabilityIsNoveltyCoordinate"] for item in descriptors)


def test_mixed_native_terminals_are_retained_without_imputation_or_silent_drop() -> (
    None
):
    rows = [
        _row(
            0,
            _terminal_outcome("development_source_terminal"),
            "development_source_terminal",
        ),
        _row(
            1,
            _terminal_outcome("stabilization_source_terminal"),
            "stabilization_source_terminal",
        ),
        _row(2, _complete_outcome(), "completed_panel"),
        _row(3, _complete_outcome(), "completed_panel"),
    ]
    descriptors = _aggregate_descriptors(E05, rows)
    by_id = {item["archiveId"]: item for item in descriptors}
    assert len(by_id) == len(E05_REGENERATION_DESCRIPTOR_SPECS)
    assert by_id[f"common:{E05}:development"]["cellEligible"] is True
    for archive_id, item in by_id.items():
        if archive_id == f"common:{E05}:development":
            continue
        assert item["supportState"] == "unavailable"
        assert item["cellEligible"] is False
        assert item["values"] is None
        assert item["cell"] is None
        assert item["comparabilityKeySha256"] is None
        assert "missing_native_phase_or_axis_support" in item["availabilityReasonCodes"]
        assert len(item["support"]["nativeStatusByScenarioFamily"]) == 4
        assert item["support"]["sourceTerminalRows"] == 2
        assert item["support"]["failedRows"] == 2
        assert item["support"]["censoredRows"] == 3


def test_known_chimera_pair_is_not_legally_deduplicated_when_commitments_differ() -> (
    None
):
    configs = [
        row
        for row in read_jsonl(S08P / "portfolio_seed_registry.jsonl")
        if row["configurationId"] in KNOWN_CHIMERA_IDS
    ]
    assert len(configs) == 2
    keys = {
        _physical_key(_logical(config, f"slot-{index}"))
        for index, config in enumerate(configs)
    }
    assert len(keys) == 2
    assert {config["members"][0]["nativeCarrier"] for config in configs} == {
        "Bubble",
        "Insertion",
    }
    assert (
        len(
            {
                json.dumps(config["portfolioStructuralCosts"], sort_keys=True)
                for config in configs
            }
        )
        == 2
    )


def test_legal_dedup_restores_each_logical_identity_and_keeps_physical_provenance() -> (
    None
):
    base = next(
        row
        for row in read_jsonl(S08P / "portfolio_seed_registry.jsonl")
        if row["taskId"] == "e07_s02_chimera_1d" and row["mode"] == "single_policy"
    )
    alias = deepcopy(base)
    alias["configurationId"] = canonical_hash(
        "E07/S08F/qualification-alias/v1", base["configurationId"]
    )
    alias["memberSetId"] = canonical_hash(
        "E07/S08F/qualification-member-set/v1", base["memberSetId"]
    )
    first_logical = _logical(base, "logical-a")
    second_logical = _logical(alias, "logical-b")
    first_key = _physical_key(first_logical)
    second_key = _physical_key(second_logical)
    assert first_key == second_key
    physical = {
        "stableEvaluationSha256": "1" * 64,
        "configurationId": base["configurationId"],
        "configurationDefinitionSha256": canonical_hash(
            "E07/S08C/configuration-definition/v1", base
        ),
    }
    first = _expand_logical_result(first_logical, physical, first_key)
    second = _expand_logical_result(second_logical, physical, second_key)
    assert first["configurationId"] == base["configurationId"]
    assert second["configurationId"] == alias["configurationId"]
    assert first["memberSetId"] == base["memberSetId"]
    assert second["memberSetId"] == alias["memberSetId"]
    assert (
        first["physicalStableEvaluationSha256"]
        == second["physicalStableEvaluationSha256"]
    )
    assert first["physicalWorkSha256"] == second["physicalWorkSha256"]
    assert first["stableEvaluationSha256"] != second["stableEvaluationSha256"]
    assert first["logicalExpansionSha256"] != second["logicalExpansionSha256"]


def test_selector_and_cost_semantic_changes_prevent_deduplication() -> None:
    single = next(
        row
        for row in read_jsonl(S08P / "portfolio_seed_registry.jsonl")
        if row["taskId"] == "e07_s02_chimera_1d" and row["mode"] == "single_policy"
    )
    changed_cost = deepcopy(single)
    changed_cost["configurationId"] = "a" * 64
    changed_cost["portfolioStructuralCosts"]["totalRuleCount"] += 1
    assert _physical_key(_logical(single, "cost-a")) != _physical_key(
        _logical(changed_cost, "cost-b")
    )

    conditioned = next(
        row
        for row in read_jsonl(S08P / "portfolio_seed_registry.jsonl")
        if row["taskId"] == E05 and row["mode"] == "environment_conditioned"
    )
    changed_selector = deepcopy(conditioned)
    changed_selector["selector"] = dict(changed_selector["selector"])
    changed_selector["selector"]["signal"] = (
        "repair.nudge_count"
        if changed_selector["selector"]["signal"] == "last_action.rejected"
        else "last_action.rejected"
    )
    changed_selector["configurationId"] = "b" * 64
    assert _physical_key(_logical(conditioned, "selector-a")) != _physical_key(
        _logical(changed_selector, "selector-b")
    )


def test_logical_expansion_hash_commits_to_restored_configuration() -> None:
    config = next(iter(read_jsonl(S08P / "portfolio_seed_registry.jsonl")))
    logical = _logical(config, "logical")
    physical = {
        "stableEvaluationSha256": canonical_sha256("qualification", {"row": 1}),
        "configurationId": config["configurationId"],
        "configurationDefinitionSha256": "2" * 64,
    }
    expanded = _expand_logical_result(logical, physical, _physical_key(logical))
    assert expanded["configurationId"] == logical["reservedConfigurationSlotId"]
    assert expanded["configurationRole"] == config["mode"]
    assert expanded["configurationDefinitionSha256"] == canonical_hash(
        "E07/S08C/configuration-definition/v1", config
    )
    assert expanded["physicalConfigurationDefinitionSha256"] == "2" * 64
