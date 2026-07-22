from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import run_quality_diversity_s05 as search_script
from src.environment_suite import EnvironmentSuite, Split, SuiteValidationError
from src.quality_diversity.core import (
    BASE_SCENARIOS,
    SPLIT_MANIFEST,
    TASK_REGISTRY,
    aggregate_candidate,
    archive_insert,
    compatible_with_task,
    counter_u64,
    derive_train_record,
    evaluate_work_item,
    finite_behavior_deduplicate,
    load_seed_records,
    mutate_policy,
    policy_body_sha256,
    validate_gate_and_inputs,
)


PROTOCOL = "configs/search/s05_qd_protocol.yaml"


@pytest.fixture(scope="module")
def seeds():
    rows = load_seed_records()
    return {row["policyId"]: row for row in rows}


def test_live_gate_and_all_frozen_hashes_pass():
    result = validate_gate_and_inputs(PROTOCOL)
    assert result["success"] is True
    assert set(result["gateRows"]) == {
        "G01_protected_split_access",
        "G02_line_E01_E04_arbitrary_DSL_episode_binding",
        "G03_E05_arbitrary_DSL_binding",
        "G04_E06_arbitrary_DSL_binding",
        "G05_objective_descriptor_extraction",
        "G06_deterministic_replay_and_leakage",
    }
    assert result["validationOutcomeEvaluations"] == 0
    assert result["confirmationOutcomeEvaluations"] == 0


def test_counter_randomness_is_addressed_not_call_order():
    forward = [counter_u64("task", 2, index, stream="mutation") for index in range(20)]
    reverse = {
        index: counter_u64("task", 2, index, stream="mutation")
        for index in reversed(range(20))
    }
    assert forward == [reverse[index] for index in range(20)]
    assert forward != [
        counter_u64("task", 2, index, stream="parent") for index in range(20)
    ]


def test_train_resampling_is_deterministic_and_rejects_protected_e05_ordinals():
    suite = EnvironmentSuite(TASK_REGISTRY, SPLIT_MANIFEST)
    task = "e07_s02_regeneration_1d"
    base = suite.records[BASE_SCENARIOS[task]]
    assert derive_train_record(base, 2) == derive_train_record(base, 2)
    assert derive_train_record(base, 2).split is Split.TRAIN
    with pytest.raises(SuiteValidationError, match="protected replicate"):
        derive_train_record(base, 4)


def test_mutations_compile_and_never_expand_permissions(seeds):
    parent = seeds["bubble_cell_view_v1"]["canonicalPolicy"]
    generated = []
    for operator in (
        "delete_rule",
        "rotate_rule_order",
        "duplicate_rule",
        "flip_comparison",
        "nudge_literal",
        "terminal_noop",
        "flip_relative_offset",
    ):
        for address in range(12):
            result = mutate_policy(parent, operator, address=address)
            if result is not None:
                generated.append(result[0])
                assert result[0]["permissions"] == parent["permissions"]
                assert policy_body_sha256(result[0]) != policy_body_sha256(parent)
                break
    assert {item["environment"] for item in generated} == {"line1d.v1"}
    assert len(generated) >= 5


def test_task_compatibility_is_typed_not_field_name_inference(seeds):
    line = seeds["bubble_cell_view_v1"]["canonicalPolicy"]
    spatial = seeds["spatial_memory_repair_v1"]["canonicalPolicy"]
    assert compatible_with_task("e07_s02_sorting_1d", line)
    assert not compatible_with_task("e07_s02_sorting_1d", spatial)
    assert compatible_with_task("e07_s02_spatial2d_memory", spatial)
    assert not compatible_with_task("e07_s02_spatial2d_memory", line)


def _synthetic_aggregate(policy_hash: str, quality: float, cell: int = 0):
    return {
        "policySha256": policy_hash,
        "validNativeEpisodes": True,
        "qualityMinimization": {"performance.metric": quality},
        "descriptors": [{"archiveId": "common:task", "cell": [cell]}],
        "complexityVector": {
            "ruleCount": 1.0,
            "expressionNodes": 1.0,
            "actionCount": 1.0,
            "persistentMemoryBits": 0.0,
            "outboundSignalBits": 0.0,
            "worstCaseOperations": 1.0,
            "canonicalBytes": 10.0,
        },
        "boundedSemanticSha256": f"semantic-{policy_hash}",
    }


def test_pareto_archive_is_order_independent():
    rows = [
        _synthetic_aggregate("c", 3.0, 1),
        _synthetic_aggregate("a", 1.0, 0),
        _synthetic_aggregate("b", 2.0, 0),
    ]
    archives = []
    for order in (rows, list(reversed(rows))):
        archive = {}
        for row in order:
            archive_insert(archive, row)
        archives.append(
            {
                key: sorted(item["policySha256"] for item in front)
                for key, front in archive.items()
            }
        )
    assert archives[0] == archives[1]
    assert archives[0][("common:task", (0,))] == ["a"]


def test_failed_policy_is_retained_as_evidence_but_not_inserted():
    failed = _synthetic_aggregate("failed", 0.0)
    failed["validNativeEpisodes"] = False
    archive = {}
    stats = archive_insert(archive, failed)
    assert archive == {}
    assert stats["rejected"] == 1


def test_generation_plan_freeze_replays_authoritative_restart_decision(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(search_script, "PLAN_CACHE", tmp_path)
    monkeypatch.setattr(search_script, "protocol_hash", lambda: "protocol")
    offspring = [
        {
            "policyId": "p",
            "policySha256": "canonical",
            "policyBodySha256": "structural",
            "parents": ["parent"],
            "mutation": {"operatorId": "delete_rule"},
            "document": {"policyId": "p", "environment": "line1d.v1"},
        }
    ]
    _, first_matches = search_script.freeze_generation_plan("task", 1, offspring)
    resolved, replay_matches = search_script.freeze_generation_plan(
        "task", 1, offspring
    )
    assert first_matches and replay_matches
    assert resolved[0]["policySha256"] == "canonical"
    changed = deepcopy(offspring)
    changed[0]["policySha256"] = "different"
    resolved, replay_matches = search_script.freeze_generation_plan("task", 1, changed)
    assert replay_matches is False
    assert resolved[0]["policySha256"] == "canonical"


def test_finite_semantic_dedup_keeps_lower_complexity_then_hash():
    a = _synthetic_aggregate("a", 1.0)
    b = deepcopy(a)
    b["policySha256"] = "b"
    b["complexityVector"] = dict(a["complexityVector"])
    b["complexityVector"]["ruleCount"] = 2.0
    retained, audit = finite_behavior_deduplicate([b, a])
    assert [row["policySha256"] for row in retained] == ["a"]
    assert audit[0]["discardedPolicySha256"] == ["b"]
    assert audit[0]["finiteTrainingSupportOnly"] is True


def test_aggregation_keeps_licensed_costs_separate(seeds):
    seed = seeds["insertion_cell_view_v1"]
    document = seed["canonicalPolicy"]
    policy = {
        "policyId": seed["policyId"],
        "policySha256": seed["policySha256"],
        "policyBodySha256": policy_body_sha256(document),
        "document": document,
        "origin": "frozen_s03_seed",
        "generation": 0,
    }
    row = {
        "taskId": "e07_s02_sorting_1d",
        "scenarioOrdinal": 0,
        "failed": False,
        "censored": False,
        "replayPass": True,
        "validation": {"exactReplay": True},
        "outcome": {
            "completed": True,
            "derived": {"finalNormalizedKendallDistance": 0.0},
            "descriptors": {
                "acceptedNativeActionFraction": 0.2,
                "committedDisplacementFraction": 0.1,
            },
        },
        "nativeLedgerFamilies": {
            "dslRuntimeLedger": {
                "licensedPrefixPredicateEvaluations": 5,
                "licensedPrefixValueReads": 8,
                "licensedPrefixValueComparisons": 3,
            }
        },
        "semanticProjectionSha256": "semantic",
        "stableEvaluationSha256": "evaluation",
    }
    aggregate = aggregate_candidate(policy, [row])
    assert (
        aggregate["costVector"]["dslRuntimeLedger.licensedPrefixPredicateEvaluations"]
        == 5
    )
    assert (
        aggregate["qualityMinimization"][
            "cost.dslRuntimeLedger.licensedPrefixValueReads"
        ]
        == 8
    )


def test_e06_common_and_phenotype_descriptors_remain_separate(seeds):
    seed = seeds["spatial_greedy_adjacent_only_v1"]
    document = seed["canonicalPolicy"]
    policy = {
        "policyId": seed["policyId"],
        "policySha256": seed["policySha256"],
        "policyBodySha256": policy_body_sha256(document),
        "document": document,
        "origin": "frozen_s03_seed",
        "generation": 0,
    }
    row = {
        "taskId": "e07_s02_spatial2d_local",
        "scenarioOrdinal": 0,
        "failed": False,
        "censored": True,
        "replayPass": True,
        "validation": {"exactReplay": True},
        "outcome": {
            "conjunctiveCompletion": False,
            "s01GlobalCompletionAudit": {"normalizedMismatch": 0.25},
            "s02LocalGrammar": {"accepted": True},
            "repairCompletion": False,
            "descriptors": {
                "acceptedNativeActionFraction": 0.2,
                "committedDisplacementFraction": 0.1,
                "committedMovementKindEntropy": 0.5,
                "stateTurnoverFraction": 0.25,
            },
        },
        "nativeLedgerFamilies": {"e06MovementLedger": {"actions": 1}},
        "semanticProjectionSha256": "semantic-spatial",
        "stableEvaluationSha256": "evaluation-spatial",
    }
    aggregate = aggregate_candidate(policy, [row])
    by_archive = {item["archiveId"]: item for item in aggregate["descriptors"]}
    assert by_archive["common:e07_s02_spatial2d_local"]["fields"] == [
        "acceptedNativeActionFraction",
        "committedDisplacementFraction",
    ]
    assert by_archive["phenotype:e06:e07_s02_spatial2d_local"]["fields"] == [
        "committed_movement_kind_entropy",
        "state_turnover_fraction",
    ]


def test_spatial_counter_schedule_resample_replays_and_changes_address(seeds):
    document = seeds["spatial_greedy_adjacent_only_v1"]["canonicalPolicy"]
    by_id = {key: row["canonicalPolicy"] for key, row in seeds.items()}
    faithful = {
        "Bubble": by_id["bubble_cell_view_v1"],
        "Insertion": by_id["insertion_cell_view_v1"],
    }
    base_work = {
        "taskId": "e07_s02_spatial2d_local",
        "document": document,
        "faithfulDocuments": faithful,
        "stage": "test",
    }
    first = evaluate_work_item({**base_work, "scenarioOrdinal": 0})
    second = evaluate_work_item({**base_work, "scenarioOrdinal": 1})
    assert first["replayPass"] and second["replayPass"]
    assert (
        first["provenance"]["baseNativeScenarioId"]
        == second["provenance"]["baseNativeScenarioId"]
    )
    assert (
        first["provenance"]["nativeScenarioId"]
        != second["provenance"]["nativeScenarioId"]
    )
    assert first["split"] == second["split"] == "train"
