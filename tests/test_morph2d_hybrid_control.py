from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from scripts.build_morph2d_s12 import (
    exploratory_contrasts,
    exploratory_tasks,
    factorial_interactions,
    pareto_frontier,
    select_confirmation_blocks,
)

from src.morph2d.hybrid_control import (
    canonical_bytes,
    load_hybrid_catalog,
    make_challenge_state,
    run_hybrid_once,
    scenario_identity,
)


def _spec(arm_id: str, *, transitions: int = 17, replicate: int = 3):
    return {
        "phase": "unit",
        "split": "unit",
        "targetId": "layers_three_ordered_tissues",
        "challengeId": "partially_correct_formation",
        "armId": arm_id,
        "replicate": replicate,
        "eventBudget": transitions,
        "retainTrace": False,
    }


def test_catalog_freezes_five_arms_absolute_and_comparative_rules() -> None:
    catalog = load_hybrid_catalog()
    assert catalog["researchStepId"] == "S12"
    assert {item["armId"] for item in catalog["controlArms"]} == {
        "local_only",
        "central_only",
        "gradient_only",
        "sparse_direct",
        "combined",
    }
    assert catalog["outcomes"]["zeroTerminalRule"].startswith(
        "A comparative morphology effect remains comparative only"
    )
    assert catalog["confirmation"]["s12SupportRule"].startswith(
        "combinedComparativeRule AND combinedAbsoluteRule"
    )
    assert catalog["topologyAndFeasibility"]["countChangingConditions"] == ("forbidden")


def test_pairing_is_arm_blind_but_run_identity_is_arm_specific() -> None:
    identities = [
        scenario_identity(
            "exploratory",
            "layers_three_ordered_tissues",
            "compact_wound_mild",
            7,
            arm,
        )
        for arm in (
            "local_only",
            "central_only",
            "gradient_only",
            "sparse_direct",
            "combined",
        )
    ]
    assert len({item["scenarioId"] for item in identities}) == 1
    assert len({item["pairingBlockId"] for item in identities}) == 1
    assert len({item["seedHex"] for item in identities}) == 1
    assert len({item["runId"] for item in identities}) == 5


@pytest.mark.parametrize(
    "challenge_id",
    [
        "exact_maintenance",
        "partially_correct_formation",
        "compact_wound_mild",
        "formed_displacement_mild",
    ],
)
def test_challenges_preserve_counts_and_feasibility(challenge_id: str) -> None:
    state, mask, external, formed_source, feasible = make_challenge_state(
        "layers_three_ordered_tissues", challenge_id, "pb1:unit-challenge"
    )
    assert len(state.occupancy) == 81
    assert len({item.occupant_id for _, item in state.occupancy}) == 81
    assert feasible
    assert external["externalDisplacedEntities"] >= 0
    assert mask["challengeId"] == challenge_id
    assert formed_source is (challenge_id != "partially_correct_formation")


@pytest.mark.parametrize(
    "arm_id",
    [
        "local_only",
        "central_only",
        "gradient_only",
        "sparse_direct",
        "combined",
    ],
)
def test_all_arms_replay_and_reconcile_budgets(arm_id: str) -> None:
    first, _, first_mask = run_hybrid_once(_spec(arm_id))
    second, _, second_mask = run_hybrid_once(_spec(arm_id))
    deterministic_fields = {
        key: first[key] for key in first if key not in {"wallSeconds"}
    }
    repeated_fields = {key: second[key] for key in second if key not in {"wallSeconds"}}
    assert canonical_bytes(deterministic_fields) == canonical_bytes(repeated_fields)
    assert first_mask == second_mask
    assert first["budgetEnvelopeSuccess"]
    assert first["invariantSuccess"]
    assert first["permissionAuditSuccess"]
    assert first["zeroTerminalClaimBoundaryAcknowledged"]
    assert first["totalInformationBitsIncludingObservation"] == (
        first["observationCommunicatedBitsUpperBound"]
        + first["channelTotalInformationBits"]
    )
    assert (
        first["metricSummarySha256"]
        == hashlib.sha256(
            b"E06/S12/metric-summary/v1\x00"
            + canonical_bytes(
                {
                    key: first[key]
                    for key in sorted(first)
                    if key.startswith(
                        (
                            "initialS01",
                            "minimumS01",
                            "terminalS01",
                            "terminalS02",
                        )
                    )
                    or key
                    in {
                        "conjunctiveCompletionByBudget",
                        "firstCompletionTransition",
                        "terminalConjunctiveCompletion",
                        "localGlobalDiscordance",
                        "absoluteTerminalSuccess",
                    }
                }
            )
        ).hexdigest()
    )


def test_central_only_has_no_native_actuation_and_one_bounded_direct_query() -> None:
    central, _, _ = run_hybrid_once(_spec("central_only"))
    combined, _, _ = run_hybrid_once(_spec("combined"))
    assert central["usedNativeActorSlots"] == 0
    assert central["foregoneNativeActorSlots"] == 68
    assert central["directRecipientQuerySlots"] == 1
    assert central["actuationAttempts"] <= 1
    assert combined["usedNativeActorSlots"] == 64
    assert combined["directRecipientQuerySlots"] == 1
    assert combined["controllerInputBits"] > 0
    assert combined["controllerComputeUnits"] > 0


def test_gradient_delivery_respects_native_information_cap_with_one_actor() -> None:
    gradient, _, _ = run_hybrid_once(_spec("gradient_only", transitions=32))
    assert gradient["usedNativeActorSlots"] == 32
    assert gradient["foregoneNativeActorSlots"] == 96
    assert gradient["policyDeliveryBits"] <= 2 * 1600
    assert gradient["configurationBits"] == 649
    assert gradient["budgetEnvelopeSuccess"]


def test_nonexact_censoring_and_exact_maintenance_are_separate() -> None:
    formation, _, _ = run_hybrid_once(_spec("central_only", transitions=1))
    maintenance_spec = {
        **_spec("central_only", transitions=1),
        "challengeId": "exact_maintenance",
    }
    maintenance, _, _ = run_hybrid_once(maintenance_spec)
    assert formation["censored"] is (not formation["conjunctiveCompletionByBudget"])
    assert not maintenance["censored"]
    assert maintenance["initialConjunctiveCompletion"]
    assert (
        maintenance["absoluteTerminalSuccess"]
        == (maintenance["terminalConjunctiveCompletion"])
    )


def test_builder_expands_frozen_screen_before_holdout() -> None:
    catalog = load_hybrid_catalog()
    tasks = exploratory_tasks(catalog)
    assert len(tasks) == 40
    assert sum(len(item["replicates"]) for item in tasks) == 10_000
    assert len({item["conditionId"] for item in tasks}) == 40
    assert all(item["phase"] == "exploratory" for item in tasks)


def test_factorial_contrasts_interactions_and_pareto_keep_cost_axes_separate() -> None:
    catalog = load_hybrid_catalog()
    rows = []
    for target_id in catalog["scenarioFactorial"]["factors"]["target"]:
        for challenge_id in catalog["scenarioFactorial"]["factors"]["challenge"]:
            for replicate in range(3):
                for ordinal, arm_id in enumerate(
                    catalog["scenarioFactorial"]["factors"]["controlArm"]
                ):
                    row = {
                        "phase": "exploratory",
                        "targetId": target_id,
                        "challengeId": challenge_id,
                        "armId": arm_id,
                        "pairingBlockId": f"{target_id}:{challenge_id}:{replicate}",
                        "replicate": replicate,
                        "failed": False,
                        "terminalConjunctiveCompletion": bool(
                            arm_id == "combined" and replicate == 0
                        ),
                        "conjunctiveCompletionByBudget": bool(
                            arm_id == "combined" and replicate == 0
                        ),
                        "terminalS01MismatchFraction": 0.20 - 0.01 * ordinal,
                        "minimumS01MismatchFraction": 0.10 - 0.005 * ordinal,
                        "terminalS02RelationalScore": 0.50 + 0.01 * ordinal,
                    }
                    for cost in (
                        "totalInformationBitsIncludingObservation",
                        "totalGraphDisplacement",
                        "directMovementGraphDisplacement",
                        "externalInterventionGraphDisplacement",
                        "totalSourceWorkUnits",
                        "totalComputationUnits",
                        "totalOpportunityCostUnits",
                    ):
                        row[cost] = ordinal + 1
                    rows.append(row)
    frame = pd.DataFrame(rows)
    contrasts = exploratory_contrasts(frame, catalog)
    blocks, decision = select_confirmation_blocks(contrasts, catalog)
    interactions = factorial_interactions(frame, catalog)
    frontier = pareto_frontier(frame, catalog)
    assert len(contrasts) == 24
    assert blocks[0] == {
        "targetId": "layers_three_ordered_tissues",
        "challengeId": "partially_correct_formation",
    }
    assert decision["writtenBeforeConfirmationScenarioConstruction"]
    assert len(interactions) == 80
    assert len(frontier) == 40
    assert not frontier["authoritySemanticsCollapsed"].any()
