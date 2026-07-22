from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from src.environment_suite.e05_semantics import (
    TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION,
    TARGET_CHANGE_SEMANTICS_VERSION,
    build_target_change_audit_projection,
    validate_persisted_target_change_audit_projection,
    validate_target_change_result_semantics,
)


ADAPTATION_BUDGET = 6400
PROBE_BUDGET = 160
SOURCE_SHA = "a" * 64


def _source(
    stop: str,
    *,
    phase: int,
    hit: int | None,
    probe: int,
    retained: bool,
) -> dict:
    completed = hit is not None
    result = {
        "targetChangeSemanticsVersion": TARGET_CHANGE_SEMANTICS_VERSION,
        "stopReason": stop,
        "targetCompleted": completed,
        "phaseActivationCount": phase,
        "adaptationTime": hit,
        "adaptationCensored": not completed,
        "overshootCensored": not completed,
        "postHitProbeOpportunities": probe,
        "postHitProbeRetained": retained,
        "postHitProbeApplicable": completed,
    }
    result["targetChangeSemanticAudit"] = validate_target_change_result_semantics(
        result,
        adaptation_budget=ADAPTATION_BUDGET,
        probe_budget=PROBE_BUDGET,
    )
    return result


def _persisted(source: dict) -> dict:
    projection = build_target_change_audit_projection(
        source,
        adaptation_budget=ADAPTATION_BUDGET,
        probe_budget=PROBE_BUDGET,
        source_result_sha256=SOURCE_SHA,
    )
    projection = json.loads(json.dumps(projection, sort_keys=True))
    return {
        "native_event": {
            "resultSha256": SOURCE_SHA,
            "targetChangeAuditProjection": projection,
        },
        "native_outcome": {
            key: source[key]
            for key in (
                "targetCompleted",
                "phaseActivationCount",
                "adaptationTime",
                "adaptationCensored",
                "overshootCensored",
            )
        },
        "stop_reason": source["stopReason"],
        "validation": {
            "targetChangeSemanticContract": source["targetChangeSemanticAudit"][
                "validNativeContract"
            ],
            "postHitProbeRetained": source["postHitProbeRetained"],
        },
        "adaptation_budget": ADAPTATION_BUDGET,
        "probe_budget": PROBE_BUDGET,
    }


@pytest.mark.parametrize(
    "source",
    [
        _source(
            "post_adaptation_probe_complete",
            phase=260,
            hit=100,
            probe=160,
            retained=True,
        ),
        _source(
            "post_adaptation_probe_complete",
            phase=6560,
            hit=6400,
            probe=160,
            retained=True,
        ),
        _source("phase_event_budget", phase=6400, hit=None, probe=0, retained=True),
        _source("controller_quiescent", phase=320, hit=None, probe=0, retained=True),
    ],
)
def test_valid_branch_round_trip_matches_s08h(source: dict) -> None:
    original = deepcopy(source)
    result = validate_persisted_target_change_audit_projection(**_persisted(source))
    assert source == original
    assert result["projectionAuthentic"] is True
    assert result["validNativeContract"] is True
    assert result["validPersistedContract"] is True
    assert (
        result["classification"]
        == source["targetChangeSemanticAudit"]["classification"]
    )


@pytest.mark.parametrize(
    "source",
    [
        _source(
            "post_adaptation_probe_complete",
            phase=6561,
            hit=6401,
            probe=160,
            retained=True,
        ),
        _source("phase_event_budget", phase=6560, hit=6500, probe=60, retained=False),
        _source("invariant_error", phase=10, hit=None, probe=0, retained=False),
        _source("phase_event_budget", phase=6400, hit=6300, probe=100, retained=False),
    ],
)
def test_invalid_branch_is_authentic_but_fails_s08h(source: dict) -> None:
    result = validate_persisted_target_change_audit_projection(**_persisted(source))
    assert result["projectionAuthentic"] is True
    assert result["validNativeContract"] is False
    assert result["validPersistedContract"] is False
    assert result["classification"] == "adapter_failure"
    assert result["semanticErrors"]


def test_missing_inconsistent_and_forged_metadata_fail_closed() -> None:
    source = _source("phase_event_budget", phase=6400, hit=None, probe=0, retained=True)
    cases = []
    missing = _persisted(source)
    missing["native_event"] = {"resultSha256": SOURCE_SHA}
    cases.append(missing)

    missing_field = _persisted(source)
    del missing_field["native_event"]["targetChangeAuditProjection"]["probeBudget"]
    cases.append(missing_field)

    forged = _persisted(source)
    forged["native_event"]["targetChangeAuditProjection"]["projectionSha256"] = "f" * 64
    cases.append(forged)

    inconsistent = _persisted(source)
    inconsistent["native_outcome"]["phaseActivationCount"] = 6399
    cases.append(inconsistent)

    detached = _persisted(source)
    detached["native_event"]["resultSha256"] = "b" * 64
    cases.append(detached)

    for case in cases:
        result = validate_persisted_target_change_audit_projection(**case)
        assert result["projectionAuthentic"] is False
        assert result["validPersistedContract"] is False
        assert result["integrityErrors"]


def test_protocol_and_runner_fix_only_the_native_event_audit_plane() -> None:
    protocol = yaml.safe_load(
        Path("configs/portfolio/s08j_target_audit_projection.yaml").read_text()
    )
    assert protocol["researchStepId"] == "S08J"
    assert protocol["scope"]["qualificationOnly"] is True
    assert protocol["scope"]["episodeEvaluations"] == 0
    assert protocol["scope"]["frozenSmokeRows"] == 0
    assert protocol["scope"]["portfolioExecutionRows"] == 0
    assert protocol["scope"]["validationOutcomeAccesses"] == 0
    assert protocol["scope"]["confirmationOutcomeAccesses"] == 0
    assert protocol["scope"]["prohibitedCacheRoot"] == "/cache/e07-s08i"
    assert protocol["projection"]["schemaVersion"] == (
        TARGET_CHANGE_AUDIT_PROJECTION_SCHEMA_VERSION
    )
    assert protocol["projection"]["nativeOutcomeFieldsChanged"] is False
    source = Path("src/environment_suite/runners.py").read_text()
    assert '"targetChangeAuditProjection": target_audit_projection' in source
