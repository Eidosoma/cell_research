"""S12X binding for the byte-frozen S12T actual-use dependency control.

The S12T registry remains byte-for-byte unchanged and governs dependency
imports, module origins, structural-input opens, caches, and signals.  S12X
adds only an authenticated output ledger for its newly authorized cache and
artifact roots; that ledger cannot authorize any dependency input.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.spatial_transfer.dependency_preflight import (
    DependencyPreflightError,
    domain_hash,
    sha256_file,
)


class OutputAccessViolation(DependencyPreflightError):
    """An S12X output operation escaped its exact authorized roots."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class OutputAccessRegistry:
    """Deny-first registry that can authorize only S12X-owned output paths."""

    def __init__(self, raw: Mapping[str, Any], *, source: Path) -> None:
        self.raw = deepcopy(dict(raw))
        self.source = source.resolve()
        if (
            self.raw.get("schemaVersion") != "e07.s12x.output-access-registry.v1"
            or self.raw.get("researchStepId") != "S12X"
        ):
            raise DependencyPreflightError("unexpected S12X output registry")
        self.base_dependency_registry_sha256 = str(
            self.raw["baseDependencyRegistrySha256"]
        )
        self.allowed_roots = tuple(
            Path(item).resolve(strict=False) for item in self.raw["allowedRoots"]
        )
        self.prohibited_roots = tuple(
            Path(item).resolve(strict=False) for item in self.raw["prohibitedRoots"]
        )
        if set(self.allowed_roots) & set(self.prohibited_roots):
            raise DependencyPreflightError("output allow/deny roots overlap")
        self.registry_sha256 = sha256_file(self.source)

    @classmethod
    def load(cls, path: Path) -> OutputAccessRegistry:
        return cls(
            json.loads(path.read_text(encoding="utf-8")),
            source=path,
        )

    def path_disposition(
        self,
        requested: os.PathLike[str] | str,
    ) -> dict[str, Any]:
        lexical = Path(os.path.abspath(os.fspath(requested)))
        resolved = lexical.resolve(strict=False)
        if any(_is_within(resolved, root) for root in self.prohibited_roots):
            allowed = False
            reason = "prohibited_path_prefix"
        elif any(_is_within(resolved, root) for root in self.allowed_roots):
            allowed = True
            reason = "authorized_s12x_output_root"
        else:
            allowed = False
            reason = "not_an_s12x_output_path"
        return {
            "allowed": allowed,
            "reason": reason,
            "requestedPath": os.fspath(requested),
            "lexicalPath": str(lexical),
            "resolvedPath": str(resolved),
        }


class OutputAccessLedger:
    """Authenticated manual ledger for cache/artifact persistence operations."""

    def __init__(
        self,
        registry: OutputAccessRegistry,
        *,
        session_commitment: str,
    ) -> None:
        self.registry = registry
        self.session_commitment = session_commitment
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def authorize_path(
        self,
        requested: os.PathLike[str] | str,
        *,
        operation: str,
        artifact_class: str,
    ) -> dict[str, Any]:
        disposition = self.registry.path_disposition(requested)
        with self._lock:
            event = {
                "ordinal": len(self.events),
                "eventKind": "s12x_output_access",
                "operation": operation,
                "artifactClass": artifact_class,
                **disposition,
            }
            event["eventSha256"] = domain_hash(
                "E07/S12X/output-event/v1",
                event,
            )
            self.events.append(event)
        if not disposition["allowed"]:
            raise OutputAccessViolation(
                f"denied S12X output access: {disposition['reason']}: "
                f"{disposition['resolvedPath']}"
            )
        return deepcopy(event)

    def finalize(self) -> dict[str, Any]:
        events = deepcopy(self.events)
        ledger_sha = domain_hash("E07/S12X/output-ledger/v1", events)
        authenticator = domain_hash(
            "E07/S12X/output-ledger-authenticator/v1",
            {
                "registrySha256": self.registry.registry_sha256,
                "baseDependencyRegistrySha256": (
                    self.registry.base_dependency_registry_sha256
                ),
                "sessionCommitment": self.session_commitment,
                "ledgerSha256": ledger_sha,
                "eventCount": len(events),
            },
        )
        return {
            "schemaVersion": "e07.s12x.output-access-ledger.v1",
            "researchStepId": "S12X",
            "registrySha256": self.registry.registry_sha256,
            "baseDependencyRegistrySha256": (
                self.registry.base_dependency_registry_sha256
            ),
            "sessionCommitment": self.session_commitment,
            "eventCount": len(events),
            "events": events,
            "ledgerSha256": ledger_sha,
            "authenticatorSha256": authenticator,
        }


def validate_output_ledger(
    raw: Mapping[str, Any] | None,
    registry: OutputAccessRegistry,
    *,
    expected_session_commitment: str,
) -> dict[str, Any]:
    if raw is None:
        return {
            "passed": False,
            "reasons": ["missing_ledger"],
            "eventCount": 0,
        }
    ledger = deepcopy(dict(raw))
    reasons: list[str] = []
    if ledger.get("schemaVersion") != "e07.s12x.output-access-ledger.v1":
        reasons.append("schema_mismatch")
    if ledger.get("registrySha256") != registry.registry_sha256:
        reasons.append("registry_commitment_mismatch")
    if (
        ledger.get("baseDependencyRegistrySha256")
        != registry.base_dependency_registry_sha256
    ):
        reasons.append("base_registry_commitment_mismatch")
    if ledger.get("sessionCommitment") != expected_session_commitment:
        reasons.append("session_commitment_mismatch")
    events = ledger.get("events")
    if not isinstance(events, list):
        return {
            "passed": False,
            "reasons": [*reasons, "events_not_list"],
            "eventCount": 0,
        }
    for ordinal, event in enumerate(events):
        body = deepcopy(event)
        claimed = body.pop("eventSha256", None)
        if body.get("ordinal") != ordinal:
            reasons.append(f"noncontiguous_ordinal:{ordinal}")
        if claimed != domain_hash("E07/S12X/output-event/v1", body):
            reasons.append(f"event_hash_mismatch:{ordinal}")
        disposition = registry.path_disposition(str(event.get("resolvedPath", "")))
        if event.get("allowed") is not True or not disposition["allowed"]:
            reasons.append(f"denied_or_forged_event:{ordinal}")
    ledger_sha = domain_hash("E07/S12X/output-ledger/v1", events)
    if ledger.get("ledgerSha256") != ledger_sha:
        reasons.append("ledger_hash_mismatch")
    authenticator = domain_hash(
        "E07/S12X/output-ledger-authenticator/v1",
        {
            "registrySha256": registry.registry_sha256,
            "baseDependencyRegistrySha256": (registry.base_dependency_registry_sha256),
            "sessionCommitment": expected_session_commitment,
            "ledgerSha256": ledger_sha,
            "eventCount": len(events),
        },
    )
    if ledger.get("authenticatorSha256") != authenticator:
        reasons.append("authenticator_mismatch")
    if ledger.get("eventCount") != len(events):
        reasons.append("event_count_mismatch")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "eventCount": len(events),
        "ledgerSha256": ledger_sha,
        "authenticatorSha256": authenticator,
    }
