"""Helpers for E07 S15 periodic-table atlas publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PERIODIC_TABLE_SCHEMA_VERSION = "eidosoma.e07.periodic_table.v1"

REQUIRED_S15_ARTIFACT_KEYS = {
    "periodic_table_html",
    "periodic_table_summary",
    "final_corpus_manifest",
    "report_bundle_handoff",
    "full_results",
}

TOP_SUMMARY_FIELDS = [
    "Research step ID",
    "Completion status",
    "Artifacts written",
    "Validation result",
    "Outcome classification",
    "Caveats or blockers",
    "Lay summary",
    "Recommended next action",
]

REQUIRED_SOURCE_STEPS = {f"S{idx:02d}" for idx in range(1, 16)}


def sanitize_json(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible values."""

    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def stable_json(payload: Any) -> str:
    return json.dumps(sanitize_json(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []


def relative_href(target: Path, from_dir: Path) -> str:
    return os.path.relpath(Path(target).resolve(), start=Path(from_dir).resolve()).replace(os.sep, "/")


def report_relative_href(target: Path, reports_dir: Path) -> str:
    return os.path.relpath(Path(target).resolve(), start=Path(reports_dir).resolve()).replace(os.sep, "/")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_local_hrefs(html_text: str) -> list[str]:
    hrefs = re.findall(r"""href=["']([^"']+)["']""", html_text)
    local: list[str] = []
    for href in hrefs:
        lowered = href.lower()
        if lowered.startswith(("http://", "https://", "mailto:", "data:", "javascript:", "#")):
            continue
        local.append(href.split("#", 1)[0])
    return [href for href in local if href]


def resolve_report_href(href: str, html_path: Path) -> Path:
    return (html_path.parent / href).resolve()


def markdown_has_top_summary(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if "## Top Summary" not in text:
        return False
    summary_text = text.split("## Top Summary", 1)[1][:2500]
    return all(field in summary_text for field in TOP_SUMMARY_FIELDS)


def validation_summary(checks: pd.DataFrame) -> dict[str, Any]:
    if checks.empty or "success" not in checks:
        return {"passed": 0, "total": 0, "allPassed": False}
    return {
        "passed": int(checks["success"].sum()),
        "total": int(len(checks)),
        "allPassed": bool(checks["success"].all()),
    }


def validate_periodic_table_artifacts(
    *,
    manifest: Mapping[str, Any],
    html_path: Path,
    summary_path: Path,
    handoff_path: Path,
    full_results_path: Path,
    law_matrix: pd.DataFrame,
    source_manifest: pd.DataFrame,
    expected_claim_status_counts: Mapping[str, int],
    expected_unsupported_ids: Sequence[str],
) -> pd.DataFrame:
    """Validate S15 atlas packaging, provenance, and claim scoping."""

    checks: list[dict[str, Any]] = []

    def add(name: str, success: bool, detail: str) -> None:
        checks.append({"validation_case": name, "success": bool(success), "detail": detail})

    required_paths = {
        "periodic_table_html": html_path,
        "periodic_table_summary": summary_path,
        "final_corpus_manifest": Path(str(manifest.get("manifestPath", ""))) if manifest.get("manifestPath") else None,
        "report_bundle_handoff": handoff_path,
        "full_results": full_results_path,
    }
    missing_outputs = [
        key
        for key, path in required_paths.items()
        if path is None or not Path(path).exists() or (Path(path).is_file() and Path(path).stat().st_size == 0)
    ]
    add("required_s15_outputs_exist", not missing_outputs, f"missing_or_empty={missing_outputs}")

    add(
        "manifest_schema_and_corpus_version_recorded",
        manifest.get("schemaVersion") == PERIODIC_TABLE_SCHEMA_VERSION and bool(manifest.get("corpusVersion")),
        f"schema={manifest.get('schemaVersion')} corpusVersion={manifest.get('corpusVersion')}",
    )

    artifacts = list(manifest.get("artifacts", []))
    artifact_keys = {str(item.get("artifactKey", "")) for item in artifacts if isinstance(item, Mapping)}
    add(
        "manifest_required_s15_artifacts_listed",
        REQUIRED_S15_ARTIFACT_KEYS <= artifact_keys,
        f"missing={sorted(REQUIRED_S15_ARTIFACT_KEYS - artifact_keys)}",
    )

    bad_hash_entries = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            bad_hash_entries.append("<non-mapping>")
            continue
        path = Path(str(item.get("path", "")))
        checksum_omitted = bool(item.get("checksumOmittedReason"))
        if not path.exists():
            bad_hash_entries.append(str(item.get("artifactKey", path)))
        elif not checksum_omitted and not item.get("sha256"):
            bad_hash_entries.append(str(item.get("artifactKey", path)))
    add("manifest_artifact_hashes_present", not bad_hash_entries, f"bad_entries={bad_hash_entries[:20]}")

    html_text = html_path.read_text(encoding="utf-8") if html_path.exists() else ""
    unresolved = [href for href in extract_local_hrefs(html_text) if not resolve_report_href(href, html_path).exists()]
    add("atlas_links_resolve", not unresolved, f"unresolved={unresolved[:20]} total={len(unresolved)}")

    actual_counts = dict(manifest.get("s14ClaimStatusCounts", {}))
    add(
        "s14_claim_status_counts_preserved",
        {str(k): int(v) for k, v in expected_claim_status_counts.items()} == {str(k): int(v) for k, v in actual_counts.items()},
        f"expected={dict(expected_claim_status_counts)} actual={actual_counts}",
    )

    unsupported_ids = {str(item) for item in manifest.get("unsupportedSpeculationLawIds", [])}
    html_unsupported_ok = "unsupported_speculation" in html_text and all(str(law_id) in html_text for law_id in expected_unsupported_ids)
    add(
        "unsupported_speculation_labels_preserved",
        set(map(str, expected_unsupported_ids)) == unsupported_ids and html_unsupported_ok,
        f"expected_ids={list(expected_unsupported_ids)} manifest_ids={sorted(unsupported_ids)} html_ok={html_unsupported_ok}",
    )

    visible_scope_text = "\n".join(
        [
            html_text,
            summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
            handoff_path.read_text(encoding="utf-8") if handoff_path.exists() else "",
        ]
    ).lower()
    scope_ok = (
        "embedded-row" in visible_scope_text
        and "only" in visible_scope_text
        and "s13" in visible_scope_text
        and "do not generalize" in visible_scope_text
    )
    add("s13_embedded_row_only_scope_visible", scope_ok, "requires S13, embedded-row, only, and do not generalize text")

    claim_scope_ok = all(
        phrase in visible_scope_text
        for phrase in [
            "computational",
            "bounded",
            "not causal",
            "not biological",
        ]
    )
    add("claim_scoping_visible", claim_scope_ok, "requires computational/bounded/not causal/not biological language")

    markdown_paths = [summary_path, handoff_path, full_results_path]
    missing_summaries = [str(path) for path in markdown_paths if not markdown_has_top_summary(path)]
    add("markdown_top_summaries_present", not missing_summaries, f"missing={missing_summaries}")

    source_steps = {str(item) for item in manifest.get("sourceStepsRepresented", [])}
    add(
        "final_corpus_source_steps_covered",
        REQUIRED_SOURCE_STEPS <= source_steps,
        f"missing={sorted(REQUIRED_SOURCE_STEPS - source_steps)} represented={sorted(source_steps)}",
    )

    law_rows = int(manifest.get("s14LawRowCount", -1))
    add("s14_law_row_count_preserved", law_rows == len(law_matrix), f"manifest={law_rows} matrix={len(law_matrix)}")

    source_hash_ok = not source_manifest.empty and source_manifest.get("sha256", pd.Series(dtype=str)).fillna("").astype(str).ne("").all()
    add("source_manifest_hashes_present", bool(source_hash_ok), f"source_rows={len(source_manifest)}")

    return pd.DataFrame(checks)
