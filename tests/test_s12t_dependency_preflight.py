from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from src.spatial_transfer.dependency_preflight import (
    AuthenticatedAccessLedger,
    DependencyRegistry,
    DependencyViolation,
    StaticImportAnalyzer,
    audit_runtime_dispatch,
    canonical_round_trip,
    cleanup_modules,
    domain_hash,
    instrument_path_opens,
    validate_access_ledger,
)

REPOSITORY = Path(__file__).resolve().parents[1]
REGISTRY = REPOSITORY / "configs/transfer/s12t_dependency_registry.json"
PROTOCOL = REPOSITORY / "configs/transfer/s12t_dependency_preflight_protocol.yaml"
TEST_ROOT = Path("/cache/e07-s12t-qualification/pytest")


@pytest.fixture()
def root() -> Path:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True)
    yield TEST_ROOT
    cleanup_modules(("s12t_test_allowed_module", "s12t_test_forbidden_module"))
    shutil.rmtree(TEST_ROOT, ignore_errors=True)


@pytest.fixture()
def registry() -> DependencyRegistry:
    return DependencyRegistry.load(REGISTRY)


def write_source(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.write_text(source.strip() + "\n", encoding="utf-8")
    return path


def test_protocol_and_registry_authorize_zero_scientific_work(
    registry: DependencyRegistry,
) -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["researchStepId"] == "S12T"
    assert protocol["authorizationBoundary"]["transferEpisodes"] == 0
    assert protocol["authorizationBoundary"]["scientificExecutionNamespaces"] == []
    assert protocol["historicalBoundary"]["patchesOrRetriesS12S"] is False
    assert (
        protocol["historicalBoundary"]["createsS12RScientificExecutionNamespace"]
        is False
    )
    assert registry.raw["scientificBoundary"]["transferEpisodes"] == 0
    assert (
        registry.raw["scientificBoundary"]["scientificExecutionNamespacesAuthorized"]
        == []
    )


def test_registry_is_deny_first_and_literals_are_not_operations(
    registry: DependencyRegistry,
    root: Path,
) -> None:
    source = write_source(
        root,
        "literal.py",
        """
FORBIDDEN = "src.surrogate_models /artifacts/research_steps/S07"
VALUE = len(FORBIDDEN)
""",
    )
    result = StaticImportAnalyzer(
        registry,
        search_roots=(REPOSITORY, root),
    ).analyze([source])
    assert result["passed"]
    assert result["violationCount"] == 0
    assert registry.module_disposition("src.surrogate_models")[0] is False
    assert registry.module_disposition("src.spatial_transfer.execution")[0] is True


@pytest.mark.parametrize(
    ("name", "source", "kind"),
    [
        (
            "direct.py",
            "import src.surrogate_models",
            "direct_import",
        ),
        (
            "dynamic.py",
            'import importlib\nimportlib.import_module("src.surrogate_remediation")',
            "dynamic_import",
        ),
        (
            "alias.py",
            (
                "from importlib import import_module as loader\n"
                'loader("src.allocation_redesign")'
            ),
            "dynamic_import",
        ),
        (
            "builtin.py",
            '__import__("src.non_surrogate_allocation")',
            "dynamic_import",
        ),
        (
            "wrapper.py",
            (
                "from importlib import import_module as loader\n"
                "def dispatch(name):\n"
                "    return loader(name)\n"
                'dispatch("src.surrogate_models")'
            ),
            "dynamic_import",
        ),
        (
            "unresolved.py",
            'import importlib\nname = "json"\nimportlib.import_module(name)',
            "unresolved_dynamic_import",
        ),
    ],
)
def test_forbidden_and_unresolved_import_operations_fail_closed(
    registry: DependencyRegistry,
    root: Path,
    name: str,
    source: str,
    kind: str,
) -> None:
    entry = write_source(root, name, source)
    result = StaticImportAnalyzer(
        registry,
        search_roots=(REPOSITORY, root),
    ).analyze([entry])
    assert not result["passed"]
    assert kind in {row["kind"] for row in result["violations"]}


def test_indirect_forbidden_import_is_resolved(
    registry: DependencyRegistry,
    root: Path,
) -> None:
    entry = write_source(root, "entry.py", "import helper")
    write_source(root, "helper.py", "from src import surrogate_remediation")
    result = StaticImportAnalyzer(
        registry,
        search_roots=(REPOSITORY, root),
    ).analyze([entry])
    assert not result["passed"]
    assert result["filesAnalyzed"] >= 2
    assert any(
        row.get("module") == "src" and row["sourcePath"].endswith("helper.py")
        for row in result["imports"]
    )
    assert any(
        row.get("module") == "src.surrogate_remediation" for row in result["violations"]
    )


def test_access_ledger_authenticates_allowed_open_and_denies_forbidden(
    registry: DependencyRegistry,
    root: Path,
) -> None:
    allowed = root / "allowed.txt"
    allowed.write_text("ok\n", encoding="utf-8")
    session = domain_hash("E07/S12T/test-session/v1", "allowed")
    ledger = AuthenticatedAccessLedger(registry, session_commitment=session)
    with instrument_path_opens(ledger):
        assert allowed.read_text(encoding="utf-8") == "ok\n"
    raw = ledger.finalize()
    assert validate_access_ledger(
        raw,
        registry,
        expected_session_commitment=session,
    )["passed"]

    denied_session = domain_hash("E07/S12T/test-session/v1", "denied")
    denied = AuthenticatedAccessLedger(
        registry,
        session_commitment=denied_session,
    )
    with pytest.raises(DependencyViolation), instrument_path_opens(denied):
        open(  # noqa: SIM115 - denial occurs before a handle exists
            "/artifacts/research_steps/S06/models/forbidden.bin", "rb"
        )
    denied_result = validate_access_ledger(
        denied.finalize(),
        registry,
        expected_session_commitment=denied_session,
    )
    assert not denied_result["passed"]
    assert "denied_event_present:0" in denied_result["reasons"]


def test_symlink_and_dotdot_normalization_fail_closed(
    registry: DependencyRegistry,
    root: Path,
) -> None:
    link = root / "link"
    link.symlink_to("/artifacts/research_steps/S06/models/forbidden.bin")
    result = registry.path_disposition(link)
    assert not result["allowed"]
    assert result["reason"] == "prohibited_path_prefix"
    dotdot = (
        root
        / ".."
        / ".."
        / ".."
        / "artifacts"
        / "research_steps"
        / "S07"
        / "signal.json"
    )
    result = registry.path_disposition(dotdot)
    assert not result["allowed"]
    assert result["resolvedPath"].endswith("/artifacts/research_steps/S07/signal.json")


def test_cache_signal_missing_and_forged_ledgers_fail_closed(
    registry: DependencyRegistry,
) -> None:
    session = domain_hash("E07/S12T/test-session/v1", "ledger")
    ledger = AuthenticatedAccessLedger(registry, session_commitment=session)
    with pytest.raises(DependencyViolation):
        ledger.authorize_cache("/cache/e07-s10h/work/outcomes.parquet")
    with pytest.raises(DependencyViolation):
        ledger.authorize_signal("S07.armMembership")
    raw = ledger.finalize()
    assert not validate_access_ledger(
        raw,
        registry,
        expected_session_commitment=session,
    )["passed"]
    assert not validate_access_ledger(
        None,
        registry,
        expected_session_commitment=session,
    )["passed"]

    valid = AuthenticatedAccessLedger(registry, session_commitment=session)
    valid.authorize_signal("qualification.structural")
    valid_raw = valid.finalize()
    assert validate_access_ledger(
        valid_raw,
        registry,
        expected_session_commitment=session,
    )["passed"]
    forged = deepcopy(valid_raw)
    forged["events"][0]["signal"] = "S07.forged"
    assert not validate_access_ledger(
        forged,
        registry,
        expected_session_commitment=session,
    )["passed"]
    forged_auth = deepcopy(valid_raw)
    forged_auth["authenticatorSha256"] = "0" * 64
    assert not validate_access_ledger(
        forged_auth,
        registry,
        expected_session_commitment=session,
    )["passed"]
    round_trip = canonical_round_trip(valid_raw)
    assert round_trip["passed"]
    assert round_trip["decoded"] == valid_raw


def test_runtime_module_provenance_allows_fixture_and_denies_preloaded_origin(
    registry: DependencyRegistry,
    root: Path,
) -> None:
    name = "s12t_test_allowed_module"
    write_source(root, f"{name}.py", "VALUE = 9")
    sys.path.insert(0, str(root))
    cleanup_modules((name,))
    try:
        allowed = audit_runtime_dispatch(
            lambda: importlib.import_module(name).VALUE,
            registry,
        )
    finally:
        sys.path.remove(str(root))
        cleanup_modules((name,))
    assert allowed["passed"]
    assert allowed["inspectedModuleCount"] == 1

    forbidden_name = "s12t_test_forbidden_module"
    module = ModuleType(forbidden_name)
    module.__file__ = "/artifacts/research_steps/S06/models/forbidden.py"
    sys.modules[forbidden_name] = module
    try:
        forbidden = audit_runtime_dispatch(
            lambda: True,
            registry,
            inspect_preloaded_names=(forbidden_name,),
        )
    finally:
        cleanup_modules((forbidden_name,))
    assert not forbidden["passed"]
    assert forbidden["violationCount"] == 1


def test_historical_s12s_source_passes_actual_use_analysis(
    registry: DependencyRegistry,
) -> None:
    result = StaticImportAnalyzer(
        registry,
        search_roots=(REPOSITORY,),
    ).analyze(
        [
            REPOSITORY / "scripts/execute_replacement_spatial_transfer_s12s.py",
            REPOSITORY / "src/spatial_transfer/execution.py",
        ]
    )
    assert result["passed"], json.dumps(result["violations"], indent=2)
    assert result["filesAnalyzed"] >= 2


def test_os_open_is_instrumented_and_denied_before_kernel_open(
    registry: DependencyRegistry,
) -> None:
    session = domain_hash("E07/S12T/test-session/v1", "os-open")
    ledger = AuthenticatedAccessLedger(registry, session_commitment=session)
    with pytest.raises(DependencyViolation), instrument_path_opens(ledger):
        os.open(
            "/artifacts/research_steps/S07/arm_signal.json",
            os.O_RDONLY,
        )
    raw = ledger.finalize()
    assert raw["eventCount"] == 1
    assert raw["events"][0]["sourceApi"] == "os.open"
