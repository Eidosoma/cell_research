"""Outcome-independent actual-use dependency controls for S12T.

The control deliberately ignores inert string constants.  It reasons about
AST import operations, resolved module origins, and actual path/signal access
events.  Qualification and any later execution authorization remain separate.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import os
import sys
import sysconfig
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any


class DependencyPreflightError(RuntimeError):
    """Base fail-closed dependency-control error."""


class DependencyViolation(DependencyPreflightError):
    """An actual or unresolved dependency operation violated the registry."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def domain_hash(domain: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class DependencyRegistry:
    """Validated external allow/deny registry with deny-first precedence."""

    def __init__(self, raw: Mapping[str, Any], *, source: Path) -> None:
        self.raw = deepcopy(dict(raw))
        self.source = source.resolve()
        if (
            self.raw.get("schemaVersion") != "e07.s12t.dependency-registry.v1"
            or self.raw.get("researchStepId") != "S12T"
        ):
            raise DependencyPreflightError("unexpected S12T dependency registry")
        module = self.raw["modulePolicy"]
        path = self.raw["pathPolicy"]
        signal = self.raw["signalPolicy"]
        boundary = self.raw["scientificBoundary"]
        if (
            boundary.get("qualificationOnly") is not True
            or boundary.get("scientificExecutionNamespacesAuthorized") != []
            or any(
                int(boundary[field]) != 0
                for field in (
                    "transferEpisodes",
                    "validationOutcomeReads",
                    "confirmationOutcomeReads",
                    "protectedOutcomeReads",
                    "civicRows",
                    "s13Rows",
                    "s14Rows",
                    "efficacyRowsPublished",
                )
            )
        ):
            raise DependencyPreflightError("registry authorizes scientific work")
        self.allowed_third_party = frozenset(module["allowedThirdPartyTopLevels"])
        self.allowed_local = frozenset(module["allowedLocalTopLevels"])
        self.prohibited_modules = tuple(module["prohibitedModulePrefixes"])
        self.allowed_roots = tuple(
            Path(item).resolve(strict=False)
            for item in (
                path["allowedRepositoryRoots"] + path["allowedQualificationRoots"]
            )
        )
        self.prohibited_roots = tuple(
            Path(item).resolve(strict=False) for item in path["prohibitedPathPrefixes"]
        )
        self.allowed_exact = {
            str(Path(item).resolve(strict=False))
            for item in path["allowedExactStructuralFiles"]
        }
        self.allowed_manifest_files = tuple(
            Path(item).resolve(strict=False)
            for item in path["allowedEvidenceManifestFiles"]
        )
        self.manifest_bound: dict[str, dict[str, Any]] = {}
        self._load_manifest_bound_paths()
        self.prohibited_signals = tuple(signal["prohibitedSignalPrefixes"])
        self.stdlib_names = frozenset(sys.stdlib_module_names)
        self.stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=False)
        self.registry_sha256 = sha256_file(self.source)

    @classmethod
    def load(cls, path: Path) -> DependencyRegistry:
        return cls(
            json.loads(path.read_text(encoding="utf-8")),
            source=path,
        )

    def _load_manifest_bound_paths(self) -> None:
        for manifest_path in self.allowed_manifest_files:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            records: list[Mapping[str, Any]] = []
            if isinstance(raw.get("artifacts"), list):
                records.extend(raw["artifacts"])
            if isinstance(raw.get("inputs"), list):
                records.extend(raw["inputs"])
            if isinstance(raw.get("files"), dict):
                records.extend(raw["files"].values())
            self.allowed_exact.add(str(manifest_path))
            for row in records:
                if not isinstance(row, Mapping) or "path" not in row:
                    continue
                normalized = str(Path(str(row["path"])).resolve(strict=False))
                self.allowed_exact.add(normalized)
                self.manifest_bound[normalized] = {
                    key: row[key] for key in ("bytes", "sha256") if key in row
                }

    def module_disposition(self, name: str) -> tuple[bool, str]:
        normalized = name.strip(".")
        if not normalized:
            return False, "empty_module_name"
        if any(
            normalized == prefix or normalized.startswith(prefix + ".")
            for prefix in self.prohibited_modules
        ):
            return False, "prohibited_module_prefix"
        top = normalized.split(".", 1)[0]
        if (
            top in self.stdlib_names
            or top in self.allowed_third_party
            or top in self.allowed_local
        ):
            return True, "allowed_module_namespace"
        return False, "unregistered_module_namespace"

    def path_disposition(
        self,
        requested: os.PathLike[str] | str,
    ) -> dict[str, Any]:
        lexical = Path(os.path.abspath(os.fspath(requested)))
        resolved = lexical.resolve(strict=False)
        normalized = str(resolved)
        if any(_is_within(resolved, root) for root in self.prohibited_roots):
            return {
                "allowed": False,
                "reason": "prohibited_path_prefix",
                "requestedPath": os.fspath(requested),
                "lexicalPath": str(lexical),
                "resolvedPath": normalized,
                "manifestCommitment": self.manifest_bound.get(normalized),
            }
        if normalized in self.allowed_exact:
            allowed = True
            reason = "allowed_exact_or_manifest_bound_path"
        elif any(_is_within(resolved, root) for root in self.allowed_roots):
            allowed = True
            reason = "allowed_repository_or_qualification_root"
        else:
            allowed = False
            reason = "unregistered_path"
        return {
            "allowed": allowed,
            "reason": reason,
            "requestedPath": os.fspath(requested),
            "lexicalPath": str(lexical),
            "resolvedPath": normalized,
            "manifestCommitment": self.manifest_bound.get(normalized),
        }

    def signal_disposition(self, signal: str) -> tuple[bool, str]:
        if any(signal.startswith(prefix) for prefix in self.prohibited_signals):
            return False, "prohibited_signal_prefix"
        return True, "allowed_signal_namespace"

    def module_origin_disposition(
        self,
        name: str,
        origin: str | None,
    ) -> tuple[bool, str, str | None]:
        module_allowed, module_reason = self.module_disposition(name)
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in self.prohibited_modules
        ):
            return False, module_reason, origin
        if origin in {None, "built-in", "frozen"}:
            top = name.split(".", 1)[0]
            allowed = top in self.stdlib_names or module_allowed
            return allowed, "builtin_or_frozen" if allowed else module_reason, origin
        resolved = Path(origin).resolve(strict=False)
        if any(_is_within(resolved, root) for root in self.prohibited_roots):
            return False, "prohibited_module_origin", str(resolved)
        if _is_within(resolved, self.stdlib_root):
            return True, "stdlib_origin", str(resolved)
        path_result = self.path_disposition(resolved)
        if path_result["allowed"]:
            return True, "allowed_registered_origin", str(resolved)
        top = name.split(".", 1)[0]
        if top in self.allowed_third_party and "site-packages" in resolved.parts:
            return True, "allowed_third_party_origin", str(resolved)
        return False, "unregistered_module_origin", str(resolved)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _resolve_alias(name: str, aliases: Mapping[str, str]) -> str:
    head, *tail = name.split(".")
    replacement = aliases.get(head, head)
    return ".".join([replacement, *tail]) if tail else replacement


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class StaticImportAnalyzer:
    """AST/import-graph analyzer that ignores inert literal constants."""

    def __init__(
        self,
        registry: DependencyRegistry,
        *,
        search_roots: Sequence[Path],
    ) -> None:
        self.registry = registry
        self.search_roots = tuple(path.resolve(strict=False) for path in search_roots)

    def _module_path(self, module: str) -> Path | None:
        relative = Path(*module.split("."))
        for root in self.search_roots:
            module_file = root / relative.with_suffix(".py")
            if module_file.is_file():
                return module_file.resolve()
            package_file = root / relative / "__init__.py"
            if package_file.is_file():
                return package_file.resolve()
        return None

    @staticmethod
    def _module_for_path(path: Path, roots: Sequence[Path]) -> str | None:
        for root in roots:
            try:
                relative = path.resolve().relative_to(root)
            except ValueError:
                continue
            parts = list(relative.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            return ".".join(parts)
        return None

    @staticmethod
    def _absolute_relative_module(
        current_module: str | None,
        current_path: Path,
        node: ast.ImportFrom,
    ) -> str:
        if node.level == 0:
            return node.module or ""
        if not current_module:
            return node.module or ""
        package = current_module.split(".")
        if current_path.name != "__init__.py":
            package = package[:-1]
        remove = max(0, node.level - 1)
        if remove:
            package = package[:-remove]
        if node.module:
            package.extend(node.module.split("."))
        return ".".join(package)

    def analyze(self, entries: Sequence[Path]) -> dict[str, Any]:
        queue: list[tuple[Path, str | None]] = [
            (
                path.resolve(),
                self._module_for_path(path.resolve(), self.search_roots),
            )
            for path in entries
        ]
        visited: set[Path] = set()
        imports: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []

        while queue:
            path, current_module = queue.pop(0)
            if path in visited:
                continue
            visited.add(path)
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as error:
                violations.append(
                    {
                        "kind": "source_parse_failure",
                        "path": str(path),
                        "detail": f"{type(error).__name__}:{error}",
                    }
                )
                continue
            files.append(
                {
                    "path": str(path),
                    "bytes": len(source.encode("utf-8")),
                    "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                }
            )
            aliases: dict[str, str] = {}
            direct_nodes: list[tuple[str, ast.AST]] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local = alias.asname or alias.name.split(".", 1)[0]
                        aliases[local] = alias.name
                        direct_nodes.append((alias.name, node))
                elif isinstance(node, ast.ImportFrom):
                    base = self._absolute_relative_module(current_module, path, node)
                    for alias in node.names:
                        full = f"{base}.{alias.name}" if base else alias.name
                        aliases[alias.asname or alias.name] = full
                        if alias.name != "*":
                            _full_allowed, full_reason = (
                                self.registry.module_disposition(full)
                            )
                            if (
                                full_reason == "prohibited_module_prefix"
                                or self._module_path(full) is not None
                            ):
                                direct_nodes.append((full, node))
                    if base:
                        direct_nodes.append((base, node))
                elif isinstance(node, ast.Assign):
                    dotted = _dotted_name(node.value)
                    if dotted:
                        resolved = _resolve_alias(dotted, aliases)
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                aliases[target.id] = resolved

            wrappers: dict[str, int] = {}
            forwarding_call_nodes: set[int] = set()
            dynamic_names = {
                "importlib.import_module",
                "builtins.__import__",
                "__import__",
            }
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                parameters = [arg.arg for arg in node.args.args]
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.Call):
                        continue
                    dotted = _dotted_name(inner.func)
                    if not dotted:
                        continue
                    resolved = _resolve_alias(dotted, aliases)
                    if resolved not in dynamic_names or not inner.args:
                        continue
                    if isinstance(inner.args[0], ast.Name):
                        try:
                            wrappers[node.name] = parameters.index(inner.args[0].id)
                            forwarding_call_nodes.add(id(inner))
                        except ValueError:
                            pass

            for module, node in direct_nodes:
                allowed, reason = self.registry.module_disposition(module)
                resolved_path = self._module_path(module)
                if (
                    not allowed
                    and reason == "unregistered_module_namespace"
                    and resolved_path is not None
                ):
                    path_policy = self.registry.path_disposition(resolved_path)
                    allowed = bool(path_policy["allowed"])
                    reason = (
                        "allowed_resolved_local_fixture"
                        if allowed
                        else str(path_policy["reason"])
                    )
                record = {
                    "kind": "direct_import",
                    "sourcePath": str(path),
                    "line": getattr(node, "lineno", None),
                    "module": module,
                    "resolvedPath": str(resolved_path) if resolved_path else None,
                    "allowed": allowed,
                    "reason": reason,
                }
                imports.append(record)
                if not allowed:
                    violations.append(record)
                elif resolved_path is not None:
                    queue.append((resolved_path, module))

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if id(node) in forwarding_call_nodes:
                    continue
                dotted = _dotted_name(node.func)
                if not dotted:
                    continue
                resolved = _resolve_alias(dotted, aliases)
                dynamic = resolved in dynamic_names
                wrapper_index = wrappers.get(dotted)
                if not dynamic and wrapper_index is None:
                    continue
                argument_index = 0 if dynamic else wrapper_index
                argument = (
                    node.args[argument_index]
                    if argument_index is not None and argument_index < len(node.args)
                    else None
                )
                target = _literal_string(argument)
                if target is None:
                    record = {
                        "kind": "unresolved_dynamic_import",
                        "sourcePath": str(path),
                        "line": getattr(node, "lineno", None),
                        "callable": resolved,
                        "allowed": False,
                        "reason": "dynamic_target_not_literal",
                    }
                    imports.append(record)
                    violations.append(record)
                    continue
                allowed, reason = self.registry.module_disposition(target)
                resolved_path = self._module_path(target)
                if (
                    not allowed
                    and reason == "unregistered_module_namespace"
                    and resolved_path is not None
                ):
                    path_policy = self.registry.path_disposition(resolved_path)
                    allowed = bool(path_policy["allowed"])
                    reason = (
                        "allowed_resolved_local_fixture"
                        if allowed
                        else str(path_policy["reason"])
                    )
                record = {
                    "kind": "dynamic_import",
                    "sourcePath": str(path),
                    "line": getattr(node, "lineno", None),
                    "callable": resolved,
                    "module": target,
                    "resolvedPath": str(resolved_path) if resolved_path else None,
                    "allowed": allowed,
                    "reason": reason,
                }
                imports.append(record)
                if not allowed:
                    violations.append(record)
                elif resolved_path is not None:
                    queue.append((resolved_path, target))

        files.sort(key=lambda row: row["path"])
        imports.sort(
            key=lambda row: (
                row["sourcePath"],
                row.get("line") or -1,
                row["kind"],
                row.get("module") or "",
            )
        )
        violations.sort(
            key=lambda row: (
                row.get("sourcePath") or row.get("path") or "",
                row.get("line") or -1,
                row["kind"],
            )
        )
        summary = {
            "schemaVersion": "e07.s12t.static-import-analysis.v1",
            "entryPaths": sorted(str(path.resolve()) for path in entries),
            "filesAnalyzed": len(files),
            "importsResolved": len(imports),
            "violationCount": len(violations),
            "files": files,
            "imports": imports,
            "violations": violations,
            "passed": not violations,
        }
        summary["semanticSha256"] = domain_hash(
            "E07/S12T/static-import-analysis/v1",
            {
                "entryPaths": summary["entryPaths"],
                "files": files,
                "imports": imports,
                "violations": violations,
            },
        )
        return summary


class AuthenticatedAccessLedger:
    """In-memory, deny-before-open file/cache/signal access ledger."""

    def __init__(
        self,
        registry: DependencyRegistry,
        *,
        session_commitment: str,
    ) -> None:
        self.registry = registry
        self.session_commitment = session_commitment
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _append(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            event = {
                "ordinal": len(self.events),
                **deepcopy(dict(body)),
            }
            event["eventSha256"] = domain_hash(
                "E07/S12T/access-event/v1",
                event,
            )
            self.events.append(event)
            return deepcopy(event)

    def authorize_path(
        self,
        requested: os.PathLike[str] | str,
        *,
        operation: str,
        source_api: str,
        access_kind: str = "file",
    ) -> dict[str, Any]:
        result = self.registry.path_disposition(requested)
        event = self._append(
            {
                "eventKind": "path_access",
                "accessKind": access_kind,
                "operation": operation,
                "sourceApi": source_api,
                **result,
            }
        )
        if not result["allowed"]:
            raise DependencyViolation(
                f"denied {access_kind} access: {result['reason']}: "
                f"{result['resolvedPath']}"
            )
        return event

    def authorize_signal(
        self, signal: str, *, operation: str = "read"
    ) -> dict[str, Any]:
        allowed, reason = self.registry.signal_disposition(signal)
        event = self._append(
            {
                "eventKind": "signal_access",
                "signal": signal,
                "operation": operation,
                "allowed": allowed,
                "reason": reason,
            }
        )
        if not allowed:
            raise DependencyViolation(f"denied signal access: {signal}")
        return event

    def authorize_cache(
        self,
        path: os.PathLike[str] | str,
        *,
        operation: str = "read",
    ) -> dict[str, Any]:
        return self.authorize_path(
            path,
            operation=operation,
            source_api="cache_broker",
            access_kind="cache",
        )

    def finalize(self) -> dict[str, Any]:
        events = deepcopy(self.events)
        ledger_sha = domain_hash(
            "E07/S12T/access-ledger/v1",
            events,
        )
        authenticator = domain_hash(
            "E07/S12T/access-ledger-authenticator/v1",
            {
                "registrySha256": self.registry.registry_sha256,
                "sessionCommitment": self.session_commitment,
                "ledgerSha256": ledger_sha,
                "eventCount": len(events),
            },
        )
        return {
            "schemaVersion": "e07.s12t.authenticated-access-ledger.v1",
            "registrySha256": self.registry.registry_sha256,
            "sessionCommitment": self.session_commitment,
            "eventCount": len(events),
            "events": events,
            "ledgerSha256": ledger_sha,
            "authenticatorSha256": authenticator,
        }


def validate_access_ledger(
    raw: Mapping[str, Any] | None,
    registry: DependencyRegistry,
    *,
    expected_session_commitment: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if raw is None:
        return {
            "passed": False,
            "reasons": ["missing_ledger"],
            "eventCount": 0,
        }
    ledger = deepcopy(dict(raw))
    if ledger.get("schemaVersion") != "e07.s12t.authenticated-access-ledger.v1":
        reasons.append("schema_mismatch")
    if ledger.get("registrySha256") != registry.registry_sha256:
        reasons.append("registry_commitment_mismatch")
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
        expected = domain_hash("E07/S12T/access-event/v1", body)
        if claimed != expected:
            reasons.append(f"event_hash_mismatch:{ordinal}")
        if event.get("allowed") is not True:
            reasons.append(f"denied_event_present:{ordinal}")
    expected_ledger = domain_hash("E07/S12T/access-ledger/v1", events)
    if ledger.get("ledgerSha256") != expected_ledger:
        reasons.append("ledger_hash_mismatch")
    expected_authenticator = domain_hash(
        "E07/S12T/access-ledger-authenticator/v1",
        {
            "registrySha256": registry.registry_sha256,
            "sessionCommitment": expected_session_commitment,
            "ledgerSha256": expected_ledger,
            "eventCount": len(events),
        },
    )
    if ledger.get("authenticatorSha256") != expected_authenticator:
        reasons.append("authenticator_mismatch")
    if ledger.get("eventCount") != len(events):
        reasons.append("event_count_mismatch")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "eventCount": len(events),
        "ledgerSha256": expected_ledger,
        "authenticatorSha256": expected_authenticator,
    }


@contextmanager
def instrument_path_opens(ledger: AuthenticatedAccessLedger):
    """Instrument common Python path-open APIs and deny before access."""

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_path_open = Path.open
    local = threading.local()

    def invoke(
        original: Callable[..., Any],
        source_api: str,
        file: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if isinstance(file, int):
            raise DependencyViolation("unresolved file-descriptor open denied")
        if getattr(local, "nested", False):
            return original(file, *args, **dict(kwargs))
        local.nested = True
        try:
            mode = kwargs.get("mode")
            if mode is None and args:
                mode = args[0]
            ledger.authorize_path(
                file,
                operation=str(mode or "r"),
                source_api=source_api,
            )
            return original(file, *args, **dict(kwargs))
        finally:
            local.nested = False

    def builtin_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke(original_builtin_open, "builtins.open", file, args, kwargs)

    def io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke(original_io_open, "io.open", file, args, kwargs)

    def os_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke(original_os_open, "os.open", file, args, kwargs)

    def path_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        return invoke(original_path_open, "pathlib.Path.open", path, args, kwargs)

    builtins.open = builtin_open
    io.open = io_open
    os.open = os_open
    Path.open = path_open
    try:
        yield ledger
    finally:
        Path.open = original_path_open
        os.open = original_os_open
        io.open = original_io_open
        builtins.open = original_builtin_open


def _module_records(modules: Mapping[str, ModuleType]) -> list[dict[str, Any]]:
    rows = []
    for name, module in modules.items():
        origin = getattr(module, "__file__", None)
        rows.append(
            {
                "module": name,
                "origin": str(origin) if origin is not None else None,
            }
        )
    return sorted(rows, key=lambda row: row["module"])


def audit_runtime_dispatch(
    callback: Callable[[], Any],
    registry: DependencyRegistry,
    *,
    inspect_preloaded_names: Iterable[str] = (),
) -> dict[str, Any]:
    before = dict(sys.modules)
    callback_result: Any = None
    callback_error: str | None = None
    try:
        callback_result = callback()
    except Exception as error:  # noqa: BLE001 - the audit records any callback failure
        callback_error = f"{type(error).__name__}:{error}"
    after = dict(sys.modules)
    names = set(after) - set(before)
    names.update(name for name in inspect_preloaded_names if name in after)
    records = []
    for name in sorted(names):
        module = after[name]
        origin = getattr(module, "__file__", None)
        allowed, reason, resolved_origin = registry.module_origin_disposition(
            name,
            str(origin) if origin is not None else None,
        )
        records.append(
            {
                "module": name,
                "origin": str(origin) if origin is not None else None,
                "resolvedOrigin": resolved_origin,
                "newlyLoaded": name not in before,
                "allowed": allowed,
                "reason": reason,
            }
        )
    violations = [row for row in records if not row["allowed"]]
    result = {
        "schemaVersion": "e07.s12t.runtime-module-provenance.v1",
        "beforeModuleCount": len(before),
        "afterModuleCount": len(after),
        "inspectedModuleCount": len(records),
        "records": records,
        "violationCount": len(violations),
        "violations": violations,
        "callbackError": callback_error,
        "callbackResultCanonicalSha256": (
            domain_hash("E07/S12T/runtime-callback-result/v1", callback_result)
            if callback_error is None
            else None
        ),
        "passed": callback_error is None and not violations,
    }
    result["semanticSha256"] = domain_hash(
        "E07/S12T/runtime-module-provenance/v1",
        {
            "records": records,
            "callbackError": callback_error,
            "callbackResultCanonicalSha256": result["callbackResultCanonicalSha256"],
        },
    )
    return result


def canonical_round_trip(value: Any) -> dict[str, Any]:
    encoded = canonical_json_bytes(value)
    decoded = json.loads(encoded.decode("ascii"))
    reencoded = canonical_json_bytes(decoded)
    return {
        "passed": encoded == reencoded,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "decoded": decoded,
    }


def cleanup_modules(prefixes: Iterable[str]) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]
