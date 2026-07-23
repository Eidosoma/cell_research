from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.run_native_event_discovery_s10 as base
import scripts.run_native_event_discovery_s10b as s10b


CALLBACK_NAMES = tuple(sorted(s10b.execution_callback_overrides()))


def _surface_and_callbacks() -> tuple[
    SimpleNamespace, dict[str, object], dict[str, object], dict[str, int]
]:
    counts = {name: 0 for name in CALLBACK_NAMES}
    originals: dict[str, object] = {}
    targets: dict[str, object] = {}
    for name in CALLBACK_NAMES:
        def original(*args: object, _name: str = name, **kwargs: object) -> str:
            del args, kwargs
            counts[_name] += 1
            return f"original:{_name}"

        def target(*args: object, _name: str = name, **kwargs: object) -> str:
            return originals[_name](*args, **kwargs)  # type: ignore[operator]

        originals[name] = original
        targets[name] = target
    return SimpleNamespace(**originals), originals, targets, counts


def test_captured_originals_are_distinct_from_installed_targets() -> None:
    targets = s10b.execution_callback_overrides()
    assert set(s10b._ORIGINAL_BASE_CALLBACKS) == set(targets)
    assert all(
        s10b._ORIGINAL_BASE_CALLBACKS[name] is not targets[name]
        for name in targets
    )
    assert (
        s10b._ORIGINAL_BASE_CALLBACKS["revalidate_frozen_inputs"]
        is base.revalidate_frozen_inputs
    )
    assert (
        s10b._ORIGINAL_BASE_CALLBACKS["prospective_freeze"]
        is base.prospective_freeze
    )


def test_installed_dispatch_invokes_original_once_and_restores() -> None:
    surface, originals, targets, counts = _surface_and_callbacks()
    with s10b.installed_base_callbacks(
        surface=surface, captured=originals, overrides=targets
    ) as audit:
        assert surface.revalidate_frozen_inputs is targets["revalidate_frozen_inputs"]
        assert surface.revalidate_frozen_inputs() == (
            "original:revalidate_frozen_inputs"
        )
        assert audit["installed"]
        assert not audit["nestedOrRepeated"]
    assert counts["revalidate_frozen_inputs"] == 1
    assert surface.revalidate_frozen_inputs is originals["revalidate_frozen_inputs"]


def test_nested_repeated_installation_is_idempotent_and_exactly_restored() -> None:
    surface, originals, targets, _ = _surface_and_callbacks()
    with s10b.installed_base_callbacks(
        surface=surface, captured=originals, overrides=targets
    ):
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ) as nested:
            assert nested["nestedOrRepeated"]
            assert surface.prospective_freeze is targets["prospective_freeze"]
        assert surface.prospective_freeze is targets["prospective_freeze"]
    assert all(getattr(surface, name) is originals[name] for name in CALLBACK_NAMES)


def test_preexisting_wrapper_conflict_fails_before_any_overwrite() -> None:
    surface, originals, targets, _ = _surface_and_callbacks()
    conflict = lambda: None
    surface.report_markdown = conflict
    before = {name: getattr(surface, name) for name in CALLBACK_NAMES}
    with pytest.raises(s10b.CallbackBindingError, match="conflicting"):
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            raise AssertionError("unreachable")
    assert all(getattr(surface, name) is before[name] for name in CALLBACK_NAMES)


def test_self_reference_target_fails_closed() -> None:
    surface, originals, targets, _ = _surface_and_callbacks()
    targets["revalidate_frozen_inputs"] = originals["revalidate_frozen_inputs"]
    with pytest.raises(s10b.CallbackBindingError, match="self-reference"):
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            raise AssertionError("unreachable")


def test_installed_callback_replacement_is_detected_and_restored() -> None:
    surface, originals, targets, _ = _surface_and_callbacks()
    with pytest.raises(s10b.CallbackBindingError, match="was replaced"):
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            surface.manifest_for_output = lambda: None
    assert all(getattr(surface, name) is originals[name] for name in CALLBACK_NAMES)


def test_original_exception_propagates_and_bindings_restore() -> None:
    surface, originals, targets, _ = _surface_and_callbacks()
    sentinel = ValueError("outcome-independent sentinel")

    def failing_original() -> None:
        raise sentinel

    originals["revalidate_frozen_inputs"] = failing_original
    surface.revalidate_frozen_inputs = failing_original
    with pytest.raises(ValueError) as raised:
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            surface.revalidate_frozen_inputs()
    assert raised.value is sentinel
    assert surface.revalidate_frozen_inputs is failing_original


def test_installer_restores_after_body_exception() -> None:
    surface, originals, targets, _ = _surface_and_callbacks()
    with pytest.raises(RuntimeError, match="body sentinel"):
        with s10b.installed_base_callbacks(
            surface=surface, captured=originals, overrides=targets
        ):
            raise RuntimeError("body sentinel")
    assert all(getattr(surface, name) is originals[name] for name in CALLBACK_NAMES)
