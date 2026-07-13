from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from reference_simulator.api import create_scenario, run_scenario
from shared_events import (
    adapt_historical_run,
    adapt_reference_result,
    read_jsonl_zstd,
    read_parquet_zstd,
    validate_event,
    validate_trace,
    write_trace_bundle,
)
from shared_events.model import bundle_with_events, sha256_file
from shared_events.schema import TRACKED_FIELDS, assign_event_id
from shared_events.schema import sha256_json
from shared_events.validation import UnsupportedSchemaVersion


S04_RUN = Path("/artifacts/research_steps/S04/smoke_outputs/generated_raw_smoke/run.json")
S05_RESULT = Path("/artifacts/research_steps/S05/smoke_sample_result.json")


class AdapterTests(unittest.TestCase):
    def test_reference_adapter_preserves_payload_and_replays_hashes(self):
        raw = json.loads(S05_RESULT.read_text())
        bundle = adapt_reference_result(raw, source_path=S05_RESULT)
        summary = validate_trace(bundle)
        self.assertEqual(summary["eventCount"], len(raw["events"]))
        self.assertTrue(summary["hashChainsValidated"])
        self.assertEqual(bundle.events[0]["backendPayload"]["sourceEvent"], raw["events"][0])
        self.assertEqual(set(bundle.events[0]["fieldAvailability"]), set(TRACKED_FIELDS))

    def test_reference_adapter_supports_atomic_batch_hashes(self):
        scenario = create_scenario(
            [5, 4, 3, 2, 1], policy="Bubble", seed=77,
            generation_key="S06-batch", permute=False, batch_width=4,
            max_activations=1000,
        )
        result = run_scenario(scenario, trace_mode="full")
        bundle = adapt_reference_result(result)
        validate_trace(bundle)
        self.assertTrue(any(event["run"]["batch"]["width"] == 4 for event in bundle.events))

    def test_reference_zero_event_terminal_run_uses_manifest_stop_metadata(self):
        scenario = create_scenario(
            [1, 2, 3], policy="Bubble", generation_key="S06-zero-event",
            permute=False,
        )
        bundle = adapt_reference_result(run_scenario(scenario, trace_mode="full"))
        self.assertEqual(bundle.events, ())
        self.assertEqual(bundle.stop_reason, "complete")
        manifest = bundle.base_manifest()
        self.assertEqual(manifest["stop"]["reason"], "complete")
        self.assertIsNone(manifest["stop"]["terminalEventIndex"])
        validate_trace(bundle, manifest)

    def test_reference_source_hash_tamper_is_rejected(self):
        raw = json.loads(S05_RESULT.read_text())
        raw["events"][0]["preStateHash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pre-state hash mismatch"):
            adapt_reference_result(raw)

    def test_historical_mapping_is_explicitly_lossy(self):
        raw = json.loads(S04_RUN.read_text())
        bundle = adapt_historical_run(raw, source_path=S04_RUN)
        validate_trace(bundle)
        event = bundle.events[0]
        self.assertEqual(event["run"]["sequenceBasis"], "recorded_swap")
        self.assertEqual(event["proposal"]["kind"], "Swap")
        self.assertTrue(event["decision"]["accepted"])
        self.assertIsNone(event["actor"]["id"])
        self.assertEqual(event["fieldAvailability"]["actor.id"]["status"], "unavailable")
        self.assertEqual(event["fieldAvailability"]["state.nativePreHash"]["status"], "unavailable")

    def test_historical_zero_swap_run_does_not_invent_terminal_event(self):
        raw = json.loads(S04_RUN.read_text())
        raw["inputValues"] = raw["finalValues"] = [1, 2]
        raw["sortingSteps"] = []
        raw["cellTypes"] = []
        raw["traceSha256"] = sha256_json([])
        raw["metrics"].update({
            "swapCount": 0, "recordedSortingSteps": 0, "recordedCellTypeSteps": 0,
        })
        bundle = adapt_historical_run(raw)
        self.assertEqual(bundle.events, ())
        manifest = bundle.base_manifest()
        self.assertEqual(manifest["stop"]["reason"], "historical_is_sorted")
        self.assertIsNone(manifest["stop"]["terminalEventIndex"])
        validate_trace(bundle, manifest)


class SchemaAndWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = adapt_reference_result(json.loads(S05_RESULT.read_text()))

    def test_required_fields_and_event_identity(self):
        validate_event(self.bundle.events[0])
        damaged = deepcopy(self.bundle.events[0])
        del damaged["actor"]
        with self.assertRaisesRegex(ValueError, "JSON Schema validation failed"):
            validate_event(damaged)

    def test_unknown_version_is_rejected(self):
        damaged = deepcopy(self.bundle.events[0])
        damaged["schemaVersion"] = "e01.shared_event.v2.0.0"
        with self.assertRaises(UnsupportedSchemaVersion):
            validate_event(damaged)

    def test_unavailable_field_cannot_carry_invented_value(self):
        historical = adapt_historical_run(json.loads(S04_RUN.read_text()))
        damaged = deepcopy(historical.events[0])
        damaged["actor"]["id"] = "invented-cell"
        damaged = assign_event_id(damaged)
        with self.assertRaisesRegex(ValueError, "actor.id is unavailable"):
            validate_event(damaged)

    def test_pre_post_chain_tamper_is_rejected(self):
        events = [deepcopy(event) for event in self.bundle.events]
        second = deepcopy(events[1])
        second["state"]["observablePreHash"] = "sha256:" + "0" * 64
        events[1] = assign_event_id(second)
        damaged = bundle_with_events(self.bundle, events)
        with self.assertRaisesRegex(ValueError, "observable post/pre hash chain"):
            validate_trace(damaged)

    def test_both_compressed_encodings_round_trip_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_trace_bundle(self.bundle, root / "first")
            second = write_trace_bundle(self.bundle, root / "second")
            self.assertEqual(read_jsonl_zstd(root / "first.jsonl.zst"), list(self.bundle.events))
            self.assertEqual(read_parquet_zstd(root / "first.parquet"), list(self.bundle.events))
            self.assertEqual(sha256_file(root / "first.jsonl.zst"), sha256_file(root / "second.jsonl.zst"))
            self.assertEqual(sha256_file(root / "first.parquet"), sha256_file(root / "second.parquet"))
            self.assertEqual(first["traceContentSha256"], second["traceContentSha256"])


if __name__ == "__main__":
    unittest.main()
