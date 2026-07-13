# E01 shared event and provenance schema

`shared_events` is the S06 interoperability layer between two deliberately
different evidence sources:

- `reference` adapts the deterministic S05 clean-room backend. Every S05 event
  is retained verbatim in `backendPayload.sourceEvent`; the adapter independently
  replays batch commits and verifies the native pre/post hashes.
- `historical_frozen_public_commit` adapts an already validated S04 `run.json`.
  Its event sequence is explicitly `recorded_swap`, not `activation`: the frozen
  `StatusProbe` emitted post-swap value snapshots only. Missing actor identities,
  observations, failed proposals, random draws, and full-state hashes remain
  null with mandatory `fieldAvailability` reasons.

Current versions are `e01.shared_event.v1.0.0`,
`e01.shared_trace_manifest.v1.0.0`, and
`e01.shared_event_adapters.v1.0.0`. Unknown versions are rejected.
Direct writer/validator dependencies are pinned in `requirements.lock`.

Each event is content-addressed from canonical JSON. `write_trace_bundle` writes
the same records as externally Zstandard-compressed canonical JSONL and as
Parquet with Zstandard-compressed columns. Parquet preserves the full canonical
event in `event_json` and adds a searchable projection; neither encoding drops
backend payload or field-level provenance.

```python
import json
from pathlib import Path
from shared_events import adapt_reference_result, write_trace_bundle

source = Path("result.json")
bundle = adapt_reference_result(json.loads(source.read_text()), source_path=source)
manifest = write_trace_bundle(bundle, Path("trace/reference"))
```

These schemas establish serialization and provenance contracts. The broader
randomized invariant/property-test gate belongs to S07 and is not implemented
here.
