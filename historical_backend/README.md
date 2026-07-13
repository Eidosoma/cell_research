# Source-external historical backend adapter

This directory contains the S04 compatibility adapter and OCI recipe. It does
not contain the frozen public source. The public repository has no detected
license grant, so the original bytes remain in the S02 read-only cache
quarantine and are mounted only at runtime.

The adapter verifies all 91 frozen files against the S02 SHA-256 ledger before
importing the original policy, thread, lock, group, and probe classes. It does
not add the paper's pre-lock random gate because that gate is absent from frozen
public commit `1fd2bd5921c1f6b423a71f691d5189106a8a1020`. That commit is a frozen
public-code reference, not a claimed publication snapshot.

The recipe pins CPython 3.11.9 because frozen bytecode supports a 3.11 lineage,
and NumPy 1.23.5 because it preserves the driver's warned ragged-object-array
conversion. These are reconstruction choices, not recovered publication
versions. Historical-style ragged arrays require trusted `allow_pickle=True`
loading; the adapter additionally emits safer per-run numeric arrays.

Build from this directory only:

```bash
docker build --pull=false -t e01-s04-legacy-adapter:py311 .
```

Run a bounded toy with the source, provenance ledger, and output mounted
separately. The hardening flags are part of the validated recipe:

```bash
docker run --rm --network none --read-only --user 0:0 --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --mount type=bind,src=/cache/e01_s02/historical-worktree,dst=/historical-src,readonly \
  --mount type=bind,src=/artifacts/research_steps/S02/source_tree_checksums.sha256,dst=/provenance/source_tree_checksums.sha256,readonly \
  --mount type=bind,src=/cache/e01_s04/container-output,dst=/output \
  e01-s04-legacy-adapter:py311 \
  --source /historical-src --checksums /provenance/source_tree_checksums.sha256 \
  run --policy bubble --values 2,1 --seed 1729 --output /output/toy
```

The validated S02 quarantine uses root-only modes (`0400` files and `0500`
directories), so this exact cache mount requires container UID 0. The runtime
still has every Linux capability dropped, no network, no privilege escalation,
a read-only root filesystem, and a read-only source mount. Environments that
grant a read-only group access to the quarantine can omit `--user 0:0` and use
the image's default unprivileged UID 65532.

The adapter's wall-clock timeout and post-run joins are safety containment, not
historical stop rules. Successful no-fault runs still stop on the original
non-strict `is_sorted` predicate. Generated `.npy` files are S04 smoke outputs,
not recovered historical data.
