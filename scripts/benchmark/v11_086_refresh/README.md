# V11-086 ordinary-protocol refresh

This is a source proposal, not deployment of the owner's live driver. The repaired 086v2 driver validates **raw** bench JSON before normalization: every row must contain a boolean `autokernel_hardened` matching parsed benchmark argv. Unknown option arities, missing/nonboolean fields, mixed protocols and unrequested hardening refuse the launch without retries or cell aggregation. The recipe, thresholds and decision logic are unchanged. Default B/C paths remain the original pins until the owning session selects approved candidates in a private driver copy.

`v11_driver_086v2.py`, its required `fa086v2_integrity.py` dependency, and the complete SHA-pinned predeclared files are colocated. Inside an owner grant, model-free portability checks are:

```bash
/usr/bin/python3 -B -I /absolute/path/v11_driver_086v2.py --help
/usr/bin/python3 -B -I /absolute/path/test_protocol.py /absolute/path
/usr/bin/python3 -B -I /absolute/path/test_driver_integration.py /absolute/path
/usr/bin/python3 -B -I /absolute/path/test_probe.py /absolute/path
```

These tests do not open a model or launch inference. The last fixture creates only captured sleeping descendants and verifies `/proc` traversal and cleanup. `probe_child.sh CAPTURED_PARENT_PID` resolves the real llama-bench descendant, including the root after an exec, or exits1. `probe.patch` shows the replacement for the historical first-child probe; it does not rewrite historical diagnostic evidence.

The build tooling is the reviewed standalone project, with GNU++17, pinned g++15.2.0, original flags/defines/include and library link order. It imports copied dependencies; no llama/ggml build subtree is configured. `compile-parity.py` rejects differences before target compilation. Only bench/main and bench implementation objects/products rebuild. Source is the exact original B/C pin plus full fix141e4bcf5, including its structural test/registration. `--help` is the supported model-free CLI check; this bench has no `--version` option. Identity is source commit + fix + implementation/product hashes.

`launch-build.sh` and `build-bench-only.sh` use `V11_REFRESH_STATE` as an explicit reviewed state root. Tool dependencies resolve beside the scripts. State contains the main-approved marker, pinned source clones `src-B`/`src-C`, `synthetic/source-{B,C}.txt`, synthetic COMPLETE, reviewed BUILD-PROPOSAL-v3.SHA256, and own canonical 18-column OWN-FILLERS-BUILD-v3.md. The launcher uses official filler8, whole-body taskset, canonical grant/HOLD/time/actual CPU-subset checks. This is a bounded reproduction launcher for this exact reviewed state, not an automatic state/bootstrap/grant writer. It refuses existing candidate/build roots; author a fresh reviewed root for another attempt. No tests or toolchains run outside grants.

`driver.patch` and `probe.patch` retain exact changes against the captured owner sources. Build-only receipts and acceptance instructions live in `data/v11-086/slow2-build-refresh-20261025/`. Native depth acceptance is still outstanding; implementation tests and build receipts do not close SLOW-2 or regrade the historical HOLD.
