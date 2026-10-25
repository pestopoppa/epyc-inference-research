# fold_gate - gate (b) of a champion fold

Compares the shared libs of a candidate store image against the previous champion image. Deterministic, read-only.

```
python3 fold_gate.py --base-image $STORE/<kind>-<old>/bin --new-image $STORE/<kind>-<new>/bin \
    --src-repo /mnt/raid0/llm/llama.cpp --base-commit <old sha> --cand-commit <new sha> \
    --expected-changed libllama-bench-impl.so \
    --record-json $RUN/fold_gate_record.json      # immutable Vidya source record; optional
```
Exit 0 = PASS. Run once for `cpu`, once for `gpu`. No manual acceptance: every difference is classified or the gate fails.

`--record-json` is opt-in, so existing stdout, exit status, and callers stay unchanged. It writes one
complete `epyc.champion_fold_gate.v1` JSON record without replacing an existing file. The record
contains every per-library class (including missing/extra), code-identity flag, differing details,
the checked symlink targets, the overall gate verdict, and the base/candidate store paths plus the
SHA-256 of each store's `SHA256SUMS` file. Every library record carries the exact identity
proposition that the Vidya verifier projects. An `EXPLAINED-UI` result also emits its own exception
record, naming `fold_gate.v4:EXPLAINED-UI` as the deterministic rule that accepted it; it does not
accept any other code difference. Writing the record requires both store `SHA256SUMS` files to be
present and refuses to replace an earlier receipt.

| class | meaning |
|---|---|
| SAME / EXPLAINED | identical bytes / identical normalised disassembly, differences only in build-id, build-info strings, layout sections |
| EXPECTED | code differs in a lib named by `--expected-changed` |
| EXPLAINED-PATH | identical normalised disassembly; the only differences are strings equal once each image's OWN build/source root is normalised (absolute path vs `-ffile-prefix-map=<src>=.`, read from that image's `CMakeCache.txt`), each image's own commit/build number (`VERSION.txt`), a linker re-ordering of the string pool (non-string bytes compared as a multiset), layout, and - for `libggml-hip` - `no_device_code` helpers whose codegen depends on `__FILE__` length (kernels must be identical). A real string change stays UNEXPLAINED. A pass. |
| EXPLAINED-UI | only the `_GLOBAL__sub_I_ui.cpp` initialiser differs, via `movq $imm` stores (gzip sizes of the embedded web UI), AND `git diff base cand -- tools/ui tools/server` is empty (proved by the driver) |
| CODE-DIFF / UNEXPLAINED / MISSING / EXTRA | FAIL |

`libggml-hip` device code (`hipdev.py`): each gfx code object is disassembled, pc-relative operands are resolved to what they
point at, and `.rodata` / kernel descriptors are compared with the build root normalised. Kernels must be identical; the only
tolerated difference is a non-kernel helper (`no_device_code`) that references a `__FILE__` path, whose codegen depends on path
length. Those are counted in the output (`mode=path-dependent-codegen`). With path-independent `__FILE__` (prefix-map) or equal-length
roots the result is `mode=exact`.

## Making the inputs deterministic (recipe changes are proposals, not applied here)
- UI: `scripts/ui-assets.cmake` priority 1 is a pre-built `tools/ui/dist/index.html`. Copying the previous champion's
  `<build>/tools/ui/dist` to `<fold worktree>/tools/ui/dist` before configure skips `npm run build` and makes ui.cpp byte-stable.
- `__FILE__`: add `-ffile-prefix-map=<worktree>=. -fmacro-prefix-map=<worktree>=.` to `CMAKE_C/CXX/HIP_FLAGS` in `build_store.sh`.

Transition folds (baseline built without prefix maps, candidate with): `fold_gate.py` prints `image-roots base|new` so the roots it normalised are on the record; `(UNREADABLE ...)` means no CMakeCache.txt was found and the legacy loose tokens are in force.

Tests (fixtures compiled at test time with gcc and ROCm clang, nothing committed): `python3 -m pytest scripts/kernel_rnd/fold_gate -q`.
`libcmp.py BASE NEW [--ui-source-identical]` classifies a single lib.
