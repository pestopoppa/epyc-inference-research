# V11-086-SLOW-2 build-only receipt, 2026-10-25

Both refreshed candidates compiled, passed compile/link parity and own-library linkage, preserved every original non-bench bin file/static common archive and symlink target, and passed the supported `--help` rc0 check. Source/manifests were checked before/after. Actual affinity72-95 matched the canonical one-quarter build grant. These are implementation observations [belief: cc]; no model or inference ran and no performance claim is made.

| arm | exact candidate root | original source | patched source commit |
|---|---|---|---|
| B | `/mnt/raid0/llm/tmp/codex-ni-v11-086-slow2-20261025/store-candidates-v3/B` | `8110933ba4b8f6be197e10360e8613de6067021d` | `16d24304ffd3e55d00f5ad108459fb96a58e48a7` |
| C | `/mnt/raid0/llm/tmp/codex-ni-v11-086-slow2-20261025/store-candidates-v3/C` | `f0c96dc1fc90a0257061db41dd95844763630969` | `03791e10b145c8e192879649c86b2a0d42e388c3` |

Both source commits cherry-pick full `141e4bcf5ae023bd3b871edcd32ad493e6c4d58e`. Original build stores remain read-only and unchanged. Candidate directories contain copied dependencies; only llama-bench and libllama-bench-impl.so are deliberately rebuilt. The small launcher executable hashes equal the originals because the fix lives in the implementation library.

| product | B SHA-256 | C SHA-256 |
|---|---|---|
| llama-bench | `90b1f55bc54b0d961a945c8e9c73b8c4f8370eef0a44065b7888a6894a112396` | `4b3231a09c443d971fe8f0aec7c52a9fc1785e30dc7665b6004f159c1a39de97` |
| libllama-bench-impl.so | `610c88494eb7190230b82e41ec30fbbc00ce7a62931d5387e32b85cb4af7a125` | `8af63a8804d5d8d3e7427b6a23edb946b9a1ea79640ee220422bce15f2a543c4` |
| libggml-cpu.so.0.16.0 (unchanged) | `01327fe9c78671d2d79fe22e437a9e2a178012c14b2d342529e05e33d9693275` | `3f520a5b6552d390c68e9555706b8ed34c27d61c6d07f60f04ed7a24d1e7e435` |

Per-arm provenance, full product SHA256SUMS, preserved-file manifest, symlink snapshots, generated compile/link parity, linkage, dynamic ELF metadata and HELP.rc/source receipts are retained here. Build binaries remain in the exact private candidate paths and are not Git artifacts.

The earlier v2 attempt produced a valid partial C build, then its harness invoked unsupported `--version` and got rc1. B was not reached. Its failure and partial C remain immutable in original `build-logs/` and `store-candidates/C`; v3 used entirely new roots and `--help`. This is a harness check correction, not a kernel/build defect. Historical diagnostic wrong-PID samples, GRAY-2-5X and whole-host HOLD remain unchanged.

## Owning-session next steps (separate inference grant)

1. Verify candidate SHA256SUMS and preserved CPU hashes. Copy the published driver plus local helper/predeclared files into a private launch directory; select the two exact refreshed candidates there at the owning session's boundary. Do not edit a live poller's script.
2. Under the owner's explicit grant, run one launch per arm with the existing Q38 model/environment/FA/f16 recipe, `-p 0 -n 32 -d 32768 -r 1 -o json`, quarter-appropriate threads/affinity and candidate-specific LD_LIBRARY_PATH. Capture original argv, candidate hash/provenance, linkage and raw row protocol attributes at write time. No hardening flag requested. Preserve all failure results.
3. SLOW-2 closes only when both actual arm outputs explicitly report boolean `autokernel_hardened:false`; the source guard must accept both raw rows. The owner then re-arms the separately scheduled whole-host 086v2 timing run under its unchanged pinned rule, cap and recipe. Build success alone never ticks the parent task.

Belief wiring must reuse the existing llama-bench protocol family/ladder. Capture native acceptance warrant at write time; implementation/build receipts are observations, not retrospective timing warrant.
