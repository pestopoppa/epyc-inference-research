# AutoKernel controller red-suite triage — 2026-11-07

Input: `/mnt/raid0/llm/tmp/ak-redloop-20261106/pytest3.log` (research origin/main). Worktree outside tmp, `EPYC_ROOT=/workspace`, run from `scripts/kernel_rnd`.
Controller files (8): before 33 red (23 F + 10 E); after 15 red (5 F + 10 E), 3 new skips, 120 passed.

| # | Cause | Count | Class | Breaking commit | Fixed |
|---|---|---|---|---|---|
| 1 | `arena_cell_runner` verifies pinned ROCm evaluator python sha; on this host `geak-v1-rocm62-py312/bin/python` is a symlink to `/home/node/.local/bin/python3.12` (sha 0a97d28d..., pin 9544d2a2...) | 12 (test_arena_cell_runner) | test-environment | `a57feba03` (pin added) | yes: setUp falls back to declared identity when the real binary is absent; the one test asserting the real binary skips |
| 2 | V8 acceptance bundle (immutable) sealed under production v9 and pinned code bytes (`actors.wrapper` digest); loader refuses: v10 identity gate, then live-tree HEAD check, then code-byte digests | 10 E (correctness_capability_acceptance) | stale test / stale fixture (sealed bundle) | `613646215` (v10 repin), then every later change to pinned files | NO (see below) |
| 3 | Portfolio authority pins `_PORTFOLIO_SEMANTIC_SHA256` / `_PORTFOLIO_FILE_SHA256` / contract in `discovery_deployment_factory.py:606-608` | 4 (deployment_factory) + 1 (supervisor `test_direct_validate_graph_reopens...`) | REVIEW-GATED pin | `f8167767a` (2026-11-06 08:50, portfolio JSON: H4/H9 numbers, new DNR `dnr-gfx90a-launch-removal-fusion-under-hip-graphs`, evidence carrier) after last pin update `5ae54e0ae`; `a4e2201bf` (isa_schedule class, schemas) also changes `mechanism_fingerprint` | NO |
| 4 | Supervisor requires `PYTHONDONTWRITEBYTECODE=1` + `sys.dont_write_bytecode` (launcher sets it, bare pytest does not) | 2 | test-environment | n/a | yes: setUpModule/tearDownModule |
| 5 | kernelfoundry / xe_forge spawn child `-m unittest autokernel...` with `cwd=parents[4]` (repo root); package lives at `scripts/kernel_rnd` | 2 | stale test (path) | `7654383ca` (Aug) layout drift | yes: `cwd=parents[2]` (the real pinned-runtime upstream test now PASSES) |
| 6 | evoengineer test imports upstream in-process; host python lacks `scipy` (pinned runtime python has it) | 1 | test-environment | n/a | yes: skipUnless scipy |
| 7 | `_SITE_MODEL` DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf absent from `/mnt/raid0/llm/models` | 1 | test-environment | model retired/absent | yes: skipTest when missing |

## Review-gated pins (not updated)
Cluster 3. Update requires a human review of: (a) the portfolio JSON diff in `f8167767a` (new/changed DNR rows, H4/H9 numbers, evidence carrier `ev-mi210-asm-port-drops-20261106`), (b) `a4e2201bf` changes to `mechanism_fingerprint` (isa_schedule class), (c) the portfolio contract digest. Current values for the reviewer: file sha256 `276623643c9b540970be73067797e217183a55212e57ffed4e9f518aec5c6f1f` (pin `f9db2032...`), semantic sha256 `9dcb51a8e3d2c037eaa6ffbf10248b31e625e67cf3f498ff71ae5c8d70c2cb43` (pin `c894690f...`). Contract sha not recomputed. Also: any live sealed deployment bundle declaring the old authority is refused by `build_static_deployment_graph` (`discovery_deployment_factory.py:2358`) until re-sealed.

## Cluster 2 (not fixed)
The v8 bundle is a historical immutable artifact; making it load means either reseal (impossible, immutable) or stubbing three independent live gates (`_verify_production`, git HEAD, code-byte digests) which defeats what the test asserts. Needs an owner decision: retire the class as historical, or point `AUTOKERNEL_V8_ACCEPTANCE_BUNDLE` at a v10-sealed bundle. Currently errors only because the bundle dir exists on disk (the class skips when absent).

## Real code bugs
None found.
