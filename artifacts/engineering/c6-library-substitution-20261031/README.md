# C6 library substitution: source and finite controls

RVP-C6-15 adds a named LIBRARY-SUBSTITUTION family to the existing native
`scripts/kernel_rnd/c6_mutants/` AST hostile suite and reporter. The three fixed
candidate sources dispatch through Torch matmul, rocBLAS GEMM and hipBLASLt
matmul APIs. The detector follows finite import, callable and same-module helper
aliases; reflective/indexed dispatch, conflicting conditional aliases, and
ambiguous or nested helper definitions are UNKNOWN. Allocation/dtype plumbing and
reference functions outside the candidate path remain allowed. The suite README
documents known calls and limits. Native C++ T0 admission is outside this change.

The accepted patch is `PROPOSED.patch`, SHA-256
`3a178f33ff5f5c9983212ea57f8ddf18de9af4757ae18ecb2e232b72102c93f8`.
Its source base is Research `57ec250b6ad7b9bd5b9c1059cac69ea3603440ef`.
The owning session reviewed the initial proposal and final repaired patch before
publication. The original task text and checkbox are owned by the ROOT session;
this source capsule changes neither.

## Verification

`test-static.log` records 16/16 native controls passing, including finite alias
coverage, conservative UNKNOWN cases, restricted in-memory vendor-stub traces,
allocation/dtype/reference controls, and missing/duplicate/zero-finding/unflagged
reporter mutations. Reporter-only injected rows verify the unchanged 25-row
legacy full-stack denominator; no GPU tier runs in that control.

`native-cli.log` and `native-cli.rows.jsonl` retain the actual native driver
invocation. Its six legacy static rows remain separate from four `L1_library`
rows: three hostile FAILs with one finding each, and one honest PASS with zero
findings. The native summary counts expected=3, flagged=3, findings=3 and labels
the result static-only/PARTIAL with GPU tiers unrun. The original six/25 legacy
denominators and three GPU task predicates remain unchanged.

`verification.json` retains the source startup/before/after hashes, exact argv,
CPU pair, original mutex, overlapping grant snapshots and both exit codes. The
actual resource interval was a finite R6 gap at 2026-10-31T18:47:27–28Z: physical
CPUs 76/77 under the original 28-29 mutex, OMP/OPENBLAS=1. Both corrected
supervisors exited zero and drained all owned descendants. Sampled owned-tree
RSS was 17,813,504 bytes for the controls and 17,616,896 bytes for the CLI;
elapsed intervals were about 0.106 s and 0.109 s. This is sampled process-tree
RSS, not a kernel-enforced peak or quiet-host guarantee. Earlier admission
refusals are retained under `admission-refusals/` rather than counted as tests.

Only repository-owned fixed fixture strings execute, through restricted fake
imports and in-memory call traces. No installed Torch, model, GPU, HIP kernel,
speed benchmark, reference-oracle outcome or native performance warrant is
claimed. Existing CI/source-control evidence only; no new claim ladder.

## Reuse and publication boundary

The existing HOST Research lane was clean before reuse. Its private Git common
directory is distinct from the canonical clone (inode 188913341 versus 96371209,
UID 1000). Its prior `6a3981e4bd23379d57f791379f64ad07f7d334a2` source was already
an ancestor of fresh main and was preserved by a fast-forward to `57ec250b6`.
All five approved paths had unchanged baselines before application. No new full
checkout or canonical HEAD/index/config mutation was used.

`lane-strict.json` records the strict guard's structural shared-clone result
(exit 3). That classification follows git-dir=common-dir for this private admin
layout; it does not identify the canonical clone. The owner explicitly
authorized reuse after the clean/distinct identity checks. The guard was not
modified or treated as passing.

Publication uses the owner-adopted existing R4 physical-relocation guard at
`/mnt/raid0/llm/tmp/codex3-r4-physical-guard-20261031/`. Its admission SHA-256 is
`5215ca929cbc320b998a48c9e91c837417bc3c2314c34783ce992387a3499abe`; the unchanged
supervisor SHA-256 is
`d0351bfc8577f3fa094d5d8f11061ec7c10f2805226c6ddb543a508886f18509`.
The earlier self-imposed 20:09Z publication floor was removed after the owner
confirmed existing physical R4 authority. Fresh physical-footprint admission,
the original 28-29 mutex and timing pause for SMT fallback remain enforced.
The 16 C6 controls are not rerun. No model/R11 smoke is authorized.

The exact canonical Research wrapper and extras are installed only in the
private Git directory. Native staged PII evidence remains private; normal hook
output and Git publication receipts are retained separately after they exist.
No metadata-only timing exception authorizes this source publication. Fresh
Research main `266c475946e62896131c19e702171321bd468baa` was fast-forwarded and
preserved; its full C6 input directory is unchanged from the control base `57ec`.

[no-belief: software source implementation and finite regression controls; no measurement]
