# AutoKernel loop-engineering and process-recovery evidence — 2026-08-12

This packet records the first governed AK-LE-1/2 planner panel and the post-hook real host-process
fault rehearsal. It is a locator and interpretation boundary; the immutable receipts below remain
authoritative.

## AK-LE evidence sequence

| Attempt | Disposition | Evidence |
|---|---|---|
| r1 | Eight raw cells completed, but the reducer refused them because the manifest pinned source bytes rather than a runnable, input-bound structural-prefilter contract. It emits no reduced result or belief rows | `/mnt/raid0/llm/autokernel/campaigns/ak-le-planner-20260812-r1/planner-reduction-refusal-r2.json`; refusal self-hash `f86cd53b4292e76ac47baed8f37f04dad000ae7e3e6058d0f6a4b8e86f14ae6` |
| r2 | Failed before one complete cell: Claude returned a malformed JSON wrapper. It emits no reduced result or belief rows | `/mnt/raid0/llm/autokernel/campaigns/ak-le-planner-20260812-r2/panel/panel.json`; file SHA-256 `a4ee273ee8b8f51d1afbfd1d71642a6c45182d3403d58838833b99f1080ef171` |
| runner repair | `loop_experiment_runner.py` now binds Claude's structured-output schema and strictly parses its wrapper while retaining read-only, captured-process behavior | research `97511843` |
| r3 | Eight of eight cells parsed; the source-pinned structural reducer emitted one planner receipt plus 32 prospective belief rows | `/mnt/raid0/llm/autokernel/campaigns/ak-le-planner-20260812-r3/planner-reduction.json`; file SHA-256 `c24122893acdd7cf10f042a447d7daa41aeb7d77161dcda8b5b5bf5344b57791`; reduction self-hash `e75683981b30bb7f1336154c7a4a70a8310e1e85f3d17557e68d209dcc9e3ecf` |

The corrected r3 panel held model/quant/context fixed inside each comparison and swept high/xhigh
effort plus absent/rendered decode-target context. One observation per cell is not a population:
these findings are descriptive and model-local.

| Model | Control high → xhigh | Target high → xhigh | Target effect at high / xhigh |
|---|---|---|---|
| Claude Opus 5 | surviving hypotheses `6→6`; wall `81.77→136.84 s` | `6→6`; wall `94.08→180.49 s` | `6→6` / `6→6` |
| GPT-5.6 Sol | surviving hypotheses `3→3`; wall `26.07→65.63 s` | `1→2`; wall `45.88→56.28 s` | `3→1` / `3→2` |

No cell emitted an already-optimized termination. Thus the predeclared higher-effort direction is a
bounded null on both control contrasts; the target arm is null for Claude and mixed/adverse for
Codex. The receipt has no campaign ranking, champion, release, or AK-LE-3 scaffold authority.

Research `16ad9c2c` prospectively emits four self-hashed search-persistence rows per complete cell
only after re-running the exact pinned reducer. The r1 refusal and failed r2 attempt remain zero-row
evidence; they were not back-filled.

## Real host-process fault rehearsal

Campaign `ak-fault-rehearsal-20260812-r2` passed all three native legs:

1. a captured child planted a journal crash; a separate captured restart child replayed and appended
   the durable journal, and both identities were verified dead;
2. an exact captured holder acknowledged revocation on a disposable fake-device claim, stayed alive
   beyond the deadline (proving non-preemption), then was terminated by exact process-group identity
   and verified dead;
3. a changed hash-bound artifact was refused.

The rehearsal touched no live claim root and started no inference, benchmark, build, GPU, kernel,
stack, release, or production action. Research `5c8714a1` emitted three self-hashed dependency rows
sharing one rehearsal-run support key. They explicitly have no performance-measurement,
corroborating-witness, belief-measurement, campaign, ranking, or release authority.

- Receipt: `/mnt/raid0/llm/autokernel/rehearsals/ak-fault-rehearsal-20260812-r2/receipt.json`
- Receipt self-hash: `8a345fbdbfff4bb04b4ac3388b8e82ff347941bc8fb93cbe8266ee2b2ed79a58`
- File SHA-256: `70e9f87bc5405be6e66c9c0f6ce49f0427f1c83ba0f14491c93401e5283d688e`

## Remaining boundary

- AK-LE-3 still needs a governed same-model direct-implement versus implement-then-exploit scaffold
  seam before any matched empirical arm.
- CPU IQK still requires the OP-16 orderly reboot, followed by the first distinct clean matched
  completed-proposal archive and AK-WM-2/AP-WM-1b observe-only evaluation.
- OP-11 and OP-17 remain human decisions in epyc-root. No experimental kernel commit or production
  attestation change is authorized by this packet.
