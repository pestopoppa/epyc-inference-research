# Matched A3/A4 Promptfix Recapture

This package recaptures the 40-task SWE-oracle raw responses for A3 then A4,
using the exact terminal Laguna promptfix question bytes. It is capture-only:
no converter, Docker scorer, benchmark verdict, lineup action, or registry
write is part of this package.

The execute gate reconstructs the 27B continuation's terminal state; the
`continuation.complete` marker alone is insufficient. It verifies the pinned
instrument identity and hashes, all four arms times both suites, complete live
statuses, exact denominators, zero request errors, clean integrity flags, and
terminal marker ordering. It also requires an idle KFD device and free port
18093. A failed, partial, or marker-spoofed continuation cannot satisfy it.

The A3 and A4 GGUFs, frozen v8 HIP binary, v4 runner, watchdog, and promptfix
questions are all pinned by SHA-256. Preflight recomputes those file hashes.
Each run snapshots complete prompts, responses, reasoning, fingerprints, live
status, server arguments, model/binary hashes, and a fail-closed validation
marker. Post-capture validation independently recomputes the model, binary,
runner, watchdog, and question hashes and enforces exact server semantics.

Run only after the active 27B continuation has completed:

```bash
cd /mnt/raid0/llm/epyc-inference-research
bash artifacts/architect-laguna-iq2-v8-20260726/a3-a4-promptfix-recapture-20260726/run_matched_a3_a4_promptfix.sh --preflight
bash artifacts/architect-laguna-iq2-v8-20260726/a3-a4-promptfix-recapture-20260726/run_matched_a3_a4_promptfix.sh --execute
```

Irreducible difference: the models have different weights and tokenizers, so
their actual tokenization, throughput, and model responses are necessarily
model-specific. The task IDs, exact prompt bytes, sampling controls, v4 raw
capture implementation, server surface, kernel, and per-arm denominator are
held constant.
