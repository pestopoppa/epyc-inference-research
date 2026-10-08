# E5 Stage-B (W1–W4) execution runbook

Prepared 2026-07-29 by `mainA` (roster id was `claude-main` until the rename in
epyc-root `4dc445a2`). Owning handoff: `epyc-root/handoffs/active/batched-decode-measurement.md`
§ *E5 — NUMA×batch interaction sweep*.

**EXECUTION UNDERWAY 2026-07-29.** W1 is running; W4 follows; W2 runs last, after its capture smoke. **W3 is REMOVED from CPU scope** (operator: `qwen36_27b_q8` is scheduled to run residently on the GPU, so a CPU dense-control sweep measures a shape that will never serve). The staging notes below remain accurate for the parts not yet run.

> **Historical plan boundary.** The execution-underway status and staged execution commands in this document describe the 2026-07-29 plan only. The current v10 refresh is the dry-only section at the end. No current host readiness or input-presence claim is made here.

## Historical Stage-B plan (2026-07-29)

The preconditions, smoke, and execution steps in this section reproduce the July staging plan; they do not establish current host state or form part of the v10 dry-only invocation.

### Preconditions (historical; verify at an actual execution window, do not assume)

| Check | Command | Required |
|---|---|---|
| Host health gate | see snippet below | **zero warnings** |
| Kernel branch | `epyc-root/scripts/session/verify_llama_cpp.sh` | `production-consolidated-v8` @ `67a433bf4` |
| Regions | `epyc-orchestrator/scripts/region-lock status` | claim, never observe (BUS_PROTOCOL rule 7) |

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'scripts/benchmark'); sys.path.insert(0,'scripts')
from server_np_sweep import host_health_warnings, collect_attestation
print(host_health_warnings(collect_attestation()) or 'NONE -> decision-grade eligible')"
```

> **Historical host-health note (2026-07-29).** The original harness health gate was treated as a run-level/per-cell gate for decision-grade execution. That July observation is not a current host-state claim. The dry-only v10 refresh below prints planned launch and affinity-preflight arguments but does not run either preflight or inspect current process state.

**Do NOT pass `--allow-host-health-warning` or `--skip-clean-check`.** Both force
`decision_grade=false` for the whole run. They were correct for W0 (scout, 20-day uptime); they
are wrong here. If the gate warns, fix the host — do not override it.

`drop_caches` between model groups (a NUMA re-read trap follows it, which the harness prewarm
handles):

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

## Step 0 — W2 capture smoke (HARD precondition for W2 only)

The historic W0 Gemma capture was **430/430 parse failures** — HTTP 200, `predicted_n=64`,
`response_text: ""`, and no reasoning field anywhere. Those rows are **unrecoverable, not
re-scoreable** — which is the only reason re-running inference is authorised for W2 at all.
The parser bug (reading only `content`/`delta.content` while gemma emits `reasoning_content`)
was real and is fixed, but see the re-attribution below: reasoning-mode-ON is what actually
consumed the budget.

```bash
RUN=e5-w2-capture-smoke-$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py \
  --manifest-dir data/batched_decode/e5_manifests_w2_capture_smoke/gemma4_26b_a4b_q4km_mtp \
  --execute --i-have-operator-grant --run-id "$RUN"

.venv/bin/python scripts/benchmark/e5_w2_capture_smoke_check.py \
  --run-dir "data/batched_decode/$RUN" \
  --output "data/batched_decode/$RUN/capture_smoke_verdict.json"
```

Exit **0** = W2 may proceed to decision-grade. Exit **1** = W2 stays quality-invalid; do not run
the decision-grade Gemma sweep, and the published artifact's W2 section keeps saying so.

The checker verifies all three required properties, and property 3 is run for real (the actual
offline scorer), not by proxy — (1) and (2) can both hold while every answer still strips to
nothing:

1. `reasoning_text` persisted separately from answer text;
2. nonempty answer-text deltas whenever tokens were generated;
3. the offline scorer sees scoreable answer text (per-cell parse-failure budget 2).

> **The smoke DETECTS the failure; `--reasoning off` is what PREVENTS it.** Re-attributed
> 2026-07-29 (research `5d6a17f2`): the harness emitted no `--reasoning` flag, so gemma4 ran at
> llama-server's `auto` default — which for `arch=gemma4` is ON — while both registries record
> `reasoning: 'off'`. The whole token budget went into the reasoning channel before the answer
> channel opened: 41/43 on the 2026-07-29 smoke, and the same signature behind W0's 430/430.
> **Before trusting a passing smoke, confirm the launch line actually carries `--reasoning off`**
> (it comes from the manifest's `reasoning` field; all 19 gemma4 manifests were amended that day).
> A green smoke on a run without the flag would be a false clearance.

Negative control already run: pointed at a *copy* of the historic W0 Gemma run it fails all three
and exits 1. (Run it against a copy — never the historic dir; the scorer writes
`offline_scores.jsonl` into the run dir and historical records are appended, never edited.)

## Steps 1–4 — the Stage-B windows

One invocation per model group (the harness is one-group-per-invocation **by design**; there is no
`--window` flag — an earlier staged `--window W0` was fabricated). Cell selection comes from the
frozen prune plan `data/batched_decode/e5_pre_reboot_20260728/stage_b_prune_plan.json`; the
per-window `--cell-manifest` lists are generated by:

```bash
python3 - <<'PY'
import json
p = json.load(open('data/batched_decode/e5_pre_reboot_20260728/stage_b_prune_plan.json'))
for w, spec in p['windows'].items():
    mk = spec['model_key']
    args = ' '.join(
        f"--cell-manifest data/batched_decode/e5_manifests/{mk}/{c}.json" for c in spec['run']
    )
    print(f"# {w} ({mk}, {len(spec['run'])} cells)\n{args}\n")
PY
```

| Window | Model group | Cells | Pruned |
|---|---|---|---|
| W1 | `qwen36_q8_0` | 11 | `C1b-np{4,8,16}` (W0: C3 wins 34.97–44.78%; C1b/C1 0.598@K4, 0.463@K8 — the documented half-pair collapse) |
| W2 | `gemma4_26b_a4b_q4km_mtp` | 8 | none |
| ~~W3~~ | ~~`qwen36_27b_q8`~~ | ~~14~~ | **REMOVED from CPU scope 2026-07-29 — GPU-resident; not run** |
| W4 | `qwen3_next_80b` | 12 | none |

Then per window:

```bash
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py \
  <the --cell-manifest list for that window> \
  --execute --i-have-operator-grant \
  --run-id e5-w1-qwen36-$(date -u +%Y%m%dT%H%M%SZ)
```

`qwen36_q8_0-C1-np16-kvu` is a **conditional operator cell** — do not generate or run it without
the split-vs-unified KV decision.

## Step 5 — R1–R4

**Withheld until clean decision-grade Stage-B cells exist.** Do not run them against partial or
scout data.

```bash
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py --summarize-run data/batched_decode/<RUN>
```

R3 refuses to price the eval lane without a fresh current-arm baseline row (`--current-arm-baseline`).

## Historical verification at staging time (2026-07-29, zero inference)

- All **45** pruned Stage-B cells dry-run clean, exit 0 (W1 11 / W2 8 / W3 14 / W4 12).
- Host-health gate returns **zero** warnings (uptime 49 min, `numa_balancing=0`, no static freq
  caps) → decision-grade eligible with no override.
- All 5 GGUFs present; launch args carry `--device none` / `--device-draft none` (the v8 binary is
  HIP-capable — without the pin, CPU cells could silently offload draft work to the MI210),
  production spec-dec, `-c` = 2048×K floored at 8192, production sampling temp 0.3 / seed 42,
  `n_predict` 256.
- Throttle gate re-scoped to each cell's pinned cores (`98cfff44`, operator-ratified) — without it
  19 of these 45 cells would have been force-demoted to observation-grade.

## Artifact obligation when decision-grade results land

Update the operator-facing artifact **in place**:

```
url: https://claude.ai/code/artifact/b0a7785f-d618-436a-a3e2-46f2fef393aa
```

Omitting `url` mints a new URL and breaks the operator's link. Also: replace the OBSERVATION-GRADE
banner (it becomes false), apply the full claim grammar `(metric, protocol-id, n/reps, date,
attestation ref)`, **retain** the W0 scout numbers alongside rather than overwriting, and keep the
W2 subsection quality-invalid until the capture smoke passes.

**Historical host statement (2026-07-29):** the staging note reported AutoPilot down and the host quiesced. This does not establish current AutoPilot or host status. Any future result must carry its own contemporaneous host-state receipt; July values remain historical.

## Current v10 readiness refresh (PREP; source/input capture pending)

The July 2026 material above is historical. Current source and selected-manifest capture is pending. The previous capture is the private PREP artifact `/mnt/raid0/llm/tmp/ni08_remaining_backlog_screen-20261007/evl49-s04-v10-readiness-PREP-20261008-v2/v10-readiness-pins-previous-capture.json` (SHA-256 `77bf80dd16fbb053bc53b5736ff4112380b6e5a0681e20ddf655a6db71ceb1e6`), captured at `2026-10-08T07:10:23.240535Z`; it is not part of the Research checkout and is retained as time-scoped custody evidence, not current readiness. The production source pin is `ffc1bac82eeca6f9099e1ccd9ba49703c460a115`; use the CPU serving alias `/mnt/raid0/llm/kernels/production/cpu/llama-server`. The frozen source tree's `build/bin` and `build-hip/bin` hold v9 binaries and are excluded. W3 is excluded from CPU scope; the exact selected cell counts are W1=11, W2=8, W4=12.

The commands below are dry invocations from the pinned Research source checkout at `c7229b461bdf06e82e12f721aad35bbd81c4dced`. They deliberately omit `--execute` and `--i-have-operator-grant`. Each uses an explicit CPU kernel-store alias, unique scratch output root, fixed run ID, and every selected cell manifest. The harness's default branch prints launch and affinity-preflight argv, writes a dry/non-decision-grade manifest, and returns before affinity preflight or server launch. It does not attest current host quiet, current model presence, or runtime readiness. No command below has been invoked.

Before these dry commands are considered for use, capture into a fresh evidence directory: exact Research HEAD and clean intended-source status; command argv/stdout/stderr/exit status; each selected manifest's path, byte length, and SHA-256; prompt file path/length/SHA-256; and source dependency paths/blob IDs/length/SHA-256. The dry branch does not require model-file presence and this proposal does not make model-path capture a prerequisite. For a later execution plan, separately capture the CPU alias symlink target, store `SHA256SUMS`, provenance file and hash, server binary SHA-256 and version/linkage receipts, model identity, and current host readiness. Do not carry any value from the prior capture forward as current. Preserve all July historical evidence unchanged.

PREP status at 2026-10-08: this package has not imported the harness or invoked the dry commands below. Any owner quiet-window receipt belongs to an actual scheduled execution; this dry invocation cannot provide one.

### W1 — 11 cells

```sh
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py --llama-server /mnt/raid0/llm/kernels/production/cpu/llama-server --output-root /mnt/raid0/llm/tmp/ni08_remaining_backlog_screen-20261007/evl49-s04-v10-readiness-PREP-20261008-v2/dryrun-results --run-id evl49-s04-v10-dry-20261008-w1 --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C1-np1.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C1-np4.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C1-np8.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C1-np16.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C1-np32.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C2-np8.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C2-np16.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C3-np1.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C3-np2.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C3-np4.json --cell-manifest data/batched_decode/e5_manifests/qwen36_q8_0/qwen36_q8_0-C3-np8.json
```

### W2 — 8 cells

```sh
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py --llama-server /mnt/raid0/llm/kernels/production/cpu/llama-server --output-root /mnt/raid0/llm/tmp/ni08_remaining_backlog_screen-20261007/evl49-s04-v10-readiness-PREP-20261008-v2/dryrun-results --run-id evl49-s04-v10-dry-20261008-w2 --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C1-np1.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C1-np8.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C1-np16.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C1-np32.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C3-np1.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C3-np2.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C3-np4.json --cell-manifest data/batched_decode/e5_manifests/gemma4_26b_a4b_q4km_mtp/gemma4_26b_a4b_q4km_mtp-C3-np8.json
```

### W4 — 12 cells

```sh
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py --llama-server /mnt/raid0/llm/kernels/production/cpu/llama-server --output-root /mnt/raid0/llm/tmp/ni08_remaining_backlog_screen-20261007/evl49-s04-v10-readiness-PREP-20261008-v2/dryrun-results --run-id evl49-s04-v10-dry-20261008-w4 --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1-np1.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1-np4.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1-np8.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1-np16.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1-np32.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1b-np4.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1b-np8.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C1b-np16.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C3-np1.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C3-np2.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C3-np4.json --cell-manifest data/batched_decode/e5_manifests/qwen3_next_80b/qwen3_next_80b-C3-np8.json
```
