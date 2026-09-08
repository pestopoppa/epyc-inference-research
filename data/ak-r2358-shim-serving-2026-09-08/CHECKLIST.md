# R23-58 PRE-FLIGHT — 30 seconds before you press go

Full design: `PREREGISTRATION.md` (freeze its sha256 first). Runner: `run_r2358.py`.
**Question:** does `GGML_NOHUGEPAGE_PROCESS=1` tighten the GPU **serving floor**?
**Unit:** one `llama-server` launch. **Plan:** 24 valid couples = 48 launches, α = 0.05.

---

## 0. The build — no rebuild needed

Use **`/mnt/raid0/llm/tmp/build-fold-ef81196d5`** (the runner's default): the FOLD-2 candidate build
of the champion, HIP/gfx90a, built 2026-09-08 09:25 from `fold-ef81196d5-src` @ `ef81196d5` (clean).

- [ ] Control 0 passes and its **negative control** rejects a pre-fold build. Both are preflight
      checks; you do not run them by hand. If you want to eyeball it:

```bash
B=/mnt/raid0/llm/tmp/build-fold-ef81196d5
LD_LIBRARY_PATH=$B/bin ldd $B/bin/llama-server | grep llama-common   # must resolve INSIDE $B
grep -c INF70_CHAMPION3_PROCESS_THP_DISABLE $B/bin/libllama-common.so.0.0.10301   # 1
```

**Scan the library, never `llama-server`** — it is an 18 KB stub and `common/common.cpp` compiles into
`libllama-common.so`. Searching the executable returns a false negative on a build that has the shim.

Do **not** use `anchor-gen-021` (`bff30cebee0d`) or `champ2/build-hip` (2026-09-01). Both are pre-fold
and genuinely lack the shim.

## 1. Host

- [ ] `uptime` — 1-minute load **≤ 8**. (During writing it was 28.1; that is another session measuring.)
- [ ] **autokernel loop DOWN** — the 4.581 % floor was calibrated with it down; a floor from a
      different host state is not the same floor.
- [ ] No CPU campaign benching on 0-95. Every logical CPU here shares a physical core with 0-95, and
      the coupling runs both ways — pinned GPU host threads have been measured degrading the CPU
      floor 9×.
- [ ] `cat /sys/class/kfd/kfd/proc | wc -l` → **0 GPU tenants**, and VRAM below 1 GiB.
      *(The runner checks all three itself, name-blind. Never `pkill`/`pgrep` a name pattern here.)*

## 2. Locks

- [ ] `/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock` — free. **The runner takes it itself**
      (`claim.hold()`, non-blocking, re-verified at close). You do not take it by hand; you just make
      sure nobody else holds it, and nobody starts the loop mid-run.
- [ ] Port **18317** free (deliberately not the loop's 18311).

## 3. Prerequisite in the tree

- [ ] `autokernel.loop.serving` carries the U3 change (research lane `b5f58b74`): `Recipe.env`,
      `Recipe.env_readback`, `with_env`, `recipe_hash`, `_spread`. The runner refuses with a clear
      message (exit 3) if not.

## 4. Expected wall clock

**≈ 60 minutes** (48 launches × 75 s, derived from the loop-memory serving artifacts), **≤ 77 min**
if the full 4-couple replacement budget is used. If the launch cost turns out to match the CPU
session's measured 228 s shape instead, budget **3.0–3.6 h**. Safe to leave unattended: it holds its
own claim, writes every launch to disk as it completes, and stops itself at the stop rule.

## 5. If an arm is lost

The runner already decides this; you do not adjudicate at the console.

| what happened | exit | what it means / what to do |
|---|---|---|
| a launch is screened out (host load > 12, residency unproven, clock moved > 50 MHz, server died) | — | the **couple is replaced, not added** — automatic, blind to the measured rate, so α is unaffected. Budget 4. |
| more than 4 invalid launches | 69 | window was too dirty. Quiet the host, start over in a fresh `--out`. Do **not** merge two windows. |
| `THP_enabled` did not match the arm's declaration | **67** | FATAL, the run stops. Either the knob did not take, or **the control arm was secretly the treatment**. Do not re-run until you know which — this is not a retry-able flake. |
| `THP_enabled` and `AnonHugePages` disagree | **68** | **The disagreement is the finding.** No verdict is issued. Report it; do not paper over it by re-running. |
| device claim lost mid-window | 69 | every measurement inside it is suspect. Discard and restart. |

Everything is on disk as it goes: `launches.jsonl` (fsynced per launch), `thp_proof.txt` (the
positive control, independent of the JSON), `state.json`. A killed run loses nothing but the verdict.

## 6. Go

```bash
python3 /mnt/raid0/llm/tmp/r2358-shim-serving-20260908/run_r2358.py --build <hip-build-of-ef81196d5> --run
```

Drop `--run` for a dry run (the default) — it prints both arms' argv, both recipe hashes, and every
preflight check, and touches nothing.

## 7. Afterwards — what this run may NOT do

It does not promote, fold, advance an anchor, or write anything under `loop-memory`. **Recalibrating
the serving floor under the ON arm is a separate, separately-authorised run**, and it is only called
for if the verdict is `D+`.
