# A7 / T3 — post-reboot runbook (decision-grade E5 Stage-B placement re-run)

**Staged 2026-08-12 by mainA, pre-reboot. Everything below is validated; nothing was executed.**

## Why this waits for the reboot

`host_health_warnings()` returns exactly one warning today:

    uptime exceeds 1 week; MEASUREMENT.md P-BENCH-1/P-BENCH-3 policy requires reboot
    before decision-grade claims

with `uptime_seconds = 1,200,315` (13.89 d) against `MAX_DECISION_GRADE_UPTIME_SECONDS = 604,800` (7 d).
The driver computes `decision_grade = not warnings and not overrides_active`, so **one warning makes
decision-grade unobtainable, and `--allow-host-health-warning` sets `overrides_active`, which also
forces it False.** There is no flag combination that banks this run decision-grade before a reboot.
The other two conditions are already clean (`numa_balancing = 0`, no resident llama processes), so a
reboot alone opens the window.

## Step 0 — re-check the gate FIRST (this is what makes it decision-grade)

    cd /mnt/raid0/llm/epyc-inference-research
    python3 - <<'PY'
    import sys; sys.path.insert(0,"scripts")
    from benchmark.server_np_sweep import collect_attestation, host_health_warnings
    a = collect_attestation(); w = host_health_warnings(a)
    print("uptime_days:", round(a["uptime_seconds"]/86400, 2))
    print("warnings:", w)
    print("DECISION-GRADE ELIGIBLE:", not w)
    PY

**If that prints anything other than `warnings: []` / `ELIGIBLE: True`, STOP and report.**
Do not pass `--allow-host-health-warning` — it silently downgrades the very claim tier A7 exists to
produce. Note the field is `uptime_seconds`, not `uptime`; a `.get("uptime", 0)` returns 0 and
manufactures a passing reading.

## Step 1 — acquire the regions (acquired, never observed; observing is TOCTOU)

Take the CPU region lock before launching. Do not infer freeness from `ps`/`lsof`.

## Step 2 — run

    cd /mnt/raid0/llm/epyc-inference-research
    for g in gemma4_26b_a4b_q4km_mtp qwen36_27b_q8 qwen36_q8_0 qwen3_next_80b; do
      setsid nohup python3 scripts/benchmark/server_numa_np_sweep.py \
        --manifest-dir data/batched_decode/e5_manifests_a7_placement1/$g \
        --run-id a7-placement1-$g-$(date -u +%Y%m%dT%H%M%SZ) \
        --execute --i-have-operator-grant \
        > /mnt/raid0/llm/tmp/a7-$g.log 2>&1 < /dev/null &
    done

`setsid` matters: a plain `nohup` died with its parent shell when a foreground wait timed out today.

## Step 3 — verify while it runs

* **Live affinity, not the topology hash.** Confirm each `llama-server`'s actual cpuset with
  `taskset -cp <pid>` / `/proc/<pid>/status:Cpus_allowed_list` against the manifest's `cpu_list`.
* **Pair speed with correctness**: `error_rate` must be `0.0` and `success_count == total_count`
  per cell. A cell reporting `0.00 tok/s` with `err=100%` is a failure the driver still banks —
  it writes rows rather than aborting (filed; inference owns the fix).
* **Per-cell persistence** is already the driver's behaviour: `cells.jsonl` grows per cell.

## Step 4 — re-check the gate IMMEDIATELY BEFORE banking

Re-run Step 0. Uptime rises during the run; if anything else changed (a llama process appeared,
`numa_balancing` moved), the claim tier changes and must be recorded, not assumed.

## What is staged and validated

* `scripts/benchmark/a7_generate_placement1_manifests.py` — regenerates confounded cells on the
  ratified grid. Placement constants **imported** from `e5_cell_manifests`
  (`CPUSET_FULL`, `CPUSET_HALF0/1`, `CONFIG_INSTANCE_COUNT`, `K_LADDER`), never retyped.
* `data/batched_decode/e5_manifests_a7_placement1/` — **40 manifests, all dry-run clean through the
  real driver** (8 + 12 + 12 + 8 across four model groups, 0 validation errors).

## Counts, stated honestly

* The row says **27 of 31**. I identified **22** confounded source cells by predicate: declares a
  `stage_b_families` value **and** uses a retired quarter cpuset. The gap is **not reconciled** —
  do not treat 22 as a refutation of 27 without checking whether the row counts something my
  predicate misses.
* 22 source cells x 2 arms = 44, but **40** land: four pairs **converge**. `C2-npN` and `C3-npN`
  differed only in quarter placement, so once placement is declared they are the *same* cell.
  The generator reports each convergence by name and refuses to overwrite silently.

## Three defects the dry-run caught in my own generator, before any compute

1. `config_id` **encodes instance count** (`C1`=1, `C1b`=2, `C2`=2, `C3`=4). Re-shaping a cell
   without remapping it is refused: *"config C3 requires exactly 4 instance(s), got 1"*.
2. `cell_id` must match `{model_key}-{config_id}-np{np}[-suffix]`, so the id must be **rebuilt**
   from the new config, not suffixed onto the old one. Original suffixes (`e1parity`) are preserved.
3. The convergence above was **silently overwriting files** — 44 emitted, 40 on disk, and nothing
   said so. Now reported.

None of these would have surfaced without dry-running through the real driver first.
