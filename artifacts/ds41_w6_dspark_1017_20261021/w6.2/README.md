# DS41-W6.2 (+ W5a kernel, W5b probe): booked slot driver (prepared 2026-10-21)

Slot: **2026-10-22T22:00Z -> 2026-10-23T00:30Z (hard)**, q0+q1 (`--cpu-list 0-47`, `-t 48`), build role
(correctness/indicative). It chains after the DS41 AK floor run (flag
`/mnt/raid0/llm/tmp/compute-flags/DS41_AK_FLOOR_ENDED`, which must be fresh, i.e. mtime >= 2026-10-22T00:00Z).

Live copy: `/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/`. The scripts here are the committed record.

**Window change (coordinator, 2026-10-21 20:5xZ): `driver2.sh` supersedes `driver.sh`.** The CS-47 quiet latency run
(90 min, full host) now goes first, right after the AK floor. `driver2.sh` (PID 1645930, armed 20:53:15Z) starts at the
later of a fresh `/mnt/raid0/llm/tmp/compute-flags/grants/cs47-latency.ended` and 2026-10-22T23:30Z. If that flag
never appears it falls back to 23:35Z, and the region-lock claim still blocks while 0-47 is held. **Hard stop
2026-10-23T02:00Z**, exported to `inner.sh`. `driver.sh` (PID 1360538) was stopped on purpose while still in its wait loop,
and its false end flag was removed. Everything below otherwise holds.

- `driver.sh`: waits (shell sleep) for now >= 22:00Z AND the fresh flag, then takes ONE
  `region-lock run --cpu-list 0-47 --role build --tag ds41-w62-slot` claim. The acquire is bounded so
  it can never start later than 00:15Z, and it never starts within 20 min of HARD. It always writes
  `/mnt/raid0/llm/tmp/compute-flags/DS41_W62_ENDED`. Timeline:
  `/mnt/raid0/llm/tmp/compute-flags/DS41_W62_TIMELINE.txt`. `DRY_RUN=1` validates everything and runs
  nothing.
- `inner.sh`: strictly sequential steps, each bounded by `timeout` to HARD:
  W5A (kernel NMSE harness, no server), then W5B (one target-only server, the probe set), then the
  W6.2 pairs p3 A,B / p4 B,A / p5 A,B / p6 B,A (ABBA). p3's arms also run the full greedy
  text-identity pass on the same server (`greedymode 1`; this saves two 9-shard loads). p6 runs only
  if the measured pair time fits.
- `memgate.sh`: before EVERY load, MemAvailable - 300 GiB must be >= 100 GiB, and no other
  `llama-server` naming a DS41 GGUF may be alive (a read-only /proc scan). It waits up to 10 min,
  then refuses.
- `arm.sh`: the W6.1 arm argv unchanged (same binary, target, drafters, `--spec-draft-n-max 3`), but
  run sequentially at `-t 48` on 0-47.
- `client.py`: byte-identical to W6.1's client. Seeds are `1000*p + 234 + prompt`, as in W6.1.
- `analyze_post.py`: for the main session after the end flag exists. It pools the complete pairs
  (p1-p2 from `../w6.1/runs`, p3+ from here) and computes a paired-t CI over per-pair diffs plus the
  W6.1 bootstraps. On the W6.1 rows alone it reproduces W6.1's `analysis.json` exactly. It also
  computes greedy identity (p3A vs p3B) and summarises W5a and W5b. It writes `analysis.json` and a
  summary page into this directory; `--commit [--push]`.

Pair-config caveat: p1-p2 ran both arms concurrently at `-t 24`, while p3 onward run sequentially at
`-t 48`. The design is paired within each pair, so the per-pair diff is the unit. Greedy text identity
compares only same-config arms (the 2026-10-21 thread-count lesson).
