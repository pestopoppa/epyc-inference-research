# E5 Stage-B v2 runbook: re-measure on the corrected placement (INF-45)

Prepared 2026-11-11 (pipeline-mgmt). Owning box: epyc-root
`handoffs/active/numa-placement-and-contention.md` → **RE-MEASURE E5 Stage-B on the corrected
placement** (absorbs S-04/S-05). Supersedes `E5_STAGE_B_RUNBOOK.md` for any new run. That file and
every banked cell stay as historical records.

## What changed vs the 2026-07-29 grid

| Binding requirement | How v2 meets it |
|---|---|
| (a) `0-95` + `numactl --interleave=all` swept for both models | `C1` = 1 × `0-95` interleave=all at np 1/2/4/8/16/32 for W1 and W4 |
| (b) straddling cpusets interleave or are declared controls | `C1b` halves (`0-47,96-143` / `48-95,144-191`, each spans 2 NPS4 nodes) load a private `--no-mmap` copy under `membind=0,1` / `membind=2,3` (was `interleave=0,1,2,3` on a shared mmap until 2026-11-11; see below). `C1b-defectctl` np 4/8 run policy `none` and are tagged `defect_replication_control: true`. The generator refuses a cell that is neither, and also refuses a cell that leaves the field implicit |
| (c) cache drop before every cell + `pages_by_node` | `cache_reset: per_cell` on every cell. The unchanged harness runs `scripts/lib/cell_cache_reset.py`: `drop_caches` when this process tree holds q0..q3 + gpu-quiet exclusive (the quiet-host grant), else the INF-70/C7 `numa_evict.py` substitute. The row records `cache_reset_method` and before/after `pages_by_node`. Cells declare `placement_protocol: P-BENCH-PLACEMENT-1`, so a numa_evict-only cell cannot grade decision-grade |
| (d) numactl policy attested per instance, fail closed | Generation: a policy must cover its cpuset's nodes, or the cell is refused. After each group, `e5_v2_stageb_grid.py attest --run-dir RUN` compares each instance's observed mmap `pages_by_node` with its declared policy. Interleave needs ≥0.85 of pages inside the declared nodes and every node ≥0.5× its even share. A cell FAILs on disagreement, on a missing witness, or on a missing cache-reset method |
| (e) harness unchanged | `server_numa_np_sweep.py` is untouched except the opt-in `no_mmap` / `membind=` support added 2026-11-11 for C1b (default off). Depth probes use the schema owner's opt-in `depth_probe` block and the harness's existing `--question-pool` override |

**Model set.** W1 is `qwen36_q8_0` (Qwen3.6-35B-A3B-MTP-Q8_0, NEXTN self-draft). W4 is re-seated on
the current CPU role Qwen3.8-Flash-Next UD-IQ4_XS (3 shards, `qwen4exp`). It runs the separate
shared-Q8_0 MTP head as `-md`, with n-max 4 and p-min 0.5. **W2 gemma is dropped**: the
`gemma-4-26B-A4B-it-ORIG-Q4_K_M` GGUF was retired on 2026-09-22 and is not on disk. W3 stays
excluded by operator ruling.

For E5 comparability, W4 keeps these stated deviations from its serving recipe:
- KV is q8_0 (recipe: f16).
- Weights are mmap+mlock (recipe: `--no-mmap`).
- `-t` equals the cpuset size (recipe: 48).

**C1b halves: private `--no-mmap` copies, per-half membind (2026-11-11, pipeline-mgmt).**
Superseded design: with a shared mmap the two C1b instances share one page cache of the GGUF, so
each half's weights interleaved over all four nodes and requirement (b) half-locality was not met.
Now every C1b cell carries `no_mmap: true` and each half runs `numactl --membind` of its own nodes
(`0-47,96-143` → `membind=0,1`, `48-95,144-191` → `membind=2,3`). Each process allocates its own
weight copy under its own bind. Harness change (amends "(e) unchanged"): `server_numa_np_sweep.py`
accepts `membind=<nodes>` and emits `--no-mmap` when a manifest sets `no_mmap: true` (default off,
so C1 and the `C1b-defectctl` controls launch byte-identically and keep the shared mmap). The
schema (`e5_cell_manifests.py`) accepts `membind=<nodes>` only with `no_mmap: true`; the generator
refuses a membind that is not exactly the cpuset's nodes. `attest` reads the preflight's anon-page
witness (it switches to anon pages when the cmdline has `--no-mmap`) and passes a membind instance on
≥0.85 locality alone (no balance requirement: bind fills nearest-first).
RAM check (2026-11-11): two copies of W4 = 2 × (87.3 GiB weights + 2.6 GiB MTP head) ≈ 180 GiB,
W1 = 2 × 35.2 GiB ≈ 70 GiB; grant budget = MemAvailable 1006.6 GiB − 100 GiB margin ≈ 906 GiB, and
each half's 2 nodes total ≈ 566 GiB. Fits. Rows that ran from the pre-change snapshot (r1) used the
old shared-mmap C1b and are not comparable on C1b.

**Depth points (>16k).** These feed INF-45 capacity curves. They use `C1` only, at np 1/2/4/8 and
target depths 16k and 32k tokens. Each depth has 8 pinned long-context prompts built by
`e5_v2_stageb_grid.py build-pools`, which is deterministic and sha256-pinned in each manifest. The
realised depth is each request's `prompt_tokens`, not the nominal target.

## Groups (one group per grant; the row resumes per group)

`data/batched_decode/e5_v2_manifests/groups.json`. Estimates come from the 2026-07-29 per-cell
walls: W1 64.6 min / 11 cells; W4 80B 88.3 min / 12 cells, scaled ×1.5 for Flash-Next bytes.

| Group | Cells | Est. |
|---|---:|---:|
| `qwen36_q8_0` | 13 | 78 min |
| `qwen36_q8_0-d16k` | 4 | 30 min |
| `qwen36_q8_0-d32k` | 4 | 60 min |
| `qwen38_flash_next_ud_iq4xs-p1` | 7 | 63 min |
| `qwen38_flash_next_ud_iq4xs-p2` | 6 | 54 min |
| `qwen38_flash_next_ud_iq4xs-d16k` | 4 | 45 min |
| `qwen38_flash_next_ud_iq4xs-d32k-p1` | 2 | 45 min |
| `qwen38_flash_next_ud_iq4xs-d32k-p2` | 2 | 45 min |

## Commands

Regenerate, validate and dry-run (zero inference):

```bash
python3 scripts/benchmark/e5_v2_stageb_grid.py build-pools      # byte-identical rebuild
python3 scripts/benchmark/e5_v2_stageb_grid.py generate --dry-run
python3 scripts/benchmark/e5_v2_stageb_grid.py generate
.venv/bin/python scripts/benchmark/e5_v2_stageb_grid.py dryrun --output-root <scratch>   # harness, no --execute; all groups rc 0
python3 scripts/benchmark/e5_v2_qualify_banked.py --stamp        # banked-cell tags + superseded_by_run stamps
```

To run one group inside a quiet-host grant (whole host + GPU quiet, held by the caller's process
tree):

```bash
G=qwen36_q8_0; M=data/batched_decode/e5_v2_manifests/$G
ARGS=(); for f in $M/*.json; do ARGS+=(--cell-manifest "$f"); done
POOL=$(python3 scripts/benchmark/e5_v2_stageb_grid.py pool-for "$(ls $M/*.json | head -1)")   # depth groups only
.venv/bin/python scripts/benchmark/server_numa_np_sweep.py --llama-server /mnt/raid0/llm/kernels/production/cpu/llama-server \
  "${ARGS[@]}" ${POOL:+--question-pool "$POOL"} --execute --i-have-operator-grant --run-id e5v2-$G-$(date -u +%Y%m%dT%H%M%SZ)
python3 scripts/benchmark/e5_v2_stageb_grid.py attest --run-dir data/batched_decode/<RUN>
```

Never pass `--allow-host-health-warning` or `--skip-clean-check`.

## Stop rules (pre-registered)

- **Stop-on-answer:** stop per pre-registered iso-T pair (`isoT_T{2,4,8,16,32}`: C1@T vs
  C1b@T/2). The pair is answered when the sign of the C-shape contrast `(C1b − C1)/C1` on aggregate
  decode tok/s lies outside the measured cell noise. The noise is 1.96 × the root sum of squares of
  each cell's per-request relative standard error. Unanswered pairs feed a re-run row; they never
  widen this one.
- **Futility:** stop the row when the host-health gate fails twice, meaning two consecutive group
  attempts are refused for health.
- Formal-gate cells are futility-only at interim looks, until OP-ABSEQ, OP-PQP and OP-ALPHAWIN are
  decided.

## After the run

- R1–R4 via `server_numa_np_sweep.py --summarize-run` on each attested group run.
- Capacity curves via `scripts/benchmark/capacity_curve_fit.py`. The W0 e1parity twins enter only
  when `banked_qualification.jsonl` tags them `QUALIFIED`, which none are today: the cache axis is
  UNRESOLVED for every banked cell.
- The T=32 `empty_trimmed_window` and the R2/R4 mixed-basis items must be re-checked against the
  new runs.

## Banked-cell qualification (2026-11-11)

`e5_v2_qualify_banked.py` tagged 249 banked manifests: 149 DISCARD, 100 UNRESOLVED, 0 QUALIFIED.
The live counts differ from the figures filed on 2026-08-11:
- W0-mismatch: 79 live (filed as 69). The 10 extra W0-window cells are the
  `gemma4_26b_a4b_q4km_nomtp_v9` scout copies in `e5_manifests/`; 79 − 10 = 69 exactly.
- Missing reasoning: 204 live (filed as 172). 103 + 69 = 172 exactly. The difference is 32 A7
  placement manifests written after the census.

All 79 regenerated W0 cells now carry `superseded_by_run`, which points to their run of record.
