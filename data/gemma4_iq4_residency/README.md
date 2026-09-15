# gemma4-26B-A4B UD-IQ4_XS — MI210 residency, MTP A/B and CPU control

Measurement evidence made durable on **2026-09-15** (NIB2-73a). The master registry cited
three artifacts in this campaign, all of which existed only as UNTRACKED files in the
shared clone `/mnt/raid0/llm/epyc-inference-research` — so the evidence gate passed in that
one checkout and reported `MISSING` in every worktree. Copied byte-for-byte within the same
clone (sha256-verified, all 16 files); the untracked originals were left in place.

| | |
|---|---|
| origin | `/mnt/raid0/llm/epyc-inference-research/data/gemma4_iq4_residency/` (untracked) |
| measured (UTC) | 2026-07-18 16:24 – 17:07 |
| made durable | 2026-09-15 |
| carried | **16 files, 28,835 bytes** — of 189 files / 298,216 bytes |
| kernel | experimental-v7 `d1e5a20ebebe567f0da6bc64ca7ea7ecd521fc24` (build 10088), `experimental_dirty: true` |

## Registry claims this backs

`orchestration/model_registry.yaml`, all under
`roles.gemma4_26b_a4b_ud_iq4xs_local.performance`. Line numbers as of 2026-09-15.

- **L9822** &nbsp;`current_v7_mi210_observation` →
  `gemma4_26b_ud_iq4xs_mi210_v7_20260718T162446Z/summary.json` (4,977 B, status `pass`).
  llama-bench `pp2048 2449.01 t/s`, `tg256 81.91 t/s` on an MI210; an 8K server/chat
  coherence probe processed 6,971 prompt tokens at 2,257.80 t/s and generated 201 tokens at
  76.02 t/s with exact repeated-token compliance (`bad_word_count: 0`). Cleanup verified no
  KFD PIDs remaining.
- **L9823** &nbsp;`current_v7_mi210_mtp_observation` →
  `mtp_ab_local_20260718T170739Z/summary.json` (4,812 B). No-spec q8-KV control at
  `pp2048 2450.51 t/s`, `tg512 81.41 t/s`; external assistant-head MTP with
  `--spec-type draft-mtp --spec-draft-n-max 2` decoded 542 eval tokens at **117.01 t/s**
  and accepted **360/362 drafts (0.99448)**, mean draft length 2.99 — roughly 1.44x decode
  over the control. **Caveat recorded in the artifact itself:** strict JSON content was not
  clean (`json_items_ok: false`, `checksum_ok: false`), and the log shows
  `draft model memory: failed to create llama_context from model`. The speed number is real
  and the output quality is not attested; do not quote the first without the second.
- **L9825** &nbsp;`current_v7_cpu_observation` → the directory
  `20260718T164058Z_cpu_only_v7_control` (14 files, 11,923 B — **no `summary.json`**; the
  numbers live in `llama_bench_stdout.json`). llama-bench with `-dev none -ngl 0 -nkvo 1
  -t 96`, q8 KV: `pp2048 261.26 t/s`, `tg512 11.15 t/s`. This is the baseline that makes
  the MI210 figures a speedup — roughly **9.4x prefill and 7.3x decode** — so it is the
  single artifact without which the other two claims are unanchored numbers.

The role's status is `current_v7_mi210_speed_coherence_pass_quality_retention_open`, and
`constraints.forbid` still carries
`production_stack_registration_without_quality_retention_gate`: **quality retention versus
ORIG Q4_K_M is unmeasured.** Nothing here licenses a production cutover.

## Why the CPU control was carried whole

The citation at L9825 names the *directory*, not a file inside it, and the directory is
11,923 bytes. Carrying a subset would leave a path that resolves to a directory whose
contents differ from what was measured, which is a worse failure than a few kilobytes of
process dumps: the `preflight_pgrep.txt` / `postflight_pgrep.txt` /
`final_cleanup_rocm_smi_showpids.txt` files are the *cleanup proof* that the sibling
observations lean on ("no KFD PIDs after cleanup"), so they are cited in substance even
though no line names them. `metadata.txt` pins host, branch
`experimental-v7-refresh-20260716`, commit `d1e5a20eb…` and the 16:40:58 → 16:43:51 window.

## What was NOT carried — 173 files, 269,381 bytes

Five other run directories in this campaign are not cited by the registry and are not
carried: `20260718T164705Z/` (20,438 B), `gemma4_26b_ud_iq4_xs_mi210_v7_20260718T163644Z/`
(46,517 B), `gemma4_26b_ud_iq4_xs_mi210_v7_gpu_20260718T164140Z/` (40,858 B),
`mtp_ab_20260718T170731Z/` (23,811 B) and
`quality_retention_optimized_v7_20260718T2212Z/` (28,395 B). **Watch the names**: the cited
`…ud_iq4xs…162446Z` differs from the uncited `…ud_iq4_xs…163644Z` and
`…ud_iq4_xs…gpu_164140Z` by one underscore and a timestamp, and those are different runs
with no `summary.json`.

Within the two cited run directories, only `summary.json` was carried. Left behind:
`request.json` (69,607 B, the 8K coherence prompt payload), `response.json`, the `logs/`
subtrees, `preflight.txt`, `cleanup_proof.txt`, the `pre`/`post` `rocm_smi` and `pgrep`
dumps, `mtp_metrics.txt`, `mtp_server.stderr.txt` and the per-arm `llama_bench.json`
files. Each `summary.json` already embeds the llama-bench rows it distils (including
`samples_ns`, `model_size`, `model_n_params`, cpu_info "AMD EPYC 9655 96-Core Processor"
and gpu_info "AMD Instinct MI210") plus the coherence-probe and acceptance results, so the
cited claims are readable without the substrate. It all remains untracked at the origin
path above.

Note for anyone considering publication: the carried files contain no PII and no
credentials (scanned: zero email addresses, zero credential-shaped strings), but they do
carry first-party operational detail — the hostname `Beelzebub` in the CPU control's
`metadata.txt`, local absolute paths under `/mnt/raid0/llm/`, local commit SHAs, and OS
PIDs in the process dumps.

## Integrity

`SHA256SUMS` seals all 16 carried files, hashed after the copy and compared against the
originals (all verified `OK`). `README.md` is deliberately not in it (documentation is not
evidence, and hashing it would make every doc edit break the seal), following the
2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/gemma4_iq4_residency/SHA256SUMS
```
