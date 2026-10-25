# INF-66 R23-64 CPU decode thread re-sweep (2026-10-25)

Champion 141e4bcf5 (build 10412, store build cpu-20261023-141e4bcf5-lbh), Qwen3.8-Flash-Next UD-IQ4_XS + shared Q8_0 MTP. Arms t48 (control) / t64 / t96, 6 launches each, unit = launch. Job code pinned to research cb86b6cfd (adds rc 68 AFF fix); finished 2026-10-25T22:29:05Z rc=0.

Result: t48 44.297 t/s (CV 1.37%); t64 37.647 (-15.01% [-17.09,-13.34], RESOLVED); t96 43.464 (-1.88% [-3.04,-0.82], MDE(observed) 1.95%, UNRESOLVED). Floor sd 0.609% (launch unit).

Affinity: all 18 launches log `live affinity OK` (union_cpus == threads, within 0-95); rc 68 never fired in the final run. sweep.log also holds the earlier halted attempts (01:16Z false rc=68; 18:41Z `AFF: unbound variable` after t48_r1's arm, whose data was reused by resume skip). Host at start clean (no KFD, 0 VRAM).

Files: ANALYSIS.*, RESULT.txt, row.txt (filler row), sweep.log, steps/, runs/ (per-launch rows/meta/timeline/argv/env/host/foreign; server logs under 5 MB included). Source: /mnt/raid0/llm/tmp/roi-prep-20261023/INF-66-R23-64/out/. Integrity: SHA256SUMS.
