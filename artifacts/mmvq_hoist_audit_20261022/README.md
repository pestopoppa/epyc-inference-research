# mul_mat_vec_q weight-hoist static ISA audit (INF37-LB1, 2026-10-22)

Zero-GPU, zero-build. Source: `/mnt/raid0/llm/kernels/production/gpu/libggml-hip.so.0.16.0`
(v10 kernel store, sha256 f26a166b..., built 2026-09-21 14:38 from `ffc1bac82`, ROCm 6.2 clang 18).
Tool: `scripts/kernel_rnd/mmvq_hoist_audit.py` (reads via `gfx90a_isa_audit.py`); tests in
`scripts/kernel_rnd/test_mmvq_hoist_audit.py`. `table.txt` is the per-cell table, `audit.json` the
full rows plus verdicts. Columns: `Wld`/`Yld` weight/activation VMEM loads per hot-loop iteration,
`R` rows per thread, `W/row` weight loads per row (flat in N = hoisted), `W/unit` = per block per column.

Result: Q8_0 and IQ4_XS hoist their weight loads (3 per row per block, flat for N=1..8). Q4_K does
not: its 3 dword `qs` loads per row are hoisted but its 3 ushort d/dmin/scale loads per row are
re-issued per column (6 -> 27 weight loads per row from N=1 to N=8), and its nibble unpack VALU
per row scales with N (21/unit -> 13.25/unit instead of falling as 1/N). N=4 is the exception
(`load_q4_K_b4_weight` path hoists, but its loop duplicates load arms, so its Y count is flagged
`load_overcount` and excluded from the verdict).
