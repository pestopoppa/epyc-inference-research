# PREDECLARATION exl3-6c5-quality (INF-80 EXL3-6c5, CORRECTNESS half only) - written 2026-10-25 BEFORE compute

Class: correctness (load + greedy smoke + perplexity). NO speed figure is read or claimed. The speed half of EXL3-6c5 (codified recipe, `GGML_FUSED_DECODE_OFF=1`,
pp5/pp256/tg128 vs U) is TIMING and needs a quiet window; it is NOT in this launcher (the 10-04 CPU-window-2 speed result stands: pp5 -30.9%, pp256/tg128 inconclusive).
Handoff box: `handoffs/active/exl3-cpu-mi210-implementation.md` EXL3-6c5 (origin/main). Runbook: `EXL3-CPU-STATUS.md` -> Window runbook (tree `llama-exl3-cpu-20261003`).

## Objects (pinned; the job refuses on any mismatch)
- Build: `/mnt/raid0/llm/worktrees/llama-exl3-cpu-20261003/build/bin`, branch `experimental/exl3-cpu-20261003`, HEAD `20192b81da1df690753e0b96295efb4ee2316ffb`, clean tracked tree;
  `libggml-cpu.so.0.16.0` sha256 `50343614c4c9b94d634dc50dbe0c715a691f974e19be771a150e860bf4134298`, mtime after HEAD's code commits, `GGML_EXL3_ISA` compiled in (strings).
- E (EXL3 3.05 bpw): `/mnt/raid0/llm/models/exl3-gguf/Qwen3.8-Flash-Next-EXL3-3.05bpw/model-00001-of-00002.gguf` (49,740,163,328 B) + shard 2 -> `qwen4exp-ple-table-ud-iq4nl.split2of2.gguf` (28,800,138,528 B).
- U (reference, the only one left): `/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf` (10,946,624 / 49,835,229,856 / 43,836,407,744 B).
  Both arms run on the SAME EXL3-tree binary (U has no EXL3 tensors), so the contrast is not confounded by the tree. The 10-04 U run used the champion build `cpu-20260925-90c12df42`.
- Text: `/mnt/raid0/llm/data/wiki.test.raw` sha256 `bfe9eb16ab9987fb88bde4ea9a30a00f2a45db01dfc14bad78d05325789c4f12`.
- No high-precision Q38FN base on disk, so no KLD; PPL is the fallback the box names.

## Units (each recorded in out/steps.txt, resumable; order E then U)
1. E_smoke_p1..p3 and U_smoke_p1..p3: `llama-completion -t <grant cores> -fa on -ctk f16 -ctv f16 --no-mmap --temp 0 -n 48 -no-cnv --no-display-prompt`, prompts
   p1 `The capital of France is`, p2 `Explain in two sentences why an L3 cache is faster than main memory:`, p3 `def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n`.
2. E_ppl and U_ppl: `llama-perplexity -f wiki.test.raw -c 512 --chunks 40 -b 512 -ub 512 -fa on -ctk f16 -ctv f16 --no-mmap` (the 10-04 shape).
Env (codified recipe): `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1 GGML_FUSED_DECODE_OFF=1 GGML_FA_SPLIT_KV=0 GGML_NOHUGEPAGE_PROCESS=1`,
`LD_LIBRARY_PATH` = the EXL3 build bin only (linkage proven by `verify_ggml_linkage.sh`), `taskset` + `numactl --interleave` from the grant.

## Decision values
- SMOKE = PASS iff E loads and every E output has >= 20 characters AND the common character prefix of E vs U text is >= 100 (p1), >= 40 (p2), >= 40 (p3)
  (10-04: p1 identical over 48 tokens, p2 diverged at token 45, p3 at token 22). Else SMOKE = FAIL (a result, with the texts).
- PPL: delta = PPL_E / PPL_U - 1 on the 40-chunk numbers. QUALITY_NOT_WORSE iff delta <= +1.0%; QUALITY_WORSE iff delta > +1.0%. |delta| below the larger of the two +/- stderr
  is reported as WITHIN_STDERR. Per the 10-04 caveat, small PPL gaps between 4-bit-class files (Q8_0 KLD floor 0.022-0.027, wikitext overlaps the memorised n-gram set) are NOT resolvable as quality;
  the claim is non-inferiority on PPL only, never "better". The project gate for quality stays a paired per-item task eval.
- REPRO flag (not a gate): PPL_U within 0.5% of 2.9925 and PPL_E within 0.5% of 2.8909 (the 10-04 values) = REPRODUCES_10_04, else DIFFERS_10_04 (reported, thread count and tree differ).
- Verdict COMPLETE iff all 8 units finished and parsed; INCOMPLETE otherwise (re-grant resumes). PASSED only on COMPLETE.

## Stopping rule
A unit starts only if remaining grant >= 1.3 x its estimate (smoke ~300 s E / ~360 s U, ppl ~900 s). Stop on STOP file / grant end / unit failure (rc 1). Completed units are never re-run.

## Belief capture
Native producer `epyc-inference-research/scripts/kernel_rnd/exl3/evidence.py` `write()` -> `epyc.exl3.measurement.v1` rows, projected by `scripts/vidya/adapters/exl3.py` (`cli.py ingest exl3-measurement`; source-table row "EXL3 CPU/gfx90a performance and quality receipts (INF-80)").
The job performs no git and does not call write(): it stages the raw logs, texts, digests (binary, model, text) and RUN_RECORD.json under `evidence-staged/data/exl3-6c5-quality/` so the owner emits the rows (this also closes the box's "rows were not emitted" item).
