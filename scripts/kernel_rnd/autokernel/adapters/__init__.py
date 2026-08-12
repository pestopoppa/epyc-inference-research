"""autokernel.adapters — the per-backend adapter plane (§13).

One domain-agnostic controller, backend adapters at the edges (AK-D8). An
adapter owns what is *specific* to its backend — its trees, its recipes, its
correctness corpus, its resource lane, its release path — and owns nothing that
the controller or the evaluator already owns.

The load-bearing asymmetry lives here: **not every backend releases the same
way.** `llama_cpu`/`llama_gpu`/`whisper_stt`/`qwentts_tts` release through a
kernel freeze, which is four human-only writes over one sealed package (§1.3).
`serving_runtime` does not release that way at all — it travels the three-gate
stack-change path (§11.6), and its adapter REFUSES the kernel-freeze path rather
than degrading to it (§13.5). An adapter that quietly accepted the wrong path
would let a scheduler change impersonate a kernel era, which AK-D9 and AK-D23
exist to prevent.

No adapter in this package freezes, cuts over, writes a production branch, moves
a stable kernel symlink, writes an era-registry row, or applies an AutoPilot
baseline. Those are human writes (`MEASUREMENT.md:140-142`), and an adapter that
offered them would be a defect, not a feature.
"""
