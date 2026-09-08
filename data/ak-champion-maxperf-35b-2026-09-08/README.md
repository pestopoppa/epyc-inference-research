# Champion maximum performance — Qwen3.6-35B-A3B-MTP-Q8_0 on MI210, 2026-09-08

**Build `ef81196d5`**, model `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` (35.2 GB),
**MTP self-drafting** (`--spec-type draft-mtp --spec-draft-n-max 4`, **no separate drafter**),
recipe `qwen3.6-35b-a3b-q8-gpu-mtp` with **only `np` varied**. 3 launches per point except np=1
which is **6**. **Unit is the LAUNCH.** All points residency `proven`.

| slots | aggregate tok/s | per slot | p95 dev | peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 112.68 (n=6) | 112.68 | 2.70% | 36.6 GB |
| 2 | 130.54 | 65.27 | 1.38% | 36.9 GB |
| 4 | 189.57 | 47.39 | 1.13% | 37.5 GB |
| 8 | 242.75 | 30.34 | 1.23% | 38.8 GB |
| 12 | 268.12 | 22.34 | 1.88% | 40.2 GB |
| 16 | **310.96** | 19.43 | 2.53% | 41.4 GB |

## What this establishes

**Not saturated at 16 slots.** +28% from 8 to 12, +16% from 12 to 16, and VRAM is only 41 of 64 GB.
**The ceiling is UNMEASURED** — 24/32 slots was offered and the operator chose to stop. That is a
recorded decision, not a gap in the data.

**It beats the dense 27B at BOTH ends** — 112.68 vs 79.25 single-user, 310.96 vs 179.12 aggregate —
while being the larger model on disk. **This is architecture, not a kernel result**: A3B is a
mixture-of-experts activating ~3B parameters per token, so it decodes faster than a dense 27B.
Do not read the gap as evidence about the kernel.

**The two models want different operating points.** The 27B curve turns over hard between 4 and 8
slots (np=4 gives 94% of peak aggregate at ~42 tok/s per user). This one is still climbing at 16.

**The np=1 spread did NOT tighten when samples doubled** (2.35% at n=3 → 2.70% at n=6), so it is a
real property of this model at one slot rather than a small-sample artifact. Any single-user headline
for the 35B must carry it. The 27B's np=1 spread was 0.44% — a 6x difference between two models on
one harness, unexplained.

## Note on how this became measurable

`Recipe.server_argv` required a separate drafter GGUF unconditionally, so a self-drafting model
raised `KeyError` and was **inexpressible as a recipe**. Fixed in `c3e362a1` (research repo):
absence of `drafter` is now the declaration that the model drafts for itself. This was the **third**
recipe-expressiveness gap found in one day, after env vars and the recipe identity hash — each
surfaced only when a slightly new question was asked.

Raw: `sweep.json` (np 1-8), `sweep-hi.json` (12, 16), `sweep-np1-n6.json` (the tightened single-slot
point), the two logs, and `sweep.py` as executed.
