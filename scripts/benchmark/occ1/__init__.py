"""OCC-1 — optical (bitmap-frame) context compression vs raw text, on a served local reader.

epyc-root handoff: handoffs/active/optical-context-compression.md (index row RTG-53).

Modules
    fixture   SQuAD v1.1 dev history fixture (hash-pinned), paired question sampling, EM/F1
    render    pixel-font text -> PNG frames aligned to the reader's 32-px token grid
    costs     reader-side image-token prediction (port of llama.cpp mtmd smart_resize)
    prompts   carrier/question prompt bytes (identical question block across arms)
    stats     paired contrasts (exact McNemar on EM, paired bootstrap on F1)
    run_occ1  CLI: plan (no inference) | run (needs a live llama-server) | report

Rendering and SQuAD handling are ported from @oh-my-pi/snapcompact research/bdf.py and
research/squad.py (MIT, oh-my-pi @ 37eee7197) — see README.md for attribution.
"""
