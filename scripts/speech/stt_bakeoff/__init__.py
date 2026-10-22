"""STT bake-off scoring package (conversation-stack CS-8.STT1 / CS-4.STT1-2 / CS-8.STT7 / CS-8.LEX0-1).

Deterministic, CPU-only scoring code. Nothing in this package runs inference, loads a
model, starts a server or downloads data. Corpus audio and reference text for private
strata are NEVER committed here (the repository is public); only manifests of hashes.
Import from the repository root as `scripts.speech.stt_bakeoff`.
"""
