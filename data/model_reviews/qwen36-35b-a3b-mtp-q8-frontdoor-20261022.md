# One-time GGUF review: Qwen3.6-35B-A3B-MTP-Q8_0.gguf (frontdoor, worker_general/summarize/math/explore, toolrunner)

Operator decision 2026-10-22 (SEC-ART-3, option B): `unsloth` and `lmstudio-community` are allowlisted
re-quantizers; this file is the one-time review of the tool-bearing frontdoor/worker GGUF that the decision asked for first.

Registry provenance before this review: `artifact_status.provenance: local_artifact_no_hf_sidecar_found`.

Findings (HF tree API, metadata only; no GGUF bytes hashed here):
- Local file: 37801097504 bytes, mtime 2026-06-25. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` @ 5bc3e238d916 holds
  `Qwen3.6-35B-A3B-Q8_0.gguf` with the byte-identical size 37801097504 (LFS oid c1283d8b80c3...), so the local file is a
  renamed copy of that artifact (or a same-size forgery; the full-hash registration is what excludes the latter).
- The repo's last modification (2026-05-20) PRE-DATES the local file (2026-06-25): the pinned revision is the one
  that was current when the file was downloaded, so no intervening re-upload explains a size-equal-but-different file.
- Repo is not gated, author `unsloth`, base_model `Qwen/Qwen3.6-35B-A3B` (first-party). The repo carries the usual quant
  set, `imatrix_unsloth.gguf_file` and three `mmproj-*.gguf`; no unexplained extra file, no `.safetensors`/pickle/refusal-direction sidecar.
- Template: the served template is the EPYC terse file; its diff against the first-party `Qwen/Qwen3.6-35B-A3B` template is
  `data/template_reviews/frontdoor.diff` (CT-1b work).
- Launch flags that change generation (registry): `--chat-template-file` (terse template; label matches function) and
  `chat_template_kwargs.enable_thinking=false`. No `--lora`, `--control-vector` or `--override-kv`.

Remaining step (not done in this review): `register_gguf_fingerprints.py register --role frontdoor` full-hashes the file
and must reproduce sha256 c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131; a mismatch voids this review.
