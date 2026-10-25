# Chat template review diffs

These unified diffs preserve the reviewed changes from upstream templates to the served template
bytes. The `template_diff_reviewed` field in `orchestration/model_registry.yaml` records the
review signature and SHA-256 for each of the four production review diffs.

| Diff | What it records | Registry entry and signature |
|---|---|---|
| `frontdoor.diff` | Qwen3.6-35B-A3B upstream template to the served `epyc-qwen3x-v1-terse.jinja`. | `server_mode.frontdoor` (`orchestration/model_registry.yaml:1007–1010`), signed by Daniele; `diff_sha256` `98b0d55c…`. |
| `architect_critic.diff` | Qwen3.8-27B upstream template to the same served v1 template. | `server_mode.architect_critic` (`orchestration/model_registry.yaml:1792–1795`), signed by Daniele; `diff_sha256` `cfdc3654…`. |
| `architect_general.diff` | Qwen3.8-Flash-Next upstream template to its served GGUF-embedded template. | `server_mode.architect_general` (`orchestration/model_registry.yaml:2012–2015`), signed by Daniele; `diff_sha256` `3f80812a…`. |
| `worker_vision.diff` | Records that the served Qwen3-VL-30B-A3B template is byte-identical to upstream. | `server_mode.worker_vision` (`orchestration/model_registry.yaml:1265–1268`), signed by Daniele; `diff_sha256` `05bdf611…`. |

The two `.v2.diff` files are regression-test fixtures used by
`scripts/test_epyc_qwen3x_v2_template.py` (SEC-ART-POST-b.1). They compare each base model's
upstream template with the proposed `epyc-qwen3x-v2` template. The four registry signatures above
attest the listed non-v2 diffs; neither `.v2.diff` has a separate `template_diff_reviewed`
signature in the registry. Their shared `review_digest_sha256` is
`50c357749c54064da46aa8f9a0e49f9fd0d63f4cb08924469d8a6767b29e2516`.

- `frontdoor.v2.diff`: Qwen3.6-35B-A3B upstream to the v2 template.
- `architect_critic.v2.diff`: Qwen3.8-27B upstream to the v2 template.

`SHA256SUMS` pins the exact bytes of all six diffs in this directory. Verify from this directory
with `sha256sum -c SHA256SUMS`.
