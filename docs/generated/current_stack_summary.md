# Current Stack Summary

Generated from structured stack truth. Do not hand-edit this file; run:

```bash
uv run python scripts/registry/stack_change_pipeline.py update
```

Source: `orchestration/derived/stack_priors.yaml`

Source fingerprints:
- orchestration/derived/stack_priors.yaml: `1823ee4e60442231f3a1e8250fe71a1b3f1ed6bbde84230ae15caa8afcfc998c`
- orchestration/model_registry.yaml: `71e7917ec7cb282ea0746698524b638cd994f367908f8cd87df8659066c27e9d`
- orchestration/model_descriptors.yaml: `fe70193dec6d38f049b05ee0b2da5900b36b78498b0e15f5927129bf8585d166`

| Role | Port | Model | Tier | Acceleration | Requirements | Throughput | Description |
|---|---:|---|---|---|---|---:|---|
| architect_critic | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-dflash (lookup=false, draft_max=8) | draft=Qwen3.8-27B-DFlash2-Q8_0.gguf | 47.79 | live_stack; binding=server_mode.direct; status=compiled |
| architect_general | 8074 | Qwen3.8-Flash-Next-UD-IQ4_XS | hot | draft-mtp (lookup=false, draft_max=4) | draft=mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | 52.7 | live_stack; binding=server_mode.direct; status=compiled |
| coder_escalation | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-dflash (lookup=false, draft_max=8) | draft=Qwen3.8-27B-DFlash2-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| frontdoor | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| ingest_long_context | 8083 | Qwen3.8-27B-Q8_0 | hot | draft-dflash (lookup=false, draft_max=8) | draft=Qwen3.8-27B-DFlash2-Q8_0.gguf | 47.79 | live_stack; binding=stack_manifest.alias->server_mode.direct; status=compiled |
| toolrunner | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| vision_escalation | 8086 | Qwen3-VL-30B-A3B-Instruct | warm | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.direct; status=compiled |
| worker_explore | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_general | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_math | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_summarize | 8070, 8080, 8180 | Qwen3.6-35B-A3B-MTP-Q8_0 | hot | draft-mtp (lookup=false, draft_max=4) | embedded_nextn=Qwen3.6-35B-A3B-MTP-Q8_0.gguf | 40.22 | live_stack; binding=server_mode.shared_with; status=compiled |
| worker_vision | 8086 | Qwen3-VL-30B-A3B-Instruct | warm | baseline | mmproj=mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf | 112.2 | live_stack; binding=server_mode.direct; status=compiled |
