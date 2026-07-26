# M-2 Omni Feasibility Probe

This is a deferred feasibility probe, not a production integration or a claim
that Path-B TTS is available on MI210. Do not start it before the Q8 download
chain is terminal and Laguna 2b powered confirmation is complete.

## Pinned Source And Isolation

The upstream README directs `feat/web-demo`, resolved read-only on 2026-07-26
to `5202b7b2f4d11f50b9f996161e7a2f8b8571b890`. Create an independent detached
worktree only:

```bash
git clone --filter=blob:none --no-checkout https://github.com/tc-mb/llama.cpp-omni.git /mnt/raid0/llm/llama.cpp-omni-experimental
git -C /mnt/raid0/llm/llama.cpp-omni-experimental checkout --detach 5202b7b2f4d11f50b9f996161e7a2f8b8571b890
```

Never use the frozen production tree or `llama.cpp-experimental` for this
probe.

## Classified Blockers

1. At the pinned `feat/web-demo` commit, `tools/omni/CMakeLists.txt` defines
   `llama-omni-cli`; the advertised `llama-omni-server` target is absent from
   the tree/CMake target inventory. The README server/API path cannot be run
   until this source/documentation mismatch is resolved.
2. The CLI has no text-prompt argument. Its `--test <prefix> <n>` feeds numbered
   WAV files and optional same-stem JPG files, then asks the model to respond.
   It is not a clean deterministic text-to-speech acceptance test.
3. Upstream documents CUDA/Metal, not ROCm. The CLI restricts the vision backend
   option to `metal` or `coreml`, and the Token2Wav GPU route is CUDA-guarded.
   A HIP build is an unproven compile/runtime probe, not supported configuration.

## Local Asset Preflight

The local MiniCPM directory contains the Q4 LLM, audio encoder, TTS model and
projector, vision model, and all five Token2Wav GGUFs required by the upstream
layout. There is no need to download model weights for the initial source/build
probe.

## Deferred Commands

First inspect the pinned target inventory, then build only the CLI. Do not run
the HIP command if the CPU configure/target inventory fails.

```bash
OMNI=/mnt/raid0/llm/llama.cpp-omni-experimental
cmake -S "$OMNI" -B "$OMNI/build-cpu" -DCMAKE_BUILD_TYPE=Release
cmake --build "$OMNI/build-cpu" --target help | rg 'llama-omni|omni'
cmake --build "$OMNI/build-cpu" --target llama-omni-cli -j "$(nproc)"
"$OMNI/build-cpu/bin/llama-omni-cli" --help

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S "$OMNI" -B "$OMNI/build-hip" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a
cmake --build "$OMNI/build-hip" --target llama-omni-cli -j "$(nproc)"
```

Only after both builds and CLI help are captured should an operator approve a
bounded bundled-input smoke. Its output needs WAV header/duration inspection,
explicit process cleanup, and a finding whether it actually demonstrates the
Path-B text-to-speech requirement.
