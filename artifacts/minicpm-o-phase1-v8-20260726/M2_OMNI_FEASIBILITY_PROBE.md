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

## Terminal Pinned-Interface Result (2026-07-26)

The exact detached pin was cloned and its CPU `llama-omni-cli` target was built
with `-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`; the latter is required on
this host because CMake otherwise stops for missing CURL development files. The
built executable SHA-256 is
`4cb1ee507de9e3965419c5824b598dbffb732b38c7836dd7f17ac3af324a8f30`.

The binary must be run with its own dynamic-library directory first:

```bash
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-omni-experimental/build-cpu/bin \
  /mnt/raid0/llm/llama.cpp-omni-experimental/build-cpu/bin/llama-omni-cli --help
```

Without that binding, the loader selects the production `libllama.so` and fails
on `llama_apply_adapter_cvec`; that is a loader-path issue, not a production
tree defect. With the local binding, help and `tools/omni/omni-cli.cpp` confirm
that the CLI accepts no text/prompt input and no output-WAV path. Its only
input-driving operation is `--test <audio-prefix> <n>`, which consumes numbered
WAV fixtures, and it calls `stream_decode(ctx, "./")` into its own nested output
directory. It therefore cannot satisfy the M-2 runner's required exact text
input and `$RUN/output.wav` contract without changing the pinned source.

**Terminal result:** `blocked-by-pinned-interface`. Do not substitute mainline
OuteTTS, modify the pinned source, or treat an audio-fixture response as the
Path-B text-to-speech observation. The M-2 manifest remains blocked.

## Local Asset Preflight

The local MiniCPM directory contains the Q4 LLM, audio encoder, TTS model and
projector, vision model, and all five Token2Wav GGUFs required by the upstream
layout. There is no need to download model weights for the initial source/build
probe.

## CPU Reproduction And Deferred Work

The following CPU commands are the exact working configure/build/help sequence
used for the terminal feasibility result:

```bash
OMNI=/mnt/raid0/llm/llama.cpp-omni-experimental
cmake -S "$OMNI" -B "$OMNI/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build "$OMNI/build-cpu" --target help | rg 'llama-omni|omni'
cmake --build "$OMNI/build-cpu" --target llama-omni-cli -j "$(nproc)"
LD_LIBRARY_PATH="$OMNI/build-cpu/bin" \
  "$OMNI/build-cpu/bin/llama-omni-cli" --help
```

The HIP build remains deferred and was not part of the terminal CPU result:

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S "$OMNI" -B "$OMNI/build-hip" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a
cmake --build "$OMNI/build-hip" --target llama-omni-cli -j "$(nproc)"
```

Any bundled-input smoke also remains deferred. It requires operator approval
and a revised test contract that does not misrepresent the pinned CLI's
audio-fixture response as the required Path-B text-to-speech observation.
