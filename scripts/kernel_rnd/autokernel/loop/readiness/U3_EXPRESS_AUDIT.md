# U3-EXPRESS pre-fix readiness draft

Source: Research 5e3cf5355bc701f6c055ce83836fcdabfca8cfcd; production llama.cpp source ffc1bac82eeca6f9099e1ccd9ba49703c460a115. This is an offline source audit, not a measurement or a claim that the full campaign is ready. `u3_express.py` is an explicitly invoked, intentionally failing unittest suite outside default test discovery. No runtime changes.

`loop/serving.py` Recipe serializes with `to_dict`/`from_dict`, hashes the serialized template and emits `server_argv`; `loop/resolved_recipe.py` validates template, pins model/drafter/executable/DSO identity and derives a capability report and execution identity. Raw argv emission alone does not establish supported resolved execution. `serving.py:1196` constructs legacy request bodies; frozen request bytes and their digest are a distinct request surface.

| Surface | Observed current behavior / readiness case |
|---|---|
| Multi-GPU | Comma device reaches argv; resolver explicitly rejects multi-device. `test_multigpu`. |
| Split modes, tensor split | Raw flags reach argv, resolver refuses unnormalized extensions. `test_split_mode`, `test_tensor_split`. Frozen `common/arg.cpp:2600` and `:2624` document actual arguments. |
| Per-device NGL | No field for independent placement; `test_per_device_ngl` demonstrates schema refusal only. Frozen `common/arg.cpp:2582` parses one N, auto or all, while tensor split takes proportions. An exact per-device placement implementation and proof need design; no invented comma-NGL CLI. |
| LoRA, control vectors | Raw flags reach argv, resolver rejects extensions. `test_lora`, `test_control_vector`. The model/drafter-only artifact set does not bind adapter content: `test_lora_content_identity` and `test_control_vector_content_identity` propose pinned artifact sets and require different execution digests for changed contents at the same path. Both currently error at strict artifact-set validation. Field names are proposed test interface, not an existing native contract. |
| RPC | `--rpc` reaches argv; resolver explicitly rejects RPC and extensions. `test_rpc`. |
| Per-arm env and absence | Multiple Recipe env entries round trip, resolve and distinguish recipe hashes; unset vs inherit differ and server_env removes the inherited value (`test_multienv_control`, `test_unset_differs_from_inherit_control`). However `runtime_arms.py:140` and `actors.py:3285` accept exactly one env key per arm. `test_runtime_arm_multi_env` proposes one atomic set of two keys and demands a real planner pair with both changes; it currently errors in arm declaration. Readback/witness policy still matters; these controls do not prove live process state. |
| Draft | Existing self-draft absence omits `-md`/`-ngld` and changes hash relative to external drafter; `test_selfdraft_absence_control`. Structured p-min, n-min and draft cache are serialized but not emitted and resolver rejects unknown spec fields. `test_draft_p_min`, `test_draft_n_min`, `test_draft_own_cache`. `--draft-p-min` extension is a supported alternate route (`test_draft_p_min_extra_flag_control`). Native campaign `scripts/benchmark/laguna_cpu_dflash_observation_runner.py:1261` uses `--spec-draft-n-min`, `--spec-draft-p-min`, and `--spec-draft-type-k/v`. |
| Grammar and sampler | The server-default flags in the readiness cases are raw argv but rejected by resolver. Frozen requests accept exact bytes and change request digest (`test_frozen_request_grammar_and_sampler_control`); this control does not demonstrate server execution or validate grammar semantics. Request-level grammar/samplers and server-default flags are distinct. |
| Cache and no KV offload | Cache K/V independently round trip into actual argv and resolve (`test_cache_types_control`); adding `--no-kv-offload` is refused as an extension (`test_no_kv_offload_cache_combination`). |

The proposed test schemas for atomic env sets, per-device placement and auxiliary artifact lists are concrete candidates, not existing interfaces. Build cost: teach the Recipe, validation and launch/command parser an unambiguous bounded representation; carry it through both resolver variants, canonical projection and execution digest; bind every referenced auxiliary artifact by digest; update runtime dimension/planner for atomic env mutations and per-key witnesses; and provide actual backend lowering or explicitly refuse where per-device placement cannot be lowered. Multi-GPU and RPC additionally require residency, distributed input and resource/claim verification. None of those requirements is a reason to dismiss the requested surface.

Run with host UID under the approved R4 pair lock, taskset 28,29 and prep supervisor: `PYTHONPATH=scripts/kernel_rnd python3 -m scripts.kernel_rnd.autokernel.loop.readiness.u3_express`. The 2026-11-01 guarded run had 24 cases: 8 pass, 12 fail, 4 errors; native reason traces and supervisor receipt are retained at `U3_EXPRESS_RUN.log` in this directory. Failures are not xfailed and the suite is opt-in. These are pre-fix expectations, not a claim of successful backend execution. Full scope and design require parent review before runtime changes.

## Native red points and matched controls

The complete original unittest traceback and the supervisor JSON receipt are in `U3_EXPRESS_RUN.log`. The following map reports the observed failure point, not a predicted future failure:

| Cases | Observed native reason |
|---|---|
| `test_multigpu` | capability `multi_device_unsupported`, `unsupported_gpu_device` |
| `test_rpc` | capability `extra_flags_unsupported`, `rpc_unsupported` |
| `test_split_mode`, `test_tensor_split`, `test_lora`, `test_control_vector`, `test_grammar`, `test_sampler_variant`, `test_no_kv_offload_cache_combination` | capability `extra_flags_unsupported` after checking actual emitted argv |
| `test_draft_p_min`, `test_draft_n_min`, `test_draft_own_cache` | requested native flags absent from the actual generated argv; the full argv is in the log |
| `test_per_device_ngl` | constructor `TypeError`: unexpected `per_device_ngl`; if accepted later, test checks exact serialization, hash, and either changed lowering or a specific unsupported capability |
| `test_lora_content_identity`, `test_control_vector_content_identity` | `ResolutionError`: unknown proposed `loras` / `control_vectors` artifact-set keys |
| `test_runtime_arm_multi_env` | `ArmDeclarationRefused`: env candidate requires exactly `key` and `value` |

The matched `test_runtime_arm_single_env_control` passes with the exact same native arm metadata, validated canonical anchor, provenance and `actors._runtime_pair` path; only its candidate representation changes from a single key to an atomic two-key set. `test_frozen_request_grammar_and_sampler_control` checks request bytes and digest; `test_grammar` separately tests the server default argv with the valid literal `root ::= "x"`. The source audit of `serving.py:1196-1220` establishes where those frozen bytes enter the HTTP request; these offline tests do not send HTTP requests.

## Pinned source and evidence SHA-256

All paths are relative to the Research root unless absolute. The audit file is excluded from its own catalog to avoid a self-referential digest; Git pins its contents.

| File | SHA-256 |
|---|---|
| `scripts/kernel_rnd/autokernel/loop/serving.py` | `772177dd47012ba18b1b05621305eccb0bd15ad49d2605ed5da07b3679f6c462` |
| `scripts/kernel_rnd/autokernel/loop/resolved_recipe.py` | `4f841f1f01c7d8329ffe5c1be2199922d8878a7d02b1224964679309f784984a` |
| `scripts/kernel_rnd/autokernel/loop/runtime_arms.py` | `62a8ef9344f89548c430b7f26344dac64f4d9dce74631cd1aed4e0e07b13a780` |
| `scripts/kernel_rnd/autokernel/loop/actors.py` | `e6232b6cb69101b5a90c31668194167d22ac2c0214eb6db753313e3fde833ce8` |
| `scripts/kernel_rnd/autokernel/loop/unified_planner.py` | `55761428bed0198ea7835810254d256120c8730e39342ee949476a6b3d45ea6d` |
| `scripts/benchmark/laguna_cpu_dflash_observation_runner.py` | `f38d5abc82175322f9225805bc118d734c1325a45af1e1eaab0c8e1c55de50a4` |
| `scripts/kernel_rnd/autokernel/loop/readiness/u3_express.py` | `763bbb60013262b0a864475388adc1510ac216e3235d761a53ddc5d2fc49b068` |
| `scripts/kernel_rnd/autokernel/loop/readiness/U3_EXPRESS_RUN.log` | `2c2be4751ca08dda8730c22e354148259a39a5152b73d451b8fc0d8962203dc1` |
| `/mnt/raid0/llm/llama.cpp/common/arg.cpp` | `72b3defe53d296a0cfe73016986fe32d9100f9e27a0aafbd0277999eda51c8e9` |
