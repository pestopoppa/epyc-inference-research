# GEAK / AgentKernelArena on MI210

`arena_adapter.py` is the narrow integration seam between AutoKernel and the
paper-era AMD agent frameworks. It does not vendor either project, score a
kernel, or put an arena on the campaign import path.

The source inputs are fixed to:

- AgentKernelArena `2dbbf1d3f676b948c04c339de50516fe80ed4ab9`
  (Apache-2.0);
- GEAK v1 `4ffba15a55f250816598b4e27eb56ca40a699cea`
  (Apache-2.0).

The preflight refuses a moved or dirty checkout, hashes the license and adapter
surface, exercises AgentKernelArena's `@register_agent` decorator, and proves
that the physical device enumerates as `gfx90a`. It records a useful vendor
limitation explicitly: at the pinned revision the decorator accepts an external
adapter, but `AgentType` cannot select it until the vendor enum/import,
prompt-builder, and postprocessor dispatch lists are overlaid. The pinned
architecture table also has no MI210 entry. AutoKernel therefore supplies the
MI210/gfx90a context and all three compile environment variables itself; it
refuses `HSA_OVERRIDE_GFX_VERSION` and any conflicting target.

The registered controller IDs are `claude_codex_actor_critic`, `evoengineer`,
`kernelfoundry`, `k_search`, `xe_forge`, `geak_v1`, and `argus`. Registration
means the same adapter contract can launch them and compare whole-agent task
results. It does not imply that an unexecuted controller has an MI210 result.

An arena-side launcher should:

1. import `register_agentkernelarena_adapter()` in a tiny
   `agents/epyc_autokernel/launch_agent.py` module;
2. pass AgentKernelArena's `register_agent` decorator and a three-argument
   wrapper around its normal prompt builder;
3. add `epyc_autokernel` to the vendor `AgentType`, import dispatch,
   prompt-builder dispatch, and general postprocessor list (the preflight records
   why these four paper-pin overlay edits are required);
4. provide `eval_config.epyc_autokernel` with a registered `controller_id`, an
   argv that reads the prompt from stdin, and optionally a C4 report path plus
   its SHA-256;
5. leave compile, correctness, timing, held-out shapes, and scoring to the
   arena/evaluator.

The returned arena score remains `whole_agent_task_only`; C4 input remains
`diagnostic_only`. Neither is an AutoKernel promotion verdict.

## On-box substrate reproduction — 2026-08-11

An isolated Python 3.12 environment at
`/mnt/raid0/llm/tools/geak-v1-rocm62-py312` resolved the paper-era runtime to
PyTorch `2.5.1+rocm6.2`, HIP `6.2.41133-dd7f95766`, and Triton `3.1.0`. Under
exclusive device claim `akd-80bf8f3910734f3a`, AgentKernelArena's Apache-2.0
`instruction2triton/rocmbench/test_add_kernel` task completed:

- source compile preflight: exit `0`;
- correctness: `3/3` selected tests passed;
- timing harness: `5/5` selected tests passed across FP16/FP32 and two sizes;
- physical device identity: `gfx90a:sramecc+:xnack-`.

The diagnostic receipt is
`/mnt/raid0/llm/autokernel/probes/inf03-geak-arena-add-roundtrip-20260811/receipt.json`
(file SHA-256
`aee866ee3ebd2fe88b37185f4226c3c20a5b439c60a40fac57ef1bb42898be8c`).
This closes compile/correctness/timing compatibility for one baseline Arena
task. It does not reproduce a GEAK-authored candidate, rank a controller, or
transfer any MI300 result to MI210.
