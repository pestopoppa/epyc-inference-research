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

That last distinction is now machine-enforced by `arena_campaign.py`.
`CONTROLLERS` is discovery metadata; it is not implementation coverage. The
primary INF-03 comparison is exactly eight arms:

1. the Arena-measured starting state (the score-zero baseline, with no authoring);
2. Claude-planner + Codex-actor/critic;
3. EvoEngineer;
4. KernelFoundry;
5. K-Search;
6. Xe-Forge; and
7. GEAK-v1; and
8. ARGUS.

ARGUS is unavailable, not omitted: its arm names the missing public licensed
source, gfx90a controller path, and hash-bound Arena wrapper/model pins. No
MI300X/gfx942 result transfers into this prospective MI210 comparison.

`arena_campaign_v1.json` fixes the comparison task, file digests, one-at-a-time
MI210 use, and the adopted RE-Bench elapsed-wall-time checkpoints of exactly 2,
8, and 32 hours per controller/task. A ready controller must bind a clean source
commit, entrypoint digest, executable digest, explicit model IDs, and argv. The
campaign refuses the entire matrix if any arm, task, source, or hardware identity
is missing; a partial panel is never rankable. Each non-baseline executor call
receives a typed cell request carrying the complete `(2.0, 8.0, 32.0)` checkpoint
tuple and the 32-hour ceiling. The baseline request instead carries an empty
checkpoint tuple and zero authoring budget. The audit also binds the exact config
file and campaign-driver module SHA-256 values, and execution rechecks both to
prevent a ready receipt from being replayed after either input changes.

The no-execution audit command is:

```bash
python3 -m scripts.kernel_rnd.autokernel.controller.arena_campaign \
  --config scripts/kernel_rnd/autokernel/controller/arena_campaign_v1.json \
  --arena-root /path/to/AgentKernelArena-at-2dbbf1d3 \
  --geak-root /path/to/GEAK-at-4ffba15a \
  --output /path/to/audit-receipt.json
```

On 2026-08-11 the physical-gfx90a audit correctly refused before inference at
**1/8 executable arms**. The starting-state baseline, pinned task, pinned vendor
sources, and MI210 identity are ready. The seven controller blockers are recorded
verbatim in the receipt. Claude Code 2.1.227 and Codex CLI 0.147.0 are visible on
the host; Cursor and a `geak` CLI are not. CLI presence explicitly implies no
controller-family coverage. Most importantly, an `argv` can no longer be labelled
EvoEngineer, KernelFoundry, K-Search, Xe-Forge, GEAK, or ARGUS and treated as coverage
without a clean source commit, entrypoint digest, executable digest, and explicit
model identity.

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

The controller-coverage audit is
`/mnt/raid0/llm/autokernel/probes/inf03-controller-ab-audit-20260811/receipt.json`
(file SHA-256
`3152e2fa97b52d9f0b91fb43449375adec66128189fddf253b0772fadfcf59c4`;
internal payload SHA-256
`3533ee0955ecbdfd61be58d45f69f15030de9dbf1a1238148a220b14b8bfd138`).
Its `status=refused` is the intended safe outcome: no controller or GPU command
ran, and no incomplete comparison was reported.
