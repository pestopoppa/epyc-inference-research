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

EvoEngineer and ARGUS are unavailable, not omitted: their public repositories
do not yet provide licensed controller implementations that can honestly occupy
those named arms. A controller reconstructed from either paper would be
``*-inspired``, not the published system. No MI300X/gfx942 result transfers into
this prospective MI210 comparison.

`arena_campaign_v1.json` fixes the comparison task, file digests, one-at-a-time
MI210 use, and the adopted RE-Bench elapsed-wall-time checkpoints of exactly 2,
8, and 32 hours per controller/task. A ready controller must bind a clean source
commit, entrypoint digest, executable digest, explicit model IDs, and argv. A
licensed port additionally binds the in-repo adapter and external upstream
checkout as separate identities, including upstream entrypoint and licence
digests. The
campaign refuses the entire matrix if any arm, task, source, or hardware identity
is missing; a partial panel is never rankable. Each non-baseline executor call
receives a typed cell request carrying the complete `(2.0, 8.0, 32.0)` checkpoint
tuple and the 32-hour ceiling. The baseline request instead carries an empty
checkpoint tuple and zero authoring budget. The audit also binds the exact config
file and campaign-driver module SHA-256 values, and execution rechecks both to
prevent a ready receipt from being replayed after either input changes.

`arena_cell_runner.py` is the concrete governed implementation of that typed
executor seam. It does not patch the pinned Arena checkout. Instead, each cell
loads Arena's workspace/evaluator modules in an isolated child process while
the parent holds AutoKernel's cross-process `mi210_0` claim and 250 ms device
sampler. The baseline runs once without authoring. Every controller runs three
independent fresh workspaces at 2 h, 8 h, and 32 h; the runner rewrites only the
two declared budget flags and refuses an adapter that does not expose them.
Task and controller identities are re-audited before every claim. Arena alone
compiles, checks correctness, and times the resulting workspace, after which
the runner writes hash-bound cell evidence plus separate Vidya-compatible
correctness and timing-validity beliefs. Partial campaign evidence is retained
but is explicitly non-rankable.

The no-execution audit command is:

```bash
python3 -m scripts.kernel_rnd.autokernel.controller.arena_campaign \
  --config scripts/kernel_rnd/autokernel/controller/arena_campaign_v1.json \
  --arena-root /path/to/AgentKernelArena-at-2dbbf1d3 \
  --geak-root /path/to/GEAK-at-4ffba15a \
  --output /path/to/audit-receipt.json
```

Once (and only once) the audit reaches 8/8, the same inputs execute through the
governed bridge with:

```bash
python3 -m scripts.kernel_rnd.autokernel.controller.arena_cell_runner \
  --config scripts/kernel_rnd/autokernel/controller/arena_campaign_v1.json \
  --arena-root /path/to/AgentKernelArena-at-2dbbf1d3 \
  --geak-root /path/to/GEAK-at-4ffba15a \
  --preflight /path/to/preflight-receipt.json \
  --output-root /new/write-once/campaign-directory
```

If any arm remains unavailable, this command writes the refusal audit and exits
before a device claim, model, compiler, or GPU command is started.

The separately labelled available-source diagnostic uses the same pinned task,
identities, evaluator, and matched budgets, but selects only the baseline plus
the five executable controller arms:

```bash
python3 -m scripts.kernel_rnd.autokernel.controller.arena_cell_runner \
  --available-source \
  --config scripts/kernel_rnd/autokernel/controller/arena_campaign_v1.json \
  --arena-root /path/to/AgentKernelArena-at-2dbbf1d3 \
  --geak-root /path/to/GEAK-at-4ffba15a \
  --preflight /path/to/preflight-receipt.json \
  --output-root /new/write-once/available-source-campaign-directory
```

This produces an availability-conditioned diagnostic only. It records
EvoEngineer and ARGUS as external exclusions, binds the refused parent
eight-arm audit, and cannot imply an eight-arm result or promotion verdict.

## Controller source availability — 2026-08-11

The source gate is split by controller rather than treating every unavailable
arm alike:

- K-Search (`53c8fab9a5e8fab2c86610d24fbec5067f90e115`) is a governed
  executable arm. Its exact `WorldModelKernelGeneratorWithBaseline.generate`
  loop receives a Task whose benchmark method is the centralized Arena
  evaluator, with GPT-5.6 Sol/high fixed as the text-model dependency;
- Xe-Forge `v0.3.0` (`4dcb5080b0f56d0b655ec8c8c9509b8e3ba0382c`) has an explicit
  gfx90a port that retains `DSPyEngine.optimize` and linear CoVeR while routing
  all compilation, correctness, and timing through Arena;
- KernelFoundry `v0.3.0`
  (`1c053e02383d12937f144923bcc1faa82fa7788f`) is a governed executable arm.
  Its inherited `Controller.run_single` retains MAP-Elites/island branching;
  the adapter activates the upstream Triton feature patterns, records measured
  parent transitions through upstream QD tracking, and sends every evaluation
  through Arena;
- GEAK-v1 `v1.0.0` (`4ffba15a55f250816598b4e27eb56ca40a699cea`)
  is a governed executable arm using its real `OptimAgent_ROCm.run`, BM25
  corpus, and reflection memory. The adapter confines upstream cleanup to a
  task-local safe root and routes its dataset evaluation through Arena. That
  controller A/B is separate from reproducing the
  paper's GEAK-eval score; GEAK-eval has no project-level licence and remains a
  corpus/reproduction gate;
- EvoEngineer's repository contains only a release-soon notice and no licence;
  ARGUS has no official source artifact. Those two are external publication
  gates and keep the fixed eight-arm campaign safely refused.

External checkouts may be addressed as `vendor://<checkout>` in the campaign
file. The default root is
`/mnt/raid0/llm/autokernel/vendor/arena-controllers`; an operator may relocate
the clean exact checkouts with `AUTOKERNEL_ARENA_CONTROLLER_ROOT`.

On 2026-08-11 the final clean physical-gfx90a audit pair at research
`26ad617883ec72d417d98815aac38aa585236305` made the two authorities explicit:

- the fixed full panel correctly **refused at 6/8** before inference. Receipt
  `/mnt/raid0/llm/autokernel/probes/inf03-final-audits-20260811-Kpg5wU/full-eight-arm-refusal.json`
  carries receipt SHA-256 `3f67b750c99dccbbe45f5c0043c8aa11973c3e014ab3610bb117786f60a79f7f`
  and file SHA-256 `b432fcb802797136444b510618966489529147aac60d73209b0c1ee946231b1d`;
- the separately labelled available-source panel is **ready at 6/6**. Receipt
  `/mnt/raid0/llm/autokernel/probes/inf03-final-audits-20260811-Kpg5wU/available-source-six-arm.json`
  carries receipt SHA-256 `d812b69ce380613bd854dd0d15206c09981899abac11b10234a8f98bb02482b8`
  and file SHA-256 `88101db4f28f909f220acb3bf906f488fe8b4f8e7c307821d325a5542fd4627d`.

Both receipts bind config SHA-256
`4cadf1e5120c5439132249f6126901207602dc22065a1b643cb41ca68b7dc5ba`
and driver SHA-256 `b839a35cb79627bb27c7f1be6902e91365a1ce8beb0fc26edff58bae5d003866`,
and record that no controller or GPU command executed. The 6/6 receipt has
availability-conditioned diagnostic authority only: it cannot imply an 8-arm
result, rank partial full-panel evidence, or authorize promotion. Only
EvoEngineer and ARGUS remain external source/port/launcher refusals. CLI presence
still implies no controller-family coverage: every ready arm binds a clean
source commit, entrypoint digest, executable digest, and explicit model identity.

The first real one-iteration/two-branch KernelFoundry smoke then tested the
boundary that a no-execution audit cannot. It exposed two integration defects
without producing rankable evidence: v1 could not import the in-tree `scripts`
package from the copied task workspace, and v2 reached both real GPT-5.6 Sol/high
model branches but raced when their shared evaluator materialized the same
workspace. Research `f8569112` supplies the immutable repository import root to
the child, and `8afd016c` serializes shared Arena evaluation while preserving
concurrent upstream authoring.

The v3 smoke completed under a cleanly released MI210 claim. Its receipt is
`/mnt/raid0/llm/autokernel/probes/inf03-kernelfoundry-real-smoke-v3-20260811-Mg6Fl9/smoke-receipt.json`
(receipt SHA-256 `cd61675e83040b196a92aa85f2c0bd951f34912bef10a37e1b07f41864f52276`;
file SHA-256 `9b47fefcb2744392923745385053aa6ee9a8a959102a17acb7eaea079a1be5b1`).
Both model calls returned successfully; the centralized evaluator passed
compilation and correctness, admitted all four baseline and four optimized
timing cases, and reported diagnostic average speedup `0.9986680991832163`.
The controller retained two programs, one occupied MAP-Elites cell, 159 added
Triton patterns, and enabled QD tracking; zero transitions is expected with only
one iteration. The 164.54-second sampler captured 659 samples. This smoke is
explicitly diagnostic, non-rankable, and does not imply the matched campaign.

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
