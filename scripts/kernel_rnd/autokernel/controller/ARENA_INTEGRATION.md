# Licensed Arena controllers on MI210

`arena_adapter.py` is the narrow integration seam between AutoKernel and the
paper-era AMD agent frameworks. It does not vendor either project, score a
kernel, or put an arena on the campaign import path.

The source inputs are fixed to:

- AgentKernelArena `2dbbf1d3f676b948c04c339de50516fe80ed4ab9`
  (Apache-2.0);
- GEAK v1 `4ffba15a55f250816598b4e27eb56ca40a699cea`
  (Apache-2.0);
- EvoToolkit/EvoEngineer paper release `data-v1.0.0`, exact commit
  `1649715a975b9022c84b5279c88aaef0b73b28dc` (MIT).

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

ARGUS is unavailable, not omitted. EvoEngineer is now source-admitted and
executable through the same parent-worker broker, controller sandbox, and
isolated evaluator boundary as the other licensed upstream arms. The
availability-conditioned diagnostic panel is therefore seven arms; it is not a
substitute for the primary eight-arm comparison. ARGUS still has no official
licensed source artifact. A controller reconstructed from the ARGUS paper would
be ``ARGUS-inspired``, not the published system. No MI300X/gfx942 result
transfers into this prospective MI210 comparison.

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
executor seam. It does not patch the pinned Arena checkout. The parent worker
loads Arena only for the immutable starting-state compilation/baseline. Every
candidate compilation, correctness check, and timing pass—intermediate and
final—runs in a fresh subprocess under the `candidate_evaluator_gpu_v1`
profile. That profile default-denies read/exec outside the copied task, pinned
Python/ROCm/Arena inputs, and exact system runtime roots; grants read/write only
to `/dev/kfd`, `/dev/dri/renderD128`, `/dev/null`, and the copied task, plus
read-only access to exact `/dev/urandom` for PyTorch initialization; denies all networking,
broker inheritance, cross-process memory, io_uring, signals, ptrace, namespaces,
modules, BPF, and mounts; and owns a fresh process group/session and cgroup.
The parent alone owns AutoKernel's cross-process `mi210_0` claim and 250 ms
device sampler around each child lifetime. `/proc` remains absent from the read
allowlist; the first claim-scoped isolated probe must fail closed if ROCr needs
a narrower evidenced exception.

Licensed upstream controllers send every intermediate candidate to a
parent-worker-owned Unix-socket broker. The parent materializes complete,
bounded candidate bytes in a fresh task copy and serializes all evaluations
against the immutable, strictly JSON-serialized starting-state baseline cases.
The socket uses an in-memory nonce, exact `SO_PEERCRED` PID plus procfs
start-time binding, a persistent descriptor connected before controller sandbox
activation, contiguous ordinals, and a durable hash chain. A bounded
pre-registration wait closes the connect-before-`Popen`-callback race. Peer
disconnect, timeout, signal, or broker teardown cancels the exact child process
group, drains/removes its cgroup, releases the claim, and emits no result.
Model subprocesses receive no broker descriptor or credential. Each measurement
window has its own atomic, self-hashed open/release/sampling receipt and embeds
the child activation/teardown receipt. Durable validation rehashes the pinned
Arena evaluator source, child request, serialized baseline, stdout/stderr,
persisted child result, exact evaluation value, and cgroup lifecycle. The
baseline runs once without authoring. Every controller runs three
independent fresh workspaces at 2 h, 8 h, and 32 h; the runner rewrites only the
two declared budget flags and refuses an adapter that does not expose them.
Task and controller identities are re-audited before every claim. Arena alone
compiles, checks correctness, and times the resulting workspace, after which
the runner writes hash-bound cell evidence plus separate Vidya-compatible
correctness and timing-validity beliefs. Partial campaign evidence is retained
but is explicitly non-rankable.

The executor is restart-safe at the expensive-checkpoint boundary. Its first
invocation writes an immutable, self-hashed campaign manifest binding the audit,
config, preflight, source/controller identities, runner digest, task order, arm
order, and checkpoint order. Repeating the exact command with the same output
root re-audits those inputs, verifies every durable checkpoint receipt, artifact
digest, belief receipt, and both released measurement-window claims, then skips only checkpoints
whose complete evidence still matches. A directory left in flight without its
atomic checkpoint receipt is preserved under `execution/abandoned/` and rerun
from a fresh workspace; a completed-but-tampered receipt refuses the resume
instead of being reused or overwritten. Per-task/arm cell receipts are also
published atomically. `execution-receipt.json` does not exist until the entire
declared matrix has been reconstructed in order, and partial evidence never
acquires ranking authority.

Validation is independently read-only and does not resume a campaign:

```bash
python3 -m scripts.kernel_rnd.autokernel.controller.arena_cell_runner \
  --validate-only --output-root /durable/campaign-directory
```

It accepts legacy v1 manifests as historical evidence. V2 manifests bind the
logical campaign to the run-directory `attempt_id`, so device-claim journals
cannot conflate repeated logical campaign IDs. Nested measurement-window and
belief identities are checked semantically against their enclosing checkpoint.

`arena_controller_sandbox.py` now provisions the reusable controller half of
that OS boundary. It copies one task into a new workspace, discovers a
fail-closed runtime allowlist from an exact real Python, licensed controller and
repository source roots, the exact Codex package/CLI and shebang chain, real
Node executable, auth and CA files, exact DNS/NSS/hosts/OpenSSL configuration,
ELF loader, and shared-library closure, and then constructs
`execution.sandbox.CONTROLLER_PROFILE`. Broad roots, devices, production trees,
campaign/evidence state, symlinks, and duplicate authority are refused. Exact
executable files are a distinct Landlock capability: the interpreter and ELF
loader can execute without granting their sibling directories. The adapter
returns `command_prefix`, fixed startup environment overrides, the activation
receipt path/policy, `process_started(pid)`, and strict activation-plus-cgroup
teardown verification for `arena_adapter.launch`; it never accepts an arbitrary
environment or command prefix.

The controller permits unnamed `socketpair` IPC because the pinned Codex
binary's Tokio signal driver requires it; such a pair has no filesystem or
external peer. Creating a new AF_UNIX socket is still denied. The pinned Codex
client also requires client-side `bind` before outbound traffic, while
`listen`/`accept` remain denied; only the wrapper-created peer-bound broker
stream is inherited.

The wrapper preconnects its one broker stream before Landlock/seccomp and before
the launch callback can register its exec-stable PID. A broker therefore queues
an accepted stream until that exact PID and procfs start time are registered,
then authenticates `SO_PEERCRED`; ancestry and uid admission are forbidden. A
live tiny-controller test proves this ordering, inherited broker use, denied
KFD/render/campaign-sibling and sibling-executable access, and descendant
cgroup cleanup. It uses a fake local Codex package and no model or GPU work.

Candidate evaluation isolation is also implemented. Every intermediate and
final candidate now runs in a fresh restricted-read GPU evaluator subprocess,
with exact device admission, networking denied, and request/result/baseline,
activation, teardown, and current vendor-evaluator identities durably
revalidated. The controller adapter is wired into the cell runner, and the
first claim-scoped evaluator compatibility probe has succeeded. Those facts
close the implementation gates for isolated execution; they do not grant the
compatibility probe ranking, aggregate, belief, champion, or release authority,
and the primary campaign remains refused while ARGUS leaves it at 7/8.

### Isolated evaluator compatibility sequence — 2026-08-12

The attempts under
`/mnt/raid0/llm/autokernel/probes/inf03-isolated-evaluator-compatibility-20260812-r*`
are a compatibility-debug sequence, not six performance repetitions. R1-r5
are immutable **invalid compatibility defect attempts**:

- r1 reached no child result because Python startup could not obtain hash
  randomness (activation file SHA-256
  `386881fb1b700dd8d0fc488f3f3bb7114b46c310077ba265d2ea8ca1b8c149b0`);
- r2 reached no durable execution result because the pinned Arena package root
  was outside the read allowlist (activation file SHA-256
  `82011249f50a99873af2efeede05b5c804ca99954bdd3498f865c96aadd4d213`);
- r3 emitted terminal receipt SHA-256
  `ea03ff11164bba0af2cb16374927fb867383ced38b456ee193339786e2694a19`
  but its result receipt
  `f948ae2148e91c8b3b73916047ddd998505567302118f61dac4b756ae2126421`
  records `pass_correctness=false` because `/dev/null` was denied;
- r4 emitted failed result receipt
  `8537f488cd45ce09bf0396e0e1cdefa39763af90306df2c466a58dc1d66257d4`
  and no terminal receipt because PyTorch could not open `/dev/urandom`; and
- r5 emitted failed result receipt
  `0306d543e07edd588622421f0bea8a8f7e2058466d1f19dd6a3da487cc937516`
  and no terminal receipt because Triton's runtime helper build could not
  execute the exact GCC `cc1` helper.

None of r1-r5 is a valid compatibility pass, measurement comparison, or input
to ranking or belief. Each defect led to a narrow evidenced allowlist change;
no attempt widened the evaluator to `/dev`, `/proc`, or a broad filesystem
root.

R6 is the first and only compatibility pass. Its terminal receipt is
`/mnt/raid0/llm/autokernel/probes/inf03-isolated-evaluator-compatibility-20260812-r6/receipt.json`,
with receipt SHA-256
`cc129a4a5e0424ae17d284e43b06425e386f5f6e6c29d929666699ede81c2405`
and file SHA-256
`ac8027a2d052d455ff042a0fe34916485a749fff5f6094f799ab2b976cb604e6`.
The isolated evaluator execution receipt is
`ee15214bbc4516849316176dd54bf9193f3074d13191231a0e64e949cdd076e1`;
its child result is
`d21b086ad659ddb882f05af782b75b60baa64657cc201a8e51956d7074682c4d`;
and its baseline/final measurement-window receipts are respectively
`5d949c2dda9d370ea9569119a81327494996c21f86365da9c4d8525c5665d603`
and `c484ad66794b89d7f714d13292f6e22ef0d4e659791f74a634c40965af100321`.
Both windows held and released `mi210_0`; the evaluator passed compilation and
correctness with 4/4 baseline and 4/4 optimized cases, denied networking, kept
`/proc` outside the read allowlist, and verified its cgroup empty and removed.
The observed `0.9991096842915198` starting-state/self speedup is deliberately
**compatibility-only**: the receipt says
`compatibility_only_no_ranking_or_promotion_authority`, no model inference ran,
and the value must not be used as a kernel or controller performance claim.

### Governed K-Search diagnostic pilot — 2026-08-12

The attempts under
`/mnt/raid0/llm/autokernel/probes/inf03-mi210-isolated-k-search-pilot-20260812-r*`
are one evolving compatibility-debug sequence, not repeated measurements. R1-r12
closed missing task-source discovery, exact module/runtime/ELF/Git/null/config
allowlisting, and ephemeral Codex-home projection. R12 reached the real Codex
client but the controller seccomp policy rejected client-side `bind(2)`. A
seccomp-only differential reproduced that failure and showed that admitting
only `bind` while retaining denied `listen`/`accept`, denied AF_UNIX creation,
and exact filesystem/device allowlists was sufficient. R13 then completed the
worker but exposed a stale parent validator that expected no controller
`/dev/null`; r14 exposed the symmetric stale deny-all evaluator-network receipt
expectation. R1-r14 are immutable invalid diagnostic attempts and must not be
resumed, ranked, aggregated, or banked.

R15 is the first terminal pass. Its receipt is
`/mnt/raid0/llm/autokernel/probes/inf03-mi210-isolated-k-search-pilot-20260812-r15/diagnostic-pilot-receipt.json`,
with internal receipt SHA-256
`3425cf579a9d6fb06f5ed76b480ae652eaed3b3685eec9116299d253528a8771`
and file SHA-256
`eb53de389450ba6ef800528a0da6f54c8508eabc39b4357f0cd9beff64aacefa`.
The exact one-task, one-round, `gpt-5.6-sol:high` K-Search controller made six
governed model calls. Its brokered intermediate candidate passed compilation
and correctness with 4/4 optimized cases and diagnostic ratio
`1.0033912745189357`; the final isolated evaluator also passed 4/4 and reported
`1.0019039030109136`. These ratios prove plumbing and candidate round-trip only;
the pilot explicitly denies matched-campaign, cross-controller ranking, belief,
promotion, and release authority.

The intermediate broker result file SHA-256 is
`c910243c8f7e11988f0246db30a4d1ad58a10928144f6d1b62ef3cb7ef522090`;
the final evaluator execution receipt is
`2d9fa2b575a021be8f385064b12f0ff665f1d0b136eedbdea99dc4cd171d7116`.
Baseline, intermediate, and final GPU windows held and released identical claim
IDs and captured 11, 21, and 21 samples respectively. The controller admitted
only `/dev/null`, remained KFD/render blind, denied `listen`/`accept` and new
AF_UNIX sockets, and inherited one peer-bound broker stream. Both evaluator
invocations denied all networking and admitted exactly `/dev/kfd`,
`/dev/dri/renderD128`, and `/dev/null`. Every cgroup was verified empty and
removed, and the ephemeral controller Codex home was scrubbed before success
publication. The warning-strict controller suite passed 578 tests plus 202
subtests after the terminal pilot.

The 2026-08-12 available-source r4 attempt is immutable defect evidence, not a
valid partial campaign. KernelFoundry performed 64 intermediate vendor
evaluations in the controller process while only the two outer checkpoint
windows were claimed and sampled. Its five checkpoint and two grouped-cell
receipts remain useful for diagnosing publication boundaries, but none has
claim-window integrity and no r4 value may enter a ranking or belief update.

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
  --output-root /durable/campaign-directory
```

If any arm remains unavailable, this command writes the refusal audit and exits
before a device claim, model, compiler, or GPU command is started.
For a ready campaign the directory must initially be absent. After interruption,
run the identical command against that same directory; changing any bound input
or attempting to reuse a different pre-existing directory fails closed.

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
  --output-root /durable/available-source-campaign-directory
```

This produces an availability-conditioned diagnostic only. It records
EvoEngineer as source-admitted/runtime-pending and ARGUS as an external source
exclusion, binds the refused parent eight-arm audit, and cannot imply an
eight-arm result or promotion verdict.

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
- EvoEngineer-Full is source-admitted from `pgg3/evotoolkit` at
  `1649715a975b9022c84b5279c88aaef0b73b28dc`. The release's CUDA tutorial cites
  the exact EvoEngineer paper; the controller, Full interface, 91-task release
  asset, and paper environment agree. The licence SHA-256 is
  `3a18133891b736252655b83391edfef51bd52aa317198fcc4374eb5f16e99de3`
  and controller SHA-256 is
  `28c56fbeb8663c9084734c8682dea39df4a539e1680eee782a8553046963e50d`.
  Current EvoToolkit `master` is not substituted because its later refactor
  removed the paper-specific CUDA task surface. The arm remains runtime-pending;
- ARGUS has no official source artifact and remains an external publication
  gate. Together with EvoEngineer's runtime dependencies, this keeps the fixed
  eight-arm campaign safely refused.

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
result, rank partial full-panel evidence, or authorize promotion. At the time of
that receipt, EvoEngineer and ARGUS were both recorded as external
source/port/launcher refusals. CLI presence
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

Three more available-source controller smokes completed under the same
one-iteration diagnostic contract, each with compilation and correctness
passing, all four baseline and four optimized timing cases admitted, and the
exclusive MI210 claim released:

- K-Search: receipt
  `/mnt/raid0/llm/autokernel/probes/inf03-k-search-real-smoke-20260811-tQf7zZ/smoke-receipt.json`
  (receipt SHA-256
  `6eea9028399635083a6aed7a4d0101aa106cc4393e4225dd768c6f27c23e7704`;
  file SHA-256
  `74f49b472dcb6b2eed1cef66e706e9471ea67f71c94de7a8ba046e3bcd7520b7`).
  One upstream round produced three model-output artifacts; 666 samples over
  166.30 seconds accompanied diagnostic average speedup
  `1.0033678996172277`.
- Xe-Forge: receipt
  `/mnt/raid0/llm/autokernel/probes/inf03-xe-forge-real-smoke-20260811-NIffwN/smoke-receipt.json`
  (receipt SHA-256
  `a53ef172b42d3fbb6008902a865eac9c884d181a6cc0fc0cc981f9e8aad1ccae`;
  file SHA-256
  `e82917d9f1f751520317700520f33b7bf62dd372749ef18ce3d822ba5b2806ea`).
  One upstream iteration produced three model artifacts; 565 samples over
  141.08 seconds accompanied diagnostic average speedup
  `0.999612223103817`.
- GEAK-v1: its first two attempts correctly released their claims after a
  cached top-level `agents` namespace shadowed the clean Arena checkout. The
  namespace-isolation repairs are research `4e01cf48` and `743f59df`. The
  terminal receipt is
  `/mnt/raid0/llm/autokernel/probes/inf03-geak-v1-real-smoke-v3-20260811-RJ4UYN/smoke-receipt.json`
  (receipt SHA-256
  `0deef125d026625055e77a63270c444101fd96ce5f6f9b2fce433e47b509a229`;
  file SHA-256
  `a4c30029a38011cfc7ca0b59be1eb826463cc336322db57e7d6b2cdea19d7487`).
  Its 739 samples over 184.62 seconds accompanied diagnostic average speedup
  `0.9955954625720872`.

These near-1.0 observations are smoke telemetry, not comparative performance
claims. They are neither rankable across controllers nor a substitute for the
governed 6/6 available-source campaign at its declared checkpoints.

The Claude/Codex actor-critic smoke then exercised the final available
controller boundary across five diagnostic attempts. V1 exposed a strict
single-fenced-JSON response that the raw parser rejected; v2 exposed a
workspace-contained absolute candidate path; v3 reached planner, actor, and
critic but the actor's nested workspace sandbox could not configure loopback;
and v4 validated the replacement container confinement but found that Docker
stdin was not attached. Research `84e2f948`, `dd0daedd`, `da677443`, and
`22e60940` repair those boundaries. The final actor runs in a digest-pinned
container with a read-only root, one writable workspace bind, dropped
capabilities, no-new-privileges, exact-name teardown, and ephemeral auth
staging; the receipt discloses that confinement rather than claiming the
failed nested sandbox remained active.

V5 completed the full planner/actor/critic authoring path and reached the
centralized evaluator.
Receipt
`/mnt/raid0/llm/autokernel/probes/inf03-actor-critic-real-smoke-v5-20260811-2Oo2cJ/smoke-receipt.json`
carries receipt SHA-256
`dcb2da77bf691c670b705149bfaec0bf6e8062594cd4297ac3247037da5937fb`
and file SHA-256
`e890240fa4e3e5134c2975f77930bf5a611e3db285dea2f01a9560d5b699d0d3`.
The actor changed the candidate and the controller ended `critic_accept`, but
the evaluator worker inherited `/usr/bin/python3`, where pytest was absent.
Consequently its apparent correctness failure and zero timing cases are an
evaluator-runtime defect, not evidence against the candidate. The 457.02-second
sampling window captured 1,829 samples and the MI210 claim released cleanly.
V5 establishes authoring-path integration only; a fresh run with the exact ROCm
evaluator Python/package identity is required before any candidate or
performance conclusion.

Research `a57feba0` then pins the evaluator interpreter and refuses identity
mismatch. The bound runtime has binary SHA-256
`9544d2a29138833e6177d45dbc57468d37710b5080c901fbb579d53f251cdd6f`
with pytest `9.1.1`, torch `2.5.1+rocm6.2`, and Triton `3.1.0`. V6 completed
under that identity: compilation and correctness passed, all four baseline and
four optimized timing cases were valid, and the diagnostic average speedup was
`0.9936027407797491`. Receipt
`/mnt/raid0/llm/autokernel/probes/inf03-actor-critic-real-smoke-v6-20260811-3hAJir/smoke-receipt.json`
carries receipt SHA-256
`5961eef441e487e787310d3bea9d4e57693a8f7a621dff1cc39190d48d952ef9`
and file SHA-256
`816ecc2d60ee48e8b0be3c6fb05ff1d562d7283697f1def9499ce6d0a98a916c`.
The nested controller receipt carries SHA-256
`fa11ee8162fb6da877358a0c26c67d58d84c75b6c284fe9ee53540bb3673e315`.
The 158.72-second window retained 635 samples and released its MI210 claim.
Like the other smokes, this proves executable integration, not comparative
controller quality or a matched-campaign result.

The campaign pin was then refreshed to the validated actor and GEAK entrypoint
identities and re-audited from clean detached research `6233cd42`. The current
full-panel receipt is
`/mnt/raid0/llm/autokernel/probes/inf03-final-audits-v2-20260811-v6/full-eight-arm-refusal.json`
(receipt SHA-256
`4a13f7d0ba91c2610efae4e51bcf7e0be8661d07657f74fecf9a5ee0a4dab3af`;
file SHA-256
`199e4e129daf561f42d59750a1c2e157da340f433c0fec3abf19cf7c1bd91195`)
and still refuses at 6/8 solely for EvoEngineer and ARGUS. That historical
available-source receipt is
`/mnt/raid0/llm/autokernel/probes/inf03-final-audits-v2-20260811-v6/available-source-six-arm.json`
(receipt SHA-256
`cf8b03df355a8124a1dd668293c1d7d6e839c9f176aebdcd082f073f92ba0581`;
file SHA-256
`625280fb92b678b8a2a24f15f9a87484a2409c0dcb94e32a777e613639f327ed`)
and remains ready at 6/6 with availability-conditioned diagnostic authority.
Both bind config SHA-256
`6d72b61a3f1f8ebff12344e038bae16c57e8a2e7ae74cadd0f3ee63e4c8a6d7c`
and driver SHA-256
`b839a35cb79627bb27c7f1be6902e91365a1ce8beb0fc26edff58bae5d003866`;
neither audit executed a controller or GPU command.

## EvoEngineer source admission and executable integration — 2026-08-12

`evoengineer_arena.py` binds the historical source release rather than the later
generic EvoToolkit `master`. It declares `EvoEngineer-Full` explicitly—Free and
Insight are distinct upstream variants—and pins the paper-matched population 4,
10 generations, 45 samples, four samplers, and four evaluators. The adapter
retains upstream `EvoEngineer.run`, rank-probability parent selection, operators
`init(0)`, `crossover(2)`, `mutation(1)`, elite trimming, and random sampling of
up to three prior thoughts. Only the task, AMD prompt, model, and evaluator
boundaries are translated. The implementation now exposes a fixed
`campaign_argv`, a stdin CLI, the exact source/runtime identities, and the
paper-matched `EvoEngineer.run` loop. Every candidate crosses
`ArenaWorkspaceEvaluator.evaluate(files)` into the authenticated parent broker;
the controller has no direct evaluator/device authority.

The admission sequence is complete: the clean exact checkout is present, the
broker backs the evaluator protocol, the campaign runner launches the arm under
the controller sandbox, and the no-execution campaign audit declares it ready.
The latest seven-arm available-source audit is
`/mnt/raid0/llm/autokernel/probes/inf03-isolation-available-source-reaudit-20260812-r3/receipt.json`
(receipt SHA-256
`5fa93695f021b869010f880ba90d265ac2a157d7c3ac4c6e50c3f5c69f1fd46d`;
file SHA-256
`082d977fa99b6f83e2d2b32583757340729c6c9014cbadf920b08025e169c990`).
It is ready at 7/7 with authority
`availability_conditioned_diagnostic_only`; its embedded primary-panel audit
remains refused at 7/8 with receipt SHA-256
`3a43cd35044da9301a1d39996e365f3e6321f4ffa8a72dfb374df1e936a0c555`
because ARGUS alone is unavailable. The audit executed neither a controller nor
a GPU command. Consequently an EvoEngineer **result** still does not exist:
source and execution-path admission are complete, but controller quality can be
measured only by a fresh governed campaign.

The returned arena score remains `whole_agent_task_only`; C4 input remains
`diagnostic_only`. Neither is an AutoKernel promotion verdict.

## Claude controller runtime confinement repair — 2026-08-12

The first fresh 7/7 campaign attempt (`inf03-available-source-seven-arm-r5-20260812`)
completed its starting-state baseline and then stopped in the first
Claude+Codex actor-critic cell. The pinned Claude planner exited by `SIGSEGV`
before stdout or stderr. This partial campaign has no comparative, ranking,
belief-update, or promotion authority.

Differential reproduction proved that the controller seccomp profile was not
the cause: the same pinned request completed under seccomp alone and failed
under the campaign's exact Landlock read allowlist. File-open tracing found the
fixed Bun/Claude startup surface: self process maps/stat/cgroup, four bounded
kernel/CPU facts, UTC zoneinfo, and `/dev/urandom`. The controller sandbox now
admits only those declared literals for Claude-bearing arms. `/proc/self/*`
binds after the wrapper becomes the controller PID, volatile values are not
misrepresented as content identities, and `/dev/urandom` receives read—not
write—authority. Claude's two credential/settings inputs are copied into a
minimal workspace-local config directory and scrubbed after the controller;
host history, projects, plugins, cache, and the rest of the home directory stay
outside the read boundary.

The governed live probe is
`/mnt/raid0/llm/autokernel/probes/inf03-claude-sandbox-runtime-20260812-r1/`.
Its activation, result, and teardown file SHA-256 values are respectively
`c84232f3b8ca6514e5b221bc3ac3d2c4d07154ebf39cb468a2a196d9494b0795`,
`3322e81aac1f68a3490875cbfcdcaf81a3496e2fe041e87946e9de2e4a1bd384`, and
`e15ca322f2835ef581e06b3940ba3670671db50289674f5bf86cc3900d80df61`.
The real pinned `claude-opus-5/high` request returned structured output with
zero stderr; the exact cgroup was empty and removed, and ephemeral Claude state
was absent after exit. This proves runtime compatibility only. A fresh
campaign ID is still required because the r5 source manifest predates this
repair.

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

## Governed raw-HIP authoring arm — 2026-08-12

`hip_authoring_arm.py` closes the smallest real Torch2HIP loop without touching
a production kernel tree. It accepts only an exact `torch2hip/<suite>/<task>`
locator from clean AgentKernelArena commit
`2dbbf1d3f676b948c04c339de50516fe80ed4ab9`, hashes every task input and the HIP
candidate, binds the evaluator Python/Ninja/hipcc identities, and compiles for
`gfx90a` while GPU-blind. The vendor PyTorch baseline and final centralized
evaluation use separate short `mi210_0` claim/sampler windows; source authoring
and compilation hold no GPU claim.

```bash
/mnt/raid0/llm/tools/geak-v1-rocm62-py312/bin/python -m \
  scripts.kernel_rnd.autokernel.controller.hip_authoring_arm \
  --task-id torch2hip/gpumode/16636_SiLU \
  --candidate-source /path/to/candidate.hip \
  --output-root /new/campaign/root \
  --campaign-id unique-campaign-id
```

The completed MI210 diagnostic is
`/mnt/raid0/llm/autokernel/campaigns/hip-arm-silu-roundtrip-20260812-r4/receipt.json`
(file SHA-256
`e682ca027781acc03e4ce33ef8584ec9660721711aadd10670791ddfbbe5fc89`;
internal payload SHA-256
`1cb7087f715a2a9ac28b187a3f2d25c41be6a82279ca4fb254ac9b481805bc48`).
The candidate compiled and passed all 11 public correctness cases. The observed
Torch-eager ratio is explicitly non-rankable: public shapes are not sealed and
the evaluator does not bind an honest vendor baseline. The receipt may prove
round-trip compatibility and harness validity; it cannot rank, promote, or
deploy a HIP candidate.
