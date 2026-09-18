# Selected actor advice to private source/build preparation

The public `UnifiedCampaignDriver.materialize_actor(outcome)` returns immutable
`SelectedActorWork`: exact catalog, transition, scheduler selection, actor request,
preparation cache key and current controller binding. Like selected-profile advice,
it has `execution_authorized=False`; it is not a grant, mutation lease or receipt.

`loop/source_build_preparation.py` connects that advice to existing preparation
consumers without a launcher, WAL, grader, scheduler settlement or champion update.

## Source input and actual materialization

`bind_source_preparation` takes the selected actor record, the actual consumer's
`PreparationResult`, resolved campaign, an explicit native `AuthoringAssignment`,
a real authored `SourcePatchManifest`, and its native legacy proposal. It validates
the request/target/config binding, accepted output, mechanism, exact single-file
surface/symbol, pinned instrument revision, assignment and native diff-size limit.
The actual changed-line count is checked before mutation. Portfolio-bound discovery
assignments require the existing full discovery-plan binder and are refused here;
their additional dispatch/regime policy is not silently discarded.
Current source advice names one file and symbol, so this adapter intentionally
refuses a multi-file manifest; it does not infer expanded scope from prose.

The unified `prepare:...` proposal ID remains unchanged. The record separately
binds it to the existing source format's owner-assigned `akp-`/`akc-` IDs. Actual
patch bytes, declaration scope, production/instrument commits and change class must
be supplied, never synthesized from `implementation_plan`. Native
`SourcePatchManifest` validation and `source_patch_manifest_bytes` own the policy
and canonical carrier; this adapter stores canonical bytes rather than mutable
native declaration dictionaries or a patch pathname.

`materialize_source(bound, campaign_driver=..., actor_worktree=...)` reopens the
actual selected request through public `materialize_actor`, rechecks current
controller binding and source ancestry, then delegates unchanged to
`source_candidate.apply_source_candidate` using the caller's existing guarded
private `Worktree`. The core proves the exact clean base, applies immutable bytes,
commits only declared paths, and re-derives policy/symbol evidence. The returned
deeply immutable record carries the actual commit/tree, patch bundle and diff
digests, derived file/symbol/hunk identities, native policy checks and mutation
receipt. Its build status is **pending**, never compiled or scientifically verified.

A serialized advice/result/assignment is not authority. The caller must already own
and serialize the selected preparation operation, supply its own private Worktree,
and use its existing durable intent/recovery mechanism. The current-controller
check is a preflight, **not an atomic admission lease**. This adapter does not make
check-then-I/O safe against an owner that concurrently starts another operation,
nor reconstruct execution authority after restart. No automatic standalone CLI
wires this mutation path into a live campaign.

## Explicit build-only input and delegation

`bind_build_preparation` accepts the selected build advice, explicit existing
`BuildPlan`, and an actual clean detached snapshot. It proves the source root,
commit and tree digest, preserves source identity, copies the plan/parallelism,
and requires actor options to exactly match the plan's sorted effective CMake
defines. There is no shell splitting or implicit flag adoption. The caller must
resolve any GPU production/divergence policy through its existing recipe authority
before handing in a plan; this generic adapter does not pretend a free-form plan
is an approved production recipe.

The plan must have explicit CPU affinity within the resolved campaign CPU set and
jobs no greater than the campaign build limit. Delegation requires finite positive
configure/build deadlines whose sum fits the campaign build timeout, explicit
environment, absolute owner log path and owned cgroup root. Source identity is
revalidated immediately before delegation.

`delegate_build` has **no default runner**. A caller supplies its already-owned
runner using the existing `worktree.run_build` signature. The exact plan, copied
environment, deadlines, fresh-build requirement and cgroup root are forwarded.
The original result or exception is returned unchanged: no sentinel/exit code/
artifact expectation is converted into `BuildResult`, `BuildIdentity`, verified
dispatch or a successful experiment. Actual identification remains
`worktree.build_identity` (and GPU `verify_build_authority`) after genuine native
execution. The caller owns grants, execution serialization, durable build intent,
spent-cost settlement, stage-count limits and recovery.

The deployed `StaticGpuSourceBuilder` remains a GPU **source** builder. This module
does not route CPU/build-only work through it or manufacture an empty patch to make
its interface fit.

## Validation scope

The focused tests use the real UnifiedCampaignDriver and CampaignController to
select/materialize source and build advice. A disposable tiny C++ fixture exercises
the actual guarded source application and exact-path commit; no production kernel
is touched. Build tests construct actual BuildPlans and substitute a recording
boundary at `worktree.run_build`, asserting exact forwarded native arguments.
No CMake, kernel execution, model inference or provider grant is performed.

Advice-only tests deliberately construct preparation result *data*: they prove
binding and native source consumption, not a completed actor/provider campaign.
They do not seed private controller receipt/projection state. The already-existing
strict selected-profile scheduler/cost binding xfail (OP-AKU-BIND / AKU-07k) remains
unchanged. This adapter is not full AKU-06d or full campaign acceptance.

## Concrete next contained authoring and planner-feedback route

The remaining source authoring stage should be a separate contained worker, not a
call to the old direct `AgentPlanner.author`:

1. From current selected advice plus an explicit AuthoringAssignment, construct a
   content-addressed authoring request binding the parent request/advice digest,
   exact production/instrument revision, allowed file:symbol scope, maximum diff
   size, and bounded read-only source context (bytes plus SHA). Distinct authoring
   request/attempt identity prevents reusing the already-finished advice reservation.
2. A new bounded source-authoring consumer uses the existing public
   ActorStageCapability reserve/invoke/finish and ActorLifecycleAdapter. It asks for
   a closed output containing patch bytes and declarations only. Authority fields
   (campaign/proposal/candidate IDs, commits, scope ceilings) come from the owner,
   not actor JSON. The worker must not build, mutate the parent's source worktree,
   allocate resources or issue an ExperimentPlan.
3. Decode only controller-authenticated bounded stdout joined to the exact native
   terminal and held-cost receipt. Apply strict output-size/UTF-8/base64/closed-field
   checks, construct the existing SourcePatchManifest, and bind it with the existing
   native validator. Invalid output settles genuine spent cost without emitting an
   authored-artifact result; bounded retry/repair uses a new attempt and existing
   independent budgets, not reinvocation of a settled reservation.
4. Feed that genuine manifest directly to this adapter. The owning transaction
   retains content-addressed advice → authored patch → source materialization →
   actual build identity references through the existing Journal/object mechanism.
   This requires a separately reviewed native-owner event/accessor integration;
   the present adapter neither adds a second WAL nor treats serialized references
   as recovery authority.
5. The planner owner reads only its producer-verified preparation projection keyed
   by exact campaign/target/parent request/advice/source/recipe identities. It marks
   the parent prerequisite consumed, schedules a separately bounded build when
   needed, and offers completed source/build artifacts to the existing immutable
   experiment-plan constructor. It does not upgrade the old pending proposal by
   changing a digest in place. Source/build drift invalidates the preparation;
   failed/unknown build or missing required evidence stays an explicit prerequisite.

Proposed next ownership is new `source_authoring.py`/tests plus main-reviewed
catalog/feedback wiring in unified_driver/unified_planner and native reservation/
artifact-projection changes by the native owner. Before those existing methods are
edited, query exact impacts and release each scope. Fixing OP-AKU-BIND is a separate
prerequisite for the real public actor/profile cost-settlement chain; a permissive
validator or fabricated held receipt is not an acceptance route.
