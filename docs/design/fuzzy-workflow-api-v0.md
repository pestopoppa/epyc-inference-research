# Fuzzy workflow API v0 — APP-owned design contract

MAIN reviewed and applied 2026-10-06. This is the APP-owned contract for FW-2, prepared privately from ROOT lane `1f7ce096dd6a715b758e84368f5034b8fb707837` and APP `origin/main` commit `48a546e90fbc202a0c2ae103203621ba29175915`. Proposed destination: `docs/design/fuzzy-workflow-api-v0.md`. Design only; no routes, page, registry rows, workflow storage, linter, or executor are implemented here.

## Scope and source ownership

APP owns the workflow document, validation schema/linter and `/dashboard/api/workflows/*` data contract. ROOT owns `/workflows` authoring page, its own nav/registry entry, and its own page-level freshness display. The page directly fetches APP; ROOT does not proxy APP data. HS-4 excludes an OpenCode plugin. HS-5b/P0.4 blocks future GUI-authored workflow execution over `/v1`, but not schema/API design, private draft authoring, or deterministic validation.

At APP `48a546e90`, `/dashboard/api/*` routes are in `src/api/routes/dashboard.py` and its existing CORS path policy is `src/api/dashboard_cors.py`; no workflow-authoring schema/route is present. FW-1 in ROOT `handoffs/active/fuzzy-workflow-authoring-gui.md` is the vocabulary source: six node types, six edge types, and lint rules L1–L11. This v0 reuses that set without making function bodies, arbitrary code, endpoints, or policy editable.

## Closed document schema `epyc.orchestrator.fuzzy_workflow.v0`

Canonical digest: RFC 8785 canonical JSON UTF-8 over exactly the semantic projection `{schema,title,nodes,edges,budgets}`. Exclude server-owned workflow ID, revision, state, timestamps and digest, and exclude the separate presentation map. Arrays preserve their declared order. Every object is closed (`additionalProperties: false`). The digest binds semantic content only; adding/reordering canvas coordinates does not change it. Position is stored separately as presentation metadata, omitted from the semantic digest and executor input.

Top-level fields exactly:

- `schema` (literal `epyc.orchestrator.fuzzy_workflow.v0`)
- `workflow_id` (opaque server-assigned ID on create; client omits it)
- `revision` (positive server-assigned integer; client omits it on create/update body)
- `title` (non-empty string)
- `state` (`draft|validated|disabled`; state is server-managed, caller cannot submit it)
- `nodes` (non-empty list of the closed variants below)
- `edges` (list of the closed variants below)
- `budgets` (list of `{id,name,limit,unit}` where `id` is stable within document, `name` non-empty, `limit` positive integer, `unit` is `attempt|call|token|millisecond`; every budget has a unique ID and unique name)
- `created_at`, `updated_at` (server-generated UTC RFC3339 strings)
- `content_sha256` (lowercase 64-hex digest over canonical semantic content as defined above)

Each node object has common fields `id` (unique non-empty stable ID), `type` (one of exact variants), `reads` (list of declared symbolic field names), `config` (closed per-type object). Edges refer to node IDs. No arbitrary Python expressions, shell commands, filesystem paths, URLs, dynamic imports, prompt execution, or unregistered function names are allowed in any config.

**Closed config variants** (fields omitted here are forbidden):

| Node `type` | Required `config` fields | Additional rules |
|---|---|---|
| `code` | `function_ref`, `output_fields` | `function_ref` is a symbolic key from an APP-owned allowlist established by a later implementation; FW-2 introduces no functions. No function body/source text. |
| `fuzzy` | `question_catalogue_id`, `question_catalogue_sha256`, `model_role_or_pin`, `parse_budget_id`, `fallback_target`, `shadow` | The catalogue itself is referenced, not embedded prose. `model_role_or_pin` is a logical role or immutable pin, never URL. `fallback_target` names an in-document node. `shadow` boolean means record-only behavior per FW-1; it grants no value edge. |
| `gate` | `predicate_ref`, `rejection_grounds`, `budget_id`, `pass_target`, `reject_target` | `predicate_ref` symbolic/allowlisted; rejection grounds non-empty; budget unique/independent; both targets resolve. |
| `expensive` | `role`, `cost_class`, `output_fields` | No endpoint; the serving router remains owner. Every path from a graph root to this node must cross a declared gate in v0. Future explicit exceptions require a separately reviewed semantic schema field; presentation metadata cannot grant permission. |
| `terminal` | `outcome`, `record_fields` | `outcome` is `success|failure`; failure terminal must preserve declared reason/gate outcomes. |
| `subflow` | `workflow_ref`, `workflow_revision` | Immutable reference to another saved workflow revision; cycles rejected by validation. |

Edge objects have exactly `id,source,target,type,condition_ref,reason_fields,value_ref`. Fields are nullable only when edge type makes them inapplicable. `type` is `flow|pass|reject|value|fallback|loopback`; `condition_ref` is a symbolic deterministic predicate, never a free expression; `reason_fields` are declared fields visible to the destination; `value_ref` is a candidate ID from the source fuzzy node catalogue. L1-L11 remain the governing lints: no dangling edges; no predicate reads confidence/probabilities/token_logprob; fuzzy fallback required and resolves; fuzzy value coverage total unless shadow; no unknown/other pseudo-option (use fallback); budget IDs/names independent; expensive-node path gate; nonempty gate rejection grounds; nonempty actor reads; loopback rejection reason is readable by destination; no model endpoint node.

Graph roots are nodes with no incoming non-loopback edge; at least one root is required. Reachability and expensive-path checks start at every root, ignoring loopback only for root discovery. Each gate/parse budget is an independent counter; validation rejects reuse of a counter by different gates or parse operations.

Canvas `position` is a separate presentation map keyed by node id and excluded from the workflow object/digest and evaluator. It is not accepted in v0 semantic POST/PUT bodies.

## Endpoint contract

All methods use JSON. The implementing owner must verify operator authentication, same-origin/CSRF protection, CORS and authorization on every mutation route; this design does not assert that existing routes already meet those requirements. Register static `/health` before `/{workflow_id}`; validation is the item route `/{workflow_id}/validate`, not a collection `/validate` route.

| Method/path | Request/response summary | Effect |
|---|---|---|
| `GET /dashboard/api/workflows` | Metadata list only: `workflow_id,revision,title,state,updated_at,content_sha256`; never emits node/catalogue content in a list response. | Read only |
| `POST /dashboard/api/workflows` | Create draft from client semantic fields (no server-owned fields); response is server-owned ID/revision/timestamps/digest. | Persist draft |
| `GET /dashboard/api/workflows/health` | Request-time APP producer probe: `schema,generated_at,status,api_version,storage_state,absence_means`; `status=ok|absent|degraded`, `storage_state=available|empty|unavailable|unreadable`. | Read only |
| `GET /dashboard/api/workflows/{workflow_id}` | Optional `revision=<positive integer>` selects an immutable saved revision; omission returns latest. Missing revision returns not-found. Return exact selected revision and digest. | Read only |
| `PUT /dashboard/api/workflows/{workflow_id}` | Replace draft using `expected_revision` precondition; stale revision returns conflict, never last-write-wins. Create a new immutable revision and recalculate timestamps/digest; preserve prior revisions for exact subflow references. | Persist draft |
| `POST /dashboard/api/workflows/{workflow_id}/validate` | Deterministic schema+L1–L11 lint only: `{valid,errors,warnings,workflow_id,revision,content_sha256}`; warnings carry typed diagnostic reasons and cannot waive v0 gate/schema errors. No model call and no execution. | No persistent state change except optional validation receipt, which is outside v0 and therefore omitted. |
| `POST /dashboard/api/workflows/{workflow_id}/execute` | Reserved path. In v0 it is not registered; caller receives route-not-found. Future activation requires separate source task, runtime/security review, HS-5b/P0.4 freeze and explicit runtime authorization. | Not implemented |

No delete endpoint in v0. `disabled` status is owner-controlled in a future route task so referenced revisions remain available. Workflow data access is single-user/operator-authenticated in the implementation, with same-origin/CSRF protections and allowlisted methods; this contract does not itself expand access.

## Health and freshness contract

APP health response is created on request and uses producer time `generated_at` (UTC RFC3339). `storage_state=empty` means the producer is healthy and has no saved workflows; `unavailable` means no storage is configured; `unreadable` means configured storage could not be read. `absence_means` is mandatory and explicit: “workflow inventory is unknown because its producer or store could not be read; absence does not mean there are zero workflows.” The APP probe is the page's data probe, not a statement about the ROOT hub transport.

ROOT registry row draft is in `artifacts/ni08/workflow-contract-design-20261006/workflow-dashboard-row-draft.json`. It is a review artifact, not an installed registry entry. The row's `/health` is ROOT transport-only per current convention. The page itself directly fetches APP `/dashboard/api/workflows/health` and the list route. It displays the returned producer timestamp and computes page freshness from that timestamp; proposed polling is 30 seconds with `stale_after_s=90`, after which retained content is visibly stale and a failed refresh cannot render as empty. These are proposed page behavior values, not APP health/SLO promises. A successful empty list remains a distinct “no saved workflows” state. No claim enters ROOT's server-side global dashboard health fold, and no hub-side proxy/reader is implied.

## Acceptance for FW-2 design/application

1. MAIN applies this APP-owned contract and the ROOT registry draft as the shared design boundary. Implementation remains a separate task.
2. APP design file records schema/hash rules, closed config fields, endpoint ordering and `execute` omission.
3. ROOT row+page draft names the APP data probe separately from ROOT `/health` and displays `generated_at` freshness without confusing stale/unreachable with an empty workflow store.
4. No shell plugin, model endpoint, arbitrary code, model call, runtime executor, serving change, or P0.4 bypass is part of FW-2.
5. FW-4 is already filed prospectively as `VB-FW-1`; do not duplicate a Vidya source row/task. Its write-side refuse-without-hook check belongs to the future executor implementation, not this design contract.

RFC 8785 defines the proposed canonicalization: [JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785). No hashing implementation or live endpoint is added by this contract.
