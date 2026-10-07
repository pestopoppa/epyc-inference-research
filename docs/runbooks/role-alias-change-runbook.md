# Runbook — Changing Role↔Server Aliases

**Created 2026-07-22** from the WP-13 boundary deploy (the coder_escalation/worker_summarize
reconciliation). Every pitfall listed below was hit live that day. Supersession note: when the
**WP-12 fleet layer** lands (`handoffs/active/wp12-fleet-layer-design.md` in epyc-root), roles
reference fleets structurally and several steps here (the models.py delegation, the WP-13
generator inheritance) are deleted — re-read that design before using this runbook afterward.

## Concepts (read first)

- A **host role** owns a physical llama-server fleet (e.g. `frontdoor`, `worker_general`).
- An **alias role** has NO process of its own — it rides its host's server(s). Same GGUF, one
  server (operator constitution: same-model roles share ONE server; roles are a remappable
  logical layer over servers).
- **The single declarative truth**: the host's `shared_with` list in `server_mode` of the
  canonical research master at `epyc-inference-research/orchestration/model_registry.yaml`.
  The orchestrator's `orchestration/model_registry.yaml` is a generated lean runtime view,
  rebuilt from that master at stack start; never edit the generated view.
  An alias must NEVER have its own `server_mode` row
  (the dead-8070 coder_escalation row was exactly that defect).
- Aliases keep their role-layer identity elsewhere: `roles:` section entry (model metadata),
  per-role timeouts (`TimeoutsConfig`), prompts/templates, sampling. Those stay per-role.

## The five layers an alias touches (until WP-12)

| Layer | File | What to change |
|---|---|---|
| 1. Registry (SoT) | `epyc-inference-research/orchestration/model_registry.yaml` | host row `server_mode.<host>.shared_with` list; the orchestrator copy is generated lean output |
| 2. Launch tagging | `scripts/server/stack_manifest.py` `ROLE_LAUNCH_META` | host's `shared_with_first_n` (+ `shared_with_first_n_count` if aliases should tag onto >1 instance) |
| 3. Operative URL default | `src/config/models.py` `ServerURLsConfig` | alias field delegates: `_server_url_default("<host>")` |
| 4. Generated priors | `orchestration/derived/stack_priors.yaml` | NEVER by hand — regenerate via the pipeline (WP-13 inheritance gives aliases the host's full port fleet) |
| 5. Fallback map | `src/roles.py` `_FALLBACK_MAP` | same-fleet fallback edges are physically no-ops — remove/avoid them (they produced 90x churn) |

## Procedure A — add an alias to a host

1. Registry: append the alias to the host's `shared_with`. Confirm the alias has a `roles:`
   entry (model metadata may be a ghost binding — that's legal and is what remappability means).
   Confirm the alias has **no** `server_mode` row of its own; delete one if present.
2. `stack_manifest.py`: add the alias to the host's `shared_with_first_n` if the launch layer
   should tag it (routing fan-out normally wants this).
3. `models.py`: point the alias's `ServerURLsConfig` field at `_server_url_default("<host>")`.
4. Role-layer config: timeouts, prompt, sampling for the alias as needed (unchanged mechanics).
5. Run the **verification chain** (below). 6. Reload (below). 7. Update
   `epyc-root/handoffs/active/within-role-placement-state-machine.md` if fleet semantics moved.

## Procedure B — remove an alias / Procedure C — re-point to a different host

Same steps with the list membership / delegation target changed. Re-pointing is the designed
capability (worker_math's Qwen2.5-Math→gemma4 ghost is the canonical precedent). If the alias
should become its OWN server (reverse of B): it gains a `server_mode` row + `NUMA_CONFIG`
entry + launch meta — **that changes cpusets and triggers the §H contention-matrix recert
cascade** (see `wp9-wp10-lineup-event-prep.md`); treat it as a lineup event, not an alias edit.

## Verification chain (mandatory, in order — each step exists because it caught a real defect)

1. **Snapshot the operative URLs BEFORE**:
   `.venv/bin/python -c "from src.config import get_config; c=get_config(); [print(r, getattr(c.server_urls,r)) for r in ('frontdoor','worker_general','worker_math','coder_escalation','worker_summarize','ingest_long_context')]"`
2. **Pipeline update**: `.venv/bin/python scripts/registry/stack_change_pipeline.py update`
   - Compiles under the **realized fleet mode** (ESC-8 Fix 6). A clean shell cannot poison it
     with default-full. If it REFUSES (`StackPriorsModeError`): nothing is listening — pass an
     explicit mode only if you are deliberately compiling for a down fleet.
   - Guard must be green. Alias semantics the guard enforces: alias serving.ports/endpoint ==
     the HOST's launch row; alias launch entries port-contained in host's; runtime validated on
     the host row only (an alias's stored runtime is historical residue — it RUNS under the
     host's runtime).
3. **Snapshot AFTER and byte-compare with BEFORE.** An alias membership change should alter
   ONLY the intended alias's URL (to its new host's fleet). Anything else changing = stop.
4. **Targeted tests**: `pytest tests/unit/test_stack_priors_compiler.py tests/unit/test_stack_numa_reader_agreement.py tests/unit/test_stack_change_guard.py tests/unit/test_stack_change_pipeline.py -q`
5. **Reload** — `.venv/bin/python scripts/server/orchestrator_stack.py reload orchestrator`
   - ⚠️ **If ANY eval/measurement is running: SIGSTOP the eval runner first, reload, SIGCONT.**
     The eval client has no reconnect backoff; a naked reload burns every in-flight and queued
     question (532-question incident, 2026-07-22; ~680-question incident before that).
   - API-only reload; do NOT stop/relaunch llama servers for an alias change (no process moves).
6. **Post-reload verification**:
   - `/proc/<api-pid>/environ`: `ORCHESTRATOR_STACK_NUMA_MODE` matches the realized fleet;
     placement flags present.
   - Runtime-facts manifest (`/mnt/raid0/llm/tmp/orchestrator_runtime_facts.json`): realized
     mode + non-empty selected_servers.
   - `curl localhost:8000/health`: probe groups green, alias grouped with its host.
   - Plain-import URL snapshot one more time (this is what plain-import consumers get).
   - **Live spread proof** if the alias carries real traffic: per-quarter llama log task counts
     (`grep -c "launch_slot" logs/llama-server-<port>.log` tails), NOT instantaneous busy-slot
     sampling — short generations make instant busy counts near-zero even at full 4-wide spread.

## Pitfalls (each was a live defect on 2026-07-22)

- `evidence.alias_overrides` exists ONLY for model-conflict aliases (ghost bindings). Same-model
  aliases are declared ONLY by `shared_with`. Don't build tooling that assumes the evidence form.
- Descriptors are model-keyed: the HOST's own record carries a copy of alias evidence. Tooling
  that treats "has alias_overrides" as "is an alias" silently exempts the HOST from validation.
- The guard, dashboard, eval fan-out, and priors compile each read the NUMA mode — all four now
  resolve realized-first, but any NEW reader must use `scripts/server/realized_fleet.py`, never
  `env_stack_numa_mode()` bare (default full).
- Never hand-edit `stack_priors.yaml`, the runtime-facts manifest, or historical records — regenerate/append.
- Same-fleet fallback edges in `_FALLBACK_MAP` are retry-the-same-metal no-ops that masquerade
  as failover. Don't add them for a new alias.

## Cross-references

- `epyc-root/handoffs/active/wp12-fleet-layer-design.md` — the structural replacement.
- `epyc-root/handoffs/active/esc8-stack-restart-landmine-audit-2026-07-22.md` — why realized-mode.
- `epyc-root/handoffs/active/within-role-placement-state-machine.md` — WP-8..WP-14 context.
- Commits: 22e32ec2 (WP-13 generator), d8b7e0bd (reconciliation + guard), 5aa29f35 (Fix 5/6).
