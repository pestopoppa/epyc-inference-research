# Actor wire admission (OAB-29)

Live loop launches write `<store>/actor-wire-request.json` after resolving their
planner, author panel, critic and lane overrides. A `--dry-run` changes nothing in
the store: it emits the same request as one complete JSON line on stdout, with
`schema: epyc.autokernel.actor_wire.v1`. Save that JSON line to an explicitly chosen
request file outside the store before running the command below. Before making an actor call,
`run.py` requires a matching passing record or `--allow-untested-actor-seats REASON`.
The override and exact reason enter the initial `loop-run.json` before the calls;
the final manifest preserves the same receipt.

Run the requested configuration against installed native clients and a fake provider:

```bash
PYTHONPATH=scripts/kernel_rnd python -m autokernel.loop.actor_wire_test \
  --request /path/to/actor-wire-request.json \
  --record /path/to/actor-wire-pass.json
```

Retry the original launch with `--actor-wire-test-record /path/to/actor-wire-pass.json`.
The record binds the effective ActorSeat fields, backend/model/effort, timeout,
installed executable digest, role, launch actor configuration and suite/adapter source
hashes. Changing one requires a new record. An absent client, unsupported backend,
failed check or stale record is refused. Orchestrator actors require the explicit
override until their own native adapter is added to this suite.

`--matrix --record PATH` runs CI's backend/role/config matrix. CI uses installed
Codex, Claude and OpenCode clients, a private HOME/CODEX_HOME/XDG tree, an environment
allowlist with dummy keys, and loopback Responses/Anthropic Messages/OpenAI Chat
Completions fixtures. It reads no host authentication state. The serving-F1 config
reader is explicitly rebased to the fixture, because its production default path is
bound at Python import time. No GPU or model server is involved.

Each effective seat proves that an oversized stdin prompt reaches the provider and
a >128 KiB Unicode final reply survives byte-exact in the actor capture, including
its tail. OpenCode's requested max_tokens and chat_template_kwargs are asserted on
the actual provider body. Codex's native reasoning effort is asserted on Responses;
Codex/Claude do not expose OpenCode's ActorSeat template/output knobs. Read-only
seats request a real denied tool, receive its error on the wire and still emit final
text; the scratch sentinel stays unchanged. A deliberately slow provider is ended
by the actor timeout. Separate process controls prove rc=1 salvage, signal deaths
as transients, and compaction/template echoes refused as answers. These controls
cannot establish a native provider pass on their own.

On this shared host every CLI/test execution still requires a canonical compute
grant. Preparing or reading a request does not waive that scheduling boundary.

OpenCode auxiliary title requests are recorded and answered separately; only a
positively identified title request is exempt from actor permission-tool selection.
Actor continuations must carry a nonempty result for the emitted call ID. OpenCode
read-only seats prove both external-directory ask-class refusal and an in-lane
`touch` refusal from the existing bash policy, with sentinel bytes and mtime unchanged.
Codex proves its native read-only sandbox with an in-lane patch attempt; these
checks require real native error results and exact final continuation. Synthetic
unit results alone never prove native permission behavior.

Claude's read-only probe uses its advertised Write tool against a new absent
in-lane path; that path must remain absent and the existing sentinel's bytes and
mtime must remain unchanged. Classifier outages and safety-monitor requests cannot
establish a native permission refusal. The fixture never supplies a severity verdict
to manufacture a refusal; it reports this auxiliary protocol as infrastructure failure.

Claude's exact native `Cannot write to <emitted path> while in plan mode.` result
counts only when bound to that emitted path and matching tool-call ID. The matrix
visits all seats after expected assertion failures with verified owned-child cleanup,
then exits failed without writing an admission PASS if any seat failed. A separate
matrix-results.json keeps every seat outcome. STOP, deadline, interrupt, unexpected
infrastructure or uncertain child cleanup aborts remaining seats explicitly; no
unsafe tree deletion or implicit success follows uncertainty.
