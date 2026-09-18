# AutoKernel standalone management CLI and packaging

The standalone client operates the existing `CampaignHTTPService`; it is not a
second daemon, controller, transport, configuration source, or WAL. Its authenticated
commands use the supervisor-fenced closed v2 command schema and the existing
worker-lifecycle v2 result validator; legacy v1 commands remain explicitly supported
without inventing the new fence. Snapshot dispatch is closed by schema: v2 uses
`validate_snapshot_v2`, v3 uses the installed `validate_snapshot_v3`, and every other
schema refuses. This integrated checkout includes the published closed v3 producer
and validator; the client exercises that exact validator rather than widening v2.

## Commands

Set `AUTOKERNEL_CONTROL_TOKEN` in a protected process environment, or use
`--token-file` pointing to a regular owner-owned mode-0600 one-link file. Token values
are never CLI arguments. The service origin must be exact HTTP numeric IPv4 loopback.

```text
python3 -m scripts.kernel_rnd.autokernel.loop.standalone_cli inspect --endpoint http://127.0.0.1:8077
python3 -m scripts.kernel_rnd.autokernel.loop.standalone_cli preflight --endpoint http://127.0.0.1:8077
python3 -m scripts.kernel_rnd.autokernel.loop.standalone_cli status --endpoint http://127.0.0.1:8077 --expect-campaign-id CAMPAIGN --expect-config-generation 1 --expect-config-digest SHA256 --expect-supervisor-incarnation 4
python3 -m scripts.kernel_rnd.autokernel.loop.standalone_cli pause --endpoint http://127.0.0.1:8077 --expect-campaign-id CAMPAIGN --expect-config-generation 1 --expect-config-digest SHA256 --expect-supervisor-incarnation 4 --request-id pause-0001 --expected-revision 7
```

`resume` and `drain` use the same control arguments. After an uncertain ACK, repeat
the exact request ID, expected revision, operation, campaign, and generation. The
payload digest is deterministic, and bounded `--retries` (maximum two) reuses the
same serialized command. Never retry with a new ID. Output distinguishes `requested`
(accepted but not completed), `applied`, and `refused`.

For a control retry only, authenticated health and snapshot may report a supervisor
incarnation newer than the command's original pin. The client preserves the exact
historical command bytes and the server returns it only when that command is already
durable; a fresh command carrying the old incarnation is refused. A current endpoint
older than the command pin also refuses. `status` retains exact current-incarnation
pinning and health/snapshot coherence is always exact.

The v2 command body and its digest bind campaign ID, configuration generation and
digest, supervisor incarnation, expected control revision, request ID, operation, and
payload. `CampaignController.apply_command` validates a fresh request under its mutex.
An exact journaled retry can return its prior result after restart without reapplying;
a fresh old-incarnation command refuses. Existing v1 commands retain their historical
campaign/generation/revision behavior and do not acquire a retroactive supervisor fence.

`inspect` reads transport health only. `preflight` is also unauthenticated and
non-mutating: it does not read a token, create a store, acquire a claim, or infer
semantic liveness. It can read a `campaign_cli` resolution/production-enrollment
envelope via `--resolved-campaign`; parsing is delegated to the service's actual
`load_resolved` path and still performs no creation. `status` reads an authenticated typed snapshot and checks its
campaign/config/supervisor identity against transport health and all caller pins.
Transport health alone never means the campaign is scientifically live.

Each attempt has one absolute deadline spanning connection, headers, and the complete
body (default two seconds, maximum 30), at most two exact retries, and a 128-KiB
response limit. Redirects are structurally absent because the
client uses a direct loopback HTTP connection. Wildcard/public binds, hostnames,
credentials in URLs, paths, queries, TLS ambiguity, browser launching, and shell
interpolation are refused. Remote refusal bodies are never rendered because they are
untrusted and may echo an Authorization value. Environment and protected-file tokens
share one 4096-byte visible-ASCII bound; token FIFOs and other non-regular objects are
opened nonblocking and refused before any read.

## Packaging status

`deploy/autokernel/autokernel-campaign.service.in` is source-only and intentionally
not ready to install or enable. It has no `[Install]` section and depends on a marker
that is not shipped. It reports the pinned checkout/interpreter/config/store identity
in its literal command while keeping the owner-only token in an environment file.

`campaign_service.main` now converts SIGTERM through a notification-only self-pipe into
an idempotent durable drain and keeps authenticated service state available while an
ownership wait is unresolved. It closes the service/controller only after the typed
drain reports quiescence. One absolute shutdown deadline bounds the initial wait;
expiry is reported while authenticated management remains available and ownership is
retained. Elapsed manager time does not release claims.
If an operator drain is already the current durable control state, shutdown reuses that
exact completed or pending receipt and waits for current quiescence instead of appending
a second command. Repeated signals and a restarted already-drained manager do the same.
Management-only snapshot v1 uses its existing journaled v1 control path; it never enters
the worker-lifecycle v2 transition path or fabricates a lifecycle receipt.
The eventual contained-child bridge must separately place worker processes in the
provider/broker-owned cgroup; the service-manager process must not become resource
authority. Until those seams and deployment review exist, the default provider and
runtime launch bridge remain unavailable and this package makes no production-ready,
live-cutover, or serving claim.
