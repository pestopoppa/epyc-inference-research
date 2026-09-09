# AutoKernel source-only service template

This directory is deliberately not an installable service package. The unit has no
`[Install]` section and its condition names an approval marker that is not shipped.
Do not create that marker until the core service has one reviewed SIGTERM-to-durable-
drain bridge and the contained workload launcher places workers in broker-owned,
separate cgroups.

Substitute every `@...@` field with a literal, reviewed value. `@PINNED_CHECKOUT@`
and `@PYTHON@` must identify immutable deployed source and interpreter locations;
the campaign, store, generation, snapshot version, port, user, group, and secret environment path are
not shell-expanded. Install the environment file mode 0600, owned by the service
account. The token must never be placed in the unit, URL, argv, logs, or status.

The snapshot-version placeholder must name a version for which that exact checkout
ships a closed validator used by both producer and standalone client. This delivery
integration includes the published closed v2 and v3 validators. Its client dispatches
the unified v3 producer to `validate_snapshot_v3`; it never parses v3 through a
permissive v2 field union.

The current template starts the existing management-only `campaign_service`; it
does not establish worker launch integration, resource/provider authority,
scientific liveness, deployment readiness, or production acceptance. `KillMode`
does not authorize killing eventual broker-owned workload cgroups, and
`SendSIGKILL=no` prevents a manager timeout from being presented as clean teardown.
Legacy v1 controls have no atomic supervisor-incarnation field. The standalone client
uses the closed v2 command envelope, whose config digest and supervisor incarnation are
validated under the controller mutex; deployment must not attribute that fence to v1.
