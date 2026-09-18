# Native retained-artifact catalog

The standalone-v3 native path installs one closed
`RETENTION_CATALOG_INSTALLED` record before runtime composition. The controller
rederives the seed from the resolved campaign, prepared runtime recipes, exact
`ScheduledModelPreparation` records, and the configured native artifact root;
serialized seed bytes are not an installation capability. Older v1/v2 controller
readers refuse a store containing this event.

The seed records permanent production, rollback, recipe, executable, DSO,
build-directory, RUNPATH, model-inventory, and native-artifact-store ownership.
Physical files shared by multiple targets have one identity with all target edges.
Model preparation records remain coverage debt until their bounded manifest is
reopened. Reopening expands every declared shard and checks that the enrolled model
entry and digest are members of that complete inventory. It never reads model bytes.

Candidate collection is two phase. The controller snapshots the durable seed and
candidate/worker/native/driver frontier while holding its mutex. Candidate objects,
small model manifests, and directory identities are checked outside the mutex. The
controller then requires the exact unchanged frontier before it binds a retention
job. The bound job includes the snapshot, policy digest, and selected bounded prefix;
another job with the same plan digest is not interchangeable.

This integration creates no expiry classification, provider hold, or deletion
authority. Permanent and active dependencies do not expire. Missing source,
inventory, artifact-root, or worker-to-issued-work joins remain explicit uncertainty,
which makes the existing retention planner refuse expiry. Dry-run reports both the
instrument and catalog as planned/unpublished and performs no model or artifact I/O.
