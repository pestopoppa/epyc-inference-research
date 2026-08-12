# IQK matched-pair preparation manifest

`epyc.autokernel.iqk_matched_pair_preparation.v1` is a campaign-input
contract, not a record of a completed measurement.  Both arms must carry the
question that authorizes the arm and the immutable store from which that
question is resolved.  A manifest that omits either binding can produce
capture plans which look complete but cannot be replayed through the
hypothesis gate.

The `intervention` and `control` objects therefore require these additional
fields:

| field | contract |
|---|---|
| `hypothesis_id` | non-empty `akh-…` identifier; it must resolve to exactly one entry in the store and its statement must equal the arm proposal's `hypothesis` | 
| `hypothesis_store` | absolute path to an existing, non-symlink JSON operator-hypothesis store |

These fields are required independently for both arms.  The A/A control has a
different generated proposal and must not inherit the intervention's
authorization.  The preparation boundary should reject missing, relative,
symlinked, or unreadable stores before creating either output directory.  The
durable result should retain the IDs and store content digests (the latter are
the source identity), so a later campaign invocation can prove which source
authorized each arm.

The store path is an input locator only; preparation must not mutate the store
or its ledger.  Resolution and falsifier checks remain the responsibility of
the campaign/evidence-path gate, while this manifest schema prevents an
unbound campaign input from being emitted in the first place.
