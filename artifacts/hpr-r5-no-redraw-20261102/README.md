# HPR R5 no-redraw control

The original native cohort has five completed cells and one partially exposed
coarse wave 2. Request 116's response and saved server state are absent; it is
UNKNOWN, not a failed or completed observation. A fresh server cannot resume
that original request, and replaying the whole cell repeats already exposed
requests. The six-cell original decision remains incomplete.

`exposure-ledger.json` records that exact corpus-manifest and selection hash
plus all six cell states from the retained R5 audit (SHA-256
`896566dc3edefa4308c2f9874f07b9d9753e84aba695aca9fae9a8da34b74390`).
Its SHA-256 is `2dd66aee9b3aefca1637c2ea341bc685bc3941d0bff8912ae09cbaecbcf18005`.
`hpr_no_redraw_guard.reserve` checks a pinned inventory, refuses missing,
unknown or already exposed cells and unknown cohort changes, and atomically
creates a durable exclusive cell reservation before launch. A reservation
survives interruption and a second attempt fails. It is conservative: even a
crash before the first request leaves the cell treated as possibly exposed.

`full_run_guarded.py` is a prospective replacement of the R5 `full_run.py`
(original SHA-256 `63162f9fd68a7360b5b837023894b9d15bf04dff7ecd34da709cc501ed300b32`).
It requires the pinned inventory and a single shared claims directory, and
calls the reservation guard before the native cell/server launch. The original
packet has not been edited, rearmed or run. The checked-in replacement must be
installed together with `hpr_no_redraw_guard.py` in a separately qualified
successor packet; all six original cells intentionally refuse. No READY row
or six-cell result is asserted. A genuinely new disjoint cohort requires its
own evidence-backed inventory and independent predeclaration; changing its
manifest hash alone does not prove disjointness.

The ledger and claims directory are ordinary same-UID files; the operator must
provide persistent shared custody and protect them from replacement/removal.
The helper is an accidental-replay and concurrency guard, not an adversarial
security boundary or proof of a restored checkpoint. Four offline controls
exercise all original cells, unknown cohort and incomplete inventory refusal,
eight-way reservation races, interrupted-reservation replay and guard order.
