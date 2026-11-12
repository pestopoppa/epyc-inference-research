# AGG-1.9 predeclared, amendment 1 (post-measurement, pre-model): the guard definitions as measured

Written 2026-11-12 by the AGG-1.9 / AGG-1.10 preparation session, **after** the zero-model guard measurement on the AGG-1.1
store and **before any AGG-1.9 model call**. `PREDECLARED_AGG19.md` (sha256
`4eed298ac95022de532f0a30844ab6d7af684e261e7e1a3e4109ae749ad7917c`) is not edited. This amendment changes no gate, no
question, no gold, no stop rule and no estimate; it replaces the two guard definitions the original file sketched with the
ones that passed its own acceptance rule, and says which were tried.

## What was tried (all on the AGG-1.1 store, 211 verified cells, before any fill; `canonical.v1` control 177 of 211 correct)

| Variant | Fixed | Broken | Decision under the predeclared rule (fix >= 1, break 0) |
|---|---|---|---|
| attributable net income, as sketched (any quote with "attributable to") | 4 net income cells (Boeing, MGM, PepsiCo, Verizon) | 2 (American Express net income: "attributable to common shareholders" is after preferred dividends; eBay net income: "attributable to Adevinta" is another entity) | REJECTED |
| thousands-unit, as sketched (value >= 100x the median of the others, or 1/1000 of a consensus) | 0 | 2 (Amazon total current liabilities, Johnson and Johnson dividends paid: the 100x clause keeps the junk peers and drops the true value) | REJECTED |
| attributable net income, narrowed: "attributable to" must be followed by the company's own name (first entity token) or by shareholders / stockholders / owners, and the quote must not carry noncontrolling, minority, preferred, redeemable, participating, discontinued or common; among attributable candidates the value most of them agree on leads, then the `canonical.v1` order | 4 (Boeing, MGM, PepsiCo, Verizon net income) | 0 | **ADOPTED** |
| thousands-unit, narrowed: only an exact x1000 or /1000 slip (within 0.5 percent) against a value at least two other candidates agree on; the 100x clause removed | 0 | 0 | NOT ADOPTED (fixes nothing on this store); implemented and opt-in, never part of the adopted rule |

The full counts and cell lists are in `agg19_guard_probe.json` (sha256 recorded in the row and the runner). The tier flip of
AGG-1.8 (prefer `l4_llm`, 7 fixed, 19 broken) was not repeated.

## What the fill row uses

* The adopted rule is `canonical.v2` with `guards=("attributable_net_income",)`, applied to the filled store; `canonical.v1`
  stays the default of the library and is what the AGG-1.1 store holds. The adopted set is read from the frozen
  `agg19_guard_probe.json` (it is not re-decided after the fill). The probe is recomputed from the base store at preflight and
  must equal the frozen one.
* After the fill the guard variants are measured again on the FILLED store and filed as informational (`variants_after_fill`);
  a guard that would be adopted only by the filled-store measurement is reported, not switched on in this row.
* The retained-cell regression gate (DONE WHEN 3) is evaluated under the adopted rule on both measures, unchanged.
* Orchestrator code: `canonical.v2` is orchestrator `32a3d859` plus `6ec701df` (the `guards=` argument through
  `refresh_canonical`); `facts_llm.py` is unchanged from `1a408ce4` (sha256
  `c64b7c9ed7df88906d38f07e71c5605bcbb4aae66549452ae42487c257dd7451`), so the L4 stage is exactly the AGG-1.3 / AGG-1.4 amended stage.

## Budget-stop calibration (a stop threshold, not a gate)

`PREDECLARED_AGG19.md` makes a projected total above 2x the booked estimate a budget stop. The box's 600 s ESTIMATE assumes 26 to
40 L4 calls; the measured rate is about 6 L4 calls per (field, document) unit (AGG-1.1: 876 calls over 144 new units; AGG-1.4 r3:
72 calls over 12 units, 54.5 s per unit), so 26 units is about 158 calls and about 1,420 s. With the 600 s figure the pilot's own
projection would trip the budget stop on measured rates. The projected-wall budget stop therefore uses 2x 1,400 s = 2,800 s; the
300 s pilot-wall bound, the pilot rule and every gate are unchanged. The compute row keeps the box's 600 s ESTIMATE as its
`estimate_s` (flagged in the row as likely low); it is resumable, and a grant-deadline stop is rc 75, a re-grant, not a gate.

## Cells fixed by neither guard (reported as not addressed)

Of the 8 cells with a correct candidate in `facts` that `canonical.v1` passed over: 4 are repaired by the adopted guard; 4 are not
(Amazon net income: the consolidated net loss line competes with a 5-fold repeated correct value and loses on magnitude; Corning,
Costco and MGM revenue: a segment or "net sales" line wins on tier). A third guard for those is NOT added in this row.

## Post-hoc caveat, repeated

The two guards were designed and then accepted on the same 211 cells that motivated them. The narrowing above was made after
seeing which cells the first drafts broke. Zero broken cells on this store is a regression check on this store, not evidence of
generalisation.
