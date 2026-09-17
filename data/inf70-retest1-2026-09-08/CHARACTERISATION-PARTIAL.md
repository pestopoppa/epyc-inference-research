# Champion characterisation — PARTIAL RUN, STOPPED. Superseded in PURPOSE, not invalid.

Stopped on operator instruction at ~14:05Z: the champion is **not final** — if the decision-grade
THP test comes out likely-positive, the shim default flips and this characterises a superseded
configuration. **These arms are valid measurements that were correctly collected under
`PREREG-CHAMPION.md` (13:14:21Z, sha256 f8c24f5d…). Nothing here is being discarded for being wrong.**

Region 13:15:03Z–14:05Z. Binary/knob state per the pre-registration; the champion knob state was set
EXPLICITLY (`SOLO_YIELD_ROWCOL=0, SCALE_SPLIT=0, TINY_SOLO_CLAMP=1, GET_ROWS_SOLO=0`) and confirmed
in the server knob readback each arm — bin-r1's compiled defaults are NOT the champion's.

| config | binary | arms complete | tw t/s | screens |
|---|---|---|---|---|
| `CP` champion plain | bin-r1 10303 | 5/5 | 27.2945*, 27.4270, 27.3738, 27.3982, 27.4197 | CP1 FLAGGED, rest CLEAN |
| `PP` pristine plain | bin-p 10221 | 3/3 | 12.6996, 12.6048*, 12.6057* | PP2/PP3 FLAGGED |
| `PM` pristine MTP | bin-p 10221 | 3/3 | 23.5070, 23.4430, 24.8856 | CLEAN |
| `CM` champion MTP | bin-r1 10303 | 4/5 (CM5 truncated by the stop) | 42.7644, 42.8547, 42.8936, 42.8415 | CLEAN |

MTP draft acceptance: **82.1%** (pristine and champion alike; 87.1% on the short arm).

**CM5 is INCOMPLETE** — truncated mid-arm by the stop — and must not be used: a partial prompt set is
a different token mix, not a comparable arm. Preserved as `CM5.rows.jsonl` with this note.

## Reusability for the THP test — the honest answer is NO

The `CP` arms are champion-default, shim-OFF, and superficially look like OFF-side sessions.
**They are not usable as such**, for three independent reasons, any one of which is disqualifying:

1. **Structure differs.** They are 5 arms in ONE session. The THP design is **one arm per session**,
   because the unit for a process-scoped knob is the process launch. Five arms from one launch are
   ONE session-level observation, not five — using them as five would be precisely the
   pseudo-replication this campaign has been correcting all day.
2. **Only one usable session-level datum** would come out of the whole `CP` block, and it cannot be
   paired: pairing requires an adjacent ON launch, and there is none.
3. **Selection.** Choosing which of these to keep, or how to fold a 5-arm session mean into a
   1-arm design, requires judgement made *after* seeing the values. The instruction was explicit:
   if inclusion needs any such judgement, exclude them all.

**Decision: the THP test starts clean.** No arm from this block, and none from the earlier 4v4 THP
block, is carried into it.

## What survives regardless

These numbers stand as a measurement of the **current** champion default state, and if THP is not
adopted they can be re-used directly as the characterisation. **One flag travels with them, and it
must be resolved before any headline is quoted** — see `CHAMPION-DIVERGENCE.md`.
