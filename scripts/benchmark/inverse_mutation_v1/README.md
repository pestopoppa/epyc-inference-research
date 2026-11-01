# CJ-7a inverse-mutation fixtures

This is a small, deterministic, judge-free edit-application instrument. It creates four tasks
from one pinned first-party Python source snapshot: arithmetic operator swap, boolean flip,
off-by-one, and removed guard. Each fixture is one case; fixture count is not a representativeness
claim. The source provenance and MIT license are recorded in `fixtures/provenance.json`.

`fixtures.generate_fixtures()` returns immutable fixture records. Give an actor only
`fixture.task_payload()` (`id`, plain-English instruction, and mutated `source`). The expected
source and provenance remain on the trusted scorer side. `scorer.score(fixture_id, candidate)`
returns the existing three-valued gate verdict plus a contemporaneous native proof record:
fixture/source/mutation/instruction pins, candidate raw and normalized SHA-256, expected normalized
SHA-256, formatter identity/configuration, UTC timestamp, verdict, and cause when undecided. Missing
values stay null when they were not observed. It formats the candidate and hidden expected source
with Ruff 0.16.3 in isolated mode with line length 100, checks the pinned executable SHA-256, and
compares formatted text exactly. Ambient Ruff config is ignored. Formatter absence, mismatch,
failure, or timeout is `out-of-coverage` with its registered cause. No candidate source is imported
or executed; there is no model judge or network path.

These fixtures measure mechanical edit application only. They do not measure software engineering,
general coding ability, quality, or a representative task distribution. The source's published
fixture-mix caveat remains: only about 6/106 tasks fall in its named “representative” categories.
No external dataset or costed task reproduction is included.
The bounded native control receipts, including source/candidate/reference hashes and run-resource
envelope, are retained in `fixtures/native_control_proof.json`.

Run the bounded native controls with:

```bash
python3 -m pytest -q scripts/benchmark/inverse_mutation_v1/tests/test_inverse_mutation.py
```
