# Prospective FA-4a exact-set control proposal

Before future collection, freeze the selected source-derived native case IDs,
candidate source/build and library hashes, exact input bytes for each case,
repeat count, arm order, backend/argv, raw output and observation custody in a
sealed plan. Candidate ON and OFF use identical input bytes. At value bias
256, require complete matched coverage and no nonfinite output in either arm;
then run the existing native finite numerical and OFF exactness controls with
their already declared thresholds. This finite-set check cannot replace those
numeric checks or prove equivalence by itself.

At bias 1024 retain all original stress cases and report per-key nonfinite
element counts separately per arm and repeat. Membership differences are
observations, not a pass, equivalence or mechanism verdict. Preserve the old
228-case outcome and original predeclaration: this is a prospective proposal,
not a retroactive regrading or authorization to bypass the parent FA gate.
The timing band comparison remains gated on the owner's accepted amendment
and all other original controls, including ROCm residency, model-level checks,
and native backend correctness. R17's published retrospective record is not
inferred from the new control.

`v11_fa4a_exactset_control.py` checks a frozen plan (`case_ids`, `repeats`),
plus complete indexed records for 256 and 1024 × ON/OFF × every repeat. Each
record identifies the raw JSON and retained actual input bytes with SHA-256.
The raw JSON must bind `case_id`, `bias`, `arm`, `repeat`, `input_sha256`, and
`nonfinite_elements`. It refuses drift, incomplete/duplicate coverage and
unmatched input bytes. It does not extract data from the existing FNV-only
diagnostic logs, determine numeric accuracy, verify GPU execution or authorize
promotion; those need the producer, native backend and grant/custody evidence.
No execution packet or new GPU observation was created by this preparation.
