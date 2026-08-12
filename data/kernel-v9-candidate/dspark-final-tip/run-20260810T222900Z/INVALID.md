INVALID — harness request-shape defect

Both arms used a nested `speculative` object, but the v9 server accepts the
flattened `speculative.n_max` request key. Both requests therefore inherited the
launch default and drafted 15 tokens. This run must not be cited for request-cap
or vanilla/DSpark parity evidence. The corrected replay is the sibling
`run-20260810T223200Z-flat-request` artifact.
