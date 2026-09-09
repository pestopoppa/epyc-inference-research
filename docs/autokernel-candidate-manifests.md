# AutoKernel candidate manifests (offline contract)

This module supplies immutable, versioned records for assembled candidates, keep treatments,
leave-one-out (LOO) plans, validation rows/batches, and the separate integration and validated
pointers. It is an offline contract only: it does not build, launch, claim resources, grade native
evidence, write the journal, or move a production reference.

Each manifest freezes its parent and production-kernel-set digest, Git sources (including explicit
SHA-1 or SHA-256 object format), a set of exact executable/DSO builds, and targets that select one
build while also binding resolved-recipe, model, optional drafter, and workload identities. CPU and
HIP targets may therefore share a source tip without falsely sharing an executable.

Validation rows bind the candidate and comparator target identities, including backend, build,
recipe, model/drafter, workload, objective, protocol, and instrument. A serialized `passed` status
is historical data, not warrant. `advance_validated` requires caller-injected registered verifiers
for every required row and every required eligible LOO result. Optional seed rows may remain
explicitly pending and cannot block required-row batch completion; they are never treated as
passed. Validation debt remains until trusted advancement; completing a genuinely attempted
inconclusive gate resets the four-keep cadence, while a pre-run prerequisite refusal does not.
Gain-trigger debt uses monotonic trigger and covered generations, so an older in-flight batch cannot
erase a newer trigger when it completes.

The offline summary command accepts `--manifest`, optional `--row-set`, `--batch`, `--state`, and
`--out`. It always emits `execution_authorized: false`, `production_promotion_authorized: false`,
and `validated_pointer_eligible: false`:

```bash
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.candidate_cli \
  --manifest candidate.json --row-set rows.json --batch batch.json --out summary.json
```

All example digests used in tests are synthetic. Actual Journal integration, registered protocol
and evidence adapters, cross-repository intent recovery, resource grants, execution, and any
production transition remain later integration work. This slice alone does not complete AKU-05.
