# AutoKernel hypothesis portfolio v2 contract

The canonical truth is `discovery_hypothesis_portfolio_v2.json`, schema
`epyc.autokernel.discovery_hypothesis_portfolio.v2`. This document is an authoring
contract, not a second corpus. The loader rejects unknown/missing fields, duplicate
JSON keys and ambiguous or mutable authority carriers.

## Intake workflow

1. Keep the stable `hypothesis_id` or `dnr_id`; increment `record_version` for a
   revision and bind `provenance` (`introduced_at`, actor, origin, note, supersedes).
2. Add absolute evidence paths, SHA-256 identities, authority class, temporal status
   and bounded claims. Evidence does not inherit promotion authority.
3. Bind exact frames: model path/SHA, measurement binary/source identity, graph
   policy, workload regime and hotspot derivation selector/raw-artifact SHA.
4. Declare the mechanism, exact regime, primary and supporting falsifiers, target
   files/symbols, dispatch signatures, interactions and portability bounds.
5. Declare lifecycle maturity and next action. A sealed candidate uses
   `candidate_identity`; dirty bytes use `diagnostic_identity` and can never be a
   candidate authority.
6. Predeclare expected value, implementation cost/risk, stop rules and numeric
   `decision_policy`: relative-percent continuation/nomination/per-replication floors,
   replication count/spread, sign/conflict handling, candidate budget and terminal
   rule.
7. Set current-bundle eligibility only when the exact current frame routes the target,
   there are no blockers, and deployment template authority covers every file, symbol
   and source-manifest change class. Deployment must call
   `validate_template_authorability`; syntactic template membership alone is not
   authority.
8. Validate the corpus and every evidence carrier before sealing it into deployment.

```bash
python3 -m scripts.kernel_rnd.autokernel.hypothesis_portfolio \
  validate scripts/kernel_rnd/autokernel/discovery_hypothesis_portfolio_v2.json \
  --verify-evidence
python3 -m scripts.kernel_rnd.autokernel.hypothesis_portfolio \
  summarize scripts/kernel_rnd/autokernel/discovery_hypothesis_portfolio_v2.json
```

## Hypothesis record

Each record has exact fields for identity/version/provenance; title/status/statement;
`primary_falsifier` plus `falsifiers`; canonical `regime`; `target` frame IDs,
files, symbols and template intent; dispatch anchors; mechanism facets and fingerprint;
evidence; interactions; portability; priority; expected value; implementation; stop
rules; current-bundle eligibility; lifecycle; numeric decision policy; and epistemic
grade/confidence/limitations.

Dispatch anchors preserve all selected exact rows as `signatures` with kernel literal,
calls, grid, workgroup and LDS. `excluded_signatures` makes nearby routes explicitly
out of scope. `total_calls` must equal selected signature calls. A family aggregate
cannot invent one geometry; `not_applicable` carries neither signatures nor calls.

Epistemic grades distinguish design priors, graphs-off routing profiles, dirty
diagnostics, correctness-only candidates, replicated nonpromotable candidate screens
and retired negatives. Graphs-off attribution is routing/device-time-ceiling evidence;
it cannot establish graphs-on whole-model reward direction.

Statuses are `queued`, `candidate_incumbent`, `retired` and `needs-template`.
`candidate_incumbent` is candidate-only and nonpromotable; it never means production
adoption. Maturity separately records design, characterization, authored/correctness
state, screen state, candidate-incumbent state, retirement or dirty diagnostic state.

## Hard do-not-repeat record

Every DNR binds a stable/versioned identity, classification, exact mechanism
fingerprint, exact regime, falsifier result, evidence and reentry conditions under
`hard_refusal_exact_mechanism_and_regime`. Classifications distinguish measured
negative, nonreplication, correctness failure, subadditivity, sign conflict, physics
constraint, prior art, configuration regression and low value. An eligible hypothesis
with the same exact mechanism/regime identity is a corpus contradiction and is refused.

## Consumer projections

`eligible_projection()` returns deeply immutable records containing the exact regime,
mechanism, primary/supporting falsifiers, target, selected/excluded dispatch rows,
frames, templates, evidence, lifecycle, epistemic state, stop rule and candidate
budget. `dnr_projection()` returns deeply immutable mechanism/regime refusal and reentry
records. Neither projection grants compute, production writes, promotion, or operator
approval.
