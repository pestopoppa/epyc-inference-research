# Model Registry Standards

Canonical format spec for `orchestration/model_registry.yaml` (this repo — the comprehensive
research benchmark record). Moved here 2026-07-30 from epyc-root
`agents/shared/ENGINEERING_STANDARDS.md` (repo-local file-format spec, not cross-repo
engineering policy). Referenced by epyc-root role files (model-engineer, benchmark-analyst).

## Scoring Fields

All `quality_score`, `vl_score`, and `blind_score` fields use an inline YAML map:

```yaml
quality_score: {pct: 65.4, raw: "159/243"}   # standard: pct + raw fraction
vl_score: {pct: 92.0, raw: "11/12"}          # same format for vision-language
blind_score: {pct: 36.0}                      # raw omitted when fraction unavailable
blind_score: {pct: null, note: "not scored"}  # null pct with note for unscored entries
```

- `pct` (float): percentage score — native YAML float for programmatic comparison. Use `null`
  when no single score applies.
- `raw` (string, optional): numerator/denominator fraction when available.
- `note` (string, optional): replaces `raw` for special cases (unscored, multi-config).
- Supplementary context (rescored dates, scale descriptions) goes in YAML inline comments.

**Anti-patterns** (never use):

```yaml
quality_score: 60.5              # bare float — missing raw fraction
quality_score: 66/69 (96%)       # unquoted string — YAML parse error risk
quality_score: "36%"             # quoted string — not programmatically comparable
vl_score: "11/12 (92%)"         # quoted string — mixed format
```

## Registry Scope

- **Research registry** (this repo): comprehensive benchmark record — all tested models, all
  quants, deprecated entries preserved with notes.
- **Orchestrator registry** (`epyc-orchestrator`): active stack only — lean, production-facing,
  compiled from the master. The production stack registry is FROZEN; lineup changes are
  operator-gated.

## Model Entry Requirements

- Paths must be absolute (not relative to any base).
- Per-model serving config (`use_chat_api`, `reasoning`, `kv_cache`, `sampling`) must be set
  before benchmarking — chat-template models produce empty output via `/completion`.
- Deprecated models retain their entry with a `deprecated: true` flag and reason in comments.
- Throughput/quality fields carry provenance — prefer structured `measured: {date, protocol}`;
  never strip the legacy free-text date/protocol comments in reformats (they are the only
  witness for older values; epyc-root `agents/shared/MEASUREMENT_POLICY.md`).
