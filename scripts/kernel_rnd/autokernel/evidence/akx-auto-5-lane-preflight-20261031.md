# AKX-AUTO-5 Launch Preflight — Gitproof Capsule

## Original acceptance text

`AKX-AUTO-5 Launch preflight cross-checks lane binding ↔ inputs ↔ anchor ↔ store (2026-10-06: lane1 bound to inputs-b0ba1d427 while running inputs-802bf9ac6)`

## Source implementation

- Repository: `epyc-inference-research`
- Branch: `codex-ni/code-20261031`
- Base: `bb49234420a28cb86e0aae44f0491a7b48020a1d`
- Implementing commit: `60443141408f97347d2c3ef45d2fd81a8bf7119b`
- Changes: serving lane startup now cross-checks the selected owner's launch, frozen prompts, worktree, anchor, branch, and store before capture or store work. An omitted store is accepted only in the `serial_roster` target-digest default layout. Champion re-anchor emits a lane-binding copy that repoints only the selected lane to the new owned-targets file; peer ownership and policy remain intact.

## Controls

Command, using `/workspace/tmp/pytestenv/bin/python`:

```text
python -m pytest scripts/kernel_rnd/autokernel/loop/test_lane_targets.py scripts/kernel_rnd/autokernel/loop/test_serial_roster.py scripts/kernel_rnd/autokernel/loop/test_champion_reanchor.py -q
```

Final result: **88 passed**. The controls include real `run.py` startup refusals before capture/store creation, required-field and mismatch refusals, the serial-roster default store identity, nonserving compatibility, and a re-anchor with a new anchor path/rebound launch whose generated binding preserves peer fields. A wrong launch remains refused even when `--new-anchor-epoch` is present.

The first focused attempt exposed a selected-owner variable overwritten while resolving peers (61 passed, 8 failed); that defect was fixed. The final run above is clean.

`git diff --check` passed. Shared Ruff `0.15.10` reports one unchanged `F401` at `run.py:552`; the same finding was confirmed on the base commit. No unrelated lint edit was made.

This is deterministic launch-control acceptance. No inference, benchmark, measurement, kernel, or promotion claim was produced.

## Handoff closure draft

Preserve the original task text above unchanged. Proposed completion note:

`✅ 2026-10-31 — Research source commit 60443141408f97347d2c3ef45d2fd81a8bf7119b adds fail-closed serving-lane preflight for lane binding ↔ launch/prompts ↔ anchor/worktree/branch ↔ store, including the serial-roster default store identity. Champion re-anchor relaunches use a copied binding that points only the selected lane at its newly validated owner file. Focused controls: 88 passed; evidence: scripts/kernel_rnd/autokernel/evidence/akx-auto-5-lane-preflight-20261031.md. [no-belief: deterministic implementation/control acceptance]`
