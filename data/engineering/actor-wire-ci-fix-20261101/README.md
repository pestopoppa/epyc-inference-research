# Actor wire CI fixture correction — 2026-11-01

Base: `c17d0d0b7ae5d697875e67fb2799426b69d65800`; failing GitHub Actions run 41294746044, workflow `actor native wire gate`.

Commit `c17d0d0b7` added an unconditional `--allow-untested-actor-seats` to the shared promotion fixture. The missing-record refusal test reused that fixture, so admission correctly honored the explicit override and reached the actor factory. The fix makes that fixture option configurable and turns it off in both missing/stale refusal cases, retaining the default for custody tests. No production code changed; admission remains before actor factories.

Commands (the first and last are the workflow steps):

```sh
PYTHONPATH=scripts/kernel_rnd python -m pytest -q scripts/kernel_rnd/autokernel/loop/test_actor_wire_gate.py
PYTHONPATH=scripts/kernel_rnd python -m pytest -q scripts/kernel_rnd/autokernel/loop/test_promotion_targets.py
PYTHONPATH=scripts/kernel_rnd python -m autokernel.loop.actor_wire_test --matrix --record actor-wire-pass.json
```

The first command produced 1 failure and 77 passes before the fix (`before.log`) and 78 passes after (`after.log`). The custody suite produced 20 passes (`custody.log`). The native matrix passed all 15 seats (`matrix.log`), against its scripted loopback provider without credentials, models, or inference.

Local execution used Python 3.12.3 (CI specifies 3.13), pytest 9.2.0, mcp 2.3.0, PyYAML 6.0.3, Node 22.23.4, Codex 0.163.0, Claude Code 2.1.310, and OpenCode 1.18.31. Packages and XDG state were isolated in scratch. The matrix ran from scratch with `scripts` pointing to the lane, preserving the workflow command while keeping its generated record out of the working tree. Python dependencies were added to PYTHONPATH. Tests ran under the R4 supervisor on CPUs 74–75 with one-thread BLAS/OpenMP settings; its sampled peak and cleanup results are at the end of each log.

Raw logs are retained byte-for-byte; their scoped whitespace attribute preserves native traceback formatting.
