# AMENDMENT (adopted by workspace-58, 2026-10-24 19:5xZ): HSF-5.P G4 log capture

Status: ADOPTED instrument amendment (operator-directed review by the owning session). G1 is evaluated with ALL arms at --log-verbosity 4 (within-run comparison). This draft does not change the G4 gate text, thresholds, or verdict rules.

The 2026-10-24 15:44Z HSF-5.P run remains `INCONCLUSIVE` as recorded in its `VERDICT.json`. Do not re-grade that run.

For a future run, each of the four server arms must use `--log-verbosity 4`. The harness captures server stdout and stderr in `server.$a.log`. This threshold admits the libllama INFO bind line required by G4. The setting must be identical for A1, B, A2, and Bf.

G4 requirements remain unchanged: B's log must contain `HSF-5.P kv host mirror ON`, a `kv-host-mirror save:` line with `tail_copies > 0` and `0 <= replay_steps <= 6`, and a `kv-host-mirror restore:` line with `device_bytes > 0`. Missing enable text remains `INCONCLUSIVE`; other G4 misses retain the existing rule.

This is a log-capture instrument change. It does not establish that the mirror improves performance or alter any grading rule. The future run must use its own predeclared, reviewed pin set.
