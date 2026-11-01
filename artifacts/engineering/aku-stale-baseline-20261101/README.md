# Shared COR stale-baseline fence — source and control evidence

Implementation commit: `2f047a7b1a1a47f30d97e27ddd4c4153d11f911f` (parent `1421b578ae4d5c9afbf3b2199f89893b093a0819`). Source file hashes are in `source-final-before.sha256`; all eight matched after the final guarded run. The earlier source pin precedes review fixes and is retained for history.

The shared COR authority supplies the derived stale state. Explicit stale recovery verifies ancestry and the existing surface journal, binds an intent before the native own-surface comparison, and rechecks the COR, tip and journal before saving its measured result. A changed authority produces an aborted receipt. Late surface initialization is unknown until its own measurement. Seed initialization refuses existing shared COR state. No live production COR, ledger, model, kernel, or performance claim was changed by these controls.

All tests used CONTROL_ONLY synthetic stores and launch stubs, the native admission and supervisor under R6 pair 124–125, two threads, 8 GiB sampled tree limit and 900-second ceiling. Gzip files preserve the raw logs byte-for-byte; they contain the supervisor receipts, peak sampled RSS and cleanup outcomes. Separate runs overlap and must not be added as a single count:

- `guarded-pytest.log.gz`: 332 passed, one outdated serving validity source assertion failed; that assertion was updated to require the effective authority predicate.
- `guarded-pytest-final.log.gz`: 349 passed, two heartbeat AST selectors failed. The base commit selects public `main` while its lifecycle implementation is `_main`; `head-baseline.txt` captures both base sources. Selectors were corrected without removing lifecycle or status assertions.
- `guarded-focused-final.log.gz`: 72 passed after those selector corrections, including heartbeat, surface, serving-gate and seed controls.
- `guarded-cold-init.log.gz`: 31 passed after guarding late seed initialization.
- `guarded-cpu-final.log.gz`: 51 passed after adding a real serving reducer CONTROL_ONLY CPU wrapper, incomplete/failure retention, and incompatible CLI option refusal.
- `missing-sparse-reproduction.log.gz`: intentionally failed single-test reproduction of the tracked recipe absent from the sparse checkout. Its committed blob and canonical copy had SHA256 `17e9e51eaae2a4af8e649fa16b1bed0d1d634982ec7822e9cb2280bf587a8fe3`; the exact blob was materialized only for the affected suite and removed afterward.

The first broad run reported 305 passed and 28 failed, including missing sparse-fixture failures and one stale source assertion. Its full raw output was streamed and truncated by the execution tool, so it is not reproduced or claimed as a complete raw log here; the later complete logs and the targeted missing-fixture reproduction document the diagnoses. Before the successful R6 runs, a physical-pair attempt was refused by admission, and two environment attempts exited before collection because their interpreters lacked pytest; no guard was bypassed.

The final complete affected suite ran at proof HEAD `06f6fde8dca44985039af07c5626df5ce904896f` with implementation `2f047a7b1a1a47f30d97e27ddd4c4153d11f911f`: **354 passed** in 29.53 seconds. `guarded-committed-final.log.gz` retains the full raw output and admission/supervision receipt; eight hashes in `source-committed-before.sha256` matched after the run. The temporary committed recipe fixture was removed afterward. This final suite is a separate run, not a sum of earlier controls.
