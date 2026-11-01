# AKX-AUTO-8 source controls

Source base: research `92b5dee62f86808732a05138765a6fe28d98c90c`.
The isolated test projection's Git HEAD resolves to that authentic commit.
`SOURCE-PROOF-FINAL.json` binds every changed Python file to the tested
projection; syntax was parsed and the file digests matched. The isolated test
projection includes the exact source copies of the canonical recipe,
belief-capture helper and program template. It is not a serving deployment.

- `watchdog.log`: 21 watchdog controls passed, including a private inert
  serial leaf, cross-entry store exclusion and inherited FD/PYTHONPATH checks.
- `factory.log`: a supervised fake-provider glue control passed; it installs
  no production provider.
- `helperfinal.log`: 80 serial and native CPU fixture tests passed.
- `finalclean.log`: 307 passed, 19 skipped, one failed. The only failure,
  `test_real_serial_interruption_new_holder_same_pending_pair_no_old_sample_mix`,
  reproduced at the same assertion on the unchanged base in `baseline.log`.
  An attempted path deselection did not apply; the test actually ran.
- `hooks-verify.log`: installed wrappers/extras and the regular committed
  helper files matched the canonical bytes before source publication.

The tests used two CPU cores per run, OMP=1 and a bounded process supervisor;
the supervisor summaries at the end of the logs record sampled RSS and child
drain. Sampled RSS is not a kernel-enforced peak. There was no inference,
kernel build or running-service reload. The installed-lock capability is a
cooperative same-UID ownership mechanism, not hostile-same-UID authentication.
