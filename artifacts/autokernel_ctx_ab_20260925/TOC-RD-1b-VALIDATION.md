# TOC-RD-1b validation (2026-10-31)

Implementation: Research `6918763bd7c8c5259a8da8be2832c564707719b7`; ROOT hook/test checkout at `85b5aa48e111c1431b15d6433cbef4104f17ab06`.

The restored run-8 fixture SHA-256 is `72c8677850d955b2598bd9d7254460b0fdead5c0ee2f150c80e897ae0ce47b99`, identical to the recorded actor-call prompt. Only seven period cells changed from the prior committed fixture. The two OAB-9 INDEX copies contain the same restored seven rows; other committed INDEX bytes were unchanged. The staged hook control uses the actual restored prompt, then adds a synthetic account-number row in the same table.

These are source-custody and staged-hook checks, not a new inference measurement.

### Recorded prompt and same-table account control

`EPYC_ROOT=/workspace python3 scripts/kernel_rnd/autokernel/loop/test_perf_period_hook.py`

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.844s

OK
```

### Recorded prompt/control fixture regression

`PYTHONPATH=scripts/kernel_rnd python3 -c import unittest; from autokernel.loop import test_actor_context as m; suite=unittest.TestSuite([m.TheFixtureIsTheRealPrompt('test_fixture_bytes_are_the_recorded_run8_prompt'),m.NodeProfileIsExactlyOneInsertedSection('test_control_fixture_is_the_recorded_prompt_plus_exactly_the_node_profile')]); r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())`

```text
test_fixture_bytes_are_the_recorded_run8_prompt (autokernel.loop.test_actor_context.TheFixtureIsTheRealPrompt.test_fixture_bytes_are_the_recorded_run8_prompt) ... ok
test_control_fixture_is_the_recorded_prompt_plus_exactly_the_node_profile (autokernel.loop.test_actor_context.NodeProfileIsExactlyOneInsertedSection.test_control_fixture_is_the_recorded_prompt_plus_exactly_the_node_profile)
Two deliberate deltas from the recorded bytes now, not one: the inserted ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.002s

OK
```

### Root staged-hook compatibility controls

`python3 -m pytest -q /workspace/scripts/hooks/tests/test_pii_staged_capture.py -k all_seven_public_perf_period_cells_are_allowed or perf_period_exception_does_not_exempt_account_value_in_same_table or later_perf_table_row_with_account_label_still_blocks`

```text
...                                                                      [100%]
3 passed, 70 deselected in 0.99s
```
