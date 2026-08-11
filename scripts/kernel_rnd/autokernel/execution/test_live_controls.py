import importlib
import os
import unittest
from unittest import mock

from . import live_controls


class InstrumentSelection(unittest.TestCase):

    def tearDown(self):
        importlib.reload(live_controls)

    def test_clean_instrument_worktree_can_be_selected_explicitly(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument"}, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_ROOT), "/tmp/ak-clean-instrument")
        self.assertEqual(
            str(module.INSTRUMENT_BINARY),
            "/tmp/ak-clean-instrument/build-v9-cpu/bin/llama-bench")

    def test_binary_override_is_separate_and_explicit(self):
        with mock.patch.dict(os.environ, {
                "AUTOKERNEL_INSTRUMENT_ROOT": "/tmp/ak-clean-instrument",
                "AUTOKERNEL_INSTRUMENT_BINARY": "/tmp/ak-build/bin/llama-bench",
        }, clear=False):
            module = importlib.reload(live_controls)
        self.assertEqual(str(module.INSTRUMENT_BINARY), "/tmp/ak-build/bin/llama-bench")


if __name__ == "__main__":
    unittest.main()
