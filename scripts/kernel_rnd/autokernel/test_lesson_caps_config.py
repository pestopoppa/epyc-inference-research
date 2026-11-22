import tempfile
import unittest
from pathlib import Path

from autokernel import lesson_cap, lesson_caps_config as cfg
from autokernel.lesson_native import _cap_source_valid
from autokernel.test_lesson_cap import make_receipt


class CapsConfigTest(unittest.TestCase):
    def setUp(self):
        self.receipt = make_receipt(Path(tempfile.mkdtemp(prefix='ak-pm12-s4-')) / 'r')

    def test_off_by_default(self):
        self.assertIsNone(cfg.load_caps({}))
        self.assertIsNone(cfg.load_caps({cfg.FLAG: '0'}))

    def test_caps_are_receipt_derived_and_admitted(self):
        caps = cfg.build_caps(self.receipt)
        (task, entry), = caps.items()
        self.assertEqual(entry['upper_cap'], lesson_cap.receipt_ceiling(self.receipt, 'gfx90a'))
        self.assertTrue(_cap_source_valid(entry['cap_provenance'], task, entry['upper_cap']))

    def test_flag_on_with_bad_evidence_is_none(self):
        import os
        old = os.environ.get(cfg.RECEIPT_ENV)
        os.environ[cfg.RECEIPT_ENV] = str(self.receipt.parent / 'missing')
        try:
            self.assertIsNone(cfg.load_caps({cfg.FLAG: '1'}))
        finally:
            os.environ.pop(cfg.RECEIPT_ENV) if old is None else os.environ.__setitem__(cfg.RECEIPT_ENV, old)

    def test_keep_path_forwards_caps_only_when_enabled(self):
        seen = []
        import os
        from autokernel.loop import keep_lesson
        orig = keep_lesson.submit_keep
        keep_lesson.submit_keep = lambda j, a, producer=None, caps=None: seen.append(caps) or {}
        try:
            os.environ.pop(cfg.FLAG, None)
            cfg.submit_keep_with_caps(None, {})
            os.environ[cfg.FLAG] = '1'
            os.environ[cfg.RECEIPT_ENV] = str(self.receipt)
            cfg.submit_keep_with_caps(None, {})
        finally:
            keep_lesson.submit_keep = orig
            os.environ.pop(cfg.FLAG, None)
            os.environ.pop(cfg.RECEIPT_ENV, None)
        self.assertIsNone(seen[0])
        self.assertIn('AK-PM-12:27b-q8-dense-n16', seen[1])


if __name__ == '__main__':
    unittest.main()
