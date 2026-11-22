import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from autokernel import lesson_cap
from autokernel.lesson_native import _cap_source_valid

PROTO = 'ak-pm12-native-27b-q8-dense-n16-v1'
BASELINE_NS = 57496800.0
BYTES = 25669140480
TASK = 'AK-PM-12-receipt-27b-q8-n16'


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def make_receipt(root, baseline_ns=BASELINE_NS, result_baseline=None):
    caps = {}
    for name, ns in (('baseline', baseline_ns), ('verification', 59098400.0)):
        d = root / 'receipt' / name
        d.mkdir(parents=True)
        out = json.dumps({'baseline_ns': ns, 'candidate_ns': 24988000.0,
                          'correctness': True, 'protocol_id': PROTO}).encode()
        (d / 'stdout.txt').write_bytes(out)
        cap = json.dumps({'stdout_sha256': _sha(out), 'returncode': 0}).encode()
        (d / 'capture.json').write_bytes(cap)
        caps[name] = _sha(cap)
    (root / 'RESULT.json').write_text(json.dumps({
        'verdict': 'RECEIPT_OK', 'complete': True, 'smoke': False, 'protocol_id': PROTO,
        'baseline_ns': baseline_ns if result_baseline is None else result_baseline,
        'bytes_moved_per_forward': BYTES, 'captures': caps}))
    return root


class LessonCapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='ak-pm12-s2-'))
        self.receipt = make_receipt(self.tmp / 'r')
        self.peak = lesson_cap.peak_bw('gfx90a')

    def test_measured_table_peak(self):
        self.assertEqual(self.peak, 1.4982e12)

    def test_unknown_part_refuses(self):
        with self.assertRaises(lesson_cap.CapEvidenceError):
            lesson_cap.peak_bw('gfx942')

    def test_tampered_table_refuses(self):
        t = json.loads(lesson_cap.PEAK_TABLE.read_text())
        t['parts']['gfx90a']['peak_bw_bytes_per_s'] = 1.6e12  # spec sheet, not measured
        p = self.tmp / 't.json'
        p.write_text(json.dumps(t))
        with self.assertRaises(lesson_cap.CapEvidenceError):
            lesson_cap.peak_bw('gfx90a', p)

    def test_deriver_on_s3_numbers(self):
        cap = lesson_cap.receipt_ceiling(self.receipt, 'gfx90a')
        self.assertAlmostEqual(cap, BASELINE_NS / (BYTES / self.peak * 1e9))
        self.assertAlmostEqual(cap, 3.356, places=3)
        self.assertGreater(cap, BASELINE_NS / 24988000.0)  # S3 speedup sits under the cap

    def test_receipt_cap_admitted_through_native(self):
        prov = lesson_cap.receipt_provenance(self.receipt, 'gfx90a', TASK)
        cap = lesson_cap.receipt_ceiling(self.receipt, 'gfx90a')
        self.assertTrue(_cap_source_valid(prov, TASK, cap))
        self.assertTrue(_cap_source_valid(prov, TASK))
        self.assertFalse(_cap_source_valid(prov, TASK, cap * 1.5))  # inflated cap
        self.assertFalse(_cap_source_valid(prov, 'other-task', cap))

    def test_paper_cap_still_refused(self):
        paper = {'kind': 'workload_roofline', 'source': 'baseline 57.5ms / (25.7GB / 1.6TB/s)'}
        self.assertFalse(_cap_source_valid(paper, TASK, 3.58))

    def test_tampered_receipt_refused(self):
        bad = make_receipt(self.tmp / 'b', result_baseline=BASELINE_NS * 2)
        self.assertFalse(lesson_cap.verify_cap_provenance(
            lesson_cap.receipt_provenance(bad, 'gfx90a', TASK), TASK))
        (self.receipt / 'receipt/baseline/stdout.txt').write_text('{}')
        self.assertFalse(lesson_cap.verify_cap_provenance(
            lesson_cap.receipt_provenance(self.receipt, 'gfx90a', TASK), TASK))

    def test_relative_or_unknown_part_refused(self):
        self.assertFalse(lesson_cap.verify_cap_provenance(
            lesson_cap.receipt_provenance('r', 'gfx90a', TASK), TASK))
        self.assertFalse(lesson_cap.verify_cap_provenance(
            lesson_cap.receipt_provenance(self.receipt, 'gfx942', TASK), TASK))


if __name__ == '__main__':
    unittest.main()
