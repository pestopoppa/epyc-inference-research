"""The finalizer indexes original files and refuses incomplete evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import kvu126_finalize


class FinalizerTest(unittest.TestCase):
    def test_incomplete_arm_or_missing_receipt_never_creates_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build.json"
            build.write_text("{}")
            cell = root / "arm" / "cell"
            cell.mkdir(parents=True)
            wave = {"np": 8, "L": 32768, "c": 524288, "kv_unified_from_log": "true", "wave": 1}
            (cell / "waves_capture.jsonl").write_text(json.dumps(wave) + "\n")
            with self.assertRaises(FileNotFoundError):
                kvu126_finalize.write_index(root, build)
            self.assertFalse((root / "kvu126-evidence-index.json").exists())


if __name__ == "__main__":
    unittest.main()
