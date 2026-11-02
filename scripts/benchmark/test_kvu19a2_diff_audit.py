import hashlib
import json

import numpy as np
import pytest

from scripts.benchmark.kvu19a2_diff_audit import audit


def test_localization_and_tamper_refusal(tmp_path):
    rows = []
    for case in (78, 80):
        for arm in ("base", "candidate"):
            for repeat in range(1, 6):
                folder = tmp_path / f"case{case}-{arm}-r{repeat}"
                folder.mkdir()
                data = np.array([1, repeat if arm == "candidate" else 1, 3], dtype="<f4").tobytes()
                (folder / f"out_{case:02d}.bin").write_bytes(data)
                rows.append({"case": case, "arm": arm, "repeat": repeat,
                             "output_sha256": hashlib.sha256(data).hexdigest()})
    (tmp_path / "result.json").write_text(json.dumps({"runs": rows}))
    result = audit(tmp_path)
    assert result["cases"]["78"]["base"]["varying_elements"] == 0
    assert result["cases"]["78"]["candidate"]["varying_flat_ranges_inclusive"] == [[1, 1]]
    (tmp_path / "case78-base-r1/out_78.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        audit(tmp_path)
