"""AKR-SG driver calls the belief writer at record time (toy endpoints; no model, no window)."""
import json

from . import run_paired_serving_ab as drv


def test_toy_driver_exports_belief_receipt_marked_pipeline_only(tmp_path):
    rc = drv.main(["--toy", "--toy-rates", "100,100", "--out-dir", str(tmp_path), "--pairs", "3",
                   "--aa-pairs", "3", "--np", "2"])
    assert rc == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    receipt = manifest["belief_receipt"]
    assert receipt.startswith(str(tmp_path / "store" / "serving-beliefs"))
    rec = json.loads((tmp_path / "serving_ab.json").read_text())
    assert rec["pipeline_only"] is True and rec["effect_unit"] == "session"
    assert rec["belief_capture"]["belief_measurements"]
