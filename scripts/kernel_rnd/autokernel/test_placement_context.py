from __future__ import annotations

import hashlib
import json

import pytest

from scripts.benchmark import autokernel_p2_5j_receipt as P
from scripts.benchmark.tests.test_autokernel_p2_5j_receipt import campaign
from scripts.kernel_rnd.autokernel import placement_context as C


def _write_receipt(tmp_path):
    value = P.finalize_campaign(campaign(tmp_path), base_dir=tmp_path)
    path = tmp_path / "placement-receipt.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, value


def test_context_preserves_all_arms_and_authority_boundary(tmp_path) -> None:
    path, value = _write_receipt(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    context = C.load_placement_context(path, expected_sha256=digest).discovery_context()
    assert [arm["arm"] for arm in context["host_placement_arms"]] == list(P.ARM_SPECS)
    assert context["placement_verdict"]["selected_arm"] == "I"
    assert context["placement_verdict"]["observed_leader_arm"] == "Lp"
    assert context["authority"] == {
        "placement_context_only": True,
        "observation_only": True,
        "kernel_speedup_claim": False,
        "carve_authorized": False,
        "production_activation_authorized": False,
    }
    assert context["evidence"]["receipt_self_sha256"] == value["receipt_sha256"]


def test_file_hash_mismatch_is_refused(tmp_path) -> None:
    path, _ = _write_receipt(tmp_path)
    with pytest.raises(C.PlacementContextError, match="receipt hash mismatch"):
        C.load_placement_context(path, expected_sha256="0" * 64)


def test_self_hash_tamper_is_refused(tmp_path) -> None:
    path, value = _write_receipt(tmp_path)
    value["arm_summaries"]["I"]["median_decode_tps"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(C.PlacementContextError, match="self-hash mismatch"):
        C.load_placement_context(path)


def test_missing_arm_is_refused_even_with_recomputed_hash(tmp_path) -> None:
    path, value = _write_receipt(tmp_path)
    value["arm_summaries"].pop("H")
    value["receipt_sha256"] = P.receipt_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(C.PlacementContextError, match="all four"):
        C.load_placement_context(path)
