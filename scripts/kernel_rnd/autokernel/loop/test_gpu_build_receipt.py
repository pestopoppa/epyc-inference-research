"""AKX-ALL-14: a loop-built GPU keep carries the build recipe digest + residency proof."""
from __future__ import annotations

from pathlib import Path

from . import hip_launch_proof as hp, serving
from ..controller import build_recipe

PROVEN_LAUNCH = {"status": "proven", "backend": "gpu", "resident": True,
                 "hip_proof": {"status": hp.PROVEN, "legs": {
                     "linkage": "proven", "kfd_own_vram": "proven", "maps": "proven"}}}


def test_row_residency_fold_carries_the_hip_proof_only_when_every_launch_proved():
    ok = serving._residency_fold([PROVEN_LAUNCH, PROVEN_LAUNCH])
    assert ok["hip_proof"]["status"] == hp.PROVEN and ok["hip_proof"]["launches"] == 2
    missing = serving._residency_fold([PROVEN_LAUNCH, {**PROVEN_LAUNCH, "hip_proof": None}])
    assert missing["hip_proof"]["status"] == hp.UNPROVEN
    refuted = serving._residency_fold([PROVEN_LAUNCH, {**PROVEN_LAUNCH, "hip_proof": {
        "status": hp.REFUTED, "legs": {"maps": "refuted"}}}])
    assert refuted["hip_proof"]["status"] == hp.REFUTED and refuted["hip_proof"]["refuted"] == 1


def test_cpu_rows_and_empty_rows_carry_no_invented_hip_proof():
    assert "hip_proof" not in serving._residency_fold([{"backend": "cpu", "status": "not_applicable"}])
    assert "hip_proof" not in serving._residency_fold(())


def test_receipt_binds_the_v10_recipe_digest_to_the_residency_proof():
    recipe = build_recipe.GFX90A_ROCM62_V10_RECIPE
    residency = serving._residency_fold([PROVEN_LAUNCH])
    receipt = hp.build_receipt(recipe_name=recipe.name, recipe_digest=recipe.sha256(),
                               residency=residency)
    assert receipt["build_recipe"] == "gfx90a-rocm62-v10"
    assert receipt["build_recipe_digest"] == recipe.sha256() and len(recipe.sha256()) == 64
    assert receipt["hip_proof"]["status"] == hp.PROVEN


def test_receipt_without_a_proof_reads_unproven_never_proven():
    for residency in (None, {}, {"status": "proven"}, serving._residency_fold(())):
        receipt = hp.build_receipt(recipe_name="r", recipe_digest="d" * 64, residency=residency)
        assert receipt["hip_proof"]["status"] == hp.UNPROVEN


def test_keep_tail_writes_the_receipt_for_gpu_keeps():
    src = Path(__file__).with_name("run.py").read_text()
    assert 'evidence["gpu_build_receipt"] = hip_launch_proof.build_receipt(' in src
    assert "recipe_digest=recipe.sha256()" in src
