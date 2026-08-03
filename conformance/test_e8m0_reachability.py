#!/usr/bin/env python3
"""Sentinels for the E8M0 divergence's reachability argument.

WHY THIS FILE EXISTS
    The divergence between the live CPU and live GPU MXFP4 decode paths is real:
    at 0xFF the CPU yields 2^127 (finite) and the GPU yields +Inf. It is not
    currently reachable in our stack. But "not currently reachable" is a claim
    about the *world*, not about the code, and an unchecked claim about the world
    decays silently.

    An earlier version of this instrument justified the divergence with "the loader
    rejects 0xFF". That was WRONG: the gate runs only under `check_tensors`, which
    defaults to FALSE and is passed by none of our launchers. The safety argument
    had to be replaced with a weaker and more honest one -- we serve no MXFP4 model
    -- and *that* is what these sentinels watch.

    Each test below fails the moment one of the load-bearing preconditions stops
    holding. They are cheap, they touch no GPU, and they are the difference between
    a documented risk and a forgotten one.

SCOPE
    These check REACHABILITY, not correctness. test_e8m0_vectors.py checks the
    decoders; this file checks whether anyone can get hurt by their disagreement.
"""
import os
import subprocess
from pathlib import Path

import pytest

LLAMA = Path(os.environ.get("LLAMA_CPP_ROOT", "/mnt/raid0/llm/llama.cpp"))
ORCH = Path(os.environ.get("EPYC_ORCH_ROOT", "/mnt/raid0/llm/epyc-orchestrator"))
MODELS = Path(os.environ.get("EPYC_MODELS_ROOT", "/mnt/raid0/llm/models"))

pytestmark = pytest.mark.skipif(
    not LLAMA.exists(), reason="production llama.cpp tree not present")


def grep(pattern: str, path: Path, *globs: str) -> list[str]:
    """Read-only grep over the FROZEN tree. Nothing here writes to it."""
    cmd = ["grep", "-rn", "--include=*.c", "--include=*.cpp", "--include=*.h",
           "--include=*.cu", "--include=*.cuh", "-e", pattern, str(path)]
    for g in globs:
        cmd.insert(2, f"--include={g}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def test_no_mxfp4_model_is_served():
    """THE load-bearing precondition. If this fails, the divergence goes live.

    It is the ONLY thing standing between the CPU/GPU disagreement and production,
    now that the loader-gate justification has been retracted.
    """
    hits = []
    for cfg in list(ORCH.glob("orchestration/*.yaml")) + list(ORCH.glob("orchestration/derived/*.yaml")):
        try:
            if "mxfp4" in cfg.read_text(errors="ignore").lower():
                hits.append(str(cfg))
        except OSError:
            continue
    if MODELS.exists():
        hits += [str(p) for p in MODELS.glob("**/*mxfp4*")][:5]
    assert not hits, (
        "AN MXFP4 MODEL HAS APPEARED — the E8M0 CPU/GPU divergence is now REACHABLE.\n"
        f"  found: {hits}\n"
        "  At 0xFF the CPU MXFP4 path yields 2^127 (finite) and the GPU path yields +Inf.\n"
        "  Before serving this model, do ONE of:\n"
        "    (a) pass --check-tensors so 0xFF is rejected at load (it is OFF by default), or\n"
        "    (b) confirm the producing quantizer cannot emit 0xFF, and record that, or\n"
        "    (c) report upstream and pin a fix.\n"
        "  See conformance/matrices/e8m0-conformance.md.")


def test_cpu_nonhalf_decoder_still_has_no_call_sites():
    """`ggml_e8m0_to_fp32` is dead today. If it gains a caller, a THIRD live answer appears."""
    hits = [h for h in grep("ggml_e8m0_to_fp32(", LLAMA / "ggml")
            if "ggml-impl.h" not in h and "_half" not in h]
    assert not hits, (
        "ggml_e8m0_to_fp32 has gained call site(s); it was dead when this instrument was "
        f"written, and it disagrees with BOTH other contracts at 0xFF:\n  " + "\n  ".join(hits))


def test_cpu_live_path_still_uses_the_half_decoder():
    """The CPU MXFP4 path must still go through `_half`; that is what the vectors pin."""
    hits = grep("GGML_E8M0_TO_FP32_HALF(", LLAMA / "ggml")
    assert len(hits) >= 3, (
        "the CPU MXFP4 decode path changed shape — expected >=3 GGML_E8M0_TO_FP32_HALF call "
        f"sites, found {len(hits)}. Re-derive the vectors before trusting them.")


def test_gpu_live_path_still_composes_full_times_half():
    """The GPU reaches the same value by `full(e) * 0.5f`, not by a fused half.

    That composition is exactly why the two disagree at 0xFF (+Inf*0.5 is still +Inf,
    while the fused half yields a finite 2^127). If the GPU ever switches to a fused
    half, the divergence closes on its own and this instrument should be revisited.
    """
    hits = grep("ggml_cuda_e8m0_to_fp32(", LLAMA / "ggml/src/ggml-cuda")
    call_sites = [h for h in hits if "common.cuh" not in h]
    assert call_sites, "ggml_cuda_e8m0_to_fp32 has no call sites — the GPU path changed"
    halved = [h for h in call_sites if "0.5f" in h]
    assert halved, (
        "no GPU call site applies *0.5f any more — the CPU/GPU composition has changed and "
        "the divergence analysis in e8m0-conformance.md is stale:\n  " + "\n  ".join(call_sites))


def test_check_tensors_is_still_off_by_default_or_we_learn_it_changed():
    """Documents the retracted justification so it cannot be quietly reinstated.

    This test PASSES when the default is false (today's reality, and the reason the
    loader gate does not protect us). If upstream flips the default, it fails and
    prompts a re-read -- at which point the gate WOULD protect us and the risk drops.
    """
    common_h = LLAMA / "common/common.h"
    if not common_h.exists():
        pytest.skip("common/common.h not present")
    txt = common_h.read_text(errors="ignore")
    assert "bool check_tensors     = false;" in txt or "check_tensors = false" in txt.replace("  ", " "), (
        "check_tensors no longer defaults to false. That is GOOD NEWS -- the loader gate would "
        "now reject 0xFF -- but e8m0-conformance.md still records it as off by default and must "
        "be updated.")
