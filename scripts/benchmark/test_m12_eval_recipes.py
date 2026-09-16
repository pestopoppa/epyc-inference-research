#!/usr/bin/env python3
"""M-12 B4/B5: the long-context reader recipes and the judge recipe say what the doc says."""

from __future__ import annotations

from pathlib import Path

import pytest

m12 = pytest.importorskip("m12_launch_argv")

EVAL = Path(__file__).resolve().parents[2] / "artifacts" / "serving-recipes" / "eval"
SHIPPED = EVAL.parent


def _spec(name: str) -> dict:
    return m12.launch_spec(EVAL / name, port=8199)


def _flag(argv: list[str], flag: str) -> str:
    """A llama-server flag's value (argv starts with ``taskset -c <cpus>``)."""
    server = argv[argv.index(next(a for a in argv if a.endswith("/llama-server"))):]
    return server[server.index(flag) + 1]


@pytest.mark.parametrize("name, np_, model", [
    ("qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json", 4, "Qwen3.6-35B-A3B-MTP-Q8_0.gguf"),
    ("qwen3.8-27b-q8-gpu-dflash2-longctx196k-np1.json", 1, "Qwen3.8-27B-Q8_0.gguf"),
])
def test_readers_give_every_slot_196k_and_checkpoints(name, np_, model):
    spec = _spec(name)
    argv = spec["argv"]
    assert spec["np"] == np_ and spec["ctx_per_slot"] == 196608
    assert _flag(argv, "-c") == str(196608 * np_) and "--no-kv-unified" in argv
    assert int(_flag(argv, "--ctx-checkpoints")) >= 2
    assert int(_flag(argv, "--cache-ram")) > 0
    assert _flag(argv, "-ub") == "2048"          # the checkpoint stride is 4 + ubatch
    assert _flag(argv, "-m").endswith(model)
    assert _flag(argv, "--device") == "ROCm0" and _flag(argv, "-ngl") == "99"
    assert argv[:3] == ["taskset", "-c", "184-191"]
    assert argv[3] == "/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server"
    assert spec["env"]["LD_LIBRARY_PATH"] == "/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin"


@pytest.mark.parametrize("variant, canonical", [
    ("qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json", "qwen3.6-35b-a3b-q8-gpu-mtp.json"),
    ("qwen3.8-27b-q8-gpu-dflash2-longctx196k-np1.json", "qwen3.8-27b-q8-gpu-dflash2-np4.json"),
])
def test_readers_differ_from_the_canonical_recipe_only_in_ctx_np_and_cache(variant, canonical):
    import json

    a = json.loads((EVAL / variant).read_text())
    b = json.loads((SHIPPED / canonical).read_text())
    differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differing <= {"name", "np", "ctx", "extra_flags"}


def test_judge_is_gemma_orig_q8_without_thinking_or_spec_decode():
    spec = _spec("gemma-4-26b-a4b-orig-q8-gpu-judge-np4.json")
    argv = spec["argv"]
    assert _flag(argv, "-m").endswith("gemma-4-26B-A4B-it-ORIG-Q8_0.gguf")
    assert _flag(argv, "--reasoning") == "off"
    assert "--spec-type" not in argv and "-md" not in argv
    assert spec["ctx_per_slot"] >= 8192


def test_the_recipe_hash_is_stable_across_ports():
    a = m12.launch_spec(EVAL / "qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json", port=1)
    b = m12.launch_spec(EVAL / "qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json", port=2)
    assert a["recipe_hash"] == b["recipe_hash"]


def test_shell_line_pins_the_loader_env():
    line = m12.shell_line(_spec("qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json"))
    assert line.startswith("env -uHSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=")
    assert "--ctx-checkpoints 8" in line
