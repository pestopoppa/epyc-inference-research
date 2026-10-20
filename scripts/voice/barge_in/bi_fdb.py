#!/usr/bin/env python3
"""CS-48 loader for the Full-Duplex-Bench v1/v1.5 scorers at the pinned commit, with the CS-45 patches applied.

FDB clone: ``FDB_REPO`` (default /mnt/raid0/llm/tmp/dive-w3-fdb-repo/repo; scratch checkout, no .git, so the pin is
proven by the file's sha256 equalling ``base_sha256`` in CS-45 ``pins.json``, not by ``git rev-parse``). Patches live in
CS-45's ``scorer_patches`` (torch/torchaudio/torch.hub removed, Silero = the ONE pinned ONNX, no ``.cuda()``). The
patched file is materialised in a cache dir and its sha256 must equal ``patched_sha256``; any difference refuses.
Nothing here calls a cloud judge: ``eval_behavior.py`` / ``eval_user_interruption.py`` (GPT-4o / gpt-4-turbo) are
never imported or run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import bi_audio as A
import bi_vad

HERE = Path(__file__).resolve().parent
COMMIT = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
GAP_READINGS = {
    "code": {"user_merge_gap_s": 0.6, "model_merge_gap_s": 0.5},
    "comment": {"user_merge_gap_s": 0.3, "model_merge_gap_s": 0.05},
}


class PinRefused(RuntimeError):
    pass


def cs45_data() -> Path:
    env = os.environ.get("CS45_DATA")
    return Path(env) if env else HERE.parent.parent.parent / "data" / "turn-taking-cs45-v1"


def fdb_repo() -> Path:
    return Path(os.environ.get("FDB_REPO", "/mnt/raid0/llm/tmp/dive-w3-fdb-repo/repo"))


def pins() -> dict:
    return json.loads((cs45_data() / "pins.json").read_text())


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def materialise(rel: str, cache: Path) -> Path:
    """Apply the CS-45 patch for FDB file ``rel`` onto the pinned base; verify both hashes; return the patched path."""
    spec = pins()["full_duplex_bench"]
    if spec["commit"] != COMMIT:
        raise PinRefused(f"pins.json FDB commit {spec['commit']} != task pin {COMMIT}")
    pf = spec["patched_files"][rel]
    base = fdb_repo() / rel
    if _sha(base) != pf["base_sha256"]:
        raise PinRefused(f"{base}: sha256 {_sha(base)} != pinned base {pf['base_sha256']} (not FDB @ {COMMIT[:7]})")
    out = cache / "fdb-patched" / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    patch = cs45_data() / pf["patch"]
    subprocess.run(["patch", "-s", "-o", str(out), str(base), str(patch)], check=True)
    if _sha(out) != pf["patched_sha256"]:
        raise PinRefused(f"patched {rel} sha256 {_sha(out)} != pinned {pf['patched_sha256']}")
    return out


def load_get_timing(cache: Path, silero_model: str | None = None):
    """Import the patched get_timing.py as a module. ``silero_model`` (mutation hook) redirects tt_silero's default
    model path BEFORE the module builds its PinnedSilero, so a wrong file is refused by the pin check inside."""
    path = materialise("v1_v1.5/evaluation/get_timing.py", cache)
    os.environ["TT_PINS_DIR"] = str(bi_vad.pins_dir())
    m = bi_vad.tt("tt_silero")
    prev = m.DEFAULT_MODEL
    if silero_model:
        m.DEFAULT_MODEL = Path(silero_model)
    try:
        spec = importlib.util.spec_from_file_location("fdb_get_timing_patched", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        m.DEFAULT_MODEL = prev
    return mod


def timing_for(gt, input_wav: Path, output_wav: Path, user_gap: float, model_gap: float) -> dict:
    """FDB's own seg_sec / overlaps / response_gaps with explicit merge gaps (module constants are not edited)."""
    user = gt.seg_sec(gt.load_wav(input_wav), user_gap)
    model = gt.seg_sec(gt.load_wav(output_wav), model_gap)
    stop = gt.overlaps(user, model)
    resp = gt.response_gaps(user, model)
    return {
        "user_segments": [[round(s, 3), round(e, 3)] for s, e in user],
        "model_segments": [[round(s, 3), round(e, 3)] for s, e in model],
        "latency_stop_list": stop,  # [[overlap_start, model_stop], ...]; t_stop = model_stop - overlap_start
        "latency_resp_list": resp,  # [[user_end, model_start], ...];  t_resp = model_start - user_end
        "t_stop_s": [round(e - s, 3) for s, e in stop],
        "t_resp_s": [round(e - s, 3) for s, e in resp],
    }


def run_official_process_folder(gt, folder: Path) -> dict:
    """FDB's unmodified-but-patched ``process_folder`` (code-reading constants); returns its latency_intervals.json."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        gt.process_folder(folder)
    return json.loads((folder / gt.OUT_FILENAME).read_text())


def run_fdb_eval_script(script: str, root: Path, interpreter: str | None = None) -> dict:
    """Run an UNMODIFIED FDB v1 evaluation script (pause_handling / smooth_turn_taking) and parse its summary lines."""
    path = fdb_repo() / "v1_v1.5" / "evaluation" / script
    spec = pins()["full_duplex_bench"]["patched_files"]
    if f"v1_v1.5/evaluation/{script}" in spec:
        raise PinRefused(f"{script} is patched in pins.json; this runner is for unpatched scripts only")
    r = subprocess.run([interpreter or sys.executable, "-I", str(path), "--root_dir", str(root)], capture_output=True,
                       text=True, env={"PATH": os.environ.get("PATH", ""), "TQDM_DISABLE": "1"})
    out = {}
    if r.returncode != 0:
        # FDB's smooth-turn script divides by len(latency_list): it crashes when NO item took the turn. Recorded, not hidden.
        return {"script_error": (r.stderr.strip().splitlines() or ["nonzero exit"])[-1]}
    for line in r.stdout.splitlines():
        if line.startswith("Average take turn:"):
            out["average_take_turn"] = float(line.split(":")[1])
        elif line.startswith("Average latency:"):
            out["average_latency_s"] = float(line.split(":")[1])
    return out
