#!/usr/bin/env python3
"""Write-side capture for the DS41-1017 validation runs (INF-77 DS41-W5/W6, VB-DS41-1017-WIRE).

REPO PATH: epyc-inference-research/scripts/benchmark/ds41_1017_capture.py

WHY THIS EXISTS. The VB-DSV41 source row (epyc-root ``scripts/vidya/adapters/README.md``) requires
that every 1017 validation record binds the 9 shard sha256s (and, for W6, the converted drafter's
digest) AT WRITE TIME. A tuple invented on read claims warrant the run never captured, so this
producer is called by the run harness at the moment each gate's number exists, and nothing here
re-derives a number from logs after the fact.

It does NOT grade. Each row is a native measurement record; the epyc-root adapter
``scripts/vidya/adapters/ds41_1017_validation.py`` projects it into a ``ClaimTuple`` and the shared
``claim_tuple.grade()`` measurement ladder decides. No new grading rule.

Row = one gate number. The sidecar is ``<run_dir>/belief_measurements.jsonl``; each row names its
scored artifact (the raw instrument output) by repo-relative path + byte sha256, and carries:
  * the 9-shard artifact identity (name, bytes, sha256 from the HF-LFS-verified download log);
  * the drafter digest, or an explicit ``null`` with the reason (W5 runs are no-drafter);
  * binary identity: source root, HEAD commit, tracked-dirty flag, binary + libggml-cpu sha256;
  * the complete runtime recipe (argv + env) the instrument ran under;
  * the CPU region claim the run held (region-lock role + cpu list);
  * category (BASELINE for the Sept Q4 comparison arm, CANDIDATE for 1017), metric direction, reps.

Usage (from a harness):
    import ds41_1017_capture as C
    C.capture(run_dir, gate="ppl", arm="1017", metric="ppl_wikitext2_c512", value=..., unit="ppl",
              metric_direction="lower_better", reps=1, reps_basis="chunks=...", scored=<raw file>,
              recipe={"argv": [...], "env": {...}}, region={"role": "build", "cpu_list": "0-95"},
              binary=<path>, claim="...")
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "epyc.ds41_1017_validation.v1"
PRODUCER = "epyc-inference-research/scripts/benchmark/ds41_1017_capture.py"
SOURCE_KIND = "ds41-1017-validation"
RESEARCH_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf"
DOWNLOAD_LOG = "/mnt/raid0/llm/tmp/ds41-1017-swap/download_1017.log"

# HF-LFS sha256 verified by download_1017.sh before each rename (download_1017.log `OK` lines,
# last one 2026-10-21T08:22:38Z). Sizes from the directory listing at capture-module authoring.
SHARDS_1017: tuple[tuple[str, int, str], ...] = (
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf", 49_353_784_544,
     "636918b09de6253bf41af29f06aac3ceeda220e46379d8a2058ef92c4e433ed8"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00002-of-00009.gguf", 49_412_635_712,
     "b26d3fc1e6ae1be6c52665542f3c1fee6a33791478f26e6cd90b5a75940a9f96"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00003-of-00009.gguf", 49_582_960_352,
     "20c7fe7adb0578336d8a5e5f9eed521aeb88e8c119ba46e0916cdaab28931c80"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00004-of-00009.gguf", 49_418_206_432,
     "67592ba15870c4292a615666adcc2736b53cc345b2cccb1c36710a0a6caf15b4"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00005-of-00009.gguf", 49_412_829_632,
     "ecd04e0aa71bfab8cfe79ea7fef7f35b6be0b6772455139e2081d6ce1dcc18ed"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00006-of-00009.gguf", 49_257_603_776,
     "88214e0cc80fbae578b80c114527e6ce725199684e0b6849475c7d00b3d0f54c"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00007-of-00009.gguf", 2_406_482_112,
     "38939faf3e4258f3ae333f0fa1954ce89515b75ceee90d37bf4ebe5fafe285e6"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00008-of-00009.gguf", 101_377_628_544,
     "a9d1634da2ac57459a64b10b69ddedfadda7dcde0e71ca6e55bea7c293c4a9f0"),
    ("DeepSeek-V4.1-Flash-1017-MXFP4-experts-00009-of-00009.gguf", 101_380_404_256,
     "b9832b355288c005da6ef444d02424464dd27f6ebdefeadbec2244264eb65285"),
)
# The Sept comparison arm. Its full-file sha256 was never recorded (only the two published part
# digests, short form, DS41-A2); it is bound by name + size + the part digests, and says so.
SEPT_Q4 = {"name": "DeepSeek-V4.1-Flash-Q4.gguf", "bytes": 518_596_067_328,
           "sha256": None, "part_sha256_short": {"part1": "6442b1f9", "part2": "7c3e1064"},
           "sha256_note": "full-file sha256 never recorded; identity = name+size+published part digests"}

GATES = {"load", "coherence", "ppl", "nmse", "bench", "served", "alpha"}
ARMS = {"1017": "CANDIDATE", "sept-q4": "BASELINE"}
DIRECTIONS = {"higher_better", "lower_better"}  # the shared ladder admits exactly these
NO_DRAFTER_REASON = "W5 runs are no-drafter: the Sept DSpark drafter does not match 1017 (W6 pending)"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_identity(arm: str) -> dict:
    """Bind the served artifact at write time; refuses if any shard is missing or resized."""
    if arm == "1017":
        shards = []
        for name, size, sha in SHARDS_1017:
            p = os.path.join(MODEL_DIR, name)
            actual = os.path.getsize(p)
            if actual != size:
                raise ValueError(f"shard {name} is {actual} B, expected {size}")
            shards.append({"name": name, "bytes": size, "sha256": sha})
        return {"arm": arm, "path": os.path.join(MODEL_DIR, SHARDS_1017[0][0]), "split_count": 9,
                "shards": shards, "sha256_source": DOWNLOAD_LOG,
                "sha256_source_sha256": sha256_file(DOWNLOAD_LOG)}
    if arm == "sept-q4":
        p = os.path.join(MODEL_DIR, SEPT_Q4["name"])
        if os.path.getsize(p) != SEPT_Q4["bytes"]:
            raise ValueError("Sept Q4 artifact size drifted")
        return {"arm": arm, "path": p, "split_count": 1, **SEPT_Q4}
    raise ValueError(f"unknown arm {arm!r}")


def binary_identity(binary: str | Path) -> dict:
    b = Path(binary).resolve()
    bindir = b.parent
    root = bindir.parents[1]
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                              check=True).stdout.strip()
    libcpu = bindir / "libggml-cpu.so"
    return {"binary": str(b), "binary_sha256": sha256_file(b),
            "libggml_cpu_sha256": sha256_file(libcpu.resolve()) if libcpu.exists() else None,
            "source_root": str(root), "head": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "binary_mtime": _dt.datetime.fromtimestamp(b.stat().st_mtime, _dt.timezone.utc).isoformat()}


def _row_hash(row: dict) -> str:
    body = {k: v for k, v in row.items() if k != "row_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def measurement_identity(*, run_id: str, gate: str, arm: str, metric: str, scored_sha256: str) -> str:
    return "ds41w:" + hashlib.sha256(f"{run_id}|{gate}|{arm}|{metric}|{scored_sha256}".encode()).hexdigest()[:24]


def validate_row(row: Any) -> list[str]:
    p: list[str] = []
    if not isinstance(row, dict):
        return ["row is not a mapping"]
    for k in ("schema", "producer", "run_id", "gate", "arm", "metric", "value", "unit", "category",
              "metric_direction", "claim", "date", "emitted_at", "scored_path", "scored_sha256",
              "artifact", "drafter", "binary", "recipe", "region", "measurement_id", "row_sha256"):
        if k not in row:
            p.append(f"missing {k}")
    if p:
        return p
    if row["schema"] != SCHEMA or row["producer"] != PRODUCER:
        p.append("schema/producer mismatch")
    if row["gate"] not in GATES:
        p.append(f"unknown gate {row['gate']!r}")
    if ARMS.get(row["arm"]) != row["category"]:
        p.append("arm/category mismatch")
    if row["metric_direction"] not in DIRECTIONS:
        p.append("bad metric_direction")
    art = row["artifact"]
    if row["arm"] == "1017":
        sh = art.get("shards") if isinstance(art, dict) else None
        if not isinstance(sh, list) or [(s.get("name"), s.get("bytes"), s.get("sha256")) for s in sh] \
                != [tuple(x) for x in SHARDS_1017]:
            p.append("1017 row does not bind the 9 verified shard sha256s")
    dr = row["drafter"]
    if not isinstance(dr, dict) or (dr.get("sha256") is None and not dr.get("reason")):
        p.append("drafter digest or explicit null reason required")
    if row["gate"] == "alpha" and (not isinstance(dr, dict) or not dr.get("sha256")):
        p.append("alpha rows must bind the drafter digest")
    reg = row["region"]
    if not isinstance(reg, dict) or reg.get("role") not in {"bench", "build"} or not reg.get("cpu_list"):
        p.append("region claim (role bench|build + cpu_list) required")
    if row["gate"] in {"bench", "served"} and isinstance(reg, dict) and reg.get("role") != "bench":
        p.append("timing gates must run under a bench-role claim")
    if not isinstance(row["recipe"], dict) or "argv" not in row["recipe"] or "env" not in row["recipe"]:
        p.append("recipe argv+env required")
    if row["measurement_id"] != measurement_identity(run_id=row["run_id"], gate=row["gate"],
                                                     arm=row["arm"], metric=row["metric"],
                                                     scored_sha256=row["scored_sha256"]):
        p.append("measurement_id does not rederive")
    if row["row_sha256"] != _row_hash(row):
        p.append("row_sha256 mismatch")
    return p


def capture(run_dir: str | Path, *, gate: str, arm: str, metric: str, value: Any, unit: str,
            metric_direction: str, claim: str, scored: str | Path, recipe: dict, region: dict,
            binary: str | Path, reps: int | None = None, reps_basis: str = "",
            protocol_id: str = "", drafter: dict | None = None, extra: dict | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    scored = Path(scored).resolve()
    if not scored.is_relative_to(RESEARCH_ROOT):
        raise ValueError("scored artifact must live in the research repo (it is committed)")
    now = _dt.datetime.now(_dt.timezone.utc)
    run_id = run_dir.name
    ssha = sha256_file(scored)
    row = {
        "schema": SCHEMA, "producer": PRODUCER, "source_kind": SOURCE_KIND,
        "run_id": run_id, "gate": gate, "arm": arm, "metric": metric, "value": value, "unit": unit,
        "category": ARMS.get(arm), "metric_direction": metric_direction, "claim": claim,
        "protocol_id": protocol_id, "reps": reps, "reps_basis": reps_basis,
        "date": now.date().isoformat(), "emitted_at": now.isoformat(),
        "scored_path": str(scored.relative_to(RESEARCH_ROOT)), "scored_sha256": ssha,
        "artifact": artifact_identity(arm),
        "drafter": drafter or {"sha256": None, "reason": NO_DRAFTER_REASON},
        "binary": binary_identity(binary), "recipe": recipe, "region": region,
        "extra": extra or {},
    }
    row["measurement_id"] = measurement_identity(run_id=run_id, gate=gate, arm=arm, metric=metric,
                                                 scored_sha256=ssha)
    row["row_sha256"] = _row_hash(row)
    problems = validate_row(row)
    if problems:
        raise ValueError("refusing to write row: " + "; ".join(problems))
    sidecar = run_dir / "belief_measurements.jsonl"
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip() and json.loads(line).get("measurement_id") == row["measurement_id"]:
                return row  # idempotent: the same scored artifact is already captured
    with open(sidecar, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


if __name__ == "__main__":
    import sys
    bad = 0
    for path in sys.argv[1:]:
        for i, line in enumerate(Path(path).read_text().splitlines()):
            if line.strip():
                pr = validate_row(json.loads(line))
                if pr:
                    bad += 1
                    print(f"{path}:{i + 1}: {'; '.join(pr)}")
    print("OK" if not bad else f"{bad} bad rows")
    sys.exit(1 if bad else 0)
