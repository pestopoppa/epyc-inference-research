"""Build the evidence index from original producer outputs; never create data."""
import hashlib
import json
from pathlib import Path


def reference(root, path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}


def write_index(root, build_receipt):
    root = Path(root).resolve()
    keys = {(8, 32768): "np8_32k", (8, 49152): "np8_48k",
            (8, 65536): "np8_64k", (4, 98304): "np4_96k"}
    arms = {name: [] for name in keys.values()}
    for path in sorted(root.rglob("waves_capture.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("np"), row.get("L"))
            if key not in keys or row.get("c") != 524288 or row.get("kv_unified_from_log") != "true" or not isinstance(row.get("wave"), int) or row["wave"] < 1:
                continue
            wave = row["wave"]
            arms[keys[key]].append({"wave_file": reference(root, path), "wave": wave,
                                    "occupancy_file": reference(root, path.parent / f"wave-{wave}-occupancy.json"),
                                    "runtime_receipt": reference(root, path.parent / f"wave-{wave}-runtime.json")})
    if any(not entries for entries in arms.values()):
        raise ValueError("original output is missing one or more predeclared arms")
    index = {"schema": "epyc.kvu126.evidence_index.v1", "build_receipt": reference(root, build_receipt), "arms": arms}
    path = root / "kvu126-evidence-index.json"
    with path.open("x") as out:
        json.dump(index, out, indent=2)
        out.write("\n")
    return path
