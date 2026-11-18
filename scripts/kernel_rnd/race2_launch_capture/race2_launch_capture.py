"""Host side of the RVP-RACE-2 launch-capture shim (launch_capture.c). No GPU.

* ``argtable(notes_text)``  -- explicit-kernarg layout per kernel from ``llvm-readelf --notes`` of a gfx90a code
  object (hidden_* args are runtime-filled and skipped), rendered as the shim's RACE2_CAPTURE_ARGTABLE lines.
* ``read_capture(path)``    -- the shim's JSONL folded to unique launches per kernel (grid in workgroups, workgroup,
  dynamic LDS, kernarg bytes) with a dispatch count.
* ``replay_plan(capture, kernel)`` -- the hipModuleLaunchKernel arguments C5-ISA-2 GCN-Gym replays for one kernel.

CLI: ``race2_launch_capture.py argtable <code_object.co|libggml-hip.so> > table.txt``
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import yaml

SHIM_SOURCE = Path(__file__).with_name("launch_capture.c")


def kernel_args(notes_text: str) -> dict[str, dict]:
    i = notes_text.find("amdhsa.kernels:")
    if i < 0:
        return {}
    j = notes_text.find("\namdhsa.", i + 1)
    doc = yaml.load(notes_text[i:j if j > 0 else None], Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
    out = {}
    for k in doc.get("amdhsa.kernels") or []:
        args = [(int(a[".offset"]), int(a[".size"])) for a in k.get(".args") or []
                if not str(a.get(".value_kind", "")).startswith("hidden_")]
        out[k[".name"]] = {"kernarg_segment_size": int(k.get(".kernarg_segment_size", 0)), "args": args}
    return out


def argtable(notes_text: str) -> str:
    lines = []
    for name, k in sorted(kernel_args(notes_text).items()):
        if " " in name:
            raise ValueError(f"kernel name with a space: {name!r}")
        lines.append(" ".join([name, str(k["kernarg_segment_size"]), str(len(k["args"]))]
                              + [f"{o}:{s}" for o, s in k["args"]]))
    return "\n".join(lines) + ("\n" if lines else "")


def read_capture(path: str | Path) -> dict[str, list[dict]]:
    agg: dict[tuple, int] = collections.Counter()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue     # a torn final line from a killed process
        agg[(r["k"], tuple(r["grid"]), tuple(r["block"]), int(r["shmem"]), r.get("kernarg"))] += 1
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for (k, g, b, sh, ka), n in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4] or "")):
        out[k].append({"grid_wg": list(g), "workgroup": list(b), "dynamic_lds": sh,
                       "kernarg_hex": ka, "waves_per_wg": -(-(b[0] * b[1] * b[2]) // 64), "dispatches": n})
    return dict(out)


def replay_plan(capture: dict[str, list[dict]], kernel: str) -> list[dict]:
    """hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, NULL, extra=kernarg) per unique launch.
    A launch without kernarg bytes cannot be replayed and is refused."""
    plan = []
    for l in capture.get(kernel, []):
        if l["kernarg_hex"] is None:
            raise ValueError(f"{kernel}: launch without captured kernarg (missing from the arg table)")
        plan.append({"gridDim": l["grid_wg"], "blockDim": l["workgroup"], "sharedMemBytes": l["dynamic_lds"],
                     "kernarg": bytes.fromhex(l["kernarg_hex"]), "dispatches": l["dispatches"]})
    return plan


def _main(argv):
    if len(argv) == 2 and argv[0] == "argtable":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import tempfile
        import gfx90a_isa_audit as ia
        text = []
        for co in ia.load_code_objects(Path(argv[1])):
            with tempfile.NamedTemporaryFile(suffix=".co") as t:
                t.write(co); t.flush()
                text.append(argtable(ia._run("llvm-readelf", "--notes", t.name)))
        sys.stdout.write("".join(text))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
