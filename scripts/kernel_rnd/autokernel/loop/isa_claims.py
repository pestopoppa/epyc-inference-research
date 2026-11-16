"""C5-ISA-1: archive the gfx90a ISA of a GPU candidate and refuse absent claimed instructions.

For a built candidate (``libggml-hip.so`` / offload bundle / code object) and the
kernels its diff touches (``--kernel-regex`` on mangled names):

1. **Archive.** The ``llvm-objdump -d`` text of every touched kernel is written gzip'd
   under ``<archive>/isa/<code_object_sha256>/<kernel_sha16>.s.gz`` with a manifest
   (``isa_manifest.json``: binary sha256, toolchain, per-kernel sha256 of the text), so
   the candidate record carries the ISA next to its source diff.
2. **Static claim check.** Each claimed mechanism maps to opcode predicates
   (``MECHANISMS``). A claim with zero matching instructions in every touched kernel is
   ``ABSENT`` and the candidate is ``REFUSED``.
3. **Execution witness.** With a ``gpu_counter_signals`` observation for the measured
   run, a claim whose executed-instruction counter (``mfma_insts`` for mfma,
   ``lds_insts`` for lds_staging) reads 0 for every touched kernel is
   ``NOT_EXECUTED`` -> REFUSED. A counter that was not captured leaves the claim
   ``STATIC_ONLY`` (present in ISA, execution unproven); it is never read as 0 and
   never refuses.

Disassembly reuses ``scripts/kernel_rnd/gfx90a_isa_audit.py`` (ELF/bundle split,
private-buffer classification); nothing is built and no GPU is touched.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.isa_claims.v1"
_AUDIT_PATH = Path(__file__).resolve().parents[2] / "gfx90a_isa_audit.py"
_audit_mod: Any = None


def audit():
    global _audit_mod
    if _audit_mod is None:
        spec = importlib.util.spec_from_file_location("_ak_isa_claims_audit", _AUDIT_PATH)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
        spec.loader.exec_module(mod)
        _audit_mod = mod
    return _audit_mod


def _op(prefixes: tuple[str, ...], *, args: str | None = None, not_args: str | None = None
        ) -> Callable[[str, str], bool]:
    def pred(op: str, a: str) -> bool:
        if not op.startswith(prefixes):
            return False
        if args is not None and args not in a:
            return False
        return not (not_args is not None and not_args in a)
    return pred


# mechanism -> (all-of groups; each group is any-of predicates, counter witness metric)
MECHANISMS: dict[str, tuple[tuple[Callable[[str, str], bool], ...], str | None]] = {
    "mfma": ((_op(("v_mfma",)),), "mfma_insts"),
    "lds_staging": ((_op(("ds_write", "ds_store")), _op(("ds_read", "ds_load"))), "lds_insts"),
    "buffer_load": ((_op(("buffer_load",), not_args=" lds"),), None),
    "lds_dma": ((_op(("buffer_load",), args=" lds"),), None),
    "dot": ((_op(("v_dot",)),), None),
    "dpp": ((lambda op, a: "_dpp" in op or "row_" in a or "quad_perm" in a,), None),
    "s_barrier": ((_op(("s_barrier",)),), None),
}

_KEYWORDS = {
    "mfma": r"\bmfma\b|matrix core|v_mfma",
    "lds_staging": r"\blds\b|shared[- ]memory|ds_read|ds_write",
    "buffer_load": r"buffer_load|buffer load",
    "lds_dma": r"lds[- ]dma|buffer_load[^\n]{0,20}\blds\b|direct-to-lds",
    "dot": r"\bv_dot|\bdot4\b|\bdp4a\b|sdot",
    "dpp": r"\bdpp\b|row_shr|quad_perm",
}


def claims_from_text(*texts: str) -> list[str]:
    """Mechanisms a hypothesis text names (lower-cased keyword match)."""
    body = " ".join(t.lower() for t in texts if t)
    return sorted(m for m, rx in _KEYWORDS.items() if re.search(rx, body))


def count_claims(insts: Iterable[Any], private_quads: set[int] | None = None) -> dict[str, list[int]]:
    """Per mechanism, the matching instruction count of each all-of group.

    A ``buffer_load`` through the private-segment buffer (spill/stack traffic) is
    not a global buffer load and is excluded when ``private_quads`` is given."""
    au = audit() if private_quads is not None else None
    counts = {m: [0] * len(groups) for m, (groups, _w) in MECHANISMS.items()}
    for ins in insts:
        if au is not None and au.classify_private(ins, private_quads):
            continue
        for mech, (groups, _w) in MECHANISMS.items():
            for i, pred in enumerate(groups):
                if pred(ins.op, ins.args):
                    counts[mech][i] += 1
    return counts


def _witness(observation: Mapping[str, Any] | None, metric: str | None,
             kernels: Sequence[str]) -> str:
    if not observation or metric is None:
        return "not_captured"
    values = [row["metrics"].get(metric) for row in observation.get("kernels", ())
              if row.get("kernel") in kernels]
    values = [v for v in values if v is not None]
    if not values:
        return "not_captured"
    return "executed" if any(v > 0 for v in values) else "zero"


def judge(per_kernel: Mapping[str, Mapping[str, list[int]]], claims: Sequence[str],
          observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    unknown = sorted(set(claims) - set(MECHANISMS))
    if unknown:
        raise ValueError(f"undeclared mechanisms {unknown}")
    kernels = sorted(per_kernel)
    results = {}
    for mech in claims:
        present_in = [k for k in kernels if all(n > 0 for n in per_kernel[k][mech])]
        if not present_in:
            status = "ABSENT"
        else:
            w = _witness(observation, MECHANISMS[mech][1], present_in)
            status = {"executed": "EXECUTED", "zero": "NOT_EXECUTED"}.get(w, "STATIC_ONLY")
        results[mech] = {"status": status, "present_in": present_in}
    refused = sorted(m for m, r in results.items() if r["status"] in ("ABSENT", "NOT_EXECUTED"))
    if not kernels:
        verdict = "REFUSED_NO_TOUCHED_KERNEL"
    else:
        verdict = "REFUSED" if refused else "ACCEPTED"
    return {"verdict": verdict, "refused_claims": refused, "claims": results,
            "kernels": kernels}


def archive_and_check(binary: Path, kernel_regex: str, claims: Sequence[str], *,
                      archive: Path | None = None,
                      observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    au = audit()
    rx = re.compile(kernel_regex)
    per_kernel: dict[str, dict[str, list[int]]] = {}
    manifest: list[dict[str, Any]] = []
    import tempfile
    for co in au.load_code_objects(binary):
        co_sha = hashlib.sha256(co).hexdigest()
        with tempfile.TemporaryDirectory(prefix="ak-isa-claims-") as tmp:
            path = Path(tmp) / "co.o"
            path.write_bytes(co)
            text = au._run("llvm-objdump", "-d", "--no-show-raw-insn", str(path))
        split = au.split_disassembly(text)
        for name, insts in split.items():
            if not rx.search(name):
                continue
            per_kernel[name] = count_claims(insts, au.private_resource_quads(insts))
            if archive is not None:
                body = "\n".join(f"{i.op} {i.args}".rstrip() for i in insts) + "\n"
                digest = hashlib.sha256(body.encode()).hexdigest()
                dest = archive / "isa" / co_sha / f"{digest[:16]}.s.gz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(gzip.compress(body.encode(), mtime=0))
                manifest.append({"kernel": name, "code_object_sha256": co_sha,
                                 "isa_sha256": digest, "path": str(dest.relative_to(archive)),
                                 "instructions": len(insts)})
    doc = {"schema": SCHEMA, "binary": str(binary), "binary_sha256": au.file_sha256(binary),
           "kernel_regex": kernel_regex, **judge(per_kernel, claims, observation),
           "counts": per_kernel}
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "isa_manifest.json").write_text(json.dumps(
            {"schema": SCHEMA + ".manifest", "binary_sha256": doc["binary_sha256"],
             "kernels": sorted(manifest, key=lambda r: r["kernel"])}, indent=1) + "\n")
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="C5-ISA-1 claimed-instruction gate")
    ap.add_argument("binary", type=Path)
    ap.add_argument("--kernel-regex", required=True)
    ap.add_argument("--claim", action="append", default=[], choices=sorted(MECHANISMS))
    ap.add_argument("--claims-from-text", default=None, help="derive claims from hypothesis text")
    ap.add_argument("--counters", type=Path, help="gpu_counter_signals observation JSON")
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    claims = sorted(set(args.claim) | set(claims_from_text(args.claims_from_text or "")))
    obs = None
    if args.counters:
        from . import gpu_counter_signals  # noqa: PLC0415
        obs = gpu_counter_signals.load_observation(args.counters)
    doc = archive_and_check(args.binary, args.kernel_regex, claims, archive=args.archive,
                            observation=obs)
    text = json.dumps(doc, indent=1, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(json.dumps({"verdict": doc["verdict"], "refused": doc["refused_claims"]}))
    return 0 if doc["verdict"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
