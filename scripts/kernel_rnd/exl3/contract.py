"""Fail-closed, content-addressed EXL3 native packed artifact v1.

Validation is a planning phase: metadata and file sizes are checked before any
payload read or reconstruction allocation. No backend or inference dependency.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "epyc.exl3.artifact.v1"
PACKING = "native-k-major-le-u16-msb32-circular256-v1"
ROLES = {"q", "k", "v", "o", "gate", "up", "down", "lm_head"}
MAX_ELEMENTS = 1_048_576  # reference tool admission bound; never infer a full model


class Refusal(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exact(obj, keys, label):
    if not isinstance(obj, dict) or set(obj) != set(keys.split()):
        raise Refusal(f"{label}: unknown or missing fields")


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def hash_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(Refusal(s)))


def _validate(manifest, *, max_elements=MAX_ELEMENTS):
    exact(manifest, "schema artifact_sha256 source matrices", "manifest")
    if manifest["schema"] != SCHEMA:
        raise Refusal("unknown artifact schema")
    body = {k: v for k, v in manifest.items() if k != "artifact_sha256"}
    if manifest["artifact_sha256"] != digest(body):
        raise Refusal("manifest digest mismatch")
    source = manifest["source"]
    exact(source, "repository revision kind", "source")
    if (not isinstance(source["repository"], str) or not source["repository"] or
            not isinstance(source["revision"], str) or
            re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source["revision"]) is None or
            source["kind"] not in {"synthetic", "real_weight", "legacy_real_capture"}):
        raise Refusal("source revision/kind must be pinned")
    if not isinstance(manifest["matrices"], list) or not manifest["matrices"]:
        raise Refusal("matrices must be nonempty")
    ids, domains, paths = set(), set(), set()
    total = 0
    for m in manifest["matrices"]:
        exact(m, "id tensor_role shape padded_shape order K rate_x2 codebook packing hadamard scaling padding expert_domain tensors source_sha256", "matrix")
        if not isinstance(m["id"], str) or not m["id"] or m["id"] in ids:
            raise Refusal("matrix identity missing/ambiguous")
        ids.add(m["id"])
        if m["tensor_role"] not in ROLES or not hash_string(m["source_sha256"]):
            raise Refusal("unknown tensor role or source digest")
        for field in ("shape", "padded_shape"):
            if not isinstance(m[field], list) or len(m[field]) != 2 or not all(integer(x, 1, max_elements) for x in m[field]):
                raise Refusal("shape must be [input, output] positive integers")
        ki, no = m["padded_shape"]
        total += ki * no
        if total > max_elements or ki % 128 or no % 128 or any(a > b for a, b in zip(m["shape"], m["padded_shape"])):
            raise Refusal("invalid padding, H128 geometry, or allocation capacity")
        if not integer(m["K"], 1, 8) or type(m["rate_x2"]) is not int or m["rate_x2"] != 2 * m["K"]:
            raise Refusal("only consistent integer K1-K8 supported; odd rate_x2 refused")
        if m["codebook"] not in {"mul1", "mcg"} or m["order"] != "input_output" or m["packing"] != PACKING:
            raise Refusal("unknown codebook/order/packing")
        if m["hadamard"] != {"input": 128, "output": 128, "normalization": "orthonormal", "rounding": "fp16_rne_each_stage", "algorithm": "normalized_matrix_ordered_fp32_fma"}:
            raise Refusal("unknown transform geometry or rounding")
        if m["scaling"] != "folded_in_suh_svh" or m["padding"] != "zero_input_crop_output_after_transform":
            raise Refusal("unknown scaling/padding policy")
        domain = m["expert_domain"]
        exact(domain, "kind local global count", "expert_domain")
        if domain["kind"] == "dense":
            if domain != {"kind": "dense", "local": None, "global": None, "count": 0}:
                raise Refusal("conflicting dense expert domain")
        elif domain["kind"] == "expert":
            if not integer(domain["count"], 1, 1000000) or not integer(domain["local"], 0, domain["count"] - 1) or not integer(domain["global"], 0, 1000000):
                raise Refusal("invalid local/global expert domain")
            key = (m["tensor_role"], domain["local"])
            if key in domains:
                raise Refusal("ambiguous expert projection")
            domains.add(key)
        else:
            raise Refusal("unknown expert domain")
        exact(m["tensors"], "trellis suh svh bias", "tensors")
        for name, size in (("trellis", ki * no * m["K"] // 8), ("suh", ki * 2), ("svh", no * 2), ("bias", no * 2)):
            desc = m["tensors"][name]
            if name == "bias" and desc is None:
                continue
            exact(desc, "path sha256 nbytes dtype shape", name)
            expected_shape = [ki // 16, no // 16, 16 * m["K"]] if name == "trellis" else [ki if name == "suh" else no]
            expected_dtype = "uint16_le" if name == "trellis" else "float16_le"
            if desc["shape"] != expected_shape or desc["dtype"] != expected_dtype or type(desc["nbytes"]) is not int or desc["nbytes"] != size or not hash_string(desc["sha256"]):
                raise Refusal(f"{name}: conflicting tensor metadata")
            path = desc["path"]
            if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts or path in paths:
                raise Refusal("tensor paths must be distinct contained relative paths")
            paths.add(path)
    return manifest


def validate(manifest, *, max_elements=MAX_ELEMENTS):
    try:
        return _validate(manifest, max_elements=max_elements)
    except (TypeError, KeyError, OverflowError, ValueError) as exc:
        if isinstance(exc, Refusal):
            raise
        raise Refusal(f'malformed metadata: {exc}') from exc


def load(path, *, max_elements=MAX_ELEMENTS):
    """Return validated manifest and native bytes, without repacking.

    All descriptors and sizes are admitted before *any* tensor file is read.
    Digests are then checked before callers allocate output matrices.
    """
    path = Path(path)
    if path.stat().st_size > 1 << 20:
        raise Refusal("manifest exceeds metadata capacity")
    manifest = validate(read_json(path), max_elements=max_elements)
    files = []
    root = path.parent.resolve()
    for m in manifest["matrices"]:
        for desc in m["tensors"].values():
            if desc is None:
                continue
            file = root / desc["path"]
            if not file.resolve().is_relative_to(root) or not file.is_file() or file.stat().st_size != desc["nbytes"]:
                raise Refusal("tensor path or length mismatch")
            files.append((file, desc))
    payload = {}
    for file, desc in files:
        data = file.read_bytes()
        if sha(data) != desc["sha256"]:
            raise Refusal("tensor digest mismatch")
        payload[desc["path"]] = data
    return manifest, payload


def bind_repack(record, manifest):
    """A backend layout is derivative evidence, never a replacement identity."""
    validate(manifest)
    exact(record, "schema canonical_sha256 backend layout payload_sha256", "repack")
    if (record["schema"] != "epyc.exl3.repack.v1" or
            record["canonical_sha256"] != manifest["artifact_sha256"] or
            not all(isinstance(record[k], str) and record[k] for k in ("backend", "layout")) or
            not hash_string(record["payload_sha256"])):
        raise Refusal("backend repack identity does not bind canonical input")
    return record
