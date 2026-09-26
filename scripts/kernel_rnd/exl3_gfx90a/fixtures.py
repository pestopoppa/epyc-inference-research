#!/usr/bin/env python3
"""Validate canonical artifacts and derive a digest-bound standalone GPU test envelope.

The packed tensor bytes are copied unchanged. Floating-point scale vectors are
widened exactly from binary16. This is a test transport, never a model format.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def export(manifest_path: Path, output: Path) -> dict:
    from scripts.kernel_rnd.exl3.contract import load, canonical
    manifest, payloads = load(manifest_path)
    fixture_path = manifest_path.parent / "fixture.json"
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    if (fixture["schema"] != "epyc.exl3.fixture.v1" or fixture["artifact_sha256"] != manifest["artifact_sha256"]
            or fixture["operator_path"] != "materialized_weight_fp32_fma_v1"
            or fixture["source_revision"] != manifest["source"]["revision"]):
        raise ValueError("fixture/canonical identity mismatch")
    matrices = [m for m in manifest["matrices"] if m["id"] == fixture["matrix_id"]]
    if len(matrices) != 1:
        raise ValueError("ambiguous fixture projection")
    m = matrices[0]
    provenance=[]
    root=manifest_path.parent.resolve()
    for desc in fixture['provenance']:
        path=(root/desc['path']).resolve()
        if not path.is_relative_to(root) or sha(path.read_bytes())!=desc['sha256']:
            raise ValueError('fixture provenance digest drift')
        provenance.append({'path':str(path),'sha256':desc['sha256']})
    i, o = m["shape"]
    pi, po = m["padded_shape"]
    activations, expected = fixture["activations"], fixture["reference"]["operator_outputs"]
    if not activations or any(len(row) != i for row in activations) or len(expected) != len(activations) or any(len(row) != o for row in expected):
        raise ValueError("operator fixture shape mismatch")
    desc = m["tensors"]
    packed = payloads[desc["trellis"]["path"]]
    floats = bytearray()
    for name in ("suh", "svh", "bias"):
        if desc[name] is not None:
            data = payloads[desc[name]["path"]]
            vals = [x[0] for x in struct.iter_unpack("<e", data)]
            if name == "bias": vals = vals[:o]
            floats.extend(struct.pack(f"<{len(vals)}f", *vals))
    for rows in (activations, expected):
        values = [value for row in rows for value in row]
        if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in values):
            raise ValueError('nonfinite activation or expected output')
        floats.extend(struct.pack(f"<{len(values)}f", *values))
    role=m['tensor_role'].encode('ascii')
    data = (b"EXL3GPU1" + manifest["artifact_sha256"].encode("ascii") +
            sha(canonical(manifest['source'])).encode('ascii')+m['source_sha256'].encode('ascii')+
            role+b'\0'*(48-len(role))+
            struct.pack("<9I", i, o, pi, po, m["K"], {"mcg":1,"mul1":2}[m["codebook"]],
                        len(activations), int(desc["bias"] is not None), len(packed)//4) + packed + floats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    record = {"schema":"epyc.exl3.gfx90a.fixture_transport.v1", "canonical_artifact_sha256":manifest["artifact_sha256"],
              "manifest":{"path":str(manifest_path.resolve()),"sha256":sha(manifest_path.read_bytes())},
              "fixture":{"path":str(fixture_path.resolve()),"sha256":sha(fixture_bytes)},
              "transport":{"path":str(output.resolve()),"sha256":sha(data)},
              "source":manifest["source"], "matrix":m["id"], "codebook":m["codebook"], "K":m["K"],
              "packed_tensor_sha256":sha(packed), "packed_tensor_changed":False,
              "provenance_reads":provenance,
              "tensor_reads":[{"path":str((manifest_path.parent/d["path"]).resolve()),"sha256":d["sha256"]}
                              for d in desc.values() if d is not None]}
    output.with_suffix(".json").write_bytes(canonical(record)+b"\n")
    return record

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--contract-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.contract_root.resolve()))
    print(json.dumps(export(a.manifest,a.output),sort_keys=True))

if __name__=="__main__":main()
