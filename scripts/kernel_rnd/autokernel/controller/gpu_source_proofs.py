"""Typed proof reducers for source-patch GPU discovery (no throughput authority)."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, os, re
from pathlib import Path
from typing import Any, Mapping, Sequence

SHA=re.compile(r"^[0-9a-f]{64}$")
class ProofError(RuntimeError): pass
def _hash(v: object)->str: return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _atomic(path:Path,v:Mapping[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("x") as f: json.dump(v,f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def load_receipt(path:Path, *, schema:str)->dict:
 """Read one immutable regular receipt and validate its native self-hash."""
 if path.is_symlink() or not path.is_file(): raise ProofError("receipt must be a regular non-symlink file")
 body=path.read_bytes(); digest=hashlib.sha256(body).hexdigest()
 try: value=json.loads(body)
 except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ProofError("receipt is not JSON") from exc
 if not isinstance(value,dict) or value.get("schema")!=schema: raise ProofError("receipt schema mismatch")
 native=value.get("receipt_sha256") or value.get("result_sha256")
 if not isinstance(native,str) or not SHA.fullmatch(native): raise ProofError("receipt native hash missing")
 return {"path":str(path.resolve()),"file_sha256":digest,"native_sha256":native,"body":value}
def require_result_file(path:Path, returned:Mapping[str,Any])->dict:
 """The runner's return object is advisory; re-read the durable bytes."""
 loaded=load_receipt(path,schema="epyc.autokernel.gpu_candidate_only_screen.v2")
 body=loaded["body"]
 if body.get("result_sha256")!=returned.get("result_sha256") or body.get("promotion_claim") is not False or body.get("non_promotable") is not True: raise ProofError("result file disagrees with runner or lacks discovery boundary")
 return loaded
@dataclass(frozen=True)
class GpuSourceProofBundle:
 manifest_sha256:str; candidate:BuildIdentity; anchor:BuildIdentity; workload_sha256:str; correctness:Mapping[str,Any]; attribution:Mapping[str,Any]; bundle_sha256:str
 def __post_init__(self):
  if not all(SHA.fullmatch(x) for x in (self.manifest_sha256,self.workload_sha256,self.bundle_sha256)): raise ProofError("bundle identity hash missing")
  expected=_hash({"manifest_sha256":self.manifest_sha256,"candidate":self.candidate.__dict__,"anchor":self.anchor.__dict__,"workload_sha256":self.workload_sha256,"correctness":dict(self.correctness),"attribution":dict(self.attribution)})
  if expected!=self.bundle_sha256: raise ProofError("bundle self-hash mismatch")
 def to_dict(self): return {"schema":"epyc.autokernel.gpu_source_proof_bundle.v1","manifest_sha256":self.manifest_sha256,"candidate":self.candidate.__dict__,"anchor":self.anchor.__dict__,"workload_sha256":self.workload_sha256,"correctness":dict(self.correctness),"attribution":dict(self.attribution),"authority":"nonpromotable_candidate_only_discovery","promotion_claim":False,"bundle_sha256":self.bundle_sha256}
@dataclass(frozen=True)
class BuildIdentity:
 source_sha256:str; binary_sha256:str; config_sha256:str
 def __post_init__(self):
  if not all(SHA.fullmatch(x) for x in (self.source_sha256,self.binary_sha256,self.config_sha256)): raise ProofError("build identity requires three SHA-256 values")
@dataclass(frozen=True)
class CorrectnessPlan:
 candidate:BuildIdentity; manifest_sha256:str; argv:tuple[str,...]; expected_summary:str; workload_sha256:str
 def __post_init__(self):
  if not self.argv or any(not isinstance(x,str) or not x for x in self.argv) or not all(SHA.fullmatch(x) for x in (self.manifest_sha256,self.workload_sha256)): raise ProofError("invalid correctness plan")
def load_and_validate_correctness(plan:CorrectnessPlan,path:Path)->dict:
 loaded=load_receipt(path,schema="epyc.autokernel.targeted_correctness_receipt.v1"); raw=loaded["body"]
 required={"authority":"nonpromotable_candidate_only_discovery","promotion_claim":False,"source_sha256":plan.candidate.source_sha256,"binary_sha256":plan.candidate.binary_sha256,"config_sha256":plan.candidate.config_sha256,"manifest_sha256":plan.manifest_sha256,"workload_sha256":plan.workload_sha256,"command_argv":list(plan.argv),"summary":plan.expected_summary,"returncode":0,"exact_case_ok":True}
 if any(raw.get(k)!=v for k,v in required.items()): raise ProofError("targeted correctness receipt does not bind its plan")
 residency=raw.get("residency"); claim=raw.get("claim")
 if not isinstance(residency,Mapping) or residency.get("overlapped") is not True or not residency.get("kfd_pids") or not isinstance(claim,Mapping) or claim.get("released") is not True: raise ProofError("correctness receipt lacks in-window residency or released claim")
 return loaded
@dataclass(frozen=True)
class DispatchExpectation:
 candidate_pattern:str; candidate_count:int; grid:int; workgroup:int; lds:int
 forbidden_pattern:str; anchor_pattern:str|None=None; anchor_count:int|None=None
 def __post_init__(self):
  if self.candidate_count<1 or self.grid<1 or self.workgroup<1 or self.lds<0 or ((self.anchor_pattern is None)!=(self.anchor_count is None)) or self.anchor_count is not None and self.anchor_count<1: raise ProofError("invalid exact dispatch expectation")
  for x in (self.candidate_pattern,self.forbidden_pattern,self.anchor_pattern):
   if x is not None: re.compile(x)
@dataclass(frozen=True)
class AttributionPlan:
 candidate:BuildIdentity; anchor:BuildIdentity; manifest_sha256:str; model_sha256:str; workload_sha256:str; expectation:DispatchExpectation; invariant_signatures:tuple[str,...]=()
 def __post_init__(self):
  if not all(SHA.fullmatch(x) for x in (self.manifest_sha256,self.model_sha256,self.workload_sha256)): raise ProofError("attribution plan hash missing")
def load_and_validate_attribution(plan:AttributionPlan,candidate_path:Path,anchor_path:Path|None=None)->dict:
 candidate=load_receipt(candidate_path,schema="epyc.autokernel.gpu_kernel_attribution.v1"); raw=candidate["body"]
 for key,value in {"authority":"nonpromotable_candidate_only_discovery","promotion_claim":False,"source_sha256":plan.candidate.source_sha256,"binary_sha256":plan.candidate.binary_sha256,"config_sha256":plan.candidate.config_sha256,"manifest_sha256":plan.manifest_sha256,"model_sha256":plan.model_sha256,"workload_sha256":plan.workload_sha256}.items():
  if raw.get(key)!=value: raise ProofError("candidate attribution receipt does not bind its plan")
 if raw.get("claim",{}).get("released") is not True or raw.get("residency",{}).get("overlapped") is not True or not raw.get("timestamps_sha256") or any(x not in raw.get("invariant_signatures",[]) for x in plan.invariant_signatures): raise ProofError("candidate attribution lacks claim/residency/timestamps/invariants")
 rows=_rows(raw); hits=_match(rows,plan.expectation.candidate_pattern)
 if len(hits)!=plan.expectation.candidate_count or _match(rows,plan.expectation.forbidden_pattern): raise ProofError("candidate exact/forbidden dispatch mismatch")
 if plan.expectation.anchor_pattern:
  if anchor_path is None: raise ProofError("inverse anchor receipt required")
  anchor=load_receipt(anchor_path,schema="epyc.autokernel.gpu_kernel_attribution.v1")["body"]
  if anchor.get("source_sha256")!=plan.anchor.source_sha256 or anchor.get("binary_sha256")!=plan.anchor.binary_sha256 or len(_match(_rows(anchor),plan.expectation.anchor_pattern))!=plan.expectation.anchor_count: raise ProofError("anchor inverse mismatch")
 return {"candidate":candidate,"anchor":None if anchor_path is None else str(anchor_path.resolve())}
def _rows(receipt:Mapping[str,Any])->Sequence[Mapping[str,Any]]:
 rows=receipt.get("dispatches")
 if not isinstance(rows,list) or not rows: raise ProofError("attribution receipt has no exact dispatch rows")
 return rows
def _match(rows,pattern): return [r for r in rows if isinstance(r,Mapping) and re.search(pattern,str(r.get("kernel","")))]
def verify_correctness(raw:Mapping[str,Any], *, source_sha256:str,binary_sha256:str,expected_summary:str)->dict:
 if not all(SHA.fullmatch(x) for x in (source_sha256,binary_sha256)): raise ProofError("bad build hash")
 if raw.get("source_sha256")!=source_sha256 or raw.get("binary_sha256")!=binary_sha256: raise ProofError("correctness identity mismatch")
 if raw.get("returncode")!=0 or raw.get("summary")!=expected_summary or raw.get("exact_case_ok") is not True: raise ProofError("targeted correctness failed")
 residency=raw.get("residency")
 if not isinstance(residency,Mapping) or residency.get("overlapped") is not True or not residency.get("kfd_pids"): raise ProofError("correctness residency did not overlap execution")
 body={"schema":"epyc.autokernel.targeted_correctness_receipt.v1","source_sha256":source_sha256,"binary_sha256":binary_sha256,"summary":expected_summary,"raw_sha256":_hash(raw),"promotion_claim":False}; body["receipt_sha256"]=_hash(body); return body
def verify_dispatch(candidate:Mapping[str,Any], *, candidate_identity:BuildIdentity,expectation:DispatchExpectation,anchor:Mapping[str,Any]|None=None,anchor_identity:BuildIdentity|None=None)->dict:
 for raw,identity in ((candidate,candidate_identity),(anchor,anchor_identity)):
  if raw is not None and (identity is None or raw.get("source_sha256")!=identity.source_sha256 or raw.get("binary_sha256")!=identity.binary_sha256 or raw.get("config_sha256")!=identity.config_sha256 or raw.get("residency",{}).get("overlapped") is not True): raise ProofError("attribution identity or residency mismatch")
 rows=_rows(candidate); hits=_match(rows,expectation.candidate_pattern)
 if len(hits)!=expectation.candidate_count or any((r.get("grid"),r.get("workgroup"),r.get("lds")) != (expectation.grid,expectation.workgroup,expectation.lds) for r in hits): raise ProofError("candidate dispatch geometry mismatch")
 if _match(rows,expectation.forbidden_pattern): raise ProofError("forbidden old dispatch remains")
 if expectation.anchor_pattern:
  if anchor is None or len(_match(_rows(anchor),expectation.anchor_pattern))!=expectation.anchor_count: raise ProofError("anchor inverse dispatch mismatch")
 body={"schema":"epyc.autokernel.gpu_kernel_attribution.v1","candidate_identity":candidate_identity.__dict__,"anchor_identity":None if anchor_identity is None else anchor_identity.__dict__,"candidate_sha256":_hash(candidate),"anchor_sha256":None if anchor is None else _hash(anchor),"expectation":expectation.__dict__,"authority":"nonpromotable_candidate_only_discovery","promotion_claim":False}; body["receipt_sha256"]=_hash(body); return body
def seal(path:Path,receipt:Mapping[str,Any])->str:
 value=dict(receipt); _atomic(path,value); return hashlib.sha256(path.read_bytes()).hexdigest()
