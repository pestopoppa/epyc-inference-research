import json,tempfile,unittest
from pathlib import Path
from . import gpu_source_proofs as P
H="a"*64
I=lambda x=H:P.BuildIdentity("commit-"+x[:8],x,x,x,x,x)
def good(): return {"source_sha256":H,"binary_sha256":H,"residency":{"overlapped":True,"kfd_pids":[1]},"dispatches":[{"kernel":"new","grid":2,"workgroup":64,"lds":0}]}
class Tests(unittest.TestCase):
 def test_path_correctness_validator_refuses_unbound_receipt(self):
  plan=P.CorrectnessPlan(I(),H,("test-backend-ops","--op","Q5"),"12/12",H)
  raw={"schema":"epyc.autokernel.targeted_correctness_receipt.v1","receipt_sha256":H}
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/"r.json"; path.write_text(json.dumps(raw))
   with self.assertRaises(P.ProofError): P.load_and_validate_correctness(plan,path)
 def test_correctness_refuses_identity_failure_and_absent_residency(self):
  raw={"source_sha256":H,"binary_sha256":H,"returncode":0,"summary":"12/12","exact_case_ok":True,"residency":{"overlapped":False,"kfd_pids":[]}}
  with self.assertRaises(P.ProofError): P.verify_correctness(raw,source_sha256=H,binary_sha256=H,expected_summary="12/12")
  raw["residency"]={"overlapped":True,"kfd_pids":[1]}; raw["binary_sha256"]="b"*64
  with self.assertRaises(P.ProofError): P.verify_correctness(raw,source_sha256=H,binary_sha256=H,expected_summary="12/12")
 def test_dispatch_refuses_geometry_forbidden_and_inverse_mismatch(self):
  e=P.DispatchExpectation("new",1,2,64,0,"old","old",1); c=good(); a=good(); a["source_sha256"]="b"*64; a["binary_sha256"]="b"*64; a["config_sha256"]="b"*64; c["config_sha256"]=H; a["dispatches"]=[{"kernel":"old","grid":1,"workgroup":64,"lds":0}]
  self.assertEqual(P.verify_dispatch(c,candidate_identity=I(),expectation=e,anchor=a,anchor_identity=I("b"*64))["schema"],"epyc.autokernel.gpu_kernel_attribution.v1")
  c["dispatches"][0]["grid"]=3
  with self.assertRaises(P.ProofError): P.verify_dispatch(c,candidate_identity=I(),expectation=e,anchor=a,anchor_identity=I("b"*64))
