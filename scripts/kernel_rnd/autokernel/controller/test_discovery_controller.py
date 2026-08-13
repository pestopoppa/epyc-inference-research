"""No-hardware replay tests for the typed discovery state machine."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from . import discovery_controller as D

H="a"*64
class Manifest: pass
class FakePlanner:
 def __init__(self): self.calls=[]
 def attest(self): return {**D.SOL,"runtime":{"real":"attested"}}
 def plan(self,*,context,workspace):
  self.calls.append(context); return D.PlannedCandidate("akh-test-"+str(len(self.calls)),"one-wave reduces cross-wave LDS","no speed improvement invalidates it",{"backend":"gpu","phase":"decode","mechanism":"one_wave"},{"id":"p"+str(len(self.calls))},Manifest(),H)
class FakeCritic:
 def __init__(self,decisions): self.decisions=iter(decisions)
 def attest(self): return {**D.TERRA,"runtime":{"real":"attested"}}
 def review(self,*args,**kw): return D.Critique(next(self.decisions),"bounded gate")
class Lease:
 def admit(self,item): return {"admitted":True,"mode":"allowed_discovery_noise"}
class FakeScreen:
 def __init__(self,values): self.values=iter(values); self.calls=0
 def screen(self,*args):
  self.calls+=1; return D.SealedScreen("receipt",H[:-1]+str(self.calls),next(self.values),"candidate",H,H,H)

class Tests(unittest.TestCase):
 def cfg(self,root,n=2): return D.ControllerConfig(root/"out",n)
 def test_exact_two_codex_no_claude(self):
  self.assertEqual(D.sealed_roster()["claude_members"],0); self.assertEqual([x["model"] for x in D.sealed_roster()["members"]],["gpt-5.6-sol","gpt-5.6-terra"])
 def test_veto_blocks_compute_and_feedback_is_next_context(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   p=FakePlanner(); s=FakeScreen([.04,.03]); r=D.run_controller(self.cfg(Path(t),3),planner=p,critic=FakeCritic(["reject","accept","accept"]),screener=s,lease=Lease())
   self.assertEqual([x["status"] for x in r["iterations"]],["critic_reject","candidate","candidate"]); self.assertEqual(s.calls,2); self.assertIn(H[:-1]+"1",p.calls[2]["prior_results"])
 def test_threshold_is_only_idempotent_operator_nomination(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); r=D.run_controller(self.cfg(root,1),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.04]),lease=Lease())
   row=(root/"out"/"promotion-queue.jsonl").read_text(); self.assertIn('"promotion_claim": false',row); self.assertIn('"operator_decision_required": true',row); self.assertEqual(len(r["iterations"]),1)
 def test_refused_screen_stops_and_resume_does_not_repeat(self):
  class Bad(FakeScreen):
   def screen(self,*args): raise RuntimeError("build failed")
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); p=FakePlanner(); r=D.run_controller(self.cfg(root,1),planner=p,critic=FakeCritic(["accept"]),screener=Bad([]),lease=Lease()); again=D.run_controller(self.cfg(root,1),planner=p,critic=FakeCritic(["accept"]),screener=Bad([]),lease=Lease())
   self.assertEqual(r["iterations"][0]["status"],"screen_refused"); self.assertEqual(again,r); self.assertEqual(len(p.calls),1)
 def test_planner_result_field_is_impossible(self):
  with patch.object(D.source_candidate,"SourcePatchManifest",Manifest):
   with self.assertRaisesRegex(D.DiscoveryControllerError,"result"):
    D.PlannedCandidate("akh-a","s","f",{}, {"effect_pct":1},Manifest(),H)
 def test_gpu_source_gate_order_before_runner(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D.gpu_discovery,"run") as run:
   root=Path(t); anchor=root/"anchor"; candidate=root/"candidate"; anchor.mkdir(); candidate.mkdir(); events=[]
   source_file=root/"source.json"; dispatch_file=root/"dispatch.json"; source_file.write_text("source"); dispatch_file.write_text("dispatch")
   build=D.GpuSourceBuild(anchor,candidate,D.gpu_source_proofs.BuildIdentity("commit-a",H,H,H,H,H),D.gpu_source_proofs.BuildIdentity("commit-b","b"*64,"b"*64,"b"*64,"b"*64,"b"*64))
   item=D.PlannedCandidate("akh-a","statement","falsifier",{}, {"id":"p"},Manifest(),H)
   args=argparse.Namespace(factor="source_patch",anchor_build=str(anchor),candidate_build=str(candidate),output_dir=str(root/"screen"))
   (root/"screen").mkdir()
   raw={"schema":"epyc.autokernel.gpu_candidate_only_screen.v2","non_promotable":True,"promotion_claim":False,"hip_residency_proved":True,"median_relative":.02,"baseline_sha256":H}; raw["result_sha256"]=D.gpu_source_proofs._hash(raw)
   (root/"screen"/"result.json").write_text(json.dumps(raw))
   run.side_effect=lambda _args: events.append("runner") or raw
   source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest(); dispatch_hash=hashlib.sha256(dispatch_file.read_bytes()).hexdigest()
   identity=D.gpu_source_proofs.BuildIdentity("commit-a",H,H,H,H,H); material={"manifest_sha256":H,"candidate":identity,"anchor":identity,"workload_sha256":H,"correctness":{"file_sha256":source_hash,"native_sha256":H},"attribution":{"file_sha256":dispatch_hash,"native_sha256":H}}
   hashed={**material,"candidate":identity.__dict__,"anchor":identity.__dict__}; bundle=D.gpu_source_proofs.GpuSourceProofBundle(**material,bundle_sha256=D.gpu_source_proofs._hash(hashed))
   screen=D.GpuSourceScreener(build_source=lambda *_: events.append("build") or build,proof_bundle=lambda *_: events.extend(["source","dispatch"]) or bundle,args_factory=lambda *_:args)
   with patch.object(D.autokernel_progression,"_gpu_screen",return_value={"stage":"candidate"}):
    got=screen.screen(item,object(),{})
   self.assertEqual(events,["build","source","dispatch","runner"]); self.assertEqual(got.dispatch_proof_sha256,dispatch_hash)
 def test_lease_wait_is_durable_without_spending_iteration(self):
  class Wait:
   def admit(self,item): return {"admitted":False,"reason":"CPU window busy"}
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   r=D.run_controller(self.cfg(Path(t),1),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.1]),lease=Wait())
   self.assertEqual(r["next"],1); self.assertEqual(r["pending"]["status"],"waiting_resource"); self.assertFalse(r["complete"])
