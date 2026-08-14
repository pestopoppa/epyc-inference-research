"""No-hardware replay tests for the typed discovery state machine."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from . import discovery_controller as D

H="a"*64
RUNTIME={"kind":"docker_workspace_bind_only","docker_path":"/docker","docker_sha256":H,"image_id":"image","codex_native_sha256":H,"code_mode_host_sha256":H,"ca_certificate_sha256":H,"writable_host_binds":["/workspace"],"host_network_mode":"docker_bridge"}
class Manifest:
 campaign_id="ak-test"; proposal_id="akp-test"; candidate_id="akc-test"; source_tree="llama.cpp"; production_base_commit="0"*40; instrument_commit="0"*40; change_class="fusion"; declared_files=("ggml/src/ggml.c",); declared_symbols={"ggml/src/ggml.c":("<file-scope>",)}; mechanism_id="test"; patch_sha256="0"*64; patch_bundle_sha256=H; patch_bytes=b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n@@ -1 +1 @@\n-x\n+y\n"
 def __init__(self, **values):
  for key, value in values.items(): setattr(self,key,value)
class FakePlanner:
 def __init__(self): self.calls=[]
 def attest(self): return {**D.SOL,"runtime":RUNTIME}
 def plan(self,*,context,workspace):
  self.calls.append(context); return D.PlannedCandidate("akh-test-"+str(len(self.calls)),"one-wave reduces cross-wave LDS","no speed improvement invalidates it",{"backend":"gpu","phase":"decode","mechanism":"one_wave"},{"id":"p"+str(len(self.calls))},Manifest(),H)
class FakeCritic:
 def __init__(self,decisions): self.decisions=iter(decisions)
 def attest(self): return {**D.TERRA,"runtime":RUNTIME}
 def review(self,*args,**kw): return D.Critique(next(self.decisions),"bounded gate")
class Lease:
 def admit(self,item): return {"admitted":True,"mode":"allowed_discovery_noise"}
class FakeScreen:
 def __init__(self,values): self.values=iter(values); self.calls=0
 def screen(self,*args):
  self.calls+=1; return D.SealedScreen("receipt",H[:-1]+str(self.calls),next(self.values),"candidate",H,H,H)
 def reconcile(self,inflight): return D.Recovery("safe_to_start")

class Tests(unittest.TestCase):
 def test_bounded_dispatch_refuses_meta_and_out_of_range_literals(self):
  valid=D.BoundedDispatchExpectation("kernel_6",2,128,64,0)
  self.assertEqual(valid.kernel_name,"kernel_6")
  for name in ("kernel.*", "kernel[0]", "kernel name"):
   with self.subTest(name=name), self.assertRaises(D.DiscoveryControllerError):
    D.BoundedDispatchExpectation(name,1,1,1,0)
  with self.assertRaises(D.DiscoveryControllerError): D.BoundedDispatchExpectation("kernel",1,1,4097,0)
 def cfg(self,root,n=2): return D.ControllerConfig(root/"out",n)
 def test_exact_two_codex_no_claude(self):
  self.assertEqual(D.sealed_roster()["claude_members"],0); self.assertEqual([x["model"] for x in D.sealed_roster()["members"]],["gpt-5.6-sol","gpt-5.6-terra"])
 def test_sealed_planner_context_is_visible_and_resume_mismatch_refuses(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); context={"hotspots":[{"symbol":"q5_hot"}],"context_sha256":H}
   cfg=D.ControllerConfig(root/"out",1,dry_run=True,planner_context=context,planner_context_sha256=H)
   p=FakePlanner(); D.run_controller(cfg,planner=p,critic=FakeCritic(["accept"]),screener=FakeScreen([.01]),lease=Lease())
   self.assertEqual(p.calls[0]["planner_context"]["hotspots"][0]["symbol"],"q5_hot")
   with self.assertRaises(D.DiscoveryControllerError):
    D.run_controller(D.ControllerConfig(root/"out",1,dry_run=True,planner_context=context,planner_context_sha256=H[:-1]+"b"),planner=p,critic=FakeCritic(["accept"]),screener=FakeScreen([.01]),lease=Lease())
 def test_sealed_deployment_identity_is_durable_resume_authority(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t)
   context={"context_sha256":H}
   def make_config(identity): return D.ControllerConfig(root/"out",1,dry_run=True,planner_context=context,planner_context_sha256=H,
       production_base_commit="0"*40,instrument_commit="1"*40,experiment_template_registry_sha256="2"*64,
       admission_corpus_sha256="3"*64,admission_corpus_version="test-v1",deployment_identity_sha256=identity)
   cfg=make_config(H)
   D.run_controller(cfg,planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.01]),lease=Lease())
   with self.assertRaises(D.DiscoveryControllerError):
    D.run_controller(make_config(H[:-1]+"b"),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.01]),lease=Lease())
 def test_veto_blocks_compute_and_feedback_is_next_context(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   p=FakePlanner(); s=FakeScreen([.04,.03,.04]); r=D.run_controller(self.cfg(Path(t),4),planner=p,critic=FakeCritic(["reject","accept","accept"]),screener=s,lease=Lease())
   self.assertEqual([x["status"] for x in r["iterations"]],["critic_reject","candidate","top_k_replicated_candidate","top_k_replicated_candidate"]); self.assertEqual(s.calls,3); self.assertEqual(len(p.calls),3); self.assertEqual([row["result_sha256"] for row in p.calls[2]["prior_results"]],[H[:-1]+"1",H[:-1]+"2"])
 def test_single_positive_screen_never_nominates(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); r=D.run_controller(self.cfg(root,1),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.04]),lease=Lease())
   self.assertEqual(r["iterations"][0]["status"],"candidate")
   self.assertFalse((root/"out"/"promotion-queue.jsonl").exists())
 def test_replicated_positive_threshold_is_idempotent_operator_nomination(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); r=D.run_controller(self.cfg(root,2),planner=FakePlanner(),critic=FakeCritic(["accept","accept"]),screener=FakeScreen([.04,.03]),lease=Lease())
   row=(root/"out"/"promotion-queue.jsonl").read_text(); self.assertIn('"promotion_claim": false',row); self.assertIn('"operator_decision_required": true',row); self.assertEqual(len(row.splitlines()),1); self.assertEqual(r["iterations"][-1]["status"],"top_k_replicated_candidate")
 def test_nomination_uses_pooled_exact_series_effect_not_s2_order(self):
  for effects, expected in (((.001,.04),True),((.04,.001),True),((.001,.002),False)):
   with self.subTest(effects=effects), tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
    root=Path(t); result=D.run_controller(D.ControllerConfig(root/"out",2,nomination_threshold=.02),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen(effects),lease=Lease())
    queue=root/"out"/"promotion-queue.jsonl"; self.assertEqual(queue.exists(),expected)
    if expected: self.assertAlmostEqual(result["iterations"][-1]["series_effect_fraction"],sum(effects)/2)
 def test_replication_spread_is_inconclusive_not_a_nomination(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); r=D.run_controller(self.cfg(root,2),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.12,.001]),lease=Lease())
   self.assertEqual(r["iterations"][-1]["status"],"inconclusive"); self.assertFalse((root/"out"/"promotion-queue.jsonl").exists())
 def test_refused_screen_stops_and_resume_does_not_repeat(self):
  class Bad(FakeScreen):
   def screen(self,*args): raise D.PrecomputeScreenRefusal("build failed before operation start")
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
   candidate_identity=build.candidate_identity; anchor_identity=build.anchor_identity; material={"manifest_sha256":H,"candidate":candidate_identity,"anchor":anchor_identity,"workload_sha256":H,"correctness":{"file_sha256":source_hash,"native_sha256":H},"attribution":{"file_sha256":dispatch_hash,"native_sha256":H}}
   hashed={**material,"candidate":candidate_identity.__dict__,"anchor":anchor_identity.__dict__}; bundle=D.gpu_source_proofs.GpuSourceProofBundle(**material,bundle_sha256=D.gpu_source_proofs._hash(hashed))
   screen=D.GpuSourceScreener(build_source=lambda *_: events.append("build") or build,proof_bundle=lambda *_: events.extend(["source","dispatch"]) or bundle,args_factory=lambda *_:args)
   with patch.object(D.autokernel_progression,"_gpu_screen",return_value={"stage":"candidate"}):
    got=screen.screen(item,object(),{})
   self.assertEqual(events,["build","source","dispatch","runner"]); self.assertEqual(got.dispatch_proof_sha256,dispatch_hash)
 def test_lease_wait_is_durable_without_spending_iteration(self):
  class Wait:
   def admit(self,item): return {"admitted":False,"reason":"CPU window busy"}
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   r=D.run_controller(self.cfg(Path(t),1),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.1]),lease=Wait())
   self.assertEqual(r["next"],1); self.assertEqual(r["pending"]["row"]["status"],"waiting_resource"); self.assertFalse(r["complete"])
 def test_pending_roundtrip_uses_real_manifest_and_skips_planner_critic(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); patch_bytes=b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n@@ -1 +1 @@\n-x\n+y\n"; manifest=D.source_candidate.SourcePatchManifest(campaign_id="ak-test",proposal_id="akp-test",candidate_id="akc-test",source_tree="llama.cpp",production_base_commit="0"*40,instrument_commit="0"*40,change_class="fusion",declared_files=("ggml/src/ggml.c",),declared_symbols={"ggml/src/ggml.c":("<file-scope>",)},mechanism_id="test",patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),patch_bytes=patch_bytes)
   item=D.PlannedCandidate("akh-real","real statement","speed does not improve",{}, {"id":"p"},manifest,manifest.patch_bundle_sha256)
   restored=D._restore_pending({"candidate":D._pending_item(item)})
   self.assertEqual(restored.source_manifest.patch_bytes,manifest.patch_bytes); self.assertEqual(restored.source_manifest.patch_bundle_sha256,manifest.patch_bundle_sha256)
 def test_discovery_negative_records_attempt_without_resolving_hypothesis(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); p=FakePlanner(); screen=FakeScreen([-.01]); screen.values=iter([-.01])
   D.run_controller(self.cfg(root,1),planner=p,critic=FakeCritic(["accept"]),screener=screen,lease=Lease())
   tracker=D._tracker(D.DurableState(root/"out")); self.assertTrue(tracker.state()["akh-test-1"].is_open)
 def test_pooled_classifier_requires_replication_and_detects_sign_conflict(self):
  self.assertEqual(D.classify_screen_series([.01]),"candidate")
  self.assertEqual(D.classify_screen_series([.01,.02]),"top_k_replicated_candidate")
  self.assertEqual(D.classify_screen_series([.01,-.01]),"inconclusive")
  self.assertEqual(D.classify_screen_series([.01,.006],component_pooled_effects=[.02]),"replicated_but_subadditive")
 def test_non_finite_or_boolean_effect_and_threshold_are_refused(self):
  for value in (True, float("nan"), float("inf"), float("-inf")):
   with self.subTest(value=value):
    with self.assertRaises(D.DiscoveryControllerError): D.SealedScreen("receipt",H,value,"candidate",H,H,H)
    with self.assertRaises(D.DiscoveryControllerError): D.ControllerConfig(Path("/tmp/controller"),1,nomination_threshold=value)
 def test_dry_run_authorizes_without_lease_or_screen(self):
  class ExplosiveLease:
   def admit(self,item): raise AssertionError("dry run may not ask for a compute lease")
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   p=FakePlanner(); s=FakeScreen([.1]); r=D.run_controller(D.ControllerConfig(Path(t)/"out",1,dry_run=True),planner=p,critic=FakeCritic(["accept"]),screener=s,lease=ExplosiveLease())
   self.assertEqual(r["iterations"][0]["status"],"dry_run_authorized"); self.assertEqual((len(p.calls),s.calls),(1,0))
 def test_canonical_projection_uses_evidence_root_not_state_root(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection") as projection:
   root=Path(t); evidence=root/"canonical-evidence"; D.run_controller(D.ControllerConfig(root/"state",1,evidence_root=evidence),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.1]),lease=Lease())
   projection.assert_called_once_with(evidence)
 def test_adapter_bundle_requires_exact_four_seams(self):
  parts=D.build_controller_adapters(planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.1]),lease=Lease())
  self.assertEqual(set(parts),{"planner","critic","screener","lease"})
