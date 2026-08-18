"""No-hardware replay tests for the typed discovery state machine."""
from __future__ import annotations
import argparse, base64, hashlib, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .. import hypothesis_portfolio
from . import discovery_controller as D

H="a"*64
RUNTIME={"kind":"docker_workspace_bind_only","docker_path":"/docker","docker_sha256":H,"image_id":"image","codex_native_sha256":H,"code_mode_host_sha256":H,"ca_certificate_sha256":H,"writable_host_binds":["/workspace"],"host_network_mode":"docker_bridge"}
CLAUDE_RUNTIME={"kind":"claude_cli_structured_critic","provider":"claude","model":"claude-fable-5","effort":"high","wrapper_path":"/sealed/claude","wrapper_sha256":H,"argv_policy_sha256":H,"auth_staging_policy":"ephemeral_0600_copy_atomic_oauth_rotation_sync_no_secret_receipt"}
class Manifest:
 campaign_id="ak-test"; proposal_id="akp-test"; candidate_id="akc-test"; source_tree="llama.cpp"; production_base_commit="0"*40; instrument_commit="0"*40; change_class="fusion"; declared_files=("ggml/src/ggml.c",); declared_symbols={"ggml/src/ggml.c":("<file-scope>",)}; mechanism_id="test"; patch_sha256="0"*64; patch_bytes=b"diff --git a/ggml/src/ggml.c b/ggml/src/ggml.c\n--- a/ggml/src/ggml.c\n+++ b/ggml/src/ggml.c\n@@ -1 +1 @@\n-x\n+y\n"
 patch_text=patch_bytes.decode("utf-8")
 def __init__(self, **values):
  for key, value in values.items(): setattr(self,key,value)
 @property
 def patch_bundle_sha256(self):
  raw=json.dumps({"schema":D.source_candidate.SCHEMA_SOURCE_PATCH,
      "campaign_id":self.campaign_id,"proposal_id":self.proposal_id,
      "candidate_id":self.candidate_id,"source_tree":self.source_tree,
      "production_base_commit":self.production_base_commit,
      "instrument_commit":self.instrument_commit,"change_class":self.change_class,
      "declared_files":list(self.declared_files),
      "declared_symbols":{key:list(value) for key,value in self.declared_symbols.items()},
      "mechanism_id":self.mechanism_id,"patch_sha256":self.patch_sha256,
      "patch_encoding":"base64","patch_base64":base64.b64encode(self.patch_bytes).decode("ascii")},
      sort_keys=True,separators=(",",":")).encode()
  return hashlib.sha256(raw).hexdigest()
class FakePlanner:
 def __init__(self): self.calls=[]
 def attest(self): return {**D.SOL,"runtime":RUNTIME}
 def plan(self,*,context,workspace):
  self.calls.append(context); manifest=Manifest(); return D.PlannedCandidate("akh-test-"+str(len(self.calls)),"one-wave reduces cross-wave LDS","no speed improvement invalidates it",{"backend":"gpu","phase":"decode","mechanism":"one_wave"},{"id":"p"+str(len(self.calls))},manifest,manifest.patch_bundle_sha256)
class FakeCritic:
 def __init__(self,decisions): self.decisions=iter(decisions)
 def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
 def review(self,*args,**kw): return D.Critique(next(self.decisions),"bounded gate")
class Lease:
 def admit(self,item,*,operation_key): return {"admitted":True,"mode":"allowed_discovery_noise","operation_key":operation_key}
 def resume(self,item,permit): return self.admit(item,operation_key=permit["operation_key"])
class FakeScreen:
 def __init__(self,values): self.values=iter(values); self.calls=0
 def screen(self,*args):
  self.calls+=1; return D.SealedScreen("receipt",H[:-1]+str(self.calls),next(self.values),"candidate",H,H,H)
 def reconcile(self,inflight): return D.Recovery("safe_to_start")

class Tests(unittest.TestCase):
 def portfolio_record(self, *, hypothesis_id="akh-portfolio-q8", rank=2,
                      eligible=True, budget=2):
  return {"hypothesis_id":hypothesis_id,"statement":"reduce exact q8 activation overhead",
          "primary_falsifier":"replicated decode gain is below the sealed floor",
          "falsifiers":["replicated decode gain is below the sealed floor"],
          "regime":{"frame_id":"qwen05","architecture":"gfx90a","phase":"decode"},
          "target":{"source_files":["ggml/src/ggml-cuda/quantize.cu"],
                    "source_symbols":["quantize_q8_1"],
                    "template_intent":"cuda-quantize-q8-v1"},
          "mechanism":{"fingerprint_sha256":H,
                       "facets":{"change_class":"arithmetic"}},
          "priority":{"rank":rank},
          "current_bundle_eligibility":{"eligible":eligible,
               "template_ids":["cuda-quantize-q8-v1"] if eligible else []},
          "decision_policy":{"frame_id":"qwen05","continuation_floor_pct":0.4,
               "nomination_floor_pct":0.8,"required_replications":2,
               "sign_policy":"all_positive","conflict_policy":"retain_inconclusive",
               "max_distinct_candidates":budget,"terminal_rule":"retire",
               "metric":"decode_tokens_per_s","effect_unit":"relative_percent",
               "min_replication_effect_pct":0.0,"max_replication_spread_pct":1.0}}
 def portfolio_config(self, root, records):
  portfolio=hypothesis_portfolio.Portfolio(
      {"hypotheses":records,"frames":[],"do_not_repeat":[]},"f"*64)
  return D.ControllerConfig(root,3,dry_run=True,
      planner_context={"portfolio_dispatch_authority": {
          row["hypothesis_id"]: [{"route_id":"cuda-quantize-q8-v1.anchor.0",
                                  "kernel_name":"quantize_q8_1", "calls":18705,
                                  "grid":1024,"workgroup":256,"lds_bytes":0}]
          for row in records if row["current_bundle_eligibility"]["eligible"]}},
      planner_context_sha256="e"*64,
      hypothesis_portfolio=portfolio,
      hypothesis_portfolio_sha256="f"*64)
 def portfolio_candidate(self, binding, *, hypothesis_id=None, mechanism_id=None,
                         regime=None):
  path=binding["target_file"]; symbols=tuple(binding["target_symbols"])
  patch=(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
         f"@@ -1 +1 @@ {symbols[0]}()\n-x\n+y\n").encode()
  manifest=D.source_candidate.SourcePatchManifest(
      campaign_id="ak-test",proposal_id="akp-test",candidate_id="akc-test",
      source_tree="llama.cpp",production_base_commit="0"*40,
      instrument_commit="1"*40,change_class=binding["change_class"],declared_files=(path,),
      declared_symbols={path:symbols},
      mechanism_id=mechanism_id or binding["mechanism_id"],
      patch_sha256=hashlib.sha256(patch).hexdigest(),patch_bytes=patch)
  intent=D.GpuSourceExperimentIntent(binding["template_id"],"gpu_decode",symbols[0],
      "backend-ops-hip-v1","decode-tg128-rocprof-v1",
      tuple(D.BoundedDispatchExpectation(**row)
            for row in binding["expected_dispatch"]))
  return D.PlannedCandidate(hypothesis_id or binding["hypothesis_id"],
      binding["statement"],binding["falsifier"],regime or binding["regime"],
      {"proposal_id":"akp-test","change_class":binding["change_class"]},
      manifest,manifest.patch_bundle_sha256,intent)
 def test_portfolio_scheduler_owns_rank_budget_and_exact_candidate_binding(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[
       self.portfolio_record(hypothesis_id="akh-lower",rank=2,budget=1),
       self.portfolio_record(hypothesis_id="akh-first",rank=1,budget=1)])
   state={"iterations":[]}
   first=D._select_portfolio_binding(state,config)
   self.assertEqual(first["hypothesis_id"],"akh-first")
   D._validate_portfolio_candidate(self.portfolio_candidate(first),first,
                                    config.hypothesis_portfolio)
   state["iterations"].append({"portfolio_hypothesis_id":"akh-first",
                               "source_manifest_sha256":"1"*64})
   second=D._select_portfolio_binding(state,config)
   self.assertEqual(second["hypothesis_id"],"akh-lower")
   with self.assertRaisesRegex(D.DiscoveryControllerError,"controller-owned"):
    D._validate_portfolio_candidate(
        self.portfolio_candidate(second,hypothesis_id="akh-invented"),second,
        config.hypothesis_portfolio)
 def test_portfolio_dispatch_rows_refuse_missing_extra_and_duplicate_precompute(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record()])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   candidate=self.portfolio_candidate(binding)
   base=candidate.experiment_intent
   for rows in ((), base.expected_dispatch + (
           D.BoundedDispatchExpectation("cuda-quantize-q8-v1.anchor.1",
                                        "quantize_q8_1_extra",1,64,64,0),)):
    altered=D.GpuSourceExperimentIntent(
        base.template_id,base.target_surface,base.target_symbol,
        base.correctness_id,base.dispatch_id,rows or (
            D.BoundedDispatchExpectation("cuda-quantize-q8-v1.anchor.1",
                                         "quantize_q8_1_missing",1,64,64,0),))
    bad=D.PlannedCandidate(candidate.hypothesis_id,candidate.statement,
        candidate.falsifier,candidate.regime,candidate.proposal,
        candidate.source_manifest,candidate.source_manifest_sha256,altered)
    with self.assertRaisesRegex(D.DiscoveryControllerError,"portfolio assignment"):
     D._validate_portfolio_candidate(bad,binding,config.hypothesis_portfolio)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"distinct"):
    D.GpuSourceExperimentIntent(
        base.template_id,base.target_surface,base.target_symbol,
        base.correctness_id,base.dispatch_id,
        (base.expected_dispatch[0],base.expected_dispatch[0]))
 def test_portfolio_exact_dnr_match_refuses_before_selected_binding(self):
  with tempfile.TemporaryDirectory() as t:
   record=self.portfolio_record(); config=self.portfolio_config(Path(t),[record])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   dnr={"mechanism":{"fingerprint_sha256":"e"*64},"regime":{"phase":"decode"}}
   portfolio=hypothesis_portfolio.Portfolio(
       {**config.hypothesis_portfolio.body,"do_not_repeat":[dnr]},"f"*64)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"DNR"):
    D._validate_portfolio_candidate(
        self.portfolio_candidate(binding,mechanism_id="e"*64,
                                 regime={"phase":"decode"}),binding,portfolio)
 def test_portfolio_decision_floor_retains_one_percent_and_refuses_weak_conflicted_nonfinite(self):
  policy=self.portfolio_record()["decision_policy"]
  self.assertEqual(D.classify_screen_series(
      [.01,.011],continuation_floor=.004,nomination_floor=.008,
      required_replications=2),"top_k_replicated_candidate")
  self.assertEqual(D.classify_screen_series(
      [.003],continuation_floor=.004,nomination_floor=.008,
      required_replications=2),"screened_out")
  self.assertEqual(D.classify_screen_series(
      [.01,-.001],continuation_floor=.004,nomination_floor=.008,
      required_replications=2),"inconclusive")
  self.assertEqual(D.classify_screen_series(
      [.01,.012],continuation_floor=.004,nomination_floor=.008,
      min_replication_effect=.005,max_replication_spread=.001,
      required_replications=2),"inconclusive")
  self.assertEqual(D.classify_screen_series(
      [.01,.003],continuation_floor=.004,nomination_floor=.008,
      min_replication_effect=.005,max_replication_spread=.02,
      required_replications=2),"screened_out")
  with self.assertRaises(D.DiscoveryControllerError):
   D.classify_screen_series([float("nan"),.01],continuation_floor=.004,
                            nomination_floor=.008,required_replications=2)
  self.assertAlmostEqual(D._decision_floor(policy,"nomination_floor_pct",.03),.008)
 def test_empty_eligible_portfolio_stops_without_planner_review_or_compute(self):
  class NeverPlanner(FakePlanner):
   def plan(self,**_kwargs): raise AssertionError("empty portfolio reached planner")
  class NeverCritic(FakeCritic):
   def review(self,*_args,**_kwargs): raise AssertionError("empty portfolio reached critic")
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record(eligible=False)])
   result=D.run_controller(config,planner=NeverPlanner(),critic=NeverCritic([]),
                           screener=FakeScreen([.01]),lease=Lease())
  self.assertTrue(result["complete"])
  self.assertEqual(result["terminal_reason"],"portfolio_exhausted")
  self.assertEqual(result["iterations"],[])
 def test_off_assignment_plan_refuses_before_fable_lease_or_screen(self):
  class Planner(FakePlanner):
   def __init__(self,candidate): super().__init__(); self.candidate=candidate
   def plan(self,**_kwargs): self.calls.append(_kwargs); return self.candidate
  class Critic(FakeCritic):
   def __init__(self): super().__init__(["accept"]); self.calls=0
   def review(self,*_args,**_kwargs): self.calls+=1; return D.Critique("accept","bad")
  class Never:
   calls=0
   def __getattr__(self,_name):
    def called(*_args,**_kwargs): self.calls+=1; raise AssertionError("compute reached")
    return called
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record()])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   planner=Planner(self.portfolio_candidate(binding,hypothesis_id="akh-off-assignment"))
   critic=Critic(); lease=Never(); screen=Never()
   result=D.run_controller(config,planner=planner,critic=critic,
                           screener=screen,lease=lease)
  self.assertEqual(critic.calls,0)
  self.assertEqual(lease.calls,0); self.assertEqual(screen.calls,0)
  self.assertTrue(result["complete"])
  self.assertEqual(result["iterations"][0]["status"],"portfolio_refused")
 def source_package(self):
  content=b"void reviewed_kernel() {}\n"; digest=hashlib.sha256(content).hexdigest()
  body={"schema":"epyc.autokernel.reviewed_source_package.v1","instrument_commit":"1"*40,
        "files":[{"relative_path":"ggml/src/ggml-cuda/reviewed.cu","sha256":digest,
                  "workspace_path":"reviewed-source/ggml/src/ggml-cuda/reviewed.cu"}]}
  return D.ReviewedSourcePackage("1"*40,
      (D.ReviewedSourceFile("ggml/src/ggml-cuda/reviewed.cu",digest,content),),D._sha(body))
 def test_reviewed_source_package_is_exact_idempotent_and_hardlink_safe(self):
  package=self.source_package()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); manifest=package.materialize(root)
   self.assertEqual(manifest["package_sha256"],package.package_sha256)
   source=root/"reviewed-source/ggml/src/ggml-cuda/reviewed.cu"
   self.assertEqual(source.read_bytes(),package.files[0].content)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"already exists"):
    package.materialize(root)
   source.chmod(0o600); alias=root/"source-hardlink"; alias.hardlink_to(source)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"source bytes changed"):
    package.revalidate_materialized(root)
  self.assertFalse(root.exists())
 def test_planner_revalidates_reviewed_source_after_actor_before_loading_plan(self):
  package=self.source_package()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},reviewed_sources=package)
   assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
   def actor(**kwargs):
    target=root/"reviewed-source/ggml/src/ggml-cuda/reviewed.cu"
    target.chmod(0o600); target.write_bytes(b"mutated\n")
    return SimpleNamespace(returncode=0,stderr="")
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=actor), \
        patch.object(D,"_load_plan",side_effect=AssertionError("mutated source reached loader")):
    with self.assertRaisesRegex(D.DiscoveryControllerError,"source bytes changed"):
     planner.plan(context={"authoring_assignment":assignment.to_dict(),
                           "planner_context":{"reviewed_source_package_sha256":
                                              package.package_sha256}},workspace=root)
 def test_planner_prompt_has_exact_schemas_source_paths_and_structural_example(self):
  package=self.source_package(); captured={}
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},reviewed_sources=package)
   assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
   def actor(**kwargs):
    captured.update(json.loads(kwargs["prompt"]))
    source=root/"reviewed-source/ggml/src/ggml-cuda/reviewed.cu"
    manifest=root/"reviewed-source/source-package.json"
    self.assertEqual(source.read_bytes(),package.files[0].content)
    self.assertEqual(json.loads(manifest.read_text())["package_sha256"],package.package_sha256)
    return SimpleNamespace(returncode=1,stderr="stop")
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=actor):
    with self.assertRaisesRegex(D.DiscoveryControllerError,"Sol actor failed"):
     planner.plan(context={"authoring_assignment":assignment.to_dict(),
                           "planner_context":{"reviewed_source_package_sha256":
                                              package.package_sha256}},workspace=root)
   self.assertEqual(captured["reviewed_source_package"]["package_sha256"],package.package_sha256)
   self.assertEqual(captured["authoring_contract"]["expected_dispatch"],"array of 1..8 exact objects")
   estimate_rule=captured["authoring_contract"]["proposal_schema"]["estimated_diff_size_rule"]
   self.assertIn("added lines plus removed lines",estimate_rule)
   hunk_rule=captured["authoring_contract"]["source_manifest_schema"]["unified_diff_hunk_rule"]
   self.assertIn("exact old/new line counts",hunk_rule)
   self.assertIn("reviewed enclosing function symbol",hunk_rule)
   self.assertIn("Blank hunk context",hunk_rule)
   self.assertEqual(captured["structural_example_only"]["source-patch.json"]["patch_encoding"],"base64")
   declarations=captured["structural_example_only"]["plan.json"]["proposal"]["change"]["files_and_symbols"]
   self.assertIsInstance(declarations,list)
   self.assertEqual(declarations,["ggml/src/ggml-cuda/example.cu:example_symbol"])
 def _write_planner_artifacts(self, workspace, assignment, *, mode="valid"):
  relative="ggml/src/ggml-cuda/reviewed.cu"; symbol="reviewed_kernel"
  patch_bytes=(f"diff --git a/{relative} b/{relative}\n"
               f"--- a/{relative}\n+++ b/{relative}\n"
               f"@@ -1 +1 @@ {symbol}()\n-old\n+new\n").encode()
  if mode == "malformed_diff":
   patch_bytes=(f"@@ -1 +1 @@ {symbol}()\n-old\n+new\n").encode()
  manifest={"schema":D.source_candidate.SCHEMA_SOURCE_PATCH,
      "campaign_id":"ak-off-assignment" if mode == "off_assignment" else assignment.campaign_id,
      "proposal_id":assignment.proposal_id,"candidate_id":assignment.candidate_id,
      "source_tree":"llama.cpp","production_base_commit":assignment.production_base_commit,
      "instrument_commit":assignment.instrument_commit,"change_class":"arithmetic",
      "declared_files":[relative],"declared_symbols":{relative:[symbol]},
      "mechanism_id":"planner-fault-test",
      "patch_sha256":hashlib.sha256(patch_bytes).hexdigest(),
      "patch_encoding":"base64","patch_base64":base64.b64encode(patch_bytes).decode()}
  plan={"hypothesis_id":"akh-planner-fault","statement":"bounded planner output",
      "falsifier":"exact runtime does not improve","regime":{"phase":"decode"},
      "proposal":{"proposal_id":assignment.proposal_id,"change_class":"arithmetic",
                  "change":{"files_and_symbols":[f"{relative}:{symbol}"],
                            "estimated_diff_size":2}},
      "source_manifest_path":"source-patch.json"}
  if mode != "missing_plan": (workspace/"plan.json").write_text(json.dumps(plan))
  if mode not in {"missing_plan","missing_manifest"}:
   (workspace/"source-patch.json").write_text(json.dumps(manifest))
 def test_planner_output_faults_are_typed_but_off_assignment_stays_terminal(self):
  package=self.source_package()
  assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
  for mode,pattern in (("malformed_diff","hunk.*file"),
                       ("missing_plan","invalid actor artifact plan.json"),
                       ("missing_manifest","invalid actor artifact source-patch.json")):
   with self.subTest(mode=mode), tempfile.TemporaryDirectory() as t:
    root=Path(t); workspace=root/"operation"/"workspace"; workspace.mkdir(parents=True)
    wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
    planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                           reviewed_sources=package)
    def actor(**_kwargs):
     self._write_planner_artifacts(workspace,assignment,mode=mode)
     return SimpleNamespace(returncode=0,stdout="",stderr="")
    with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
         patch.object(D.codex_container_actor,"run_actor",side_effect=actor), \
         self.assertRaisesRegex(D.PlannerOutputRefusal,pattern):
     planner.plan(context={"authoring_assignment":assignment.to_dict(),
                           "planner_context":{"reviewed_source_package_sha256":
                                              package.package_sha256}},
                  workspace=workspace,checkpoint_path=root/"operation"/"actor-result.json")
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); workspace=root/"operation"/"workspace"; workspace.mkdir(parents=True)
   wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                          reviewed_sources=package)
   def off_assignment(**_kwargs):
    self._write_planner_artifacts(workspace,assignment,mode="off_assignment")
    return SimpleNamespace(returncode=0,stdout="",stderr="")
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=off_assignment), \
        self.assertRaisesRegex(D.DiscoveryControllerError,"invent campaign") as caught:
    planner.plan(context={"authoring_assignment":assignment.to_dict(),
                          "planner_context":{"reviewed_source_package_sha256":
                                             package.package_sha256}},
                 workspace=workspace,checkpoint_path=root/"operation"/"actor-result.json")
   self.assertNotIsInstance(caught.exception,D.PlannerOutputRefusal)
 def test_rc0_actor_checkpoint_resumes_without_rerunning_sol(self):
  class StopAfterCheckpoint(BaseException): pass
  package=self.source_package(); assignment=D.AuthoringAssignment(
      "ak-test","akp-test","akc-test","0"*40,"1"*40)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); operation=root/"operation"; workspace=operation/"workspace"
   workspace.mkdir(parents=True); checkpoint=operation/"actor-result.json"
   wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                          reviewed_sources=package)
   def actor(**_kwargs):
    self._write_planner_artifacts(workspace,assignment)
    return SimpleNamespace(returncode=0,stdout="ok",stderr="")
   context={"authoring_assignment":assignment.to_dict(),
            "planner_context":{"reviewed_source_package_sha256":package.package_sha256}}
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=actor), \
        patch.object(D,"_load_plan",side_effect=StopAfterCheckpoint("stop")), \
        self.assertRaises(StopAfterCheckpoint):
    planner.plan(context=context,workspace=workspace,checkpoint_path=checkpoint)
   self.assertTrue(checkpoint.is_file())
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=AssertionError("Sol reran")):
    candidate=planner.resume_plan(context=context,workspace=workspace,
                                  checkpoint_path=checkpoint)
   self.assertEqual(candidate.hypothesis_id,"akh-planner-fault")
 def test_planner_refusals_retry_same_portfolio_without_science_attempt(self):
  class RefusingPlanner:
   def __init__(self): self.contexts=[]
   def attest(self): return {**D.SOL,"runtime":RUNTIME}
   def plan(self,*,context,workspace):
    self.contexts.append(context)
    raise D.PlannerOutputRefusal("SourceCandidateError: malformed unified diff")
  class NeverCritic:
   def __init__(self): self.calls=0
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def review(self,*_args,**_kwargs):
    self.calls+=1; raise AssertionError("refused planner reached critic")
  class NeverCompute:
   def __init__(self): self.calls=0
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     self.calls+=1; raise AssertionError("refused planner reached compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); planner=RefusingPlanner(); critic=NeverCritic()
   lease=NeverCompute(); screen=NeverCompute()
   base=self.portfolio_config(root,[self.portfolio_record(budget=3)])
   config=D.ControllerConfig(
       root,3,dry_run=True,planner_context=base.planner_context,
       planner_context_sha256=base.planner_context_sha256,
       production_base_commit="0"*40,instrument_commit="1"*40,
       hypothesis_portfolio=base.hypothesis_portfolio,
       hypothesis_portfolio_sha256=base.hypothesis_portfolio_sha256)
   with patch.object(D,"_ensure_question",side_effect=AssertionError("hypothesis opened")):
    result=D.run_controller(config,planner=planner,critic=critic,
                            screener=screen,lease=lease)
   self.assertTrue(result["complete"]); self.assertEqual(len(planner.contexts),3)
   self.assertEqual(critic.calls,0); self.assertEqual(lease.calls,0)
   self.assertEqual(screen.calls,0)
   self.assertEqual([len(row["prior_authoring_refusals"])
                     for row in planner.contexts],[0,1,2])
   self.assertEqual({row["authoring_assignment"]["portfolio_binding"]["hypothesis_id"]
                     for row in planner.contexts},{"akh-portfolio-q8"})
   self.assertEqual([row["status"] for row in result["iterations"]],
                    ["planner_refused"]*3)
   self.assertTrue(all("source_manifest_sha256" not in row
                       for row in result["iterations"]))
   again=D.run_controller(config,planner=planner,critic=critic,
                          screener=screen,lease=lease)
   self.assertEqual(again,result); self.assertEqual(len(planner.contexts),3)
 def test_valid_plan_and_accepted_critic_checkpoints_skip_actors_on_restart(self):
  class StopAfterSave(BaseException): pass
  class CountingCritic(FakeCritic):
   def __init__(self): super().__init__(["accept"]); self.calls=0
   def review(self,*args,**kwargs):
    self.calls+=1; return super().review(*args,**kwargs)
  for crash_phase in ("planner_checkpointed","critic_checkpointed"):
   with self.subTest(crash_phase=crash_phase), tempfile.TemporaryDirectory() as t, \
        patch.object(D.source_candidate,"SourcePatchManifest",Manifest), \
        patch.object(D,"_write_projection"):
    root=Path(t); planner=FakePlanner(); critic=CountingCritic()
    screen=FakeScreen([.01]); original=D.DurableState.save; crashed=[False]
    def save_then_stop(store,state,phase):
     original(store,state,phase)
     if phase == crash_phase and not crashed[0]:
      crashed[0]=True; raise StopAfterSave(phase)
    with patch.object(D.DurableState,"save",new=save_then_stop), \
         self.assertRaises(StopAfterSave):
     D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                      screener=screen,lease=Lease())
    done=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                          screener=screen,lease=Lease())
    self.assertTrue(done["complete"]); self.assertEqual(len(planner.calls),1)
    self.assertEqual(critic.calls,1); self.assertEqual(screen.calls,1)
 def test_late_source_candidate_error_remains_raw_and_ambiguous(self):
  class LateSourceFailure(FakeScreen):
   def screen(self,*_args):
    self.calls+=1
    raise D.source_candidate.SourceCandidateError("late non-planner failure")
  with tempfile.TemporaryDirectory() as t, \
       patch.object(D.source_candidate,"SourcePatchManifest",Manifest), \
       patch.object(D,"_write_projection"):
   root=Path(t); screen=LateSourceFailure([])
   with self.assertRaisesRegex(D.source_candidate.SourceCandidateError,
                               "late non-planner failure"):
    D.run_controller(self.cfg(root,1),planner=FakePlanner(),
                     critic=FakeCritic(["accept"]),screener=screen,lease=Lease())
   state=json.loads((root/"out"/"state.json").read_text())
   self.assertIn("inflight",state); self.assertNotIn("pending",state)
   self.assertEqual(state["inflight"]["exception"]["type"],
                    "SourceCandidateError")
 def test_load_plan_binds_exact_flat_manifest_symbols_before_critic(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); relative="ggml/src/ggml-cuda/vecdotq.cuh"
   symbols=["vec_dot_q5_0_q8_1","vec_dot_q5_0_q8_1_impl"]
   patch_bytes=(f"diff --git a/{relative} b/{relative}\n"
                f"--- a/{relative}\n+++ b/{relative}\n"
                "@@ -1 +1 @@ vec_dot_q5_0_q8_1_impl\n-old\n+new\n"
                "@@ -3 +3 @@ vec_dot_q5_0_q8_1\n-before\n+after\n").encode()
   assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
   manifest={"schema":D.source_candidate.SCHEMA_SOURCE_PATCH,
       "campaign_id":assignment.campaign_id,"proposal_id":assignment.proposal_id,
       "candidate_id":assignment.candidate_id,"source_tree":"llama.cpp",
       "production_base_commit":assignment.production_base_commit,
       "instrument_commit":assignment.instrument_commit,"change_class":"arithmetic",
       "declared_files":[relative],"declared_symbols":{relative:symbols},
       "mechanism_id":"exact-q5-dequant","patch_sha256":hashlib.sha256(patch_bytes).hexdigest(),
       "patch_encoding":"base64","patch_base64":base64.b64encode(patch_bytes).decode()}
   (root/"source-patch.json").write_text(json.dumps(manifest))
   base={"hypothesis_id":"akh-q5-exact","statement":"reuse q5 high bits",
       "falsifier":"exact kernel duration does not improve","regime":{"phase":"decode"},
       "proposal":{"proposal_id":assignment.proposal_id,"change_class":"arithmetic",
                   "change":{"files_and_symbols":[f"{relative}:{symbol}" for symbol in symbols],
                             "estimated_diff_size":4}},
       "source_manifest_path":"source-patch.json"}
   (root/"plan.json").write_text(json.dumps(base))
   candidate=D._load_plan(root/"plan.json",root,assignment=assignment)
   self.assertEqual(candidate.source_manifest.declared_symbols[relative],tuple(symbols))
   undersized=json.loads(json.dumps(base))
   undersized["proposal"]["change"]["estimated_diff_size"]=3
   (root/"plan.json").write_text(json.dumps(undersized))
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               r"actual changed-line count \(3 < 4\)"):
    D._load_plan(root/"plan.json",root,assignment=assignment)
   for malformed in ([f"{relative}:{symbols[0]}"],
                     {relative:symbols},
                     [f"{relative}:{symbol}" for symbol in symbols]+[
                         "ggml/src/ggml-cuda/offscope.cu:foreign_kernel"]):
    with self.subTest(malformed=malformed):
     plan=json.loads(json.dumps(base)); plan["proposal"]["change"]["files_and_symbols"]=malformed
     (root/"plan.json").write_text(json.dumps(plan))
     with self.assertRaisesRegex(D.source_candidate.SourceCandidateError,
                                 "exactly equal"):
      D._load_plan(root/"plan.json",root,assignment=assignment)
 def test_fable_context_binds_selected_reviewed_source_preimage(self):
  content=b"template <typename T>\nvoid quantize_q8_1(T * x) {\n    x[0] = T{};\n}\n"
  digest=hashlib.sha256(content).hexdigest()
  package_body={"schema":"epyc.autokernel.reviewed_source_package.v1",
      "instrument_commit":"1"*40,
      "files":[{"relative_path":"ggml/src/ggml-cuda/quantize.cu",
                "sha256":digest,
                "workspace_path":"reviewed-source/ggml/src/ggml-cuda/quantize.cu"}]}
  package=D.ReviewedSourcePackage("1"*40,(D.ReviewedSourceFile(
      "ggml/src/ggml-cuda/quantize.cu",digest,content),),D._sha(package_body))
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"claude"; wrapper.write_bytes(b"claude"); wrapper.chmod(0o700)
   binding=D._portfolio_binding(self.portfolio_config(root,[self.portfolio_record()]),
                                self.portfolio_record())
   candidate=self.portfolio_candidate(binding)
   captured={}
   def critic_call(**kwargs):
    captured.update(kwargs)
    return SimpleNamespace(decision="accept",reason="source-visible")
   critic=D.ClaudeCritic(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                         reviewed_sources=package)
   with patch.object(D.claude_fable5_critic_actor,"runtime_identity",
                     return_value=CLAUDE_RUNTIME), \
        patch.object(D.claude_fable5_critic_actor,"run_critic",
                     side_effect=critic_call):
    result=critic.review(candidate,context={"sealed":"context"},workspace=root)
  self.assertEqual(result.decision,"accept")
  prompt=json.loads(captured["prompt"])
  source=prompt["context"]["selected_source_preimage"]
  self.assertEqual(source["source_sha256"],digest)
  self.assertIn("quantize_q8_1",source["excerpts"][0]["text"])
  self.assertEqual(captured["bindings"]["context_sha256"],
                   D._sha(prompt["context"]))
 def test_reviewed_source_package_refuses_symlinked_parent(self):
  package=self.source_package()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); outside=root/"outside"; outside.mkdir()
   (root/"reviewed-source").symlink_to(outside, target_is_directory=True)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"already exists|symlink"):
    package.materialize(root)
 def test_bounded_dispatch_refuses_meta_and_out_of_range_literals(self):
  valid=D.BoundedDispatchExpectation("template-v1.anchor.0","kernel_6",2,128,64,0)
  self.assertEqual(valid.kernel_name,"kernel_6")
  for name in ("kernel.*", "kernel[0]", "kernel name"):
   with self.subTest(name=name), self.assertRaises(D.DiscoveryControllerError):
    D.BoundedDispatchExpectation("template-v1.anchor.0",name,1,1,1,0)
  with self.assertRaises(D.DiscoveryControllerError): D.BoundedDispatchExpectation("template-v1.anchor.0","kernel",1,1,4097,0)
  full="void kernel<float>(float const*, void*) [clone .kd]"
  self.assertEqual(D.BoundedDispatchExpectation("template-v1.anchor.0",full,2,128,64,0).kernel_name,full)
 def cfg(self,root,n=2): return D.ControllerConfig(root/"out",n)
 def test_old_v3_and_v4_durable_state_refuse(self):
  for version in ("v3","v4"):
   with self.subTest(version=version), tempfile.TemporaryDirectory() as t:
    root=Path(t); store=D.DurableState(root)
    body={"schema":f"epyc.autokernel.discovery_controller.{version}",
          "authority":D.AUTHORITY,"roster":D.sealed_roster(),
          "iterations":[],"next":1,"complete":False}
    body["state_sha256"]=D._sha(body)
    D._atomic(store.path,body)
    with self.assertRaisesRegex(D.DiscoveryControllerError,"wrong controller journal"):
     store.load()
 def test_exact_sol_planner_and_fable5_critic(self):
  self.assertEqual(D.sealed_roster()["claude_members"],1); self.assertEqual([x["model"] for x in D.sealed_roster()["members"]],["gpt-5.6-sol","claude-fable-5"])
 def test_legacy_terra_roster_state_refuses_resume(self):
  with tempfile.TemporaryDirectory() as t:
   store=D.DurableState(Path(t))
   legacy={"schema":"epyc.autokernel.discovery_controller.v2","authority":D.AUTHORITY,
           "roster":{"schema":"epyc.autokernel.discovery_roster.v2","members":[D.SOL,{"provider":"codex","model":"gpt-5.6-terra","effort":"high","role":"critic"}],"claude_members":0,"member_count":2},
           "iterations":[],"next":1,"complete":False}
   D._atomic(store.path,legacy)
   with self.assertRaises(D.DiscoveryControllerError): store.load()
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
 def test_fable_veto_precedes_lease_and_screen(self):
  class ExplosiveLease:
   def admit(self,*args,**kwargs): raise AssertionError("critic veto must precede lease")
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   screen=FakeScreen([.1]); result=D.run_controller(self.cfg(Path(t),1),planner=FakePlanner(),critic=FakeCritic(["reject"]),screener=screen,lease=ExplosiveLease())
   self.assertEqual(result["iterations"][0]["status"],"critic_reject"); self.assertEqual(screen.calls,0)
 def test_fable_critic_sees_full_patch_and_binds_all_authority_digests(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"claude"; wrapper.write_bytes(b"claude"); wrapper.chmod(0o700)
   catalog={"fattn":{"allowed_files":["ggml/src/ggml-cuda/fattn.cu"]}}
   critic=D.ClaudeCritic(wrapper=wrapper,environment={"PATH":"/usr/bin"},template_catalog=catalog,
       wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),runtime_identity=CLAUDE_RUNTIME)
   with patch.object(D.source_candidate,"SourcePatchManifest",Manifest):
    candidate=D.PlannedCandidate("akh-fable","statement","falsifier",{"backend":"gpu"},{"proposal_id":"akp-test"},Manifest(),H)
   context={"turn":1,"planner_context_sha256":H}
   captured={}
   def run_critic(**kwargs):
    captured.update(kwargs); return SimpleNamespace(decision="accept",reason="bounded")
   with patch.object(D.claude_fable5_critic_actor,"runtime_identity",return_value=CLAUDE_RUNTIME), \
        patch.object(D.claude_fable5_critic_actor,"run_critic",side_effect=run_critic):
    self.assertEqual(critic.review(candidate,context=context,workspace=root).decision,"accept")
   body=json.loads(captured["prompt"])
   self.assertEqual(body["candidate"]["manifest"]["patch_text"],Manifest.patch_bytes.decode())
   self.assertEqual(captured["bindings"],body["required_output_bindings"])
   self.assertEqual(set(captured["bindings"]),{"proposal_sha256","source_manifest_sha256","candidate_patch_sha256","context_sha256","template_catalog_sha256"})
 def test_fable_critic_refuses_patch_above_full_visibility_limit(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"claude"; wrapper.write_bytes(b"claude")
   manifest=Manifest(patch_text="x"*65537,patch_bytes=b"x"*65537)
   with patch.object(D.source_candidate,"SourcePatchManifest",Manifest):
    candidate=D.PlannedCandidate("akh-fable","statement","falsifier",{}, {"proposal_id":"akp-test"},manifest,H)
   critic=D.ClaudeCritic(wrapper=wrapper,environment={"PATH":"/usr/bin"})
   with self.assertRaisesRegex(D.DiscoveryControllerError,"bounded critic visibility"), \
        patch.object(D.claude_fable5_critic_actor,"run_critic",side_effect=AssertionError("no actor")):
    critic.review(candidate,context={},workspace=root)
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
   args._device_claim_acquirer=lambda *_args,**_kwargs: None; args._expected_outer_claim_id="akd-outer"
   (root/"screen").mkdir()
   phase={"schema":"epyc.autokernel.borrowed_device_claim_phase.v1","mode":"borrowed_outer_reservation","outer_claim_id":"akd-outer","device_id":"mi210_0","campaign_id":"ak-test","phase_ended_at":"done","physical_release":False}
   opened={"claim_id":"akd-outer"}; raw={"schema":"epyc.autokernel.gpu_candidate_only_screen.v2","non_promotable":True,"promotion_claim":False,"hip_residency_proved":True,"median_relative":.02,"baseline_sha256":H,"device_claim_mode":"borrowed_outer_reservation","device_claim_open":opened,"device_claim_borrowed_phase_end":phase}; raw["result_sha256"]=D.gpu_source_proofs._hash(raw)
   (root/"screen"/"result.json").write_text(json.dumps(raw))
   governance={"status":"borrowed_phase_ended","device_claim_mode":"borrowed_outer_reservation","device_claim_open":opened,"device_claim_borrowed_phase_end":phase}
   (root/"screen"/"live-governance.json").write_text(json.dumps(governance))
   run.side_effect=lambda _args: events.append("runner") or raw
   source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest(); dispatch_hash=hashlib.sha256(dispatch_file.read_bytes()).hexdigest()
   candidate_identity=build.candidate_identity; anchor_identity=build.anchor_identity; material={"manifest_sha256":H,"candidate":candidate_identity,"anchor":anchor_identity,"workload_sha256":H,"correctness":{"file_sha256":source_hash,"native_sha256":H},"attribution":{"file_sha256":dispatch_hash,"native_sha256":H}}
   hashed={**material,"candidate":candidate_identity.__dict__,"anchor":anchor_identity.__dict__}; bundle=D.gpu_source_proofs.GpuSourceProofBundle(**material,bundle_sha256=D.gpu_source_proofs._hash(hashed))
   screen=D.GpuSourceScreener(build_source=lambda *_: events.append("build") or build,proof_bundle=lambda *_: events.extend(["source","dispatch"]) or bundle,args_factory=lambda *_:args)
   with patch.object(D.autokernel_progression,"_gpu_screen",return_value={"stage":"candidate"}):
    got=screen.screen(item,object(),{})
   self.assertEqual(events,["build","source","dispatch","runner"]); self.assertEqual(got.dispatch_proof_sha256,dispatch_hash)
   (root/"screen"/"live-governance.json").unlink()
   with patch.object(D.autokernel_progression,"_gpu_screen",return_value={"stage":"candidate"}), self.assertRaisesRegex(D.DiscoveryControllerError,"governance"):
    screen.screen(item,object(),{})
   wrong_phase={**phase,"outer_claim_id":"akd-wrong"}; wrong_opened={"claim_id":"akd-wrong"}
   wrong={**raw,"device_claim_open":wrong_opened,"device_claim_borrowed_phase_end":wrong_phase}; wrong.pop("result_sha256"); wrong["result_sha256"]=D.gpu_source_proofs._hash(wrong)
   (root/"screen"/"result.json").write_text(json.dumps(wrong))
   (root/"screen"/"live-governance.json").write_text(json.dumps({**governance,"device_claim_open":wrong_opened,"device_claim_borrowed_phase_end":wrong_phase}))
   run.side_effect=lambda _args: wrong
   with patch.object(D.autokernel_progression,"_gpu_screen",return_value={"stage":"candidate"}), self.assertRaisesRegex(D.DiscoveryControllerError,"exact outer claim"):
    screen.screen(item,object(),{})
 def test_lease_wait_is_durable_without_spending_iteration(self):
  class Wait:
   def admit(self,item,*,operation_key): return {"admitted":False,"reason":"CPU window busy","operation_key":operation_key}
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   r=D.run_controller(self.cfg(Path(t),1),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=FakeScreen([.1]),lease=Wait())
   self.assertEqual(r["next"],1); self.assertEqual(r["pending"]["row"]["status"],"waiting_resource"); self.assertFalse(r["complete"])
 def test_postbuild_resource_wait_retries_exact_candidate_without_replanning(self):
  class Race(FakeScreen):
   def __init__(self,root): super().__init__([]); self.root=root
   def screen(self,*args):
    self.calls+=1
    if self.calls == 1:
     operation_key=args[2]["operation_key"]; directory=self.root/operation_key/"resource-waits"; directory.mkdir(parents=True,mode=0o700)
     contention={"admitted":False,"phase":"pre_executor_reservation","reason":"device_busy","operation_key":operation_key,"promotion_claim":False}
     body={"schema":"epyc.autokernel.gpu_source_resource_wait.v1","authority":D.AUTHORITY,"promotion_claim":False,"operation_key":operation_key,"gpu_executor_started":False,"proof_root_created":False,"runner_plan_created":False,"runner_output_created":False,"contention":contention}; body["receipt_sha256"]=D._sha(body)
     path=directory/"wait-0001.json"; path.write_text(json.dumps(body,sort_keys=True)); digest=hashlib.sha256(path.read_bytes()).hexdigest()
     raise D.ResourceWait("device race",receipt={**contention,"stage_receipt_path":str(path),"stage_receipt_sha256":digest})
    return D.SealedScreen("receipt",H,.04,"candidate",H,H,H)
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); planner=FakePlanner(); critic=FakeCritic(["accept"]); screen=Race(root/"operations"); lease=Lease()
   waiting=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,screener=screen,lease=lease)
   self.assertEqual((waiting["next"],waiting["pending"]["row"]["status"]),(1,"waiting_resource"))
   done=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,screener=screen,lease=lease)
   self.assertTrue(done["complete"]); self.assertEqual((len(planner.calls),screen.calls),(1,2))
 def test_forged_resource_wait_cannot_erase_ambiguous_inflight(self):
  class Forged(FakeScreen):
   def screen(self,*args):
    operation_key=args[2]["operation_key"]
    raise D.ResourceWait("forged",receipt={"admitted":False,"phase":"pre_executor_reservation","operation_key":operation_key,"promotion_claim":False})
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"durable stage"):
    D.run_controller(self.cfg(root,1),planner=FakePlanner(),critic=FakeCritic(["accept"]),screener=Forged([]),lease=Lease())
   state=json.loads((root/"out"/"state.json").read_text()); self.assertIn("inflight",state); self.assertNotIn("pending",state)
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
