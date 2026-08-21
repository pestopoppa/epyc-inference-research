"""No-hardware replay tests for the typed discovery state machine."""
from __future__ import annotations
import argparse, base64, dataclasses, hashlib, json, os, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
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
  symbol_authority={}
  for row in records:
   if not row["current_bundle_eligibility"]["eligible"]:
    continue
   template_id=row["current_bundle_eligibility"]["template_ids"][0]
   symbol_authority[template_id]={
       path:list(row["target"]["source_symbols"])
       for path in row["target"]["source_files"]}
  return D.ControllerConfig(root,3,dry_run=True,
      planner_context={"template_symbol_authority":symbol_authority,
       "portfolio_dispatch_authority": {
          row["hypothesis_id"]: [{"route_id":"cuda-quantize-q8-v1.anchor.0",
                                  "kernel_name":"quantize_q8_1", "calls":18705,
                                  "grid":1024,"workgroup":256,"lds_bytes":0}]
          for row in records if row["current_bundle_eligibility"]["eligible"]}},
      planner_context_sha256="e"*64,
      hypothesis_portfolio=portfolio,
      hypothesis_portfolio_sha256="f"*64)
 def portfolio_candidate(self, binding, *, hypothesis_id=None, mechanism_id=None,
                         regime=None, new_line="y"):
  paths=tuple(binding["target_files"])
  symbols_by_file={
      path:tuple(binding["target_symbols_by_file"][path]) for path in paths}
  patch="".join(
      f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
      f"@@ -1 +1 @@ {symbols_by_file[path][0]}()\n-x\n+{new_line}\n"
      for path in paths).encode()
  manifest=D.source_candidate.SourcePatchManifest(
      campaign_id="ak-test",proposal_id="akp-test",candidate_id="akc-test",
      source_tree="llama.cpp",production_base_commit="0"*40,
      instrument_commit="1"*40,change_class=binding["change_class"],declared_files=paths,
      declared_symbols=symbols_by_file,
      mechanism_id=mechanism_id or binding["mechanism_id"],
      patch_sha256=hashlib.sha256(patch).hexdigest(),patch_bytes=patch)
  intent=D.GpuSourceExperimentIntent(binding["template_id"],"gpu_decode",
      binding["target_symbols"][0],
      "backend-ops-hip-v1","decode-tg128-rocprof-v1",
      tuple(D.BoundedDispatchExpectation(**row)
            for row in binding["expected_dispatch"]))
  return D.PlannedCandidate(hypothesis_id or binding["hypothesis_id"],
      binding["statement"],binding["falsifier"],regime or binding["regime"],
      {"proposal_id":"akp-test","change_class":binding["change_class"]},
      manifest,manifest.patch_bundle_sha256,intent)
 def reseal_carry_forward(self, value):
  body={key:item for key,item in value.items()
        if key!="carry_forward_sha256"}
  receipt_sha256=D._sha(body)
  return {**body,"carry_forward_sha256":receipt_sha256},receipt_sha256
 def carry_forward(self, *, patch_sha256=None,
                   cross_campaign_sha256=None):
  digest=lambda label:hashlib.sha256(label.encode()).hexdigest()
  body={"schema":"epyc.autokernel.discovery_carry_forward.v1",
      "predecessor_state_file_sha256":digest("state-file"),
      "predecessor_journal_file_sha256":digest("journal-file"),
      "predecessor_state_semantic_sha256":digest("state-semantic"),
      "portfolio_outcomes":{
          "akh-v2-q5-type-specific-dequant":"nominated",
          "akh-v2-q8-quantizer-new-mechanism":"retire",
          "akh-v2-fa-gqa7-pair-tail":"bounded_authoring_skip",
          "akh-v2-rms-direct-load-reduction":"bounded_authoring_skip"},
      "candidate_semantic_sha256":sorted(digest(f"semantic-{i}") for i in range(12)),
      "candidate_patch_sha256":sorted(
          [patch_sha256 or digest("patch-repeat")]+
          [digest(f"patch-{i}") for i in range(6)]),
      "cross_campaign_candidate_sha256":sorted(
          [cross_campaign_sha256 or digest("cross-repeat")]+
          [digest(f"cross-{i}") for i in range(6)])}
  return self.reseal_carry_forward(body)
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
                               "source_manifest_sha256":"1"*64,
                               "result_sha256":"2"*64,
                               "evidence":{"source":"3"*64}})
   second=D._select_portfolio_binding(state,config)
   self.assertEqual(second["hypothesis_id"],"akh-lower")
   with self.assertRaisesRegex(D.DiscoveryControllerError,"controller-owned"):
    D._validate_portfolio_candidate(
        self.portfolio_candidate(second,hypothesis_id="akh-invented"),second,
        config.hypothesis_portfolio)
 def test_portfolio_scheduler_round_robins_negative_science_before_second_candidate(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[
       self.portfolio_record(hypothesis_id="akh-first",rank=1,budget=2),
       self.portfolio_record(hypothesis_id="akh-second",rank=2,budget=2)])
   state={"iterations":[{
       "portfolio_hypothesis_id":"akh-first",
       "candidate_semantic_sha256":"1"*64,
       "result_sha256":"2"*64,
       "evidence":{"source":"3"*64}}]}
   self.assertEqual(
       D._select_portfolio_binding(state,config)["hypothesis_id"],
       "akh-second")
   state["iterations"].append({
       "portfolio_hypothesis_id":"akh-second",
       "candidate_semantic_sha256":"4"*64,
       "result_sha256":"5"*64,
       "evidence":{"source":"6"*64}})
   self.assertEqual(
       D._select_portfolio_binding(state,config)["hypothesis_id"],
       "akh-first")
 def test_portfolio_scheduler_yields_after_non_scientific_authoring_failure(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[
       self.portfolio_record(hypothesis_id="akh-first",rank=1,budget=2),
       self.portfolio_record(hypothesis_id="akh-second",rank=2,budget=2)])
   state={"iterations":[{
       "status":"planner_refused",
       "portfolio_hypothesis_id":"akh-first"}],
       "portfolio_authoring_failures":{"akh-first":1}}
   self.assertEqual(
       D._select_portfolio_binding(state,config)["hypothesis_id"],
       "akh-second")
 def test_portfolio_binding_requires_exact_template_symbol_authority(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record()])
   context=dict(config.planner_context)
   context["template_symbol_authority"]={}
   malformed=dataclasses.replace(config,planner_context=context)
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               "per-file symbol authority"):
    D._select_portfolio_binding({"iterations":[]},malformed)
 def test_predecessor_carry_forward_refuses_terminal_family_and_patch_repeat(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)
   record=self.portfolio_record()
   config=self.portfolio_config(root,[record])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   candidate=self.portfolio_candidate(binding)
   carry,carry_sha=self.carry_forward(
       patch_sha256=candidate.source_manifest.patch_sha256)
   config=dataclasses.replace(config,carry_forward=carry,
                              carry_forward_sha256=carry_sha)
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               "repeats predecessor source semantics"):
    D._validate_portfolio_candidate(candidate,binding,
                                    config.hypothesis_portfolio,carry)
   terminal_record=self.portfolio_record(
       hypothesis_id="akh-v2-q8-quantizer-new-mechanism")
   terminal_config=self.portfolio_config(root, [terminal_record])
   terminal_binding=D._select_portfolio_binding(
       {"iterations":[]},terminal_config)
   terminal_candidate=self.portfolio_candidate(terminal_binding)
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               "predecessor-terminal hypothesis"):
    D._validate_portfolio_candidate(
        terminal_candidate,terminal_binding,
        terminal_config.hypothesis_portfolio,carry)
 def test_predecessor_carry_forward_exact_grammar_and_hash_are_required(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record()])
   valid,_valid_sha=self.carry_forward()
   mutations={}
   missing=dict(valid); missing.pop("predecessor_state_file_sha256")
   mutations["missing_key"]=self.reseal_carry_forward(missing)
   extra={**valid,"unexpected":"closed"}
   mutations["extra_key"]=self.reseal_carry_forward(extra)
   bad_lengths={**valid,"candidate_semantic_sha256":
                valid["candidate_semantic_sha256"][:-1]}
   mutations["bad_lengths"]=self.reseal_carry_forward(bad_lengths)
   bad_outcomes={**valid,"portfolio_outcomes":{
       **valid["portfolio_outcomes"],
       "akh-v2-q8-quantizer-new-mechanism":"nominated"}}
   mutations["bad_outcomes"]=self.reseal_carry_forward(bad_outcomes)
   mutations["bad_hash"]=({**valid,"carry_forward_sha256":H},H)
   for label,(value,receipt_sha256) in mutations.items():
    with self.subTest(label=label), self.assertRaisesRegex(
            D.DiscoveryControllerError,
            "invalid predecessor carry-forward authority"):
     dataclasses.replace(config,carry_forward=value,
                         carry_forward_sha256=receipt_sha256)
 def test_predecessor_cross_campaign_repeat_refuses_but_new_patch_passes(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record()])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   repeated=self.portfolio_candidate(binding)
   carry,carry_sha=self.carry_forward(
       cross_campaign_sha256=D._cross_campaign_candidate_identity(repeated))
   config=dataclasses.replace(config,carry_forward=carry,
                              carry_forward_sha256=carry_sha)
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               "repeats predecessor source semantics"):
    D._validate_portfolio_candidate(
        repeated,binding,config.hypothesis_portfolio,carry)
   novel=self.portfolio_candidate(binding,new_line="genuinely_new")
   D._validate_portfolio_candidate(
       novel,binding,config.hypothesis_portfolio,carry)
 def test_durable_v6_resume_binds_carry_digest_and_refuses_legacy_state(self):
  class RefusingPlanner(FakePlanner):
   def plan(self,**_kwargs):
    raise D.PlannerOutputRefusal("bounded legacy authoring refusal")
  class Never:
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     raise AssertionError("carry-forward fixture reached compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); carry,carry_sha=self.carry_forward()
   base=self.portfolio_config(
       root,[self.portfolio_record(eligible=False)])
   config=dataclasses.replace(base,carry_forward=carry,
                              carry_forward_sha256=carry_sha)
   result=D.run_controller(
       config,planner=FakePlanner(),critic=FakeCritic([]),
       screener=FakeScreen([.01]),lease=Lease())
   self.assertEqual(result["schema"],D.SCHEMA)
   self.assertEqual(result["carry_forward_sha256"],carry_sha)
   changed=dict(carry)
   changed["predecessor_state_file_sha256"]="9"*64
   changed,changed_sha=self.reseal_carry_forward(changed)
   changed_config=dataclasses.replace(
       base,carry_forward=changed,carry_forward_sha256=changed_sha)
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               "predecessor carry-forward changed"):
    D.run_controller(
        changed_config,planner=FakePlanner(),critic=FakeCritic([]),
        screener=FakeScreen([.01]),lease=Lease())
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); legacy=self.portfolio_config(
       root,[self.portfolio_record(budget=3)])
   never=Never()
   result=D.run_controller(
       legacy,planner=RefusingPlanner(),critic=never,
       screener=never,lease=never)
   self.assertTrue(result["iterations"])
   carry,carry_sha=self.carry_forward()
   upgraded=dataclasses.replace(
       legacy,carry_forward=carry,carry_forward_sha256=carry_sha)
   with self.assertRaisesRegex(D.DiscoveryControllerError,
                               "legacy durable state lacks predecessor"):
    D.run_controller(
        upgraded,planner=RefusingPlanner(),critic=never,
        screener=never,lease=never)
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
 def test_portfolio_exact_dnr_match_has_canonical_receipt(self):
  with tempfile.TemporaryDirectory() as t:
   record=self.portfolio_record(); config=self.portfolio_config(Path(t),[record])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   dnr={"dnr_id":"dnr-exact-test",
        "mechanism":{"fingerprint_sha256":binding["mechanism_id"]},
        "regime":dict(binding["regime"])}
   portfolio=hypothesis_portfolio.Portfolio(
       {**config.hypothesis_portfolio.body,"do_not_repeat":[dnr]},"f"*64)
   config=dataclasses.replace(config,hypothesis_portfolio=portfolio)
   candidate=self.portfolio_candidate(binding)
   receipt=D._portfolio_exact_dnr_check(config,candidate,binding)
   self.assertEqual(receipt["outcome"],D.schemas.FAIL)
   self.assertEqual(receipt["matched_dnr_ids"],["dnr-exact-test"])
   self.assertEqual(receipt["candidate_mechanism_id"],binding["mechanism_id"])
   self.assertEqual(receipt["canonical_regime_sha256"],
                    D.schemas.content_hash(binding["regime"]))
   self.assertEqual(receipt["receipt_sha256"],D.schemas.content_hash(
       {key:value for key,value in receipt.items() if key!="receipt_sha256"}))

 def test_portfolio_exact_dnr_refuses_with_zero_critic_auth_lease_or_screen(self):
  class Planner(FakePlanner):
   def __init__(self,candidate): super().__init__(); self.candidate=candidate
   def plan(self,**kwargs): self.calls.append(kwargs); return self.candidate
  class NeverCritic(FakeCritic):
   def __init__(self): super().__init__([]); self.calls=0
   def review(self,*args,**kwargs):
    self.calls+=1; raise AssertionError("portfolio DNR reached critic")
  class Never:
   calls=0
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     self.calls+=1; raise AssertionError("portfolio DNR reached compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   record=self.portfolio_record(); config=self.portfolio_config(Path(t),[record])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   dnr={"dnr_id":"dnr-exact-test",
        "mechanism":{"fingerprint_sha256":binding["mechanism_id"]},
        "regime":dict(binding["regime"])}
   portfolio=hypothesis_portfolio.Portfolio(
       {**config.hypothesis_portfolio.body,"do_not_repeat":[dnr]},"f"*64)
   config=dataclasses.replace(config,hypothesis_portfolio=portfolio)
   critic=NeverCritic(); lease=Never(); screen=Never()
   with patch.object(D.hypotheses.HypothesisTracker,"authorize_claim",
                     side_effect=AssertionError("portfolio DNR reached authorization")) as auth:
    result=D.run_controller(config,
        planner=Planner(self.portfolio_candidate(binding)),critic=critic,
        screener=screen,lease=lease)
   row=result["iterations"][0]
   self.assertEqual(row["status"],"portfolio_dnr_refused")
   self.assertEqual(row["portfolio_exact_dnr_check"]["matched_dnr_ids"],
                    ["dnr-exact-test"])
   self.assertEqual((critic.calls,auth.call_count,lease.calls,screen.calls),(0,0,0,0))

 def test_new_portfolio_path_fails_closed_on_missing_or_mismatched_mechanism_and_receipt(self):
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record()])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   candidate=self.portfolio_candidate(binding)
   missing=dict(binding); missing.pop("mechanism_id")
   tracker=D._tracker(D.DurableState(config.output_root))
   with self.assertRaisesRegex(D.DiscoveryControllerError,"structural mechanism"):
    D._ensure_question(tracker,candidate,missing)
   mismatched=self.portfolio_candidate(binding,mechanism_id="e"*64)
   with self.assertRaisesRegex(D.DiscoveryControllerError,"structural mechanism"):
    D._ensure_question(tracker,mismatched,binding)
   row={"portfolio_binding":dict(binding)}
   with self.assertRaisesRegex(D.DiscoveryControllerError,"receipt is missing"):
    D._revalidate_portfolio_checkpoint(config,candidate,row)
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
  self.assertEqual(result["iterations"][0]["status"],"planner_contract_refused")

 def test_portfolio_dnr_receipt_round_trips_before_resumed_critic_and_authorization(self):
  class Planner(FakePlanner):
   def __init__(self,candidate): super().__init__(); self.candidate=candidate
   def plan(self,**kwargs): self.calls.append(kwargs); return self.candidate
  class CrashCritic(FakeCritic):
   def __init__(self): super().__init__([])
   def review(self,*_args,**_kwargs): raise RuntimeError("critic interrupted")
  class NoReplan(FakePlanner):
   def plan(self,**_kwargs): raise AssertionError("resume replanned a sealed candidate")
  class Never:
   def __getattr__(self,_name):
    def called(*_args,**_kwargs): raise AssertionError("dry-run reached compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   record=self.portfolio_record(budget=1)
   config=self.portfolio_config(Path(t),[record])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   planner=Planner(self.portfolio_candidate(binding))
   with self.assertRaisesRegex(RuntimeError,"critic interrupted"):
    D.run_controller(config,planner=planner,critic=CrashCritic(),
                     screener=Never(),lease=Never())
   checkpoint=json.loads((config.output_root/"state.json").read_text())
   receipt=dict(checkpoint["pending"]["row"]["portfolio_exact_dnr_check"])
   self.assertEqual(receipt["outcome"],D.schemas.PASS)
   result=D.run_controller(config,planner=NoReplan(),critic=FakeCritic(["accept"]),
                           screener=Never(),lease=Never())
   row=result["iterations"][0]
   self.assertEqual(row["portfolio_exact_dnr_check"],receipt)
   self.assertEqual(row["campaign_ledger_dnr_outcome"],D.schemas.PASS)
   self.assertEqual(row["status"],"dry_run_authorized")
   self.assertEqual(len(planner.calls),1)

 def test_portfolio_family_budget_allows_two_distinct_candidates(self):
  class Planner(FakePlanner):
   def __init__(self,candidates): super().__init__(); self.candidates=iter(candidates)
   def plan(self,**kwargs): self.calls.append(kwargs); return next(self.candidates)
  class Never:
   def __getattr__(self,_name):
    def called(*_args,**_kwargs): raise AssertionError("dry-run reached compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   config=self.portfolio_config(Path(t),[self.portfolio_record(budget=2)])
   binding=D._select_portfolio_binding({"iterations":[]},config)
   first=self.portfolio_candidate(binding)
   second_patch=first.source_manifest.patch_bytes.replace(b"+y\n",b"+z\n")
   second_manifest=dataclasses.replace(
       first.source_manifest,patch_bytes=second_patch,
       patch_sha256=hashlib.sha256(second_patch).hexdigest())
   second=D.PlannedCandidate(
       first.hypothesis_id,first.statement,first.falsifier,first.regime,
       first.proposal,second_manifest,second_manifest.patch_bundle_sha256,
       first.experiment_intent)
   result=D.run_controller(
       config,planner=Planner([first,second]),
       critic=FakeCritic(["accept","accept"]),screener=Never(),lease=Never())
   rows=result["iterations"]
   self.assertEqual([row["status"] for row in rows], ["dry_run_authorized"])
   self.assertEqual(len({row["source_manifest_sha256"] for row in rows}),1)
   self.assertEqual([row["campaign_ledger_dnr_outcome"] for row in rows],
                    [D.schemas.PASS])
   self.assertIn(binding["hypothesis_id"], result["portfolio_validations"])
   self.assertNotIn(binding["hypothesis_id"], result.get("portfolio_skips", {}))
   tracked=D._tracker(D.DurableState(config.output_root)).state()[
       binding["hypothesis_id"]]
   self.assertEqual(tracked.hypothesis.regime["mechanism"],
                    binding["mechanism_id"])

 def test_legacy_generic_question_without_mechanism_remains_could_not_check(self):
  class Planner(FakePlanner):
   def plan(self,*,context,workspace):
    self.calls.append(context); manifest=Manifest()
    return D.PlannedCandidate(
        "akh-legacy-generic","legacy question without structural identity",
        "no speed improvement invalidates it",{"backend":"gpu","phase":"decode"},
        {"id":"legacy"},manifest,manifest.patch_bundle_sha256)
  with tempfile.TemporaryDirectory() as t, \
       patch.object(D.source_candidate,"SourcePatchManifest",Manifest), \
       patch.object(D,"_write_projection"):
   result=D.run_controller(
       D.ControllerConfig(Path(t)/"out",1,dry_run=True),planner=Planner(),
       critic=FakeCritic(["accept"]),screener=FakeScreen([.1]),lease=Lease())
  self.assertEqual(result["iterations"][0]["campaign_ledger_dnr_outcome"],
                   D.schemas.COULD_NOT_CHECK)
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
   self.assertIn("deployed anchor objects",captured["authoring_contract"]["expected_dispatch"])
   self.assertIn("Never substitute predicted candidate subroutes",
                 captured["authoring_contract"]["expected_dispatch_rule"])
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

 def test_planner_catalog_hides_controller_owned_fa_geometry(self):
  catalog={"cuda-fattn-gqa7-common-v1":{
      "template_id":"cuda-fattn-gqa7-common-v1",
      "semantics":{"candidate_dispatch_variants":{
          "gqa7_bulk_pairs":{"calls":3096,"grid":3072},
          "gqa7_scalar_tail":{"calls":3096,"grid":1024}}}}}
  planner=D.CodexPlanner(wrapper=Path("/not-invoked"),environment={},
                         template_catalog=catalog)
  projected=planner._planner_catalog()
  semantics=projected["cuda-fattn-gqa7-common-v1"]["semantics"]
  self.assertNotIn("candidate_dispatch_variants",semantics)
  self.assertEqual(semantics["candidate_dispatch_strategy"],{
      "strategy_id":"gqa7_pair_tail",
      "selection_authority":"controller_owned",
      "expected_dispatch_source":
          "controller_owned_portfolio_binding.expected_dispatch",
      "instruction":(
          "Author the bounded six-head pair plus one-head tail source mechanism. "
          "Do not emit candidate route IDs, call counts, or geometry; the controller "
          "derives and validates those after authorization.")})
  self.assertNotIn("3072",json.dumps(projected))
  self.assertIn("3072",json.dumps(planner.template_catalog))
 def _write_planner_artifacts(self, workspace, assignment, *, mode="valid"):
  relative="ggml/src/ggml-cuda/reviewed.cu"; symbol="reviewed_kernel"
  patch_bytes=(f"diff --git a/{relative} b/{relative}\n"
               f"--- a/{relative}\n+++ b/{relative}\n"
               f"@@ -1 +1 @@ {symbol}()\n-old\n+new\n").encode()
  if mode == "malformed_diff":
   patch_bytes=(f"@@ -1 +1 @@ {symbol}()\n-old\n+new\n").encode()
  if mode == "underestimated_diff_v16":
   patch_bytes=(f"diff --git a/{relative} b/{relative}\n"
                f"--- a/{relative}\n+++ b/{relative}\n"
                f"@@ -1,8 +1,7 @@ {symbol}()\n"
                "-old1\n-old2\n-old3\n-old4\n-old5\n-old6\n-old7\n-old8\n"
                "+new1\n+new2\n+new3\n+new4\n+new5\n+new6\n+new7\n").encode()
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
                            "estimated_diff_size":(
                                14 if mode == "underestimated_diff_v16" else 2)}},
      "source_manifest_path":"source-patch.json"}
  if mode != "missing_plan": (workspace/"plan.json").write_text(json.dumps(plan))
  if mode not in {"missing_plan","missing_manifest"}:
   (workspace/"source-patch.json").write_text(json.dumps(manifest))
 def test_planner_output_faults_are_typed_but_off_assignment_stays_terminal(self):
  package=self.source_package()
  assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
  for mode,pattern in (("malformed_diff","hunk.*file"),
                       ("missing_plan","invalid actor artifact plan.json"),
                       ("missing_manifest","invalid actor artifact source-patch.json"),
                       ("underestimated_diff_v16", "14 < 15")):
   with self.subTest(mode=mode), tempfile.TemporaryDirectory() as t:
    root=Path(t); workspace=root/"operation"/"workspace"; workspace.mkdir(parents=True)
    wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
    telemetry=D.discovery_telemetry.DiscoveryTelemetry(root/"live")
    planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                           reviewed_sources=package,telemetry=telemetry)
    def actor(**_kwargs):
     self._write_planner_artifacts(workspace,assignment,mode=mode)
     return SimpleNamespace(returncode=0,stdout="",stderr="")
    with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
         patch.object(D.codex_container_actor,"run_actor",side_effect=actor), \
         self.assertRaisesRegex(D.PlannerOutputRefusal,pattern) as caught:
     planner.plan(context={"authoring_assignment":assignment.to_dict(),
                           "planner_context":{"reviewed_source_package_sha256":
                                              package.package_sha256}},
                  workspace=workspace,checkpoint_path=root/"operation"/"actor-result.json")
    events=[json.loads(line) for line in (root/"live/planner.jsonl").read_text().splitlines()]
    self.assertEqual([row["event"] for row in events],
                     ["planner_started","planner_refused"])
    self.assertEqual(events[-1]["result"]["refusal_reason_sha256"],
                     hashlib.sha256(str(caught.exception).encode()).hexdigest())
    self.assertNotIn(str(caught.exception),json.dumps(events[-1]))
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

 def test_v19_fa_candidate_subroutes_are_typed_secret_free_and_reopen_without_actor(self):
  package=self.source_package()
  assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); workspace=root/"operation/workspace"; workspace.mkdir(parents=True)
   wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   telemetry=D.discovery_telemetry.DiscoveryTelemetry(root/"live")
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                          reviewed_sources=package,telemetry=telemetry)
   calls=0
   def actor(**_kwargs):
    nonlocal calls
    calls+=1
    self._write_planner_artifacts(workspace,assignment)
    plan=json.loads((workspace/"plan.json").read_text())
    plan["experiment_intent"]={
        "template_id":"cuda-fattn-tile-v1","target_surface":"gpu_decode",
        "target_symbol":"launch_fattn_tile_switch_ncols2",
        "correctness_id":"backend-ops-hip-v1",
        "dispatch_id":"decode-tg128-rocprof-v3",
        "expected_dispatch":[
            {"route_id":"cuda-fattn-tile-v1.gqa7_bulk_pairs",
             "kernel_name":"void flash_attn_tile<64, 64, 1, 2, false>",
             "calls":3096,"grid":3072,"workgroup":64,"lds_bytes":5120},
            {"route_id":"cuda-fattn-tile-v1.gqa7_scalar_tail",
             "kernel_name":"void flash_attn_tile<64, 64, 2, 1, false>",
             "calls":3096,"grid":1024,"workgroup":64,"lds_bytes":5120}]}
    (workspace/"plan.json").write_text(json.dumps(plan))
    return SimpleNamespace(returncode=0,stdout="",stderr="")
   context={"authoring_assignment":assignment.to_dict(),
            "planner_context":{"reviewed_source_package_sha256":package.package_sha256}}
   checkpoint=root/"operation/actor-result.json"
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=actor), \
        self.assertRaisesRegex(D.PlannerOutputRefusal,
                               "planner experiment intent violates deployed authority"):
    planner.plan(context=context,workspace=workspace,checkpoint_path=checkpoint)
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",
                     side_effect=AssertionError("completed actor replayed")), \
        self.assertRaisesRegex(D.PlannerOutputRefusal,
                               "planner experiment intent violates deployed authority"):
    planner.resume_plan(context=context,workspace=workspace,checkpoint_path=checkpoint)
   self.assertEqual(calls,1)
   rows=[json.loads(line) for line in (root/"live/planner.jsonl").read_text().splitlines()]
   self.assertEqual([row["event"] for row in rows],["planner_started","planner_refused"])
   serialized=json.dumps(rows)
   self.assertNotIn("gqa7_bulk_pairs",serialized)
   self.assertNotIn("dispatch route id is not deployed authority",serialized)
 def test_planner_refusal_survives_telemetry_schema_or_io_failure(self):
  package=self.source_package()
  assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
  for failure in (D.discovery_telemetry.TelemetryError("schema drift"),
                  OSError("telemetry disk unavailable")):
   with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as t:
    root=Path(t); workspace=root/"operation/workspace"; workspace.mkdir(parents=True)
    wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
    telemetry=Mock()
    telemetry.emit.side_effect=lambda _channel,event,**_kwargs: (
        (_ for _ in ()).throw(failure) if event == "planner_refused" else None)
    planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                           reviewed_sources=package,telemetry=telemetry)
    def actor(**_kwargs):
     self._write_planner_artifacts(workspace,assignment,
                                   mode="underestimated_diff_v16")
     return SimpleNamespace(returncode=0,stdout="",stderr="")
    with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
         patch.object(D.codex_container_actor,"run_actor",side_effect=actor), \
         self.assertRaisesRegex(D.PlannerOutputRefusal,"14 < 15") as caught:
     planner.plan(context={"authoring_assignment":assignment.to_dict(),
                           "planner_context":{"reviewed_source_package_sha256":
                                              package.package_sha256}},
                  workspace=workspace,
                  checkpoint_path=root/"operation/actor-result.json")
    self.assertEqual(caught.exception.telemetry_status,"emit_failed")
    self.assertEqual(caught.exception.telemetry_failure["type"],
                     type(failure).__name__)
 def test_actor_telemetry_is_observational_for_success_and_primary_failures(self):
  package=self.source_package()
  assignment=D.AuthoringAssignment("ak-test","akp-test","akc-test","0"*40,"1"*40)
  telemetry=Mock()
  telemetry.emit.side_effect=OSError("telemetry unavailable")
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   success_workspace=root/"success/workspace"; success_workspace.mkdir(parents=True)
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                          reviewed_sources=package,telemetry=telemetry)
   def success(**_kwargs):
    self._write_planner_artifacts(success_workspace,assignment)
    return SimpleNamespace(returncode=0,stdout="ok",stderr="")
   context={"authoring_assignment":assignment.to_dict(),
            "planner_context":{"reviewed_source_package_sha256":
                               package.package_sha256}}
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=success):
    candidate=planner.plan(
        context=context,workspace=success_workspace,
        checkpoint_path=root/"success/actor-result.json")
   self.assertEqual(candidate.hypothesis_id,"akh-planner-fault")
   failed_workspace=root/"failed/workspace"; failed_workspace.mkdir(parents=True)
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",
                     side_effect=RuntimeError("planner transport primary")), \
        self.assertRaisesRegex(RuntimeError,"planner transport primary"):
    planner.plan(context=context,workspace=failed_workspace,
                 checkpoint_path=root/"failed/actor-result.json")
   claude=root/"claude"; claude.write_bytes(b"claude"); claude.chmod(0o700)
   critic=D.ClaudeCritic(wrapper=claude,environment={"PATH":"/usr/bin"},
                         telemetry=telemetry)
   critic_result=SimpleNamespace(
       decision="accept",reason="bounded",stdout_sha256="c"*64,
       stderr_sha256="d"*64)
   with patch.object(D.claude_fable5_critic_actor,"runtime_identity",
                     return_value=CLAUDE_RUNTIME), \
        patch.object(D.claude_fable5_critic_actor,"run_critic",
                     return_value=critic_result):
    self.assertEqual(critic.review(candidate,context=context,
                                   workspace=root).decision,"accept")
   with patch.object(D.claude_fable5_critic_actor,"runtime_identity",
                     return_value=CLAUDE_RUNTIME), \
        patch.object(D.claude_fable5_critic_actor,"run_critic",
                     side_effect=RuntimeError("critic transport primary")), \
        self.assertRaisesRegex(RuntimeError,"critic transport primary"):
    critic.review(candidate,context=context,workspace=root)
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
 def test_concrete_v16_refusal_retries_are_bounded_without_science_or_compute(self):
  outer=self; package=self.source_package()
  class Never:
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     raise AssertionError("planner refusal reached critic or compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   telemetry=D.discovery_telemetry.DiscoveryTelemetry(root/"live")
   planner=D.CodexPlanner(wrapper=wrapper,environment={"PATH":"/usr/bin"},
                          reviewed_sources=package,telemetry=telemetry)
   actor_calls=[]
   def actor(**kwargs):
    prompt=json.loads(kwargs["prompt"])
    assignment=D.AuthoringAssignment(
        **prompt["context"]["authoring_assignment"])
    actor_calls.append(assignment.candidate_id)
    outer._write_planner_artifacts(
        kwargs["workspace"],assignment,mode="underestimated_diff_v16")
    return SimpleNamespace(returncode=0,stdout="",stderr="")
   base=self.portfolio_config(root,[self.portfolio_record(
       hypothesis_id="akh-v2-q5-type-specific-dequant",rank=1,budget=3)])
   planner_context={**base.planner_context,
       "reviewed_source_package_sha256":package.package_sha256}
   config=dataclasses.replace(
       base,planner_context=planner_context,
       planner_context_sha256=D._sha(planner_context))
   never=Never()
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=actor):
    result=D.run_controller(config,planner=planner,critic=never,
                            screener=never,lease=never)
   events=[json.loads(line)["event"]
           for line in (root/"live/planner.jsonl").read_text().splitlines()]
  self.assertEqual(actor_calls,
                   ["akc-discovery-1","akc-discovery-2","akc-discovery-3"])
  self.assertEqual(result["portfolio_authoring_failures"],{
      "akh-v2-q5-type-specific-dequant":3})
  self.assertEqual(result["portfolio_skips"][
      "akh-v2-q5-type-specific-dequant"]["disposition"],
      "bounded_authoring_skip")
  self.assertEqual(result["scientific_attempts"],0)
  self.assertEqual(result["terminal_reason"],"portfolio_exhausted")
  self.assertTrue(all(row["status"] == "planner_refused"
                      and row["scientific_budget_spent"] is False
                      and row["telemetry_status"] == "emitted"
                      for row in result["iterations"]))
  self.assertEqual(events,["planner_started","planner_refused"]*3)
 def test_six_bounded_planner_skips_reach_every_strategy_with_one_science_slot(self):
  class RefusingPlanner:
   def __init__(self): self.bindings=[]
   def attest(self): return {**D.SOL,"runtime":RUNTIME}
   def plan(self,*,context,workspace):
    self.bindings.append(context["authoring_assignment"]["portfolio_binding"])
    raise D.PlannerOutputRefusal("bounded authored artifact refusal")
  class Never:
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     raise AssertionError("refused portfolio reached critic or compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   records=[self.portfolio_record(
       hypothesis_id=f"akh-strategy-{rank}",rank=rank,budget=3)
       for rank in range(1,7)]
   config=dataclasses.replace(self.portfolio_config(Path(t),records),
                              max_iterations=1)
   planner=RefusingPlanner(); never=Never()
   result=D.run_controller(config,planner=planner,critic=never,
                           screener=never,lease=never)
  selected=[row["hypothesis_id"] for row in planner.bindings]
  self.assertEqual(selected,
                   [f"akh-strategy-{rank}" for _ in range(3)
                    for rank in range(1,7)])
  self.assertEqual([row["turn"] for row in result["iterations"]],
                   list(range(1,19)))
  self.assertEqual(len({row["planner_operation_key"]
                        for row in result["iterations"]}),18)
  self.assertEqual(result["scientific_attempts"],0)
  self.assertEqual(result["terminal_reason"],"portfolio_exhausted")
  self.assertTrue(result["complete"])
 def test_completed_legacy_state_reentry_does_not_insert_science_counter(self):
  class RefusingPlanner:
   def attest(self): return {**D.SOL,"runtime":RUNTIME}
   def plan(self,*,context,workspace):
    raise D.PlannerOutputRefusal("legacy completed refusal")
  class Never:
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); config=D.ControllerConfig(root,1,dry_run=True)
   result=D.run_controller(config,planner=RefusingPlanner(),
                           critic=Never(),screener=Never(),lease=Never())
   self.assertTrue(result["complete"])
   legacy=dict(result); legacy.pop("scientific_attempts")
   legacy["state_sha256"]=D._sha({
       key:value for key,value in legacy.items() if key != "state_sha256"})
   D._atomic(root/"state.json",legacy)
   before=(root/"state.json").read_bytes()
   reopened=D.run_controller(config,planner=RefusingPlanner(),
                             critic=Never(),screener=Never(),lease=Never())
   self.assertNotIn("scientific_attempts",reopened)
   self.assertEqual((root/"state.json").read_bytes(),before)
 def test_controller_persists_visibility_degraded_without_masking_refusal(self):
  outer=self; package=self.source_package()
  class BrokenTelemetry:
   def emit(self,*_args,**_kwargs):
    raise OSError("injected telemetry transaction failure")
  class Never:
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     raise AssertionError("planner refusal reached critic or compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); wrapper=root/"codex"; wrapper.write_bytes(b"codex"); wrapper.chmod(0o700)
   planner=D.CodexPlanner(
       wrapper=wrapper,environment={"PATH":"/usr/bin"},
       reviewed_sources=package,telemetry=BrokenTelemetry())
   def actor(**kwargs):
    assignment=D.AuthoringAssignment(**json.loads(
        kwargs["prompt"])["context"]["authoring_assignment"])
    outer._write_planner_artifacts(
        kwargs["workspace"],assignment,mode="underestimated_diff_v16")
    return SimpleNamespace(returncode=0,stdout="",stderr="")
   base=self.portfolio_config(root,[self.portfolio_record(
       hypothesis_id="akh-visibility-q5",rank=1,budget=3)])
   planner_context={**base.planner_context,
       "reviewed_source_package_sha256":package.package_sha256}
   config=dataclasses.replace(
       base,max_iterations=1,planner_context=planner_context,
       planner_context_sha256=D._sha(planner_context))
   never=Never()
   with patch.object(D.codex_container_actor,"runtime_identity",return_value=RUNTIME), \
        patch.object(D.codex_container_actor,"run_actor",side_effect=actor):
    result=D.run_controller(config,planner=planner,critic=never,
                            screener=never,lease=never)
  self.assertEqual(result["terminal_reason"],"portfolio_exhausted")
  self.assertEqual({item["event"] for item in result["visibility_degraded"]},
                   {"planner_started","planner_refused"})
  self.assertTrue(all(row["status"] == "planner_refused"
                      and row["visibility_degraded"] is True
                      and row["telemetry_status"] == "emit_failed"
                      for row in result["iterations"]))
 def test_visibility_degradation_does_not_mask_successful_planner_or_critic(self):
  outer=self
  class Planner:
   def __init__(self): self.telemetry_failures=[]
   def attest(self): return {**D.SOL,"runtime":RUNTIME}
   def plan(self,*,context,workspace):
    self.telemetry_failures.append({
        "event":"planner_completed","operation_key":"4"*64,
        "error_type":"OSError","error_sha256":"5"*64})
    return outer.portfolio_candidate(
        context["authoring_assignment"]["portfolio_binding"])
  class Critic:
   def __init__(self): self.telemetry_failures=[]
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def review(self,*_args,**_kwargs):
    self.telemetry_failures.append({
        "event":"critic_completed","operation_key":"6"*64,
        "error_type":"OSError","error_sha256":"7"*64})
    return D.Critique("accept","bounded")
  class Never:
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     raise AssertionError("dry run reached compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   config=dataclasses.replace(self.portfolio_config(
       Path(t),[self.portfolio_record(rank=1,budget=1)]),max_iterations=1)
   result=D.run_controller(config,planner=Planner(),critic=Critic(),
                           screener=Never(),lease=Never())
  row=result["iterations"][0]
  self.assertEqual(row["status"],"dry_run_authorized")
  self.assertTrue(row["visibility_degraded"])
  self.assertEqual({item["event"] for item in row["telemetry_failures"]},
                   {"planner_completed","critic_completed"})
  self.assertEqual({item["event"] for item in result["visibility_degraded"]},
                   {"planner_completed","critic_completed"})
 def test_v16_telemetry_terminal_recovers_checkpoint_and_rederives_refusal(self):
  outer=self
  class StopAfterRefusal(BaseException): pass
  class LegacyThenCurrentPlanner:
   def __init__(self): self.plan_calls=0; self.resume_calls=0
   def attest(self): return {**D.SOL,"runtime":RUNTIME}
   def plan(self,*,context,workspace,checkpoint_path):
    self.plan_calls+=1
    assignment=D.AuthoringAssignment(**context["authoring_assignment"])
    outer._write_planner_artifacts(
        workspace,assignment,mode="underestimated_diff_v16")
    D._seal_planner_actor_checkpoint(
        workspace,checkpoint_path,context=context,
        result={"returncode":0,"stdout_sha256":"a"*64,
                "stderr_sha256":"b"*64})
    raise D.discovery_telemetry.TelemetryError(
        "telemetry result contains a non-allowlisted field")
   def resume_plan(self,*,context,workspace,checkpoint_path):
    self.resume_calls+=1
    checkpoint=D._reopen_planner_actor_checkpoint(
        workspace,checkpoint_path,context=context)
    self.assert_checkpoint_rc=checkpoint["result"]["returncode"]
    return D._load_plan(
        workspace/"plan.json",workspace,
        assignment=D.AuthoringAssignment(**context["authoring_assignment"]))
  class Never:
   def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
   def __getattr__(self,_name):
    def called(*_args,**_kwargs):
     raise AssertionError("planner refusal reached critic or compute")
    return called
  with tempfile.TemporaryDirectory() as t:
   root=Path(t); planner=LegacyThenCurrentPlanner(); never=Never()
   config=dataclasses.replace(
       self.portfolio_config(root,[self.portfolio_record(
           hypothesis_id="akh-v2-q5-type-specific-dequant",rank=1,budget=3)]),
       max_iterations=1)
   with self.assertRaises(D.discovery_telemetry.TelemetryError):
    D.run_controller(config,planner=planner,critic=never,
                     screener=never,lease=never)
   crashed=json.loads((root/"state.json").read_text())
   self.assertEqual(crashed["planning"]["failure"],{
       "type":"TelemetryError",
       "message":"telemetry result contains a non-allowlisted field"})
   original_save=D.DurableState.save
   def stop_after_refusal(store,state,phase):
    original_save(store,state,phase)
    if phase == "planner_refused": raise StopAfterRefusal(phase)
   with patch.object(D.DurableState,"save",new=stop_after_refusal), \
        self.assertRaises(StopAfterRefusal):
    D.run_controller(config,planner=planner,critic=never,
                     screener=never,lease=never)
   result=json.loads((root/"state.json").read_text())
  self.assertEqual((planner.plan_calls,planner.resume_calls),(1,1))
  self.assertEqual(planner.assert_checkpoint_rc,0)
  self.assertEqual(result["next"],2)
  self.assertEqual(result.get("planner_provider_attempt",0),0)
  self.assertEqual(result["portfolio_authoring_failures"],{
      "akh-v2-q5-type-specific-dequant":1})
  self.assertNotIn("portfolio_terminals",result)
  row=result["iterations"][0]
  self.assertEqual(row["status"],"planner_refused")
  self.assertEqual(row["refusal_type"],"planner_output_refusal")
  self.assertFalse(row["scientific_budget_spent"])
  self.assertEqual(row["portfolio_hypothesis_id"],
                   "akh-v2-q5-type-specific-dequant")
  self.assertTrue(row["planner_checkpoint_reused"])
  self.assertEqual(row["telemetry_recovery"]["disposition"],
                   "resume_checkpoint_and_rederive_refusal")
  self.assertIn("14 < 15",row["reason"])
 def test_v16_telemetry_terminal_refuses_missing_or_changed_checkpoint_closure(self):
  # The legacy exception name alone is not recovery authority.  The private
  # rc=0 checkpoint must still bind the exact actor artifact closure.
  for mutation in ("missing_checkpoint","extra_artifact","hardlink_checkpoint",
                   "result_rc1","extra_result","missing_stdout","bad_stdout",
                   "extra_top_level","missing_top_level"):
   with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as t:
    root=Path(t); config=D.ControllerConfig(root,1,dry_run=True)
    context=D._context({"iterations":[]},D._tracker(D.DurableState(root)),1,
                       config,None)
    planning=D._planning_intent(config,turn=1,context=context,
                                portfolio_binding=None)
    workspace=Path(planning["workspace"])
    D._prepare_planner_workspace(config,planning["operation_key"],workspace)
    (workspace/"plan.json").write_text("{}")
    checkpoint=workspace.parent/"actor-result.json"
    D._seal_planner_actor_checkpoint(
        workspace,checkpoint,context=context,
        result={"returncode":0,"stdout_sha256":"a"*64,
                "stderr_sha256":"b"*64})
    planning["phase"]="actor_entering"
    planning["failure"]={"type":"TelemetryError","message":
        "telemetry result contains a non-allowlisted field"}
    if mutation == "missing_checkpoint": checkpoint.unlink()
    elif mutation == "extra_artifact": (workspace/"extra").write_text("x")
    elif mutation == "hardlink_checkpoint":
     os.link(checkpoint,workspace.parent/"checkpoint-alias")
    else:
     value=json.loads(checkpoint.read_text())
     if mutation == "result_rc1": value["result"]["returncode"]=1
     elif mutation == "extra_result": value["result"]["unexpected"]=True
     elif mutation == "missing_stdout": value["result"].pop("stdout_sha256")
     elif mutation == "bad_stdout": value["result"]["stdout_sha256"]="bad"
     elif mutation == "extra_top_level": value["unexpected_top_level"]=True
     else: value.pop("assignment_sha256")
     value["receipt_sha256"]=D._sha({
         key:item for key,item in value.items() if key != "receipt_sha256"})
     checkpoint.write_text(json.dumps(value))
    state={"schema":D.SCHEMA,"authority":D.AUTHORITY,"roster":D.sealed_roster(),
           "iterations":[],"next":1,"complete":False,"planning":planning}
    store=D.DurableState(root); store.save(state,"fixture")
    class Planner:
     def attest(self): return {**D.SOL,"runtime":RUNTIME}
    class Never:
     def attest(self): return {**D.FABLE5_CRITIC,"runtime":CLAUDE_RUNTIME}
    with self.assertRaises(D.DiscoveryControllerError):
     D.run_controller(config,planner=Planner(),critic=Never(),
                      screener=Never(),lease=Never())
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
  sources=prompt["context"]["selected_source_preimage"]
  self.assertEqual(len(sources),1)
  source=sources[0]
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
 def test_post_proof_transport_interruption_pauses_and_resumes_same_candidate(self):
  class Interrupted(FakeScreen):
   def screen(self,*args):
    self.calls+=1
    if self.calls==1:
     raise D.ResumableScreenInterruption("runner parser exit 2")
    return D.SealedScreen("receipt",H[:-1]+"1",.01,"candidate",H,H,H)
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); planner=FakePlanner(); critic=FakeCritic(["accept"]); screen=Interrupted([])
   first=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,screener=screen,lease=Lease())
   self.assertEqual(first["next"],1); self.assertEqual(first["iterations"],[])
   self.assertTrue(first["inflight"]["interruption"]["resumable"])
   second=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,screener=screen,lease=Lease())
   self.assertEqual(second["iterations"][0]["status"],"candidate")
   self.assertEqual(len(planner.calls),1); self.assertEqual(screen.calls,2)
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
   with self.assertRaisesRegex(
           D.DiscoveryControllerError, "separate target-runtime stage"):
    screen.screen(item,object(),{})
   self.assertEqual(events,["build","source","dispatch"])
   run.assert_not_called()
 def test_gpu_source_maps_timed_divergence_and_never_runs_graphs_on(self):
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest):
   root=Path(t); anchor=root/"anchor"; candidate=root/"candidate"; anchor.mkdir(); candidate.mkdir()
   source_file=root/"source.json"; dispatch_file=root/"dispatch.json"
   source_file.write_text("source"); dispatch_file.write_text("dispatch")
   refusal_file=root/"correctness-divergence.json"
   refusal_file.write_text(json.dumps({"status":"correctness_falsified"}))
   refusal_sha=hashlib.sha256(refusal_file.read_bytes()).hexdigest()
   build=D.GpuSourceBuild(
       anchor,candidate,
       D.gpu_source_proofs.BuildIdentity("commit-a",H,H,H,H,H),
       D.gpu_source_proofs.BuildIdentity("commit-b","b"*64,"b"*64,"b"*64,"b"*64,"b"*64))
   item=D.PlannedCandidate("akh-a","statement","falsifier",{}, {"id":"p"},Manifest(),H)
   off=root/"off"; on=root/"on"; off.mkdir(); on.mkdir()
   target=argparse.Namespace(
       factor="source_patch",anchor_build=str(anchor),candidate_build=str(candidate),
       output_dir=str(on))
   args=argparse.Namespace(
       factor="source_patch",anchor_build=str(anchor),candidate_build=str(candidate),
       output_dir=str(off),_target_runtime_args=target)
   source_hash=hashlib.sha256(source_file.read_bytes()).hexdigest()
   dispatch_hash=hashlib.sha256(dispatch_file.read_bytes()).hexdigest()
   comparison={"relative_improvement_fraction":.01}
   material={"manifest_sha256":H,"candidate":build.candidate_identity,
             "anchor":build.anchor_identity,"workload_sha256":H,
             "correctness":{"file_sha256":source_hash,"native_sha256":H},
             "attribution":{"file_sha256":dispatch_hash,"native_sha256":H,
                            "body":{"exact_duration_comparison":comparison}}}
   hashed={**material,"candidate":build.candidate_identity.__dict__,
           "anchor":build.anchor_identity.__dict__}
   bundle=D.gpu_source_proofs.GpuSourceProofBundle(
       **material,bundle_sha256=D.gpu_source_proofs._hash(hashed))
   screen=D.GpuSourceScreener(
       build_source=lambda *_:build,proof_bundle=lambda *_:bundle,
       args_factory=lambda *_:args)
   native=D.gpu_discovery.CandidateCorrectnessDivergence(
       "candidate timed outputs differ bitwise from the sealed anchor",
       receipt_path=str(refusal_file.resolve()),receipt_sha256=refusal_sha,
       result_sha256="e"*64, operation_key="9"*64)
   with patch.object(D.gpu_discovery,"run",side_effect=native) as run, \
       self.assertRaises(D.TimedOutputCorrectnessRefusal) as raised:
    screen.screen(item,object(),{})
   self.assertEqual(run.call_count,1)
   self.assertEqual(raised.exception.receipt_sha256,refusal_sha)
   self.assertEqual(raised.exception.result_sha256,"e"*64)
   self.assertEqual(raised.exception.operation_key,"9"*64)
   self.assertTrue(raised.exception.scientific_budget_spent)
 def test_lease_wait_is_durable_without_spending_iteration(self):
  class Wait:
   def __init__(self): self.calls=0
   def admit(self,item,*,operation_key):
    self.calls+=1
    return {"admitted":False,"phase":"prebuild_probe",
            "reason":"device_busy","detail":"CPU window busy",
            "operation_key":operation_key,"promotion_claim":False,
            "mode":"cold_serialized","device_id":"mi210_0",
            "inference_window_lock":"/tmp/test-inference-window.lock",
            "model_sha256":H,"load_admission":{"decision_sha256":H}}
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); planner=FakePlanner(); critic=FakeCritic(["accept"])
   screen=FakeScreen([.1]); lease=Wait()
   r=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                      screener=screen,lease=lease)
   self.assertEqual(r["next"],1); self.assertEqual(r["pending"]["row"]["status"],"waiting_resource"); self.assertFalse(r["complete"])
   reopened=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                             screener=screen,lease=lease)
   self.assertEqual((reopened["scientific_attempts"],reopened["next"]),(0,1))
   self.assertEqual((lease.calls,len(planner.calls),screen.calls),(2,1,0))
 def test_postbuild_resource_wait_retries_exact_candidate_without_replanning(self):
  class Race(FakeScreen):
   def __init__(self,root):
    super().__init__([]); self.root=root; self.wait_receipt=None; self.operations=[]
   def screen(self,*args):
    self.calls+=1; self.operations.append(args[2]["operation_key"])
    if self.calls == 1:
     operation_key=args[2]["operation_key"]; directory=self.root/operation_key/"resource-waits"; directory.mkdir(parents=True,mode=0o700)
     contention={"admitted":False,"phase":"pre_executor_reservation","reason":"device_busy","operation_key":operation_key,"promotion_claim":False}
     body={"schema":"epyc.autokernel.gpu_source_resource_wait.v1","authority":D.AUTHORITY,"promotion_claim":False,"operation_key":operation_key,"gpu_executor_started":False,"proof_root_created":False,"runner_plan_created":False,"runner_output_created":False,"contention":contention}; body["receipt_sha256"]=D._sha(body)
     path=directory/"wait-0001.json"; path.write_text(json.dumps(body,sort_keys=True)); digest=hashlib.sha256(path.read_bytes()).hexdigest()
     self.wait_receipt={**contention,"stage_receipt_path":str(path),"stage_receipt_sha256":digest}
     raise D.ResourceWait("device race",receipt=self.wait_receipt)
    return D.SealedScreen("receipt",H,.04,"candidate",H,H,H)
   def reconcile(self,inflight):
    return D.Recovery("resource_wait",wait_receipt=self.wait_receipt)
  class TrackingLease(Lease):
   def __init__(self): self.admits=[]; self.resumes=[]
   def admit(self,item,*,operation_key):
    self.admits.append(operation_key)
    return super().admit(item,operation_key=operation_key)
   def resume(self,item,permit):
    self.resumes.append(dict(permit))
    if len(self.resumes) == 1:
     return {"admitted":False,"reason":"foreign_kfd_busy",
             "operation_key":permit["operation_key"]}
    return Lease.admit(self,item,operation_key=permit["operation_key"])
  class CountingCritic(FakeCritic):
   def __init__(self): super().__init__(["accept"]); self.calls=0
   def review(self,*args,**kwargs):
    self.calls+=1
    return super().review(*args,**kwargs)
  with tempfile.TemporaryDirectory() as t, patch.object(D.source_candidate,"SourcePatchManifest",Manifest), patch.object(D,"_write_projection"):
   root=Path(t); planner=FakePlanner(); critic=CountingCritic(); screen=Race(root/"operations"); lease=TrackingLease()
   waiting=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,screener=screen,lease=lease)
   self.assertEqual((waiting["next"],waiting["pending"]["row"]["status"]),(1,"waiting_resource"))
   self.assertEqual((waiting["scientific_attempts"],waiting["iterations"]),(0,[]))
   checkpoint=waiting["pending"]["resource_wait"]
   self.assertEqual(checkpoint["schema"],D.RESOURCE_WAIT_CHECKPOINT_SCHEMA)
   self.assertTrue(checkpoint["inflight"]["lease"]["admitted"])
   self.assertEqual(checkpoint["wait_receipt"],screen.wait_receipt)
   tampered=json.loads(json.dumps(waiting))
   tampered["pending"]["resource_wait"]["wait_receipt_sha256"]="0"*64
   D.DurableState(root/"out").save(tampered,"test_tampered_wait")
   with self.assertRaisesRegex(
       D.DiscoveryControllerError,"resource-wait checkpoint"):
    D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                     screener=screen,lease=lease)
   self.assertEqual((len(planner.calls),critic.calls,screen.calls,
                     len(lease.resumes)),(1,1,1,0))
   D.DurableState(root/"out").save(waiting,"test_restore_wait")
   still_waiting=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                                  screener=screen,lease=lease)
   self.assertEqual(still_waiting["pending"]["resource_wait"],checkpoint)
   self.assertEqual((still_waiting["scientific_attempts"],still_waiting["next"]),(0,1))
   self.assertEqual((len(planner.calls),critic.calls,screen.calls),(1,1,1))
   done=D.run_controller(self.cfg(root,1),planner=planner,critic=critic,screener=screen,lease=lease)
   self.assertTrue(done["complete"])
   self.assertEqual((done["scientific_attempts"],done["next"]),(1,2))
   self.assertEqual((len(planner.calls),critic.calls,screen.calls),(1,1,2))
   self.assertEqual((len(lease.admits),len(lease.resumes)),(1,2))
   self.assertEqual(screen.operations,[lease.admits[0],lease.admits[0]])
   self.assertEqual(lease.resumes,[checkpoint["resume_permit"]]*2)
 def test_postbuild_resource_wait_partial_or_resealed_checkpoint_fails_closed(self):
  class Race(FakeScreen):
   def __init__(self,root):
    super().__init__([]); self.root=root; self.wait_receipt=None
   def screen(self,*args):
    self.calls+=1
    if self.calls != 1:
     raise AssertionError("partial wait checkpoint re-entered the screener")
    operation_key=args[2]["operation_key"]
    directory=self.root/operation_key/"resource-waits"
    directory.mkdir(parents=True,mode=0o700)
    contention={"admitted":False,"phase":"pre_executor_reservation",
                "reason":"device_busy","device_id":"mi210_0",
                "operation_key":operation_key,"promotion_claim":False}
    body={"schema":"epyc.autokernel.gpu_source_resource_wait.v1",
          "authority":D.AUTHORITY,"promotion_claim":False,
          "operation_key":operation_key,"gpu_executor_started":False,
          "proof_root_created":False,"runner_plan_created":False,
          "runner_output_created":False,"contention":contention}
    body["receipt_sha256"]=D._sha(body)
    path=directory/"wait-0001.json"
    path.write_text(json.dumps(body,sort_keys=True))
    self.wait_receipt={**contention,"stage_receipt_path":str(path),
                       "stage_receipt_sha256":
                           hashlib.sha256(path.read_bytes()).hexdigest()}
    raise D.ResourceWait("device race",receipt=self.wait_receipt)
   def reconcile(self,inflight):
    return D.Recovery("resource_wait",wait_receipt=self.wait_receipt)
  class TrackingLease(Lease):
   def __init__(self): self.admits=[]; self.resumes=[]
   def admit(self,item,*,operation_key):
    self.admits.append(operation_key)
    return super().admit(item,operation_key=operation_key)
   def resume(self,item,permit):
    self.resumes.append(dict(permit))
    raise AssertionError("malformed wait reached lease.resume")
  mutations={
      "deleted_checkpoint": lambda value:
          value["pending"].pop("resource_wait"),
      "null_checkpoint": lambda value:
          value["pending"].__setitem__("resource_wait",None),
      "missing_field": lambda value:
          value["pending"]["resource_wait"].pop("wait_receipt_sha256"),
      "extra_field": lambda value:
          value["pending"]["resource_wait"].__setitem__("extra",True),
      "coherently_resealed_wait": self._reseal_resource_wait_mutation,
      "deleted_stage_receipt": None,
  }
  for label,mutate in mutations.items():
   with self.subTest(label=label), tempfile.TemporaryDirectory() as t, \
       patch.object(D.source_candidate,"SourcePatchManifest",Manifest), \
       patch.object(D,"_write_projection"):
    root=Path(t); planner=FakePlanner(); critic=FakeCritic(["accept"])
    screen=Race(root/"operations"); lease=TrackingLease()
    waiting=D.run_controller(
        self.cfg(root,1),planner=planner,critic=critic,
        screener=screen,lease=lease)
    tampered=json.loads(json.dumps(waiting))
    if mutate is None:
     Path(tampered["pending"]["resource_wait"]
          ["wait_receipt"]["stage_receipt_path"]).unlink()
    else:
     mutate(tampered)
    D.DurableState(root/"out").save(tampered,"test_partial_wait")
    with self.assertRaises(D.DiscoveryControllerError):
     D.run_controller(self.cfg(root,1),planner=planner,critic=critic,
                      screener=screen,lease=lease)
    self.assertEqual((len(planner.calls),screen.calls,
                      len(lease.admits),len(lease.resumes)),(1,1,1,0))
 @staticmethod
 def _reseal_resource_wait_mutation(value):
  checkpoint=value["pending"]["resource_wait"]
  wait=checkpoint["wait_receipt"]
  wait["reason"]="device_free"
  checkpoint["wait_receipt_sha256"]=D._sha(wait)
  checkpoint["resume_permit"]={**checkpoint["inflight"]["lease"],**wait}
  value["pending"]["row"]["lease"]=dict(wait)
  checkpoint["checkpoint_sha256"]=D._sha({
      key:item for key,item in checkpoint.items()
      if key != "checkpoint_sha256"})
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
