import json, sys
sys.path.insert(0, "/mnt/raid0/llm/tmp/ds41-w5-20261021/research-wt/scripts/benchmark")
import ds41_1017_capture as C
D = "/mnt/raid0/llm/tmp/ds41-w5-20261021/research-wt/data/ds41-1017-w5-20261021"
R = D + "/raw"; B = "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin"
REG = {"role": "build", "cpu_list": "0-23", "quarter": "q0"}
def pplv(arm):
    for l in open(f"{R}/ppl-{arm}.log"):
        if "Final estimate" in l:
            v = l.split("PPL = ")[1].split(); return float(v[0]), float(v[2])
for arm in ("1017", "sept-q4"):
    v, e = pplv(arm)
    C.capture(D, gate="ppl", arm=arm, metric="ppl_wikitext2_test_c512_chunks32", value=v, unit="ppl",
              metric_direction="lower_better", claim=f"DS41 {arm} wiki.test.raw ppl c=512 32 chunks = {v} +/- {e} (port 3376147f8, -t 24, q0)",
              scored=f"{R}/ppl-{arm}.log", recipe={"argv": open(f"{R}/ppl-{arm}.argv").read().split(), "env": json.load(open(f"{R}/coh-{arm}.recipe.json"))["env"]},
              region=REG, binary=f"{B}/llama-perplexity", reps=32, reps_basis="chunks (one run)", extra={"stderr": e})
    rec = json.load(open(f"{R}/coh-{arm}.recipe.json"))
    C.capture(D, gate="load", arm=arm, metric="split_load_ok", value=1, unit="bool", metric_direction="higher_better",
              claim=f"DS41 {arm} loads in the port (server /health ok)" + ("; 1046 tensors, deepseek41, alignment 32, 9 shards" if arm == "1017" else ""),
              scored=f"{R}/coh-{arm}.server.log", recipe=rec, region=REG, binary=f"{B}/llama-server", reps=1, reps_basis="one load")
v = json.load(open(f"{R}/coherence_verdict.json"))["aggregate"]
C.capture(D, gate="coherence", arm="1017", metric="coherence_gate_regressions_vs_sept_q4", value=v["regressions"], unit="items",
          metric_direction="lower_better", claim=f"DS41 1017 vs Sept Q4 paired coherence_gate: gate {v['gate']}, {v['regressions']} regressions / {v['n']} items (7-2041 prompt tokens); shared empty-EOS quirk on 'Answer:' after article prompts",
          scored=f"{R}/coherence_verdict.json", recipe=json.load(open(f"{R}/coh-1017.recipe.json")), region=REG, binary=f"{B}/llama-server",
          reps=5, reps_basis="prompts", extra={"aggregate": v})
print(open(D + "/belief_measurements.jsonl").read().count("\n"))
