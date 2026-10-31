"""Model-free pipeline control: time the same integer summation twice."""
import json
import sys
import time

prompt = sys.stdin.read()
if not prompt:
    raise SystemExit("prompt required")
start = time.perf_counter_ns()
a = sum(range(50000))
middle = time.perf_counter_ns()
b = sum(range(50000))
end = time.perf_counter_ns()
print(json.dumps({"baseline_ns": middle - start, "candidate_ns": end - middle,
                  "correctness": a == b == 1249975000,
                  "protocol_id": "ak-pm12-model-free-control/v1"}))
