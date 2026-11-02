# KVU-16a load-only prospective producer

This is a candidate for a new run, not evidence from November 2. It uses the
current pinned orchestrator's real `architect_critic` compiler and changes only
port and slot path. It never submits a generation request. Run it from a clean,
pinned orchestrator checkout with that checkout on `PYTHONPATH` and its
Python environment; retain the checkout and output directory. First select the
port, number of healthy samples, and maximum permitted sampling gap in the
measurement plan. The output directory must not exist. For example, replace
all placeholders with the predeclared values:

```sh
cd /path/to/pinned/epyc-orchestrator
PYTHONPATH="$PWD" /path/to/orchestrator/.venv/bin/python /path/to/producer.py \
  --belief-adapter /path/to/epyc-root/scripts/vidya/adapters/kvu16a_native.py \
  --output-dir /path/to/new/private/output --port PORT \
  --max-gap-seconds MAX_GAP --healthy-samples COUNT
sha256sum /path/to/new/private/output/plan.json
```

Execution requires the same checkout, values and rendered launch, the recorded
plan hash, and the project's existing GPU grant (`FILLER_NEEDGPU`,
`FILLER_DEADLINE`, `FILLER_STOPFILE`, `FILLER_PIDFILE`). Add
`--run-plan-sha256 PLAN_SHA256` to the same command inside the grant. The
producer hashes model files when planning and again when validating live
identity, which can take substantial time; account for it in the grant deadline.
It rechecks the binary, models, template, exact process argv and selected
runtime environment, and hashes mapped ggml/llama libraries. It verifies owned
child cleanup. Incomplete KFD reads and uninitialized own residency are unknown,
not zero. Sampling coverage is false if any gap exceeds the predeclared limit,
any own residency sample is unknown, the target KV log line does not match, or
cleanup and other checks fail.

The post-health sample maximum minus the target KV allocation is a **sampled
non-KV value**. Neither this value nor the largest startup sample is a measured
lifetime peak. This producer does not itself authorize a registry change. A real
run requires independent review of its plan, evidence, device lease and belief
capture; no real model was loaded during its development tests.

Tests: `python3 -m unittest -v test_producer` in this directory. The HTTP test
starts and terminates only its own Python server; it does not load a model.
