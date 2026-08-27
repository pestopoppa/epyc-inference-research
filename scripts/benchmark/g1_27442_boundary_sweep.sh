#!/bin/bash
# g1_27442_boundary_sweep.sh — SC49 G1: #27442 greedy long-prefill boundary sweep (CPU)
#
# Reproduces the upstream #27442 exposure question on OUR frozen v9 CPU path:
# at long cold prefills, does the model emit a VALID EOS as its FIRST sampled
# token? Two prompt classes (repeated-pangram filler = NEGATIVE CONTROL, and a
# semantically meaningful document ending with a real instruction), five target
# prompt lengths (15401 / 16501 / 17601 / 19801 / 23981 tokens), greedy
# (--temp 0, fixed seed), cache_prompt=false, -np 1, no speculation.
#
# THE BINARY (verified against the frozen production tree, no execution needed):
#   llama-completion is used instead of llama-cli: in the frozen v9 build the
#   llama-cli is an interactive chat client whose UI never emits token ids or a
#   prompt-token count, while llama-completion (same frozen build dir) supports
#   --verbose-prompt (reports the exact prefilled token count) and --special
#   (renders the first sampled token as its name, which llama-tokenize maps back
#   to an id). All three live in <tree>/build/bin from the SAME frozen build.
#
# Prompt construction: llama-tokenize --show-count gives the EXACT token count
# of any candidate prompt (vocab-only load, no weights), so prompts are built to
# the target length by iteration (bounded budget) and the ACTUAL prefilled count
# is re-recorded from the run's own --verbose-prompt line. Both the estimated
# target and the run-reported actual are carried in the trial record.
#
# Output (SC49 G1 contract, one JSONL row per trial, EXACTLY these fields):
#   prompt_length_target, prompt_length_actual, prompt_class,
#   first_sampled_token_id, stop_reason, seed, trial_ts_utc
# Plus run_manifest.json — the attestation the adapter sha256s:
#   schema, protocol_id (epyc.g1_27442.boundary_sweep.v1), date, binary_path,
#   binary_sha256, model_path, model_sha256, research_commit, launch, trials_file,
#   trials_sha256, manifest_sha256 (self-hash over everything else).
#
# Operation hygiene (house rules):
#   * Resumable — a trial whose (target, class) row already exists is skipped.
#   * Refuses to run while another g1 sweep holds the flock at
#     <research>/data/.g1-27442-sweep.lock.
#   * Kills ONLY its own PIDs (tracked children), verifies death (ps -p),
#     escalates TERM -> KILL. Never pkill/pgrep by name.
#   * No compute here: assertions are git rev-parse, --version, file existence.
#
# Run: bash scripts/benchmark/g1_27442_boundary_sweep.sh

set -euo pipefail

# ============================================================
# Configuration (env-overridable)
# ============================================================

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/mnt/raid0/llm/llama.cpp}"
EXPECTED_COMMIT="0db32c06e3e550065b78311a6031ef3dd2c4f27c"   # production-consolidated-v9
EXPECTED_VERSION_LINE="version: 10125 (0db32c06e)"
BIN="${LLAMA_CPP_DIR}/build/bin/llama-completion"
TOKENIZER="${LLAMA_CPP_DIR}/build/bin/llama-tokenize"

# The frontdoor role's model, per the orchestrator compiled registry
# (orchestration/model_registry.yaml frontdoor entry + stack_priors.yaml
# requirements.model_path): /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
MODEL="${G1_MODEL:-/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf}"

SEED="${G1_SEED:-27442}"            # fixed seed for greedy determinism
CTX_SIZE="${G1_CTX_SIZE:-32768}"    # bounded; matches serving KV quant, see below
KV_K="${G1_KV_K:-q8_0}"             # frontdoor serves kv k:q8_0 v:q8_0 (registry)
KV_V="${G1_KV_V:-q8_0}"
THREADS="${G1_THREADS:-}"           # empty = llama default (all cores)

PROTOCOL_ID="epyc.g1_27442.boundary_sweep.v1"
MANIFEST_SCHEMA="epyc.g1_27442.run_manifest.v1"

TARGETS=(15401 16501 17601 19801 23981)
CLASSES=(pangram meaningful)
TOKENIZE_BUDGET=45                 # bounded llama-tokenize calls per trial

RESEARCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${RESEARCH_DIR}/data"
RUN_NAME="g1-27442-$(date -u +%Y%m%dT%H%MZ)"
OUT_DIR="${DATA_DIR}/${RUN_NAME}"
TRIALS="${OUT_DIR}/trials.jsonl"
MANIFEST="${OUT_DIR}/run_manifest.json"
LOCK_FILE="${DATA_DIR}/.g1-27442-sweep.lock"

CHILD_PIDS=()

# ============================================================
# Kill only our own children, and verify death
# ============================================================

kill_children() {
  local pid sig
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for _ in 1 2 3 4 5; do
    local alive=0
    for pid in "${CHILD_PIDS[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then alive=1; fi
    done
    [ "${alive}" -eq 0 ] && break
    sleep 1
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "g1_27442: escalating TERM -> KILL for ${pid}" >&2
      kill -KILL "${pid}" 2>/dev/null || true
      for _ in 1 2 3; do
        if ! kill -0 "${pid}" 2>/dev/null; then break; fi
        sleep 1
      done
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      echo "g1_27442: ERROR child ${pid} survived SIGKILL" >&2
    fi
  done
}

trap 'kill_children' EXIT INT TERM

# ============================================================
# Identity assertions (no compute): frozen tree + binary + model
# ============================================================

for f in "${BIN}" "${TOKENIZER}" "${MODEL}"; do
  [ -f "$f" ] || { echo "g1_27442: missing ${f}" >&2; exit 1; }
done

HEAD="$(git -C "${LLAMA_CPP_DIR}" rev-parse HEAD 2>/dev/null || true)"
if [ "${HEAD}" != "${EXPECTED_COMMIT}" ]; then
  echo "g1_27442: llama.cpp tree is NOT at the frozen v9 commit" >&2
  echo "  expected ${EXPECTED_COMMIT}, got ${HEAD:-<no git>}" >&2
  echo "  refusing to sweep against a non-production tree" >&2
  exit 1
fi

VERSION_LINE="$(env LD_LIBRARY_PATH="$(dirname "${BIN}")" LANG=C LC_ALL=C \
  "${BIN}" --version 2>&1 | head -n 1 || true)"
if [ "${VERSION_LINE}" != "${EXPECTED_VERSION_LINE}" ]; then
  echo "g1_27442: binary version mismatch" >&2
  echo "  expected '${EXPECTED_VERSION_LINE}', got '${VERSION_LINE:-<no output>}'" >&2
  exit 1
fi

if [ "${CTX_SIZE}" -lt $((23981 + 4)) ]; then
  echo "g1_27442: CTX_SIZE ${CTX_SIZE} too small for the longest target (23981)" >&2
  exit 1
fi

# Single-runner lock: another g1 sweep running -> refuse (never block).
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "g1_27442: another g1 sweep holds ${LOCK_FILE} — refusing to run" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# ============================================================
# Trial helpers
# ============================================================

trial_done() { # target class
  [ -f "${TRIALS}" ] || return 1
  python3 - "${TRIALS}" "$1" "$2" <<'PYEOF' && return 0 || return 1
import json, sys
path, target, cls = sys.argv[1], int(sys.argv[2]), sys.argv[3]
for line in open(path):
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    if int(row.get("prompt_length_target", -1)) == target and \
       row.get("prompt_class") == cls:
        sys.exit(0)
sys.exit(1)
PYEOF
}

build_prompt() { # target class -> prints ACTUAL token count of the built prompt
  python3 - "${TOKENIZER}" "${MODEL}" "$1" "$2" "${OUT_DIR}/prompt-$1-$2.txt" \
    <<'PYEOF'
import random
import subprocess
import sys

TOKENIZER, MODEL, TARGET, CLASS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

SEED = 27442  # fixed: deterministic, reproducible prompt construction

def count(text):
    p = subprocess.run(
        [TOKENIZER, "-m", MODEL, "--stdin", "--show-count", "--no-escape"],
        input=text.encode("utf-8"), capture_output=True, timeout=300)
    if p.returncode != 0:
        raise SystemExit(f"tokenizer failed: {p.stderr.decode(errors='replace')}")
    for line in p.stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("Total number of tokens: "):
            return int(line.rsplit(" ", 1)[1])
    raise SystemExit("tokenizer did not report a count")

# --- class-specific, deterministic content (seed 27442) ---------------------
PANG_RMS = [
    "The quick brown fox jumps over the lazy dog, and then a quiet vixen waits by the river.",
    "Pack my box with five dozen liquor jugs while the jinxed wizard keeps flipping that vexed fish.",
    "How vexingly quick daft zebras jump over the boxcar that sits beside the frozen quay wall.",
    "Sphinx of black quartz, judge my vow, and do not let the wry giant break that strange drum.",
    "The five boxing wizards jump quickly beneath the luminous moon above the tranquil bayou.",
]

DOC_PARAGRAPHS = [
    ("Long-context inference servers keep a key-value cache of every token position "
     "they have seen. A cold request with no cache reuse must prefill the full prompt "
     "before the first token can be sampled, and the cache state after that prefill is "
     "the only input the decoder has. Any defect that appears only at long prefills is "
     "therefore invisible in short-prompt regression suites, which is why boundary "
     "sweeps deliberately exercise lengths far beyond the unit-test envelope."),
    ("The prefill phase computes attention states for every position in parallel and "
     "writes them into the cache. Greedy decoding then selects the single most probable "
     "token for the next position. When the sampled distribution at the final prompt "
     "position is dominated by the end-of-sequence token, the model produces an empty "
     "response: generation stops before a single content token is emitted, which "
     "surfaces to the caller as a blank completion."),
    ("An empty response after a very long prefill has three plausible readings. The "
     "first is a real defect: the model genuinely assigns overwhelming probability to "
     "the end token once the prompt exceeds some length. The second is degenerate "
     "input: the prompt itself is unstructured filler, so the model has no instruction "
     "to follow and exits early. The third is a harness artefact: the cache was not "
     "really empty, or sampling was not really greedy, so the run does not measure "
     "what it claims to measure."),
    ("Repeated-filler prompts are the standard control for the second reading. When "
     "the same sentence is concatenated thousands of times, the model sees a text "
     "with no coherent intent, and ending the response immediately is plausible model "
     "behaviour on degenerate input rather than evidence of a length-dependent "
     "defect. A control of this kind must never be scored as a quality signal; it "
     "only calibrates what the model does when there is nothing to answer."),
    ("A semantically meaningful document controls for the first reading. A coherent "
     "multi-paragraph technical text that ends with a direct instruction gives the "
     "model a real task, so an early end-of-sequence token under those conditions is "
     "not degenerate-input behaviour but a genuine failure to engage with the "
     "instruction. The instruction must come last, so the final prompt position is "
     "exactly the position where the model must decide whether to continue."),
    ("Deterministic reproduction is a precondition for diagnosis. Greedy sampling "
     "with a fixed seed removes every random choice from the decode, so re-running a "
     "trial must reproduce the same first token byte for byte. Prompt construction "
     "must be deterministic for the same reason: the exact prompt text, not merely "
     "its token count, is part of the trial identity."),
    ("The tokenizer is the source of truth for prompt length. Counting characters "
     "and dividing by a per-character average is only an estimate, because the "
     "vocabulary maps some substrings to single tokens and others to several. The "
     "reliable procedure is to tokenize the candidate prompt and read the exact "
     "token count, then adjust the text until the count lands on the target, "
     "recording the achieved count alongside the target in the trial record."),
    ("Measurement discipline requires that the recorded count come from the run "
     "itself, not from the construction tool. The generation binary reports how many "
     "prompt tokens it actually prefilled, and that reported number is what the "
     "trial record carries as the actual length. Any discrepancy between the "
     "constructed count and the prefilled count is itself a finding worth "
     "preserving, because it means the two tools tokenized the same bytes "
     "differently."),
    ("Hosting constraints matter when a sweep shares a machine. The binary must be "
     "the frozen production build, verified by its version line, because a drifted "
     "build makes the whole sweep unreproducible. The sweep must also refuse to "
     "start when another sweep is already running, and it must only ever kill the "
     "processes it started itself, checking each one is actually gone before "
     "continuing."),
]

DOC_TASK = (
    "Based on the document above, answer the following question. State your answer "
    "in at most three sentences, and begin your response with the word \"Answer:\". "
    "What distinguishes a length-dependent defect in long-context decoding from "
    "degenerate-input behaviour, and why does a repeated-filler prompt fail to "
    "distinguish them?"
)

def pangram_text(k):
    rng = random.Random(SEED)
    order = list(PANG_RMS)
    rng.shuffle(order)
    sentences = [order[i % len(order)] for i in range(k)]
    lines, line = [], []
    for i, s in enumerate(sentences):
        line.append(s)
        if (i + 1) % 10 == 0:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    return "\n\n".join(lines)

def doc_text(k):
    rng = random.Random(SEED)
    order = list(range(len(DOC_PARAGRAPHS)))
    rng.shuffle(order)
    mid = [DOC_PARAGRAPHS[order[i % len(order)]] for i in range(k)]
    return ("Long-Context Inference: Prefill Boundaries and First-Token Behaviour\n\n"
            + "Abstract. This note collects the background for a boundary sweep of "
            "first-token behaviour after cold long prefills.\n\n"
            + "\n\n".join(mid)
            + "\n\n" + DOC_TASK)

units_for = {"pangram": pangram_text, "meaningful": doc_text}
if CLASS not in units_for:
    raise SystemExit(f"unknown prompt_class {CLASS!r}")
build = units_for[CLASS]

# --- iterate to the target token count (bounded, deterministic) --------------
GRAN = 12 if CLASS == "pangram" else 25   # unit granularity in tokens
k = 1
actual = count(build(k))
best = (abs(actual - TARGET), k, actual)
budget = TOKENIZE_BUDGET
while budget > 0 and actual != TARGET:
    budget -= 1
    if actual > TARGET:
        step = max(1, (actual - TARGET) // GRAN)
        k -= step
    else:
        step = max(1, (TARGET - actual) // GRAN)
        k += step
    k = max(1, k)
    actual = count(build(k))
    cand = (abs(actual - TARGET), k, actual)
    if cand < best:
        best = cand

# final local search around the best k (k-3..k+3), deterministic tie-break
for kk in range(max(1, best[1] - 3), best[1] + 4):
    cand = (abs(count(build(kk)) - TARGET), kk, count(build(kk)))
    if cand < best:
        best = cand

final_k, final_actual = best[1], best[2]
text = build(final_k)
if text.endswith("\n"):
    text = text.rstrip("\n")
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(text)
print(final_actual)
PYEOF
}

parse_run() { # stdout_file stderr_file -> prints "stop_reason<TAB>first_token_id<TAB>prompt_tokens"
  python3 - "$1" "$2" "${TOKENIZER}" "${MODEL}" <<'PYEOF'
import subprocess, sys

stdout_path, stderr_path, TOKENIZER, MODEL = sys.argv[1:5]
out = open(stdout_path, encoding="utf-8", errors="replace").read()
err = open(stderr_path, encoding="utf-8", errors="replace").read()

import re
m = re.findall(r"number of tokens in prompt = (\d+)", err)
prompt_tokens = m[-1] if m else ""

# stdout carries ONLY generated text + "[end of text]" marker + perf tail
# (all LOG_INF noise goes to stderr). The perf block starts after the last "\n\n".
region = out.rsplit("\n\n", 1)[0]
eog_marker = "[end of text]" in region
piece = region.split("[end of text]", 1)[0].strip()
stop = "eog" if eog_marker else "completed"

first_id = ""
if piece:
    p = subprocess.run(
        [TOKENIZER, "-m", MODEL, "-p", piece, "--ids", "--special",
         "--no-bos", "--no-escape"],
        capture_output=True, timeout=300, text=True)
    ids = re.findall(r"\d+", p.stdout)
    if len(ids) == 1:
        first_id = ids[0]
    elif not p.stdout.strip() or p.stdout.strip() == "[]":
        pass  # empty piece rendered no token id: honest null
    else:
        sys.stderr.write(f"g1_27442: first-token round-trip ambiguous "
                         f"(piece {piece!r} -> {p.stdout.strip()!r}); recording null\n")

print(f"{stop}\t{first_id}\t{prompt_tokens}")
PYEOF
}

append_trial() { # target actual class first_id stop ts
  python3 - "${TRIALS}" "$@" <<'PYEOF'
import json, sys
path = sys.argv[1]
target, actual = int(sys.argv[2]), int(sys.argv[3])
cls, first_id, stop, seed, ts = sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]), sys.argv[8]
row = {
    "prompt_length_target": target,
    "prompt_length_actual": actual,
    "prompt_class": cls,
    "first_sampled_token_id": None if first_id == "" else int(first_id),
    "stop_reason": stop,
    "seed": seed,
    "trial_ts_utc": ts,
}
assert set(row) == {
    "prompt_length_target", "prompt_length_actual", "prompt_class",
    "first_sampled_token_id", "stop_reason", "seed", "trial_ts_utc",
}, "SC49 G1 row must carry EXACTLY the seven contract fields"
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, sort_keys=True) + "\n")
PYEOF
}

# ============================================================
# The sweep
# ============================================================

echo "g1_27442: sweep -> ${OUT_DIR}"
echo "g1_27442: binary ${BIN} (${VERSION_LINE})"
echo "g1_27442: model  ${MODEL}"

for target in "${TARGETS[@]}"; do
  for cls in "${CLASSES[@]}"; do
    if trial_done "${target}" "${cls}"; then
      echo "g1_27442: skip ${target}/${cls} (row already recorded)"
      continue
    fi
    echo "g1_27442: building ${cls} prompt at target ${target} tokens"
    actual="$(build_prompt "${target}" "${cls}")"

    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    run_out="${OUT_DIR}/run-${target}-${cls}.stdout"
    run_err="${OUT_DIR}/run-${target}-${cls}.stderr"
    echo "g1_27442: trial ${target}/${cls} (built ${actual} tokens), cold prefill + 1 greedy token"

    set +e
    "$BIN" -m "${MODEL}" -f "${OUT_DIR}/prompt-${target}-${cls}.txt" \
      -n 1 --temp 0 -s "${SEED}" --no-cache-prompt -no-cnv -np 1 \
      -c "${CTX_SIZE}" --cache-type-k "${KV_K}" --cache-type-v "${KV_V}" \
      --special --no-display-prompt --verbose-prompt --no-escape \
      ${THREADS:+--threads "${THREADS}"} \
      >"${run_out}" 2>"${run_err}" &
    child_pid=$!
    CHILD_PIDS+=("${child_pid}")
    wait "${child_pid}"
    rc=$?
    set -e
    CHILD_PIDS=("${CHILD_PIDS[@]/${child_pid}}")

    if [ "${rc}" -ne 0 ]; then
      echo "g1_27442: trial ${target}/${cls} FAILED rc=${rc}; run log: ${run_err}" >&2
      echo "g1_27442: rerun the same command to resume (no row was written)" >&2
      exit 1
    fi

    parsed="$(parse_run "${run_out}" "${run_err}")"
    stop="${parsed%%$'\t'*}"
    rest="${parsed#*$'\t'}"
    first_id="${rest%%$'\t'*}"
    run_actual="${rest#*$'\t'}"
    if [ -n "${run_actual}" ] && [ "${run_actual}" != "${actual}" ]; then
      echo "g1_27442: WARNING run reported ${run_actual} prompt tokens, builder estimated ${actual} — recording the run's number" >&2
      actual="${run_actual}"
    fi
    append_trial "${target}" "${actual}" "${cls}" "${first_id}" "${stop}" \
      "${SEED}" "${ts}"
    echo "g1_27442: trial ${target}/${cls} -> stop=${stop} first_token_id=${first_id:-null} tokens=${actual}"
  done
done

# ============================================================
# Manifest (the attestation): written only after all trials
# ============================================================

python3 - "${MANIFEST}" "${PROTOCOL_ID}" "${MANIFEST_SCHEMA}" "${TRIALS}" \
  "${BIN}" "${MODEL}" "${RESEARCH_DIR}" "${CTX_SIZE}" "${KV_K}" "${KV_V}" \
  "${SEED}" "${THREADS}" <<'PYEOF'
import hashlib, json, os, subprocess, sys

manifest_path, protocol_id, schema = sys.argv[1:4]
trials_path, binary, model = sys.argv[4:7]
research_dir, ctx_size, kv_k, kv_v, seed, threads = sys.argv[7:13]

def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()

# research_commit: the tree that will pin the artifact (git check, no compute)
head = ""
try:
    out = subprocess.run(["git", "-C", research_dir, "rev-parse", "HEAD"],
                         capture_output=True, text=True, timeout=10)
    if out.returncode == 0:
        head = out.stdout.strip()
except (OSError, subprocess.SubprocessError):
    pass

launch = {
    "binary": "llama-completion",
    "n_predict": 1,
    "temp": 0,
    "seed": int(seed),
    "cache_prompt": False,
    "n_parallel": 1,
    "conversation_mode": False,
    "ctx_size": int(ctx_size),
    "kv_type_k": kv_k,
    "kv_type_v": kv_v,
    "special": True,
    "display_prompt": False,
    "verbose_prompt": True,
    "escape": False,
    "threads": int(threads) if threads else None,
}
manifest = {
    "schema": schema,
    "protocol_id": protocol_id,
    "date": os.path.basename(os.path.dirname(manifest_path))[len("g1-27442-"):],
    "binary_path": binary,
    "binary_sha256": sha256(binary),
    "model_path": model,
    "model_sha256": sha256(model),
    "research_commit": head,
    "launch": launch,
    "trials_file": os.path.basename(trials_path),
    "trials_sha256": sha256(trials_path),
}
# canonical form (the adapter recomputes this exact digest)
body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
manifest["manifest_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
with open(manifest_path, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
print(f"manifest written: {manifest_path}")
print(f"  protocol       {manifest['protocol_id']}")
print(f"  binary         {manifest['binary_sha256'][:16]}… {binary}")
print(f"  model          {manifest['model_sha256'][:16]}… {model}")
print(f"  research_commit {manifest['research_commit'] or '<none>'}")
print(f"  trials         {manifest['trials_sha256'][:16]}… {manifest['trials_file']}")
print(f"  manifest_sha256 {manifest['manifest_sha256'][:16]}…")
PYEOF

echo "g1_27442: DONE — ${OUT_DIR}"
