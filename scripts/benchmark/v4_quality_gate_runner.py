#!/usr/bin/env python3
"""V4 quality-gate runner: capture token-level logprobs on the 20-prompt set.

Loops `benchmarks/prompts/v1/deepseek-v4-quality-gate.yaml` through a llama-server
instance loaded with the V4 model, captures the first 64 token-IDs + logprobs of
the greedy output at `--temp 0 --seed 1 --top-k 1`, and writes structured JSON
ready for diff-against-reference.

Usage:
    v4_quality_gate_runner.py --model PATH --output PATH [--port 18072] [--n-tokens 64]

The script:
1. Verifies the canonical recipe via canonical_recipe.py validate --v4-fork
2. Confirms no other llama-server is on the requested port (feedback_no_concurrent_inference)
3. Launches the V4 fork's llama-server with --no-mmap --mlock + canonical env
4. Waits for /health to return "ok"
5. POSTs each prompt to /v1/completions with seed=1, temp=0, top_k=1, logprobs=5
6. Captures token IDs + logprobs into structured JSON
7. Tears down the server

Output JSON shape:
    {
      "model_path": "...",
      "binary": "/mnt/raid0/llm/llama.cpp-deepseek-v4/build/bin/llama-server",
      "n_tokens_requested": 64,
      "n_prompts": 20,
      "prompts": [
        {"id": "factual_01", "category": "short_factual", "prompt": "...",
         "token_ids": [...], "tokens_text": [...], "logprobs": [...]},
        ...
      ]
    }

This output is the EPYC side. The reference side (Mac running antirez fork or
ds4 reference engine) must produce the same shape; v4_quality_gate_compare.py
diffs them.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Resolve canonical_recipe + lib path
REPO_DIR = Path(__file__).resolve().parents[2]  # epyc-inference-research/
sys.path.insert(0, str(REPO_DIR / "scripts" / "lib"))
import canonical_recipe as r  # noqa: E402

# Optional yaml for prompt loading; fall back to a minimal parser if missing
try:
    import yaml  # type: ignore
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_yaml_prompts(path: Path) -> list[dict]:
    """Load the V4 quality-gate prompt YAML."""
    if HAVE_YAML:
        with path.open() as f:
            data = yaml.safe_load(f)
        return data["prompts"]
    raise RuntimeError(
        f"PyYAML not installed. Install with: pip install --user pyyaml\n"
        f"Or convert {path} to JSON and pass the .json path instead."
    )


def port_in_use(port: int) -> bool:
    """Check if a TCP port on localhost is bound by another process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def wait_for_health(port: int, timeout_s: int = 600) -> None:
    """Wait for /health to return 200 with `status: ok`. Tolerates load delays."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None  # type: ignore[name-defined]
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = json.load(resp)
            if body.get("status") in ("ok", "loaded"):
                return
        except (urllib.error.URLError, ConnectionRefusedError, json.JSONDecodeError) as e:
            last_err = e
        time.sleep(2)
    raise TimeoutError(
        f"server on :{port} did not become healthy within {timeout_s}s "
        f"(last error: {last_err})"
    )


def post_completion(port: int, prompt: str, n_tokens: int, seed: int = 1) -> dict:
    """POST /v1/completions with greedy + logprobs and return the JSON response."""
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": n_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "n_predict": n_tokens,
        "logprobs": 5,
        "stream": False,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


def extract_logprobs(completion_response: dict, n_tokens: int) -> dict:
    """Extract token-level logprobs from a /v1/completions response.

    Handles the OpenAI-compatible shape:
        choices[0].logprobs.tokens          -> list[str]
        choices[0].logprobs.token_logprobs  -> list[float]
        choices[0].logprobs.top_logprobs    -> list[dict[str, float]]
    """
    choice = completion_response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens", [])[:n_tokens]
    token_logprobs = logprobs.get("token_logprobs", [])[:n_tokens]
    return {
        "tokens_text": tokens,
        "logprobs": token_logprobs,
        "token_count": len(tokens),
        "finish_reason": choice.get("finish_reason"),
        "raw_text": choice.get("text", ""),
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="V4 quality-gate runner (EPYC side)")
    p.add_argument("--model", required=True, help="Path to V4 GGUF")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--port", type=int, default=18072, help="llama-server port")
    p.add_argument("--n-tokens", type=int, default=64,
                   help="Tokens to capture per prompt (default 64 per §Merge Gates)")
    p.add_argument("--n-threads", type=int, default=96)
    p.add_argument("--ctx-size", type=int, default=8192)
    p.add_argument(
        "--prompts-yaml",
        default=str(REPO_DIR / "benchmarks" / "prompts" / "v1" / "deepseek-v4-quality-gate.yaml"),
    )
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip canonical_recipe validation (testing only)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    prompts_path = Path(args.prompts_yaml)
    if not prompts_path.is_file():
        print(f"ERROR: prompts YAML not found: {prompts_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_validate:
        try:
            r.validate_canonical_env(check_host=True, skip_perf_paranoid=True)
            binary, _libs = r.discover_v4_fork_bench()
            print(f"Canonical recipe + V4 fork validated.")
        except (FileNotFoundError, r.CanonicalRecipeViolation) as e:
            print(f"VALIDATION FAILED:\n{e}", file=sys.stderr)
            return 3

    if port_in_use(args.port):
        print(f"ERROR: port {args.port} is already in use; refuse to start "
              f"(feedback_no_concurrent_inference)", file=sys.stderr)
        return 4

    # V4 fork's llama-server (sibling of llama-bench in same build dir)
    server_binary = "/mnt/raid0/llm/llama.cpp-deepseek-v4/build/bin/llama-server"
    if not Path(server_binary).is_file():
        print(f"ERROR: V4 fork llama-server not found: {server_binary}", file=sys.stderr)
        return 2

    prompts = parse_yaml_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    # Launch the server. Canonical env via canonical_recipe.build_canonical_env().
    env = r.build_canonical_env()
    # taskset/numactl prefix per CANONICAL_PREFIX; --no-mmap --mlock for THP-pool
    # warmup + anonymous allocation (matches production launches per the
    # project_gemma4_mtp_launch_recipe pattern).
    cmd = list(r.CANONICAL_PREFIX) + [
        server_binary,
        "-m", str(model_path),
        "--port", str(args.port),
        "--host", "127.0.0.1",
        "-t", str(args.n_threads),
        "-c", str(args.ctx_size),
        "--no-mmap",
        "--mlock",
        "--flash-attn", "on",
        "-fa", "1",
        "-np", "1",
        "--jinja",
        # Logprob support flag (llama-server enables logprobs by default when
        # requested per-request; no global flag needed)
    ]
    print(f"Launching V4 server (this may take 1-2 min to mlock 153 GiB):")
    print("  " + " ".join(cmd))
    server_log = output_path.with_suffix(".server.log")
    with server_log.open("w") as logf:
        proc = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # so we can kill the process group
        )

    try:
        try:
            wait_for_health(args.port, timeout_s=900)  # large model = long load
            print(f"V4 server healthy on :{args.port}")
        except TimeoutError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print(f"  see {server_log} for server-side errors", file=sys.stderr)
            return 5

        results = {
            "model_path": str(model_path),
            "binary": server_binary,
            "n_tokens_requested": args.n_tokens,
            "n_prompts": len(prompts),
            "seed": 1,
            "temperature": 0,
            "top_k": 1,
            "prompts": [],
        }
        for i, prompt_obj in enumerate(prompts, 1):
            pid = prompt_obj["id"]
            text = prompt_obj["text"].rstrip()
            print(f"[{i}/{len(prompts)}] {pid} ({prompt_obj['category']})")
            try:
                resp = post_completion(args.port, text, args.n_tokens, seed=1)
                lp = extract_logprobs(resp, args.n_tokens)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                lp = {"error": str(e), "token_count": 0, "logprobs": [], "tokens_text": []}
            results["prompts"].append({
                "id": pid,
                "category": prompt_obj["category"],
                "prompt": text,
                **lp,
            })
            # incremental save so a crash mid-run doesn't lose results
            with output_path.open("w") as f:
                json.dump(results, f, indent=2)

        print(f"\nDONE. Results -> {output_path}")
        return 0
    finally:
        # Teardown the server process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"V4 server torn down. Log: {server_log}")


if __name__ == "__main__":
    sys.exit(main())
