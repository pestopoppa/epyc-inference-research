#!/usr/bin/env python3
"""Bounded UFH14-A6 native inference canary; private server + proxy3 + Landlock client."""

import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sandbox3

MODEL = pathlib.Path("/mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf")
BIN = pathlib.Path("/mnt/raid0/llm/kernels/production/cpu/llama-server")
LIB = BIN.parent.resolve()
VERIFY = pathlib.Path(
    "/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh"
)
RUN = pathlib.Path(os.environ["A6_OUTPUT_ROOT"]).resolve()
if os.environ.get("A6_SMOKE") != "1":
    raise SystemExit("standalone R11 normal diagnostic requires A6_SMOKE=1")
MODE = os.environ.get("A6_MODE", "normal")
if MODE not in ("normal", "cutoff"):
    raise SystemExit("invalid A6 mode")
out = sandbox3.claim_fresh_output(RUN, MODE)
work = out / "workspace"
work.mkdir(exist_ok=True)
result = out / "result"
result.mkdir(exist_ok=True)
wire = out / "wire"
wire.mkdir(exist_ok=True)
end = float(os.environ.get("FILLER_END", str(time.time() + 280)))
stopfile = pathlib.Path(os.environ.get("FILLER_STOPFILE", str(out / "STOP")))
cpus = os.environ.get("FILLER_CPUS", "")
actual = (
    pathlib.Path("/proc/self/status")
    .read_text()
    .split("Cpus_allowed_list:\t")[1]
    .splitlines()[0]
    .strip()
)


def expand_cpu_list(text):
    out = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.update(range(lo, hi + 1))
        elif part:
            out.add(int(part))
    return out


if expand_cpu_list(cpus) != expand_cpu_list(actual) or not cpus:
    raise SystemExit("invalid R11 CPU claim")
if len(os.sched_getaffinity(0)) != 2 or not all(96 <= c <= 183 for c in os.sched_getaffinity(0)):
    raise SystemExit("R11 standalone diagnostic requires exactly two SMT CPUs 96-183")
if not MODEL.is_file() or not BIN.is_file():
    raise SystemExit("source missing")


# Pin the actual binary, source, and canonical model without copying weights.
def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


pins = {
    str(p): digest(p)
    for p in [
        HERE / "native_canary.py",
        HERE / "sandbox3.py",
        HERE / "proxy3.py",
        HERE / "native_client.py",
        MODEL,
        BIN,
    ]
}
(out / "pins.json").write_text(json.dumps(pins, indent=2) + "\n")
env = {
    **os.environ,
    "LD_LIBRARY_PATH": str(LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    "OMP_NUM_THREADS": "1",
    "TMPDIR": str(work),
}
link = subprocess.run(
    ["bash", str(VERIFY), str(BIN), str(LIB)], env=env, capture_output=True, text=True, timeout=30
)
(out / "linkage.log").write_text(link.stdout + link.stderr)
if link.returncode:
    raise SystemExit("ggml linkage failed")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
server = None
proxy = None
c = None
proof = {"mode": f"R11-standalone-{MODE}", "cpu_claim": cpus, "pins": pins, "partial": True}


def birth(pid):
    fields = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[19])


def durable_new(path, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def drain(p):
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=5)


try:
    meta = json.loads((sandbox3.BASE / "contexts/C2/meta.json").read_text())
    pol = sandbox3.c95_policy(
        cutoff=sandbox3.iso_epoch(meta["asran_started"]),
        ws_root=work,
        call_dir=result,
        scratch_clone=sandbox3.BASE / "lanes/0c8b896ee",
        context_id="C2",
        prompt_text=(sandbox3.BASE / "contexts/C2/prompt.inline.txt").read_text(),
        extra_allow=[{"path": str(HERE / "native_client.py"), "mode": "ro"}],
    )
    pf = out / "policy.json"
    pf.write_text(json.dumps(pol, indent=2) + "\n")
    proof["policy_sha256"] = digest(pf)
    provenance = {
        "schema": "ds41-c95.preexposure.v1",
        "purpose": "CANARY",
        "run_id": uuid.uuid4().hex,
        "call_locator": str(out),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "boot_id": pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid": os.getpid(),
        "uid": os.getuid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "context_id": "C2",
        "context_meta_sha256": digest(sandbox3.BASE / "contexts/C2/meta.json"),
        "prompt_sha256": hashlib.sha256((
            b"Count upward from 1, one integer per line. Continue until 1000."
            if MODE == "cutoff" else b"Reply with the single word READY."
        )).hexdigest(),
        "policy_sha256": proof["policy_sha256"],
        "source_sha256": {
            str(p): pins[str(p)]
            for p in (
                HERE / "native_canary.py",
                HERE / "native_client.py",
                HERE / "sandbox3.py",
                HERE / "proxy3.py",
            )
        },
        "grade_source_sha256": digest(sandbox3.BASE / "grade.py")
        if (sandbox3.BASE / "grade.py").is_file()
        else None,
        "declared_model": str(MODEL),
        "declared_endpoint": f"http://127.0.0.1:{port}",
        "model_file_sha256": pins[str(MODEL)],
        "binary_file_sha256": pins[str(BIN)],
        "verified_backend_identity": None,
        "verified_model_identity": None,
        "grade": None,
    }
    proof["preexposure_provenance"] = sandbox3.emit_preexposure(out, provenance)
    slog = (out / "server.log").open("w")
    server = subprocess.Popen(
        [
            str(BIN),
            "-m",
            str(MODEL),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            "0",
            "-t",
            "2",
            "-c",
            "2048",
            "-np",
            "1",
            "--no-warmup",
        ],
        env=env,
        stdout=slog,
        stderr=subprocess.STDOUT,
    )
    proof["server_pid"] = server.pid
    proof["server_birth"] = birth(server.pid)
    load_deadline = min(end - 5, time.time() + 150)
    while time.time() < load_deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited {server.returncode}")
        if stopfile.exists():
            raise TimeoutError("STOP during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError("server load cutoff")
    plog = (out / "proxy.stdout").open("w")
    proxy = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "proxy3.py"),
            "--port",
            "0",
            "--log",
            str(wire),
            "--upstream",
            f"http://127.0.0.1:{port}",
        ],
        env=env,
        stdout=plog,
        stderr=subprocess.STDOUT,
    )
    proof["proxy_pid"] = proxy.pid
    proof["proxy_birth"] = birth(proxy.pid)
    proxy_port = None
    for _ in range(80):
        if proxy.poll() is not None:
            raise RuntimeError("proxy exited")
        plog.flush()
        txt = (out / "proxy.stdout").read_text()
        for line in txt.splitlines():
            if line.startswith("PORT "):
                proxy_port = int(line.split()[1])
                break
        if proxy_port:
            break
        time.sleep(0.1)
    if not proxy_port:
        raise RuntimeError("proxy no PORT")
    cenv = {
        **env,
        "A6_WORK": str(work),
        "A6_RESULT": str(result),
        "A6_PROXY_URL": f"http://127.0.0.1:{proxy_port}",
        "A6_MODEL_NAME": MODEL.stem,
        "A6_MODE": MODE,
        "A6_CUTOFF_STOP": str(result / "STOP"),
    }
    remaining = end - time.time() - 5
    if remaining <= 0:
        raise TimeoutError("client deadline before request")
    cmd = [sys.executable, str(HERE / "sandbox3.py"), "exec", "--policy", str(pf),
           "--report", str(out / "sandbox-report.json"), "--", sys.executable,
           str(HERE / "native_client.py")]
    if MODE == "normal":
        c = subprocess.run(cmd, env=cenv, capture_output=True, text=True,
                           timeout=min(100, remaining))
        (out / "client.stdout").write_text(c.stdout)
        (out / "client.stderr").write_text(c.stderr)
    else:
        with (out / "client.stdout").open("w") as cout, (out / "client.stderr").open("w") as cerr:
            c = subprocess.Popen(cmd, env=cenv, stdout=cout, stderr=cerr)
            proof["client_pid"] = c.pid
            proof["client_birth"] = birth(c.pid)
            request_path = result / "request-start.json"
            progress_path = result / "stream-progress.json"
            checks_path = result / "filesystem-checks.json"
            client_deadline = min(end - 5, time.time() + 100)
            request = progress = checks = None
            while time.time() < client_deadline:
                if c.poll() is not None:
                    raise RuntimeError(f"client exited before cutoff: {c.returncode}")
                if stopfile.exists():
                    raise TimeoutError("external STOP before in-flight progress")
                try:
                    request = json.loads(request_path.read_text())
                    progress = json.loads(progress_path.read_text())
                    checks = json.loads(checks_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(0.01)
                    continue
                if not all(isinstance(item, dict) for item in (request, progress, checks)):
                    raise RuntimeError("stream marker schema invalid")
                break
            else:
                raise TimeoutError("no live stream progress before deadline")
            required = ("future_denied", "live_denied", "anchor_read", "workspace_write", "result_write")
            if not all(checks.get(key) for key in required):
                raise RuntimeError("filesystem isolation failed before cutoff")
            if request["pid"] != c.pid or progress["pid"] != c.pid or progress["nonterminal_content_events"] < 2:
                raise RuntimeError("stream markers do not identify live client")
            if not request["monotonic_ns"] < progress["monotonic_ns"]:
                raise RuntimeError("stream chronology invalid")
            if c.poll() is not None:
                raise RuntimeError("stream ended before owned STOP")
            stop_ns = time.monotonic_ns()
            durable_new(result / "STOP", {"monotonic_ns": stop_ns, "client_pid": c.pid,
                                           "client_birth": proof["client_birth"]})
            c.wait(timeout=min(20, max(1, client_deadline - time.time())))
            ack = json.loads((result / "cutoff-ack.json").read_text())
            if (c.returncode != 75 or ack["pid"] != c.pid or
                not progress["monotonic_ns"] < stop_ns < ack["monotonic_ns"] or
                ack["request_id"] != progress["request_id"] or
                ack["nonterminal_content_events"] < 2 or
                ack["finish_observed"] or ack["done_observed"]):
                raise RuntimeError("cutoff did not interrupt verified live stream")
            proof["cutoff"] = {"request": request, "progress": progress,
                               "stop_monotonic_ns": stop_ns, "ack": ack,
                               "client_birth": proof["client_birth"]}
    proof["client_rc"] = c.returncode
    proof["client_checks"] = (
        json.loads((result / "client-proof.json").read_text())
        if (result / "client-proof.json").exists()
        else {}
    )
    if MODE == "cutoff":
        for _ in range(200):
            if (wire / "wire.jsonl").exists() and (wire / "wire.jsonl").stat().st_size:
                break
            time.sleep(0.01)
    proof["wire_rows"] = (
        len((wire / "wire.jsonl").read_text().splitlines()) if (wire / "wire.jsonl").exists() else 0
    )
    if MODE == "normal" and (c.returncode or not proof["wire_rows"]):
        raise RuntimeError("native client or wire failed")
    if MODE == "cutoff" and (c.returncode != 75 or not proof["wire_rows"]):
        raise RuntimeError("client cutoff or wire row invalid")
    proof["partial"] = MODE == "cutoff"
    rc = 75 if MODE == "cutoff" else 0
except (TimeoutError, subprocess.TimeoutExpired) as exc:
    proof["stop_reason"] = str(exc)
    rc = 75
except Exception as exc:
    proof["error"] = f"{type(exc).__name__}: {exc}"
    rc = 1
finally:
    if isinstance(c, subprocess.Popen):
        drain(c)
    drain(proxy)
    drain(server)
    proof["client_dead"] = c is None or not isinstance(c, subprocess.Popen) or c.poll() is not None
    proof["proxy_dead"] = proxy is None or proxy.poll() is not None
    proof["server_dead"] = server is None or server.poll() is not None
    proof["finished_at"] = time.time()
    (out / "result.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps({"rc": rc, "result": str(out / "result.json"), "partial": proof["partial"]}))
raise SystemExit(rc)
