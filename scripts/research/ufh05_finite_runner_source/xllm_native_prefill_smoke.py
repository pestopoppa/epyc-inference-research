#!/usr/bin/env python3
"""Prospective finite actual xLLM CPU batch-one feasibility actor.

Source preparation does not authorize execution. This ungraded operational
recipe invokes native terminal_kv_forward/prefill_banks, never the paper's
MIN_BATCH=8 evaluator. No model imports occur until the guarded worker phase.
"""
from __future__ import annotations
import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback

SOURCE_PIN = "3af99f493e14fba162625d5dd687abea551a299e"
SHIM_HASHES = {
    "xllm/paper_part2/artifacts.py": "a90fb0942ccfc7dac23e6580eef576be07f3c73f1f9b360814763158d27bd8e7",
    "xllm/paper_part2/distill.py": '04d12863a3dc61f607dfb2218548935f1272f59955a63bc8329d03f7594b0c43',
}
TEACHER_REPOSITORY = "IFM/LoopedLM-P2-huginn-s-learned-entropy0p01"
STUDENT_REPOSITORY = "IFM/LoopedLM-P2-distilled-s-teacher-init"
MEMORY_BYTES = 8 * 1024 ** 3
WALL_SECONDS = 1800
DEPTH = 5
STATE_SEED = 42
MAX_CASES = 4
DECODE_TOKENS = 4
SMALL_MODEL = {"arch": "huginn", "huginn_depth_control": True, "num_layers": 4,
               "model_dim": 1536, "num_heads": 24, "num_kv_heads": 6,
               "head_dim": 64, "ffn_hidden_dim": 4352, "vocab_size": 64256,
               "loop_times": 5, "loop_start_layer": 1, "loop_end_layers": 3}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(path, kind):
    candidate = Path(os.path.abspath(path))
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("explicit input traverses a symlink: " + str(candidate))
    info = candidate.lstat()
    if not (stat.S_ISREG(info.st_mode) if kind == "file" else stat.S_ISDIR(info.st_mode)):
        raise ValueError("explicit input has wrong type: " + str(candidate))
    return candidate


def stat_identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_regular(path, limit=16 * 1024 ** 2):
    path = checked_path(path, "file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("bounded regular metadata input required")
        chunks = []
        size = 0
        while chunk := os.read(fd, 1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError("metadata input exceeded bound")
            chunks.append(chunk)
        if stat_identity(before) != stat_identity(os.fstat(fd)) or stat_identity(before) != stat_identity(path.lstat()) or size != before.st_size:
            raise ValueError("metadata identity changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def sha_file(path):
    path = checked_path(path, "file")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        result = hashlib.file_digest(handle, "sha256").hexdigest()
        if stat_identity(before) != stat_identity(os.fstat(handle.fileno())) or stat_identity(before) != stat_identity(path.lstat()):
            raise ValueError("original input changed while hashing")
        return result


def strict_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out
    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(read_regular(path), object_pairs_hook=unique, parse_constant=nonfinite)


def original_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args])


def source_identity(source):
    source = checked_path(source, "directory")
    if git(source, "rev-parse", "HEAD").decode().strip() != SOURCE_PIN:
        raise ValueError("actual native Git parent differs")
    changes = set(git(source, "diff", "--name-only", "-z", "HEAD").decode().split("\0")) - {""}
    if changes != set(SHIM_HASHES) or git(source, "ls-files", "--others", "-z"):
        raise ValueError("source must contain only the exact two reviewed shims and no extra ignored/untracked files")
    files = []
    for raw in git(source, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        if not raw:
            continue
        info, relative = raw.split(b"\t", 1)
        mode, kind, blob = info.decode().split()
        name = relative.decode()
        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise ValueError("nested Git/submodule source requires a separate bound recipe")
        path = source / name
        if mode == "120000":
            if not path.is_symlink():
                raise ValueError("original tracked link changed type")
            content = os.readlink(path).encode()
            if hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != blob:
                raise ValueError("original tracked link changed")
            files.append({"path": name, "mode": mode, "blob": blob, "target": content.decode()})
            continue
        value = sha_file(path)
        if name in SHIM_HASHES:
            if value != SHIM_HASHES[name]:
                raise ValueError("source device shim differs from exact source proposal")
        else:
            content = read_regular(path, limit=128 * 1024 ** 2)
            if hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != blob:
                raise ValueError("unchanged actual Git source differs: " + name)
        files.append({"path": name, "mode": mode, "blob": blob, "bytes": path.stat().st_size, "sha256": value})
    return {"commit": SOURCE_PIN, "exact_shims": SHIM_HASHES, "files": files}


def validate_inputs(payload):
    if not isinstance(payload, dict) or set(payload) != {"schema", "cases"} or payload["schema"] != "epyc.xllm.native_prefill_inputs.v1":
        raise ValueError("frozen native input schema differs")
    cases = payload["cases"]
    if not isinstance(cases, list) or not 2 <= len(cases) <= MAX_CASES:
        raise ValueError("two to four finite development/reserved-conformance inputs required")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"id", "phase", "text", "token_ids"}:
            raise ValueError("frozen native input fields differ")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case["id"]):
            raise ValueError("frozen input identity invalid")
        if case["phase"] not in {"development", "conformance"} or not isinstance(case["text"], str) or not 1 <= len(case["text"]) <= 2048:
            raise ValueError("frozen input phase or short text invalid")
        if not isinstance(case["token_ids"], list) or not 2 <= len(case["token_ids"]) <= 64 or any(type(t) is not int or not 0 <= t < 64256 for t in case["token_ids"]):
            raise ValueError("short native vocabulary token IDs invalid")
    if cases[0]["phase"] != "development" or {c["phase"] for c in cases} != {"development", "conformance"}:
        raise ValueError("development warm-up and reserved conformance inputs required")
    if any(len({key(c) for c in cases}) != len(cases) for key in
           (lambda c: c["id"], lambda c: c["text"], lambda c: tuple(c["token_ids"]), lambda c: len(c["token_ids"]))):
        raise ValueError("IDs, texts, token sequences and lengths must all be distinct")
    if [c["phase"] for c in cases] != sorted((c["phase"] for c in cases), key=lambda v: 0 if v == "development" else 1):
        raise ValueError("reserved conformance follows development without tuning")
    return cases


def validate_binding(args):
    if sha_file(args.binding) != args.binding_sha256:
        raise ValueError("exact future MAIN binding bytes differ")
    binding = strict_json(args.binding)
    fields = {"schema", "recipe_id", "source_pin", "runner_sha256", "shim_map_sha256",
              "inputs_sha256", "core_claim_sha256", "teacher", "student", "source_enrollment"}
    if set(binding) != fields or binding["schema"] != "epyc.xllm.native_prefill_binding.v1" or binding["recipe_id"] != "RC-XLLM-PREFILL-1" or binding["source_pin"] != SOURCE_PIN:
        raise ValueError("future source/recipe binding scope differs")
    for key, path in [("runner_sha256", Path(__file__)), ("shim_map_sha256", args.shim_map),
                      ("inputs_sha256", args.inputs), ("core_claim_sha256", args.core_claim)]:
        if binding[key] != sha_file(path):
            raise ValueError("exact bound original differs: " + key)
    shims = strict_json(args.shim_map)
    actual = {r["path"]: r["proposed_sha256"] for r in shims["actual_source_patch"]}
    if actual != SHIM_HASHES:
        raise ValueError("a caller-supplied shim map cannot expand the two reviewed paths")
    for role, path, repository in [("teacher", args.teacher, TEACHER_REPOSITORY), ("student", args.student, STUDENT_REPOSITORY)]:
        checked_path(path, "directory")
        row = binding[role]
        if set(row) != {"repository", "revision", "manifest_sha256", "manifest_file_sha256", "config_sha256"} or row["repository"] != repository or not re.fullmatch(r"[0-9a-f]{40}", row["revision"]):
            raise ValueError("exact public S teacher/student catalog binding differs")
        if row["manifest_file_sha256"] != sha_file(path / "artifact_manifest.json") or row["config_sha256"] != sha_file(path / "config.json"):
            raise ValueError("actual artifact manifest/config bytes differ")
        meta = strict_json(path / "artifact_manifest.json")
        if not re.fullmatch(r"[0-9a-f]{64}", row["manifest_sha256"]) or meta.get("manifest_sha256") != row["manifest_sha256"]:
            raise ValueError("actual native artifact logical manifest binding differs")
    teacher = strict_json(args.teacher / "config.json")
    student = strict_json(args.student / "config.json")
    if any(teacher.get("model", {}).get(k) != v for k, v in SMALL_MODEL.items()):
        raise ValueError("only the pinned native Part II S teacher may be loaded")
    if student.get("model", {}).get("model_dim") != 1536 or student.get("model", {}).get("num_layers") != 2:
        raise ValueError("only the two-block native S student may be loaded")
    if student.get("student", {}).get("teacher_manifest_sha256") != binding["teacher"]["manifest_sha256"]:
        raise ValueError("actual student must bind this exact teacher before any state load")
    enrollment = binding["source_enrollment"]
    if set(enrollment) != {"root_commit", "source_table_blob", "task_id", "capture_contract"} or not re.fullmatch(r"[0-9a-f]{40}", enrollment["root_commit"]) or not re.fullmatch(r"[0-9a-f]{40}", enrollment["source_table_blob"]) or enrollment["task_id"] != "VB-RI-OPS-WIRE" or enrollment["capture_contract"] != "prospective-original-native-record-no-new-grade":
        raise ValueError("exact existing prospective write-side enrollment required")
    root = checked_path(args.root_context, "directory")
    if git(root, "rev-parse", "HEAD").decode().strip() != enrollment["root_commit"]:
        raise ValueError("actual prospective ROOT enrollment checkout differs")
    table_path = "scripts/vidya/adapters/README.md"
    task_path = "handoffs/active/vidya-belief-substrate-program.md"
    table_blob = git(root, "rev-parse", "HEAD:" + table_path).decode().strip()
    if table_blob != enrollment["source_table_blob"]:
        raise ValueError("actual prospective source-table Git blob differs")
    for relative in (table_path, task_path):
        original = git(root, "show", "HEAD:" + relative)
        actual_path = checked_path(root / relative, "file")
        if actual_path.read_bytes() != original or b"VB-RI-OPS-WIRE" not in original:
            raise ValueError("actual unchanged write-side enrollment marker required: " + relative)
        if relative == table_path and not any(b"RC-XLLM-PREFILL-1" in line and b"VB-RI-OPS-WIRE" in line for line in original.splitlines()):
            raise ValueError("actual native prefill source row must name its existing wiring owner")
    # These pinned bytes do not prove permission or a held CPU claim. The owning
    # model-run wrapper retains and validates the actual claim before this actor.
    return binding, teacher, student


def runtime_files():
    rows = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file():
                rows[str(path)] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    executable = Path(sys.executable).resolve()
    rows[str(executable)] = {"bytes": executable.stat().st_size, "sha256": sha_file(executable)}
    original_maps = Path("/proc/self/maps").read_text()
    for line in original_maps.splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) == 6 and pieces[5].startswith("/"):
            path = Path(pieces[5]).resolve()
            if path.is_file():
                rows[str(path)] = {"bytes": path.stat().st_size, "sha256": sha_file(path)}
    return {"schema": "epyc.xllm.loaded_runtime_identity.v1",
            "loaded_files": rows, "original_proc_self_maps": original_maps}


def output_inventory(root):
    if root.is_symlink() or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError("output root must be an original directory")
    members = {".": {"kind": "directory"}}
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    members[relative] = {"kind": "symlink", "target": os.readlink(entry.path)}
                elif stat.S_ISDIR(info.st_mode):
                    members[relative] = {"kind": "directory"}
                    pending.append((Path(entry.path), relative))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    members[relative] = {"kind": "regular", "bytes": info.st_size, "sha256": sha_file(Path(entry.path))}
                else:
                    raise ValueError(f"unsupported output object: {relative}")
    return dict(sorted(members.items()))


def worker(args):
    output = args.output
    started = time.monotonic()
    stage = "resource context"
    source_before = None
    native_manifests = None
    verify_artifact = None
    def event(kind, **fields):
        row = {"kind": kind, "utc": utc(), "elapsed_s": time.monotonic() - started, **fields}
        with (output / "original-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    def operation(name):
        nonlocal stage
        stage = name
        event("operation_started", operation=name)
    def expired(signum, frame):
        raise TimeoutError("thirty-minute native CPU wall limit reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(WALL_SECONDS)
    failed = True
    code = 1
    try:
        cpus = [int(value) for value in args.cpus.split(",")]
        if len(cpus) != 8 or len(set(cpus)) != 8 or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("exact eight available granted CPU IDs required")
        os.sched_setaffinity(0, cpus)
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        topology = []
        for cpu in cpus:
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            topology.append({"cpu": cpu, "package_id": (root / "physical_package_id").read_text().strip(),
                             "core_id": (root / "core_id").read_text().strip(),
                             "thread_siblings_list": (root / "thread_siblings_list").read_text().strip()})
        if len({(r["package_id"], r["core_id"]) for r in topology}) != 8:
            raise ValueError("SMT siblings do not count as distinct granted physical cores")
        event("budget", cpu_ids=cpus, observed_topology=topology,
              memory_cap_bytes=MEMORY_BYTES, memory_cap_kind="conservative address-space ceiling, not RSS",
              wall_cap_seconds=WALL_SECONDS, held_claim_proof="external model-run wrapper; opaque receipt identity alone does not prove heldness")
        if not sys.flags.isolated or not sys.dont_write_bytecode or sys.prefix == sys.base_prefix:
            raise ValueError("actual native worker requires isolated -I/-B interpreter in explicit venv")
        operation("source and frozen input binding before native imports")
        binding, teacher_config, student_config = validate_binding(args)
        cases = validate_inputs(strict_json(args.inputs))
        source_before = source_identity(args.source)
        original_json(output / "source-tree-before.json", source_before)
        original_json(output / "bound-input-context.json", {"binding": binding, "binding_sha256": args.binding_sha256,
            "source_tree_sha256": sha_file(output / "source-tree-before.json"), "cases": cases,
            "teacher_config": teacher_config, "student_config": student_config,
            "device": "cpu", "dtype": "float32", "depth": DEPTH, "seed": STATE_SEED})
        operation("actual native imports; unsupported imports are retained without stubs or builds")
        sys.path.insert(0, str(args.source))
        import torch
        from xllm.config import TokenizerConf
        from xllm.data.dataset_streamer.tokenizer import build_tokenizer
        from xllm.paper_part2.artifacts import open_artifact, build_model, verify_artifact
        from xllm.paper_part2.distill import load_student, prefill_banks
        # Bind what was actually imported before any model or native operations.
        original_json(output / "runtime-loaded-files-before-native-use.json", runtime_files())
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.manual_seed(STATE_SEED)
        torch.set_grad_enabled(False)
        event("native_runtime", torch_version=torch.__version__, torch_build=torch.__config__.show(),
              packages={item.metadata["Name"]: item.version for item in importlib.metadata.distributions()},
              model_parallel_context="actual native singleton accessors; no synthetic process group or patched rank helper")
        operation("actual unchanged artifact verification before loading tensors")
        teacher_manifest = verify_artifact(args.teacher)
        student_manifest = verify_artifact(args.student)
        if teacher_manifest["manifest_sha256"] != binding["teacher"]["manifest_sha256"] or student_manifest["manifest_sha256"] != binding["student"]["manifest_sha256"]:
            raise ValueError("native artifact verifier differs from bound manifests")
        native_manifests = (teacher_manifest, student_manifest)
        operation("actual native S teacher and exact bound student cold load")
        cold_start = time.perf_counter()
        manifest, config, state = open_artifact(args.teacher)
        if manifest != teacher_manifest or config != teacher_config:
            raise ValueError("actual loader artifact/config changed after binding")
        tokenizer = build_tokenizer(TokenizerConf(type="huggingface", path=str(args.teacher / config["tokenizer"]["path"])))
        teacher = build_model(config["model"], state, tokenizer, device="cpu", dtype=torch.float32)
        del state
        gc.collect()
        student = load_student(args.student, manifest, config["model"], device="cpu", dtype=torch.float32)
        event("cold_load", seconds=time.perf_counter() - cold_start, teacher_manifest=manifest,
              student_manifest=student_manifest, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        def valid_cache(cache, length):
            if len(cache) != 4 or teacher.num_layers != 4:
                raise ValueError("actual terminal bank count differs from native S teacher")
            shapes = []
            for bank in cache:
                if len(bank) != 3 or type(bank[2]) is not int or bank[2] != length:
                    raise ValueError("actual terminal cache length metadata differs")
                key, value, _ = bank
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor) or list(key.shape) != [1, length, 6, 64] or key.shape != value.shape:
                    raise ValueError("actual terminal bank tensor shape differs")
                if key.device.type != "cpu" or value.device.type != "cpu" or not torch.isfinite(key).all() or not torch.isfinite(value).all():
                    raise FloatingPointError("nonfinite or non-CPU terminal cache")
                shapes.append({"key": list(key.shape), "value": list(value.shape), "length": bank[2]})
            return shapes
        def prefill(tokens, ids, arm):
            prefix = tokens[:, :-1]
            before = time.perf_counter()
            cache = (teacher.terminal_kv_forward(prefix, ids, STATE_SEED, depth=DEPTH)[1]
                     if arm == "teacher" else prefill_banks(teacher, student, prefix))
            prefix_seconds = time.perf_counter() - before
            valid_cache(cache, prefix.shape[1])
            before = time.perf_counter()
            logits, cache = teacher.terminal_kv_forward(tokens[:, -1:], ids, STATE_SEED, cache, depth=DEPTH)
            terminal_seconds = time.perf_counter() - before
            if list(logits.shape) != [1, 1, 64256] or logits.device.type != "cpu" or not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite, non-CPU or wrong-shape native logits")
            return logits, cache, prefix_seconds, terminal_seconds, valid_cache(cache, tokens.shape[1])
        for case in cases:
            if tokenizer.encode(case["text"], bos=True, eos=False) != case["token_ids"]:
                raise ValueError("frozen input token IDs differ from exact bound tokenizer")
        warmup_case = cases[0]
        warmup = torch.tensor([warmup_case["token_ids"]], dtype=torch.long, device="cpu")
        warmup_ids = torch.tensor([0], dtype=torch.long, device="cpu")
        with torch.inference_mode():
            for arm in ("teacher", "student_endpoint_teacher_cache"):
                operation("development-only warm-up actual " + arm)
                logits, cache, prefix_s, terminal_s, shapes = prefill(warmup, warmup_ids, arm)
                event("warmup", input_id=warmup_case["id"], phase="development", arm=arm,
                      prefix_seconds=prefix_s, terminal_seconds=terminal_s, terminal_bank_shapes=shapes,
                      reserved_conformance_used=False, not_a_warm_measurement=True)
                del logits, cache
        gc.collect()
        for index, case in enumerate(cases):
            operation("frozen input " + case["id"])
            tokens = torch.tensor([case["token_ids"]], dtype=torch.long, device="cpu")
            ids = torch.tensor([index], dtype=torch.long, device="cpu")
            outputs = {}
            with torch.inference_mode():
                for arm in ("teacher", "student_endpoint_teacher_cache"):
                    operation(f"input {case['id']} {arm} warm prefill")
                    logits, cache, prefix_s, terminal_s, shapes = prefill(tokens, ids, arm)
                    raw_path = output / f"input-{index:03d}-{arm}.pt"
                    with raw_path.open("xb") as handle:
                        torch.save({"logits": logits.cpu(), "cache": cache}, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    outputs[arm] = logits[:, -1].float().cpu()
                    continuation = []
                    decode_logits = []
                    before = time.perf_counter()
                    for turn in range(DECODE_TOKENS):
                        operation(f"input {case['id']} {arm} actual teacher decode {turn}")
                        next_token = logits[:, -1].argmax(dim=-1).reshape(1, 1)
                        continuation.append(int(next_token.item()))
                        logits, cache = teacher.terminal_kv_forward(next_token, ids, STATE_SEED, cache, depth=DEPTH)
                        if list(logits.shape) != [1, 1, 64256] or not torch.isfinite(logits).all():
                            raise FloatingPointError("nonfinite or wrong-shape native decode logits")
                        valid_cache(cache, len(case["token_ids"]) + turn + 1)
                        decode_logits.append(logits.cpu())
                    decode_elapsed = time.perf_counter() - before
                    decode_path = output / f"input-{index:03d}-{arm}-decode.pt"
                    with decode_path.open("xb") as handle:
                        torch.save({"logits": decode_logits, "continuation": continuation, "final_cache": cache}, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    event("arm", input_id=case["id"], phase=case["phase"], arm=arm, batch=1,
                          token_ids=case["token_ids"], depth=DEPTH, state_seed=STATE_SEED,
                          warm_prefix_seconds=prefix_s, warm_terminal_seconds=terminal_s,
                          decode_seconds=decode_elapsed, continuation=continuation,
                          raw_decode_artifact=decode_path.name, raw_decode_sha256=sha_file(decode_path),
                          terminal_bank_shapes=shapes, raw_artifact=raw_path.name, raw_sha256=sha_file(raw_path),
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          performance_warrant=False)
                    del logits, cache, decode_logits
                mse = float(torch.mean((outputs["teacher"].double() - outputs["student_endpoint_teacher_cache"].double()) ** 2))
                event("approximation", input_id=case["id"], initial_logit_mse=mse,
                      initial_argmax_agrees=bool(outputs["teacher"].argmax(-1).eq(outputs["student_endpoint_teacher_cache"].argmax(-1)).all()),
                      approximation_is_execution_conformance=False, quality_warrant=False)
            del outputs, tokens, ids
            gc.collect()
        operation("full original runtime/source/artifact custody after native operations")
        original_json(output / "runtime-loaded-files.json", runtime_files())
        if source_identity(args.source) != source_before or verify_artifact(args.teacher) != teacher_manifest or verify_artifact(args.student) != student_manifest:
            raise ValueError("full original source/artifact custody changed")
        event("completed", arms=len(cases) * 2, cases=len(cases),
              scope="actual native CPU batch-one execution feasibility only; no paper benchmark, approximation-quality, production-role or graded performance warrant")
        failed = False
        code = 0
    except BaseException as error:
        event("refused_or_failed", operation=stage, error_type=type(error).__name__, error=str(error),
              traceback=traceback.format_exc(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        code = 1
    finally:
        signal.alarm(0)
        # No deletion or repair of original outputs, even after a failed write.
        if source_before is not None:
            try:
                after = source_identity(args.source)
                original_json(output / "source-tree-after.json", after)
                event("source_custody", unchanged=after == source_before, operation_failed=failed)
                if after != source_before:
                    code = 1
            except BaseException as error:
                code = 1
                event("source_custody_refused", error_type=type(error).__name__, error=str(error))
        try:
            original_json(output / "runtime-loaded-files-final.json", runtime_files())
        except BaseException as error:
            code = 1
            event("runtime_custody_refused", error_type=type(error).__name__, error=str(error))
        if native_manifests is not None and verify_artifact is not None:
            try:
                values = [verify_artifact(args.teacher), verify_artifact(args.student)]
                original_json(output / "artifact-manifests-after.json", values)
                event("artifact_custody", unchanged=tuple(values) == native_manifests)
                if tuple(values) != native_manifests:
                    code = 1
            except BaseException as error:
                code = 1
                event("artifact_custody_refused", error_type=type(error).__name__, error=str(error))

    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "teacher", "student", "inputs", "shim-map", "core-claim", "binding", "root-context", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("source", "teacher", "student", "inputs", "shim_map", "core_claim", "binding", "root_context", "output"):
        setattr(args, name, Path(os.path.abspath(getattr(args, name))))
    if not args.execute:
        parser.error("actual execution belongs only to the separately authorized model-run owner")
    if args.worker:
        checked_path(args.output, "directory")
        request = strict_json(args.output / "execution-request.json")
        if request.get("capture_parent_pid") != os.getppid() or os.environ.get("XLLM_CAPTURE_PARENT_PID") != str(os.getppid()) or set(p.name for p in args.output.iterdir()) != {"execution-request.json", "original-command.log"}:
            raise ValueError("only the fresh owned capture child may enter native worker")
        return worker(args)
    if os.path.lexists(args.output):
        raise ValueError("fresh exclusive output directory required")
    checked_path(args.output.parent, "directory")
    args.output.mkdir(mode=0o700)
    argv = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    original_json(args.output / "execution-request.json", {"schema": "epyc.xllm.native_cpu_prefill_request.v1",
        "utc": utc(), "capture_parent_pid": os.getpid(), "argv": argv, "script_sha256": sha_file(Path(__file__)), "binding_sha256": args.binding_sha256,
        "recipe_id": "RC-XLLM-PREFILL-1", "work_wall_cap_seconds": WALL_SECONDS, "cleanup_grace_seconds": 5,
        "memory_cap_bytes": MEMORY_BYTES, "maximum_cases": MAX_CASES, "batch": 1, "depth": DEPTH,
        "scope": "prospective original native operational record, no new grading ladder or permission assertion"})
    env = dict(os.environ, XLLM_CAPTURE_PARENT_PID=str(os.getpid()), TORCHDYNAMO_DISABLE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8",
               MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", NUMEXPR_NUM_THREADS="8",
               CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    child = None
    started = time.monotonic()
    timed_out = False
    interrupted = None
    def forward(signum, frame):
        nonlocal interrupted
        interrupted = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise InterruptedError("external owner interrupted native actor")
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    with (args.output / "original-command.log").open("xb") as log:
        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=max(1, WALL_SECONDS - (time.monotonic() - started)))
        except (subprocess.TimeoutExpired, InterruptedError):
            timed_out = interrupted is None
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            code = 124 if timed_out else 128 + interrupted
    if child.poll() is None:
        raise RuntimeError("owned child termination not confirmed")
    members = output_inventory(args.output)
    original_json(args.output / "exit-status.json", {"schema": "epyc.xllm.native_cpu_prefill_exit.v1", "utc": utc(),
        "exit_code": code, "owned_pid": child.pid, "owned_pid_confirmed_dead": True, "timed_out": timed_out,
        "external_interrupt_signal": interrupted, "elapsed_seconds": time.monotonic() - started,
        "original_output_members_before_exit_record": members, "new_grade_authored": False,
        "scope": "actual CPU batch-one feasibility only; output timing and approximation are ungraded observations"})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
