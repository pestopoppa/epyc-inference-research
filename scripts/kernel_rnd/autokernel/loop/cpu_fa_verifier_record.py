"""Prospective native real-mask verifier custody; no grading rule or replay claim."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

SCHEMA = "epyc.autokernel.ds41_fa_verifier.v1"
PROPOSITION = ("For the sixteen captured DS41 raw-plus-compressed top-k F16 masks at "
               "KV widths 4096, 8192, 32768, 65536 and query rows 2, 3, 4, 5, "
               "each recorded anchor/candidate probe configuration has identical inputs, "
               "three internally stable output repetitions, and bit-identical output rows.")
AUTHORITY = "captured_mask_probe_identity_no_serving_performance_or_promotion"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def exclusive(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


class NativeVerifierRecord:
    """Create before execution, persist every original terminal before reduction."""

    def __init__(self, capture_dir, anchor_build, candidate_build, source_root,
                 cases, configs, reps, arms):
        self.capture_dir = Path(capture_dir).resolve(strict=True)
        self.directory = self.capture_dir / "verifier-runs" / uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.observations = []
        self.pins = []
        self.started_at = utc()
        manifest = json.loads((self.capture_dir / "capture-manifest.json").read_bytes())
        self.mode = manifest.get("evidence_mode")
        if self.mode not in ("native_ds41_capture", "synthetic_source_control"):
            raise ValueError("capture predates the prospective verifier evidence-mode hook")
        self.proposition = ("Synthetic source-conformance fixture only: " + PROPOSITION
                            if self.mode == "synthetic_source_control" else PROPOSITION)
        if manifest.get("decided_proposition") != self.proposition or manifest.get("verifier_schema") != SCHEMA:
            raise ValueError("capture predates the prospective verifier hook")
        capture_root = Path(manifest["source_root"]).resolve(strict=True)
        for index, (name, expected) in enumerate(manifest["capture_sources"].items()):
            pin = self.pin(capture_root / name, f"capture-model-source-{index}.txt")
            if pin["sha256"] != expected:
                raise ValueError("capture model source changed before verification")
        for index, (name, expected) in enumerate(manifest["build_images"].items()):
            pin = self.pin(Path(name), f"capture-build-image-{index}.bin")
            if pin["sha256"] != expected:
                raise ValueError("capture build changed before verification")
        for name in ("capture-manifest.json", "recipe.json", "prompt.txt"):
            self.pin(self.capture_dir / name, "capture-" + name)
        for case in cases:
            for suffix in ("mask.f16", "mask.json"):
                name = f"{case.name}.{suffix}"
                self.pin(self.capture_dir / name, name)
        source_root = Path(source_root).resolve(strict=True)
        for name, path in (("probe-source.cpp", Path(__file__).with_name("cpu_fa_reference_probe.cpp")),
                           ("verifier-source.py", Path(__file__)),
                           ("reference-source.py", Path(__file__).with_name("cpu_fa_reference.py")),
                           ("capture-source.py", Path(__file__).with_name("cpu_fa_mask_capture.py")),
                           ("ggml-header.h", source_root / "ggml/include/ggml.h")):
            self.pin(path, name)
        self.builds = {}
        for role, build in (("anchor", anchor_build), ("candidate", candidate_build)):
            build = Path(build).resolve(strict=True)
            self.builds[role] = str(build)
            for name in ("libggml.so", "libggml-base.so", "libggml-cpu.so"):
                self.pin(build / "bin" / name, role + "-" + name)
        self.request = {"schema": SCHEMA + ".request", "capture_id": self.directory.name,
            "started_at": self.started_at, "decided_proposition": self.proposition, "evidence_mode": self.mode,
            "authority": AUTHORITY, "metric": "captured_mask_anchor_bit_identity",
            "metric_direction": "higher_better", "category": "CANDIDATE",
            "source_root": str(source_root), "builds": self.builds,
            "cases": [vars(case) for case in cases], "configs": configs, "reps": reps,
            "arms": arms, "pins": list(self.pins), "capture_manifest": manifest}
        exclusive(self.directory / "request.json", canonical(self.request))

    def pin(self, path, name):
        path = Path(path).absolute()
        raw = path.read_bytes()
        exclusive(self.directory / name, raw)
        pin = {"name": name, "sha256": digest(raw), "bytes": len(raw), "original_path": str(path),
               "resolved_path": str(path.resolve(strict=True))}
        self.pins.append(pin)
        return pin

    def runner(self, runner):
        def observed(argv, **kwargs):
            index = len(self.observations)
            role = next(role for role in ("anchor", "candidate") if
                (argv[0] == "c++" and argv[-1].endswith("probe-" + role)) or
                any(value.endswith("probe-" + role) for value in argv))
            libraries = [item for item in self.request["pins"]
                         if item["name"].startswith(role + "-libggml")]
            library_before = {item["name"]: digest(Path(item["original_path"]).read_bytes()) for item in libraries}
            if library_before != {item["name"]: item["sha256"] for item in libraries}:
                raise ValueError("original probe libraries changed before execution")
            probe_pin = next((item for item in self.pins if item["name"].startswith("compiled-probe-")
                              and item["original_path"] in argv), None)
            probe_before = digest(Path(probe_pin["original_path"]).read_bytes()) if probe_pin else None
            if probe_pin is not None and probe_before != probe_pin["sha256"]:
                raise ValueError("original compiled probe changed before execution")
            mask_file = Path(argv[-1]) if "--mask-file" in argv else None
            mask_sha = digest(mask_file.read_bytes()) if mask_file is not None else None
            if mask_file is not None:
                expected = next(item["sha256"] for item in self.request["pins"]
                                if item["original_path"] == str(mask_file.resolve(strict=True)))
                if mask_sha != expected:
                    raise ValueError("original consumed mask changed before probe execution")
            launch = {"sequence": index, "argv": list(argv), "env": dict(kwargs["env"]),
                      "cwd": os.getcwd(), "timeout_seconds": kwargs["timeout"], "started_at": utc(),
                      "mask_sha256_before": mask_sha, "library_sha256_before": library_before,
                      "probe_binary_sha256_before": probe_before,
                      "output_capture": "original_binary_bytes"}
            exclusive(self.directory / f"launch-{index:04d}.json", canonical(launch))
            start = time.monotonic()
            error = None
            stdout, stderr, returncode = None, None, None
            try:
                done = runner(argv, **dict(kwargs, text=False))
                stdout, stderr, returncode = done.stdout, done.stderr, done.returncode
            except BaseException as exc:
                error = type(exc).__name__
                stdout, stderr, returncode = getattr(exc, "stdout", None), getattr(exc, "stderr", None), None
                raise
            finally:
                def binary(value):
                    return value if isinstance(value, bytes) else (value or "").encode("utf-8")
                stdout_raw, stderr_raw = binary(stdout), binary(stderr)
                output_pins = {}
                for stream, data in (("stdout", stdout_raw), ("stderr", stderr_raw)):
                    name = f"{stream}-{index:04d}.bin"
                    exclusive(self.directory / name, data)
                    output_pin = {"name": name, "sha256": digest(data), "bytes": len(data)}
                    self.pins.append(output_pin)
                    output_pins[stream] = output_pin
                observation = {**launch, "ended_at": utc(), "wait_seconds": time.monotonic() - start,
                    "returncode": returncode, "stdout": stdout_raw.decode("utf-8", errors="replace"),
                    "stderr": stderr_raw.decode("utf-8", errors="replace"), "original_outputs": output_pins,
                    "error": error, "mask_sha256_after": digest(mask_file.read_bytes()) if mask_file is not None else None,
                    "probe_binary_sha256_after": digest(Path(probe_pin["original_path"]).read_bytes()) if probe_pin else None,
                    "library_sha256_after": {item["name"]: digest(Path(item["original_path"]).read_bytes()) for item in libraries}}
                exclusive(self.directory / f"observation-{index:04d}.json", canonical(observation))
                self.observations.append(observation)
            if observation["mask_sha256_after"] != mask_sha:
                raise ValueError("original consumed mask changed during probe execution")
            if observation["library_sha256_after"] != library_before:
                raise ValueError("original probe libraries changed during execution")
            if observation["probe_binary_sha256_after"] != probe_before:
                raise ValueError("original compiled probe changed during execution")
            if argv[0] == "c++" and returncode == 0:
                binary = Path(argv[argv.index("-o") + 1])
                self.pin(binary, f"compiled-probe-{index}.bin")
            done.stdout, done.stderr = observation["stdout"], observation["stderr"]
            return done
        return observed

    def finish(self, result):
        record = {"schema": SCHEMA, "request_sha256": digest(canonical(self.request)),
            "capture_id": self.directory.name, "started_at": self.started_at, "ended_at": utc(),
            "decided_proposition": self.proposition, "evidence_mode": self.mode, "authority": AUTHORITY,
            "metric": self.request["metric"], "metric_direction": "higher_better", "category": "CANDIDATE",
            "verdict": result.status, "value": {"pass": True, "wrong": False}.get(result.status),
            "reason": result.reason, "detail": result.detail,
            "observations": [{"name": f"observation-{i:04d}.json", "sha256": digest(canonical(row))}
                             for i, row in enumerate(self.observations)],
            "pins": self.pins}
        record["record_sha256"] = digest(canonical(record))
        exclusive(self.directory / "record.json", canonical(record))
        return self.directory / "record.json"
