"""Immutable, offline resolution of one serving launch recipe.

Resolution consumes caller-pinned artifact digests and an explicit environment policy.
It performs no hashing, building, downloading, process launch, or resource admission.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from . import serving


ARTIFACT_SCHEMA = "epyc.autokernel.launch_artifact.v1"
ENVIRONMENT_POLICY_SCHEMA = "epyc.autokernel.environment_policy.v1"
CAPABILITY_SCHEMA = "epyc.autokernel.recipe_capability.v1"
RESOLVED_RECIPE_SCHEMA = "epyc.autokernel.resolved_recipe.v1"
CANONICAL_RESOLVED_RECIPE_SCHEMA = "epyc.autokernel.canonical_launch.v1"
SUPPORTED_BACKENDS = frozenset({"cpu", "gpu"})
WITNESS_KINDS = frozenset({"recipe_readback", "runtime_set", "master_off"})
SPECULATION_TYPES = frozenset({"none", "draft-dflash", "draft-mtp"})
_GPU_DEVICE = re.compile(r"ROCm[0-9]+")
_CREDENTIAL_MARKERS = ("PASSWORD", "PASSWD", "TOKEN", "SECRET", "CREDENTIAL",
                       "AUTHORIZATION", "API_KEY", "COOKIE", "PRIVATE_KEY")
_STRUCTURED_FLAGS = frozenset({
    "-m", "--model", "-np", "--parallel", "-c", "--ctx-size", "-t", "-tb",
    "-b", "-ub", "-ctk", "-ctv", "--device", "-ngl", "--gpu-layers", "-fa",
    "--flash-attn", "--host", "--port", "--metrics", "--slots", "-md", "-ngld",
    "--spec-type", "--spec-draft-n-max", "--kv-unified", "--no-kv-unified"})


class ResolutionError(ValueError):
    """A recipe-resolution input is malformed or contradicts a pinned identity."""


class UnsupportedRecipeCapability(RuntimeError):
    """A well-formed resolved recipe cannot be launched by this consumer."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResolutionError(f"{label} must be an object")
    return dict(value)


def _keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise ResolutionError(f"{label}: " + "; ".join(details))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResolutionError(f"{label} must be a non-empty string")
    return value


def _env_key(value: Any, label: str) -> str:
    key = _text(value, label)
    if "=" in key or "\0" in key:
        raise ResolutionError(f"{label} is not a valid environment key")
    if any(marker in key.upper() for marker in _CREDENTIAL_MARKERS):
        raise ResolutionError(f"{label} names credential-bearing state, which is forbidden")
    return key


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ResolutionError(f"{label} must be lowercase SHA-256")
    return digest


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResolutionError(f"{label} must be an array of strings")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise ResolutionError(f"{label} contains duplicates")
    return result


def _argv_sequence(value: Any, label: str) -> tuple[str, ...]:
    """Validate ordered argv without treating repeated tokens or values as aliases."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResolutionError(f"{label} must be an array of strings")
    result = tuple(_text(item, f"{label}[]") for item in value)
    if not result:
        raise ResolutionError(f"{label} must not be empty")
    return result


@dataclass(frozen=True)
class ArtifactDigest:
    role: str
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any, *, role: str) -> "ArtifactDigest":
        row = _object(value, f"artifact {role}")
        _keys(row, {"schema", "role", "path", "sha256"}, f"artifact {role}")
        if row["schema"] != ARTIFACT_SCHEMA or row["role"] != role:
            raise ResolutionError(f"artifact {role} identity does not match its role")
        path = _text(row["path"], f"artifact {role}.path")
        if not Path(path).is_absolute():
            raise ResolutionError(f"artifact {role}.path must be absolute")
        return cls(role=role, path=path,
                   sha256=_sha(row["sha256"], f"artifact {role}.sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"schema": ARTIFACT_SCHEMA, "role": self.role,
                "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class EnvironmentPolicy:
    version: str
    measurement_keys: tuple[str, ...]
    allowed_inherit_keys: tuple[str, ...]
    witnesses: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, value: Any) -> "EnvironmentPolicy":
        row = _object(value, "environment policy")
        _keys(row, {"schema", "version", "measurement_keys", "allowed_inherit_keys",
                    "witnesses"}, "environment policy")
        if row["schema"] != ENVIRONMENT_POLICY_SCHEMA:
            raise ResolutionError(
                f"environment policy: unsupported schema {row['schema']!r}")
        measurement = tuple(sorted(
            _env_key(key, "environment policy.measurement_keys[]")
            for key in _string_sequence(row["measurement_keys"],
                                        "environment policy.measurement_keys")))
        inherited = tuple(sorted(
            _env_key(key, "environment policy.allowed_inherit_keys[]")
            for key in _string_sequence(row["allowed_inherit_keys"],
                                        "environment policy.allowed_inherit_keys")))
        if set(measurement) & set(serving.LOADER_OWNED_ENV):
            raise ResolutionError("loader-owned variables cannot be measurement-policy keys")
        if set(inherited) & set(serving.LOADER_OWNED_ENV):
            raise ResolutionError("loader-owned variables cannot be inherited by policy")
        witness_row = _object(row["witnesses"], "environment policy.witnesses")
        if set(witness_row) != set(measurement):
            raise ResolutionError(
                "environment policy.witnesses must name every measurement key exactly")
        witnesses = []
        for key, kind in witness_row.items():
            witness_key = _env_key(key, "environment policy.witnesses key")
            witness_kind = _text(kind, f"environment policy.witnesses.{key}")
            if witness_kind not in WITNESS_KINDS:
                raise ResolutionError(
                    f"environment policy.witnesses.{key} has unsupported declaration")
            witnesses.append((witness_key, witness_kind))
        return cls(version=_text(row["version"], "environment policy.version"),
                   measurement_keys=measurement, allowed_inherit_keys=inherited,
                   witnesses=tuple(sorted(witnesses)))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": ENVIRONMENT_POLICY_SCHEMA, "version": self.version,
                "measurement_keys": list(self.measurement_keys),
                "allowed_inherit_keys": list(self.allowed_inherit_keys),
                "witnesses": dict(self.witnesses)}


@dataclass(frozen=True)
class WitnessReport:
    key: str
    kind: str
    status: str
    field: str | None = None
    expected: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind, "status": self.status,
                "field": self.field, "expected": self.expected, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: Any) -> "WitnessReport":
        row = _object(value, "witness report")
        _keys(row, {"key", "kind", "status", "field", "expected", "reason"},
              "witness report")
        status = _text(row["status"], "witness report.status")
        if status not in {"declared", "unknown"}:
            raise ResolutionError("witness report.status must be declared or unknown")
        for field in ("field", "expected", "reason"):
            if row[field] is not None and not isinstance(row[field], str):
                raise ResolutionError(f"witness report.{field} must be a string or null")
        return cls(key=_env_key(row["key"], "witness report.key"),
                   kind=_text(row["kind"], "witness report.kind"), status=status,
                   field=row["field"], expected=row["expected"], reason=row["reason"])


@dataclass(frozen=True)
class CapabilityReason:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: Any) -> "CapabilityReason":
        row = _object(value, "capability reason")
        _keys(row, {"code", "detail"}, "capability reason")
        return cls(_text(row["code"], "capability reason.code"),
                   _text(row["detail"], "capability reason.detail"))


@dataclass(frozen=True)
class CapabilityReport:
    supported: bool
    backend: str
    speculation: str
    reasons: tuple[CapabilityReason, ...]
    witnesses: tuple[WitnessReport, ...]
    gpu_residency: str
    cpu_placement: str
    contention: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CAPABILITY_SCHEMA, "supported": self.supported,
                "backend": self.backend, "speculation": self.speculation,
                "reasons": [reason.to_dict() for reason in self.reasons],
                "witnesses": [witness.to_dict() for witness in self.witnesses],
                "gpu_residency": self.gpu_residency,
                "cpu_placement": self.cpu_placement, "contention": self.contention}

    @classmethod
    def from_dict(cls, value: Any) -> "CapabilityReport":
        row = _object(value, "capability report")
        _keys(row, {"schema", "supported", "backend", "speculation", "reasons",
                    "witnesses", "gpu_residency", "cpu_placement", "contention"},
              "capability report")
        if row["schema"] != CAPABILITY_SCHEMA or not isinstance(row["supported"], bool):
            raise ResolutionError("malformed capability report schema/supported flag")
        if not isinstance(row["reasons"], list) or not isinstance(row["witnesses"], list):
            raise ResolutionError("capability reasons/witnesses must be arrays")
        reasons = tuple(CapabilityReason.from_dict(item) for item in row["reasons"])
        if row["supported"] == bool(reasons):
            raise ResolutionError("capability supported flag and reasons disagree")
        return cls(supported=row["supported"],
                   backend=_text(row["backend"], "capability report.backend"),
                   speculation=_text(row["speculation"], "capability report.speculation"),
                   reasons=reasons,
                   witnesses=tuple(WitnessReport.from_dict(item) for item in row["witnesses"]),
                   gpu_residency=_text(row["gpu_residency"], "capability.gpu_residency"),
                   cpu_placement=_text(row["cpu_placement"], "capability.cpu_placement"),
                   contention=_text(row["contention"], "capability.contention"))


@dataclass(frozen=True)
class WorkloadSpec:
    n_predict: int
    temperature: float
    top_p: float
    top_k: int
    metric: str

    def to_dict(self) -> dict[str, Any]:
        return {"n_predict": self.n_predict, "temperature": self.temperature,
                "top_p": self.top_p, "top_k": self.top_k, "metric": self.metric}

    @classmethod
    def from_dict(cls, value: Any) -> "WorkloadSpec":
        row = _object(value, "resolved workload")
        _keys(row, {"n_predict", "temperature", "top_p", "top_k", "metric"},
              "resolved workload")
        result = cls(row["n_predict"], row["temperature"], row["top_p"], row["top_k"],
                     _text(row["metric"], "resolved workload.metric"))
        _validate_workload(result)
        return result


@dataclass(frozen=True)
class ResolvedRecipe:
    template_hash: str
    backend: str
    build_dir: str
    port: int
    argv: tuple[str, ...]
    launch_env: tuple[tuple[str, str], ...]
    absent_environment: tuple[str, ...]
    relevant_environment: tuple[tuple[str, str | None], ...]
    readback_expectations: tuple[tuple[str, str], ...]
    workload: WorkloadSpec
    environment_policy: EnvironmentPolicy
    model: ArtifactDigest
    drafter: ArtifactDigest | None
    executable: ArtifactDigest
    dsos: tuple[ArtifactDigest, ...]
    capability: CapabilityReport
    snapshot_digest: str
    execution_digest: str

    def _snapshot_dict(self) -> dict[str, Any]:
        return {"template_hash": self.template_hash, "backend": self.backend,
                "build_dir": self.build_dir, "port": self.port, "argv": list(self.argv),
                "launch_env": dict(self.launch_env),
                "absent_environment": list(self.absent_environment),
                "relevant_environment": {
                    key: ({"state": "absent"} if value is None
                          else {"state": "value", "value": value})
                    for key, value in self.relevant_environment},
                "readback_expectations": [list(item) for item in self.readback_expectations],
                "workload": self.workload.to_dict(),
                "environment_policy": self.environment_policy.to_dict(),
                "model": self.model.to_dict(),
                "drafter": self.drafter.to_dict() if self.drafter else None,
                "executable": self.executable.to_dict(),
                "dsos": [item.to_dict() for item in self.dsos],
                "capability": self.capability.to_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESOLVED_RECIPE_SCHEMA, **self._snapshot_dict(),
                "snapshot_digest": self.snapshot_digest,
                "execution_digest": self.execution_digest}

    def _normalized_execution_dict(self) -> dict[str, Any]:
        argv = list(self.argv)
        executable_index = 3 if len(argv) >= 4 and argv[:2] == ["taskset", "-c"] else 0
        argv[executable_index] = f"<executable:{self.executable.sha256}>"
        replacements = {"-m": f"<model:{self.model.sha256}>",
                        "--port": "<listen-port>"}
        if self.drafter is not None:
            replacements["-md"] = f"<drafter:{self.drafter.sha256}>"
        for index, token in enumerate(argv[:-1]):
            if token in replacements:
                argv[index + 1] = replacements[token]
        normalized_env = dict(self.launch_env)
        normalized_env["LD_LIBRARY_PATH"] = "<sealed-dso-set>"
        return {"backend": self.backend, "argv": argv,
                "launch_env": normalized_env,
                "absent_environment": list(self.absent_environment),
                "relevant_environment": {
                    key: ({"state": "absent"} if value is None
                          else {"state": "value", "value": value})
                    for key, value in self.relevant_environment},
                "readback_expectations": [list(item) for item in self.readback_expectations],
                "environment_policy_version": self.environment_policy.version,
                "workload": self.workload.to_dict(),
                "artifacts": {"model": self.model.sha256,
                              "drafter": self.drafter.sha256 if self.drafter else None,
                              "executable": self.executable.sha256,
                              "dsos": sorted(
                                  ({"load_name": Path(item.path).name,
                                    "sha256": item.sha256} for item in self.dsos),
                                  key=lambda item: (item["load_name"], item["sha256"]))}}

    def validate_launch(self, template: serving.Recipe, build_dir: Path | str,
                        port: int) -> None:
        if self.snapshot_digest != _digest(self._snapshot_dict()):
            raise ResolutionError("resolved recipe snapshot integrity check failed")
        if self.execution_digest != _digest(self._normalized_execution_dict()):
            raise ResolutionError("resolved recipe execution integrity check failed")
        _validate_resolved_consistency(self)
        if not self.capability.supported:
            codes = ",".join(reason.code for reason in self.capability.reasons)
            raise UnsupportedRecipeCapability(f"resolved recipe is unsupported: {codes}")
        if template.recipe_hash != self.template_hash:
            raise ResolutionError("resolved recipe does not match the passed template")
        if str(Path(build_dir)) != self.build_dir or port != self.port:
            raise ResolutionError("resolved recipe does not match the passed build/port")
        if tuple(template.server_argv(Path(build_dir), port)) != self.argv:
            raise ResolutionError("resolved launch argv no longer matches the template")
        if WorkloadSpec(template.n_predict, template.temperature, template.top_p,
                        template.top_k, template.metric) != self.workload:
            raise ResolutionError("resolved workload no longer matches the template")
        relevant = dict(self.relevant_environment)
        if ((set(template.env or {}) | set(template.explicit_unsets))
                - set(self.environment_policy.measurement_keys)):
            raise ResolutionError("template environment is not covered by resolved policy")
        if any(relevant.get(key) != value for key, value in (template.env or {}).items()):
            raise ResolutionError("template environment set values differ from frozen launch")
        if any(relevant.get(key, "<missing>") is not None
               for key in template.explicit_unsets):
            raise ResolutionError("template environment unsets differ from frozen launch")
        effective_env = dict(template.env or {})
        effective_unsets = set(template.explicit_unsets)
        for key, value in self.relevant_environment:
            if value is None:
                effective_env.pop(key, None)
                effective_unsets.add(key)
            else:
                effective_env[key] = value
                effective_unsets.discard(key)
        effective_template = replace(
            template, env=effective_env, explicit_unsets=tuple(sorted(effective_unsets)))
        if effective_template.readback_expectations() != self.readback_expectations:
            raise ResolutionError("resolved readback expectations no longer match the launch")
        expected_capability = _capability(
            template, self.backend, self.drafter,
            _witness_reports(template, self.environment_policy,
                             self.readback_expectations))
        if expected_capability != self.capability:
            raise ResolutionError("resolved capability is not derived from the actual launch")

    @classmethod
    def from_dict(cls, value: Any) -> "ResolvedRecipe":
        row = _object(value, "resolved recipe")
        required = {"schema", "template_hash", "backend", "build_dir", "port", "argv",
                    "launch_env", "absent_environment", "relevant_environment",
                    "readback_expectations", "workload", "environment_policy", "model",
                    "drafter", "executable", "dsos", "capability", "snapshot_digest",
                    "execution_digest"}
        _keys(row, required, "resolved recipe")
        if row["schema"] != RESOLVED_RECIPE_SCHEMA:
            raise ResolutionError(f"resolved recipe: unsupported schema {row['schema']!r}")
        launch_row = _object(row["launch_env"], "resolved recipe.launch_env")
        launch_env = tuple(sorted((_env_key(key, "resolved launch env key"),
                                   item if isinstance(item, str) else
                                   (_ for _ in ()).throw(ResolutionError(
                                       "resolved launch env values must be strings")))
                                  for key, item in launch_row.items()))
        relevant_row = _object(row["relevant_environment"], "relevant environment")
        relevant = []
        for key, state_value in relevant_row.items():
            state = _object(state_value, f"relevant environment {key}")
            env_key = _env_key(key, "relevant environment key")
            if state == {"state": "absent"}:
                relevant.append((env_key, None))
            elif set(state) == {"state", "value"} and state["state"] == "value" \
                    and isinstance(state["value"], str):
                relevant.append((env_key, state["value"]))
            else:
                raise ResolutionError(f"relevant environment {key} has malformed state")
        if not isinstance(row["dsos"], list):
            raise ResolutionError("resolved recipe.dsos must be an array")
        resolved = cls(
            template_hash=_sha(row["template_hash"], "resolved recipe.template_hash"),
            backend=_text(row["backend"], "resolved recipe.backend"),
            build_dir=_text(row["build_dir"], "resolved recipe.build_dir"),
            port=_port(row["port"]), argv=_argv_sequence(row["argv"], "resolved recipe.argv"),
            launch_env=launch_env,
            absent_environment=tuple(sorted(
                _env_key(key, "resolved absent environment[]")
                for key in _string_sequence(row["absent_environment"],
                                            "resolved absent environment"))),
            relevant_environment=tuple(sorted(relevant)),
            readback_expectations=_readback_pairs(
                row["readback_expectations"], "resolved recipe.readback_expectations"),
            workload=WorkloadSpec.from_dict(row["workload"]),
            environment_policy=EnvironmentPolicy.from_dict(row["environment_policy"]),
            model=ArtifactDigest.from_dict(row["model"], role="model"),
            drafter=(None if row["drafter"] is None else
                     ArtifactDigest.from_dict(row["drafter"], role="drafter")),
            executable=ArtifactDigest.from_dict(row["executable"], role="executable"),
            dsos=tuple(ArtifactDigest.from_dict(item, role="dso") for item in row["dsos"]),
            capability=CapabilityReport.from_dict(row["capability"]),
            snapshot_digest=_sha(row["snapshot_digest"], "resolved recipe.snapshot_digest"),
            execution_digest=_sha(row["execution_digest"], "resolved recipe.execution_digest"))
        if not resolved.dsos:
            raise ResolutionError("resolved recipe needs a non-empty DSO identity set")
        if resolved.snapshot_digest != _digest(resolved._snapshot_dict()):
            raise ResolutionError("resolved recipe snapshot_digest does not match its contents")
        if resolved.execution_digest != _digest(resolved._normalized_execution_dict()):
            raise ResolutionError("resolved recipe execution_digest does not match normalized launch")
        _validate_resolved_consistency(resolved)
        return resolved


_CANONICAL_VALUE_FLAGS = frozenset({
    "-m", "--host", "--port", "-np", "-c", "-t", "-tb", "-b", "-ub",
    "--flash-attn", "-fa", "-ctk", "-ctv", "--chat-template-file", "--spec-type",
    "--spec-draft-n-max", "--spec-draft-p-min", "--reasoning", "--slot-save-path", "--device", "-lv",
    "--device-draft", "-ngl", "-md", "-ngld",
})
_CANONICAL_SWITCH_FLAGS = frozenset({
    "--jinja", "--mlock", "--no-mmap", "--kv-unified", "--no-kv-unified",
    "--metrics", "--slots", "--no-webui",
})


def _canonical_command(command: tuple[str, ...]) -> tuple[str, dict[str, str | bool]]:
    if not command:
        raise ResolutionError("canonical command must not be empty")
    executable = _text(command[0], "canonical command executable")
    parsed: dict[str, str | bool] = {}
    index = 1
    while index < len(command):
        token = command[index]
        if token in _CANONICAL_SWITCH_FLAGS:
            if token in parsed:
                raise ResolutionError(f"canonical command repeats {token}")
            parsed[token] = True
            index += 1
            continue
        if token in _CANONICAL_VALUE_FLAGS:
            if token in parsed or index + 1 >= len(command):
                raise ResolutionError(f"canonical command repeats or omits value for {token}")
            parsed[token] = _text(command[index + 1], f"canonical command {token}")
            index += 2
            continue
        # Equals spellings make duplicate detection ambiguous with launcher output and
        # are not emitted by the supported canonical builder grammar.
        raise ResolutionError(f"canonical command has unsupported flag {token!r}")
    for required in ("-m", "--host", "--port", "-np", "-c", "-t"):
        if required not in parsed:
            raise ResolutionError(f"canonical command omits required {required}")
    if parsed["--host"] != "127.0.0.1":
        raise ResolutionError("canonical command must use the loopback host")
    if "--spec-draft-p-min" in parsed:
        try:
            probability = float(parsed["--spec-draft-p-min"])
        except (TypeError, ValueError) as exc:
            raise ResolutionError("canonical draft probability must be finite in [0,1]") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ResolutionError("canonical draft probability must be finite in [0,1]")
    if "-lv" in parsed:
        _canonical_int(parsed, "-lv")
    return executable, parsed


def _canonical_prefix(prefix: tuple[str, ...]) -> str | None:
    if not prefix:
        return None
    offset = 0
    if prefix[0] == "numactl":
        if len(prefix) < 4 or prefix[2] != "--":
            raise ResolutionError("malformed canonical numactl prefix")
        policy = prefix[1]
        if not re.fullmatch(r"--(?:interleave=(?:all|[0-9]+(?:,[0-9]+)*)|"
                            r"membind=[0-9]+(?:,[0-9]+)*)", policy):
            raise ResolutionError("unsupported canonical numactl policy")
        offset = 3
    if len(prefix) != offset + 3 or prefix[offset:offset + 2] != ("taskset", "-c"):
        raise ResolutionError("canonical topology prefix must end in one taskset declaration")
    cpu_list = _text(prefix[-1], "canonical topology CPU list")
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", cpu_list):
        raise ResolutionError("canonical topology CPU list is malformed")
    return cpu_list


def canonical_recipe_projection(*, name: str, command_argv: Sequence[str],
                                topology_prefix: Sequence[str], n_predict: int = 256,
                                temperature: float = 0.6, top_p: float = 0.95,
                                top_k: int = 20, metric: str = "aggregate_tok_s"
                                ) -> serving.Recipe:
    """Parse the closed production grammar into Recipe's semantic fields."""
    command = _argv_sequence(command_argv, "canonical command_argv")
    executable, parsed = _canonical_command(command)
    del executable
    if isinstance(topology_prefix, (str, bytes)) or not isinstance(topology_prefix, Sequence):
        raise ResolutionError("topology_prefix must be an array")
    prefix = tuple(_text(x, "topology_prefix[]") for x in topology_prefix)
    cpu_list = _canonical_prefix(prefix)
    device = str(parsed.get("--device", "none"))
    raw_ngl = parsed.get("-ngl")
    ngl = 99 if raw_ngl == "all" else (
        0 if raw_ngl is None and device == "none" else _canonical_int(parsed, "-ngl"))
    spec_type = str(parsed.get("--spec-type", "none"))
    spec: dict[str, Any] = {"type": spec_type}
    if "-md" in parsed:
        spec["drafter"] = str(parsed["-md"])
        spec["ngld"] = _canonical_int(parsed, "-ngld", ngl)
    if "--spec-draft-n-max" in parsed:
        spec["draft_n_max"] = _canonical_int(parsed, "--spec-draft-n-max")
    extra: tuple[str, ...] = ()
    if "--device-draft" in parsed:
        extra = ("--device-draft", str(parsed["--device-draft"]))
    if "-tb" in parsed and _canonical_int(parsed, "-tb") != _canonical_int(parsed, "-t"):
        raise ResolutionError("canonical -tb differs from -t")
    if "--flash-attn" in parsed and "-fa" in parsed:
        raise ResolutionError("canonical command repeats flash-attention through aliases")
    kv_unified = "--kv-unified" in parsed
    if kv_unified and "--no-kv-unified" in parsed:
        raise ResolutionError("canonical command declares conflicting KV-unified states")
    return serving.Recipe(
        name=_text(name, "canonical recipe name"), model=str(parsed["-m"]),
        device=device, ngl=ngl, spec_decode=spec, np=_canonical_int(parsed, "-np"),
        ctx=_canonical_int(parsed, "-c"), threads=_canonical_int(parsed, "-t"),
        batch=_canonical_int(parsed, "-b", 2048), ubatch=_canonical_int(parsed, "-ub", 2048),
        ctk=str(parsed.get("-ctk", "f16")), ctv=str(parsed.get("-ctv", "f16")),
        fa=str(parsed.get("--flash-attn", parsed.get("-fa", "off"))),
        kv_unified=kv_unified, extra_flags=extra, cpu_list=cpu_list,
        n_predict=n_predict, temperature=temperature, top_p=top_p, top_k=top_k,
        metric=metric)


@dataclass(frozen=True)
class CanonicalResolvedRecipe:
    """Exact production command snapshot under a closed, versioned launch grammar."""

    template_hash: str
    template: serving.Recipe
    backend: str
    build_dir: str
    port: int
    command_argv: tuple[str, ...]
    topology_prefix: tuple[str, ...]
    argv: tuple[str, ...]
    launch_env: tuple[tuple[str, str], ...]
    absent_environment: tuple[str, ...]
    relevant_environment: tuple[tuple[str, str | None], ...]
    readback_expectations: tuple[tuple[str, str], ...]
    workload: WorkloadSpec
    environment_policy: EnvironmentPolicy
    model: ArtifactDigest
    drafter: ArtifactDigest | None
    executable: ArtifactDigest
    dsos: tuple[ArtifactDigest, ...]
    capability: CapabilityReport
    runtime_binary_dir: str | None
    runtime_ld_paths: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]
    snapshot_digest: str
    execution_digest: str

    def _snapshot_dict(self) -> dict[str, Any]:
        return {"launch_contract": "orchestrator-production/v1",
                "template_hash": self.template_hash, "template": self.template.to_dict(),
                "backend": self.backend,
                "build_dir": self.build_dir, "port": self.port,
                "command_argv": list(self.command_argv),
                "topology_prefix": list(self.topology_prefix), "argv": list(self.argv),
                "launch_env": dict(self.launch_env),
                "absent_environment": list(self.absent_environment),
                "relevant_environment": {
                    key: ({"state": "absent"} if value is None else
                          {"state": "value", "value": value})
                    for key, value in self.relevant_environment},
                "readback_expectations": [list(item) for item in self.readback_expectations],
                "workload": self.workload.to_dict(),
                "environment_policy": self.environment_policy.to_dict(),
                "model": self.model.to_dict(),
                "drafter": self.drafter.to_dict() if self.drafter else None,
                "executable": self.executable.to_dict(),
                "dsos": [item.to_dict() for item in self.dsos],
                "capability": self.capability.to_dict(),
                "runtime_binary_dir": self.runtime_binary_dir,
                "runtime_ld_paths": list(self.runtime_ld_paths),
                "provenance": dict(self.provenance)}

    def _normalized_execution_dict(self) -> dict[str, Any]:
        command = list(self.command_argv)
        command[0] = f"<executable:{self.executable.sha256}>"
        replacements = {"-m": f"<model:{self.model.sha256}>", "--port": "<listen-port>"}
        if self.drafter:
            replacements["-md"] = f"<drafter:{self.drafter.sha256}>"
        for index, token in enumerate(command[:-1]):
            if token in replacements:
                command[index + 1] = replacements[token]
        env = dict(self.launch_env)
        env["LD_LIBRARY_PATH"] = "<sealed-dso-set>"
        return {"launch_contract": "orchestrator-production/v1", "backend": self.backend,
                "command_argv": command, "topology_prefix": list(self.topology_prefix),
                "launch_env": env, "absent_environment": list(self.absent_environment),
                "relevant_environment": dict(self.relevant_environment),
                "workload": self.workload.to_dict(),
                "environment_policy_version": self.environment_policy.version,
                "artifacts": {"model": self.model.sha256,
                              "drafter": self.drafter.sha256 if self.drafter else None,
                              "executable": self.executable.sha256,
                              "dsos": sorted((Path(x.path).name, x.sha256) for x in self.dsos)}}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CANONICAL_RESOLVED_RECIPE_SCHEMA, **self._snapshot_dict(),
                "snapshot_digest": self.snapshot_digest,
                "execution_digest": self.execution_digest}

    def validate_launch(self, template: serving.Recipe, build_dir: Path | str,
                        port: int) -> None:
        if self.snapshot_digest != _digest(self._snapshot_dict()) \
                or self.execution_digest != _digest(self._normalized_execution_dict()):
            raise ResolutionError("canonical launch integrity check failed")
        _validate_canonical_consistency(self, template)
        if str(Path(build_dir)) != self.build_dir or port != self.port:
            raise ResolutionError("canonical launch does not match build/port")
        if not self.capability.supported:
            codes = ",".join(item.code for item in self.capability.reasons)
            raise UnsupportedRecipeCapability(f"canonical launch is unsupported: {codes}")

    @classmethod
    def from_dict(cls, value: Any) -> "CanonicalResolvedRecipe":
        row = _object(value, "canonical resolved recipe")
        required = {"schema", "launch_contract", "template_hash", "template", "backend", "build_dir",
                    "port", "command_argv", "topology_prefix", "argv", "launch_env",
                    "absent_environment", "relevant_environment", "readback_expectations",
                    "workload", "environment_policy", "model", "drafter", "executable",
                    "dsos", "capability", "runtime_binary_dir", "runtime_ld_paths",
                    "provenance", "snapshot_digest", "execution_digest"}
        _keys(row, required, "canonical resolved recipe")
        if row["schema"] != CANONICAL_RESOLVED_RECIPE_SCHEMA \
                or row["launch_contract"] != "orchestrator-production/v1":
            raise ResolutionError("unsupported canonical launch schema/contract")
        policy = EnvironmentPolicy.from_dict(row["environment_policy"])
        launch = _object(row["launch_env"], "canonical launch env")
        relevant_row = _object(row["relevant_environment"], "canonical relevant env")
        relevant = []
        for key, raw in relevant_row.items():
            state = _object(raw, f"canonical relevant env {key}")
            if state == {"state": "absent"}:
                relevant.append((_env_key(key, "canonical relevant env key"), None))
            elif set(state) == {"state", "value"} and state["state"] == "value" \
                    and isinstance(state["value"], str):
                relevant.append((_env_key(key, "canonical relevant env key"), state["value"]))
            else:
                raise ResolutionError("malformed canonical relevant environment")
        runtime_binary = row["runtime_binary_dir"]
        if runtime_binary is not None:
            runtime_binary = _text(runtime_binary, "runtime_binary_dir")
        try:
            frozen_template = serving.Recipe.from_dict(row["template"])
        except Exception as exc:
            raise ResolutionError(f"malformed canonical template: {exc}") from exc
        resolved = cls(
            _sha(row["template_hash"], "template_hash"), frozen_template,
            _text(row["backend"], "backend"), _text(row["build_dir"], "build_dir"),
            _port(row["port"]), _argv_sequence(row["command_argv"], "command_argv"),
            tuple(_text(x, "topology_prefix[]") for x in
                  (row["topology_prefix"] if isinstance(row["topology_prefix"], list) else
                   (_ for _ in ()).throw(ResolutionError("topology_prefix must be an array")))),
            _argv_sequence(row["argv"], "argv"),
            tuple(sorted((_env_key(k, "launch env key"),
                          v if isinstance(v, str) else (_ for _ in ()).throw(
                              ResolutionError("launch env values must be strings")))
                         for k, v in launch.items())),
            tuple(sorted(_env_key(x, "absent env") for x in
                         _string_sequence(row["absent_environment"], "absent env"))),
            tuple(sorted(relevant)),
            _readback_pairs(row["readback_expectations"], "readback expectations"),
            WorkloadSpec.from_dict(row["workload"]), policy,
            ArtifactDigest.from_dict(row["model"], role="model"),
            None if row["drafter"] is None else ArtifactDigest.from_dict(row["drafter"], role="drafter"),
            ArtifactDigest.from_dict(row["executable"], role="executable"),
            tuple(ArtifactDigest.from_dict(x, role="dso") for x in row["dsos"]),
            CapabilityReport.from_dict(row["capability"]), runtime_binary,
            tuple(_text(x, "runtime_ld_paths[]") for x in
                  _string_sequence(row["runtime_ld_paths"], "runtime_ld_paths")),
            tuple(sorted((_text(k, "provenance key"), _text(v, f"provenance.{k}"))
                         for k, v in _object(row["provenance"], "provenance").items())),
            _sha(row["snapshot_digest"], "snapshot_digest"),
            _sha(row["execution_digest"], "execution_digest"))
        if resolved.snapshot_digest != _digest(resolved._snapshot_dict()) \
                or resolved.execution_digest != _digest(resolved._normalized_execution_dict()):
            raise ResolutionError("canonical resolved recipe integrity check failed")
        _validate_canonical_consistency(resolved, resolved.template)
        return resolved


def _port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ResolutionError("port must be an integer from 1 through 65535")
    return value


def _readback_pairs(value: Any, label: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResolutionError(f"{label} must be an array")
    result = []
    for item in value:
        if isinstance(item, (str, bytes)) or not isinstance(item, Sequence) or len(item) != 2:
            raise ResolutionError(f"{label} entries must be [field, expected] pairs")
        result.append((_text(item[0], f"{label}.field"),
                       _text(item[1], f"{label}.expected")))
    return tuple(result)


def _validate_workload(workload: WorkloadSpec) -> None:
    for name in ("n_predict", "top_k"):
        value = getattr(workload, name)
        minimum = 1 if name == "n_predict" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ResolutionError(f"recipe.{name} must be a finite integer >= {minimum}")
    for name in ("temperature", "top_p"):
        value = getattr(workload, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise ResolutionError(f"recipe.{name} must be finite")
    if not 0.0 <= float(workload.top_p) <= 1.0 or float(workload.temperature) < 0.0:
        raise ResolutionError("recipe temperature/top_p are outside their valid ranges")


def _validate_template(template: serving.Recipe) -> None:
    integer_fields = ("ngl", "np", "ctx", "threads", "batch", "ubatch")
    for name in integer_fields:
        value = getattr(template, name)
        minimum = 0 if name in {"ngl", "top_k"} else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ResolutionError(f"recipe.{name} must be a finite integer >= {minimum}")
    _validate_workload(WorkloadSpec(template.n_predict, template.temperature, template.top_p,
                                    template.top_k, template.metric))
    if not isinstance(template.extra_flags, tuple) or not all(
            isinstance(flag, str) and flag for flag in template.extra_flags):
        raise ResolutionError("recipe.extra_flags must be non-empty strings")
    spec = _object(template.spec_decode, "recipe.spec_decode")
    spec_type = _text(spec.get("type", "none"), "recipe.spec_decode.type")
    if spec_type not in SPECULATION_TYPES:
        raise ResolutionError(f"unknown recipe speculation type {spec_type!r}")
    allowed_spec = ({"type"} if spec_type == "none" else
                    {"type", "drafter", "ngld", "draft_n_max"})
    if set(spec) - allowed_spec:
        raise ResolutionError(
            f"recipe.spec_decode has unknown fields {sorted(set(spec) - allowed_spec)}")
    if spec_type == "none" and spec.get("drafter"):
        raise ResolutionError("speculation type none cannot declare a drafter")
    if "ngld" in spec:
        ngld = spec["ngld"]
        if isinstance(ngld, bool) or not isinstance(ngld, int) or ngld < 0:
            raise ResolutionError("recipe.spec_decode.ngld must be a non-negative integer")
        if not spec.get("drafter"):
            raise ResolutionError("recipe.spec_decode.ngld requires an external drafter")
    if "draft_n_max" in spec:
        draft_n = spec["draft_n_max"]
        if isinstance(draft_n, bool) or not isinstance(draft_n, int) or draft_n <= 0:
            raise ResolutionError("recipe.spec_decode.draft_n_max must be a positive integer")
    flags = tuple(template.extra_flags)
    conflicts = sorted({flag.split("=", 1)[0] for flag in flags
                        if flag.split("=", 1)[0] in _STRUCTURED_FLAGS})
    if conflicts:
        raise ResolutionError(
            f"recipe.extra_flags override structured launch fields: {conflicts}")


def _artifact_set(value: Any) -> tuple[ArtifactDigest, ArtifactDigest | None,
                                       ArtifactDigest, tuple[ArtifactDigest, ...]]:
    row = _object(value, "artifact identities")
    _keys(row, {"model", "drafter", "executable", "dsos"}, "artifact identities")
    if isinstance(row["dsos"], (str, bytes)) or not isinstance(row["dsos"], Sequence):
        raise ResolutionError("artifact identities.dsos must be an array")
    model = ArtifactDigest.from_dict(row["model"], role="model")
    drafter = (None if row["drafter"] is None else
               ArtifactDigest.from_dict(row["drafter"], role="drafter"))
    executable = ArtifactDigest.from_dict(row["executable"], role="executable")
    dsos = tuple(sorted((ArtifactDigest.from_dict(item, role="dso")
                         for item in row["dsos"]), key=lambda item: item.path))
    if (not dsos or len({item.path for item in dsos}) != len(dsos)
            or len({Path(item.path).name for item in dsos}) != len(dsos)):
        raise ResolutionError(
            "artifact identities need a non-empty DSO set with unique loader names")
    return model, drafter, executable, dsos


def _flag_value(flags: tuple[str, ...], name: str) -> str | None:
    values = []
    for index, flag in enumerate(flags):
        if flag == name:
            values.append(flags[index + 1] if index + 1 < len(flags) else "<missing>")
        elif flag.startswith(name + "="):
            values.append(flag.split("=", 1)[1])
    if len(values) > 1:
        return "<multiple>"
    if not values:
        return None
    return values[0]


def _required_argv_value(argv: tuple[str, ...], name: str) -> str:
    value = _flag_value(argv, name)
    if value in {None, "<multiple>", "<missing>"}:
        raise ResolutionError(f"resolved argv needs exactly one valid {name}")
    return value


def _validate_resolved_consistency(resolved: ResolvedRecipe) -> None:
    """Recheck cross-field claims independently of the content digest."""
    if resolved.capability.backend != resolved.backend:
        raise ResolutionError("resolved backend disagrees with capability report")
    if (len({item.path for item in resolved.dsos}) != len(resolved.dsos)
            or len({Path(item.path).name for item in resolved.dsos}) != len(resolved.dsos)):
        raise ResolutionError("resolved DSO paths and loader names must be unique")
    executable = (resolved.argv[3] if len(resolved.argv) >= 4
                  and resolved.argv[:2] == ("taskset", "-c") else resolved.argv[0])
    expected_executable = str(Path(resolved.build_dir) / "bin" / "llama-server")
    if resolved.executable.path != expected_executable or executable != expected_executable:
        raise ResolutionError("resolved executable/build/argv identities disagree")
    if _required_argv_value(resolved.argv, "-m") != resolved.model.path:
        raise ResolutionError("resolved model identity disagrees with argv")
    if _required_argv_value(resolved.argv, "--port") != str(resolved.port):
        raise ResolutionError("resolved port disagrees with argv")
    device = _required_argv_value(resolved.argv, "--device")
    try:
        ngl = int(_required_argv_value(resolved.argv, "-ngl"))
    except ValueError as exc:
        raise ResolutionError("resolved -ngl is not an integer") from exc
    spec_type = _flag_value(resolved.argv, "--spec-type")
    drafter_path = _flag_value(resolved.argv, "-md")
    if spec_type is not None and spec_type not in SPECULATION_TYPES - {"none"}:
        raise ResolutionError("resolved argv has unknown speculation type")
    speculation = ("none" if spec_type is None else
                   "external_draft" if drafter_path is not None else "self_draft")
    if resolved.capability.speculation != speculation:
        raise ResolutionError("resolved speculation disagrees with argv")
    if drafter_path is None and resolved.drafter is not None:
        raise ResolutionError("resolved drafter identity is absent from argv")
    if drafter_path is not None and resolved.drafter is None:
        if (resolved.capability.supported or "drafter_identity_missing" not in
                {reason.code for reason in resolved.capability.reasons}):
            raise ResolutionError("resolved drafter identity is missing without refusal")
    elif (drafter_path is not None and resolved.drafter is not None
          and resolved.drafter.path != drafter_path):
        raise ResolutionError("resolved drafter identity disagrees with argv")

    expected_residency = ("not_applicable" if resolved.backend == "cpu" else "required")
    if resolved.capability.gpu_residency != expected_residency:
        raise ResolutionError("resolved GPU-residency requirement disagrees with backend")
    if resolved.capability.supported:
        if resolved.backend == "cpu" and (device != "none" or ngl != 0):
            raise ResolutionError("supported CPU capability contradicts resolved argv")
        if resolved.backend == "gpu" and (
                not _GPU_DEVICE.fullmatch(device) or ngl <= 0):
            raise ResolutionError("supported GPU capability contradicts resolved argv")
        if resolved.backend not in SUPPORTED_BACKENDS:
            raise ResolutionError("supported capability names an unsupported backend")
        if speculation == "external_draft":
            try:
                ngld = int(_required_argv_value(resolved.argv, "-ngld"))
            except ValueError as exc:
                raise ResolutionError("resolved -ngld is not an integer") from exc
            draft_device = _flag_value(resolved.argv, "--device-draft")
            if resolved.backend == "cpu" and (ngld != 0 or draft_device != "none"):
                raise ResolutionError("supported CPU capability has mixed draft placement")
            if resolved.backend == "gpu" and (
                    ngld <= 0 or (draft_device is not None and draft_device != device)):
                raise ResolutionError("supported GPU capability has mixed draft placement")
        if ("," in device or any("rpc" in token.lower() for token in resolved.argv)):
            raise ResolutionError("supported capability contains multi-device/RPC argv")

    policy = resolved.environment_policy
    if EnvironmentPolicy.from_dict(policy.to_dict()) != policy:
        raise ResolutionError("resolved environment policy is not canonical")
    relevant = dict(resolved.relevant_environment)
    launch = dict(resolved.launch_env)
    if set(relevant) != set(policy.measurement_keys):
        raise ResolutionError("relevant environment does not match measurement policy")
    absent = {key for key, value in relevant.items() if value is None}
    if absent != set(resolved.absent_environment):
        raise ResolutionError("explicit absent environment does not match relevant state")
    allowed_launch = (set(policy.measurement_keys) | set(policy.allowed_inherit_keys)
                      | {"LD_LIBRARY_PATH"})
    if set(launch) - allowed_launch or "HSA_OVERRIDE_GFX_VERSION" in launch:
        raise ResolutionError("resolved launch environment exceeds its allowlist")
    if launch.get("LD_LIBRARY_PATH") != str(Path(resolved.build_dir) / "bin"):
        raise ResolutionError("resolved loader environment does not match build")
    for key, value in relevant.items():
        if value is None and key in launch:
            raise ResolutionError("resolved absent environment is present at launch")
        if value is not None and launch.get(key) != value:
            raise ResolutionError("resolved relevant environment differs from launch")
    witness_by_key = {item.key: item for item in resolved.capability.witnesses}
    if set(witness_by_key) != set(policy.measurement_keys):
        raise ResolutionError("capability witnesses do not match environment policy")
    for key, kind in policy.witnesses:
        if witness_by_key[key].kind != kind:
            raise ResolutionError("capability witness kind disagrees with policy")
        witness = witness_by_key[key]
        if witness.status == "declared" and (witness.field, witness.expected) not in \
                resolved.readback_expectations:
            raise ResolutionError("declared capability witness is absent from readback checks")


def _witness_reports(template: serving.Recipe, policy: EnvironmentPolicy,
                     readback_expectations: tuple[tuple[str, str], ...]
                     ) -> tuple[WitnessReport, ...]:
    declarations: dict[str, tuple[str, str]] = {}
    for check, (field, expected) in zip(
            template.env_readback, readback_expectations, strict=True):
        key = check.get("env") if isinstance(check, Mapping) else None
        if key:
            declarations[key] = (field, expected)
    reports = []
    for key, kind in policy.witnesses:
        if kind == "recipe_readback" and key in declarations:
            field, expected = declarations[key]
            reports.append(WitnessReport(key, kind, "declared", field, expected))
        elif kind == "recipe_readback":
            reports.append(WitnessReport(
                key, kind, "unknown", reason="recipe_readback_not_declared"))
        else:
            reports.append(WitnessReport(
                key, kind, "unknown", reason="witness_sampler_not_implemented"))
    return tuple(reports)


def _capability(template: serving.Recipe, backend: str, drafter: ArtifactDigest | None,
                witnesses: tuple[WitnessReport, ...]) -> CapabilityReport:
    reasons: list[CapabilityReason] = []
    spec = dict(template.spec_decode)
    spec_type = spec.get("type", "none")
    external = spec_type != "none" and bool(spec.get("drafter"))
    speculation = "none" if spec_type == "none" else (
        "external_draft" if external else "self_draft")
    flags = tuple(template.extra_flags)
    draft_device = _flag_value(flags, "--device-draft")
    consumed = set()
    for index, flag in enumerate(flags):
        if flag == "--device-draft":
            consumed.update({index, index + 1})
        elif flag.startswith("--device-draft="):
            consumed.add(index)
    if any(index not in consumed for index in range(len(flags))):
        reasons.append(CapabilityReason(
            "extra_flags_unsupported",
            "resolved execution does not normalize arbitrary extension flags"))
    if backend not in SUPPORTED_BACKENDS:
        reasons.append(CapabilityReason("unsupported_backend",
                                        "only one explicit CPU or GPU backend is implemented"))
    if ("," in template.device or draft_device == "<multiple>"
            or (draft_device is not None and "," in draft_device)):
        reasons.append(CapabilityReason("multi_device_unsupported",
                                        "multi-device launch is not implemented"))
    if (external and draft_device not in {None, "none"}
            and not _GPU_DEVICE.fullmatch(draft_device)):
        reasons.append(CapabilityReason(
            "unsupported_draft_device", "draft device spelling is not supported"))
    if any("rpc" in flag.lower() for flag in flags) or "rpc" in template.device.lower():
        reasons.append(CapabilityReason("rpc_unsupported", "RPC launch is not implemented"))
    if backend == "cpu":
        if template.ngl != 0:
            reasons.append(CapabilityReason("cpu_requires_ngl_zero",
                                            "CPU main-model launch requires ngl=0"))
        if template.device != "none":
            reasons.append(CapabilityReason("unsupported_cpu_device",
                                            "CPU main-model device must be exactly 'none'"))
        if external and (spec.get("ngld", template.ngl) != 0 or draft_device != "none"):
            reasons.append(CapabilityReason(
                "mixed_cpu_gpu_draft_unsupported",
                "external CPU draft must explicitly use --device-draft none and ngld=0"))
    elif backend == "gpu":
        if template.ngl <= 0:
            reasons.append(CapabilityReason("gpu_offload_required",
                                            "GPU launch requires positive ngl"))
        if not _GPU_DEVICE.fullmatch(template.device):
            reasons.append(CapabilityReason("unsupported_gpu_device",
                                            "GPU device must name one explicit ROCm device"))
        if external and (spec.get("ngld", template.ngl) <= 0 or draft_device == "none"):
            reasons.append(CapabilityReason(
                "mixed_cpu_gpu_draft_unsupported",
                "CPU-offloaded draft beside a GPU main model is not implemented"))
        if external and draft_device is not None and draft_device != template.device:
            reasons.append(CapabilityReason(
                "multi_device_unsupported",
                "main and external draft must use the same explicit GPU device"))
    if external and drafter is None:
        reasons.append(CapabilityReason("drafter_identity_missing",
                                        "external speculation requires a pinned drafter"))
    if not external and drafter is not None:
        reasons.append(CapabilityReason("unexpected_drafter_identity",
                                        "none/self draft has no separate drafter artifact"))
    if any(witness.kind == "recipe_readback" and witness.status == "unknown"
           for witness in witnesses):
        reasons.append(CapabilityReason(
            "recipe_readback_missing", "required process readback is not declared"))
    return CapabilityReport(
        supported=not reasons, backend=backend, speculation=speculation,
        reasons=tuple(reasons), witnesses=witnesses,
        gpu_residency="not_applicable" if backend == "cpu" else "required",
        cpu_placement="unproven", contention="unproven")


def _canonical_int(parsed: Mapping[str, str | bool], name: str,
                   default: int | None = None) -> int:
    raw = parsed.get(name)
    if raw is None and default is not None:
        return default
    if not isinstance(raw, str):
        raise ResolutionError(f"canonical command {name} must be an integer")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ResolutionError(f"canonical command {name} must be an integer") from exc
    if value < 0:
        raise ResolutionError(f"canonical command {name} must be non-negative")
    return value


def _validate_canonical_consistency(resolved: CanonicalResolvedRecipe,
                                    template: serving.Recipe) -> None:
    _validate_template(template)
    if template.recipe_hash != resolved.template_hash or template != resolved.template:
        raise ResolutionError("canonical launch template identity mismatch")
    if resolved.argv != resolved.topology_prefix + resolved.command_argv:
        raise ResolutionError("canonical actual argv is not topology prefix plus command")
    projected = canonical_recipe_projection(
        name=template.name, command_argv=resolved.command_argv,
        topology_prefix=resolved.topology_prefix, n_predict=template.n_predict,
        temperature=template.temperature, top_p=template.top_p, top_k=template.top_k,
        metric=template.metric)
    if projected.to_dict() != template.to_dict():
        raise ResolutionError("canonical command semantic projection differs from template")
    cpu_list = _canonical_prefix(resolved.topology_prefix)
    if cpu_list != template.cpu_list:
        raise ResolutionError("canonical topology CPU list differs from template")
    executable, parsed = _canonical_command(resolved.command_argv)
    if executable != resolved.executable.path \
            or executable != str(Path(resolved.build_dir) / "bin" / "llama-server"):
        raise ResolutionError("canonical executable/build identities disagree")
    if (resolved.runtime_binary_dir is not None
            and resolved.runtime_binary_dir != str(Path(executable).parent)):
        raise ResolutionError("canonical runtime binary directory differs from executable")
    if parsed["-m"] != template.model or parsed["-m"] != resolved.model.path:
        raise ResolutionError("canonical model identity differs from template")
    if _canonical_int(parsed, "--port") != resolved.port:
        raise ResolutionError("canonical port differs from snapshot")
    for flag, expected in (("-np", template.np), ("-c", template.ctx),
                           ("-t", template.threads), ("-ub", template.ubatch)):
        if _canonical_int(parsed, flag) != expected:
            raise ResolutionError(f"canonical {flag} differs from template")
    if "-b" in parsed and _canonical_int(parsed, "-b") != template.batch:
        raise ResolutionError("canonical batch differs from template")
    if str(parsed.get("-ctk", "f16")) != template.ctk \
            or str(parsed.get("-ctv", "f16")) != template.ctv:
        raise ResolutionError("canonical KV types differ from template")
    flash = str(parsed.get("--flash-attn", parsed.get("-fa", "off")))
    if flash != template.fa:
        raise ResolutionError("canonical flash-attention state differs from template")
    device = parsed.get("--device", "none")
    if device != template.device:
        raise ResolutionError("canonical device differs from template")
    ngl_raw = parsed.get("-ngl")
    if resolved.backend == "cpu":
        if (0 if ngl_raw is None else _canonical_int(parsed, "-ngl")) != 0 \
                or template.ngl != 0:
            raise ResolutionError("canonical CPU offload state differs from template")
    elif ngl_raw == "all":
        if template.ngl <= 0:
            raise ResolutionError("canonical GPU offload differs from template")
    elif _canonical_int(parsed, "-ngl") != template.ngl:
        raise ResolutionError("canonical GPU offload differs from template")
    spec = template.spec_decode
    if str(parsed.get("--spec-type", "none")) != spec.get("type", "none"):
        raise ResolutionError("canonical speculation differs from template")
    if parsed.get("-md") != spec.get("drafter"):
        raise ResolutionError("canonical drafter differs from template")
    if spec.get("draft_n_max") is not None \
            and _canonical_int(parsed, "--spec-draft-n-max") != spec["draft_n_max"]:
        raise ResolutionError("canonical draft limit differs from template")
    if (resolved.drafter is None) != (parsed.get("-md") is None):
        raise ResolutionError("canonical drafter artifact presence differs from command")
    if resolved.drafter is not None and resolved.drafter.path != parsed.get("-md"):
        raise ResolutionError("canonical drafter artifact differs from command")
    if len({Path(item.path).name for item in resolved.dsos}) != len(resolved.dsos):
        raise ResolutionError("canonical DSO loader names must be unique")
    launch = dict(resolved.launch_env)
    ld_dirs = {part for part in launch.get("LD_LIBRARY_PATH", "").split(":") if part}
    dso_dirs = {str(Path(item.path).parent) for item in resolved.dsos}
    permitted_dso_dirs = ld_dirs | {str(Path(resolved.executable.path).parent)}
    if not resolved.dsos or any(path not in permitted_dso_dirs for path in dso_dirs):
        raise ResolutionError("canonical DSO identities are outside the loader path")
    if any(path not in ld_dirs for path in resolved.runtime_ld_paths):
        raise ResolutionError("canonical runtime loader path differs from launch environment")
    if any(path not in dso_dirs for path in resolved.runtime_ld_paths):
        raise ResolutionError("canonical runtime loader paths lack sealed DSO identities")
    policy = EnvironmentPolicy.from_dict(resolved.environment_policy.to_dict())
    relevant = dict(resolved.relevant_environment)
    if set(relevant) != set(policy.measurement_keys):
        raise ResolutionError("canonical relevant environment differs from policy")
    allowed = set(policy.measurement_keys) | set(policy.allowed_inherit_keys) | {"LD_LIBRARY_PATH"}
    if set(launch) - allowed:
        raise ResolutionError("canonical launch environment exceeds policy")
    if any(marker in key.upper() for key in launch for marker in _CREDENTIAL_MARKERS):
        raise ResolutionError("canonical launch environment contains credential-like keys")
    if set(resolved.absent_environment) != {k for k, v in relevant.items() if v is None}:
        raise ResolutionError("canonical absent environment differs from relevant state")
    if any(value is not None and launch.get(key) != value for key, value in relevant.items()) \
            or any(value is None and key in launch for key, value in relevant.items()):
        raise ResolutionError("canonical relevant environment differs from launch")
    if any(relevant.get(key) != value for key, value in (template.env or {}).items()) \
            or any(relevant.get(key, "<missing>") is not None
                   for key in template.explicit_unsets):
        raise ResolutionError("canonical template environment differs from frozen launch")
    if template.readback_expectations() != resolved.readback_expectations:
        raise ResolutionError("canonical readback expectations differ from template")
    provenance = dict(resolved.provenance)
    _sha(provenance.get("export_sha256"), "provenance.export_sha256")
    if (provenance.get("instance_mode") not in {"full", "quarter", "both"}
            or not any(key.startswith("source:") for key in provenance)):
        raise ResolutionError("canonical provenance is incomplete")
    expected_capability = _capability(
        template, resolved.backend, resolved.drafter,
        _witness_reports(template, policy, resolved.readback_expectations))
    if resolved.capability != expected_capability:
        raise ResolutionError("canonical capability is not derived from its template")
    if WorkloadSpec(template.n_predict, template.temperature, template.top_p,
                    template.top_k, template.metric) != resolved.workload:
        raise ResolutionError("canonical workload differs from template")


def resolve_canonical_launch(template: serving.Recipe, *, build_dir: Path | str,
                             command_argv: Sequence[str], topology_prefix: Sequence[str],
                             launch_environment: Mapping[str, str],
                             artifact_identities: Mapping[str, Any], backend: str,
                             environment_policy: EnvironmentPolicy | Mapping[str, Any],
                             port: int, runtime_binary_dir: str | None,
                             runtime_ld_paths: Sequence[str],
                             provenance: Mapping[str, str]) -> CanonicalResolvedRecipe:
    """Freeze one exact production command without reformatting its argv."""
    _validate_template(template)
    command = _argv_sequence(command_argv, "canonical command_argv")
    if isinstance(topology_prefix, (str, bytes)) or not isinstance(topology_prefix, Sequence):
        raise ResolutionError("topology_prefix must be an array")
    prefix = tuple(_text(x, "topology_prefix[]") for x in topology_prefix)
    _canonical_prefix(prefix)
    root = Path(build_dir)
    if not root.is_absolute():
        raise ResolutionError("build_dir must be absolute")
    resolved_port = _port(port)
    policy = EnvironmentPolicy.from_dict(
        environment_policy.to_dict() if isinstance(environment_policy, EnvironmentPolicy)
        else environment_policy)
    launch = _object(launch_environment, "canonical launch environment")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in launch.items()):
        raise ResolutionError("canonical launch environment must map strings to strings")
    model, drafter, executable, dsos = _artifact_set(artifact_identities)
    relevant = tuple(sorted((key, launch.get(key)) for key in policy.measurement_keys))
    readbacks = template.readback_expectations()
    witnesses = _witness_reports(template, policy, readbacks)
    capability = _capability(template, backend, drafter, witnesses)
    workload = WorkloadSpec(template.n_predict, template.temperature, template.top_p,
                            template.top_k, template.metric)
    if runtime_binary_dir is not None:
        runtime_binary_dir = _text(runtime_binary_dir, "runtime_binary_dir")
    runtime_ld = tuple(_text(x, "runtime_ld_paths[]") for x in runtime_ld_paths)
    provenance_pairs = tuple(sorted((_text(k, "provenance key"),
                                     _text(v, f"provenance.{k}"))
                                    for k, v in _object(provenance, "provenance").items()))
    provisional = CanonicalResolvedRecipe(
        template.recipe_hash, template, backend, str(root), resolved_port, command, prefix,
        prefix + command, tuple(sorted(launch.items())),
        tuple(sorted(key for key, value in relevant if value is None)), relevant, readbacks,
        workload, policy, model, drafter, executable, dsos, capability,
        runtime_binary_dir, runtime_ld, provenance_pairs, "0" * 64, "0" * 64)
    resolved = replace(provisional, snapshot_digest=_digest(provisional._snapshot_dict()),
                       execution_digest=_digest(provisional._normalized_execution_dict()))
    _validate_canonical_consistency(resolved, template)
    return resolved


def resolve_recipe(template: serving.Recipe, *, build_dir: Path | str,
                   artifact_identities: Mapping[str, Any], backend: str,
                   environment_policy: EnvironmentPolicy | Mapping[str, Any],
                   inherited_environment: Mapping[str, str], port: int) -> ResolvedRecipe:
    """Resolve one launch without touching artifacts or granting execution authority."""
    if not isinstance(template, serving.Recipe):
        raise ResolutionError("template must be a serving.Recipe")
    _validate_template(template)
    resolved_port = _port(port)
    root = Path(build_dir)
    if not root.is_absolute():
        raise ResolutionError("build_dir must be absolute")
    root_text = str(root)
    policy = EnvironmentPolicy.from_dict(
        environment_policy.to_dict() if isinstance(environment_policy, EnvironmentPolicy)
        else environment_policy)
    inherited = _object(inherited_environment, "inherited environment")
    for key, value in inherited.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ResolutionError("inherited environment keys and values must be strings")
    declared = set(template.env or {}) | set(template.explicit_unsets)
    uncovered = declared - set(policy.measurement_keys)
    if uncovered:
        raise ResolutionError(
            f"recipe environment keys are not covered by policy: {sorted(uncovered)}")
    selected_parent = {key: inherited[key] for key in policy.allowed_inherit_keys
                       if key in inherited}
    launch = template.server_env(root, base=selected_parent)
    relevant = []
    for key in policy.measurement_keys:
        if key in template.explicit_unsets:
            relevant.append((key, None))
        elif key in (template.env or {}):
            relevant.append((key, (template.env or {})[key]))
        elif key in policy.allowed_inherit_keys:
            relevant.append((key, inherited.get(key)))
        elif key in inherited:
            raise ResolutionError(
                f"measurement key {key!r} would inherit without policy permission")
        else:
            relevant.append((key, None))
    model, drafter, executable, dsos = _artifact_set(artifact_identities)
    argv = tuple(template.server_argv(root, resolved_port))
    expected_executable = str(root / "bin" / "llama-server")
    if model.path != template.model:
        raise ResolutionError("model artifact path does not match the recipe")
    draft_path = template.spec_decode.get("drafter")
    if draft_path is not None and drafter is not None and drafter.path != draft_path:
        raise ResolutionError("drafter artifact path does not match the recipe")
    if draft_path is None and drafter is not None:
        raise ResolutionError("recipe has no external drafter but an identity was supplied")
    if executable.path != expected_executable:
        raise ResolutionError("executable artifact path does not match the build recipe")
    effective_env = dict(template.env or {})
    effective_unsets = set(template.explicit_unsets)
    for key, value in relevant:
        if value is None:
            effective_env.pop(key, None)
            effective_unsets.add(key)
        else:
            effective_env[key] = value
            effective_unsets.discard(key)
    effective_template = replace(
        template, env=effective_env, explicit_unsets=tuple(sorted(effective_unsets)))
    readback_expectations = effective_template.readback_expectations()
    witnesses = _witness_reports(template, policy, readback_expectations)
    capability = _capability(template, backend, drafter, witnesses)
    workload = WorkloadSpec(template.n_predict, template.temperature, template.top_p,
                            template.top_k, template.metric)
    provisional = ResolvedRecipe(
        template_hash=template.recipe_hash, backend=backend, build_dir=root_text,
        port=resolved_port, argv=argv, launch_env=tuple(sorted(launch.items())),
        absent_environment=tuple(sorted(key for key, value in relevant if value is None)),
        relevant_environment=tuple(sorted(relevant)),
        readback_expectations=readback_expectations, workload=workload,
        environment_policy=policy,
        model=model, drafter=drafter, executable=executable, dsos=dsos,
        capability=capability, snapshot_digest="0" * 64, execution_digest="0" * 64)
    resolved = ResolvedRecipe(**{**provisional.__dict__,
                                 "snapshot_digest": _digest(provisional._snapshot_dict()),
                                 "execution_digest": _digest(
                                     provisional._normalized_execution_dict())})
    _validate_resolved_consistency(resolved)
    return resolved


def resolved_recipe_from_dict(value: Any) -> ResolvedRecipe | CanonicalResolvedRecipe:
    """Explicit schema dispatcher; unknown records never fall back to a looser parser."""
    row = _object(value, "resolved recipe")
    schema = row.get("schema")
    if schema == RESOLVED_RECIPE_SCHEMA:
        return ResolvedRecipe.from_dict(row)
    if schema == CANONICAL_RESOLVED_RECIPE_SCHEMA:
        return CanonicalResolvedRecipe.from_dict(row)
    raise ResolutionError(f"resolved recipe: unsupported schema {schema!r}")


__all__ = ["ARTIFACT_SCHEMA", "CANONICAL_RESOLVED_RECIPE_SCHEMA", "CAPABILITY_SCHEMA",
           "ENVIRONMENT_POLICY_SCHEMA", "RESOLVED_RECIPE_SCHEMA", "ArtifactDigest",
           "CanonicalResolvedRecipe", "CapabilityReason", "CapabilityReport",
           "EnvironmentPolicy", "ResolutionError", "ResolvedRecipe",
           "UnsupportedRecipeCapability", "WitnessReport", "WorkloadSpec",
           "canonical_recipe_projection", "resolve_canonical_launch", "resolve_recipe",
           "resolved_recipe_from_dict"]
