"""Pure, pinned export of production launcher obligations for AutoKernel.

This module describes what the current launcher *would* execute.  It never starts a
process, creates runtime directories, evicts memory, or grants admission.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence
import types

import yaml

EXPORT_SCHEMA = "autokernel-production-enrollment/v1"
CONTEXT_SCHEMA = "autokernel-production-export-context/v1"
#: Backend scopes an export may declare.  "all" is the default and the
#: historical shape; a scoped export omits every out-of-scope target.
BACKEND_SCOPES = ("cpu", "gpu", "all")
RECIPE_ARTIFACT_SCHEMA = "autokernel-production-launch-recipe/v1"
_SAFE_ENV_EXACT = frozenset({"LD_LIBRARY_PATH", "PATH", "HSA_OVERRIDE_GFX_VERSION"})
_SAFE_ENV_PREFIXES = ("GGML_", "OMP_", "KMP_")
_CREDENTIAL_MARKERS = ("PASSWORD", "PASSWD", "TOKEN", "SECRET", "CREDENTIAL",
                       "AUTH", "BEARER", "API_KEY", "PRIVATE_KEY")


class EnrollmentExportError(ValueError):
    """The caller's pinned context cannot describe the current launcher exactly."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise EnrollmentExportError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise EnrollmentExportError(f"{label} must be a SHA-256 digest")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(_plain(value), sort_keys=True, separators=(",", ":")) + "\n").encode()


def _plain(value: Any, label: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EnrollmentExportError(f"{label} must be finite")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or key in out:
                raise EnrollmentExportError(f"{label} keys must be unique non-empty strings")
            out[key] = _plain(item, f"{label}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item, f"{label}[]") for item in value]
    raise EnrollmentExportError(f"{label} is not canonical JSON")


def _configuration(value: Any, label: str = "configuration") -> Any:
    """JSON projection for trusted launcher data whose YAML maps may use int keys."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, (str, int)) or isinstance(raw_key, bool):
                raise EnrollmentExportError(f"{label} contains an unsupported mapping key")
            key = str(raw_key)
            if not key or key in out:
                raise EnrollmentExportError(f"{label} contains colliding mapping keys")
            out[key] = _configuration(item, f"{label}.{key}")
        return out
    if isinstance(value, tuple):
        return [_configuration(item, f"{label}[]") for item in value]
    if isinstance(value, list):
        return [_configuration(item, f"{label}[]") for item in value]
    return _plain(value, label)


@dataclass(frozen=True)
class SourcePin:
    name: str
    path: str
    sha256: str
    revision: str
    revision_kind: str = "caller_declared"

    @classmethod
    def from_dict(cls, value: Any) -> "SourcePin":
        if not isinstance(value, Mapping) or set(value) != {
            "name", "path", "sha256", "revision", "revision_kind"}:
            raise EnrollmentExportError("source pin has unknown or missing fields")
        path = _text(value["path"], "source.path")
        if not Path(path).is_absolute():
            raise EnrollmentExportError("source.path must be absolute")
        kind = _text(value["revision_kind"], "source.revision_kind")
        if kind != "caller_declared":
            raise EnrollmentExportError("source.revision_kind is unsupported")
        return cls(_text(value["name"], "source.name"), path,
                   _sha(value["sha256"], "source.sha256"),
                   _text(value["revision"], "source.revision"), kind)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path, "sha256": self.sha256,
                "revision": self.revision, "revision_kind": self.revision_kind}


@dataclass(frozen=True)
class ArtifactPin:
    use: str
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactPin":
        if not isinstance(value, Mapping) or set(value) != {"use", "path", "sha256"}:
            raise EnrollmentExportError("artifact pin has unknown or missing fields")
        path = _text(value["path"], "artifact.path")
        if not Path(path).is_absolute():
            raise EnrollmentExportError("artifact.path must be absolute")
        use = _text(value["use"], "artifact.use")
        if use not in {"model", "drafter", "executable", "dso"}:
            raise EnrollmentExportError(f"unsupported artifact use {use!r}")
        return cls(use, path, _sha(value["sha256"], "artifact.sha256"))

    def to_dict(self) -> dict[str, str]:
        return {"use": self.use, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class ExportContext:
    export_id: str
    created_at: str
    instance_mode: str
    sources: tuple[SourcePin, ...]
    artifacts: tuple[ArtifactPin, ...]
    base_environment: tuple[tuple[str, str], ...]
    requested_roles: tuple[str, ...]
    loaded_builder_sha256: str
    seed_roles: tuple[str, ...] = ()
    #: Which backends this export is allowed to describe.  ``all`` is the historical
    #: behaviour (every selected target); ``cpu``/``gpu`` omit every out-of-scope
    #: target from the export entirely, so a reader cannot reconstitute one.  The
    #: scope travels inside the context, hence under ``export_sha256``: a short
    #: roster must always carry the reason it is short.
    backend_scope: str = "all"

    @classmethod
    def from_dict(cls, value: Any) -> "ExportContext":
        # ``backend_scope`` is additive and omitted when it is the default, so an
        # unscoped export stays byte-identical to every export produced before it.
        if not isinstance(value, Mapping) or set(value) - {"backend_scope"} != {
            "schema", "export_id", "created_at", "instance_mode", "sources",
            "artifacts", "base_environment", "requested_roles", "seed_roles",
            "loaded_builder_sha256",
        } or value["schema"] != CONTEXT_SCHEMA:
            raise EnrollmentExportError("malformed or unsupported export context")
        backend_scope = _text(value.get("backend_scope", "all"), "backend_scope")
        if backend_scope not in BACKEND_SCOPES:
            raise EnrollmentExportError("backend_scope must be cpu, gpu, or all")
        mode = _text(value["instance_mode"], "instance_mode")
        if mode not in {"full", "quarter", "both"}:
            raise EnrollmentExportError("instance_mode must be full, quarter, or both")
        try:
            datetime.fromisoformat(_text(value["created_at"], "created_at").replace("Z", "+00:00"))
        except ValueError as exc:
            raise EnrollmentExportError("created_at must be an ISO-8601 timestamp") from exc
        env = value["base_environment"]
        if not isinstance(env, Mapping) or not all(isinstance(k, str) and k and isinstance(v, str)
                                                   for k, v in env.items()):
            raise EnrollmentExportError("base_environment must map strings to strings")
        unsafe = sorted(key for key in env if key not in _SAFE_ENV_EXACT
                        and not key.startswith(_SAFE_ENV_PREFIXES))
        credentials = sorted(key for key in env
                             if any(marker in key.upper() for marker in _CREDENTIAL_MARKERS))
        if unsafe:
            raise EnrollmentExportError(
                f"base_environment contains unsupported keys: {unsafe}; arbitrary parent "
                "environment is never enrollment provenance")
        if credentials:
            raise EnrollmentExportError(
                f"base_environment contains credential-like keys: {credentials}")
        roles = value["requested_roles"]
        seeds = value["seed_roles"]
        if (not isinstance(roles, list) or not isinstance(seeds, list)
                or not isinstance(value["sources"], list)
                or not isinstance(value["artifacts"], list)):
            raise EnrollmentExportError("requested_roles and seed_roles must be arrays")
        requested = tuple(_text(item, "requested_roles[]") for item in roles)
        seed_roles = tuple(_text(item, "seed_roles[]") for item in seeds)
        if len(set(requested)) != len(requested) or len(set(seed_roles)) != len(seed_roles):
            raise EnrollmentExportError("requested_roles and seed_roles may not contain duplicates")
        sources = tuple(SourcePin.from_dict(item) for item in value["sources"])
        artifacts = tuple(ArtifactPin.from_dict(item) for item in value["artifacts"])
        if len({item.name for item in sources}) != len(sources):
            raise EnrollmentExportError("source names must be unique")
        if len({(item.use, item.path) for item in artifacts}) != len(artifacts):
            raise EnrollmentExportError("artifact use/path pairs must be unique")
        return cls(_text(value["export_id"], "export_id"), value["created_at"], mode,
                   sources, artifacts, tuple(sorted(env.items())), requested,
                   _sha(value["loaded_builder_sha256"], "loaded_builder_sha256"), seed_roles,
                   backend_scope)

    def to_dict(self) -> dict[str, Any]:
        row = {"schema": CONTEXT_SCHEMA, "export_id": self.export_id,
               "created_at": self.created_at, "instance_mode": self.instance_mode,
               "sources": [item.to_dict() for item in self.sources],
               "artifacts": [item.to_dict() for item in self.artifacts],
               "base_environment": dict(self.base_environment),
               "requested_roles": list(self.requested_roles),
               "loaded_builder_sha256": self.loaded_builder_sha256,
               "seed_roles": list(self.seed_roles)}
        if self.backend_scope != "all":
            row["backend_scope"] = self.backend_scope
        return row


def _loaded_builder_identity() -> str:
    """Hash the callable code actually loaded, separately from disk provenance."""
    from scripts.server import orchestrator_stack as launcher
    from scripts.server import stack_env, stack_numa
    functions = (launcher.build_server_command, launcher._build_role_command,
                 launcher._build_worker_general_command, launcher._build_worker_fast_command,
                 launcher._build_vision_command, launcher._build_embedding_command,
                 launcher._build_eval_batch_frontdoor_command,
                 launcher._build_gpu_shadow_lane_command, launcher._dispatch_prior_role,
                 stack_env.build_launch_env, stack_numa._numa_prefix)
    digest = hashlib.sha256()
    def projection(code: types.CodeType) -> dict[str, Any]:
        def constant(value: Any) -> Any:
            if isinstance(value, types.CodeType):
                return {"code": projection(value)}
            if isinstance(value, bytes):
                return {"bytes": value.hex()}
            if isinstance(value, tuple):
                return {"tuple": [constant(item) for item in value]}
            if isinstance(value, frozenset):
                rows = [constant(item) for item in value]
                return {"frozenset": sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))}
            if value is None or isinstance(value, (bool, int, str)):
                return value
            if isinstance(value, float):
                return {"float": repr(value)}
            return {"type": f"{type(value).__module__}.{type(value).__qualname__}",
                    "repr": repr(value)}
        return {"argcount": code.co_argcount, "posonlyargcount": code.co_posonlyargcount,
                "kwonlyargcount": code.co_kwonlyargcount, "nlocals": code.co_nlocals,
                "stacksize": code.co_stacksize, "flags": code.co_flags,
                "code": code.co_code.hex(), "exceptiontable": code.co_exceptiontable.hex(),
                "consts": [constant(item) for item in code.co_consts],
                "names": list(code.co_names), "varnames": list(code.co_varnames),
                "freevars": list(code.co_freevars), "cellvars": list(code.co_cellvars)}
    for function in functions:
        digest.update(f"{function.__module__}.{function.__qualname__}\0".encode())
        digest.update(json.dumps(projection(function.__code__), sort_keys=True,
                                 separators=(",", ":")).encode())
    return digest.hexdigest()


def capture_current_context(*, master_registry: Path | str, revision: str,
                            instance_mode: str, artifacts: Sequence[ArtifactPin] = (),
                            requested_roles: Sequence[str] = (),
                            seed_roles: Sequence[str] = (),
                            base_environment: Mapping[str, str] | None = None,
                            export_id: str = "production-current",
                            backend_scope: str = "all") -> ExportContext:
    """Capture source bytes from the exact modules loaded by this process.

    Artifact digests remain caller-supplied declarations; this helper never hashes models,
    executables, or DSOs.  ``export_production_enrollment`` immediately rechecks every
    captured source before using it.
    """
    from datetime import timezone
    from scripts.server import orchestrator_stack as launcher
    from scripts.server import stack_env, stack_manifest, stack_numa, stack_paths, stack_runtime
    from src.registry import stack_priors
    paths = {
        "master_registry": Path(master_registry),
        "lean_registry": Path(stack_manifest._LEAN_REGISTRY_PATH),
        "launcher": Path(launcher.__file__),
        "launch_manifest": Path(stack_manifest._LAUNCH_MANIFEST_PATH),
        "topology": Path(stack_numa._TOPOLOGY_PATH),
        "stack_env": Path(stack_env.__file__), "stack_runtime": Path(stack_runtime.__file__),
        "stack_priors": Path(launcher.STACK_PRIORS_PATH),
        "stack_manifest": Path(stack_manifest.__file__),
        "stack_numa": Path(stack_numa.__file__), "stack_paths": Path(stack_paths.__file__),
        "descriptors": Path(stack_priors.DEFAULT_DESCRIPTORS),
    }
    pins = tuple(SourcePin(name, str(path.resolve()), _digest(path.read_bytes()), revision)
                 for name, path in paths.items())
    raw = ExportContext(export_id, datetime.now(timezone.utc).isoformat(), instance_mode,
                        pins, tuple(artifacts), tuple(sorted((base_environment or {}).items())),
                        tuple(requested_roles), _loaded_builder_identity(), tuple(seed_roles),
                        backend_scope)
    return ExportContext.from_dict(raw.to_dict())


def _flag(argv: Sequence[str], name: str) -> str | None:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == name:
            values.append(argv[index + 1] if index + 1 < len(argv) else "")
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    return values[0] if len(values) == 1 and values[0] else None


def _mode_kwargs(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "embedding_mode": bool(entry.get("embedding")),
        "worker_pool_mode": bool(entry.get("worker_pool")),
        "worker_type": entry.get("worker_type"),
        "vision_mode": bool(entry.get("vision")),
        "vision_type": entry.get("vision_type"),
        "eval_batch_frontdoor_mode": bool(entry.get("eval_batch_frontdoor")),
        "gpu_shadow_lane_mode": bool(entry.get("gpu_shadow_lane")),
    }


def _source_role(entry: Mapping[str, Any]) -> str:
    role = str(entry["roles"][0])
    if entry.get("eval_batch_frontdoor"):
        return "frontdoor"
    if entry.get("worker_pool"):
        return "worker"
    return role


def _apply_declared_runtime_env(env: dict[str, str], *, binary_override: str | None,
                                ld_paths: list[str] | None) -> None:
    """Side-effect-free equivalent of the launcher's environment-only transform."""
    from scripts.server.stack_env import compose_ld_library_path
    if binary_override:
        for key in tuple(env):
            if key.startswith("GGML_") and key != "GGML_IQK":
                del env[key]
        env["KMP_BLOCKTIME"] = "10"
    if ld_paths:
        env["LD_LIBRARY_PATH"] = compose_ld_library_path(
            ld_paths, env.get("LD_LIBRARY_PATH", ""), "prepend")


def _artifact_rows(argv: Sequence[str], env: Mapping[str, str],
                   pins: tuple[ArtifactPin, ...]) -> tuple[list[dict[str, str]], list[str]]:
    by_key = {(pin.use, pin.path): pin for pin in pins}
    paths = [("executable", argv[0]), ("model", _flag(argv, "-m"))]
    draft = _flag(argv, "-md")
    if draft:
        paths.append(("drafter", draft))
    # llama.cpp loads every sibling of a split GGUF, not only the argv shard.
    # Expand the declared filenames without scanning directories or hashing here.
    expanded = []
    for use, path in paths:
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", str(path)) if path else None
        if use in {"model", "drafter"} and match:
            index, total = int(match[2]), int(match[3])
            if not 1 <= index <= total:
                raise EnrollmentExportError(f"invalid split GGUF filename: {path}")
            expanded.extend((use, f"{match[1]}-{part:05d}-of-{total:05d}.gguf")
                            for part in range(1, total + 1))
        else:
            expanded.append((use, path))
    paths = expanded
    ld_dirs = {part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part}
    if argv:
        ld_dirs.add(str(Path(argv[0]).parent))  # binary RUNPATH/adjacent ggml DSOs
    for pin in pins:
        if pin.use == "dso" and str(Path(pin.path).parent) in ld_dirs:
            paths.append(("dso", pin.path))
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for use, path in paths:
        if not path:
            missing.append(use)
            continue
        pin = by_key.get((use, str(path)))
        if pin is None:
            missing.append(f"{use}:{path}")
        else:
            rows.append(pin.to_dict())
    if not any(row["use"] == "dso" for row in rows):
        missing.append("dso_set")
    return rows, missing


def _verify_artifact_pins(pins: tuple[ArtifactPin, ...]) -> dict[tuple[str, str], str]:
    """Verify exact bytes once per path within one explicitly opted-in export."""
    cache: dict[str, tuple[str | None, str | None]] = {}
    failures: dict[tuple[str, str], str] = {}
    for pin in pins:
        if pin.path not in cache:
            path = Path(pin.path)
            try:
                info = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(info.st_mode):
                    cache[pin.path] = (None, "not_regular_file")
                else:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    cache[pin.path] = (digest.hexdigest(), None)
            except OSError as exc:
                cache[pin.path] = (None, f"unreadable:{type(exc).__name__}")
        actual, error = cache[pin.path]
        if error is not None:
            failures[(pin.use, pin.path)] = error
        elif actual != pin.sha256:
            failures[(pin.use, pin.path)] = "sha256_mismatch"
    return failures


def export_production_enrollment(context: ExportContext | Mapping[str, Any], *,
                                 verify_artifacts: bool = False) -> dict[str, Any]:
    """Resolve the exact current registry/launcher context without side effects."""
    if not isinstance(context, ExportContext):
        context = ExportContext.from_dict(context)
    # Re-roundtrip direct dataclass instances so they get the same checks as JSON.
    context = ExportContext.from_dict(context.to_dict())
    if context.loaded_builder_sha256 != _loaded_builder_identity():
        raise EnrollmentExportError("loaded canonical builder identity changed")
    if not isinstance(verify_artifacts, bool):
        raise EnrollmentExportError("verify_artifacts must be bool")

    from scripts.server import orchestrator_stack as launcher
    from scripts.server import stack_env, stack_manifest, stack_numa, stack_paths, stack_runtime
    from src.registry import stack_priors
    from scripts.server.stack_manifest import (AUX_SERVICES, HOT_SERVERS, PORT_MAP,
                                                ROLE_LAUNCH_META, WARM_SERVERS,
                                                _filter_by_numa_mode)
    from scripts.server.stack_numa import NUMA_CONFIG, _numa_prefix
    from scripts.server.stack_numa_evict import pre_evict_gib_for_role
    from src.registry.registry_compiler import active_roles_from_launch_meta, compile_lean
    from src.registry_loader import RegistryLoader

    source_by_name = {item.name: item for item in context.sources}
    required_sources = {"master_registry", "lean_registry", "launcher", "launch_manifest",
                        "topology", "stack_env", "stack_runtime", "stack_priors",
                        "stack_manifest", "stack_numa", "stack_paths", "descriptors"}
    missing_sources = required_sources - set(source_by_name)
    if missing_sources:
        raise EnrollmentExportError(f"missing pinned sources: {sorted(missing_sources)}")
    for source in context.sources:
        try:
            actual = _digest(Path(source.path).read_bytes())
        except OSError as exc:
            raise EnrollmentExportError(f"cannot read pinned source {source.name}: {exc}") from exc
        if actual != source.sha256:
            raise EnrollmentExportError(f"pinned source drift: {source.name}")
    loaded_paths = {
        "lean_registry": Path(stack_manifest._LEAN_REGISTRY_PATH),
        "launcher": Path(launcher.__file__),
        "launch_manifest": Path(stack_manifest._LAUNCH_MANIFEST_PATH),
        "topology": Path(stack_numa._TOPOLOGY_PATH),
        "stack_env": Path(stack_env.__file__),
        "stack_runtime": Path(stack_runtime.__file__),
        "stack_priors": Path(launcher.STACK_PRIORS_PATH),
        "stack_manifest": Path(stack_manifest.__file__),
        "stack_numa": Path(stack_numa.__file__),
        "stack_paths": Path(stack_paths.__file__),
        "descriptors": Path(stack_priors.DEFAULT_DESCRIPTORS),
    }
    for name, loaded_path in loaded_paths.items():
        if Path(source_by_name[name].path).resolve() != loaded_path.resolve():
            raise EnrollmentExportError(f"pinned source path is not the loaded {name}")
    # Source bytes describe the files now on disk; separately prove that the tables
    # already loaded into this process still equal those pinned inputs.  This catches
    # an import followed by a file replacement even though no callable bytecode moved.
    try:
        loaded_numa, loaded_shapes = stack_numa._load_numa_config(
            Path(source_by_name["topology"].path))
        loaded_manifest = stack_manifest._load_launch_manifest(
            Path(source_by_name["launch_manifest"].path))
        loaded_master = stack_manifest._load_master_registry(
            Path(source_by_name["lean_registry"].path))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise EnrollmentExportError(
            f"cannot reconstruct pinned loaded configuration: {exc}") from exc
    if (loaded_numa != stack_numa.NUMA_CONFIG
            or loaded_shapes != stack_numa.NUMA_INSTANCE_SHAPES
            or loaded_manifest != stack_manifest._MANIFEST
            or loaded_master != stack_manifest._MASTER
            or stack_paths._get_paths() != stack_paths._PATHS
            or stack_manifest._build_role_launch_meta() != stack_manifest.ROLE_LAUNCH_META
            or stack_manifest._build_servers_from_classification()
            != (stack_manifest.HOT_SERVERS, stack_manifest.WARM_SERVERS)):
        raise EnrollmentExportError(
            "loaded launcher configuration differs from pinned source inputs")
    active_roles = active_roles_from_launch_meta(ROLE_LAUNCH_META)
    # DRAFT-SEL-1: the drafter selection is a compile input; use the PINNED topology.
    compiled = compile_lean(Path(source_by_name["master_registry"].path), active_roles,
                            Path(source_by_name["topology"].path))
    loaded_lean = yaml.safe_load(Path(source_by_name["lean_registry"].path).read_text())
    if compiled != loaded_lean:
        raise EnrollmentExportError("compiled lean registry is stale for pinned master/launch roles")
    prior = yaml.safe_load(Path(source_by_name["stack_priors"].path).read_text())
    prior_sources = prior.get("source_artifacts") if isinstance(prior, Mapping) else None
    closure = {"registry": "lean_registry", "descriptors": "descriptors",
               "launch_manifest": "launch_manifest", "stack_topology": "topology",
               "stack_runtime": "stack_runtime", "stack_paths": "stack_paths"}
    if not isinstance(prior_sources, Mapping):
        raise EnrollmentExportError("compiled priors omit their source dependency closure")
    for prior_name, context_name in closure.items():
        declaration = prior_sources.get(prior_name)
        if (not isinstance(declaration, Mapping)
                or declaration.get("sha256") != source_by_name[context_name].sha256):
            raise EnrollmentExportError(f"compiled priors are stale for {context_name}")
    verification_failures = (_verify_artifact_pins(context.artifacts)
                             if verify_artifacts else {})

    # Consume the registry whose bytes were just pinned and checked.  An injected
    # RegistryLoader could carry stale in-memory role data under a current file path.
    registry = RegistryLoader(source_by_name["lean_registry"].path, validate_paths=False)
    entries = _filter_by_numa_mode(HOT_SERVERS + WARM_SERVERS, context.instance_mode)
    # Freeze production membership before appending explicitly requested launcher-only
    # seeds.  Otherwise merely making a seed resolvable turns it into an obligation.
    production_roles = {str(role) for entry in entries for role in entry["roles"]}
    explicit = set(context.requested_roles) | set(context.seed_roles)
    for role in sorted(explicit):
        meta = ROLE_LAUNCH_META.get(role)
        if not isinstance(meta, Mapping) or meta.get("launcher_only") is not True:
            continue
        port = meta.get("port", PORT_MAP.get(role))
        if not isinstance(port, int):
            raise EnrollmentExportError(f"launcher-only role {role!r} has no exact port")
        entry: dict[str, Any] = {"port": port, "roles": [role]}
        mode = meta.get("mode")
        if isinstance(mode, str):
            entry[mode] = True
        entries.append(entry)
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        for role in entry["roles"]:
            by_role.setdefault(str(role), []).append(entry)
    live_primary = list(dict.fromkeys(str(entry["roles"][0]) for entry in entries))
    requested = list(live_primary)
    for role in context.requested_roles:
        if role not in requested:
            requested.append(role)
    for seed in context.seed_roles:
        if seed not in requested:
            requested.append(seed)

    targets: list[dict[str, Any]] = []
    seen_instances: set[tuple[str, int]] = set()

    def in_scope(row: Mapping[str, Any]) -> bool:
        """A scoped export carries only targets of that backend, and nothing else.

        Omission, not annotation: a campaign that cannot name a resource cannot
        reach it.  ``context.backend_scope`` (sealed under ``export_sha256``) is
        what tells the reader the roster is deliberately short.
        """
        return context.backend_scope == "all" or row.get("backend") == context.backend_scope

    for requested_role in requested:
        matches = by_role.get(requested_role, [])
        if not matches:
            row = {"target_id": requested_role, "primary_role": requested_role,
                   "aliases": [], "obligations": [requested_role],
                   "optional_seed": requested_role in context.seed_roles,
                   "status": "unsupported", "reasons": ["role_not_in_selected_fleet"]}
            if in_scope(row):
                targets.append(row)
            continue
        for entry in matches:
            primary = str(entry["roles"][0])
            instance = int(entry.get("numa_instance", 0))
            key = (primary, int(entry["port"]))
            if key in seen_instances:
                # Alias and primary requests union obligations; never boost a seed.
                for row in targets:
                    if row.get("primary_role") == primary and row.get("port") == int(entry["port"]):
                        row["obligations"] = sorted(set(row["obligations"]) | {requested_role})
                        row["optional_seed"] = row["optional_seed"] and requested_role in context.seed_roles
                        break
                continue
            seen_instances.add(key)
            meta = ROLE_LAUNCH_META[primary]
            mode = str(meta.get("mode", "default"))
            unsupported: list[str] = []
            if mode in {"embedding", "vision"}:
                unsupported.append("multimodal_or_embedding_instrument_unsupported")
            try:
                role_config = registry.get_role(primary)
            except Exception:
                role_config = None
            if entry.get("worker_pool"):
                binary_dir, ld_paths = launcher._runtime_requirements_for_role(
                    registry, primary)
            elif entry.get("embedding") or entry.get("eval_batch_frontdoor"):
                binary_dir, ld_paths = None, None
            else:
                override_role = (launcher.GPU_SHADOW_LANE_TENANT_ROLE
                                 if entry.get("gpu_shadow_lane") else primary)
                binary_override, ld_paths = launcher._stack_prior_runtime_overrides(override_role)
                binary_dir = str(Path(binary_override).parent) if binary_override else None
            if role_config is None and mode == "default":
                unsupported.append("registry_role_missing")
                argv: list[str] = []
            else:
                try:
                    argv = launcher.build_server_command(
                        role_config, int(entry["port"]), numa_instance=instance,
                        binary_override=(str(Path(binary_dir) / "llama-server")
                                         if entry.get("worker_pool") and binary_dir else None),
                        prepare_runtime_dirs=False, **_mode_kwargs(entry))
                except Exception as exc:
                    argv = []
                    unsupported.append(f"launcher_resolution_failed:{type(exc).__name__}")
            env_role = _source_role(entry)
            env = launcher.build_launch_env(env_role, dict(context.base_environment))
            _apply_declared_runtime_env(
                env, binary_override=(str(Path(binary_dir) / "llama-server")
                                      if binary_dir else None), ld_paths=ld_paths)
            artifacts, missing = _artifact_rows(argv, env, context.artifacts) if argv else ([], [])
            selected_failures = [f"artifact_verification:{item['use']}:{reason}"
                                 for item in artifacts
                                 for reason in [verification_failures.get(
                                     (item["use"], item["path"]))] if reason is not None]
            missing.extend(selected_failures)
            device = _flag(argv, "--device") if argv else None
            backend = "cpu" if device == "none" else "gpu" if (
                isinstance(device, str) and device.startswith("ROCm")) else "unknown"
            if backend == "unknown":
                unsupported.append("backend_or_device_unsupported")
            if any("rpc" in item.lower() for item in argv):
                unsupported.append("rpc_unsupported")
            status = "unsupported" if unsupported else "waiting_artifact" if missing else "ready"
            topology = NUMA_CONFIG.get(primary, {})
            topology_prefix = _numa_prefix(primary, instance)
            target = {
                "target_id": f"{primary}@{entry['port']}", "primary_role": primary,
                "port": int(entry["port"]), "numa_instance": instance,
                "aliases": [str(x) for x in entry["roles"][1:]],
                "obligations": sorted(str(x) for x in entry["roles"]),
                "optional_seed": (requested_role in context.seed_roles
                                  and requested_role not in production_roles),
                "status": status, "reasons": sorted(set(unsupported + missing)),
                "backend": backend, "device": device,
                "speculation": (_flag(argv, "--spec-type") or
                                ("external" if _flag(argv, "-md") else "none")),
                "argv": topology_prefix + list(argv), "command_argv": list(argv),
                "environment": dict(sorted(env.items())),
                "environment_unsets": sorted(set(dict(context.base_environment)) - set(env)),
                "artifacts": artifacts,
                "artifact_verification": ("failed" if selected_failures else
                                          "passed" if verify_artifacts else
                                          "not_requested"),
                "workload": {"np": _flag(argv, "-np"), "context": _flag(argv, "-c"),
                             "threads": _flag(argv, "-t")},
                "topology": {"argv_prefix": topology_prefix,
                             "declaration": _configuration(topology),
                             "pre_evict_gib": pre_evict_gib_for_role(topology)},
                "runtime_requirements": {"binary_dir": binary_dir,
                                         "ld_library_path": ld_paths},
                "source_revisions": {pin.name: pin.revision for pin in context.sources},
                "source_revision_kinds": {pin.name: pin.revision_kind
                                          for pin in context.sources},
            }
            if in_scope(target):
                targets.append(target)

    for name in ("whisper", "tts"):
        service = AUX_SERVICES.get(name)
        if service is not None:
            row = {"target_id": f"speech:{name}", "primary_role": name,
                   "aliases": [], "obligations": [name], "optional_seed": False,
                   "status": "unsupported", "reasons": ["speech_instrument_unsupported"],
                   "backend": service.backend, "argv": list(service.argv),
                   "environment": dict(service.env)}
            if in_scope(row):
                targets.append(row)

    body = {"schema": EXPORT_SCHEMA, "context": context.to_dict(), "targets": targets,
            "disposition": {status: sum(row["status"] == status for row in targets)
                            for status in ("ready", "waiting_artifact", "unsupported")}}
    body["export_sha256"] = _digest(json.dumps(body, sort_keys=True, separators=(",", ":"))
                                     .encode("utf-8"))
    return body


_RECIPE_FIELDS = (
    "backend", "port", "numa_instance", "command_argv", "environment",
    "environment_unsets", "workload", "topology", "runtime_requirements",
    "source_revisions", "source_revision_kinds", "speculation",
)


def _recipe_body(target: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in _RECIPE_FIELDS if field not in target]
    if missing:
        raise EnrollmentExportError(
            f"target {target.get('target_id')!r} lacks recipe fields {missing}")
    artifacts = [dict(item) for item in target.get("artifacts", [])
                 if isinstance(item, Mapping) and item.get("use") != "recipe"]
    return {"schema": RECIPE_ARTIFACT_SCHEMA,
            "launch": {field: _plain(target[field], f"recipe.{field}")
                       for field in _RECIPE_FIELDS},
            "artifacts": sorted(artifacts,
                                key=lambda row: (str(row.get("use")), str(row.get("path"))))}


def _fsync_directory(fd: int) -> None:
    os.fsync(fd)


def _write_exact_at(directory_fd: int, name: str, body: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(body)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_owned_at(directory_fd: int, name: str, *, linked: bool = False
                   ) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink not in ({1, 2} if linked else {1})):
            raise EnrollmentExportError(f"unsafe immutable export artifact {name!r}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        os.fsync(fd)
        return b"".join(chunks), info
    finally:
        os.close(fd)


def _read_exact_at(directory_fd: int, name: str) -> bytes:
    return _read_owned_at(directory_fd, name)[0]


def _publish_exact_at(directory_fd: int, name: str, body: bytes) -> None:
    """Recover or publish one create-only file while holding the parent lock."""
    stage = f".{name}.stage"
    names = set(os.listdir(directory_fd))
    if name in names:
        existing, final_info = _read_owned_at(directory_fd, name, linked=True)
        if stage in names:
            staged, stage_info = _read_owned_at(directory_fd, stage, linked=True)
            same = (final_info.st_dev, final_info.st_ino) == (stage_info.st_dev, stage_info.st_ino)
            if same and final_info.st_nlink == stage_info.st_nlink == 2 and staged == existing:
                os.unlink(stage, dir_fd=directory_fd)
                _fsync_directory(directory_fd)
            elif stage_info.st_nlink == 1:
                os.unlink(stage, dir_fd=directory_fd)
                _fsync_directory(directory_fd)
            else:
                raise EnrollmentExportError("conflicting recipe publication stage")
        if existing != body or _read_exact_at(directory_fd, name) != body:
            raise EnrollmentExportError(f"immutable export artifact {name!r} differs")
        _fsync_directory(directory_fd)
        return
    if stage in names:
        staged, stage_info = _read_owned_at(directory_fd, stage)
        if staged != body:
            os.unlink(stage, dir_fd=directory_fd)
            _fsync_directory(directory_fd)
            _write_exact_at(directory_fd, stage, body)
    else:
        _write_exact_at(directory_fd, stage, body)
    os.link(stage, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            follow_symlinks=False)
    _fsync_directory(directory_fd)
    os.unlink(stage, dir_fd=directory_fd)
    _fsync_directory(directory_fd)
    if _read_exact_at(directory_fd, name) != body:
        raise EnrollmentExportError("published export artifact failed exact verification")


def seal_export_bundle(value: Mapping[str, Any], output: Path | str) -> dict[str, Any]:
    """Create exact recipe sidecars and enrollment JSON; never replace a path."""
    body = _plain(value, "production enrollment")
    if not isinstance(body, dict) or body.get("schema") != EXPORT_SCHEMA:
        raise EnrollmentExportError("cannot seal malformed production enrollment")
    out = Path(output)
    if not out.is_absolute():
        out = out.resolve()
    parent = out.parent
    parent_info = parent.lstat()
    if parent.is_symlink() or not stat.S_ISDIR(parent_info.st_mode):
        raise EnrollmentExportError("export parent must be a real directory")
    recipes_name = out.name + ".recipes"
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        recipe_files: dict[str, bytes] = {}
        for target in body["targets"]:
            if target.get("backend") not in {"cpu", "gpu"} or not target.get("command_argv"):
                continue
            artifacts = target.get("artifacts", [])
            if not isinstance(artifacts, list) or not all(
                    isinstance(item, Mapping) for item in artifacts):
                raise EnrollmentExportError("target artifacts must be an array of objects")
            target["artifacts"] = [item for item in artifacts if item.get("use") != "recipe"]
            recipe_bytes = _canonical_bytes(_recipe_body(target))
            recipe_digest = _digest(recipe_bytes)
            recipe_files[f"{recipe_digest}.json"] = recipe_bytes
            target.setdefault("artifacts", []).append({
                "use": "recipe", "path": str(parent / recipes_name / f"{recipe_digest}.json"),
                "sha256": recipe_digest})

        try:
            recipes_fd = os.open(
                recipes_name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd)
        except FileNotFoundError:
            os.mkdir(recipes_name, 0o700, dir_fd=parent_fd)
            recipes_fd = os.open(
                recipes_name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd)
        try:
            info = os.fstat(recipes_fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise EnrollmentExportError("unsafe recipe bundle directory ownership or mode")
            existing = set(os.listdir(recipes_fd))
            allowed = set(recipe_files) | {f".{name}.stage" for name in recipe_files}
            if existing - allowed:
                raise EnrollmentExportError("recipe bundle contains unexpected files")
            for name, recipe_bytes in recipe_files.items():
                _publish_exact_at(recipes_fd, name, recipe_bytes)
            _fsync_directory(recipes_fd)
        finally:
            os.close(recipes_fd)
        _fsync_directory(parent_fd)

        unsigned = {key: item for key, item in body.items() if key != "export_sha256"}
        body["export_sha256"] = _digest(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":")).encode())
        encoded = json.dumps(body, indent=2, sort_keys=True).encode() + b"\n"
        _publish_exact_at(parent_fd, out.name, encoded)
        _fsync_directory(parent_fd)
        return body
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        finally:
            os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export production enrollment without launching")
    parser.add_argument("--master-registry", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--instance-mode", choices=("full", "quarter", "both"), required=True)
    parser.add_argument("--backend", choices=BACKEND_SCOPES, default="all",
                        help="restrict the export to one backend; out-of-scope targets "
                             "are omitted entirely and the scope is sealed in the export")
    parser.add_argument("--role", action="append", default=[])
    parser.add_argument("--seed-role", action="append", default=[])
    parser.add_argument("--artifact-pins", type=Path)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        artifacts: tuple[ArtifactPin, ...] = ()
        if args.artifact_pins:
            raw = json.loads(args.artifact_pins.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise EnrollmentExportError("artifact pin file must be an array")
            artifacts = tuple(ArtifactPin.from_dict(item) for item in raw)
        context = capture_current_context(
            master_registry=args.master_registry, revision=args.revision,
            instance_mode=args.instance_mode, artifacts=artifacts,
            requested_roles=args.role, seed_roles=args.seed_role,
            backend_scope=args.backend)
        body = export_production_enrollment(context, verify_artifacts=args.verify_artifacts)
        encoded = json.dumps(body, indent=2, sort_keys=True) + "\n"
        if args.out is None:
            sys.stdout.write(encoded)
        else:
            seal_export_bundle(body, args.out)
    except (EnrollmentExportError, OSError, json.JSONDecodeError) as exc:
        print(f"production enrollment refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ArtifactPin", "BACKEND_SCOPES", "CONTEXT_SCHEMA", "EXPORT_SCHEMA",
           "RECIPE_ARTIFACT_SCHEMA",
           "EnrollmentExportError",
           "ExportContext", "SourcePin", "capture_current_context",
           "export_production_enrollment", "seal_export_bundle", "main"]
