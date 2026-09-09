"""Closed child entry point for one previously admitted native build request."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Mapping

from ..execution import worktree
from . import lifecycle_observation, measurement_capture

REQUEST_SCHEMA = "epyc.autokernel.source_build_worker_request.v1"
RESULT_SCHEMA = "epyc.autokernel.source_build_worker_result.v1"
PRODUCER_SCHEMA = "epyc.autokernel.source_build_worker_producer.v1"


class SourceBuildWorkerRefused(RuntimeError):
    pass


def producer_identity() -> dict:
    # Import only while constructing the identity.  The entrypoint imports this
    # module, so importing it at module load time would create a cycle.
    from . import source_build_entrypoint
    callables = (
        producer_identity, run, source_build_entrypoint.main, _plan,
        worktree.run_build, worktree._run_owned,
        worktree._sealed_process_receipt,
        worktree.process_sandbox.SandboxPolicy.wrap,
        worktree.process_sandbox._join_owned_cgroup,
        worktree.process_sandbox.cleanup_cgroup,
    )
    identities = tuple(lifecycle_observation.callable_identity(item) for item in callables)
    if any(row["implementation_status"] != "pinned"
           or row["configuration_status"] != "pinned" for row in identities):
        raise SourceBuildWorkerRefused("build worker loaded callable closure is incomplete")
    modules = {}
    for module in (sys.modules[__name__], worktree, worktree.process_sandbox,
                   measurement_capture):
        path = Path(module.__file__).resolve()
        modules[module.__name__] = {
            "path": str(path), "sha256": worktree._sha256_file(path)}
    entrypoint = Path(__file__).with_name("source_build_entrypoint.py").resolve()
    modules["autokernel.loop.source_build_entrypoint"] = {
        "path": str(entrypoint), "sha256": worktree._sha256_file(entrypoint)}
    body = {"schema": PRODUCER_SCHEMA, "callables": list(identities),
            "modules": modules, "python_version": sys.version,
            "constants": {"request_schema": REQUEST_SCHEMA, "result_schema": RESULT_SCHEMA}}
    return body | {"sha256": worktree.schemas.content_hash(body)}


def _plan(value) -> worktree.BuildPlan:
    if not isinstance(value, Mapping) or set(value) != {
            "source_root", "build_dir", "actor_worktree", "parallelism", "targets",
            "build_type", "cmake_defines", "generator", "allow_ccache",
            "configure_command", "build_command"}:
        raise SourceBuildWorkerRefused("build plan fields differ")
    parallel = value["parallelism"]
    if not isinstance(parallel, Mapping) or set(parallel) != {
            "jobs", "cpu_list", "load_average_cap"}:
        raise SourceBuildWorkerRefused("build parallelism fields differ")
    cmake_defines = value["cmake_defines"]
    if (not isinstance(cmake_defines, (list, tuple))
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   for row in cmake_defines)):
        raise SourceBuildWorkerRefused("build definitions differ")
    configure = value["configure_command"]
    if not isinstance(configure, (list, tuple)) or not configure:
        raise SourceBuildWorkerRefused("configure command differs")
    prefix = 3 if tuple(configure[:2]) == ("taskset", "-c") else 0
    if len(configure) <= prefix or Path(configure[prefix]).name != "cmake":
        raise SourceBuildWorkerRefused("build plan does not use the admitted cmake program")
    result = worktree.BuildPlan(
        source_root=worktree.SandboxPath.create(value["source_root"], label="source root"),
        build_dir=worktree.SandboxPath.create(value["build_dir"], label="build dir"),
        actor_worktree=worktree.SandboxPath.create(
            value["actor_worktree"], label="actor worktree"),
        parallelism=worktree.BuildParallelism(**parallel),
        targets=tuple(value["targets"]), build_type=value["build_type"],
        cmake_defines=tuple(tuple(row) for row in cmake_defines),
        generator=value["generator"], allow_ccache=value["allow_ccache"],
        cmake=configure[prefix])
    if result.to_dict() != measurement_capture._plain(value):
        raise SourceBuildWorkerRefused("reconstructed build plan differs")
    return result


def run(request) -> dict:
    if not isinstance(request, Mapping) or set(request) != {
            "schema", "build_plan", "log_path", "configure_timeout_s",
            "build_timeout_s", "env", "sandbox_cgroup_root", "producer_identity"} \
            or request["schema"] != REQUEST_SCHEMA:
        raise SourceBuildWorkerRefused("worker request fields/schema differ")
    supplied_identity = measurement_capture._plain(request["producer_identity"])
    current_identity = producer_identity()
    if supplied_identity != current_identity:
        differing = sorted(key for key in current_identity
                           if supplied_identity.get(key) != current_identity[key])
        details = {
            "modules": [key for key in current_identity["modules"]
                        if supplied_identity.get("modules", {}).get(key)
                        != current_identity["modules"][key]],
            "callables": [index for index, value in enumerate(current_identity["callables"])
                          if index >= len(supplied_identity.get("callables", ()))
                          or supplied_identity["callables"][index] != value],
        }
        raise SourceBuildWorkerRefused(
            f"build worker producer identity differs: {differing}: {details}")
    plan = _plan(request["build_plan"])
    result = worktree.run_build(
        plan, log_path=request["log_path"],
        configure_timeout_s=request["configure_timeout_s"],
        build_timeout_s=request["build_timeout_s"], env=request["env"],
        require_fresh_build_dir=True,
        sandbox_cgroup_root=request["sandbox_cgroup_root"])
    if result.result_receipt_path is None or result.result_receipt_sha256 is None:
        raise SourceBuildWorkerRefused("build produced no sealed result receipt")
    return {"schema": RESULT_SCHEMA,
            "result_receipt_path": result.result_receipt_path,
            "result_receipt_sha256": result.result_receipt_sha256}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SourceBuildWorkerRefused("expected artifact root, locator and digest")
    root, locator, sha256 = args
    store = measurement_capture.ArtifactStore(Path(root))
    try:
        request = store.read(locator, sha256)
    finally:
        store.close()
    result = run(dict(request))
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by owned child integration
    raise SystemExit(main())

