#!/usr/bin/env python3
"""Run the pinned SWE-bench evaluator with atomic container CPU isolation.

The stock harness creates containers through the Docker daemon, so an outer
``taskset`` on the evaluator client does not constrain test processes. This
adapter changes only the container-create call: every evaluator container gets
the canonical q2 cpuset before it can be started. It intentionally leaves
patch application, test execution, and scoring in the pinned harness.
"""

from __future__ import annotations

import importlib
import hashlib
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping


CPUSET_ENV = "SWEBENCH_EVAL_CPUSET"
CANONICAL_CPUSET = "112-119"
DOCKER_BUILD_SHA256 = "5278842b60a7d38256f95f93c915dc84de2b8b4f286e9baae1b19280f768e484"


class CpuIsolationError(RuntimeError):
    """Raised before a container can start without the canonical cpuset."""


def require_cpuset(environ: Mapping[str, str] | None = None) -> str:
    """Return the one permitted evaluator cpuset or fail closed."""
    value = (os.environ if environ is None else environ).get(CPUSET_ENV)
    if value != CANONICAL_CPUSET:
        raise CpuIsolationError(
            f"{CPUSET_ENV} must be exactly {CANONICAL_CPUSET!r}; got {value!r}"
        )
    return value


def attest_container_cpuset(container: Any, expected: str) -> None:
    """Inspect Docker's persisted host config before the caller starts it."""
    container.reload()
    actual = container.attrs.get("HostConfig", {}).get("CpusetCpus")
    if actual != expected:
        raise CpuIsolationError(
            f"Docker persisted CpusetCpus={actual!r}, expected {expected!r}"
        )


def attest_docker_build(docker_build: Any) -> None:
    """Refuse to patch a different harness implementation at runtime."""
    source = Path(docker_build.__file__).resolve()
    actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else "<missing>"
    if actual != DOCKER_BUILD_SHA256:
        raise CpuIsolationError(
            f"pinned docker_build SHA-256 drifted: {source}: {actual}"
        )


def install_adapter(docker_build: Any, expected: str) -> None:
    """Replace only the stock container factory with a cpuset-attesting copy."""

    def build_container(
        test_spec: Any,
        client: Any,
        run_id: str,
        logger: Any,
        nocache: bool,
        force_rebuild: bool = False,
    ) -> Any:
        if force_rebuild:
            docker_build.remove_image(client, test_spec.instance_image_key, "quiet")
        if not test_spec.is_remote_image:
            docker_build.build_instance_image(test_spec, client, logger, nocache)
        else:
            try:
                client.images.get(test_spec.instance_image_key)
            except docker_build.docker.errors.ImageNotFound:
                try:
                    client.images.pull(test_spec.instance_image_key)
                except docker_build.docker.errors.NotFound as exc:
                    raise docker_build.BuildImageError(test_spec.instance_id, str(exc), logger) from exc
                except Exception as exc:
                    raise Exception(
                        f"Error occurred while pulling image {test_spec.base_image_key}: {str(exc)}"
                    ) from exc

        container = None
        try:
            logger.info(f"Creating CPU-isolated container for {test_spec.instance_id}...")
            run_args = test_spec.docker_specs.get("run_args", {})
            cap_add = run_args.get("cap_add", [])
            container = client.containers.create(
                image=test_spec.instance_image_key,
                name=test_spec.get_instance_container_name(run_id),
                user=docker_build.DOCKER_USER,
                detach=True,
                command="tail -f /dev/null",
                platform=test_spec.platform,
                cap_add=cap_add,
                cpuset_cpus=expected,
            )
            attest_container_cpuset(container, expected)
            logger.info(
                f"Container for {test_spec.instance_id} created with CpusetCpus={expected}: "
                f"{container.id}"
            )
            return container
        except Exception as exc:
            logger.error(f"Error creating CPU-isolated container for {test_spec.instance_id}: {exc}")
            logger.info(docker_build.traceback.format_exc())
            docker_build.cleanup_container(client, container, logger)
            raise docker_build.BuildImageError(test_spec.instance_id, str(exc), logger) from exc

    docker_build.build_container = build_container


def main(argv: list[str] | None = None) -> int:
    """Install the adapter, then execute the stock CLI with its original argv."""
    expected = require_cpuset()
    argv = list(sys.argv[1:] if argv is None else argv)
    docker_build = importlib.import_module("swebench.harness.docker_build")
    attest_docker_build(docker_build)
    install_adapter(docker_build, expected)

    # The package imports run_evaluation eagerly. Dropping that cached module
    # makes its ``from docker_build import build_container`` bind the adapter.
    sys.modules.pop("swebench.harness.run_evaluation", None)
    sys.argv = ["swebench.harness.run_evaluation", *argv]
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
