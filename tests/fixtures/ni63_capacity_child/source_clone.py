"""Build an external, source-pinned clone for NI63's import-capacity child probes.

This fixture copies only the Python modules in the import-time caller closure,
plus the immutable declared GPU-capacity policy input and owned YAML fixtures.
It reads real /proc/meminfo; it never edits or substitutes host memory values.
"""

from __future__ import annotations

import hashlib
import os
import sys
import sysconfig
import tempfile
from pathlib import Path

PINNED_APP_COMMIT = "6bb8d860d410217895efb7806e9ef907a62a4350"
# (Git blob SHA-1, content SHA-256), independently resolved from the pinned tree.
SOURCE_PINS: dict[str, tuple[str, str]] = {
    "scripts/server/bench_core_claim.py": ("58462c114e341beab0d0480d570f26e80b0072ba", "66a04cb66895b2898c65820088a23a14a0de9db54df93b74fe0860aef055eb5e"),
    "scripts/server/fleet_markers.py": ("070b6ab8af62fbd33cffa4069133cca915c3258e", "8ef8df7703252f41f52c2f9033e941d28d4fea7017d220a7e3dc617e1a2aec68"),
    "scripts/server/orchestrator_stack.py": ("a63a00b10e73edc9b46a5610800812395344c4eb", "645cef6e3ec5196734c3bf27a84fe1f1670302e5ac958431a54625b19104d833"),
    "scripts/server/realized_fleet.py": ("12f01d3c17a2ccb13c193728d79e24d52148de4e", "9099aa115a22799ba2446bf2082e0d7ebb360a0df5f57aac654b6286c1ea9a31"),
    "scripts/server/runtime_facts_manifest.py": ("c18bb563aabfa092e6b378fdddbd5f516352cd5d", "e2bfec7bb580fac727ca184b2dd4d56bd487e443e89d4720cc037743ab1dc374"),
    "scripts/server/stack_env.py": ("0ad2da92f14a50e2145e36fe32e31011d0da8606", "c2dd2a40abf2ac3048e392a11bd910a27f6991e2a99cefdbd232172b67ea06a4"),
    "scripts/server/stack_health.py": ("99d48f391565e0befb10040a0fea252087fc0313", "19ede42e88efeda2ab1fc5eb82c1272ac9d992f59313103c7d8d83917d4dfcad"),
    "scripts/server/stack_log_banner.py": ("acdf77435557f3e06635725dc17eb90b29d5b3c9", "70daa945b70e0d74962016aac85c21fda4c56e4c44d02fa6621cc5cdcf2d2ef5"),
    "scripts/server/stack_manifest.py": ("8ec58a7e5d28e3b214f3f4e18edefba136828930", "5f7acb6d067e4420edfc2bcc662148cd8fc7aff6c90682c51ece207666171610"),
    "scripts/server/stack_numa.py": ("78ba16a1d289fa7f44480aa5f0f225ce70103908", "edfdb41035ec638d3eab60631024ebece56a5e2d9b577851bf02e05a446e0c32"),
    "scripts/server/stack_numa_evict.py": ("09116f5040a575405f0d7807bb7af211085f906b", "575b811b72b31ed95f7296ee7aecb9e245aae9d603496e4baa1137ab42cb77de"),
    "scripts/server/stack_numa_mode.py": ("115bb4c3d5b61802452512f98819fbf86ae67e9c", "e17e1c7260dda722b5e1f555e5fc92204a4e8d93a397a7128a0ed76c057afe63"),
    "scripts/server/stack_paths.py": ("18c2daab3dd0d2941a112c686d73e5bad853f2a9", "85291c79110d3ab7683b0acbffb9158f320c138ad61f8acbdde269ee7a47abd5"),
    "scripts/server/stack_processes.py": ("df278a70d5801b81dc54c5ec663d19859ad4a50e", "e9ac8d7951fb6a11664c1d69e71050dfd44ed51ae3e215858ffd698198459812"),
    "scripts/server/stack_runtime.py": ("2d4489a3d4e6046973255dbf2b9a2d47b65c6a7e", "984d27262e1f1d5a5332d0b4a3e6fb96fb8095fc890ddaa93da07bff52eb1c99"),
    "scripts/server/stack_state.py": ("5d7ad9483260e4e56f8105df702fc5ff1b7f782f", "a047f152ac478e8fead80700b73bcb782dfe61969f513aa69da422518c89da8c"),
    "src/__init__.py": ("ff893046a8e7d2598657b950e9e41449f358d25f", "16e20bce77e6470646b90def0e3d653a3bd9fb024b484f7cd034f96ff8a0a5f8"),
    "src/config/__init__.py": ("90c83b49ed877c1615661f4a3ac188efac02d644", "bd0a888d06c952e0899093233241029b6cac2b6c2e1d605fea4b56878fe04d17"),
    "src/config/models.py": ("a846546e725d448cbf9321fba84c2128905ccaf3", "9ff0f384f74ffd315be19311f52b32cb9db78a2209248a9040dc7a9b185576b7"),
    "src/config/validation.py": ("64f122cf6a2d7f0d29baac9bcc955029ec8f9cf5", "dc33a052fb1e0962b3546336017d239929a36a8942502a7aa99da064b20e32da"),
    "src/env_parsing.py": ("11e3b7c80df1a6f921d467f7a924d287e1ac28dc", "2626fb2c47012c56a7f7e99fe03f088977f9b061c77bcd1edf0dfa1e12b7d52f"),
    "src/registry/__init__.py": ("287333a63889217a52c0082a7e316dc64670ee1f", "50ddc75e291c2caaae688bbe94c4f3144a5751eb11e0a65ee35b5632ea3a9c6d"),
    "src/registry/kernel_paths.py": ("6bb3dad23130a763dc4c456c7c432a9535536f30", "f5cd510ad2887a6276350c8939147ce66ba78a8eab573d42ee120094c0795066"),
    "src/registry/registry_loader.py": ("0bddc98c3a27c40d4e117783f70808f22551fc4f", "c4a4bf71f0f90384ffb64c673897c9fd496889e4e395a2e3eaf94933550ed18b"),
    "src/registry/model_descriptors.py": ("81d056260371f0da2f097aa1a184ee8c892e3f9c", "ba868c939eba8ebc1274bd2d05bcff865c702fcdef1e3d6ec7184833aad44f69"),
    "src/registry/stack_priors.py": ("22a04ef5aba708f2f351acf740573eaba9f659f0", "188e4093675afc23308ad1c3b8afd0c4d104f9626d6658ae78849b3ab90fca7f"),
    "src/registry_loader.py": ("c68c5918b94e4a54a07a7c6642d92365480bf656", "81f3dc46dea4af9eb8df9adb4a73d2e900e1d040d107a73a04053f8ea72b77be"),
    "src/roles.py": ("59432d52c3c5ebb661c336f70627267073cfa4cb", "408f10be964705638562ec9a6d0e1cd56b26ac849fdb964e8bac0c9a7fe4be61"),
    "src/scheduling/__init__.py": ("e81d5c94b0479ae82472dfceec8393cd8f8286d2", "a65b963b3d38ca2d680827bfe3debe3e7d91095ea50723c5c40ce6b19dd0df51"),
    "src/scheduling/device_model.py": ("78e0ede88801c39a842fadcbf6b4322d1c1a83ac", "90e3f9e001bbcb940c4cb493caa4df29b4da244b7d9834e6b09632c9fb49171b"),
}
CAPACITY_ARTIFACT_PIN = (
    "orchestration/gpu_shadow_lane_np_ceiling.yaml",
    "46e79920d1b88da5397f0cb0058295f9660f8f65",
    "d4a45fbbbb33ce37ec1176ed2252596db4546a7755d9757151aca670c42fb1dd",
)
FIXTURE_PINS: dict[str, str] = {
    "launch_manifest.yaml": "583bbb9ee8910c5633e454f5a3528e4f9a86146b6641448e92ebf6b573cfd41b",
    "stack_topology.yaml": "d4fcf1ee0583df9072450129278b55339df3f9f115d3c18529e72f29b44135e4",
    "model_registry_fit.yaml": "b1afae32f0cd1256eef7e43e55fb25241e96f0b962adc87896fd2ce3a4c41080",
    "model_registry_oversized.yaml": "5547c56b5730ca528dfd446a7f1c655d64eecb4fe3e303275f2f73e0526c8b64",
}


def _git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def read_real_meminfo() -> dict[str, int]:
    """Read the runner's actual MemTotal and MemAvailable in kB, without writes."""
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            result[key] = int(rest.split()[0])
    if set(result) != {"MemTotal", "MemAvailable"}:
        raise RuntimeError("/proc/meminfo did not expose both required host facts")
    return result


def assert_real_available_headroom(
    required_gib: float, reserve_gib: float, meminfo_kb: dict[str, int]
) -> None:
    """Require actual MemAvailable to cover fixture arithmetic plus its reserve."""
    total_gib = meminfo_kb["MemTotal"] / (1024.0 * 1024.0)
    available_gib = meminfo_kb["MemAvailable"] / (1024.0 * 1024.0)
    if available_gib > total_gib:
        raise AssertionError("actual MemAvailable exceeds actual MemTotal")
    if required_gib + reserve_gib > available_gib:
        raise AssertionError(
            f"fixture needs {required_gib:.6f} GiB plus {reserve_gib:.3f} GiB reserve, "
            f"but actual MemAvailable is {available_gib:.3f} GiB "
            f"of {total_gib:.3f} GiB MemTotal"
        )


def materialize_source_clone(
    source_root: Path,
    private_parent: Path,
    *,
    variant: str,
    project_venv_python: Path,
) -> Path:
    """Copy only pinned import-time Python sources and the selected fixture data."""
    if variant not in {"fit", "oversized"}:
        raise ValueError("variant must be 'fit' or 'oversized'")
    source_root = source_root.resolve(strict=True)
    private_parent = private_parent.resolve(strict=True)
    if private_parent == source_root or source_root in private_parent.parents:
        raise ValueError("fixture source clone must live outside the checked-out repository")
    data_root = Path(__file__).resolve().parent
    clone = Path(tempfile.mkdtemp(prefix="ni63-capacity-child-", dir=private_parent))
    clone.chmod(0o700)

    for relative, (expected_blob, expected_sha256) in SOURCE_PINS.items():
        source = source_root / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"pinned source is absent or linked: {relative}")
        contents = source.read_bytes()
        got_blob = _git_blob_sha1(contents)
        got_sha256 = hashlib.sha256(contents).hexdigest()
        if (got_blob, got_sha256) != (expected_blob, expected_sha256):
            raise RuntimeError(f"source differs from accepted APP pin: {relative}")
        target = clone / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(0o600)

    for fixture_name, expected_sha256 in FIXTURE_PINS.items():
        source = data_root / fixture_name
        contents = source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise RuntimeError(f"fixture bytes changed without updating the pin: {fixture_name}")
        if fixture_name.startswith("model_registry_"):
            if (variant == "fit") != (fixture_name == "model_registry_fit.yaml"):
                continue
            target_name = "model_registry.yaml"
        else:
            target_name = fixture_name
        target = clone / "orchestration" / target_name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(contents)
        target.chmod(0o600)

    capacity_path, capacity_blob, capacity_sha256 = CAPACITY_ARTIFACT_PIN
    capacity_source = source_root / capacity_path
    contents = capacity_source.read_bytes()
    if (_git_blob_sha1(contents), hashlib.sha256(contents).hexdigest()) != (
        capacity_blob,
        capacity_sha256,
    ):
        raise RuntimeError("declared GPU-capacity source differs from accepted APP pin")
    capacity_target = clone / capacity_path
    capacity_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    capacity_target.write_bytes(contents)
    capacity_target.chmod(0o600)

    real_venv_python = project_venv_python.resolve(strict=True)
    venv_link = clone / ".venv" / "bin" / "python"
    venv_link.parent.mkdir(mode=0o700, parents=True)
    venv_link.symlink_to(real_venv_python)
    (clone / "scratch" / "tmp").mkdir(mode=0o700, parents=True)
    (clone / "scratch" / "logs").mkdir(mode=0o700, parents=True)
    (clone / "scratch" / "cache").mkdir(mode=0o700, parents=True)
    return clone


def probe_environment(clone: Path) -> dict[str, str]:
    """Explicit child allowlist; preserve inherited HOME and exclude provider/memory overrides."""
    clone = clone.resolve(strict=True)
    purelib = sysconfig.get_paths()["purelib"]
    scratch = clone / "scratch"
    absent = clone / "absent"
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(purelib).resolve(strict=True)),
        "TMPDIR": str(scratch / "tmp"),
        "ORCHESTRATOR_PATHS_PROJECT_ROOT": str(clone),
        "ORCHESTRATOR_PATHS_LLM_ROOT": str(absent / "llm"),
        "ORCHESTRATOR_PATHS_MODELS_DIR": str(absent / "models"),
        "ORCHESTRATOR_PATHS_MODEL_BASE": str(absent / "model-base"),
        "ORCHESTRATOR_PATHS_LLAMA_CPP_BIN": str(absent / "kernel-bin"),
        "ORCHESTRATOR_PATHS_LLAMA_MTMD": str(absent / "llama-mtmd"),
        "ORCHESTRATOR_PATHS_LOG_DIR": str(scratch / "logs"),
        "ORCHESTRATOR_PATHS_CACHE_DIR": str(scratch / "cache"),
        "ORCHESTRATOR_PATHS_TMP_DIR": str(scratch / "tmp"),
        "ORCHESTRATOR_PATHS_REGISTRY_PATH": str(clone / "orchestration" / "model_registry.yaml"),
        "ORCHESTRATOR_PATHS_TOOL_REGISTRY_PATH": str(absent / "tool_registry.yaml"),
        "ORCHESTRATOR_PATHS_SCRIPT_REGISTRY_DIR": str(absent / "script-registry"),
        "ORCHESTRATOR_PATHS_STACK_PRIORS_PATH": str(absent / "stack-priors.yaml"),
        "ORCHESTRATOR_PATHS_VISION_DIR": str(absent / "vision"),
        "ORCHESTRATOR_PATHS_RAID_PREFIX": str(absent / "raid"),
    }
    forbidden = {
        "HOME",
        "ORCHESTRATOR_STACK_REEXEC",
        "ORCHESTRATOR_HOST_MEMTOTAL_GIB",
        "ORCHESTRATOR_HOST_MEMAVAILABLE_GIB",
        "ORCHESTRATOR_VRAM_TOTAL_GIB",
        "ORCHESTRATOR_VRAM_HEADROOM_GIB",
    }
    if forbidden & env.keys():
        raise AssertionError(f"probe allowlist contains forbidden overrides: {sorted(forbidden & env.keys())}")
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    if (absent / "kernel-bin").exists():
        raise AssertionError("fixture kernel path unexpectedly exists")
    return env
