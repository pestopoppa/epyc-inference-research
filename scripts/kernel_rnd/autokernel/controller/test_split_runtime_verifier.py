from __future__ import annotations

from pathlib import Path
import hashlib
import tempfile
import unittest

from . import split_runtime_verifier as V


COMMON = (
    "libllama-common.so", "libllama.so", "libggml.so",
    "libggml-base.so", "libggml-cpu.so",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chain(root: Path, stem: str, payload: bytes) -> Path:
    version = root / f"{stem}.0.16.0"
    version.write_bytes(payload)
    (root / f"{stem}.0").symlink_to(version.name)
    (root / stem).symlink_to(f"{stem}.0")
    return version


def _runtime(root: Path) -> tuple[Path, Path]:
    common = root / "common"
    anchor = root / "anchor-hip"
    candidate = root / "candidate-hip"
    for directory in (common, anchor, candidate):
        directory.mkdir(parents=True)
    reward = common / "llama-bench"
    reward.write_bytes(b"reward")
    reward.chmod(0o755)
    (common / "libllama-bench-impl.so").write_bytes(b"bench-impl")
    for stem in COMMON:
        _chain(common, stem, stem.encode())
    _chain(anchor, "libggml-hip.so", b"anchor-hip")
    _chain(candidate, "libggml-hip.so", b"candidate-hip")
    return reward, common


def _fake_elf(path: Path) -> V.ElfIdentity:
    if path.name == "llama-bench":
        return V.ElfIdentity(None, ("libllama-bench-impl.so",), ("$ORIGIN",))
    if path.name.startswith("libggml-hip.so."):
        return V.ElfIdentity("libggml-hip.so.0", ("libggml-base.so.0",),
                             ("/opt/rocm/lib",))
    soname = path.name.rsplit(".0.", 1)[0] + ".0" if ".0." in path.name else path.name
    needed = ("libggml-hip.so.0", "libggml-base.so.0") \
        if path.name.startswith("libggml.so.") else ("libc.so.6",)
    return V.ElfIdentity(soname, needed, ("$ORIGIN", "/opt/rocm/lib"))


def _maps_line(path: Path) -> str:
    return f"7f000000-7f001000 r-xp 00000000 00:00 1 {path.resolve()}"


def _maps(manifest: V.SplitRuntimeManifest, arm: str, model: Path) -> str:
    paths = {path.resolve() for path in manifest.common_dir.iterdir() if path.is_file()}
    hip = manifest.anchor_hip_dir if arm == "anchor" else manifest.candidate_hip_dir
    paths.add((hip / "libggml-hip.so.0").resolve())
    paths.add(model.resolve())
    return "\n".join(_maps_line(path) for path in sorted(paths))


class SplitRuntimeVerifierTests(unittest.TestCase):
    def test_exact_layout_elf_identity_and_sanitized_environments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _runtime(root)
            manifest = V.verify_split_runtime(root, elf_reader=_fake_elf)
        self.assertNotEqual(manifest.anchor_hip_sha256,
                            manifest.candidate_hip_sha256)
        self.assertEqual(dict(manifest.arm_environment("anchor")), {
            "PATH": "/usr/bin:/bin",
            "LD_LIBRARY_PATH":
                f"{manifest.anchor_hip_dir}:{manifest.common_dir}:/opt/rocm/lib",
        })
        self.assertNotIn("LD_PRELOAD", manifest.arm_environment("candidate"))

    def test_extra_object_and_escaping_symlink_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _, common = _runtime(root)
            (common / "libreward-cheat.so").write_bytes(b"cheat")
            with self.assertRaisesRegex(V.SplitRuntimeError, "membership differs"):
                V.verify_split_runtime(root, elf_reader=_fake_elf)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _, common = _runtime(root)
            (common / "libggml.so").unlink()
            (common / "libggml.so").symlink_to("../common/libggml.so.0")
            with self.assertRaisesRegex(V.SplitRuntimeError, "escapes exact closure"):
                V.verify_split_runtime(root, elf_reader=_fake_elf)

    def test_wrong_hip_soname_and_relative_runpath_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _runtime(root)
            def wrong_soname(path: Path) -> V.ElfIdentity:
                value = _fake_elf(path)
                if path.name.startswith("libggml-hip.so."):
                    return V.ElfIdentity("libggml-hip.so.9", value.needed,
                                         value.runpath)
                return value
            with self.assertRaisesRegex(V.SplitRuntimeError, "matching SONAME"):
                V.verify_split_runtime(root, elf_reader=wrong_soname)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _runtime(root)
            def relative(path: Path) -> V.ElfIdentity:
                value = _fake_elf(path)
                if path.name == "llama-bench":
                    return V.ElfIdentity(value.soname, value.needed, ("../unsafe",))
                return value
            with self.assertRaisesRegex(V.SplitRuntimeError, "unsealed ELF RUNPATH"):
                V.verify_split_runtime(root, elf_reader=relative)

    def test_wrong_common_chain_target_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _, common = _runtime(root)
            (common / "libggml.so").unlink()
            (common / "libggml.so").symlink_to("libllama.so.0")
            with self.assertRaisesRegex(V.SplitRuntimeError, "topology/ELF identity"):
                V.verify_split_runtime(root, elf_reader=_fake_elf)

    def test_maps_bind_actual_arm_model_kfd_and_process_lifetime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _runtime(root)
            manifest = V.verify_split_runtime(root, elf_reader=_fake_elf)
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"model")
            model_sha = _sha(model)
            anchor = V.verify_runtime_maps(
                manifest, arm="anchor", maps_text=_maps(manifest, "anchor", model),
                model_path=model, model_sha256=model_sha, device_id="mi210_0",
                kfd_pid=123, boot_id="boot-a", process_start_ticks=456)
            same = V.verify_runtime_maps(
                manifest, arm="anchor", maps_text=_maps(manifest, "anchor", model),
                model_path=model, model_sha256=model_sha, device_id="mi210_0",
                kfd_pid=123, boot_id="boot-a", process_start_ticks=456)
            restarted = V.verify_runtime_maps(
                manifest, arm="anchor", maps_text=_maps(manifest, "anchor", model),
                model_path=model, model_sha256=model_sha, device_id="mi210_0",
                kfd_pid=123, boot_id="boot-a", process_start_ticks=457)
            candidate = V.verify_runtime_maps(
                manifest, arm="candidate", maps_text=_maps(manifest, "candidate", model),
                model_path=model, model_sha256=model_sha, device_id="mi210_0",
                kfd_pid=124, boot_id="boot-a", process_start_ticks=458)
            V.validate_arm_pair(anchor, candidate)
        self.assertTrue(anchor.same_resident_process(same))
        self.assertFalse(anchor.same_resident_process(restarted))
        self.assertEqual(anchor.to_dict()["schema"], V.RESIDENCY_SCHEMA)

    def test_maps_refuse_wrong_arm_and_unsealed_local_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            _runtime(root)
            manifest = V.verify_split_runtime(root, elf_reader=_fake_elf)
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"model")
            text = _maps(manifest, "anchor", model) + "\n" + _maps_line(
                (manifest.candidate_hip_dir / "libggml-hip.so.0").resolve())
            with self.assertRaisesRegex(V.SplitRuntimeError, "opposite HIP arm"):
                V.verify_runtime_maps(
                    manifest, arm="anchor", maps_text=text, model_path=model,
                    model_sha256=_sha(model), device_id="mi210_0", kfd_pid=1,
                    boot_id="boot", process_start_ticks=1)

    def test_default_readelf_parser_on_existing_real_build_is_read_only(self) -> None:
        binary = Path(
            "/mnt/raid0/llm/autokernel/worktrees/ak-gpu-q5-onewave-20260813/"
            "build-ak-gpu-q5-onewave/bin/llama-bench")
        if not binary.is_file():
            self.skipTest("read-only real HIP build fixture is absent")
        identity = V.readelf_identity(binary)
        self.assertIn("libllama-bench-impl.so", identity.needed)
        self.assertIsNone(identity.soname)


if __name__ == "__main__":
    unittest.main()
