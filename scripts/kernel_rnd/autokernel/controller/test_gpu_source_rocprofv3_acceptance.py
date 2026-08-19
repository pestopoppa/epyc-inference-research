import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from . import discovery_deployment_factory as F
from . import gpu_source_evidence as E
from . import gpu_source_proofs as P
from . import split_runtime_verifier as V


V3_HEADER = E.ROCPROF_V3_COLUMNS


def v3_row(
        *, correlation: int, kernel: str, group_segment: int,
        workgroup: tuple[int, int, int], grid: tuple[int, int, int],
        begin: int = 100, end: int = 200, kind: str = "KERNEL_DISPATCH",
        agent: int = 4, queue: int = 3) -> dict[str, object]:
    return {
        "Kind": kind, "Agent_Id": agent, "Queue_Id": queue,
        "Kernel_Id": correlation + 10, "Kernel_Name": kernel,
        "Correlation_Id": correlation, "Start_Timestamp": begin,
        "End_Timestamp": end, "Private_Segment_Size": 0,
        "Group_Segment_Size": group_segment,
        "Workgroup_Size_X": workgroup[0],
        "Workgroup_Size_Y": workgroup[1],
        "Workgroup_Size_Z": workgroup[2],
        "Grid_Size_X": grid[0], "Grid_Size_Y": grid[1],
        "Grid_Size_Z": grid[2],
    }


def write_v3(path: Path, rows: list[dict[str, object]],
             *, header: tuple[str, ...] = V3_HEADER) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


class RocprofV3ParserAcceptance(unittest.TestCase):
    """Hardware-free migration gate derived from the governed v13 traces."""

    Q5 = "void mul_mat_vec_q<(ggml_type)6, 1, true, true>(args)"
    Q8 = "void mul_mat_vec_q<(ggml_type)8, 1, true, true>(args)"
    Q8_TAIL = "void mul_mat_vec_q<(ggml_type)8, 1, false, true>(args)"
    FA_TILE = "void flash_attn_tile<64, 64, 2, 1, false>(args)"
    FA_COMBINE = "void flash_attn_combine_results<64>(args)"
    RMS = "void rms_norm_f32<256, true, false>(args)"

    def load(self, rows: list[dict[str, object]], *, expected: int | None = None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace_kernel_trace.csv"
        write_v3(path, rows)
        return E._load_dispatches(
            path, profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
            expected_rows=len(rows) if expected is None else expected)

    def governed_rows(self) -> list[dict[str, object]]:
        # Requested LDS values are the exact rocprofv3 values observed on the
        # governed v13 gfx90a tg128 diagnostic.  The v1 route contract uses the
        # 512-byte allocation-rounded values asserted below.
        return [
            v3_row(correlation=1, kernel=self.Q5, group_segment=1024,
                   workgroup=(128, 1, 1), grid=(57344, 1, 1)),
            v3_row(correlation=2, kernel=self.Q8, group_segment=6144,
                   workgroup=(64, 4, 1), grid=(2048, 4, 1)),
            v3_row(correlation=3, kernel=self.Q8_TAIL, group_segment=3072,
                   workgroup=(64, 4, 1), grid=(2430976, 4, 1)),
            v3_row(correlation=4, kernel=self.FA_TILE, group_segment=4992,
                   workgroup=(64, 1, 1), grid=(7168, 1, 1)),
            v3_row(correlation=5, kernel=self.FA_COMBINE, group_segment=64,
                   workgroup=(64, 1, 1), grid=(896, 1, 1)),
            v3_row(correlation=6, kernel=self.RMS, group_segment=128,
                   workgroup=(256, 1, 1), grid=(256, 1, 1)),
        ]

    def test_gfx90a_requested_lds_is_mapped_to_v1_allocation_authority(self):
        rows = self.load(self.governed_rows())
        self.assertEqual(
            [row["group_segment_size"] for row in rows],
            [1024, 6144, 3072, 4992, 64, 128])
        self.assertEqual(
            [row["lds"] for row in rows],
            [1024, 6144, 3072, 5120, 512, 512])
        self.assertEqual(
            [(row["grid"], row["workgroup"]) for row in rows],
            [(57344, 128), (8192, 256), (9723904, 256),
             (7168, 64), (896, 64), (256, 256)])

    def test_full_structural_fingerprint_is_order_independent(self):
        rows = self.load(self.governed_rows())
        self.assertEqual(
            E._profiler_structural_fingerprint(rows),
            E._profiler_structural_fingerprint(list(reversed(rows))))
        mutated = [dict(row) for row in rows]
        mutated[0]["grid_xyz"] = [57345, 1, 1]
        self.assertNotEqual(
            E._profiler_structural_fingerprint(rows),
            E._profiler_structural_fingerprint(mutated))

    def test_exact_dispatch_count_is_not_learned_from_the_trace(self):
        with self.assertRaisesRegex(E.EvidenceProducerError, "exact expected row count"):
            self.load(self.governed_rows(), expected=59925)

    def test_duplicate_correlation_refuses(self):
        rows = self.governed_rows()
        rows[1]["Correlation_Id"] = rows[0]["Correlation_Id"]
        with self.assertRaisesRegex(E.EvidenceProducerError, "correlation IDs"):
            self.load(rows)

    def test_correlation_ids_must_be_the_complete_contiguous_dispatch_domain(self):
        rows = self.governed_rows()
        for index, row in enumerate(rows, start=2):
            row["Correlation_Id"] = index
        with self.assertRaisesRegex(E.EvidenceProducerError, "correlation.*contiguous"):
            self.load(rows)

    def test_non_dispatch_record_refuses(self):
        rows = self.governed_rows()
        rows[0]["Kind"] = "MEMORY_COPY"
        with self.assertRaisesRegex(E.EvidenceProducerError, "non-dispatch"):
            self.load(rows)

    def test_header_must_be_exact_and_ordered(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "trace_kernel_trace.csv"
            reordered = (V3_HEADER[1], V3_HEADER[0], *V3_HEADER[2:])
            write_v3(path, self.governed_rows(), header=reordered)
            with self.assertRaisesRegex(E.EvidenceProducerError, "exact rocprofv3 columns"):
                E._load_dispatches(
                    path, profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
                    expected_rows=6)

    def test_invalid_duration_and_nonintegral_geometry_refuse(self):
        rows = self.governed_rows()
        rows[0]["End_Timestamp"] = rows[0]["Start_Timestamp"]
        with self.assertRaisesRegex(E.EvidenceProducerError, "invalid duration"):
            self.load(rows)

    def test_governed_tg128_trace_has_exact_q5_q8_fa_rms_route_authority(self):
        path = Path(
            "/mnt/raid0/llm/autokernel/diagnostics/"
            "v13-rocprofv3-tg128-20260819/raw/v13_sdk_kernel_trace.csv")
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "a018347428ab50672d3ce87b3c02c2689645b55e12e62b16a761cad24657d1dc")
        rows = E._load_dispatches(
            path, profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
            expected_rows=59_925)

        def family(kernel: str) -> str | None:
            for label, prefix in (
                ("q5", "void mul_mat_vec_q<(ggml_type)6,"),
                ("q8", "void mul_mat_vec_q<(ggml_type)8,"),
                ("quantize_q8", "quantize_q8_1("),
                ("fa_tile", "void flash_attn_tile<64, 64, 2, 1, false>("),
                ("fa_combine", "void flash_attn_combine_results<64>("),
                ("rms", "void rms_norm_f32<256, true, false>("),
            ):
                if kernel.startswith(prefix):
                    return label
            return None

        actual = Counter(
            (label, row["grid"], row["workgroup"], row["lds"])
            for row in rows if (label := family(str(row["kernel"]))) is not None)
        self.assertEqual(actual, Counter({
            ("q5", 57344, 128, 1024): 6063,
            ("q5", 8192, 128, 1024): 4644,
            ("q5", 311296, 128, 1024): 3096,
            ("q5", 57344, 128, 512): 129,
            ("quantize_q8", 1024, 256, 0): 15609,
            ("quantize_q8", 5120, 256, 0): 3096,
            ("q8", 8192, 256, 6144): 1548,
            ("q8", 9723904, 256, 3072): 129,
            ("fa_tile", 7168, 64, 5120): 3096,
            ("fa_combine", 896, 64, 512): 3096,
            ("rms", 256, 256, 512): 6321,
        }))
        rows = self.governed_rows()
        rows[0]["Grid_Size_X"] = 57345
        with self.assertRaisesRegex(E.EvidenceProducerError, "non-integral blocks"):
            self.load(rows)

    def test_agent_info_binds_trace_agent_to_exact_gfx90a_mi210(self):
        path = Path(
            "/mnt/raid0/llm/autokernel/diagnostics/"
            "v13-rocprofv3-tg128-20260819/raw/v13_sdk_agent_info.csv")
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "50189a58f15ffb0008e840a8a6d18db1a88f73e3492b686b167d773de6b9323e")
        identity = E._load_rocprofv3_agent_info(path, trace_agent_ids={4})
        self.assertEqual((identity["agent_id"], identity["name"],
                          identity["product_name"]),
                         (4, "gfx90a", "AMD Instinct MI210"))
        with self.assertRaisesRegex(E.EvidenceProducerError, "gfx90a MI210"):
            E._load_rocprofv3_agent_info(path, trace_agent_ids={5})


class RocprofV3ClosureAcceptance(unittest.TestCase):
    def bound_manifest(self, root: Path, role: str) -> E.BoundInputFile:
        body = E.profiler_prefix_snapshot(root)
        path = root.parent / f"{role}.json"
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        path.write_bytes(raw)
        return E.BoundInputFile(role, path.resolve(), hashlib.sha256(raw).hexdigest())

    def test_runtime_manifest_cannot_bind_an_empty_unrelated_prefix(self):
        with tempfile.TemporaryDirectory() as raw:
            root = (Path(raw) / "empty-sdk").resolve()
            root.mkdir()
            bound = self.bound_manifest(root, "profiler_runtime_manifest")
            with self.assertRaisesRegex(E.EvidenceProducerError,
                                        "rocprofv3.*closure"):
                E._verify_profiler_runtime_manifest(bound)

    def test_dependency_manifest_must_bind_the_aql_profile_dso(self):
        with tempfile.TemporaryDirectory() as raw:
            root = (Path(raw) / "empty-dependency").resolve()
            root.mkdir()
            bound = self.bound_manifest(root, "profiler_aqlprofile_manifest")
            with self.assertRaisesRegex(E.EvidenceProducerError,
                                        "aqlprofile"):
                E._verify_profiler_runtime_manifest(bound)

    def test_prefix_snapshot_refuses_external_symlink_escape(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prefix"
            root.mkdir()
            outside = Path(raw) / "outside.so"
            outside.write_bytes(b"mutable external dependency")
            (root / "lib.so").symlink_to(outside)
            with self.assertRaisesRegex(E.EvidenceProducerError, "symlink.*escape"):
                E.profiler_prefix_snapshot(root.resolve())

    def test_prefix_snapshot_refuses_special_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prefix"
            root.mkdir()
            os.mkfifo(root / "unexpected.fifo")
            with self.assertRaisesRegex(E.EvidenceProducerError, "special file"):
                E.profiler_prefix_snapshot(root.resolve())

    def test_runtime_maps_refuses_path_replaced_after_object_was_mapped(self):
        """A current pathname hash must not stand in for the mapped inode.

        Linux retains the mapped inode after unlink and reports the old mapping
        as ``PATH (deleted)``.  Recreating PATH with the sealed bytes must not
        let that different inode satisfy required_mapped_files.
        """
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            common = root / "common"
            anchor = root / "anchor-hip"
            candidate = root / "candidate-hip"
            for directory in (common, anchor, candidate):
                directory.mkdir()
            reward = common / "llama-bench"
            reward.write_bytes(b"reward")
            anchor_hip = anchor / "libggml-hip.so.0"
            candidate_hip = candidate / "libggml-hip.so.0"
            anchor_hip.write_bytes(b"anchor")
            candidate_hip.write_bytes(b"candidate")
            model = root / "model.gguf"
            model.write_bytes(b"model")
            profiler = root / "librocprofiler-sdk.so.0.4.0"
            profiler.write_bytes(b"old mapped bytes")
            mapped_stat = profiler.stat()
            # Model the exact interval between reading /proc/PID/maps and
            # hashing PATH: the sampled line still names the original inode,
            # while PATH now resolves to a sealed-byte replacement.
            profiler.unlink()
            profiler.write_bytes(b"sealed replacement bytes")

            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = V.SplitRuntimeManifest(
                root=root, common_dir=common, anchor_hip_dir=anchor,
                candidate_hip_dir=candidate, reward_binary=reward,
                common_files=(V.RuntimeFile(
                    name=reward.name, kind="file", sha256=digest(reward)),),
                anchor_hip_files=(), candidate_hip_files=(),
                anchor_hip_sha256=digest(anchor_hip),
                candidate_hip_sha256=digest(candidate_hip),
                manifest_sha256="a" * 64)

            def mapped(path: Path, inode: int, *, deleted: bool = False,
                       device: int | None = None) -> str:
                suffix = " (deleted)" if deleted else ""
                dev = os.makedev(0, 86) if device is None else device
                return (f"7f000000-7f001000 r-xp 00000000 "
                        f"{os.major(dev):02x}:{os.minor(dev):02x} {inode} "
                        f"{path}{suffix}")

            maps_text = "\n".join((
                mapped(reward, 1), mapped(candidate_hip, 2), mapped(model, 3),
                # The cached maps image is authentic but PATH was replaced
                # before verification; hashing PATH alone proves the wrong
                # inode even without relying on the `(deleted)` marker.
                mapped(profiler, mapped_stat.st_ino,
                       device=mapped_stat.st_dev),
            ))
            with self.assertRaisesRegex(V.SplitRuntimeError, "mapped inode|deleted"):
                V.verify_runtime_maps(
                    manifest, arm="candidate", maps_text=maps_text,
                    model_path=model, model_sha256=digest(model),
                    device_id="mi210_0", kfd_pid=1, boot_id="boot",
                    process_start_ticks=1,
                    required_mapped_files={str(profiler): digest(profiler)})

    def test_runtime_maps_binds_model_digest_to_the_mapped_inode(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            common = root / "common"
            anchor = root / "anchor-hip"
            candidate = root / "candidate-hip"
            for directory in (common, anchor, candidate):
                directory.mkdir()
            reward = common / "llama-bench"
            reward.write_bytes(b"reward")
            anchor_hip = anchor / "libggml-hip.so.0"
            candidate_hip = candidate / "libggml-hip.so.0"
            anchor_hip.write_bytes(b"anchor")
            candidate_hip.write_bytes(b"candidate")
            model = root / "model.gguf"
            model.write_bytes(b"old mapped model bytes")
            mapped_model_handle = model.open("rb")
            self.addCleanup(mapped_model_handle.close)
            mapped_model_stat = model.stat()
            model.unlink()
            model.write_bytes(b"sealed replacement model bytes")

            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = V.SplitRuntimeManifest(
                root=root, common_dir=common, anchor_hip_dir=anchor,
                candidate_hip_dir=candidate, reward_binary=reward,
                common_files=(V.RuntimeFile(
                    name=reward.name, kind="file", sha256=digest(reward)),),
                anchor_hip_files=(), candidate_hip_files=(),
                anchor_hip_sha256=digest(anchor_hip),
                candidate_hip_sha256=digest(candidate_hip),
                manifest_sha256="a" * 64)

            def mapped(path: Path, stat: os.stat_result) -> str:
                return (f"7f000000-7f001000 r-xp 00000000 "
                        f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x} "
                        f"{stat.st_ino} {path}")

            maps_text = "\n".join((
                mapped(reward, reward.stat()),
                mapped(candidate_hip, candidate_hip.stat()),
                mapped(model, mapped_model_stat),
            ))
            with self.assertRaisesRegex(V.SplitRuntimeError, "mapped inode"):
                V.verify_runtime_maps(
                    manifest, arm="candidate", maps_text=maps_text,
                    model_path=model, model_sha256=digest(model),
                    device_id="mi210_0", kfd_pid=1, boot_id="boot",
                    process_start_ticks=1)


class RocprofV3GovernedMapsAcceptance(unittest.TestCase):
    """Bind the residency contract to both governed tg128 map captures."""

    DIAGNOSTICS = Path("/mnt/raid0/llm/autokernel/diagnostics")
    ARMS = (
        ("v13-rocprofv3-tg128-20260819", "candidate"),
        ("v13-rocprofv3-anchor-tg128-20260819", "anchor"),
    )

    @staticmethod
    def file_sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def required_profiler_mappings() -> dict[str, str]:
        return {str(path): digest for path, digest in (
            (F._ROCPROF_V3_SDK_LIB, F._ROCPROF_V3_SDK_LIB_SHA256),
            (F._ROCPROF_V3_TOOL_LIB, F._ROCPROF_V3_TOOL_LIB_SHA256),
            (F._ROCPROF_V3_AQL_LIB, F._ROCPROF_V3_AQL_LIB_SHA256),
            (F._ROCPROF_V3_HSA_LIB, F._ROCPROF_V3_HSA_LIB_SHA256),
            (F._ROCPROF_V3_REGISTER_LIB, F._ROCPROF_V3_REGISTER_LIB_SHA256),
        )}

    def test_governed_maps_complete_without_libpci_and_refuse_missing_required_dso(
            self) -> None:
        model = F._SITE_MODEL.resolve(strict=True)
        model_sha = self.file_sha(model)
        required = self.required_profiler_mappings()
        self.assertNotIn(str(F._ROCPROF_V3_PCI_LIB), required)
        self.assertNotIn("profiler_libpci_library", E.PROFILER_MAPPED_ROLES)
        self.assertIn("profiler_libpci_library", E.PROFILER_V3_INPUT_ROLES)

        for diagnostic_name, arm in self.ARMS:
            with self.subTest(arm=arm):
                root = self.DIAGNOSTICS / diagnostic_name
                result = json.loads((root / "result.json").read_text())
                argv = result["argv"]
                binary = Path(argv[argv.index("/usr/bin/taskset") + 3])
                manifest = V.verify_split_runtime(binary.parent.parent)
                samples = json.loads((root / "samples.json").read_text())

                # Sample 3 carries an already-vanished /tmp/comgr linked.bc.
                # It is unrelated to the sealed runtime and must not prevent a
                # complete classification for either arm.
                sample3_pid, sample3_maps = next(iter(samples[3]["maps"].items()))
                self.assertIn("/tmp/comgr-", sample3_maps)
                comgr_path = next(
                    Path(line.split(maxsplit=5)[5])
                    for line in sample3_maps.splitlines()
                    if "/tmp/comgr-" in line)
                self.assertFalse(comgr_path.exists())
                V.verify_runtime_maps(
                    manifest, arm=arm, maps_text=sample3_maps,
                    model_path=model, model_sha256=model_sha,
                    device_id="mi210_0", kfd_pid=int(sample3_pid),
                    boot_id="governed-fixture", process_start_ticks=1,
                    required_mapped_files=required)

                complete = 0
                for sample_index in (4, 5, 6):
                    pid, maps_text = next(iter(samples[sample_index]["maps"].items()))
                    self.assertNotIn("libpciaccess", maps_text)
                    V.verify_runtime_maps(
                        manifest, arm=arm, maps_text=maps_text,
                        model_path=model, model_sha256=model_sha,
                        device_id="mi210_0", kfd_pid=int(pid),
                        boot_id="governed-fixture", process_start_ticks=1,
                        required_mapped_files=required)
                    complete += 1
                self.assertGreaterEqual(complete, 2)

                missing_path = str(F._ROCPROF_V3_SDK_LIB)
                pid, maps_text = next(iter(samples[4]["maps"].items()))
                without_required = "\n".join(
                    line for line in maps_text.splitlines()
                    if missing_path not in line)
                with self.assertRaisesRegex(
                        V.RuntimeMapsIncomplete,
                        "omit required profiler objects"):
                    V.verify_runtime_maps(
                        manifest, arm=arm, maps_text=without_required,
                        model_path=model, model_sha256=model_sha,
                        device_id="mi210_0", kfd_pid=int(pid),
                        boot_id="governed-fixture", process_start_ticks=1,
                        required_mapped_files=required)


class RocprofV3RawClosureAcceptance(unittest.TestCase):
    REAL_V14_RAW = Path(
        "/mnt/raid0/llm/autokernel/deployments/"
        "gpu-discovery-quant-ladder-occupancy-v14/operations/"
        "0846800a4806aeb98cc3d5ce60fcf2b2426264cb8afc8f8818a142302c7efda8/"
        "proof/attribution-candidate/attempt-01/raw")

    @staticmethod
    def invocation(raw: Path) -> E.CommandInvocation:
        return E.CommandInvocation(
            kind="rocprof", arm="candidate", argv=("rocprofv3",),
            stdout_path=raw / "stdout.txt",
            stderr_path=raw / "stderr.txt",
            timestamp_csv_path=raw / "trace_kernel_trace.csv",
            working_directory=raw.parent.resolve(strict=True),
            environment=(("LD_LIBRARY_PATH", "/opt/rocm/lib"),))

    @staticmethod
    def synthetic_raw(root: Path) -> Path:
        raw = root / "raw"
        raw.mkdir(parents=True)
        (raw / ".rocprofv3").mkdir()
        (raw / "stdout.txt").write_bytes(b"[{\"n_gen\": 128}]\n")
        (raw / "stderr.txt").write_bytes(b"")
        (raw / "trace_kernel_trace.csv").write_bytes(b"header\nrow\n")
        (raw / "trace_agent_info.csv").write_bytes(b"header\nagent\n")
        return raw

    def test_exact_v14_raw_closure_seals_and_reopens_without_mutation(self) -> None:
        raw = self.REAL_V14_RAW.resolve(strict=True)
        self.assertEqual(set(path.name for path in raw.iterdir()), {
            ".rocprofv3", "stdout.txt", "stderr.txt",
            "trace_kernel_trace.csv", "trace_agent_info.csv"})
        self.assertEqual(list((raw / ".rocprofv3").iterdir()), [])
        invocation = self.invocation(raw)
        sealed = E._rocprofv3_raw_artifacts(invocation)
        reopened = E._rocprofv3_raw_artifacts(invocation)
        self.assertEqual(sealed, reopened)
        self.assertEqual(len(sealed), 5)
        bookkeeping = sealed[0]
        self.assertEqual(bookkeeping["name"], ".rocprofv3")
        self.assertEqual(bookkeeping["kind"],
                         "profiler_bookkeeping_directory")
        self.assertFalse(bookkeeping["scientific_evidence"])
        self.assertEqual(bookkeeping["mode"], "0755")
        self.assertEqual(bookkeeping["links"], 2)
        self.assertEqual(bookkeeping["entries"], 0)
        self.assertEqual(
            bookkeeping["metadata_sha256"],
            E.schemas.content_hash({
                key: value for key, value in bookkeeping.items()
                if key != "metadata_sha256"}))
        self.assertEqual({
            row["name"]: row["sha256"] for row in sealed
            if row["kind"] == "regular_file"}, {
            "stdout.txt":
                "e2afec30810f5c47cc6150d9f40f9553eed0fbdb25fa80d4eac3335d2992280e",
            "stderr.txt":
                "7bd641d8844ab4a2c46754ef9a8b8b55e7ad28a27533a653806b19bdda86afec",
            "trace_agent_info.csv":
                "50189a58f15ffb0008e840a8a6d18db1a88f73e3492b686b167d773de6b9323e",
            "trace_kernel_trace.csv":
                "e1012fbd79f638d8ca39ecf7f92deb7755e846db3b8a3487683f9a733164497e",
        })

    def test_raw_closure_refuses_every_unreviewed_topology(self) -> None:
        mutations = (
            "missing_metadata", "metadata_file", "metadata_symlink_escape",
            "metadata_file_content", "metadata_directory_content",
            "metadata_fifo", "metadata_writable", "metadata_mount",
            "extra_file", "extra_directory", "stdout_symlink_escape",
            "stdout_hardlink", "stderr_fifo",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                raw = self.synthetic_raw(root)
                metadata = raw / ".rocprofv3"
                outside = root / "outside"
                if mutation == "missing_metadata":
                    metadata.rmdir()
                elif mutation == "metadata_file":
                    metadata.rmdir(); metadata.write_bytes(b"")
                elif mutation == "metadata_symlink_escape":
                    metadata.rmdir(); outside.mkdir()
                    metadata.symlink_to(outside, target_is_directory=True)
                elif mutation == "metadata_file_content":
                    (metadata / "state.json").write_bytes(b"{}")
                elif mutation == "metadata_directory_content":
                    (metadata / "nested").mkdir()
                elif mutation == "metadata_fifo":
                    metadata.rmdir(); os.mkfifo(metadata)
                elif mutation == "metadata_writable":
                    metadata.chmod(0o777)
                elif mutation == "metadata_mount":
                    with mock.patch.object(E.os.path, "ismount", return_value=True):
                        with self.assertRaises(E.EvidenceProducerError):
                            E._rocprofv3_raw_artifacts(self.invocation(raw))
                    continue
                elif mutation == "extra_file":
                    (raw / "extra.txt").write_bytes(b"extra")
                elif mutation == "extra_directory":
                    (raw / "extra").mkdir()
                elif mutation == "stdout_symlink_escape":
                    (raw / "stdout.txt").unlink(); outside.write_bytes(b"stdout")
                    (raw / "stdout.txt").symlink_to(outside)
                elif mutation == "stdout_hardlink":
                    os.link(raw / "stdout.txt", outside)
                elif mutation == "stderr_fifo":
                    (raw / "stderr.txt").unlink(); os.mkfifo(raw / "stderr.txt")
                with self.assertRaises(E.EvidenceProducerError):
                    E._rocprofv3_raw_artifacts(self.invocation(raw))

    def test_raw_root_symlink_escape_and_reopen_byte_mutation_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = self.synthetic_raw(root / "real-parent")
            escaped = root / "escaped-parent"; escaped.mkdir()
            link = escaped / "raw"; link.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "real contained path"):
                E._rocprofv3_raw_artifacts(self.invocation(link))

        with tempfile.TemporaryDirectory() as temporary:
            raw = self.synthetic_raw(Path(temporary).resolve())
            invocation = self.invocation(raw)
            sealed = E._rocprofv3_raw_artifacts(invocation)
            (raw / "stdout.txt").write_bytes(b"mutated after receipt sealing")
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "changed after sealing"):
                E._revalidate_rocprofv3_raw_artifacts(invocation, sealed)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            raw = self.synthetic_raw(root)
            invocation = self.invocation(raw)
            sealed = E._rocprofv3_raw_artifacts(invocation)
            original = raw / ".rocprofv3"
            retained = root / "retained-original-metadata"
            original.rename(retained)
            original.mkdir()
            with self.assertRaisesRegex(
                    E.EvidenceProducerError, "changed after sealing"):
                E._revalidate_rocprofv3_raw_artifacts(invocation, sealed)


class RocprofV3FactoryAcceptance(unittest.TestCase):
    def test_factory_builds_only_the_exact_v3_plan_without_hardware(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            operations = root / "operations"
            operation_key = "d" * 64
            (operations / operation_key).mkdir(parents=True, mode=0o700)
            candidate_build = root / "candidate-build"
            anchor_build = root / "anchor-build"
            candidate_build.mkdir(); anchor_build.mkdir()

            def bound(role: str, path: Path, body: bytes,
                      *, executable: bool = False) -> E.BoundInputFile:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                if executable:
                    path.chmod(0o755)
                return E.BoundInputFile(
                    role, path.resolve(), hashlib.sha256(body).hexdigest())

            candidate_hip = bound(
                "hip_library", root / "candidate-runtime/libggml-hip.so.0",
                b"candidate hip")
            anchor_hip = bound(
                "hip_library", root / "anchor-runtime/libggml-hip.so.0",
                b"anchor hip")
            reward = bound(
                "measurement_binary", root / "common/llama-bench",
                b"shared reward", executable=True)
            runtime_receipt = bound(
                "runtime_receipt", root / "runtime-receipt.json", b"{}")
            shared = E.SharedRewardRuntimeFiles(
                reward, runtime_receipt, anchor_hip, candidate_hip)

            def build_files(label: str, hip: E.BoundInputFile):
                return E.BuildIdentityFiles(
                    source_identity=bound(
                        "source_identity", root / f"{label}-source.json", label.encode()),
                    binary=bound(
                        "binary", root / f"{label}-binary", f"{label} binary".encode()),
                    hip_library=hip,
                    config=bound(
                        "config", root / f"{label}-config", f"{label} config".encode()),
                    linkage=bound(
                        "linkage", root / f"{label}-linkage", f"{label} linkage".encode()))

            candidate_files = build_files("candidate", candidate_hip)
            anchor_files = build_files("anchor", anchor_hip)
            manifest = bound("manifest", root / "manifest.json", b"manifest")
            model = bound("model", root / "model.gguf", b"model")
            workload = bound("workload", root / "workload.json", b"workload")
            runtime_config = bound(
                "runtime_config", root / "runtime-config.json", b"runtime config")
            materialization = bound(
                "materialization", root / "materialization.json", b"materialization")
            identities = E.EvidenceIdentityFiles(
                candidate=candidate_files, anchor=anchor_files,
                manifest=manifest, model=model, workload=workload,
                runtime_config=runtime_config, materialization=materialization,
                shared_runtime=shared)
            candidate_identity = P.BuildIdentity(
                "candidate", "1" * 64, candidate_files.binary.sha256,
                candidate_hip.sha256, candidate_files.config.sha256,
                candidate_files.linkage.sha256)
            anchor_identity = P.BuildIdentity(
                "anchor", "2" * 64, anchor_files.binary.sha256,
                anchor_hip.sha256, anchor_files.config.sha256,
                anchor_files.linkage.sha256)
            correctness = bound(
                "executable", root / "candidate-build/bin/test-backend-ops",
                b"correctness", executable=True)
            capability = bound(
                "correctness_capability", root / "correctness-capability.json",
                b"capability")
            dispatch = E.DispatchContract(
                candidate_exact=(
                    E.ExactDispatch(
                        "candidate.bulk", r"^kernel_bulk$", 3096,
                        3072, 64, 5120, 48),
                    E.ExactDispatch(
                        "candidate.tail", r"^kernel_tail$", 3096,
                        1024, 64, 5120, 16),
                ),
                anchor_exact=(E.ExactDispatch(
                    "anchor.route", r"^kernel_anchor$", 3096,
                    7168, 64, 5120, 112),))
            template = SimpleNamespace(
                template_id="cuda-fattn-tile-v1",
                semantics={
                    "correctness_op": "FLASH_ATTN_EXT",
                    "expected_correctness_cases": 1, "suite_seed": 7,
                    "required_correctness_cases": [{"case_id": "odd-gqa7"}],
                },
                bind_dispatch=lambda _intent: dispatch)
            candidate = SimpleNamespace(
                source_manifest=SimpleNamespace(campaign_id="ak-test"),
                source_manifest_sha256=manifest.sha256,
                experiment_intent=SimpleNamespace())
            build = SimpleNamespace(
                materialization_receipt=materialization.path,
                operation_key=operation_key,
                candidate_identity=candidate_identity,
                anchor_identity=anchor_identity,
                candidate_build=candidate_build,
                anchor_build=anchor_build)
            config = SimpleNamespace(
                operations_root=operations, config_sha256="3" * 64,
                device_id="mi210_0", model=model, workload=workload,
                runtime_config=runtime_config)

            def write_carrier(_config, _key, name, body, _label):
                path = operations / operation_key / name
                path.write_bytes(body)
                return path

            with mock.patch.object(
                    F.discovery_static_registry, "evidence_identity_files_for_build",
                    return_value=identities), mock.patch.object(
                    F.discovery_static_registry, "correctness_capability_files_for_build",
                    return_value=(correctness, capability)), mock.patch.object(
                    F, "_manifest_file", return_value=manifest), mock.patch.object(
                    F, "_operation_carrier_root",
                    return_value=operations / operation_key), mock.patch.object(
                    F, "_write_operation_carrier", side_effect=write_carrier):
                plan = F._evidence_binding(config).build(
                    candidate, build, template, 1)

            exact_prefix = (
                str(F._ROCPROF_V3_PYTHON.resolve(strict=True)),
                str(F._ROCPROF_V3.resolve(strict=True)),
                "--kernel-trace", "-d", E.ROCPROF_OUTPUT_DIRECTORY,
                "-o", E.ROCPROF_OUTPUT_BASENAME,
                "--output-format", "csv", "--")
            self.assertEqual(plan.candidate_rocprof_argv[:len(exact_prefix)], exact_prefix)
            self.assertEqual(plan.anchor_rocprof_argv, plan.candidate_rocprof_argv)
            self.assertNotIn(E.ROCPROF_TIMESTAMP_OUTPUT, plan.candidate_rocprof_argv)
            self.assertNotIn("-i", plan.candidate_rocprof_argv)
            self.assertEqual(plan.profiler_trace_schema_id, E.ROCPROF_V3_TRACE_ID)
            self.assertEqual(
                (plan.expected_candidate_profiler_dispatch_rows,
                 plan.expected_anchor_profiler_dispatch_rows), (63_021, 59_925))
            self.assertEqual(plan.profiler_transport_policy,
                             E.ROCPROF_V3_TRANSPORT_POLICY)
            for inputs in (plan.candidate_rocprof_inputs,
                           plan.anchor_rocprof_inputs):
                roles = {item.role for item in inputs}
                self.assertTrue(E.PROFILER_MAPPED_ROLES.issubset(roles))
                self.assertIn("profiler_libpci_library", roles)
                self.assertNotIn(
                    "profiler_libpci_library", E.PROFILER_MAPPED_ROLES)
                self.assertNotIn("timestamp_input", roles)
                for item in inputs:
                    if item.role.endswith("_manifest"):
                        E._verify_profiler_runtime_manifest(item)
            candidate_ld = dict(plan.candidate_rocprof_environment)["LD_LIBRARY_PATH"]
            anchor_ld = dict(plan.anchor_rocprof_environment)["LD_LIBRARY_PATH"]
            self.assertTrue(candidate_ld.startswith(str(candidate_hip.path.parent) + ":"))
            self.assertTrue(anchor_ld.startswith(str(anchor_hip.path.parent) + ":"))
            for required in (F._ROCPROF_V3_SDK / "lib", F._ROCPROF_V3_OLD_LIB,
                             F._ROCPROF_V3_PCI_LIB_DIR, Path("/opt/rocm/lib")):
                self.assertIn(str(required), candidate_ld.split(":"))
                self.assertIn(str(required), anchor_ld.split(":"))
            self.assertFalse(hasattr(F, "_rocprof_v1_policy"))
            self.assertNotIn("_rocprof_v1_policy", F._evidence_binding.__code__.co_names)

            anchor_rows = E._load_dispatches(
                F._PROFILE_V3_TRACE_CSV,
                profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
                expected_rows=plan.expected_anchor_profiler_dispatch_rows)
            self.assertEqual(len(anchor_rows), 59_925)
            with self.assertRaisesRegex(E.EvidenceProducerError,
                                        "exact expected row count"):
                E._load_dispatches(
                    F._PROFILE_V3_TRACE_CSV,
                    profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
                    expected_rows=plan.expected_candidate_profiler_dispatch_rows)

            synthetic_candidate = root / "candidate_kernel_trace.csv"
            write_v3(synthetic_candidate, (
                v3_row(correlation=index, kernel="kernel_bulk",
                       group_segment=5120, workgroup=(64, 1, 1),
                       grid=(3072, 1, 1))
                for index in range(1, 63_022)))
            candidate_rows = E._load_dispatches(
                synthetic_candidate,
                profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
                expected_rows=plan.expected_candidate_profiler_dispatch_rows)
            self.assertEqual(len(candidate_rows), 63_021)
            with self.assertRaisesRegex(E.EvidenceProducerError,
                                        "exact expected row count"):
                E._load_dispatches(
                    synthetic_candidate,
                    profiler_trace_schema_id=E.ROCPROF_V3_TRACE_ID,
                    expected_rows=plan.expected_anchor_profiler_dispatch_rows)


if __name__ == "__main__":
    unittest.main()
