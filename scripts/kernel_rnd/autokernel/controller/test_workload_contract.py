"""The screening workload must dispatch the kernels production dispatches.

The defect: `Qwen2.5-Coder-0.5B-Q4_K_M.gguf` revalidated cleanly for a month while
being 132x Q5_0 and 12x Q4_K. n_embd=896 is not divisible by the 256-element K-quant
superblock, so llama.cpp fell back silently. The loop optimised `mul_mat_vec_q<Q5_0>`
-- a path production never dispatches -- and its flagship hypothesis was
`akh-v2-q5-type-specific-dequant`.

The filename said Q4_K_M. Only the tensor table said otherwise.
"""
import struct
import unittest
from pathlib import Path

from autokernel.controller import workload_contract as wc


def _write_gguf(path: Path, *, architecture: str, n_embd: int,
                tensors: dict[int, int]) -> Path:
    """A minimal but real GGUF header: metadata KVs then the tensor table."""
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", sum(tensors.values()))
    out += struct.pack("<Q", 2)

    def kv_string(key: str, value: str) -> bytes:
        blob = struct.pack("<Q", len(key)) + key.encode()
        blob += struct.pack("<I", 8)
        blob += struct.pack("<Q", len(value)) + value.encode()
        return blob

    def kv_uint32(key: str, value: int) -> bytes:
        blob = struct.pack("<Q", len(key)) + key.encode()
        blob += struct.pack("<I", 4) + struct.pack("<I", value)
        return blob

    out += kv_string("general.architecture", architecture)
    out += kv_uint32(f"{architecture}.embedding_length", n_embd)

    index = 0
    for type_id, count in tensors.items():
        for _ in range(count):
            name = f"blk.{index}.weight".encode()
            index += 1
            out += struct.pack("<Q", len(name)) + name
            out += struct.pack("<I", 1) + struct.pack("<Q", 4096)
            out += struct.pack("<I", type_id) + struct.pack("<Q", 0)

    path.write_bytes(bytes(out))
    return path


Q4_K, Q6_K, Q5_0, Q4_1, Q8_0, F32, F16 = 12, 14, 6, 3, 8, 0, 1


def _census(tmp: Path, name: str, *, architecture: str = "qwen35",
            n_embd: int = 5120, tensors: dict[int, int]) -> "wc.WorkloadCensus":
    """A censused fixture model. The default shape is production's own
    (Qwen3.8-27B-Q8_0: arch qwen35, n_embd 5120), so the family checks below run
    against a FIXTURE reference and never depend on the real 29 GB file."""
    return wc.read_census(_write_gguf(Path(tmp) / name, architecture=architecture,
                                      n_embd=n_embd, tensors=tensors))


def _production(tmp: Path) -> "wc.WorkloadCensus":
    """The declared production model's shape, as a fixture: Q8_0-dominant, 5120."""
    return _census(tmp, "Qwen3.8-27B-Q8_0.gguf", tensors={Q8_0: 506, F32: 360})


class Census(unittest.TestCase):

    def test_it_reads_the_tensor_table_not_the_filename(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Named like a K-quant, built like a legacy quant. The real defect.
            path = _write_gguf(Path(tmp) / "Model-Q4_K_M.gguf", architecture="qwen2",
                               n_embd=896, tensors={Q5_0: 132, F32: 121, Q4_K: 12})
            census = wc.read_census(path)
            self.assertEqual(census.dominant_quant, "Q5_0")
            self.assertEqual(census.n_embd, 896)
            self.assertFalse(census.superblock_compatible)

    def test_float_tensors_never_count_as_the_dominant_quant(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "m.gguf", architecture="qwen3",
                               n_embd=1024, tensors={F32: 500, Q4_K: 10})
            self.assertEqual(wc.read_census(path).dominant_quant, "Q4_K")


class Gate(unittest.TestCase):
    """`verify_workload` against a FIXTURE production reference (Q8_0-dominant,
    the 2026-08-14 cutover's shape) -- never the real 29 GB file."""

    def test_the_real_defect_is_refused(self):
        """896 / 256 = 3.5. This is the run that cost a month."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "Qwen2.5-Coder-0.5B-Q4_K_M.gguf",
                               architecture="qwen2", n_embd=896,
                               tensors={Q5_0: 132, F32: 121, Q8_0: 13, Q6_K: 12})
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.verify_workload(path, production=_production(tmp))
            self.assertIn("not divisible by 256", str(caught.exception))

    def test_a_superblock_workload_is_accepted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "ok.gguf", architecture="qwen2",
                               n_embd=1536, tensors={Q4_K: 169, F32: 141, Q6_K: 29})
            census = wc.verify_workload(path, production=_production(tmp))
            self.assertEqual(census.dominant_quant, "Q4_K")

    def test_q8_0_is_accepted_when_production_is_q8_0_dominant(self):
        """THE §5.1 bug fix: the stale hard-coded family refused production's own
        model after the 2026-08-14 Q8_0 cutover. Censused, it cannot drift."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "confirm.gguf", architecture="qwen35",
                               n_embd=5120, tensors={Q8_0: 506, F32: 360})
            census = wc.verify_workload(path, production=_production(tmp))
            self.assertEqual(census.dominant_quant, "Q8_0")

    def test_a_divisible_model_dominated_by_a_legacy_quant_is_still_refused(self):
        """The two failure modes are independent -- and a Q8_0-dominant PRODUCTION
        must not re-admit the silent-fallback targets it does not dispatch.

        A hidden dim divisible by 256 does not make a Q5_0-dominated model a valid
        screening surface: it still exercises a kernel production never dispatches.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production = _production(tmp)
            for name, type_id in (("legacy.gguf", Q5_0), ("Bonsai.gguf", Q4_1)):
                with self.subTest(dominant=type_id):
                    path = _write_gguf(Path(tmp) / name, architecture="qwen2",
                                       n_embd=1536,
                                       tensors={type_id: 169, F32: 141})
                    with self.assertRaises(wc.WorkloadContractError) as caught:
                        wc.verify_workload(path, production=production)
                    self.assertIn("outside the production family",
                                  str(caught.exception))

    def test_an_iquant_workload_is_in_the_production_family(self):
        """I-quants are superblock quants -- deliberate choices, never fallback
        artifacts -- and stay screen-admissible whatever production's dominant is."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_gguf(Path(tmp) / "iq.gguf", architecture="qwen3",
                               n_embd=4096, tensors={16: 200, F32: 50})
            census = wc.verify_workload(path, production=_production(tmp))
            self.assertEqual(census.dominant_quant, "IQ2_XXS")

    def test_a_non_gguf_file_is_refused_clearly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not.gguf"
            path.write_bytes(b"this is not a gguf file at all")
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.read_census(path)
            self.assertIn("not a GGUF", str(caught.exception))


class TheProductionCensusFailsLoud(unittest.TestCase):
    """An unreadable production reference must refuse EVERY workload with a clear
    message -- never fail open into a pass nobody censused."""

    def test_a_missing_production_model_refuses_the_workload_loudly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            good = _write_gguf(Path(tmp) / "ok.gguf", architecture="qwen2",
                               n_embd=1536, tensors={Q4_K: 169, F32: 141})
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.verify_workload(good,
                                   production_model=Path(tmp) / "absent.gguf")
            self.assertIn("declared production model", str(caught.exception))
            self.assertIn("absent.gguf", str(caught.exception))

    def test_a_garbage_production_model_refuses_too(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            good = _write_gguf(Path(tmp) / "ok.gguf", architecture="qwen2",
                               n_embd=1536, tensors={Q4_K: 169, F32: 141})
            bad = Path(tmp) / "prod.gguf"
            bad.write_bytes(b"not a gguf")
            with self.assertRaises(wc.WorkloadContractError):
                wc.verify_workload(good, production_model=bad)

    def test_a_float_only_production_model_cannot_define_a_family(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production = _census(tmp, "f16.gguf", tensors={F16: 200, F32: 100})
            with self.assertRaises(wc.WorkloadContractError) as caught:
                wc.production_quant_family(production)
            self.assertIn("no quantised tensors", str(caught.exception))

    def test_the_family_is_censused_from_production_not_hard_coded(self):
        """Mutation guard on the census wiring itself: the family must FOLLOW the
        reference. Same workload, two productions, two verdicts."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            q8_workload = _write_gguf(Path(tmp) / "w.gguf", architecture="qwen35",
                                      n_embd=5120, tensors={Q8_0: 506, F32: 360})
            wc.verify_workload(q8_workload, production=_production(tmp))
            k_production = _census(tmp, "k.gguf", n_embd=1536,
                                   tensors={Q4_K: 169, F32: 141, Q6_K: 29})
            with self.assertRaises(wc.WorkloadContractError):
                wc.verify_workload(q8_workload, production=k_production)


class RungParityAgainstProduction(unittest.TestCase):
    """`rung_matches_production` (§5.1): exact for CONFIRM, recorded-but-waived
    for SCREEN. The waiver is a visible artifact, never a silent pass."""

    def _shapes(self, tmp):
        production = _production(tmp)
        exact = _census(tmp, "exact.gguf", tensors={Q8_0: 400, F32: 300})
        screen = _census(tmp, "screen.gguf", architecture="qwen2", n_embd=1536,
                         tensors={Q4_K: 169, F32: 141, Q6_K: 29})
        return production, exact, screen

    def test_an_exact_match_is_exact_on_both_rungs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production, exact, _screen = self._shapes(tmp)
            for rung in (wc.SCREEN_RUNG, wc.CONFIRM_RUNG):
                with self.subTest(rung=rung):
                    parity = wc.rung_matches_production(exact, production,
                                                        rung=rung)
                    self.assertTrue(parity.dominant_quant_match)
                    self.assertTrue(parity.n_embd_class_match)
                    self.assertTrue(parity.exact)
                    self.assertFalse(parity.waived,
                                     "an exact match has nothing to waive")

    def test_a_screen_mismatch_is_waived_and_the_waiver_is_visible(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production, _exact, screen = self._shapes(tmp)
            parity = wc.rung_matches_production(screen, production,
                                                rung=wc.SCREEN_RUNG)
            self.assertFalse(parity.exact)
            self.assertTrue(parity.waived)
            record = parity.to_dict()
            self.assertFalse(record["dominant_quant_match"])
            self.assertFalse(record["n_embd_class_match"])
            self.assertTrue(record["waived"], "the waiver must be IN the record")
            self.assertIn("WAIVED", record["detail"])
            self.assertEqual(record["production_dominant"], "Q8_0")
            self.assertEqual(record["workload_dominant"], "Q4_K")

    def test_a_confirm_mismatch_is_not_waivable(self):
        """Deleting or inverting the parity check must fail HERE: a non-exact
        confirm rung may never read as acceptable."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production, _exact, screen = self._shapes(tmp)
            parity = wc.rung_matches_production(screen, production,
                                                rung=wc.CONFIRM_RUNG)
            self.assertFalse(parity.exact)
            self.assertFalse(parity.waived,
                             "a confirm rung mismatch has NO waiver path")
            self.assertIn("not waivable", parity.detail)

    def test_each_axis_alone_breaks_exactness(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production = _production(tmp)
            right_width = _census(tmp, "w.gguf", tensors={Q4_K: 400, F32: 300})
            right_quant = _census(tmp, "q.gguf", n_embd=2560,
                                  tensors={Q8_0: 400, F32: 300})
            for census, broken in ((right_width, "dominant_quant_match"),
                                   (right_quant, "n_embd_class_match")):
                with self.subTest(broken=broken):
                    parity = wc.rung_matches_production(census, production,
                                                        rung=wc.CONFIRM_RUNG)
                    self.assertFalse(parity.exact)
                    self.assertFalse(parity.to_dict()[broken])

    def test_an_unknown_rung_role_is_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            production, exact, _screen = self._shapes(tmp)
            with self.assertRaises(wc.WorkloadContractError):
                wc.rung_matches_production(exact, production, rung="headline")


class TheRealModelsOnDisk(unittest.TestCase):
    """Guard rails against the actual files, skipped when they are absent.
    These run against the DEFAULT production reference (the real censused
    Qwen3.8-27B-Q8_0), so they double as the wiring check for PRODUCTION_MODEL."""

    OLD = Path("/mnt/raid0/llm/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/"
               "Qwen2.5-Coder-0.5B-Q4_K_M.gguf")
    NEW = Path("/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")

    def _need_production(self):
        if not wc.PRODUCTION_MODEL.is_file():
            self.skipTest("declared production model not present")

    def test_the_superseded_workload_is_refused(self):
        if not self.OLD.is_file():
            self.skipTest("superseded workload not present")
        self._need_production()
        with self.assertRaises(wc.WorkloadContractError):
            wc.verify_workload(self.OLD)

    def test_the_replacement_workload_passes_and_dispatches_q4_k(self):
        if not self.NEW.is_file():
            self.skipTest("replacement workload not present")
        self._need_production()
        census = wc.verify_workload(self.NEW)
        self.assertEqual(census.dominant_quant, "Q4_K")
        self.assertEqual(census.n_embd, 1536)
        self.assertGreaterEqual(census.tensor_types.get("Q4_K", 0), 100)

    def test_production_accepts_its_own_model(self):
        """The recur-proof for the 2026-08-14 drift: `verify_workload` on the real
        declared production model can never again refuse it, because the family is
        censused from that same file."""
        self._need_production()
        census = wc.verify_workload(wc.PRODUCTION_MODEL)
        self.assertEqual(census.dominant_quant, "Q8_0")
        self.assertEqual(census.n_embd, 5120)
        parity = wc.rung_matches_production(
            census, wc.production_census(), rung=wc.CONFIRM_RUNG)
        self.assertTrue(parity.exact, parity.detail)


if __name__ == "__main__":
    unittest.main()
