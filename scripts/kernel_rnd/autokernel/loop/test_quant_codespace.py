"""AKX-ALL-31.QDEC: reference decoders, enumerator, coverage, GGUF scanner and judge (no native binary)."""
import json
import os
import struct
import subprocess
import tempfile
import unittest
from fractions import Fraction

import numpy as np

from autokernel.loop import quant_codespace as qc

LLAMA = "/mnt/raid0/llm/llama.cpp"


def _f16(x: float) -> int:
    return int(np.float16(x).view(np.uint16))


def _has_c12() -> bool:
    try:
        return subprocess.run(["git", "-C", LLAMA, "cat-file", "-e", qc.C12], capture_output=True).returncode == 0
    except OSError:
        return False


class Tables(unittest.TestCase):
    def test_embedded_tables_bound_and_sized(self):
        t = qc.load_tables()
        self.assertEqual({k: len(v) for k, v in t.items()},
                         {"iq2xxs_grid": 256, "iq2xs_grid": 512, "iq2s_grid": 1024, "iq3xxs_grid": 256,
                          "iq3s_grid": 512, "kvalues_iq4nl": 16, "kvalues_fp4": 16})
        self.assertEqual(t["kvalues_iq4nl"][0], -127)
        self.assertEqual(t["iq2xxs_grid"][0], 0x0808080808080808)

    def test_derived_ksigns_parity(self):
        ks = qc.ksigns()
        self.assertEqual(ks[1], 129)          # one sign bit -> parity bit set
        self.assertEqual(ks[3], 3)
        self.assertEqual(ks[127], 255)
        self.assertTrue(all(bin(int(v)).count("1") % 2 == 0 for v in ks))

    @unittest.skipUnless(_has_c12(), "llama.cpp C12 objects not available")
    def test_tables_match_c12_source(self):
        src = qc.source_tables(LLAMA)
        emb = qc.load_tables()
        for k in qc.EMBEDDED:
            self.assertEqual(src[k], emb[k], k)
        self.assertEqual(src["ksigns_iq2xs"], qc.ksigns().tolist())


class HandComputed(unittest.TestCase):
    def test_q8_0(self):
        qt = qc.qtype("Q8_0")
        b = qt.pack({"d": np.array([[_f16(0.5)]]), "q": (np.arange(32)[None, :] * 8 - 128) & 0xFF})
        y = qt.decode(b)[0]
        self.assertEqual(y[0], -64.0)          # code 0x80 = -128, times 0.5
        self.assertEqual(y[31], 60.0)          # 31 * 8 - 128 = 120

    def test_q4_k(self):
        qt = qc.qtype("Q4_K")
        f = {"d": np.array([[_f16(1.0)]]), "dmin": np.array([[_f16(0.5)]]),
             "sc": np.array([[3, 0, 0, 0, 0, 0, 0, 37]]), "m": np.array([[2, 0, 0, 0, 0, 0, 0, 50]]),
             "q": np.zeros((1, 256), np.int64)}
        f["q"][0, 0] = 5; f["q"][0, 255] = 15
        b = qt.pack(f)
        # sub-block 7 uses the packed high bits: sc 37 = 0b100101, m 50 = 0b110010
        self.assertEqual(b[0, 4 + 3] >> 6, 37 >> 4)
        self.assertEqual(b[0, 12 + 3] & 0xF, 37 & 0xF)
        self.assertEqual(b[0, 12 + 3] >> 4, 50 & 0xF)
        # element 255 lives in the high nibble of qs[127]
        self.assertEqual(b[0, 16 + 127] >> 4, 15)
        y = qt.decode(b)[0]
        self.assertEqual(y[0], 1 * 3 * 5 - 0.5 * 2)
        self.assertEqual(y[255], 37 * 15 - 0.5 * 50)
        self.assertEqual(y[1], -1.0)

    def test_q6_k(self):
        qt = qc.qtype("Q6_K")
        b = qt.empty(1)
        b[0, 208:210] = np.frombuffer(np.float16(0.25).tobytes(), np.uint8)
        b[0, 192:208] = np.array([-3] + [1] * 15, np.int8).view(np.uint8)
        b[0, 0] = 0x7A           # element 0 low nibble 0xA, element 64 low nibble 0x7
        b[0, 128] = 0b11100110   # qh: element 0 bits 2, element 32 bits 1, element 64 bits 2, element 96 bits 3
        y = qt.decode(b)[0]
        self.assertEqual(y[0], 0.25 * -3 * ((0xA | (2 << 4)) - 32))
        self.assertEqual(y[64], 0.25 * 1 * ((0x7 | (2 << 4)) - 32))
        self.assertEqual(y[32], 0.25 * 1 * ((0 | (1 << 4)) - 32))
        self.assertEqual(y[96], 0.25 * 1 * ((0 | (3 << 4)) - 32))

    def test_iq4_xs(self):
        qt = qc.qtype("IQ4_XS")
        s = np.array([[0, 33, 63, 32, 1, 2, 3, 40]])
        q = np.zeros((1, 256), np.int64); q[0, 16] = 15; q[0, 32] = 8
        y = qt.decode(qt.pack({"d": np.array([[_f16(2.0)]]), "s": s, "q": q}))[0]
        self.assertEqual(y[0], 2.0 * (0 - 32) * -127)
        self.assertEqual(y[16], 2.0 * (0 - 32) * 113)
        self.assertEqual(y[32], 2.0 * (33 - 32) * 1)

    def test_mxfp4(self):
        qt = qc.qtype("MXFP4")
        b = qt.empty(3); b[:, 0] = [128, 0, 255]; b[:, 1] = 0x97   # elements 0 -> code 7, 16 -> code 9
        y = qt.decode(b)
        self.assertEqual(y[0, 0], 12.0)
        self.assertEqual(y[0, 16], -1.0)
        self.assertEqual(y[1, 0], np.float32(12 * 2.0 ** -128))        # float32 subnormal, exact
        self.assertTrue(np.isinf(y[2, 0]))                               # 12 * 2^127 overflows as in C
        self.assertEqual(y[2, 16], np.float32(-(2.0 ** 127)))

    def test_iq2_xxs(self):
        qt = qc.qtype("IQ2_XXS")
        b = qt.empty(1)
        b[0, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)
        aux1 = (3 << 28) | 1        # scale 3, group 0 sign index 1 -> ksigns 129 (bits 0 and 7)
        b[0, 6:10] = np.frombuffer(struct.pack("<I", aux1), np.uint8)
        y = qt.decode(b)[0]
        db = 1.0 * (0.5 + 3) * 0.25
        self.assertEqual(y[0], -db * 8)
        self.assertEqual(y[1], db * 8)
        self.assertEqual(y[7], -db * 8)
        self.assertEqual(y[32], 1.0 * 0.5 * 0.25 * 8)

    def test_iq3_s_high_bit_and_signs(self):
        qt = qc.qtype("IQ3_S")
        grid = np.array(qc.load_tables()["iq3s_grid"], np.uint64)
        b = qt.empty(1)
        b[0, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)
        b[0, 2 + 1] = 5          # quad 1 low byte 5
        b[0, 66] = 0b10          # qh bit 1 -> quad 1 index 256 + 5
        b[0, 74] = 0b00010000    # sign bit 4 -> element 4 (first element of quad 1) negative
        b[0, 106] = 0x02         # sub-block 0 scale 2 -> db = 1 + 4
        y = qt.decode(b)[0]
        g = int(grid[256 + 5])
        self.assertEqual(y[4], -5.0 * (g & 0xFF))
        self.assertEqual(y[5], 5.0 * ((g >> 8) & 0xFF))


class Exactness(unittest.TestCase):
    def test_round_sub_f32_is_correctly_rounded(self):
        rng = np.random.default_rng(3)
        d = qc.f16(rng.integers(0, 0x7BFF, 600)); dmin = qc.f16(rng.integers(0, 0x7BFF, 600))
        a = d * rng.integers(0, 64, 600) * rng.integers(0, 16, 600)
        bb = dmin * rng.integers(0, 64, 600)
        # adversarial: operands far apart in exponent, and exact float32 midpoints
        a = np.concatenate([a, [2.0 ** 20 + 2.0 ** -3, 1.0 + 2.0 ** -24, 3.0, 1.0 + 3 * 2.0 ** -24]])
        bb = np.concatenate([bb, [2.0 ** -40, 2.0 ** -60, 2.0 ** -30, 2.0 ** -60]])
        got = qc.round_sub_f32(a, bb)
        for x, y, g in zip(a, bb, got):
            exact = Fraction(float(x)) - Fraction(float(y))
            lo = np.float32(float(exact))
            cands = sorted({float(lo), float(np.nextafter(lo, np.float32(np.inf))),
                            float(np.nextafter(lo, np.float32(-np.inf)))},
                           key=lambda c: (abs(Fraction(c) - exact),
                                          int(np.float32(c).view(np.uint32)) & 1))
            self.assertEqual(float(g), cands[0], (x, y))


class RoundTripAndEnumeration(unittest.TestCase):
    def test_pack_unpack_identity_on_random_blocks(self):
        rng = np.random.default_rng(11)
        for name in qc.TYPES:
            qt = qc.qtype(name)
            b = rng.integers(0, 256, size=(64, qt.bsize), dtype=np.uint8)
            self.assertTrue(np.array_equal(qt.pack(qt.unpack(b)), b), name)

    def test_enumerator_complete_for_every_type(self):
        dset = [_f16(1.0), _f16(-0.75), 0x0001, 0x7BFF]
        for name in qc.TYPES:
            qt = qc.qtype(name)
            blocks = qc.pad_to_rows(qt, qc.enumerate_blocks(qt, dset, dset[:2]))
            self.assertEqual(len(blocks) * qt.qk % qc.ROW, 0)
            cov = qc.coverage(qt, blocks, {"d": dset, "dmin": dset[:2]})
            for field, v in cov["addressing"].items():
                self.assertEqual(v["fraction"], 1.0, f"{name}.{field}")
            for axis, v in cov["arith"].items():
                self.assertEqual(v["fraction"], 1.0, f"{name}.{axis}")
            # finite everywhere except MXFP4 e=255 with |kv| >= 4, which overflows exactly as in C
            self.assertFalse(np.isnan(qt.decode(blocks)).any(), name)

    def test_coverage_detects_missing_code(self):
        qt = qc.qtype("IQ2_XXS")
        blocks = qc.enumerate_blocks(qt)
        f = qt.unpack(blocks)
        f["grid"][f["grid"] == 200] = 201
        cov = qc.coverage(qt, qt.pack(f))
        self.assertLess(cov["addressing"]["grid"]["fraction"], 1.0)

    def test_random_suite_model(self):
        rep = qc.random_suite_report((4, 64), seed=1, types=("Q8_0", "MXFP4", "IQ2_S"))
        self.assertEqual(rep["Q8_0"][0]["model"], "emulated_ref_quantizer")
        self.assertEqual(rep["IQ2_S"][0]["model"], "random_code_proxy")
        # the reference Q8_0 quantizer never emits code -128, so the suite cannot reach it
        self.assertLess(rep["Q8_0"][1]["fields"]["q"], 1.0)
        # uniform [-1,1] inputs hit only a couple of E8M0 exponents
        self.assertLess(rep["MXFP4"][1]["fields"]["e"], 0.02)

    def test_suite_seed_mix_is_splitmix64(self):
        self.assertEqual(int(qc.suite_seed_mix(np.uint64(0))), 0xE220A8397B1DCDAF)

    def test_q8_0_reference_quantizer(self):
        qt = qc.qtype("Q8_0")
        x = np.linspace(-1, 0.5, 32, dtype=np.float32)
        b = qc.quantize_ref(qt, x)
        f = qt.unpack(b)
        self.assertEqual(f["d"][0, 0], _f16(np.float32(1) / np.float32(127)))
        self.assertEqual(f["q"][0, 0], (-127) & 0xFF)


class Judge(unittest.TestCase):
    def _enum(self, name):
        qt = qc.qtype(name)
        return qt, qc.pad_to_rows(qt, qc.enumerate_blocks(qt))

    def test_reference_passes_itself(self):
        for name in qc.TYPES:
            qt, b = self._enum(name)
            self.assertEqual(qc.judge(qt, b, qt.decode(b), "self")["verdict"], "pass", name)

    def test_planted_wrong_nibble_order_goes_red(self):
        qt, b = self._enum("Q4_K")
        mut = b.copy()
        mut[:, 16:] = ((mut[:, 16:] & 0xF) << 4) | (mut[:, 16:] >> 4)   # decoder reading nibbles swapped
        res = qc.judge(qt, b, qt.decode(mut), "planted_nibble_swap")
        self.assertEqual(res["verdict"], "fail")
        self.assertGreaterEqual(res["mismatches"], 64)   # caught by the position-digit blocks
        self.assertIn("sc", res["first"][0]["codes"])

    def test_planted_wrong_grid_entry_goes_red_only_with_enumeration(self):
        qt, b = self._enum("IQ2_XXS")
        tables = qc.load_tables()
        grid = list(tables["iq2xxs_grid"]); grid[200] ^= 0x0000000000110000   # one byte of one entry
        mutant = qc.IQ2_XXS(grid=grid)
        res = qc.judge(qt, b, mutant.decode(b), "planted_grid_entry")
        self.assertEqual(res["verdict"], "fail")
        self.assertTrue(all(m["codes"]["mag"] >= 0 for m in res["first"]))
        # a 4-block random-code proxy almost surely never selects entry 200 at the right byte
        rb, _ = qc.random_suite_blocks(qt, 1, seed=5)
        if not (qt.unpack(rb)["grid"] == 200).any():
            self.assertEqual(qc.judge(qt, rb, mutant.decode(rb), "proxy")["verdict"], "pass")

    def test_planted_q6_k_high_bit_shift_goes_red(self):
        qt, b = self._enum("Q6_K")
        mut = b.copy()
        mut[:, 128:192] = ((mut[:, 128:192] << 2) | (mut[:, 128:192] >> 6)) & 0xFF   # qh 2-bit lanes rotated
        self.assertEqual(qc.judge(qt, b, qt.decode(mut), "planted_qh")["verdict"], "fail")

    def test_planted_iq3_s_sign_bit_order_goes_red(self):
        qt, b = self._enum("IQ3_S")
        mut = b.copy()
        s = mut[:, 74:106]
        mut[:, 74:106] = np.array([int(f"{v:08b}"[::-1], 2) for v in s.ravel()], np.uint8).reshape(s.shape)
        self.assertEqual(qc.judge(qt, b, qt.decode(mut), "planted_sign_order")["verdict"], "fail")

    def test_exact_mode_classifies_zero_sign_and_ftz(self):
        qt = qc.qtype("MXFP4")
        b = qt.empty(8); b[:, 0] = 0; b[:, 1] = 0x01          # e=0: element 0 = 1 * 2^-128 (subnormal)
        ref = qt.decode(b).ravel()
        got = ref.copy(); got[0] = 0.0; got[1] = -0.0
        res = qc.judge(qt, b, got, "classes")
        self.assertEqual(res["verdict"], "fail")
        self.assertEqual(res["classes"]["ftz_subnormal"], 1)
        self.assertEqual(res["classes"]["zero_sign"], 1)

    def test_identity_mode_bound(self):
        qt, b = self._enum("Q4_K")
        ref = qt.decode(b).astype(np.float64) * qc.ONEHOT
        r32 = ref.astype(np.float32)
        ok = np.where(r32 == 0, r32, np.nextafter(r32, np.float32(np.inf)))
        self.assertEqual(qc.judge(qt, b, ok, "mul_mat:CPU", mode="identity")["verdict"], "pass")
        bad = ref.astype(np.float32).ravel().copy(); bad[777] = bad[777] * 1.01 + 1.0   # outside every benign class
        self.assertEqual(qc.judge(qt, b, bad, "mul_mat:CPU", mode="identity")["verdict"], "fail")

    def test_chunked_judge_reports_global_block(self):
        qt, b = self._enum("IQ4_XS")
        got = qt.decode(b).ravel().copy()
        got[(len(b) - 1) * qt.qk + 3] += 1.0
        res = qc.judge(qt, b, got, "chunked", chunk_blocks=7)
        self.assertEqual((res["mismatches"], res["first"][0]["block"]), (1, len(b) - 1))

    def test_shards_partition_scale_rows(self):
        qt = qc.qtype("Q4_K")
        ds = [_f16(v) for v in (1.0, 2.0, 3.0, 5.0, 7.0)]
        whole = qc.enumerate_blocks(qt, ds, ds[:2])
        parts = [qc.enumerate_blocks(qt, ds, ds[:2], (i, 3)) for i in range(3)]
        n_addr = len(qc.enumerate_blocks(qt, ds[:1], ds[:1])) - 8   # addressing part, repeated per shard
        self.assertEqual(sum(len(p) - n_addr for p in parts), len(whole) - n_addr)
        tiles = np.concatenate([p[n_addr:] for p in parts])
        self.assertEqual(sorted(map(bytes, tiles)), sorted(map(bytes, whole[n_addr:])))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(qc.main(["enumerate", "--type", "Q6_K", "--all-finite-fp16", "--shard", "5/32",
                                      "--out-dir", tmp]), 0)
            with open(os.path.join(tmp, "Q6_K.shard5of32.manifest.json")) as fh:
                man = json.load(fh)
            self.assertTrue(man["coverage"].get("structural"))
            self.assertTrue(man["scale_bytes_ok"])
            self.assertEqual(man["scale_rows"], len(range(5, 63490, 32)))

    def test_size_mismatch_is_invalid(self):
        qt, b = self._enum("Q8_0")
        self.assertEqual(qc.judge(qt, b, np.zeros(5, np.float32), "x")["verdict"], "invalid")


def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def _write_gguf(path: str, tensors: list) -> None:
    """Minimal GGUF v3: alignment kv, a string kv, a string array kv, tensor infos, aligned data."""
    kv = [("general.alignment", 4, struct.pack("<I", 32)),
          ("general.name", 8, _gguf_string("qdec-synthetic")),
          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _gguf_string("a") + _gguf_string("bc"))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, vt, val in kv:
        head += _gguf_string(k) + struct.pack("<I", vt) + val
    data, infos = b"", b""
    for name, ggml_type, dims, payload in tensors:
        data += b"\0" * ((-len(data)) % 32)
        infos += _gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", ggml_type, len(data))
        data += payload
    body = head + infos
    body += b"\0" * ((-len(body)) % 32)
    with open(path, "wb") as fh:
        fh.write(body + data)


class GGUF(unittest.TestCase):
    def test_header_parse_and_harvest(self):
        q8, q4 = qc.qtype("Q8_0"), qc.qtype("Q4_K")
        d8 = [_f16(v) for v in (0.5, 0.25, 0.5, 3.0)]
        b8 = q8.pack({"d": np.array(d8)[:, None], "q": np.zeros((4, 32), np.int64)})
        f4 = q4.unpack(np.zeros((2, q4.bsize), np.uint8))
        f4["d"] = np.array([[_f16(1.5)], [_f16(-2.0)]]); f4["dmin"] = np.array([[_f16(0.125)], [_f16(0.125)]])
        b4 = q4.pack(f4)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.gguf")
            _write_gguf(p, [("blk.0.attn_q.weight", 8, [64, 2], b8.tobytes()),
                            ("blk.0.ffn_up.weight", 12, [256, 2], b4.tobytes()),
                            ("output_norm.weight", 0, [8], b"\0" * 32)])
            hdr = qc.read_gguf_header(p)
            self.assertEqual(hdr["version"], 3)
            self.assertEqual(hdr["kv"]["general.name"], "qdec-synthetic")
            self.assertEqual(hdr["kv"]["tokenizer.ggml.tokens"], {"array_of": 8, "len": 2})
            self.assertEqual([t["type"] for t in hdr["tensors"]], [8, 12, 0])
            self.assertEqual(hdr["data_offset"] % 32, 0)
            doc = qc.harvest_scales([p])
            self.assertEqual(doc["types"]["Q8_0"]["d"], sorted(set(d8)))
            self.assertEqual(doc["types"]["Q8_0"]["blocks"], 4)
            self.assertEqual(doc["types"]["Q4_K"]["d"], sorted({_f16(1.5), _f16(-2.0)}))
            self.assertEqual(doc["types"]["Q4_K"]["dmin"], [_f16(0.125)])
            sp = os.path.join(tmp, "scales.json")
            with open(sp, "w") as fh:
                json.dump(doc, fh)
            out = os.path.join(tmp, "enum")
            self.assertEqual(qc.main(["enumerate", "--type", "Q4_K", "--scales", sp, "--out-dir", out]), 0)
            with open(os.path.join(out, "Q4_K.manifest.json")) as fh:
                man = json.load(fh)
            self.assertEqual(man["ggml_type_name"], "q4_K")
            self.assertEqual(man["scale_counts"], {"d": 2, "dmin": 1})
            with open(os.path.join(out, "Q4_K.blocks"), "rb") as fh:
                blocks = fh.read()
            native = os.path.join(tmp, "native.f32")
            q4.decode(blocks).tofile(native)
            self.assertEqual(qc.main(["judge", "--type", "q4_k", "--blocks", os.path.join(out, "Q4_K.blocks"),
                                      "--native", native, "--leg", "to_float",
                                      "--out", os.path.join(tmp, "v.json")]), 0)

    def test_not_gguf_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.bin")
            with open(p, "wb") as fh:
                fh.write(b"NOPE" + b"\0" * 64)
            with self.assertRaises(ValueError):
                qc.read_gguf_header(p)


class Receipts(unittest.TestCase):
    def test_parse_receipts(self):
        text = "noise\nAK_QDEC_V1 {\"leg\": \"to_float\", \"status\": \"ok\"}\nAK_QDEC_V1 {bad\n"
        r = qc.receipts(text)
        self.assertEqual(r[0]["leg"], "to_float")
        self.assertEqual(r[1]["status"], "malformed")


if __name__ == "__main__":
    unittest.main()
