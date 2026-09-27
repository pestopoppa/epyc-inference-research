"""Hardware-free tests for the independent fixed CPU quant oracle."""
import json
import struct
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import cpu_quant_reference as fixture


def _f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def q8_encoded_row(expert: int, row: int, block_order: tuple[int, ...] = tuple(range(8))) -> bytes:
    """Q8_0 row of the fixture; `block_order` lets a test mis-assign block scales."""
    values = [fixture._source_weight(expert, row, column) for column in range(fixture.K)]
    scales, quants = [], []
    for block in range(8):
        chunk = values[block * 32:(block + 1) * 32]
        scale = _f16(max(abs(value) for value in chunk) / 127)
        scales.append(scale)
        quants.append([max(-127, min(127, round(value / scale))) for value in chunk])
    return b"".join(struct.pack("<e32b", scales[block_order[block]], *quants[block])
                    for block in range(8))


def _pack_k_scales(scales: list[int], minimums: list[int]) -> bytes:
    """Inverse of `fixture._scale_min`: ggml's 12-byte 6-bit K-quant layout."""
    packed = [0] * 12
    for group in range(4):
        packed[group] = scales[group] | ((scales[group + 4] >> 4) << 6)
        packed[group + 4] = minimums[group] | ((minimums[group + 4] >> 4) << 6)
        packed[group + 8] = (scales[group + 4] & 15) | ((minimums[group + 4] & 15) << 4)
    return bytes(packed)


def k_encoded_row(quant: str, expert: int, row: int,
                  scale_order: tuple[int, ...] = tuple(range(8))) -> bytes:
    """Independent Q4_K/Q5_K encoder for the fixture row.

    `scale_order` stores sub-block g's (scale, min) pair from sub-block
    `scale_order[g]` while keeping the quants: the byte image a kernel with a
    sub-scale layout/permutation bug effectively decodes.
    """
    levels = 15 if quant == "Q4_K" else 30  # 30: the fixture's 31-level grid
    values = [fixture._source_weight(expert, row, column) for column in range(fixture.K)]
    steps, minimums = [], []
    for group in range(8):
        chunk = values[group * 32:(group + 1) * 32]
        low = min(0.0, min(chunk))
        steps.append((max(chunk) - low) / levels)
        minimums.append(-low)
    d, dmin = _f16(max(steps) / 63), _f16(max(minimums) / 63)
    scale_codes = [round(step / d) for step in steps]
    minimum_codes = [round(minimum / dmin) if dmin else 0 for minimum in minimums]
    top = 15 if quant == "Q4_K" else 31
    quants = []
    for group in range(8):
        step = d * scale_codes[group]
        for value in values[group * 32:(group + 1) * 32]:
            quant_value = round((value + dmin * minimum_codes[group]) / step) if step else 0
            quants.append(max(0, min(top, quant_value)))
    packed, high = bytearray(128), bytearray(32)
    for group in range(8):
        for column32 in range(32):
            quant_value = quants[group * 32 + column32]
            if quant_value & 16:
                high[column32] |= 1 << group
            packed[(group // 2) * 32 + column32] |= (quant_value & 15) << (4 * (group % 2))
    header = struct.pack("<ee", d, dmin) + _pack_k_scales(
        [scale_codes[scale_order[g]] for g in range(8)],
        [minimum_codes[scale_order[g]] for g in range(8)])
    return header + (bytes(packed) if quant == "Q4_K" else bytes(high) + bytes(packed))


def encoded_row(quant: str, expert: int, row: int,
                order: tuple[int, ...] = tuple(range(8))) -> bytes:
    return (q8_encoded_row(expert, row, order) if quant == "Q8_0"
            else k_encoded_row(quant, expert, row, order))


def permuted_scale_output(quant: str, op: str, width: int,
                          order: tuple[int, ...]) -> str:
    """Probe output whose stored bytes are right but whose outputs were computed
    by a kernel that reads the per-block scales in `order`."""
    experts = 2 if op == "MUL_MAT_ID" else 1
    lines = [f"{fixture.MARKER} {quant} {op} 256 40 {width} {fixture.ROW_BYTES[quant]}"]
    misread = {}
    for expert in range(experts):
        for row in range(fixture.ROWS):
            lines.append(f"A {expert} {row} {encoded_row(quant, expert, row).hex()}")
            misread[expert, row] = encoded_row(quant, expert, row, order)
    for token in range(width):
        for row in range(fixture.ROWS):
            value = fixture._reference(quant, op, misread, token, row)
            lines.append(f"O {token} {row} {value.hex()}")
    return "\n".join(lines) + "\n"


def _constant_scale_weight(expert: int, row: int, column: int) -> float:
    # The pre-DS41-C53 fixture: every 32-block spans the same +/-15/16.
    return ((column * 13 + row * 7 + expert * 17) % 31 - 15) / 16.0


IDENTITY = tuple(range(8))
ADJACENT_SWAP = (1, 0, 3, 2, 5, 4, 7, 6)
REVERSED = tuple(range(7, -1, -1))


def q8_fused_output(width: int = fixture.TOKENS,
                    expert_mode: str = "alternating") -> str:
    op = fixture.FUSED_OP
    suffix = " single_expert" if expert_mode == "single" else ""
    lines = [f"{fixture.MARKER} Q8_0 {op} 256 40 {width} 272{suffix}"]
    up, gate = {}, {}
    for expert in range(2):
        for row in range(fixture.ROWS):
            up[expert, row] = q8_encoded_row(expert, row)
            gate[expert, row] = q8_encoded_row(expert + 3, row + 5)
            lines.append(f"A {expert} {row} {up[expert, row].hex()}")
            lines.append(f"G {expert} {row} {gate[expert, row].hex()}")
    for token in range(width):
        for row in range(fixture.ROWS):
            value = fixture._reference("Q8_0", op, up, token, row, gate, expert_mode)
            lines.append(f"O {token} {row} {value.hex()}")
    return "\n".join(lines) + "\n"


def q8_output(op: str = "MUL_MAT_ID", width: int = fixture.TOKENS,
              expert_mode: str = "alternating") -> str:
    experts = 2 if op == "MUL_MAT_ID" else 1
    suffix = " single_expert" if expert_mode == "single" else ""
    lines = [f"{fixture.MARKER} Q8_0 {op} 256 40 {width} 272{suffix}"]
    rows = {}
    for expert in range(experts):
        for row in range(fixture.ROWS):
            encoded = q8_encoded_row(expert, row)
            rows[expert, row] = encoded
            lines.append(f"A {expert} {row} {encoded.hex()}")
    for token in range(width):
        for row in range(fixture.ROWS):
            value = fixture._reference("Q8_0", op, rows, token, row,
                                       expert_mode=expert_mode)
            lines.append(f"O {token} {row} {value.hex()}")
    return "\n".join(lines) + "\n"


class QuantReferenceTest(unittest.TestCase):
    def test_all_activation_widths_cover_dynamic_expert_and_fused_rows(self):
        for width in fixture.SUPPORTED_WIDTHS:
            for expert_mode in ("alternating", "single"):
                for op in (*fixture.OPS, fixture.FUSED_OP):
                    with self.subTest(width=width, op=op, expert_mode=expert_mode):
                        output = (q8_fused_output(width, expert_mode)
                                  if op == fixture.FUSED_OP else
                                  q8_output(op, width, expert_mode))
                        result = fixture._parse_and_compare(output, "Q8_0", op, width,
                                                            expert_mode)
                        self.assertEqual(result.status, "pass")
                        metric = json.loads(result.detail.split(" ", 1)[1])
                        self.assertEqual((metric["width"], metric["outputs"],
                                          metric["expert_mode"]),
                                         (width, width * fixture.ROWS, expert_mode))

    def test_single_expert_mode_is_identity_bound(self):
        output = q8_fused_output(4, "single")
        with self.assertRaisesRegex(ValueError, "identity"):
            fixture._parse_and_compare(output, "Q8_0", fixture.FUSED_OP, 4)
        with self.assertRaisesRegex(ValueError, "identity"):
            fixture._parse_and_compare(q8_fused_output(4), "Q8_0",
                                       fixture.FUSED_OP, 4, "single")

    def test_width_mismatch_missing_last_output_and_extra_output_refused(self):
        output = q8_output(width=8)
        with self.assertRaisesRegex(ValueError, "identity"):
            fixture._parse_and_compare(output, "Q8_0", "MUL_MAT_ID", 4)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixture._parse_and_compare("\n".join(output.splitlines()[:-1]),
                                       "Q8_0", "MUL_MAT_ID", 8)
        lines = q8_output(width=4).splitlines()
        lines.append("O 4 0 0x0p+0")
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID", 4)

    def test_width_suite_passes_width_to_probe_and_preserves_default(self):
        completed = [subprocess.CompletedProcess([], 0, "", "")]
        completed += [subprocess.CompletedProcess([], 0, q8_output(width=width), "")
                      for width in (1, 2, 4, 8)]
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(subprocess, "run", side_effect=completed) as run:
            result = fixture.check_cpu_quant_suite(
                Path("/build"), Path("/source"), quants=("Q8_0",),
                ops=("MUL_MAT_ID",), widths=(1, 2, 4, 8))
        self.assertEqual(result.status, "pass")
        self.assertEqual([call.args[0][-1] for call in run.call_args_list[1:]],
                         ["1", "2", "4", "8"])
        rows = [json.loads(line.split(" ", 1)[1]) for line in result.detail.splitlines()]
        self.assertEqual([row["width"] for row in rows], [1, 2, 4, 8])

    def test_compile_uses_toolchain_env_and_run_uses_launch_env(self):
        # DS41 run 10i: the launch env is an allowlist with no PATH, so compiling
        # under it failed with "cannot execute 'cc1plus'" and every widened-route
        # reference came back `unavailable`.
        completed = [subprocess.CompletedProcess([], 0, "", ""),
                     subprocess.CompletedProcess([], 0, q8_output(width=1), "")]
        launch = {"OMP_NUM_THREADS": "48"}
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.dict("os.environ", {"PATH": "/usr/bin:/bin"}), \
             mock.patch.object(subprocess, "run", side_effect=completed) as run:
            result = fixture.check_cpu_quant_suite(
                Path("/build"), Path("/source"), quants=("Q8_0",),
                ops=("MUL_MAT_ID",), widths=(1,), launch_env=launch)
        self.assertEqual(result.status, "pass")
        compile_env = run.call_args_list[0].kwargs["env"]
        run_env = run.call_args_list[1].kwargs["env"]
        self.assertEqual(compile_env.get("PATH"), "/usr/bin:/bin")
        self.assertNotIn("PATH", run_env)
        self.assertEqual(run_env["OMP_NUM_THREADS"], "48")
        self.assertTrue(run_env["LD_LIBRARY_PATH"].startswith("/build/bin"))

    def test_single_expert_suite_passes_cli_mode(self):
        completed = [subprocess.CompletedProcess([], 0, "", ""),
                     subprocess.CompletedProcess([], 0,
                                                 q8_output(width=4, expert_mode="single"), "")]
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(subprocess, "run", side_effect=completed) as run:
            result = fixture.check_cpu_quant_suite(
                Path("/build"), Path("/source"), quants=("Q8_0",),
                ops=("MUL_MAT_ID",), widths=(4,), expert_mode="single")
        self.assertEqual(result.status, "pass")
        self.assertEqual(run.call_args.args[0][-2:], ["4", "--single-expert"])

    def test_dot_witness_forces_single_expert_and_rowexact_off(self):
        from autokernel.loop import iqk_witness
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0, "", "")), \
             mock.patch.object(iqk_witness, "run_fused_probe",
                               return_value=(mock.Mock(status="pass"), "invalid")) as witness:
            result = fixture.check_cpu_quant_suite(
                Path("/build"), Path("/source"), quants=("Q4_K",),
                ops=(fixture.FUSED_OP,), widths=(4,), require_dot_hit=True)
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.reason, "Q4_K FUSED_UP_GATE width=4 probe output invalid")
        self.assertEqual(witness.call_args.kwargs["expected_width"], 4)
        self.assertTrue(witness.call_args.kwargs["single_expert"])
        self.assertEqual(witness.call_args.kwargs["dot_quant"], "Q4_K")
        self.assertEqual(witness.call_args.kwargs["launch_env"]["GGML_ROWEXACT_N"], "0")

    def test_fused_reference_checks_both_matrices_and_output(self):
        output = q8_fused_output()
        self.assertEqual(fixture._parse_and_compare(output, "Q8_0", fixture.FUSED_OP).status,
                         "pass")
        lines = output.splitlines()
        lines = [line for line in lines if not line.startswith("G 1 39 ")]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixture._parse_and_compare("\n".join(lines), "Q8_0", fixture.FUSED_OP)
        lines = output.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        lines[index] = "O 0 0 0x1.0p+10"
        self.assertEqual(fixture._parse_and_compare(
            "\n".join(lines), "Q8_0", fixture.FUSED_OP).status, "wrong")
        lines = output.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("G 0 0 "))
        lines[index] = "G 0 0 " + bytes(fixture.ROW_BYTES["Q8_0"]).hex()
        self.assertEqual(fixture._parse_and_compare(
            "\n".join(lines), "Q8_0", fixture.FUSED_OP).status, "wrong")

    def test_existing_cpu_fixture_reports_observed_error_without_new_gate(self):
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        baseline = float.fromhex(lines[index].split()[3])
        lines[index] = f"O 0 0 {(baseline + 0.009).hex()}"
        result = fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID")
        self.assertEqual(result.status, "pass")
        marker, receipt = result.detail.split(" ", 1)
        self.assertEqual(marker, fixture.METRIC_MARKER)
        metric = json.loads(receipt)
        self.assertEqual((metric["quant"], metric["op"], metric["outputs"]),
                         ("Q8_0", "MUL_MAT_ID", 80))
        self.assertLess(metric["max_quant_abs_error"], fixture.QUANT_ABS_TOL["Q8_0"])
        self.assertAlmostEqual(metric["max_output_abs_error"], 0.009)
        self.assertAlmostEqual(metric["max_output_limit_fraction"], 0.9)
        self.assertEqual((metric["output_abs_tol"], metric["output_rel_tol"]),
                         (fixture.ABS_TOL, fixture.REL_TOL))

    def test_suite_carries_each_existing_case_metric(self):
        completed = [subprocess.CompletedProcess([], 0, "", "")]
        completed += [subprocess.CompletedProcess([], 0, q8_output(op), "")
                      for op in fixture.OPS]
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(subprocess, "run", side_effect=completed):
            result = fixture.check_cpu_quant_suite(
                Path("/build"), Path("/source"), quants=("Q8_0",), ops=fixture.OPS)
        self.assertEqual(result.status, "pass")
        rows = [json.loads(line.split(" ", 1)[1]) for line in result.detail.splitlines()]
        self.assertEqual([row["op"] for row in rows], list(fixture.OPS))
        self.assertTrue(all(row["width"] == 2 and
                            row["expert_mode"] == "alternating" for row in rows))
        self.assertTrue(all(row["schema"] == "epyc.autokernel.cpu_quant_metric.v1"
                            for row in rows))

    def test_q8_scalar_oracle_accepts_both_ops_without_claiming_dispatch(self):
        for op in fixture.OPS:
            with self.subTest(op=op):
                result = fixture._parse_and_compare(q8_output(op), "Q8_0", op)
                self.assertEqual(result.status, "pass")
                self.assertFalse(result.path_verified)

    def test_changed_output_is_wrong_not_unavailable(self):
        output = q8_output().replace("O 0 0 ", "O 0 0 0x1.0p+10\nO 0 0 ", 1)
        with self.assertRaisesRegex(ValueError, "duplicate output"):
            fixture._parse_and_compare(output, "Q8_0", "MUL_MAT_ID")
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        lines[index] = "O 0 0 0x1.0p+10"
        result = fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID")
        self.assertEqual(result.status, "wrong")
        self.assertIn("mismatch", result.reason)

    def test_near_boundary_wrong_output_is_rejected(self):
        # Below |2| the absolute allowance dominates. A +0.012 perturbation
        # exceeds both the 0.01 absolute and 0.005 relative allowances.
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("O 0 0 "))
        baseline = float.fromhex(lines[index].split()[3])
        self.assertLess(abs(baseline), 1.0)
        lines[index] = f"O 0 0 {(baseline + 0.009).hex()}"
        self.assertEqual(fixture._parse_and_compare("\n".join(lines),
                                                    "Q8_0", "MUL_MAT_ID").status, "pass")
        lines[index] = f"O 0 0 {(baseline + 0.012).hex()}"
        self.assertEqual(fixture._parse_and_compare("\n".join(lines),
                                                    "Q8_0", "MUL_MAT_ID").status, "wrong")

    def test_broken_quant_encoding_is_wrong(self):
        lines = q8_output().splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("A 0 0 "))
        lines[index] = "A 0 0 " + bytes(fixture.ROW_BYTES["Q8_0"]).hex()
        result = fixture._parse_and_compare("\n".join(lines), "Q8_0", "MUL_MAT_ID")
        self.assertEqual(result.status, "wrong")
        self.assertIn("quant encoding mismatch", result.reason)

    def test_q4_and_q5_packed_high_bits(self):
        scales = bytes([1] * 12)
        q4 = struct.pack("<ee", 1, 0) + scales + bytes([0x21] * 128)
        decoded4 = fixture._decode_row("Q4_K", q4)
        self.assertEqual(decoded4[:32], (1.0,) * 32)
        self.assertEqual(decoded4[32:64], (2.0,) * 32)
        high = bytes([0x01] * 32)
        q5 = struct.pack("<ee", 1, 0) + scales + high + bytes([0x21] * 128)
        decoded5 = fixture._decode_row("Q5_K", q5)
        self.assertEqual(decoded5[:32], (17.0,) * 32)
        self.assertEqual(decoded5[32:64], (2.0,) * 32)

    def test_fixture_scales_vary_per_block_and_sub_block(self):
        # DS41-C53: every stored scale must vary, or a scale-layout bug is
        # invisible. Checked on the encoder's stored codes, for the up and the
        # fused gate matrices (expert + 3, row + 5).
        row_scales = set()
        for expert, row in [(e, r) for e in range(2) for r in range(fixture.ROWS)] + \
                [(e + 3, r + 5) for e in range(2) for r in range(fixture.ROWS)]:
            q8 = q8_encoded_row(expert, row)
            q8_scales = [struct.unpack_from("<e", q8, 34 * block)[0] for block in range(8)]
            # amax/127 can coincide for two non-adjacent blocks; never for a pair
            # an adjacent-block mis-read would confuse.
            self.assertGreaterEqual(len(set(q8_scales)), 7, (expert, row))
            self.assertTrue(all(q8_scales[b] != q8_scales[b ^ 1] for b in range(8)))
            for quant in ("Q4_K", "Q5_K"):
                encoded = k_encoded_row(quant, expert, row)
                pairs = [fixture._scale_min(encoded[4:16], group) for group in range(8)]
                self.assertEqual(len({scale for scale, _ in pairs}), 8, (quant, expert, row))
                self.assertEqual(len({minimum for _, minimum in pairs}), 8,
                                 (quant, expert, row))
                # Scales and mins vary independently: a scale/min swap is visible.
                self.assertTrue(any(scale != minimum for scale, minimum in pairs))
                row_scales.add((quant, struct.unpack_from("<ee", encoded)))
        self.assertGreaterEqual(len({v for q, v in row_scales if q == "Q4_K"}), 4)

    def test_scale_layout_permutation_is_caught(self):
        # intake-1825#record: a gfx90a scale-layout bug passed constant-scale
        # fixtures. A kernel that reads per-block scales (Q8_0 d, K-quant
        # sub-scale/min pairs) in the wrong order must fail the oracle at every
        # width, including width 1, where each token touches only 4 blocks.
        for quant in fixture.QUANTS:
            for op in fixture.OPS:
                for width in (1, 2, 8):
                    for order in (ADJACENT_SWAP, REVERSED):
                        with self.subTest(quant=quant, op=op, width=width, order=order):
                            result = fixture._parse_and_compare(
                                permuted_scale_output(quant, op, width, order), quant, op, width)
                            self.assertEqual(result.status, "wrong")
                            self.assertIn(f"{quant} {op} mismatch", result.reason)
                with self.subTest(quant=quant, op=op, order="identity"):
                    self.assertEqual(fixture._parse_and_compare(
                        permuted_scale_output(quant, op, 8, IDENTITY), quant, op, 8).status,
                        "pass")

    def test_constant_scale_fixture_was_blind_to_permutation(self):
        # The control that makes the test above meaningful: under the old
        # constant-amplitude fixture the same mis-read passes.
        with mock.patch.object(fixture, "_source_weight", _constant_scale_weight):
            for quant in fixture.QUANTS:
                with self.subTest(quant=quant):
                    self.assertEqual(fixture._parse_and_compare(
                        permuted_scale_output(quant, "MUL_MAT", 8, ADJACENT_SWAP),
                        quant, "MUL_MAT", 8).status, "pass")

    # --- float weights (float_tinyblas_plan route, DS41 inbox 52) ---------------------

    @staticmethod
    def _float_row(quant: str, expert: int, row: int) -> bytes:
        values = [fixture._source_weight(expert, row, column) for column in range(fixture.K)]
        if quant == "F16":
            return struct.pack(f"<{fixture.K}e", *values)
        if quant == "F32":
            return struct.pack(f"<{fixture.K}f", *values)
        halves = []
        for value in values:  # round-to-nearest-even, as ggml_fp32_to_bf16 does
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            halves.append((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
        return struct.pack(f"<{fixture.K}H", *halves)

    def _float_output(self, quant: str, width: int, tile_shift: int = 0,
                      nudge: float = 0.0) -> str:
        """Probe output; `tile_shift` computes row r from row r+shift (a mis-planned tile)."""
        lines = [f"{fixture.MARKER} {quant} MUL_MAT 256 40 {width} {fixture.ROW_BYTES[quant]}"]
        rows = {(0, row): self._float_row(quant, 0, row) for row in range(fixture.ROWS)}
        lines += [f"A 0 {row} {rows[0, row].hex()}" for row in range(fixture.ROWS)]
        for token in range(width):
            for row in range(fixture.ROWS):
                value = fixture._reference(quant, "MUL_MAT", rows, token,
                                           (row + tile_shift) % fixture.ROWS)
                lines.append(f"O {token} {row} {(value + nudge).hex()}")
        return "\n".join(lines) + "\n"

    def test_float_weights_decode_exactly_and_pass(self):
        for quant in fixture.FLOAT_TYPES:
            for width in (1, 3, 8):
                with self.subTest(quant=quant, width=width):
                    result = fixture._parse_and_compare(self._float_output(quant, width),
                                                        quant, "MUL_MAT", width)
                    self.assertEqual(result.status, "pass", result.detail)
                    metric = json.loads(result.detail.split(" ", 1)[1])
                    self.assertLessEqual(metric["max_quant_abs_error"],
                                         fixture.QUANT_ABS_TOL[quant] / 2)
                    self.assertEqual(metric["output_abs_tol"], fixture.FLOAT_ABS_TOL)

    def test_float_weights_catch_a_misplanned_tile_and_a_tiny_error(self):
        for quant in fixture.FLOAT_TYPES:
            with self.subTest(quant=quant, defect="row tile"):
                self.assertEqual(fixture._parse_and_compare(
                    self._float_output(quant, 3, tile_shift=8), quant, "MUL_MAT", 3).status,
                    "wrong")
            with self.subTest(quant=quant, defect="1e-4 error"):
                # the float bound is analytic (~3.6e-7 worst case), not the 0.01 quant bound
                self.assertEqual(fixture._parse_and_compare(
                    self._float_output(quant, 3, nudge=1e-4), quant, "MUL_MAT", 3).status,
                    "wrong")

    def test_float_weights_are_mul_mat_only_and_not_in_the_default_suite(self):
        self.assertTrue(set(fixture.FLOAT_TYPES).isdisjoint(fixture.QUANTS))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            fixture.check_cpu_quant_suite(Path("/build"), Path("/source"),
                                          quants=("F16",), ops=("MUL_MAT", "MUL_MAT_ID"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            fixture.check_cpu_quant_suite(Path("/build"), Path("/source"),
                                          quants=("F16", "Q8_0"), ops=("MUL_MAT_ID",))

    def test_missing_rows_and_identity_are_untrusted(self):
        output = q8_output()
        with self.assertRaisesRegex(ValueError, "identity"):
            fixture._parse_and_compare(output, "Q8_0", "MUL_MAT")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            fixture._parse_and_compare("\n".join(output.splitlines()[:-1]),
                                       "Q8_0", "MUL_MAT_ID")


if __name__ == "__main__":
    unittest.main()
