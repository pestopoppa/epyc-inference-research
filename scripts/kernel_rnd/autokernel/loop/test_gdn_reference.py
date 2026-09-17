"""Hardware-free checks for the independent GDN scalar fixture."""
import math
import unittest

from .gdn_reference import D, H, K, T, MARKER, _parse_output, expected_output


class GDNReferenceTest(unittest.TestCase):
    def test_exact_dyadic_outputs_and_layout(self):
        values = expected_output()
        self.assertEqual(len(values), D*H*T + K*D*D*H)
        # All expected values round-trip through F32, so zero tolerance is valid.
        import struct
        self.assertTrue(all(struct.unpack("f", struct.pack("f", v))[0] == v
                            for v in values))
        self.assertTrue(all(math.isfinite(v) for v in values))
        self.assertNotEqual(values[0], values[D*H])  # time changes the output
        self.assertNotEqual(values[D*H*T:D*H*T + D*D*H],
                            values[D*H*T + D*D*H:])  # snapshots differ

    def test_probe_output_requires_complete_single_receipt(self):
        values = expected_output()
        output = f"{MARKER} {len(values)}\n" + "\n".join(v.hex() for v in values)
        self.assertEqual(_parse_output(output), values)
        with self.assertRaises(ValueError):
            _parse_output(output.rsplit("\n", 1)[0])
        with self.assertRaises(ValueError):
            _parse_output(output + "\n" + output)


if __name__ == "__main__":
    unittest.main()
