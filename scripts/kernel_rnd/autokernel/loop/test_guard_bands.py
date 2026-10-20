"""AKX-ALL-29.5: guard-band logic on CPU buffers (no GPU)."""
import unittest

from autokernel.loop import guard_bands as gb


def arena(**views):
    a = gb.GuardedArena(band_bytes=256)
    for name, extent in views.items():
        a.add_view(name, extent)
    a.seal()
    return a


class GuardBandTests(unittest.TestCase):
    def test_in_extent_writes_leave_every_band_intact(self):
        a = arena(dst=1000, kv=512)
        a.view("dst")[:] = b"\xff" * 1000
        a.view("kv")[:] = b"\x00" * 512
        self.assertEqual(a.check(), [])
        self.assertTrue(gb.verdict_from_receipt(
            a.receipt([]), required_views=("dst", "kv")).passed)

    def test_write_one_byte_past_dst_is_named_as_tail_band(self):
        a = arena(dst=1000)
        a.buf[a.offset("dst") + 1000] ^= 0xFF
        (v,) = a.check()
        self.assertEqual((v.view, v.band, v.first_offset, v.n_bytes), ("dst", "tail", 0, 1))
        verdict = gb.verdict_from_receipt(a.receipt([v]))
        self.assertFalse(verdict.passed)
        self.assertIn("guard_band_violation", verdict.reason)
        self.assertIn("after view 'dst'", verdict.reason)

    def test_write_before_kv_is_named_as_head_band(self):
        a = arena(dst=64, kv=512)
        a.buf[a.offset("kv") - 3] = 0
        (v,) = a.check()
        self.assertEqual((v.view, v.band), ("kv", "head"))
        self.assertEqual(v.first_offset, a.band_bytes - 3)
        self.assertIn("before view 'kv'", v.reason())

    def test_alignment_pad_after_a_short_extent_is_guarded(self):
        a = arena(dst=1000)            # 1000 is not a multiple of 256: pad bytes 1000..1023
        a.buf[a.offset("dst") + 1010] = 0xA5
        self.assertEqual(len(a.check()), 1)

    def test_overwrite_with_constants_or_fill_is_still_caught(self):
        for fill in (0x00, 0xFF, 0x5A, 0xA5):
            a = arena(dst=256)
            end = a.offset("dst") + 256
            a.buf[end:end + 64] = bytes([fill]) * 64
            self.assertEqual(len(a.check()), 1, fill)

    def test_adjacent_views_do_not_share_a_band(self):
        a = arena(dst=256, kv=256)
        self.assertGreaterEqual(a.offset("kv") - (a.offset("dst") + 256), 2 * a.band_bytes)
        a.view("dst")[:] = b"\x11" * 256
        self.assertEqual(a.check(), [])

    def test_missing_or_incomplete_receipt_is_not_a_pass(self):
        self.assertFalse(gb.verdict_from_receipt(None).passed)
        self.assertFalse(gb.verdict_from_receipt({"schema": "other"}).passed)
        only_dst = arena(dst=256).receipt([])
        v = gb.verdict_from_receipt(only_dst, required_views=("dst", "kv"))
        self.assertFalse(v.passed)
        self.assertIn("guard_band_receipt_incomplete", v.reason)
        self.assertIn("kv", v.reason)

    def test_readback_of_the_wrong_size_is_refused(self):
        a = arena(dst=256)
        with self.assertRaises(ValueError):
            a.check(bytes(10))
        a2 = arena(dst=256)
        corrupted = bytearray(a2.buf)
        corrupted[0] ^= 1
        self.assertEqual(len(a2.check(corrupted)), 1)

    def test_construction_refuses_bad_inputs(self):
        with self.assertRaises(ValueError):
            gb.GuardedArena(band_bytes=100)
        a = gb.GuardedArena(256)
        a.add_view("dst", 8)
        with self.assertRaises(ValueError):
            a.add_view("dst", 8)
        with self.assertRaises(ValueError):
            a.add_view("z", 0)
        a.seal()
        with self.assertRaises(RuntimeError):
            a.add_view("late", 8)


if __name__ == "__main__":
    unittest.main()
