import hashlib
import json
import os
import tempfile
import unittest

from autokernel.evaluator import oracle_cases as oc

HDR = "enum ggml_op {\n GGML_OP_NONE = 0,\n GGML_OP_DUP,\n GGML_OP_MUL_MAT,\n GGML_OP_FLASH_ATTN_EXT,\n GGML_OP_COUNT,\n};\n"
WIT = "fast path ran"


class OracleCaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, lines, **over):
        data = ("\n".join(lines) + "\n").encode()
        open(os.path.join(self.tmp.name, "c.txt"), "wb").write(data)
        desc = {"schema": oc.SCHEMA, "op": "FLASH_ATTN_EXT", "name": "hd128", "cases_file": "c.txt",
                "cases_sha256": hashlib.sha256(data).hexdigest(), "n_cases": len(lines), "witness": WIT}
        desc.update(over)
        path = os.path.join(self.tmp.name, "c.json")
        json.dump(desc, open(path, "w"))
        return path

    def test_op_ids_parse(self):
        self.assertEqual(oc.ggml_op_ids(HDR)["FLASH_ATTN_EXT"], 3)

    def test_valid_set_argv_and_witness(self):
        cs = oc.load_case_set(self.write(["3 0 1 -", "3 0 2 -"]), ggml_h_text=HDR)
        self.assertEqual(oc.oracle_argv(cs, backend="ROCm0"),
                         ["test", "-b", "ROCm0", "--test-file", cs.cases_path])
        self.assertEqual(oc.check_witness(cs, "x fast path ran\nfast path ran"), 2)
        self.assertEqual(oc.check_witness(cs, "nothing", expect_ran=False), 0)

    def test_refusals(self):
        bad = [
            (["2 0 1 -"], {}),                       # wrong op id
            (["3 0 1 -"], {"cases_sha256": "0" * 64}),
            (["3 0 1 -"], {"n_cases": 2}),
            (["3 0 1 -"], {"witness": ""}),
            (["3 0 1 -"], {"cases_file": "../c.txt"}),
            (["3 0 1 -"], {"op": "SOFT_MAX"}),        # not in enum
            (["3 0 1 -"], {"extra": 1}),
        ]
        for lines, over in bad:
            with self.subTest(over=over, lines=lines), self.assertRaises(oc.OracleCaseError):
                oc.load_case_set(self.write(lines, **over), ggml_h_text=HDR)

    def test_witness_gates(self):
        cs = oc.load_case_set(self.write(["3 0 1 -"]), ggml_h_text=HDR)
        with self.assertRaises(oc.OracleCaseError):
            oc.check_witness(cs, "stock path only")
        with self.assertRaises(oc.OracleCaseError):
            oc.check_witness(cs, "fast path ran", expect_ran=False)
        with self.assertRaises(oc.OracleCaseError):
            oc.oracle_argv(cs, backend="ROCm0; rm")

    def test_committed_registry_entries_are_self_consistent(self):
        hdr = ("enum ggml_op {\n GGML_OP_NONE = 0,\n"
               + "".join(f" GGML_OP_X{i},\n" for i in range(1, 76))
               + " GGML_OP_FLASH_ATTN_EXT,\n GGML_OP_COUNT,\n};\n")
        found = 0
        for root, _dirs, files in os.walk(oc.REGISTRY_DIR):
            for name in files:
                if name.endswith(".json"):
                    cs = oc.load_case_set(os.path.join(root, name), ggml_h_text=hdr)
                    self.assertEqual(os.path.basename(root), cs.op)
                    found += 1
        self.assertGreaterEqual(found, 1)


if __name__ == "__main__":
    unittest.main()
