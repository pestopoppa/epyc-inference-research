import importlib.util
import copy
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("runner", ROOT / "finetune_bench_runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def dual_section_fixture() -> str:
    return """gguf_ex_read_0: n_tensors: 1
gguf_ex_read_0: tensor[0]: name = duplicate, size = 9, offset = 0, type = f32, n_elts = 1
gguf_ex_read_1: kv[0]: key = qwen35.nextn_predict_layers
gguf_ex_read_1: n_tensors: 1
gguf_ex_read_1: tensor[0]: name = blk.64.attn_k.weight, size = 4, offset = 0, type = f32, n_elts = 1
gguf_ex_read_1: reading tensor 0 data
gguf_ex_read_1: tensor[0]: n_dims = 1, ne = (1, 1, 1, 1), name = blk.64.attn_k.weight, data = 0x1
"""


def tensor(name: str, index: int) -> dict:
    return {
        "index": index,
        "name": name,
        "size": 4,
        "offset": index * 4,
        "type": "f32",
        "n_elts": 1,
        "n_dims": 1,
        "ne": (1, 1, 1, 1),
    }


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.data = runner.load_manifest()

    def test_dual_section_parser_uses_only_read_1(self):
        parsed = runner.parse_gguf_header(
            Path(__file__),
            runner=lambda command: dual_section_fixture(),
        )
        self.assertEqual(parsed["tensor_count"], 1)
        self.assertEqual(parsed["tensors"][0]["name"], "blk.64.attn_k.weight")

    def test_unknown_recipe_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown recipe"):
            runner.render_recipe(self.data, "unknown")

    def test_no_spec_recipe_renders_exact_suites(self):
        plan = runner.render_recipe(self.data, "OBS-27B-V8-NOSPEC-v1")
        self.assertEqual(
            {contrast["id"] for contrast in plan["contrasts"]},
            {"A3-tc-quality", "A3-ff-quality"},
        )
        for contrast in plan["contrasts"]:
            for arm in contrast["arms"]:
                self.assertNotIn("--spec-type", arm["server_argv"])
                self.assertEqual(set(arm["suites"]), {"swe_oracle", "lcb_hard"})
                self.assertIn("--questions-in", arm["suites"]["swe_oracle"]["evaluator_argv"])

    def test_embedded_mtp_only_applies_to_fable_mtp(self):
        plan = runner.render_recipe(
            self.data, "OBS-27B-V8-FABLE-EMBEDDED-MTP-v1"
        )
        arms = {arm["model"]: arm for arm in plan["contrasts"][0]["arms"]}
        self.assertIn("--spec-type", arms["fable_mtp"]["server_argv"])
        self.assertNotIn("--spec-type", arms["fable_non_mtp"]["server_argv"])

    def test_fable_contract_and_tamper(self):
        names = self.data["fable_tensor_contract"]["mtp_only_names"]
        base_rows = [tensor(f"base.{index}", index) for index in range(851)]
        mtp_rows = base_rows + [tensor(name, 851 + index) for index, name in enumerate(names)]
        base = {"tensor_count": 851, "tensors": base_rows, "keys": []}
        mtp = {
            "tensor_count": 866,
            "tensors": mtp_rows,
            "keys": ["qwen35.nextn_predict_layers"],
        }
        rows = iter((base, mtp))
        runner.validate_fable_contract(self.data, header_reader=lambda path: next(rows))
        mtp_rows[0] = {**mtp_rows[0], "type": "q8_0"}
        rows = iter((base, mtp))
        with self.assertRaisesRegex(RuntimeError, "specifications"):
            runner.validate_fable_contract(self.data, header_reader=lambda path: next(rows))

    def test_manifest_component_paths_are_hash_bound(self):
        expected = {path: digest for path, digest in runner.COMPONENT_BINDINGS.values()}
        witness = runner.validate_component_roles(
            self.data, hash_reader=lambda path: expected[str(path)]
        )
        self.assertEqual(set(witness), set(runner.COMPONENT_BINDINGS))
        swapped = copy.deepcopy(self.data)
        swapped["components"]["quality_runner"], swapped["components"]["lcb_scorer"] = (
            swapped["components"]["lcb_scorer"],
            swapped["components"]["quality_runner"],
        )
        with self.assertRaisesRegex(RuntimeError, "role/path"):
            runner.validate_component_roles(swapped, hash_reader=lambda path: expected[str(path)])

    def test_evaluator_argv_is_parseable_and_has_no_port_default_leak(self):
        source = Path(self.data["components"]["quality_runner"]).read_text()
        tree = ast.parse(source)
        accepted = {
            arg.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }
        plan = runner.render_recipe(self.data, "OBS-27B-V8-NOSPEC-v1")
        argv = plan["contrasts"][0]["arms"][0]["suites"]["swe_oracle"]["evaluator_argv"]
        flags = [value for value in argv[1:] if value.startswith("--")]
        self.assertTrue(set(flags) <= accepted)
        self.assertEqual(argv[argv.index("--port") + 1], "18092")
        self.assertNotEqual(argv[argv.index("--port") + 1], "18072")
        self.assertEqual(
            argv[argv.index("--kernel") + 1], "production-consolidated-v8"
        )
        self.assertNotIn("v7-candidate", argv)
        self.assertTrue(argv[argv.index("--output") + 1].endswith(".summary.json"))
        for required in ("--host", "--endpoint", "--concurrency", "--repeats",
                         "--arm", "--binary", "--models", "--questions-in",
                         "--per-question-out"):
            self.assertIn(required, argv)

    def test_fixed_denominators_and_split_budgets(self):
        plan = runner.render_recipe(self.data, "OBS-27B-V8-NOSPEC-v1")
        arm = plan["contrasts"][0]["arms"][0]
        self.assertEqual(self.data["inputs"]["swe_oracle"]["denominator"], 40)
        self.assertEqual(self.data["inputs"]["lcb_hard"]["denominator"], 53)
        self.assertEqual(arm["suites"]["swe_oracle"]["request_kwargs"]["max_tokens"], 3072)
        self.assertEqual(arm["suites"]["lcb_hard"]["request_kwargs"]["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
