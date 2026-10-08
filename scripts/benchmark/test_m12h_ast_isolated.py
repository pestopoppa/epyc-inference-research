"""Offhost-only AST-isolated M-12h controls. No Research package imports."""
import ast, re, string, unicodedata, unittest
from pathlib import Path

def load(source):
    tree = ast.parse(Path(source).read_bytes())
    names = {"_normalise_token", "_tokenise", "_token_f1", "score_f1_list", "_extract_list_from_response"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TulvingEpisodicAdapter")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_f1_for_result")
    method.decorator_list = []
    selected.append(method)
    ns = dict(re=re, string=string, unicodedata=unicodedata)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(source), "exec"), ns)
    return ns["compute_f1_for_result"]

class JudgeMatchControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.score = staticmethod(load(Path(__file__).with_name("tulving_episodic_adapter.py")))
    def result(self, response, gt, judge):
        return self.score(response, {"metadata": {"ground_truth_items": gt, "get_style": "chronological", "retrieval_type": "Times"}}, llm_judge=judge)
    def test_full_match_retained_with_low_judge(self):
        r = self.result("- Alpha\n- Beta", ["Alpha", "Beta"], lambda *a: 0.2)
        self.assertEqual(r["matched_gt_items"], ["Alpha", "Beta"])
        self.assertEqual((r["precision"], r["recall"], r["f1"]), (0.2, 0.2, 0.2))
        self.assertEqual(r["match_source"], "deterministic")
    def test_partial_match_not_completed_by_high_judge(self):
        r = self.result("- Alpha", ["Alpha", "Beta"], lambda *a: 1.0)
        self.assertEqual(r["matched_gt_items"], ["Alpha"])
        self.assertEqual((r["nb_pred"], r["nb_gt"]), (1, 2))
    def test_no_semantic_matches_invented(self):
        r = self.result("- unrelated", ["Alpha", "Beta"], lambda *a: 1.0)
        self.assertEqual(r["matched_gt_items"], [])
        self.assertEqual(r["f1"], 1.0)
    def test_none_judge_identical_to_no_judge(self):
        self.assertEqual(self.result("- Alpha", ["Alpha", "Beta"], lambda *a: None), self.result("- Alpha", ["Alpha", "Beta"], None))
    def test_empty_ground_truth(self):
        r = self.result("None", [], lambda *a: 0.5)
        self.assertEqual((r["nb_pred"], r["nb_gt"], r["matched_gt_items"]), (0, 0, []))
    def test_reversed_response_not_order_claim(self):
        r = self.result("- Beta\n- Alpha", ["Alpha", "Beta"], lambda *a: 0.8)
        self.assertEqual(r["matched_gt_items"], ["Alpha", "Beta"])
        self.assertNotIn("kendall_tau", r)
    def test_duplicate_prediction_does_not_fill_missing_match(self):
        r = self.result("- Alpha\n- Alpha", ["Alpha", "Beta"], lambda *a: 0.9)
        self.assertEqual(r["matched_gt_items"], ["Alpha"])
    def test_judge_called_once_with_native_arguments(self):
        calls = []
        def judge(*args):
            calls.append(args)
            return 0.7
        r = self.result("- Alpha", ["Alpha"], judge)
        self.assertEqual(calls, [(["Alpha"], ["Alpha"], "Times")])
        self.assertEqual((r["get_style"], r["retrieval_type"], r["source"]), ("chronological", "Times", "llm_judge"))

if __name__ == "__main__":
    unittest.main()
