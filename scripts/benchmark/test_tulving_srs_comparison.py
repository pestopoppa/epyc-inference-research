"""Prospective stdlib controls; off-host only, not executed locally."""
import copy, unittest
from tulving_srs_comparison import require_comparable_srs, BINS
class SRSComparisonControls(unittest.TestCase):
    def fixture(self):
        return {"scorer_version":2,"gold_binding":"chapter_set_v1","simple_recall_bin_basis":"nb_events",
                "simple_recall_questions":4,"simple_recall_bins":{b:{"count":0 if b=="6+" else 1,"avg_f1":0.0}for b in BINS}}
    def test_matching_populated_bins_preserves_comparison_without_reweighting(self):
        a=self.fixture();b=copy.deepcopy(a);b["simple_recall_bins"]["0"]["count"]=3;b["simple_recall_questions"]=6
        self.assertEqual(require_comparable_srs(a,b)[-1],("0","1","2","3-5"))
    def test_empty_six_plus_vs_populated_refuses(self):
        a=self.fixture();b=copy.deepcopy(a);b["simple_recall_bins"]["6+"]["count"]=1;b["simple_recall_questions"]=5
        with self.assertRaisesRegex(ValueError,"quantities differ"):require_comparable_srs(a,b)
    def test_old_scorer_mixed_missing_or_different_bin_basis_refuses(self):
        for key,value in (("scorer_version",1),("simple_recall_bin_basis","mixed(3/4 nb_events)"),("simple_recall_bin_basis",None),("simple_recall_bin_basis","nb_gt_fallback"),("gold_binding",None)):
            a=self.fixture();b=copy.deepcopy(a);b[key]=value
            with self.assertRaises(ValueError):require_comparable_srs(a,b)
    def test_unknown_missing_bins_and_empty_population_refuse(self):
        for action in ("missing","extra","empty"):
            a=self.fixture();b=copy.deepcopy(a)
            if action=="missing":b["simple_recall_bins"].pop("0")
            elif action=="extra":b["simple_recall_bins"]["other"]={"count":1,"avg_f1":0.0}
            else:
                for bucket in b["simple_recall_bins"].values():bucket["count"]=0
                b["simple_recall_questions"]=0
            with self.assertRaises(ValueError):require_comparable_srs(a,b)
    def test_nonfinite_mean_bool_count_and_inconsistent_population_refuse(self):
        for key,value in (("avg_f1",float("nan")),("avg_f1",float("inf")),("avg_f1",1.1),("count",True),("count",-1),("count",3)):
            a=self.fixture();b=copy.deepcopy(a);b["simple_recall_bins"]["0"][key]=value
            with self.assertRaises(ValueError):require_comparable_srs(a,b)
if __name__=="__main__":unittest.main()
