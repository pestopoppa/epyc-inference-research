import pytest

from . import integrity_pack as ip


def run(i, seed, inp, out):
    return {"run_id": f"r{i}", "input_seed": seed, "input_digest": inp, "output_digest": out}


def test_fresh_seeds_distinct_deterministic_and_not_suite_seed():
    seeds = ip.plan_fresh_seeds(1234, 50)
    assert len(set(seeds)) == 50 and 1234 not in seeds
    assert seeds == ip.plan_fresh_seeds(1234, 50)
    assert ip.plan_fresh_seeds(1235, 3) != seeds[:3]


@pytest.mark.parametrize("bad", [(True, 3), (-1, 3), (2**64, 3), (5, 0), (5, True)])
def test_fresh_seed_plan_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ip.plan_fresh_seeds(*bad)


def good_runs(n=3, suite=7):
    return [run(i, s, f"in{i}", f"out{i}") for i, s in enumerate(ip.plan_fresh_seeds(suite, n))]


def test_recheck_passes_on_fresh_distinct_runs():
    v = ip.fresh_input_recheck(good_runs(), suite_seed=7)
    assert v.status == ip.PASS and not v.refuses and v.authority == "report_only"


def test_recheck_fails_on_reused_suite_seed():
    runs = good_runs()
    runs[1]["input_seed"] = 7
    v = ip.fresh_input_recheck(runs, suite_seed=7)
    assert v.status == ip.FAIL and "suite seed" in v.findings[0]


def test_recheck_fails_on_shared_seed_between_runs():
    runs = good_runs()
    runs[2]["input_seed"] = runs[0]["input_seed"]
    assert ip.fresh_input_recheck(runs, suite_seed=7).status == ip.FAIL


def test_recheck_fails_on_cached_constant_output():
    runs = good_runs()
    for r in runs:
        r["output_digest"] = "same"
    v = ip.fresh_input_recheck(runs, suite_seed=7)
    assert v.status == ip.FAIL and "cached or constant" in v.findings[0]


def test_recheck_fails_on_nondeterminism():
    runs = good_runs()
    runs.append(run(9, 99, "in0", "other"))
    v = ip.fresh_input_recheck(runs, suite_seed=7)
    assert v.status == ip.FAIL and "nondeterminism" in v.findings[0]


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("input_seed"), lambda r: r.update(input_digest=""),
    lambda r: r.update(output_digest=None), lambda r: r.update(input_seed=True)])
def test_recheck_missing_facts_are_indeterminate_and_refuse(mutate):
    runs = good_runs()
    mutate(runs[0])
    v = ip.fresh_input_recheck(runs, suite_seed=7)
    assert v.status == ip.INDETERMINATE and v.refuses


def test_recheck_too_few_runs_indeterminate():
    assert ip.fresh_input_recheck(good_runs(1), suite_seed=7).status == ip.INDETERMINATE
    assert ip.fresh_input_recheck([], suite_seed=7).refuses


def _diff(line, path="ggml/src/ggml-cuda/k.cu"):
    return f"--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,2 @@\n x\n+{line}\n"


@pytest.mark.parametrize("line,key", [
    ("const char* e = getenv(\"X\");", "environment_probe"),                 # RVP-C6-9
    ("static std::unordered_map<void*, int> memo;", "pointer_memoization"),   # RVP-C6-9
    ("unsigned int m = __ballot(p);", "solx.wave64_32bit_ballot"),            # SOL/wave64
    ("static bool result_computed = false;", "solx.cached_output_across_iterations"),
])
def test_static_blocklist_unions_both_scanners(line, key):
    hits = ip.static_blocklist(_diff(line))
    assert key in hits and ip.static_blocklist_refusal(_diff(line))


def test_static_blocklist_clean_diff_has_no_refusal():
    assert ip.static_blocklist(_diff("float y = x * 2.0f;")) == {}
    assert ip.static_blocklist_refusal(_diff("float y = x * 2.0f;")) is None


def test_v_d_tradeoff_is_pre_registered_and_scope_names_sealed_anchor():
    c = ip.PRE_REGISTERED_COST
    assert "pre-registered cost" in c.statement and c.observed_in_source["TXN-Sched"] == "49% -> 3.9%"
    assert "SEALED anchor" in ip.DIVERGENCE_SEARCH_SCOPE["applies_to"]
    assert ip.DIVERGENCE_SEARCH_SCOPE["status"] == "scoped_not_implemented"
