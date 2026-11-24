"""AK-SURFACE-SERVING-1: serving-knob schema, actuator, screening evaluator, seed, planner emission."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from . import seed as seed_mod, serving, serving_knobs as sk, typed_planner

SEED_FILE = Path(__file__).parent / "seeds" / "hypotheses" / "14-architect-critic-serving-knobs-20261124.serving-knob.json"
MODEL = "/m/target.gguf"
REGISTRY = {
    "server_mode": {"srv": {
        "slots": 4, "cache_ram": 65536, "ctx_checkpoints": 32, "model_role": "big", "shared_with": ["coder"],
        "serving_shape": {"n_ctx": 393216, "batch": 512, "ubatch": 512, "kv_unified": True,
                          "kv_quant": {"k": "q8_0", "v": "q8_0"}},
        "acceleration": {"draft_max": 7, "drafter_selection": {"drafter": "dflash2"}}, "model": MODEL}},
    "roles": {
        "big": {"generation_defaults": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                "drafters": {"dflash2": {"spec_type": "draft-dflash", "draft_model": "/m/dflash.gguf", "draft_max": 7, "ngld": 99},
                             "mtp": {"spec_type": "draft-mtp", "draft_model": MODEL, "draft_max": 4},
                             "dspark": {"spec_type": "draft-dspark", "draft_model": "/m/dspark.gguf", "draft_max": 4}}},
        "coder": {"generation_defaults": {"temperature": 0.3}}}}


@pytest.fixture
def base():
    return sk.base_from_registry("srv", REGISTRY)


@pytest.fixture
def recipe():
    return serving.Recipe(name="r", model=MODEL, np=4, batch=2048, ubatch=2048, kv_unified=False, cpu_list="184-191",
                          temperature=1.0, spec_decode={"type": "draft-dflash", "drafter": "/m/dflash.gguf",
                                                        "ngld": 99, "draft_n_max": 8})


def argv_of(recipe):
    return recipe.server_argv(Path("/k/gpu"), 8083)


def delta(**knobs):
    return sk.KnobDelta.of("srv", knobs)


# --- schema + actuator ---
def test_base_reads_the_registry_shape(base):
    assert base.values == {"np": 4, "batch": 512, "ubatch": 512, "kv_unified": True, "cache_ram": 65536,
                           "ctx_checkpoints": 32, "spec_draft_n_max": 7, "drafter": "dflash2",
                           "n_ctx": 393216, "ctk": "q8_0", "ctv": "q8_0"}
    assert base.sampling_allowed["temperature"] == (0.3, 1.0)


def test_missing_registry_entry_is_refused_not_invented():
    with pytest.raises(sk.KnobError, match="no serving_shape"):
        sk.base_from_registry("nope", REGISTRY)


def test_render_overlays_the_registry_then_the_delta_over_the_recipe_argv(base, recipe):
    out = sk.render(argv_of(recipe), base, delta(batch=1024, ubatch=1024, ctx_checkpoints=16))
    assert sk.read_knobs(out.base_argv) == {"np": 4, "batch": 512, "ubatch": 512, "kv_unified": True,
                                            "cache_ram": 65536, "ctx_checkpoints": 32, "spec_draft_n_max": 7,
                                            "n_ctx": 393216, "ctk": "q8_0", "ctv": "q8_0"}
    got = sk.read_knobs(out.argv)
    assert (got["batch"], got["ubatch"], got["ctx_checkpoints"], got["cache_ram"]) == (1024, 1024, 16, 65536)
    assert {f for f, _, _ in out.argv_delta} == {"-b", "-ub", "--ctx-checkpoints"}
    assert out.argv[:3] == ("taskset", "-c", "184-191"), "taskset's own -c must not be mistaken for --ctx-size"
    assert out.argv[out.argv.index("-c") + 1] == "184-191" and got["n_ctx"] == 393216
    assert out.delta_digest == delta(batch=1024, ubatch=1024, ctx_checkpoints=16).digest


def test_the_base_argv_is_never_mutated_and_the_render_is_deterministic(base, recipe):
    argv = argv_of(recipe)
    before = list(argv)
    a, b = sk.render(argv, base, delta(np=2)), sk.render(argv, base, delta(np=2))
    assert argv == before and a == b


def test_kv_unified_switch_flips_in_place(base, recipe):
    out = sk.render(argv_of(recipe), base, delta(kv_unified=False))
    assert "--no-kv-unified" in out.argv and "--kv-unified" not in out.argv
    assert out.argv_delta == (("--kv-unified", None, None), ("--no-kv-unified", None, None))


def test_drafter_swap_to_self_draft_drops_md_and_takes_the_drafters_n_max(base, recipe):
    out = sk.render(argv_of(recipe), base, delta(drafter="mtp"))
    assert "-md" not in out.argv and "-ngld" not in out.argv
    assert out.argv[out.argv.index("--spec-type") + 1] == "draft-mtp"
    assert out.argv[out.argv.index("--spec-draft-n-max") + 1] == "4"
    explicit = sk.render(argv_of(recipe), base, delta(drafter="mtp", spec_draft_n_max=6))
    assert explicit.argv[explicit.argv.index("--spec-draft-n-max") + 1] == "6"


@pytest.mark.parametrize("knobs,why", [
    ({"nope": 1}, "unknown serving knob"), ({"batch": 0}, "outside"), ({"np": True}, "not a valid int"),
    ({"kv_unified": "yes"}, "not a valid switch"), ({"batch": 512}, "not a delta"),
    ({"batch": 256, "ubatch": 600}, "exceeds batch"), ({"temperature": 0.6}, "role does not allow"),
    ({"drafter": "ghost"}, "not eligible"), ({"drafter": "dspark"}, "np > 1"),
])
def test_invalid_deltas_are_refused(base, knobs, why):
    with pytest.raises(sk.KnobError, match=why):
        sk.KnobDelta.of("srv", knobs).validate(base)


def test_sampling_allowed_only_at_a_value_a_served_role_declares(base, recipe):
    out = sk.render(argv_of(recipe), base, delta(temperature=0.3))
    assert out.argv[out.argv.index("--temp") + 1] == "0.3"


def test_empty_delta_and_wrong_server_are_refused(base):
    with pytest.raises(sk.KnobError):
        sk.KnobDelta("srv", ())
    with pytest.raises(sk.KnobError, match="base is srv"):
        sk.KnobDelta.of("other", {"np": 2}).validate(base)


def test_delta_round_trips_and_recipe_gains_no_field_and_keeps_its_hash(base, recipe):
    d = delta(np=2, cache_ram=0)
    assert sk.KnobDelta.from_dict(json.loads(json.dumps(d.to_dict()))) == d
    h = recipe.recipe_hash
    sk.render(argv_of(recipe), base, d)
    assert recipe.recipe_hash == h
    assert not [f.name for f in dataclasses.fields(serving.Recipe) if "knob" in f.name]


# --- KV type and context knobs ---
def test_kv_type_and_ctx_render_ctk_ctv_c_over_the_taskset_prefix(base, recipe):
    out = sk.render(argv_of(recipe), base, delta(ctk="f16", ctv="q8_0", n_ctx=262144))
    assert out.argv[:3] == ("taskset", "-c", "184-191")
    got = sk.read_knobs(out.argv)
    assert (got["ctk"], got["ctv"], got["n_ctx"]) == ("f16", "q8_0", 262144)
    assert sk.read_knobs(out.base_argv)["ctk"] == "q8_0"
    assert {f for f, _, _ in out.argv_delta} == {"-ctk", "-c"}, "ctv equals the registry value, so only K and ctx move"
    assert "-ctv" not in {f for f, _, _ in out.argv_delta}


@pytest.mark.parametrize("kind", ["f16", "bf16", "q8_0", "q4_0"])
def test_every_allowed_kv_type_is_accepted(base, kind):
    knobs = {"ctk": kind} if kind != "q8_0" else {"ctv": "f16"}
    sk.KnobDelta.of("srv", knobs).validate(base)


@pytest.mark.parametrize("knobs,why", [
    ({"ctk": "q5_0"}, r"one of \['f16', 'bf16', 'q8_0', 'q4_0'\]"), ({"ctv": "iq4_nl"}, "not a valid enum"),
    ({"ctk": 8}, "not a valid enum"), ({"n_ctx": 0}, "outside"), ({"n_ctx": -4}, "outside"),
    ({"n_ctx": 262145}, "not a multiple of np 4"), ({"np": 5}, "not a multiple of np 5"),
    ({"n_ctx": 393216}, "not a delta"), ({"n_ctx": 262144.0}, "not a valid int"),
])
def test_kv_and_ctx_bounds(base, knobs, why):
    with pytest.raises(sk.KnobError, match=why):
        sk.KnobDelta.of("srv", knobs).validate(base)


def test_np_and_n_ctx_multiple_is_checked_on_the_effective_pair(base):
    sk.KnobDelta.of("srv", {"np": 2, "n_ctx": 100}).validate(base)
    sk.KnobDelta.of("srv", {"np": 3}).validate(base)                     # 393216 % 3 == 0
    with pytest.raises(sk.KnobError, match="multiple"):
        sk.KnobDelta.of("srv", {"np": 2, "n_ctx": 101}).validate(base)


def test_quantized_v_cache_is_refused_when_the_argv_has_flash_attn_off(base, recipe):
    off = [("off" if i and argv_of(recipe)[i - 1] == "-fa" else t) for i, t in enumerate(argv_of(recipe))]
    with pytest.raises(sk.KnobError, match="needs flash-attn"):
        sk.render(off, base, delta(ctv="q4_0"))
    assert sk.read_knobs(sk.render(off, base, delta(ctv="f16", ctk="f16")).argv)["ctv"] == "f16"  # f16 V needs no FA
    with pytest.raises(sk.KnobError, match="needs flash-attn"):
        sk.render(off, base, delta(ctk="f16"))        # the registry V is q8_0: still quantized, so still refused


# --- screening evaluator ---
def launches(tok, itl=100.0, *, n=3, used=30.0, ok=True, warm=1):
    rows = [{"warmup": True, "aggregate_tok_s": tok / 2, "itl_prefill_ms": itl * 3, "output_ok": True,
             "mem_used_gib": used, "mem_capacity_gib": 64.0} for _ in range(warm)]
    rows += [{"warmup": False, "aggregate_tok_s": tok + i, "itl_prefill_ms": itl, "output_ok": ok,
              "mem_used_gib": used, "mem_capacity_gib": 64.0} for i in range(n)]
    return rows


def test_screen_ranks_by_median_aggregate_tok_s_ignoring_warmup():
    r = sk.screen({"control": launches(100), "a": launches(120), "b": launches(110)})
    assert [v["arm_id"] for v in r["ranking"]] == ["a", "b", "control"]
    assert r["best_arm"] == "a" and r["ranking"][0]["median_tok_s"] == 121.0
    assert r["ranking"][0]["vs_control_pct"] == pytest.approx(100 * 20 / 101, abs=1e-3)
    assert r["screening_only"] is True and r["promotable"] is False and not r["rejected"]


def test_screen_best_is_none_when_no_arm_beats_control():
    r = sk.screen({"control": launches(100), "a": launches(90)})
    assert r["best_arm"] is None and r["ranking"][0]["arm_id"] == "control"


@pytest.mark.parametrize("arm,reason", [
    (launches(200, itl=121.0), "itl-regression>20%"),
    (launches(200, ok=False), "incorrect-output"),
    (launches(200, used=65.0), "does-not-fit-memory"),
    (launches(200, n=2), "insufficient-launches:2<3"),
    ([{**r, "status": "oom"} for r in launches(200)], "does-not-fit-memory"),
    ([{k: v for k, v in r.items() if k != "itl_prefill_ms"} for r in launches(200)], "itl-unmeasured"),
    ([{k: v for k, v in r.items() if k != "output_ok"} for r in launches(200)], "incorrect-output"),
    ([{k: v for k, v in r.items() if k != "mem_used_gib"} for r in launches(200)], "memory-unmeasured"),
])
def test_a_faster_arm_failing_a_guard_is_rejected_not_ranked(arm, reason):
    r = sk.screen({"control": launches(100), "fast": arm, "slow": launches(105)})
    assert [v["arm_id"] for v in r["ranking"]] == ["slow", "control"]
    assert reason in {x for v in r["rejected"] for x in v["reasons"]} and r["rejected"][0]["arm_id"] == "fast"


def test_itl_regression_exactly_at_the_limit_passes():
    r = sk.screen({"control": launches(100), "a": launches(130, itl=120.0)})
    assert [v["arm_id"] for v in r["ranking"]] == ["a", "control"]
    assert r["ranking"][0]["itl_regression_pct"] == 20.0


@pytest.mark.parametrize("arms", [
    {"a": launches(100)}, {"control": launches(100, ok=False)}, {"control": launches(100, n=1)},
    {"control": [{**r, "aggregate_tok_s": float("nan")} for r in launches(100)]},
])
def test_an_unsound_or_missing_control_refuses_to_screen(arms):
    with pytest.raises(sk.ScreeningInvalid):
        sk.screen(arms)


# --- seed ---
def test_shipped_seed_loads_and_every_arm_validates_and_renders(base):
    seed = sk.load_seed(SEED_FILE)
    reg = json.loads(json.dumps(REGISTRY))
    reg["server_mode"]["architect_critic"] = reg["server_mode"].pop("srv")
    sk.load_seed(SEED_FILE, registry=reg)           # every arm valid against the server's shape
    assert seed["kind"] == "serving-knob" and len(seed["arms"]) >= 6
    kv = {a["arm_id"]: a["knobs"] for a in seed["arms"] if a["arm_id"].startswith("kv-") and "f16" in a["arm_id"]}
    assert kv == {"kv-f16": {"ctk": "f16", "ctv": "f16"}, "kv-k-f16-v-q8": {"ctk": "f16", "ctv": "q8_0"},
                  "kv-f16-ctx262144": {"ctk": "f16", "ctv": "f16", "n_ctx": 262144},
                  "kv-f16-ub2048": {"ctk": "f16", "ctv": "f16", "batch": 2048, "ubatch": 2048}}
    argv = argv_of(serving.Recipe(name="r", model=MODEL, temperature=1.0))
    b = sk.base_from_registry("architect_critic", reg)
    for a in seed["arms"]:
        assert sk.render(argv, b, sk.KnobDelta.of("architect_critic", a["knobs"])).argv_delta


def test_seed_py_installs_the_seed_as_json_plus_planner_visible_markdown(tmp_path):
    result = seed_mod.install(tmp_path)
    name = SEED_FILE.name
    assert (tmp_path / "inbox" / name).read_bytes() == SEED_FILE.read_bytes()
    md = (tmp_path / "inbox" / name.replace(".serving-knob.json", ".serving-knob.md")).read_text()
    assert "architect-critic-serving-20261124" in md and "never typed" in md and '"ubatch": 1024' in md
    assert any(f.endswith(".serving-knob.md") for f in result["inbox_files"])


def test_a_malformed_seed_fails_install(tmp_path):
    seeds = tmp_path / "seeds"
    shutil_tree(seeds)
    bad = json.loads(SEED_FILE.read_text())
    bad["arms"][0]["knobs"] = {"not_a_knob": 1}
    (seeds / "hypotheses" / "99-bad.serving-knob.json").write_text(json.dumps(bad))
    with pytest.raises(sk.KnobError, match="unknown serving knob"):
        seed_mod.install(tmp_path / "store", seeds=seeds)


def shutil_tree(seeds):
    import shutil
    shutil.copytree(seed_mod.SEEDS, seeds)


@pytest.mark.parametrize("mutate", [lambda s: s.pop("falsifier"), lambda s: s.update(kind="hypothesis"),
                                    lambda s: s.update(arms=[]),
                                    lambda s: s["arms"].append(dict(s["arms"][0])),
                                    lambda s: s["arms"][0].update(arm_id="control")])
def test_seed_structure_is_enforced(tmp_path, mutate):
    s = json.loads(SEED_FILE.read_text())
    mutate(s)
    p = tmp_path / "x.serving-knob.json"
    p.write_text(json.dumps(s))
    with pytest.raises(sk.KnobError):
        sk.load_seed(p)


# --- planner emission ---
def test_planner_emits_a_serving_knob_candidate_with_a_rendered_argv(base, recipe):
    s = {"seed_id": "s", "server": "srv", "summary": "x", "falsifier": "y",
         "arms": [{"arm_id": "ub", "knobs": {"batch": 1024, "ubatch": 1024}}, {"arm_id": "n3", "knobs": {"spec_draft_n_max": 3}}]}
    options = typed_planner.build_options([{"mechanism_id": "m1", "summary": "src"}, *sk.portfolio_rows(s)])
    assert [o["kind"] for o in options] == ["row", "serving-knob", "serving-knob", "control", "control"]
    reply = {"choice": "serving-knob:s:ub", "statement": "bigger batches", "falsifier": "no tok/s gain"}
    choice = typed_planner.parse_choice(reply, options)          # no target_surface/target_symbol needed
    assert choice["kind"] == "serving-knob" and choice["serving_knob"]["knobs"] == {"batch": 1024, "ubatch": 1024}
    cand = sk.emit_candidate(choice, base, argv_of(recipe), base_recipe_hash=recipe.recipe_hash)
    assert cand["kind"] == "serving-knob" and cand["screening_only"] and cand["base_recipe_hash"] == recipe.recipe_hash
    assert sk.read_knobs(cand["argv"])["ubatch"] == 1024 and cand["delta_digest"] == sk.KnobDelta.from_dict(cand["delta"]).digest
    assert {f for f, _, _ in map(tuple, cand["argv_delta"])} == {"-b", "-ub"}


def test_planner_still_requires_free_text_for_plain_rows_and_rejects_bad_serving_rows():
    options = typed_planner.build_options([{"mechanism_id": "m1", "summary": "src"}])
    with pytest.raises(typed_planner.TypedPlannerRefused, match="lacks"):
        typed_planner.parse_choice({"choice": "m1", "statement": "s", "falsifier": "f"}, options)
    with pytest.raises(typed_planner.TypedPlannerRefused, match="serving_knob"):
        typed_planner.build_options([{"mechanism_id": "k", "kind": "serving-knob"}])
    with pytest.raises(typed_planner.TypedPlannerRefused, match="lacks"):
        o = typed_planner.build_options(sk.portfolio_rows({"seed_id": "s", "server": "srv", "summary": "x",
                                                          "arms": [{"arm_id": "a", "knobs": {"np": 2}}]}))
        typed_planner.parse_choice({"choice": "serving-knob:s:a", "statement": "s"}, o)


def test_emit_refuses_a_non_knob_choice_and_an_invalid_delta(base, recipe):
    with pytest.raises(sk.KnobError, match="not a serving-knob"):
        sk.emit_candidate({"kind": "row", "choice": "m1"}, base, argv_of(recipe), base_recipe_hash="0" * 64)
    bad = {"kind": "serving-knob", "choice": "k", "serving_knob": delta(batch=512).to_dict()}
    with pytest.raises(sk.KnobError, match="not a delta"):
        sk.emit_candidate(bad, base, argv_of(recipe), base_recipe_hash="0" * 64)
