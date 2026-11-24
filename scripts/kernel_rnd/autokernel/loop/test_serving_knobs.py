"""AK-SURFACE-SERVING-1: serving-knob schema, actuator, seed, planner emission."""
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
    # DEFECT 25: --temp is server-wide, so top_p / top_k (declared by `big` only, not by `coder`) are not allowed
    assert base.sampling_allowed["top_p"] == () and base.sampling_allowed["top_k"] == ()
    with pytest.raises(sk.KnobError, match="role does not allow"):
        delta(top_p=0.95).validate(base)


def test_single_role_server_allows_what_its_one_role_declares_and_vram_capacity_is_the_card_ceiling():
    reg = {"server_mode": {"s1": {"model_role": "big", "serving_shape": {"n_ctx": 8}}},
           "roles": {"big": {"generation_defaults": {"top_p": 0.95}, "vram_gib": 36.7}}}
    assert sk.base_from_registry("s1", reg).sampling_allowed["top_p"] == (0.95,)
    assert sk.vram_capacity_gib("s1", reg) == 62.0      # the loop's card ceiling, not the role's 36.7 footprint
    assert sk.vram_capacity_gib("s2", {"server_mode": {"s2": {"device": "ROCm0", "model_role": "x"}}}) == 62.0
    for bad in ("srv", "nope"):                         # CPU server (no device, no vram_gib) / unknown server
        with pytest.raises(sk.KnobError, match="MISSING_INPUT"):
            sk.vram_capacity_gib(bad, REGISTRY)


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


def test_emit_candidate_carries_base_sha_and_patch_sha_from_the_caller(base, recipe):
    s = {"seed_id": "s", "server": "srv", "summary": "x", "falsifier": "y", "arms": [{"arm_id": "ub", "knobs": {"batch": 1024}}]}
    options = typed_planner.build_options(sk.portfolio_rows(s))
    choice = typed_planner.parse_choice({"choice": "serving-knob:s:ub", "statement": "a", "falsifier": "b"}, options)
    sha = "ab" * 20
    cand = sk.emit_candidate(choice, base, argv_of(recipe), base_recipe_hash=recipe.recipe_hash, champion_sha=sha)
    assert cand["base_sha"] == cand["patch_sha"] == sha   # a knob screen has no patch: both are the champion
    with pytest.raises(sk.KnobError, match="champion_sha"):
        sk.emit_candidate(choice, base, argv_of(recipe), base_recipe_hash=recipe.recipe_hash, champion_sha="HEAD")


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


def test_candidate_from_choice_routes_serving_knob_to_emit_candidate_and_leaves_rows_alone(base, recipe):
    row = {"kind": "row", "choice": "m1"}
    assert typed_planner.candidate_from_choice(row) is None
    knob = {"kind": "serving-knob", "choice": "k", "serving_knob": delta(batch=1024, ubatch=1024).to_dict()}
    cand = typed_planner.candidate_from_choice(knob, base=base, recipe_argv=argv_of(recipe),
                                               base_recipe_hash=recipe.recipe_hash)
    assert cand["kind"] == "serving-knob" and sk.read_knobs(cand["argv"])["batch"] == 1024
    with pytest.raises(typed_planner.TypedPlannerRefused, match="needs the registry base"):
        typed_planner.candidate_from_choice(knob)
    bad = knob | {"serving_knob": delta(batch=512).to_dict()}
    with pytest.raises(typed_planner.TypedPlannerRefused, match="not a delta"):
        typed_planner.candidate_from_choice(bad, base=base, recipe_argv=argv_of(recipe),
                                            base_recipe_hash=recipe.recipe_hash)


def test_emit_refuses_a_non_knob_choice_and_an_invalid_delta(base, recipe):
    with pytest.raises(sk.KnobError, match="not a serving-knob"):
        sk.emit_candidate({"kind": "row", "choice": "m1"}, base, argv_of(recipe), base_recipe_hash="0" * 64)
    bad = {"kind": "serving-knob", "choice": "k", "serving_knob": delta(batch=512).to_dict()}
    with pytest.raises(sk.KnobError, match="not a delta"):
        sk.emit_candidate(bad, base, argv_of(recipe), base_recipe_hash="0" * 64)
