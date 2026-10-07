"""cpu_fa_schedule (ak-longctx audit 2026-10-04 C2) and the cpu_fa_longctx_v1 FA case
set (C3): route scope on the real anchor tree, the anchor bit-identity gate, the case-set
correctness corpus and the paired perf screen -- all on injected runners, so nothing here
builds, runs a kernel or measures anything.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from . import cpu_fa_reference as fa
from . import cpu_route_witness, gates, loop

ANCHOR = Path("/mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925")
OPS = "ggml/src/ggml-cpu/ops.cpp"
needs_anchor = pytest.mark.skipif(not (ANCHOR / OPS).is_file(),
                                  reason="DS41 anchor tree not present")
FA_SYMBOL = "ggml_compute_forward_flash_attn_ext_f16"


def _head():
    return (ANCHOR / OPS).read_text(encoding="utf-8")


def _patch(head, candidate):
    return "".join(difflib.unified_diff(head.splitlines(True), candidate.splitlines(True), n=0))


def _after_line(text, marker, block):
    at = text.index("\n", text.index(marker)) + 1
    return text[:at] + block + text[at:]


def _before(text, marker, block):
    at = text.index(marker)
    return text[:at] + block + text[at:]


def _gqa_pass_patch(*, pragma=False, in_tiled=False):
    """A GQA-shared KV pass in miniature: a new static helper, a dispatch-body hunk and
    a per-row-walk hunk (the content is irrelevant to scope; the anchor gate judges it)."""
    text = _head()
    helper = ("static int64_t fa_kv_group(int64_t q_head, int64_t rk2) {\n"
              "    return q_head / rk2;\n}\n\n")
    text = _before(text, "static void ggml_compute_forward_flash_attn_ext_f16_one_chunk(", helper)
    text = _after_line(text, "    const bool kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || "
                             "k->type == GGML_TYPE_F16);",
                       "    const int64_t group_rows = neq2 / nek2; (void) group_rows;\n" +
                       ("    #pragma omp barrier\n" if pragma else ""))
    text = _after_line(text, "    const bool write_partials = (partials != nullptr);",
                       "    (void) fa_kv_group;\n")
    if in_tiled:
        start = text.index("static void ggml_compute_forward_flash_attn_ext_tiled(")
        text = text[:start] + _after_line(text[start:], "        int ir0, int ir1) {",
                                          "    int tiled_edit = 0; (void) tiled_edit;\n")
    return text


def _scope(text, symbol=FA_SYMBOL):
    head = _head()
    return gates.affected_op_scope((OPS,), target_surface=OPS, target_symbol=symbol,
                                   source_text=text, pre_source_text=head,
                                   patch_text=_patch(head, text))


# ------------------------------------------------------------------ route scope (C2)

@needs_anchor
def test_fa_scheduling_patch_lands_on_cpu_fa_schedule():
    text = _gqa_pass_patch()
    assert _scope(text) == ("FLASH_ATTN_EXT",)
    route, refusal = gates.admit_cpu_route(OPS, FA_SYMBOL, text, _head(), _patch(_head(), text))
    assert refusal is None and route.route == "cpu_fa_schedule"
    assert route.model_identity and route.identity_repeats == 3 and route.long_identity
    assert route.identity_targets == "peers" and route.new_helpers
    for symbol in route.symbols:
        assert [r.route for r in gates.cpu_source_routes(OPS, symbol)] == ["cpu_fa_schedule"]


@needs_anchor
def test_fa_route_refuses_tiled_body_and_pragma_omp():
    scope = _scope(_gqa_pass_patch(in_tiled=True))
    assert not scope.passed and "outside every admitted body" in scope.reason
    scope = _scope(_gqa_pass_patch(pragma=True))
    assert not scope.passed and ("forbidden pattern" in scope.reason or "refused lexically" in scope.reason)


@needs_anchor
def test_fa_body_markers_resolve_once_on_both_trees():
    route = gates.cpu_route_named("cpu_fa_schedule")
    for tree in (ANCHOR, Path("/mnt/raid0/llm/llama.cpp")):
        if (tree / OPS).is_file():
            bounds = gates._cpu_route_bounds((tree / OPS).read_text(encoding="utf-8"),
                                             "HEAD", route)
            assert not isinstance(bounds, str), bounds
            assert [label for label, *_ in bounds[0]] == [label for label, _ in route.bodies]


def test_existing_routes_are_unaffected():
    names = [r.route for r in gates.CPU_SOURCE_ROUTES]
    # 2026-10-06: the six low-bit ppl_contract routes land right after
    # iqk_dense_dispatch (their natural neighbour: iqk_type_whitelist shares its
    # dispatch boundary, the rest share its iqk/ directory); every PRE-EXISTING
    # route keeps its own relative order around them, which is what this test guards.
    assert names[:-1] == ["dense_q8_tinyblas", "iqk_mmid_dispatch", "iqk_dense_dispatch",
                          "iqk_iquants_dequant", "iqk_legacy_iq4nl", "iqk_type_whitelist",
                          "cpu_repack_mmid", "iqk_kquants_q6_iq4xs_dequant",
                          "iqk_kquants_set_kernels",
                          "cpu_graph_sync", "cpu_graph_sched", "float_tinyblas_plan",
                          "cpu_norm_rowsplit", "cpu_norm_numerics", "cpu_mul_mat_body",
                          "cpu_weight_placement"]
    assert names[-1] == "cpu_fa_schedule"
    assert [r.route for r in gates.cpu_source_routes(OPS, "ggml_compute_forward_rms_norm_f32")] \
        == ["cpu_norm_rowsplit", "cpu_norm_numerics"]
    assert not any(r.long_identity for r in gates.CPU_SOURCE_ROUTES
                   if r.route != "cpu_fa_schedule")
    # the tiled body is not a symbol of any route: still unresolved
    assert gates.cpu_source_routes(OPS, "ggml_compute_forward_flash_attn_ext_tiled") == ()
    # ops.cpp's GATED_DELTA_NET rule is untouched by the new route
    assert gates.cpu_source_routes(OPS, "ggml_compute_forward_gated_delta_net") == ()


def test_witness_defers_to_the_loop(tmp_path):
    witness = cpu_route_witness.WITNESSES["cpu_fa_schedule"]
    assert witness.reference == "fa_anchor_bits" and witness.breakpoint is None
    (tmp_path / "ggml/src/ggml-cpu").mkdir(parents=True)
    (tmp_path / OPS).write_text("")
    recipe = SimpleNamespace(backend="cpu", launch_env={"GGML_IQK": "1"}, template=None,
                             port=None, validate_launch=lambda *a: None, topology_prefix=())
    result = cpu_route_witness.check(tmp_path, resolved_recipe=recipe, source_root=tmp_path,
                                     route="cpu_fa_schedule", source_path=OPS)
    assert result.status == "unavailable" and "cpu_fa_reference" in result.reason


def test_planner_surfaces_and_run_wiring_name_the_route():
    here = Path(__file__).parent
    program = (here / "program.md").read_text(encoding="utf-8")
    assert "**`cpu_fa_schedule`**" in program and fa.CASE_SET_ID in program
    run = (here / "run.py").read_text(encoding="utf-8")
    for needle in ("gates.check_cpu_fa_case_set(", "gates.check_cpu_fa_reference(",
                   "gates.check_cpu_fa_perf_screen(", '"fa_anchor_bits"',
                   "long_identity_targets: list = [None]", "return targets + long_targets(route)",
                   "(cpu_fa_schedule in program.md)"):
        assert needle in run, needle
    assert {"cpu_fa_reference.py", "cpu_fa_reference_probe.cpp"} <= set(loop.ORACLE_MODULES)


# ------------------------------------------------------------------ case set (C3)

def test_case_set_shapes():
    # 2026-10-07 (ak-longctx-identity-oracle, part C): widened from {1, 5} to every
    # query-row count a decode/verify step can present (route admitted text: "Decode/
    # verify steps have N <= 5 query rows") so N=2,3,4 get bit-exact FA coverage too, not
    # only whatever query-row count the identity gate's greedy serving requests happen
    # to produce.
    assert fa.SERVED_QUERY_ROWS == (1, 2, 3, 4, 5)
    q38 = [c for c in fa.CASE_SET if c.name.startswith("q38fn")]
    ds41 = [c for c in fa.CASE_SET if c.name.startswith("ds41")]
    assert len(q38) == 3 * len(fa.SERVED_QUERY_ROWS)
    assert len(ds41) == 4 * len(fa.SERVED_QUERY_ROWS)
    assert {(c.hsk, c.n_kv_heads, c.gqa, c.sinks) for c in q38} == {(256, 2, 12, False)}
    assert {c.kv for c in q38} == {8192, 65536, 131072}
    assert {c.nb for c in q38} == set(fa.SERVED_QUERY_ROWS)
    assert {(c.hsk, c.n_kv_heads, c.n_q_heads, c.sinks) for c in ds41} == {(512, 1, 64, True)}
    assert {c.nb for c in ds41} == set(fa.SERVED_QUERY_ROWS)
    assert all(c.backend_ops for c in fa.CASE_SET)
    assert not any(c.backend_ops for c in fa.GUARD_CASES)
    assert len({c.name for c in fa.CASE_SET}) == len(fa.CASE_SET)  # every name unique


def test_vars_regex_and_patch_block_agree():
    import re
    regex = re.compile(fa.CASE_SET_REGEX)
    q = fa.CASE_SET[0]
    assert q.vars() == ("hsk=256,hsv=256,nh=2,nr23=[12,1],kv=8192,nb=1,mask=1,sinks=0,"
                        "max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,"
                        "type_V=f16,permute=[0,1,2,3]")
    assert all(regex.search(c.vars()) for c in fa.CASE_SET)
    generic = q.vars().replace("kv=8192", "kv=4096")
    assert not regex.search(generic) and not regex.search("x" + q.vars())
    block = fa.backend_ops_patch_block()
    assert f'"{fa.CASE_SET_ID}"' in block and f'"{fa.CASE_SET_ENV}"' in block
    assert all(c.cpp() in block for c in fa.CASE_SET)
    patch = Path("/mnt/raid0/llm/tmp/ak-cpu-fa-route-20261004/"
                 "test-backend-ops-cpu-fa-longctx-v1.patch")
    if patch.is_file():   # the llama-tree patch is generated from CASE_SET
        text = patch.read_text(encoding="utf-8")
        assert fa.THREADS_ENV in text
        # 2026-10-07 (ak-longctx-identity-oracle, part C): CASE_SET widened its query-row
        # coverage to nb 1..5; this on-disk patch predates that widening (it was
        # generated from the pre-widening set) and is therefore a SUBSET of today's
        # CASE_SET, not an exact match, until it is regenerated and reapplied to the
        # llama tree and test-backend-ops rebuilt (reported separately -- no new source
        # line here was produced by writing C++, only by adding Python FaCase entries
        # that the SAME generator (`backend_ops_patch_block`) will turn into one when
        # that follow-up runs). What this still must catch: no line in the on-disk patch
        # may claim a case this module no longer generates (that would be real drift,
        # e.g. an edited shape the patch was never updated for).
        # The patch may also carry an unrelated precedent case set (odd_gqa7_d64_q1_v1)
        # further down the same file; scope the search to THIS module's generated
        # helper body only (every "+"-added line up to its closing "+}").
        marker = "autokernel_add_cpu_fa_longctx_cases"
        assert marker in text, "expected this module's generated helper in the patch"
        body_lines = []
        for ln in text[text.index(marker):].splitlines()[1:]:
            if ln.strip() == "+}":
                break
            body_lines.append(ln)
        patch_lines = {ln.strip().lstrip("+").strip() for ln in body_lines
                      if "test_cases.emplace_back" in ln}
        current_cpp = {c.cpp() for c in fa.CASE_SET}
        assert patch_lines, "expected at least one case line in the on-disk patch"
        orphaned = patch_lines - current_cpp
        assert not orphaned, f"on-disk patch claims case(s) CASE_SET no longer generates: " \
                             f"{sorted(orphaned)[:3]}"
        pending = current_cpp - patch_lines
        assert pending, ("expected the on-disk patch to predate the 2026-10-07 nb "
                        "widening (nb=2,3,4 should be pending regeneration)")


def test_split_kv_semantics_and_configs():
    assert fa.split_kv_enabled({}) and fa.split_kv_enabled({"GGML_FA_SPLIT_KV": "1"})
    assert not fa.split_kv_enabled({"GGML_FA_SPLIT_KV": "0"})
    assert not fa.split_kv_enabled({"GGML_FA_SPLIT_KV": "junk"})
    configs = fa.probe_configs({"GGML_FA_SPLIT_KV": "0"}, 48)
    assert [(v, t) for _l, v, t in configs] == [("0", 7), ("0", 48), ("1", 48)]


# -------------------------------------------- real-mask coverage (N>1, review item 1)

def _write_capture_fixture(root, cases):
    import hashlib
    from .cpu_fa_verifier_record import PROPOSITION, SCHEMA, canonical
    root.mkdir(exist_ok=True)
    (root / "prompt.txt").write_bytes(b"synthetic fixture prompt\n")
    (root / "recipe.json").write_bytes(b'{"mode":"synthetic_fixture"}\n')
    capture_source = root / "capture-src"
    for name in ("src/models/deepseek41.cpp", "src/models/deepseek4-fa-mask-capture.h", "ggml/src/ggml-cpu/ops.cpp"):
        path = capture_source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic capture source\n")
    capture_image = root / "capture-lib.bin"
    capture_image.write_bytes(b"synthetic capture build\n")
    env = {"LD_LIBRARY_PATH": "/synthetic/fixture"}
    manifest = {"schema": "epyc.autokernel.ds41_fa_capture.v1", "architecture": "deepseek41",
                "capture_contract": "ds41_real_mask_n2_5_v1", "source_commit": "a" * 40,
                "model": "/engineering-fixture.gguf", "model_sha256": "b" * 64,
                "recipe_sha256": hashlib.sha256((root / "recipe.json").read_bytes()).hexdigest(),
                "prompt_sha256": hashlib.sha256((root / "prompt.txt").read_bytes()).hexdigest(),
                "run_id": "engineering-fixture", "started_at": "2026-10-07T00:00:00Z",
                "verifier_schema": SCHEMA, "evidence_mode": "synthetic_source_control",
                "decided_proposition": "Synthetic source-conformance fixture only: " + PROPOSITION,
                "source_root": str(capture_source.resolve()), "launch_env": env,
                "launch_env_sha256": hashlib.sha256(canonical(env)).hexdigest(),
                "binary_sha256": hashlib.sha256(capture_image.read_bytes()).hexdigest(),
                "libllama_sha256": hashlib.sha256(capture_image.read_bytes()).hexdigest(),
                "capture_sources": {name: hashlib.sha256((capture_source / name).read_bytes()).hexdigest()
                    for name in ("src/models/deepseek41.cpp", "src/models/deepseek4-fa-mask-capture.h", "ggml/src/ggml-cpu/ops.cpp")},
                "build_images": {str(capture_image.resolve()): hashlib.sha256(capture_image.read_bytes()).hexdigest()}}
    (root / "capture-manifest.json").write_text(json.dumps(manifest))
    for case in cases:
        data = b"\x00" * (case.kv * case.nb * 2)
        fa.real_mask_path(root, case).write_bytes(data)
        digest = 0xcbf29ce484222325
        for byte in data:
            digest = ((digest ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
        metadata = {key: manifest[key] for key in
                    ("source_commit", "model", "model_sha256", "run_id", "recipe_sha256", "prompt_sha256")}
        metadata.update(schema="epyc.autokernel.ds41_fa_mask.v1", type="f16", byte_order="little",
                        layout="token_kv", ne=[case.kv, case.nb, 1, 1], hsk=case.hsk, hsv=case.hsv,
                        n_q_heads=case.n_q_heads, n_kv_heads=case.n_kv_heads, mask_kind="raw_plus_compressed_top_k",
                        mask_hash_algorithm="fnv1a64", mask_bytes=len(data), mask_hash=f"{digest:016x}",
                        captured_at=manifest["started_at"], original_ne=[case.kv, case.nb, 1, 1],
                        original_nb=[2, case.kv*2, case.kv*case.nb*2, case.kv*case.nb*2],
                        slice={"row_start": 0, "row_count": case.nb, "head_index": 0, "stream_index": 0},
                        layer=0, compressed_ratio=1)
        (root / f"{case.name}.mask.json").write_text(json.dumps(metadata))


@pytest.mark.parametrize("defect", ["no_manifest", "no_sidecar", "shape", "model", "run",
                                    "digest", "short", "source", "stale", "slice", "stride"])
def test_real_mask_native_provenance_refuses_before_probe(monkeypatch, tmp_path, defect):
    monkeypatch.setattr(fa, "probe_supports_mask_file", lambda: True)
    cases = fa.ds41_real_mask_cases()
    _write_capture_fixture(tmp_path, cases)
    case = cases[0]
    sidecar = tmp_path / f"{case.name}.mask.json"
    metadata = json.loads(sidecar.read_text())
    if defect == "no_manifest":
        (tmp_path / "capture-manifest.json").unlink()
    elif defect == "no_sidecar":
        sidecar.unlink()
    elif defect == "short":
        fa.real_mask_path(tmp_path, case).write_bytes(b"\x00")
    else:
        key, value = {"shape": ("ne", [case.kv, case.nb, 2, 1]),
                      "model": ("model_sha256", "e" * 64), "run": ("run_id", "another-run"),
                      "digest": ("mask_hash", "0" * 16), "source": ("source_commit", "e" * 40),
                      "slice": ("slice", {"row_start": 1, "row_count": case.nb}),
                      "stride": ("original_nb", [4, case.kv*4, case.kv*case.nb*4, case.kv*case.nb*4]),
                      "stale": ("captured_at", "2026-10-06T23:59:59Z")}[defect]
        metadata[key] = value
        sidecar.write_text(json.dumps(metadata))
    result = fa.check_real_mask_identity(tmp_path / "a", tmp_path / "c", tmp_path / "src",
        capture_dir=tmp_path, check_anchor_identity_fn=lambda *a, **k: pytest.fail("probe ran"))
    assert result.status == "unavailable" and "provenance refused" in result.reason

def test_ds41_real_mask_cases_cover_nb_2_through_5_at_every_served_kv_depth():
    cases = fa.ds41_real_mask_cases()
    assert len(cases) == 4 * len(fa.DS41_REAL_MASK_QUERY_ROWS)
    assert all(c.name.startswith("ds41_realmask_") for c in cases)
    assert {c.nb for c in cases} == set(fa.DS41_REAL_MASK_QUERY_ROWS) == {2, 3, 4, 5}
    assert {c.kv for c in cases} == {4096, 8192, 32768, 65536}
    assert all((c.hsk, c.n_kv_heads, c.n_q_heads, c.sinks, c.mask) ==
              (512, 1, 64, True, "captured") for c in cases)
    # Never registered in test-backend-ops (which has no concept of an external mask),
    # and never swept into the approximate-mask corpora.
    assert not any(c.backend_ops for c in cases)
    assert not (set(c.name for c in cases) & set(c.name for c in fa.CASE_SET))
    assert not (set(c.name for c in cases) & set(c.name for c in fa.GUARD_CASES))
    # 1 is deliberately excluded: live serving already covers N=1 under the REAL mask
    # (model_identity.check, after longctx._oracle_arm strips speculative decoding).
    assert 1 not in {c.nb for c in cases}


def test_probe_supports_mask_file_is_true_for_the_probe_as_shipped():
    assert fa.PROBE_MASK_FILE_FLAG in fa.PROBE.read_text(encoding="utf-8")
    assert fa.probe_supports_mask_file() is True


def test_real_mask_identity_fails_closed_without_probe_support(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "probe_supports_mask_file", lambda: False)
    result = fa.check_real_mask_identity(tmp_path / "a", tmp_path / "c", tmp_path / "src",
                                         capture_dir=tmp_path / "masks")
    assert result.status == "unavailable"
    assert fa.PROBE_MASK_FILE_FLAG in result.reason
    assert "cannot be judged yet" in result.reason


def test_real_mask_identity_fails_closed_without_a_capture_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "probe_supports_mask_file", lambda: True)
    result = fa.check_real_mask_identity(tmp_path / "a", tmp_path / "c", tmp_path / "src")
    assert result.status == "unavailable" and "no capture_dir was given" in result.reason


def test_real_mask_identity_fails_closed_on_a_partial_capture(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "probe_supports_mask_file", lambda: True)
    capture_dir = tmp_path / "masks"
    capture_dir.mkdir()
    cases = fa.ds41_real_mask_cases()
    # Only the first case's mask file exists -- never a silent partial corpus.
    fa.real_mask_path(capture_dir, cases[0]).write_bytes(b"\x00" * (cases[0].kv * cases[0].nb * 2))
    result = fa.check_real_mask_identity(tmp_path / "a", tmp_path / "c", tmp_path / "src",
                                         capture_dir=capture_dir,
                                         check_anchor_identity_fn=lambda *a, **k: pytest.fail("ran"))
    assert result.status == "unavailable"
    assert f"{len(cases) - 1}/{len(cases)}" in result.reason


def test_real_mask_identity_defers_to_anchor_identity_once_everything_is_ready(monkeypatch, tmp_path):
    """Validated native inputs reach the anchor check with a prospective observer."""
    monkeypatch.setattr(fa, "probe_supports_mask_file", lambda: True)
    capture_dir = tmp_path / "masks"
    capture_dir.mkdir()
    cases = fa.ds41_real_mask_cases()
    _write_capture_fixture(capture_dir, cases)
    builds, source = _builds(tmp_path)
    seen = {}

    def fake_check_anchor_identity(anchor_build, candidate_build, source_root, *, cases,
                                   capture_dir=None, anchor_recipe=None, candidate_recipe=None, window=None,
                                   native_record=None):
        assert native_record.request["cases"] == [vars(case) for case in cases]
        seen.update(cases=cases, anchor_build=anchor_build, candidate_build=candidate_build,
                   source_root=source_root, capture_dir=capture_dir)
        return fa.FaResult("pass", "ok")

    result = fa.check_real_mask_identity(
        builds["anchor"], builds["candidate"], source, capture_dir=capture_dir,
        anchor_recipe=_recipe(), candidate_recipe=_recipe(),
        check_anchor_identity_fn=fake_check_anchor_identity)
    assert result.status == "pass"
    assert seen["cases"] == fa.ds41_real_mask_cases()
    assert seen["capture_dir"] == capture_dir
    assert seen["source_root"] == tmp_path / "src"


def test_gates_wraps_real_mask_identity_as_oracle_unavailable_by_default(tmp_path):
    verdict = gates.check_cpu_fa_real_mask_identity(tmp_path / "a", tmp_path / "c",
                                                    tmp_path / "src")
    assert verdict.gate == "oracle_unavailable" and not verdict.passed
    assert "no capture_dir" in verdict.reason


# ------------------------------------------------------------------ anchor identity

def _probe_text(case, threads, *, lib, digests=None, rows=None, input_hash="0" * 16,
                reps=fa.REPS):
    head = (f"{fa.MARKER} {case.hsk} {case.hsv} {case.n_kv_heads} {case.gqa} {case.kv} "
            f"{case.nb} {int(case.sinks)} {case.mask} {case.layout} {threads} {reps} {fa.SEED}")
    n_rows = case.nb * case.n_q_heads
    digests = digests or ["a" * 16] * reps
    rows = rows or [f"{i:016x}" for i in range(n_rows)]
    return "\n".join([head, f"L {lib}", f"I {input_hash}",
                      *(f"D {i} {d}" for i, d in enumerate(digests)),
                      *(f"R {i} {r}" for i, r in enumerate(rows))]) + "\n"


SMALL = fa.FaCase("tiny", 64, 64, 2, 3, 128, 2)


def test_parse_probe_is_strict():
    run = fa.parse_probe(_probe_text(SMALL, 7, lib="/x/libggml-cpu.so"), SMALL, 7)
    assert run.library == "/x/libggml-cpu.so" and len(run.rows) == 12
    with pytest.raises(ValueError):
        fa.parse_probe(_probe_text(SMALL, 8, lib="/x"), SMALL, 7)      # other team
    with pytest.raises(ValueError):
        fa.parse_probe(_probe_text(SMALL, 7, lib="/x") + "junk\n", SMALL, 7)
    with pytest.raises(ValueError):
        fa.parse_probe(_probe_text(SMALL, 7, lib="/x", digests=["a" * 16] * 2), SMALL, 7)


def test_compare_runs_verdicts(tmp_path):
    a_lib, c_lib = tmp_path / "a.so", tmp_path / "c.so"
    a_lib.write_text(""), c_lib.write_text("")
    parse = lambda **kw: fa.parse_probe(_probe_text(SMALL, 7, **kw), SMALL, 7)
    anchor = parse(lib=a_lib)
    assert fa.compare_runs(SMALL, "cfg", anchor, parse(lib=c_lib), a_lib, c_lib) is None
    rows = [f"{i:016x}" for i in range(12)]
    rows[7] = "f" * 16
    wrong = fa.compare_runs(SMALL, "cfg", anchor, parse(lib=c_lib, digests=["b" * 16] * 3,
                                                        rows=rows), a_lib, c_lib)
    assert wrong.status == "wrong" and "token 1, query head 1" in wrong.reason
    race = fa.compare_runs(SMALL, "cfg", anchor,
                           parse(lib=c_lib, digests=["a" * 16, "a" * 16, "c" * 16]),
                           a_lib, c_lib)
    assert race.status == "wrong" and "race" in race.reason
    shaky = fa.compare_runs(SMALL, "cfg", parse(lib=a_lib, digests=["a" * 16, "d" * 16,
                                                                    "a" * 16]),
                            parse(lib=c_lib), a_lib, c_lib)
    assert shaky.status == "unavailable" and "ANCHOR disagrees" in shaky.reason
    other = fa.compare_runs(SMALL, "cfg", anchor, parse(lib=a_lib), a_lib, c_lib)
    assert other.status == "unavailable" and "candidate probe was served by" in other.reason


def _builds(tmp_path):
    builds = {}
    for role in ("anchor", "candidate"):
        bin_dir = tmp_path / role / "bin"
        bin_dir.mkdir(parents=True)
        for name in ("libggml.so", "libggml-base.so", "libggml-cpu.so"):
            (bin_dir / name).write_text("")
        builds[role] = tmp_path / role
    source = tmp_path / "src"
    (source / "ggml/include").mkdir(parents=True)
    (source / "ggml/include/ggml.h").write_text("")
    return builds, source


def _recipe(prefix=("taskset", "-c", "0-47"), split="0"):
    return SimpleNamespace(launch_env={"GGML_FA_SPLIT_KV": split, "LD_LIBRARY_PATH": "/x"},
                           topology_prefix=prefix, template=SimpleNamespace(threads=48))


def _fake_runner(builds, calls, *, cases=(SMALL,), diverge_on=None, capture_dir=None):
    def run(argv, **kwargs):
        calls.append((argv, kwargs.get("env")))
        if argv[0] == "c++":
            Path(argv[argv.index("-o") + 1]).write_bytes(b"synthetic injected compiler output")
            return subprocess.CompletedProcess(argv, 0, "", "")
        binary = next(a for a in argv if "fa-reference-probe-" in a)
        role = binary.rsplit("-", 1)[1]
        positional = argv[argv.index(binary):argv.index(binary) + 13]
        team = int(positional[10])
        case = next(c for c in cases
                    if fa.probe_argv(Path(binary), c, team,
                                     mask_file=fa.real_mask_path(capture_dir, c)
                                     if c.mask == "captured" else None)[1:] ==
                    argv[argv.index(binary) + 1:])
        digest = "a" * 16
        if role == "candidate" and diverge_on and diverge_on(case, kwargs["env"]):
            digest = "b" * 16
        out = _probe_text(case, team, lib=builds[role] / "bin/libggml-cpu.so",
                          digests=[digest] * fa.REPS)
        return subprocess.CompletedProcess(argv, 0, out, "")
    return run


def test_real_mask_paths_reach_both_probe_arms(monkeypatch, tmp_path):
    monkeypatch.setattr(fa, "probe_supports_mask_file", lambda: True)
    builds, source = _builds(tmp_path)
    capture_dir = tmp_path / "masks"
    capture_dir.mkdir()
    cases = fa.ds41_real_mask_cases()
    _write_capture_fixture(capture_dir, cases)
    calls = []

    def check(*args, **kwargs):
        return fa.check_anchor_identity(*args, **kwargs,
            runner=_fake_runner(builds, calls, cases=cases, capture_dir=capture_dir))

    result = fa.check_real_mask_identity(builds["anchor"], builds["candidate"], source,
        capture_dir=capture_dir, anchor_recipe=_recipe(), candidate_recipe=_recipe(),
        check_anchor_identity_fn=check)
    assert result.status == "pass", result
    probes = [argv for argv, _env in calls if argv[0] != "c++"]
    assert len(probes) == len(cases) * 3 * 2
    for role in ("anchor", "candidate"):
        selected = [argv for argv in probes if any(a.endswith(f"probe-{role}") for a in argv)]
        assert {Path(argv[-1]) for argv in selected} == {
            fa.real_mask_path(capture_dir, case) for case in cases}
        assert all(argv[-2] == "--mask-file" for argv in selected)


def test_anchor_identity_passes_and_covers_both_split_settings(tmp_path):
    builds, source = _builds(tmp_path)
    calls = []
    windows = []

    class Window:
        def __enter__(self):
            windows.append("in")

        def __exit__(self, *exc):
            windows.append("out")

    result = fa.check_anchor_identity(builds["anchor"], builds["candidate"], source,
                                      anchor_recipe=_recipe(), candidate_recipe=_recipe(),
                                      cases=(SMALL,), window=Window,
                                      runner=_fake_runner(builds, calls))
    assert result.status == "pass", result
    assert windows == ["in", "out"]
    probes = [(argv, env) for argv, env in calls if argv[0] != "c++"]
    assert len(probes) == 6                                     # 3 configs x 2 arms
    assert {env["GGML_FA_SPLIT_KV"] for _a, env in probes} == {"0", "1"}
    assert all(argv[:3] == ["taskset", "-c", "0-47"] for argv, _e in probes)
    for argv, env in probes:
        role = "anchor" if "probe-anchor" in " ".join(argv) else "candidate"
        assert env["LD_LIBRARY_PATH"].startswith(str(builds[role] / "bin") + ":")
    compiles = [argv for argv, _e in calls if argv[0] == "c++"]
    assert {c[c.index("-L") + 1] for c in compiles} == {str(builds["anchor"] / "bin"),
                                                         str(builds["candidate"] / "bin")}
    receipt = json.loads(result.detail)
    assert receipt["case_set"] == fa.CASE_SET_ID and len(receipt["passes"]) == 3


def test_anchor_identity_flags_a_split_kv_only_divergence(tmp_path):
    builds, source = _builds(tmp_path)
    result = fa.check_anchor_identity(
        builds["anchor"], builds["candidate"], source, anchor_recipe=_recipe(),
        candidate_recipe=_recipe(), cases=(SMALL,),
        runner=_fake_runner(builds, [], diverge_on=lambda c, env: env["GGML_FA_SPLIT_KV"] == "1"))
    assert result.status == "wrong" and "split_kv=1 t48" in result.reason
    verdict = gates.Verdict("reference_comparison" if result.status != "unavailable"
                            else "oracle_unavailable", False)
    assert verdict.gate == "reference_comparison"


def test_anchor_identity_missing_inputs_is_unavailable(tmp_path):
    builds, source = _builds(tmp_path)
    (builds["anchor"] / "bin/libggml-cpu.so").unlink()
    result = fa.check_anchor_identity(builds["anchor"], builds["candidate"], source,
                                      anchor_recipe=_recipe(), candidate_recipe=_recipe(),
                                      runner=lambda *a, **k: pytest.fail("ran"))
    assert result.status == "unavailable"


# ------------------------------------------------------------------ perf screen + corpus

def _perf_text(scale=1.0):
    lines = ["Backend 1/1: CPU"]
    for i, case in enumerate(fa.CASE_SET):
        lines.append(f"  \x1b[1mFLASH_ATTN_EXT({case.vars()}):\x1b[0m        "
                     f"   12 runs - {100.0 * (i + 1) * scale:8.2f} us/run - 1 GFLOP")
    return "\n".join(lines) + "\n"


def test_parse_perf_requires_exactly_the_case_set():
    found = fa.parse_perf(_perf_text())
    assert len(found) == len(fa.CASE_SET) and found[fa.CASE_SET[0].vars()] == 100.0
    with pytest.raises(ValueError):
        fa.parse_perf("\n".join(_perf_text().splitlines()[:-1]))


def _perf_builds(tmp_path, *, literal=True):
    builds = {}
    for role in ("anchor", "candidate"):
        (tmp_path / role / "bin").mkdir(parents=True)
        (tmp_path / role / "bin/test-backend-ops").write_bytes(
            b"\0ELF" + (fa.CASE_SET_ID.encode() if literal else b"") + b"\0")
        builds[role] = tmp_path / role
    return builds


@pytest.mark.parametrize("scale,status", [(0.9, "pass"), (0.99, "slower")])
def test_perf_screen_pairs_abab(tmp_path, scale, status):
    builds = _perf_builds(tmp_path)
    order = []

    def run(argv, **kwargs):
        role = "anchor" if str(builds["anchor"]) in argv[3] else "candidate"
        order.append(role)
        env = kwargs["env"]
        assert env[fa.CASE_SET_ENV] == fa.CASE_SET_ID and env[fa.THREADS_ENV] == "48"
        assert argv[:3] == ["taskset", "-c", "0-47"] and argv[4:8] == ["perf", "-o",
                                                                       "FLASH_ATTN_EXT", "-b"]
        assert argv[-1] == fa.CASE_SET_REGEX
        return subprocess.CompletedProcess(argv, 0, _perf_text(scale if role == "candidate"
                                                               else 1.0), "")

    result = fa.perf_screen(builds["anchor"], builds["candidate"], anchor_recipe=_recipe(),
                            candidate_recipe=_recipe(), runner=run)
    assert order == ["anchor", "candidate", "candidate", "anchor", "anchor", "candidate"]
    assert result.status == status
    assert json.loads(result.detail)["geomean_ratio"] == pytest.approx(scale)


def test_gate_wrappers_skip_without_the_case_set(tmp_path):
    builds = _perf_builds(tmp_path, literal=False)
    verdict = gates.check_cpu_fa_case_set(builds["candidate"], resolved_recipe=_recipe())
    assert verdict.passed and verdict.reason.startswith("SKIPPED")
    verdict = gates.check_cpu_fa_perf_screen(builds["anchor"], builds["candidate"],
                                             anchor_recipe=_recipe(), candidate_recipe=_recipe())
    assert verdict.passed and verdict.gate == "cpu_fa_perf_screen"
    assert verdict.reason.startswith("SKIPPED")


def test_case_set_corpus_selects_exactly_the_set(tmp_path, monkeypatch):
    builds = _perf_builds(tmp_path)
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, env=kwargs["env"])
        total = seen.get("total", len(fa.CASE_SET))
        out = (f"Testing 1 devices\n\nBackend 1/1: CPU\n  {total}/{total} tests passed\n"
               "  Backend CPU: OK\n1/1 backends passed\n")
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(gates.subprocess, "run", run)
    recipe = SimpleNamespace(backend="cpu", template=None, port=None,
                             validate_launch=lambda *a: None,
                             launch_env={"GGML_IQK": "1"}, topology_prefix=("taskset", "-c", "1"))
    verdict = gates.check_cpu_fa_case_set(builds["candidate"], resolved_recipe=recipe)
    assert verdict.passed and verdict.gate == "correctness", verdict
    assert seen["argv"][-2:] == ["-p", fa.CASE_SET_REGEX]
    assert seen["env"][fa.CASE_SET_ENV] == fa.CASE_SET_ID and seen["env"]["GGML_IQK"] == "1"
    seen["total"] = 3
    verdict = gates.check_cpu_fa_case_set(builds["candidate"], resolved_recipe=recipe)
    assert not verdict.passed and verdict.gate == "oracle_unavailable"
    # without a case set, op_correctness is unchanged: no -p, no override
    gates.op_correctness(builds["candidate"], op="FLASH_ATTN_EXT", backend="CPU",
                         resolved_recipe=recipe)
    assert "-p" not in seen["argv"] and fa.CASE_SET_ENV not in seen["env"]


# ------------------------------------------------------------ scope hardening (2026-10-04)

@needs_anchor
@pytest.mark.parametrize("inserted, match", [
    ("    _Pragma(\"omp barrier\")\n", "forbidden pattern"),
    ("    #define ggml_vec_dot_f16 fa_evil_dot\n", "forbidden pattern"),
    ("    #undef MIN\n", "forbidden pattern"),
    ("    #include \"fa-evil.h\"\n", "forbidden pattern"),
    ("    omp_set_num_threads(1);\n", "forbidden pattern"),
])
def test_fa_route_refuses_preprocessor_and_openmp_spellings_inside_a_body(inserted, match):
    """A macro defined inside an FA body stays in force to the end of ops.cpp (the tiled
    body, ssm_conv and gated_delta_net follow it); `_Pragma` and omp_* calls are OpenMP
    in another spelling."""
    text = _after_line(_head(), "    const bool write_partials = (partials != nullptr);",
                       inserted)
    scope = _scope(text)
    # round-7 resolution C: the lexical refusal (no `#` outside a literal) may fire
    # before the route's own forbidden pattern; either is a refusal.
    assert not scope.passed and (match in scope.reason or "refused lexically" in scope.reason), scope.reason


@needs_anchor
def test_fa_route_new_static_objects_must_be_const_and_call_free():
    marker = "static void ggml_compute_forward_flash_attn_ext_f16_one_chunk("
    table = _before(_head(), marker, "static const int fa_kv_groups[4] = {1, 2, 3, 4};\n\n")
    assert _scope(table) == ("FLASH_ATTN_EXT",)
    for bad, match in (("static int fa_env = (setenv(\"GGML_FA\", \"1\", 1), 0);\n\n", "const"),
                       ("static struct fa_init { fa_init() {} } fa_init_once;\n\n", "const"),
                       ("static const float fa_scale = fa_compute_scale();\n\n", "call")):
        scope = _scope(_before(_head(), marker, bad))
        # round-7 resolution C: the lexical refusal (no `#` outside a literal) may fire
    # before the route's own forbidden pattern; either is a refusal.
    assert not scope.passed and (match in scope.reason or "refused lexically" in scope.reason), scope.reason
