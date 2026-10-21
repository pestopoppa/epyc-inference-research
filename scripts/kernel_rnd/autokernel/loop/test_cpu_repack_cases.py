"""Unit tests for the CPU_REPACK MUL_MAT_ID gate (`cpu_repack_cases.py`, 2026-10-21) and its
gates.py / run.py wiring (`check_cpu_repack_case_set`, `ppl_contract_op_nmse`'s
`cpu_repack_manifest`, the `cpu_repack_gemv` route).

Pure Python: probe output is synthesised here. The real-hardware half -- the anchor
calibration, the broken-kernel fixture modes and the GGML_IQK=1 engagement negative -- runs
in `fixture_regression` from the runner under a build-role claim.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest import mock

import pytest

from . import cpu_repack_cases as crc
from . import gates
from . import served_shape_cases as ssc

RESEARCH_ROOT = Path(__file__).resolve().parents[4]
W5A_TENSORS = RESEARCH_ROOT / "data/ds41-1017-w5-20261021/w5a/gguf_tensors.json"
DS41_MODEL = ssc.LANE_PROFILES["ds41"].model
Q38FN_MODEL = ssc.LANE_PROFILES["q38fn"].model
ANCHOR_1017 = Path("/mnt/raid0/llm/worktrees/llama-ds41-1017-ak-20261021")


def _served_recipe():
    return SimpleNamespace(topology_prefix=("taskset", "-c", "0-95"),
                           command_argv=("llama-server", "-t", "48", "-tb", "48"),
                           launch_env={"GGML_IQK": "1"}, backend="cpu")


# ------------------------------------------------------------------ synthetic probe output
def _case_row(case, **over):
    row = {"kind": "case", "case": case.name, "type": case.type_a, "k": case.k, "m": case.m,
           "n_as": case.n_as, "n_used": case.n_used, "bcast": int(case.bcast),
           "repack_buft": "CPU_REPACK", "extra": True, "layout_changed": True,
           "witness_changed": 3, "witness_unexpected": 0, "witness_restored": True,
           "engaged": True, "raw_fnv64": f"{abs(hash(case.name)) % (1 << 60):016x}",
           "decode_maxdiff": 0.0, "decode_inexact": 0, "witness_block": 0}
    row.update(over)
    return row


def _row(case, n, **over):
    row = {"kind": "row", "case": case.name, "n": n, "input_fnv64": f"{n:016x}",
           "finite_repack": True, "finite_plain": True, "nmse_repack": 1.2e-5,
           "maxrel_repack": 1.5e-2, "nmse_plain": 1.1e-5, "maxrel_plain": 1.4e-2,
           "nmse_repack_vs_plain": 1e-7, "ref_rms": 3.0}
    row.update(over)
    return row


def _output(cases, *, fixture="none", case_over=None, row_over=None, drop_rows=(),
            end=True, header_over=None):
    header = {"kind": "header", "marker": crc.PROBE_MARKER, "case_set": crc.CASE_SET_ID,
              "types": crc.PROBE_TYPES, "fixture": fixture, "threads": 48, "ggml_iqk": "1",
              "repack_buft": True, "n_cases": len(cases)}
    header.update(header_over or {})
    lines = [header]
    for case in cases:
        lines.append(_case_row(case, **(case_over or {}).get(case.name, {})))
        for n in crc.WIDTHS:
            if (case.name, n) in drop_rows:
                continue
            lines.append(_row(case, n, **(row_over or {}).get((case.name, n), {})))
    if end:
        lines.append({"kind": "end", "n_cases": len(cases)})
    return "\n".join(json.dumps(line) for line in lines) + "\n"


def _manifest(lane="ds41"):
    return crc.manifest_from_measurement(crc.parse_output(_output(crc.all_cases(lane))),
                                         lane=lane, threads=48)


def _write_manifest(tmp_path, body=None):
    path = tmp_path / "store" / crc.STORE_SUBDIR / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body if body is not None else _manifest()), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- case set
def test_ds41_lane_cases_are_real_1017_mxfp4_experts_at_their_gguf_offsets():
    """The W5a set, cross-checked against the header-only tensor index W5a committed."""
    cases = crc.LANE_CASES["ds41"]
    assert [c.type_a for c in cases] == ["MXFP4"] * 4
    profile = ssc.LANE_PROFILES["ds41"]
    if not W5A_TENSORS.is_file():
        pytest.skip("W5a tensor index not in this checkout")
    index = json.loads(W5A_TENSORS.read_text(encoding="utf-8"))
    for case in cases:
        entry = index[case.tensor]
        assert case.src == "gguf" and Path(case.path).name == entry["file"]
        assert case.offset == entry["offset"] and entry["type"] == "MXFP4"
        assert entry["shape"] == [case.k, case.m, case.n_exp_total] == \
            [case.k, case.m, profile.expert_count]
        assert case.n_used == profile.expert_used and len(case.expert_ids) == case.n_as == 8
        assert case.bcast == ("down" not in case.tensor)
        assert entry["nbytes"] == case.n_exp_total * case.m * case.k // 32 * 17


def test_shared_cases_cover_the_other_two_template_instantiations():
    types = {c.type_a for c in crc.SHARED_CASES}
    assert types == {"Q4_0", "IQ4_NL"}
    # Q4_0 is diverted from the repack under GGML_IQK=1, so its cases run with it off.
    assert all(c.ggml_iqk == "0" for c in crc.SHARED_CASES if c.type_a == "Q4_0")
    assert all(c.ggml_iqk is None for c in crc.SHARED_CASES if c.type_a == "IQ4_NL")
    q38 = ssc.LANE_PROFILES["q38fn"]
    assert any((c.k, c.m, c.n_used) == (q38.expert_ff, q38.hidden, q38.expert_used)
               for c in crc.SHARED_CASES if c.type_a == "IQ4_NL")
    assert all(c.tier == "shared" and c.src == "synth" for c in crc.SHARED_CASES)


def test_widths_cover_the_served_widths_and_the_prefill_shapes():
    assert set(ssc.SERVED_WIDTHS) <= set(crc.WIDTHS) and max(crc.WIDTHS) >= 128


def test_repack_routes_are_exactly_the_no_op_oracle_routes_and_exist():
    assert gates._REPACK_ROUTES == crc.REPACK_ROUTES == gates.PPL_CONTRACT_NO_OP_ORACLE
    for name in crc.REPACK_ROUTES:
        route = gates.cpu_route_named(name)
        assert route is not None and route.numerics == "ppl_contract"
        assert route.path in crc.REPACK_PATHS
        assert gates.ppl_contract_relevant(route.path)


def test_tsv_lines_carry_the_probes_fourteen_columns():
    for case in crc.all_cases("ds41"):
        fields = case.tsv().split("\t")
        assert len(fields) == 14
        assert fields[7] == ",".join(str(w) for w in crc.WIDTHS)
        if case.src == "synth":
            assert fields[9] == "-" and fields[10] == "0" and fields[12] == "-"


def test_probe_source_carries_the_literals_and_keeps_the_fixture_behind_its_macro():
    text = crc.PROBE.read_text(encoding="utf-8")
    for literal in (crc.PROBE_MARKER, crc.CASE_SET_ID, crc.PROBE_TYPES):
        assert f'"{literal}"' in text
    defined = dict(re.findall(r"#(?:el)?if AK_REPACK_FIXTURE == (\d)\n#define AK_FIXTURE_NAME \"(\w+)\"",
                              text))
    assert {int(k): v for k, v in defined.items()} == crc.FIXTURE_MODES
    # every code line naming a fixture literal sits inside an `#if defined(AK_REPACK_FIXTURE)`
    depth, inside = 0, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#if"):
            inside.append("AK_REPACK_FIXTURE" in stripped)
        elif stripped.startswith("#endif"):
            inside.pop()
        elif crc.FIXTURE_LITERAL in line and not stripped.startswith("//"):
            assert any(inside), f"fixture literal outside the fixture macro: {line!r}"
    assert not inside and depth == 0


def test_touches_repack():
    assert crc.touches_repack(["ggml/src/ggml-cpu/arch/x86/repack.cpp"])
    assert crc.touches_repack(("ggml/src/ggml-cpu/repack.cpp", "x"))
    assert not crc.touches_repack(["ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp"])
    assert not crc.touches_repack(None)


# ------------------------------------------------------------------------------- bounds
def test_maxrel_bound_is_anchor_times_factor_floored_and_capped():
    assert crc.maxrel_bound(0.01) == pytest.approx(0.03)
    assert crc.maxrel_bound(0.0) == crc.MAXREL_FLOOR
    assert crc.maxrel_bound(0.2) == crc.MAXREL_CAP
    for bad in (crc.MAXREL_CAP, float("nan"), -1.0, True):
        with pytest.raises(ValueError):
            crc.maxrel_bound(bad)


def test_calibration_uses_the_served_shape_nmse_formula():
    manifest = _manifest()
    row = manifest["cases"][0]["rows"][0]
    assert row["nmse_bound"] == ssc.tightened_nmse_bound(1.2e-5, factor=3.0)
    assert row["maxrel_bound"] == pytest.approx(4.5e-2)
    assert [e["case"]["name"] for e in manifest["cases"]] == \
        [c.name for c in crc.all_cases("ds41")]
    assert manifest["probe_sha256"] == crc.probe_sha256()


@pytest.mark.parametrize("over,needle", [
    ({"extra": False}, "did not engage"),
    ({"layout_changed": False}, "did not engage"),
    ({"witness_changed": 0}, "did not engage"),
    ({"decode_maxdiff": 1e-3}, "independent decode"),
])
def test_calibration_refuses_an_anchor_that_cannot_back_a_bound(over, needle):
    first = crc.LANE_CASES["ds41"][0]
    parsed = crc.parse_output(_output(crc.all_cases("ds41"), case_over={first.name: over}))
    with pytest.raises(ValueError, match=needle):
        crc.manifest_from_measurement(parsed, lane="ds41")


def test_calibration_refuses_an_anchor_at_the_generic_cap_and_a_fixture_build():
    first = crc.LANE_CASES["ds41"][0]
    parsed = crc.parse_output(_output(crc.all_cases("ds41"),
                                      row_over={(first.name, 4): {"nmse_repack": 6e-4}}))
    with pytest.raises(ValueError, match="generic"):
        crc.manifest_from_measurement(parsed, lane="ds41")
    parsed = crc.parse_output(_output(crc.all_cases("ds41"), fixture=crc.FIXTURE_MODES[1]))
    with pytest.raises(ValueError, match="FIXTURE"):
        crc.manifest_from_measurement(parsed, lane="ds41")
    parsed = crc.parse_output(_output(crc.all_cases("ds41"), drop_rows={(first.name, 512)}))
    with pytest.raises(ValueError, match="no row"):
        crc.manifest_from_measurement(parsed, lane="ds41")


# ----------------------------------------------------------------------------- manifest
def test_manifest_round_trips_and_refuses_anything_stale(tmp_path):
    path = _write_manifest(tmp_path)
    assert crc.load_manifest(path, lane="ds41")["lane"] == "ds41"
    with pytest.raises(crc.ManifestRefused, match="no CPU_REPACK manifest"):
        crc.load_manifest(tmp_path / "missing.json", lane="ds41")
    good = _manifest()
    for mutate, needle in (
            (lambda b: b.update(lane="q38fn"), "lane"),
            (lambda b: b.update(case_set="other"), "case set"),
            (lambda b: b.update(probe_sha256="0" * 64), "probe source"),
            (lambda b: b.update(schema="x"), "schema"),
            (lambda b: b["cases"].pop(), "case list"),
            (lambda b: b["cases"][0]["case"].update(offset=1), "case list"),
            (lambda b: b["cases"][0]["rows"].pop(), "widths"),
            (lambda b: b["cases"][0]["rows"][0].update(nmse_bound=0.0), "nmse_bound"),
            (lambda b: b["cases"][0].pop("raw_fnv64"), "malformed")):
        body = json.loads(json.dumps(good))
        mutate(body)
        bad = tmp_path / f"bad-{needle.replace(' ', '_')}.json"
        bad.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(crc.ManifestRefused, match=needle):
            crc.load_manifest(bad, lane="ds41")


def test_a_lane_without_a_case_set_has_none():
    assert crc.lane_has_cases("ds41") and not crc.lane_has_cases("q38fn")
    assert not crc.lane_has_cases(None)
    with pytest.raises(ValueError):
        crc.all_cases("q38fn")


# -------------------------------------------------------------------------------- judge
def _judge(cases=None, **kw):
    cases = cases or crc.LANE_CASES["ds41"]
    return crc.judge(_manifest(), crc.parse_output(_output(cases, **kw)), cases)


def test_judge_passes_the_anchor_itself():
    result = _judge()
    assert result.status == "pass", result.reason
    assert len(result.records) == 4 * len(crc.WIDTHS)


def test_judge_refuses_an_nmse_violation_as_numerical():
    case = crc.LANE_CASES["ds41"][1]
    result = _judge(row_over={(case.name, 1): {"nmse_repack": 2e-4}})
    assert result.status == "wrong" and "nmse" in result.reason and case.name in result.reason


def test_judge_catches_a_localized_fault_aggregate_nmse_cannot_see():
    """One wrong block scale on one row: NMSE stays within the anchor bound, max error not."""
    case = crc.LANE_CASES["ds41"][0]
    result = _judge(row_over={(case.name, 32): {"nmse_repack": 1.3e-5, "maxrel_repack": 0.4}})
    assert result.status == "wrong" and "maxrel" in result.reason and "nmse 1" not in result.reason


def test_judge_refuses_non_finite_repack_output_as_numerical():
    case = crc.LANE_CASES["ds41"][2]
    result = _judge(row_over={(case.name, 8): {"finite_repack": False, "nmse_repack": -1.0}})
    assert result.status == "wrong" and "non-finite" in result.reason


@pytest.mark.parametrize("over,needle", [
    ({"extra": False, "engaged": False}, "tensor->extra unset"),
    ({"repack_buft": "CPU"}, "not in CPU_REPACK"),
    ({"layout_changed": False}, "equal the source bytes"),
    ({"witness_changed": 0}, "changed no output"),
    ({"witness_unexpected": 2}, "outside expert 0"),
    ({"witness_restored": False}, "did not reproduce"),
    ({"engaged": False}, "not engaged"),
    ({"raw_fnv64": "deadbeefdeadbeef"}, "weight bytes"),
    ({"decode_inexact": 3}, "independent decoder"),
])
def test_judge_fails_closed_when_the_repack_path_is_not_proven(over, needle):
    case = crc.LANE_CASES["ds41"][3]
    result = _judge(case_over={case.name: over})
    assert result.status == "unavailable" and needle in result.reason
    if "extra" in over or "layout_changed" in over or "witness" in str(over):
        assert "did not engage" in result.reason


def test_judge_fails_closed_on_missing_extra_or_mismatched_rows():
    case = crc.LANE_CASES["ds41"][0]
    assert "no row" in _judge(drop_rows={(case.name, 3)}).reason
    assert _judge(drop_rows={(case.name, 3)}).status == "unavailable"
    assert "inputs" in _judge(row_over={(case.name, 2): {"input_fnv64": "x"}}).reason
    assert "plain-arm" in _judge(row_over={(case.name, 2): {"nmse_plain": 1e-2}}).reason
    assert "did not finish" in _judge(end=False).reason
    parsed = crc.parse_output(_output(crc.LANE_CASES["ds41"]) + json.dumps(
        _row(case, 7)) + "\n")
    result = crc.judge(_manifest(), parsed, crc.LANE_CASES["ds41"])
    assert result.status == "unavailable" and "not in the manifest" in result.reason
    missing = crc.parse_output(_output(crc.LANE_CASES["ds41"][:3]))
    result = crc.judge(_manifest(), missing, crc.LANE_CASES["ds41"])
    assert result.status == "unavailable" and "no such case" in result.reason


def test_judge_refuses_a_fixture_or_foreign_probe_header():
    result = _judge(fixture=crc.FIXTURE_MODES[2])
    assert result.status == "unavailable" and "FIXTURE" in result.reason
    assert _judge(fixture=crc.FIXTURE_MODES[2], header_over={}).status == "unavailable"
    assert "is not" in _judge(header_over={"case_set": "served_shape_lowbit_v1"}).reason
    assert "no CPU_REPACK buffer" in _judge(header_over={"repack_buft": False}).reason
    cases = crc.LANE_CASES["ds41"]
    allowed = crc.judge(_manifest(), crc.parse_output(_output(cases, fixture=crc.FIXTURE_MODES[2])),
                        cases, allow_fixture=True)
    assert allowed.status == "pass"   # the fixture flag alone is not a numerical verdict


def test_parse_output_refuses_duplicates_and_garbage():
    case = crc.LANE_CASES["ds41"][0]
    text = _output([case])
    with pytest.raises(ValueError, match="twice"):
        crc.parse_output(text + json.dumps(_row(case, 1)) + "\n")
    with pytest.raises(ValueError, match="not JSON"):
        crc.parse_output("nope\n")
    with pytest.raises(ValueError, match="unknown kind"):
        crc.parse_output('{"kind": "x"}\n')


# -------------------------------------------------------------------------- binary / run
def test_binary_refusal_requires_the_literals_and_refuses_a_fixture_build(tmp_path):
    good = tmp_path / "good"
    good.write_bytes(b"\0".join(x.encode() for x in (crc.PROBE_MARKER, crc.CASE_SET_ID,
                                                       crc.PROBE_TYPES)))
    assert crc.binary_refusal(good) is None
    lacking = tmp_path / "lacking"
    lacking.write_bytes(crc.PROBE_MARKER.encode())
    assert "lacks" in crc.binary_refusal(lacking)
    fixture = tmp_path / "fixture"
    fixture.write_bytes(good.read_bytes() + b"\0" + crc.FIXTURE_MODES[1].encode())
    assert "TEST FIXTURE" in crc.binary_refusal(fixture)
    assert "unreadable" in crc.binary_refusal(tmp_path / "absent")


def test_run_probe_runs_one_process_per_ggml_iqk_group(tmp_path):
    calls = []

    def fake_run(argv, *, capture_output, text, timeout, env):
        tsv, out = Path(argv[-2]), Path(argv[-1])
        names = [line.split("\t")[0] for line in tsv.read_text().splitlines()]
        calls.append((argv[:3], env.get("GGML_IQK"), env["LD_LIBRARY_PATH"], names))
        by = {c.name: c for c in crc.all_cases("ds41")}
        out.write_text(_output([by[n] for n in names]))
        return SimpleNamespace(returncode=0, stderr="")

    with mock.patch.object(crc.subprocess, "run", fake_run):
        parsed = crc.run_probe(Path("/bin/probe"), Path("/build"), crc.all_cases("ds41"),
                               threads=48, launch_env={"GGML_IQK": "1"},
                               topology_prefix=("taskset", "-c", "0-95"), workdir=tmp_path)
    assert len(parsed["cases"]) == 8 and len(parsed["headers"]) == 2
    groups = {iqk: names for _argv, iqk, _ld, names in calls}
    assert sorted(groups["0"]) == sorted(c.name for c in crc.SHARED_CASES if c.type_a == "Q4_0")
    assert len(groups["1"]) == 6
    assert all(argv == ["taskset", "-c", "0-95"] for argv, *_ in calls)
    assert all(ld.startswith("/build/bin") for _a, _i, ld, _n in calls)

    def failing(argv, **kw):
        return SimpleNamespace(returncode=4, stderr="compute failed")
    with mock.patch.object(crc.subprocess, "run", failing):
        with pytest.raises(RuntimeError, match="exit 4"):
            crc.run_probe(Path("/bin/probe"), Path("/build"), crc.LANE_CASES["ds41"],
                          threads=1, workdir=tmp_path)


def _stub_compile(tmp_path, literal_extra=b""):
    def compile_(build, out):
        out.write_bytes(b"\0".join(x.encode() for x in (crc.PROBE_MARKER, crc.CASE_SET_ID,
                                                          crc.PROBE_TYPES)) + literal_extra)
        return out, ""
    return compile_


def test_check_case_set_fails_closed_on_every_missing_piece(tmp_path):
    manifest = _write_manifest(tmp_path)
    ok_run = lambda binary, build, cases, **kw: crc.parse_output(_output(cases))  # noqa: E731
    common = dict(lane="ds41", include_shared=False, threads=48)
    assert crc.check_case_set(tmp_path, manifest, _compile=_stub_compile(tmp_path), _run=ok_run,
                              **common).status == "pass"
    unknown = crc.check_case_set(tmp_path, manifest, lane="q38fn", include_shared=False,
                                 threads=48)
    assert unknown.status == "unavailable" and "no CPU_REPACK case set" in unknown.reason
    absent = crc.check_case_set(tmp_path, tmp_path / "none.json", **common)
    assert absent.status == "unavailable" and "manifest refused" in absent.reason
    no_compile = crc.check_case_set(tmp_path, manifest,
                                    _compile=lambda b, o: (None, "compile failed"), **common)
    assert no_compile.status == "unavailable" and "compile failed" in no_compile.reason
    fixture = crc.check_case_set(tmp_path, manifest, _run=ok_run,
                                 _compile=_stub_compile(tmp_path, b"\0AK_REPACK_FIXTURE_X"),
                                 **common)
    assert fixture.status == "unavailable" and "TEST FIXTURE" in fixture.reason

    def boom(*a, **k):
        raise RuntimeError("probe (GGML_IQK=1) exit 4")
    crashed = crc.check_case_set(tmp_path, manifest, _compile=_stub_compile(tmp_path),
                                 _run=boom, **common)
    assert crashed.status == "unavailable" and "exit 4" in crashed.reason


def test_check_case_set_runs_the_shared_cases_only_when_asked(tmp_path):
    manifest = _write_manifest(tmp_path)
    seen = []

    def run(binary, build, cases, **kw):
        seen.append(tuple(c.name for c in cases))
        return crc.parse_output(_output(cases))
    for shared in (False, True):
        result = crc.check_case_set(tmp_path, manifest, lane="ds41", include_shared=shared,
                                    threads=48, _compile=_stub_compile(tmp_path), _run=run)
        assert result.status == "pass"
    assert seen[0] == tuple(c.name for c in crc.LANE_CASES["ds41"])
    assert seen[1] == tuple(c.name for c in crc.all_cases("ds41"))


# --------------------------------------------------------------------------- gates wiring
def test_gates_check_maps_status_to_verdicts():
    for status, gate, passed in (("pass", "cpu_repack_case_set", True),
                                 ("wrong", "cpu_repack_case_set", False),
                                 ("unavailable", "oracle_unavailable", False)):
        seen = {}

        def check(build, manifest, **kw):
            seen.update(kw)
            return crc.RepackResult(status, "why", "detail")
        verdict = gates.check_cpu_repack_case_set(
            Path("/b"), resolved_recipe=_served_recipe(), manifest_path=Path("/m"),
            lane="ds41", include_shared=True, _check=check)
        assert (verdict.gate, verdict.passed) == (gate, passed)
        assert seen["threads"] == 48 and seen["include_shared"] is True
        assert seen["topology_prefix"] == ("taskset", "-c", "0-95")
        assert seen["launch_env"] == {"GGML_IQK": "1"}


def test_gates_check_refuses_an_unknown_lane_or_thread_count():
    verdict = gates.check_cpu_repack_case_set(Path("/b"), resolved_recipe=_served_recipe(),
                                              manifest_path=Path("/m"), lane=None)
    assert verdict.gate == "oracle_unavailable" and not verdict.passed
    verdict = gates.check_cpu_repack_case_set(
        Path("/b"), resolved_recipe=SimpleNamespace(topology_prefix=(), command_argv=("x",)),
        manifest_path=Path("/m"), lane="ds41", _check=lambda *a, **k: 1 / 0)
    assert verdict.gate == "oracle_unavailable" and "thread count" in verdict.reason


def _op_nmse(**kw):
    passing = lambda *a, **k: gates.Verdict("correctness", True, "ok")  # noqa: E731
    return gates.ppl_contract_op_nmse(Path("/candidate"), resolved_recipe=_served_recipe(),
                                      _op_correctness=passing, routed_lane="ds41", **kw)


def test_a_repack_route_needs_the_cpu_repack_manifest():
    for route in sorted(crc.REPACK_ROUTES):
        verdict = _op_nmse(route_name=route)
        assert not verdict.passed and verdict.gate == "ppl_contract_nmse_unavailable"
        assert "no op-level oracle" in verdict.reason


def test_a_repack_route_with_the_manifest_is_judged_by_the_repack_oracle():
    calls = []

    def repack(status):
        def check(build, **kw):
            calls.append(kw)
            return gates.Verdict("cpu_repack_case_set" if status != "unavailable"
                                 else "oracle_unavailable", status == "pass", f"r-{status}")
        return check
    ok = _op_nmse(route_name="cpu_repack_gemv", cpu_repack_manifest=Path("/s/cpu_repack/m.json"),
                  cpu_repack_shared=True, _check_cpu_repack=repack("pass"))
    assert ok.passed and "CPU_REPACK suite passed" in ok.reason
    assert calls[-1]["include_shared"] is True and calls[-1]["lane"] == "ds41"
    wrong = _op_nmse(route_name="cpu_repack_mmid", cpu_repack_manifest=Path("/m"),
                     _check_cpu_repack=repack("wrong"))
    assert (wrong.passed, wrong.gate) == (False, "ppl_contract_nmse")
    unavailable = _op_nmse(route_name="iqk_legacy_iq4nl", cpu_repack_manifest=Path("/m"),
                           _check_cpu_repack=repack("unavailable"))
    assert (unavailable.passed, unavailable.gate) == (False, "ppl_contract_nmse_unavailable")
    assert calls[-1]["include_shared"] is False


def test_the_store_manifest_is_derived_only_for_a_repack_lane(tmp_path):
    served = tmp_path / "store" / "served_shape" / "manifest.json"
    assert gates._cpu_repack_manifest_for(served, DS41_MODEL) == \
        tmp_path / "store" / "cpu_repack" / "manifest.json" == crc.manifest_path(tmp_path / "store")
    assert gates._cpu_repack_manifest_for(served, Q38FN_MODEL) is None
    assert gates._cpu_repack_manifest_for(None, DS41_MODEL) is None


def test_the_ppl_contract_candidate_and_bundle_tiers_pass_the_repack_manifest(tmp_path):
    served = tmp_path / "store" / "served_shape" / "manifest.json"
    seen = []

    def fake_nmse(build, **kw):
        seen.append(kw)
        return gates.Verdict("ppl_contract_nmse", True, "ok")
    common = dict(resolved_recipe=None, threads=1, cpu_list="0", env={}, log_dir=tmp_path,
                  served_shape_manifest=served)
    with mock.patch.object(gates, "ppl_contract_op_nmse", fake_nmse):
        assert gates.ppl_contract_gate(Path("/a"), Path("/c"), model=Path(DS41_MODEL),
                                       route=gates.cpu_route_named("cpu_repack_gemv"),
                                       **common).passed
        assert gates.ppl_contract_gate(Path("/a"), Path("/c"), model=Path(DS41_MODEL),
                                       route=gates.cpu_route_named("iqk_legacy_iq4nl"),
                                       **common).passed
        assert gates.ppl_contract_gate(Path("/a"), Path("/c"), model=Path(Q38FN_MODEL),
                                       route=gates.cpu_route_named("iqk_legacy_iq4nl"),
                                       **common).passed
    repack = tmp_path / "store" / "cpu_repack" / "manifest.json"
    assert (seen[0]["cpu_repack_manifest"], seen[0]["cpu_repack_shared"]) == (repack, True)
    assert (seen[1]["cpu_repack_manifest"], seen[1]["cpu_repack_shared"]) == (repack, False)
    assert seen[2]["cpu_repack_manifest"] is None
    passing = {name: (lambda: gates.Verdict(name, True, "ok"))
               for name in ("production_reference_load", "ppl", "coherence", "long_canary")}
    with mock.patch.object(gates, "ppl_contract_op_nmse", fake_nmse):
        gates.ppl_contract_bundle_gate(Path("/a"), Path("/c"), resolved_recipe=None,
                                       model=Path(DS41_MODEL), threads=1, cpu_list="0", env={},
                                       log_dir=tmp_path, reference_build=Path("/ref"),
                                       served_shape_manifest=served, _layers=passing)
    assert (seen[3]["cpu_repack_manifest"], seen[3]["cpu_repack_shared"]) == (repack, True)


# ------------------------------------------------------------------- cpu_repack_gemv route
def _edit(lines, line_no, text=" // edited"):
    before = lines[line_no - 1]
    after = before.rstrip("\n") + text + "\n"
    patch = f"@@ -{line_no} +{line_no} @@\n-{before.rstrip(chr(10))}\n+{after.rstrip(chr(10))}\n"
    new = list(lines)
    new[line_no - 1] = after
    return "".join(new), patch


def test_cpu_repack_gemv_admits_the_gemv_bodies_and_refuses_the_gemm_and_getenv():
    route = gates.cpu_route_named("cpu_repack_gemv")
    path = ANCHOR_1017 / route.path
    if not path.is_file():
        pytest.skip("DS41-1017 anchor tree not present")
    head = path.read_text(encoding="utf-8")
    lines = head.splitlines(keepends=True)

    def first(prefix):
        return next(i for i, line in enumerate(lines, 1) if line.startswith(prefix))
    gemv = first("static void gemv_q4_b32_8x8_q8_0_lut_avx(") + 7
    entry = first("void ggml_gemv_mxfp4_8x8_q8_0(") + 2
    gemm = first("static void gemm_q4_b32_8x8_q8_0_lut_avx(") + 7
    for symbol, line_no in (("gemv_q4_b32_8x8_q8_0_lut_avx", gemv),
                            ("gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>", gemv),
                            ("ggml_gemv_mxfp4_8x8_q8_0", entry)):
        candidate, patch = _edit(lines, line_no)
        admitted, refusal = gates.admit_cpu_route(route.path, symbol, candidate, head, patch)
        assert admitted is not None and admitted.route == "cpu_repack_gemv", refusal
        assert refusal is None
        assert gates.affected_op_scope((route.path,), target_surface=route.path,
                                       target_symbol=symbol, source_text=candidate,
                                       pre_source_text=head, patch_text=patch) == route.ops
    candidate, patch = _edit(lines, gemm)
    _admitted, refusal = gates.admit_cpu_route(route.path, "gemv_q4_b32_8x8_q8_0_lut_avx",
                                               candidate, head, patch)
    assert refusal is not None
    candidate, patch = _edit(lines, gemv, '  const char * k = getenv("X");')
    _admitted, refusal = gates.admit_cpu_route(route.path, "gemv_q4_b32_8x8_q8_0_lut_avx",
                                               candidate, head, patch)
    assert refusal is not None
