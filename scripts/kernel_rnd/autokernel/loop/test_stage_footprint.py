"""CTC-1 acceptance (§9 'Generic targets'): footprints come from recipes, not names."""
from __future__ import annotations

import argparse
import itertools
from types import SimpleNamespace

import pytest

from . import resolved_recipe, serving, stage_footprint as sf
from .stage_footprint import FootprintRefused

#: Deterministic topology: logical 96..191 are SMT siblings of 0..95.
SIBLINGS = {cpu: cpu % 96 for cpu in range(192)}


def _resources(cpus=range(192), gpu=("ROCm0",), jobs=24):
    return SimpleNamespace(cpu_logical=tuple(cpus), gpu_ids=tuple(gpu), build_jobs=jobs)


def _target(name, backend, model):
    payload = {"target_ids": [name], "backend": backend, "model": model}
    return SimpleNamespace(execution=SimpleNamespace(backend=backend),
                           to_dict=lambda: dict(payload))


def _launch(backend, cpu_list, *, device=None, numa=None, supported=True, codes=()):
    template = SimpleNamespace(cpu_list=cpu_list,
                               device=device or ("ROCm0" if backend == "gpu" else "none"))
    prefix = (("numactl", numa) if numa else ()) + ("taskset", "-c", cpu_list)
    capability = SimpleNamespace(supported=supported,
                                 reasons=tuple(SimpleNamespace(code=c) for c in codes))
    return SimpleNamespace(backend=backend, template=template, topology_prefix=prefix,
                           capability=capability, build_dir=f"/b/{backend}",
                           execution_digest=f"exec-{cpu_list}", launch_env=())


def _argv(backend, model):
    flag = "--cpu-serving-launch" if backend == "cpu" else "--gpu-serving-launch"
    return ["--model", model, flag, "/r/launch.json"]


#: Three synthetic model identities with different affinities (two CPU, one GPU).
ROSTER = [("alpha", "cpu", "/m/alpha.gguf", "0-23", None),
          ("bravo", "cpu", "/m/bravo.gguf", "48-71", "--interleave=all"),
          ("charlie", "gpu", "/m/charlie.gguf", "184-191", None)]


def _footprints(roster):
    out = []
    for name, backend, model, cpus, numa in roster:
        launch = _launch(backend, cpus, numa=numa)
        out.append(sf.runtime_footprint(
            "measure", _target(name, backend, model), launch, _resources(),
            adapter=sf.execution_adapter(_argv(backend, model)), attempt="a1",
            environment={}, sibling_map=SIBLINGS))
    return out


def _decisions(fps):
    return {(i, j): sf.conflicts(fps[i], fps[j])
            for i, j in itertools.combinations(range(len(fps)), 2)}


def test_three_synthetic_targets_get_footprints_from_their_recipes():
    alpha, bravo, charlie = _footprints(ROSTER)
    assert alpha.regions == ("q0",) and alpha.devices == () and alpha.memory is None
    assert bravo.regions == ("q2",) and bravo.memory == "--interleave=all"
    assert charlie.devices == ("mi210_0",) and charlie.regions == ("q3",)
    assert {fp.recipe for fp in (alpha, bravo, charlie)} == {
        "exec-0-23", "exec-48-71", "exec-184-191"}
    assert len({fp.target for fp in (alpha, bravo, charlie)}) == 3
    assert _decisions([alpha, bravo, charlie]) == {(0, 1): (), (0, 2): (), (1, 2): ()}


def test_swapping_names_leaves_decisions_unchanged():
    swapped = [(n2, b, m2, c, numa) for (n2, _, m2, _, _), (_, b, _, c, numa)
               in zip(reversed(ROSTER), ROSTER)]
    renamed = [("zz-" + n, b, "/other/" + m.rsplit("/", 1)[1], c, numa)
               for n, b, m, c, numa in ROSTER]
    base = _decisions(_footprints(ROSTER))
    assert _decisions(_footprints(swapped)) == base
    assert _decisions(_footprints(renamed)) == base


def test_smt_siblings_fold_into_one_physical_region():
    gpu_helper = sf.runtime_footprint(
        "measure", _target("g", "gpu", "/m/g"), _launch("gpu", "184-191"), _resources(),
        adapter=sf.execution_adapter(_argv("gpu", "/m/g")), attempt="a", environment={},
        sibling_map=SIBLINGS)
    cpu_worker = sf.runtime_footprint(
        "measure", _target("c", "cpu", "/m/c"), _launch("cpu", "88"), _resources(),
        adapter=sf.execution_adapter(_argv("cpu", "/m/c")), attempt="a", environment={},
        sibling_map=SIBLINGS)
    assert gpu_helper.cores == frozenset(range(88, 96))
    assert 88 in cpu_worker.cores
    assert sf.conflicts(gpu_helper, cpu_worker) == ("region:q3",)


def test_mixed_offload_keeps_its_explicit_refusal_from_the_real_capability_owner():
    template = serving.Recipe(name="t", model="/m/x.gguf", device="ROCm0", ngl=99,
                              spec_decode={"type": "draft", "drafter": "/m/d.gguf", "ngld": 0},
                              np=1, ctx=4096, threads=8, cpu_list="88-95")
    capability = resolved_recipe._capability(template, "gpu", object(), ())
    assert not capability.supported
    launch = _launch("gpu", "88-95")
    launch.capability = capability
    with pytest.raises(FootprintRefused, match="mixed_cpu_gpu_draft_unsupported"):
        sf.runtime_footprint("measure", _target("x", "gpu", "/m/x"), launch, _resources(),
                             adapter=sf.execution_adapter(_argv("gpu", "/m/x")),
                             attempt="a", environment={}, sibling_map=SIBLINGS)


def test_affinity_outside_owned_resources_is_refused():
    with pytest.raises(FootprintRefused, match="exceeds declared"):
        sf.runtime_footprint("measure", _target("x", "cpu", "/m/x"), _launch("cpu", "0-47"),
                             _resources(cpus=range(24)),
                             adapter=sf.execution_adapter(_argv("cpu", "/m/x")),
                             attempt="a", environment={}, sibling_map=SIBLINGS)


def test_build_footprint_uses_declared_allocation_and_one_build_slot():
    target = _target("x", "cpu", "/m/x")
    adapter = sf.execution_adapter(_argv("cpu", "/m/x"))
    a = sf.build_footprint(target, _resources(cpus=range(0, 24), jobs=12), adapter=adapter,
                           attempt="a", build="/b/1", sibling_map=SIBLINGS)
    b = sf.build_footprint(target, _resources(cpus=range(48, 72)), adapter=adapter,
                           attempt="b", build="/b/2", sibling_map=SIBLINGS)
    assert a.build_jobs == 12 and a.regions == ("q0",)
    assert sf.conflicts(a, b) == ("build_slot",)


def test_execution_adapter_is_read_not_rechosen():
    legacy = sf.execution_adapter(["--model", "/m/g", "--pairs", "8"])
    assert legacy.route == sf.ADAPTER_GPU_LEGACY_SCREEN and legacy.pairs == "8"
    matched = sf.execution_adapter(_argv("cpu", "/m/c") + ["--serving-instrument",
                                                           "matched_process_v2"])
    assert (matched.route, matched.instrument) == (sf.ADAPTER_CPU_SERVING, "matched_process_v2")
    assert sf.execution_adapter(_argv("gpu", "/m/g")).route == sf.ADAPTER_GPU_SERVING
    with pytest.raises(FootprintRefused):
        sf.execution_adapter(["--cpu-serving-launch", "a", "--gpu-serving-launch", "b"])


def test_max_active_targets_defaults_to_one_and_refuses_more_without_admission():
    parser = argparse.ArgumentParser()
    sf.add_max_active_targets(parser)
    args = parser.parse_args([])
    sf.check_max_active_targets(parser, args)
    assert args.max_active_targets == 1
    with pytest.raises(SystemExit):
        sf.check_max_active_targets(parser, parser.parse_args(["--max-active-targets", "2"]))


def test_scheduling_code_names_no_model():
    text = (sf.__file__ and open(sf.__file__).read()).lower()
    for name in ("glm", "qwen", "gemma", "llama-", "deepseek"):
        assert name not in text
