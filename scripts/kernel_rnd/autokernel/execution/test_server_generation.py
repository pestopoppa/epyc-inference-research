"""Owning v1 content reduction, not a claim of native issuer authority."""
from dataclasses import FrozenInstanceError, replace

import pytest

from ..evaluator import correctness
from . import server_generation as sg
from . import t0_provider as t0
from .test_t0_provider import (FakeClaim, anchor_capture, evaluation_request,
                               execution_plan, t0_policy)


def generation(**changes):
    args = dict(prompt="original prompt", prompt_ref="prompt-0", n_predict=3,
        temperature=0.0, top_k=1, slot_index=0, request_sha256="a" * 64,
        response_sha256="b" * 64, content="original answer", delivered_n=3,
        receipt_ref="artifact:original-response", error=None)
    return sg.ServerGenerationEvidence(**(args | changes))


class NoLaunch:
    def run(self, *args, **kwargs):
        raise AssertionError("server response replay must not launch a process")


def provider(value, anchor=None):
    return t0.ExecutedT0EvidenceProvider(plan=execution_plan(generation=value),
        runner=NoLaunch(), claim=FakeClaim(held=False), anchor_capture=anchor)


def test_content_byte_coherence_positive_with_unknown_seed_and_original_anchor():
    value = generation()
    anchor = anchor_capture(output_digests=(t0.sha256_text(value.content),),
                            output_lengths=(len(value.content),))
    evidence = provider(value, anchor).collect_coherence(t0._Collected())
    gate, verdict = correctness.check_output_coherence(
        evaluation_request(anchor=anchor.identity()), evidence, t0_policy())
    assert gate.check.outcome == "PASS"
    assert verdict.label == correctness.COHERENCE_BYTE_IDENTICAL
    assert evidence.seed is None
    assert evidence.token_agreement_ratio is None
    assert evidence.recorded_anchor() == anchor.identity()
    assert "llama-server/http-v1" in evidence.sampler_id
    assert not hasattr(value, "tokens")


def test_no_seed_repeats_or_dispatch_are_synthesized_and_count_is_observed():
    value = generation()
    owning = provider(value)
    collected = t0._Collected()
    collected.captures.append(t0.CompletedProcess(("unrelated",), (), "/", 0,
        "", "eval time = 99 ms / 999 runs", 0.1, False, False))
    assert owning.collect_determinism(collected) is None
    assert owning.collect_dispatch_trace(collected) is None
    assert owning._delivered_units(collected) == 3
    assert len(collected.notes) == 2


@pytest.mark.parametrize("temperature,top_k", [(1.0, 1), (0.0, 40), (0.0, 0)])
def test_non_greedy_never_byte_equivalence(temperature, top_k):
    value = generation(temperature=temperature, top_k=top_k)
    anchor = anchor_capture(output_digests=(t0.sha256_text(value.content),),
                            output_lengths=(len(value.content),))
    evidence = sg.collect_server_coherence(value, anchor)
    gate, _ = correctness.check_output_coherence(
        evaluation_request(anchor=anchor.identity()), evidence, t0_policy())
    assert gate.check.outcome != "PASS"


@pytest.mark.parametrize("change", [dict(content="different"), dict(content="")])
def test_output_mutation_is_not_equivalence(change):
    anchor = anchor_capture(output_digests=(t0.sha256_text("original answer"),),
                            output_lengths=(len("original answer"),))
    evidence = sg.collect_server_coherence(generation(**change), anchor)
    gate, _ = correctness.check_output_coherence(
        evaluation_request(anchor=anchor.identity()), evidence, t0_policy())
    assert gate.check.outcome != "PASS"


def test_failed_response_count_remains_diagnostic_not_comparable_content():
    value = generation(content=None, error="HTTPError: unavailable", delivered_n=2)
    evidence = provider(value).collect_coherence(t0._Collected())
    assert provider(value)._delivered_units(t0._Collected()) == 2
    assert evidence.candidate_output_sha256 is None
    assert evidence.candidate_output_len == 0


@pytest.mark.parametrize("changes", [dict(delivered_n=2), dict(delivered_n=True),
    dict(response_sha256=None), dict(content=None), dict(error="failed"),
    dict(temperature=float("nan")), dict(top_k=True), dict(slot_index=-1),
    dict(content=b"bytes"), dict(request_sha256="not-a-digest"),
    dict(content="x" * (sg.MAX_CONTENT_BYTES + 1))])
def test_closed_server_facts_refuse_one_fact_mutation(changes):
    with pytest.raises((TypeError, ValueError)):
        generation(**changes)


def test_frozen_record_exact_type_and_revalidation():
    value = generation()
    with pytest.raises(FrozenInstanceError):
        value.seed = 42
    class Foreign(sg.ServerGenerationEvidence):
        pass
    foreign = object.__new__(Foreign)
    for name, item in vars(value).items():
        object.__setattr__(foreign, name, item)
    with pytest.raises(TypeError):
        execution_plan(generation=foreign)
    with pytest.raises(TypeError):
        sg.collect_server_coherence(foreign, None)
    object.__setattr__(value, "seed", 42)
    with pytest.raises(ValueError, match="seed"):
        execution_plan(generation=value)


def test_cli_generation_plan_acceptance_unchanged():
    class Legacy(t0.GenerationPlan):
        pass
    value = Legacy(prompt="prompt", prompt_ref="ref", n_predict=3, seed=42)
    assert execution_plan(generation=value).generation is value
    assert replace(execution_plan(), generation=generation()).generation.seed is None
