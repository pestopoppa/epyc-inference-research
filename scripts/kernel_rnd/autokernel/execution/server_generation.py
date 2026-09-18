"""Original llama-server v1 response facts for owning T0 reduction.

This is not a launcher, an issuance registry, or a scientific verdict. The native
parent adapter independently reopens the original full unit before constructing
these records. v1 has no declared seed/repeat/token-return capability: neither
content bytes nor a successful HTTP response supplies those missing facts.
"""
from dataclasses import dataclass
import math
import re

from ..evaluator import correctness

SERVER_GENERATION_SCHEMA = "epyc.autokernel.server_generation_evidence.v1"
MAX_CONTENT_BYTES = 4 * 1024 * 1024
_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, init=False)
class ServerGenerationEvidence:
    prompt: str
    prompt_ref: str
    n_predict: int
    temperature: float
    top_k: int
    slot_index: int
    request_sha256: str
    response_sha256: str | None
    content: str | None
    delivered_n: int | None
    receipt_ref: str
    error: str | None
    seed: None
    schema: str

    def __init__(self, *, prompt: str, prompt_ref: str, n_predict: int,
                 temperature: float, top_k: int, slot_index: int,
                 request_sha256: str, response_sha256: str | None,
                 content: str | None, delivered_n: int | None,
                 receipt_ref: str, error: str | None = None) -> None:
        for name, value in (("prompt", prompt), ("prompt_ref", prompt_ref),
                ("n_predict", n_predict), ("temperature", temperature), ("top_k", top_k),
                ("slot_index", slot_index), ("request_sha256", request_sha256),
                ("response_sha256", response_sha256), ("content", content),
                ("delivered_n", delivered_n), ("receipt_ref", receipt_ref),
                ("error", error), ("seed", None), ("schema", SERVER_GENERATION_SCHEMA)):
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema != SERVER_GENERATION_SCHEMA or self.seed is not None:
            raise ValueError("server generation v1 has no seed/repeat capability")
        for name in ("prompt", "prompt_ref", "receipt_ref"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"server generation {name} must be non-empty text")
        for name, minimum in (("n_predict", 1), ("top_k", 0), ("slot_index", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"server generation {name} is invalid")
        if (type(self.temperature) not in (int, float)
                or not math.isfinite(self.temperature)):
            raise ValueError("server generation temperature must be finite")
        for name in ("request_sha256", "response_sha256"):
            value = getattr(self, name)
            if value is None and name == "response_sha256":
                continue
            if type(value) is not str or not _SHA.fullmatch(value):
                raise ValueError(f"server generation {name} is invalid")
        if self.content is not None and (type(self.content) is not str
                or len(self.content.encode("utf-8")) > MAX_CONTENT_BYTES):
            raise ValueError("server generation content is invalid or oversized")
        if self.delivered_n is not None and (type(self.delivered_n) is not int
                                            or self.delivered_n < 0):
            raise ValueError("server delivered count is invalid")
        if self.error is not None and (type(self.error) is not str or not self.error):
            raise ValueError("server generation error must be absent or non-empty")
        if self.error is None and (self.response_sha256 is None or self.content is None
                                  or self.delivered_n != self.n_predict):
            raise ValueError("successful server generation requires exact complete observed work")
        if self.error is not None and self.content is not None:
            raise ValueError("failed server generation cannot carry comparable output")

    def is_greedy(self) -> bool:
        # Identical owning predicate to GenerationPlan; do not broaden it.
        return float(self.temperature) == 0.0 and self.top_k == 1


def collect_server_coherence(value: ServerGenerationEvidence, anchor):
    """Content-byte coherence only, with the original owning anchor triple."""
    from . import t0_provider as t0
    if type(value) is not ServerGenerationEvidence:
        raise TypeError("concrete server generation evidence required")
    value.__post_init__()
    commit, binary, linkage = t0._anchor_triple(anchor)
    text = value.content or ""
    anchor_digest = None if anchor is None else anchor.first_output_digest()
    digest = t0.sha256_text(text) if text else None
    return correctness.CoherenceEvidence(
        candidate_output_sha256=digest, candidate_output_len=len(text),
        anchor_output_sha256=anchor_digest,
        anchor_output_len=None if anchor is None else anchor.first_output_length(),
        sampler_id=f"llama-server/http-v1/temp={value.temperature},top_k={value.top_k}",
        sampler_is_greedy=value.is_greedy(), seed=None, tokens_requested=value.n_predict,
        token_agreement_ratio=None, divergence_first_index=None,
        anchor_determinism_class=None if anchor is None else anchor.determinism_class,
        anchor_source_commit=commit, anchor_binary_sha256=binary,
        anchor_linkage_sha256=linkage, prompt_ref=value.prompt_ref,
        receipt_ref=value.receipt_ref, produced_by=t0.PRODUCER)
