"""Domain-specific exceptions for the orchestrator system.

Replaces generic ``except Exception`` catches with typed exceptions that
allow callers to handle specific failure modes.  Infrastructure exceptions
(timeouts, connection failures) are distinguished from application-level
errors (bad input, delegation failure) so callers can retry vs. bail out.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""


# -- Inference / LLM --------------------------------------------------------

class InferenceError(OrchestratorError):
    """LLM call failed (timeout, backend down, malformed response)."""


class InferenceTimeoutError(InferenceError):
    """LLM call timed out."""


class BackendUnavailableError(InferenceError):
    """Backend server is unreachable or returned 502/503."""


class AdmissionDenied(BackendUnavailableError):
    """Admission refused before dispatch; waiting may make the same request fit."""


class AdmissionDeniedText(str):
    """Legacy error text retaining its typed admission cause for API callers."""

    def __new__(cls, error: AdmissionDenied):
        value = super().__new__(cls, f"[ERROR: {error}]")
        value.error = error
        return value


class ContextOverflowError(InferenceError, RuntimeError):
    """llama-server refused or aborted a request because the KV context ran out.

    Two server failures map here (frozen tree ``production-consolidated-v10``,
    ``tools/server/server-context.cpp``):

    * ``kind == "request_too_large"`` — HTTP 400, ``type:
      exceed_context_size_error``: the prompt alone does not fit the slot's
      ``n_ctx`` ("request (N tokens) exceeds the available context size (M
      tokens), try increasing it", :3303-3310; or "input (N tokens) is larger
      than the max context size (M tokens). skipping", :3292-3299). Retrying the
      same request on the same role can never succeed.
    * ``kind == "pool_exhausted"`` — HTTP 500 / mid-stream error, ``type:
      server_error``, message "Context size has been exceeded." (:3759-3764):
      ``llama_decode`` found no free KV cells even at batch size 1. Under a
      unified KV pool this is concurrency (other in-flight requests filled the
      shared pool) and every in-flight request fails with it; the request may
      well fit on its own.

    Subclasses ``RuntimeError`` as well as ``InferenceError`` so every existing
    ``except RuntimeError`` caller still treats it as an inference failure,
    while callers that care can catch the specific type.
    """

    REQUEST_TOO_LARGE = "request_too_large"
    POOL_EXHAUSTED = "pool_exhausted"

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        role: str = "",
        backend_url: str = "",
        n_prompt_tokens: int | None = None,
        n_ctx: int | None = None,
        http_status: int | None = None,
        server_message: str = "",
        source: str = "server",
        recovery: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.role = role
        self.backend_url = backend_url
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx
        self.http_status = http_status
        self.server_message = server_message
        # "server": the llama-server said so. "admission": the orchestrator's
        # shared-pool admission refused to send it (no server call was made).
        self.source = source
        # Ordered record of the recovery attempts already made (backoff
        # retries, reroutes), so the final error says what was tried.
        self.recovery: list[dict] = list(recovery or [])

    @property
    def retryable(self) -> bool:
        """True when waiting can help (pool pressure), False when it cannot."""
        return self.kind == self.POOL_EXHAUSTED

    def to_dict(self) -> dict:
        return {
            "error": "context_overflow",
            "kind": self.kind,
            "role": self.role,
            "backend_url": self.backend_url,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_ctx": self.n_ctx,
            "http_status": self.http_status,
            "server_message": self.server_message,
            "source": self.source,
            "recovery": list(self.recovery),
            "detail": str(self),
        }


class RoleParkedError(BackendUnavailableError):
    """The role's server is parked: its GPU is lent to AutoKernel work.

    Raised BEFORE any lock, admission or HTTP call (``src/runtime/gpu_window.py``),
    so a parked role fails in microseconds with an explicit reason instead of a
    connection refusal or a slow timeout. Deliberately NOT a ``RuntimeError``:
    the primitives' same-tier model fallback must not silently answer with a
    different model. Surfaces as HTTP 503 ``role_parked`` + ``Retry-After``
    (``src/api/__init__.py``) or, swallowed by ``llm_call`` into the in-band
    ``[ERROR: role_parked: ...]`` sentinel, as a /chat infra failure that
    ``_annotate_error`` maps to the same 503.

    The message format is parsed back by ``gpu_window.parse_parked_sentinel``;
    change both together.
    """

    error_type = "role_parked"

    def __init__(
        self,
        *,
        role: str | None,
        port: int | None,
        holder: str,
        retry_after_s: int,
        expected_end: str | None = None,
        request_id: str | None = None,
        refusal: dict | None = None,
        preempt: dict | None = None,
    ) -> None:
        self.role = role
        self.port = port
        self.holder = holder
        self.retry_after_s = int(retry_after_s)
        self.expected_end = expected_end
        self.request_id = request_id
        self.preempt = preempt
        self.refusal = dict(refusal or {"gate": self.error_type, "holder": holder})
        if preempt is not None:
            self.refusal["preempt"] = preempt.get("status")
        super().__init__(
            f"role_parked: role={role or ''} port={port or ''} holder={holder} "
            f"retry_after_s={self.retry_after_s} — the GPU serving this role is lent to "
            f"{holder} work until {expected_end or 'unknown'}; preempt "
            f"{(preempt or {}).get('status', 'not requested')}; service unavailable"
        )

    @classmethod
    def from_info(cls, info, *, request_id=None, preempt=None) -> "RoleParkedError":
        return cls(
            role=info.role,
            port=info.port,
            holder=info.holder,
            retry_after_s=info.retry_after_s,
            expected_end=info.expected_end,
            request_id=request_id,
            refusal=info.refusal(request_id=request_id),
            preempt=preempt,
        )

    def to_dict(self) -> dict:
        """HTTP body (shared by every route that refuses a parked role)."""
        detail = str(self)
        return {
            "error": self.error_type,
            "type": self.error_type,
            "detail": detail,
            "error_code": 503,
            "error_detail": detail,
            "retry_after_s": self.retry_after_s,
            "holder": self.holder,
            "role": self.role,
            "port": self.port,
            "expected_end": self.expected_end,
            "refusal": dict(self.refusal),
        }


# -- Delegation / Routing ---------------------------------------------------

class DelegationError(OrchestratorError):
    """Architect delegation failed (bad decision, specialist failure)."""


class DelegationLoopError(DelegationError):
    """Delegation entered a zero-progress loop."""


# -- Vision ------------------------------------------------------------------

class VisionAnalysisError(OrchestratorError):
    """Vision pipeline failed (OCR, VL inference, image processing)."""


# -- Archive / Document ------------------------------------------------------

class ArchiveExtractionError(OrchestratorError):
    """Archive extraction failed (corrupt, too large, unsupported format)."""


class DocumentProcessingError(OrchestratorError):
    """Document preprocessing or parsing failed."""


# -- REPL / Execution -------------------------------------------------------

class REPLExecutionError(OrchestratorError):
    """REPL code execution failed."""


class REPLTimeoutError(REPLExecutionError):
    """REPL execution timed out."""


# -- Configuration -----------------------------------------------------------

class ConfigurationError(OrchestratorError):
    """Configuration is invalid or missing required values."""


class RegistryError(OrchestratorError):
    """Model registry loading or validation failed."""
