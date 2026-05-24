"""HF backbone integration via a wrapper-layer subclass.

The Engram module needs *both* the residual hidden state and the original
`input_ids` (for hashing). HF DecoderLayer signatures only take hidden
states + attention masks, so we cannot just register a `forward_pre_hook`
without losing input_ids.

The cleanest solution that works for any HF DecoderLayer is:

  1. Wrap the chosen layer in `EngramLayerWrapper(hf_layer, engram)`.
  2. The training loop calls `wrapper.set_input_ids(batch_ids)` before each
     `model(input_ids=batch_ids, ...)` call. The wrapper caches the ids.
  3. Inside `forward(hidden_states, ...)`, the wrapper:
       a. Reads the cached input_ids,
       b. Reshapes hidden_states to [B, L, HC_MULT, D] for Engram input,
       c. Adds Engram's residual delta,
       d. Reshapes back to [B, L, D] and forwards to the original layer.

For hc_mult=1 (vanilla backbones like Qwen3.6) the reshape is a free
view, no extra memory. For hc_mult>1 (hyper-connected backbones, e.g. the
paper's own architecture) the wrapper expands then collapses; same
semantics as upstream demo's expand-at-input.

`splice_engram_into(model, layer_index, engram)` is the convenience entry
point that wraps `model.model.layers[layer_index]` for HF causal-LM
naming, or accepts a custom `layer_accessor` for non-standard backbones.
"""
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from engram.modules import Engram


class EngramLayerWrapper(nn.Module):
    """Wrap an HF DecoderLayer so its residual stream gets an Engram add
    BEFORE the wrapped layer's attention + MLP run.

    The wrapped layer's `forward` signature is preserved — extra args/kwargs
    (attention_mask, position_ids, past_key_values, etc.) are forwarded
    untouched. The Engram add happens on the first positional arg, which
    is hidden_states by HF convention.

    Args:
        hf_layer: an HF DecoderLayer (Qwen3DecoderLayer, GemmaDecoderLayer,
            LlamaDecoderLayer, …)
        engram: an `Engram` module (typically zero-initialized via
            `apply_identity_init` at retrofit time)
        hc_mult: hyper-connection multiplicity. 1 for standard HF backbones.

    Usage:
        engram = Engram(eng_cfg, bb_cfg, layer_id=2, ...)
        apply_identity_init(engram)
        wrapper = EngramLayerWrapper(model.model.layers[2], engram, hc_mult=1)
        model.model.layers[2] = wrapper
        # ... in the training loop:
        wrapper.set_input_ids(batch["input_ids"])
        outputs = model(input_ids=batch["input_ids"], ...)
    """

    def __init__(self, hf_layer: nn.Module, engram: Engram, hc_mult: int = 1):
        super().__init__()
        self.hf_layer = hf_layer
        self.engram = engram
        self.hc_mult = hc_mult
        self._cached_input_ids: Optional[torch.Tensor] = None

    def set_input_ids(self, input_ids: torch.Tensor) -> None:
        """Cache the current batch's input_ids. Call this from the training
        loop immediately before each `model(input_ids=...)` call.

        Stored as a Tensor (not Parameter), excluded from state_dict.
        """
        self._cached_input_ids = input_ids

    def _engram_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the Engram residual delta to add to hidden_states.

        hidden_states shape: [B, L, D] (standard HF). Internally we add an
        hc_mult dimension for the Engram call, then collapse it back.
        """
        if self._cached_input_ids is None:
            raise RuntimeError(
                "EngramLayerWrapper has no cached input_ids. "
                "Call wrapper.set_input_ids(batch_ids) before each forward."
            )
        ids = self._cached_input_ids
        B, L, D = hidden_states.shape
        if ids.shape != (B, L):
            raise RuntimeError(
                f"Cached input_ids shape {tuple(ids.shape)} != hidden_states leading shape ({B}, {L}). "
                "Did the training loop forget to refresh set_input_ids between batches?"
            )

        # Promote to [B, L, HC_MULT, D]; for hc_mult=1 this is a free view.
        h_hc = hidden_states.unsqueeze(2).expand(B, L, self.hc_mult, D)
        delta_hc = self.engram(hidden_states=h_hc, input_ids=ids)  # [B, L, HC_MULT, D]

        # Collapse back. For hc_mult=1 this is a view; for >1 we average
        # across the hc axis (matches upstream demo which takes hc=0 at the
        # head but we want all gates' contributions averaged to a single
        # residual delta for the HF backbone).
        if self.hc_mult == 1:
            return delta_hc.squeeze(2)
        return delta_hc.mean(dim=2)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Any:
        delta = self._engram_residual(hidden_states)
        modified_hidden = hidden_states + delta
        return self.hf_layer(modified_hidden, *args, **kwargs)


def splice_engram_into(
    model: nn.Module,
    layer_index: int,
    engram: Engram,
    hc_mult: int = 1,
    layer_accessor: Optional[Callable[[nn.Module], List[nn.Module]]] = None,
) -> EngramLayerWrapper:
    """Replace one decoder layer with an `EngramLayerWrapper` in-place.

    Args:
        model: HF causal-LM (or any model with a list-like `.model.layers`)
        layer_index: which layer to wrap
        engram: the Engram module to graft on
        hc_mult: hyper-connection multiplicity (1 for standard HF)
        layer_accessor: function that returns the list of decoder layers
            given the model. Default: lambda m: m.model.layers (HF
            convention for Qwen / Llama / Gemma / Mistral).

    Returns:
        the EngramLayerWrapper that now lives at layers[layer_index].
        Keep a reference so the training loop can call set_input_ids() on it.
    """
    if layer_accessor is None:
        layer_accessor = lambda m: m.model.layers  # noqa: E731
    layers = layer_accessor(model)
    original = layers[layer_index]
    wrapper = EngramLayerWrapper(original, engram, hc_mult=hc_mult)
    layers[layer_index] = wrapper
    return wrapper


def collect_wrappers(model: nn.Module) -> List[EngramLayerWrapper]:
    """Find all EngramLayerWrapper instances in a model.

    Useful for the training loop's broadcast `set_input_ids` step when
    multiple Engram layers are spliced in.
    """
    return [m for m in model.modules() if isinstance(m, EngramLayerWrapper)]
