"""Reader-side image-token prediction for Qwen3-VL (qwen3vl_merger) at the champion kernel.

Port of llama.cpp `tools/mtmd/mtmd-image.cpp::calc_size_preserved_ratio` (smart_resize) and
`clip-model.h::set_limit_image_tokens` as of champion ef81196d5:
    align = patch_size * n_merge = 16 * 2 = 32 px per output token side
    token bounds (8, 4096), overridable by --image-min-tokens / --image-max-tokens

The PREDICTION is used only for planning and as a cross-check. The authoritative billed count
is the server's `usage.prompt_tokens` (the runner records both and flags disagreement).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderGrid:
    name: str
    patch_size: int
    n_merge: int
    min_tokens: int
    max_tokens: int

    @property
    def align(self) -> int:
        return self.patch_size * self.n_merge

    @property
    def patch_area(self) -> int:
        return self.align * self.align


# Production worker_vision launch passes --image-min-tokens 1024 (launch_manifest.yaml).
QWEN3VL_PROD = ReaderGrid("qwen3vl_merger", patch_size=16, n_merge=2, min_tokens=1024, max_tokens=4096)


def smart_resize(width: int, height: int, grid: ReaderGrid) -> tuple[int, int]:
    """Exact port of calc_size_preserved_ratio(inp, align, min_pixels, max_pixels)."""
    a = grid.align
    min_pixels = grid.min_tokens * grid.patch_area
    max_pixels = grid.max_tokens * grid.patch_area

    def round_by(x: float) -> int:
        # std::round rounds half away from zero; Python round() is banker's rounding.
        return int(math.floor(x / a + 0.5)) * a

    h_bar = max(a, round_by(height))
    w_bar = max(a, round_by(width))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(a, int(math.floor((height / beta) / a)) * a)
        w_bar = max(a, int(math.floor((width / beta) / a)) * a)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil((height * beta) / a)) * a
        w_bar = int(math.ceil((width * beta) / a)) * a
    return w_bar, h_bar


def image_tokens(width: int, height: int, grid: ReaderGrid = QWEN3VL_PROD) -> int:
    """Embedding tokens for one image (excludes the <|vision_start|>/<|vision_end|> markers)."""
    w, h = smart_resize(width, height, grid)
    return (w // grid.align) * (h // grid.align)


def is_resample_free(width: int, height: int, grid: ReaderGrid = QWEN3VL_PROD) -> bool:
    """True when the reader consumes the frame pixel-for-pixel (no resize blur on glyphs)."""
    return smart_resize(width, height, grid) == (width, height)
