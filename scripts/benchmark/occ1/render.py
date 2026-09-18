"""Deterministic pixel-font renderer: text -> list of PNG frames on the reader's token grid.

Ported from @oh-my-pi/snapcompact research/bdf.py (MIT) with two OCC-1 changes:
  * frames are paginated (a history chunk becomes 1..N frames) and every frame dimension is a
    multiple of the reader's 32-px token cell, so llama.cpp consumes it with NO resample;
  * the last frame hugs its printed rows (upstream "frame height hugs printed rows") but never
    drops below min_height, so the production --image-min-tokens 1024 floor never upsamples it.

"DPI" is not meaningful for bitmap fonts: glyph size is fixed in pixels by the font cell
(advance x pitch). Fonts are hash-pinned; files live in the OCC-1 cache (never in git).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .fixture import DEFAULT_CACHE, sha256_file

XORG_RAW = "https://gitlab.freedesktop.org/xorg/font/misc-misc/-/raw/master/{name}.bdf"
UNSCII_HEX = "https://raw.githubusercontent.com/viznut/unscii/master/fontfiles/{name}.hex"

FONT_SHA256 = {
    "6x10.bdf": "99abf27fa2ca5a5171cad3df13aed2d4c55e73698cb141981b3e2268c733f64c",
    "8x13.bdf": "57cf39e2f24007c5e6bd0d4b9c03ebe5ae0003277a005f819c3aab4cc5ecb623",
    "unscii-8.hex": "03094f7fbab7085cf6a6b624cee61e47e71ce5d0c2f308c2f4436afdc17f776c",
}

FRAME_W = 1568  # 49 x 32: the upstream Anthropic-derived frame edge, and resample-free for Qwen3-VL
FRAME_H_MAX = 1568
FRAME_H_MIN = 672  # 21 x 32 -> 49*21 = 1029 tokens >= the production 1024-token floor
ALIGN = 32


@dataclass(frozen=True)
class FontCfg:
    """A density configuration: a bitmap font drawn on an adv x pitch cell grid."""

    name: str
    source: str  # font file name in the cache
    adv: int
    pitch: int
    ascent: int | None = None
    native: tuple[int, int] | None = None  # rasterize at this cell, then Lanczos to adv x pitch


FONTS: dict[str, FontCfg] = {
    "6x10": FontCfg("6x10", "6x10.bdf", 6, 10),
    "8x13": FontCfg("8x13", "8x13.bdf", 8, 13),
    "8x8u": FontCfg("8x8u", "unscii-8.hex", 8, 8),
    "12x12u": FontCfg("12x12u", "unscii-8.hex", 12, 12, native=(8, 8)),
    "16x16u": FontCfg("16x16u", "unscii-8.hex", 16, 16, native=(8, 8)),
}

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
# "color" variant: per-row pale band + dark glyph hue (6-hue cycle), as upstream.
_HUES = [0.0, 0.08, 0.3, 0.5, 0.62, 0.78]


def _hls(h: float, lum: float, s: float) -> tuple[int, int, int]:
    import colorsys

    return tuple(int(c * 255) for c in colorsys.hls_to_rgb(h, lum, s))  # type: ignore[return-value]


_DARK = [_hls(h, 0.22, 0.95) for h in _HUES]
_PALE = [_hls(h, 0.94, 0.6) for h in _HUES]
VARIANTS = ("bw", "color")


def ensure_font(cfg: FontCfg, cache: Path = DEFAULT_CACHE, download: bool = False) -> Path:
    path = cache / cfg.source
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} missing; re-run with --download")
        import urllib.request

        stem, ext = cfg.source.rsplit(".", 1)
        url = UNSCII_HEX.format(name=stem) if ext == "hex" else XORG_RAW.format(name=stem)
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    want = FONT_SHA256.get(cfg.source)
    if want is not None and sha256_file(path) != want:
        raise ValueError(f"font hash drift for {path}")
    return path


def parse_bdf(text: str) -> tuple[dict[int, dict], int]:
    """({codepoint: {bbx, rows}}, font_ascent)."""
    glyphs: dict[int, dict] = {}
    ascent = 0
    cur: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("FONT_ASCENT"):
            ascent = int(ln.split()[1])
        elif ln.startswith("STARTCHAR"):
            cur = {"rows": []}
        elif ln.startswith("ENCODING"):
            cur["enc"] = int(ln.split()[1])
        elif ln.startswith("BBX"):
            cur["bbx"] = tuple(map(int, ln.split()[1:5]))
        elif ln.startswith("BITMAP"):
            i += 1
            while not lines[i].startswith("ENDCHAR"):
                cur["rows"].append(int(lines[i], 16))
                i += 1
            if cur.get("enc", -1) >= 0:
                glyphs[cur["enc"]] = cur
        i += 1
    return glyphs, ascent


def parse_hex(text: str) -> tuple[dict[int, dict], int]:
    """Unifont-style .hex, 8x8 (one byte per row). Baseline at row 7."""
    glyphs: dict[int, dict] = {}
    for line in text.splitlines():
        cp, _, bits = line.partition(":")
        if not bits:
            continue
        data = bytes.fromhex(bits.strip())
        if len(data) == 8:
            glyphs[int(cp, 16)] = {"bbx": (8, 8, 0, -1), "rows": list(data)}
    return glyphs, 7


def load_font(cfg: FontCfg, cache: Path = DEFAULT_CACHE, download: bool = False) -> tuple[dict, int]:
    path = ensure_font(cfg, cache, download)
    text = path.read_text(errors="replace")
    return parse_hex(text) if path.suffix == ".hex" else parse_bdf(text)


def frame_capacity(cfg: FontCfg, width: int = FRAME_W, height: int = FRAME_H_MAX) -> tuple[int, int]:
    """(cols per row, rows per full frame)."""
    return width // cfg.adv, height // cfg.pitch


def _align_up(x: int, a: int = ALIGN) -> int:
    return -(-x // a) * a


def paginate(n_chars: int, cfg: FontCfg, width: int = FRAME_W, height_max: int = FRAME_H_MAX,
             height_min: int = FRAME_H_MIN) -> list[tuple[int, int, int, int]]:
    """Layout plan: [(char_start, char_end, frame_w, frame_h)] with 32-aligned dims."""
    cols, rows_max = frame_capacity(cfg, width, height_max)
    per_frame = cols * rows_max
    plan = []
    for start in range(0, n_chars, per_frame):
        end = min(start + per_frame, n_chars)
        rows = -(-(end - start) // cols)
        h = height_max if rows == rows_max else max(height_min, _align_up(rows * cfg.pitch))
        plan.append((start, end, width, min(h, height_max)))
    return plan


def normalize_text(text: str) -> str:
    """Collapse whitespace; the fixture is already single-spaced prose."""
    return " ".join(text.split())


class Renderer:
    def __init__(self, cfg: FontCfg, variant: str = "bw", cache: Path = DEFAULT_CACHE,
                 download: bool = False):
        if variant not in VARIANTS:
            raise ValueError(f"variant {variant!r} not in {VARIANTS}")
        from PIL import Image  # noqa: F401  (fail early if Pillow is missing)

        self.cfg = cfg
        self.variant = variant
        self.glyphs, font_ascent = load_font(cfg, cache, download)
        self.ascent = cfg.ascent if cfg.ascent is not None else font_ascent
        self.cell = cfg.native or (cfg.adv, cfg.pitch)
        self._masks: dict[int, object] = {}

    def _mask(self, cp: int):
        """1-bit-ish 'L' mask of one glyph in a native cell; None when blank/unknown."""
        if cp in self._masks:
            return self._masks[cp]
        from PIL import Image

        g = self.glyphs.get(cp)
        m = None
        if g is not None:
            cw, ch = self.cell
            m = Image.new("L", (cw, ch), 0)
            px = m.load()
            w, h, xoff, yoff = g["bbx"]
            top = self.ascent - h - yoff
            shift = 0x80 if w <= 8 else 0x8000
            ink = False
            for r, bits in enumerate(g["rows"]):
                y = top + r
                if not 0 <= y < ch:
                    continue
                for b in range(w):
                    if bits & (shift >> b):
                        x = xoff + b
                        if 0 <= x < cw:
                            px[x, y] = 255
                            ink = True
            if not ink:
                m = None
        self._masks[cp] = m
        return m

    def render_frame(self, text: str, frame_w: int, frame_h: int):
        from PIL import Image

        cfg = self.cfg
        cols = frame_w // cfg.adv
        rows = -(-len(text) // cols) if text else 0
        cw, ch = self.cell
        canvas = Image.new("RGB", (cols * cw, max(1, rows) * ch), _WHITE)
        for row in range(rows):
            if self.variant == "color":
                bg, fg = _PALE[row % 6], _DARK[row % 6]
                canvas.paste(bg, (0, row * ch, cols * cw, (row + 1) * ch))
            else:
                fg = _BLACK
            line = text[row * cols:(row + 1) * cols]
            for col, c in enumerate(line):
                m = self._mask(ord(c))
                if m is not None:
                    canvas.paste(fg, (col * cw, row * ch), m)
        if cfg.native is not None:
            canvas = canvas.resize((cols * cfg.adv, max(1, rows) * cfg.pitch), Image.LANCZOS)
        frame = Image.new("RGB", (frame_w, frame_h), _WHITE)
        frame.paste(canvas.crop((0, 0, min(canvas.width, frame_w), min(canvas.height, frame_h))), (0, 0))
        return frame

    def render(self, text: str, width: int = FRAME_W, height_max: int = FRAME_H_MAX,
               height_min: int = FRAME_H_MIN):
        """Returns [(PIL.Image, char_start, char_end)] covering all of `text` in reading order."""
        text = normalize_text(text)
        out = []
        for start, end, w, h in paginate(len(text), self.cfg, width, height_max, height_min):
            out.append((self.render_frame(text[start:end], w, h), start, end))
        return out


def png_bytes(img) -> bytes:
    import io

    buf = io.BytesIO()
    # optimize=False + fixed compress level keeps bytes deterministic for a given Pillow build.
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def pixel_digest(img) -> str:
    """Hash of raw pixels (independent of PNG encoder version)."""
    return hashlib.sha256(img.tobytes() + repr(img.size).encode()).hexdigest()
