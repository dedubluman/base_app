"""Parametric deep-abyss "look" (global layer) shared by abyss_fit.py and abyss_grade.py.

The look is a handful of numbers, not a LUT, so it can be fitted to a reference still
(abyss_fit.py), saved as JSON, hand-tuned, and applied identically to every frame:

    tone       Y = gain * softmax(n - black, 0) ** gamma     n = subject-free base luma (+ detail)
    vignette   V = exp(-(vig_x*ux**p + vig_y*uy**p))        ux,uy = distance from (cx, cy)
    chroma     R/G = rg[0] + rg[1]*(Rs/Gs) + rg[2]*Y      (s = smoothed, subject-free source ratios)
               B/G = bg[0] + bg[1]*(Bs/Gs) + bg[2]*Y      (linear ratios: the deep look all but removes red,
               RGB is then scaled so that its BT.709 luma equals Y exactly.   log-ratios would explode)

Everything is computed on a *subject-free* base: SAM masks of divers and fish are excluded
from the blur (normalised convolution), so a dark diver cannot pull a dark halo into the water.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from abyss_common import LUMA, gblur, lowpass

EPS = 1e-4


@dataclass
class Look:
    # --- tone on the base luma (source units, 0..1) -------------------------------------
    black: float = 0.38          # source luma that maps to black
    gain: float = 1.30           # slope above the black point
    gamma: float = 1.0
    toe: float = 0.04            # softness of the black clip: shadows keep texture, no hard digital black
    base_sigma: float = 30.0     # px: scale separating "base" from "detail"
    detail: float = 1.4          # gain on the mid-frequency detail (seabed texture, rays)
    # --- vignette / depth falloff ----------------------------------------------------------
    cx: float = 0.5
    cy: float = 0.5
    vig_x: float = 0.9
    vig_y: float = 0.25
    vig_p: float = 2.0
    # --- chroma model (see module doc) -------------------------------------------------------
    rg: list[float] = field(default_factory=lambda: [-0.4, 1.4, 0.3])
    bg: list[float] = field(default_factory=lambda: [0.1, 1.0, 0.0])
    chroma_sigma: float = 40.0
    chroma_gain: float = 1.0     # >1 = more colour, <1 = more neutral (applied on the ratios)
    # --- extras (not fitted) ----------------------------------------------------------------
    floor: float = 0.008         # lifted black: never flat digital black
    particles: float = 0.8       # keep source specks (marine snow) at this gain

    @staticmethod
    def load(path: Path | None) -> "Look":
        look = Look()
        if path and Path(path).exists():
            d = json.loads(Path(path).read_text())
            names = {f.name for f in fields(Look)}
            for k, v in d.items():
                if k in names:
                    setattr(look, k, v)
        return look

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    def override(self, items: list[str]) -> "Look":
        for it in items or []:
            k, v = it.split("=", 1)
            cur = getattr(self, k)
            setattr(self, k, json.loads(v) if isinstance(cur, list) else type(cur)(v))
        return self


# ------------------------------------------------------------------------------ helpers
def nblur(x: np.ndarray, excl: np.ndarray | None, s: float) -> np.ndarray:
    """Gaussian blur that ignores excluded pixels (normalised convolution): the excluded
    region is filled in from its surroundings."""
    blur = (lambda a: lowpass(a, s)) if s >= 10 else (lambda a: gblur(a, s))
    if excl is None:
        return blur(x)
    w = (1.0 - excl).astype(np.float32)
    return blur(x * w) / np.maximum(blur(w), 1e-3)


def softpos(x: np.ndarray, eps: float) -> np.ndarray:
    """Smooth max(x, 0) (C-infinity at 0, no kink)."""
    return 0.5 * (x + np.sqrt(x * x + eps * eps))


def tone(n: np.ndarray, look: Look) -> np.ndarray:
    return look.gain * np.power(softpos(n - look.black, look.toe) + EPS, look.gamma)


def vignette(h: int, w: int, look: Look) -> np.ndarray:
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    ux = np.abs(x - look.cx) / max(look.cx, 1 - look.cx)
    uy = np.abs(y - look.cy) / max(look.cy, 1 - look.cy)
    return np.exp(-(look.vig_x * ux ** look.vig_p + look.vig_y * uy ** look.vig_p)).astype(np.float32)


def src_ratios(rgb: np.ndarray, excl: np.ndarray | None, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed R/G and B/G of the (subject-free) source water colour."""
    ch = [nblur(rgb[..., c], excl, s) for c in range(3)]
    g = np.maximum(ch[1], 1e-3)
    return ch[0] / g, ch[2] / g


def colorize(Y: np.ndarray, r_rg: np.ndarray, r_bg: np.ndarray, look: Look) -> np.ndarray:
    """Luma Y + smoothed source ratios -> RGB with BT.709 luma == Y. chroma_gain scales the
    ratios' departure from neutral (1.0)."""
    rg = np.clip(look.rg[0] + look.rg[1] * r_rg + look.rg[2] * Y, 0.0, 4.0)
    bg = np.clip(look.bg[0] + look.bg[1] * r_bg + look.bg[2] * Y, 0.0, 4.0)
    c = look.chroma_gain
    rgb = np.stack([1 + c * (rg - 1), np.ones_like(rg), 1 + c * (bg - 1)], -1)
    rgb = np.clip(rgb, 0.0, 6.0)
    return (rgb * (Y / np.maximum(rgb @ LUMA, 1e-6))[..., None]).astype(np.float32)
