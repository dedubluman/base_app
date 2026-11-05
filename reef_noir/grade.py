"""Environment and diver grades, as float32 RGB (0..1, BT.709 gamma) maths.

Look (rebuilt from the reference Reel "The power of Colour Grading *GoPro*"):
  * deep / distant water and rocks -> near-black navy with a blue cast
  * mid tones                       -> petrol / teal-blue
  * lit sand                        -> icy blue-white with caustic texture
  * skin                            -> warm, natural, restored red
The environment look is a luminance-driven palette ("gradient map") blended
with the source chroma, so texture comes from the source luminance and
colour separation from the palette. All parameters are fixed per clip
(no per-frame auto-exposure), which keeps the grade temporally stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Palette anchors (tone position 0..1 -> sRGB 0..255). Derived from the
# reference's measured final state: upper water ~ (25,31,37)-(30,42,53),
# shadow/mid ~ (34,72,88), sand ~ (140,190,205) up to near white.
DEFAULT_PALETTE = [
    (0.00, (5, 9, 16)),
    (0.12, (12, 25, 39)),
    (0.28, (24, 54, 74)),
    (0.46, (46, 104, 130)),
    (0.64, (92, 160, 188)),
    (0.82, (152, 203, 220)),
    (1.00, (228, 243, 248)),
]


@dataclass
class EnvGrade:
    # tone stretch of the working luminance (fixed per clip, from percentiles)
    lo: float = 0.0
    hi: float = 1.0
    gamma: float = 1.0
    contrast: float = 1.0             # S-curve strength around mid grey
    palette: list = field(default_factory=lambda: list(DEFAULT_PALETTE))
    chroma_keep: float = 0.22         # how much source chroma rides on top
    depth_darken: float = 0.55        # max exposure cut in "deep" water (linear gain)
    depth_navy: float = 0.35          # extra pull of deep water towards navy
    clarity: float = 0.35             # mid-scale local contrast (caustics)
    clarity_sigma: float = 12.0
    vignette: float = 0.18
    bloom: float = 0.06
    bloom_thresh: float = 0.78


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def work_luma(rgb: np.ndarray) -> np.ndarray:
    # Red is almost absent (and noisy) underwater; weight green/blue.
    return rgb[..., 0] * 0.12 + rgb[..., 1] * 0.58 + rgb[..., 2] * 0.30


def _lut_from_palette(palette, n=1024):
    xs = np.array([p[0] for p in palette], np.float32)
    cs = np.array([p[1] for p in palette], np.float32) / 255.0
    t = np.linspace(0, 1, n, dtype=np.float32)
    # interpolate in linear light for smooth ramps
    lin = srgb_to_lin(cs)
    out = np.stack([np.interp(t, xs, lin[:, c]) for c in range(3)], -1)
    return lin_to_srgb(out).astype(np.float32)


def _s_curve(x, k):
    if k == 1.0:
        return x
    # smooth contrast around 0.5 that keeps 0 and 1 fixed
    a = np.clip(x, 0, 1)
    return np.where(a < 0.5, 0.5 * (2 * a) ** k, 1 - 0.5 * (2 * (1 - a)) ** k)


def soft_clip(x, knee=0.85):
    """Highlight roll-off: identity below `knee`, asymptotic to 1 above."""
    over = np.clip(x - knee, 0, None)
    r = 1 - knee
    return np.where(x > knee, knee + r * (1 - np.exp(-over / r)), x)


class EnvGrader:
    def __init__(self, g: EnvGrade, shape_hw):
        self.g = g
        self.lut = _lut_from_palette(g.palette)
        h, w = shape_hw
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.yn = yy / (h - 1)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        self.vig = 1 - g.vignette * np.clip((r - 0.55) / 0.9, 0, 1) ** 1.6
        self._depth_prev = None

    def depth_map(self, rgb: np.ndarray) -> np.ndarray:
        """0 = lit sand / near, 1 = deep open water. Large-scale and
        temporally smoothed so it reads as atmosphere, never as a matte."""
        small = cv2.resize(rgb, (rgb.shape[1] // 8, rgb.shape[0] // 8),
                           interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        L, a, b = lab[..., 0] / 100.0, lab[..., 1], lab[..., 2]
        # open water is bluer (b strongly negative, a less negative) and darker
        blue = np.clip((-b - 18) / 18.0, 0, 1) * np.clip((a + 22) / 14.0, 0, 1)
        dark = np.clip((0.56 - L) / 0.12, 0, 1)
        d = 0.6 * blue + 0.4 * dark
        d = cv2.GaussianBlur(d, (0, 0), 6)
        yn = cv2.resize(self.yn, d.shape[::-1])
        d = np.clip(0.55 * d + 0.45 * np.clip(1.05 - yn * 1.6, 0, 1) ** 1.3, 0, 1)
        d = cv2.GaussianBlur(d, (0, 0), 4)
        if self._depth_prev is not None:
            d = 0.85 * self._depth_prev + 0.15 * d
        self._depth_prev = d
        return cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

    def __call__(self, rgb: np.ndarray, depth: np.ndarray | None = None) -> np.ndarray:
        g = self.g
        if depth is None:
            depth = self.depth_map(rgb)
        Y = work_luma(rgb)
        # local clarity on the luminance only (caustics, rock texture). The
        # detail layer is taken from a lightly smoothed copy so 8x8 H.264
        # block edges in the source are not amplified.
        Ys_ = cv2.bilateralFilter(Y, 5, 0.02, 3)
        base = cv2.GaussianBlur(Y, (0, 0), g.clarity_sigma)
        Yc = Y + g.clarity * (Ys_ - base)
        t = np.clip((Yc - g.lo) / (g.hi - g.lo), 0, 1.2)
        t = np.clip(t, 0, 1) ** g.gamma
        t = _s_curve(t, g.contrast)
        # depth: darken & push to navy (in linear light)
        t = t * (1 - g.depth_darken * depth * (1 - 0.5 * t))
        t = np.clip(t, 0, 1)
        idx = (t * (len(self.lut) - 1)).astype(np.int32)
        pal = self.lut[idx]
        # source chroma riding on top (keeps natural variation in the water)
        Ys = work_luma(rgb)[..., None]
        chroma = rgb - Ys
        out = pal + g.chroma_keep * chroma * (0.4 + 0.6 * t[..., None])
        navy = np.array([0.02, 0.05, 0.10], np.float32)
        out = out + (navy - out) * (g.depth_navy * depth * (1 - t))[..., None]
        out = out * self.vig[..., None]
        if g.bloom > 0:
            lin = srgb_to_lin(np.clip(out, 0, 1))
            hl = np.clip(lin - srgb_to_lin(g.bloom_thresh), 0, None)
            glow = cv2.GaussianBlur(hl, (0, 0), 14) + cv2.GaussianBlur(hl, (0, 0), 40)
            lin = lin + g.bloom * glow * np.array([0.85, 1.0, 1.05], np.float32)
            out = lin_to_srgb(lin)
        return soft_clip(out, 0.86).astype(np.float32)


@dataclass
class DiverGrade:
    # target skin colour in Lab for a lit, slightly cool underwater scene
    skin_a: float = 8.0
    skin_b: float = 11.0
    l_gain: float = 1.0       # diver luminance relative to graded environment
    chroma_scale: float = 0.7
    dev_clip: float = 10.0    # max per-pixel chroma deviation from the skin median
    red_mix: float = 0.55     # rebuild red from green (classic underwater fix)


def grade_diver(src: np.ndarray, env: np.ndarray, g: DiverGrade,
                hard_mask: np.ndarray, state: dict | None = None) -> np.ndarray:
    """Return a full-frame diver grade (only meaningful inside the mask).

    1. Red rebuild + white balance computed from the diver's own pixels.
    2. Luminance taken from the environment grade (so the diver sits in the
       new exposure / contrast) with a little original detail.
    3. Chroma mapped so the median skin lands on a natural target hue.
    """
    m = hard_mask > 0.5
    x = src.copy()
    x[..., 0] = x[..., 0] + g.red_mix * np.clip(x[..., 1] - x[..., 0], 0, None)
    if m.sum() < 20:
        return env
    lab = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2LAB)
    env_lab = cv2.cvtColor(np.clip(env, 0, 1), cv2.COLOR_RGB2LAB)
    # skin = brighter half of the diver (shorts/hair/mask are the dark rest)
    Ld = lab[..., 0][m]
    skin = m & (lab[..., 0] >= np.percentile(Ld, 45))
    a_med = float(np.median(lab[..., 1][skin]))
    b_med = float(np.median(lab[..., 2][skin]))
    if state is not None:  # temporal smoothing of the white-balance anchor
        if "ab" in state:
            a_med = 0.8 * state["ab"][0] + 0.2 * a_med
            b_med = 0.8 * state["ab"][1] + 0.2 * b_med
        state["ab"] = (a_med, b_med)
    out = np.empty_like(lab)
    # luminance: env-graded luminance, a touch brighter, keeps source detail
    out[..., 0] = np.clip(env_lab[..., 0] * g.l_gain
                          + 0.35 * (lab[..., 0] - cv2.GaussianBlur(lab[..., 0], (0, 0), 2)),
                          0, 100)
    # chroma: shift so median skin -> target, scale down slightly; darker
    # pixels (shorts, hair) get proportionally less colour.
    w = np.clip((lab[..., 0] - np.percentile(Ld, 15)) / 15.0, 0, 1)
    da = np.clip(lab[..., 1] - a_med, -g.dev_clip, g.dev_clip)
    db = np.clip(lab[..., 2] - b_med, -g.dev_clip, g.dev_clip)
    # speculars on wet skin go neutral, as in the reference
    hl = np.clip((97.0 - out[..., 0]) / 14.0, 0, 1)
    out[..., 1] = (da * g.chroma_scale + g.skin_a) * w * hl
    out[..., 2] = ((db * g.chroma_scale + g.skin_b) * w
                   + (1 - w) * -6.0) * hl   # dark parts sit slightly cool
    return lab_to_rgb_gamut(out)


def lab_to_rgb_gamut(lab: np.ndarray) -> np.ndarray:
    """Lab -> RGB, pulling out-of-gamut pixels toward neutral (keeping L)
    instead of hard-clipping channels, which creates hue-shifted specks."""
    lab = lab.copy()
    for _ in range(6):
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        bad = (rgb < -1e-3).any(-1) | (rgb > 1 + 1e-3).any(-1)
        if not bad.any():
            break
        lab[..., 1][bad] *= 0.75
        lab[..., 2][bad] *= 0.75
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
