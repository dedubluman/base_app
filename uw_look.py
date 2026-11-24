"""Region-aware cinematic underwater look (shared by calibration and render).

Pipeline for one frame (float32 RGB 0..1, BT.709 gamma-encoded):

1. Region weights  - SAM 3.1 masks (subject / water / sky) plus, underwater,
                     a luminance-driven "sunlit zone" -> three soft weights
                     (top, body, subject) that sum to 1. Masks are snapped to
                     image edges with a guided filter, then feathered.
2. Regional colour - per scene, per region a cubic RGB polynomial (fitted to
                     the reference stills by uw_fit.py, shared part + per-region
                     delta). Output = sum_r weight_r * poly_r(frame): the
                     weights are soft, so there are no hard seams.
3. Roll-off        - tanh soft-clip of the highlights (no digital clipping).
4. Halation        - warm glow around bright bubbles / sun (screen blend of a
                     blurred, amber-tinted highlight layer).
5. Grain           - very subtle, luminance-dependent, per-frame film grain.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
REGIONS = ("top", "body", "subject")
SCENES = ("stand", "entry", "under")
import os
DEG = int(os.environ.get('UW_DEG', 3))


# ------------------------------------------------------------------ helpers
def smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def mono(x: np.ndarray, deg: int = DEG) -> np.ndarray:
    """x [...,3] -> polynomial features [..., n] (all monomials up to `deg`)."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                cols.append(r ** i * g ** j * b ** k)
    return np.stack(cols, -1).astype(np.float32)


def feat_degrees(deg: int = DEG) -> np.ndarray:
    d = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                d.append(i + j + k)
    return np.array(d)


def n_feat(deg: int = DEG) -> int:
    return (deg + 1) * (deg + 2) * (deg + 3) // 6


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter, gray guide, single-channel src (float32 0..1)."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    mg, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mg * ms
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return mean(a) * guide + mean(b)


def refine_mask(mask_small: np.ndarray, rgb: np.ndarray, r: int = 6, eps: float = 2e-3,
                feather: float = 2.5) -> np.ndarray:
    """uint8/float mask at SAM resolution -> soft full-res mask snapped to edges."""
    h, w = rgb.shape[:2]
    m = cv2.resize(mask_small.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
    guide = (rgb @ LUMA).astype(np.float32)
    m = np.clip(guided_filter(guide, m, r, eps), 0, 1)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m


def sunlit_zone(rgb: np.ndarray) -> np.ndarray:
    """Underwater 'upper water column': bright, bubble-filled light entering
    from above. Luminance of a heavily blurred copy x a vertical prior."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    lum = cv2.GaussianBlur(small @ LUMA, (0, 0), 6)
    lum = cv2.resize(lum, (w, h), interpolation=cv2.INTER_LINEAR)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    vert = 1.0 - smoothstep(0.32, 0.85, y)
    light = smoothstep(0.30, 0.62, lum)
    return np.clip(vert * (0.35 + 0.65 * light), 0, 1).astype(np.float32)


def region_weights(scene: str, rgb: np.ndarray, subject: np.ndarray | None,
                   water: np.ndarray | None, sky: np.ndarray | None) -> dict[str, np.ndarray]:
    """Soft weights for (top, body, subject); each full-res float32, sum = 1.
    Inputs are already-refined full-res masks (or None)."""
    h, w = rgb.shape[:2]
    zero = np.zeros((h, w), np.float32)
    s = zero if subject is None else subject
    if scene == "under":
        top = sunlit_zone(rgb)
    elif water is not None:
        # surface scenes: everything that is neither water nor the swimmer
        # (sky, mountains, town = hazy, sun-lit distance) is the "top" zone
        top = np.clip(1 - water, 0, 1)
        # sunlit shallows: non-water pixels in the lower part of the frame are
        # seabed rocks, not sky/mountains -> they belong to the body region
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        top = top * (1.0 - smoothstep(0.48, 0.64, y))
    else:
        top = zero if sky is None else sky
    if scene == "under":
        # Underwater colour follows depth / light, not identity: the silhouette
        # sits in whichever light zone it is in. The subject mask instead drives
        # a protection step (see `protect`), so the figure stays grounded in the
        # water column without amber light bleeding into it.
        return {"top": top, "body": 1 - top, "subject": zero, "protect": s}
    top = top * (1 - s)
    body = np.clip(1 - s - top, 0, 1)
    tot = s + top + body
    return {"top": top / tot, "body": body / tot, "subject": s / tot, "protect": s}


def protect(img: np.ndarray, mask: np.ndarray, desat: float = 0.30) -> np.ndarray:
    """Pull amber/teal saturation back inside the subject mask (luma kept)."""
    y = (img @ LUMA)[..., None]
    return y + (img - y) * (1.0 - desat * mask[..., None])


# ------------------------------------------------------------------- model
class Look:
    """Per scene and region: a cubic polynomial (`coef`), a safe affine
    fallback (`fb`), and a 3D colour-support grid (`support`, 16^3) saying how
    well the scene's reference still covered each input colour. Where support
    is low the cubic model is blended into the affine one, so unseen colours
    (e.g. dark olive algae) cannot extrapolate into wild hues."""

    GRID = 16

    def __init__(self, coef: dict, post: dict, fb: dict | None = None, support: dict | None = None):
        cv = lambda d: {s: {r: np.asarray(d[s][r], np.float32) for r in REGIONS} for s in d}  # noqa: E731
        self.coef = cv(coef)
        self.fb = cv(fb) if fb else None
        self.support = {s: {r: np.asarray(v[r], np.float32) for r in REGIONS} for s, v in (support or {}).items()}
        self.post = post

    @staticmethod
    def load(path: Path) -> "Look":
        d = json.loads(Path(path).read_text())
        return Look(d["coef"], d["post"], d.get("fb"), d.get("support"))

    def save(self, path: Path) -> None:
        ser = lambda d: {s: {r: d[s][r].tolist() for r in REGIONS} for s in d}  # noqa: E731
        Path(path).write_text(json.dumps({
            "coef": ser(self.coef), "fb": ser(self.fb) if self.fb else None,
            "support": {s: {r: v[r].tolist() for r in REGIONS} for s, v in self.support.items()},
            "post": self.post}))

    def support_at(self, rgb_small: np.ndarray, scene: str, region: str, shape) -> np.ndarray:
        """Support of each pixel colour for (scene, region); evaluated on a
        down-scaled frame (support is smooth) and resized back."""
        from scipy.ndimage import map_coordinates
        g = self.GRID
        c = np.clip(rgb_small.reshape(-1, 3) * g - 0.5, 0, g - 1).T
        v = map_coordinates(self.support[scene][region], c, order=1, mode="nearest")
        v = v.reshape(rgb_small.shape[:2]).astype(np.float32)
        return cv2.resize(v, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def regional(self, f: np.ndarray, scene: str, weights: dict, rgb: np.ndarray | None = None) -> np.ndarray:
        """sum_r weight_r * model_r(rgb) for one scene (no post effects).
        `f` = mono(rgb) [h,w,n], computed once per frame by the caller."""
        out = np.zeros(f.shape[:2] + (3,), np.float32)
        use_sup = self.fb is not None and scene in self.support and rgb is not None
        if use_sup:
            small = cv2.resize(rgb, (rgb.shape[1] // 4, rgb.shape[0] // 4), interpolation=cv2.INTER_AREA)
        for r in REGIONS:
            wr = weights[r]
            if wr.max() < 1e-4:
                continue
            m = f @ self.coef[scene][r]
            if use_sup:
                sup = self.support_at(small, scene, r, f.shape[:2])[..., None]
                m = sup * m + (1 - sup) * (f @ self.fb[scene][r])
            out += wr[..., None] * m
        return out


# ------------------------------------------------------------ finishing FX
def soft_clip(x: np.ndarray, knee: float = 0.86) -> np.ndarray:
    x = np.maximum(x, 0.0)
    over = x > knee
    y = x.copy()
    y[over] = knee + (1 - knee) * np.tanh((x[over] - knee) / (1 - knee))
    return y


def halation(img: np.ndarray, strength: float, threshold: float, tint: tuple, sigma_frac=(0.010, 0.035)):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    lum = img @ LUMA
    hl = np.clip((lum - threshold) / (1 - threshold), 0, 1) ** 1.5
    small = cv2.resize(hl, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    glow = 0.6 * cv2.GaussianBlur(small, (0, 0), sigma_frac[0] * w / 4) \
        + 0.4 * cv2.GaussianBlur(small, (0, 0), sigma_frac[1] * w / 4)
    glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
    layer = np.clip(glow[..., None] * np.asarray(tint, np.float32) * strength, 0, 0.9)
    return 1 - (1 - img) * (1 - layer)                    # screen blend


def grain(img: np.ndarray, frame_idx: int, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(1000 + frame_idx)
    n = rng.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)
    n /= max(n.std(), 1e-6)
    lum = img @ LUMA
    amp = amount * (1.15 - 0.75 * lum)                    # a bit more in shadows/mids
    return img + (n * amp)[..., None]


def finish(img: np.ndarray, frame_idx: int, post: dict) -> np.ndarray:
    img = soft_clip(img, post.get("knee", 0.86))
    img = halation(img, post.get("halation", 0.30), post.get("halation_thr", 0.78),
                   tuple(post.get("halation_tint", (1.0, 0.58, 0.26))))
    img = grain(img, frame_idx, post.get("grain", 0.009))
    return np.clip(img, 0, 1)


# --------------------------------------------------------------- timeline
def scene_weights(t: int, stand_to_entry=(96, 141), entry_to_under=(218, 236)) -> dict[str, float]:
    """Frame-index driven blend between the three reference looks.
    stand -> entry fades while the camera lowers into the water; entry ->
    under fades across the plunge through the foam (no hard cut exists)."""
    u = float(smoothstep(*entry_to_under, t))
    e = float(smoothstep(*stand_to_entry, t))
    return {"stand": (1 - u) * (1 - e), "entry": (1 - u) * e, "under": u}
