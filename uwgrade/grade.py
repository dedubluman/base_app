"""Node-based underwater grade.

    NODE 1  global base      per-channel levels (veil removal + cast neutralising), chroma de-noise
    NODE 2  water            lightness quantile-matched to the references, hue/chroma along the
                             reference's lightness-dependent petrol-blue ramp
    NODE 3  rocks / seabed   same machinery, warm brown / bronze ramp, source variation kept
    NODE 4  subject          skin, white shorts and dark gear graded separately
    NODE 5  surface          water node with reflection-highlight protection
    NODE 6  finishing        filmic shoulder, halo-limited micro-contrast, dither

All parameters are constant for the whole clip (calibrated once, numerically,
from keyframes), so the grade cannot "breathe". Only the masks change per frame.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import color as C
from .color import Curve, blur, hue_deg, lab2rgb, polar, ramp, rgb2lab, smoothstep

Q_LEVELS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
# Reference-derived lightness quantiles (L*), mean of REF1/REF2 (see README_uwgrade.md).
TGT_WATER_Q = [1.2, 3.1, 4.9, 10.0, 18.6, 27.2, 34.0, 38.0, 47.2]
TGT_ROCK_Q = [2.8, 6.9, 9.4, 16.1, 25.2, 34.7, 44.0, 50.0, 58.0]   # top capped: REF2 tail is reflections
TGT_SURF_Q = [1.5, 4.0, 6.5, 12.0, 21.0, 31.0, 42.0, 56.0, 82.0]   # reflections may go bright
TGT_SKIN_Q = [24.0, 30.0, 36.0, 47.0, 60.0, 70.0, 77.0, 80.0, 86.0]
TGT_SHORTS_Q = [38.0, 50.0, 57.0, 68.0, 80.0, 87.0, 91.0, 93.0, 95.0]


@dataclass
class Params:
    # ---- NODE 1: per-channel levels (0..1), strength blends identity -> full levels
    l1_black: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    l1_white: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    l1_strength: float = 0.85
    chroma_denoise_sigma: float = 1.0
    # ---- per-region calibration (filled by calibrate.py): source lightness quantiles and
    # ---- mean ab-versus-L ramp measured on the clip after node 1
    cal: dict = field(default_factory=dict)
    # ---- node behaviour
    keep_water: float = 0.35      # share of the source's local colour variation kept
    keep_rock: float = 0.45
    keep_skin: float = 0.75
    keep_shorts: float = 0.25
    water_chroma_gain: float = 1.0
    rock_chroma_gain: float = 1.0
    rock_shadow_cool: float = 0.65   # share of deep-shadow rock pushed to the blue-black water ramp
    refl_start: float = 52.0      # L* where reflection highlights start to lose water chroma
    refl_end: float = 80.0
    refl_chroma: float = 5.0
    refl_hue: float = 75.0
    shadow_lift: float = 0.0      # added L* in the deepest tones (keeps shadows from crushing)
    # ---- skin / shorts targets
    skin_hue: float = 56.0
    skin_chroma: float = 14.0
    skin_chroma_shadow: float = 6.0
    shorts_a: float = -1.5
    shorts_b: float = -5.0
    gear_chroma: float = 3.0
    gear_hue: float = 250.0
    gear_L_cut: float = 22.0
    # ---- NODE 6
    shoulder_knee: float = 84.0
    shoulder_cap: float = 97.5
    sharpen_amount: float = 0.30
    sharpen_sigma: float = 1.2
    sharpen_limit: float = 6.0
    clarity_amount: float = 0.10
    clarity_sigma: float = 9.0
    dither: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_json(s: str) -> "Params":
        d = json.loads(s)
        p = Params()
        for k, v in d.items():
            setattr(p, k, v)
        return p


def quantile_curve(src_q, tgt_q, shadow_lift: float = 0.0) -> Curve:
    """Monotone L* -> L* curve through (source quantile -> target quantile).
    Below the lowest knot it continues toward black with the local slope; above the
    highest it continues with slope ~1 and is capped at 100."""
    sq = np.asarray(src_q, np.float64)
    tq = np.asarray(tgt_q, np.float64).copy()
    tq[:4] += shadow_lift * np.array([1.0, 0.8, 0.5, 0.2])
    # enforce strictly increasing knots (quantile ties on flat regions)
    for i in range(1, len(sq)):
        if sq[i] <= sq[i - 1] + 0.05:
            sq[i] = sq[i - 1] + 0.05
        if tq[i] <= tq[i - 1] + 0.02:
            tq[i] = tq[i - 1] + 0.02
    s0 = max(0.5, min(3.0, (tq[1] - tq[0]) / (sq[1] - sq[0])))
    x_lo = max(sq[0] - tq[0] / s0, 0.0)
    xs = [x_lo, *sq]
    ys = [0.0 if x_lo > 0 else max(tq[0] - s0 * sq[0], 0.0), *tq]
    x_hi = sq[-1] + (100.0 - tq[-1])
    xs += [min(x_hi, 100.0)] if x_hi > sq[-1] + 1 else []
    ys += [min(100.0, tq[-1] + (xs[-1] - sq[-1]))] if x_hi > sq[-1] + 1 else []
    return Curve(xs, ys)


class Grader:
    def __init__(self, p: Params):
        self.p = p
        self._build()

    # ------------------------------------------------------------------ setup
    def _build(self):
        p = self.p
        cal = p.cal
        self.curves = {}
        self.src_ab = {}
        tgt = {"water": TGT_WATER_Q, "rock": TGT_ROCK_Q, "surf": TGT_SURF_Q, "skin": TGT_SKIN_Q, "shorts": TGT_SHORTS_Q}
        for k, tq in tgt.items():
            if k in cal:
                self.curves[k] = quantile_curve(cal[k]["L_q"], tq, p.shadow_lift if k in ("water", "rock", "surf") else 0.0)
                Lk, ak, bk = cal[k]["ab_ramp"]
                self.src_ab[k] = (np.asarray(Lk, np.float32), np.asarray(ak, np.float32), np.asarray(bk, np.float32))

    # ------------------------------------------------------------------ nodes
    def node1(self, rgb: np.ndarray) -> np.ndarray:
        p = self.p
        bp = np.asarray(p.l1_black, np.float32)
        wp = np.asarray(p.l1_white, np.float32)
        full = (rgb - bp) / np.maximum(wp - bp, 1e-3)
        out = rgb + p.l1_strength * (full - rgb)
        return np.clip(out, 0.0, 1.0)

    def _src_mean(self, key, L):
        Lk, ak, bk = self.src_ab[key]
        return np.interp(L, Lk, ak).astype(np.float32), np.interp(L, Lk, bk).astype(np.float32)

    def _colorize(self, lab, key, C_fn, H_fn, keep, gain=1.0, refl=False, refl_w=1.0, shadow_cool=0.0):
        p = self.p
        L = lab[..., 0]
        Lo = self.curves[key](L)
        a0, b0 = self._src_mean(key, L)
        da, db = lab[..., 1] - a0, lab[..., 2] - b0
        H_src = hue_deg(a0, b0)
        Ht = H_fn(Lo)
        d = np.deg2rad(Ht - H_src)
        cd, sd = np.cos(d), np.sin(d)
        Cs = np.hypot(a0, b0)
        scale = np.clip(C_fn(Lo) * gain / np.maximum(Cs, 4.0), 0.35, 1.6)
        ra = (da * cd - db * sd) * scale
        rb = (da * sd + db * cd) * scale
        Ct = C_fn(Lo) * gain
        if refl:
            s = smoothstep(p.refl_start, p.refl_end, Lo) * refl_w
            Ct = Ct * (1 - s) + p.refl_chroma * s
            Ht = Ht * (1 - s) + p.refl_hue * s
        ta, tb = polar(Ct, Ht)
        if shadow_cool > 0:
            # deep crevices / shadows go blue-black like the references' water-filled shadows,
            # instead of staying brown: blend toward the water ramp as lightness falls
            w = shadow_cool * (1.0 - smoothstep(2.0, 16.0, Lo))
            wa, wb = polar(C.WATER_C(Lo) * 0.9, C.WATER_H(Lo))
            ta, tb = ta * (1 - w) + wa * w, tb * (1 - w) + wb * w
        return np.stack([Lo, ta + keep * ra, tb + keep * rb], -1)

    def node_water(self, lab, surface_w=None):
        p = self.p
        key = "water"
        out = self._colorize(lab, key, C.WATER_C, C.WATER_H, p.keep_water, p.water_chroma_gain,
                             refl=True, refl_w=0.55)
        if surface_w is not None and "surf" in self.curves:
            sf = self._colorize(lab, "surf", C.WATER_C, C.WATER_H, p.keep_water, p.water_chroma_gain,
                                refl=True, refl_w=1.0)
            out = out + surface_w[..., None] * (sf - out)
        return out

    def node_rock(self, lab):
        p = self.p
        return self._colorize(lab, "rock", C.ROCK_C, C.ROCK_H, p.keep_rock, p.rock_chroma_gain,
                              shadow_cool=p.rock_shadow_cool)

    def node_skin(self, lab):
        p = self.p
        L = lab[..., 0]
        Lo = self.curves["skin"](L)
        a0, b0 = self._src_mean("skin", L)
        da, db = lab[..., 1] - a0, lab[..., 2] - b0
        Ct = np.interp(Lo, [20, 45, 60, 100], [p.skin_chroma_shadow, 11.0, p.skin_chroma, p.skin_chroma - 2]).astype(np.float32)
        ta, tb = polar(Ct, p.skin_hue)
        H_src = hue_deg(a0, b0)
        d = np.deg2rad(p.skin_hue - H_src) * 0.0   # skin is already warm: keep deviations unrotated
        return np.stack([Lo, ta + p.keep_skin * da, tb + p.keep_skin * db], -1)

    def node_shorts(self, lab):
        p = self.p
        L = lab[..., 0]
        Lo = self.curves["shorts"](L)
        a0, b0 = self._src_mean("shorts", L)
        da, db = lab[..., 1] - a0, lab[..., 2] - b0
        k = np.clip(Lo / 90.0, 0.0, 1.0)
        return np.stack([Lo, p.shorts_a * k + p.keep_shorts * da, p.shorts_b * k + p.keep_shorts * db], -1)

    def node_gear(self, lab):
        """Hair, snorkel, fins: deep neutral-cool blacks with their shape kept."""
        p = self.p
        L = lab[..., 0]
        Lo = np.interp(L, [0, 8, 20, 35, 60, 100], [0, 2.5, 8.0, 18.0, 40.0, 80.0]).astype(np.float32)
        ta, tb = polar(np.full_like(Lo, p.gear_chroma), p.gear_hue)
        return np.stack([Lo, ta, tb], -1)

    # ------------------------------------------------------------------ pipeline
    def __call__(self, rgb: np.ndarray, al: dict) -> np.ndarray:
        """rgb float32 HxWx3 in 0..1; al: dict of float32 HxW alphas:
        diver, skin, shorts, gear (parts partition the diver), rock, surface (water = remainder)."""
        p = self.p
        x = self.node1(rgb)
        lab = rgb2lab(x)
        if p.chroma_denoise_sigma > 0:
            lab[..., 1:] = blur(lab[..., 1:], p.chroma_denoise_sigma)
        D = al["diver"].astype(np.float32)
        rock = np.clip(al["rock"], 0, 1)
        surf = np.clip(al["surface"], 0, 1)
        water = np.clip(1.0 - rock - surf, 0, 1)
        s = rock + surf + water + 1e-6
        rock, surf, water = rock / s, surf / s, water / s
        out = np.zeros_like(lab)
        env_w = 1.0 - D
        if env_w.max() > 0.001:
            lw = self.node_water(lab, surface_w=surf / np.maximum(surf + water, 1e-3))
            lr = self.node_rock(lab)
            out += env_w[..., None] * ((rock[..., None]) * lr + (1 - rock)[..., None] * lw)
        if D.max() > 0.001:
            ps = al["skin"][..., None]
            ph = al["shorts"][..., None]
            pg = al["gear"][..., None]
            tot = np.maximum(ps + ph + pg, 1e-3)
            subj = (ps * self.node_skin(lab) + ph * self.node_shorts(lab) + pg * self.node_gear(lab)) / tot
            out += D[..., None] * subj
        return self.finish(out, D)

    def finish(self, lab: np.ndarray, D: np.ndarray) -> np.ndarray:
        p = self.p
        L = lab[..., 0]
        # filmic shoulder
        k, cap = p.shoulder_knee, p.shoulder_cap
        over = L > k
        L = np.where(over, k + (cap - k) * np.tanh((L - k) / (cap - k)), L)
        # halo-limited micro-contrast + mid-scale clarity (on lightness only)
        if p.sharpen_amount > 0:
            fine = np.clip(L - blur(L, p.sharpen_sigma), -p.sharpen_limit, p.sharpen_limit)
            L = L + p.sharpen_amount * fine
        if p.clarity_amount > 0:
            mid = np.clip(L - blur(L, p.clarity_sigma), -10, 10)
            L = L + p.clarity_amount * mid
        lab = np.stack([np.clip(L, 0, 100), lab[..., 1], lab[..., 2]], -1)
        rgb = lab2rgb(lab)
        if p.dither:
            rng = np.random.default_rng(getattr(self, "seed", 1234))
            n = (rng.random(rgb.shape, np.float32) - rng.random(rgb.shape, np.float32)) / 1023.0
            rgb = np.clip(rgb + n, 0.0, 1.0)
        return rgb


def save_params(p: Params, path: Path):
    Path(path).write_text(p.to_json())


def load_params(path: Path) -> Params:
    return Params.from_json(Path(path).read_text())
