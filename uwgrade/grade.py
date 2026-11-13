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
TGT_SKIN_Q = [8.0, 14.0, 22.0, 40.0, 57.0, 67.0, 74.0, 77.0, 83.0]   # low end stays deep: dark 'skin' pixels are hair / fins / deep shade
TGT_SHORTS_Q = [38.0, 50.0, 57.0, 68.0, 80.0, 87.0, 91.0, 93.0, 95.0]


@dataclass
class Params:
    # ---- NODE 1: per-channel levels (0..1), strength blends identity -> full levels
    l1_black: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    l1_white: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    l1_strength: float = 0.85
    l1_subject_relax: float = 0.55   # share of the black-point subtraction removed on the diver
    haze_lo: float = 0.22       # underwater-dark-channel range over the clip's rock pixels (calibrated)
    haze_hi: float = 0.38
    chroma_denoise_sigma: float = 1.0
    # ---- per-region calibration (filled by calibrate.py): source lightness quantiles and
    # ---- mean ab-versus-L ramp measured on the clip after node 1
    cal: dict = field(default_factory=dict)
    # ---- node behaviour
    keep_water: float = 0.35      # share of the source's local colour variation kept
    keep_rock: float = 0.35
    keep_skin: float = 0.60
    skin_cyan_damp: float = 0.30   # damp the green/cyan share of the kept skin variation
    keep_shorts: float = 0.25
    water_chroma_gain: float = 1.0
    water_dark_blend: float = 0.70   # ambiguous dark pixels (below the water's own low percentiles): share of the rock tone curve
    water_hue_tol: float = 14.0     # deg: soft limit of the hue's distance from the reference water hue ramp
    rock_chroma_gain: float = 0.95
    rock_hue_clamp: list = field(default_factory=lambda: [44.0, 72.0])
    haze_start: float = 0.45      # haze (0..1) where distant rock starts to fade into the water ramp
    haze_end: float = 1.0
    haze_cool: float = 0.85       # share of the rock grade replaced by the water grade at maximum haze
    haze_subject: float = 0.30    # same for the diver: a far / near-surface diver stays partly veiled
    rock_shadow_cool: float = 0.85   # share of dark rock pushed to the blue-black water ramp
    rock_cool_lo: float = 8.0        # L* below which rock is fully 'shadow-cooled'
    rock_cool_hi: float = 36.0       # L* above which rock keeps its warm bronze (lit tones)
    refl_start: float = 52.0      # L* where reflection highlights start to lose water chroma
    refl_end: float = 80.0
    refl_chroma: float = 3.5
    refl_hue: float = 65.0
    shadow_lift: float = 0.0      # added L* in the deepest tones (keeps shadows from crushing)
    # ---- skin / shorts targets
    skin_hue: float = 56.0
    skin_chroma: float = 14.0
    skin_chroma_shadow: float = 6.0
    shorts_a: float = -2.5
    shorts_b: float = -7.5
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

    CALIB_KEYS = ("l1_black", "l1_white", "l1_strength", "haze_lo", "haze_hi", "cal")

    def to_json(self) -> str:
        """Only what is *measured from the footage* is authoritative in the file (calibration); behaviour
        parameters come from the code defaults, optionally changed through the explicit 'overrides' dict.
        'behaviour_snapshot' is documentation only and is ignored on load."""
        d = asdict(self)
        return json.dumps({"calibration": {k: d[k] for k in self.CALIB_KEYS},
                           "behaviour_snapshot": {k: v for k, v in d.items() if k not in self.CALIB_KEYS},
                           "overrides": {}}, indent=1)

    @staticmethod
    def from_json(s: str) -> "Params":
        d = json.loads(s)
        p = Params()
        calib = d.get("calibration", {k: d[k] for k in Params.CALIB_KEYS if k in d})   # legacy flat files: calibration only
        for k, v in calib.items():
            setattr(p, k, v)
        for k, v in d.get("overrides", {}).items():
            setattr(p, k, v)
        return p

    def effective_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


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
    sq = np.minimum(sq, 99.0 + 0.001 * np.arange(len(sq)))          # no knot beyond L*=100
    s0 = max(0.5, min(3.0, (tq[1] - tq[0]) / (sq[1] - sq[0])))
    x_lo = max(sq[0] - tq[0] / s0, 0.0)
    xs = [x_lo, *sq]
    ys = [0.0 if x_lo > 0 else max(tq[0] - s0 * sq[0], 0.0), *tq]
    x_hi = min(sq[-1] + (100.0 - tq[-1]), 100.0)
    if x_hi > sq[-1] + 1:
        xs.append(x_hi)
        ys.append(min(100.0, tq[-1] + (x_hi - sq[-1])))
    xs, ys = np.asarray(xs), np.asarray(ys)
    keep = np.concatenate([[True], (np.diff(xs) > 1e-3) & (np.diff(ys) > -1e-6)])   # strictly increasing x
    xs, ys = xs[keep], ys[keep]
    return Curve(xs, ys)


def haze_dc(rgb: np.ndarray, diver: np.ndarray | None = None) -> np.ndarray:
    """Underwater dark channel (G and B only: red is already attenuated) at 1/4 resolution, smoothed.
    High = hazy / far, low = clear / near. The diver's pixels are excluded (white shorts or a dark fin would
    otherwise set the haze) and the hole is filled from the surrounding water by normalised convolution."""
    import cv2

    m = np.minimum(rgb[..., 1], rgb[..., 2]).astype(np.float32)
    h, w = m.shape
    sm = cv2.resize(m, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    if diver is not None:
        dq = cv2.resize(diver.astype(np.float32), (w // 4, h // 4), interpolation=cv2.INTER_AREA) > 0.05
        dq = cv2.dilate(dq.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        sm = np.where(dq, 1.0, sm).astype(np.float32)
    dc = cv2.erode(sm, np.ones((13, 13), np.uint8))
    valid = (dc < 0.99).astype(np.float32)                 # patches fully inside the diver stay 1.0 -> filled
    return blur(dc * valid, 8.0) / np.maximum(blur(valid, 8.0), 0.02)


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
    def node1(self, rgb: np.ndarray, diver: np.ndarray | None = None) -> np.ndarray:
        p = self.p
        bp = np.asarray(p.l1_black, np.float32)
        wp = np.asarray(p.l1_white, np.float32)
        if diver is not None and p.l1_subject_relax > 0:
            # the black point is the veil measured on the environment; the diver is nearer the lens and carries less
            # of it, so subtracting the full veil crushes his darkest pixels (hair, fins) to zero
            bp = bp[None, None, :] * (1.0 - p.l1_subject_relax * diver[..., None])
        full = (rgb - bp) / np.maximum(wp - bp, 1e-3)
        out = rgb + p.l1_strength * (full - rgb)
        return np.clip(out, 0.0, 1.0)

    def _src_mean(self, key, L):
        Lk, ak, bk = self.src_ab[key]
        return np.interp(L, Lk, ak).astype(np.float32), np.interp(L, Lk, bk).astype(np.float32)

    def _colorize(self, lab, key, C_fn, H_fn, keep, gain=1.0, refl=False, refl_w=1.0, shadow_cool=0.0, hue_clamp=None, cool_range=(2.0, 16.0), hue_tol=None):
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
            w = shadow_cool * (1.0 - smoothstep(cool_range[0], cool_range[1], Lo))
            wa, wb = polar(C.WATER_C(Lo) * 0.9, C.WATER_H(Lo))
            ta, tb = ta * (1 - w) + wa * w, tb * (1 - w) + wb * w
        oa, ob = ta + keep * ra, tb + keep * rb
        if hue_clamp is not None:
            # keep lit-tone hue inside the reference's range (no red/orange tail from the kept source variation);
            # blue-black shadows (below L~14) are left alone
            h = hue_deg(oa, ob)
            c = np.hypot(oa, ob)
            mid = 0.5 * (hue_clamp[0] + hue_clamp[1])
            d = ((h - mid + 180.0) % 360.0) - 180.0
            dc = np.clip(d, hue_clamp[0] - mid, hue_clamp[1] - mid)
            wgt = smoothstep(14.0, 24.0, Lo)
            oa, ob = polar(c, mid + d + wgt * (dc - d))
        if hue_tol is not None:
            # the kept source variation is rotated by a large angle; on a low-chroma colour it can swing the hue
            # (e.g. violet in dark water). Keep the hue within a soft tolerance of the reference hue ramp.
            h = hue_deg(oa, ob)
            c = np.hypot(oa, ob)
            d = ((h - Ht + 180.0) % 360.0) - 180.0
            oa, ob = polar(c, Ht + hue_tol * np.tanh(d / hue_tol))
        return np.stack([Lo, oa, ob], -1)

    def node_water(self, lab, surface_w=None):
        p = self.p
        key = "water"
        out = self._colorize(lab, key, C.WATER_C, C.WATER_H, p.keep_water, p.water_chroma_gain,
                             refl=True, refl_w=0.55, hue_tol=p.water_hue_tol)
        if surface_w is not None and "surf" in self.curves:
            sf = self._colorize(lab, "surf", C.WATER_C, C.WATER_H, p.keep_water, p.water_chroma_gain,
                                refl=True, refl_w=1.0, hue_tol=p.water_hue_tol)
            out = out + surface_w[..., None] * (sf - out)
        if p.water_dark_blend > 0 and "rock" in self.curves:
            # The water curve is steep (the clip's water spans a narrow lightness range) and floors at the water's own
            # 1st percentile, so a pixel darker than nearly all surface water -- typically hazy dark rock the classifier
            # called water -- collapses to black. Below the water's p1..p25 blend toward the rock tone curve instead.
            Lq = self.p.cal["water"]["L_q"]
            L = lab[..., 0]
            b = p.water_dark_blend * (1.0 - smoothstep(Lq[0], Lq[3], L))
            out[..., 0] = out[..., 0] * (1.0 - b) + self.curves["rock"](L) * b
        return out

    def node_rock(self, lab):
        p = self.p
        return self._colorize(lab, "rock", C.ROCK_C, C.ROCK_H, p.keep_rock, p.rock_chroma_gain,
                              shadow_cool=p.rock_shadow_cool, hue_clamp=tuple(p.rock_hue_clamp),
                              cool_range=(p.rock_cool_lo, p.rock_cool_hi))

    def node_skin(self, lab):
        p = self.p
        L = lab[..., 0]
        Lo = self.curves["skin"](L)
        a0, b0 = self._src_mean("skin", L)
        da, db = lab[..., 1] - a0, lab[..., 2] - b0
        da = np.where(da < 0, da * p.skin_cyan_damp, da)     # cyan/green contamination (caustics) -> mostly removed
        db = np.where(db < 0, db * p.skin_cyan_damp, db)
        # warm chroma fades out in the darks: hair, fins and deep shade stay dark and neutral, not auburn
        Ct = np.interp(Lo, [12, 22, 34, 48, 60, 100], [1.0, 2.5, p.skin_chroma_shadow, 11.0, p.skin_chroma, p.skin_chroma - 2]).astype(np.float32)
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
        Lo = np.interp(L, [0, 8, 20, 35, 60, 100], [0, 5.0, 13.0, 24.0, 46.0, 82.0]).astype(np.float32)
        ta, tb = polar(np.full_like(Lo, p.gear_chroma), p.gear_hue)
        return np.stack([Lo, ta, tb], -1)

    def haze(self, rgb: np.ndarray, diver=None) -> np.ndarray:
        import cv2

        p = self.p
        hz = np.clip((haze_dc(rgb, diver) - p.haze_lo) / max(p.haze_hi - p.haze_lo, 1e-3), 0.0, 1.0)
        return cv2.resize(hz, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ pipeline
    def __call__(self, rgb: np.ndarray, al: dict) -> np.ndarray:
        """rgb float32 HxWx3 in 0..1; al: dict of float32 HxW alphas:
        diver, skin, shorts, gear (parts partition the diver), rock, surface (water = remainder)."""
        p = self.p
        D = al["diver"].astype(np.float32)
        x = self.node1(rgb, D)
        lab = rgb2lab(x)
        if p.chroma_denoise_sigma > 0:
            lab[..., 1:] = blur(lab[..., 1:], p.chroma_denoise_sigma)
        rock = np.clip(al["rock"], 0, 1)
        # atmospheric perspective: hazy (far) rock fades into the water grade, near rock stays warm
        hz = smoothstep(p.haze_start, p.haze_end, self.haze(rgb, D))
        rock = rock * (1.0 - p.haze_cool * hz)
        surf = np.clip(al["surface"], 0, 1)
        water = np.clip(1.0 - rock - surf, 0, 1)
        s = rock + surf + water + 1e-6
        rock, surf, water = rock / s, surf / s, water / s
        out = np.zeros_like(lab)
        env_w = 1.0 - D
        lw = self.node_water(lab, surface_w=surf / np.maximum(surf + water, 1e-3))
        if env_w.max() > 0.001:
            lr = self.node_rock(lab)
            out += env_w[..., None] * ((rock[..., None]) * lr + (1 - rock)[..., None] * lw)
        if D.max() > 0.001:
            ps = al["skin"][..., None]
            ph = al["shorts"][..., None]
            pg = al["gear"][..., None]
            tot = np.maximum(ps + ph + pg, 1e-3)
            subj = (ps * self.node_skin(lab) + ph * self.node_shorts(lab) + pg * self.node_gear(lab)) / tot
            cs = (p.haze_subject * hz)[..., None]          # far / near-surface diver stays partly veiled
            subj = subj * (1.0 - cs) + lw * cs
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
