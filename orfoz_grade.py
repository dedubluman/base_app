#!/usr/bin/env python3
"""Aquarium relight of a grouper (orfoz) with SAM 3.1 masks.

Layers (float32, BT.709 display-referred RGB 0..1):
  base       light temporal + chroma denoise of the very dark source, then a
             luminance-preserving exposure lift with a soft highlight shoulder
  aquarium   turquoise/teal water tones (luma gradient map mixed with a
             hue-pushed copy of the source), warm coral/gold highlights,
             vibrance, lifted-but-not-milky blacks
  spotlight  feathered light pool that follows the grouper mask; the rest of
             the frame is a little darker and less saturated
  grouper    SAM 3.1 mask refined against the lifted luma (guided filter), lit
             about +2 stops, white-balanced from its own pale spots, recoloured
             with a luma gradient map (keeps every spot) and given local contrast

    python orfoz_grade.py --input inputs/orfoz.mov --masks masks/orfoz_sam__cuttle \
        --objs 1 --output output/orfoz_graded.mp4 [--frames 0,100 --stills DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from shades.config import Encode
from shades.io import Encoder, iter_frames, probe, read_frames_at

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

P = dict(
    # denoise (source is 8-bit and very dark: lifting it amplifies noise)
    t_alpha_still=0.35,     # weight of the new frame where nothing moves
    t_motion_thr=0.020,     # abs diff (0..1) at which the new frame wins fully
    t_shadow_only=0.45,     # luma above which temporal smoothing fades out
    chroma_sigma=2.2,       # chroma blur in dark areas (px)
    shadow_bilat=0.028,     # bilateral sigmaColor on the dark, pre-lift frame (hides 8x8 codec blocks)
    # base exposure for the whole frame (stops, linear light)
    bg_a=2.0, bg_g=0.80, bg_k=0.70,      # shadow-lift curve: (a*y)^g, soft knee at k
    veil=0.55,              # share of the clip's black floor to remove
    # aquarium look
    hue_cyan=1.35,          # how far olive/green water is pushed to cyan
    sat=1.45, vibrance=1.25,
    gm_mix=0.70,            # luma gradient-map share
    gm=[(0.00, "#03141a"), (0.15, "#05303b"), (0.32, "#09687a"), (0.48, "#18a7b5"),
        (0.60, "#6fd3c4"), (0.70, "#f2c9a0"), (0.85, "#ffb98a"), (1.00, "#fff0e0")],
    warm_hi=0.40,           # coral/gold in the brightest rock and sand
    black_lift=(0.010, 0.035, 0.045),
    contrast=1.02,
    clarity=0.14, clarity_sigma=9.0,
    # spotlight
    spot_pool=0.90,         # stops of extra light inside the pool
    spot_dim=-0.50,         # stops removed outside it
    spot_sat=0.80,          # saturation outside the pool
    spot_inner=0.45,        # sigma of the pool core, x fish radius
    spot_outer=1.10,        # sigma of the wide skirt, x fish radius
    spot_warm=0.10,
    spot_ema=0.55,          # temporal smoothing of the pool map
    # grouper
    fish_a=6.5, fish_g=0.58, fish_k=0.78,
    fish_gm_mix=0.55,
    fish_gm=[(0.00, "#07100f"), (0.12, "#1a2a26"), (0.28, "#3d4632"), (0.44, "#7a7248"),
             (0.62, "#c9b57f"), (0.80, "#f3e3b8"), (1.00, "#fff8e8")],
    fish_warm=(1.10, 1.00, 0.84),
    fish_detail=0.40, fish_detail_sigma=5.0,
    fish_floor=0.14, fish_bilat=0.07, fish_volume=0.30, fish_top=0.28, fish_rim=0.18,
    grain=0.010,            # luma grain, strongest in shadows: hides banding after the big lift
    fish_erode=2, fish_feather=4.0,
    fish_glow=0.55,         # extra stops on the fish itself
    fade_frames=10,
)


# ---------------------------------------------------------------- helpers --
def luma(x):
    return x @ LUMA


def smoothstep(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def gblur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def gradient_map(t, stops):
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.stack([hexrgb(s[1]) for s in stops])
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, xs, cs[:, c]) for c in range(3)], -1).astype(np.float32)


def tone(y, stops, white):
    """Gamma-encoded luma -> gamma-encoded luma: exposure in linear light plus
    an extended-Reinhard shoulder so highlights roll off instead of clipping."""
    lin = np.power(np.maximum(y, 0), 2.2) * (2.0 ** stops)
    out = lin * (1 + lin / (white * white)) / (1 + lin)
    return np.power(out, 1 / 2.2)


def curve_rgb(x, a, g, k, gain_cap=14.0):
    """Shadow-lifting power curve on luma, (a*y)^g, with a tanh knee at k so
    highlights roll off instead of clipping; applied as a per-pixel ratio."""
    y = luma(x)
    v = np.power(np.maximum(a * y, 0), g)
    v = np.where(v < k, v, k + (1 - k) * np.tanh((v - k) / (1 - k)))
    r = np.clip(v / np.maximum(y, 1e-4), 0, gain_cap)[..., None]
    return x * r


def lift_rgb(x, stops, white, gain_cap=7.0):
    y = luma(x)
    y2 = tone(y, stops, white)
    r = np.clip(y2 / np.maximum(y, 1e-4), 0, gain_cap)[..., None]
    return x * r


def sat(x, s, vib=0.0):
    y = luma(x)[..., None]
    c = np.abs(x - y).max(-1, keepdims=True)
    s_eff = s * (1 + vib * (1 - np.clip(c * 3.0, 0, 1)))
    return y + (x - y) * s_eff


def guided(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.blur(I, k), cv2.blur(p, k)
    a = (cv2.blur(I * p, k) - mI * mp) / (cv2.blur(I * I, k) - mI * mI + eps)
    b = mp - a * mI
    return cv2.blur(a, k) * I + cv2.blur(b, k)


def hue_push(x, amount):
    """Rotate olive/green hues (40..140 deg) toward cyan, weighted by saturation."""
    hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    w = smoothstep(35, 70, h) * (1 - smoothstep(130, 165, h)) * np.clip(s * 2.5, 0, 1)
    hsv[..., 0] = (h + amount * 70.0 * w) % 360.0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ----------------------------------------------------------------- masks --
def shift(m, dx, dy, w, h):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)


def load_masks(mdir: Path, objs: list[int], n: int, w: int, h: int, log=print):
    """Union of the chosen SAM objects per frame, with area-outlier repair."""
    per = np.zeros((n, h, w), bool)
    have = np.zeros(n, bool)
    for g in objs:
        d = mdir / f"obj_{g}"
        for p in d.glob("*.png"):
            i = int(p.stem)
            if i < n:
                per[i] |= cv2.imread(str(p), 0) > 127
                have[i] = True
    area = per.reshape(n, -1).sum(1).astype(np.float64)
    idx = np.flatnonzero(have & (area > 0))
    bad = set()
    for k, i in enumerate(idx):
        nb = [area[j] for j in idx[max(k - 3, 0):k + 4] if j != i]
        if nb and abs(area[i] / np.median(nb) - 1) > 0.30:
            bad.add(int(i))
    for i in sorted(bad):
        prev = max((j for j in idx if j < i and j not in bad), default=None)
        nxt = min((j for j in idx if j > i and j not in bad), default=None)
        src = [j for j in (prev, nxt) if j is not None]
        if not src:
            continue
        cs = {j: np.argwhere(per[j]).mean(0) for j in src}
        if len(src) == 2:
            t = (i - prev) / (nxt - prev)
            tgt = cs[prev] * (1 - t) + cs[nxt] * t
            acc = sum(shift(per[j], *(tgt - cs[j])[::-1], w, h) * (1 - abs(i - j) / (nxt - prev))
                      for j in src)
        else:
            acc = per[src[0]].astype(np.float32)
        log(f"  repaired mask {i}: area {int(area[i])} -> {int((acc > .5).sum())}")
        per[i] = acc > 0.5
    present = np.flatnonzero(per.reshape(n, -1).any(1))
    for i in np.flatnonzero(~per.reshape(n, -1).any(1)):
        if len(present):
            j = int(present[np.argmin(np.abs(present - i))])
            if abs(j - i) <= 8:
                per[i] = per[j]
                have[i] = True
                log(f"  filled frame {i} from {j}")
    return per, have


# ----------------------------------------------------------------- grade --
class Grader:
    def __init__(self, p, masks, have, veil, fish_gain):
        self.p, self.masks, self.have = p, masks, have
        self.n = len(masks)
        self.veil = veil
        self.fish_gain = fish_gain
        self.prev = None        # recursive temporal denoise state
        self.pool = None        # EMA of the spotlight map
        area = masks.reshape(self.n, -1).sum(1).astype(np.float64)
        pres = (have & (area > 200)).astype(np.float64)
        k = max(p["fade_frames"], 1)
        pad = np.pad(pres, k, mode="edge")
        self.presence = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")
        self.area = area

    def denoise(self, x):
        p = self.p
        y = luma(x)
        if self.prev is not None:
            d = np.abs(x - self.prev).max(-1)
            d = cv2.blur(d, (7, 7))
            move = smoothstep(0.0, p["t_motion_thr"], d)
            dark = 1 - smoothstep(0.08, p["t_shadow_only"], y)
            a = 1 - (1 - p["t_alpha_still"]) * (1 - move) * dark
            x = self.prev * (1 - a[..., None]) + x * a[..., None]
        self.prev = x
        # chroma denoise only where it is dark (noise lives there)
        y = luma(x)[..., None]
        ch = gblur(x - y, p["chroma_sigma"])
        wd = 1 - smoothstep(0.10, 0.45, y)
        x = y + (x - y) * (1 - wd) + ch * wd
        if p["shadow_bilat"]:
            xb = cv2.bilateralFilter(np.ascontiguousarray(x, dtype=np.float32), 9,
                                     p["shadow_bilat"], 5)
            wb = (1 - smoothstep(0.08, 0.30, luma(x)))[..., None]
            x = x * (1 - wb) + xb * wb
        return x

    def background(self, x):
        p = self.p
        x = np.clip((x - p["veil"] * self.veil) / (1 - p["veil"] * self.veil), 0, 1)
        x = curve_rgb(x, p["bg_a"], p["bg_g"], p["bg_k"])
        # olive/green -> turquoise, then vibrance
        xh = hue_push(np.clip(x, 0, 1), p["hue_cyan"])
        xh = sat(xh, p["sat"], p["vibrance"])
        y = luma(np.clip(x, 0, 1))
        gm = gradient_map(y, p["gm"])
        # keep texture: gradient map carries the colour, source carries detail
        out = xh * (1 - p["gm_mix"]) + gm * p["gm_mix"] * (y / np.maximum(luma(gm), 1e-3))[..., None] ** 0.55
        # warm coral/gold in the brightest rock and sand
        warm = smoothstep(0.50, 0.85, y)[..., None] * p["warm_hi"]
        out = out * (1 + warm * np.array([0.35, 0.05, -0.30], np.float32))
        out = np.maximum(out, 0) + np.array(p["black_lift"], np.float32)
        # clarity
        yb = luma(out)
        det = yb - gblur(yb, p["clarity_sigma"])
        out = out + (det * p["clarity"])[..., None]
        # gentle S-curve
        c = p["contrast"]
        out = np.clip(out, 0, 1)
        out = 0.5 + (out - 0.5) * c
        return np.clip(out, 0, 1)

    def spotlight(self, i, alpha, h, w):
        p = self.p
        f = float(self.presence[i])
        if alpha.sum() < 50:
            S = np.zeros((h, w), np.float32)
        else:
            ys, xs = np.nonzero(alpha > 0.5)
            r = max(float(np.sqrt(len(xs) / np.pi)), 12.0)
            a = alpha.astype(np.float32)
            inner = gblur(a, p["spot_inner"] * r)
            inner /= max(inner.max(), 1e-3)
            outer = gblur(a, p["spot_outer"] * r)
            outer /= max(outer.max(), 1e-3)
            S = np.clip(0.65 * inner + 0.55 * outer, 0, 1)
        if self.pool is not None:
            e = p["spot_ema"]
            S = self.pool * e + S * (1 - e)
        self.pool = S
        return S, f

    def apply_spot(self, bg, S, f):
        p = self.p
        stops = f * (p["spot_pool"] * S + p["spot_dim"] * (1 - S))
        out = bg * (2.0 ** stops)[..., None]
        out = sat(out, 1 - (1 - p["spot_sat"]) * f * (1 - S)[..., None] * 1.0)
        warm = (p["spot_warm"] * f * S)[..., None]
        out = out * (1 + warm * np.array([1.0, 0.25, -0.8], np.float32))
        return out

    def fish(self, x, alpha_soft):
        p = self.p
        x = x * self.fish_gain
        x = x * np.array(p["fish_warm"], np.float32)
        x = curve_rgb(x, p["fish_a"], p["fish_g"], p["fish_k"])
        y = np.clip(luma(x), 0, 1)
        # the body is crushed to ~0 in the source: smooth codec noise, then
        # float the black floor so the body reads as dark bronze, not holes
        ys = cv2.bilateralFilter(y.astype(np.float32), 7, p["fish_bilat"], 3)
        dk = 1 - smoothstep(0.08, 0.40, ys)
        ys = ys * (1 - dk) + gblur(ys, 3.5) * dk
        t = p["fish_floor"] + (1 - p["fish_floor"]) * ys
        # soft key light from above: volume from the mask's own geometry
        am = (alpha_soft > 0.5).astype(np.uint8)
        dist = np.zeros(am.shape, np.float32)
        if am.any():
            dist = cv2.distanceTransform(am, cv2.DIST_L2, 5)
            dn = np.clip(dist / max(float(dist.max()), 1.0), 0, 1) ** 0.6
            rows = np.flatnonzero(am.any(1))
            v = np.clip((np.arange(am.shape[0], dtype=np.float32) - rows[0])
                        / max(rows[-1] - rows[0], 1), 0, 1)[:, None]
            shade = 1 + p["fish_volume"] * (dn - 0.5) + p["fish_top"] * (0.5 - v)
            t = np.clip(t * shade, 0, 1)
        gm = gradient_map(t, p["fish_gm"])
        out = x * (1 - p["fish_gm_mix"]) + gm * p["fish_gm_mix"] * (t / np.maximum(luma(gm), 1e-3))[..., None] ** 0.6
        yb = luma(out)
        det = yb - gblur(yb, p["fish_detail_sigma"])
        out = out + (det * p["fish_detail"])[..., None]
        out = out * (2.0 ** p["fish_glow"])
        # rim light on the upward-facing edge detaches the fish from the rock
        if am.any() and p["fish_rim"]:
            a8 = gblur(am.astype(np.float32), 5.0)
            gy = cv2.Sobel(a8, cv2.CV_32F, 0, 1, ksize=3)
            up = np.clip(gy / max(float(np.abs(gy).max()), 1e-3) * 1.6, 0, 1)
            band = (1 - smoothstep(0.0, 16.0, dist)) * (dist > 0)
            out = out + (up * band * p["fish_rim"])[..., None] * np.array([1.0, 0.86, 0.62], np.float32)
        return np.clip(out, 0, 1)

    def refine_alpha(self, i, guide):
        p = self.p
        lo, hi = max(i - 1, 0), min(i + 1, self.n - 1)
        mf = (0.25 * self.masks[lo] + 0.5 * self.masks[i] + 0.25 * self.masks[hi]).astype(np.float32)
        m8 = (mf > 0.5).astype(np.uint8)
        if p["fish_erode"]:
            m8 = cv2.erode(m8, np.ones((3, 3), np.uint8), iterations=p["fish_erode"])
        a = gblur(0.6 * m8.astype(np.float32) + 0.4 * mf, p["fish_feather"])
        a = guided(guide.astype(np.float32), a, 4, 1e-3)
        return np.clip(a, 0, 1)

    def frame(self, i, x):
        x = self.denoise(x)
        h, w = x.shape[:2]
        m = self.masks[i]
        bg = self.background(x)
        guide = np.clip(luma(lift_rgb(x, 1.8, 3.0)), 0, 1)
        alpha = self.refine_alpha(i, guide) if m.any() else np.zeros((h, w), np.float32)
        S, f = self.spotlight(i, alpha, h, w)
        bg = self.apply_spot(bg, S, f)
        fish = self.fish(x, alpha)
        a = (alpha * min(1.0, 0.0 + f * 1.0 + (1.0 if m.any() else 0.0)))[..., None]
        out = bg * (1 - a) + fish * a
        if self.p["grain"]:
            rng = np.random.default_rng(1000 + i)
            gw = (1.0 - 0.6 * np.clip(luma(out), 0, 1))[..., None]
            out = out + rng.standard_normal(out.shape[:2] + (1,), dtype=np.float32) * self.p["grain"] * gw
        return np.clip(out, 0, 1)


def estimate_fish_gain(frames_idx, src, info, masks, have):
    """White-balance gain from the pale spots on the grouper (clip-wide)."""
    cols = []
    for i, x in read_frames_at(src, info, frames_idx).items():
        m = masks[i]
        if not m.any():
            continue
        y = luma(x)
        thr = np.percentile(y[m], 85)
        sel = m & (y >= thr)
        if sel.sum() > 20:
            cols.append(x[sel].mean(0))
    if not cols:
        return np.ones(3, np.float32)
    c = np.median(np.stack(cols), 0)
    g = (c.mean() / np.maximum(c, 1e-4)).astype(np.float32)
    return np.clip(g, 0.6, 2.5)


def estimate_veil(src, info, step=12):
    lo = []
    for i, x in enumerate(iter_frames(src, info)):
        if i % step == 0:
            lo.append(np.percentile(x.reshape(-1, 3), 0.5, axis=0))
    return np.median(np.stack(lo), 0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--objs", default="1", help="comma list of SAM object ids")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--frames", help="comma list of frame indices to dump as stills")
    ap.add_argument("--stills", type=Path)
    ap.add_argument("--params", help="JSON overrides for P")
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()
    if a.params:
        P.update(json.loads(a.params))
    for k in ("gm", "fish_gm"):
        P[k] = [tuple(s) for s in P[k]]

    info = probe(a.input)
    w, h, n = info.width, info.height, info.n_frames
    print(f"[grade] {w}x{h} {info.fps} {n} frames audio={info.has_audio}", flush=True)
    objs = [int(s) for s in a.objs.split(",") if s != ""]
    masks, have = load_masks(a.masks, objs, n, w, h)
    print(f"[grade] masks on {int(have.sum())}/{n} frames", flush=True)

    veil = estimate_veil(a.input, info)
    gain = estimate_fish_gain(list(range(0, n, 20)), a.input, info, masks, have)
    print(f"[grade] veil {veil.round(4)} fish WB gain {gain.round(3)}", flush=True)
    g = Grader(P, masks, have, veil, gain)

    want = None
    if a.frames:
        want = sorted(int(s) for s in a.frames.split(","))
        a.stills.mkdir(parents=True, exist_ok=True)
        last = want[-1]
    enc = None
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        enc = Encoder(a.input, a.output, info, Encode(crf=a.crf))
    for i, x in enumerate(iter_frames(a.input, info)):
        if want is not None and not a.output and i > last:
            break
        y = g.frame(i, x)           # temporal state needs every frame in order
        if want is not None and i in want:
            pair = np.concatenate([np.clip(lift_rgb(x, 1.6, 3.0), 0, 1), y], 1)
            bgr = (np.clip(y, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(str(a.stills / f"graded_{i:05d}.png"), bgr)
            cv2.imwrite(str(a.stills / f"pair_{i:05d}.jpg"),
                        (pair[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if enc is not None:
            enc.write(y)
            if i % 25 == 0:
                print(f"[grade] frame {i}/{n}", flush=True)
    if enc is not None:
        enc.close()
        print(f"[grade] wrote {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
