#!/usr/bin/env python3
"""Masked grouper grade, take 2: one common base, small fish-only difference.

  base       gray-world white balance from the NEUTRAL references (pale rock,
             gravel), one shadow-lifting tone curve - identical for fish and
             background (no water tint, no gradient map, no teal wash)
  background base + saturation held in 0.15-0.30, pale rock/gravel/highlight
             forced neutral (S <= 0.12), red-brown algae (10-40 deg) kept,
             shadows only a faint cool lean (205-210 deg), at most ~12 % darker
  grouper    inside the SAM alpha x a soft luma weight (0 at 3 %, 1 at 10 %,
             1 until 85 %, 0 at 95 %) so crevice shadow and bright rock stay put:
             body warm brown (hue ~28, S ~0.50), spots golden cream (hue ~44,
             S ~0.38), +0.5 stop at most (gamma), +12 % local contrast, no clip

    python orfoz_grade2.py --input inputs/orfoz.mov --alpha masks/orfoz_v2/alpha \
        --output output/orfoz_graded_2.mp4 [--frames 20,212 --stills DIR] [--metrics]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from orfoz_grade import gblur, luma, smoothstep
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe, read_frames_at

P = dict(
    # white balance from neutral references
    wb_ylo=0.20, wb_yhi=0.85,
    # common base tone: log curve log(1+c*y)/log(1+c); lifts the toe (fish body ~0.12, so the
    # luma weight is on) yet keeps true blacks (crushed 0.002 -> 0.02) and rolls highlights off
    base_c=100.0,
    # background
    sat_gain=1.15, sat_cap=0.55, shadow_sat=0.14,
    neutral_sat=0.08,              # pale rock / gravel / top-right highlight
    shadow_hue=207.0, shadow_lean=1.0,
    bg_dim=0.12,                   # darker than base outside the fish pool, at most
    pool_sigma=1.6,                # pool sigma, x fish radius
    # grouper
    body_h=28.0, body_s=0.50, spot_h=44.0, spot_s=0.38,
    gamma=0.80,                    # <= +0.5 stop on mid-tones (0.18 -> 0.25 is +0.47 stop)
    local_contrast=1.12, lc_sigma=7.0,
    knee=0.85, ceil=0.90,          # keeps the fish 99th percentile <= 0.92
    gate=(0.03, 0.10, 0.85, 0.95),
    grain=0.008,
    # denoise (very dark 8-bit source)
    t_alpha_still=0.35, t_motion_thr=0.020, chroma_sigma=2.2, shadow_bilat=0.028,
)


def hsv(x):
    return cv2.cvtColor(np.ascontiguousarray(np.clip(x, 0, 1), dtype=np.float32), cv2.COLOR_RGB2HSV)


def rgb(h):
    return cv2.cvtColor(h.astype(np.float32), cv2.COLOR_HSV2RGB)


def knee_curve(v, k, top=1.0):
    return np.where(v < k, v, k + (top - k) * np.tanh((v - k) / max(top - k, 1e-3)))


# ----------------------------------------------------------------- alpha --
def load_alpha(adir: Path, i: int, w: int, h: int):
    p = adir / f"{i:05d}.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    return np.zeros((h, w), np.float32) if a is None else a.astype(np.float32) / 255


# ----------------------------------------------------------------- grade --
class Grader:
    def __init__(self, p, adir, w, h, gains):
        self.p, self.adir, self.w, self.h, self.gains = p, adir, w, h, gains
        self.prev = None
        self.pool = None

    def denoise(self, x):
        p = self.p
        y = luma(x)
        if self.prev is not None:
            d = cv2.blur(np.abs(x - self.prev).max(-1), (7, 7))
            move = smoothstep(0.0, p["t_motion_thr"], d)
            dark = 1 - smoothstep(0.08, 0.45, y)
            a = 1 - (1 - p["t_alpha_still"]) * (1 - move) * dark
            x = self.prev * (1 - a[..., None]) + x * a[..., None]
        self.prev = x
        y = luma(x)[..., None]
        ch = gblur(x - y, p["chroma_sigma"])
        wd = 1 - smoothstep(0.10, 0.45, y)
        x = y + (x - y) * (1 - wd) + ch * wd
        xb = cv2.bilateralFilter(np.ascontiguousarray(x, dtype=np.float32), 9, p["shadow_bilat"], 5)
        wb = (1 - smoothstep(0.08, 0.30, luma(x)))[..., None]
        return x * (1 - wb) + xb * wb

    def base(self, x):
        """Common base for fish and background: white balance + tone curve."""
        p = self.p
        x = np.clip(x * self.gains, 0, 1)
        y = luma(x)
        v = np.log1p(p["base_c"] * np.maximum(y, 0)) / np.log1p(p["base_c"])
        return np.clip(x * np.clip(v / np.maximum(y, 1e-4), 0, 14)[..., None], 0, 1)

    def background(self, b):
        """Base -> background look. No gradient map, no hue push toward water."""
        p = self.p
        hh = hsv(b)
        H, S, V = hh[..., 0], hh[..., 1], hh[..., 2]
        algae = smoothstep(0, 8, H) * (1 - smoothstep(46, 56, H)) * smoothstep(0.20, 0.35, S)  # red-brown 10-40 deg
        S2 = np.clip(S * p["sat_gain"], 0, p["sat_cap"])
        # neutral surfaces: bright, low chroma -> truly neutral
        neu = smoothstep(0.28, 0.45, V) * (1 - smoothstep(0.22, 0.40, S)) * (1 - algae)
        S2 = S2 * (1 - neu) + np.minimum(S2, p["neutral_sat"]) * neu
        out = rgb(np.stack([H, S2, V], -1))
        out = out * (luma(b) / np.maximum(luma(out), 1e-4))[..., None]
        # shadows: mix toward a faint cool colour (205-210 deg) in RGB, so the olive never
        # passes through green on the way (hue interpolation would)
        cool = rgb(np.stack([np.full_like(H, p["shadow_hue"]), np.full_like(H, p["shadow_sat"]), V], -1))
        cool = cool * (luma(b) / np.maximum(luma(cool), 1e-4))[..., None]
        sh = ((1 - smoothstep(0.10, 0.18, V)) * p["shadow_lean"])[..., None]   # darkest shadows only; ceiling algae (V>0.17) untouched
        out = out * (1 - sh) + cool * sh
        return np.clip(out, 0, 1)

    def pool_map(self, m, h, w):
        p = self.p
        if m.sum() < 50:
            S = np.zeros((h, w), np.float32)
        else:
            ys, xs = np.nonzero(m > 0.5)
            r = max(float(np.sqrt(len(xs) / np.pi)), 12.0)
            S = gblur(m.astype(np.float32), p["pool_sigma"] * r)
            S = np.clip(S / max(float(S.max()), 1e-3) * 1.4, 0, 1)
        if self.pool is not None:
            S = 0.55 * self.pool + 0.45 * S
        self.pool = S
        return S

    def fish(self, b):
        """Base -> fish look (before the alpha x luma-gate blend)."""
        p = self.p
        yb = luma(b)
        th = yb - cv2.morphologyEx(yb.astype(np.float32), cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        s = gblur(smoothstep(0.03, 0.10, th).astype(np.float32), 1.5)
        Ht = p["body_h"] + (p["spot_h"] - p["body_h"]) * s
        St = p["body_s"] + (p["spot_s"] - p["body_s"]) * s
        # target luma: mild gamma lift (<= +0.5 stop), +12 % local contrast, knee below the ceiling
        Y = np.power(np.clip(yb, 1e-4, 1), p["gamma"])
        lo = gblur(Y, p["lc_sigma"])
        Y = lo + (Y - lo) * p["local_contrast"]
        Y = np.clip(knee_curve(Y, p["knee"], p["ceil"]), 0, 1)
        col = rgb(np.stack([Ht, St, np.clip(Y, 0.02, 1)], -1))
        col = col * (Y / np.maximum(luma(col), 1e-4))[..., None]
        return np.clip(col, 0, 1), s

    def gate(self, yb):
        """0 at 3 %, 1 at 10 % (on the locally averaged base luma, so noise and the black gaps
        between spots do not switch it off), 1 until 85 %, 0 at 95 % (per pixel)."""
        a, b, c, d = self.p["gate"]
        return smoothstep(a, b, gblur(yb, 10.0)) * (1 - smoothstep(c, d, yb))

    def frame(self, i, x):
        p = self.p
        x = self.denoise(x)
        b = self.base(x)
        alpha = load_alpha(self.adir, i, self.w, self.h)
        bg = self.background(b)
        S = self.pool_map(alpha, self.h, self.w)
        bg = bg * (1 - p["bg_dim"] * (1 - S))[..., None]
        fish, s = self.fish(b)
        wgt = (alpha * self.gate(luma(b)))[..., None]
        # fish edit rides on the common base, not on the stylised background
        look = b + (fish - b) * wgt
        out = bg * (1 - alpha[..., None]) + look * alpha[..., None]
        if p["grain"]:
            rng = np.random.default_rng(1000 + i)
            gw = (1.0 - 0.6 * np.clip(luma(out), 0, 1))[..., None]
            out = out + rng.standard_normal(out.shape[:2] + (1,), dtype=np.float32) * p["grain"] * gw
        return np.clip(out, 0, 1), dict(base=b, alpha=alpha, spot=s)


# --------------------------------------------------------------- metrics --
def circ_mean(h, wt=None):
    r = np.deg2rad(h)
    wt = np.ones_like(h) if wt is None else wt
    return float(np.rad2deg(np.arctan2((np.sin(r) * wt).sum(), (np.cos(r) * wt).sum())) % 360)


def measure(out, aux):
    m = aux["alpha"] > 0.5
    if m.sum() < 200:
        return None
    hh = hsv(out)
    H, S, V = hh[..., 0], hh[..., 1], hh[..., 2]
    y = luma(out)
    sp = aux["spot"]
    body = m & (sp < 0.15) & (V >= 0.06)      # hue is undefined on crushed-black body pixels
    spots = m & (sp > 0.6)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bgm = ~(cv2.dilate(m.astype(np.uint8), k) > 0)
    h_, w_ = m.shape
    low = np.zeros_like(m); low[int(0.45 * h_):, :] = True
    neutral = bgm & low & (V > 0.30)
    tr = np.zeros_like(m); tr[: int(0.30 * h_), int(0.80 * w_):] = True
    top = bgm & tr & (V > 0.60)
    ceil_ = bgm & (H > 5) & (H < 45) & (S > 0.25) & (V > 0.12)
    res = dict(
        body_h=circ_mean(H[body]) if body.sum() > 50 else None, body_s=float(S[body].mean()) if body.sum() > 50 else None,
        spot_h=circ_mean(H[spots]) if spots.sum() > 50 else None, spot_s=float(S[spots].mean()) if spots.sum() > 50 else None,
        fish_luma=float(y[m].mean()), fish_p1=float(np.percentile(y[m], 1)), fish_p99=float(np.percentile(y[m], 99)),
        bg_s=float(S[bgm].mean()), bg_luma=float(y[bgm].mean()),
        bg_h=circ_mean(H[bgm], S[bgm]),
        neutral_s=float(S[neutral].mean()) if neutral.sum() > 50 else None,
        top_s=float(S[top].mean()) if top.sum() > 50 else None,
        ceil_h=circ_mean(H[ceil_]) if ceil_.sum() > 50 else None,
    )
    sh = bgm & (V < 0.10) & (S > 0.05)
    res["shadow_h"] = circ_mean(H[sh], S[sh]) if sh.sum() > 50 else None
    return res


def wb_gains(src, info, alpha_dir, p, step=15):
    """Gray-world gains from the neutral references: bright-ish surfaces outside
    the fish (pale rock, gravel, the bright slit). One clip-wide value."""
    acc = []
    for i, x in enumerate(iter_frames(src, info)):
        if i % step:
            continue
        y = luma(x)
        a = load_alpha(alpha_dir, i, info.width, info.height) > 0.2
        sel = (y > p["wb_ylo"]) & (y < p["wb_yhi"]) & ~(cv2.dilate(a.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
        mx, mn = x.max(-1), x.min(-1)
        sel &= (mx - mn) < 0.35 * np.maximum(mx, 1e-3)            # not the red-brown algae
        if sel.sum() > 500:
            acc.append(x[sel].mean(0))
    c = np.median(np.stack(acc), 0)
    g = (c.mean() / np.maximum(c, 1e-4)).astype(np.float32)
    return g / (0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--alpha", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--frames")
    ap.add_argument("--stills", type=Path)
    ap.add_argument("--params")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()
    if a.params:
        P.update(json.loads(a.params))
    P["gate"] = tuple(P["gate"])
    info = probe(a.input)
    w, h, n = info.width, info.height, info.n_frames
    gains = wb_gains(a.input, info, a.alpha, P)
    print(f"[grade2] {w}x{h} {n} frames; WB gains {gains.round(3)}", flush=True)
    g = Grader(P, a.alpha, w, h, gains)
    want = sorted(int(s) for s in a.frames.split(",")) if a.frames else []
    if want:
        a.stills.mkdir(parents=True, exist_ok=True)
    enc = Encoder(a.input, a.output, info, Encode(crf=a.crf)) if a.output else None
    if enc:
        a.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, x in enumerate(iter_frames(a.input, info)):
        if i < a.start - 8:
            continue
        if not enc and want and i > want[-1] and not a.metrics:
            break
        y, aux = g.frame(i, x)
        if a.metrics and i >= a.start and (enc or i in want or i % 10 == 0):
            r = measure(y, aux)
            if r:
                r["frame"] = i
                rows.append(r)
        if want and i in want:
            cv2.imwrite(str(a.stills / f"g2_{i:05d}.png"), (y[..., ::-1] * 255 + 0.5).astype(np.uint8))
        if enc:
            enc.write(y)
            if i % 25 == 0:
                print(f"[grade2] frame {i}/{n}", flush=True)
        if not enc and a.metrics and not want and i >= a.start + 60:
            break
    if enc:
        enc.close()
        print(f"[grade2] wrote {a.output}", flush=True)
    if rows:
        keys = [k for k in rows[0] if k != "frame"]
        print("[metrics] mean over %d frames:" % len(rows))
        for k in keys:
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                if k.endswith("_h"):
                    print(f"   {k}: {circ_mean(np.array(v, np.float32)):.1f} deg")
                else:
                    print(f"   {k}: mean {np.mean(v):.3f}  min {np.min(v):.3f}  max {np.max(v):.3f}")
        Path("output/work/metrics2.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
