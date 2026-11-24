#!/usr/bin/env python3
"""Cinematic deep-abyss grade with SAM 3.1 guided divers / bubbles / fish.

Layers (float32, BT.709 display-referred RGB 0..1), composited per frame:

  water     the whole frame, but built from a SUBJECT-FREE base (divers and fish are filled in
            from their surroundings before the tone curve, so a dark wetsuit cannot drag a dark
            halo into the water): fitted tone curve + vignette (abyss_look.Look), source-derived
            chroma pushed to midnight/petrol blue, the source's own mid-frequency detail and
            particles kept. Soft volumetric light above the divers and light on the divers'
            own bubble specks (nothing is invented: only existing bright pixels are lifted).
  fish      SAM mask -> per-fish-class tone mapping of their own luma range (silver separates
            from the dark seabed), local clarity, soft top light. Anatomy and position untouched.
  divers    SAM mask -> shadow-lifted wetsuit/gear, clarity, top light + rim light.

All strengths are plain numbers: edit abyss_look.json (global look) or pass --set key=value
(any Look or Local field). Run with --stills first, then render the full clip.

    python abyss_grade.py --input inputs/dive.mov --stills output/stills --frames 241,0,120
    python abyss_grade.py --input inputs/dive.mov --output output/dive_abyss_1080p.mp4
    python abyss_grade.py --input inputs/dive.mov --output out.mp4 --no-masks      # global grade only
"""
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from abyss_common import (LUMA, gblur, gradient_map, guided, hexrgb, lowpass, luma, open_encoder,
                          probe, read_frames)
from abyss_look import Look, colorize, nblur, src_ratios, tone, vignette

REPO = Path(__file__).resolve().parent
REF_H = 1920.0                                     # pixel scales below are defined for a 1920 px tall frame


@dataclass
class Local:
    """Strength of every mask-guided effect. 0 switches an effect off."""
    mix: float = 1.0               # final mix graded <-> original (1 = full grade; for A/B only)
    # --- divers ---------------------------------------------------------------------------
    diver_floor: float = 0.050     # output luma of the darkest wetsuit pixels (lifted, never flat black)
    diver_peak: float = 0.19       # output luma of the brightest diver body pixels (rim/top light add on top)
    diver_gamma: float = 1.1       # >1 darker mids, <1 brighter mids
    diver_clarity: float = 1.0     # local contrast on gear / straps / fins
    diver_top_light: float = 0.20  # broad light on upward-facing surfaces
    diver_rim: float = 0.18        # thin rim light on edges facing the light
    diver_chroma: float = 1.0
    diver_own_color: float = 0.2   # 0 = colour purely from the surrounding water, 1 = from the diver's own pixels
    # --- volumetric light + bubbles -------------------------------------------------------
    beam: float = 0.070            # luma added by the soft light column above/around divers
    rays: float = 1.0              # gain on the source's own vertical light shafts (upper water)
    beam_rise: float = 130.0       # px the column is shifted upward from the divers
    beam_w: float = 70.0           # px sigma (horizontal)
    beam_h: float = 230.0          # px sigma (vertical)
    bubble_light: float = 12.0     # gain on the source's own bubble specks inside the column
    bubble_cap: float = 0.30       # max luma a bubble speck may add
    # --- fish -----------------------------------------------------------------------------
    fish_floor: float = 0.19
    fish_peak: float = 0.70
    fish_gamma: float = 0.95
    fish_clarity: float = 0.9
    fish_top_light: float = 0.30
    fish_rim: float = 0.20
    fish_chroma: float = 0.65      # < 1 keeps the fish silver, not teal
    fish_own_color: float = 0.5
    # --- finishing ------------------------------------------------------------------------
    knee: float = 0.80             # soft highlight roll-off start
    dither: float = 1.0            # TPDF dither amplitude (LSB of the 8 bit output)
    light_tint: str = "#7ED3EA"    # colour of added light (beam, bubbles, rim)


def smooth_series(series, k=12):
    """Median then mean filter of an (n, m) series; NaN gaps are filled by interpolation."""
    s = np.array(series, np.float64)
    for c in range(s.shape[1]):
        v = s[:, c]
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            continue
        v[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), v[ok])
        pad = np.pad(v, k, mode="edge")
        med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(v))])
        pad = np.pad(med, k, mode="edge")
        s[:, c] = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), "valid")
    return s


class Masks:
    """Lazy reader of the propagated soft alpha masks (abyss_propagate.py)."""

    def __init__(self, root: Path | None, job: str, size: tuple[int, int]):
        self.dir = (root / f"{job}_full") if root else None
        self.size = size
        self.ok = bool(self.dir and self.dir.exists() and any(self.dir.glob("*.png")))

    def get(self, i: int) -> np.ndarray | None:
        if not self.ok:
            return None
        m = cv2.imread(str(self.dir / f"{i:05d}.png"), 0)
        if m is None:
            return None
        if (m.shape[1], m.shape[0]) != self.size:
            m = cv2.resize(m, self.size, interpolation=cv2.INTER_LINEAR)
        return m.astype(np.float32) / 255


class Grader:
    def __init__(self, look: Look, loc: Local, w: int, h: int):
        self.look, self.loc, self.w, self.h = look, loc, w, h
        self.k = h / REF_H                            # px scale relative to the 1080x1920 design size
        self.V = vignette(h, w, look)
        t = hexrgb(loc.light_tint)
        self.tint = t / float(t @ LUMA)               # luma-normalised: adding a*tint adds luma a
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ helpers
    def _excl(self, ad, af):
        e = np.zeros((self.h, self.w), np.float32)
        r = max(int(round(7 * self.k)), 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        for a in (ad, af):
            if a is not None and a.max() > 0.04:
                e = np.maximum(e, cv2.dilate((a > 0.04).astype(np.uint8), ker).astype(np.float32))
        return e

    def _column(self, ad, rise, sx, sy):
        """Soft vertical light column anchored on the divers (peak-normalised, 0 with no diver)."""
        h, w = self.h, self.w
        if ad is None or ad.max() < 0.05:
            return None
        f = 4
        s = cv2.resize(ad, (w // f, h // f), interpolation=cv2.INTER_AREA)
        M = np.float32([[1, 0, 0], [0, 1, -rise * self.k / f]])
        s = cv2.warpAffine(s, M, (w // f, h // f))
        s = cv2.GaussianBlur(s, (0, 0), sigmaX=sx * self.k / f, sigmaY=sy * self.k / f)
        s /= max(float(s.max()), 0.04)                # tiny / fading masks fade the light, never boost it
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ water layer
    def water(self, f, L, excl, ad):
        k, lk, lc = self.k, self.look, self.loc
        soft = gblur(excl, 2.0 * k) if excl.any() else excl
        lden = gblur(L, 1.2 * k)
        base = nblur(lden, excl if excl.any() else None, lk.base_sigma * k)
        lf2 = gblur(L, 2.0 * k)
        d_mid = (lf2 - base) * (1 - soft)
        d_fine = (L - lf2) * (1 - soft)
        n = base + lk.detail * d_mid + lk.particles * d_fine
        Y = tone(n, lk) * self.V + lk.floor

        if lc.rays > 0:   # the source's own light shafts: large-scale positive high-pass, stretched vertically
            hp = np.clip(nblur(lden, None, 14 * k) - nblur(lden, excl if excl.any() else None, 60 * k), 0, None)
            rays = cv2.blur(hp, (max(int(5 * k) | 1, 3), max(int(121 * k) | 1, 3)))
            wy = np.clip(1.1 - 1.6 * np.linspace(0, 1, self.h, dtype=np.float32), 0, 1)[:, None]
            Y = Y + lc.rays * 1.5 * rays * wy * self.V * (1 - soft)
        col = self._column(ad, lc.beam_rise, lc.beam_w, lc.beam_h)
        bub = np.zeros_like(Y)
        if col is not None:
            Y = Y + lc.beam * col * (0.55 + 0.45 * self.V)
            # bubbles: the source's own bright specks (positive band-pass), only inside the rising column
            hp = np.clip(gblur(L, 0.7 * k) - gblur(L, 5.0 * k), 0, None)
            gate = np.clip(self._column(ad, lc.beam_rise * 1.8, lc.beam_w * 1.3, lc.beam_h * 1.6), 0, 1)
            bub = np.minimum(lc.bubble_light * hp * gate * (1 - soft), lc.bubble_cap)
            bub = bub + 0.5 * gblur(bub, 2.2 * k)          # faint scatter around the existing specks
        r_rg, r_bg = src_ratios(f, excl if excl.any() else None, lk.chroma_sigma * k)
        rgb = colorize(Y, r_rg, r_bg, lk)
        rgb = rgb + bub[..., None] * self.tint
        return rgb, (r_rg, r_bg)

    # ------------------------------------------------------------------ subject layers
    def subject(self, f, L, alpha, lv, water_ratios, kind):
        """Grade one subject class inside its (soft) mask. Returns (rgb crop, bbox) or None."""
        lc, k = self.loc, self.k
        ys, xs = np.nonzero(alpha > 0.01)
        if len(ys) < 12:
            return None
        pad = int(72 * k)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, self.h)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, self.w)
        sl = (slice(y0, y1), slice(x0, x1))
        a, Lc, fc = alpha[sl], L[sl], f[sl]
        P = {n: getattr(lc, f"{kind}_{n}") for n in ("floor", "peak", "gamma", "clarity", "chroma", "top_light")}
        rim_gain = getattr(lc, f"{kind}_rim")
        lo, hi = lv
        span = max(hi - lo, 0.02)
        # Edge decontamination: partially covered edge pixels still carry background luma/colour
        # (bright water around a dark diver -> white outline). Replace everything outside the solid
        # core by values extrapolated from the core, then grade.
        core = cv2.erode((a > 0.85).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(int(round(2 * k)), 1))
        wcore = core.astype(np.float32)
        if wcore.sum() > 20:
            den = np.maximum(gblur(wcore, 3.0 * k), 1e-4)
            keep = (wcore > 0.5) | (gblur(wcore, 3.0 * k) < 1e-3)
            Lc = np.where(keep, Lc, gblur(Lc * wcore, 3.0 * k) / den)
            fc = np.where(keep[..., None], fc, np.stack([gblur(fc[..., c] * wcore, 3.0 * k) / den for c in range(3)], -1))
        Ls = gblur(Lc, 0.8 * k)
        dd = Ls - gblur(Lc, 3.5 * k)
        t = np.clip((Ls - lo) / span + P["clarity"] * dd / span, 0, 1) ** P["gamma"]
        Y = P["floor"] + (P["peak"] - P["floor"]) * t

        # directional light from above (slightly from the left): Lambert on the mask's own normals
        ldir = np.array([-0.25, -0.97], np.float32)
        ldir /= np.linalg.norm(ldir)

        def facing(sigma):
            sm = gblur(a, sigma * k)
            gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / 8
            gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / 8
            mag = np.sqrt(gx * gx + gy * gy) + 1e-6
            lam = np.clip((-gx * ldir[0] - gy * ldir[1]) / mag, 0, 1)     # outward normal . light dir
            return lam * np.clip(mag * 6.0 * sigma * k, 0, 1)

        Y = Y + P["top_light"] * 0.12 * facing(18.0) * a + rim_gain * 0.16 * facing(3.0) * a

        r_rg, r_bg = water_ratios
        s_rg = gblur(fc[..., 0], 2.5 * k) / np.maximum(gblur(fc[..., 1], 2.5 * k), 1e-3)
        s_bg = gblur(fc[..., 2], 2.5 * k) / np.maximum(gblur(fc[..., 1], 2.5 * k), 1e-3)
        own = getattr(lc, f"{kind}_own_color")
        s_rg = own * np.clip(s_rg, 0.25, 1.2) + (1 - own) * r_rg[sl]
        s_bg = own * np.clip(s_bg, 0.6, 1.8) + (1 - own) * r_bg[sl]
        lk = Look(**{f_.name: getattr(self.look, f_.name) for f_ in fields(Look)})
        lk.chroma_gain = self.look.chroma_gain * P["chroma"]
        rgb = colorize(Y, s_rg, s_bg, lk)
        return rgb, sl

    # ------------------------------------------------------------------ frame
    def grade(self, f, ad, af, lv_d, lv_f):
        lc = self.loc
        L = luma(f)
        excl = self._excl(ad, af)
        out, ratios = self.water(f, L, excl, ad)
        for kind, a, lv in (("fish", af, lv_f), ("diver", ad, lv_d)):
            if a is None or lv is None:
                continue
            res = self.subject(f, L, a, lv, ratios, kind)
            if res is None:
                continue
            rgb, sl = res
            al = a[sl][..., None]
            out[sl] = out[sl] * (1 - al) + rgb * al
        # soft highlight knee (surface glints roll off, nothing clips to white)
        kn = lc.knee
        out = np.where(out > kn, kn + (1 - kn) * np.tanh((out - kn) / (1 - kn)), out)
        if lc.mix < 1:
            out = f * (1 - lc.mix) + out * lc.mix
        return np.clip(out, 0, 1)

    def finish16(self, out):
        """TPDF dither at the 8 bit step, then 16 bit for the encoder (kills banding in the dark gradient)."""
        if self.loc.dither > 0:
            r = self.rng
            out = out + self.loc.dither * (r.random(out.shape, np.float32) - r.random(out.shape, np.float32)) / 255
        return (np.clip(out, 0, 1) * 65535 + 0.5).astype(np.uint16)


def class_levels(alpha, L, lo_p, hi_p):
    """Robust tone range of a subject class, from its solid core only (edge pixels still carry background)."""
    if alpha is None:
        return [np.nan, np.nan]
    core = cv2.erode((alpha > 0.9).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    v = L[core]
    if v.size < 25:
        return [np.nan, np.nan]
    return [np.percentile(v, lo_p), np.percentile(v, hi_p)]


def target_size(w: int, h: int) -> tuple[int, int]:
    """1080p framing: the short edge becomes 1080 (1080x1920 portrait, 1920x1080 landscape)."""
    s = 1080 / min(w, h)
    return int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="H.264 MP4 to render")
    ap.add_argument("--look", type=Path, default=REPO / "abyss_look.json")
    ap.add_argument("--masks", type=Path, help="mask root (default masks/<stem>__abyss)")
    ap.add_argument("--no-masks", action="store_true", help="global grade only")
    ap.add_argument("--frames", default=None, help="comma list of frame indices -> stills instead of a render")
    ap.add_argument("--stills", type=Path, default=REPO / "output" / "stills")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Look / Local field")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--limit", type=int, help="render only the first N frames (test)")
    a = ap.parse_args()

    info = probe(a.input)
    sw, sh, n = info["w"], info["h"], info["frames"]
    tw, th = target_size(sw, sh)
    look, loc = Look.load(a.look), Local()
    look_names = {f.name for f in fields(Look)}
    for item in a.set:
        key, val = item.split("=", 1)
        if key in look_names:
            look.override([item])
        else:
            cur = getattr(loc, key)
            setattr(loc, key, type(cur)(val))
    root = None if a.no_masks else (a.masks or REPO / "masks" / f"{a.input.stem}__abyss")
    M = {"diver": Masks(root, "diver", (tw, th)), "fish": Masks(root, "fish", (tw, th))}
    print(f"[grade] {sw}x{sh} -> {tw}x{th}, {n} frames @ {float(info['fps']):.3f} fps; "
          f"masks: diver={'yes' if M['diver'].ok else 'no'} fish={'yes' if M['fish'].ok else 'no'}", flush=True)
    G = Grader(look, loc, tw, th)
    scale_to = (tw, th) if (tw, th) != (sw, sh) else None

    # ---- pass 1: per-class tone levels, smoothed over time (no exposure pumping) -------------
    lev = {"diver": np.full((n, 2), np.nan), "fish": np.full((n, 2), np.nan)}
    want = {int(x) for x in a.frames.split(",")} if a.frames else None
    if any(m.ok for m in M.values()):
        t0 = time.time()
        for i, f in enumerate(read_frames(a.input, sw, sh, scale_to=scale_to)):
            if i % 1 == 0:
                L = luma(f)
                lev["diver"][i] = class_levels(M["diver"].get(i), L, 3, 90)
                lev["fish"][i] = class_levels(M["fish"].get(i), L, 5, 99)
        for k_ in lev:
            if not np.all(np.isnan(lev[k_])):
                lev[k_] = smooth_series(lev[k_])
        print(f"[grade] levels pass {time.time() - t0:.0f}s", flush=True)

    enc = None
    if want:
        a.stills.mkdir(parents=True, exist_ok=True)
    elif a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        enc = open_encoder(a.output, tw, th, info["fps"], a.input if info["audio"] else None, a.crf, a.preset)
    else:
        raise SystemExit("give --output (render) or --frames (stills)")

    t0, written = time.time(), 0
    for i, f in enumerate(read_frames(a.input, sw, sh, scale_to=scale_to)):
        if a.limit and i >= a.limit:
            break
        if want is not None and i not in want:
            continue
        ad, af = M["diver"].get(i), M["fish"].get(i)
        lv_d = None if np.isnan(lev["diver"][i, 0]) else lev["diver"][i]
        lv_f = None if np.isnan(lev["fish"][i, 0]) else lev["fish"][i]
        out = G.grade(f, ad, af, lv_d, lv_f)
        if want is not None:
            cv2.imwrite(str(a.stills / f"g{i:05d}.png"), (np.clip(out, 0, 1)[..., ::-1] * 65535 + .5).astype(np.uint16))
            print(f"[grade] still {i}", flush=True)
        else:
            enc.stdin.write(G.finish16(out).tobytes())
            written += 1
            if i % 30 == 0:
                el = time.time() - t0
                print(f"[grade] frame {i}/{n}  {el:.0f}s  eta {el / (i + 1) * (n - i - 1):.0f}s", flush=True)
    if enc:
        enc.stdin.close()
        rc = enc.wait()
        print(f"[grade] encoder exit {rc}; {written} frames in {time.time() - t0:.0f}s -> {a.output}", flush=True)
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
