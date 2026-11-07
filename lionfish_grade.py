#!/usr/bin/env python3
"""Lionfish subject isolation + cinematic grade, masks from SAM 3.1 (CPU).

Foreground = every fish (SAM 3.1 text prompt "fish", two overlapping square
crops so the small fish are not squashed). The lionfish (largest object) is
recoloured towards the reference photo (brick-orange bands, cream lines,
dark spotted fins); the other fish only get haze removal, contrast and
natural colour. Background = desaturated, low contrast, dark teal, vignette,
defocus (fish-excluded blur, no colour bleed) and grain.

    python lionfish_grade.py --input clip.mov --masks A_DIR B_DIR --stride 3 \
        --output out.mp4 [--stills 0,85,170,255,339]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

W3 = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma(x):
    return x @ W3


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- masks ----
def index_objects(mask_dirs):
    """-> {name: {proxy_idx: Path}} for every SAM object (nothing loaded)."""
    objs = {}
    for tag, d in enumerate(mask_dirs):
        for od in sorted(Path(d).glob("obj_*")):
            m = {int(p.stem): p for p in od.glob("*.png")}
            if m:
                objs[f"{'AB'[tag]}{od.name[4:]}"] = (tag, m)
    return objs


def load_mask(path, cy, h, w):
    a = cv2.imread(str(path), 0) > 127
    if a.shape[0] != h:
        full = np.zeros((h, w), bool)
        full[cy:cy + a.shape[0], :a.shape[1]] = a
        a = full
    return a


def build_union_masks(objs, crop_ys, h, w, out_dir, min_frames=6, iou_t=0.3):
    """The two crops overlap, so one fish can be found twice: group objects
    by IoU (sampled frames, quarter res), then write per-proxy-frame
    union ('all') and lionfish ('lion') masks to disk. Returns n proxy frames."""
    q = 4
    sm = {n: {f: cv2.resize(load_mask(p, crop_ys[tag], h, w).astype(np.uint8), (w // q, h // q),
                            interpolation=cv2.INTER_AREA) > 0
              for f, p in list(fr.items())[::8]}
          for n, (tag, fr) in objs.items() if len(fr) >= min_frames}
    area = {n: np.median([m.sum() for m in v.values()]) for n, v in sm.items()}
    names = sorted(sm, key=lambda k: -area[k])
    groups = []
    for n in names:
        for g in groups:
            sh = [f for f in sm[n] if f in sm[g[0]]]
            if sh and np.mean([(sm[n][f] & sm[g[0]][f]).sum() /
                               max((sm[n][f] | sm[g[0]][f]).sum(), 1) for f in sh]) > iou_t:
                g.append(n)
                break
        else:
            groups.append([n])
    lion = groups[0]
    print(f"[grade] {len(groups)} fish groups; lionfish = {lion} (area {area[lion[0]] * q * q:.0f})", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted({f for n in names for f in objs[n][1]})
    for f in frames:
        allm = np.zeros((h, w), bool)
        lm = np.zeros((h, w), bool)
        for g in groups:
            for n in g:
                tag, fr = objs[n]
                if f in fr:
                    m = load_mask(fr[f], crop_ys[tag], h, w)
                    allm |= m
                    if g is lion:
                        lm |= m
        cv2.imwrite(str(out_dir / f"all_{f:05d}.png"), allm.astype(np.uint8) * 255)
        cv2.imwrite(str(out_dir / f"lion_{f:05d}.png"), lm.astype(np.uint8) * 255)
    return len(frames)


class MaskStore:
    def __init__(self, d):
        self.d = d
        self.cache = {}

    def get(self, kind, k):
        key = (kind, k)
        if key not in self.cache:
            p = self.d / f"{kind}_{k:05d}.png"
            self.cache[key] = (cv2.imread(str(p), 0) > 127) if p.exists() else None
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def interp_masks(store, kind, f, stride, gray_f, gray_proxy, h, w):
    """Mask at source frame f from the two neighbouring proxy frames, each
    warped into f with Farneback flow (no ghosting from fish motion)."""
    k0 = f // stride
    k1 = k0 + 1
    t = (f - k0 * stride) / stride
    acc, wsum = np.zeros((h, w), np.float32), 0.0
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for k, wt in ((k0, 1 - t), (k1, t)):
        mk = store.get(kind, k) if wt > 0 else None
        if mk is None or k not in gray_proxy:
            continue
        if k * stride == f:
            m = mk.astype(np.float32)
        else:
            fl = cv2.calcOpticalFlowFarneback(gray_f, gray_proxy[k], None, 0.5, 3, 25, 3, 5, 1.1, 0)
            m = cv2.remap(mk.astype(np.float32), gx + fl[..., 0], gy + fl[..., 1],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        acc += wt * m
        wsum += wt
    return acc / wsum if wsum else acc


# ---------------------------------------------------------------- grade ----
LION_RAMP = np.array([  # normalised lionfish luminance -> target colour (sRGB)
    [0.00, 0.060, 0.025, 0.020],   # dark brown-black (stripes / spots)
    [0.18, 0.40, 0.09, 0.04],      # deep brick red
    [0.38, 0.78, 0.22, 0.08],      # orange-red (main bands)
    [0.58, 0.93, 0.42, 0.16],      # bright orange / copper
    [0.80, 0.97, 0.74, 0.52],      # cream (fine white lines)
    [1.00, 1.00, 0.95, 0.86],
], np.float32)


def gradient_map(l):
    out = np.empty(l.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(l, LION_RAMP[:, 0], LION_RAMP[:, 1 + c])
    return out


def s_curve(x, k):
    return 0.5 + (x - 0.5) * k if k == 1 else x + k * x * (1 - x) * (x - 0.5) * 4


def estimate_veil(frames):
    """Constant airlight colour + veil strength from sample frames (stable
    in time by construction, so no flicker)."""
    air = np.median(np.stack([np.percentile(f.reshape(-1, 3), 97, axis=0) for f in frames]), 0)
    return air.astype(np.float32)


def grade_fg(rgb, air, veil, fish_a, hi_const):
    """Dehaze + exposure normalised on the fish pixels + natural colour."""
    j = (rgb - veil * air) / (1 - veil)
    j = np.clip(j, 0, 1.5) * np.array([1.45, 1.0, 1.0], np.float32)
    j = j * (0.80 / max(hi_const, 0.05))        # one clip-wide exposure (no flicker)
    j = np.clip(j, 0, 1)
    j = j ** 0.75                                # lift shadows (no crushed silhouettes)
    blur = cv2.GaussianBlur(j, (0, 0), 6)
    j = np.clip(j + 0.30 * (j - blur), 0, 1)     # clarity, moderate
    ll = luma(j)[..., None]
    j = np.clip(ll + (j - ll) * 1.45, 0, 1)      # vibrance
    return j


def natural_fish(j):
    """Small fish: keep their own luminance structure but pull the colour
    to a natural silver-olive instead of the neon cyan the dehaze leaves."""
    ll = luma(j)[..., None]
    tint = np.array([1.00, 0.95, 0.84], np.float32)
    return np.clip(ll * tint * 1.05 + (j - ll) * 0.35, 0, 1)


def grade_lion(rgb, fg_graded, lion_a):
    """Gradient-map the lionfish luminance to the reference palette, keep a
    little of the original chroma/texture."""
    l = cv2.GaussianBlur(luma(fg_graded), (0, 0), 1.0)
    lo, hi = np.percentile(l[lion_a > 0.5], [2, 98]) if (lion_a > 0.5).sum() > 500 else (0.0, 0.6)
    ln = np.clip((l - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.62
    target = gradient_map(ln)
    return np.clip(0.85 * target + 0.15 * fg_graded, 0, 1)


def grade_bg(rgb, bg_blur_src, vig, yy):
    x = bg_blur_src
    l = luma(x)[..., None]
    x = l + (x - l) * 0.40                     # desaturate
    x = 0.5 + (x - 0.5) * 0.78                 # low contrast
    x = x ** 1.15 * 0.78                       # darker but readable
    x = x * np.array([0.62, 0.94, 1.02], np.float32)  # cold teal-blue shift
    x = x + np.array([0.004, 0.018, 0.026], np.float32)  # blue-black floor
    x = x * vig[..., None]
    return np.clip(x, 0, 1)


def make_vignette(h, w, strength=0.78):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w * 0.62)) ** 2 + ((yy - h * 0.55) / (h * 0.62)) ** 2)
    v = 1 - strength * smoothstep(0.28, 1.05, d)
    return v.astype(np.float32)


def feather_alpha(m, px):
    a = cv2.GaussianBlur(m, (0, 0), px)
    return np.clip((a - 0.5) * 1.6 + 0.5, 0, 1) if px > 0 else m


def process(rgb, alpha_all, alpha_lion, air, veil, vig, rng_noise, hi_const, grain=0.012):
    h, w = alpha_all.shape
    a = feather_alpha(alpha_all, 1.2)
    al = feather_alpha(alpha_lion, 1.2)
    # fish-excluded defocus of the original so no fish colour bleeds
    wgt = 1 - cv2.dilate(alpha_all, np.ones((9, 9), np.uint8))
    wgt = wgt[..., None].astype(np.float32)
    num = cv2.GaussianBlur(rgb * wgt, (0, 0), 3.5)
    den = cv2.GaussianBlur(wgt[..., 0], (0, 0), 3.5)[..., None]
    bgsrc = num / np.maximum(den, 1e-3)
    bgsrc = np.where(den > 0.05, bgsrc, cv2.GaussianBlur(rgb, (0, 0), 14))
    bg = grade_bg(rgb, bgsrc, vig, None)
    fg = grade_fg(rgb, air, veil, a, hi_const)
    fg_other = natural_fish(fg)
    # each small fish normalised to a consistent body tone (their raw tone only
    # reflects the water behind them), no matter how haze-bright the backdrop
    oth = ((a > 0.5) & (al < 0.5)).astype(np.uint8)
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(oth, connectivity=8)
    for c in range(1, n_lab):
        x0, y0, bw, bh, ar = st[c]
        if ar < 40:
            continue
        sl = (slice(max(y0 - 3, 0), y0 + bh + 3), slice(max(x0 - 3, 0), x0 + bw + 3))
        mc = lab[sl] == c
        lc = luma(fg_other[sl])
        med = float(np.median(lc[mc]))
        gain = float(np.clip(0.50 / max(med, 0.03), 0.35, 7.0))
        fg_other[sl] = np.clip(fg_other[sl] * (1 + (gain - 1) * 1.0), 0, 1)
    lion = grade_lion(rgb, fg, al)
    fg = fg_other * (1 - al[..., None]) + lion * al[..., None]
    out = bg * (1 - a[..., None]) + fg * a[..., None]
    # grain on the background mostly
    g = rng_noise * grain
    out = out + g[..., None] * (1 - 0.6 * a[..., None])
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", nargs="+", type=Path, required=True)
    ap.add_argument("--crop-ys", nargs="+", type=int, required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--stills", default="")
    ap.add_argument("--stills-dir", type=Path, default=None)
    ap.add_argument("--veil", type=float, default=0.45)
    ap.add_argument("--min-frames", type=int, default=6)
    a = ap.parse_args()

    info = probe(a.input)
    h, w = info.height, info.width
    objs = index_objects(a.masks)
    mdir = a.workdir / "union_masks"
    if not (mdir / "done").exists():
        build_union_masks(objs, a.crop_ys, h, w, mdir, a.min_frames)
        (mdir / "done").write_text("1")
    store = MaskStore(mdir)

    # sample frames for airlight
    samples = list(iter_frames(a.input, info, scale_w=270))
    air = estimate_veil(samples[::40])
    print(f"[grade] airlight {air}", flush=True)
    vig = make_vignette(h, w)
    rng = np.random.default_rng(7)

    # proxy grays at full res (only for flow); read all frames once
    still_idx = {int(s) for s in a.stills.split(",") if s}
    enc = Encoder(a.input, a.output, info, Encode()) if a.output else None
    frames_cache = {}
    gray_proxy = {}
    his = []
    # first pass: grays of proxy frames
    for i, f in enumerate(iter_frames(a.input, info)):
        if i % a.stride == 0:
            gray_proxy[i // a.stride] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if i % (a.stride * 8) == 0:
                mk = store.get('all', i // a.stride)
                if mk is not None and mk.sum() > 300:
                    jj = np.clip((f - a.veil * air) / (1 - a.veil), 0, 1.5) * np.array([1.45, 1, 1], np.float32)
                    his.append(np.percentile(luma(jj)[mk], 97))
    hi_const = float(np.median(his))
    print(f'[grade] fish exposure ref {hi_const:.3f}', flush=True)
    for i, f in enumerate(iter_frames(a.input, info)):
        if not enc and i not in still_idx:
            continue
        g = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        am = interp_masks(store, 'all', i, a.stride, g, gray_proxy, h, w)
        al = interp_masks(store, 'lion', i, a.stride, g, gray_proxy, h, w)
        am = (am > 0.5).astype(np.float32)
        al = (al > 0.5).astype(np.float32) * am
        noise = rng.standard_normal((h // 2, w // 2)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        out = process(f, am, al, air, a.veil, vig, noise, hi_const)
        if enc:
            enc.write(out)
        if i in still_idx:
            d = a.stills_dir or a.output.parent
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"still_{i:05d}.jpg"),
                        cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(d / f"mask_{i:05d}.png"), (am * 255).astype(np.uint8))
            cv2.imwrite(str(d / f"src_{i:05d}.jpg"),
                        cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if i % 20 == 0:
            print(f"[grade] frame {i}/{info.n_frames}", flush=True)
    if enc:
        enc.close()


if __name__ == "__main__":
    main()
