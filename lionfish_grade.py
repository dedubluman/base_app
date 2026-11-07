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
def load_objects(mask_dirs, crop_ys, n_proxy, h, w):
    """-> dict name -> {proxy_idx: bool mask (h, w)} for every SAM object."""
    objs = {}
    for tag, (d, cy) in enumerate(zip(mask_dirs, crop_ys)):
        for od in sorted(Path(d).glob("obj_*")):
            m = {}
            for p in od.glob("*.png"):
                a = cv2.imread(str(p), 0) > 127
                if a.shape[0] != h:
                    full = np.zeros((h, w), bool)
                    full[cy:cy + a.shape[0], :a.shape[1]] = a
                    a = full
                m[int(p.stem)] = a
            if m:
                objs[f"{'AB'[tag]}{od.name[4:]}"] = m
    return objs


def merge_duplicates(objs, iou_t=0.3):
    """The two crops overlap, so one fish can be found twice: merge by IoU."""
    names = sorted(objs, key=lambda k: -np.median([m.sum() for m in objs[k].values()]))
    groups = []
    for n in names:
        placed = False
        for g in groups:
            sh = [f for f in objs[n] if f in objs[g[0]]]
            if not sh:
                continue
            iou = np.mean([(objs[n][f] & objs[g[0]][f]).sum() /
                           max((objs[n][f] | objs[g[0]][f]).sum(), 1) for f in sh])
            if iou > iou_t:
                g.append(n)
                placed = True
                break
        if not placed:
            groups.append([n])
    out = []
    for g in groups:
        fr = set().union(*[set(objs[n]) for n in g])
        out.append({f: np.any([objs[n][f] for n in g if f in objs[n]], 0) for f in fr})
    return out


def interp_masks(fish_by_proxy, f, stride, gray_f, gray_proxy, h, w):
    """Mask at source frame f from the two neighbouring proxy frames, each
    warped into f with Farneback flow (no ghosting from fish motion)."""
    k0 = f // stride
    k1 = k0 + 1
    t = (f - k0 * stride) / stride
    acc, wsum = np.zeros((h, w), np.float32), 0.0
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for k, wt in ((k0, 1 - t), (k1, t)):
        if wt <= 0 or k not in gray_proxy or k not in fish_by_proxy:
            continue
        if k * stride == f:
            m = fish_by_proxy[k].astype(np.float32)
        else:
            fl = cv2.calcOpticalFlowFarneback(gray_f, gray_proxy[k], None, 0.5, 3, 25, 3, 5, 1.1, 0)
            m = cv2.remap(fish_by_proxy[k].astype(np.float32), gx + fl[..., 0], gy + fl[..., 1],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        acc += wt * m
        wsum += wt
    return acc / wsum if wsum else acc


# ---------------------------------------------------------------- grade ----
LION_RAMP = np.array([  # luminance of the dehazed lionfish -> target colour (sRGB)
    [0.00, 0.045, 0.020, 0.015],   # near-black brown (dark stripes / spots)
    [0.20, 0.30, 0.075, 0.035],    # deep brick red
    [0.40, 0.66, 0.205, 0.075],    # burnt orange-red (main bands)
    [0.58, 0.86, 0.40, 0.17],      # warm orange / copper
    [0.78, 0.94, 0.74, 0.55],      # cream (fine white lines)
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


def grade_fg(rgb, air, veil):
    # dehaze: invert I = J(1-v) + A v with a constant v
    j = (rgb - veil * air) / (1 - veil)
    j = np.clip(j, 0, 1.5)
    # restore red lost to water: gentle channel gain
    j = j * np.array([1.55, 1.0, 1.0], np.float32)
    l = luma(j)
    # mid-contrast + clarity
    j = np.clip(j, 0, 1)
    j = j ** 0.82
    blur = cv2.GaussianBlur(j, (0, 0), 9)
    j = np.clip(j + 0.55 * (j - blur), 0, 1)
    ll = luma(j)[..., None]
    j = np.clip(ll + (j - ll) * 1.25, 0, 1)  # natural vibrance
    return j


def grade_lion(rgb, fg_graded, lion_a):
    """Gradient-map the lionfish luminance to the reference palette, keep a
    little of the original chroma/texture."""
    l = luma(fg_graded)
    lo, hi = np.percentile(l[lion_a > 0.5], [2, 98]) if (lion_a > 0.5).sum() > 500 else (0.0, 0.6)
    ln = np.clip((l - lo) / max(hi - lo, 1e-3), 0, 1)
    target = gradient_map(ln)
    return np.clip(0.80 * target + 0.20 * fg_graded, 0, 1)


def grade_bg(rgb, bg_blur_src, vig, yy):
    x = bg_blur_src
    l = luma(x)[..., None]
    x = l + (x - l) * 0.40                     # desaturate
    x = 0.5 + (x - 0.5) * 0.72                 # low contrast
    x = x ** 1.35 * 0.62                       # darker, lifted blacks come back below
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


def process(rgb, alpha_all, alpha_lion, air, veil, vig, rng_noise, grain=0.012):
    h, w = alpha_all.shape
    a = feather_alpha(alpha_all, 1.2)
    al = feather_alpha(alpha_lion, 1.2)
    # fish-excluded defocus of the original so no fish colour bleeds
    wgt = 1 - cv2.dilate(alpha_all, np.ones((9, 9), np.uint8))
    wgt = wgt[..., None].astype(np.float32)
    num = cv2.GaussianBlur(rgb * wgt, (0, 0), 7)
    den = cv2.GaussianBlur(wgt[..., 0], (0, 0), 7)[..., None]
    bgsrc = num / np.maximum(den, 1e-3)
    bgsrc = np.where(den > 0.05, bgsrc, cv2.GaussianBlur(rgb, (0, 0), 14))
    bg = grade_bg(rgb, bgsrc, vig, None)
    fg = grade_fg(rgb, air, veil)
    lion = grade_lion(rgb, fg, al)
    fg = fg * (1 - al[..., None]) + lion * al[..., None]
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
    ap.add_argument("--stills", default="")
    ap.add_argument("--stills-dir", type=Path, default=None)
    ap.add_argument("--veil", type=float, default=0.55)
    ap.add_argument("--min-frames", type=int, default=6)
    a = ap.parse_args()

    info = probe(a.input)
    h, w = info.height, info.width
    n_proxy = (info.n_frames + a.stride - 1) // a.stride
    objs = load_objects(a.masks, a.crop_ys, n_proxy, h, w)
    fish = merge_duplicates(objs)
    fish = [f for f in fish if len(f) >= a.min_frames]
    areas = [np.median([m.sum() for m in f.values()]) for f in fish]
    lion_i = int(np.argmax(areas))
    print(f"[grade] {len(fish)} fish objects, lionfish idx {lion_i} (median area {areas[lion_i]:.0f})", flush=True)
    all_by = {k: np.any([f[k] for f in fish if k in f], 0) for k in set().union(*[set(f) for f in fish])}
    lion_by = fish[lion_i]

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
    # first pass: grays of proxy frames
    for i, f in enumerate(iter_frames(a.input, info)):
        if i % a.stride == 0:
            gray_proxy[i // a.stride] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    for i, f in enumerate(iter_frames(a.input, info)):
        if not enc and i not in still_idx:
            continue
        g = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        am = interp_masks(all_by, i, a.stride, g, gray_proxy, h, w)
        al = interp_masks(lion_by, i, a.stride, g, gray_proxy, h, w)
        am = (am > 0.5).astype(np.float32)
        al = (al > 0.5).astype(np.float32) * am
        noise = rng.standard_normal((h // 2, w // 2)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        out = process(f, am, al, air, a.veil, vig, noise)
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
