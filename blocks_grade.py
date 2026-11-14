#!/usr/bin/env python3
"""Reel-matched grade for inputs/blocks_dive.mov (preset "blocks_dive").

This is a preset for the second clip; reel_match.py keeps the first clip's
tuned values and only its helpers are imported here.

Why different numbers: this clip is already blue (H196-203) but flat and
hazy (V 0.59-0.83 in every frame), so the reference's hue move would turn it
violet. The work is (0) a fixed de-haze, (1) the steep tone curve with the
top-heavy vertical weighting, (3) less saturation, lit sand/concrete toward
blue-tinted white and shaded sides toward blue-grey. The diver (Step 2) is
rebuilt from his own luminance: the red channel of the raw footage is nearly
empty, so white-balance gains would only amplify noise.

All parameters are fixed for the whole clip. Sub-commands:

    python blocks_grade.py stills                  # background alone, 8 stills
    python blocks_grade.py preview --frames 210:300
    python blocks_grade.py render                  # full clip -> output/
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np

from reel_match import hsv2rgb, luma, rgb2hsv, smoothstep

REPO = Path(__file__).resolve().parent
WORK = REPO / "work"
VIDEO = REPO / "inputs" / "blocks_dive.mov"
STEM = VIDEO.stem

# ------------------------------------------------------------ background --
BG = dict(
    # Step 0  de-haze: fixed black / white point on V (max channel) for the whole
    # clip. Per-frame V p1/p50/p99 is 0.59/0.74/0.83; the cube face reaches 0.86.
    bp=0.550, wp=0.925, toe=0.05,
    # Step 1  Darken: tone curve + vertical weighting (screen space, fixed)
    gamma_top=1.60, gamma_bot=1.45,
    g_top=0.88, g_bot=1.00, g_y0=0.10, g_y1=0.88,
    shoulder=0.78, peak=0.90,          # highlights roll off above `shoulder`, never past `peak`
    floor=0.050,                       # luminance floor: foam never clips to flat black
    # Step 3  blue water, less saturation, hue by role
    sat_water=0.64,                    # S x this: 0.86 -> 0.55
    hue_top=214.0, hue_mid=205.0, hue_lit=191.0, hue_shade=194.0,
    hue_keep=0.55,                     # share of the raw hue variation that survives
    hue_dark_lo=0.10, hue_dark_hi=0.40,
    # lit sand / concrete -> blue-tinted near white; shaded sides -> blue-grey
    lit_lo=0.64, lit_hi=0.80, lit_hp_lo=0.04, lit_hp_hi=0.13, lit_sat=0.30,
    shade_sat=0.70, shade_lo=0.020, shade_hi=0.060,
    # local contrast (on the de-hazed value): structures and sand texture
    clar_large=1.10, clar_large_sigma=28.0, clar_small=0.65, clar_small_sigma=3.0,
    clar_neg=0.50,                     # shaded sides get half the boost: blue-grey, not black
    clar_y0=0.15, clar_y1=0.60,        # the haze is in the lower half: more clarity there
    # banding: reconstruct the large-scale gradient, core away <= 1 LSB wiggle
    deband_sigma=9.0, deband_core=0.0090,
    grain=0.0016,                      # fine luma grain, temporally independent
)


def _blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)


def _small_blur(a, s, scale=4):
    """Large-scale blur at 1/scale resolution (context terms only)."""
    h, w = a.shape[:2]
    sm = cv2.resize(a, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), s / scale)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def plate_fill(a, hole):
    """Replace the masked pixels by what the surrounding water looks like
    (multi-scale normalised convolution), so the diver never leaks into the
    context blurs that drive local contrast."""
    if hole is None or not hole.any():
        return a
    out, left, w = a.copy(), hole.copy(), (~hole).astype(np.float32)
    for s in (4, 8, 16, 32, 64, 128):
        num, den = _small_blur(a * w, s), _small_blur(w, s)
        ok = left & (den > 0.12)
        out[ok] = num[ok] / den[ok]
        left &= ~ok
        if not left.any():
            break
    if left.any():
        out[left] = float(np.median(a[~hole]))
    return out


def vertical(h, lo, hi):
    return smoothstep(lo, hi, np.linspace(0, 1, h, dtype=np.float32))[:, None]


def add_grain(out, amount, rng=None):
    """Very fine luma grain, new every frame (breaks up what banding is left)."""
    if amount <= 0:
        return out
    rng = rng or np.random.default_rng(7)
    h, w = out.shape[:2]
    gr = _blur(rng.standard_normal((h, w)).astype(np.float32), 0.6) * amount * 1.8
    return out + gr[..., None]


def grade_background(x, p=BG, hole=None, rng=None, grain=True):
    """x: float RGB 0..1 raw frame. hole: optional bool mask of the diver
    (dilated); only the context blurs ignore it, the output is defined everywhere."""
    h, w = x.shape[:2]
    hue, s, v = rgb2hsv(np.clip(x, 0, 1))

    # ---- de-band: smooth large-scale gradient + cored detail
    g = _blur(v, p["deband_sigma"])
    d = v - g
    thr = p["deband_core"]
    v = g + np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)

    # ---- Step 0: fixed de-haze with a soft toe
    t = (v - p["bp"]) / (p["wp"] - p["bp"])
    toe = p["toe"]
    t = toe * np.log1p(np.exp(np.clip(t / toe, -30, 30)))            # softplus: soft knee at the black point
    t = np.clip(t, 0, 1.2)
    vp = plate_fill(t, hole)

    # ---- local contrast on the de-hazed value (more in the hazy lower half)
    ramp = vertical(h, p["clar_y0"], p["clar_y1"])
    d_large = t - _small_blur(vp, p["clar_large_sigma"])
    d_small = t - _blur(t, p["clar_small_sigma"])
    gain_l = p["clar_large"] * (0.35 + 0.65 * ramp)
    neg = np.where(d_large < 0, p["clar_neg"], 1.0)
    t2 = np.clip(t + gain_l * neg * d_large + p["clar_small"] * d_small * (0.5 + 0.5 * ramp), 0, 1.15)

    # ---- masks: lit sand/concrete, shaded structure
    # lit = bright AND brighter than its surroundings: broad bright open water
    # (the bottom of the frame in the first seconds) stays water
    lit = smoothstep(p["lit_lo"], p["lit_hi"], t2) * smoothstep(p["lit_hp_lo"], p["lit_hp_hi"], d_large)
    struct = smoothstep(p["shade_lo"], p["shade_hi"], np.abs(d_large) / 3.4)   # raw-V units
    shade = struct * (1 - lit)

    # ---- Step 1: darken (curve + vertical weighting)
    ramp_g = vertical(h, p["g_y0"], p["g_y1"])
    gam = p["gamma_top"] + (p["gamma_bot"] - p["gamma_top"]) * ramp_g
    wt = p["g_top"] + (p["g_bot"] - p["g_top"]) * ramp_g
    vo = np.clip(t2, 0, 1) ** gam * wt
    sh = p["shoulder"]
    pk = p["peak"]
    vo = np.where(vo > sh, sh + (pk - sh) * (1 - np.exp(-(vo - sh) / (pk - sh))), vo)   # soft shoulder
    vo = np.maximum(vo, p["floor"])

    # ---- Step 3: saturation and hue by role
    so = np.clip(s * p["sat_water"], 0, 1)
    so = so * (1 - (1 - p["shade_sat"]) * shade)
    so = so + (p["lit_sat"] - so) * lit
    dark = 1 - smoothstep(p["hue_dark_lo"], p["hue_dark_hi"], vo)
    hue_base = p["hue_mid"] + (p["hue_top"] - p["hue_mid"]) * dark
    ho = hue_base + (hue - 197.0) * p["hue_keep"]
    ho = ho + (p["hue_shade"] - ho) * shade
    ho = ho + (p["hue_lit"] - ho) * lit

    out = hsv2rgb(ho, np.clip(so, 0, 1), np.clip(vo, 0, 1)).astype(np.float32)
    if grain:
        out = add_grain(out, p["grain"], rng)
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------- diver --
# Step 2  Restore skin tones. Two classes inside the SAM 3.1 mask:
#   SKIN       flesh tone H21 S0.29 built from his own luminance (the raw red
#              channel is nearly empty, so no white-balance gains)
#   DARK GEAR  shorts, fins, hair, mask, strap: neutral near-black with a faint
#              blue reflection, own luminance (pattern contrast) kept
# There is no off-white shorts class and no "cool and bright" key in this clip.
DV = dict(
    skin_rgb=(1.186, 0.961, 0.839),    # luma-normalised flesh ratio: H21, S0.293
    # raw skin luma (haze keeps it at 0.5-0.85) -> flesh luma of the reference's
    # lit 0.77 / shaded 0.57:  Y' = skin_off + skin_gain * Y
    skin_off=0.114, skin_gain=0.82,
    far_ease=0.20,                     # share of brightness he loses at his smallest
    far_veil=0.05,                     # share of the water's colour mixed in at his smallest
    ambient=0.03,                      # water light in his shadows
    # dark gear: luminance relative to the diver's lit-skin level (90th percentile
    # of his own luma; the median is dominated by the gear itself), on a lightly
    # smoothed luma so the light patches of the shorts' pattern stay gear
    dark_rel=(0.50, 0.66), key_sigma=2.0,
    # raw gear luma 0.25-0.35 (hazy black) -> near black, pattern kept:
    # Y' = max(0, gear_gain * (Y - gear_black)) + gear_floor
    gear_rgb=(0.80, 1.00, 1.14),       # neutral, faintly blue
    gear_black=0.20, gear_gain=0.95, gear_floor=0.030,
    size_near=360.0, size_far=215.0,   # bbox diagonal (px) -> distance 0..1
    rim_px=3,                          # edge pixels take their colour from the inside
)


def split_classes(x, y_hi, mask=None, p=DV):
    """Weights (skin, gear), summing to 1, from luminance alone."""
    y = luma(np.clip(x, 0, 1))
    ys = y
    if mask is not None and mask.any():
        m = mask.astype(np.float32)
        num = cv2.GaussianBlur(y * m, (0, 0), p["key_sigma"])
        den = cv2.GaussianBlur(m, (0, 0), p["key_sigma"])
        ys = np.where(den > 0.05, num / np.maximum(den, 1e-3), y)
    rel = ys / max(y_hi, 1e-3)
    k_dark = 1 - smoothstep(*p["dark_rel"], rel)
    return 1 - k_dark, k_dark, y


def grade_diver(x, y_hi, dist, plate_rgb, mask=None, p=DV):
    """Graded colours for every pixel (use through the diver alpha).
    y_hi: his lit-skin luma level; dist: 0 (near, large) .. 1 (far, small);
    plate_rgb: the graded water behind him."""
    w_skin, w_gear, y = split_classes(x, y_hi, mask, p)
    ease = 1 - p["far_ease"] * dist
    ys = np.clip(p["skin_off"] + p["skin_gain"] * y, 0, 1) * ease
    skin = ys[..., None] * np.array(p["skin_rgb"], np.float32)
    # a little water light in his shadows, and a veil when he is far away
    amb = p["ambient"] * (1 - smoothstep(0.55, 1.0, y / max(y_hi, 1e-3)))[..., None]
    plate = np.asarray(plate_rgb, np.float32)
    plate_dir = plate / max(float(luma(plate)), 1e-3)
    veil = p["far_veil"] * dist
    skin = skin * (1 - amb) + luma(skin)[..., None] * plate_dir * amb
    skin = skin * (1 - veil) + plate * veil
    yg = (np.maximum(0, p["gear_gain"] * (y - p["gear_black"])) + p["gear_floor"]) * (1 - 0.5 * p["far_ease"] * dist)
    gear = np.clip(yg, 0, 1)[..., None] * np.array(p["gear_rgb"], np.float32)
    out = skin * w_skin[..., None] + gear * w_gear[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def fill_rim(img, alpha_hard, px):
    """Edge pixels still carry some water colour from the original frame:
    replace the outer `px` pixels by the weighted colour of the pixels just
    inside (normalised convolution from the interior)."""
    inner = cv2.erode(alpha_hard.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0
    if not inner.any():
        return img
    w = inner.astype(np.float32)
    sig = max(1.0, px * 0.7)
    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(w, (0, 0), sig)
    rim = (~inner) & (den > 0.02)
    out = img.copy()
    out[rim] = num[rim] / den[rim][:, None]
    return out


# ------------------------------------------------------------------ masks --
def read_png(d, i):
    p = d / f"{i:05d}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    return None if m is None else m > 127


def drop_specks(m, keep=0.05):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    return np.isin(lab, 1 + np.nonzero(areas >= keep * areas.max())[0])


def merged_masks(idx, shape):
    """Per frame: the tracker mask of the nearest seed (primary) with the extra
    fin objects unioned in; the alternate candidate of the overlapping chunk
    fills a frame whose primary collapsed; remaining dropouts come from the
    neighbouring frames. Returns ({frame: bool mask}, report)."""
    d = REPO / "masks" / f"{STEM}__diver" / "subject"
    alt = REPO / "masks" / f"{STEM}__diver_alt" / "subject"
    fd = REPO / "masks" / f"{STEM}__fins" / "subject"
    lo, hi = min(idx) - 4, max(idx) + 4
    raw, fins, alts = {}, {}, {}
    for i in range(lo, hi + 1):
        raw[i], fins[i], alts[i] = read_png(d, i), read_png(fd, i), read_png(alt, i)
    area = {i: (int(drop_specks(m).sum()) if m is not None else 0) for i, m in raw.items()}
    out, rep = {}, {}
    for i in idx:
        m = raw[i]
        m = drop_specks(m) if m is not None else np.zeros(shape, bool)
        nb = [area[j] for j in range(i - 3, i + 4) if j != i and area.get(j, 0) > 0]
        med = float(np.median(nb)) if nb else 0.0
        mode = "tracker"
        if med and m.sum() < 0.6 * med:
            a = alts[i]
            if a is not None and drop_specks(a).sum() > m.sum():
                m, mode = drop_specks(a), "alt candidate"
            else:
                pv = next((raw[j] for j in range(i - 1, lo - 1, -1) if area.get(j, 0) >= 0.6 * med and raw[j] is not None), None)
                nx = next((raw[j] for j in range(i + 1, hi + 1) if area.get(j, 0) >= 0.6 * med and raw[j] is not None), None)
                if pv is not None and nx is not None:
                    m, mode = drop_specks(pv & nx), "neighbour fill"
                elif pv is not None or nx is not None:
                    m, mode = drop_specks(pv if pv is not None else nx), "neighbour fill"
        f = fins[i]
        if f is not None and f.any():
            near = cv2.dilate(m.astype(np.uint8), np.ones((201, 201), np.uint8)) > 0
            if f.sum() <= 0.35 * max(m.sum(), 1) and (f & near).sum() >= 0.8 * f.sum():
                m = m | f
                mode += "+fins"
        out[i], rep[i] = m, mode
    return out, rep


def build_alpha(raw_mask, x):
    """matte.py edge snap, then erode 1 px; uint8 0..255."""
    import matte as M
    al = M.snap_diver_alpha(raw_mask, x)
    al = cv2.erode(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return np.clip(al * 255 + 0.5, 0, 255).astype(np.uint8)


def median3(a, b, c):
    return np.median(np.stack([a, b, c]), 0).astype(np.uint8)


def diver_size(alpha_u8):
    ys, xs = np.nonzero(alpha_u8 > 127)
    if len(ys) < 50:
        return np.nan
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def feather_sigma(size):
    return float(np.interp(size, [150.0, 420.0], [0.8, 2.0]))


# --------------------------------------------------------------- pipeline --
def grade_frame(task):
    """One frame: background layer, diver layer, composite. Returns float RGB."""
    i, x, al, y_ref, dist, sig = task
    hard = al > 127
    hole = cv2.dilate((al > 10).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    bg = grade_background(x, hole=hole if hole.any() else None, grain=False)
    if hard.any():
        ring = (cv2.dilate(hard.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & \
               ~(cv2.dilate(hard.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
        plate = np.median(bg[ring], 0) if ring.sum() > 50 else np.median(bg.reshape(-1, 3), 0)
        dv = fill_rim(grade_diver(x, y_ref, dist, plate, mask=hard), hard, DV["rim_px"])
        a = cv2.GaussianBlur(al.astype(np.float32) / 255, (0, 0), sig)[..., None]
        out = bg * (1 - a) + dv * a
    else:
        out = bg
    out = add_grain(out, BG["grain"], np.random.default_rng(1000 + i))
    return i, np.clip(out, 0, 1).astype(np.float32)


def smooth1d(y, sigma, r):
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(y, r, mode="edge"), k / k.sum(), "valid")


def prepare(f0, f1, log=print):
    """Pass 1: merged SAM 3.1 masks -> snapped alphas, diver luma, size.
    Frames f0-1 .. f1 are read so the 3-frame median has both neighbours."""
    from shades.io import iter_frames, probe
    info = probe(VIDEO)
    n, w, h = info.n_frames, info.width, info.height
    lo, hi = max(0, f0 - 1), min(n - 1, f1)
    ext = list(range(lo, hi + 1))
    masks, rep = merged_masks(ext, (h, w))
    missing = [i for i in ext if not masks[i].any()]
    modes = {}
    for m in rep.values():
        modes[m] = modes.get(m, 0) + 1
    log(f"[mask] frames {f0}-{f1 - 1}: merge modes {modes}; empty masks {missing[:12]}")
    alpha, yd = {}, {}
    for k, x in enumerate(iter_frames(VIDEO, info)):
        if k < lo:
            continue
        if k > hi:
            break
        alpha[k] = build_alpha(masks[k], x)
        core = cv2.erode((alpha[k] > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        yd[k] = float(np.percentile(luma(np.clip(x, 0, 1))[core], 90)) if core.sum() > 200 else np.nan
    idx = list(range(f0, f1))
    med = {i: median3(alpha[max(lo, i - 1)], alpha[i], alpha[min(hi, i + 1)]) for i in idx}
    y = np.array([yd.get(i, np.nan) for i in idx], np.float64)
    ok = ~np.isnan(y)
    y = np.interp(idx, np.array(idx)[ok], y[ok]) if ok.any() else np.full(len(idx), 0.6)
    ys = smooth1d(y, 6.0, 18)
    size = np.array([diver_size(med[i]) for i in idx], np.float64)
    ok = ~np.isnan(size)
    size = np.interp(idx, np.array(idx)[ok], size[ok]) if ok.any() else np.full(len(idx), 300.0)
    ss = smooth1d(size, 8.0, 24)
    dist = np.clip((DV["size_near"] - ss) / (DV["size_near"] - DV["size_far"]), 0, 1)
    tasks = {i: dict(al=med[i], y_ref=float(ys[k]), dist=float(dist[k]), sig=feather_sigma(float(ss[k])),
                     size=float(ss[k])) for k, i in enumerate(idx)}
    return info, tasks, masks, rep


def run_pipeline(f0, f1, out=None, qc_ids=(), workers=3, log=print):
    import multiprocessing as mp
    import dataclasses
    from shades.config import Encode
    from shades.io import Encoder, iter_frames
    info, tasks, masks, rep = prepare(f0, f1, log)
    enc = None
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        enc = Encoder(VIDEO, Path(out), dataclasses.replace(info, has_audio=False), Encode(crf=16, preset="slow"))
    keep = {}

    def gen():
        for k, x in enumerate(iter_frames(VIDEO, info)):
            if k < f0:
                continue
            if k >= f1:
                break
            t = tasks[k]
            if k in qc_ids:
                keep[k] = [x.copy(), None]
            yield (k, x, t["al"], t["y_ref"], t["dist"], t["sig"])

    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    it = pool.imap(grade_frame, gen(), chunksize=1) if pool else map(grade_frame, gen())
    done = 0
    for i, frame in it:
        if enc is not None:
            enc.write(frame)
        if i in qc_ids:
            keep[i][1] = frame
        done += 1
        if done % 50 == 0:
            log(f"[grade] {done}/{f1 - f0} frames")
    if pool:
        pool.close()
        pool.join()
    if enc is not None:
        enc.close()
    return keep, tasks, masks, rep


def thumb(img, tw=380):
    th = int(round(img.shape[0] * tw / img.shape[1]))
    return cv2.resize((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)


def sheets(keep, tasks, masks, ref_tile=None, name="preview", cols=4):
    ids = sorted(keep)
    WORK.mkdir(exist_ok=True)

    def grid(tiles):
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        return np.concatenate([np.concatenate(tiles[j:j + cols], 1) for j in range(0, len(tiles), cols)], 0)

    def overlay(i):
        x = keep[i][0]
        t = thumb(x, 400).copy()
        m = cv2.resize(tasks[i]["al"], (t.shape[1], t.shape[0])) > 127
        edge = m ^ (cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        t[m] = (0.6 * t[m] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
        t[edge] = (255, 255, 0)
        cv2.putText(t, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return t

    def ba(i):
        x, o = keep[i]
        pair = np.concatenate([thumb(x, 400), thumb(o, 400)], 0)
        cv2.putText(pair, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return pair

    ov = grid([overlay(i) for i in ids])
    ba_tiles = [ba(i) for i in ids]
    if ref_tile is not None:
        th = ba_tiles[0].shape[0]
        r = cv2.resize(ref_tile, (int(ref_tile.shape[1] * th / ref_tile.shape[0]), th), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, 400, 3), np.uint8)
        canvas[:, :min(400, r.shape[1])] = r[:, :400]
        cv2.putText(canvas, "graded reference", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ba_tiles.append(canvas)
    bsheet = grid(ba_tiles)
    cv2.imwrite(str(WORK / f"{name}_mask_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(WORK / f"{name}_before_after.jpg"), cv2.cvtColor(bsheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def reference_tile():
    p = WORK / "ref" / "ref_5.6.png"
    if not p.exists():
        return None
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return im[300:2100]


def skin_and_gear_stats(x, out, al):
    """Graded skin and dark-gear medians inside the diver core."""
    core = cv2.erode((al > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if core.sum() < 200:
        return None, None
    y = luma(np.clip(x, 0, 1))
    y_ref = float(np.percentile(y[core], 90))
    w_skin, w_gear, _ = split_classes(x, y_ref, core)
    skin = core & (w_skin > 0.9)
    gear = core & (w_gear > 0.9)
    return region_stats(out, (0, 1, 0, 1), skin), region_stats(out, (0, 1, 0, 1), gear)


def cmd_preview(a):
    f0, f1 = map(int, a.frames.split(":"))
    qc = set(np.linspace(f0, f1 - 1, 8).round().astype(int).tolist())
    out = WORK / "preview" / f"preview_{f0}_{f1 - 1}.mp4"
    keep, tasks, masks, rep = run_pipeline(f0, f1, out=out, qc_ids=qc, workers=a.workers)
    sheets(keep, tasks, masks, ref_tile=reference_tile(), name=f"preview_{f0}_{f1 - 1}")
    for i in sorted(keep):
        x, o = keep[i]
        sk, ge = skin_and_gear_stats(x, o, tasks[i]["al"])
        print(i, "skin", sk, "gear", ge, "size", round(tasks[i]["size"]), "dist", round(tasks[i]["dist"], 2), rep.get(i))


def cmd_render(a):
    from shades.io import probe
    n = probe(VIDEO).n_frames
    out = REPO / "output" / "blocks_dive_graded.mp4"
    qc = set(a.qc)
    keep, tasks, masks, rep = run_pipeline(0, n, out=out, qc_ids=qc, workers=a.workers)
    (WORK / "final_alpha").mkdir(exist_ok=True)
    for i in qc:
        cv2.imwrite(str(WORK / "final_alpha" / f"{i:05d}.png"), tasks[i]["al"])
    sheets(keep, tasks, masks, ref_tile=reference_tile(), name="final", cols=5)
    json.dump({str(i): dict(size=t["size"], dist=t["dist"]) for i, t in tasks.items()}, open(WORK / "final_tasks.json", "w"))


# --------------------------------------------------------- mask QC / table --
def cmd_qc(a):
    """Per-frame numbers on the merged masks: area, jumps against the neighbours,
    dark pixels (hair, mask, fins) left outside, and which fallback was used."""
    from shades.io import probe
    n = probe(VIDEO).n_frames
    d = REPO / "masks" / f"{STEM}__diver" / "subject"
    have = [i for i in range(n) if (d / f"{i:05d}.png").exists()]
    if not have:
        print("no masks yet")
        return
    f0, f1 = (min(have), max(have) + 1) if not a.frames else map(int, a.frames.split(":"))
    idx = [i for i in range(f0, f1) if i in set(have)]
    info = __import__("shades.io", fromlist=["probe"]).probe(VIDEO)
    masks, rep = merged_masks(idx, (info.height, info.width))
    frames_dir = REPO / ".cache" / f"sam_frames_{STEM}_bwd"
    rows = {}
    areas = np.array([masks[i].sum() for i in idx], np.float64)
    for k, i in enumerate(idx):
        x = cv2.cvtColor(cv2.imread(str(frames_dir / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        y = luma(x)
        m = masks[i]
        near = cv2.dilate(m.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
        nb = areas[max(0, k - 3):k + 4]
        med = float(np.median(np.delete(nb, min(3, k) if k < 3 else 3))) if len(nb) > 1 else float(areas[k])
        rows[i] = dict(area=int(areas[k]), jump=round(float(areas[k] / max(med, 1) - 1), 2),
                       dark_out=int(((y < 0.34) & near & ~m).sum()), mode=rep[i])
    bad = {i: r for i, r in rows.items() if abs(r["jump"]) > 0.25 or r["dark_out"] > 150 or r["area"] == 0
           or r["mode"] not in ("tracker", "tracker+fins")}
    print(f"[qc] frames {idx[0]}-{idx[-1]}: {len(idx)} masks, area {int(areas.min())}-{int(areas.max())}, "
          f"{len(bad)} flagged")
    for i, r in list(bad.items())[:60]:
        print("   ", i, r)
    (WORK / "mask_qc.json").write_text(json.dumps(rows))


TARGETS = {
    "top_water": "V 0.09-0.21, H 205-225",
    "mid_water": "H 200-210, S 0.45-0.65",
    "bright_struct": "S <= 0.35, V >= 0.70",
    "skin": "H 14-26 (20 +- 6), S 0.24-0.34",
}


def cmd_measure(a):
    """Target against result, measured on the encoded output."""
    from shades.io import probe, read_frames_at
    info_out = probe(Path(a.video))
    info_in = probe(VIDEO)
    ids = a.frames
    raw = read_frames_at(VIDEO, info_in, ids)
    out = read_frames_at(Path(a.video), info_out, ids)
    rows = {}
    for i in ids:
        alp = cv2.imread(str(WORK / "final_alpha" / f"{i:05d}.png"), 0)
        al = alp if alp is not None else np.zeros(raw[i].shape[:2], np.uint8)
        free = cv2.dilate((al > 10).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
        rec = {k: region_stats(out[i], b, free) for k, b in ROLES.items()}
        rec["bright_struct"] = bright_structure(raw[i], out[i], al > 10)
        rec["skin"], rec["gear"] = skin_and_gear_stats(raw[i], out[i], al)
        rows[i] = rec
    for i, rec in rows.items():
        f = lambda k: (rec[k]["rgb"], rec[k]["hsv"]) if rec.get(k) else "n/a"
        print(f"f{i}: top {f('top_water')} | mid {f('mid_water')} | lit {f('bright_struct')} | skin {f('skin')} | gear {f('gear')}")
    (WORK / "measure_final.json").write_text(json.dumps(rows, indent=1))


# ----------------------------------------------------------------- stills --
STILL_FRAMES = [0, 150, 240, 360, 480, 600, 720, 849]


def read_frame(i):
    from shades.io import probe, read_frames_at
    info = probe(VIDEO)
    return read_frames_at(VIDEO, info, [i])[i]


def hsv_of(rgb):
    hh, ss, vv = colorsys.rgb_to_hsv(*[float(c) for c in rgb])
    return int(round(hh * 360)), round(ss, 2), round(vv, 2)


def region_stats(img, box, sel=None):
    H, W = img.shape[:2]
    y0, y1, x0, x1 = int(box[2] * H), int(box[3] * H), int(box[0] * W), int(box[1] * W)
    reg = img[y0:y1, x0:x1]
    px = reg[sel[y0:y1, x0:x1]] if sel is not None else reg.reshape(-1, 3)
    if len(px) < 50:
        return None
    m = np.median(px, 0)
    return dict(rgb=[int(round(c * 255)) for c in m], hsv=hsv_of(m))


ROLES = {   # free of the diver in every frame of the clip? checked per frame below
    "top_water": (0.62, 0.95, 0.03, 0.17),
    "mid_water": (0.62, 0.95, 0.31, 0.46),
    "bottom_water": (0.62, 0.95, 0.80, 0.95),
}


def bright_structure(raw, out, mask=None):
    """Lit sand/concrete: raw V >= 0.84 (the cube face level) in the lower 70 % of
    the frame, away from the diver. None when the frame has no such pixels."""
    h, w = raw.shape[:2]
    _, _, v = rgb2hsv(raw)
    v = v.copy()
    v[: int(0.30 * h)] = 0
    if mask is not None:
        v[cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0] = 0
    sel = v >= 0.84
    if sel.sum() < 30:
        return None
    px = out[sel]
    m = np.median(px, 0)
    return dict(rgb=[int(round(c * 255)) for c in m], hsv=hsv_of(m), n=int(sel.sum()))


def cmd_stills(a):
    out_dir = WORK / "stills_graded"
    out_dir.mkdir(parents=True, exist_ok=True)
    from shades.io import probe, read_frames_at
    info = probe(VIDEO)
    frames = read_frames_at(VIDEO, info, STILL_FRAMES)
    rows, tiles = {}, []
    for i in STILL_FRAMES:
        x = frames[i]
        g = grade_background(x)
        rec = {k: region_stats(g, b) for k, b in ROLES.items()}
        rec["raw_top"] = region_stats(x, ROLES["top_water"])
        rec["bright_struct"] = bright_structure(x, g)
        rec["raw_bright"] = bright_structure(x, x)
        rows[i] = rec
        t0 = cv2.resize((x * 255 + .5).astype(np.uint8), (400, 350), interpolation=cv2.INTER_AREA)
        t1 = cv2.resize((g * 255 + .5).astype(np.uint8), (400, 350), interpolation=cv2.INTER_AREA)
        pair = np.concatenate([t0, t1], 0)
        cv2.putText(pair, f"f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(pair)
        cv2.imwrite(str(out_dir / f"bg_{i:03d}.png"), cv2.cvtColor((g * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    sheet = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
    cv2.imwrite(str(WORK / "bg_stills_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    for i, rec in rows.items():
        print(i, json.dumps(rec))
    (WORK / "bg_stills.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stills").set_defaults(fn=cmd_stills)
    pv = sub.add_parser("preview")
    pv.add_argument("--frames", default="210:300")
    pv.add_argument("--workers", type=int, default=3)
    pv.set_defaults(fn=cmd_preview)
    q = sub.add_parser("qc")
    q.add_argument("--frames", default=None)
    q.set_defaults(fn=cmd_qc)
    ms = sub.add_parser("measure")
    ms.add_argument("--video", default=str(REPO / "output" / "blocks_dive_graded.mp4"))
    ms.add_argument("--frames", type=int, nargs="+", default=[0, 150, 240, 360, 480, 600, 720, 849])
    ms.set_defaults(fn=cmd_measure)
    rd = sub.add_parser("render")
    rd.add_argument("--workers", type=int, default=4)
    rd.add_argument("--qc", type=int, nargs="+", default=[0, 60, 150, 240, 360, 480, 600, 720, 849])
    rd.set_defaults(fn=cmd_render)
    args = ap.parse_args()
    raise SystemExit(args.fn(args))
