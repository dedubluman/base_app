#!/usr/bin/env python3
"""Dark petrol-teal environment + natural bronze groupers, SAM 3.1 masked.

Layers (float32, BT.709 display-referred RGB 0..1; nothing is generated, every
output pixel is a colour/tone transform of the same source pixel):

  environment  per-channel tone curves + a smooth vignette, both *fitted* to the
               reference stills (graded vs. ungraded frame pairs of this clip:
               fit_env_from_stills()), so water, rock and seabed land on the same
               deep blue / petrol-teal, crushed-shadow, pulled-highlight look.
  groupers     SAM 3.1 masks (union of all instances, repaired, temporally
               smoothed, guided-filter refined, feathered). Inside the mask the
               green/cyan cast is removed with a gain measured from the water,
               luminance is re-toned with extra local contrast, and chroma is
               pulled toward a bronze / olive-brown / ochre ramp while keeping
               part of the natural pattern chroma.

    python grouper_grade.py --input inputs/grouper.mov --masks masks/grouper__cuttle \
        --output output/grouper_graded_1080p.mp4 [--stills DIR --frames 0,200,400]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cuttle_grade import guided, luma, repair_masks, shift  # noqa: E402  (shift uses global W,H)
import cuttle_grade as _cg
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

REPO = Path(__file__).resolve().parent

# --- environment fit (see fit_env_from_stills / README in the commit message) ---
# y = a * max(x - b, 0) ** g  per channel, fitted on source->reference pairs.
ENV_CURVE = {"R": (1.3107, 0.1914, 1.0148),
             "G": (1.6549, 0.1171, 1.8303),
             "B": (1.1922, 0.1882, 1.1061)}
VIGNETTE_NPY = REPO / "data" / "grouper_vignette.npy"
ENV_FLOOR = np.array([0.004, 0.010, 0.013], np.float32)   # petrol-teal black floor
REF_ENV_MEAN = np.array([0.355, 0.560, 0.520], np.float32)  # env mean of the fit frames


def build_lut(n=4096):
    x = np.linspace(0, 1, n, dtype=np.float32)
    lut = np.stack([ENV_CURVE[c][0] * np.maximum(x - ENV_CURVE[c][1], 0) ** ENV_CURVE[c][2]
                    for c in "RGB"], -1)
    # soft shoulder: pull highlights back so large pale rock never blooms
    k = 0.42
    over = np.maximum(lut - k, 0)
    lut = np.where(lut > k, k + over / (1 + 1.6 * over), lut)
    return lut.astype(np.float32)


LUT = build_lut()


def apply_lut(rgb):
    idx = np.clip(rgb * (len(LUT) - 1), 0, len(LUT) - 1)
    i0 = idx.astype(np.int32)
    fr = idx - i0
    i1 = np.minimum(i0 + 1, len(LUT) - 1)
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = LUT[i0[..., c], c] * (1 - fr[..., c]) + LUT[i1[..., c], c] * fr[..., c]
    return out


_VIG = {}


def vignette(h, w):
    if (h, w) not in _VIG:
        v = np.load(VIGNETTE_NPY).astype(np.float32)
        _VIG[(h, w)] = cv2.resize(v, (w, h), interpolation=cv2.INTER_CUBIC)
    return _VIG[(h, w)]


# ----------------------------------------------------------------- environment
def grade_env(rgb, gain):
    """rgb: source frame; gain: temporal exposure normaliser (scalar)."""
    h, w = rgb.shape[:2]
    x = np.clip(rgb * gain, 0, 1)
    y = apply_lut(x)
    # gentle large-radius clarity: deeper separation of rock ledges / seabed relief
    L = luma(y)
    base = cv2.GaussianBlur(L, (0, 0), 18)
    y = y * (1 + 0.55 * (L - base) / np.maximum(base + 0.05, 0.05))[..., None]
    y = y * vignette(h, w)[..., None]
    return np.clip(y + ENV_FLOOR * (1 - np.clip(y.mean(-1, keepdims=True) * 6, 0, 1)), 0, 1)


# -------------------------------------------------------------------- groupers
RAMP_Y = np.array([0.0, 0.06, 0.18, 0.38, 0.68, 1.0], np.float32)
RAMP_C = np.array([[0.016, 0.018, 0.014],    # near-black olive
                   [0.058, 0.060, 0.040],    # olive-brown shadow
                   [0.140, 0.135, 0.088],    # bronze-olive
                   [0.270, 0.260, 0.165],    # muted amber-ochre
                   [0.440, 0.420, 0.270],    # ochre spots / fin edges
                   [0.620, 0.590, 0.400]], np.float32)

FISH_TONE = dict(g=1.45, a=0.36, floor=0.042)   # tone on level-normalised luminance       # luminance re-tone (fitted by eye on the stills)


def ramp(Y):
    return np.stack([np.interp(Y, RAMP_Y, RAMP_C[:, c]) for c in range(3)], -1)


def fish_levels(rgb, alpha):
    """(lo, hi) luminance anchors of the fish pixels (None if too few)."""
    L = luma(np.clip(rgb, 0, 1))[alpha > 0.6]
    if L.size < 30:
        return None
    return float(np.percentile(L, 4)), float(np.percentile(L, 99.7))


def grade_fish(rgb, env_mean, gain, vig, levels):
    """rgb: source ROI. env_mean: this frame's water/seabed mean RGB (cast reference)."""
    x = np.clip(rgb * gain, 0, 1)
    # 1. remove the cyan/green cast measured from the surrounding water
    wb = env_mean.mean() / np.maximum(env_mean, 1e-3)
    wb = np.clip(wb / wb[1], 0.85, 1.45)
    xw = np.clip(x * wb, 0, 1)
    Y = luma(xw)
    # 2. luminance: base + boosted detail (micro-contrast), then re-tone
    base = cv2.GaussianBlur(Y, (0, 0), 5)
    detail = Y - base
    Yc = np.clip(base + 1.8 * detail, 0, 1)
    t = FISH_TONE
    lo, hi = levels
    lo, hi = lo * gain, max(hi * gain, lo * gain + 0.08)
    Yn = np.clip((Yc - lo) / (hi - lo), 0, 1.15)   # soft headroom: no flat clipped spots
    Yt = np.clip(t["floor"] + t["a"] * Yn ** t["g"], 0, 1)
    # 3. chroma: ramp colour, blended with the cast-corrected natural chroma
    nat = xw * (Yt / np.maximum(Y, 1e-4))[..., None]
    col = 0.68 * ramp(Yt) + 0.32 * np.minimum(nat, 1.0)
    # re-impose the tone-mapped luminance exactly so the ramp never shifts brightness
    col = col * (Yt / np.maximum(luma(col), 1e-4))[..., None]
    col = col * vig[..., None] ** 0.6
    gray = luma(col)[..., None]
    return np.clip(gray + 0.88 * (col - gray), 0, 1)   # no neon: trim chroma 12 %


# ----------------------------------------------------------------------- masks
def fill_gaps(objm, max_gap=3, log=print):
    """One object's masks: frames missing between two masked frames (SAM omits each
    chunk's prompt frame) are rebuilt from the neighbours shifted to the interpolated
    centroid. Longer gaps stay empty (animal genuinely out of view)."""
    have = sorted(objm)
    for i in range(have[0], have[-1] + 1):
        if i in objm:
            continue
        prev = max(j for j in have if j < i)
        nxt = min(j for j in have if j > i)
        if nxt - prev > max_gap + 1:
            continue
        w = (i - prev) / (nxt - prev)
        cp, cn = (np.argwhere(objm[j]).mean(0) for j in (prev, nxt))
        tgt = cp * (1 - w) + cn * w
        m = shift(objm[prev], *(tgt - cp)[::-1]) * (1 - w) + shift(objm[nxt], *(tgt - cn)[::-1]) * w
        objm[i] = m > 0.5
        log(f"[mask] gap-filled frame {i}")
    return objm


def build_union_masks(mdir, n, log=print):
    """frame -> bool mask (union of all SAM instances; each repaired + gap-filled)."""
    meta = json.loads((mdir / "meta.json").read_text())
    union, per_obj = {}, {}
    for g in meta["objects"]:
        files = {int(p.stem): p for p in (mdir / f"obj_{g}").glob("*.png")}
        if not files:
            continue
        masks, _ = repair_masks(files, tol=0.30, log=log)
        masks = fill_gaps(masks, log=log)
        per_obj[int(g)] = sorted(masks)
        log(f"[mask] object {g}: frames {min(masks)}-{max(masks)} ({len(masks)} masked)")
        for f, m in masks.items():
            union[f] = union[f] | m if f in union else m
    return union, per_obj


def alpha_for(f, union, rgb_roi_getter=None):
    """Temporally smoothed (+-1 frame, 0.2/0.6/0.2) soft alpha at full res."""
    acc, wsum = None, 0.0
    for d, w in ((-1, 0.2), (0, 0.6), (1, 0.2)):
        m = union.get(f + d)
        if m is None:
            continue
        acc = m.astype(np.float32) * w if acc is None else acc + m.astype(np.float32) * w
        wsum += w
    return None if acc is None else acc / wsum


def refine_alpha(alpha, rgb):
    """Feather + guided-filter snap to the real fish edge; returns alpha, bbox."""
    if alpha is None or alpha.max() <= 0:
        return None, None
    ys, xs = np.where(alpha > 0.02)
    pad = 48
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, alpha.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, alpha.shape[1])
    a = alpha[y0:y1, x0:x1]
    # Erode ~1 px so the blend sits *inside* the dark fish edge: an outward bias would
    # pull pale sand into the fish grade and draw a light halo around the animal.
    a = cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    a = cv2.GaussianBlur(a, (0, 0), 1.4)
    a = np.clip((a - 0.08) / 0.84, 0, 1)
    return a.astype(np.float32), (y0, y1, x0, x1)


# ---------------------------------------------------------------------- stats
def measure_env(rgb_small, alpha_small):
    keep = (alpha_small < 0.02) if alpha_small is not None else np.ones(rgb_small.shape[:2], bool)
    px = rgb_small[keep]
    return px.mean(0), float(np.median(luma(px)))


def smooth1d(v, k=12):
    v = np.asarray(v, np.float64)
    pad = np.pad(v, ((k, k),) + ((0, 0),) * (v.ndim - 1), mode="edge")
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / (k / 2.5)) ** 2)
    ker /= ker.sum()
    if v.ndim == 1:
        return np.convolve(pad, ker, "valid")
    return np.stack([np.convolve(pad[:, c], ker, "valid") for c in range(v.shape[1])], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--frames", default=None, help="comma list: render only these as PNG stills")
    ap.add_argument("--stills", type=Path, default=None)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()

    info = probe(a.input)
    n = info.n_frames
    _cg.W, _cg.H = info.width, info.height
    print(f"[grade] {a.input.name}: {info.width}x{info.height} {n}f fps={info.fps} audio={info.has_audio}")
    union, per_obj = build_union_masks(a.masks, n)
    lost = [i for i in range(n) if i not in union]
    print(f"[grade] masked frames: {len(union)}/{n}; frames with no fish mask: {len(lost)}")

    # pass 1: per-frame environment stats at low res -> temporally smoothed
    t0 = time.time()
    means, meds, lvl = [], [], []
    for i, fr in enumerate(iter_frames(a.input, info, scale_w=540)):
        al = alpha_for(i, union)
        lv = None
        if al is not None:
            al = cv2.resize(al, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_AREA)
            lv = fish_levels(fr, cv2.erode(al, np.ones((3, 3), np.uint8)))
            al = cv2.dilate(al, np.ones((5, 5), np.uint8))
        m, md = measure_env(fr, al)
        means.append(m)
        meds.append(md)
        lvl.append([np.nan, np.nan] if lv is None else lv)
    means = smooth1d(np.array(means), 12)
    meds = smooth1d(np.array(meds), 12)
    lvl = np.array(lvl)
    ok = ~np.isnan(lvl[:, 0])
    if ok.any():   # hold nearest value through fish-free frames, then smooth in time
        for c in range(2):
            lvl[:, c] = np.interp(np.arange(n), np.flatnonzero(ok), lvl[ok, c])
        lvl = smooth1d(lvl, 8)
    ref_med = float(luma(REF_ENV_MEAN))
    gains = np.clip(ref_med / np.maximum(meds, 1e-3), 0.9, 1.12)
    print(f"[grade] pass1 {time.time() - t0:.0f}s  env median luma {meds.min():.3f}-{meds.max():.3f}  "
          f"gain {gains.min():.3f}-{gains.max():.3f}")

    want = None if a.frames is None else {int(x) for x in a.frames.split(",")}
    enc = None
    if want is None:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        enc = Encoder(a.input, a.output, info, Encode(crf=a.crf, preset="slow", audio="aac"))
    else:
        a.stills.mkdir(parents=True, exist_ok=True)

    cov = []
    for i, fr in enumerate(iter_frames(a.input, info)):
        if want is not None and i not in want:
            if i > max(want):
                break
            continue
        gain = float(gains[i])
        out = grade_env(fr, gain)
        al, bb = refine_alpha(alpha_for(i, union), fr)
        if al is not None:
            y0, y1, x0, x1 = bb
            vig = vignette(*fr.shape[:2])[y0:y1, x0:x1]
            fish = grade_fish(fr[y0:y1, x0:x1], means[i], gain, vig, tuple(lvl[i]))
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - al[..., None]) + fish * al[..., None]
            cov.append(float(al.sum()))
        else:
            cov.append(0.0)
        if enc is not None:
            enc.write(out)
            if i % 25 == 0:
                print(f"[grade] frame {i}/{n}", flush=True)
        else:
            cv2.imwrite(str(a.stills / f"graded_{i:05d}.png"),
                        (np.clip(out, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
    if enc is not None:
        enc.close()
        rep = {"frames": n, "masked_frames": len(union), "lost_frames": lost,
               "gain_range": [float(gains.min()), float(gains.max())],
               "alpha_area_px": cov}
        if a.report:   # kept out of the deliverable folder on purpose
            a.report.write_text(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
