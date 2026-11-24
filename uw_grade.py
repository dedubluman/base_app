#!/usr/bin/env python
"""Reference-matched underwater grade with SAM 3.1 subject masks.

    python uw_grade.py --input clip.mov --masks masks/clip --out graded.mkv
    python uw_grade.py --input clip.mov --masks masks/clip --preview 20,80,120 --out prev/

Two passes, blended with the (temporally smoothed, guided-filter refined) SAM
mask ``alpha``:

* WORLD pass (whole frame): deep controlled blacks, filmic S-curve and highlight
  roll-off, richer blue-teal water (chroma up, hue nudged toward blue, residual
  green cast removed) and a warm amber bloom that follows the sun glare only.
* SUBJECT pass (inside alpha): removes the cyan cast the water put on skin and
  replaces it with a natural warm tan / peach chroma (Lab a*/b*, lightness kept),
  lifts shadow detail on the backlit body, and keeps near-black gear (mask,
  snorkel, stick) neutral.  Pale low-chroma gear such as shorts is excluded
  through an optional second SAM mask (``--shorts-masks``, any stride).

The output is a lossless 10-bit FFV1 intermediate at source resolution/fps.
Every constant lives in ``P`` below.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------- params
P = dict(
    # --- world tone
    black=0.012,          # black point (display-referred, 0..1)
    gamma=1.06,           # >1 sinks mid-tones a touch
    contrast=0.55,        # S-curve strength
    shoulder=0.86,        # highlight roll-off knee
    # --- world colour (Lab)
    water_chroma=1.32,    # chroma gain for blue/teal hues
    water_hue_shift=7.0,  # deg toward blue (Lab hue), only on teal hues
    green_fix=0.55,       # remove residual green cast (0..1)
    global_sat=1.04,
    # --- sun bloom
    glow_thresh=0.68,     # luminance where bloom starts
    glow_sigma=140.0,      # px (at 1232 wide) spread of the amber bloom
    glow_amount=0.85,
    glow_rgb=(1.55, 0.97, 0.66),   # max multiplicative tint at full bloom
    # --- subject (Lab a*, b* targets and strength)
    skin_a=15.0, skin_b=18.0,      # target chroma of sunlit skin
    skin_strength=0.92,
    skin_l_lift=0.50,              # shadow lift on the subject (0..1)
    skin_l_gain=1.28,
    skin_dark_floor=7.0,           # L* below which subject stays neutral
    # --- mask handling
    mask_erode=1,
    mask_feather=2.2,
    guide_radius=6,
    guide_eps=2e-3,
    temporal=(0.25, 0.5, 0.25),
)


# ----------------------------------------------------------------------------- helpers
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)


def guided_filter(guide, src, r, eps):
    """He et al. guided filter, single-channel guide (0..1)."""
    mg, ms = box(guide, r), box(src, r)
    cov = box(guide * src, r) - mg * ms
    var = box(guide * guide, r) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return box(a, r) * guide + box(b, r)


def to_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


# ----------------------------------------------------------------------------- tone
def tone_curve(x, p):
    x = np.clip((x - p["black"]) / (1 - p["black"]), 0, None)
    x = x ** p["gamma"]
    c = p["contrast"]
    y = np.clip(x, 0, 1)
    y = y + c * y * (1 - y) * (2 * y - 1) * 0.9          # S-curve, fixed ends
    k = p["shoulder"]
    over = np.maximum(y - k, 0)
    y = np.where(y > k, k + (1 - k) * np.tanh(over / (1 - k)), y)
    return np.clip(y, 0, 1)


# ----------------------------------------------------------------------------- world
def world_pass(rgb, p):
    # tone: apply the curve to luminance, keep chroma ratios
    L = luma(rgb)
    L2 = tone_curve(L, p)
    gain = (L2 + 1e-4) / (L + 1e-4)
    rgb = np.clip(rgb * gain[..., None], 0, 1)

    lab = to_lab(rgb)
    Lc, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360          # teal/cyan sit near 200-250 deg
    # weight of "water-like" hues (cyan -> blue), fades out for warm hues
    w_blue = smoothstep(150, 195, h) * (1 - smoothstep(270, 310, h))
    # darker water gets more saturation, bright foam / surface light less
    w_blue = w_blue * (1 - 0.65 * smoothstep(62, 92, Lc))
    # green cast removal: hues 165-205 (green-cyan) are rotated to blue-cyan
    w_green = smoothstep(120, 175, h) * (1 - smoothstep(205, 230, h))
    h2 = h + p["water_hue_shift"] * w_blue + 22.0 * p["green_fix"] * w_green * w_blue
    C2 = C * (1 + (p["water_chroma"] - 1) * w_blue)
    C2 = C2 * (1 + (p["global_sat"] - 1))
    a2 = C2 * np.cos(np.radians(h2))
    b2 = C2 * np.sin(np.radians(h2))
    out = from_lab(np.dstack([Lc, a2, b2]))

    # warm amber bloom around the sun glare
    L = luma(out)
    sig = p["glow_sigma"] * out.shape[1] / 1232.0
    glare = np.clip((L - p["glow_thresh"]) / (1 - p["glow_thresh"]), 0, 1)
    glare = cv2.GaussianBlur(glare, (0, 0), sig) * 3.0
    g = (np.clip(glare, 0, 1) ** 1.15)[..., None] * p["glow_amount"]
    # a small, very smooth tint, neutralising cyan highlight inside the bloom
    tint = 1 + (np.array(p["glow_rgb"], np.float32) - 1) * g
    out = np.clip(out * tint, 0, 1)
    return out


# ----------------------------------------------------------------------------- subject
def subject_pass(rgb, p, cast, shorts=None):
    """rgb = already world-graded frame.  cast = (a, b) median chroma of the
    subject's skin in that space (measured once for the whole clip)."""
    lab = to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    # keep deep blacks (mask, snorkel, stick) neutral
    w = smoothstep(p["skin_dark_floor"], p["skin_dark_floor"] + 14, L) * p["skin_strength"]
    w = w * (1 - 0.45 * smoothstep(62, 90, L))     # pale highlights stay less tinted
    if shorts is not None:                         # pale gear: only a hint of warmth
        w = w * (1 - 0.8 * shorts)
    # remove the water cast, add skin chroma. 0.78 keeps natural chroma variation
    a2 = a + w * ((a - cast[0]) * -0.0 + (p["skin_a"] - cast[0])) * 1.0
    b2 = b + w * (p["skin_b"] - cast[1])
    # lift shadows / mids on the backlit body, then a slight gain
    x = np.clip(L / 100.0, 0, 1)
    lift = p["skin_l_lift"]
    x2 = x + lift * (1 - x) ** 2 * x * 2.2
    # gain acts on shadows/mids only: bright, sun-lit skin is left alone
    g = 1 + (p["skin_l_gain"] - 1) * (1 - smoothstep(0.40, 0.72, x))
    x2 = np.clip(x2 * g, 0, 1)
    L2 = 100.0 * x2
    return from_lab(np.dstack([L2, a2, b2]))


# ----------------------------------------------------------------------------- masks
def load_soft_masks(mdir: Path, n: int, shape, p):
    h, w = shape
    out = np.zeros((n, h, w), np.float32)
    k = 2 * p["mask_erode"] + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for i in range(n):
        m = cv2.imread(str(mdir / "subject" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        m = (m > 127).astype(np.uint8) if m is not None else np.zeros((h, w), np.uint8)
        if p["mask_erode"] > 0:
            m = cv2.erode(m, ker)
        out[i] = m.astype(np.float32)
    # single-frame dropouts (SAM lost the body for one frame): rebuild from the
    # intersection of both neighbours when the area collapses
    area = out.reshape(n, -1).sum(1)
    fixed = []
    for i in range(1, n - 1):
        nb = min(area[i - 1], area[i + 1])
        if nb > 5000 and area[i] < 0.6 * nb:
            out[i] = np.maximum(out[i], out[i - 1] * out[i + 1])
            fixed.append(i)
    if fixed:
        print(f"[mask] repaired dropout frames {fixed}")
    # temporal smoothing of the (hard) masks -> soft, jitter-free alpha
    # (union with 85 % of each neighbour keeps motion-blurred limbs that SAM
    #  drops for a frame from flickering in and out)
    pad = np.concatenate([out[:1], out, out[-1:]])
    out = np.maximum(pad[1:-1], 0.85 * np.maximum(pad[:-2], pad[2:]))
    return out


def load_stride_masks(mdir: Path, n: int, shape, stride: int, p):
    """Masks computed on every `stride`-th frame -> per-frame soft masks."""
    h, w = shape
    ks = sorted(int(f.stem) for f in (mdir / "subject").glob("*.png"))
    base = np.zeros((len(ks), h, w), np.float32)
    for j, k in enumerate(ks):
        m = cv2.imread(str(mdir / "subject" / f"{k:05d}.png"), cv2.IMREAD_GRAYSCALE)
        base[j] = (m > 127).astype(np.float32)
    out = np.zeros((n, h, w), np.float32)
    for i in range(n):
        f = min(i / stride, len(ks) - 1)
        j0 = int(np.floor(f)); j1 = min(j0 + 1, len(ks) - 1); t = f - j0
        out[i] = (1 - t) * base[j0] + t * base[j1]
    return out


def refine_alpha(alpha, rgb, p):
    a = cv2.GaussianBlur(alpha, (0, 0), p["mask_feather"])
    guide = luma(rgb).astype(np.float32)
    a = guided_filter(guide, a, p["guide_radius"], p["guide_eps"])
    return np.clip(a, 0, 1)


# ----------------------------------------------------------------------------- io
def probe(path):
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)]))["streams"][0]
    num, den = map(int, j["r_frame_rate"].split("/"))
    return j["width"], j["height"], num, den


def read_frames(path, w, h):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-an",
                           "-fps_mode", "passthrough", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    sz = w * h * 3
    while True:
        buf = pr.stdout.read(sz)
        if len(buf) < sz:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
    pr.wait()


def grade_frame(rgb, alpha, p, cast, shorts=None):
    world = world_pass(rgb, p)
    if alpha is None or alpha.max() <= 0.001:
        return world, world
    sh = None
    if shorts is not None:
        sh = cv2.GaussianBlur(shorts, (0, 0), 3.0)
    subj = subject_pass(world, p, cast, sh)
    al = refine_alpha(alpha, rgb, p)[..., None]
    return world * (1 - al) + subj * al, world


def measure_cast(frames, masks, p):
    """Median Lab (a*, b*) of subject pixels in world-graded frames."""
    vals = []
    for rgb, m in zip(frames, masks):
        if m.max() < 0.5 or (m > 0.9).sum() < 4000:
            continue
        lab = to_lab(world_pass(rgb, p))
        sel = (m > 0.9) & (lab[..., 0] > 15) & (lab[..., 0] < 75)
        if sel.sum() > 500:
            vals.append(np.median(lab[sel][:, 1:], axis=0))
    return tuple(np.median(vals, axis=0)) if vals else (-8.0, -8.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--shorts-masks", type=Path, default=None,
                    help="SAM 'shorts' masks (pale gear gets only a hint of skin warmth)")
    ap.add_argument("--shorts-stride", type=int, default=1,
                    help="shorts masks were computed on every Nth frame")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preview", default="", help="comma frame list -> PNG previews in --out")
    ap.add_argument("--set", action="append", default=[], help="PARAM=VALUE override")
    a = ap.parse_args()

    p = dict(P)
    for kv in a.set:
        k, v = kv.split("=")
        p[k] = json.loads(v)
    w, h, num, den = probe(a.input)
    frames = list(read_frames(a.input, w, h))
    n = len(frames)
    masks = load_soft_masks(a.masks, n, (h, w), p)
    shorts = None
    if a.shorts_masks is not None:
        shorts = load_stride_masks(a.shorts_masks, n, (h, w), a.shorts_stride, p)
    cast = measure_cast(frames[::6], masks[::6], p)
    print(f"[grade] {n} frames {w}x{h}; subject cast a*,b* = {cast[0]:.1f},{cast[1]:.1f}")

    if a.preview:
        a.out.mkdir(parents=True, exist_ok=True)
        for i in map(int, a.preview.split(",")):
            g, wd = grade_frame(frames[i], masks[i], p, cast, None if shorts is None else shorts[i])
            al = refine_alpha(masks[i], frames[i], p)
            tile = np.concatenate([frames[i], g, np.repeat(al[..., None], 3, -1)], 1)
            cv2.imwrite(str(a.out / f"prev_{i:04d}.jpg"),
                        cv2.cvtColor((tile * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
         "-s", f"{w}x{h}", "-framerate", f"{num}/{den}", "-i", "-",
         "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p10le", "-colorspace", "bt709",
         "-color_primaries", "bt709", "-color_trc", "bt709", str(a.out)],
        stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    for i in range(n):
        g, _ = grade_frame(frames[i], masks[i], p, cast, None if shorts is None else shorts[i])
        g = g * 65535.0 + rng.uniform(-0.5, 0.5, g.shape).astype(np.float32) * 16
        enc.stdin.write(np.clip(g, 0, 65535).astype("<u2").tobytes())
        if i % 20 == 0:
            print(f"[grade] frame {i}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("[grade] done ->", a.out)


if __name__ == "__main__":
    main()
