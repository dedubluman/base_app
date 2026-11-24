#!/usr/bin/env python3
"""Cinematic underwater grade for the striped-fish clip, with SAM 3.1 mattes.

Stages (subcommands)
  alpha   SAM 3.1 keyframe masks (fish_masks.py) -> soft per-frame mattes for
          every frame: odd frames are optical-flow interpolations of the two
          neighbouring SAM masks; all mattes are snapped to real edges with a
          guided filter and feathered. Also reports the interpolation error on
          held-out SAM keyframes.
  still   grade chosen frames -> PNG (original | grade | matte) for QC
  render  full clip -> 1080x1920 H.264 MP4 (+ source audio copied)
  qc      contact sheet + temporal-stability statistics of a rendered file

Look (all in display-referred BT.709 RGB, float32; no per-frame adaptation, so
the grade cannot flicker):
  background  CDL fitted to the reference still (fit_reference.py): red
              crushed hardest -> deep teal/slate-green; flattened highlights;
              lifted teal-navy blacks; radial + vertical vignette; softer
              micro-contrast and desaturation toward the perimeter
  fish        its own grade through the matte: warm/white-balance correction,
              exposure lift, contrast, edge-aware clarity, vibrance, amber tail,
              soft highlight protection; only ~35% of the vignette
  finish      gentle filmic S-curve + highlight roll-off, subtle grain

    python fish_grade.py alpha  --input inputs/fish.mov --masks masks/fish
    python fish_grade.py still  --input inputs/fish.mov --masks masks/fish --frames 0,264,528
    python fish_grade.py render --input inputs/fish.mov --masks masks/fish --output output/fish_graded.mp4
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)          # parallelism comes from worker processes
REPO = Path(__file__).resolve().parent
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

# --------------------------------------------------------------------------
# Look parameters. bg_cdl / vignette are the numbers fitted by fit_reference.py
# on the reference still; the rest is creative direction.
# --------------------------------------------------------------------------
LOOK = {
    "bg": {
        "slope": [1.43, 1.68, 2.00], "offset": [-0.035, -0.058, -0.041],
        "power": [2.05, 1.47, 1.27],
        "sat": 1.20,                 # fitted 1.18
        "knee": 0.52, "ceiling": 0.76,   # luma soft-knee: flattens bright sand patches
        "hi_cool": [0.86, 1.00, 0.88],   # highlights go cool-neutral (never warm/pink)
        "lift": [0.003, 0.012, 0.018], "lift_pow": 2.6,   # teal-navy blacks, never crushed
        "slate": 0.10, "hi_desat": 0.55,   # slate mids; highlights near-neutral (ref sat 0.17)
        "vig_k": 0.66, "vig_q": 1.98, "vig_top": 0.16, "vig_bottom": 0.20,
        "edge_desat": 0.15,          # extra desaturation toward the perimeter
        "edge_contrast": 0.20,       # micro-contrast falloff toward the perimeter
    },
    "fg": {
        "exposure": 1.10, "wb": [0.97, 0.97, 1.10],     # neutralise the yellow-green water cast
        "slope": [1.30, 1.28, 1.30], "offset": [-0.075, -0.055, -0.040],
        "power": [1.12, 1.12, 1.10],
        "clarity": 0.7, "texture": 0.3,       # edge-aware local contrast
        "vibrance": 0.12, "sat": 1.04,
        "amber_hue": 46.0, "amber_sat": 0.18, "amber_pull": 0.35,
        "hi_knee": 0.74, "hi_ceiling": 0.95,
        "vig_share": 0.35,
        "alpha_bias": 0.12, "alpha_gamma": 1.3,   # tighten the matte (>0 / >1) against bright fringes
    },
    "finish": {
        "black": 0.004, "scurve": 0.30, "roll_knee": 0.86,
        "grain": 0.011, "grain_chroma": 0.30, "grain_blur": 0.65,
    },
}


def luma(x):
    return x @ LUMA


def gblur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def guided(I, p, r, eps):
    """He et al. guided filter, grey guide I and input p (float32, same shape)."""
    k = (2 * r + 1, 2 * r + 1)
    mI, mp_ = cv2.blur(I, k), cv2.blur(p, k)
    cov = cv2.blur(I * p, k) - mI * mp_
    var = cv2.blur(I * I, k) - mI * mI
    a = cov / (var + eps)
    b = mp_ - a * mI
    return cv2.blur(a, k) * I + cv2.blur(b, k)


# ------------------------------------------------------------------ mattes
def probe(path: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_packets,r_frame_rate",
         "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_read_packets"]), s["r_frame_rate"]


def decode_gray(path: Path, cache: Path, n: int, W: int, H: int) -> np.memmap:
    f = cache / f"{path.stem}_gray.u8"
    if not f.exists() or f.stat().st_size != n * W * H:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-fps_mode", "passthrough",
                        "-vf", "scale=in_range=tv:in_color_matrix=bt709,format=gray",
                        "-f", "rawvideo", str(f)], check=True)
    return np.memmap(f, np.uint8, "r", shape=(n, H, W))


def roi_of(masks, H, W, pad):
    ys, xs = [], []
    for m in masks:
        if m is not None and m.any():
            yy, xx = np.nonzero(m)
            ys += [yy.min(), yy.max()]
            xs += [xx.min(), xx.max()]
    if not ys:
        return None
    return max(min(xs) - pad, 0), max(min(ys) - pad, 0), min(max(xs) + pad + 1, W), min(max(ys) + pad + 1, H)


_dis = None


def flow(a, b):
    """Dense flow a -> b (uint8 grey crops), DIS medium."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(a, b, None)


def warp_soft(mask_f, fl, x0, y0, x1, y1):
    """Sample a soft mask (full frame float) at x + flow(x) inside the ROI."""
    h, w = y1 - y0, x1 - x0
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask_f[y0:y1, x0:x1], gx + fl[..., 0], gy + fl[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def interp_prior(t, nb_lo, nb_hi, gray, key_mask, H, W, pad=70):
    """Soft prior at frame t from the SAM masks at frames nb_lo and nb_hi."""
    ml, mh = key_mask(nb_lo), key_mask(nb_hi)
    roi = roi_of([ml, mh], H, W, pad)
    prior = np.zeros((H, W), np.float32)
    if roi is None:
        return prior, None
    x0, y0, x1, y1 = roi
    g = np.ascontiguousarray(gray[t, y0:y1, x0:x1])
    acc = 0
    for nb, m in ((nb_lo, ml), (nb_hi, mh)):
        fl = flow(g, np.ascontiguousarray(gray[nb, y0:y1, x0:x1]))   # t -> nb
        acc = acc + warp_soft(m.astype(np.float32), fl, x0, y0, x1, y1)
    prior[y0:y1, x0:x1] = acc / 2
    return prior, roi


def refine(prior, g_full, roi, H, W):
    """Soft prior -> matte: guided filter on the real image, constrained to a
    band around the prior (no leaks, no holes), then a ~1 px feather."""
    x0, y0, x1, y1 = roi
    p = prior[y0:y1, x0:x1]
    I = g_full[y0:y1, x0:x1].astype(np.float32) / 255
    a = np.clip(guided(I, p, 5, 3e-4), 0, 1)
    hard = (p > 0.5).astype(np.uint8)
    lo = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
    hi = cv2.dilate((p > 0.04).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    a = np.maximum(np.minimum(a, hi), lo * 0.98)
    a = np.clip(gblur(a, 1.0), 0, 1)
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = (a * 255 + 0.5).astype(np.uint8)
    return out


def load_meta(mdir: Path, video: Path):
    """meta.json once SAM has finished; while it is still running, rebuild what
    alpha needs from the keyframe PNGs on disk (partial dev runs only)."""
    f = mdir / "meta.json"
    if f.exists():
        return json.loads(f.read_text()), False
    W, H, n, _ = probe(video)
    keys = sorted(int(p.stem) for p in (mdir / "key").glob("*.png"))
    return {"width": W, "height": H, "num_frames": n, "keyframes": keys,
            "stride": 2, "missing_keyframes": []}, True


def cmd_alpha(a):
    mdir = a.masks
    meta, partial = load_meta(mdir, a.input)
    W, H, n = meta["width"], meta["height"], meta["num_frames"]
    keys = meta["keyframes"]
    assert not meta["missing_keyframes"], f"missing SAM keyframes: {meta['missing_keyframes'][:10]}"
    frames = [int(s) for s in a.frames.split(",")] if a.frames else list(range(n))
    if a.frames:
        a.no_validate = True
    if partial and not a.frames:
        raise SystemExit("SAM has not finished (no meta.json): pass --frames for a partial dev run")
    cache = REPO / ".cache"
    gray = decode_gray(a.input, cache, n, W, H)
    kd = mdir / "key"
    kcache: dict[int, np.ndarray] = {}

    def key_mask(i):
        if i not in kcache:
            if len(kcache) > 8:
                kcache.pop(next(iter(kcache)))
            kcache[i] = cv2.imread(str(kd / f"{i:05d}.png"), 0) > 127
        return kcache[i]

    adir = mdir / "alpha"
    adir.mkdir(exist_ok=True)
    stride = meta["stride"]
    kset = set(keys)

    # --- held-out validation of the interpolation (distance = stride, i.e.
    #     twice as far as the real in-between frames)
    rep = {}
    if not a.no_validate:
        ious = []
        for k in keys[1:-1:max(1, len(keys) // 40)]:
            lo, hi = k - stride, k + stride
            if lo in kset and hi in kset:
                pr, roi = interp_prior(k, lo, hi, gray, key_mask, H, W)
                truth = key_mask(k)
                pm = pr > 0.5
                u = np.logical_or(pm, truth).sum()
                ious.append(float(np.logical_and(pm, truth).sum() / u) if u else 1.0)
        rep = {"heldout_interp_iou_mean": float(np.mean(ious)), "heldout_interp_iou_min": float(np.min(ious)),
               "heldout_interp_n": len(ious)}
        print("[alpha] held-out interpolation (2x real gap):", rep, flush=True)

    areas = {}
    t0 = time.time()
    for t in frames:
        if t in kset:
            m = key_mask(t)
            prior = m.astype(np.float32)
            roi = roi_of([m], H, W, 70)
        else:
            lo = max(k for k in keys if k < t)
            hi = min((k for k in keys if k > t), default=lo)
            prior, roi = interp_prior(t, lo, hi, gray, key_mask, H, W)
        if roi is None:
            al = np.zeros((H, W), np.uint8)
        else:
            al = refine(prior, gray[t], roi, H, W)
        cv2.imwrite(str(adir / f"{t:05d}.png"), al)
        areas[t] = float(al.sum() / 255)
        if t % 100 == 0:
            print(f"[alpha] {t}/{n} {time.time() - t0:.0f}s", flush=True)
    if not a.frames:
        ar = np.array([areas[t] for t in range(n)])
        # temporal sanity: relative area change between neighbouring frames
        d = np.abs(np.diff(ar)) / np.maximum(ar[:-1], 1)
        rep.update(area_min=float(ar.min()), area_max=float(ar.max()),
                   max_frame_to_frame_area_change=float(d.max()),
                   frames_area_jump_gt_12pct=[int(i) + 1 for i in np.nonzero(d > 0.12)[0]])
        (mdir / "alpha_report.json").write_text(json.dumps(rep, indent=1))
    print("[alpha] done", json.dumps(rep), flush=True)


# -------------------------------------------------------------------- look
def cdl_lut(slope, offset, power, n=2048):
    xs = np.linspace(0, 1, n, dtype=np.float32)
    return [np.clip(xs * s + o, 0, None) ** p for s, o, p in zip(slope, offset, power)]


def apply_lut(x, luts):
    xs = np.linspace(0, 1, len(luts[0]), dtype=np.float32)
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = np.interp(np.clip(x[..., c], 0, 1), xs, luts[c])
    return out


def saturate(x, s):
    L = luma(x)[..., None]
    return L + s * (x - L)


def soft_knee(L, knee, ceil):
    return np.where(L > knee, knee + (ceil - knee) * np.tanh((L - knee) / max(ceil - knee, 1e-4)), L)


def flatten_highlights(x, knee, ceil):
    L = luma(x)
    k = soft_knee(L, knee, ceil) / np.maximum(L, 1e-4)
    return x * k[..., None]


def geometry(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = xx / (W - 1), yy / (H - 1)
    r = np.hypot((xn - .5) * 2, (yn - .5) * 2)
    return r.astype(np.float32), yn.astype(np.float32)


def vignette_map(r, yn, P):
    v = np.exp(-P["vig_k"] * r ** P["vig_q"])
    v = v * (1 - P["vig_top"] * np.clip(1 - yn / 0.35, 0, 1) ** 1.5)
    v = v * (1 - P["vig_bottom"] * np.clip((yn - 0.75) / 0.25, 0, 1) ** 1.5)
    return v.astype(np.float32)


def bg_look(x, r, v, P, luts):
    t = apply_lut(x, luts)
    t = saturate(t, P["sat"])
    t = flatten_highlights(t, P["knee"], P["ceiling"])
    Lh = np.clip((luma(t) - 0.28) / 0.40, 0, 1)
    Lh = Lh * Lh * (3 - 2 * Lh)
    t = t * (1 + (np.array(P["hi_cool"], np.float32) - 1) * Lh[..., None])
    # edge-aware depth falloff: perimeter loses micro-contrast and saturation
    edge = np.clip((r - 0.35) / 0.9, 0, 1) ** 1.5
    if P["edge_contrast"] > 0:
        base = gblur(t, 10)
        t = base + (t - base) * (1 - P["edge_contrast"] * edge)[..., None]
    Lt = luma(t)[..., None]
    t = Lt + (t - Lt) * (1 - P["edge_desat"] * edge)[..., None]
    # slate: upper mids give up some saturation (cool, heavy), shadows stay teal
    Lm = np.clip(luma(t), 0, 1)
    w = (np.clip((Lm - 0.18) / 0.5, 0, 1) * P["slate"]
         + np.clip((Lm - 0.46) / 0.34, 0, 1) * P["hi_desat"])[..., None]
    t = luma(t)[..., None] + (t - luma(t)[..., None]) * (1 - w)
    # teal-navy lifted blacks (so the shadows keep colour and detail)
    sh = ((1 - np.clip(Lm / 0.5, 0, 1)) ** P["lift_pow"])[..., None]
    t = t + sh * np.array(P["lift"], np.float32)
    return t * v[..., None]


def rgb2hsv(x):
    """float32 RGB 0..1 -> HSV with H in degrees (0..360), S and V in 0..1."""
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)


def hsv2rgb(h):
    return cv2.cvtColor(h.astype(np.float32), cv2.COLOR_HSV2RGB)


def fg_look(x, vshare, P, luts):
    """x: source crop (h,w,3) float32. Returns the fish grade for that crop."""
    y = luma(x)
    # edge-preserving clarity: boost luminance detail with a self-guided base
    base = guided(y, y, 8, 3e-3)
    fine = y - gblur(y, 1.2)
    ratio = (y + P["clarity"] * (y - base) + P["texture"] * fine * 0.5) / np.maximum(y, 1e-3)
    ratio = np.clip(ratio, 0.5, 2.0)
    t = x * ratio[..., None] * P["exposure"] * np.array(P["wb"], np.float32)
    t = apply_lut(t, luts)
    t = saturate(t, P["sat"])
    # vibrance (low-sat colours get more) + amber tail/fins
    h = rgb2hsv(t)
    H_, S_, V_ = h[..., 0], h[..., 1], h[..., 2]
    S_ = S_ * (1 + P["vibrance"] * (1 - S_))
    dh = (H_ - P["amber_hue"] + 180) % 360 - 180
    wgt = np.exp(-(dh / 20.0) ** 2) * np.clip((S_ - 0.38) / 0.30, 0, 1)   # only truly saturated gold (tail, fins)
    S_ = S_ * (1 + P["amber_sat"] * wgt)
    H_ = (H_ - dh * P["amber_pull"] * wgt) % 360
    t = hsv2rgb(np.stack([H_, np.clip(S_, 0, 1), V_], -1))
    t = flatten_highlights(t, P["hi_knee"], P["hi_ceiling"])
    return t * vshare[..., None]


def finish(x, idx, P, shape):
    # lifted black + gentle filmic S-curve + highlight roll-off
    x = P["black"] + (1 - P["black"]) * x
    s = x * x * (3 - 2 * np.clip(x, 0, 1))
    x = x + P["scurve"] * (s - x)
    L = luma(x)
    k = soft_knee(L, P["roll_knee"], 1.0) / np.maximum(L, 1e-4)
    x = np.where((L > P["roll_knee"])[..., None], x * k[..., None], x)
    # grain: luminance-weighted, slightly soft, deterministic per frame
    rng = np.random.default_rng(1000003 + idx)
    g = gblur(rng.standard_normal(shape[:2]).astype(np.float32), P["grain_blur"])
    g = g / max(g.std(), 1e-6)
    Lc = np.clip(luma(x), 0, 1)
    wL = 0.55 + 1.6 * Lc * (1 - Lc)
    gc = (rng.standard_normal(shape).astype(np.float32) * P["grain_chroma"])
    x = x + (g * wL * P["grain"])[..., None] + gc * (wL * P["grain"] * 0.5)[..., None]
    return np.clip(x, 0, 1)


class Grader:
    def __init__(self, W, H, look=None):
        self.L = copy.deepcopy(LOOK)
        if look:
            for sec, d in look.items():
                self.L[sec].update(d)
        self.W, self.H = W, H
        self.r, self.yn = geometry(H, W)
        B, F = self.L["bg"], self.L["fg"]
        self.v = vignette_map(self.r, self.yn, B)
        self.bg_luts = cdl_lut(B["slope"], B["offset"], B["power"])
        self.fg_luts = cdl_lut(F["slope"], F["offset"], F["power"])

    def grade(self, rgb_u8, alpha_u8, idx):
        """rgb_u8 (H,W,3) uint8, alpha_u8 (H,W) uint8 -> float32 (H,W,3) 0..1."""
        x = rgb_u8.astype(np.float32) / 255
        out = bg_look(x, self.r, self.v, self.L["bg"], self.bg_luts)
        ys, xs = np.nonzero(alpha_u8 > 2)
        if len(ys):
            m = 40
            x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, self.W)
            y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, self.H)
            a = alpha_u8[y0:y1, x0:x1].astype(np.float32)[..., None] / 255
            F = self.L["fg"]
            a = np.clip((a - F["alpha_bias"]) / (1 - F["alpha_bias"]), 0, 1) ** F["alpha_gamma"]
            vsh = self.v[y0:y1, x0:x1] ** self.L["fg"]["vig_share"]
            fg = fg_look(x[y0:y1, x0:x1], vsh, self.L["fg"], self.fg_luts)
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + fg * a
        return finish(out, idx, self.L["finish"], out.shape)


# ---------------------------------------------------------------- frame io
def read_frame(path: Path, idx: int, W: int, H: int):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
                        "-vf", f"select=eq(n\\,{idx}),scale=in_range=tv:in_color_matrix=bt709,format=rgb24",
                        "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(H, W, 3)


def read_alpha(mdir: Path, idx: int):
    return cv2.imread(str(mdir / "alpha" / f"{idx:05d}.png"), 0)


def load_look(arg):
    return json.loads(Path(arg).read_text()) if arg else None


def cmd_still(a):
    W, H, n, _ = probe(a.input)
    G = Grader(W, H, load_look(a.look))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for i in [int(s) for s in a.frames.split(",")]:
        src = read_frame(a.input, i, W, H)
        al = read_alpha(a.masks, i)
        if al is None:
            al = np.zeros((H, W), np.uint8)
        g = (G.grade(src, al, i) * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(str(a.out_dir / f"g{i:05d}.png"), g[..., ::-1])
        cv2.imwrite(str(a.out_dir / f"s{i:05d}.png"), src[..., ::-1])
        cv2.imwrite(str(a.out_dir / f"a{i:05d}.png"), al)
        print("still", i, flush=True)


# ------------------------------------------------------------------ render
def _work(args):
    i, rgb, mdir, W, H, look = args
    global _G
    if _G is None:
        _G = Grader(W, H, look)
    al = read_alpha(Path(mdir), i)
    if al is None:
        al = np.zeros((H, W), np.uint8)
    return i, (_G.grade(rgb, al, i) * 65535 + 0.5).astype(np.uint16)


_G = None


def cmd_render(a):
    W, H, n, fps = probe(a.input)
    look = load_look(a.look)
    limit = a.limit or n
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(a.input), "-fps_mode", "passthrough",
         "-vf", "scale=in_range=tv:in_color_matrix=bt709,format=rgb24",
         "-f", "rawvideo", "-"], stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                     "stream=index", "-of", "csv=p=0", str(a.input)],
                                    capture_output=True, text=True).stdout.strip())
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}",
           "-framerate", fps, "-i", "-", "-i", str(a.input), "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
            "-c:v", "libx264", "-profile:v", "high", "-preset", a.preset, "-crf", str(a.crf),
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-movflags", "+faststart"]
    if a.limit:     # truncated test render: cut the copied audio to the same length
        cmd += ["-t", f"{limit * 1001 / 30000:.4f}"]
    cmd += [str(a.output)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fsz = W * H * 3
    pool = mp.get_context("fork").Pool(a.workers)
    pending: dict[int, mp.pool.AsyncResult] = {}
    nxt_write, nxt_read, t0 = 0, 0, time.time()
    try:
        while nxt_write < limit:
            while nxt_read < limit and len(pending) < a.workers * 2:   # bounded window
                b = dec.stdout.read(fsz)
                if len(b) < fsz:
                    limit = nxt_read
                    break
                rgb = np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
                pending[nxt_read] = pool.apply_async(_work, ((nxt_read, rgb, str(a.masks), W, H, look),))
                nxt_read += 1
            if nxt_write >= limit:
                break
            _, out = pending.pop(nxt_write).get()
            enc.stdin.write(out.tobytes())
            nxt_write += 1
            if nxt_write % 25 == 0:
                el = time.time() - t0
                print(f"[render] {nxt_write}/{limit} {el:.0f}s ({el / nxt_write:.2f}s/frame)", flush=True)
    finally:
        pool.close()
        pool.join()
        dec.stdout.close()
        dec.kill()
        enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg encode failed")
    print(f"[render] wrote {a.output} ({nxt_write} frames, {time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------- qc
def cmd_qc(a):
    """Temporal stability of source vs graded: per-frame mean luma/saturation of
    the fish layer and the background layer; flicker = std of the frame-to-frame
    change after removing the source's own motion (graded / source ratio)."""
    W, H, n, _ = probe(a.input)
    gw, gh, gn, _ = probe(a.graded)
    assert (gw, gh) == (W, H), (gw, gh)
    step = 2

    def frames(path):
        p = subprocess.Popen(["ffmpeg", "-v", "quiet", "-i", str(path), "-fps_mode", "passthrough",
                              "-vf", f"select=not(mod(n\\,{step})),scale=270:480:flags=area,format=rgb24",
                              "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
        sz = 270 * 480 * 3
        while True:
            b = p.stdout.read(sz)
            if len(b) < sz:
                break
            yield np.frombuffer(b, np.uint8).reshape(480, 270, 3).astype(np.float32) / 255
        p.kill()
        p.wait()

    rows = []
    for k, (s, g) in enumerate(zip(frames(a.input), frames(a.graded))):
        al = read_alpha(a.masks, k * step)
        m = cv2.resize(al, (270, 480), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        fgm, bgm = m > 0.6, m < 0.02
        r = {}
        for nm, im in (("src", s), ("gr", g)):
            L = luma(im)
            mx, mn = im.max(-1), im.min(-1)
            sat = mx - mn                      # chroma: stable in very dark colours (HSV S is not)
            r[nm] = [float(L[bgm].mean()), float(sat[bgm].mean()),
                     float(L[fgm].mean()) if fgm.any() else np.nan,
                     float(sat[fgm].mean()) if fgm.any() else np.nan]
        rows.append(r["src"] + r["gr"])
    R = np.array(rows)
    names = ["bgL", "bgChroma", "fgL", "fgChroma"]
    rep = {}
    for j, nm in enumerate(names):
        s, g = R[:, j], R[:, 4 + j]
        ok = ~np.isnan(s) & ~np.isnan(g)
        ds, dg = np.diff(s[ok]), np.diff(g[ok])
        # A static grade turns a source fluctuation ds into k*ds (k = the tone
        # curve's gain). Flicker the grade ADDS is what is left after that fit.
        k = float((ds @ dg) / max(ds @ ds, 1e-12))
        resid = dg - k * ds
        rep[nm] = {"src_mean": float(s[ok].mean()), "graded_mean": float(g[ok].mean()),
                   "step_corr_src_vs_graded": float(np.corrcoef(ds, dg)[0, 1]),
                   "curve_gain_k": k,
                   "added_flicker_std": float(resid.std()),
                   "added_flicker_over_graded_step_std": float(resid.std() / max(dg.std(), 1e-9)),
                   "graded_max_abs_step": float(np.abs(dg).max()), "n": int(ok.sum())}
    (a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# ------------------------------------------------------------------- sheet
def cmd_sheet(a):
    """QC contact sheet from the RENDERED file: per frame a column of
    original | graded | matte outline on the original (zoomed on the fish)."""
    W, H, n, _ = probe(a.input)
    cols = []
    for i in [int(x) for x in a.frames.split(",")]:
        src = read_frame(a.input, i, W, H)[..., ::-1].copy()
        gr = read_frame(a.graded, i, W, H)[..., ::-1].copy()
        al = read_alpha(a.masks, i)
        ov = src.copy()
        cnt, _ = cv2.findContours((al > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnt, -1, (0, 255, 255), 2)
        ys, xs = np.nonzero(al > 8)
        cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        x0, y0 = int(np.clip(cx - 270, 0, W - 540)), int(np.clip(cy - 200, 0, H - 400))
        col = [cv2.resize(src, (270, 480), interpolation=cv2.INTER_AREA),
               cv2.resize(gr, (270, 480), interpolation=cv2.INTER_AREA),
               cv2.resize(ov[y0:y0 + 400, x0:x0 + 540], (270, 200), interpolation=cv2.INTER_AREA),
               cv2.resize(gr[y0:y0 + 400, x0:x0 + 540], (270, 200), interpolation=cv2.INTER_AREA)]
        for im, lab in zip(col, (f"orig {i}", "graded", "SAM matte", "graded (fish)")):
            cv2.putText(im, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cols.append(np.vstack(col))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("alpha")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--frames", default=None, help="comma list: partial dev run (no validation)")
    p.add_argument("--no-validate", action="store_true")
    p.set_defaults(f=cmd_alpha)
    p = sp.add_parser("still")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out-dir", type=Path, default=REPO / "output" / "stills")
    p.add_argument("--look", default=None, help="JSON overriding LOOK sections")
    p.set_defaults(f=cmd_still)
    p = sp.add_parser("render")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--look", default=None)
    p.add_argument("--crf", type=int, default=17)
    p.add_argument("--preset", default="slow")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(f=cmd_render)
    p = sp.add_parser("qc")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--graded", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO / "output" / "qc_stability.json")
    p.set_defaults(f=cmd_qc)
    p = sp.add_parser("sheet")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--graded", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", type=Path, default=REPO / "output" / "qc_contact_sheet.jpg")
    p.set_defaults(f=cmd_sheet)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
