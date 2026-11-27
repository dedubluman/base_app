#!/usr/bin/env python3
"""Reference-matched cinematic underwater grade (float pipeline, temporally stable).

Stages per frame (all float32, 0..1 sRGB/Rec.709-encoded):
  1. exposure / white-balance normalisation to the reference frame (smoothed, clamped gain)
  2. colour map fitted from the reference pair (uw_fit.py): environment polynomial,
     metal polynomial blended through the SAM matte
  3. organic vignette / light falloff field (fitted from the reference, heavily smoothed)
  4. depth-aware haze + local contrast (foreground richer, background softer / darker teal)
  5. subtle static volumetric light from the upper right
  6. 1080p-height resize, fine film grain (per-frame), dither, encode

Masks come from SAM 3.1 keyframes (uw_masks.py), warped to every frame by optical flow
between keyframes and refined with a guided filter (no halos).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from uw_fit import apply_poly

SRC_FPS_DEN = "30000/1001"


def decode(video: Path):
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", str(video)]))["streams"][0]
    w, h = info["width"], info["height"]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], bufsize=1 << 28)
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def guided_filter(guide, src, r=4, eps=2e-3):
    k = (2 * r + 1, 2 * r + 1)
    mg, ms = cv2.boxFilter(guide, -1, k), cv2.boxFilter(src, -1, k)
    cov = cv2.boxFilter(guide * src, -1, k) - mg * ms
    var = cv2.boxFilter(guide * guide, -1, k) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def load_keymasks(d: Path):
    ks = {}
    for p in sorted(d.glob("[0-9]*.png")):
        ks[int(p.stem)] = cv2.imread(str(p), 0)
    return ks


def interp_masks(frames, keys, stride):
    """Per-frame soft matte from keyframe masks: Farneback flow warp of both neighbours."""
    n, h, w = frames.shape[:3]
    kf = sorted(keys)
    gray_s = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
              for f in frames]
    out = np.zeros((n, h, w), np.float32)
    gx, gy = np.meshgrid(np.arange(w // 2, dtype=np.float32), np.arange(h // 2, dtype=np.float32))

    def warp(k, f):
        # flow f -> k tells where each pixel of f sits in k
        fl = cv2.calcOpticalFlowFarneback(gray_s[f], gray_s[k], None, 0.5, 4, 21, 3, 7, 1.5, 0)
        m = cv2.resize(keys[k], (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        wm = cv2.remap(m, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(wm, (w, h), interpolation=cv2.INTER_LINEAR)

    for f in range(n):
        lo = max([k for k in kf if k <= f], default=kf[0])
        hi = min([k for k in kf if k >= f], default=kf[-1])
        if lo == hi:
            out[f] = keys[lo].astype(np.float32) / 255 if lo == f else warp(lo, f)
        else:
            t = (f - lo) / (hi - lo)
            out[f] = (1 - t) * warp(lo, f) + t * warp(hi, f)
    return out


def refine_and_smooth(masks, frames):
    """Guided-filter edges to the image, then a light temporal blend (motion is small)."""
    n = len(masks)
    ref = np.zeros_like(masks)
    for f in range(n):
        g = cv2.cvtColor(frames[f], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        ref[f] = np.clip(guided_filter(g, masks[f]), 0, 1)
    sm = ref.copy()
    for f in range(1, n - 1):
        sm[f] = 0.25 * ref[f - 1] + 0.5 * ref[f] + 0.25 * ref[f + 1]
    return sm


def exposure_gains(frames, target_rgb, sigma=5.0, clamp=(0.86, 1.16)):
    n = len(frames)
    m = np.array([f[::8, ::8].reshape(-1, 3).mean(0) for f in frames], np.float64)
    k = int(sigma * 4) | 1
    pad = np.pad(m, ((k, k), (0, 0)), mode="edge")
    ker = cv2.getGaussianKernel(k, sigma).ravel()
    sm = np.stack([np.convolve(pad[:, c], ker, "same")[k:-k] for c in range(3)], 1)
    g = np.clip(target_rgb / sm, *clamp)
    # one more smoothing of the gains themselves
    padg = np.pad(g, ((k, k), (0, 0)), mode="edge")
    g = np.stack([np.convolve(padg[:, c], ker, "same")[k:-k] for c in range(3)], 1)
    return g


def smooth_vignette(V0):
    """Degree-4 polynomial surface fitted to log V (organic elliptical falloff, no scene blobs)."""
    h, w = V0.shape
    sm = cv2.resize(V0, (64, int(round(64 * h / w))), interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape
    yy, xx = np.mgrid[0:sh, 0:sw]
    x, y = (xx / (sw - 1) * 2 - 1).ravel(), (yy / (sh - 1) * 2 - 1).ravel()
    terms = [(i, j) for i in range(5) for j in range(5) if i + j <= 4]
    A = np.stack([x ** i * y ** j for i, j in terms], 1)
    coef, *_ = np.linalg.lstsq(A, np.log(np.clip(sm.ravel(), 0.05, 3)), rcond=None)
    gx, gy = np.meshgrid(np.linspace(-1, 1, w, dtype=np.float32), np.linspace(-1, 1, h, dtype=np.float32))
    logv = sum(c * gx ** i * gy ** j for c, (i, j) in zip(coef, terms))
    V = np.exp(logv).astype(np.float32)
    return V / V[h // 3:2 * h // 3, w // 3:2 * w // 3].mean()


def make_light(h, w):
    """Static, very soft diffuse light from the upper right (cool cyan)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = xx / w, yy / h
    # base gradient: stronger toward the top-right corner
    base = np.exp(-(((x - 0.80) / 0.42) ** 2 + ((y - 0.0) / 0.50) ** 2))
    # soft slanted bands (rays) following the same direction, low contrast
    ang = np.deg2rad(24)
    u = (x - 0.85) * np.cos(ang) + (y) * np.sin(ang) * (h / w)
    bands = 0.5 + 0.5 * (np.sin(u * 26 + 0.6) * 0.5 + np.sin(u * 47 + 2.1) * 0.3 + np.sin(u * 13) * 0.2)
    rays = base * (0.55 + 0.45 * bands) * np.clip(1 - y * 1.25, 0, 1)
    rays = cv2.GaussianBlur(rays, (0, 0), 0.01 * w)
    return rays.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--metal-masks", type=Path, default=None)
    ap.add_argument("--rock-masks", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-h", type=int, default=1080)
    ap.add_argument("--frames", default=None, help="a:b subset for tests")
    ap.add_argument("--dump", type=Path, default=None, help="dir for PNG dumps of selected frames")
    ap.add_argument("--dump-frames", default="0,75,150,225,299")
    ap.add_argument("--no-encode", action="store_true")
    ap.add_argument("--vig", type=float, default=1.0, help="vignette strength multiplier")
    ap.add_argument("--haze", type=float, default=0.10)
    ap.add_argument("--clarity", type=float, default=0.22)
    ap.add_argument("--light", type=float, default=0.045)
    ap.add_argument("--grain", type=float, default=0.007)
    ap.add_argument("--metal-sat", type=float, default=1.12, help="chroma gain inside the metal matte")
    ap.add_argument("--vig-bottom", type=float, default=0.12, help="extra darkening in the lower corners")
    ap.add_argument("--save-mattes", type=Path, default=None)
    ap.add_argument("--load-mattes", type=Path, default=None, help="reuse mattes saved by --save-mattes")
    ap.add_argument("--warm-cap", type=float, default=0.75, help="strength of red-over-green cap outside the metal matte")
    a = ap.parse_args()

    fit = np.load(a.fit, allow_pickle=True)
    coef, coef_m, V0 = fit["coef"], fit["coef_metal"], fit["V"].astype(np.float32)
    frames = decode(a.input)
    n, h, w = frames.shape[:3]
    sel = range(n)
    if a.frames:
        s, e = (int(v) for v in a.frames.split(":"))
        sel = range(s, e)
    print(f"[grade] {n} frames {w}x{h}", flush=True)

    # ---- mattes ------------------------------------------------------------
    metal = rock = None
    if a.load_mattes:
        metal = np.load(a.load_mattes / "metal.npy").astype(np.float32) / 255
        rock = np.load(a.load_mattes / "rock.npy").astype(np.float32) / 255
    elif a.metal_masks:
        metal = refine_and_smooth(interp_masks(frames, load_keymasks(a.metal_masks), a.stride), frames)
        print("[grade] metal mattes ready", flush=True)
    if a.rock_masks and not a.load_mattes:
        rock = refine_and_smooth(interp_masks(frames, load_keymasks(a.rock_masks), a.stride), frames)
        print("[grade] rock mattes ready", flush=True)
    if a.save_mattes:
        a.save_mattes.mkdir(parents=True, exist_ok=True)
        for nm, mm in (("metal", metal), ("rock", rock)):
            if mm is not None:
                np.save(a.save_mattes / f"{nm}.npy", (mm * 255).astype(np.uint8))

    # ---- exposure ---------------------------------------------------------
    target = frames[0][::8, ::8].reshape(-1, 3).mean(0).astype(np.float64)
    gains = exposure_gains(frames, target)
    np.save(Path(str(a.out) + ".gains.npy"), gains)

    # ---- static fields ------------------------------------------------------
    # vignette: reference falloff, heavily smoothed so it carries no scene content
    V = smooth_vignette(V0)
    V = np.power(np.clip(V, 0.05, 2), a.vig)
    gx_ = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    gy_ = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    bot = np.clip((gy_ - 0.55) / 0.45, 0, 1) ** 2 * (0.35 + 0.65 * np.abs(gx_) ** 1.5)
    V = V * (1 - a.vig_bottom * bot)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    depth = np.clip((0.78 - yy) / 0.78, 0, 1) ** 1.3          # 0 foreground (bottom) -> 1 far (top)
    light = make_light(h, w)
    haze_col = np.array([0.030, 0.105, 0.135], np.float32)
    light_col = np.array([0.40, 0.80, 0.90], np.float32)
    out_w = int(round(a.out_h * w / h / 2)) * 2
    out_h = a.out_h
    print(f"[grade] output {out_w}x{out_h}", flush=True)

    dump_set = {int(v) for v in a.dump_frames.split(",")} if a.dump else set()
    if a.dump:
        a.dump.mkdir(parents=True, exist_ok=True)
    enc = None
    if not a.no_encode:
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
             "-s", f"{out_w}x{out_h}", "-r", SRC_FPS_DEN, "-i", "-",
             "-i", str(a.input), "-map", "0:v", "-map", "1:a?",
             "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
             "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-profile:v", "high",
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
             "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(a.out)],
            stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)

    for f in sel:
        x = frames[f].astype(np.float32) / 255
        x *= gains[f].astype(np.float32)
        x = np.clip(x, 0, 1)
        flat = x.reshape(-1, 3)
        env = apply_poly(coef, flat).astype(np.float32).reshape(h, w, 3)
        if metal is not None and coef_m.size:
            met = apply_poly(coef_m, flat).astype(np.float32).reshape(h, w, 3)
            mm = metal[f][..., None]
            Lm = met @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            met = Lm[..., None] + (met - Lm[..., None]) * a.metal_sat      # gentle, bounded chroma lift
            # rocks / seabed stay olive-gray: soft cap on red exceeding green (no copper surfaces)
            ex = np.maximum(env[..., 0] - 1.0 * env[..., 1], 0)
            env = env.copy(); env[..., 0] -= a.warm_cap * ex
            col = env * (1 - mm) + met * mm
        else:
            col = env
        col = col * V[..., None] ** 1.0

        L = col @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        mm1 = metal[f] if metal is not None else np.zeros((h, w), np.float32)
        # depth haze (not on metal): lift toward dark teal, soften far contrast
        hz = (a.haze * depth ** 1.2 * (1 - mm1))[..., None]
        col = col * (1 - hz) + haze_col * hz
        # local contrast: richer in foreground / on metal, restrained far away
        low = cv2.GaussianBlur(L, (0, 0), 0.018 * w)
        rr = rock[f] if rock is not None else 0
        amt = a.clarity * (1 - 0.8 * depth) * (1 + 0.5 * mm1 + 0.25 * rr)
        gain = np.clip((L + amt * (L - low) + 1e-4) / (L + 1e-4), 0.75, 1.35)
        col = col * gain[..., None]
        # volumetric light, screen blend, tiny slow breathing so it feels like water (+/-6%)
        lt = (a.light * (1 + 0.06 * np.sin(2 * np.pi * f / 300 * 2)) * light)[..., None] * light_col
        col = 1 - (1 - np.clip(col, 0, 1)) * (1 - lt)
        col = np.clip(col, 0, 1)

        img = cv2.resize(col, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        # film grain: luma-weighted, slightly softened, new every frame
        g = rng.standard_normal((out_h, out_w)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.7) * 1.6
        Ls = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        wgt = (a.grain * (0.4 + 1.6 * np.sqrt(np.clip(Ls, 0, 1)) * (1 - 0.5 * Ls)))[..., None]
        img = img + g[..., None] * wgt * np.array([1.0, 1.0, 1.1], np.float32)
        img = np.clip(img, 0, 1)
        if f in dump_set:
            cv2.imwrite(str(a.dump / f"graded_{f:04d}.png"), cv2.cvtColor((img * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(a.dump / f"source_{f:04d}.png"), cv2.cvtColor(frames[f], cv2.COLOR_RGB2BGR))
        if enc is not None:
            enc.stdin.write((img * 65535 + rng.random(img.shape, dtype=np.float32)).clip(0, 65535).astype(np.uint16).tobytes())
        if f % 25 == 0:
            print(f"[grade] frame {f}", flush=True)
    if enc is not None:
        enc.stdin.close()
        enc.wait()
        print("[grade] encode finished, rc", enc.returncode, flush=True)


if __name__ == "__main__":
    main()
