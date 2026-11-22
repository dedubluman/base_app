#!/usr/bin/env python3
"""Cinematic underwater grade: orange starfish -> electric violet, dark petrol-teal
environment, blue/black fin preserved. SAM 3.1 masks drive every selective step.

    python starfish_grade.py --input inputs/starfish.mov --test 0 120 240 361   # stills
    python starfish_grade.py --input inputs/starfish.mov --render                # full clip

Masks (from sam_masks.py):  masks/<stem>__starfish/obj_*/<f>.png
                            masks/<stem>__fin/obj_*/<f>.png
Weights per frame: starfish ws, fin wf (minus starfish), environment wb = 1-ws-wf.
Everything is float32 and the per-clip statistics are global, so the grade is
temporally stable; only a gently smoothed exposure gain follows the footage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent

P = dict(
    # environment
    bg_gamma=2.55, bg_gain=0.92, bg_chroma_keep=0.10,
    bg_ramp=[(0.0, (0.008, 0.022, 0.040)), (0.12, (0.020, 0.075, 0.110)),
             (0.30, (0.050, 0.170, 0.215)), (0.55, (0.110, 0.330, 0.390)),
             (0.80, (0.330, 0.600, 0.650)), (1.0, (0.800, 0.920, 0.930))],
    vignette_strength=0.62, vignette_center=(0.5, 0.58), vignette_power=2.2,
    # fin
    fin_gamma=1.55, fin_gain=1.0,
    fin_ramp=[(0.0, (0.004, 0.006, 0.030)), (0.25, (0.015, 0.070, 0.300)),
              (0.50, (0.050, 0.240, 0.800)), (0.75, (0.280, 0.520, 1.000)),
              (0.92, (0.700, 0.820, 1.000)), (1.0, (0.960, 0.975, 1.000))],
    fin_keep_orig=0.25,
    # starfish
    star_ramp=[(0.0, (0.070, 0.025, 0.150)), (0.25, (0.200, 0.070, 0.400)),
               (0.55, (0.440, 0.190, 0.770)), (0.80, (0.620, 0.360, 0.950)),
               (1.0, (0.830, 0.700, 1.000))],
    star_luma_w=(0.60, 0.30, 0.10), star_vignette=0.35,
    # exposure stabiliser
    stab_sigma=10.0, stab_power=0.7, stab_clip=(0.8, 1.25),
)


def ramp_lut(stops, n=1024):
    x = np.array([s[0] for s in stops])
    c = np.array([s[1] for s in stops], np.float32)
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, x, c[:, k]) for k in range(3)], 1).astype(np.float32)


def apply_ramp(lut, y):
    idx = np.clip(y, 0, 1) * (len(lut) - 1)
    i0 = idx.astype(np.int32)
    i1 = np.minimum(i0 + 1, len(lut) - 1)
    f = (idx - i0)[..., None]
    return lut[i0] * (1 - f) + lut[i1] * f


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def load_union(root: Path, f: int, shape):
    m = np.zeros(shape, bool)
    for d in sorted(root.glob("obj_*")):
        p = d / f"{f:05d}.png"
        if p.exists():
            m |= cv2.imread(str(p), 0) > 127
    return m


def star_weight(rgb, mask):
    """SAM mask is the prior; the real edge is found inside a narrow band by
    redness so no orange halo survives and thin arm tips are kept."""
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m8 = mask.astype(np.uint8)
    core = cv2.erode(m8, k(3)).astype(np.float32)
    outer = cv2.dilate(m8, k(6)).astype(bool)
    redness = smoothstep(0.07, 0.22, rgb[..., 0] - rgb[..., 1])
    band = outer & (core == 0)
    w = np.where(band, redness, 0).astype(np.float32)
    # inside the SAM mask but within the eroded band, still trust SAM strongly
    inner = (m8 > 0) & (core == 0)
    w = np.where(inner, np.maximum(w, 0.55 + 0.45 * redness), w)
    w = np.maximum(w, core)
    return cv2.GaussianBlur(w, (0, 0), 0.8)


def vignette(h, w, p):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = p["vignette_center"]
    d = np.sqrt(((xx / w - cx) * 1.15) ** 2 + ((yy / h - cy) * 0.75) ** 2)
    d = d / 0.62
    return (1 - p["vignette_strength"] * np.clip(d, 0, 1.4) ** p["vignette_power"]).clip(0.15, 1).astype(np.float32)


class Grader:
    def __init__(self, stem, shape, star_stats, gains):
        self.h, self.w = shape
        self.stem = stem
        self.star_root = REPO / "masks" / f"{stem}__starfish"
        self.fin_root = REPO / "masks" / f"{stem}__fin"
        self.bg_l, self.fin_l, self.star_l = (ramp_lut(P[k]) for k in ("bg_ramp", "fin_ramp", "star_ramp"))
        self.vig = vignette(self.h, self.w, P)[..., None]
        self.star_lo, self.star_hi = star_stats
        self.gains = gains

    def weights(self, rgb, f):
        sm = load_union(self.star_root, f, (self.h, self.w))
        ws = star_weight(rgb, sm) if sm.any() else np.zeros((self.h, self.w), np.float32)
        if self.fin_root.exists():
            fm = load_union(self.fin_root, f, (self.h, self.w))
            wf = cv2.GaussianBlur(fm.astype(np.float32), (0, 0), 1.2)
        else:  # fallback only: blue-dominant / very dark pixels
            wf = np.zeros_like(ws)
        wf = np.clip(wf, 0, 1) * (1 - ws)
        return ws, wf

    def grade(self, rgb, f):
        ws, wf = self.weights(rgb, f)
        wb = np.clip(1 - ws - wf, 0, 1)
        g = float(self.gains[f]) if f < len(self.gains) else 1.0
        src = np.clip(rgb * g, 0, 1)
        Y = (src * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1)

        # environment: darker curve + teal gradient map, a little original chroma kept
        yb = np.clip(Y ** P["bg_gamma"] * P["bg_gain"] / 0.78, 0, 1)
        B = apply_ramp(self.bg_l, yb)
        keep = P["bg_chroma_keep"]
        if keep:
            dark = src * (yb / np.maximum(Y, 1e-3))[..., None]
            gray = dark.mean(-1, keepdims=True)
            B = B * (1 - keep) + np.clip(gray + (dark - gray) * 0.5, 0, 1) * keep * np.array([0.7, 0.95, 1.1], np.float32)
        # fin: blue-black identity, SUBEA lettering / caustics ride on luma
        yf = np.clip(Y ** P["fin_gamma"] * P["fin_gain"] * 1.15, 0, 1)
        F = apply_ramp(self.fin_l, yf)
        ko = P["fin_keep_orig"]
        orig = np.clip(src * (yf / np.maximum(Y, 1e-3))[..., None], 0, 1)
        F = F * (1 - ko) + orig * ko * np.array([0.8, 0.95, 1.1], np.float32)
        # starfish: gradient map on a red-weighted luma, global normalisation (no flicker)
        wl = np.array(P["star_luma_w"], np.float32)
        Ys = (src * wl).sum(-1)
        ys = np.clip((Ys - self.star_lo) / (self.star_hi - self.star_lo), 0, 1) ** 0.95
        S = apply_ramp(self.star_l, ys)

        out = B * wb[..., None] + F * wf[..., None] + S * ws[..., None]
        vig = self.vig
        # the starfish only receives a fraction of the vignette so it stays the focal point
        out = out * (vig + (1 - vig) * (1 - P["star_vignette"]) * ws[..., None])
        return np.clip(out, 0, 1), ws, wf


def decode_all(path, scale_w=None):
    """Yield float32 RGB frames, passthrough timing, 16-bit decode."""
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "json", str(path)]))["streams"][0]
    w, h = info["width"], info["height"]
    vf = []
    if scale_w:
        h = int(round(h * scale_w / w / 2) * 2)
        w = scale_w
        vf = ["-vf", f"scale={w}:{h}"]
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough", *vf,
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 6
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield (np.frombuffer(b, np.uint16).reshape(h, w, 3).astype(np.float32) / 65535.0)
    p.wait()


def prepass(video, stem, cache):
    """Per-clip statistics: exposure gains + starfish luma normalisation (global)."""
    f = cache / f"stats_{stem}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["star_stats"], np.array(d["gains"], np.float32)
    star_root = REPO / "masks" / f"{stem}__starfish"
    ml, ys = [], []
    wl = np.array(P["star_luma_w"], np.float32)
    for i, fr in enumerate(decode_all(video, 270)):
        sm = load_union(star_root, i, (fr.shape[0], fr.shape[1]) if False else (1920, 1080))
        sm = cv2.resize(sm.astype(np.uint8), (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        Y = fr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        ml.append(float(Y[~cv2.dilate(sm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)].mean()))
        if sm.any():
            ys.append((fr[cv2.erode(sm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] * wl).sum(-1))
    ml = np.array(ml)
    sig = P["stab_sigma"]
    k = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    pad = np.pad(ml, (len(k) // 2,) * 2, mode="edge")
    sm_ml = np.convolve(pad, k / k.sum(), "valid")
    gains = np.clip((sm_ml / ml) ** P["stab_power"], *P["stab_clip"])
    allv = np.concatenate(ys)
    stats = [float(np.percentile(allv, 3)), float(np.percentile(allv, 99.5))]
    f.write_text(json.dumps({"star_stats": stats, "gains": gains.tolist(), "bg_luma": ml.tolist()}))
    return stats, gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--test", type=int, nargs="*")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "output" / "starfish_grade")
    ap.add_argument("--scale", type=int, default=None, help="test width (render is always full-res)")
    a = ap.parse_args()
    stem = a.input.stem
    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    stats, gains = prepass(a.input, stem, cache)
    print("star luma norm", stats, "gain range", float(gains.min()), float(gains.max()))
    H, W = 1920, 1080
    G = Grader(stem, (H, W), stats, gains)

    if a.test:
        want = set(a.test)
        outs = []
        for i, fr in enumerate(decode_all(a.input)):
            if i in want:
                g, ws, wf = G.grade(fr, i)
                outs.append(np.hstack([fr, g]))
                cv2.imwrite(str(a.out / f"test_{i:05d}.png"), cv2.cvtColor((g * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(a.out / f"test_{i:05d}_ws.png"), (ws * 255).astype(np.uint8))
            if i > max(want):
                break
        sheet = np.hstack([cv2.resize(o, (o.shape[1] // 3, o.shape[0] // 3), interpolation=cv2.INTER_AREA) for o in outs])
        cv2.imwrite(str(a.out / "test_sheet.jpg"), cv2.cvtColor((sheet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return 0

    if a.render:
        fps = "362000/12097"  # source average rate; 362 frames, 1:1
        master = a.out / f"{stem}_graded_master_prores422hq.mov"
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{W}x{H}", "-r", fps,
             "-i", "-", "-i", str(a.input), "-map", "0:v", "-map", "1:a?", "-c:v", "prores_ks", "-profile:v", "3",
             "-vendor", "apl0", "-pix_fmt", "yuv422p10le", "-vf", "scale=out_color_matrix=bt709:out_range=tv",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-c:a", "copy",
             "-shortest", str(master)], stdin=subprocess.PIPE)
        rng = np.random.default_rng(1)
        for i, fr in enumerate(decode_all(a.input)):
            g, _, _ = G.grade(fr, i)
            g = g + (rng.random(g.shape[:2], dtype=np.float32)[..., None] - 0.5) / 600.0  # 10-bit dither
            enc.stdin.write((np.clip(g, 0, 1) * 65535 + .5).astype(np.uint16).tobytes())
            if i % 20 == 0:
                print("frame", i, flush=True)
        enc.stdin.close()
        enc.wait()
        (a.out / "grade_params.json").write_text(json.dumps(P, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
