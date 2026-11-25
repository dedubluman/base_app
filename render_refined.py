#!/usr/bin/env python3
"""Refined render: environment grade everywhere; skin and shorts get class-specific corrections.

    python render_refined.py --input inputs/player_export.mov --classes masks_classes/player_export \
        --output output/UNDERWATER_CINEMATIC_FINAL_REFINED.mp4

Pass 1 measures a robust per-frame median source colour of the skin and of the shorts (inside the SAM 3.1
class masks), smooths it over time, and derives per-frame white-balance gains toward the target
chromaticity (skin: warm peach-beige, shorts: neutral / slightly cool). Pass 2 grades each frame:
    out = env_LUT(src)                                           (everything, unchanged look)
    out = out*(1-w_skin-w_shorts) + skin_corr(src)*w_skin + shorts_corr(src)*w_shorts
Fins, hair, dive mask and snorkel carry no class weight, so they keep the environment grade only.
Output keeps the source frame size exactly (no padding, no scaling): H.264 High yuv420p BT.709 + faststart."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from shades.io import probe, iter_frames
import underwater_grade as ug

REPO = Path(__file__).resolve().parent
L = ug.LUMA.astype(np.float32)


def load_w(cdir, i):
    z = np.load(cdir / f"classes_{i:05d}.npz")
    return z["skin"].astype(np.float32), z["shorts"].astype(np.float32), z["skin_core"]


def robust_median(px, lo=20, hi=80):
    if len(px) < 200:
        return None
    y = px @ L
    a, b = np.percentile(y, [lo, hi])
    sel = px[(y >= a) & (y <= b)]
    return np.median(sel, 0)


def measure(src, cdir, info):
    sk, sh = [], []
    for i, f in enumerate(iter_frames(src, info)):
        ws, wh, core = load_w(cdir, i)
        sk.append(robust_median(f[core & (ws > .9)]))
        sh.append(robust_median(f[wh > .9]))
        if i % 100 == 0: print("measure", i, flush=True)
    return sk, sh


def smooth_series(vals, sigma):
    arr = np.array([v if v is not None else [np.nan] * 3 for v in vals], float)
    idx = np.arange(len(arr))
    for c in range(3):                                   # fill gaps by interpolation
        ok = ~np.isnan(arr[:, c]); arr[:, c] = np.interp(idx, idx[ok], arr[ok, c])
    return gaussian_filter1d(arr, sigma, axis=0, mode="nearest")


def chroma(rgb):
    return rgb / (rgb @ L)[..., None]


def gains_for(meas, target, strength):
    g = (np.array(target, float)[None] / chroma(meas)) ** strength
    return g / (g @ L)[:, None]                          # luma-preserving


def correct(f, g, p):
    x = np.clip(f * g, 0, 1)
    y = (f @ L)[..., None]
    # fade the gains out in deep shadow so noise/cast in darks is not tinted
    k = ug.smoothstep(y / 0.10)
    x = np.clip(f * (1 + (g - 1) * k), 0, 1)
    x = x ** p["gamma"] * p["exposure"]
    if p.get("pull"):
        # Pull each pixel's R/G and B/G toward natural skin ratios (luma preserved). Shaded skin underwater is lit only by
        # blue-green ambient light and comes out olive-gray (ashy) under one global gain; sunlit spots can overshoot to
        # orange. The pull warms the former and tames the latter; it fades out only in near-black.
        yo = (x @ L)[..., None]
        g = np.maximum(x[..., 1:2], 1e-4)
        w = p["pull"] * ug.smoothstep(yo / 0.06)
        r = x[..., 0:1] / g; b = x[..., 2:3] / g
        r = r + w * (p["rg"] - r); b = b + w * (p["bg"] - b)
        x = np.concatenate([r * g, g, b * g], -1)
        x = x * (yo / np.maximum(x @ L, 1e-4)[..., None])
    gray = (x @ L)[..., None]
    return np.clip(gray + (x - gray) * p["sat"], 0, 1).astype(np.float32)


def correct_shorts(f, g, p):
    """White fabric: any chroma is water cast, so keep only a fraction of it (fabric shading survives),
    then tint neutral-to-cool. A single WB gain would overcorrect the brighter, less-cast folds into pink."""
    x = correct(f, g, {**p, "sat": 1.0})
    gray = (x @ L)[..., None]
    tint = np.array(p["tint"], np.float32)
    return np.clip(gray * tint + (x - gray) * p["keep_chroma"], 0, 1).astype(np.float32)


class Out:
    def __init__(self, src, dst, info, crf, preset):
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{info.width}x{info.height}",
               "-r", str(info.fps), "-i", "-", "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
               "-c:v", "libx264", "-profile:v", "high", "-preset", preset, "-crf", str(crf),
               "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", str(dst)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE); self.rng = np.random.default_rng(1234)

    def write(self, x):
        n = (self.rng.random(x.shape[:2] + (1,), np.float32) - self.rng.random(x.shape[:2] + (1,), np.float32)) / 255.0
        self.p.stdin.write((np.clip(x + n, 0, 1) * 65535 + .5).astype(np.uint16).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait(): raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True); ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--params", type=Path, default=REPO / "grade_data/params_refined.json")
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--crf", type=int, default=17); ap.add_argument("--preset", default="slow")
    ap.add_argument("--stills", help="comma list of frames -> PNGs beside --output (no video)")
    ap.add_argument("--stats", type=Path, default=REPO / "grade_data/class_stats.json")
    a = ap.parse_args()
    P = json.loads(a.params.read_text()); info = probe(a.input)
    if a.stats.exists() and not P.get("remeasure"):
        S = json.loads(a.stats.read_text()); sk_m = np.array(S["skin"]); sh_m = np.array(S["shorts"])
    else:
        sk, sh = measure(a.input, a.classes, info)
        sk_m, sh_m = smooth_series(sk, P["skin"]["smooth_sigma"]), smooth_series(sh, P["shorts"]["smooth_sigma"])
        a.stats.write_text(json.dumps({"skin": sk_m.tolist(), "shorts": sh_m.tolist()}))
    g_sk = gains_for(sk_m, P["skin"]["chroma"], P["skin"]["strength"]); g_sh = gains_for(sh_m, P["shorts"]["chroma"], P["shorts"]["strength"])
    env = ug.build_env_lut(33, P["env"])
    want = {int(x) for x in a.stills.split(",")} if a.stills else None
    out = None if want else Out(a.input, a.output, info, a.crf, a.preset)
    for i, f in enumerate(iter_frames(a.input, info)):
        if want is not None and i not in want: continue
        base = ug.apply_lut(f, env); ws, wh, _ = load_w(a.classes, i)
        res = base * (1 - ws - wh)[..., None] + correct(f, g_sk[i], P["skin"]) * ws[..., None] + correct_shorts(f, g_sh[i], P["shorts"]) * wh[..., None]
        if want: cv2.imwrite(str(a.output.with_name(f"{a.output.stem}_{i:05d}.png")), cv2.cvtColor((np.clip(res, 0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            out.write(res)
            if i % 50 == 0: print("frame", i, flush=True)
        if want and i >= max(want): break
    if out: out.close()


if __name__ == "__main__":
    main()
