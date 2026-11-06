#!/usr/bin/env python3
"""Dark-abyss grade with SAM 3.1 bioluminescent cuttlefish.

Layers (all float32, BT.709 display-referred RGB 0..1):
  water      depth-graded tone remap of the dehazed source (midnight blue ->
             petrol -> teal), brightest under the surface, darkest mid-water
  seabed     soft texture/position layer: teal-green, more local contrast
  rays       the source's own light shafts and particles, extracted from its
             luminance high-pass and lifted (never invented)
  cuttlefish one SAM 3.1 instance mask per animal, refined against the full-
             res luma (guided filter), graded with a luminance-driven
             violet -> cobalt -> electric-cyan gradient map that keeps every
             skin pattern; per-animal levels are smoothed over time

    python cuttle_grade.py --input inputs/cuttle.mov --masks masks/cuttle__cuttle \
        --output output/cuttle_graded.mov [--frames 0,140,283 --stills DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

W, H = 1080, 1920


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def gradient_map(t, stops):
    """t: HxW in 0..1 -> HxWx3 by piecewise-linear interpolation of stops."""
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.stack([hexrgb(s[1]) for s in stops])
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, xs, cs[:, c]) for c in range(3)], -1).astype(np.float32)


# --- palettes ---------------------------------------------------------------
WATER = [(0.00, "#010307"), (0.10, "#020A12"), (0.25, "#05202C"),
         (0.45, "#0B3F4C"), (0.65, "#177684"), (0.85, "#4FB8C4"), (1.00, "#D2F6F6")]
SEABED = [(0.00, "#010508"), (0.12, "#031117"), (0.30, "#082B30"),
          (0.50, "#11535A"), (0.70, "#2C8C8C"), (1.00, "#B7EEE0")]
CUTTLE_BLUE = [(0.00, "#0C0426"), (0.18, "#2A0C78"), (0.40, "#2436D8"),
               (0.62, "#1F8CFF"), (0.82, "#4FE3FF"), (1.00, "#E4FBFF")]
CUTTLE_VIOLET = [(0.00, "#14042A"), (0.18, "#43108C"), (0.40, "#6A2AE6"),
                 (0.62, "#7A6BFF"), (0.82, "#9FB8FF"), (1.00, "#F1E8FF")]


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def gblur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def guided(I, p, r, eps):
    """He et al. guided filter (grey guide)."""
    k = (2 * r + 1, 2 * r + 1)
    mI, mp = cv2.blur(I, k), cv2.blur(p, k)
    cov = cv2.blur(I * p, k) - mI * mp
    var = cv2.blur(I * I, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.blur(a, k) * I + cv2.blur(b, k)


class Grader:
    def __init__(self, stats):
        self.lo, self.hi = stats["lo"], stats["hi"]
        y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        # Depth exposure (brightness ceiling per row): surface light at the
        # top, near-black mid-water, a dim seabed lit from above.
        self.depth = np.interp(y, [0, .08, .30, .52, .62, .80, 1],
                               [.78, .62, .36, .17, .26, .46, .48]).astype(np.float32) * np.ones((1, W), np.float32)
        self.bed_prior = np.interp(y, [0, .50, .62, 1], [0, 0, 1, 1]).astype(np.float32) * np.ones((1, W), np.float32)
        xs = np.linspace(-1, 1, W, dtype=np.float32)[None]
        ys = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        self.vign = (1 - 0.28 * np.clip(xs ** 2 * 0.9 + (ys * 0.75) ** 2, 0, 1)).astype(np.float32)

    # ---------------------------------------------------------------- water
    def background(self, rgb):
        L0 = luma(rgb)
        # 8-bit H.264 source: macroblock edges would be amplified by the
        # detail expansion, so detail is taken from a lightly smoothed luma
        L = gblur(L0, 1.3)
        base = gblur(L, 40)
        det = L - base                       # local detail (rock texture, rays)
        fine = L0 - gblur(L0, 2.5)             # particles / grain-scale detail
        n = (base - self.lo) / (self.hi - self.lo)
        n = np.clip(n, 0, 1)
        # dehaze: expand detail by 3.2x relative to the flat source
        t = np.clip(n + 1.9 * det / (self.hi - self.lo), 0, 1.2)
        # seabed layer: lower frame with real texture
        tex = gblur(np.abs(det), 25)
        tex = np.clip((tex - 0.004) / 0.012, 0, 1)
        bed = np.clip(self.bed_prior * (0.55 + 0.45 * tex), 0, 1)
        # light shafts: positive large-scale high-pass, stretched along the
        # (near vertical) ray direction so the source rays read as volume
        hp = np.clip(L - gblur(L, 18), 0, None)
        shafts = cv2.blur(hp, (5, 81)) * (1 - 0.8 * self.bed_prior)
        tone = self.depth * (t ** 1.35)
        tone = tone + 2.0 * shafts * self.depth ** 0.5
        tone = tone * self.vign
        # soft highlight knee: surface glints roll off instead of clipping
        k = 0.45
        tone = np.where(tone > k, k + (1 - k) * np.tanh((tone - k) / (1 - k) * 1.2) / 1.2, tone)
        # particles: keep visible but controlled
        tone = tone + np.clip(fine, 0, None) * 0.8 * self.depth ** 0.7
        # bed gets its own (slightly greener) map and more local contrast
        bed_tone = np.clip(tone * 1.15 + 1.3 * det * self.depth, 0, 1)
        water = gradient_map(tone, WATER)
        seabed = gradient_map(bed_tone, SEABED)
        out = water * (1 - bed[..., None]) + seabed * bed[..., None]
        # highlight bloom (surface glints only)
        hl = np.clip(luma(out) - 0.35, 0, None)
        bloom = gblur(hl, 14)[..., None] * hexrgb("#7FE0F0")
        return np.clip(out + 0.45 * bloom, 0, 1)

    # ------------------------------------------------------------ cuttlefish
    def cuttle_alpha(self, Lsrc, m):
        """Hard SAM mask (bool) -> soft, edge-aware alpha aligned to the
        full-res frame. Guided filter snaps it to real edges (thin fins keep
        their partial coverage); a 1 px erosion of the support keeps the
        feather inside the animal so no blue reaches the water."""
        mf = m.astype(np.float32)
        ys, xs = np.nonzero(m)
        if not len(ys):
            return mf
        y0, y1 = max(ys.min() - 24, 0), min(ys.max() + 25, H)
        x0, x1 = max(xs.min() - 24, 0), min(xs.max() + 25, W)
        sub = mf[y0:y1, x0:x1]
        a = guided(Lsrc[y0:y1, x0:x1], sub, 4, 1e-4)
        a = np.clip(a, 0, 1)
        support = cv2.dilate(sub, np.ones((3, 3), np.uint8))
        a = gblur(a * support, 0.7)
        out = np.zeros_like(mf)
        out[y0:y1, x0:x1] = a
        return out

    def cuttle(self, rgb, alpha, lvl):
        L = luma(rgb)
        lo, hi = lvl
        t = np.clip((L - lo) / max(hi - lo, 1e-3), 0, 1)
        # local skin detail (chromatophore pattern) kept and slightly boosted
        d = L - gblur(L, 3)
        t = np.clip(t + 2.0 * d / max(hi - lo, 1e-3), 0, 1)
        # iridescence: violet in the darker pattern and toward the fin edges,
        # electric blue/cyan on the lit mantle
        edge = 1 - np.clip(gblur(alpha, 3) * 1.4, 0, 1)
        mix = np.clip(0.55 * (1 - t) + 0.9 * edge, 0, 1)[..., None]
        col = gradient_map(t, CUTTLE_BLUE) * (1 - mix) + gradient_map(t, CUTTLE_VIOLET) * mix
        # luminous core: a gentle self-glow lift on the brightest skin
        col = col + 0.18 * (t ** 3)[..., None] * hexrgb("#7FEFFF")
        return np.clip(col, 0, 1)


def levels_for(rgb, alpha):
    L = luma(rgb)[alpha > 0.6]
    if L.size < 20:
        return None
    return np.percentile(L, 3), np.percentile(L, 99)


def smooth(series, k=15):
    """Median then mean filter of a (n,2) series with NaN gaps filled."""
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


def read_frames(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path),
                          "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb48le",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n = W * H * 3 * 2
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint16).reshape(H, W, 3).astype(np.float32) / 65535
    p.wait()


def load_masks(mdir, n):
    meta = json.loads((mdir / "meta.json").read_text())
    objs = {}
    for g in meta["objects"]:
        d = mdir / f"obj_{g}"
        objs[int(g)] = {int(p.stem): p for p in d.glob("*.png")}
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--frames", default=None, help="comma list: stills only")
    ap.add_argument("--stills", type=Path)
    ap.add_argument("--min-frames", type=int, default=10,
                    help="drop SAM objects seen on fewer frames (spurious)")
    a = ap.parse_args()

    stats_f = a.input.with_suffix(".stats.json")
    if not stats_f.exists():  # one global stretch for the whole clip: no flicker
        Ls = [luma(f[::8, ::8]) for i, f in enumerate(read_frames(a.input)) if i % 10 == 0]
        Ls = np.concatenate([gblur(l, 5).ravel() for l in Ls])
        stats_f.write_text(json.dumps({"lo": float(np.percentile(Ls, 0.5)),
                                       "hi": float(np.percentile(Ls, 99.5))}))
    G = Grader(json.loads(stats_f.read_text()))

    objs = load_masks(a.masks, 0) if a.masks else {}
    objs = {g: v for g, v in objs.items() if len(v) >= a.min_frames}
    want = set(map(int, a.frames.split(","))) if a.frames else None

    # pass 1: per-animal levels, smoothed over time
    levels = {}
    if objs:
        n = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
                                         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                                         str(a.input)]).strip())
        raw = {g: [[np.nan, np.nan]] * n for g in objs}
        for i, f in enumerate(read_frames(a.input)):
            for g, fr in objs.items():
                if i in fr:
                    m = cv2.imread(str(fr[i]), 0) > 127
                    lv = levels_for(f, m.astype(np.float32))
                    if lv:
                        raw[g][i] = list(lv)
        levels = {g: smooth(v) for g, v in raw.items()}

    enc = None
    if a.output and not want:
        audio = ["-i", str(a.input), "-map", "0:v", "-map", "1:a?", "-c:a", "copy"]
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
             "-s", f"{W}x{H}", "-framerate", "30000/1001", "-i", "-", *audio,
             "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv422p10le",
             "-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0",
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
             "-color_range", "tv", str(a.output)], stdin=subprocess.PIPE)

    rng = np.random.default_rng(0)
    for i, f in enumerate(read_frames(a.input)):
        if want and i not in want:
            continue
        out = G.background(f)
        Lsrc = luma(f)
        amax = np.zeros((H, W), np.float32)
        for g, fr in objs.items():
            if i not in fr:
                continue
            m = cv2.imread(str(fr[i]), 0) > 127
            al = G.cuttle_alpha(Lsrc, m)
            col = G.cuttle(f, al, levels[g][i])
            out = out * (1 - al[..., None]) + col * al[..., None]
            amax = np.maximum(amax, al)
        if want:
            a.stills.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(a.stills / f"g{i:05d}.png"), (out[..., ::-1] * 65535).astype(np.uint16))
            cv2.imwrite(str(a.stills / f"a{i:05d}.png"), (amax * 255).astype(np.uint8))
            print("still", i, flush=True)
        else:
            # TPDF dither at 16 bit before the 10-bit ProRes conversion
            d = (rng.random(out.shape, np.float32) - rng.random(out.shape, np.float32)) / 1023
            enc.stdin.write((np.clip(out + d, 0, 1) * 65535).astype(np.uint16).tobytes())
            if i % 20 == 0:
                print("frame", i, flush=True)
    if enc:
        enc.stdin.close()
        enc.wait()


if __name__ == "__main__":
    main()
