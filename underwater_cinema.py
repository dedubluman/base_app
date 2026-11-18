#!/usr/bin/env python3
"""Claustrophobic blue underwater grade with SAM 3.1 matte-driven layers.

Layers (all composited per frame, float32):
  background  deep blue, crushed blacks, heavy vignette that closes in around
              the diver (a small pocket of light), top-down depth gradient,
              bloom/halation on highlights, chromatic aberration, film grain
  diver       cast removed against own white shorts, skin pulled to bronze
              #D49A72 (keeps luminance shading), shorts/fins/hair untouched
              except for the cast removal; faint cool rim for separation
  shells      luminance-preserving gold + soft emissive glow

    python underwater_cinema.py --input inputs/player_export.mov \
        --diver masks/player_export__diver/obj_1 \
        --shells masks/player_export__shell --out output/player_export_graded.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SKIN = np.array([0xD4, 0x9A, 0x72], np.float32) / 255
GOLD = np.array([1.00, 0.82, 0.26], np.float32)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return s["width"], s["height"], f"{n}/{d}"


def read_frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def load_mask(d, i, shape):
    if d is None:
        return np.zeros(shape, np.float32)
    f = Path(d) / f"{i:05d}.png"
    if not f.exists():
        return np.zeros(shape, np.float32)
    return (cv2.imread(str(f), 0).astype(np.float32) / 255)


def load_union(dirs, i, shape):
    m = np.zeros(shape, np.float32)
    for d in dirs:
        m = np.maximum(m, load_mask(d, i, shape))
    return m


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def luma(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def s_curve(x, k):
    """filmic contrast around 0.5"""
    return np.clip(0.5 + (x - 0.5) * (1 + k) - k * 0.5 * np.sin(2 * np.pi * (x - 0.5)) / (2 * np.pi) * 2, 0, 1)


def bg_grade(img, cx, cy, P):
    """img float sRGB 0..1 -> claustrophobic deep-blue grade."""
    h, w = img.shape[:2]
    lin = srgb2lin(img)
    # 1. cast removal: water is green-teal; pull red back up, green down
    lin = lin * np.array(P["wb"], np.float32)
    # 2. exposure down (claustrophobic dimness)
    lin = lin * 2 ** P["exposure"]
    # 3. vignette: tight pocket of light around the subject
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / (0.55 * w)) ** 2 + ((yy - cy) / (0.62 * h)) ** 2)
    vig = 1 - P["vignette"] * smoothstep(0.25, 1.35, r) ** 1.3
    # 4. depth gradient: darker toward the surface (top)
    grad = 1 - P["top_dark"] * (1 - yy / h) ** 1.6
    lin = lin * (vig * grad)[..., None]
    out = lin2srgb(lin)
    # 5. contrast + crushed shadows toward navy, hue pushed teal -> blue
    out = s_curve(out, P["contrast"])
    L = luma(out)[..., None]
    shadow = (1 - smoothstep(0.0, 0.35, L)) * 1.0
    navy = np.array(P["navy"], np.float32)
    out = out * (1 - shadow * 0.55) + navy * shadow * 0.55 * (0.4 + L)
    hsv = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    # OpenCV float hue is 0..360; teal ~175-190 -> blue ~ 212
    hsv[..., 0] = hsv[..., 0] + P["hue_shift"] * smoothstep(150, 200, hsv[..., 0]) * (1 - smoothstep(235, 260, hsv[..., 0]))
    hsv[..., 1] = np.clip(hsv[..., 1] * P["sat"], 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(out, 0, 1)


def diver_grade(img, m, P):
    """Remove the cast against the diver's own white shorts, then pull skin to bronze."""
    if m.sum() < 30:
        return img, np.zeros_like(m)
    lin = srgb2lin(img)
    mb = m > 0.5
    L = luma(img)
    cand = mb & (L > np.percentile(L[mb], 80)) & (L < 0.97)
    if cand.sum() < 20:
        cand = mb & (L < 0.97)
    ref = lin[cand].mean(0)
    gain = ref.mean() / np.maximum(ref, 1e-4)
    gain = np.clip(gain, 0.6, 3.0)
    wb = np.clip(lin * gain, 0, 1)
    wb_s = lin2srgb(wb)
    Lw = luma(wb_s)
    # skin = inside matte, neither near-white (shorts) nor near-black (fins/hair/mask)
    Ls = cv2.GaussianBlur(L, (0, 0), 1.2)  # gate on the SOURCE luma: shorts >= .72, skin .4-.68
    skin_w = smoothstep(0.20, 0.34, Ls) * (1 - smoothstep(0.66, 0.74, Ls))
    # luminance-preserving recolour: keep local shading, scale bronze to it
    tgt_l = luma(SKIN[None, None, :])
    shade = (Lw / (np.median(Lw[mb & (skin_w > 0.3)]) + 1e-4) if (mb & (skin_w > 0.3)).any() else 1.0)
    bronze = np.clip(SKIN[None, None, :] * np.clip(shade, 0.35, 1.7)[..., None] * P["skin_exposure"], 0, 1)
    a = (skin_w * P["skin_strength"] * m)[..., None]
    out = wb_s * (1 - a) + bronze * a
    # non-skin parts of the diver just get the cast-free version, slightly cool
    return out, m


def shell_grade(img, m, P):
    """Gold, luminance preserving, with specular-ish lift."""
    L = luma(img)[..., None]
    Ln = np.clip((L - 0.1) / 0.8, 0, 1)
    gold = GOLD[None, None, :] * (0.35 + 1.35 * Ln)
    a = np.clip(m, 0, 1)[..., None]
    return img * (1 - a * P["shell_strength"]) + np.clip(gold, 0, 1.4) * a * P["shell_strength"]


def bloom(img, thresh, radius, amount, tint):
    hi = np.clip(img - thresh, 0, None)
    g = cv2.GaussianBlur(hi, (0, 0), radius)
    return img + g * amount * np.array(tint, np.float32)


def chroma_ab(img, px):
    h, w = img.shape[:2]
    r, g, b = cv2.split(img)
    def sh(c, s):
        M = np.float32([[1 + s, 0, -s * w / 2], [0, 1 + s, -s * h / 2]])
        return cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([sh(r, px), g, sh(b, -px)])


DEFAULT = dict(
    wb=(1.30, 0.80, 1.00), exposure=-1.25, vignette=0.92, top_dark=0.45, contrast=0.30,
    navy=(0.01, 0.05, 0.10), hue_shift=14.0, sat=0.82,
    skin_strength=0.88, skin_exposure=0.78,
    shell_strength=0.95, shell_glow=0.9,
    grain=0.018, bloom_amount=0.45, ca=0.0016,
)


def process(frame, i, a, P, state):
    h, w = frame.shape[:2]
    img = frame.astype(np.float32) / 255
    md = load_mask(a.diver, i, (h, w))
    ms = load_union(a.shell_dirs, i, (h, w))
    md = cv2.GaussianBlur(cv2.erode(md, np.ones((3, 3), np.uint8)), (0, 0), 1.4)
    ms = cv2.GaussianBlur(ms, (0, 0), 1.0)

    # light pocket follows the diver (smoothed centroid)
    if md.sum() > 30:
        ys, xs = np.nonzero(md > 0.5)
        c = np.array([xs.mean(), ys.mean()], np.float32)
        state["c"] = c if state.get("c") is None else state["c"] * 0.85 + c * 0.15
    cx, cy = state.get("c") if state.get("c") is not None else (w / 2, h / 2)

    bg = bg_grade(img, cx, cy, P)
    dv, _ = diver_grade(img, md, P)
    # match the diver to the dim world: same vignette + a little exposure, so it is not cut-out bright
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dv = dv * (0.78 + 0.22 * np.clip(1 - 0.35 * np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2), 0, 1))[..., None]
    # cool rim: thin band just outside the matte
    rim = np.clip(cv2.GaussianBlur(md, (0, 0), 5) - md, 0, 1)[..., None]
    bg = bg + rim * np.array([0.02, 0.10, 0.22], np.float32) * 0.8

    a_d = md[..., None]
    out = bg * (1 - a_d) + dv * a_d

    # shells: gold on top of whatever is below, plus emissive glow
    if ms.max() > 0:
        gold_layer = shell_grade(img, ms, P)
        a_s = np.clip(ms, 0, 1)[..., None]
        out = out * (1 - a_s) + gold_layer * a_s
        glow = cv2.GaussianBlur(ms, (0, 0), 7)[..., None] * np.array([1.0, 0.62, 0.12], np.float32)
        out = out + glow * P["shell_glow"] * 0.35

    out = bloom(out, 0.55, 14, P["bloom_amount"], (0.7, 0.9, 1.0))
    out = chroma_ab(out, P["ca"])
    rng = np.random.default_rng(i)
    Lm = luma(out)[..., None]
    out = out + rng.normal(0, P["grain"], (h, w, 1)).astype(np.float32) * (0.5 + (1 - Lm))
    return np.clip(out, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--diver", type=Path, required=True)
    ap.add_argument("--shells", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--png-dir", type=Path, default=None, help="write PNG frames instead of mp4")
    ap.add_argument("--set", nargs="*", default=[], help="PARAM=VALUE overrides")
    a = ap.parse_args()
    a.shell_dirs = a.shells
    P = dict(DEFAULT)
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = tuple(float(x) for x in v.split(",")) if "," in v else float(v)

    w, h, fps = probe(a.input)
    args = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-r", fps, "-i", "-", "-i", str(a.input), "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "copy", "-shortest", str(a.out)]
    enc = None
    if a.png_dir:
        a.png_dir.mkdir(parents=True, exist_ok=True)
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(args, stdin=subprocess.PIPE)
    state = {}
    for i, fr in enumerate(read_frames(a.input, w, h)):
        if i < a.start or (a.end is not None and i >= a.end):
            continue
        o = process(fr, i, a, P, state)
        u8 = (o * 255 + 0.5).astype(np.uint8)
        if a.png_dir:
            cv2.imwrite(str(a.png_dir / f"{i:05d}.png"), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        else:
            enc.stdin.write(u8.tobytes())
        if i % 40 == 0:
            print(f"frame {i}", flush=True)
    if enc:
        enc.stdin.close()
        enc.wait()


if __name__ == "__main__":
    main()
