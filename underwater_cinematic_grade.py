#!/usr/bin/env python3
"""Deep-navy cinematic underwater grade with a SAM 3.1 diver mask.

Needs masks from `diver_masks.py` (masks/<stem>__fwd/subject/*.png).
    python underwater_cinematic_grade.py --input inputs/diver.mov \
        --output output/diver_cinematic.mp4 [--preview 0,118,235]

Pure colour processing in float32 (no generative content): the water is
re-mapped to a navy gradient driven by the footage's own luminance (so ripples,
seabed and particles stay), a vignette darkens corners, and the SAM mask
drives a separate warm skin/subject grade. Settings live in SETTINGS and are
dumped next to the output as JSON.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent

SETTINGS = dict(
    # --- environment -------------------------------------------------
    water_black=0.030,        # input luminance mapped to 0 (after WB)
    water_white=0.80,
    water_gamma=2.15,         # >1 sinks the water into deep navy
    water_detail_keep=0.45,   # share of original chroma kept (0..1)
    ramp=[(0.00, (1, 4, 12)), (0.15, (3, 16, 38)), (0.40, (8, 44, 82)),
          (0.70, (40, 100, 150)), (1.00, (140, 195, 225))],  # navy gradient map (RGB 0-255)
    vignette_strength=0.80,   # corner darkening
    vignette_power=2.1,
    bottom_dark=0.45,         # extra darkening toward the lower frame
    surface_boost=0.22,       # cool highlight lift on the bright surface
    # --- subject -----------------------------------------------------
    skin_wb=(1.85, 1.00, 0.72),   # recover red lost to water absorption
    skin_exposure=0.92,
    skin_contrast=1.12,
    skin_warm=0.30,              # push toward golden bronze
    skin_sat=1.0,
    skin_hi_gold=0.55,
    subject_rim=0.10,            # slight lift for separation
    mask_feather=1.2,
    # --- temporal ----------------------------------------------------
    exposure_ema=0.85,
)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def gradient_map(l, ramp):
    xs = np.array([r[0] for r in ramp], np.float32)
    out = np.empty(l.shape + (3,), np.float32)
    for c in range(3):
        ys = np.array([r[1][c] for r in ramp], np.float32) / 255.0
        out[..., c] = np.interp(l, xs, ys)
    return out


def vignette(h, w, s):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # slightly off-centre (subject sits centre-right/upper)
    cx, cy = 0.52 * w, 0.46 * h
    d = np.sqrt(((xx - cx) / (0.62 * w)) ** 2 + ((yy - cy) / (0.64 * h)) ** 2)
    v = 1.0 - s * np.clip(d, 0, 1.6) ** s_pow
    v = np.clip(v, 0.04, 1.0)
    bot = 1.0 - s_bot * smoothstep(0.35, 1.0, yy / h)
    return (v * bot).astype(np.float32)


s_pow = SETTINGS["vignette_power"]
s_bot = SETTINGS["bottom_dark"]


def grade_water(img, S, gain, vig):
    # neutralise the turquoise cast mildly so luminance reflects light, not tint
    l = lum(img)
    l = np.clip((l * gain - S["water_black"]) / (S["water_white"] - S["water_black"]), 0, 1)
    # local contrast: unsharp on luminance keeps ripples crisp
    blur = cv2.GaussianBlur(l, (0, 0), 14)
    l = np.clip(l + 0.35 * (l - blur), 0, 1)
    l = l ** S["water_gamma"]
    base = gradient_map(l, S["ramp"])
    # keep a little of the original colour variation (blue-ish only)
    orig = img / np.maximum(lum(img)[..., None], 1e-3)
    orig = np.clip(orig, 0.2, 3.0)
    tint = np.array([0.55, 0.95, 1.55], np.float32)  # cool target chroma
    chroma = (1 - S["water_detail_keep"]) * tint + S["water_detail_keep"] * orig * np.array([0.5, 0.9, 1.3], np.float32)
    # re-colour: scale base so luminance stays, bias chroma subtly
    out = base * (0.6 + 0.4 * chroma / chroma.mean(-1, keepdims=True))
    # cool surface highlights
    hi = smoothstep(0.55, 0.95, l)[..., None]
    out = out + S["surface_boost"] * hi * np.array([0.10, 0.22, 0.30], np.float32)
    return np.clip(out * vig[..., None], 0, 1)


def grade_subject(img, S, gain, vig, top=None):
    wb = np.array(S["skin_wb"], np.float32)
    x = np.clip(img * wb * gain * S["skin_exposure"], 0, 1.5)
    # base tone: contrast around mid-grey, soft highlight roll-off
    bl = cv2.GaussianBlur(x, (0, 0), 6)
    x = x + 0.45 * (x - bl)
    x = 0.45 + (x - 0.45) * S["skin_contrast"]
    x = np.clip(x, 0, 1)
    x = x / (1 + 0.35 * x)  # gentle shoulder
    l = lum(x)
    # skin weight: warm, mid-bright, not near white (shorts) or black (fins)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    warm = smoothstep(0.0, 0.10, r - b)
    midw = smoothstep(0.06, 0.20, l) * (1 - smoothstep(0.78, 0.97, l))
    skin = warm * midw
    # golden-bronze target: shift hue toward amber while holding luminance
    gold = np.stack([l * 1.18, l * 0.93, l * 0.58], -1)
    gold = gold / np.maximum(lum(gold)[..., None], 1e-3) * l[..., None]
    x = x * (1 - S["skin_warm"] * skin[..., None]) + gold * S["skin_warm"] * skin[..., None]
    # golden highlights
    hi = smoothstep(0.30, 0.8, l) * skin
    if top is not None:
        hi = hi * top
        x = x * (1 + 0.18 * top[..., None] * skin[..., None])
    x = x + S["skin_hi_gold"] * hi[..., None] * np.array([0.10, 0.045, -0.02], np.float32)
    # saturation
    l2 = lum(x)[..., None]
    x = l2 + (x - l2) * S["skin_sat"]
    # shorts: pull a hint of cool back so they read white-blue, not beige
    near_white = smoothstep(0.62, 0.9, l) * (1 - skin)
    x = x * (1 - 0.5 * near_white[..., None]) + (lum(x)[..., None] * np.array([0.93, 0.98, 1.06], np.float32)) * 0.5 * near_white[..., None]
    # vignette shared but weaker on the subject
    return np.clip(x * (0.55 + 0.45 * vig[..., None]) + S["subject_rim"] * 0.0, 0, 1)


def merge_masks(mdir: Path, n, shape):
    ms = []
    for i in range(n):
        m = cv2.imread(str(mdir / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        ms.append(np.zeros(shape, np.uint8) if m is None else (m > 127).astype(np.uint8))
    return ms


def probe(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"], int(s["nb_frames"])


def read_frames(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    sz = w * h * 3
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def process(frames, masks, S, w, h):
    vig = vignette(h, w, S["vignette_strength"])
    ema, out = None, []
    for i, fr in enumerate(frames):
        img = fr.astype(np.float32) / 255.0
        m = (masks[i - 1] | masks[i] | masks[min(i + 1, len(masks) - 1)]) if False else masks[i]
        # temporal exposure normalisation (EMA of median luminance)
        med = float(np.median(lum(img)))
        ema = med if ema is None else S["exposure_ema"] * ema + (1 - S["exposure_ema"]) * med
        gain = float(np.clip(0.30 / max(ema, 1e-3), 0.7, 1.4)) if False else 1.0
        sm = cv2.GaussianBlur(m.astype(np.float32), (0, 0), S["mask_feather"])[..., None]
        bg = grade_water(img, S, gain, vig)
        ys = np.nonzero(m.any(1))[0]
        top = None
        if len(ys):
            yy = (np.arange(h, dtype=np.float32) - ys[0]) / max(ys[-1] - ys[0], 1)
            top = (1 - smoothstep(0.0, 0.6, yy))[:, None] * np.ones((1, w), np.float32)
        sub = grade_subject(img, S, gain, vig, top)
        # selfie stick: dark, neutral, outside the diver mask, touching the lower frame -> keep original look
        mx, mn = img.max(-1), img.min(-1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        dark = ((sat < 0.42) & (m == 0) & (np.arange(h)[:, None] > 0.30 * h)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        nl, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
        stick = np.zeros((h, w), np.float32)
        for k in range(1, nl):
            if st[k, cv2.CC_STAT_AREA] > 4000 and st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] >= h - 2:
                cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pts = np.vstack(cs)
                cv2.fillConvexPoly(stick, cv2.convexHull(pts), 1.0)
        stick = cv2.GaussianBlur(cv2.dilate(stick, np.ones((3, 3), np.uint8)), (0, 0), 1.0)[..., None]
        l0 = lum(img)[..., None]
        st_col = np.clip((l0 + (img - l0) * 0.25) * 0.85 * (0.5 + 0.5 * vig[..., None]), 0, 1)
        bg = bg * (1 - stick) + st_col * stick
        yield np.clip((bg * (1 - sm) + sub * sm) * 255 + 0.5, 0, 255).astype(np.uint8), sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--preview", default="")
    ap.add_argument("--set", action="append", default=[], help="KEY=JSON value")
    a = ap.parse_args()
    S = dict(SETTINGS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        S[k] = json.loads(v)
    global s_pow, s_bot
    s_pow, s_bot = S["vignette_power"], S["bottom_dark"]
    w, h, fps, n = probe(a.input)
    mdir = a.masks or REPO / "masks" / f"{a.input.stem}__fwd" / "subject"
    masks = merge_masks(mdir, n, (h, w))
    frames = list(read_frames(a.input, w, h)) if a.preview else read_frames(a.input, w, h)
    if a.preview:
        idx = [int(x) for x in a.preview.split(",")]
        sel = {i: frames[i] for i in idx}
        rows = []
        for i in idx:
            g, sm = next(process([sel[i]], [masks[i]], S, w, h))
            rows.append(np.hstack([sel[i], g]))
        cv2.imwrite(str(REPO / "output" / "preview.jpg"),
                    cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_suffix(".video.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", "12",
                            "-preset", "slow", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    for g, _ in process(frames, masks, S, w, h):
        enc.stdin.write(g.tobytes())
    enc.stdin.close(); enc.wait()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(a.input), "-map", "0:v",
                    "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", "-shortest", str(a.output)], check=True)
    tmp.unlink()
    a.output.with_suffix(".settings.json").write_text(json.dumps(S, indent=2))


if __name__ == "__main__":
    main()
