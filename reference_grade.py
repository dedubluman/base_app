#!/usr/bin/env python3
"""Render the whole clip with the reference-matched grade from fit_grade.py.

Water (outside the SAM 3.1 diver matte): light NLM denoise -> cubic colour map
-> smooth gain map G(x,y). Diver (inside the matte): quadratic colour map fitted
on the diver pixels -> same G. Mattes are the per-frame SAM 3.1 diver masks.
Optional --gold-shells paints the tracked shell gold (off by default).

    python reference_grade.py --input inputs/player_export.mov --fit grade_fit.npz \
        --diver masks/player_export__diver/obj_1 --out output/player_export_ref_graded.mp4
"""
import argparse, subprocess
from pathlib import Path
import cv2, numpy as np
from fit_grade import feats3, feats2
from underwater_cinema import probe, read_frames, load_mask, shell_grade, DEFAULT

def apply(C, f, img):
    h, w = img.shape[:2]
    return (f(img.reshape(-1, 3)) @ C).reshape(h, w, 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--diver", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--png-dir", type=Path, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--gold-shells", type=Path, default=None)
    ap.add_argument("--denoise", type=float, default=3.0)
    ap.add_argument("--detail", type=float, default=0.5)
    ap.add_argument("--grain", type=float, default=0.006)
    a = ap.parse_args()
    fit = np.load(a.fit)
    Cw, Cd, cap = fit["Cw"], fit["Cd"], float(fit["cap"])
    w, h, fps = probe(a.input)
    G = cv2.resize(fit["G"], (w, h), interpolation=cv2.INTER_CUBIC)
    enc = None
    if a.png_dir:
        a.png_dir.mkdir(parents=True, exist_ok=True)
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", fps, "-i", "-", "-i", str(a.input), "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "copy", "-shortest", str(a.out)], stdin=subprocess.PIPE)
    for i, fr in enumerate(read_frames(a.input, w, h)):
        if i < a.start or (a.end is not None and i >= a.end):
            continue
        den = fr
        if a.denoise > 0:  # light NLM, mixed back with the raw frame so sand keeps fine texture
            den = cv2.addWeighted(cv2.fastNlMeansDenoisingColored(fr, None, a.denoise, a.denoise, 5, 15), 0.6, fr, 0.4, 0)
        s_bg = den.astype(np.float32) / 255
        s_dv = cv2.GaussianBlur(fr.astype(np.float32) / 255, (0, 0), 0.6)
        # attenuate the 1-6 px band the steep colour map would otherwise blow up into blotches
        lo = cv2.GaussianBlur(s_bg, (0, 0), 6.0)
        water = apply(Cw, feats3, np.clip(lo, 0, 1)) + a.detail * (apply(Cw, feats3, np.clip(s_bg, 0, 1)) - apply(Cw, feats3, np.clip(lo, 0, 1)))
        water = np.clip(water * G, 0, 1)
        diver = np.clip(apply(Cd, feats2, np.clip(s_dv, 0, 1)) * G, 0, 1).astype(np.float32)
        lab = cv2.cvtColor(diver, cv2.COLOR_RGB2Lab)
        c = np.hypot(lab[..., 1], lab[..., 2])
        k = np.where(c > cap, (cap + (c - cap) * 0.15) / np.maximum(c, 1e-3), 1.0)
        lab[..., 1] *= k; lab[..., 2] *= k
        diver = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
        m = load_mask(a.diver, i, (h, w))
        m = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (0, 0), 1.2)[..., None]
        out = water * (1 - m) + diver * m
        if a.gold_shells is not None:
            sm = cv2.GaussianBlur(load_mask(a.gold_shells, i, (h, w)), (0, 0), 1.0)
            if sm.max() > 0:
                out = out * (1 - sm[..., None]) + shell_grade(out, sm, DEFAULT) * sm[..., None]
        out = out + np.random.default_rng(i).normal(0, a.grain, (h, w, 1)).astype(np.float32)
        u8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
        if a.png_dir:
            cv2.imwrite(str(a.png_dir / f"{i:05d}.png"), cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        else:
            enc.stdin.write(u8.tobytes())
        if i % 40 == 0:
            print("frame", i, flush=True)
    if enc:
        enc.stdin.close(); enc.wait()

if __name__ == "__main__":
    main()
