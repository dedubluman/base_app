#!/usr/bin/env python3
"""Numeric QC of the grade against the reference stills (and temporal-stability checks).

    python fish_qc.py stills --graded DIR --frames 0,90,... --refs REF1.jpg REF2.jpg
    python fish_qc.py video  --video output/fish_graded.mp4 --source inputs/fish.mov

`stills` : region statistics (top third / middle / lower third mean RGB, corner-to-centre luma,
           luma quantiles) and fish OKLCH hue / chroma / luma on graded stills vs the references.
`video`  : per-frame mean luma / saturation of the background, seabed and fish regions of the
           ENCODED video, high-pass energy of those series (flicker), mean abs frame-to-frame
           change inside the fish matte edge band, and an ffprobe summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from fish_color import lab_to_lch, luma, rgb_to_oklab
from fish_grade import fish_ref_lab

REPO = Path(__file__).resolve().parent


def thirds(rgb):
    h = rgb.shape[0]
    return [rgb[sl].reshape(-1, 3).mean(0) * 255 for sl in (slice(0, h // 3), slice(h // 3, 2 * h // 3), slice(2 * h // 3, h))]


def corner_centre(L):
    h, w = L.shape
    c = L[h // 3:2 * h // 3, w // 4:3 * w // 4].mean()
    e = np.mean([L[:h // 8, :w // 6].mean(), L[:h // 8, -w // 6:].mean(), L[-h // 8:, :w // 6].mean(), L[-h // 8:, -w // 6:].mean()])
    return e / c


def stats_img(rgb):
    L = luma(rgb)
    t = thirds(rgb)
    return {"mean_L": float(L.mean()), "L_pctl_1_10_50_90_99": np.percentile(L, [1, 10, 50, 90, 99]).round(3).tolist(),
            "top": t[0].round(0).tolist(), "mid": t[1].round(0).tolist(), "low": t[2].round(0).tolist(),
            "corner/centre": float(corner_centre(L))}


def fish_stats(lab, m):
    L, C, h = lab_to_lch(lab[m])
    return {"L_pctl_2_25_50_75_98": np.percentile(L, [2, 25, 50, 75, 98]).round(2).tolist(),
            "C_pctl_10_50_90": np.percentile(C, [10, 50, 90]).round(3).tolist(),
            "hue_pctl_5_25_50_75_95": np.percentile(h, [5, 25, 50, 75, 95]).round(0).tolist()}


def cmd_stills(a):
    print("== references")
    rf = []
    for p in a.refs:
        im = cv2.imread(p)[..., ::-1].astype(np.float32) / 255
        s = stats_img(im)
        lab, m = fish_ref_lab(Path(p))
        rf.append((s, fish_stats(lab, m)))
        print(f"  {Path(p).name[:14]:14s} {json.dumps(s)}")
        print(f"                 fish {json.dumps(rf[-1][1])}")
    print("== graded")
    for t in [int(x) for x in a.frames.split(",")]:
        g = cv2.imread(str(Path(a.graded) / f"g{t:05d}.png"))[..., ::-1].astype(np.float32) / 255
        s = stats_img(g)
        al = cv2.imread(str(REPO / "masks/fish__mattes/fish" / f"{t:05d}.png"), 0) > 200
        print(f"  frame {t:3d} {json.dumps(s)}")
        if al.sum() > 300:
            print(f"            fish {json.dumps(fish_stats(rgb_to_oklab(g), al))}")


def region_series(video: Path, source: Path, root: Path, every=1, pre=""):
    """Per-frame region means from the encoded video, regions from the mattes."""
    w, h = 1080, 1920
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-fps_mode", "passthrough", "-vf", pre + "scale=in_range=tv:out_range=pc,format=rgb24",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    out = []
    i = 0
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            break
        if i % every == 0:
            f = np.frombuffer(b, np.uint8).reshape(h, w, 3).astype(np.float32) / 255
            rd = lambda nm: cv2.imread(str(root / nm / f"{i:05d}.png"), 0).astype(np.float32) / 255
            L = luma(f)
            mx, mn = f.max(-1), f.min(-1)
            sat = (mx - mn) / np.maximum(mx, 1e-3)
            row = {"frame": i}
            for nm, key in (("background", "bg"), ("seabed", "sand"), ("fish", "fish")):
                wgt = rd(nm)
                if nm != "fish":
                    wgt = wgt * (rd("fish") < 0.02)
                    wgt = (wgt > 0.9).astype(np.float32)
                else:
                    wgt = (wgt > 0.9).astype(np.float32)
                s = wgt.sum()
                row[key + "_L"] = float((L * wgt).sum() / s) if s > 50 else np.nan
                row[key + "_S"] = float((sat * wgt).sum() / s) if s > 50 else np.nan
            out.append(row)
        i += 1
    p.wait()
    return out


def hf_energy(x, k=5):
    """RMS of the series minus its centred moving average (what the eye reads as flicker)."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    x = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    ma = np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, "valid")
    return float(np.sqrt(np.mean((x - ma) ** 2)))


def cmd_video(a):
    def probe(p):
        r = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                     "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames,duration,bit_rate,color_space,codec_type",
                                     "-of", "json", str(p)])
        return json.loads(r)["streams"]
    for name, p in (("source", a.source), ("graded", a.video)):
        print(f"== ffprobe {name}")
        for s in probe(p):
            print("  ", {k: v for k, v in s.items()})
    root = REPO / "masks" / f"{a.source.stem}__mattes"
    pre = f"crop={a.content_crop},scale=1080:1920:flags=lanczos," if a.content_crop else ""
    rows = region_series(a.video, a.source, root, pre=pre)
    keys = ["bg_L", "sand_L", "fish_L", "bg_S", "sand_S", "fish_S"]
    print("== region means from the encoded video (0..1)")
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        print(f"  {k:7s} mean {np.nanmean(v):.3f}  min {np.nanmin(v):.3f}  max {np.nanmax(v):.3f}  "
              f"flicker (HF rms vs 5-frame mean) {hf_energy(v):.4f}  "
              f"max |frame-to-frame| {np.nanmax(np.abs(np.diff(v))):.3f}")
    # the same series on the source for comparison (flicker the grade must not add)
    rows_s = region_series(a.source, a.source, root)
    print("== same series on the untouched source (reference for flicker)")
    for k in keys:
        v = np.array([r[k] for r in rows_s], float)
        print(f"  {k:7s} flicker {hf_energy(v):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stills")
    s.add_argument("--graded", required=True)
    s.add_argument("--frames", required=True)
    s.add_argument("--refs", nargs="+", required=True)
    v = sub.add_parser("video")
    v.add_argument("--video", type=Path, required=True)
    v.add_argument("--source", type=Path, required=True)
    v.add_argument("--content-crop", default=None, help="w:h:x:y of the picture inside a padded canvas, e.g. 608:1080:656:0")
    a = ap.parse_args()
    {"stills": cmd_stills, "video": cmd_video}[a.cmd](a)


if __name__ == "__main__":
    main()
