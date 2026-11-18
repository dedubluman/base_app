#!/usr/bin/env python3
"""QC for the pufferfish grade: export specs, contact sheets, temporal
stability and mask spill, all measured on the RENDERED file.

    python puffer_qc.py --source inputs/pufferfish.mov \
        --graded output/pufferfish_graded.mp4 --masks masks/pufferfish__puffer \
        --out output/qc
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe


def ffjson(p):
    return json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
        check=True, capture_output=True, text=True).stdout)


def lab(x):
    return cv2.cvtColor(np.ascontiguousarray(x, np.float32), cv2.COLOR_RGB2Lab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--graded", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--times", default="0.3,3.6,7.2,10.8,14.3")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # 1) container / stream specs
    s, g = ffjson(a.source), ffjson(a.graded)
    def vs(d): return next(x for x in d["streams"] if x["codec_type"] == "video")
    def au(d): return [x for x in d["streams"] if x["codec_type"] == "audio"]
    sv, gv = vs(s), vs(g)
    specs = {k: [sv.get(k), gv.get(k)] for k in
             ("codec_name", "profile", "width", "height", "r_frame_rate", "avg_frame_rate",
              "pix_fmt", "color_space", "color_transfer", "color_primaries", "nb_frames")}
    specs["duration"] = [float(s["format"]["duration"]), float(g["format"]["duration"])]
    specs["video_bitrate_mbps"] = [round(int(sv["bit_rate"]) / 1e6, 1), round(int(gv.get("bit_rate", g["format"]["bit_rate"])) / 1e6, 1)]
    specs["audio"] = [[x["codec_name"] for x in au(s)], [x["codec_name"] for x in au(g)]]

    info_s, info_g = probe(a.source), probe(a.graded)
    n = info_s.n_frames
    z = np.load(a.masks / "alpha.npz")
    alpha = z["alpha"]
    fps = float(info_s.fps)
    want = [min(n - 1, int(round(float(t) * fps))) for t in a.times.split(",")]

    # 2) per-frame stats + stills
    stats = []
    keep_s, keep_g = {}, {}
    ring_in, ring_out = [], []
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    k40 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    for i, (xs, xg) in enumerate(zip(iter_frames(a.source, info_s, 540), iter_frames(a.graded, info_g, 540))):
        m = cv2.resize(alpha[i], (540, 960), interpolation=cv2.INTER_AREA) > 127
        ls, lg = lab(xs), lab(xg)
        bgm = ~(cv2.dilate(m.astype(np.uint8), k15) > 0)
        row = {"bg_src": ls[bgm].mean(0).tolist(), "bg_grd": lg[bgm].mean(0).tolist()}
        if m.sum() > 50:
            core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            if core.sum() > 20:
                row["fish_src"] = ls[core].mean(0).tolist()
                row["fish_grd"] = lg[core].mean(0).tolist()
            # spill: grade-induced b* shift just outside the fish vs farther out
            m8 = m.astype(np.uint8)
            near = (cv2.dilate(m8, k3) > 0) & ~m
            far = (cv2.dilate(m8, k40) > 0) & ~(cv2.dilate(m8, k15) > 0)
            d = lg - ls
            ring_in.append(d[near].mean(0))
            ring_out.append(d[far].mean(0))
        stats.append(row)
        if i in want:
            keep_s[i], keep_g[i] = xs, xg

    def series(key, c):
        return np.array([r[key][c] for r in stats if key in r])

    tem = {}
    for key in ("bg_src", "bg_grd", "fish_src", "fish_grd"):
        for c, name in enumerate("Lab"):
            v = series(key, c)
            if len(v) > 1:
                tem[f"{key}_{name}_mean"] = round(float(v.mean()), 2)
                tem[f"{key}_{name}_max_step"] = round(float(np.abs(np.diff(v)).max()), 2)
                tem[f"{key}_{name}_sd"] = round(float(v.std()), 2)
    ring_in, ring_out = np.array(ring_in), np.array(ring_out)
    spill = {"near_ring_dLab_mean": ring_in.mean(0).round(2).tolist(),
             "far_ring_dLab_mean": ring_out.mean(0).round(2).tolist(),
             "near_minus_far_b_max_abs": round(float(np.abs(ring_in[:, 2] - ring_out[:, 2]).max()), 2)}

    # 3) contact sheet: original row over graded row at the QC timestamps
    def u8(x): return cv2.cvtColor((np.clip(x, 0, 1) * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    W = 270
    top = np.concatenate([cv2.resize(u8(keep_s[i]), (W, 480)) for i in want], 1)
    bot = np.concatenate([cv2.resize(u8(keep_g[i]), (W, 480)) for i in want], 1)
    cv2.imwrite(str(a.out / "contact_sheet.jpg"), np.concatenate([top, bot], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 4) full-resolution fish crops: original | graded | graded + matte edge
    full_s = dict(zip(want, [None] * len(want)))
    full_g = dict(full_s)
    for i, x in enumerate(iter_frames(a.source, info_s)):
        if i in full_s:
            full_s[i] = x
    for i, x in enumerate(iter_frames(a.graded, info_g)):
        if i in full_g:
            full_g[i] = x
    tiles = []
    for i in want:
        al = alpha[i]
        if al.max() < 128:
            c = (960, 540)
        else:
            ys, xs = np.nonzero(al > 127)
            c = (int(ys.mean()), int(xs.mean()))
        y0 = int(np.clip(c[0] - 200, 0, 1920 - 400))
        x0 = int(np.clip(c[1] - 120, 0, 1080 - 240))
        cs_, cg = u8(full_s[i][y0:y0 + 400, x0:x0 + 240]), u8(full_g[i][y0:y0 + 400, x0:x0 + 240])
        edge = cg.copy()
        e = cv2.morphologyEx((al[y0:y0 + 400, x0:x0 + 240] > 127).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge[e] = (0, 0, 255)
        tiles.append(np.concatenate([cs_, cg, edge], 1))
    cv2.imwrite(str(a.out / "fish_crops_fullres.png"), np.concatenate(tiles, 0))

    rep = {"specs": specs, "qc_frames": want, "temporal": tem, "spill": spill,
           "frames_with_fish": int(sum(1 for r in stats if "fish_grd" in r))}
    (a.out / "qc_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
