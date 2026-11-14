#!/usr/bin/env python3
"""QC for orfoz_grade.py output.

- ffprobe comparison (resolution, fps, frame count, duration, codec, audio)
- mask-overlay contact sheet and before/after contact sheet
- numbers: graded luma of the grouper vs a ring around it (the fish must be
  brighter), largest frame-to-frame jump of mean luma (flicker), mask area
  jumps and frames without a mask

    python orfoz_qc.py --src inputs/orfoz.mov --out output/orfoz_graded.mp4 \
        --masks masks/orfoz_sam__cuttle --objs 1 --work output/work --report output/orfoz_qc.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from orfoz_grade import curve_rgb, load_masks, luma

SW, SH = 308, 270


def probe(p):
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_packets,pix_fmt"
         ":format=duration", "-of", "json", str(p)], text=True))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    return dict(codec=v["codec_name"], w=int(v["width"]), h=int(v["height"]),
                fps=v["r_frame_rate"], frames=int(v["nb_read_packets"]),
                pix_fmt=v["pix_fmt"], duration=float(j["format"]["duration"]),
                audio=any(s["codec_type"] == "audio" for s in j["streams"]))


def read_small(p, w=SW, h=SH):
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(p), "-fps_mode", "passthrough",
         "-vf", f"scale=in_color_matrix=bt709:in_range=tv,scale={w}:{h}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255


def sheet(tiles, cols):
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, t in enumerate(tiles):
        out[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = t
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--objs", default="1")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)

    ps, po = probe(a.src), probe(a.out)
    checks = {
        "resolution": (ps["w"], ps["h"]) == (po["w"], po["h"]),
        "fps": ps["fps"] == po["fps"],
        "frame_count": ps["frames"] == po["frames"],
        "duration_within_0.05s": abs(ps["duration"] - po["duration"]) < 0.05,
        "h264_yuv420p": po["codec"] == "h264" and po["pix_fmt"] == "yuv420p",
        "audio_kept": po["audio"] == ps["audio"],
    }
    print("[qc] src", ps, "\n[qc] out", po, "\n[qc] checks", checks, flush=True)

    src, out = read_small(a.src), read_small(a.out)
    n = min(len(src), len(out))
    masks, have = load_masks(a.masks, [int(s) for s in a.objs.split(",")],
                             ps["frames"], ps["w"], ps["h"], log=lambda *_: None)
    ms = np.stack([cv2.resize(masks[i].astype(np.uint8), (SW, SH), interpolation=cv2.INTER_NEAREST) > 0
                   for i in range(n)])

    # fish vs ring luma, per frame
    ly = luma(out)
    k = np.ones((3, 3), np.uint8)
    ratio, fish_l, ring_l, whole_l = [], [], [], ly.reshape(n, -1).mean(1)
    for i in range(n):
        m = ms[i]
        if m.sum() < 30:
            ratio.append(np.nan); fish_l.append(np.nan); ring_l.append(np.nan)
            continue
        ring = cv2.dilate(m.astype(np.uint8), k, iterations=14).astype(bool) & ~cv2.dilate(m.astype(np.uint8), k, iterations=2).astype(bool)
        f, r = float(ly[i][m].mean()), float(ly[i][ring].mean())
        fish_l.append(f); ring_l.append(r); ratio.append(f / max(r, 1e-4))
    ratio = np.array(ratio); fish_l = np.array(fish_l)
    jump_whole = np.abs(np.diff(whole_l))
    ok = ~np.isnan(fish_l)
    jump_fish = np.abs(np.diff(fish_l[ok])) if ok.sum() > 1 else np.array([0.0])
    area = ms.reshape(n, -1).sum(1).astype(np.float64)
    nz = area > 0
    area_jump = np.abs(np.diff(area[nz])) / np.maximum(area[nz][:-1], 1) if nz.sum() > 1 else np.array([0.0])

    # contact sheets
    idx = list(range(0, n, 30)) + ([n - 1] if (n - 1) % 30 else [])
    lifted = [np.clip(curve_rgb(src[i], 3.0, 0.7, 0.8), 0, 1) for i in idx]
    ov = []
    for t, i in zip(lifted, idx):
        im = (t * 255).astype(np.uint8)[..., ::-1].copy()
        cnts, _ = cv2.findContours(ms[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 255, 255), 1)
        cv2.putText(im, str(i), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ov.append(im)
    cv2.imwrite(str(a.work / "qc_mask_overlay.jpg"), sheet(ov, 4), [cv2.IMWRITE_JPEG_QUALITY, 90])
    ba = []
    for t, i in zip(lifted, idx):
        b = (t * 255).astype(np.uint8)[..., ::-1]
        g = (np.clip(out[i], 0, 1) * 255).astype(np.uint8)[..., ::-1]
        ba.append(np.concatenate([b, g], 0))
    cv2.imwrite(str(a.work / "qc_before_after.jpg"), sheet(ba, 6), [cv2.IMWRITE_JPEG_QUALITY, 90])

    rep = dict(
        src=ps, out=po, checks=checks,
        mask_frames=int(have.sum()), mask_missing_frames=[int(i) for i in np.flatnonzero(~nz)],
        fish_over_ring_luma_median=float(np.nanmedian(ratio)) if ok.any() else None,
        fish_over_ring_luma_min=float(np.nanmin(ratio)) if ok.any() else None,
        fish_brighter_than_surround_all_frames=bool(np.all(ratio[ok] > 1.0)) if ok.any() else None,
        max_jump_mean_luma_whole=float(jump_whole.max()),
        max_jump_mean_luma_fish=float(jump_fish.max()),
        max_mask_area_jump=float(area_jump.max()),
        sheets=[str(a.work / "qc_mask_overlay.jpg"), str(a.work / "qc_before_after.jpg")],
    )
    a.report.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("src", "out")}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
