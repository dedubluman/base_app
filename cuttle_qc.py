#!/usr/bin/env python3
"""QC for cuttle_grade.py output.

- ffprobe comparison (resolution, fps, frame count, duration, codec)
- contact sheet: original | graded | graded + SAM 3.1 instance outlines
- temporal stability: per-frame mean luma of the background and of each
  cuttlefish in the graded video, and the largest frame-to-frame jump
- colour spill: share of strongly blue/violet pixels in a 3-12 px ring
  outside the cuttlefish masks (should stay ~ background level)

    python cuttle_qc.py --src inputs/cuttle.mov --out output/cuttle_graded_prores422hq.mov \
        --masks masks/cuttle__cuttle --exclude 5,6 --report output/cuttle_qc.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from cuttle_grade import W, H, luma, read_frames, repair_masks

COLORS = [(255, 80, 80), (80, 255, 80), (80, 160, 255), (255, 220, 60)]


def probe(p):
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,profile,width,height,r_frame_rate,nb_read_frames,pix_fmt,"
         "color_primaries,color_transfer,color_space:format=duration", "-of", "json", str(p)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = [s["codec_name"] for s in j["streams"] if s["codec_type"] == "audio"]
    return {**{k: v.get(k) for k in ("codec_name", "profile", "width", "height", "r_frame_rate",
                                     "nb_read_frames", "pix_fmt", "color_primaries",
                                     "color_transfer", "color_space")},
            "audio": a, "duration": float(j["format"]["duration"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--sheet-frames", default="0,45,95,140,190,240,268,283")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()

    ex = {int(x) for x in a.exclude.split(",") if x}
    objs = {}
    for d in sorted(a.masks.glob("obj_*")):
        g = int(d.name[4:])
        if g in ex:
            continue
        objs[g], _ = repair_masks({int(p.stem): p for p in d.glob("*.png")}, log=lambda *_: None)

    sheet_f = [int(x) for x in a.sheet_frames.split(",")]
    rows, stats = [], {"bg": [], **{g: [] for g in objs}}
    ring_blue, bg_blue = [], []
    for i, (fs, fo) in enumerate(zip(read_frames(a.src), read_frames(a.out))):
        union = np.zeros((H, W), bool)
        for g, fr in objs.items():
            m = fr.get(i)
            if m is None:
                stats[g].append(None)
                continue
            union |= m
            stats[g].append(float(luma(fo)[cv2.erode(m.astype(np.uint8), np.ones((3, 3))) > 0].mean()))
        Lo = luma(fo)
        stats["bg"].append(float(Lo[~cv2.dilate(union.astype(np.uint8), np.ones((25, 25))).astype(bool)].mean()))
        # spill: "blue/violet" = B dominant and R or B well above G
        hsv = cv2.cvtColor((fo * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[..., 0].astype(int) * 2, hsv[..., 1], hsv[..., 2]
        bv = (hue >= 215) & (hue <= 290) & (sat > 110) & (val > 40)
        u8 = union.astype(np.uint8)
        ring = cv2.dilate(u8, np.ones((25, 25))).astype(bool) & ~cv2.dilate(u8, np.ones((7, 7))).astype(bool)
        far = ~cv2.dilate(u8, np.ones((61, 61))).astype(bool)
        ring_blue.append(float(bv[ring].mean()) if ring.any() else 0.0)
        bg_blue.append(float(bv[far].mean()))
        if i in sheet_f:
            ov = (np.clip(fo, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
            for k, (g, fr) in enumerate(objs.items()):
                if i in fr:
                    cnt, _ = cv2.findContours(fr[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ov, cnt, -1, COLORS[k % 4], 2)
                    y, x = np.argwhere(fr[i]).min(0)
                    cv2.putText(ov, str(g), (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLORS[k % 4], 2)
            tiles = [(fs * 255).astype(np.uint8)[..., ::-1], (fo * 255).astype(np.uint8)[..., ::-1], ov]
            tiles = [cv2.resize(t, (270, 480), interpolation=cv2.INTER_AREA) for t in tiles]
            row = np.hstack(tiles)
            cv2.putText(row, f"f{i}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            rows.append(row)
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    if len(right) < len(left):
        right.append(np.zeros_like(left[0]))
    sheet = np.hstack([np.vstack(left), np.full((len(left) * 480, 12, 3), 255, np.uint8), np.vstack(right)])
    sheet_p = a.report.with_name(a.report.stem.replace("_qc", "") + "_contact_sheet.jpg")
    cv2.imwrite(str(sheet_p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def jumps(v):
        v = np.array([x for x in v if x is not None])
        d = np.abs(np.diff(v))
        return {"mean": round(float(v.mean()), 4), "max_step": round(float(d.max()), 4),
                "p99_step": round(float(np.percentile(d, 99)), 4)}

    rep = {"source": probe(a.src), "output": probe(a.out),
           "temporal_luma": {str(k): jumps(v) for k, v in stats.items()},
           "spill": {"blue_violet_share_ring_3_12px": round(float(np.mean(ring_blue)), 4),
                     "blue_violet_share_far_background": round(float(np.mean(bg_blue)), 4),
                     "ring_max_frame": round(float(np.max(ring_blue)), 4)},
           "contact_sheet": str(sheet_p)}
    a.report.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
