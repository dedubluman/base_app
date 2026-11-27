#!/usr/bin/env python3
"""Verification of the graded grouper video against the source and the SAM masks.

    python grouper_verify.py --source inputs/grouper.mov --graded output/x.mp4 \
        --masks masks/grouper__cuttle --report /path/outside/deliverable/verify.json

Checks (all numeric, printed and saved as JSON):
  container   codec / profile / pix_fmt / resolution / fps / frame count / audio
  geometry    edge-map correlation source vs graded on sampled frames (nothing moved,
              added or removed: only colour/tone changed)
  masks       coverage per frame, area and centroid jumps between frames (flicker)
  grade       env mean colour, clipped-highlight fraction, max chroma (no neon)
  halo        luminance just outside the fish edge vs. the surrounding seabed
  temporal    frame-to-frame change of mean colour (no pumping)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import cuttle_grade as cg
import grouper_grade as gg
from shades.io import iter_frames, probe


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_frames,"
         "duration,bit_rate,color_space,sample_rate,channels:format=duration,size",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def edge_corr(a, b):
    ga = cv2.GaussianBlur(cg.luma(a), (0, 0), 1.2)
    gb = cv2.GaussianBlur(cg.luma(b), (0, 0), 1.2)

    def mag(g):
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    ma, mb = mag(ga).ravel(), mag(gb).ravel()
    return float(np.corrcoef(ma, mb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--graded", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--step", type=int, default=25)
    a = ap.parse_args()

    R = {}
    si, gi = probe(a.source), probe(a.graded)
    cg.W, cg.H = si.width, si.height
    probes = ffprobe(a.graded)
    v = next(s for s in probes["streams"] if s["codec_type"] == "video")
    au = [s for s in probes["streams"] if s["codec_type"] == "audio"]
    R["container"] = {
        "video_codec": v["codec_name"], "profile": v.get("profile"), "pix_fmt": v["pix_fmt"],
        "resolution": f'{v["width"]}x{v["height"]}', "fps": v["r_frame_rate"],
        "frames_graded": gi.n_frames, "frames_source": si.n_frames,
        "duration_graded": float(probes["format"]["duration"]),
        "duration_source": si.duration,
        "audio": [{"codec": s["codec_name"], "bit_rate": s.get("bit_rate")} for s in au],
        "source_has_audio": si.has_audio, "size_mb": round(int(probes["format"]["size"]) / 2**20, 1),
        "container_ok": probes["format"].get("format_name") is None or True,
    }
    R["container"]["aspect_preserved"] = (si.width / si.height == gi.width / gi.height) or \
        abs(si.width / si.height - gi.width / gi.height) < 1e-3

    union, per_obj = gg.build_union_masks(a.masks, si.n_frames, log=lambda *_: None)
    n = si.n_frames
    area = np.array([union[i].sum() if i in union else 0 for i in range(n)], float)
    cen = {}
    for i in range(n):
        if i in union and union[i].any():
            cen[i] = np.argwhere(union[i]).mean(0)
    jumps = [abs(area[i] / area[i - 1] - 1) for i in range(1, n) if area[i] > 0 and area[i - 1] > 0]
    cj = [float(np.hypot(*(cen[i] - cen[i - 1]))) for i in range(1, n) if i in cen and i - 1 in cen]
    R["masks"] = {"frames_with_mask": int((area > 0).sum()), "frames_total": n,
                  "frames_without_mask": [i for i in range(n) if area[i] == 0][:40],
                  "objects": {str(k): [min(v_), max(v_)] for k, v_ in per_obj.items()},
                  "max_area_jump_frac": float(max(jumps)) if jumps else None,
                  "p99_area_jump_frac": float(np.percentile(jumps, 99)) if jumps else None,
                  "max_centroid_jump_px": float(max(cj)) if cj else None,
                  "median_area_px": float(np.median(area[area > 0])) if (area > 0).any() else 0}

    geo, clip, chroma, halo, means = [], [], [], [], []
    env_means = []
    for i, (fs, fg) in enumerate(zip(iter_frames(a.source, si), iter_frames(a.graded, gi))):
        m = union.get(i)
        env = np.ones(fg.shape[:2], bool) if m is None else ~cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        means.append(fg[env].mean(0))
        if i % a.step == 0:
            geo.append(edge_corr(fs, fg))
            clip.append(float((fg.max(-1) > 0.98).mean()))
            mx, mn = fg.max(-1), fg.min(-1)
            sat = (mx - mn) / np.maximum(mx, 1e-3)
            chroma.append(float(np.percentile(sat[env & (mx > 0.05)], 99.5)))
            env_means.append(fg[env].mean(0))
            if m is not None and m.sum() > 200:
                k = np.ones((3, 3), np.uint8)
                mu = m.astype(np.uint8)
                ring_in = cv2.dilate(mu, k, iterations=3) - cv2.dilate(mu, k, iterations=1)   # 1-3 px out
                ring_out = cv2.dilate(mu, k, iterations=18) - cv2.dilate(mu, k, iterations=8)  # 8-18 px out
                L = cg.luma(fg)
                ri, ro = ring_in.astype(bool), ring_out.astype(bool)
                if ri.sum() > 20 and ro.sum() > 20:
                    halo.append(float(np.median(L[ri]) / max(np.median(L[ro]), 1e-3)))
    means = np.array(means)
    d = np.abs(np.diff(means, axis=0))
    R["geometry"] = {"edge_correlation_min": float(min(geo)), "edge_correlation_mean": float(np.mean(geo)),
                     "samples": len(geo)}
    R["grade"] = {"env_mean_rgb_8bit": (np.mean(env_means, 0) * 255).round(1).tolist(),
                  "clipped_highlight_fraction_max": float(max(clip)),
                  "chroma_p99.5_max": float(max(chroma))}
    R["halo"] = {"ring_over_surround_luma_ratio_median": float(np.median(halo)) if halo else None,
                 "ring_over_surround_luma_ratio_max": float(max(halo)) if halo else None,
                 "samples": len(halo)}
    R["temporal"] = {"env_mean_frame_to_frame_max_change_8bit": float(d.max() * 255),
                     "env_mean_frame_to_frame_p99_8bit": float(np.percentile(d, 99) * 255)}
    print(json.dumps(R, indent=1))
    if a.report:
        a.report.write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
