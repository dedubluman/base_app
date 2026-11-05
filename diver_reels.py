#!/usr/bin/env python3
"""Reels-style 3-step underwater grade with a bidirectional SAM 3.1 diver mask.

Inputs (made beforehand, see prompts/sam31-diver-underwater-grade.md):
    masks/<stem>__fwd/subject/*.png   diver_masks.py --direction fwd (detector+tracker)
    masks/<stem>__bwd/subject/*.png   diver_masks_bwd.py (tracker, backwards)

    python diver_reels.py --input inputs/diver.mov --output output/diver_graded.mp4
    python diver_reels.py --input inputs/diver.mov --qc   # QC sheets only

Mask merge per frame:
  both passes agree (IoU >= 0.6) -> pixels in both = 1, in only one = 0.5,
                                    so a spill in a single pass is halved
  they disagree                  -> the pass closer to the previous frame wins
  only one pass has the diver    -> that pass
Then: drop specks, erode 1-2 px, feather (scaled to the diver's size),
temporal median over 3 frames (symmetric, so no lag).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import dataclasses

import reels_grade as R
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

REPO = Path(__file__).resolve().parent


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def read_mask(d: Path, i: int, shape):
    p = d / f"{i:05d}.png"
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
    return np.zeros(shape, bool) if m is None else m > 127


def drop_specks(m: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.nonzero(areas >= 0.05 * areas.max())[0]
    return np.isin(lab, keep)


def merge_masks(stem: str, n: int, shape):
    fd, bd = REPO / "masks" / f"{stem}__fwd" / "subject", REPO / "masks" / f"{stem}__bwd" / "subject"
    out, report, prev = [], [], None
    for i in range(n):
        a, b = read_mask(fd, i, shape), read_mask(bd, i, shape)
        a, b = drop_specks(a), drop_specks(b)
        sa, sb = a.sum(), b.sum()
        rec = {"frame": i, "fwd": int(sa), "bwd": int(sb)}
        if sa and sb:
            j = iou(a, b)
            rec["iou"] = round(j, 3)
            if j >= 0.6:
                soft = 0.5 * a + 0.5 * b
                rec["mode"] = "consensus"
            else:
                pick = a if prev is None or iou(a, prev) >= iou(b, prev) else b
                soft = pick.astype(np.float32)
                rec["mode"] = "fwd" if pick is a else "bwd"
        elif sa or sb:
            soft = (a if sa else b).astype(np.float32)
            rec["mode"] = "fwd-only" if sa else "bwd-only"
        else:
            soft = np.zeros(shape, np.float32)
            rec["mode"] = "lost"
        prev = soft >= 0.5
        # refine right away and keep 8 bit: 508 float frames would be ~2.7 GB
        out.append(np.clip(refine(soft) * 255 + 0.5, 0, 255).astype(np.uint8))
        report.append(rec)
    return out, report


def refine(soft: np.ndarray) -> np.ndarray:
    hard = soft >= 0.5
    area = hard.sum()
    if area == 0:
        return np.zeros_like(soft)
    r = 1 if area < 4000 else 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(hard.astype(np.uint8), k).astype(np.float32)
    # keep the half-weight (one-pass-only) band, but only next to the core
    a = np.maximum(core, cv2.erode((soft > 0).astype(np.uint8), k) * 0.5 * (soft < 0.75))
    sigma = float(np.clip(np.sqrt(area) / 70.0, 1.2, 3.0))
    return cv2.GaussianBlur(a, (0, 0), sigma)


def alphas(stem, n, shape):
    ref, report = merge_masks(stem, n, shape)
    out = []
    for i in range(n):
        st = np.stack([ref[max(0, i - 1)], ref[i], ref[min(n - 1, i + 1)]])
        out.append(np.median(st, 0).astype(np.uint8))
    return out, report  # uint8 alpha, 255 = diver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=REPO / "output" / "diver_graded.mp4")
    ap.add_argument("--qc", action="store_true", help="QC sheets only, no video")
    a = ap.parse_args()
    info = probe(a.input)
    w, h, n = info.width, info.height, info.n_frames
    stem = a.input.stem
    qc = REPO / "output" / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    al, report = alphas(stem, n, (h, w))
    (qc / f"{stem}_mask_report.json").write_text(json.dumps(report, indent=0))
    modes = {}
    for r in report:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    ious = [r["iou"] for r in report if "iou" in r]
    print(f"[mask] modes {modes}; fwd/bwd IoU median {np.median(ious):.3f}, "
          f"min {min(ious):.3f}" if ious else f"[mask] modes {modes}", flush=True)

    # Pass 1: diver colour per frame (eroded mask core, away from the water at
    # the edges) -> white-balance gains, smoothed over time (Gaussian, sigma
    # 10 frames) so the skin tone never flickers.
    means = np.full((n, 3), np.nan, np.float32)
    for i, x in enumerate(iter_frames(a.input, info)):
        core = cv2.erode((al[i] > 230).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if core.sum() > 300:
            means[i] = np.median(x[core], 0)
    gains = np.full((n, 3), np.nan, np.float32)
    ok = ~np.isnan(means[:, 0])
    for i in np.nonzero(ok)[0]:
        gains[i] = R.diver_gains(means[i])
    depth = np.array([R.diver_depth(m) if o else np.nan for m, o in zip(means, ok)], np.float32)
    t = np.arange(n)

    def smooth(v, fallback, sigma):
        v = np.interp(t, t[ok], v[ok]) if ok.any() else np.full(n, fallback, np.float32)
        r = int(3 * sigma)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        return np.convolve(np.pad(v, r, mode="edge"), k / k.sum(), "valid").astype(np.float32)

    # fill frames without a usable diver, then smooth (WB 10 frames, depth 15)
    gains = np.stack([smooth(gains[:, c], R.DV["gains"][c], 10.0) for c in range(3)], 1)
    depth = smooth(depth, 0.0, 15.0)
    for r, g, d in zip(report, gains, depth):
        r["gains"] = [round(float(v), 3) for v in g]
        r["depth"] = round(float(d), 3)
    (qc / f"{stem}_mask_report.json").write_text(json.dumps(report, indent=0))
    print(f"[wb] diver gains R {gains[:, 0].min():.2f}-{gains[:, 0].max():.2f}, "
          f"B {gains[:, 2].min():.2f}-{gains[:, 2].max():.2f}; depth "
          f"{depth.min():.2f}-{depth.max():.2f}", flush=True)

    grad = R.vertical_gradient(h, w, R.BG["grad_top"], R.BG["grad_bottom"], R.BG["grad_gamma"])
    sheet_idx = sorted(set([0, n - 1] + list(range(0, n, int(round(2 * float(info.fps)))))))
    ov_tiles, ba_tiles = [], []
    enc = None
    if not a.qc:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        # silent delivery: the encoder copies audio only if the info says so
        enc = Encoder(a.input, a.output, dataclasses.replace(info, has_audio=False),
                      Encode(crf=16, preset="slow", profile="high", dither=True))
    lum = []
    for i, x in enumerate(iter_frames(a.input, info)):
        if a.qc and i not in sheet_idx:
            continue
        y = R.grade_frame(x, al[i].astype(np.float32) / 255, grad, gains[i], float(depth[i]))
        lum.append((i, float(R.luma(y[::4, ::4]).mean())))
        if enc is not None:
            enc.write(y)  # TPDF-dithered, BT.709 H.264
        if i in sheet_idx:
            y8 = np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)
            bgr = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1].copy()
            m = al[i] > 127
            ov = bgr.copy()
            ov[m] = (0.5 * ov[m] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, (0, 255, 255), 2)
            cv2.putText(ov, f"{i}  {report[i]['mode']}", (20, 60), 0, 1.6, (255, 255, 255), 3)
            ov_tiles.append(cv2.resize(ov, (w // 2, h // 2)))
            ba_tiles.append(cv2.resize(np.hstack([bgr, y8[..., ::-1]]), (w, h // 2)))
        if i % 50 == 0:
            print(f"[render] frame {i}/{n}", flush=True)
    if enc is not None:
        enc.close()

    def grid(tiles, cols):
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        return np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])

    cv2.imwrite(str(qc / f"{stem}_mask_overlay.jpg"), grid(ov_tiles, 3), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(qc / f"{stem}_before_after.jpg"), grid(ba_tiles, 2), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[qc] sheets in {qc}", flush=True)
    if len(lum) > 1 and not a.qc:
        jumps = np.abs(np.diff([v for _, v in lum]))
        print(f"[qc] mean luma {min(v for _, v in lum):.3f}-{max(v for _, v in lum):.3f}, "
              f"max frame-to-frame jump {jumps.max():.4f} at frame {lum[int(jumps.argmax()) + 1][0]}",
              flush=True)
    if enc is not None:
        o = probe(a.output)
        print(f"[out] {a.output}: {o.width}x{o.height} {o.fps} fps, {o.n_frames} frames "
              f"(source {n}), audio={o.has_audio}, {o.duration:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
