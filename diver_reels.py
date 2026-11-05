#!/usr/bin/env python3
"""Reels-style 3-step underwater grade with a bidirectional SAM 3.1 diver mask.

Inputs (made beforehand, see prompts/sam31-diver-underwater-grade.md):
    masks/<stem>__fwd/subject/*.png   diver_masks.py --direction fwd (detector+tracker)
    masks/<stem>__bwd/subject/*.png   diver_masks_bwd.py (tracker, backwards)

    python diver_reels.py --input inputs/diver.mov --output output/diver_graded.mp4
    python diver_reels.py --input inputs/diver.mov --qc   # QC sheets only

Mask merge per frame: the forward pass is the mask; the backward pass fills
frames where it is missing or collapses (see merge_masks).
Then, per frame (matte.py): part masks checked against the frame's colours
(tracker drift, bubbles over hair/fins), water/bubbles at the diver's edge
removed, and the edge snapped to the body with a guided filter and choked
into the SAM 3.1 boundary (+1 px out at most); temporal median over 3 frames
(symmetric, so no lag).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import dataclasses

import matte as M
import reels_grade as R
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

REPO = Path(__file__).resolve().parent
# diver classes masked by diver_parts.py (all but skin)
PARTS = ("hair", "shorts", "fins", "goggles")


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
    """Forward pass (SAM 3.1 detector + tracker on every frame) is the
    primary mask. The backward pass (tracker only) proved weaker on this
    clip - it drops limbs/torso on fast motion - so it is used only to fill
    a frame where the forward mask is missing or collapses (area < 60% of
    its neighbours' median), and to report agreement."""
    fd, bd = REPO / "masks" / f"{stem}__fwd" / "subject", REPO / "masks" / f"{stem}__bwd" / "subject"
    fw = [drop_specks(read_mask(fd, i, shape)) for i in range(n)]
    areas = np.array([m.sum() for m in fw], np.float64)
    out, report = [], []
    for i in range(n):
        a = fw[i]
        b = drop_specks(read_mask(bd, i, shape))
        sa, sb = int(a.sum()), int(b.sum())
        rec = {"frame": i, "fwd": sa, "bwd": sb}
        if sa and sb:
            rec["iou"] = round(iou(a, b), 3)
        nb = [areas[j] for j in range(max(0, i - 3), min(n, i + 4)) if j != i and areas[j] > 0]
        collapsed = sa == 0 or (nb and sa < 0.6 * float(np.median(nb)))
        if collapsed and sb > sa:
            m = a | b
            rec["mode"] = "fwd+bwd fill"
        elif sa:
            m = a
            rec["mode"] = "fwd"
        else:
            m = a
            rec["mode"] = "lost"
        out.append(m.astype(np.uint8))   # hard 0/1; the matte is built per frame later
        report.append(rec)
    return out, report


def median3(seq):
    """Temporal median over 3 frames (symmetric, so no lag)."""
    n = len(seq)
    return [np.median(np.stack([seq[max(0, i - 1)], seq[i], seq[min(n - 1, i + 1)]]), 0)
            .astype(np.uint8) for i in range(n)]


def build_mattes(src: Path, info, stem: str, raw: list, log=print):
    """Per frame: the diver alpha snapped to the body edge (matte.snap_diver_alpha)
    and every SAM 3.1 part mask checked against this frame's skin/water colour
    models (matte.refine_parts). Returns uint8 alphas (diver, {part: alphas})
    and the per-frame fraction of each part mask that was given back."""
    n, shape = len(raw), raw[0].shape
    part_dirs = {name: REPO / "masks" / f"{stem}__{name}" / "subject" for name in PARTS}
    part_dirs = {k: d for k, d in part_dirs.items() if d.exists()}
    diver = []
    parts = {k: [] for k in part_dirs}
    removed = []
    k3 = np.ones((3, 3), np.uint8)
    parts["bubbles"] = []
    for i, x in enumerate(iter_frames(src, info)):
        m = raw[i] > 0
        pm = {k: read_mask(d, i, shape) & cv2.dilate(m.astype(np.uint8), k3).astype(bool)
              for k, d in part_dirs.items()}
        pm = {k: v for k, v in pm.items() if v.any()}
        # 1) part masks checked against this frame's colours (drift, bubbles)
        ref, rem = M.refine_parts(x, m, pm)
        # 2) water/bubbles hugging the diver's boundary leave the diver mask
        m, edge_rem = M.clean_diver_edge(x, m, {k: v for k, v in ref.items() if k != "bubbles"})
        rem["diver_edge"] = edge_rem
        removed.append(rem)
        # 3) the edge snapped to the body and choked into the SAM boundary
        diver.append(np.clip(M.snap_diver_alpha(m, x) * 255 + 0.5, 0, 255).astype(np.uint8))
        for k in list(part_dirs) + ["bubbles"]:
            r = ref.get(k)
            if r is None or not r.any():
                parts[k].append(np.zeros(shape, np.uint8))
                continue
            # grown by 1 px before a tight feather: at a skin/part boundary
            # the part treatment wins, so no skin warmth reaches hair/fabric
            a = cv2.GaussianBlur(cv2.dilate(r.astype(np.uint8), k3).astype(np.float32), (0, 0), 1.0)
            parts[k].append(np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8))
        if i % 100 == 0:
            log(f"[matte] frame {i}/{n}", flush=True)
    return median3(diver), {k: median3(v) for k, v in parts.items() if any(f.any() for f in v)}, removed


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

    raw, report = merge_masks(stem, n, (h, w))
    al, part_al, removed = build_mattes(a.input, info, stem, raw, log=print)
    del raw
    for r, rem in zip(report, removed):
        if rem:
            r["part_removed"] = rem
    print(f"[mask] SAM 3.1 part masks: {sorted(part_al) or 'none'} "
          f"(missing classes fall back to luminance keys)", flush=True)
    for k in list(part_al) + ["diver_edge"]:
        if k == "bubbles":
            continue
        fr = [(r["frame"], r["part_removed"][k]) for r in report
              if r.get("part_removed", {}).get(k, 0) > 0.15]
        print(f"[matte] {k}: {len(fr)} frames had >15% of the SAM mask given back "
              f"to skin/water" + (f", e.g. {fr[:6]}" if fr else ""), flush=True)
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
        for v in part_al.values():  # white balance from the skin only
            core &= v[i] < 128
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
        y = R.grade_frame(x, al[i].astype(np.float32) / 255, grad, gains[i], float(depth[i]),
                          {k: v[i].astype(np.float32) / 255 for k, v in part_al.items()
                           if v[i].any()})
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
