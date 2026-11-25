#!/usr/bin/env python3
"""Per-frame audit of diver masks + zoomed contour sheets.

    python mask_audit.py metrics --masks masks/player_export__fwd/subject --out work/audit_raw.json
    python mask_audit.py sheet --masks DIR [--parts masks/player_export__shorts/subject ...] \
        --frames 0,8,16 --out sheet.jpg

metrics (all 437 frames): area, bbox, centroid, flow-warped IoU against the previous frame,
contour/edge alignment (mean image gradient along the mask contour relative to the frame's
p90 gradient; low = mask edge lies in flat water, i.e. spill or drift), and the mask's
bright-foam fraction (pixels that are near-white, unsaturated and highly textured)."""
import argparse, json
from pathlib import Path
import cv2, numpy as np
from shades.io import probe, iter_frames

SRC = Path("inputs/player_export.mov")
COL = {"main": (0, 255, 255), "shorts": (255, 255, 255), "fins": (255, 0, 255), "hair": (0, 140, 255), "skin": (0, 255, 0)}


def load(d, i):
    m = cv2.imread(str(Path(d) / f"{i:05d}.png"), 0)
    return None if m is None else m


def metrics(a):
    info = probe(SRC); rows = []; prev = None; prev_g = None
    for i, f in enumerate(iter_frames(SRC, info)):
        m = load(a.masks, i); mb = (m > 127) if m is not None else np.zeros(f.shape[:2], bool)
        g8 = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gs = cv2.GaussianBlur(g8, (0, 0), 1.2).astype(np.float32)
        gm = cv2.magnitude(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
        ys, xs = np.nonzero(mb)
        r = dict(i=i, area=int(mb.sum()))
        if len(ys):
            r.update(cx=float(xs.mean()), cy=float(ys.mean()), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
            cnt, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            edge = np.zeros(mb.shape, np.uint8); cv2.drawContours(edge, cnt, -1, 1, 3)
            # ignore contour pixels on the image border
            edge[:3] = 0; edge[-3:] = 0; edge[:, :3] = 0; edge[:, -3:] = 0
            r["edge_align"] = float(gm[edge > 0].mean() / (np.percentile(gm, 90) + 1e-6)) if edge.any() else None
            r["components"] = int(len(cnt))
            hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            tex = cv2.GaussianBlur(np.abs(g8.astype(np.float32) - cv2.GaussianBlur(g8, (0, 0), 2)), (0, 0), 2)
            foam = (hsv[..., 1] < 40) & (hsv[..., 2] > 215) & (tex > 6) & mb
            r["foam_frac"] = float(foam.sum() / mb.sum())
        if prev is not None and len(ys):
            fl = cv2.calcOpticalFlowFarneback(g8, prev_g, None, 0.5, 4, 25, 4, 7, 1.5, 0)  # cur->prev
            h, w = mb.shape; gx, gy = np.meshgrid(np.arange(w), np.arange(h))
            pw = cv2.remap(prev.astype(np.float32), (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32), cv2.INTER_LINEAR) > 0.5
            u = (pw | mb).sum(); r["flow_iou"] = float((pw & mb).sum() / u) if u else None
        rows.append(r); prev, prev_g = mb.copy(), g8
    json.dump(rows, open(a.out, "w"))
    area = np.array([r["area"] for r in rows], float)
    print("frames", len(rows), "empty", [r["i"] for r in rows if r["area"] < 500])
    med = np.array([np.median(area[max(0, i - 7):i + 8]) for i in range(len(area))])
    print("area outliers (<0.7x or >1.35x local median):", [i for i in range(len(area)) if not .7 * med[i] <= area[i] <= 1.35 * med[i]])
    fi = np.array([r.get("flow_iou") or np.nan for r in rows]); print("flow_iou low (<0.80):", [int(i) for i in np.flatnonzero(fi < .8)])
    ea = np.array([r.get("edge_align") or np.nan for r in rows]); print("edge_align p10/p50", np.nanpercentile(ea, [10, 50]).round(2), "low (<0.35):", [int(i) for i in np.flatnonzero(ea < .35)])
    ff = np.array([r.get("foam_frac") or 0 for r in rows]); print("foam_frac high (>0.04):", [int(i) for i in np.flatnonzero(ff > .04)])


def sheet(a):
    idx = [int(x) for x in a.frames.split(",")]
    info = probe(SRC); want = set(idx); fr = {}
    for i, f in enumerate(iter_frames(SRC, info)):
        if i in want: fr[i] = (f * 255).astype(np.uint8)[..., ::-1].copy()
        if len(fr) == len(want): break
    tiles = []
    for i in idx:
        f = fr[i]; m = load(a.masks, i)
        mb = (m > 127) if m is not None else np.zeros(f.shape[:2], bool)
        ys, xs = np.nonzero(mb)
        if len(ys):
            pad = 40; x0, x1 = max(0, xs.min() - pad), min(f.shape[1], xs.max() + pad); y0, y1 = max(0, ys.min() - pad), min(f.shape[0], ys.max() + pad)
        else: x0, y0, x1, y1 = 0, 0, f.shape[1], f.shape[0]
        # fixed aspect 4:3 crop
        cw, ch = x1 - x0, y1 - y0; s = max(cw / 4, ch / 3); cw, ch = int(4 * s), int(3 * s)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0 = int(np.clip(cx - cw // 2, 0, max(0, f.shape[1] - cw))); y0 = int(np.clip(cy - ch // 2, 0, max(0, f.shape[0] - ch)))
        layers = [("main", mb)] + [(Path(p).parent.name.split("__")[-1], (load(p, i) or np.zeros(mb.shape, np.uint8)) > 127) for p in (a.parts or [])]
        big = cv2.resize(f, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for name, lm in layers:
            lm2 = cv2.resize(lm.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cnt, _ = cv2.findContours(lm2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); cv2.drawContours(big, cnt, -1, COL.get(name, (0, 255, 0)), 1)
        crop = big[2 * y0:2 * (y0 + ch), 2 * x0:2 * (x0 + cw)]
        crop = cv2.resize(crop, (a.tile_w, a.tile_w * 3 // 4)); cv2.putText(crop, f"f{i} {i/29.97:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        tiles.append(crop)
    cols = a.cols; rows = [np.hstack(tiles[k:k + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[k:k + cols]))) for k in range(0, len(tiles), cols)]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["metrics", "sheet"]); ap.add_argument("--masks", required=True)
    ap.add_argument("--parts", nargs="*"); ap.add_argument("--frames"); ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=4); ap.add_argument("--tile-w", type=int, default=520)
    a = ap.parse_args(); (metrics if a.cmd == "metrics" else sheet)(a)
