#!/usr/bin/env python3
"""SAM 3.1 binary masks -> soft, temporally stable alpha mattes.

Per frame (ROI around the fish only):
  1. hygiene: keep the connected component(s) that belong to the fish, fill
     interior holes (no unmasked patches on the body).
  2. edge-aware matte: a colour guided filter (He et al.) on the SAM mask with
     the frame as guide. Alpha is pinned to 1 in the eroded core and to 0
     outside a narrow band around the SAM edge, so the filter can only move
     the boundary by a few pixels (fin rays / translucent fin edge) and can
     never pull in sand or the fish's shadow.
  3. temporal: alpha of t-1 and t+1 is warped into t with DIS optical flow
     and blended inside the edge band only (supporting role: SAM decides the
     shape, flow only removes 1-2 px edge shimmer, no ghosting because the
     neighbours are motion-compensated).

Also flags SAM area outliers (QC) into matte_qc.json.

    python weever_matte.py --input inputs/weever.mov
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent


def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter_color(I, p, r, eps):
    """He et al. colour guided filter. I: HxWx3 float32 0..1, p: HxW float32."""
    mI = box(I, r)
    mp = box(p, r)
    mIp = box(I * p[..., None], r)
    cov_Ip = mIp - mI * mp[..., None]
    # covariance of I (3x3 per pixel)
    var = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = box(I[..., i] * I[..., j], r) - mI[..., i] * mI[..., j]
            var[..., i, j] = v
            var[..., j, i] = v
    var += eps * np.eye(3, dtype=np.float32)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mp - (a * mI).sum(-1)
    return (box(a, r) * I).sum(-1) + box(b, r)


def clean(m: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        keep = np.zeros(n, bool)
        keep[big] = True
        for k in range(1, n):  # keep fragments that are fins of the same fish
            a = st[k, cv2.CC_STAT_AREA]
            if a > 0.02 * st[big, cv2.CC_STAT_AREA]:
                comp = lab == k
                near = cv2.dilate((lab == big).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
                if (comp & near).any() or (prev is not None and (comp & prev).sum() > 0.5 * a):
                    keep[k] = True
        m = keep[lab].astype(np.uint8)
    # fill interior holes
    ff = np.pad(m, 1)
    inv = (ff == 0).astype(np.uint8)
    cv2.floodFill(inv, None, (0, 0), 0)
    m = m | inv[1:-1, 1:-1]
    return m.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--eps", type=float, default=2e-4)
    ap.add_argument("--band-in", type=int, default=3, help="px the edge may move inward")
    ap.add_argument("--band-out", type=int, default=4, help="px the edge may move outward")
    ap.add_argument("--temporal", type=float, default=0.5, help="weight of warped neighbours")
    a = ap.parse_args()

    stem = a.input.stem
    mdir = REPO / "masks" / f"{stem}__fish"
    fdir = REPO / ".cache" / f"frames_{stem}"
    odir = mdir / "alpha"
    odir.mkdir(exist_ok=True)
    files = sorted((mdir / "fine").glob("*.png"))
    n = len(list(fdir.glob("*.jpg")))
    H, W = cv2.imread(str(fdir / "00000.jpg")).shape[:2]
    have = {int(p.stem) for p in files}
    print(f"[matte] {len(have)}/{n} SAM frames", flush=True)

    raw = {}
    for f in range(n):
        if f in have:
            raw[f] = cv2.imread(str(mdir / "fine" / f"{f:05d}.png"), 0) > 127

    # --- QC: area outliers vs local median ---
    areas = np.array([raw[f].sum() if f in raw else 0 for f in range(n)], float)
    qc = {"missing": [f for f in range(n) if f not in raw], "outliers": []}
    for f in range(n):
        nb = [areas[g] for g in range(max(0, f - 6), min(n, f + 7)) if g != f and areas[g] > 0]
        med = np.median(nb) if nb else 0
        if areas[f] == 0 or (med and not (0.8 < areas[f] / med < 1.25)):
            qc["outliers"].append({"frame": f, "area": int(areas[f]), "local_median": float(med)})
    print(f"[matte] missing {len(qc['missing'])}, area outliers {len(qc['outliers'])}: "
          f"{[o['frame'] for o in qc['outliers']][:40]}", flush=True)

    # --- per frame clean + guided matte (soft alpha, full frame float) ---
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    alphas = {}
    grays = {}
    prev = None
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.band_in + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.band_out + 1,) * 2)
    bands = {}
    for f in range(n):
        img = cv2.imread(str(fdir / f"{f:05d}.jpg"))
        grays[f] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if f not in raw:
            continue
        m = clean(raw[f], prev)
        prev = m
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 40), min(H, ys.max() + 41)
        x0, x1 = max(0, xs.min() - 40), min(W, xs.max() + 41)
        I = img[y0:y1, x0:x1].astype(np.float32) / 255
        mm = m[y0:y1, x0:x1].astype(np.uint8)
        core = cv2.erode(mm, k_in) > 0
        outer = cv2.dilate(mm, k_out) > 0
        p = cv2.GaussianBlur(mm.astype(np.float32), (0, 0), 1.0)
        q = guided_filter_color(I, p, a.r, a.eps)
        q = np.clip(q, 0, 1)
        q[core] = 1
        q[~outer] = 0
        al = np.zeros((H, W), np.float32)
        al[y0:y1, x0:x1] = q
        alphas[f] = al
        band = np.zeros((H, W), bool)
        band[y0:y1, x0:x1] = outer & ~core
        bands[f] = band

    # --- temporal: flow-warped neighbour alphas, edge band only ---
    def warp_to(f, g):
        """alpha of frame g warped into frame f's geometry."""
        flow = dis.calc(grays[f], grays[g], None)  # f -> g
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        return cv2.remap(alphas[g], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # frames SAM did not return (should be none): flow-warp the nearest SAM matte
    filled = []
    for f in range(n):
        if f in alphas:
            continue
        cand = [g for g in alphas if g not in filled]
        if not cand:
            continue
        g = min(cand, key=lambda g: abs(g - f))
        alphas[f] = warp_to(f, g)
        bands[f] = bands[g]
        filled.append(f)
    qc["flow_filled"] = filled
    if filled:
        print(f"[matte] flow-filled frames without SAM output: {filled}", flush=True)

    stats = []
    for f in range(n):
        if f not in alphas:
            continue
        al = alphas[f]
        acc, wsum = al.copy(), np.ones_like(al)
        for g in (f - 1, f + 1):
            if g in alphas:
                wa = warp_to(f, g)
                w = np.where(bands[f], a.temporal, 0).astype(np.float32)
                acc += w * wa
                wsum += w
        out = acc / wsum
        diff = float(np.abs(out - al)[bands[f]].mean()) if bands[f].any() else 0
        stats.append(diff)
        cv2.imwrite(str(odir / f"{f:05d}.png"), np.round(out * 65535).astype(np.uint16))
        if f % 50 == 0:
            print(f"[matte] {f}/{n} band change {diff:.3f}", flush=True)
    qc["mean_temporal_band_change"] = float(np.mean(stats))
    (mdir / "matte_qc.json").write_text(json.dumps(qc, indent=1))
    print(f"[matte] done, {len(alphas)} alphas", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
