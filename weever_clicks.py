#!/usr/bin/env python3
"""Find frames where the SAM mask contains the dorsal-fin SHADOW streak (a
thin dark segment running parallel to the body, separated from it by sand)
and write SAM 3.1 refinement clicks for them:
  negative clicks on the streak, positive clicks on the body midline.

Output JSON: [[frame, x, y, label], ...] in full-frame pixels, plus the
affected frame list on stdout. Clicks are placed every --every frames inside
each affected run (SAM propagates between them).

    python weever_clicks.py --out output/clicks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent


def segments(row):
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def analyse(m):
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    L = y1 - y0
    strip_rows, body_rows = [], []
    for r in range(int(y0 + 0.25 * L), int(y1 - 0.12 * L)):
        seg = segments(m[r])
        if len(seg) < 2:
            continue
        widths = [b - a for a, b in seg]
        main = int(np.argmax(widths))
        for k, (a, b) in enumerate(seg):
            gap = seg[main][0] - b if k < main else a - seg[main][1]
            if k != main and (b - a) < 0.6 * widths[main] and gap >= 3:
                strip_rows.append((r, (a + b) / 2))
                body_rows.append((r, (seg[main][0] + seg[main][1]) / 2))
    rows = (y1 - y0) * 0.63
    return len(strip_rows) / max(rows, 1), strip_rows, body_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="weever")
    ap.add_argument("--min-frac", type=float, default=0.12)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rich", action="store_true")
    ap.add_argument("--bands", type=int, default=7)
    ap.add_argument("--src", default="fine")
    a = ap.parse_args()
    d = REPO / "masks" / f"{a.stem}__fish" / a.src
    files = sorted(d.glob("*.png"))
    info = {}
    for p in files:
        m = cv2.imread(str(p), 0) > 127
        info[int(p.stem)] = analyse(m)
    bad = sorted(f for f, (fr, *_) in info.items() if fr >= a.min_frac)
    runs = []
    for f in bad:
        if runs and f - runs[-1][1] <= 3:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    clicks = []
    for s, e in runs:
        for f in list(range(s, e + 1, a.every)) + [e]:
            if f not in info or info[f][0] < a.min_frac:
                continue
            _, sr, br = info[f]
            for q in (0.2, 0.5, 0.8):
                r, x = sr[int(q * (len(sr) - 1))]
                clicks.append([f, float(x), float(r), 0])
            if not a.rich:
                for q in (0.15, 0.5, 0.85):
                    r, x = br[int(q * (len(br) - 1))]
                    clicks.append([f, float(x), float(r), 1])
                continue
            # rich positives: whole anatomy of the streak-free mask
            m = cv2.imread(str(d / f"{f:05d}.png"), 0) > 127
            for r, x in sr:  # drop the streak segment on its rows
                seg = [sg for sg in segments(m[r]) if sg[0] <= x <= sg[1]]
                for s0, s1 in seg:
                    m[r, s0:s1] = False
            dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
            ys, xs = np.nonzero(m)
            y0, y1 = ys.min(), ys.max()
            nb = a.bands
            for b in range(nb):
                ra, rb = int(y0 + (y1 - y0) * b / nb), int(y0 + (y1 - y0) * (b + 1) / nb)
                band = dt[ra:rb]
                if band.max() < 2:
                    continue
                yy, xx = np.unravel_index(np.argmax(band), band.shape)
                clicks.append([f, float(xx), float(yy + ra), 1])
                # second positive in the band: a separate lobe (pectoral fin)
                gy, gx = np.mgrid[0:band.shape[0], 0:band.shape[1]]
                far = band * ((gx - xx) ** 2 + (gy - yy) ** 2 > 28 ** 2)
                if far.max() >= 2 and b < nb * 0.5:
                    y2, x2 = np.unravel_index(np.argmax(far), far.shape)
                    clicks.append([f, float(x2), float(y2 + ra), 1])
            # pectoral fins: outermost well-inside pixel each side, upper body
            gray = cv2.cvtColor(cv2.imread(str(REPO / ".cache" / f"frames_{a.stem}" / f"{f:05d}.jpg")),
                                cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.GaussianBlur(gray, (0, 0), 2)
            light = gray > 0.85 * np.median(gray[m])   # fins are pale, shadows are black
            up = (dt >= 3) & light & (np.arange(m.shape[0])[:, None] < y0 + 0.35 * (y1 - y0))
            uy, ux = np.nonzero(up)
            if len(ux):
                mid = np.median(ux)
                for i in (np.argmin(ux), np.argmax(ux)):
                    if abs(ux[i] - mid) > 25:
                        clicks.append([f, float(ux[i]), float(uy[i]), 1])
    a.out.write_text(json.dumps(clicks))
    print(f"affected frames: {len(bad)}; runs: {runs}; clicks: {len(clicks)}")


if __name__ == "__main__":
    main()
