#!/usr/bin/env python3
"""Measure the skin of the target grade (IMAGE 2) in CIE L*a*b* and save the
numbers the grade script matches: L* percentiles and a*/b* as a function of L*.
Skin = warm pixels (R > B, a*,b* > 0) of the diver; the reference water is blue
so it never passes the test.    python ref_skin_stats.py reference/image2_target.png
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

im = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "reference/image2_target.png")
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
h = L.shape[0]
skin = (rgb[..., 0] > rgb[..., 2] + 0.03) & (L > 6) & (a > 2) & (b > 4)
skin[: int(0.12 * h)] = False
n, lab_cc, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8))
skin = np.isin(lab_cc, [k for k in range(1, n) if st[k, 4] > 400])
pct = [1, 5, 25, 50, 75, 95, 99]
out = {"L_pct": dict(zip(map(str, pct), np.percentile(L[skin], pct).round(2).tolist())), "bands": []}
edges = [0, 15, 25, 35, 50, 65, 78, 100]
for lo, hi in zip(edges[:-1], edges[1:]):
    s = skin & (L >= lo) & (L < hi)
    if s.sum() > 80:
        out["bands"].append({"L": float(L[s].mean()), "a": float(a[s].mean()), "b": float(b[s].mean()), "n": int(s.sum())})
Path("skin_reference.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
