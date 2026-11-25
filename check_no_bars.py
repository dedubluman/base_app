#!/usr/bin/env python3
"""Verify a video has no black bars: for every frame, test the outermost columns/rows (first/last 6) for TV-black
(Y<=18 over >=97% of their length) and report the active-area extent over the whole clip.
    python check_no_bars.py VIDEO"""
import sys
import cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); n = 0; bad = []; W = int(cap.get(3)); H = int(cap.get(4))
ext = [W, 0, H, 0]; colmax = np.zeros(W); rowmax = np.zeros(H)
while True:
    ok, f = cap.read()
    if not ok: break
    y = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    dark = (y <= 18)
    edges = {"left": dark[:, :6].mean(), "right": dark[:, -6:].mean(), "top": dark[:6, :].mean(), "bottom": dark[-6:, :].mean()}
    if any(v >= 0.97 for v in edges.values()): bad.append((n, {k: round(float(v), 2) for k, v in edges.items() if v >= .97}))
    colmax = np.maximum(colmax, y.max(0)); rowmax = np.maximum(rowmax, y.max(1)); n += 1
cx = np.flatnonzero(colmax > 24); ry = np.flatnonzero(rowmax > 24)
print(f"{sys.argv[1]}: {W}x{H}, {n} frames; active (non-black) extent cols {cx.min()}-{cx.max()} rows {ry.min()}-{ry.max()}")
print("frames with a black edge band:", len(bad), bad[:3])
sys.exit(1 if bad else 0)
