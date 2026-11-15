#!/usr/bin/env python3
"""Which head parts does SAM 3.1 find with a text prompt plus a box around the head? (one 1-frame session per frame)"""
import pickle
from pathlib import Path
import cv2, numpy as np
from diver_masks3 import SamB, bbox, iou
from diver_classes import read_mask
REPO = Path(__file__).resolve().parent
cdir = REPO / "masks" / "diver__cls"
rd = lambda c, f: (cv2.imread(str(cdir / c / f"{f:05d}.png"), 0) > 127)
sam = SamB(REPO / ".cache" / "sam_frames_diver_bwd", REPO / ".cache")
FR = [10, 20, 33, 45, 55, 65, 75, 87]
TEXTS = ["hair", "face", "diving mask", "goggles", "snorkel", "head"]
out = {}
for f in FR:
    head = rd("HAIR", f) | rd("MASK", f)
    box = bbox(head, 30)
    sid = sam.start(f, f + 1, f"hd{f}")
    out[f] = {"box": box}
    for t in TEXTS:
        ids, ms = sam.text_box(sid, f, f, t, [box])
        out[f][t] = [m for m in ms]
        print(f"f{f} '{t}' box {box}: {len(ids)} inst, areas {[int(m.sum()) for m in ms]}", flush=True)
    sam.pred.handle_request({"type": "close_session", "session_id": sid})
pickle.dump(out, open(REPO / "masks" / "diver__cls" / "probe_head.pkl", "wb"))
print("done", flush=True)
