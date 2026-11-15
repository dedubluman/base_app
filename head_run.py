#!/usr/bin/env python3
"""Driver: seeds -> tracked seeds -> watershed head labels for f in FR; writes labels to .cache/head_labels.pkl"""
import pickle, sys
from pathlib import Path
import cv2, numpy as np
from head_split import *

REPO = Path(__file__).resolve().parent
v4 = REPO / ".cache" / "cls_v4"


def rd(c, f):
    im = cv2.imread(str(v4 / c / f"{f:05d}.png"), 0)
    return im > 127


def main(frames_sel=None):
    frames = {f: cv2.imread(str(REPO / ".cache" / "sam_frames_diver_bwd" / f"{f:05d}.jpg")) for f in range(N)}
    B = {f: rd("HAIR", f) | rd("MASK", f) for f in range(N)}
    SK = {f: rd("SKIN", f) for f in range(N)}
    # allocation region = head blob + body skin within 45 px of it
    A, near = {}, {}
    k45 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))
    for f in range(N):
        d = cv2.dilate(B[f].astype(np.uint8), k45) > 0 if B[f].any() else np.zeros((H, W), bool)
        A[f] = B[f] | (SK[f] & d)
    region = {f: cv2.dilate(A[f].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for f in range(N)}
    tr = {k: track(KEY[k], k, frames, region) for k in KEY}
    pickle.dump((tr, A, B), open(REPO / ".cache" / "head_tracks.pkl", "wb"))
    return tr, A, B, frames


if __name__ == "__main__":
    main()
