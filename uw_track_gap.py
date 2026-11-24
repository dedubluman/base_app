"""Fill the underwater subject-silhouette gaps that SAM 3.1 could not track.

Underwater the diver is only a dark bubble cloud. SAM 3.1 (text "person")
finds it on some frames and loses it on others, and re-prompting did not make
it stable. We keep the SAM detections as anchors and, for the key frames in
between/after, chain-warp the nearest anchor with dense optical flow between
consecutive key frames, then snap the result back to the dark silhouette
(luminance below its local surround) so the mask does not drift.

Frames that come from SAM are marked "sam", propagated ones "tracked"; the
report is saved next to the masks so the final QC can state it.

    python uw_track_gap.py --video clip.mov --first masks_uw/clip/subject.npz \
        --box masks_uw/under_box/subject.npz --out masks_uw/under_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe

RANGE = (228, 320)
STRIDE = 4


def largest(mask: np.ndarray, min_area: int = 5000) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:
        return np.zeros_like(mask, bool)
    i = 1 + int(np.argmax(st[1:, 4]))
    return (lab == i) if st[i, 4] >= min_area else np.zeros_like(mask, bool)


def flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h, w = a.shape
    s = lambda x: cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)  # noqa: E731
    f = cv2.calcOpticalFlowFarneback(s(a), s(b), None, 0.5, 5, 31, 4, 7, 1.5, 0)
    return cv2.resize(f, (w, h)) * 2.0


def warp(mask: np.ndarray, fl: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask.astype(np.float32), gx + fl[..., 0], gy + fl[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 0.5


def snap_dark(warped: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Keep the warped silhouette where it is darker than its surround."""
    if not warped.any():
        return warped
    lum = cv2.GaussianBlur(gray, (0, 0), 3)
    local = cv2.GaussianBlur(gray, (0, 0), 28)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    cand = cv2.dilate(warped.astype(np.uint8), k) > 0
    dark = lum < local - 1.5
    ref = (cand & dark)
    ref = cv2.morphologyEx(ref.astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    ref = cv2.morphologyEx(ref.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    ref = largest(ref, 3000)
    # the silhouette is solid: fill the bubble holes inside it
    cs, _ = cv2.findContours(ref.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(ref.shape, np.uint8)
    cv2.drawContours(filled, cs, -1, 1, cv2.FILLED)
    ref = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 3) > 0.5
    # never wander far from the flow-warped shape: agreement must be high
    inter = (ref & warped).sum() / max(warped.sum(), 1)
    return ref if inter > 0.55 else warped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--first", required=True, type=Path, help="full-clip SAM subject.npz")
    ap.add_argument("--box", required=True, type=Path, help="box-prompted SAM run on the underwater range")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    f1, fb = np.load(a.first), np.load(a.box)
    keys = list(range(RANGE[0], RANGE[1], STRIDE))
    h, w = f1["masks"].shape[1:]
    anchors: dict[int, np.ndarray] = {}
    for src, m in zip(f1["src_idx"], f1["masks"]):
        if int(src) in keys and (m > 127).any():
            b = largest(m > 127, 20000)          # silhouette blob only (drop the detached limb)
            if b.any():
                anchors[int(src)] = b
    for src, m in zip(fb["src_idx"], fb["masks"]):
        if int(src) in keys and (m > 127).any():
            anchors.setdefault(int(src), largest(m > 127, 20000))
    info = probe(a.video)
    gray = {}
    for t, f in enumerate(iter_frames(a.video, info, scale_w=w)):
        if t in keys:
            gray[t] = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)[:h].astype(np.float32)
        if t >= keys[-1]:
            break

    res = {k: anchors.get(k) for k in keys}
    status = {k: ("sam" if k in anchors else None) for k in keys}
    lo, hi = min(anchors), max(anchors)
    # forward from the last anchor
    cur = anchors[hi]
    for k in keys:
        if k > hi:
            cur = snap_dark(warp(cur, flow(gray[k], gray[k - STRIDE])), gray[k])
            res[k], status[k] = cur, "tracked"
    # backward from the first anchor
    cur = anchors[lo]
    for k in reversed(keys):
        if k < lo:
            cur = snap_dark(warp(cur, flow(gray[k], gray[k + STRIDE])), gray[k])
            res[k], status[k] = cur, "tracked"
    # frames before the plunge is complete: blob not yet formed -> no subject
    for k in keys:
        if k < 236:
            res[k], status[k] = None, "none (foam transition)"
        if res[k] is None:
            res[k] = np.zeros((h, w), bool)
            status[k] = status[k] or "none"
    stack = np.stack([res[k].astype(np.uint8) * 255 for k in keys])
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "subject.npz", masks=stack, src_idx=np.array(keys),
                        prob=np.full(len(keys), np.nan, np.float32))
    (a.out / "report.json").write_text(json.dumps({str(k): status[k] for k in keys}, indent=1))
    print({k: (status[k], round(float((res[k]).mean() * 100), 1)) for k in keys})


if __name__ == "__main__":
    main()
