#!/usr/bin/env python3
"""Learn a reference-matched grade from one video frame and its graded still.

The graded reference photo is the same frame as the source video at frame N
(aligned by SAM 3.1 diver mask overlay). Water and diver are fitted separately:

  water : ridge regression, cubic polynomial of source sRGB -> reference sRGB,
          on pixels outside the dilated diver mask
  gain  : smooth multiplicative per-channel map G(x,y) = blur(ref / pred) over
          the water (vignette, top-down depth falloff), inpainted under the diver
  diver : quadratic polynomial fitted on the eroded diver mask, on ref / G

    python fit_grade.py --frame 203 --ref <graded.jpg> --out grade_fit.npz
"""
import argparse
from pathlib import Path
import cv2, numpy as np

def feats3(c):
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    return np.stack([np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b, r**3, g**3, b**3], 1)

def feats2(c):
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    return np.stack([np.ones_like(r), r, g, b, r*r, g*g, b*b, r*g, r*b, g*b], 1)

def ridge(F, Y, lam):
    A = F.T @ F + lam * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ Y)

def smooth_gain(ratio, valid, sigma):
    """normalised convolution: blur only over valid pixels, fills the diver hole."""
    num = cv2.GaussianBlur(ratio * valid[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(valid, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=Path("inputs/player_export.mov"))
    ap.add_argument("--mask", type=Path, default=Path("masks/player_export__diver/obj_1"))
    ap.add_argument("--out", type=Path, default=Path("grade_fit.npz"))
    a = ap.parse_args()
    # decode through the same ffmpeg rgb24 path the renderer uses (BT.709); the SAM jpg cache is BT.601
    from underwater_cinema import probe, read_frames
    W, H, _ = probe(a.video)
    for i, fr in enumerate(read_frames(a.video, W, H)):
        if i == a.frame:
            src = fr.astype(np.float32) / 255
            break
    ref = cv2.cvtColor(cv2.resize(cv2.imread(str(a.ref)), (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    m = cv2.imread(str(a.mask / f"{a.frame:05d}.png"), 0) > 127
    far = ~cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    core = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    # light smoothing of both so sensor/compression grain does not drive the fit
    s_ = cv2.GaussianBlur(src, (0, 0), 1.5); r_ = cv2.GaussianBlur(ref, (0, 0), 1.5)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(far.ravel()); idx = rng.choice(idx, 250000, replace=False)
    Cw = ridge(feats3(s_.reshape(-1, 3)[idx]), r_.reshape(-1, 3)[idx], 1e-3)
    pred = (feats3(s_.reshape(-1, 3)) @ Cw).reshape(H, W, 3)
    ratio = np.clip(r_ / np.maximum(pred, 0.01), 0.3, 3.0)
    G = smooth_gain(ratio, far.astype(np.float32), 90)
    # second pass: refit the colour map on gain-corrected targets (gain absorbed the vignette)
    tgt = np.clip(r_ / np.maximum(G, 0.05), 0, 1.5)
    Cw = ridge(feats3(s_.reshape(-1, 3)[idx]), tgt.reshape(-1, 3)[idx], 1e-3)
    pred = (feats3(s_.reshape(-1, 3)) @ Cw).reshape(H, W, 3) * G
    err_w = np.abs(np.clip(pred, 0, 1) - r_)[far].mean()
    idd = np.flatnonzero(core.ravel())
    Cd = ridge(feats2(s_.reshape(-1, 3)[idd]), tgt.reshape(-1, 3)[idd], 1e-3)
    pd = (feats2(s_.reshape(-1, 3)) @ Cd).reshape(H, W, 3) * G
    err_d = np.abs(np.clip(pd, 0, 1) - r_)[core].mean()
    print(f"water mean|err|={err_w:.4f}  diver mean|err|={err_d:.4f}  pixels bg={len(idx)} diver={len(idd)}")
    # gain stored low-res (smooth by construction)
    # chroma ceiling for the diver (Lab chroma of the reference skin/shorts): the quadratic map is
    # fitted on one distant frame and can over-saturate skin on brighter, closer frames
    lab = cv2.cvtColor(np.clip(r_, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    chroma = np.hypot(lab[..., 1], lab[..., 2])[core]
    cap = float(np.percentile(chroma, 97))
    print(f"diver chroma cap (Lab) = {cap:.1f}")
    Gs = cv2.resize(G, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    np.savez(a.out, Cw=Cw, Cd=Cd, G=Gs, cap=cap, size=np.array([W, H]), frame=a.frame)
    out = np.where(m[..., None], np.clip(pd, 0, 1), np.clip(pred, 0, 1))
    cv2.imwrite("/tmp/claude-0/fit_check.png", cv2.cvtColor(np.hstack([src, out, ref]) .__mul__(255).astype(np.uint8), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
