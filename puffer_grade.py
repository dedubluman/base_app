#!/usr/bin/env python3
"""Reference-matched pufferfish grade with SAM 3.1 subject masks.

Two independent stages, both fitted ONCE for the whole clip (no per-frame
improvisation, so no hue drift or exposure pumping):

  Stage B  environment  lightness-conditional colour transfer toward the
                        reference photograph: the clip's pooled lightness
                        distribution is quantile-mapped part of the way to
                        the reference's, and chroma follows the reference's
                        a*/b* curve at each lightness (dark teal / deep blue),
                        keeping each pixel's own deviation from the source's
                        curve (texture, algae, pebbles). Then a gentle large-
                        radius local contrast. Applied to the background only.
  Stage A  pufferfish   the same idea fitted fish-to-fish: SAM 3.1 fish pixels
                        of the clip -> SAM 3.1 fish pixels of the reference.
                        Lightness is remapped monotonically (white spots stay
                        the brightest, every spot keeps its shape), chroma is
                        rebuilt from the reference fish's blue a*/b* curve.
                        Applied only through the refined fish alpha.

Matte: SAM 3.1 binary masks -> outlier repair (area / flow-warped IoU against
neighbours) -> edge-aware alpha (guided filter, choked to the SAM boundary
band, never more than 1 px outside) -> optical-flow-compensated temporal
smoothing (DIS flow, previous/next alphas warped onto the current frame).

    python puffer_grade.py --input inputs/pufferfish.mov \
        --masks masks/pufferfish__cuttle --ref inputs/pufferfish_reference.png \
        --ref-mask masks/pufferfish_reference__cuttle --output output/pufferfish_graded.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from matte import snap_diver_alpha
from shades.config import Encode
from shades.io import Encoder, iter_frames, probe

# ---------------------------------------------------------------- parameters
P = dict(
    # Stage B (environment)
    bg_L_strength=0.75,      # 0 = source lightness, 1 = reference distribution
    bg_L_floor=7.0,          # shadows never mapped below this L* (no crushing)
    bg_chroma_strength=0.92,  # how far a*/b* move onto the reference curve
    bg_resid_a=1.10,         # keep/expand the source's own chroma texture
    bg_resid_b=1.60,
    clarity=0.32,            # large-radius local contrast on L*
    clarity_sigma=18.0,
    # Stage A (fish)
    fish_L_strength=0.80,
    fish_chroma_strength=1.0,
    fish_resid=0.40,
    fish_sat=1.0,
    # matte
    temporal_w=(0.25, 0.5, 0.25),  # warped prev, current, warped next
)

BINS = np.arange(0, 101, 4.0)


def to_lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2Lab)


def to_rgb(lab):
    return np.clip(cv2.cvtColor(np.ascontiguousarray(lab, np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def quantiles(L, n=101):
    return np.percentile(L.astype(np.float64), np.linspace(0, 100, n))


def ab_curve(lab, min_n=150):
    """Median a*, b* per lightness bin, gaps interpolated, lightly smoothed."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    cs, ca, cb = [], [], []
    for lo in BINS[:-1]:
        s = (L >= lo) & (L < lo + 4)
        if s.sum() >= min_n:
            cs.append(lo + 2)
            ca.append(np.median(a[s]))
            cb.append(np.median(b[s]))
    cs, ca, cb = map(np.asarray, (cs, ca, cb))
    grid = BINS[:-1] + 2
    A, B = np.interp(grid, cs, ca), np.interp(grid, cs, cb)
    k = np.array([0.25, 0.5, 0.25])
    A = np.convolve(np.pad(A, 1, mode="edge"), k, "valid")
    B = np.convolve(np.pad(B, 1, mode="edge"), k, "valid")
    return grid, A, B


class LabTransfer:
    """Pointwise colour function: lightness quantile map + L-conditional
    chroma transfer. Same function for every frame."""

    def __init__(self, src_lab, ref_lab, L_strength, chroma_strength, ra, rb,
                 L_floor=0.0):
        self.qs, self.qr = quantiles(src_lab[:, 0]), quantiles(ref_lab[:, 0])
        self.Ls, self.Lf = L_strength, L_floor
        self.src = ab_curve(src_lab)
        self.ref = ab_curve(ref_lab)
        self.cs, self.ra, self.rb = chroma_strength, ra, rb

    def map_L(self, L):
        # extend the maps linearly past the sampled range
        qs = np.concatenate([[0.0], self.qs, [100.0]])
        qr = np.concatenate([[0.0], self.qr, [100.0]])
        qs, idx = np.unique(qs, return_index=True)
        tgt = np.interp(L, qs, qr[idx])
        out = (1 - self.Ls) * L + self.Ls * tgt
        if self.Lf:  # soft floor: keeps the deepest shadows readable
            f = self.Lf
            out = np.where(out < 2 * f, f + (out ** 2) / (4 * f) * (out < 2 * f), out)
        return out.astype(np.float32)

    def __call__(self, lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        L2 = self.map_L(L)
        g, sa, sb = self.src
        _, ta, tb = self.ref
        ra = a - np.interp(L, g, sa)
        rb = b - np.interp(L, g, sb)
        a_t = np.interp(L2, g, ta) + ra * self.ra
        b_t = np.interp(L2, g, tb) + rb * self.rb
        a2 = (1 - self.cs) * a + self.cs * a_t
        b2 = (1 - self.cs) * b + self.cs * b_t
        return np.stack([L2, a2, b2], -1).astype(np.float32)

    def describe(self):
        g, sa, sb = self.src
        _, ta, tb = self.ref
        pick = [3, 6, 9, 12, 15]
        return {"L_quantiles_src_p5_50_95": [round(float(self.qs[i]), 1) for i in (5, 50, 95)],
                "L_quantiles_ref_p5_50_95": [round(float(self.qr[i]), 1) for i in (5, 50, 95)],
                "ab_curve_src": {f"L{g[i]:.0f}": [round(float(sa[i]), 1), round(float(sb[i]), 1)] for i in pick},
                "ab_curve_ref": {f"L{g[i]:.0f}": [round(float(ta[i]), 1), round(float(tb[i]), 1)] for i in pick}}


# ---------------------------------------------------------------- masks
def load_sam_masks(mdir: Path, n: int, h: int, w: int) -> np.ndarray:
    """Union of the SAM 3.1 object tracks that are the pufferfish (tracks
    that cover a meaningful share of the clip); returns bool [n,h,w]."""
    meta = json.loads((mdir / "meta.json").read_text())
    keep = [g for g, o in meta["objects"].items() if o["frames"] >= min(5, n)]
    M = np.zeros((n, h, w), bool)
    for g in keep:
        for p in (mdir / f"obj_{g}").glob("*.png"):
            M[int(p.stem)] |= cv2.imread(str(p), 0) > 127
    return M


def dis_flow():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def warp(img, flow):
    """Pull img (frame j) onto frame i using flow i->j (half-res flow)."""
    h, w = img.shape[:2]
    fh, fw = flow.shape[:2]
    f = cv2.resize(flow, (w, h)) * np.array([w / fw, h / fh], np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def build_alpha(video: Path, info, M: np.ndarray, log=print):
    """SAM masks -> repaired -> edge-aware alpha -> flow-smoothed alpha.
    Returns (alpha uint8 [n,h,w], report dict)."""
    n, h, w = M.shape
    small = []
    for x in iter_frames(video, info, scale_w=w // 2):
        small.append(cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    dis = dis_flow()
    fwd = [dis.calc(small[i], small[i + 1], None) for i in range(n - 1)]  # i -> i+1
    bwd = [dis.calc(small[i + 1], small[i], None) for i in range(n - 1)]  # i+1 -> i
    log(f"[matte] optical flow done ({n - 1} pairs)")

    # 1) outlier repair: a frame whose SAM mask disagrees with BOTH flow-
    #    warped neighbours (and whose area jumps) is replaced by the warped
    #    previous mask. Fish present in neighbours but missing here = dropout.
    area = M.reshape(n, -1).sum(1).astype(np.float64)
    repaired = []
    for i in range(n):
        if i == 0 or i == n - 1:
            continue
        prev_w = warp(M[i - 1].astype(np.float32), bwd[i - 1]) > 0.5
        next_w = warp(M[i + 1].astype(np.float32), fwd[i]) > 0.5
        med = np.median(area[max(0, i - 5):i + 6])
        jump = med > 0 and abs(area[i] - med) / med > 0.35
        if (iou(M[i], prev_w) < 0.6 and iou(M[i], next_w) < 0.6 and jump
                and iou(prev_w, next_w) > 0.6):
            M[i] = prev_w | next_w if (prev_w & next_w).sum() > 0.8 * prev_w.sum() else prev_w
            area[i] = M[i].sum()
            repaired.append(i)
    log(f"[matte] repaired frames: {repaired}")

    # 2) edge-aware alpha per frame (guided filter on full-res luma)
    alpha = np.zeros((n, h, w), np.float32)
    for i, x in enumerate(iter_frames(video, info)):
        if M[i].any():
            alpha[i] = snap_diver_alpha(M[i], x)
    # 3) temporal: average with flow-warped neighbours (prev -> i uses flow
    #    i -> i-1 = bwd[i-1]; next -> i uses flow i -> i+1 = fwd[i]), then
    #    clamp to a 1-px dilation of this frame's SAM mask: smoothing may
    #    soften a flickering edge but never spreads colour onto the seabed.
    wp, wc, wn = P["temporal_w"]
    out = np.zeros((n, h, w), np.uint8)
    flick = []
    for i in range(n):
        acc, wsum = alpha[i] * wc, wc
        if i > 0:
            acc += warp(alpha[i - 1], bwd[i - 1]) * wp
            wsum += wp
        if i < n - 1:
            acc += warp(alpha[i + 1], fwd[i]) * wn
            wsum += wn
        a = acc / wsum
        band = cv2.dilate(M[i].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        core = cv2.erode(M[i].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        a = np.maximum(np.minimum(a, band), core * np.minimum(alpha[i], 1.0))
        out[i] = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
        if i > 0:
            pw = warp(out[i - 1].astype(np.float32), bwd[i - 1]) / 255
            cur = out[i] / 255
            u = np.maximum(pw, cur).sum()
            flick.append(float(1 - np.minimum(pw, cur).sum() / u) if u > 0 else 0.0)
    rep = {"repaired_frames": repaired,
           "frames_with_fish": int((area > 0).sum()),
           "area_px_min_med_max": [int(area[area > 0].min()) if (area > 0).any() else 0,
                                   int(np.median(area[area > 0])) if (area > 0).any() else 0,
                                   int(area.max())],
           "edge_instability_mean": round(float(np.mean(flick)), 4) if flick else 0.0,
           "edge_instability_p95": round(float(np.percentile(flick, 95)), 4) if flick else 0.0}
    return out, rep


# ---------------------------------------------------------------- fitting
def fit(video, info, M, ref_rgb, ref_mask, log=print):
    n = M.shape[0]
    sample = set(range(0, n, 8))
    bg_pix, fish_pix = [], []
    rng = np.random.default_rng(0)
    for i, x in enumerate(iter_frames(video, info)):
        if i not in sample:
            continue
        lab = to_lab(x)
        excl = cv2.dilate(M[i].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        bgl = lab[~excl]
        bg_pix.append(bgl[rng.choice(len(bgl), 40000, replace=False)])
        core = cv2.erode(M[i].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if core.sum() > 200:
            fish_pix.append(lab[core])
    src_bg, src_fish = np.concatenate(bg_pix), np.concatenate(fish_pix)
    rlab = to_lab(ref_rgb)
    rcore = cv2.erode(ref_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    rbg = ~(cv2.dilate(ref_mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    ref_bg, ref_fish = rlab[rbg], rlab[rcore]
    log(f"[fit] bg samples {len(src_bg)}, fish samples {len(src_fish)}, "
        f"ref fish px {len(ref_fish)}")
    B = LabTransfer(src_bg, ref_bg, P["bg_L_strength"], P["bg_chroma_strength"],
                    P["bg_resid_a"], P["bg_resid_b"], P["bg_L_floor"])
    A = LabTransfer(src_fish, ref_fish, P["fish_L_strength"], P["fish_chroma_strength"],
                    P["fish_resid"], P["fish_resid"])
    A.ref = ab_curve(ref_fish, min_n=40)
    A.src = ab_curve(src_fish, min_n=40)
    return A, B


def grade_frame(x, a8, A, B):
    lab = to_lab(x)
    bl = B(lab)
    if P["clarity"]:
        L = bl[..., 0]
        bl[..., 0] = L + P["clarity"] * (L - cv2.GaussianBlur(L, (0, 0), P["clarity_sigma"]))
    bg = to_rgb(bl)
    if not a8.any():
        return bg
    ys, xs = np.nonzero(a8)
    y0, y1 = max(0, ys.min() - 2), ys.max() + 3
    x0, x1 = max(0, xs.min() - 2), xs.max() + 3
    fl = A(lab[y0:y1, x0:x1])
    if P["fish_sat"] != 1.0:
        fl[..., 1:] *= P["fish_sat"]
    fish = to_rgb(fl)
    a = (a8[y0:y1, x0:x1].astype(np.float32) / 255)[..., None]
    out = bg.copy()
    out[y0:y1, x0:x1] = bg[y0:y1, x0:x1] * (1 - a) + fish * a
    return out


def write_cube(path: Path, A_or_B, size=33, title="pufferfish environment grade"):
    g = np.linspace(0, 1, size, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # red fastest in .cube
    rgb = np.stack([r, gg, b], -1).reshape(-1, 1, 3)
    out = to_rgb(A_or_B(to_lab(rgb))).reshape(-1, 3)
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        for v in out:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--ref-mask", type=Path, required=True,
                    help="SAM 3.1 mask folder of the reference photograph")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stills", type=Path, default=None,
                    help="only grade these frames (--frames) to this folder, no video")
    ap.add_argument("--frames", default="0,108,217,326,434")
    ap.add_argument("--crf", type=int, default=14)
    ap.add_argument("--set", action="append", default=[], help="PARAM=VALUE override")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=")
        P[k] = type(P[k])(float(v)) if not isinstance(P[k], tuple) else P[k]

    info = probe(a.input)
    n, h, w = info.n_frames, info.height, info.width
    M = load_sam_masks(a.masks, n, h, w)
    ref = cv2.cvtColor(cv2.imread(str(a.ref)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rm = load_sam_masks(a.ref_mask, 1, *ref.shape[:2])[0]
    alpha_cache = a.masks / "alpha.npz"
    if alpha_cache.exists() and alpha_cache.stat().st_mtime > (a.masks / "meta.json").stat().st_mtime:
        z = np.load(alpha_cache)
        alpha, mrep = z["alpha"], json.loads(str(z["rep"]))
        M = alpha > 127
    else:
        alpha, mrep = build_alpha(a.input, info, M)
        np.savez_compressed(alpha_cache, alpha=alpha, rep=json.dumps(mrep))
    print("[matte]", mrep, flush=True)
    A, B = fit(a.input, info, M, ref, rm)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    rep = {"params": {k: v for k, v in P.items()}, "matte": mrep,
           "environment_fit": B.describe(), "fish_fit": A.describe()}
    if a.stills:
        a.stills.mkdir(parents=True, exist_ok=True)
        want = [int(s) for s in a.frames.split(",")]
        for i, x in enumerate(iter_frames(a.input, info)):
            if i in want:
                y = grade_frame(x, alpha[i], A, B)
                cv2.imwrite(str(a.stills / f"src_{i:05d}.png"), cv2.cvtColor((x * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(a.stills / f"grd_{i:05d}.png"), cv2.cvtColor((y * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(a.stills / f"alpha_{i:05d}.png"), alpha[i])
        print(json.dumps(rep, indent=1))
        return 0

    enc = Encoder(a.input, a.output, info, Encode(crf=a.crf, preset="slow", profile="high"))
    for i, x in enumerate(iter_frames(a.input, info)):
        enc.write(grade_frame(x, alpha[i], A, B))
        if i % 50 == 0:
            print(f"[render] {i}/{n}", flush=True)
    enc.close()
    write_cube(a.output.with_name(a.output.stem + "_environment.cube"), B)
    a.output.with_name(a.output.stem + "_report.json").write_text(json.dumps(rep, indent=1))
    print(f"[render] wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
