#!/usr/bin/env python3
"""Split-sea grade: a central rock ridge divides two differently coloured seas.

SAM 3.1 (local checkpoint, CPU port when no GPU) masks the rock. The water on
each side of the rock's per-frame centre line becomes the LEFT and RIGHT sea.
Each zone gets its own grade (left: turquoise glass, rock: warm amber/copper,
right: deep ink-navy with haze) and the three are blended with feathered,
temporally smoothed masks that sum to 1.

    python split_sea.py --input clip.mov --masks-only     # SAM 3.1 only
    python split_sea.py --input clip.mov --stills         # QC stills
    python split_sea.py --input clip.mov --output out.mov # full render

See prompts/sam31-split-sea-grade.md for the creative brief.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.io import probe

REPO = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------- masking --

def make_proxy(src: Path, dst: Path, stride: int) -> None:
    """Every `stride`-th frame of the source as a near-lossless proxy clip."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
         "-vf", f"select='not(mod(n\\,{stride}))',setpts=N/TB",
         "-fps_mode", "passthrough", "-c:v", "libx264", "-crf", "6",
         "-pix_fmt", "yuv420p", str(dst)], check=True)


# Seed clicks on the rock ridge in the first frame (source pixels):
# positives along the ridge, negatives in the open water on each side.
# SAM 3.1's text detector does not find this rock ("rock", "reef", "stone"
# all return nothing), but its point-prompted tracker head segments it cleanly.
SEED_POINTS = [[616, 731, 1], [578, 540, 1], [640, 930, 1], [640, 250, 1],
               [193, 290, 0], [1080, 580, 0]]


def points_from_mask(m: np.ndarray, n_pos: int = 8, ring_neg: bool = False):
    """Positive clicks along the mask's spine from its very top to its
    bottom (thin necks included), negatives in the water on both sides,
    including the band near the surface where water and the upper rock meet.
    With ring_neg, extra negatives sit just outside the mask edge."""
    h, w = m.shape
    m8 = m.astype(np.uint8)
    # light erosion only, so thin parts of the ridge still get clicks
    er = cv2.erode(m8, np.ones((7, 7), np.uint8))
    src = er if er.sum() > 500 else m8
    rows = np.where(src.sum(1) >= 6)[0]
    if len(rows) == 0:
        rows = np.where(src.any(1))[0]
    pts = []
    lo, hi = rows.min(), rows.max()
    for y in np.linspace(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo), n_pos).astype(int):
        xs = np.where(src[y])[0]
        if len(xs):
            pts.append([float(xs[len(xs) // 2]), float(y), 1])
    dist = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5)
    for rs in (slice(0, h // 6), slice(h // 6, 5 * h // 6)):
        for side in (slice(0, w // 3), slice(2 * w // 3, w)):
            d = dist[rs, side]
            if d.max() > 60:
                y, x = np.unravel_index(np.argmax(d), d.shape)
                pts.append([float(x + side.start), float(y + rs.start), 0])
    if ring_neg:
        ring = (dist > 28) & (dist < 40)
        ys, xs = np.nonzero(ring)
        if len(ys):
            for k in np.linspace(0, len(ys) - 1, 6).astype(int):
                pts.append([float(xs[k]), float(ys[k]), 0])
    return pts


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def run_rock_masks(src: Path, mask_root: Path, stride: int, prompt: str) -> Path:
    """SAM 3.1 (multiplex predictor, local checkpoint) on every `stride`-th
    frame (stride 1 = every source frame). Each frame is segmented on its own
    from point prompts derived from the previous frame's mask, then checked:
    if it disagrees with the previous frame (IoU / area jump), the frame is
    re-segmented with alternative prompts and the most consistent result is
    kept. Every decision is logged to qc.json."""
    import shutil
    import time

    import torch

    from shades.masking import _build_predictor, extract_jpeg_frames

    if stride == 1:
        proxy = src
    else:
        proxy = REPO / ".cache" / f"{src.stem}_stride{stride}.mp4"
        if not proxy.exists():
            make_proxy(src, proxy, stride)
    mdir = mask_root / f"{src.stem}__rock_s{stride}"
    sub = mdir / "subject"
    sub.mkdir(parents=True, exist_ok=True)
    frames = REPO / ".cache" / f"sam_frames_{proxy.stem}_s{stride}"
    n = extract_jpeg_frames(proxy, frames, 2)
    if len(list(sub.glob("*.png"))) == n and (mdir / "meta.json").exists():
        log(f"[sam] cached rock masks: {mdir}")
        return mdir
    h, w = cv2.imread(str(frames / "00000.jpg")).shape[:2]
    pred, on_gpu = _build_predictor(Masking(), REPO, REPO / ".cache")
    log(f"[sam] SAM 3.1 loaded on {'CUDA' if on_gpu else 'CPU'}; {n} frames")
    one = REPO / ".cache" / "sam_one"

    def segment(i, pts):
        shutil.rmtree(one, ignore_errors=True)
        one.mkdir(parents=True)
        (one / "00000.jpg").symlink_to(frames / f"{i:05d}.jpg")
        sid = pred.handle_request({"type": "start_session",
                                   "resource_path": str(one)})["session_id"]
        # a text prompt makes the wrapper process (and cache) this frame;
        # without it the point-prompt result is dropped in post-processing
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": 0, "text": prompt})
        r = pred.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": 0, "obj_id": 1,
            "points": torch.tensor([[x / w, y / h] for x, y, _ in pts], dtype=torch.float32),
            "point_labels": torch.tensor([int(l) for *_, l in pts], dtype=torch.int32)})
        pred.handle_request({"type": "close_session", "session_id": sid})
        o = r["outputs"]
        ids = list(o["out_obj_ids"])
        return (np.asarray(o["out_binary_masks"][ids.index(1)]) > 0) if 1 in ids else None

    qc_f = mdir / "qc.json"
    qc = json.loads(qc_f.read_text()) if qc_f.exists() else {}
    hist: list[np.ndarray] = []
    iou_ok, area_lo, area_hi = (0.82, 0.85, 1.18) if stride == 1 else (0.78, 0.75, 1.3)
    for i in range(n):
        out_png = sub / f"{i:05d}.png"
        if out_png.exists():
            hist.append(cv2.imread(str(out_png), cv2.IMREAD_GRAYSCALE) > 127)
            hist = hist[-6:]
            continue
        t0 = time.time()
        prev = hist[-1] if hist else None
        if prev is None or prev.sum() < 2000:
            m = segment(i, SEED_POINTS)
            rec = {"try": 1, "iou_prev": None}
        else:
            # consensus of the recent frames: one bad frame cannot drag the
            # next ones off (prompts and the QC reference both come from it)
            basis = np.mean(np.stack(hist[-6:]), 0) >= 0.5 if len(hist) >= 3 else prev

            def score(c):
                if c is None:
                    return -1.0, 0.0
                ar = c.sum() / max(basis.sum(), 1)
                return _iou(c, basis), ar
            cands = [("spine", segment(i, points_from_mask(basis)))]
            sc = score(cands[0][1])
            if not (sc[0] >= iou_ok and area_lo <= sc[1] <= area_hi):
                # re-segment instead of trusting a drifting mask
                cands.append(("dense+ring", segment(i, points_from_mask(basis, 12, True))))
                cands.append(("prev", segment(i, points_from_mask(prev, 10, True))))
            best = max(cands, key=lambda c: score(c[1])[0])
            m = best[1] if best[1] is not None else prev
            sc = score(m)
            rec = {"try": len(cands), "chosen": best[0], "iou_prev": round(_iou(m, prev), 4),
                   "iou_consensus": round(sc[0], 4),
                   "area_ratio": round(float(sc[1]), 4),
                   "flag": not (sc[0] >= iou_ok and area_lo <= sc[1] <= area_hi)}
        rec["area"] = int(m.sum())
        qc[str(i)] = rec
        cv2.imwrite(str(out_png), m.astype(np.uint8) * 255)
        hist.append(m)
        if len(hist) > 6:
            hist.pop(0)
        qc_f.write_text(json.dumps(qc))
        log(f"[sam] frame {i + 1}/{n}: area {rec['area']}, iou {rec.get('iou_prev')}, "
            f"cons {rec.get('iou_consensus')}, "
            f"tries {rec['try']}{' FLAG' if rec.get('flag') else ''}, {time.time() - t0:.0f}s")
    flagged = [k for k, v in qc.items() if v.get("flag")]
    (mdir / "meta.json").write_text(json.dumps(
        {"num_frames": n, "stride": stride, "model": "SAM 3.1 multiplex (local)",
         "prompt": "points (auto from previous mask, QC re-segmentation)",
         "flagged_frames": flagged}, indent=1))
    log(f"[sam] done; flagged frames: {flagged}")
    return mdir


def stabilize_masks(mdir: Path, frames_dir: Path, radius: int = 6,
                    flow_scale: float = 0.5) -> Path:
    """Motion-compensated temporal vote over the per-frame SAM 3.1 masks.

    Each frame's mask is rebuilt from its own SAM 3.1 mask plus the masks of
    the `radius` frames on either side, warped into this frame with dense
    optical flow (Farneback) and combined with Gaussian weights. Regions SAM
    flips in and out of between single frames (the far, blurry rock under
    the surface) settle to the majority answer; the main body, where all
    frames agree, is unchanged. Output: soft masks in <mdir>/stable/."""
    out_dir = mdir / "stable"
    files = sorted((mdir / "subject").glob("*.png"))
    n = len(files)
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) == n:
        return out_dir
    out_dir.mkdir(exist_ok=True)
    masks = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for f in files]
    h, w = masks[0].shape
    sw, sh = int(w * flow_scale), int(h * flow_scale)
    gray = [cv2.resize(cv2.imread(str(frames_dir / f"{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE),
                       (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wts = {d: float(np.exp(-0.5 * (d / (radius * 0.6)) ** 2)) for d in range(-radius, radius + 1)}
    stats = {}
    for i in range(n):
        acc = masks[i] * wts[0]
        tot = wts[0]
        for d in range(-radius, radius + 1):
            j = i + d
            if d == 0 or not 0 <= j < n:
                continue
            fl = cv2.calcOpticalFlowFarneback(gray[i], gray[j], None, 0.5, 4, 21, 3, 5, 1.1, 0)
            fl = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) / flow_scale
            warped = cv2.remap(masks[j], gx + fl[..., 0], gy + fl[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            acc += warped * wts[d]
            tot += wts[d]
        vote = acc / tot
        # majority with a wide soft ramp: persistent changes fade over a few
        # frames instead of popping; agreed regions stay at 1
        soft = np.clip((vote - 0.5) / 0.6 + 0.5, 0, 1)
        stats[i] = round(float(np.abs((soft > 0.5).astype(np.float32) - masks[i]).mean()), 5)
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), (soft * 255).astype(np.uint8))
        if i % 40 == 0:
            log(f"[stable] frame {i}/{n}, changed {stats[i] * 100:.2f}% of pixels")
    (mdir / "stable_stats.json").write_text(json.dumps(stats))
    return out_dir


def load_rock_soft(mdir: Path, n: int, stride: int, h: int, w: int) -> np.ndarray:
    """Per-source-frame soft rock mask (float32 0..1), interpolating the
    frames SAM skipped."""
    sub = mdir / "stable" if (mdir / "stable").exists() else mdir / "subject"
    files = sorted(sub.glob("*.png"))
    keys = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            for p in files]
    out = np.zeros((n, h, w), np.float32)
    for i in range(n):
        k, r = divmod(i, stride)
        a = keys[min(k, len(keys) - 1)]
        b = keys[min(k + 1, len(keys) - 1)]
        t = r / stride
        out[i] = a * (1 - t) + b * t
    return out


# ------------------------------------------------------------- zone masks --

def clean_rock(m: np.ndarray) -> np.ndarray:
    """Fill holes, drop small islands, keep the main rock body."""
    b = (m > 0.5).astype(np.uint8)
    if not b.any():
        return b.astype(np.float32)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros_like(b)
    big = stats[1:, cv2.CC_STAT_AREA].max()
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= 0.04 * big:
            keep[lab == j] = 1
    # fill holes
    inv = (1 - keep).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = keep.shape
    for j in range(1, n2):
        x, y, bw, bh, _ = st2[j]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            keep[lab2 == j] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, k)
    return keep.astype(np.float32)


def divide_line(rock: np.ndarray) -> np.ndarray:
    """x position of the dividing line for every row: the rock mask's
    weighted centre per row, interpolated through rows without rock and
    smoothed vertically."""
    h, w = rock.shape
    xs = np.arange(w, dtype=np.float32)
    wsum = rock.sum(1)
    cx = np.full(h, np.nan, np.float32)
    ok = wsum > 3
    cx[ok] = (rock[ok] * xs).sum(1) / wsum[ok]
    if not ok.any():
        return np.full(h, w / 2, np.float32)
    rows = np.arange(h)
    cx = np.interp(rows, rows[ok], cx[ok]).astype(np.float32)
    cx = cv2.GaussianBlur(cx.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=h * 0.03).ravel()
    return cx


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (grey guide): edge-aware refinement of a soft
    mask so its transition snaps to the real rock/water edge of this frame."""
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.boxFilter(I, -1, k)
    mp = cv2.boxFilter(p, -1, k)
    cov = cv2.boxFilter(I * p, -1, k) - mI * mp
    var = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def refine_rock(rock_soft: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sub-pixel, edge-aware rock alpha for this exact frame (no halo, no
    spill): guided filter on the frame's luma, then a tiny anti-alias."""
    I = _luma(x).astype(np.float32)
    a = guided_filter(I, rock_soft.astype(np.float32), 6, 2e-3)
    a = cv2.GaussianBlur(np.clip(a, 0, 1), (0, 0), 1.2)
    return np.clip(a, 0, 1)


def side_field(x: np.ndarray, line: np.ndarray, rock: np.ndarray, w_px: float) -> np.ndarray:
    """0 = left sea, 1 = right sea. Not a straight cut: the split follows the
    rock line but bends with the scene's own large-scale light — sunlit water
    leans turquoise, naturally shadowed water leans navy — so the two seas
    meet the way light falls off underwater."""
    h, w = rock.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    geo = (xs - line[:, None]) / w_px
    # large-scale illumination of the water only (rock excluded)
    small = cv2.resize(_luma(x), (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    wt = cv2.resize(1 - rock, (w // 8, h // 8), interpolation=cv2.INTER_AREA) + 1e-3
    lum = cv2.GaussianBlur(small * wt, (0, 0), 7) / cv2.GaussianBlur(wt, (0, 0), 7)
    lum = cv2.resize(lum, (w, h), interpolation=cv2.INTER_CUBIC)
    med = float(np.median(lum))
    spread = float(np.percentile(lum, 90) - np.percentile(lum, 10)) + 1e-3
    z = (med - lum) / spread          # >0 where the water is darker
    t = geo + 1.1 * z
    return (1.0 / (1.0 + np.exp(-t))).astype(np.float32)


def zone_masks(rock_soft: np.ndarray, line: np.ndarray, feather_px: float, x=None):
    """Feathered (left, rock, right) weights that sum to 1."""
    h, w = rock_soft.shape
    r = refine_rock(rock_soft, x) if x is not None else np.clip(rock_soft, 0, 1)
    side = side_field(x, line, r, 0.075 * w) if x is not None else \
        1.0 / (1.0 + np.exp(-(np.arange(w)[None, :] - line[:, None]) / (feather_px * 1.6)))
    # atmospheric perspective: the far (upper) rock is seen through more
    # water, so it takes less of the warm grade and more of each sea's
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    t = np.clip(ys / 0.55, 0, 1)
    r = r * (0.40 + 0.60 * t * t * (3 - 2 * t))
    left = (1 - r) * (1 - side)
    right = (1 - r) * side
    return left, r, right


# ------------------------------------------------------------------ grade --

def _luma(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def _sat(x, s):
    y = _luma(x)[..., None]
    return y + (x - y) * s


def _lgg(x, lift, gamma, gain):
    """ASC-CDL-like lift / gamma / gain per channel (RGB arrays)."""
    lift, gamma, gain = map(lambda v: np.asarray(v, np.float32), (lift, gamma, gain))
    y = x * gain + lift * (1 - x)
    return np.clip(y, 0, None) ** (1.0 / gamma)


def _split_tone(x, shadow_rgb, high_rgb, s_amt, h_amt):
    y = _luma(x)[..., None]
    ws = (1 - y) ** 2 * s_amt
    wh = y ** 2 * h_amt
    sh = np.asarray(shadow_rgb, np.float32)
    hi = np.asarray(high_rgb, np.float32)
    # overlay-style tint that preserves luma
    x = x + ws * (sh - _luma(sh[None, None])[..., None]) + wh * (hi - _luma(hi[None, None])[..., None])
    return x


def _soft_knee(x, knee=0.78):
    """Highlight roll-off: compresses values above the knee toward 1."""
    over = np.maximum(x - knee, 0)
    return np.where(x > knee, knee + (1 - knee) * (1 - np.exp(-over / (1 - knee) * 1.6)) / (1 - np.exp(-1.6)), x)


def _hue_tame(x, center, width, target, shift, desat, v_lo, v_hi):
    """Pull a hue band toward `target` and desaturate it, weighted by
    brightness (used for yellow-green contamination in bright water and
    yellow patches on the rock)."""
    hsv = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    d = (H - center + 180) % 360 - 180
    band = np.exp(-0.5 * (d / width) ** 2)
    vt = np.clip((V - v_lo) / max(v_hi - v_lo, 1e-3), 0, 1)
    wgt = band * vt * vt * (3 - 2 * vt) * np.clip(S * 3, 0, 1)
    dt = (target - H + 180) % 360 - 180
    hsv[..., 0] = (H + np.clip(dt, -shift, shift) * wgt) % 360
    hsv[..., 1] = S * (1 - desat * wgt)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def grade_left(x):
    # luminous turquoise / emerald glass
    y = _lgg(x, lift=(0.0, 0.02, 0.025), gamma=(0.96, 1.05, 1.07),
             gain=(0.88, 1.03, 1.07))
    y = _split_tone(y, shadow_rgb=(0.05, 0.34, 0.33), high_rgb=(0.72, 1.0, 0.97),
                    s_amt=0.30, h_amt=0.14)
    # bright yellow-green contamination -> cleaner cyan, a little calmer
    y = _hue_tame(y, center=75, width=28, target=165, shift=22, desat=0.22,
                  v_lo=0.45, v_hi=0.9)
    return np.clip(_sat(y, 1.08), 0, 1)


def grade_rock(x):
    # natural bronze / earthy brown base, copper-gold only in the highlights;
    # no sharpening — only a gentle large-radius tonal contrast
    y = _lgg(x, lift=(0.006, 0.002, -0.002), gamma=(1.02, 1.0, 0.98),
             gain=(1.05, 1.0, 0.93))
    y = _split_tone(np.clip(y, 0, 1), shadow_rgb=(0.27, 0.22, 0.14),
                    high_rgb=(1.0, 0.74, 0.48), s_amt=0.14, h_amt=0.17)
    # yellow patches -> copper/bronze, and calmer
    y = _hue_tame(y, center=55, width=14, target=36, shift=12, desat=0.16,
                  v_lo=0.35, v_hi=0.85)
    base = cv2.GaussianBlur(y, (0, 0), 28)
    y = y + (y - base) * 0.10
    y = _soft_knee(np.clip(y, 0, None), 0.74)
    return np.clip(_sat(y, 1.05), 0, 1)


def grade_right(x, haze):
    # deep cobalt / navy, detailed shadows (no crush), light depth haze
    y = _lgg(x, lift=(0.0, 0.012, 0.035), gamma=(0.95, 0.99, 1.06),
             gain=(0.78, 0.90, 1.04))
    y = _sat(y, 0.88)
    y = _split_tone(y, shadow_rgb=(0.03, 0.08, 0.24), high_rgb=(0.58, 0.74, 0.95),
                    s_amt=0.30, h_amt=0.12)
    y = _hue_tame(y, center=75, width=28, target=200, shift=22, desat=0.25,
                  v_lo=0.40, v_hi=0.85)
    fog = np.asarray((0.08, 0.17, 0.30), np.float32)
    y = np.clip(y, 0, 1)
    return y * (1 - haze[..., None]) + fog * haze[..., None]


def global_finish(x):
    # gentle S-curve around mid grey, a soft toe (no crushed blacks) and a
    # natural highlight roll-off
    x = np.clip(x, 0, 1)
    p = 0.42
    c = 1.06
    y = np.where(x < p, p * (x / p) ** c, 1 - (1 - p) * ((1 - x) / (1 - p)) ** c)
    y = 0.012 + (1 - 0.012) * y
    y = _soft_knee(y, 0.86)
    return np.clip(y, 0, 1)


def haze_map(h, w, line):
    """Haze grows toward the right edge and toward the top (distance)."""
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    span = np.maximum(w - line[:, None], 1)
    t = np.clip((xs - line[:, None]) / span, 0, 1)
    depth = 1 - ys / h
    return np.clip(0.06 + 0.16 * t ** 1.3 + 0.10 * depth, 0, 0.30).astype(np.float32)


def grade_frame(rgb8: np.ndarray, rock_soft: np.ndarray, line: np.ndarray,
                feather_px: float, bits: int = 8):
    x = rgb8.astype(np.float32) / 255.0
    h, w = rock_soft.shape
    L, R, Rt = zone_masks(rock_soft, line, feather_px, x)
    out = (grade_left(x) * L[..., None] + grade_rock(x) * R[..., None]
           + grade_right(x, haze_map(h, w, line)) * Rt[..., None])
    out = global_finish(out)
    if bits == 16:
        q = (out * 65535 + np.random.uniform(-0.5, 0.5, out.shape)).clip(0, 65535).astype(np.uint16)
    else:
        q = (out * 255 + np.random.uniform(-0.5, 0.5, out.shape)).clip(0, 255).astype(np.uint8)
    return q, (L, R, Rt)


# ------------------------------------------------------------------ video --

def read_all_frames(src: Path, w: int, h: int):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
                         stdout=subprocess.PIPE)
    sz = w * h * 3
    while True:
        buf = p.stdout.read(sz)
        if len(buf) < sz:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.wait()


def prepare_masks(rock_raw: np.ndarray, feather_px: float):
    rock = np.stack([clean_rock(m) for m in rock_raw])
    # light temporal smoothing (weights 1-2-1) against edge flicker; kept
    # short so the mask never lags behind the camera move (no ghost edges)
    pad = np.concatenate([rock[:1], rock, rock[-1:]])
    rock = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    lines = np.stack([divide_line((m > 0.5).astype(np.float32)) for m in rock])
    # temporal smoothing of the split line
    lines = cv2.GaussianBlur(lines, (1, 0), sigmaX=0, sigmaY=2.5)
    return rock.astype(np.float32), lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--masks-dir", type=Path, default=REPO / "masks")
    ap.add_argument("--stride", type=int, default=2,
                    help="run SAM 3.1 on every Nth frame, interpolate between")
    ap.add_argument("--prompt", default="rock")
    ap.add_argument("--feather", type=float, default=0.018,
                    help="feather as a fraction of frame width")
    ap.add_argument("--masks-only", action="store_true")
    ap.add_argument("--codec", choices=["prores", "h264"], default="prores",
                    help="prores = ProRes 422 HQ 10-bit (default), h264 = preview")
    ap.add_argument("--stills", type=Path, default=None,
                    help="write QC stills (overlay + graded) to this folder and exit")
    a = ap.parse_args()

    info = probe(a.input)
    w, h, n = info.width, info.height, info.n_frames
    log(f"[probe] {w}x{h} {float(info.fps):.3f} fps {n} frames audio={info.has_audio}")
    mdir = run_rock_masks(a.input, a.masks_dir, a.stride, a.prompt)
    if a.stride == 1:
        stabilize_masks(mdir, REPO / ".cache" / f"sam_frames_{a.input.stem}_s1")
    if a.masks_only:
        return 0
    rock_raw = load_rock_soft(mdir, n, a.stride, h, w)
    feather_px = a.feather * w
    rock, lines = prepare_masks(rock_raw, feather_px)

    if a.stills:
        a.stills.mkdir(parents=True, exist_ok=True)
        picks = {int(round(s * float(info.fps))) for s in (0, 3, 6, 9)}
        for i, f in enumerate(read_all_frames(a.input, w, h)):
            if i not in picks:
                continue
            g, (L, R, Rt) = grade_frame(f, rock[i], lines[i], feather_px)
            ov = f.astype(np.float32)
            tint = (L[..., None] * (0, 220, 200) + R[..., None] * (255, 140, 0)
                    + Rt[..., None] * (40, 60, 255))
            ov = (ov * 0.55 + tint * 0.45).astype(np.uint8)
            cv2.polylines(ov, [np.stack([lines[i], np.arange(h)], 1).astype(np.int32)],
                          False, (255, 255, 255), 2)
            cv2.imwrite(str(a.stills / f"overlay_{i:04d}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(a.stills / f"compare_{i:04d}.jpg"),
                        cv2.cvtColor(np.concatenate([f, g], 1), cv2.COLOR_RGB2BGR))
        log(f"[stills] wrote {a.stills}")
        return 0

    out = a.output or REPO / "output" / f"{a.input.stem}_split-sea_grade.mov"
    out.parent.mkdir(parents=True, exist_ok=True)
    rate = f"{info.fps.numerator}/{info.fps.denominator}"
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv"]
    if a.codec == "prores":
        venc = ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0",
                "-pix_fmt", "yuv422p10le", "-bits_per_mb", "8000"]
        bits, pix = 16, "rgb48le"
    else:
        venc = ["-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p"]
        bits, pix = 8, "rgb24"
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", rate,
         "-i", "-", "-i", str(a.input),
         "-map", "0:v", "-map", "1:a?", "-c:a", "copy", *venc, *tags, str(out)],
        stdin=subprocess.PIPE)
    for i, f in enumerate(read_all_frames(a.input, w, h)):
        g, _ = grade_frame(f, rock[min(i, n - 1)], lines[min(i, n - 1)], feather_px, bits)
        enc.stdin.write(g.tobytes())
        if i % 50 == 0:
            log(f"[render] frame {i}/{n}")
    enc.stdin.close()
    enc.wait()
    log(f"[render] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
