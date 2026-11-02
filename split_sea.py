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


def points_from_mask(m: np.ndarray, n_pos: int = 8):
    """Positive clicks along the mask's spine from its very top to its
    bottom (thin necks included), negatives in the water on both sides."""
    h, w = m.shape
    # light erosion only, so thin parts of the ridge still get clicks
    er = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8))
    src = er if er.sum() > 500 else m.astype(np.uint8)
    rows = np.where(src.sum(1) >= 6)[0]
    if len(rows) == 0:
        rows = np.where(src.any(1))[0]
    pts = []
    lo, hi = rows.min(), rows.max()
    for y in np.linspace(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo), n_pos).astype(int):
        xs = np.where(src[y])[0]
        if len(xs):
            pts.append([float(xs[len(xs) // 2]), float(y), 1])
    dist = cv2.distanceTransform((1 - m.astype(np.uint8)), cv2.DIST_L2, 5)
    for side in (slice(0, w // 3), slice(2 * w // 3, w)):
        d = dist[h // 6: 5 * h // 6, side]
        if d.max() > 60:
            y, x = np.unravel_index(np.argmax(d), d.shape)
            pts.append([float(x + side.start), float(y + h // 6), 0])
    return pts


def run_rock_masks(src: Path, mask_root: Path, stride: int, prompt: str) -> Path:
    """SAM 3.1 (multiplex predictor, local checkpoint) on every `stride`-th
    frame. Each frame is segmented on its own from point prompts derived
    from the previous frame's mask, so there is no tracker drift to manage."""
    import shutil
    import time

    import torch

    from shades.masking import _build_predictor, extract_jpeg_frames

    proxy = REPO / ".cache" / f"{src.stem}_stride{stride}.mp4"
    if not proxy.exists():
        make_proxy(src, proxy, stride)
    mdir = mask_root / f"{src.stem}__rock_s{stride}"
    sub = mdir / "subject"
    sub.mkdir(parents=True, exist_ok=True)
    frames = REPO / ".cache" / f"sam_frames_{proxy.stem}"
    n = extract_jpeg_frames(proxy, frames, 2)
    if len(list(sub.glob("*.png"))) == n:
        log(f"[sam] cached rock masks: {mdir}")
        return mdir
    h, w = cv2.imread(str(frames / "00000.jpg")).shape[:2]
    pred, on_gpu = _build_predictor(Masking(), REPO, REPO / ".cache")
    log(f"[sam] SAM 3.1 loaded on {'CUDA' if on_gpu else 'CPU'}; {n} frames")
    one = REPO / ".cache" / "sam_one"
    prev = None
    for i in range(n):
        out_png = sub / f"{i:05d}.png"
        if out_png.exists():
            prev = cv2.imread(str(out_png), cv2.IMREAD_GRAYSCALE) > 127
            continue
        t0 = time.time()
        pts = SEED_POINTS if prev is None or prev.sum() < 2000 else points_from_mask(prev)
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
        m = (np.asarray(o["out_binary_masks"][ids.index(1)]) > 0) if 1 in ids \
            else (prev if prev is not None else np.zeros((h, w), bool))
        cv2.imwrite(str(out_png), m.astype(np.uint8) * 255)
        prev = m
        log(f"[sam] frame {i + 1}/{n}: area {int(m.sum())}, "
            f"{len(pts)} pts, {time.time() - t0:.0f}s")
    (mdir / "meta.json").write_text(json.dumps(
        {"num_frames": n, "stride": stride, "model": "SAM 3.1 multiplex",
         "prompt": "points (auto from previous mask)"}, indent=1))
    log("[sam] done")
    return mdir


def load_rock_soft(mdir: Path, n: int, stride: int, h: int, w: int) -> np.ndarray:
    """Per-source-frame soft rock mask (float32 0..1), interpolating the
    frames SAM skipped."""
    files = sorted((mdir / "subject").glob("*.png"))
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


def zone_masks(rock_soft: np.ndarray, line: np.ndarray, feather_px: float):
    """Feathered (left, rock, right) weights that sum to 1."""
    h, w = rock_soft.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    # signed distance to the line -> soft side split
    # the sea/sea split is wide and gentle (it crosses open sand below the
    # rock); the rock edge itself stays tighter
    side = 1.0 / (1.0 + np.exp(-(xs - line[:, None]) / (feather_px * 1.6)))
    r = cv2.GaussianBlur(rock_soft, (0, 0), feather_px * 0.5)
    # atmospheric perspective: the far (upper) rock is seen through more
    # water, so it takes less of the warm grade and more of each sea's
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    t = np.clip(ys / 0.55, 0, 1)
    r = np.clip(r, 0, 1) * (0.35 + 0.65 * t * t * (3 - 2 * t))
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


def _local_contrast(x, amount, sigma):
    blur = cv2.GaussianBlur(x, (0, 0), sigma)
    return x + (x - blur) * amount


def grade_left(x):
    # luminous turquoise glass
    y = _lgg(x, lift=(0.0, 0.025, 0.03), gamma=(0.95, 1.06, 1.08),
             gain=(0.86, 1.04, 1.08))
    y = _split_tone(y, shadow_rgb=(0.05, 0.35, 0.36), high_rgb=(0.70, 1.0, 0.98),
                    s_amt=0.35, h_amt=0.18)
    return np.clip(_sat(y, 1.12), 0, 1)


def grade_rock(x):
    # warm amber/copper concentrated in the highlights, umber shadows,
    # crisp texture; dark/blue pixels (water seen through gaps) stay quieter
    y = _lgg(x, lift=(0.008, 0.0, -0.006), gamma=(1.05, 1.0, 0.94),
             gain=(1.10, 1.02, 0.88))
    y = _split_tone(np.clip(y, 0, 1), shadow_rgb=(0.30, 0.20, 0.13),
                    high_rgb=(1.0, 0.80, 0.52), s_amt=0.18, h_amt=0.20)
    y = _local_contrast(y, 0.22, 6)
    return np.clip(_sat(y, 1.04), 0, 1)


def grade_right(x, haze):
    # deep ink-navy, cooler, slightly desaturated, with depth haze
    y = _lgg(x, lift=(-0.005, 0.005, 0.03), gamma=(0.90, 0.96, 1.06),
             gain=(0.70, 0.86, 1.04))
    y = _sat(y, 0.85)
    y = _split_tone(y, shadow_rgb=(0.02, 0.06, 0.22), high_rgb=(0.55, 0.72, 0.95),
                    s_amt=0.45, h_amt=0.15)
    fog = np.asarray((0.06, 0.14, 0.26), np.float32)
    y = np.clip(y, 0, 1)
    return y * (1 - haze[..., None]) + fog * haze[..., None]


def global_finish(x):
    # gentle S-curve around mid grey + soft highlight roll-off
    x = np.clip(x, 0, 1)
    p = 0.42
    c = 1.10
    y = np.where(x < p, p * (x / p) ** c, 1 - (1 - p) * ((1 - x) / (1 - p)) ** c)
    y = np.where(y > 0.85, 0.85 + (1 - np.exp(-(y - 0.85) * 4)) * 0.15 / (1 - np.exp(-0.6)), y)
    return np.clip(y, 0, 1)


def haze_map(h, w, line):
    """Haze grows toward the right edge and toward the top (distance)."""
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    span = np.maximum(w - line[:, None], 1)
    t = np.clip((xs - line[:, None]) / span, 0, 1)
    depth = 1 - ys / h
    return np.clip(0.10 + 0.22 * t ** 1.3 + 0.14 * depth, 0, 0.42).astype(np.float32)


def grade_frame(rgb8: np.ndarray, rock_soft: np.ndarray, line: np.ndarray,
                feather_px: float):
    x = rgb8.astype(np.float32) / 255.0
    h, w = rock_soft.shape
    L, R, Rt = zone_masks(rock_soft, line, feather_px)
    out = (grade_left(x) * L[..., None] + grade_rock(x) * R[..., None]
           + grade_right(x, haze_map(h, w, line)) * Rt[..., None])
    return (global_finish(out) * 255 + np.random.uniform(-0.5, 0.5, out.shape)).clip(0, 255).astype(np.uint8), (L, R, Rt)


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
    # temporal smoothing (centred moving average, 5 frames)
    k = 5
    pad = np.concatenate([rock[:1].repeat(k // 2, 0), rock, rock[-1:].repeat(k // 2, 0)])
    cs = np.cumsum(pad, 0, dtype=np.float32)
    cs = np.concatenate([np.zeros_like(cs[:1]), cs])
    rock = (cs[k:] - cs[:-k]) / k
    lines = np.stack([divide_line((m > 0.5).astype(np.float32)) for m in rock])
    # temporal smoothing of the line
    lines = cv2.GaussianBlur(lines, (1, 0), sigmaX=0, sigmaY=2.0)
    return rock, lines


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
    ap.add_argument("--stills", type=Path, default=None,
                    help="write QC stills (overlay + graded) to this folder and exit")
    a = ap.parse_args()

    info = probe(a.input)
    w, h, n = info.width, info.height, info.n_frames
    log(f"[probe] {w}x{h} {float(info.fps):.3f} fps {n} frames audio={info.has_audio}")
    mdir = run_rock_masks(a.input, a.masks_dir, a.stride, a.prompt)
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
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{info.fps.numerator}/{info.fps.denominator}" if hasattr(info.fps, "numerator") else str(info.fps),
         "-i", "-", "-i", str(a.input),
         "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
         "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-color_range", "tv", "-shortest", str(out)], stdin=subprocess.PIPE)
    for i, f in enumerate(read_all_frames(a.input, w, h)):
        g, _ = grade_frame(f, rock[min(i, n - 1)], lines[min(i, n - 1)], feather_px)
        enc.stdin.write(g.tobytes())
        if i % 50 == 0:
            log(f"[render] frame {i}/{n}")
    enc.stdin.close()
    enc.wait()
    log(f"[render] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
