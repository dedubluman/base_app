"""SAM 3.1 keyframe masks -> stable per-frame region weights.

    python -m deepgrade.build_masks --video inputs/player_export.mov [--range 0 62]

Diver: SAM-tracked every 2nd frame (+ snorkel keyframes), interpolated to the
other frames by optical-flow warping, 3-tap flow-compensated temporal smoothing,
then guided-filter edge refinement against the real frame.
Rocks / water / sky / mountain: SAM keyframes every 3rd frame, linear in time,
smoothed over keyframes, feathered. Final weights D,S,R,W,U are written as 8-bit
PNGs (masks/<stem>/final/<name>/<frame>.png) together with ``gate.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from shades.io import iter_frames, probe

from .params import Params
from .regions import compose, feather, refine_diver

REPO = Path(__file__).resolve().parent.parent
_dis = None


def flow(a_gray: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
    """Flow that maps pixels of frame A to frame B, half-res uint8 inputs."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(a_gray, b_gray, None)


def warp_to(mask: np.ndarray, fl_t_to_k: np.ndarray, size) -> np.ndarray:
    """Pull ``mask`` (frame k) into frame t using flow t->k (half-res flow)."""
    W, H = size
    f = cv2.resize(fl_t_to_k, (W, H), interpolation=cv2.INTER_LINEAR) * (W / fl_t_to_k.shape[1])
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(mask, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def read_mask(d: Path, f: int, shape):
    p = d / f"{f:05d}.png"
    if not p.exists():
        return None
    return cv2.imread(str(p), 0).astype(np.float32) / 255.0


def keyframes(d: Path, shape) -> dict[int, np.ndarray]:
    out = {}
    if d.exists():
        for p in sorted(d.glob("*.png")):
            out[int(p.stem)] = cv2.imread(str(p), 0).astype(np.float32) / 255.0
    return out


def interp_linear(keys: dict[int, np.ndarray], t: int, shape, fade: int = 6) -> np.ndarray:
    """Linear-in-time blend of the two surrounding keyframes. Outside the keyframe
    range the mask fades to zero over ``fade`` frames (no pop)."""
    if not keys:
        return np.zeros(shape, np.float32)
    ks = sorted(keys)
    if t <= ks[0]:
        return keys[ks[0]] * max(0.0, 1 - (ks[0] - t) / fade)
    if t >= ks[-1]:
        return keys[ks[-1]] * max(0.0, 1 - (t - ks[-1]) / fade)
    lo = max(k for k in ks if k <= t)
    hi = min(k for k in ks if k >= t)
    if lo == hi:
        return keys[lo]
    a = (t - lo) / (hi - lo)
    return (1 - a) * keys[lo] + a * keys[hi]


def smooth_keys(keys: dict[int, np.ndarray], sigma: float) -> dict[int, np.ndarray]:
    if sigma <= 0 or len(keys) < 3:
        return keys
    ks = sorted(keys)
    out = {}
    for i, k in enumerate(ks):
        acc, wsum = 0, 0
        for j in range(max(0, i - 2), min(len(ks), i + 3)):
            wgt = np.exp(-0.5 * ((j - i) / sigma) ** 2)
            acc = acc + wgt * keys[ks[j]]
            wsum += wgt
        out[k] = acc / wsum
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=REPO / "inputs/player_export.mov")
    ap.add_argument("--range", nargs=2, type=int, default=None)
    ap.add_argument("--params", type=Path, default=None)
    a = ap.parse_args()
    p = Params.load(a.params) if a.params else Params()
    info = probe(a.video)
    n, W, H = info.n_frames, info.width, info.height
    mdir = REPO / "masks" / a.video.stem
    fin = mdir / "final"
    for k in "DSRWU":
        (fin / k).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lo, hi = a.range if a.range else (0, n - 1)

    # half-res gray for flow
    grays = []
    for fr in iter_frames(a.video, info, scale_w=W // 2):
        g = (0.2126 * fr[..., 0] + 0.7152 * fr[..., 1] + 0.0722 * fr[..., 2])
        grays.append(np.clip(g * 255 + .5, 0, 255).astype(np.uint8))
    assert len(grays) == n, (len(grays), n)
    print(f"[masks] decoded {n} gray frames {time.time()-t0:.0f}s", flush=True)

    shape = (H, W)
    diver_k = keyframes(mdir / "diver", shape)
    snork_k = smooth_keys(keyframes(mdir / "snorkel", shape), 0.0)
    rocks_k = smooth_keys(keyframes(mdir / "rocks", shape), p.temporal_sigma_keyframes)
    mount_k = smooth_keys(keyframes(mdir / "mountain", shape), p.temporal_sigma_keyframes)
    water_k = smooth_keys(keyframes(mdir / "water", shape), p.temporal_sigma_keyframes)
    sky_k = smooth_keys(keyframes(mdir / "sky", shape), p.temporal_sigma_keyframes)
    print(f"[masks] keyframes diver={len(diver_k)} snorkel={len(snork_k)} rocks={len(rocks_k)} "
          f"water={len(water_k)} sky={len(sky_k)} mountain={len(mount_k)}", flush=True)

    flow_cache: dict[tuple[int, int], np.ndarray] = {}

    def fl(t, k):
        if (t, k) not in flow_cache:
            flow_cache[(t, k)] = flow(grays[t], grays[k])
        return flow_cache[(t, k)]

    def diver_raw(t, keys):
        """Keyframe mask, or flow-compensated blend of the nearest keyframes."""
        if t in keys:
            return keys[t]
        ks = sorted(keys)
        before = [k for k in ks if k < t]
        after = [k for k in ks if k > t]
        parts = []
        if before:
            k = before[-1]
            if t - k <= 3:
                parts.append((1.0 / (t - k), warp_to(keys[k], fl(t, k), (W, H))))
        if after:
            k = after[0]
            if k - t <= 3:
                parts.append((1.0 / (k - t), warp_to(keys[k], fl(t, k), (W, H))))
        if not parts:
            return None
        ws = sum(w for w, _ in parts)
        return sum(w * m for w, m in parts) / ws

    # --- pass 1: raw diver (+snorkel) per frame, flows cached per neighbour pair
    raw: dict[int, np.ndarray] = {}
    t1 = time.time()
    for t in range(max(0, lo - 1), min(n - 1, hi + 1) + 1):
        d = diver_raw(t, diver_k)
        if d is None:
            continue
        s = diver_raw(t, snork_k) if snork_k else None
        raw[t] = d if s is None else np.maximum(d, s)
    print(f"[masks] raw diver masks {len(raw)} frames {time.time()-t1:.0f}s", flush=True)

    # --- pass 2: temporal smoothing + refinement + composition
    wcovs, gate_rows = {}, {}
    stage2 = {}
    decode_idx = {t: None for t in range(lo, hi + 1)}
    for t, fr in enumerate(iter_frames(a.video, info)):
        if t < lo or t > hi:
            continue
        luma = (0.2126 * fr[..., 0] + 0.7152 * fr[..., 1] + 0.0722 * fr[..., 2]).astype(np.float32)
        D = raw.get(t)
        if D is None:
            D = np.zeros(shape, np.float32)
        else:
            acc, ws = 0.5 * D, 0.5
            for k in (t - 1, t + 1):
                if k in raw:
                    acc = acc + 0.25 * warp_to(raw[k], fl(t, k), (W, H))
                    ws += 0.25
            D = acc / ws
            D = refine_diver(D, luma, p.diver_feather_px, p.diver_guided_r, p.diver_guided_eps)
        S = interp_linear(sky_k, t, shape)
        R = np.maximum(interp_linear(rocks_k, t, shape), interp_linear(mount_k, t, shape))
        Wt = interp_linear(water_k, t, shape)
        wcov = float((Wt * (1 - D)).mean() / max((1 - D).mean(), 1e-3))
        wcovs[t] = wcov
        stage2[t] = (D, S, R, Wt)
        if t % 20 == 0:
            print(f"[masks] frame {t} wcov={wcov:.3f} {time.time()-t0:.0f}s", flush=True)
        # write intermediate state so the compose pass can use the smoothed gate
        np.save(fin / f"_tmp_{t:05d}_D.npy", D.astype(np.float16))
        np.save(fin / f"_tmp_{t:05d}_S.npy", S.astype(np.float16))
        np.save(fin / f"_tmp_{t:05d}_R.npy", R.astype(np.float16))
        np.save(fin / f"_tmp_{t:05d}_W.npy", Wt.astype(np.float16))
        del stage2[t]

    # --- sand/water gate: from water coverage, smoothed in time
    ts = sorted(wcovs)
    wc = np.array([wcovs[t] for t in ts], np.float32)
    k = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
    k /= k.sum()
    wcs = np.convolve(np.pad(wc, 9, mode="edge"), k, mode="valid")
    s_ = np.clip((wcs - 0.03) / (0.30 - 0.03), 0, 1)
    gate = 1 - s_ * s_ * (3 - 2 * s_)
    gate_json = {int(t): float(g) for t, g in zip(ts, gate)}
    (fin / "gate.json").write_text(json.dumps({"wcov": {int(t): float(w) for t, w in zip(ts, wc)},
                                               "gate": gate_json}, indent=0))
    for t in ts:
        D = np.load(fin / f"_tmp_{t:05d}_D.npy").astype(np.float32)
        S = np.load(fin / f"_tmp_{t:05d}_S.npy").astype(np.float32)
        R = np.load(fin / f"_tmp_{t:05d}_R.npy").astype(np.float32)
        Wt = np.load(fin / f"_tmp_{t:05d}_W.npy").astype(np.float32)
        w = compose(D, S, R, Wt, gate_json[t], p.region_feather_px)
        for kname, arr in w.items():
            cv2.imwrite(str(fin / kname / f"{t:05d}.png"), np.clip(arr * 255 + .5, 0, 255).astype(np.uint8))
        for nm in "DSRW":
            (fin / f"_tmp_{t:05d}_{nm}.npy").unlink(missing_ok=True)
    print(f"[masks] done frames {lo}-{hi} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
