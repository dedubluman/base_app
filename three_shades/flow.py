"""Optical-flow frame interpolation + automatic speed ramp on the ascent.

Interpolation (for a fractional source position i + t):
  1. Bidirectional flow F0->1 and F1->0 between source frames i and i+1.
  2. Intermediate flows with the Super SloMo approximation (Jiang et al. 2018):
       Ft->0 = -(1-t) t F0->1 + t^2 F1->0
       Ft->1 = (1-t)^2 F0->1 - t (1-t) F1->0
  3. Backward-warp both frames to time t and blend by temporal distance.
     Where the two warps disagree (occlusions: limbs, bubbles, the diver
     leaving frame), use the temporally nearer frame, which prevents ghosting.
The subject mask is warped with the same flows, so the grade's depth
separation stays glued to the diver in slow motion.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import FlowParams, RampParams
from .grade import smoothstep


class FlowEngine:
    def __init__(self, p: FlowParams):
        self.p = p
        algo = p.algorithm.lower()
        if algo == "dis":
            preset = {
                "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
                "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
                "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
            }[p.dis_preset]
            dis = cv2.DISOpticalFlow_create(preset)
            dis.setFinestScale(p.dis_finest_scale)
            dis.setPatchSize(p.dis_patch_size)
            dis.setPatchStride(p.dis_patch_stride)
            dis.setVariationalRefinementIterations(p.dis_var_refine_iters)
            dis.setUseSpatialPropagation(True)
            self._dis = dis
        elif algo in ("deepflow", "sf"):
            if not hasattr(cv2, "optflow"):
                raise RuntimeError(f"'{algo}' needs opencv-contrib-python (cv2.optflow)")
            if algo == "deepflow":
                self._deep = cv2.optflow.createOptFlow_DeepFlow()
        elif algo != "farneback":
            raise ValueError(f"unknown flow algorithm: {p.algorithm}")
        self.algo = algo

    def _prep(self, img: np.ndarray) -> np.ndarray:
        s = self.p.work_scale
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else img
        return (np.clip(small, 0, 1) * 255).astype(np.uint8)

    def flow(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Dense flow a->b at full resolution, (H, W, 2) float32, in pixels."""
        h, w = a.shape[:2]
        A, B = self._prep(a), self._prep(b)
        if self.algo == "sf":
            f = cv2.optflow.calcOpticalFlowSF(
                cv2.cvtColor(A, cv2.COLOR_RGB2BGR), cv2.cvtColor(B, cv2.COLOR_RGB2BGR),
                self.p.sf_layers, self.p.sf_averaging_block, self.p.sf_max_flow)
        else:
            ga, gb = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY), cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
            if self.algo == "dis":
                f = self._dis.calc(ga, gb, None)
            elif self.algo == "deepflow":
                f = self._deep.calc(ga, gb, None)
            else:
                f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 5, 21, 5, 7, 1.5, 0)
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / self.p.work_scale)
        return f.astype(np.float32)


class Interpolator:
    """Synthesizes in-between frames; caches the flow pair of the current interval."""

    def __init__(self, engine: FlowEngine, occlusion_threshold: float):
        self.engine = engine
        self.thr = occlusion_threshold
        self._key: Optional[int] = None
        self._flows: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._grid: Optional[tuple[np.ndarray, np.ndarray]] = None

    def _grid_for(self, h: int, w: int):
        if self._grid is None or self._grid[0].shape != (h, w):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid = (gx, gy)
        return self._grid

    def __call__(self, idx: int, f0: np.ndarray, f1: np.ndarray, m0: np.ndarray,
                 m1: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self._key != idx:
            self._flows = (self.engine.flow(f0, f1), self.engine.flow(f1, f0))
            self._key = idx
        F01, F10 = self._flows  # type: ignore[misc]
        Ft0 = -(1 - t) * t * F01 + t * t * F10
        Ft1 = (1 - t) * (1 - t) * F01 - t * (1 - t) * F10
        gx, gy = self._grid_for(*f0.shape[:2])

        def warp(img, F):
            return cv2.remap(img, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

        w0, w1 = warp(f0, Ft0), warp(f1, Ft1)
        blend = (1 - t) * w0 + t * w1
        err = np.abs(w0 - w1).mean(axis=-1, keepdims=True)
        occluded = smoothstep(self.thr, 2 * self.thr, err)
        nearest = w0 if t < 0.5 else w1
        frame = blend * (1 - occluded) + nearest * occluded

        mw0, mw1 = warp(m0.astype(np.float32), Ft0), warp(m1.astype(np.float32), Ft1)
        mask = (1 - t) * mw0 + t * mw1
        return frame, mask


# ─── Speed ramp ─────────────────────────────────────────────────────────────


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def detect_ascent(masks: np.ndarray, fps: float, p: RampParams) -> Optional[tuple[float, float]]:
    """Find the upward swim from the SAM mask; returns (start, end) source seconds.

    Velocity is the net rise of the mask centroid over ``velocity_window``
    (frame heights / second); short dips are bridged. If the diver is still
    rising when the mask disappears, they swam out of frame, and the ascent
    runs to the end of the clip.
    """
    n, h, _ = masks.shape
    cy = np.full(n, np.nan, dtype=np.float64)
    min_area = 0.002 * masks.shape[1] * masks.shape[2]
    for i, m in enumerate(masks):
        if m.sum() > min_area:
            rows = m.sum(axis=1)
            cy[i] = float((np.arange(h) * rows).sum() / rows.sum())
    valid = ~np.isnan(cy)
    if valid.sum() < 3:
        return None
    cy = np.interp(np.arange(n), np.nonzero(valid)[0], cy[valid])
    cy = _smooth(cy, p.smooth_window)
    # Net rise over a window (robust to limb jitter and noisy mask frames).
    k = max(1, int(round(p.velocity_window * fps / 2)))
    ahead = np.concatenate([cy[k:], np.full(k, cy[-1])])
    behind = np.concatenate([np.full(k, cy[0]), cy[:-k]])
    up = (behind - ahead) / h * fps / (2 * k)
    rising = up > p.min_up_velocity

    # Bridge short dips, then take the longest rising run.
    gap = int(round(p.max_gap * fps))
    runs, cur_start = [], None
    for i in range(n + 1):
        if i < n and rising[i]:
            cur_start = i if cur_start is None else cur_start
        elif cur_start is not None:
            runs.append([cur_start, i - 1])
            cur_start = None
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    best = max(merged, key=lambda r: r[1] - r[0]) if merged else None
    if best is None or (best[1] - best[0] + 1) / fps < p.min_duration:
        return None
    start, end = best
    last_visible = int(np.nonzero(valid)[0].max())
    if end >= last_visible - p.smooth_window:
        # Still rising when the mask ends: the diver swam out through the
        # top of the frame, so keep the slow motion through the exit.
        end = n - 1
    return start / fps, end / fps


def speed_at(t_src: float, start: float, end: float, p: RampParams) -> float:
    """Playback speed at a source time; smooth (C1) ease in and out."""
    s0 = start - p.pre_roll
    down = smoothstep(s0, s0 + p.ramp_in, np.float64(t_src))
    up = smoothstep(end, end + p.ramp_out, np.float64(t_src))
    k = float(down * (1 - up))
    return 1.0 + (p.slow_speed - 1.0) * k


def build_timeline(n_frames: int, fps: float, p: RampParams,
                   ascent: Optional[tuple[float, float]]) -> np.ndarray:
    """Source frame position (float) for every output frame."""
    if not p.enabled or ascent is None:
        return np.arange(n_frames, dtype=np.float64)
    start, end = ascent
    positions, pos = [], 0.0
    while pos <= n_frames - 1 + 1e-9:
        positions.append(pos)
        pos += speed_at(pos / fps, start, end, p)
    return np.asarray(positions)
