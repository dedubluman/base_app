"""Raw -> graded reveal curve, evaluated per frame index (deterministic)."""

from __future__ import annotations

from fractions import Fraction

from .config import Transition


def ease(t: float, kind: str) -> float:
    t = min(max(t, 0.0), 1.0)
    if kind == "linear":
        return t
    if kind == "cubic_in_out":
        return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2
    return t * t * (3 - 2 * t)  # smoothstep


def grade_mix(frame_idx: int, fps: Fraction, cfg: Transition) -> float:
    """0 = untouched source, 1 = fully graded. Uses the frame's timestamp
    (index / fps), never wall-clock time, so renders are frame-accurate."""
    t = frame_idx / float(fps)
    if t < cfg.raw_hold_end:
        return 0.0
    if cfg.xfade_dur <= 0:
        return 1.0
    return ease((t - cfg.raw_hold_end) / cfg.xfade_dur, cfg.easing)
