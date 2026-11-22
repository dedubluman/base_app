"""Mask alignment for retimed footage.

The retime map (`retime_map.json`) gives every output frame's exact source position. For
output frames that fall between two source frames, masks (SAM 3.1 or otherwise) must NOT be
re-used from the nearest frame: pass them as `aux` to `slowmo_overunder.synth`, which warps
them with the same optical flow and occlusion weights as the picture. For exact source
frames use the mask at that frame index directly. Where the ghost guard fired heavily
(stats['guard_cov'] high) re-run SAM on the synthesised frame instead."""
from __future__ import annotations
import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0
