"""High-order resampling helpers: re-render RIFE's own flow/occlusion mask with a
Lanczos kernel so interpolated frames keep more fine detail than the bilinear warp."""
from __future__ import annotations
import cv2
import numpy as np


def remap(img: np.ndarray, flow_xy: np.ndarray, interp=cv2.INTER_LANCZOS4) -> np.ndarray:
    """Backward-warp img by flow (2,H,W) in pixels."""
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow_xy[0], gy + flow_xy[1], interp, borderMode=cv2.BORDER_REPLICATE)


def render(a, b, flow, mask, interp=cv2.INTER_LANCZOS4, mask_sharpen: float = 1.0):
    w0 = remap(a, flow[:2], interp)
    w1 = remap(b, flow[2:4], interp)
    m = mask
    if mask_sharpen != 1.0:
        m = 1 / (1 + np.exp(-mask_sharpen * np.log(np.clip(m, 1e-4, 1 - 1e-4) / np.clip(1 - m, 1e-4, 1))))
    m = m[..., None]
    return np.clip(w0 * m + w1 * (1 - m), 0, 1)


def ghost_guard(w0, w1, out, t, thr=0.06, soft=0.10, blur=5.0):
    """Where the two motion-compensated candidates still disagree (misaligned water /
    occlusion, i.e. where blending would ghost or double a structure), fade to the warp
    from the temporally nearer source frame. The weight is blurred so no seam appears."""
    d = np.abs(w0 - w1).max(2)
    d = cv2.GaussianBlur(d, (0, 0), 1.5)
    g = np.clip((d - thr) / soft, 0, 1)
    g = cv2.GaussianBlur(g, (0, 0), blur)[..., None]
    near = w0 if t < 0.5 else w1
    return out * (1 - g) + near * g, float(g.mean())


def restore_detail(out, w0, w1, m, sigma=1.0, ref_sigma=8.0, gmax=1.5, strength=1.0):
    """Interpolated frames are slightly softer than camera frames wherever the two warps are not
    perfectly aligned (rippled water), which makes fine water texture pulse at half the output
    rate when originals alternate with synthesised frames. Re-inject only as much texture energy
    as the aligned source frames locally carry (bounded by gmax) so the pulse disappears
    without inventing structure."""
    def detail(x):
        return x - cv2.GaussianBlur(x, (0, 0), sigma)

    def energy(d):
        return cv2.GaussianBlur(np.abs(d).mean(2), (0, 0), ref_sigma)

    d_out = detail(out)
    e_out = energy(d_out)
    mm = m
    e_ref = mm * energy(detail(w0)) + (1 - mm) * energy(detail(w1))
    g = np.clip(e_ref / (e_out + 1e-4), 1.0, gmax)
    g = 1.0 + strength * (cv2.GaussianBlur(g, (0, 0), 6) - 1.0)
    return np.clip(out + (g[..., None] - 1.0) * d_out, 0, 1), float(g.mean())
