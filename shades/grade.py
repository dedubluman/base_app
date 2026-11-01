"""Cinematic underwater grade.

The grade is a pure per-pixel colour function, so it is baked into a 3D LUT:
the render applies the LUT (fast, identical everywhere) and the same LUT is
exported as a .cube file for Resolve / Premiere / ffmpeg `lut3d`.

The look is defined by the target palette (shadow / mid / highlight anchors),
never by the raw pixels: the raw footage only supplies its black/white points
so the LUT can stretch this clip's flat range before shaping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Grade, Swatch

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # BT.709


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def rgb_to_hex(rgb: np.ndarray) -> str:
    v = np.clip(np.round(np.asarray(rgb) * 255), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*v)


@dataclass
class SourceStats:
    """Per-channel black/white points of the raw clip (robust percentiles)."""

    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)

    @classmethod
    def from_frames(cls, frames: list[np.ndarray]) -> "SourceStats":
        px = np.concatenate([f[::4, ::4].reshape(-1, 3) for f in frames])
        return cls(np.percentile(px, 0.5, axis=0).astype(np.float32),
                   np.percentile(px, 99.7, axis=0).astype(np.float32))


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-6
    rc = np.where(m, (mx - r) / np.where(m, d, 1), 0)
    gc = np.where(m, (mx - g) / np.where(m, d, 1), 0)
    bc = np.where(m, (mx - b) / np.where(m, d, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(m, (h / 6.0) % 1.0, 0.0)
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.astype(np.int32) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


def _hue_dist(h, center):
    """Signed shortest distance on the hue circle (units of turns)."""
    return (h - center + 0.5) % 1.0 - 0.5


def apply_grade(rgb: np.ndarray, g: Grade, palette: list[Swatch],
                stats: SourceStats) -> np.ndarray:
    """rgb: float32 [..., 3] in 0..1 (BT.709 gamma-encoded). Returns graded."""
    src = rgb.astype(np.float32)
    x = src

    # 1) Levels from the clip's own black/white points. Underwater the red
    #    channel is squeezed into a narrow band; a per-channel stretch restores
    #    the colour separation the water absorbed. `channel_restore` mixes
    #    between a common stretch (0) and a full per-channel stretch (1): a
    #    full one over-amplifies red and turns bright surface ripples copper.
    lo = stats.lo.mean() + (stats.lo - stats.lo.mean()) * g.channel_restore
    hi = stats.hi.mean() + (stats.hi - stats.hi.mean()) * g.channel_restore
    x = (x - lo) / np.maximum(hi - lo, 1e-3)
    x = np.clip(x, 0.0, 1.2)

    # 2) White balance against the residual green/cyan cast. The gains fade
    #    out toward white so highlights stay clean instead of tinting warm.
    lum = (x * LUMA).sum(-1, keepdims=True)
    roll = 1.0 - _smoothstep(0.55, 1.0, lum) * g.highlight_rolloff
    gains = 1.0 + (np.asarray(g.wb_gains, np.float32) - 1.0) * g.green_cast_removal
    x = x * (1.0 + (gains - 1.0) * roll)

    # 3) Output levels + S-curve contrast on luminance (hue-preserving), plus
    #    a midtone gamma that sinks the water for a deeper, moodier base.
    x = g.black_point + np.clip(x, 0, None) * (g.white_point - g.black_point)
    lum = np.maximum((x * LUMA).sum(-1, keepdims=True), 1e-5)
    lum_g = lum ** g.mid_gamma
    p = g.pivot
    lo_part = p * (np.clip(lum_g, 0, p) / p) ** g.contrast
    hi_part = 1 - (1 - p) * ((1 - np.clip(lum_g, p, 1)) / (1 - p)) ** g.contrast
    new_lum = np.where(lum_g < p, lo_part, hi_part)
    x = x * (new_lum / lum)

    # 4) Hue work in HSV: rotate green-cyan water toward teal/blue, calm the
    #    greens, enrich warm sand/skin hues while protecting them from magenta.
    h, s, v = _rgb_to_hsv(np.clip(x, 0, None))
    green_w = np.exp(-(_hue_dist(h, 0.42) / 0.09) ** 2)   # ~150 deg
    h = (h + green_w * g.green_to_teal_shift / 360.0) % 1.0
    s = s * (1 - g.green_desat * green_w * 0.5)
    # warm band ~15-75 deg: skin, sand, and the olive/khaki rock the water
    # leaves behind (those are pulled toward the sand hue below).
    warm_w = np.exp(-(_hue_dist(h, g.warm_center / 360) / (g.warm_width / 360)) ** 2)
    # Pull warm hues toward the Golden Sand anchor hue so rock reads tan, not
    # rust; boost their saturation only in the mid-tones (bright warm pixels
    # are usually surface reflections and must stay clean).
    sand_h = _rgb_to_hsv(hex_to_rgb([sw.hex for sw in palette if sw.role == "mid"][0]))[0]
    h = (h + _hue_dist(sand_h, h) * warm_w * g.warm_hue_pull) % 1.0
    hi_w = _smoothstep(0.55, 0.9, v)
    s = s * (1 + (g.warm_saturation - 1) * warm_w * (1 - hi_w))
    s = s * (1 - g.highlight_desat * hi_w * warm_w)
    # violet/magenta guard: hues between ~250 and ~345 deg are pulled back
    # toward blue (water) and their saturation reduced, so no lavender water.
    vio_w = np.exp(-(_hue_dist(h, 0.80) / 0.08) ** 2)
    h = (h + g.warm_protect * vio_w * _hue_dist(0.60, h) * 0.5) % 1.0
    s = s * (1 - 0.5 * g.warm_protect * vio_w)
    # calm the cyan-blue water so it reads deep rather than electric
    water_w = np.exp(-(_hue_dist(h, 0.55) / 0.08) ** 2)
    s = s * (1 - (1 - g.water_saturation) * water_w)
    s = np.clip(s * g.saturation, 0, 1)
    x = _hsv_to_rgb(h, s, v)

    # 5) Split toning toward the palette anchors. Weights come from luminance
    #    so shadows pick up the Abyssal Teal, highlights the Crystal Highlight,
    #    and warm mid-tones are nudged toward Golden Sand.
    anchors = {sw.role: hex_to_rgb(sw.hex) for sw in palette}
    lum = (x * LUMA).sum(-1, keepdims=True)
    w_sh = (1 - _smoothstep(0.05, 0.45, lum)) * g.shadow_tint_amount
    w_hi = _smoothstep(0.62, 0.95, lum) * g.highlight_tint_amount
    w_mid = (_smoothstep(0.22, 0.45, lum) * (1 - _smoothstep(0.62, 0.85, lum))
             * warm_w[..., None] * g.mid_tint_amount)

    def tint(x, anchor, w):
        # Move chroma toward the anchor while keeping the pixel's luminance,
        # so tinting never crushes or lifts exposure.
        a_l = float((anchor * LUMA).sum())
        target = anchor * (lum / max(a_l, 1e-4))
        return x + (target - x) * w

    # Lift warm mid-tones (sand / rock / skin) a little so they glow against
    # the deep water instead of sinking with it.
    x = x * (1 + g.warm_lift * (w_mid / max(g.mid_tint_amount, 1e-4)))
    x = tint(x, anchors["shadow"], w_sh)
    x = tint(x, anchors["mid"], w_mid)
    x = tint(x, anchors["highlight"], w_hi)

    x = np.clip(x, 0.0, 1.0)
    return src + (x - src) * g.grade_strength


# --------------------------------------------------------------------------
# LUT bake / apply / export
# --------------------------------------------------------------------------

def bake_lut(g: Grade, palette: list[Swatch], stats: SourceStats) -> np.ndarray:
    n = g.lut_size
    ax = np.linspace(0, 1, n, dtype=np.float32)
    # index order [r, g, b]
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1)
    return apply_grade(grid.reshape(-1, 3), g, palette, stats).reshape(n, n, n, 3)


def apply_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear 3D-LUT lookup. rgb float32 [H,W,3] in 0..1."""
    n = lut.shape[0]
    p = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.minimum(p.astype(np.int32), n - 2)
    f = p - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c000 = lut[r0, g0, b0]; c100 = lut[r0 + 1, g0, b0]
    c010 = lut[r0, g0 + 1, b0]; c110 = lut[r0 + 1, g0 + 1, b0]
    c001 = lut[r0, g0, b0 + 1]; c101 = lut[r0 + 1, g0, b0 + 1]
    c011 = lut[r0, g0 + 1, b0 + 1]; c111 = lut[r0 + 1, g0 + 1, b0 + 1]
    c00 = c000 + (c100 - c000) * fr; c10 = c010 + (c110 - c010) * fr
    c01 = c001 + (c101 - c001) * fr; c11 = c011 + (c111 - c011) * fr
    c0 = c00 + (c10 - c00) * fg; c1 = c01 + (c11 - c01) * fg
    return c0 + (c1 - c0) * fb


def write_cube(lut: np.ndarray, path: Path, title: str) -> None:
    n = lut.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {n}",
             "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    # .cube order: red changes fastest, then green, then blue
    for b in range(n):
        for gi in range(n):
            for r in range(n):
                c = lut[r, gi, b]
                lines.append(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")
