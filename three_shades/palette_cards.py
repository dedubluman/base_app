"""Title + three palette cards with eased, staggered motion.

Each card is pre-rendered once (PIL, supersampled for crisp edges) as two
premultiplied RGBA layers: the card body with its soft shadow, and the label
(name + hex). Per frame the layers are only translated/scaled with sub-pixel
precision (cv2.warpAffine) and alpha-composited, so animation is cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import IntroTimings, Layout, Palette

SS = 3  # supersampling factor for drawing

FONT_CANDIDATES = {
    "bold": [
        "/usr/share/fonts/opentype/inter/InterDisplay-SemiBold.otf",
        "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
        "/Library/Fonts/Inter-SemiBold.otf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "medium": [
        "/usr/share/fonts/opentype/inter/Inter-Medium.otf",
        "/Library/Fonts/Inter-Medium.otf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


def _font(kind: str, override: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    for path in ([override] if override else []) + FONT_CANDIDATES[kind]:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# ─── Easing ─────────────────────────────────────────────────────────────────


def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - clamp01(t)) ** 3


def ease_in_cubic(t: float) -> float:
    return clamp01(t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp01(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float, overshoot: float = 1.25) -> float:
    """Lands with a small, soft overshoot: the 'drop' feel."""
    t = clamp01(t) - 1
    return 1 + (overshoot + 1) * t ** 3 + overshoot * t ** 2


# ─── Layers ─────────────────────────────────────────────────────────────────


@dataclass
class Layer:
    rgba: np.ndarray  # float32, premultiplied, (h, w, 4)
    anchor: tuple[float, float]  # where the layer's (0, 0) sits at rest, in frame px


def _premultiplied(img: Image.Image) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def _ink_for(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (12, 28, 33) if y > 0.55 else (246, 250, 249)


def _tracked_text(draw: ImageDraw.ImageDraw, xy, text, font, fill, tracking: float) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def _tracked_width(draw: ImageDraw.ImageDraw, text, font, tracking: float) -> float:
    return sum(draw.textlength(ch, font=font) for ch in text) + tracking * (len(text) - 1)


class IntroOverlay:
    def __init__(self, palette: Palette, layout: Layout, timings: IntroTimings,
                 width: int, height: int):
        self.lay, self.tim, self.w, self.h = layout, timings, width, height
        n = 3
        margin = layout.side_margin_frac * width
        gap = layout.gap_frac * width
        self.card_w = (width - 2 * margin - (n - 1) * gap) / n
        self.card_h = self.card_w * layout.card_aspect
        top = layout.cards_top_frac * height
        self.cards: list[tuple[Layer, Layer]] = []
        for i, sw in enumerate(palette.swatches()):
            x = margin + i * (self.card_w + gap)
            self.cards.append(self._render_card(sw.name, sw.hex.upper(), sw.rgb, x, top))
        self.title = self._render_title(layout.title) if layout.title else None

    # ── rendering ──

    def _render_card(self, name, hex_code, rgb, x, y) -> tuple[Layer, Layer]:
        lay = self.lay
        cw, ch = self.card_w * SS, self.card_h * SS
        pad = int(lay.shadow_blur_frac * self.card_w * SS * 3)
        size = (int(cw + 2 * pad), int(ch + 2 * pad))
        radius = lay.corner_radius_frac * cw
        box = (pad, pad, pad + cw, pad + ch)

        # Body: soft drop shadow + rounded card + hairline inner highlight.
        shadow = Image.new("L", size, 0)
        off = lay.shadow_offset_frac * cw
        ImageDraw.Draw(shadow).rounded_rectangle(
            (box[0], box[1] + off, box[2], box[3] + off), radius, fill=int(255 * lay.shadow_opacity))
        shadow = shadow.filter(ImageFilter.GaussianBlur(lay.shadow_blur_frac * cw))
        body = Image.new("RGBA", size, (0, 0, 0, 0))
        body.putalpha(shadow)
        card = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(card)
        fill = tuple(int(round(c * 255)) for c in rgb)
        d.rounded_rectangle(box, radius, fill=fill + (255,))
        d.rounded_rectangle(box, radius, outline=(255, 255, 255, 46), width=max(1, SS))
        body = Image.alpha_composite(body, card)

        # Label: palette name + hex, bottom-left, ink chosen for contrast.
        label = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(label)
        ink = _ink_for(rgb)
        f_name = _font("bold", lay.font_bold, int(lay.name_size_frac * cw))
        f_hex = _font("medium", lay.font_medium, int(lay.hex_size_frac * cw))
        tp = lay.text_pad_frac * cw
        hex_y = box[3] - tp - f_hex.size
        words = name.split()
        lines = [" ".join(words[:1]), " ".join(words[1:])] if len(words) > 1 else [name]
        line_h = f_name.size * 1.08
        name_y = hex_y - tp * 0.55 - line_h * len(lines)
        for k, line in enumerate(lines):
            _tracked_text(d, (box[0] + tp, name_y + k * line_h), line, f_name,
                          ink + (255,), tracking=-0.01 * f_name.size)
        _tracked_text(d, (box[0] + tp, hex_y), hex_code, f_hex, ink + (175,),
                      tracking=0.08 * f_hex.size)

        def to_layer(img: Image.Image) -> Layer:
            img = img.resize((size[0] // SS, size[1] // SS), Image.LANCZOS)
            return Layer(_premultiplied(img), (x - pad / SS, y - pad / SS))

        return to_layer(body), to_layer(label)

    def _render_title(self, text: str) -> Layer:
        lay = self.lay
        f = _font("bold", lay.font_bold, int(lay.title_size_frac * self.w * SS))
        tracking = 0.06 * f.size
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        tw = _tracked_width(probe, text, f, tracking)
        pad = int(f.size * 0.6)
        size = (int(tw + 2 * pad), int(f.size * 1.4 + 2 * pad))
        shadow = Image.new("L", size, 0)
        _tracked_text(ImageDraw.Draw(shadow), (pad, pad + f.size * 0.06), text, f, 140, tracking)
        shadow = shadow.filter(ImageFilter.GaussianBlur(f.size * 0.12))
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        img.putalpha(shadow)
        fg = Image.new("RGBA", size, (0, 0, 0, 0))
        _tracked_text(ImageDraw.Draw(fg), (pad, pad), text, f, (255, 255, 255, 255), tracking)
        img = Image.alpha_composite(img, fg).resize((size[0] // SS, size[1] // SS), Image.LANCZOS)
        x = (self.w - img.width) / 2
        y = lay.title_y_frac * self.h - img.height / 2
        return Layer(_premultiplied(img), (x, y))

    # ── compositing ──

    def _blit(self, frame: np.ndarray, layer: Layer, dx: float, dy: float,
              scale: float, alpha: float) -> None:
        if alpha <= 1e-3:
            return
        lh, lw = layer.rgba.shape[:2]
        ax, ay = layer.anchor[0] + dx, layer.anchor[1] + dy
        cx, cy = lw / 2, lh / 2  # scale about the layer center
        x0 = int(math.floor(ax + cx - cx * scale)) - 2
        y0 = int(math.floor(ay + cy - cy * scale)) - 2
        bw, bh = int(math.ceil(lw * scale)) + 4, int(math.ceil(lh * scale)) + 4
        M = np.array([[scale, 0, ax + cx - cx * scale - x0],
                      [0, scale, ay + cy - cy * scale - y0]], dtype=np.float32)
        patch = cv2.warpAffine(layer.rgba, M, (bw, bh), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) * alpha
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + bw, self.w), min(y0 + bh, self.h)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        p = patch[fy0 - y0: fy1 - y0, fx0 - x0: fx1 - x0]
        region = frame[fy0:fy1, fx0:fx1]
        region *= 1.0 - p[..., 3:4]
        region += p[..., :3]

    def active(self, t: float) -> bool:
        tim = self.tim
        end = tim.cards_out_start + 2 * tim.card_out_stagger + tim.card_out_dur
        return t < end + 1e-6

    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Composite the intro graphics for output time ``t`` (in place)."""
        tim, lay = self.tim, self.lay
        if not self.active(t):
            return frame

        if self.title is not None:
            a_in = ease_out_cubic((t - tim.title_in) / tim.title_fade)
            a_out = 1 - ease_in_cubic((t - tim.title_out) / tim.title_fade)
            rise = (1 - a_in) * lay.title_rise_frac * self.h
            self._blit(frame, self.title, 0, rise, 1.0, min(a_in, a_out))

        drop = lay.drop_distance_frac * self.h
        for i, (body, label) in enumerate(self.cards):
            t_in = (t - tim.cards_in_start - i * tim.card_stagger) / tim.card_in_dur
            t_out = (t - tim.cards_out_start - i * tim.card_out_stagger) / tim.card_out_dur
            if t_in <= 0:
                continue
            pos = ease_out_back(t_in)  # slide-down drop with a soft landing
            fade = ease_out_cubic(t_in * 1.6)
            leave = ease_in_cubic(t_out)
            dy = -(1 - pos) * drop - leave * drop * 1.4
            scale = lay.start_scale + (1 - lay.start_scale) * ease_out_cubic(t_in)
            scale *= 1 - 0.04 * leave
            alpha = fade * (1 - leave)
            self._blit(frame, body, 0, dy, scale, alpha)
            t_lbl = (t - tim.cards_in_start - i * tim.card_stagger - tim.label_delay) / tim.label_dur
            lbl = ease_out_cubic(t_lbl)
            self._blit(frame, label, 0, dy + (1 - lbl) * self.card_h * 0.04, scale, alpha * lbl)
        return frame


def grade_wipe_alpha(t: float, height: int, tim: IntroTimings) -> Optional[np.ndarray]:
    """Per-row blend weight (H, 1, 1) of the graded image; None = fully graded."""
    p = ease_in_out_cubic((t - tim.grade_wipe_start) / tim.grade_wipe_dur)
    if p >= 1.0:
        return None
    f = tim.grade_wipe_feather
    edge = -f + p * (1 + 2 * f)  # sweeps from above the frame to below it
    y = np.linspace(0, 1, height, dtype=np.float32)
    a = np.clip((edge - y) / f, 0, 1)
    a = a * a * (3 - 2 * a)
    return a[:, None, None]
