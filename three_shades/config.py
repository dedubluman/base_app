"""All tunable parameters of the pipeline, grouped by stage.

Every field can be overridden without touching code via a JSON file passed
with ``--config`` (see ``config.example.json``). Times are in seconds, sizes
that end in ``_frac`` are fractions of the output frame (width for x/horizontal
values, height for y/vertical values) so the layout scales with resolution.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Palette ────────────────────────────────────────────────────────────────
# The swatches describe the TARGET grade, not the raw footage. The same three
# colors drive the split-toning in grade.py, so the cards and the final look
# always agree.


@dataclass
class Swatch:
    name: str
    hex: str

    @property
    def rgb(self) -> tuple[float, float, float]:
        h = self.hex.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class Palette:
    shadow: Swatch = field(default_factory=lambda: Swatch("Abyssal Teal", "#0B3D47"))
    midtone: Swatch = field(default_factory=lambda: Swatch("Golden Sand", "#C9A46A"))
    highlight: Swatch = field(default_factory=lambda: Swatch("Crystal Highlight", "#E3F6F3"))

    def swatches(self) -> list[Swatch]:
        return [self.shadow, self.midtone, self.highlight]


# ─── Intro: title + palette cards (output-timeline seconds) ─────────────────


@dataclass
class IntroTimings:
    # Title ("3 SHADES METHOD"). Set Layout.title to "" to disable it.
    title_in: float = 0.20
    title_out: float = 1.10
    title_fade: float = 0.35

    # Cards drop in one after another.
    cards_in_start: float = 1.50
    card_stagger: float = 0.20  # delay between card 1, 2 and 3
    card_in_dur: float = 0.80  # fade-in + slide-down duration of one card
    label_delay: float = 0.30  # label fades in this long after its card starts
    label_dur: float = 0.45

    # Cards leave (slide up + fade), staggered.
    cards_out_start: float = 4.40
    card_out_stagger: float = 0.10
    card_out_dur: float = 0.60

    # Flat -> graded wipe (top to bottom, feathered), starts as cards leave.
    grade_wipe_start: float = 4.50
    grade_wipe_dur: float = 1.20
    grade_wipe_feather: float = 0.35  # soft edge, fraction of frame height

    # Fade to black at the very end of the reel (0 disables).
    outro_fade: float = 0.60


@dataclass
class Layout:
    title: str = "3 SHADES METHOD"
    title_y_frac: float = 0.115  # vertical center of the title
    title_size_frac: float = 0.058  # font size, fraction of frame width
    title_rise_frac: float = 0.012  # title drifts up by this much while fading in

    cards_top_frac: float = 0.085  # top edge of the card row (safe area for Reels/TikTok)
    side_margin_frac: float = 0.065
    gap_frac: float = 0.030
    card_aspect: float = 1.28  # height / width
    corner_radius_frac: float = 0.11  # of card width
    drop_distance_frac: float = 0.055  # how far above its rest position a card starts
    start_scale: float = 0.94  # cards grow to 1.0 while dropping
    shadow_blur_frac: float = 0.06  # of card width
    shadow_opacity: float = 0.45
    shadow_offset_frac: float = 0.025  # of card width

    name_size_frac: float = 0.085  # font size, fraction of card width
    hex_size_frac: float = 0.066
    text_pad_frac: float = 0.085  # inner padding, fraction of card width

    font_bold: Optional[str] = None  # path to a .ttf/.otf; None = auto (Inter, then DejaVu)
    font_medium: Optional[str] = None


# ─── Grade ──────────────────────────────────────────────────────────────────


@dataclass
class GradeParams:
    # Neutralize the green underwater cast (0 = off, 1 = full gray-world).
    wb_strength: float = 0.70
    # Auto levels from percentiles of the whole clip (stable over time).
    black_percentile: float = 0.5
    white_percentile: float = 99.6
    levels_strength: float = 0.65  # 1 = full stretch to the percentiles, 0 = off
    # Filmic S-curve.
    contrast: float = 1.18
    pivot: float = 0.42
    highlight_knee: float = 0.82  # soft roll-off above this value

    # Split-toning toward the palette (0 = off).
    shadow_tint: float = 0.45  # chroma of Abyssal Teal added to shadows
    shadow_lift: float = 0.18  # blacks are lifted INTO the teal (rich, not crushed)
    midtone_tint: float = 0.10  # Golden Sand chroma in the midtones
    highlight_tint: float = 0.35  # Crystal Highlight chroma in the highlights
    highlight_mix: float = 0.10

    # Warm/cool separation: sand and skin toward Golden Sand, water toward teal.
    sand_boost: float = 0.45  # how strongly sand/skin are pulled to the Golden Sand hue
    sand_chroma: float = 0.12  # saturation of that target (caps how orange it can get)
    water_teal: float = 0.22
    saturation: float = 1.04
    vibrance: float = 0.22

    # Sun rays / surface glow.
    bloom: float = 0.18
    bloom_threshold: float = 0.72
    bloom_radius_frac: float = 0.02  # of frame width

    # Subject (SAM 3.1 mask) vs background: depth separation.
    mask_feather_frac: float = 0.006  # of frame width
    subject_wb_strength: float = 0.85  # diver-only white balance (needs the mask)
    subject_teal_reduction: float = 0.65  # how much less teal toning the diver gets
    skin_restore: float = 0.55  # pull of the diver's midtones toward a Golden Sand skin tone
    skin_chroma: float = 0.11  # saturation of that skin tone
    subject_exposure: float = 1.07
    subject_warmth: float = 0.15  # keep skin warm against the teal water
    subject_detail: float = 0.25  # local-contrast boost on the diver
    bg_exposure: float = 0.93
    bg_desaturate: float = 0.08
    bg_blur_frac: float = 0.0016  # gentle depth-of-field on the background
    vignette: float = 0.28


# ─── Optical flow + speed ramp ──────────────────────────────────────────────


@dataclass
class FlowParams:
    # "dis" (default, fast & robust), "deepflow" and "sf" (calcOpticalFlowSF)
    # need opencv-contrib-python; "farneback" is in plain OpenCV.
    algorithm: str = "dis"
    work_scale: float = 0.5  # flow is computed at this scale, then upsampled
    dis_preset: str = "medium"  # ultrafast | fast | medium
    dis_finest_scale: int = 1
    dis_patch_size: int = 12
    dis_patch_stride: int = 4
    dis_var_refine_iters: int = 5
    sf_layers: int = 3
    sf_averaging_block: int = 2
    sf_max_flow: int = 8
    # Pixels where the two warped frames disagree by more than this (0-1 RGB)
    # are treated as occluded and taken from the temporally nearer frame,
    # which removes ghosting around fast limbs and bubbles.
    occlusion_threshold: float = 0.06


@dataclass
class RampParams:
    enabled: bool = True
    slow_speed: float = 0.35  # playback speed during the ascent (0.35 = ~2.9x slow-mo)
    ramp_in: float = 0.50  # source seconds to ease from 1.0x down to slow_speed
    ramp_out: float = 0.50  # source seconds to ease back to 1.0x
    pre_roll: float = 0.25  # start slowing this long before the ascent begins

    # Automatic ascent detection from the subject mask.
    min_up_velocity: float = 0.05  # frame heights / second
    velocity_window: float = 0.50  # seconds; velocity = net rise over this window
    max_gap: float = 0.40  # seconds; dips shorter than this don't split the ascent
    min_duration: float = 0.60  # seconds
    smooth_window: int = 9  # frames

    # Manual override (source seconds). Set both to bypass auto-detection.
    start: Optional[float] = None
    end: Optional[float] = None


# ─── Segmentation ───────────────────────────────────────────────────────────


@dataclass
class SegParams:
    # "sam3.1": SAM 3.1 Object Multiplex video predictor (needs a CUDA GPU).
    # "maskrcnn-preview": torchvision Mask R-CNN, CPU-friendly preview only.
    backend: str = "sam3.1"
    checkpoint: Optional[str] = "checkpoints/sam3.1/sam3.1_multiplex.pt"
    text_prompt: str = "person"
    # Optional positive click (x, y in output pixels) on frame 0 that selects
    # the diver; otherwise the largest tracked person is chosen.
    subject_point: Optional[list[float]] = None
    prompt_frame: int = 0
    jpeg_quality: int = 95


# ─── Export ─────────────────────────────────────────────────────────────────


@dataclass
class ExportParams:
    width: int = 1080  # 9:16 vertical
    height: int = 1920
    fps: float = 30.0
    codec: str = "libx264"  # or "libx265"
    crf: int = 16
    preset: str = "slow"
    dither: bool = True  # breaks up banding in deep-water gradients
    music: Optional[str] = None  # optional audio track, faded out at the end
    music_fade: float = 1.0


@dataclass
class Config:
    palette: Palette = field(default_factory=Palette)
    intro: IntroTimings = field(default_factory=IntroTimings)
    layout: Layout = field(default_factory=Layout)
    grade: GradeParams = field(default_factory=GradeParams)
    flow: FlowParams = field(default_factory=FlowParams)
    ramp: RampParams = field(default_factory=RampParams)
    seg: SegParams = field(default_factory=SegParams)
    export: ExportParams = field(default_factory=ExportParams)


def _update(obj, data: dict, path: str = "") -> None:
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Unknown config key: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def load_config(path: Optional[str | Path]) -> Config:
    cfg = Config()
    if path:
        _update(cfg, json.loads(Path(path).read_text()))
    return cfg


def dump_config(cfg: Config) -> str:
    return json.dumps(dataclasses.asdict(cfg), indent=2)
