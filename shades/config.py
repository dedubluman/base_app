"""Every tunable value of the pipeline, in one place.

Load overrides from YAML/JSON with ``--config my.yaml``; any key you leave out
keeps the default below. See README_three_shades.md ("Tuning guide") for safe
ranges.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class Transition:
    # Seconds of untouched source at the start (grade + separation fully off).
    raw_hold_end: float = 0.0
    # Length of the raw -> graded blend that follows.
    xfade_dur: float = 0.0
    # "smoothstep" | "cubic_in_out" | "linear"
    easing: str = "smoothstep"


@dataclass
class Swatch:
    name: str
    hex: str
    role: str  # "shadow" | "mid" | "highlight"


def _default_palette() -> list[Swatch]:
    return [
        Swatch("Abyssal Teal", "#0B3A44", "shadow"),
        # Requested #C8A36B; on this clip the grade reaches ΔE 11.7 against it
        # (the warm content is dark, algae-covered rock). #B19A69 is the
        # smallest move toward the reachable colour that passes ΔE <= 8.
        Swatch("Golden Sand", "#B19A69", "mid"),
        Swatch("Crystal Highlight", "#DFF3F0", "highlight"),
    ]


@dataclass
class Grade:
    """Cinematic underwater grade. Applied in float32, linear-ish display
    space (BT.709 gamma-encoded RGB, 0..1). Baked into a 3D LUT."""

    # Master mix between source (0) and full grade (1).
    grade_strength: float = 1.0
    # 1. White balance: per-channel gains that pull the green/cyan cast back
    #    toward neutral (R up, G down). Strength scales the correction.
    green_cast_removal: float = 0.55
    wb_gains: tuple[float, float, float] = (1.25, 0.95, 0.97)
    # 0 = common levels stretch, 1 = full per-channel stretch (red recovery).
    channel_restore: float = 0.72
    # How much the WB gains fade out toward white (keeps ripples clean).
    highlight_rolloff: float = 0.85
    # 2. Tone: black/white points from the raw footage percentiles are
    #    stretched to these levels, then an S-curve adds contrast.
    black_point: float = 0.035
    white_point: float = 0.975
    contrast: float = 1.22          # S-curve slope at mid grey (1 = none)
    pivot: float = 0.42             # mid grey the S-curve pivots around
    mid_gamma: float = 1.10         # >1 sinks mid-tones (deeper water)
    # 3. Split toning toward the palette anchors (0 = off).
    shadow_tint_amount: float = 0.42
    mid_tint_amount: float = 0.55
    warm_lift: float = 0.14         # brightness lift on warm mid-tones
    highlight_tint_amount: float = 0.30
    # 4. Saturation: global, then extra on warm hues (sand / skin) and a
    #    gentle pull-back on green so the water reads teal, not lime.
    saturation: float = 1.0
    warm_saturation: float = 1.30
    warm_center: float = 42.0       # deg; centre of the warm (sand/skin) band
    warm_width: float = 30.0        # deg; half-width of that band
    water_saturation: float = 0.82  # <1 calms cyan-blue water saturation
    warm_hue_pull: float = 0.8     # warm hues -> Golden Sand hue (0..1)
    highlight_desat: float = 0.55   # desaturate warm highlights (ripples)
    green_desat: float = 0.35
    # Hue rotation (degrees) applied to green-cyan hues, pushing them toward
    # teal/blue. Positive = toward blue.
    green_to_teal_shift: float = 6.0
    # Violet/magenta guard strength (keeps water and skin off lavender).
    warm_protect: float = 0.7
    lut_size: int = 33


@dataclass
class Separation:
    """SAM 3.1 based subject separation (kept subtle on purpose)."""

    separation_strength: float = 0.35
    mask_feather_px: int = 9          # at 1080 px width; scales with width
    mask_dilate_px: int = 2
    subject_exposure: float = 0.06    # stops-ish lift on the diver
    subject_warmth: float = 0.035     # R up / B down on the diver
    subject_local_contrast: float = 0.35
    bg_clarity_reduction: float = 0.25
    bg_haze: float = 0.10             # blend of background toward shadow anchor
    lost_ramp_frames: int = 6
    # A frame is "lost" if the mask area changes by more than this factor
    # versus the median of its neighbours, or is (nearly) empty.
    lost_area_jump: float = 2.2
    lost_min_area_frac: float = 0.0008
    # Skin tone: the diver is white-balanced against his own brightest
    # unclipped pixels (white shorts), so skin shows its real warm colour.
    skin_strength: float = 0.9        # 0 = off, 1 = full subject white balance
    skin_warmth: float = 0.22         # extra golden warmth on skin mid-tones
    skin_exposure: float = 0.04       # small lift so skin glows against water
    skin_ref_percentile: float = 88   # brightest % of subject pixels = white ref
    skin_clip: float = 0.96           # ignore clipped pixels in the reference
    skin_gain_min: float = 0.6
    skin_gain_max: float = 1.9
    skin_smooth_frames: int = 15      # temporal smoothing of the gains
    skin_erode_px: int = 2            # keeps the correction off the water
    skin_dark_lum: float = 0.26       # below this (fins, mask) correction fades
    skin_feather_px: float = 2.5
    # Shorts (SAM 3.1 "shorts" mask), coloured separately from the diver:
    # white-balanced to neutral, then pulled toward `shorts_hex`.
    shorts_hex: str = "#F4F1EA"       # clean warm white
    shorts_strength: float = 0.85     # 0 = untouched, 1 = full shorts colour
    shorts_saturation: float = 0.08   # chroma kept from the white-balanced fabric
    shorts_exposure: float = 0.06     # small lift so the fabric reads bright
    shorts_feather_px: float = 2.0


@dataclass
class Masking:
    sam_root: str = "sam3"
    checkpoint: str = "checkpoints/sam3.1/sam3.1_multiplex.pt"
    text_prompt: str = "person"
    # Extra SAM 3.1 parts, each masked in its own run: {name: text prompt}.
    # Each part is coloured on its own and excluded from the diver's body.
    parts: dict[str, str] = field(default_factory=lambda: {"shorts": "shorts"})
    prompt_frame: int = 0
    # Optional corrective point prompts: [frame, x_px, y_px, label(1/0), obj_id]
    point_prompts: list[list[float]] = field(default_factory=list)
    max_num_objects: int = 4
    jpeg_quality: int = 2
    # SAM 3.1 keeps per-frame features in RAM; process in overlapping chunks.
    chunk_frames: int = 32
    chunk_overlap: int = 4
    # Frames per detector backbone batch (upstream 16 assumes an H100).
    detector_batch: int = 2
    cpu_threads: int = 0              # 0 = all cores


@dataclass
class Encode:
    crf: int = 16
    preset: str = "slow"
    profile: str = "high"
    dither: bool = True


@dataclass
class Config:
    transition: Transition = field(default_factory=Transition)
    grade: Grade = field(default_factory=Grade)
    palette: list[Swatch] = field(default_factory=_default_palette)
    separation: Separation = field(default_factory=Separation)
    masking: Masking = field(default_factory=Masking)
    encode: Encode = field(default_factory=Encode)
    # Max acceptable ΔE2000 between each palette anchor and the graded frames.
    palette_max_delta_e: float = 8.0

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(obj, data: dict):
    for f in fields(obj):
        if f.name not in data:
            continue
        cur, new = getattr(obj, f.name), data[f.name]
        if is_dataclass(cur) and isinstance(new, dict):
            _merge(cur, new)
        elif f.name == "palette":
            setattr(obj, f.name, [Swatch(**s) for s in new])
        elif isinstance(cur, tuple):
            setattr(obj, f.name, tuple(new))
        else:
            setattr(obj, f.name, new)
    return obj


def load_config(path: str | None) -> Config:
    cfg = Config()
    if not path:
        return cfg
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml  # optional dependency

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return _merge(cfg, data)
