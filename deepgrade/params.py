"""All tunable numbers of the deep-water grade (dumped next to every render).

Colour anchors (OKLab-measured):  Stratos #070F54  L .228 h 268 C .122
                                  Mocha   #4F5733  L .440 h 119 C .056
                                  Silver  #BFC1C2  L .810 neutral
Reference photographs (measured on the matched frames 0 and 117):
    water  L .24-.30  h ~250  C ~.067  (source L .49-.56, h ~210)
    skin   L .57      h ~85   C ~.026  (source L .66, h ~177 = cyan cast)
    shadow floor L ~.19
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Params:
    # ---------------- anchors
    stratos: str = "#070F54"
    mocha: str = "#4F5733"
    silver: str = "#BFC1C2"

    # ---------------- open water (Stratos navy-indigo)
    water_L_mid: float = 0.265        # target L of the median water pixel
    water_L_slope: float = 1.45       # amplification of gradients / texture around the anchor
    water_L_follow: float = 0.55      # share of the source's depth-driven brightness drift kept
    water_L_floor: float = 0.17       # soft floor: shadows stay open
    water_hue_deep: float = 266.0     # near Stratos (h 268)
    water_hue_mid: float = 252.0      # reference water (h ~250)
    water_hue_light: float = 236.0    # lighter water drifts cool
    water_C_deep: float = 0.088
    water_C_mid: float = 0.070
    water_C_light: float = 0.050
    water_detail_chroma: float = 0.35 # how much high-frequency source chroma survives

    # ---------------- seabed sand (Mocha earth)
    sand_L_top: float = 0.245         # far / top of frame (haze, darker)
    sand_L_bottom: float = 0.355      # near / bottom of frame
    sand_L_slope: float = 1.5
    sand_local_gain: float = 1.9      # micro-contrast on texture (reference ~3x)
    sand_hue_shadow: float = 160.0    # dark sand leans to the water
    sand_hue_mocha: float = 110.0     # Mocha h 119 (a touch browner)
    sand_hue_light: float = 92.0
    sand_C_shadow: float = 0.030
    sand_C_mocha: float = 0.040       # Mocha C .056, restrained
    sand_C_light: float = 0.020
    sand_detail_chroma: float = 0.30
    sand_mocha_top: float = 0.18      # share of the Mocha palette at the far/top (rest = water haze)
    sand_mocha_bottom: float = 0.82   # ... at the near/bottom of the frame
    rock_mocha: float = 0.90          # cap of the Mocha share on rocks
    rock_mocha_extra: float = 0.32    # rocks are this much earthier than the sand around them

    # ---------------- rocks / coral / land
    rock_L_delta: float = 0.022       # slightly lighter than sand so texture reads
    rock_local_gain: float = 2.5
    rock_C_boost: float = 1.0 

    # ---------------- sky / above water
    sky_L_gain: float = 0.74          # compresses the bright pastel sky
    sky_L_offset: float = -0.02
    sky_chroma_keep: float = 0.70     # keep some of the warm horizon glow
    sky_hue_cool: float = 236.0
    sky_C_cool: float = 0.026

    # ---------------- diver
    skin_target_L: float = 0.585      # reference lit skin .57, source .66
    skin_L_slope: float = 1.45        # curve slope around the skin anchor
    skin_L_shoulder: float = 0.85     # highlights roll off above this L
    skin_hue: float = 72.0            # warm, restrained (natural skin ~55-70)
    skin_hue_pull: float = 0.55       # share of the way the hue moves to skin_hue
    skin_C_target: float = 0.034      # ref .026, natural .05-.07: subtle warmth
    skin_C_min: float = 0.020
    skin_C_max: float = 0.070
    skin_local_gain: float = 1.25
    gear_protect: float = 0.9         # dark gear / white shorts keep their own colour

    # ---------------- highlights / particles (Silver Sand)
    particle_keep: float = 0.92       # fraction of source brightness kept for particles
    particle_neutral: float = 0.80    # how far particle chroma moves to silver
    particle_scale_px: int = 11       # top-hat structuring element
    particle_lo: float = 0.020
    particle_hi: float = 0.075
    highlight_start: float = 0.66     # L0 where highlights are treated as light (silver)
    highlight_end: float = 0.88
    shoulder_knee: float = 0.68       # global highlight shoulder: no clipping, silver cap
    shoulder_top: float = 0.845       # Silver Sand L .81 + a little headroom

    # ---------------- depth / atmosphere
    vignette: float = 0.08
    vignette_power: float = 2.2
    saturation: float = 1.0

    # ---------------- masks
    diver_feather_px: float = 1.2
    diver_guided_r: int = 5
    diver_guided_eps: float = 4e-4
    region_feather_px: float = 12.0
    temporal_sigma_keyframes: float = 0.8

    def dump(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: Path) -> "Params":
        d = json.loads(Path(path).read_text())
        return Params(**d)
