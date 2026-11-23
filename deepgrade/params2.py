"""Parameters of grade v2: one shared underwater base look + natural above-water path.

Everything not listed here is inherited from ``params.Params`` (colour anchors, skin, palette ramps).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .params import Params


@dataclass
class Params2(Params):
    # ---- shared underwater base: ONE tone map + palette for water, sand and the unclaimed remainder
    water_L_follow: float = 0.30      # share of the source's slow depth-driven brightness drift kept
    depth_grad: float = 0.040         # top-of-frame darkening on shadows/mids (constant, all frames)
    anchor_sigma_frames: float = 10.0 # temporal Gaussian on the background anchor (frames)
    skin_sigma_frames: float = 6.0

    # ---- seabed = restrained earth on top of the base (never a separate colour regime)
    sand_L_offset: float = 0.004      # tiny lift so sand reads as a surface, not as water
    sand_base_earth: float = 0.04     # constant Mocha share on sand (hue ~221, the reference's sand)
    sand_earth_max: float = 0.55      # max Mocha share on lit sand texture
    sand_earth_lo: float = 0.003      # texture level (band-pass, source L units) where the tint starts
    sand_earth_hi: float = 0.012      # ... and where it reaches its maximum
    rock_mocha: float = 0.88          # constant Mocha share on rocks (hazes out with the gate)
    rock_L_delta: float = 0.050       # rocks sit a little above the water base so texture reads
    rock_local_gain: float = 3.2      # micro-contrast on rocks (reference ~4x, source 1x)
    sand_local_gain: float = 2.4
    rock_C_boost: float = 1.05
    rock_mocha_extra: float = 0.0     # unused in v2 (rocks use rock_mocha directly)

    # ---- above-water (sky / land / sea): natural, protected from the underwater grade
    air_lo: float = 0.03              # sky+land frame fraction where 'above water' starts
    air_hi: float = 0.25              # ... and where it is complete
    sea_natural_share: float = 0.50   # share of the natural look kept on the sea once above water
    nat_black: float = 0.012          # natural path: black lift
    nat_gamma: float = 1.06           # natural path: gentle mid density
    nat_sat: float = 0.96
    nat_knee: float = 0.80            # highlight shoulder for sky/land (natural)
    nat_top: float = 0.92
    nat_local_gain: float = 1.20      # land micro-contrast
    nat_hill_dim: float = 0.0

    @staticmethod
    def load(path: Path) -> "Params2":
        return Params2(**json.loads(Path(path).read_text()))
