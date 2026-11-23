"""Run all SAM 3.1 segmentation jobs for one clip.

    python -m deepgrade.run_sam --video inputs/player_export.mov --stages preview,diver,stuff
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .sam_runner import Sam31, prepare_frames

REPO = Path(__file__).resolve().parent.parent

# Preview keyframes (all multiples of 6 -> in every stride grid).
PREVIEW = [0, 30, 60, 78, 108, 150, 186, 204, 228]

# prompt -> (detector threshold, output prob thresh, keep prob)
STUFF = {
    "rocks": dict(prompt="rocks", det=0.1, out_prob=0.05, keep_prob=0.15),
    "water": dict(prompt="water", det=0.1, out_prob=0.05, keep_prob=0.10),
    "sky":   dict(prompt="sky",   det=0.4, out_prob=0.5,  keep_prob=0.5),
}


# Extra classes run after the main stages: land (hills) in the above-water finale,
# and the snorkel, which SAM's "person" masklet leaves out once it clears the head.
EXTRA = {
    "mountain": dict(prompt="mountain", det=0.1, out_prob=0.05, keep_prob=0.12, min_area=0.002),
    "snorkel":  dict(prompt="snorkel",  det=0.3, out_prob=0.3,  keep_prob=0.30, min_area=0.0003),
}


def stuff_frames(name: str, n: int, stride: int) -> list[int]:
    lo, hi = {"rocks": (0, 120), "water": (45, n - 1), "sky": (165, n - 1),
              "mountain": (186, n - 1), "snorkel": (60, n - 1)}[name]
    fr = list(range(lo, hi + 1, stride))
    if name != "rocks" and fr[-1] != n - 1:
        fr.append(n - 1)
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--stages", default="preview,diver,stuff")
    ap.add_argument("--stuff-stride", type=int, default=3)
    ap.add_argument("--diver-stride", type=int, default=2)
    a = ap.parse_args()

    video = a.video if a.video.is_absolute() else REPO / a.video
    cache = REPO / ".cache"
    cache.mkdir(exist_ok=True)
    mdir = REPO / "masks" / video.stem
    mdir.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)

    frames, n, w, h = prepare_frames(video, cache, log)
    sam = Sam31(REPO, cache, frames, log)
    stages = a.stages.split(",")
    t0 = time.time()
    if "preview" in stages:
        # single-frame detections on the preview keyframes: used for grade tuning
        # and later as an independent cross-check of the tracked diver masks.
        sam.detect_frames(mdir, "diver_single", "person", PREVIEW,
                          det=0.4, out_prob=0.5, keep_prob=0.5)
        for name, c in STUFF.items():
            fr = [f for f in PREVIEW if stuff_frames(name, n, a.stuff_stride)[0] <= f
                  <= stuff_frames(name, n, a.stuff_stride)[-1]]
            sam.detect_frames(mdir, name, c["prompt"], fr, c["det"], c["out_prob"], c["keep_prob"])
    if "diver" in stages:
        sam.track_diver(mdir, n, stride=a.diver_stride)
    if "stuff" in stages:
        for name, c in STUFF.items():
            sam.detect_frames(mdir, name, c["prompt"], stuff_frames(name, n, a.stuff_stride),
                              c["det"], c["out_prob"], c["keep_prob"])
    if "extra" in stages:
        for name, c in EXTRA.items():
            sam.detect_frames(mdir, name, c["prompt"], stuff_frames(name, n, a.stuff_stride),
                              c["det"], c["out_prob"], c["keep_prob"], c["min_area"])
    log(f"[sam] all stages done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
