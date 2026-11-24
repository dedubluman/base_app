#!/usr/bin/env python3
"""Runner for the whole pipeline with per-stage caching (re-run any stage with --force STAGE).

    python run_grade.py --input inputs/fish.mov --refs inputs/reference/gold_a.jpg inputs/reference/gold_b.jpg \
        --look look_gold.json --gold --canvas 1920x1080 --output output/fish_gold_1080p.mp4

Stages (module that implements each):
  probe      ffprobe summary                                         (here)
  mask       SAM 3.1 keyframe masks  masks/<stem>__{fish,sand}       fish_masks.py   (skipped if cached)
  refine     per-frame mattes        masks/<stem>__mattes            fish_refine.py  (skipped if cached)
  calibrate  measure the references  grade_params*.json             fish_grade.py   (skipped if cached)
  render     grade + H.264 encode                                    fish_grade.py
  verify     ffprobe + numeric QC of the encoded file                fish_qc.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
STAGES = ["probe", "mask", "refine", "calibrate", "render", "verify"]


def run(cmd, log=print):
    log("[run] " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--refs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("--look", type=Path, default=None, help="LOOK override JSON")
    ap.add_argument("--gold", action="store_true", help="reference fish is golden/ochre (measure its tables)")
    ap.add_argument("--canvas", default=None, help="e.g. 1920x1080: fit + pad with black")
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES, help="re-run these stages even if cached")
    a = ap.parse_args()
    stem = a.input.stem
    py = sys.executable
    params = a.params or REPO / (f"grade_params_{stem}.json")
    masks = REPO / "masks"
    todo = lambda st: st in a.stages

    if todo("probe"):
        run(["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,duration,color_space,color_range:stream_side_data=rotation",
             "-of", "compact", a.input])
    if todo("mask") and ("mask" in a.force or not (masks / f"{stem}__sand" / "meta.json").exists()
                         or not (masks / f"{stem}__fish" / "meta.json").exists()):
        run([py, "fish_masks.py", "--input", a.input, "--chunk", 32, "--overlap", 4, "--resume",
             "--regions", "fish:fish:3", "sand:sandy bottom:6"])
    if todo("refine") and ("refine" in a.force or not (masks / f"{stem}__mattes" / "report.json").exists()):
        parts = ["--parts", "fins"] if (masks / f"{stem}__fins" / "meta.json").exists() else []
        run([py, "fish_refine.py", "--input", a.input, "--fish-gids", 1, *parts])
    if todo("calibrate") and ("calibrate" in a.force or not params.exists()):
        run([py, "fish_grade.py", "calibrate", "--input", a.input, "--refs", *a.refs, "--params", params,
             *(["--gold"] if a.gold else [])])
    if todo("render"):
        run([py, "fish_grade.py", "render", "--input", a.input, "--output", a.output, "--params", params,
             "--crf", a.crf, *(["--canvas", a.canvas] if a.canvas else []),
             *(["--look-json", a.look] if a.look else [])])
    if todo("verify"):
        run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
             "stream=codec_name,profile,width,height,sar,pix_fmt,r_frame_rate,nb_read_frames,duration,bit_rate,color_space,color_range",
             "-of", "compact", a.output])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
