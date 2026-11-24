#!/usr/bin/env python
"""Merge targeted SAM 3.1 re-runs into the main mask cache (union).

The main text-prompt pass can lose a body part for a few frames (e.g. the body
that is only half in frame at the start, a motion-blurred hand).  Those spans
are re-segmented from a frame where SAM sees the whole body, tracking forward or
backward, and OR-ed into the main masks:

    python uw_merge_masks.py --main masks/clip --out masks/merged \
        --aux masks/auxA:20:-1 --aux masks/auxB:64:1

``--aux DIR:START:STEP``: aux mask k belongs to source frame START + STEP*k.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aux", action="append", default=[])
    a = ap.parse_args()

    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "subject").mkdir(parents=True)
    frames = sorted(int(f.stem) for f in (a.main / "subject").glob("*.png"))
    merged = {i: cv2.imread(str(a.main / "subject" / f"{i:05d}.png"), 0) > 127 for i in frames}
    for spec in a.aux:
        d, start, step = spec.split(":")
        start, step = int(start), int(step)
        for f in sorted((Path(d) / "subject").glob("*.png")):
            i = start + step * int(f.stem)
            if i in merged:
                m = cv2.imread(str(f), 0) > 127
                before = int(merged[i].sum())
                merged[i] = merged[i] | m
                print(f"frame {i:3d}: {before // 1000:4d}k -> {int(merged[i].sum()) // 1000:4d}k px")
    for i, m in merged.items():
        cv2.imwrite(str(a.out / "subject" / f"{i:05d}.png"), m.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
