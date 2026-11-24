#!/usr/bin/env python
"""SAM 3.1 subject masks for an underwater clip (cache for uw_grade.py).

    python uw_masks.py --input clip.mov --masks masks/clip

Writes ``<masks>/subject/<frame:05d>.png`` (uint8 0/255, source resolution) and
``meta.json``. Reuses the chunked / CPU-port runner in ``shades/masking.py``;
the swimmer is found with the SAM 3.1 text prompt "person" (the whole body:
skin, hair, mask/snorkel, arms, legs and gear held or worn). Frames where the
swimmer is absent come out as empty masks.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from shades.config import Masking
from shades.masking import masks_cached, run_sam31

REPO = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True, help="output mask dir")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    n = int(json.loads(__import__("subprocess").check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(a.input)]
    ))["streams"][0]["nb_read_frames"])
    if masks_cached(a.masks, n) and not a.force:
        print(f"[masks] cache hit: {a.masks}")
        return
    cfg = dataclasses.replace(Masking(), text_prompt=a.prompt, parts={},
                              chunk_frames=a.chunk, chunk_overlap=a.overlap)
    run_sam31(a.input.resolve(), a.masks.resolve(), cfg, REPO, REPO / ".cache")


if __name__ == "__main__":
    main()
