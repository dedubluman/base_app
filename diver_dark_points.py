"""Point-prompted SAM 3.1 masks for small diver parts (shorts, watch, mask).

SAM 3.1's text detector finds none of "shorts", "swim trunks", "swimsuit",
"briefs", "watch", "wristwatch", "goggles", "mask" in this clip, but its
point-prompted tracker segments them from a few clicks. Same mechanism as
diver_dark_stick.py (split_sea.run_rock_masks: per-frame segmentation from
points derived from the previous mask, with IoU/area QC re-segmentation) on
the stride-4 proxy; result linked to masks/diver__<part>_s4.

    python diver_dark_points.py shorts
"""

import shutil
import sys
from pathlib import Path

import split_sea

REPO = Path(__file__).resolve().parent

# first-frame clicks (source px): [x, y, 1=on the part / 0=not]
SEEDS = {
    "shorts": [[680, 320, 1], [700, 335, 1], [660, 355, 1], [720, 345, 1],
               [770, 290, 0], [560, 320, 0], [740, 425, 0], [650, 470, 0]],
    "watch": [[740, 420, 1], [735, 440, 1], [725, 405, 1],
              [650, 480, 0], [800, 450, 0], [690, 340, 0], [560, 400, 0]],
    "mask": [[905, 60, 1], [890, 105, 1], [870, 140, 1],
             [780, 40, 0], [980, 80, 0], [760, 200, 0], [640, 120, 0]],
}


def main(part: str) -> None:
    split_sea.SEED_POINTS = SEEDS[part]
    base = REPO / ".cache" / "diver__stick_s4.mp4"  # stride-4 proxy, frames 0-587
    proxy = REPO / ".cache" / f"diver__{part}_s4pts.mp4"
    if not proxy.exists():
        shutil.copy(base, proxy)
    mdir = split_sea.run_rock_masks(proxy, REPO / "masks", 1, part)
    link = REPO / "masks" / f"diver__{part}_s4"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(mdir.name)
    print("linked", link, "->", mdir.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
