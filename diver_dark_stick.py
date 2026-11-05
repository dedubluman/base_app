"""SAM 3.1 point-prompted selfie-stick masks for diver_dark.py.

SAM 3.1's text detector finds no "selfie stick" / "pole" / "stick" /
"monopod" in this clip, but its point-prompted tracker head segments the
stick cleanly. Reuses split_sea.run_rock_masks (per-frame segmentation from
points derived from the previous mask, with IoU/area QC re-segmentation) on
the stride-4 proxy made by diver_dark_sam.py, then links the result to
masks/<stem>__stick_s4 for diver_dark.py.
"""

from pathlib import Path

import split_sea

REPO = Path(__file__).resolve().parent
# first-frame clicks (source px): positives along the stick and its grip
# ring, negatives in the water on both sides and on the hand
split_sea.SEED_POINTS = [[650, 700, 1], [690, 800, 1], [730, 900, 1],
                         [770, 1000, 1], [635, 590, 1],
                         [480, 820, 0], [940, 760, 0], [640, 470, 0]]

proxy = REPO / ".cache" / "diver__stick_s4.mp4"
mdir = split_sea.run_rock_masks(proxy, REPO / "masks", 1, "stick")
link = REPO / "masks" / "diver__stick_s4"
if link.is_symlink() or link.exists():
    import shutil
    shutil.rmtree(link) if link.is_dir() and not link.is_symlink() else link.unlink()
link.symlink_to(mdir.name)
print("linked", link, "->", mdir.name)
