"""Assemble the deliverables folder.

    python -m deepgrade.package --stem player_export

output/<stem>_deliverables/
    video/       graded H.264 MP4 + 10-bit ProRes master
    previews/    source | graded frames, contact sheet
    masks/       final region weights (D diver, S sky, R rocks/land, W water, U sand),
                 SAM 3.1 raw keyframe masks, gate / QC json
    config/      grade parameters (params.json), render stats, palette report
    logs/        SAM / render logs
    scripts/     snapshot of the deepgrade package
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def cp(src: Path, dst: Path) -> None:
    if not src.exists():
        print("  (missing)", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("_tmp_*", "progress.json", "__pycache__"))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="player_export")
    ap.add_argument("--logs", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    out = REPO / "output" / f"{a.stem}_deliverables"
    o = REPO / "output"
    m = REPO / "masks" / a.stem
    cp(o / f"{a.stem}_graded.mp4", out / "video" / f"{a.stem}_graded.mp4")
    cp(o / f"{a.stem}_master_prores.mov", out / "video" / f"{a.stem}_master_prores.mov")
    cp(o / f"{a.stem}_contact_sheet.jpg", out / "previews" / "contact_sheet.jpg")
    for p in sorted((o / "final_frames").glob("*")):
        cp(p, out / "previews" / p.name)
    for name in ("final",):
        cp(m / name, out / "masks" / "final_region_weights")
    for name in ("diver", "diver_single", "rocks", "water", "sky", "mountain", "snorkel"):
        cp(m / name, out / "masks" / "sam31_keyframes" / name)
    cp(m / "qc_masks.json", out / "masks" / "qc_masks.json")
    cp(o / f"{a.stem}_graded.params.json", out / "config" / "grade_params.json")
    cp(o / f"{a.stem}_graded.stats.json", out / "config" / "render_anchor_stats.json")
    cp(o / "palette_report.json", out / "config" / "palette_report.json")
    for lg in a.logs:
        cp(lg, out / "logs" / lg.name)
    cp(REPO / "deepgrade", out / "scripts" / "deepgrade")
    cp(REPO / "README_deepgrade.md", out / "README_deepgrade.md")
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"deliverables -> {out}  ({n} files, {size:.0f} MB)")


if __name__ == "__main__":
    main()
