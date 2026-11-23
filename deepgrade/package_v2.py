"""Assemble the v2 deliverables folder (final 15 s video, master, previews, masks, configs, logs, scripts)."""
from __future__ import annotations

import shutil
import subprocess
import sys
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
    logs = [Path(p) for p in sys.argv[1:]]
    o, v2, m = REPO / "output", REPO / "output/v2", REPO / "masks/player_export"
    out = o / "underwater_cinematic_final_15s_deliverables"
    mp4 = o / "underwater_cinematic_final_15s.mp4"
    cp(mp4, out / "video" / mp4.name)
    cp(o / "underwater_cinematic_final_15s_master_prores.mov", out / "video" / "underwater_cinematic_final_15s_master_prores.mov")
    cp(o / "underwater_cinematic_final_15s_contact_sheet.jpg", out / "previews" / "contact_sheet.jpg")
    for k, name in ((0, "000_opening"), (88, "088_old_1p5s"), (128, "128"), (148, "148_old_2p5s"), (220, "220_middle"),
                    (340, "340_final_underwater"), (378, "378_waterline"), (384, "384_waterline"), (420, "420_above_water"), (449, "449_ending")):
        dst = out / "previews" / f"frame_{name}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", f"select=eq(n\\,{k})", "-frames:v", "1", str(dst)])
    for f in ("qc_sheet_final_grade.jpg", "interp_waterline_strip.jpg", "interp_hands_1to1.jpg", "rock_crop_1to1.jpg"):
        cp(v2 / f, out / "previews" / f)
    cp(m / "final_v2", out / "masks" / "final_region_weights_v2")
    for name in ("diver", "diver_single", "rocks", "water", "sky", "mountain", "snorkel", "shorts"):
        cp(m / name, out / "masks" / "sam31_keyframes" / name)
    cp(m / "scene_cuts.json", out / "masks" / "scene_cuts.json")
    for f in ("params2.json", "anchors.json", "retime_plan.json", "interp_qc.json", "interp_loo.json", "diag_final.json", "diag_v2.json"):
        cp(v2 / f, out / "config" / f)
    cp(o / "diag_v1.json", out / "config" / "diag_v1_previous_render.json")
    for lg in logs:
        cp(lg, out / "logs" / lg.name)
    cp(REPO / "deepgrade", out / "scripts" / "deepgrade")
    cp(REPO / "README_deepgrade.md", out / "README_deepgrade.md")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"deliverables -> {out} ({n} files, {sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
