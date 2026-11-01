#!/usr/bin/env python3
"""Raw -> cinematic underwater reveal with SAM 3.1 subject separation.

    python three_shades.py --input inputs/video_1.mov --output output/video_1_graded.mp4
    python three_shades.py --input ... --masks-only          # segmentation only
    python three_shades.py --input ... --config my.yaml      # override defaults

The video opens on the untouched source, then eases into the graded look.
Nothing is drawn on top of the picture. See README_three_shades.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from shades.config import load_config
from shades.grade import SourceStats, apply_lut, bake_lut, hex_to_rgb, write_cube
from shades.io import Encoder, probe, read_frames_at, iter_frames
from shades.masking import load_subject_masks, mask_dir_for, masks_cached, run_sam31
from shades.separation import (apply_separation, feather, find_lost_frames,
                               lost_spans, separation_weights)
from shades.transition import grade_mix
from shades import verify

REPO = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(msg, flush=True)


def get_masks(args, cfg, info):
    """Return (masks or None, meta or None, status string)."""
    import torch

    mdir = mask_dir_for(Path(args.masks_dir), args.input.stem)
    if masks_cached(mdir, info.n_frames) and not args.resegment:
        masks, meta = load_subject_masks(mdir)
        return masks, meta, f"cached SAM 3.1 masks ({meta['device']})"
    if args.no_sam:
        return None, None, "disabled (--no-sam)"
    if not torch.cuda.is_available() and not args.sam_cpu:
        log("[warn] No CUDA GPU and no cached masks: rendering WITHOUT subject "
            "separation. Run with --sam-cpu (slow) or run --masks-only on a "
            "GPU machine and copy the masks/ folder here.")
        return None, None, "skipped (no GPU, no cache)"
    run_sam31(args.input, mdir, cfg.masking, REPO, REPO / ".cache", log=log,
              resume=not args.resegment)
    masks, meta = load_subject_masks(mdir)
    return masks, meta, f"SAM 3.1 ({meta['device']})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--config", default=None, help="YAML/JSON overrides")
    ap.add_argument("--masks-dir", default=str(REPO / "masks"))
    ap.add_argument("--masks-only", action="store_true",
                    help="run SAM 3.1 segmentation, write the mask cache, exit")
    ap.add_argument("--resegment", action="store_true", help="ignore mask cache")
    ap.add_argument("--sam-cpu", action="store_true",
                    help="allow SAM 3.1 on CPU (CPU port; ~10 s/frame)")
    ap.add_argument("--no-sam", action="store_true", help="skip separation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    info = probe(args.input)
    log(f"[probe] {info.width}x{info.height} {float(info.fps):.3f} fps, "
        f"{info.n_frames} frames, {info.duration:.2f}s, {info.pix_fmt}, "
        f"audio={info.has_audio}")

    if args.masks_only:
        args.sam_cpu = True
        mdir = mask_dir_for(Path(args.masks_dir), args.input.stem)
        (mdir / "meta.json").unlink(missing_ok=True)  # force a (resumable) run
        get_masks(args, cfg, info)
        return 0

    out = args.output or REPO / "output" / f"{args.input.stem}_graded.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    t_start = time.time()

    masks, mmeta, mask_status = get_masks(args, cfg, info)
    log(f"[mask] {mask_status}")

    # --- Grade: clip black/white points -> LUT -> .cube --------------------
    sample_idx = [int(i) for i in np.linspace(3, info.n_frames - 4, 12)]
    samples = read_frames_at(args.input, info, sample_idx, scale_w=540)
    stats = SourceStats.from_frames(list(samples.values()))
    lut = bake_lut(cfg.grade, cfg.palette, stats)
    cube = out.parent / f"{stem}_grade.cube"
    write_cube(lut, cube, f"{stem} cinematic underwater")
    log(f"[grade] LUT {cfg.grade.lut_size}^3 -> {cube}")

    # --- Separation weights -------------------------------------------------
    weights = np.zeros(info.n_frames, np.float32)
    spans: list = []
    coverage = None
    if masks is not None:
        areas = (masks > 127).reshape(len(masks), -1).sum(1)
        coverage = areas / float(info.width * info.height)
        lost = find_lost_frames(areas, info.width * info.height, cfg.separation)
        spans = lost_spans(lost)
        weights = separation_weights(lost, cfg.separation.lost_ramp_frames)
        log(f"[mask] coverage median {np.median(coverage) * 100:.2f}% "
            f"(min {coverage.min() * 100:.2f}%, max {coverage.max() * 100:.2f}%), "
            f"lost spans: {spans or 'none'}")
    haze = hex_to_rgb(next(s.hex for s in cfg.palette if s.role == "shadow"))

    # --- Render -------------------------------------------------------------
    full_from = int(np.ceil((cfg.transition.raw_hold_end + cfg.transition.xfade_dur)
                            * float(info.fps)))
    measure_idx = set(int(i) for i in np.linspace(
        min(full_from, info.n_frames - 1), info.n_frames - 1, 10))
    measured_frames = []
    enc = Encoder(args.input, out, info, cfg.encode)
    t0 = time.time()
    for i, frame in enumerate(iter_frames(args.input, info)):
        mix = grade_mix(i, info.fps, cfg.transition)
        if mix <= 0.0:
            enc.write(frame)   # untouched source
            continue
        graded = apply_lut(frame, lut)
        if masks is not None and weights[i] > 0:
            alpha = feather(masks[i], cfg.separation, info.width)
            graded = apply_separation(graded, alpha, float(weights[i]),
                                      cfg.separation, haze)
        if i in measure_idx:
            measured_frames.append(graded[::2, ::2].copy())
        enc.write(frame + (graded - frame) * mix if mix < 1.0 else graded)
        if i % 60 == 0:
            log(f"[render] frame {i}/{info.n_frames} mix={mix:.2f} "
                f"sep={weights[i]:.2f} {time.time() - t0:.0f}s")
    enc.close()
    log(f"[render] done in {time.time() - t0:.0f}s -> {out}")

    # --- Palette check + sidecars ------------------------------------------
    pal = verify.measure_palette(measured_frames, cfg.palette)
    max_de = cfg.palette_max_delta_e
    for r in pal.values():
        r["pass"] = r["delta_e"] is not None and r["delta_e"] <= max_de
        log(f"[palette] {r['name']:<18} {r['hex']} measured {r['measured']} "
            f"ΔE2000 {r['delta_e']}  {'PASS' if r['pass'] else 'CHECK'}")
    (out.parent / "palette.json").write_text(json.dumps(
        {"max_delta_e": max_de, "anchors": pal}, indent=1, ensure_ascii=False))
    verify.palette_strip(pal, out.parent / "palette.png")

    # --- Verification ---------------------------------------------------------
    src_p, out_p = verify.ffprobe_summary(args.input), verify.ffprobe_summary(out)
    sv = next(s for s in src_p["streams"] if s["codec_type"] == "video")
    ov = next(s for s in out_p["streams"] if s["codec_type"] == "video")
    checks = {
        "frames_match": sv["nb_read_packets"] == ov["nb_read_packets"],
        "size_match": (sv["width"], sv["height"]) == (ov["width"], ov["height"]),
        "fps_match": sv["r_frame_rate"] == ov["r_frame_rate"],
        "audio_present": any(s["codec_type"] == "audio" and s["codec_name"] == "aac"
                             for s in out_p["streams"]),
        "duration_delta_s": round(abs(float(src_p["format"]["duration"])
                                      - float(out_p["format"]["duration"])), 4),
    }
    checks["duration_ok"] = checks["duration_delta_s"] <= 1.0 / float(info.fps) + 1e-3
    raw_t = min(1.0, cfg.transition.raw_hold_end / 2)
    checks["raw_section_mad_8bit"] = round(verify.raw_frame_diff(args.input, out, raw_t), 3)
    checks["raw_section_ok"] = checks["raw_section_mad_8bit"] < 2.0
    sheet = out.parent / f"{stem}_contact_sheet.jpg"
    verify.contact_sheet(out, [0.5, 1.5, 2.3, 2.7, 3.5, 8.0, 14.0], sheet)
    for k, v in checks.items():
        log(f"[verify] {k}: {v}")

    report = {
        "input": str(args.input), "output": str(out), "cube": str(cube),
        "mask_status": mask_status,
        "mask_meta": None if mmeta is None else {
            k: mmeta[k] for k in ("device", "text_prompt", "chunks", "seconds")
            if k in mmeta},
        "coverage_per_frame": None if coverage is None else
            [round(float(c), 5) for c in coverage],
        "lost_spans": spans, "palette": pal, "checks": checks,
        "contact_sheet": str(sheet), "ffprobe_output": out_p,
        "source_stats": {"lo": stats.lo.tolist(), "hi": stats.hi.tolist()},
        "config": cfg.to_dict(), "seconds": round(time.time() - t_start, 1),
    }
    (out.parent / f"{stem}_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str))
    ok = all(v for k, v in checks.items() if k.endswith(("_ok", "_match", "_present")))
    log(f"[done] {'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
