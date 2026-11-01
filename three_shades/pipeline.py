"""3 Shades underwater reel: palette intro -> grade wipe -> optical-flow ascent ramp.

    python -m three_shades all     video_2.mov -o reel.mp4          # GPU: SAM 3.1 + render
    python -m three_shades segment video_2.mov -o reel.mp4          # GPU: masks only
    python -m three_shades render  video_2.mov -o reel.mp4          # any machine, cached masks
    python -m three_shades render  video_2.mov -o preview.mp4 --preview \
        --segmenter maskrcnn-preview                                 # CPU-only preview
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import flow as flowmod
from .config import Config, dump_config, load_config
from .grade import analyze, export_cube, grade_frame
from .palette_cards import IntroOverlay, grade_wipe_alpha
from .segmentation import load_masks, save_masks, segment
from .video_io import SequentialFrameSource, VideoWriter, read_frames


def mask_cache_path(args) -> Path:
    if args.masks:
        return Path(args.masks)
    out = Path(args.output)
    return out.with_name(out.stem + f".{args.segmenter_name}.masks.npz")


def ensure_masks(args, cfg: Config) -> np.ndarray:
    """Masks are always computed at the full export size and cached."""
    W, H = cfg.export.width, cfg.export.height
    path = mask_cache_path(args)
    if path.exists() and not args.resegment:
        print(f"[masks] using cache {path}")
        return load_masks(path, W, H)
    t0 = time.time()
    masks = segment(args.input, W, H, cfg.seg)
    save_masks(path, masks, {"backend": cfg.seg.backend, "prompt": cfg.seg.text_prompt})
    print(f"[masks] {len(masks)} masks in {time.time() - t0:.0f}s -> {path}")
    return masks


def render(args, cfg: Config) -> None:
    ex = cfg.export
    full_masks = ensure_masks(args, cfg)
    scale = 0.5 if args.preview else 1.0
    W, H = int(ex.width * scale) // 2 * 2, int(ex.height * scale) // 2 * 2
    masks = full_masks if scale == 1.0 else load_masks(mask_cache_path(args), W, H)
    n = len(masks)
    fps = ex.fps

    # Clip-wide grade measurements (sampled, small) so the look never flickers.
    sample = [f for i, f in enumerate(read_frames(args.input, 270, 480)) if i % 8 == 0]
    stats = analyze(sample, cfg.grade)
    print(f"[grade] wb gains {np.round(stats.wb_gains, 3)}, "
          f"levels {stats.black:.3f}..{stats.white:.3f}")

    # Speed ramp on the ascent.
    r = cfg.ramp
    if r.start is not None and r.end is not None:
        ascent = (r.start, r.end)
    else:
        ascent = flowmod.detect_ascent(full_masks, fps, r)
    timeline = flowmod.build_timeline(n, fps, r, ascent)
    print(f"[ramp] ascent (source s): {ascent}; {n} source frames -> "
          f"{len(timeline)} output frames ({len(timeline) / fps:.2f}s)")

    overlay = IntroOverlay(cfg.palette, cfg.layout, cfg.intro, W, H)
    interp = flowmod.Interpolator(flowmod.FlowEngine(cfg.flow), cfg.flow.occlusion_threshold)
    source = SequentialFrameSource(args.input, W, H)
    duration = len(timeline) / fps
    writer = VideoWriter(args.output, W, H, fps, ex.codec, ex.crf, ex.preset, ex.dither,
                         ex.music, ex.music_fade, duration)
    tim = cfg.intro
    t0 = time.time()
    try:
        for k, pos in enumerate(timeline):
            t_out = k / fps
            i = int(np.floor(pos))
            frac = pos - i
            f0 = source.get(i)
            if f0 is None:
                break
            if frac > 1e-3 and i + 1 < n and (f1 := source.get(i + 1)) is not None:
                frame, mask = interp(i, f0, f1, masks[i], masks[i + 1], float(frac))
            else:
                frame, mask = f0, masks[i].astype(np.float32)

            # Flat footage during the palette presentation, then the wipe.
            if t_out < tim.grade_wipe_start:
                out = frame.copy()
            else:
                graded = grade_frame(frame, mask, stats, cfg.grade, cfg.palette)
                a = grade_wipe_alpha(t_out, H, tim)
                out = graded if a is None else frame * (1 - a) + graded * a
            out = overlay.draw(out, t_out)
            if tim.outro_fade > 0:
                out = out * min(1.0, (duration - t_out) / tim.outro_fade)
            writer.write(out)
            if k % 30 == 0:
                el = time.time() - t0
                print(f"[render] {k}/{len(timeline)} frames, {el:.0f}s elapsed, "
                      f"~{el / (k + 1) * (len(timeline) - k - 1):.0f}s left", flush=True)
    finally:
        writer.close()

    out = Path(args.output)
    export_cube(str(out.with_suffix(".cube")), stats, cfg.grade, cfg.palette)
    report = {
        "output": str(out), "size": [W, H], "fps": fps,
        "source_frames": n, "output_frames": len(timeline), "duration_s": round(duration, 3),
        "ascent_source_s": ascent, "segmenter": cfg.seg.backend,
        "palette": {s.name: s.hex for s in cfg.palette.swatches()},
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"[done] {out} ({time.time() - t0:.0f}s); LUT -> {out.with_suffix('.cube')}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="three_shades", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["all", "segment", "render", "dump-config"])
    ap.add_argument("input", nargs="?", help="target video (e.g. video_2.mov)")
    ap.add_argument("-o", "--output", default="reel.mp4")
    ap.add_argument("--config", help="JSON file overriding any field of config.py")
    ap.add_argument("--segmenter", choices=["sam3.1", "maskrcnn-preview"])
    ap.add_argument("--masks", help="mask cache (.npz); default: next to the output")
    ap.add_argument("--resegment", action="store_true", help="ignore an existing mask cache")
    ap.add_argument("--subject-point", help="x,y click on the diver in frame 0 (1080x1920 px)")
    ap.add_argument("--text-prompt", help="SAM 3.1 concept prompt (default: person)")
    ap.add_argument("--music", help="audio track to lay under the reel")
    ap.add_argument("--preview", action="store_true", help="half resolution, faster")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "dump-config":
        print(dump_config(cfg))
        return
    if not args.input:
        ap.error("input video is required")
    if args.segmenter:
        cfg.seg.backend = args.segmenter
    if args.subject_point:
        cfg.seg.subject_point = [float(v) for v in args.subject_point.split(",")]
    if args.text_prompt:
        cfg.seg.text_prompt = args.text_prompt
    if args.music:
        cfg.export.music = args.music
    args.segmenter_name = cfg.seg.backend.replace(".", "")

    if args.command == "segment":
        args.resegment = True
        ensure_masks(args, cfg)
    else:
        render(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
