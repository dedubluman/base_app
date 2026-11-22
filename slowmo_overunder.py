#!/usr/bin/env python3
"""Cinematic over-under slow motion: smooth speed ramp + optical-flow frame synthesis.

    python slowmo_overunder.py --input clip.mov --map-only              # print the retime map
    python slowmo_overunder.py --input clip.mov --preview prev.mp4       # short full-res preview
    python slowmo_overunder.py --input clip.mov --output out.mov         # full render (+ retimed audio)

Normal-speed sections are bit-for-bit original frames (no interpolation). Only frames that fall
between two source frames are synthesised with RIFE v4.26 optical flow, re-rendered with a
Lanczos kernel, plus a ghost guard. The retime map (JSON) records, for every output frame, the
exact source time, so SAM masks / grades can be evaluated at the matching timestamp (see
`slowmo/masks.py` for flow-consistent mask propagation).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from slowmo import engine as E
from slowmo import resample as RS
from slowmo.rife import Rife

SNAP = 0.03        # |t| below this is treated as an exact source frame (invisible, keeps detail)


def synth(rife: Rife, src, p: float, guard_thr: float, stats: dict | None = None, aux=None):
    """Frame (float32 RGB) at fractional source position p.

    aux: optional (aux_a, aux_b) float32 HxWxC arrays defined on the two bracketing source frames
    (e.g. SAM masks). They are warped with exactly the same flow/occlusion mask as the picture and
    returned as stats['aux'], so a mask can never be reused at the wrong timestamp."""
    i = int(np.floor(p + 1e-9)); t = p - i
    if t < SNAP or i + 1 >= len(src):
        return E.to_f32(src[min(i, len(src) - 1)]), "src"
    if t > 1 - SNAP:
        return E.to_f32(src[i + 1]), "src"
    a, b = E.to_f32(src[i]), E.to_f32(src[i + 1])
    _, fl, m = rife(a, b, t, return_flow=True)
    w0, w1 = RS.remap(a, fl[:2]), RS.remap(b, fl[2:4])
    mm = m[..., None]
    if aux is not None and stats is not None:
        wa = lambda x, f: RS.remap(x, f, cv2.INTER_LINEAR).reshape(x.shape)
        stats["aux"] = wa(aux[0], fl[:2]) * mm + wa(aux[1], fl[2:4]) * (1 - mm)
    out = np.clip(w0 * mm + w1 * (1 - mm), 0, 1)
    out, cov = RS.ghost_guard(w0, w1, out, t, thr=guard_thr)
    out, gain = RS.restore_detail(out, w0, w1, m, gmax=1.6)
    if stats is not None:
        stats["guard_cov"] = cov
        stats["detail_gain"] = gain
    return out, "rife"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--preview", type=Path, help="render only the ramp-in + early hold at full res")
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--ramp-in-start", type=int, default=24)
    ap.add_argument("--ramp-in-frames", type=int, default=14)
    ap.add_argument("--ramp-out-end", type=int, default=124)
    ap.add_argument("--ramp-out-frames", type=int, default=18)
    ap.add_argument("--preview-src", type=float, nargs=2, metavar=("A", "B"),
                    help="source-frame range for --preview (default: ramp-in region)")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--guard", type=float, default=0.12)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--map-json", type=Path)
    ap.add_argument("--dump", type=Path, help="write 8-bit PNG of every output frame here (QC)")
    a = ap.parse_args()

    info = E.probe_video(a.input)
    w, h, fps = info["width"], info["height"], info["r_frame_rate"]
    src = E.decode_frames(a.input, w, h)
    spec = E.RampSpec(a.ramp_in_start, a.ramp_in_frames, a.speed, a.ramp_out_end, a.ramp_out_frames, len(src))
    pos = E.build_speed_map(spec)
    rows = [{"out": k, "src_pos": round(float(p), 5), "src_a": int(np.floor(p + 1e-9)),
             "t": round(float(p - np.floor(p + 1e-9)), 5)} for k, p in enumerate(pos)]
    print(f"{len(src)} source frames -> {len(pos)} output frames ({len(pos) / 29.97:.2f}s), "
          f"{sum(1 for r in rows if SNAP <= r['t'] <= 1 - SNAP)} synthesised", flush=True)
    if a.map_json:
        a.map_json.write_text(json.dumps({"fps": fps, "spec": vars(a) if False else spec.__dict__, "frames": rows}, indent=1))
    if a.map_only:
        return 0

    idx = np.arange(len(pos))
    if a.preview:
        lo, hi = a.preview_src or (a.ramp_in_start - 4, a.ramp_in_start + 30)
        idx = idx[(pos >= lo) & (pos <= hi)]
    rife = Rife()
    enc = E.Encoder(a.preview or a.output, w, h, fps, crf=a.crf if not a.preview else 18)
    if a.dump:
        a.dump.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, k in enumerate(idx):
        st = {}
        f, how = synth(rife, src, float(pos[k]), a.guard, st)
        enc.write(E.to_u16(f))
        if a.dump:
            cv2.imwrite(str(a.dump / f"o{k:04d}.png"), cv2.cvtColor((f * 255 + .5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if n % 10 == 0:
            print(f"  out {k:3d} pos {pos[k]:7.3f} {how:4s} guard {st.get('guard_cov', 0):.3f}  [{time.time() - t0:5.0f}s]", flush=True)
    enc.close()
    print("done", a.preview or a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
