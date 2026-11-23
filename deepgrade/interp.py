"""Optical-flow slow motion: 230 graded frames -> 450 frames (exactly 15.000 s at 30 fps).

    python -m deepgrade.interp plan                      # retiming plan -> output/v2/retime_plan.json
    python -m deepgrade.interp run                       # RIFE v4.26 (CPU), resumable
    python -m deepgrade.interp qc                        # artifact detector + conservative fallback
    python -m deepgrade.interp loo --frames 190,...      # leave-one-out accuracy test vs. ground truth

Timing: the n_in source frames last exactly n_out output frames, so output frame k shows source position
s_k = min(k * n_in / n_out, n_in-1) (speed = 230/450 = 51.1 % = 7.674 s / 15 s). The original order is kept,
nothing is reversed or dropped; the last source frame stays up for the last output frame(s), which is simply
how long it was on screen, slowed down. Positions within ``snap`` of an integer reuse the source frame.

Interpolation: RIFE v4.26 (Practical-RIFE) estimates intermediate flow and a fusion mask for an arbitrary
time t in one pass (occlusion-aware, no ghosting). Frames are the graded frames, so colour stays exactly as
graded. Conservative fallbacks (only for frames the detector flags): coarser RIFE flow scale, then a short
dissolve between the two neighbouring source frames. No content is invented.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
V2 = REPO / "output/v2"


def load16(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), -1)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0


def save16(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR))


def make_plan(n_in: int, n_out: int, snap: float) -> list[dict]:
    plan = []
    for k in range(n_out):
        s = min(k * n_in / n_out, n_in - 1.0)         # time-faithful: n_in frames of video last n_out frames
        i = min(int(np.floor(s)), n_in - 2)
        f = s - i
        if f < snap:
            plan.append({"k": k, "s": s, "mode": "src", "i": i})
        elif f > 1 - snap:
            plan.append({"k": k, "s": s, "mode": "src", "i": i + 1})
        else:
            plan.append({"k": k, "s": s, "mode": "rife", "i": i, "f": f})
    return plan


def small_gray(rgb: np.ndarray, w: int = 308) -> np.ndarray:
    h = int(round(rgb.shape[0] * w / rgb.shape[1]))
    g = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    return 0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "qc", "loo"])
    ap.add_argument("--frames-dir", type=Path, default=V2 / "graded_frames")
    ap.add_argument("--out", type=Path, default=V2 / "retimed_frames")
    ap.add_argument("--n-out", type=int, default=450)
    ap.add_argument("--snap", type=float, default=0.04)
    ap.add_argument("--frames", default="190,191,192,193,194,195,196,197,198,60,100,140")
    a = ap.parse_args()
    n_in = len(list(a.frames_dir.glob("*.png")))
    plan = make_plan(n_in, a.n_out, a.snap)
    plan_f = V2 / "retime_plan.json"

    if a.cmd == "plan":
        plan_f.write_text(json.dumps({"n_in": n_in, "n_out": a.n_out, "fps_out": 30, "snap": a.snap,
                                      "model": "RIFE v4.26 (Practical-RIFE), CPU, scale 1.0",
                                      "mapping": "s_k = min(k*n_in/n_out, n_in-1)", "plan": plan}, indent=0))
        nr = sum(p["mode"] == "rife" for p in plan)
        print(f"plan: {a.n_out} output frames from {n_in}; {nr} interpolated, {a.n_out-nr} reuse a source frame")
        print(f"speed: source runs at {n_in/a.n_out*100:.2f}% of real time; "
              f"duration {a.n_out/30:.3f} s at 30 fps")
        return

    from .rife_model import RIFE

    a.out.mkdir(parents=True, exist_ok=True)
    cache: dict[int, np.ndarray] = {}

    def src(i: int) -> np.ndarray:
        if i not in cache:
            if len(cache) > 3:
                cache.pop(min(cache))
            cache[i] = load16(a.frames_dir / f"{i:05d}.png")
        return cache[i]

    if a.cmd == "run":
        rife = RIFE()
        t0, done = time.time(), 0
        for p in plan:
            dst = a.out / f"{p['k']:05d}.png"
            if dst.exists():
                continue
            if p["mode"] == "src":
                save16(dst, src(p["i"]))
            else:
                save16(dst, rife(src(p["i"]), src(p["i"] + 1), p["f"]))
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"[interp] k={p['k']}/{a.n_out} done={done} {el:.0f}s ({el/done:.1f}s/frame)", flush=True)
        print("[interp] finished", flush=True)

    elif a.cmd == "qc":
        # artifact detector: the frame-to-frame change of the output must follow the change of the
        # source between the two frames it was made from (scaled by the time step); an interpolated
        # frame that adds extra change (ghost, tear, warp) shows up as a spike.
        outs = {p["k"]: small_gray(load16(a.out / f"{p['k']:05d}.png")) for p in plan}
        srcs = {i: small_gray(src(i)) for i in range(n_in)}
        d = np.array([np.abs(outs[k] - outs[k - 1]).mean() for k in range(1, a.n_out)])
        sd = np.array([np.abs(srcs[i + 1] - srcs[i]).mean() for i in range(n_in - 1)])
        step = n_in / a.n_out
        expect = np.array([step * sd[min(int(np.floor(plan[k]["s"] - 1e-6)), n_in - 2)] for k in range(1, a.n_out)])
        ratio = d / np.maximum(expect, 0.002)
        # local spike: output change versus the median of its neighbours
        spike = np.array([d[j] / max(np.median(np.r_[d[max(0, j - 4):j], d[j + 1:j + 5]]), 1e-4)
                          for j in range(len(d))])
        flagged = [int(j + 1) for j in range(len(d)) if ratio[j] > 1.9 and spike[j] > 1.6]
        print(f"output/expected change ratio: median {np.median(ratio):.2f}  p99 {np.percentile(ratio, 99):.2f}  max {ratio.max():.2f}")
        print(f"local spike: median {np.median(spike):.2f}  p99 {np.percentile(spike, 99):.2f}  max {spike.max():.2f}")
        print("flagged output frames:", flagged)
        fixed = []
        if flagged:
            rife = RIFE()
            for k in flagged:
                p = plan[k]
                if p["mode"] != "rife":
                    continue
                A, B = src(p["i"]), src(p["i"] + 1)
                cand = rife(A, B, p["f"], scale=0.5)             # 1) coarser flow
                g = small_gray(cand)
                bad = np.abs(g - outs[k - 1]).mean() > 1.9 * expect[k - 1]
                if bad:                                          # 2) conservative: short dissolve
                    cand = (1 - p["f"]) * A + p["f"] * B
                    fixed.append((k, "dissolve"))
                else:
                    fixed.append((k, "scale0.5"))
                save16(a.out / f"{k:05d}.png", cand)
        (V2 / "interp_qc.json").write_text(json.dumps({
            "ratio_median": float(np.median(ratio)), "ratio_p99": float(np.percentile(ratio, 99)),
            "ratio_max": float(ratio.max()), "spike_max": float(spike.max()),
            "flagged": flagged, "fallbacks": fixed, "d": d.tolist(), "expected": expect.tolist()}))
        print("fallbacks applied:", fixed)

    elif a.cmd == "loo":
        rife = RIFE()
        rows = []
        for i in [int(x) for x in a.frames.split(",")]:
            A, T, B = src(i - 1), src(i), src(i + 1)
            M = rife(A, B, 0.5)
            avg = (A + B) / 2
            mse = lambda X: float(np.mean((X - T) ** 2))
            psnr = lambda X: 10 * np.log10(1.0 / max(mse(X), 1e-10))
            rows.append((i, psnr(M), psnr(avg), psnr(A)))
            print(f"frame {i:3d}: RIFE {rows[-1][1]:5.1f} dB | plain average {rows[-1][2]:5.1f} dB | repeat-previous {rows[-1][3]:5.1f} dB", flush=True)
        r = np.array(rows)
        print(f"mean PSNR  RIFE {r[:,1].mean():.1f} dB  average {r[:,2].mean():.1f} dB  repeat {r[:,3].mean():.1f} dB")
        (V2 / "interp_loo.json").write_text(json.dumps({"frames": r[:, 0].tolist(), "rife": r[:, 1].tolist(),
                                                        "average": r[:, 2].tolist(), "repeat": r[:, 3].tolist()}))


if __name__ == "__main__":
    main()
