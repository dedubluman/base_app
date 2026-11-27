#!/usr/bin/env python3
"""SAM 3.1 point-prompted masks of the wheel, shaft and rocks, on every Nth frame.

SAM 3.1's CPU-port video propagation returns no tracks on this footage, so each
keyframe gets its own single-frame SAM 3.1 session with point prompts. The prompt
points of keyframe k are taken from the previous keyframe's mask, warped by
Farneback optical flow (keyframe 0 uses hand-placed anchors). A SAM mask that
disagrees with the flow-warped previous mask (IoU < thr) is rejected and the
warped mask is kept instead; those keyframes are listed in report.json.

    python uw_masks.py --input clip.mov --out masks_uw --stride 6
Output: <out>/<part>/<frame:05d>.png (0/255) for parts wheel, shaft, rock_r, rock_l, rock_bg
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, time
from pathlib import Path
import cv2, numpy as np, torch
from shades.config import Masking
from shades.masking import _build_predictor

REPO = Path(__file__).resolve().parent
# keyframe-0 anchors in source pixels (x, y); obj ids are per part
ANCHORS = {
    "wheel":   (1, [(495, 350), (735, 330), (610, 248), (620, 350)]),
    "shaft":   (2, [(585, 550), (565, 760), (545, 980)]),
    "rock_r":  (3, [(900, 650), (1000, 560)]),
    "rock_l":  (4, [(270, 390), (300, 300)]),
}
KPTS = {"wheel": 4, "shaft": 4, "rock_r": 3, "rock_l": 2}


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def warp_mask(m, g_prev, g_cur):
    h, w = m.shape
    hs, ws = h // 2, w // 2
    a, b = cv2.resize(g_cur, (ws, hs)), cv2.resize(g_prev, (ws, hs))
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    gx, gy = np.meshgrid(np.arange(ws, dtype=np.float32), np.arange(hs, dtype=np.float32))
    mm = cv2.resize(m.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA)
    out = cv2.remap(mm, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.resize(out, (w, h)) > 0.5


def interior_points(m, k):
    er = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8))
    if er.sum() < 30:
        er = m.astype(np.uint8)
    ys, xs = np.nonzero(er)
    if len(xs) == 0:
        return []
    pts = np.stack([xs, ys], 1).astype(np.float32)
    k = min(k, len(pts))
    _, _, cen = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    dt = cv2.distanceTransform(er, cv2.DIST_L2, 3)
    out = []
    for c in cen:
        d2 = (xs - c[0]) ** 2 + (ys - c[1]) ** 2
        near = np.argsort(d2)[:200]
        j = near[np.argmax(dt[ys[near], xs[near]])]
        out.append((int(xs[j]), int(ys[j])))
    return out


def load_seed(seed_dir: Path):
    """Keyframe-0 SAM 3.1 masks (from the interactive probe): wheel, shaft, rock_r, rock_l."""
    return {"wheel": seed_dir / "wheel.png", "shaft": seed_dir / "shaft.png",
            "rock_r": seed_dir / "rock_r.png", "rock_l": seed_dir / "rock_l.png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=Path, required=True, help="dir with keyframe-0 wheel/shaft/rock_r/rock_l PNGs")
    ap.add_argument("--prompt", default="rock")
    ap.add_argument("--det-thresh", type=float, default=0.15)
    a = ap.parse_args()
    cache = REPO / ".cache"
    kdir = cache / f"uw_keyframes_{a.input.stem}_s{a.stride}"
    if not kdir.exists():
        kdir.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(a.input), "-vf", f"select='not(mod(n\\,{a.stride}))'",
                        "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0", str(kdir / "%05d.jpg")], check=True)
    files = sorted(kdir.glob("*.jpg"))[:a.limit]
    pred, _ = _build_predictor(Masking(max_num_objects=8, detector_batch=2), REPO, cache)
    for k in ("score_threshold_detection", "new_det_thresh", "image_only_det_thresh"):
        setattr(pred.model, k, a.det_thresh)
    for part in ANCHORS:
        (a.out / part).mkdir(parents=True, exist_ok=True)
    H, W = cv2.imread(str(files[0])).shape[:2]
    prev, prev_g, report = {}, None, {}
    t_all = time.time()
    for i, f in enumerate(files):
        fr = i * a.stride
        t0 = time.time()
        img = cv2.imread(str(f))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warped = {p: warp_mask(prev[p], prev_g, g) for p in prev}
        cdir = cache / "uw_single"
        shutil.rmtree(cdir, ignore_errors=True); cdir.mkdir()
        (cdir / "00000.jpg").symlink_to(f.resolve())
        sid = pred.handle_request({"type": "start_session", "resource_path": str(cdir)})["session_id"]
        rep, cur = {}, {}
        if i == 0:
            for part, p in load_seed(a.seed).items():
                cur[part] = cv2.imread(str(p), 0) > 127
                rep[part] = f"seed area={int(cur[part].sum())}"
        else:
            det = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": a.prompt})["outputs"]
            dmasks = {int(oid): np.asarray(m) > 0 for oid, m in zip(det["out_obj_ids"], det["out_binary_masks"])}
            used_ids = set()
            order = ["rock_r", "rock_l", "wheel", "shaft"]   # rocks claim their own instance first
            for part in order:
                best = max(((iou(m, warped[part]), oid) for oid, m in dmasks.items() if oid not in used_ids),
                           default=(0.0, None))
                if part in ("wheel", "shaft"):
                    # the detector does not segment metal as "rock": any free instance slot is only
                    # a vehicle for the point-prompted refinement (SAM re-segments from the points)
                    free = [oid for oid in dmasks if oid not in used_ids]
                    best = (best[0], free[-1] if free else None)
                if best[1] is None or (part.startswith("rock") and best[0] < 0.2):
                    cur[part] = warped[part]; rep[part] = f"WARPED (no instance, best iou={best[0]:.2f})"; continue
                used_ids.add(best[1])
                pts = interior_points(warped[part], KPTS[part])
                r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                         "points": torch.tensor([[x / W, y / H] for x, y in pts], dtype=torch.float32),
                                         "point_labels": torch.ones(len(pts), dtype=torch.int32), "obj_id": best[1]})
                o = r["outputs"]
                ids = [int(x) for x in o["out_obj_ids"]]
                m = np.asarray(o["out_binary_masks"][ids.index(best[1])]) > 0 if best[1] in ids else np.zeros((H, W), bool)
                s = iou(m, warped[part])
                if s >= a.thr and m.sum() > 0:
                    cur[part] = m; rep[part] = f"sam iou={s:.2f} (inst {best[1]})"
                else:
                    cur[part] = warped[part]; rep[part] = f"WARPED (sam iou={s:.2f}, sam area={int(m.sum())})"
        pred.handle_request({"type": "close_session", "session_id": sid})
        for part, m in cur.items():
            cv2.imwrite(str(a.out / part / f"{fr:05d}.png"), m.astype(np.uint8) * 255)
        prev, prev_g = cur, g
        report[fr] = rep
        (a.out / "report.json").write_text(json.dumps(report, indent=1))
        print(f"[sam] frame {fr}: {rep} {time.time()-t0:.0f}s (elapsed {time.time()-t_all:.0f}s)", flush=True)
    print("[sam] DONE", flush=True)


if __name__ == "__main__":
    main()
