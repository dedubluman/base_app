"""SAM 3.1 (local checkpoint, CPU port when no GPU) text-prompt segmentation of key frames.

Prompts that SAM 3.1 detects on this footage: sky, rock cliff, green vegetation, water
(plus plain 'rock' for seabed rocks). Result per key frame: {prompt: uint8 mask} in .npz."""
from __future__ import annotations
import shutil, sys, time
from pathlib import Path
import cv2, numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
PROMPTS = ["sky", "rock cliff", "green vegetation", "water", "rock"]


def run(frames_dir: Path, kf_dir: Path, keys: list[int], work: Path):
    sys.path.insert(0, str(REPO))
    from shades.config import Masking
    from shades.masking import _build_predictor
    pred, gpu = _build_predictor(Masking(), REPO, REPO / ".cache")
    print("SAM 3.1 loaded; gpu:", gpu, flush=True)
    kf_dir.mkdir(parents=True, exist_ok=True)
    jpg = work / "jpg"; jpg.mkdir(parents=True, exist_ok=True)
    for k in keys:
        out = kf_dir / f"kf{k:04d}.npz"
        if out.exists():
            continue
        t0 = time.time()
        cv2.imwrite(str(jpg / f"{k:05d}.jpg"), cv2.imread(str(frames_dir / f"o{k:04d}.png")), [cv2.IMWRITE_JPEG_QUALITY, 97])
        one = work / "one"; shutil.rmtree(one, ignore_errors=True); one.mkdir()
        (one / "00000.jpg").symlink_to(jpg / f"{k:05d}.jpg")
        res, probs = {}, {}
        for p in PROMPTS:
            sid = pred.handle_request({"type": "start_session", "resource_path": str(one)})["session_id"]
            r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": p})
            pred.handle_request({"type": "close_session", "session_id": sid})
            o = r["outputs"]; m = np.asarray(o["out_binary_masks"])
            res[p] = (m.any(0) if len(m) else np.zeros(cv2.imread(str(jpg / f"{k:05d}.jpg"), 0).shape, bool)).astype(np.uint8)
            probs[p] = float(max(np.asarray(o.get("out_probs", [0])).tolist() or [0]))
        np.savez_compressed(out, **res, probs=np.array([probs[p] for p in PROMPTS]))
        print(f"kf {k}: " + ", ".join(f"{p}={int(res[p].sum())}" for p in PROMPTS) + f"  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    frames, kf, work = map(Path, sys.argv[1:4])
    keys = [int(x) for x in sys.argv[4].split(",")]
    run(frames, kf, keys, work)
