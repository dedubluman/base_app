#!/usr/bin/env python3
"""SAM 3.1 text-prompt masks for single frames (parts of the diver). Same checkpoint/CPU port as the video tracker:
every frame is its own 1-frame session. Frames are fed at ~960 px width; masks are saved at that size.

  python sam_parts.py --tag after_fin --image path.jpg
  python sam_parts.py --video v.mov --frames 15 30 524 --prompts "swim shorts" fins hair
Writes .cache/parts/<tag>.npz  (person, shorts, fins, hair -> bool arrays at 960 px width)
"""
import argparse, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np, torch
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from shades.config import Masking
from shades.masking import _build_predictor

PROMPTS = {"person": "person", "shorts": "swim shorts", "fins": "fins", "hair": "hair"}


def run(pred, img_path: Path, keys, work: Path):
    out = {}
    for key in keys:
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        shutil.copy(img_path, work / "00000.jpg")
        sid = pred.handle_request({"type": "start_session", "resource_path": str(work)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": PROMPTS[key]})
        ms = []
        for r in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            o = r["outputs"]
            ms += [np.asarray(m) > 0 for m in o["out_binary_masks"]]
        pred.handle_request({"type": "close_session", "session_id": sid})
        h, w = cv2.imread(str(img_path)).shape[:2]
        out[key] = np.any(ms, axis=0) if ms else np.zeros((h, w), bool)
        out[key + "_n"] = len(ms)
        print(f"  {img_path.name} '{PROMPTS[key]}': {len(ms)} obj, area {int(out[key].sum())}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag"); ap.add_argument("--image", type=Path, action="append")
    ap.add_argument("--video", type=Path); ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--keys", nargs="*", default=["person", "shorts", "fins", "hair"])
    a = ap.parse_args()
    cache = REPO / ".cache"; pd = cache / "parts"; pd.mkdir(parents=True, exist_ok=True)
    jobs = []   # (tag, jpg path)
    if a.image:
        for p in a.image:
            im = cv2.imread(str(p)); s = 960 / im.shape[1]
            jp = pd / f"{p.stem}_in.jpg"; cv2.imwrite(str(jp), cv2.resize(im, (960, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            jobs.append((p.stem, jp))
    if a.video:
        sel = "+".join(f"eq(n\\,{i})" for i in a.frames)
        tmp = pd / "kf_tmp"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a.video), "-vf", f"select='{sel}',scale=960:-2", "-fps_mode", "passthrough", "-q:v", "2", str(tmp / "%04d.jpg")], check=True)
        for f, jp in zip(sorted(a.frames), sorted(tmp.glob("*.jpg"))):
            jobs.append((f"f{f:04d}", jp))
    cfg = Masking(text_prompt="person", parts={}, max_num_objects=4)
    pred, _ = _build_predictor(cfg, REPO, cache)
    for tag, jp in jobs:
        dst = pd / f"{tag}.npz"
        if dst.exists():
            continue
        res = run(pred, jp, a.keys, cache / "sam_one")
        np.savez_compressed(dst, **{k: v for k, v in res.items()})
    print("PARTS_DONE", flush=True)

if __name__ == "__main__":
    main()
