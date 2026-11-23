"""SAM 3.1 segmentation for the deep-water grade.

Everything goes through the SAM 3.1 multiplex video predictor that the repo
already ships (``shades.masking._build_predictor``: repo weights, CPU port of
the *same* model when CUDA is missing). No other segmentation model is used.

Two modes, both resumable (one PNG per frame, written as soon as it exists):

* ``track_diver``  - video session with the text prompt "person", chunked
  (SAM keeps ~100 MB of features per frame), run on every ``stride``-th frame.
  The diver is matched between chunks by IoU and the overlap is cross-faded.
* ``detect_frames`` - single-frame sessions for stuff-classes ("rocks",
  "water", "sky"). SAM 3.1's detector scores amorphous regions low, so the
  detection thresholds are lowered for those prompts and every confident
  instance is unioned. Independent per-keyframe detection avoids a tracker
  dropping a weak masklet.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from shades.config import Masking
from shades.masking import _build_predictor, extract_jpeg_frames


def prepare_frames(video: Path, cache: Path, log=print) -> tuple[Path, int, int, int]:
    d = cache / f"dg_frames_{video.stem}"
    if d.exists() and (d / "00000.jpg").exists():
        n = len(list(d.glob("*.jpg")))
    else:
        n = extract_jpeg_frames(video, d, 2)
    h, w = cv2.imread(str(d / "00000.jpg")).shape[:2]
    log(f"[sam] {n} frames at {w}x{h} in {d}")
    return d, n, w, h


class Sam31:
    def __init__(self, repo: Path, cache: Path, frames: Path, log=print):
        self.log, self.frames, self.cache = log, frames, cache
        t0 = time.time()
        cfg = Masking(chunk_frames=32, detector_batch=2)
        self.pred, on_gpu = _build_predictor(cfg, repo, cache)
        self.model = self.pred.model
        log(f"[sam] SAM 3.1 loaded ({'CUDA' if on_gpu else 'CPU port'}) in {time.time()-t0:.0f}s")

    def set_thresholds(self, det: float, new: float) -> None:
        self.model.score_threshold_detection = det
        self.model.new_det_thresh = new

    # ---------------------------------------------------------------- stuff
    def detect_one(self, frame: int, prompt: str, out_prob: float):
        """Single-frame session -> list of (prob, bool mask)."""
        cdir = self.cache / "dg_one"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True)
        (cdir / "00000.jpg").symlink_to(self.frames / f"{frame:05d}.jpg")
        sid = self.pred.handle_request(
            {"type": "start_session", "resource_path": str(cdir)})["session_id"]
        r = self.pred.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": 0,
            "text": prompt, "output_prob_thresh": out_prob})
        outs = r.get("outputs")
        if not outs:
            for resp in self.pred.handle_stream_request(
                    {"type": "propagate_in_video", "session_id": sid}):
                outs = resp["outputs"]
                break
        self.pred.handle_request({"type": "close_session", "session_id": sid})
        probs = [float(p) for p in outs.get("out_probs", [])]
        masks = [np.asarray(m) > 0 for m in outs["out_binary_masks"]]
        return list(zip(probs, masks))

    def detect_frames(self, mdir: Path, name: str, prompt: str, frames: list[int],
                      det: float, out_prob: float, keep_prob: float,
                      min_area: float = 0.002) -> None:
        out = mdir / name
        out.mkdir(parents=True, exist_ok=True)
        meta_f = out / "meta.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        self.set_thresholds(det, det)
        todo = [f for f in frames if not (out / f"{f:05d}.png").exists()]
        self.log(f"[sam] {name}: '{prompt}' {len(todo)}/{len(frames)} frames to do")
        for k, f in enumerate(todo):
            t0 = time.time()
            res = self.detect_one(f, prompt, out_prob)
            union, kept = None, []
            for p, m in res:
                if p >= keep_prob and m.mean() >= min_area:
                    union = m if union is None else (union | m)
                    kept.append({"prob": round(p, 3), "area": round(float(m.mean()), 4)})
            if union is None:
                union = np.zeros(res[0][1].shape if res else (1, 1), bool)
                if not res:  # shape unknown: take it from the frame
                    h, w = cv2.imread(str(self.frames / f"{f:05d}.jpg")).shape[:2]
                    union = np.zeros((h, w), bool)
            cv2.imwrite(str(out / f"{f:05d}.png"), union.astype(np.uint8) * 255)
            meta[str(f)] = {"kept": kept, "raw": [round(p, 3) for p, _ in res]}
            meta_f.write_text(json.dumps(meta, indent=1))
            self.log(f"[sam] {name} f{f:03d} ({k+1}/{len(todo)}) {time.time()-t0:.0f}s "
                     f"kept={kept}")

    # ---------------------------------------------------------------- diver
    def track_diver(self, mdir: Path, n_total: int, stride: int = 2,
                    chunk: int = 32, overlap: int = 4, prompt: str = "person") -> None:
        out = mdir / "diver"
        out.mkdir(parents=True, exist_ok=True)
        samples = list(range(0, n_total, stride))
        if samples[-1] != n_total - 1:
            samples.append(n_total - 1)
        spans, s = [], 0
        while True:
            e = min(len(samples), s + chunk)
            spans.append((s, e))
            if e == len(samples):
                break
            s = e - overlap
        prog_f = out / "progress.json"
        prog = json.loads(prog_f.read_text()) if prog_f.exists() else {"done": [], "frames": {}}
        key = {"samples": samples, "spans": spans, "prompt": prompt}
        if prog.get("key") != key:
            prog = {"key": key, "done": [], "frames": {}}
        self.set_thresholds(0.4, 0.65)
        h, w = cv2.imread(str(self.frames / "00000.jpg")).shape[:2]
        prev_tail: dict[int, np.ndarray] = {}   # frame -> soft mask of previous chunk
        for ci, (cs, ce) in enumerate(spans):
            ids = samples[cs:ce]
            if ci in prog["done"]:
                # reload overlap masks for IoU matching / cross-fade of next chunk
                prev_tail = {f: cv2.imread(str(out / f"{f:05d}.png"), 0) / 255.0
                             for f in ids[-overlap:]}
                continue
            t0 = time.time()
            cdir = self.cache / "dg_chunk"
            shutil.rmtree(cdir, ignore_errors=True)
            cdir.mkdir(parents=True)
            for j, f in enumerate(ids):
                (cdir / f"{j:05d}.jpg").symlink_to(self.frames / f"{f:05d}.jpg")
            sid = self.pred.handle_request(
                {"type": "start_session", "resource_path": str(cdir)})["session_id"]
            self.pred.handle_request({"type": "add_prompt", "session_id": sid,
                                      "frame_index": 0, "text": prompt})
            per_obj: dict[int, dict[int, np.ndarray]] = {}
            probs: dict[int, dict[int, float]] = {}
            for resp in self.pred.handle_stream_request(
                    {"type": "propagate_in_video", "session_id": sid}):
                j = int(resp["frame_index"])
                o = resp["outputs"]
                pr = list(o.get("out_probs", [])) or [None] * len(o["out_obj_ids"])
                for oid, m, p in zip(o["out_obj_ids"], o["out_binary_masks"], pr):
                    per_obj.setdefault(int(oid), {})[j] = np.asarray(m) > 0
                    probs.setdefault(int(oid), {})[j] = None if p is None else float(p)
            self.pred.handle_request({"type": "close_session", "session_id": sid})

            def score(oid):
                fr = per_obj[oid]
                ov = [j for j, f in enumerate(ids) if f in prev_tail]
                if ov:
                    return float(np.mean([
                        _iou(fr.get(j, np.zeros((h, w), bool)), prev_tail[ids[j]] > 0.5)
                        for j in ov]))
                areas = [m.sum() for m in fr.values() if m.any()]
                return len(areas) * float(np.median(areas)) if areas else 0.0

            chosen = max(per_obj, key=score) if per_obj else None
            for j, f in enumerate(ids):
                m = per_obj.get(chosen, {}).get(j) if chosen is not None else None
                new = np.zeros((h, w), np.float32) if m is None else m.astype(np.float32)
                if f in prev_tail:                       # cross-fade over the overlap
                    k = ids.index(f)
                    a = (k + 1) / (overlap + 1)
                    new = a * new + (1 - a) * prev_tail[f].astype(np.float32)
                cv2.imwrite(str(out / f"{f:05d}.png"), np.round(new * 255).astype(np.uint8))
                prog["frames"][str(f)] = {
                    "chunk": ci, "obj": chosen, "prob": (probs.get(chosen, {}).get(j)
                                                         if chosen is not None else None),
                    "area": float(new.mean())}
            prev_tail = {f: cv2.imread(str(out / f"{f:05d}.png"), 0) / 255.0
                         for f in ids[-overlap:]}
            prog["done"].append(ci)
            prog_f.write_text(json.dumps(prog))
            self.log(f"[sam] diver chunk {ci+1}/{len(spans)} frames {ids[0]}-{ids[-1]} "
                     f"objs={sorted(per_obj)} -> {chosen} score={score(chosen) if chosen is not None else 0:.2f} "
                     f"{time.time()-t0:.0f}s")
            del per_obj


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0
