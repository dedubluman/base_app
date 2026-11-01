"""SAM 3.1 subject masking with an on-disk cache.

Output layout (one folder per clip):
    masks/<stem>/subject/<frame:05d>.png   uint8 0/255, source resolution
    masks/<stem>/meta.json                 per-frame chunk, object id, prob, area

Rendering only reads this cache, so segmentation can run once (or on a
different, GPU-equipped machine via ``--masks-only``) and every later render
skips it.
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .config import Masking


def mask_dir_for(masks_root: Path, stem: str) -> Path:
    return masks_root / stem


def masks_cached(mdir: Path, n_frames: int) -> bool:
    meta = mdir / "meta.json"
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text()).get("num_frames") == n_frames
    except json.JSONDecodeError:
        return False


def extract_jpeg_frames(video: Path, out_dir: Path, quality: int) -> int:
    """SAM's loader accepts a JPEG folder named <index>.jpg. Extracting with
    ffmpeg (not OpenCV) guarantees HEVC 10-bit decodes the same as the render
    path does."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-fps_mode", "passthrough", "-q:v", str(quality),
         "-start_number", "0", str(out_dir / "%05d.jpg")],
        check=True,
    )
    return len(list(out_dir.glob("*.jpg")))


def _build_predictor(cfg: Masking, repo_root: Path, cache_dir: Path):
    import torch

    from .sam31_cpu import import_sam3

    if cfg.cpu_threads:
        torch.set_num_threads(cfg.cpu_threads)
    on_gpu = torch.cuda.is_available()
    mb = import_sam3(repo_root / cfg.sam_root, cache_dir)
    import sam3

    try:  # FlashAttention 3 is optional and GPU-only
        import flash_attn_interface  # noqa: F401

        use_fa3 = on_gpu
    except ImportError:
        use_fa3 = False
    pred = mb.build_sam3_multiplex_video_predictor(
        checkpoint_path=str(repo_root / cfg.checkpoint),
        bpe_path=str(Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz"),
        use_fa3=use_fa3,
        use_rope_real=use_fa3,
        async_loading_frames=False,
        max_num_objects=cfg.max_num_objects,
    )
    # The detector runs `batched_grounding_batch_size` frames through the
    # backbone at once (16 upstream, sized for an H100). Each 1008 px frame
    # costs GBs of activations on CPU, so make it configurable.
    pred.model.batched_grounding_batch_size = cfg.detector_batch
    # Upstream bug (sam3 @ SAM 3.1 release): the base predictor passes
    # `offload_state_to_cpu` to a multiplex init_state() that does not accept
    # it. Drop keyword arguments the target does not declare.
    orig_init = pred.model.init_state
    accepted = set(inspect.signature(orig_init).parameters)
    pred.model.init_state = lambda **kw: orig_init(
        **{k: v for k, v in kw.items() if k in accepted}
    )
    return pred, on_gpu


class _PeakRSS:
    """Background sampler of this process's resident memory (for the log)."""

    def __init__(self):
        import threading

        import psutil

        self.proc, self.peak, self._stop = psutil.Process(), 0, False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.5)

    def take(self) -> float:
        p, self.peak = self.peak, 0
        return p / 2**30


def _chunks(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    out, s = [], 0
    while True:
        e = min(n, s + size)
        out.append((s, e))
        if e == n:
            return out
        s = e - overlap


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def run_sam31(video: Path, mdir: Path, cfg: Masking, repo_root: Path,
              cache_dir: Path, log=print, resume: bool = True) -> dict:
    """Segment the subject through the whole clip and write the mask cache.

    SAM 3.1 keeps backbone features of every processed frame in its session
    state (~100 MB/frame at 1008 px), so a whole clip in one session runs out
    of RAM on modest machines. The clip is processed in overlapping chunks,
    each its own session prompted with the text prompt; the diver is matched
    across chunks by mask IoU on the overlapping frames.
    """
    import gc

    import torch

    frames_dir = cache_dir / f"sam_frames_{video.stem}"
    n = extract_jpeg_frames(video, frames_dir, cfg.jpeg_quality)
    h, w = cv2.imread(str(frames_dir / "00000.jpg")).shape[:2]
    log(f"[sam] {n} frames extracted ({w}x{h})")

    rss = _PeakRSS()
    t0 = time.time()
    pred, on_gpu = _build_predictor(cfg, repo_root, cache_dir)
    log(f"[sam] SAM 3.1 loaded on {'CUDA' if on_gpu else 'CPU (port)'} "
        f"in {time.time() - t0:.0f}s, peak RSS {rss.take():.1f} GB")

    chunks = _chunks(n, cfg.chunk_frames, cfg.chunk_overlap)
    subject = np.zeros((n, h, w), bool)
    have = np.zeros(n, bool)
    frames_meta: dict[int, dict] = {}
    # Resume: a crash (e.g. OOM) keeps every finished chunk on disk.
    prog_f = mdir / "progress.json"
    key = {"n": n, "chunks": [list(c) for c in chunks], "text": cfg.text_prompt,
           "points": cfg.point_prompts}
    done_chunks: set[int] = set()
    if resume and prog_f.exists() and json.loads(prog_f.read_text())["key"] == key:
        prog = json.loads(prog_f.read_text())
        done_chunks = set(prog["done"])
        frames_meta = {int(k): v for k, v in prog["frames"].items()}
        for i in frames_meta:
            subject[i] = cv2.imread(str(mdir / "subject" / f"{i:05d}.png"),
                                    cv2.IMREAD_GRAYSCALE) > 127
            have[i] = True
        log(f"[sam] resuming: {len(done_chunks)}/{len(chunks)} chunks already done")
    else:
        if mdir.exists():
            shutil.rmtree(mdir)
        (mdir / "subject").mkdir(parents=True)
    t_all = time.time()
    for ci, (cs, ce) in enumerate(chunks):
        if ci in done_chunks:
            continue
        t0 = time.time()
        cdir = cache_dir / f"sam_chunk_{video.stem}"
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir()
        for i in range(cs, ce):  # chunk-local <index>.jpg names for SAM
            (cdir / f"{i - cs:05d}.jpg").symlink_to(frames_dir / f"{i:05d}.jpg")
        sid = pred.handle_request(
            {"type": "start_session", "resource_path": str(cdir)})["session_id"]
        pred.handle_request({"type": "add_prompt", "session_id": sid,
                             "frame_index": 0, "text": cfg.text_prompt})
        for f_idx, x, y, label, obj_id in cfg.point_prompts:
            if cs <= int(f_idx) < ce:
                pred.handle_request({
                    "type": "add_prompt", "session_id": sid,
                    "frame_index": int(f_idx) - cs,
                    "points": torch.tensor([[x / w, y / h]], dtype=torch.float32),
                    "point_labels": torch.tensor([int(label)], dtype=torch.int32),
                    "obj_id": int(obj_id)})
        per_obj: dict[int, dict[int, np.ndarray]] = {}
        probs: dict[int, dict[int, float]] = {}
        for resp in pred.handle_stream_request(
                {"type": "propagate_in_video", "session_id": sid}):
            fi = cs + int(resp["frame_index"])
            out = resp["outputs"]
            pr = list(out.get("out_probs", [])) or [None] * len(out["out_obj_ids"])
            for oid, m, p in zip(out["out_obj_ids"], out["out_binary_masks"], pr):
                per_obj.setdefault(int(oid), {})[fi] = np.asarray(m) > 0
                probs.setdefault(int(oid), {})[fi] = None if p is None else float(p)
        pred.handle_request({"type": "close_session", "session_id": sid})

        # Which object in this chunk is the diver?
        overlap = [i for i in range(cs, ce) if have[i]]
        def score(oid):
            fr = per_obj[oid]
            if overlap:  # continuity with what we already have
                return float(np.mean([_iou(fr[i], subject[i]) if i in fr else 0.0
                                      for i in overlap]))
            areas = [m.sum() for m in fr.values() if m.any()]
            return len(areas) * float(np.median(areas)) if areas else 0.0
        chosen = max(per_obj, key=score) if per_obj else None
        for i in range(cs, ce):
            m = per_obj.get(chosen, {}).get(i) if chosen is not None else None
            if m is None:
                continue
            if have[i]:
                # overlap: union keeps the mask stable across the seam
                subject[i] |= m
            else:
                subject[i] = m
                have[i] = True
            frames_meta[i] = {"chunk": ci, "obj_id": chosen,
                              "prob": probs.get(chosen, {}).get(i),
                              "objects_in_chunk": sorted(per_obj)}
        log(f"[sam] chunk {ci + 1}/{len(chunks)} frames {cs}-{ce - 1}: "
            f"objects {sorted(per_obj)} -> subject {chosen} "
            f"(score {score(chosen) if chosen is not None else 0:.2f}), "
            f"{time.time() - t0:.0f}s, peak RSS {rss.take():.1f} GB")
        del per_obj
        for i in range(cs, ce):
            if have[i]:
                cv2.imwrite(str(mdir / "subject" / f"{i:05d}.png"),
                            subject[i].astype(np.uint8) * 255)
        done_chunks.add(ci)
        prog_f.write_text(json.dumps({"key": key, "done": sorted(done_chunks),
                                      "frames": {str(k): v for k, v in frames_meta.items()}}))
        gc.collect()

    for i in range(n):
        cv2.imwrite(str(mdir / "subject" / f"{i:05d}.png"),
                    subject[i].astype(np.uint8) * 255)
        frames_meta.setdefault(i, {"chunk": None, "obj_id": None, "prob": None})
        frames_meta[i]["area"] = int(subject[i].sum())
    meta = {"num_frames": n, "width": w, "height": h,
            "device": "cuda" if on_gpu else "cpu",
            "text_prompt": cfg.text_prompt,
            "chunks": chunks,
            "frames": {str(k): v for k, v in sorted(frames_meta.items())},
            "seconds": round(time.time() - t_all, 1)}
    (mdir / "meta.json").write_text(json.dumps(meta, indent=1))
    prog_f.unlink(missing_ok=True)
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(cache_dir / f"sam_chunk_{video.stem}", ignore_errors=True)
    log(f"[sam] done: {n} frames in {meta['seconds']}s")
    return meta


def load_subject_masks(mdir: Path) -> tuple[np.ndarray, dict]:
    """Return (masks[T,H,W] uint8 0/255, meta). Frames where the subject was
    not found are all zeros (handled as "lost" by the separation step)."""
    meta = json.loads((mdir / "meta.json").read_text())
    t, h, w = meta["num_frames"], meta["height"], meta["width"]
    masks = np.zeros((t, h, w), np.uint8)
    for p in (mdir / "subject").glob("*.png"):
        masks[int(p.stem)] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return masks, meta
