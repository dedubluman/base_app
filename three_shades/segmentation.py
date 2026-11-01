"""Subject isolation: one soft-edged diver mask per source frame.

Primary backend is SAM 3.1 (Object Multiplex video predictor) from
github.com/facebookresearch/sam3, which needs a CUDA GPU. Masks are cached to
an .npz file, so SAM runs once and every later render (new timings, grade,
ramp...) reuses them; a GPU machine or Colab can produce the cache and a
laptop can render from it.

"maskrcnn-preview" is a CPU-friendly stand-in (torchvision Mask R-CNN) to tune
the edit without a GPU. Its edges are visibly worse; use SAM 3.1 for finals.
"""

from __future__ import annotations

import functools
import inspect
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import SegParams
from .video_io import read_frames


# ─── Cache ──────────────────────────────────────────────────────────────────


def save_masks(path: str | Path, masks: np.ndarray, meta: dict) -> None:
    n, h, w = masks.shape
    np.savez_compressed(
        path, bits=np.packbits(masks.reshape(n, -1), axis=1), shape=np.array([n, h, w]),
        **{f"meta_{k}": np.array(v) for k, v in meta.items()},
    )


def load_masks(path: str | Path, width: int, height: int) -> np.ndarray:
    data = np.load(path)
    n, h, w = (int(v) for v in data["shape"])
    masks = np.unpackbits(data["bits"], axis=1, count=h * w).reshape(n, h, w).astype(bool)
    if (w, h) != (width, height):
        masks = np.stack([
            cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR) > 0
            for m in masks
        ])
    return masks


# ─── SAM 3.1 ────────────────────────────────────────────────────────────────


def _export_jpegs(video: str, width: int, height: int, out_dir: Path, quality: int) -> int:
    n = 0
    for n, frame in enumerate(read_frames(video, width, height)):
        bgr = cv2.cvtColor((frame * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return n + 1


def _patch_init_state(predictor) -> None:
    """Compatibility shim for the current sam3 code.

    Sam3BasePredictor.start_session() always forwards ``offload_state_to_cpu``,
    but the SAM 3.1 multiplex model's init_state() does not accept it, so
    start_session raises TypeError. Drop keyword arguments it doesn't know.
    """
    model = predictor.model
    original = model.init_state
    accepted = set(inspect.signature(original).parameters)

    @functools.wraps(original)
    def init_state(**kwargs):
        return original(**{k: v for k, v in kwargs.items() if k in accepted})

    model.init_state = init_state


def _select_subject(per_frame: dict[int, dict], point: Optional[tuple[float, float]],
                    prompt_frame: int) -> Optional[int]:
    """Pick the diver among all tracked objects.

    With a click point: the object whose mask contains it on the prompt frame.
    Otherwise: the object with the largest total mask area over the clip
    (the hero is close to camera; background swimmers are small).
    """
    if point is not None and prompt_frame in per_frame:
        x, y = int(point[0]), int(point[1])
        for obj_id, mask in per_frame[prompt_frame].items():
            if mask[y, x]:
                return obj_id
    area: dict[int, int] = {}
    for objs in per_frame.values():
        for obj_id, mask in objs.items():
            area[obj_id] = area.get(obj_id, 0) + int(mask.sum())
    return max(area, key=area.get) if area else None


def run_sam31(video: str, width: int, height: int, p: SegParams) -> np.ndarray:
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3.1's multiplex video predictor requires a CUDA GPU. Run the "
            "'segment' step on a GPU machine (or Colab) and copy the .npz mask "
            "cache back, or use --segmenter maskrcnn-preview for a CPU preview."
        )

    with tempfile.TemporaryDirectory(prefix="sam31_frames_") as tmp:
        frames_dir = Path(tmp)
        print("[sam3.1] exporting frames ...", flush=True)
        n_frames = _export_jpegs(video, width, height, frames_dir, p.jpeg_quality)

        print("[sam3.1] loading model ...", flush=True)
        ckpt = p.checkpoint if p.checkpoint and Path(p.checkpoint).exists() else None
        predictor = build_sam3_multiplex_video_predictor(checkpoint_path=ckpt)
        _patch_init_state(predictor)

        session_id = predictor.handle_request(
            dict(type="start_session", resource_path=str(frames_dir))
        )["session_id"]

        # Open-vocabulary concept prompt: every person in the clip is detected
        # and tracked (detector + tracker on every frame), and the diver is
        # selected afterwards; subject_point only picks among the tracked people.
        # Point-only prompts are NOT used: in the current sam3 code the
        # multiplex model's point (tracker-only) propagation drops its masks
        # on frames without a prior text/box propagation (_build_sam2_output
        # returns {}), so a points-only session yields empty masks.
        predictor.handle_request(dict(
            type="add_prompt", session_id=session_id,
            frame_index=p.prompt_frame, text=p.text_prompt,
        ))

        per_frame: dict[int, dict[int, np.ndarray]] = {}
        t0 = time.time()
        for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video", session_id=session_id,
            start_frame_index=p.prompt_frame, propagation_direction="both",
        )):
            out = resp["outputs"]
            ids = out["out_obj_ids"].tolist()
            masks = out["out_binary_masks"]
            masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
            per_frame[resp["frame_index"]] = {i: m.astype(bool) for i, m in zip(ids, masks)}
        print(f"[sam3.1] propagated {len(per_frame)} frames in {time.time() - t0:.1f}s")
        predictor.handle_request(dict(type="close_session", session_id=session_id))

    subject = _select_subject(per_frame, p.subject_point, p.prompt_frame)
    print(f"[sam3.1] objects: {sorted({i for o in per_frame.values() for i in o})}, "
          f"subject: {subject}")
    result = np.zeros((n_frames, height, width), dtype=bool)
    for idx, objs in per_frame.items():
        if subject in objs and idx < n_frames:
            m = objs[subject]
            if m.shape != (height, width):
                m = cv2.resize(m.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
            result[idx] = m
    return result


# ─── CPU preview fallback ──────────────────────────────────────────────────


def run_maskrcnn_preview(video: str, width: int, height: int, p: SegParams,
                         work_long_side: int = 960) -> np.ndarray:
    import torch
    import torchvision
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights).eval()
    scale = work_long_side / max(width, height)
    ww, wh = int(round(width * scale)), int(round(height * scale))
    person = weights.meta["categories"].index("person")

    masks: list[np.ndarray] = []
    prev: Optional[np.ndarray] = None
    t0 = time.time()
    for idx, frame in enumerate(read_frames(video, width, height)):
        small = cv2.resize(frame, (ww, wh), interpolation=cv2.INTER_AREA)
        with torch.inference_mode():
            det = model([torch.from_numpy(small).permute(2, 0, 1)])[0]
        keep = (det["labels"] == person) & (det["scores"] > 0.5)
        cands = (det["masks"][keep, 0] > 0.5).numpy()
        chosen = np.zeros((wh, ww), dtype=bool)
        if len(cands):
            if prev is None:
                if p.subject_point is not None:
                    x, y = int(p.subject_point[0] * scale), int(p.subject_point[1] * scale)
                    hits = [i for i, c in enumerate(cands) if c[y, x]]
                    best = hits[0] if hits else int(np.argmax(cands.sum((1, 2))))
                else:
                    best = int(np.argmax(cands.sum((1, 2))))
            else:  # follow the same person: best IoU with the previous mask
                inter = (cands & prev).sum((1, 2))
                union = (cands | prev).sum((1, 2)) + 1
                best = int(np.argmax(inter / union))
                if inter[best] == 0:
                    best = -1
            if best >= 0:
                chosen = cands[best]
        if chosen.any():
            prev = chosen
        masks.append(cv2.resize(chosen.astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_LINEAR) > 0)
        if idx % 25 == 0:
            print(f"[maskrcnn-preview] frame {idx} ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(masks)


def segment(video: str, width: int, height: int, p: SegParams) -> np.ndarray:
    if p.backend == "sam3.1":
        return run_sam31(video, width, height, p)
    if p.backend == "maskrcnn-preview":
        return run_maskrcnn_preview(video, width, height, p)
    raise ValueError(f"unknown segmentation backend: {p.backend}")
