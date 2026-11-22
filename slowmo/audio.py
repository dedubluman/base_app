"""Retime the original audio to follow the picture's speed ramp.

Normal-speed stretches are copied sample-exact. Slowed stretches are time-stretched
(pitch preserved) with Rubber Band's smooth/laminar mode; the ramps are cut into short
constant-tempo pieces (2-3 frames) whose steps are < 0.06 in tempo, joined with 12 ms
equal-power crossfades so there is no click, wobble or pitch shift."""
from __future__ import annotations
import subprocess
import numpy as np

SR = 44100


def _read(path, af=None, sr=SR):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn"]
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    return np.frombuffer(subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout, np.float32).reshape(-1, 2)


def _stretch(src: np.ndarray, tempo: float) -> np.ndarray:
    if abs(tempo - 1.0) < 1e-4:
        return src
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
         "-af", f"rubberband=tempo={tempo:.5f}:pitch=1:transients=smooth:detector=soft:phase=laminar:window=long",
         "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
        input=np.ascontiguousarray(src).tobytes(), check=True, stdout=subprocess.PIPE)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2)


def plan_segments(pos: np.ndarray, chunk: int = 3):
    """[(first_out_frame, last_out_frame_exclusive, tempo)] covering every output frame.
    Unity-speed runs and the constant-speed hold are single segments; ramps are cut into
    `chunk`-frame pieces."""
    steps = np.diff(pos)                       # source frames advanced per output frame
    segs, i = [], 0
    while i < len(steps):
        j = i + 1
        if abs(steps[i] - 1.0) < 1e-6 or (abs(steps[i] - steps[i + 1 if i + 1 < len(steps) else i]) < 1e-6
                                          and abs(steps[i] - 0.5) < 1e-6):
            while j < len(steps) and abs(steps[j] - steps[i]) < 1e-6:
                j += 1
        else:
            while j < len(steps) and j - i < chunk and abs(steps[j] - 1.0) > 1e-6 and abs(steps[j] - 0.5) > 1e-6:
                j += 1
        segs.append((i, j, float((pos[j] - pos[i]) / (j - i))))
        i = j
    return segs


def retime_audio(path, pos: np.ndarray, fps: float, src_fps: float, xfade_ms: float = 12.0) -> np.ndarray:
    a = _read(path)
    n_out = int(round(len(pos) / fps * SR))
    out = np.zeros((n_out + SR, 2), np.float32)
    xf = int(SR * xfade_ms / 1000)
    half = xf // 2
    segs = plan_segments(pos)
    for si, (i, j, tempo) in enumerate(segs):
        T0, T1 = int(round(i / fps * SR)), int(round(j / fps * SR))          # nominal output span (samples)
        s0, s1 = pos[i] / src_fps, pos[j] / src_fps                          # source span (s)
        ctx = int(0.08 * SR)   # context so Rubber Band's latency/edge handling never touches the kept span
        a0 = max(0, int(s0 * SR) - ctx); a1 = min(len(a), int(s1 * SR) + ctx)
        if a1 <= a0:
            continue
        piece = _stretch(a[a0:a1], tempo)
        o0 = T0 - int(round((s0 * SR - a0) / tempo))                         # output index of piece[0]
        idx = o0 + np.arange(len(piece))
        w = np.ones(len(piece), np.float32)
        if si > 0:
            u = np.clip((idx - (T0 - half)) / xf, 0, 1)
            w *= np.sin(u * np.pi / 2).astype(np.float32)
        if si < len(segs) - 1:
            u = np.clip((idx - (T1 - half)) / xf, 0, 1)
            w *= np.cos(u * np.pi / 2).astype(np.float32)
        k0 = max(o0, 0)
        out[k0:o0 + len(piece)] += piece[k0 - o0:] * w[k0 - o0:, None]
    return out[:n_out]


def write_wav(path, x: np.ndarray):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-c:a", "pcm_f32le", str(path)], input=np.ascontiguousarray(x).tobytes(), check=True)
