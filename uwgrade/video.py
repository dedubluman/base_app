"""Video I/O: probe, high-bit-depth decode through ffmpeg pipes, ProRes encode.

The source is 8-bit HEVC, BT.709, limited range (tagged). Decoding goes
through swscale to *full-range RGB* with the BT.709 matrix, 16-bit for the
grade (``rgb48le``) so no precision is lost before the float pipeline, 8-bit for
the mask/QC cache.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

SCALE = ("scale=in_range=tv:out_range=pc:in_color_matrix=bt709:"
         "flags=accurate_rnd+full_chroma_int+bicubic")


def probe(path: Path) -> dict:
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {"w": int(v["width"]), "h": int(v["height"]), "fps_num": num, "fps_den": den,
            "fps": num / den, "frames": int(v.get("nb_frames") or 0),
            "duration": float(j["format"]["duration"]), "codec": v["codec_name"],
            "pix_fmt": v["pix_fmt"], "range": v.get("color_range"), "matrix": v.get("color_space"),
            "trc": v.get("color_transfer"), "audio": a["codec_name"] if a else None,
            "audio_duration": float(a["duration"]) if a else None}


def decode(path: Path, bits: int = 8, start: int = 0, count: int | None = None):
    """Yield (frame_index, HxWx3 array) of full-range BT.709-matrixed RGB.
    bits=8 -> uint8, bits=16 -> uint16."""
    info = probe(path)
    w, h = info["w"], info["h"]
    pix = "rgb24" if bits == 8 else "rgb48le"
    dt, bpp = (np.uint8, 3) if bits == 8 else (np.dtype("<u2"), 6)
    vf = f"select='gte(n\\,{start})'," if start else ""
    vf += SCALE
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", vf, "-fps_mode", "passthrough",
           "-pix_fmt", pix, "-f", "rawvideo", "-"]
    if count is not None:
        cmd[cmd.index("-an"):cmd.index("-an")] = ["-frames:v", str(count)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    size = w * h * bpp
    i = start
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield i, np.frombuffer(buf, dt).reshape(h, w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def frame_cache(path: Path, cache: Path) -> np.ndarray:
    """uint8 RGB cache of every frame as a read-only memmap (T,H,W,3)."""
    info = probe(path)
    shape = (info["frames"], info["h"], info["w"], 3)
    if cache.exists():
        mm = np.load(cache, mmap_mode="r")
        if mm.shape == shape:
            return mm
    mm = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=shape)
    n = 0
    for i, fr in decode(path, 8):
        mm[i] = fr
        n += 1
    mm.flush()
    assert n == shape[0], f"decoded {n} frames, expected {shape[0]}"
    return np.load(cache, mmap_mode="r")


class ProResWriter:
    """Pipe float/uint16 RGB frames into ffmpeg -> ProRes 422 HQ (10-bit 4:2:2,
    BT.709 tagged, limited range), copying the source audio untouched."""

    def __init__(self, out: Path, src: Path, w: int, h: int, fps_num: int, fps_den: int):
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}", "-r", f"{fps_num}/{fps_den}", "-i", "-",
               "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
               "-vf", "scale=out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int+bicubic,format=yuv422p10le",
               "-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0", "-bits_per_mb", "8000",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-movflags", "+write_colr", str(out)]   # no -shortest: it trimmed the last video frame (video is 2 ms longer than the audio)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb16: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(rgb16, "<u2").tobytes())

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg ProRes encode failed (rc={rc})")
