"""Delivery checks + QC sheets for the Instagram render."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,codec_name,profile,width,height,r_frame_rate,"
                          "avg_frame_rate,pix_fmt,nb_read_frames,color_primaries,color_transfer,"
                          "color_space,bit_rate:format=duration,bit_rate", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def check(path: Path) -> dict:
    d = ffprobe(path)
    st = d["streams"]
    v = [s for s in st if s["codec_type"] == "video"][0]
    res = {
        "video_streams": sum(s["codec_type"] == "video" for s in st),
        "audio_streams": sum(s["codec_type"] == "audio" for s in st),
        "codec": v["codec_name"], "profile": v.get("profile"),
        "size": f'{v["width"]}x{v["height"]}', "r_frame_rate": v["r_frame_rate"],
        "avg_frame_rate": v["avg_frame_rate"], "pix_fmt": v["pix_fmt"],
        "frames": int(v["nb_read_frames"]), "duration": float(d["format"]["duration"]),
        "bitrate_mbps": round(int(d["format"]["bit_rate"]) / 1e6, 2),
        "tags": [v.get("color_primaries"), v.get("color_transfer"), v.get("color_space")],
    }
    res["ok"] = (res["audio_streams"] == 0 and res["codec"] == "h264" and res["profile"] == "High"
                 and res["size"] == "1080x1920" and res["r_frame_rate"] == "30/1"
                 and res["pix_fmt"] == "yuv420p" and res["tags"] == ["bt709"] * 3)
    return res


def temporal_stats(path: Path) -> dict:
    """Per-frame mean luma / chroma; flags jumps (pumping, colour shifts)."""
    cap = cv2.VideoCapture(str(path))
    L, A, B = [], [], []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
        L.append(lab[..., 0].mean()); A.append(lab[..., 1].mean()); B.append(lab[..., 2].mean())
    L, A, B = map(np.array, (L, A, B))
    j = lambda x: float(np.abs(np.diff(x)).max()) if len(x) > 1 else 0.0
    return {"mean_L_range": [float(L.min()), float(L.max())], "max_dL": j(L),
            "max_da": j(A), "max_db": j(B)}


def contact_sheet(path: Path, out: Path, n: int = 12, cols: int = 6) -> None:
    cap = cv2.VideoCapture(str(path))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ims = []
    for i in np.linspace(0, tot - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            ims.append(cv2.resize(im, (270, 480), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(ims[r:r + cols]) for r in range(0, len(ims), cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    p = Path(sys.argv[1])
    r = check(p)
    r.update(temporal_stats(p))
    print(json.dumps(r, indent=1))
    if len(sys.argv) > 2:
        contact_sheet(p, Path(sys.argv[2]))
