"""reef_noir: reference-driven cinematic underwater grade -> Instagram 9:16.

  python -m reef_noir segment --source in.mov            # SAM 3.1 (two passes)
  python -m reef_noir preview --source in.mov --frames 0 200 420
  python -m reef_noir render  --source in.mov --out out.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shades.io import probe

from .grade import DiverGrade, EnvGrade
from .masks import segment
from .render import render

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(prog="reef_noir")
    ap.add_argument("cmd", choices=["segment", "preview", "render"])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path, default=REPO / "work")
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--params", type=Path, help="JSON with env/diver/rays overrides")
    ap.add_argument("--crop", type=int, default=400)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    info = probe(a.source)
    mask_dir = REPO / "masks" / f"{a.source.stem}_full"
    if a.cmd == "segment":
        segment(a.source, info.n_frames, (info.width, info.height), a.work, REPO, crop=a.crop)
        return
    p = json.loads(a.params.read_text()) if a.params else {}
    env, diver = EnvGrade(**p.get("env", {})), DiverGrade(**p.get("diver", {}))
    if a.cmd == "preview":
        rep = render(a.source, None, mask_dir, env, diver, p.get("rays", 0.0),
                     frames=a.frames, preview_dir=a.work / "preview")
    else:
        rep = render(a.source, a.out, mask_dir, env, diver, p.get("rays", 0.0))
        (a.work / "render_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("env", "diver")}))


if __name__ == "__main__":
    main()
