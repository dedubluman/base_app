# fish_grade: cinematic underwater grade with SAM 3.1 mattes

Takes a handheld underwater clip of a striped fish (1080x1920, 29.97 fps, with
audio) and grades it toward a reference still: a deep, claustrophobic
teal / slate-green world with the fish as the one bright, warm, clearly
defined subject. Output is **1080x1920 H.264 (libx264) yuv420p MP4**, same
frame rate and frame count as the source, source AAC audio copied untouched.

```
python fish_masks.py --input inputs/fish.mov --name fish --crop-y 600     # SAM 3.1 keyframe masks (slow)
python fish_grade.py alpha  --input inputs/fish.mov --masks masks/fish    # soft mattes for every frame
python fish_grade.py still  --input inputs/fish.mov --masks masks/fish --frames 0,264,528   # QC stills
python fish_grade.py render --input inputs/fish.mov --masks masks/fish --output output/fish_graded.mp4
python fish_grade.py sheet  --input inputs/fish.mov --graded output/fish_graded.mp4 --masks masks/fish --frames 0,100,200,300,400,528,640
python fish_grade.py qc     --input inputs/fish.mov --graded output/fish_graded.mp4 --masks masks/fish
```

Setup is the one in `README_three_shades.md` (Python 3.12 venv, CPU or CUDA
torch, `pip install -e sam3/`, `requirements_three_shades.txt`) plus
`pip install einops ninja`. SAM runs through the repo's CPU port
(`shades/sam31_cpu.py`) when there is no CUDA GPU.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Segment | `fish_masks.py` | SAM 3.1 multiplex video predictor, text prompt `"fish"`. SAM is fed a **square 1080x1080 crop** (rows 600-1680 cover the fish's whole travel range) so its 1008x1008 resize does not squash the fish. **Every 2nd frame** is segmented (13-15 s/frame on 4 CPU cores); keyframes are cut into overlapping 40-frame chunks, the fish is re-identified across chunk seams by mask IoU, and every finished chunk is saved so a crash resumes. |
| 2. Matte | `fish_grade.py alpha` | Odd frames: both neighbouring SAM masks are warped onto the frame with DIS optical flow and averaged. Every frame: a guided filter on the real image snaps the matte to the true edge (fins, motion blur keep partial coverage), it is held inside a band around the SAM mask (no leaks or holes) and feathered ~1 px. The interpolation is validated against real SAM masks on held-out keyframes (see `masks/fish/alpha_report.json`). |
| 3. Grade | `fish_grade.py render` | Per-frame, static parameters (no per-frame adaptation, so no grade flicker): background look, fish look through the matte, film finish. |
| 4. Encode | same | ffmpeg pipe: 16-bit RGB -> BT.709 limited-range yuv420p, libx264 `-crf 17 -preset slow`, source AAC `-c:a copy`, `+faststart`. |

## The look (`LOOK` in `fish_grade.py`)

* **Background** (inverse matte). A per-channel CDL *fitted to the reference
  still* by `fit_reference.py` (the reference is frame 528 of this clip, so the
  two images pair pixel for pixel): red is crushed hardest (power 2.05 vs 1.47
  green, 1.27 blue), which turns the yellow-green water into deep teal and
  slate-green. On top: soft luminance knee that flattens bright sand, cool-neutral
  highlights, lifted teal-navy blacks (never crushed to flat black), slate
  desaturation of the upper mids, and a radial + top/bottom vignette with
  perimeter desaturation and micro-contrast falloff. Fit quality on the
  reference frame: mean RGB within 0.001, vignette ring luminance within ~4%.
* **Fish** (matte). Its own grade: white-balance correction of the yellow-green
  water cast (cream segments read ivory), small exposure lift, contrast curve
  (crisp espresso bars), edge-aware clarity (guided-filter base, so no halos),
  vibrance, an amber push limited to *truly saturated* gold (tail, fins, belly
  patch), highlight protection (stays below 0.95), and only ~35% of the vignette
  so it stays the focal point.
* **Finish.** Lifted black point (0.004), gentle filmic S-curve, highlight
  roll-off above 0.86, fine luminance-weighted grain (sigma 0.011, deterministic
  per frame) that also dithers the 8-bit encode.

## Tuning

`--look overrides.json` (any subset of `LOOK`, e.g.
`{"bg": {"vig_k": 0.5}, "fg": {"exposure": 1.0}}`) works for `still` and `render`.

| Parameter | Effect |
|---|---|
| `bg.vig_k` | vignette strength (0.66 = about two thirds of the reference's) |
| `bg.knee`, `bg.ceiling` | where / how hard bright sand is flattened |
| `bg.slope/offset/power` | the fitted CDL (red power up = more teal) |
| `bg.lift` | colour of the lifted blacks |
| `fg.exposure`, `fg.wb` | fish brightness / ivory vs yellow |
| `fg.clarity`, `fg.texture` | local contrast on the fish (keep modest: the source is 8-bit H.264) |
| `fg.amber_*` | strength of the golden tail / fins |
| `finish.grain` | grain amount |

## Verified on the supplied clip

| Check | Result |
|---|---|
| SAM 3.1 keyframes | 323/323 masked in 81 min on 4 CPU cores; chunk-seam IoU 0.97-0.99 |
| Flow interpolation (held-out keyframes, 2x the real gap) | IoU mean 0.981, min 0.948 |
| Matte stability | max frame-to-frame area change 2.7%; no jump above 12% |
| Output | H.264 High, 1080x1920, yuv420p, 29.97 fps, 645 frames, 21.5215 s (= source), BT.709; AAC copied untouched; full decode error-free |
| Flicker added by the grade | <= 1% of layer mean (0.0006-0.0012 in 0..1); step correlation with the source 0.93 (bg luma) / 0.997 (fish luma) |
| Reference frame 528 | background mean RGB within ~8%, luminance rings/bands within ~6%; fish luminance p2/p98 0.022/0.75 vs reference 0.024/0.726; fish dE76 median 6.5 |
| Artifacts | warm/pink background pixels <= 0.09%, clipped highlights <= 0.001%, no visible fringe at the matte edge |

The CRF 17 file is large (about 25 Mb/s, 67 MB) because grain costs bits; use
`--crf 20` for a smaller file.

## Limits

* SAM 3.1 runs on **every 2nd frame**; the frames in between use flow-warped
  SAM masks (see the held-out IoU in `alpha_report.json`). Run `fish_masks.py
  --stride 1` for a SAM mask on every frame (about 2.4 h on 4 CPU cores).
* The grade is a single static look. Frames with very different light share it
  by design.
* Thin seagrass blades crossing in front of the fish are only as clean as SAM's
  mask there.
