# Deep-abyss grade (SAM 3.1 guided)

Turns flat turquoise reef footage into a cinematic *deep abyss* look: midnight / petrol-blue water, desaturated
teal seabed, natural depth falloff and vignette, with the **divers, their bubble trails and the fish** lifted
out of the dark by gentle directional light. Geometry, poses, equipment, bubbles and fish positions are never
altered: every effect is a tone / colour / light change driven by the source pixels.

Output: **1080p H.264 MP4** (`libx264`, CRF 16, `yuv420p`, BT.709, `+faststart`), same frame count / fps as the
source, original audio stream-copied. A vertical 1080x1920 source stays 1080x1920; any other size is scaled so the
short edge is 1080 (4K landscape -> 1920x1080).

```
inputs/dive.mov
   |  abyss_masks.py      SAM 3.1 on every 6th frame: "scuba diver" + "fish"  -> masks/<stem>__abyss/<job>/kf_*.png
   |  abyss_propagate.py  optical-flow carry to every frame + guided-filter edges -> masks/<stem>__abyss/<job>_full/*.png
   |  abyss_fit.py        (optional) fit the global look to a reference still -> abyss_look.json
   |  abyss_grade.py      global grade + mask-guided layers + H.264 render
   v  abyss_qc.py         compare / sheet / check
output/dive_abyss_1080p.mp4
```

## Run it

```bash
# one-time setup (see requirements_abyss.txt): torch, deps, SAM 3 code + SAM 3.1 weights (HF_TOKEN)
pip install -r requirements_abyss.txt && ./scripts/download_sam3.sh

# 1. masks (SAM 3.1; ~12 s per keyframe on 4 CPU cores, minutes on a GPU) - resumable, cached
python abyss_masks.py     --input inputs/dive.mov
python abyss_propagate.py --input inputs/dive.mov

# 2. (optional) re-fit the global look to another reference still of one of the video's frames
python abyss_fit.py --video inputs/dive.mov --frame 241 --reference reference.jpg --out abyss_look.json

# 3. test on stills first (seconds), then render
python abyss_grade.py --input inputs/dive.mov --frames 241,0,300 --stills output/stills
python abyss_qc.py compare --video inputs/dive.mov --frame 241 --graded output/stills/g00241.png --reference reference.jpg --sheet output/compare.jpg
python abyss_grade.py --input inputs/dive.mov --output output/dive_abyss_1080p.mp4

# 4. verify
python abyss_qc.py check --video inputs/dive.mov --graded output/dive_abyss_1080p.mp4
python abyss_qc.py sheet --video inputs/dive.mov --graded output/dive_abyss_1080p.mp4 --out output/sheet.jpg --overlay
```

`--no-masks` renders the global grade only (no SAM needed) - useful for a quick look; the divers and fish then
simply stay dim like the rest of the scene.

## How the grade works

| layer | what it does |
|---|---|
| **water / seabed** (whole frame) | Subject-free base luma (divers and fish are *filled in* from their surroundings by a normalised blur, so a dark wetsuit cannot pull a dark halo into the water) -> fitted tone curve `Y = gain * softmax(n - black, 0)^gamma` times an elliptical vignette (darker toward edges, far water and the distant seabed). Chroma comes from the source's own smoothed R/G and B/G ratios through a fitted linear model, so the teal-green seabed and the bluer water keep their natural separation. Mid-frequency detail and marine snow are kept (1.1x / 0.8x); a soft toe and a small floor keep shadows textured, never flat black. |
| **light shafts + bubbles** | The source's own vertical rays are gently lifted in the upper water. A soft column of light is anchored above every diver; inside that column the *existing* bright bubble specks are lifted (band-pass of the source, gated, capped, faint scatter). No bubble is added, moved or sharpened into existence. |
| **divers** (SAM mask) | Own tone mapping from their measured luma range to a lifted but dark range (silhouettes stay silhouettes), local clarity on straps / fins, light from above (Lambert on the mask's own normals: broad top light + thin rim). Edge pixels are *decontaminated* first (bright water mixed into the mask edge is replaced by values extrapolated from the solid core) so no white outline or cut-out edge appears. |
| **fish** (SAM mask) | Same machinery with a brighter silver range, gentler contrast, reduced chroma so they stay silver, and a soft top light. |
| **finish** | Soft highlight knee, TPDF dither at the 8-bit step (no banding in the dark gradients), x264 `aq-mode=3` (bits go to dark flat areas). |

### Temporal consistency
* One fixed global look for the whole clip (the source's mean luma varies only 0.531-0.544, so there is no exposure
  pumping to chase; nothing is auto-exposed per frame).
* Masks: SAM keyframes -> bidirectional DIS optical-flow warps blended by distance -> 5-tap triangular temporal
  smoothing -> guided filter against the full-res luma (thin fins keep partial coverage, no jitter).
* Per-class tone levels (diver / fish) are measured on the solid mask core, then median + mean filtered over time.
* Light columns are built from the smoothed masks, so they cannot flicker.

## Tuning

Everything is a number. Global look: `abyss_look.json` (written by `abyss_fit.py`; fields documented in
`abyss_look.py`). Local strengths: the `Local` dataclass at the top of `abyss_grade.py`. Override any of them on
the command line, render stills, compare:

```bash
python abyss_grade.py --input inputs/dive.mov --frames 241 --stills output/t \
    --set chroma_gain=0.85 beam=0.04 diver_peak=0.28 fish_peak=0.62 mix=1.0
```

| want | change |
|---|---|
| darker / brighter water | `black` up/down (0.34-0.40), `gain` |
| stronger / weaker vignette | `vig_x`, `vig_y` |
| more / less colour in the water | `chroma_gain` (0.8-1.2) |
| deeper shadows | `toe` down, `floor` down |
| brighter divers | `diver_peak`, `diver_top_light`, `diver_rim` |
| softer light columns | `beam`, `beam_w`, `beam_h`, `rays` |
| bubble glow | `bubble_light`, `bubble_cap` |
| fish more / less pop | `fish_floor`, `fish_peak`, `fish_clarity`, `fish_chroma` |
| A/B against the original | `mix` (0 = source, 1 = full grade) |

## SAM usage (why and how)

Global tone / colour cannot separate a dark diver or a silver fish from the water without bleeding into it, so
SAM 3.1 is used for exactly those two classes - and only for them. Bubbles are *not* segmented (SAM is unreliable on
sub-pixel specks); they are lit from the source's own luma inside the diver light column.

* The repo's SAM 3.1 multiplex predictor is used (`shades/masking.py`, CPU port in `shades/sam31_cpu.py` when no CUDA).
* Prompts: `"scuba diver"` and `"fish"`, one session per prompt and chunk; all instances of a class are merged into
  one mask per keyframe, one-off detections are dropped (`--min-prob 0.45`, `--min-life 3`).
* SAM resizes input to a 1008x1008 square, which squashes a 9:16 frame; each job therefore feeds SAM a *square crop*
  (top square for divers, bottom square for fish) so thin fins and tails keep their shape.
* SAM runs on every 6th frame (`--stride`) to keep the CPU cost at ~80 keyframes per class; use `--stride 1` on a GPU.

## Files

| file | role |
|---|---|
| `abyss_masks.py` | SAM 3.1 keyframe masks (resumable) |
| `abyss_propagate.py` | optical-flow propagation + temporal smoothing + edge refinement |
| `abyss_look.py` / `abyss_look.json` | parametric global look (tone, vignette, chroma) and its fitted values |
| `abyss_fit.py` | fit the look to a reference still |
| `abyss_grade.py` | the grade and the H.264 renderer |
| `abyss_qc.py` | region-by-region comparison to the reference, contact sheets, output checks |
| `abyss_common.py` | ffprobe / ffmpeg IO and image maths |
