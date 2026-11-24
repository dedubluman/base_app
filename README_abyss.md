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
| **divers** (SAM mask) | Their median luma is anchored to a fixed, lifted-but-dark output level (silhouettes stay silhouettes), and only a capped span of their own contrast around it is kept (`*_min_span`), so haze and noise are never stretched into texture. Local clarity on straps / fins, light from above (Lambert on the mask's own normals: broad top light + thin rim). Edge pixels are *decontaminated* first (bright water mixed into the mask edge is replaced by values extrapolated from the solid core) and the alpha is inset a little (`inset`) so no white outline, light ring or cut-out edge appears. |
| **fish** (SAM mask) | Same machinery with a brighter silver level, gentler contrast, the fish's own colour ratios (they land on a silver-blue), and a soft top light. The seabed right next to a fish keeps its texture (detail is only removed under a tight mask). |
| **finish** | Soft highlight knee, TPDF dither at the 8-bit step (no banding in the dark gradients), x264 `aq-mode=3` (bits go to dark flat areas). |

### Temporal consistency
* One fixed global look for the whole clip (the source's mean luma varies only 0.531-0.544, so there is no exposure
  pumping to chase; nothing is auto-exposed per frame).
* Masks: SAM keyframes -> bidirectional DIS optical-flow warps blended by distance -> 5-tap triangular temporal
  smoothing -> guided filter against the full-res luma (thin fins keep partial coverage, no jitter).
* Per-class tone levels (diver / fish: low / high / median luma) are measured on the solid mask core, then median + mean filtered over time.
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
| brighter / darker divers | `diver_floor`, `diver_peak` (median sits mid-range), `diver_top_light`, `diver_rim`; `diver_min_span` / `diver_clarity` = how much of their own contrast is kept |
| softer light columns | `beam`, `beam_w`, `beam_h`, `rays` |
| bubble glow | `bubble_light`, `bubble_cap` |
| fish more / less pop | `fish_floor`, `fish_peak`, `fish_clarity`, `fish_min_span`; colour: `fish_chroma`, `fish_own_color` |
| tighter / looser subject edges | `inset` (0.35), `--min-prob` / `--guide-radius` upstream |
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
* `abyss_propagate.py` drops diver components that are not at least 0.018 darker than the ring around them
  (`--min-contrast`): on the sample clip that removes a far, hazy third diver whose contrast is only 0.006-0.014
  (clear divers: 0.024-0.044). Stretching such a patch into a dark silhouette would invent a blob stronger than the
  footage; unmasked, it is graded with the scene and stays as faint as it is.

## Measured on the sample clip (1080x1920, 480 frames, 16 s)

Reference = your graded still, which is source frame 241 (found by pixel MSE). Numbers are from the **encoded MP4**,
frame 241, region by region (`abyss_qc.py compare`):

| region | graded Y / Lab | reference Y / Lab |
|---|---|---|
| whole frame | 0.171 / 19.2, -9.3, -10.5 | 0.168 / 18.9, -9.5, -9.8 |
| every 20 % row band, edges, centre | within \|dY\| <= 0.018 | |
| fish (mask) | 0.412 / 45.2, -15.7, -5.6 | 0.415 / 45.3, -13.6, -7.0 |
| divers (mask) | 0.139 / 15.5, -6.7, -13.5 | 0.170 / 19.4, -7.6, -14.5 |

Temporal (whole clip, from the MP4): mean luma 0.164-0.183, largest frame-to-frame step 0.0014; mean Y inside the
diver mask 0.129-0.144 (median step 0.0004), inside the fish mask 0.413-0.454 (median step 0.001).
Output checks: H.264 High / yuv420p / 1080x1920 / BT.709 / 30000:1001 fps / 480 frames / 16.016 s / AAC copied, 18.5 MB.

Cost on 4 CPU cores (AMX-BF16): SAM 3.1 ~49 min for both classes (80 keyframes each), propagation ~2.5 min,
render ~5.5 min (0.5 s/frame grade + libx264 `slow`).

## Limitations (read before trusting it blindly)

* The look is fitted to **one** reference frame. It generalised well on the rest of this clip (visual check of six
  frames and whole-clip statistics), but it was not checked against other references or other footage.
* The reference's gear detail on the divers and the scale texture on the fish are synthesised in that image. This
  pipeline only changes tone, colour and light, so the divers stay as soft as the (hazy) source and the fish keep their
  real markings: diver mean luma is 0.139 vs the reference's 0.170 for that reason.
* Masks come from every 6th frame plus optical-flow propagation, not from SAM on every frame. It tracked smoothly on
  this clip, but very fast motion between keyframes would need `--stride 3` or `--stride 1`.
* The source is already 1080x1920, so the scale-to-1080p path in `abyss_grade.py` was not exercised.
* Light shafts / bubble glow are driven by the existing pixels (rays, bubble specks); they cannot be stronger than the
  source allows without inventing detail.

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
