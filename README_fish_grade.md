# fish_grade — SAM 3.1 mattes + reference-locked underwater grade → H.264 1080p

Moody, high-contrast, enclosed underwater look for a raw clip (`inputs/fish.mov`, 1080×1920, 29.97 fps,
429 frames), matched to two graded reference stills. The fish gets its own matte and its own grade;
the seabed (with the grass/debris touching it) and the background (rock, dense weed, open water) are
graded separately.

```
scripts/run_fish_grade.sh inputs/fish.mov REF_STILL_1.jpg [REF_STILL_2.jpg]     # whole pipeline
```

| Step | Script | Output |
|---|---|---|
| SAM 3.1 keyframe masks | `fish_masks.py` | `masks/<stem>__{fish,sand,fins}/obj_*/<frame>.png` |
| Mattes (clean, per frame) | `fish_refine.py` | `masks/<stem>__mattes/{fish,seabed,midground,background}/<frame>.png`, `overlay.mp4`, `report.json` |
| Reference measurement | `fish_grade.py calibrate` | `grade_params.json` (tone curves, tints, fish chroma window) |
| Grade + encode | `fish_grade.py render` | `output/<stem>_graded.mp4` (libx264 High, yuv420p, BT.709, CRF 15, source AAC copied) |
| LUTs / numbers | `fish_grade.py luts`, `spec` | `output/<stem>_luts/*.cube`, `output/<stem>_grade_numbers.txt` |
| QC | `fish_qc.py stills`, `video` | numeric comparison with the references, flicker on the encoded file |

Colour maths (`fish_color.py`) is OKLab/OKLCH, float32; nothing is 8-bit until the final encode.

## 1. Masking with SAM 3.1

SAM 3.1 (`checkpoints/sam3.1/sam3.1_multiplex.pt`) runs through the repo's CPU port
(`shades/sam31_cpu.py`); the upstream `sam3/` clone is untouched. Install per `README_three_shades.md`
(plus `einops`, which the SAM 3.1 video code imports but does not declare).

**What SAM 3.1 answers on this clip** (text prompt, `add_prompt` response, frames 0 / 240 / 425):

| Prompt | Result |
|---|---|
| `fish` | 1 instance, p 0.94 — used |
| `sandy bottom` | 1 instance on all three frames, p 0.85 / 0.88 / 0.93 — used for the seabed (`sand` found it on 2 of 3 frames, `sea floor` on 1 of 3) |
| `fin` | 1–3 instances, p 0.65–0.88 — used for the pectoral fin where the fish mask drops it |
| `seagrass`, `sea grass`, `eelgrass`, `grass blade`, `seaweed`, `algae`, `vegetation`, `plant`, `rock`, `rock wall`, `stick`, `debris`, `log`, `bush`, `reed`, `leaves`, `ground`, `seabed` | **no detections** (also with the output threshold lowered to 0.05) |

So the three regions are built like this:

* **Fish** — SAM 3.1 `fish`, tracked through 5 overlapping 32-keyframe chunks (one global object id all
  the way, 144/144 keyframes found, no gaps). Frames 222–333 additionally run `fin`; instances that come within 25 px of the
  fish silhouette are merged (117 instances: pectoral, dorsal, tail, pelvic), then a 9 px closing bridges the seam. The fish mask excludes the blade crossing the fish, so
  that blade stays part of the environment.
* **Seabed** — SAM 3.1 `sandy bottom`, 73 keyframes, hole-filled (debris and blades lying on the sand count
  as seabed).
* **Midground (immediate grass / debris)** — *derived*, not segmented: a soft contact band above the sand
  edge (7.5 % of the frame height, falling off with distance) plus the blades/stones that lie on the sand.
* **Background** — *derived*: `(1 − fish) × (1 − max(seabed, midground))`: upper rock, dense weed, open water.

If you want the weed/rock split to come from SAM itself, SAM 3.1 needs visual prompts (clicks/boxes) for
those classes; the text route does not detect them on this footage.

**Temporal handling** (`fish_refine.py`)

* SAM runs on every 3rd frame (fish) / 6th frame (seabed), 15 s per segmented frame on 4 CPU cores; the rest is
  DIS optical-flow interpolation of the soft masks (flow-warped from both neighbouring keyframes, blended).
* Fish matte: keyframe area-outlier repair (none were needed), flow-aligned 1-2-1 temporal filter
  (removes the period-3 jitter that keyframes + interpolation create), then a guided filter against the
  frame's own luma (edges snap to fin rays and blade edges; thin fins keep partial coverage) and a 0.8 px feather.
* Seabed: quarter-res flow, binomial temporal smoothing, closing + hole filling, 7 px feather.
* All grade constants are clip-level; the few per-frame quantities (fish exposure, env exposure, light
  pocket centre) are Gaussian-smoothed over 6–10 frames.

## 2. The grade (numbers)

Everything below is also in `output/fish_grade_numbers.txt` (measured, not hand-typed) and the point-wise
parts are baked into `output/fish_luts/fish_{subject,seabed,background}.cube` (33³, R fastest).

**Calibration source.** `watermarked_img_10881920203034563569.jpg` was not among the uploaded files; the two
orange-fish stills `IMG_1650` / `IMG_1649` were used as the reference. Tone curves and tints are measured on
the *central* columns (20–80 %) of the reference's top 30 % (background) and 58–88 % (seabed), fish removed;
the edge falloff is handled by the vignette instead.

### Global (before the layers)
* White balance on the sand: gain R 1.015, G 0.965, B 1.308 (80 % of full neutralisation), luma preserved.
* Per-frame environment exposure normalisation 50 %, clamp 0.80–1.30 (sunlit vs shaded sand).

### Background (rock / weed / open water) — matte `background`
Neutral-axis curve after the mid-tone lift (input → output, 8-bit codes):

| in | 0 | 31 | 63 | 95 | 127 | 159 | 191 | 223 | 255 |
|---|---|---|---|---|---|---|---|---|---|
| out | 0 | 18 | 49 | 80 | 94 | 96 | 102 | 112 | 117 |

(ASC-CDL approximation: slope 0.27, offset −0.03, power 0.50 — rms 6 codes; use the points as a curve.)
Tonal-zone colour offsets (wheel colour as RGB shift at the zone centre): shadows R−8 G+2 B+2,
mids R−17 G+7 B−10, lights R−28 G+14 B−45. Chroma ×0.80. Clarity +0.45 (σ 12 px), particle highlights +0.85.
Result: near-black teal-slate shadows (floor 0.020), olive-charcoal mids.

### Seabed + midground — matte `max(seabed, midground)`

| in | 0 | 31 | 63 | 95 | 127 | 159 | 191 | 223 | 255 |
|---|---|---|---|---|---|---|---|---|---|
| out | 0 | 16 | 43 | 71 | 86 | 98 | 116 | 133 | 145 |

(ASC-CDL approximation: slope 0.48, offset −0.04, power 0.68 — rms 2.6 codes.) Zone offsets: shadows R−14 G+4 B−2,
mids R−12 G+6 B−14, lights R−5 G+6 B−33. Chroma ×0.85, clarity +0.20. The lift (bg ×1.42, seabed ×1.22 on OKLab L)
acts on the mid-tones only and fades out above L 0.52, so sunlit ridges keep the reference's white point.

### Fish — matte `fish` (OKLCH)
* Luma: quantile curve from the clip's fish pixels to the reference's (raw L 0.14 / 0.21 / 0.29 / 0.37 / 0.55 →
  0.19 / 0.29 / 0.39 / 0.50 / 0.67), after a smoothed per-frame exposure stabiliser (median-normalised, clamp 0.75–1.6).
  The darkest 2 % of the fish (the bars) land at L 0.17–0.22 (reference 0.18–0.19, ≈ RGB 29,10,2, espresso); scale texture is kept (micro-contrast ×1.10).
* Hue and chroma follow the law measured on the reference's fish pixels, by output lightness:

| L | 0.10 | 0.14 | 0.21 | 0.28 | 0.36 | 0.44 | 0.52 | 0.60 | 0.70 | 0.85 |
|---|---|---|---|---|---|---|---|---|---|---|
| hue° | 38 | 38 | 44 | 48 | 47 | 47 | 54 | 63 | 72 | 76 |
| chroma | .040 | .050 | .051 | .064 | .082 | .107 | .121 | .123 | .117 | .100 |

  (source fish hue p5/50/95 = 89 / 103 / 121°, yellow-ochre → crimson 38° … amber 76°.)
* Warmth: the source chroma (window 0.022–0.067, smoothed σ 2 px) sets a weight w. Tail, fin rays and flank blush
  (high raw chroma) get hue −12° × w and lightness ×(1 − 0.16 w): deep burnt orange / crimson, never glowing pink.
* Belly: low chroma and lighter than the bars → blended (max 0.72, σ 3 px) toward a cooled silver
  (hue 245°, chroma 0.020, L cap 0.60), so form and separation stay without washing out the orange wrap.

### Atmosphere
Elliptical vignette (strength 0.58, power 2.1), top-down falloff 0.30 over the upper 42 %, a light pocket that
follows the smoothed fish centroid (floor 0.70, σ 0.60 W × 0.34 H), luma-dependent grain (σ 0.011, new per
frame, 0.7 px soft). Corner/centre luma 0.22–0.38 (reference 0.24–0.31).

### Node-tree translation (Resolve / any node grader)
1. **Global**: gain R 1.015 G 0.965 B 1.308.
2. **Background node** (matte = `background` PNG sequence, or qualifier + window): curve above, shadow/mid/light
   colour offsets above, saturation 80 %, mid-detail +, black floor ≈ 5/255.
3. **Seabed node** (matte = `seabed` ∪ `midground`): curve above, offsets above, saturation 85 %.
4. **Fish node** (matte = `fish`): luma curve as above; *Lum vs Hue* and *Lum vs Sat* with the table above;
   a second qualifier on *low saturation + luma above the bars* → desaturate to ~0.02 chroma, hue 245°.
5. **Outside node**: inverted soft-edged power window (vignette) −0.45 at the corners, top gradient, a moving
   circular window on the fish for the light pocket, film grain ≈ 1 %.
`fish_luts/*.cube` give steps 2–4 exactly except the smoothing-based parts (belly blend and body-pattern
terms, clarity, particles, vignette, pocket, grain).

## 3. QC results

Reference vs graded (stills, same region definitions):

| | reference | graded |
|---|---|---|
| mean luma | 0.18 | 0.12–0.15 |
| corner / centre luma | 0.24 / 0.31 | 0.22–0.38 |
| top third RGB | (13,19,12) / (40,34,16) | (8–34, 10–27, 8–16) |
| fish luma median (OKLab) | 0.39 | 0.38–0.40 |
| fish hue median / chroma median | 45–59° / 0.084–0.087 | 40–47° / 0.078–0.093 |
| fish hue p95 | 90–95° | 49–53° (the reference has more yellow-amber in its lightest scales) |

The environment is darker than the reference on average (the reference frames are mostly fish and lit sand);
the graded frames are matched on the sand's lower third and on the vignette depth, not on mean luma.

Temporal stability on the **encoded** file (high-pass rms of the per-frame region means against a 5-frame mean;
the same series on the untouched source in brackets): background luma 0.0005 (0.0013), seabed luma 0.0003 (0.0010),
fish luma 0.0015 (0.0010), background saturation 0.0050 (0.0054), seabed saturation 0.0018 (0.0008), fish saturation
0.0030 (0.0050). Fish luma and seabed saturation flicker a little more than the source (both well under 0.002 in
absolute terms); the others less. Largest frame-to-frame step of any region mean: 0.026. Fish matte area:
25.9k–166k px, 144/144 SAM keyframes found. Output verified with ffprobe: 429 frames, 14.3143 s, identical to the source.

## 4. Known limits

* Weed/rock and seagrass could not be segmented by SAM 3.1 text prompts (table above); those two regions are
  derived from the SAM seabed + fish masks. The contact band is geometric, not semantic.
* The fish and seabed masks are keyframed (every 3rd / 6th frame) and flow-interpolated; fast fin flicks between
  keyframes can lag by a frame. The fin pass only covers frames 222–333, where the fish mask visibly dropped the
  pectoral fin; at other times the main fish mask already includes it.
* The dorsal fin enters the frame around frame 275 (fish matte area +25 %, a real change, not a mask error).
  Fins on frames outside 222–333 rely on the main fish mask; a scan of the keyframe areas shows no single-keyframe
  dropouts after the fin merge, but fins that SAM drops for a whole span elsewhere would need another `fin` pass.
* The belly patch stays the brightest part of the fish by design; on some frames (e.g. 0, 420) it still reads as a
  pale spot.
* The source is 8-bit H.264; the grade amplifies its compression noise in the dark layers, so the output is encoded
  at CRF 15 (~26 Mbit/s, 47 MB). A lighter copy needs a higher CRF.
* Output is 1080×1920 (portrait 1080p), same size, frame rate (30000/1001) and duration (14.3143 s, 429 frames)
  as the source; no scaling was applied.

## 5. Golden-amber variant (`look_gold.json`, `grade_params_gold.json`)

Same clip, second look: warm golden-amber hero fish on a dense, dark environment, 1920×1080 output.
The SAM 3.1 mattes are reused (the second upload decodes to byte-identical video and audio, only the container metadata differs).

```
python run_grade.py --input inputs/fish.mov --refs inputs/reference/gold_a.jpg inputs/reference/gold_b.jpg \
    --look look_gold.json --gold --canvas 1920x1080 --output output/fish_gold_1080p.mp4
```

* `calibrate --gold` measures the reference fish: hue 90–98° (OKLCH), chroma 0.04 → 0.10 rising with lightness, L median 0.39–0.45.
  The raw fish is already hue ≈ 103°, so this look is richer chroma + brighter scales + a slight shift toward amber, not a recolour.
* Fish toe 0.16 keeps the black stripes dark (the reference mask leaves them out, so a plain quantile match would lift them);
  micro-contrast ×1.25 and a σ 5 px clarity term; belly cooled to a silver (hue 235°).
* Seabed gets a soft ceiling at OKLab L 0.50 so sunlit sand cannot out-shine the fish; vignette strength 0.70.
* Halation: pixels above luma 0.42 bleed a warm two-radius glow (σ 9 and 26 px, gain 0.22); grain 1.1 %.
* `--canvas 1920x1080` fits the portrait frame inside 1920×1080 (`scale … force_original_aspect_ratio=decrease:force_divisible_by=2`,
  `pad` centred, `setsar=1`): a 608×1080 picture with black pillars. Encode: libx264 High, yuv420p, CRF 17, BT.709, source AAC copied.
* Result vs reference (stills): fish L median 0.42–0.44 (ref 0.39–0.45), chroma median 0.053–0.082 (ref 0.056–0.076), hue median 89–94° (ref 94–96°).
  Encoded-file flicker (HF rms): bg luma 0.0008, seabed luma 0.0004, fish luma 0.0021, fish saturation 0.0043 (source 0.0013 / 0.0010 / 0.0010 / 0.0050).

## 6. Matte refinement pass (video_1: spurs, edge margin, intermittent fins)

Findings on the v1 mattes (contour sheets at 100 %, `fish_diag.py`): a 3-8 px loose edge margin on the dorsal/tail boundary
against dark water, thin spurs under the belly where a stem touches (f90-f150, f255), the pectoral fin dropped around
frames 231-300, and fins that were merged on some keyframes and not on others (pop). The stems that cross the fish were
already excluded by SAM. (The yellow "fan" left of the fish at 5.0-6.5 s is a dry grass tuft on the sand, not a fin.)

* `fish_resegment.py`: SAM 3.1 re-segmentation with dense point prompts (9 positives spread over body, stripes and fins,
  10-24 negatives on stems / spurs / thin gaps and a ring around the fish; SAM 3.1 does not accept points together with a
  box). 66 keyframes (0-0.6 s, 5.0-7.5 s, 11.5-14 s, plus tests). Point-prompted SAM under-segments the belly by ~15 %
  on some frames (f0, f9), so it is used only as *evidence*: parts of the keyframe mask that SAM rejects by more than
  10 px are removed.
* `fin` text pass on frames 135-260 (`fins2`) added to the 222-333 pass (`fins`); parts within 25 px of the fish are merged.
* `fish_refine.py --clean --tmedian --wide 140 235 --edge-shift 0.10 --choke 0.10`: spur / hole / fragment clean-up,
  flow-aligned majority vote over three consecutive keyframes (kills single-keyframe fin dropouts and stem bits),
  5-tap flow-aligned temporal window on frames 140-235 (3-tap elsewhere), edge pulled in ~1 px before the guided filter.
* `fish_verify.py`: verification strip and stability table.
