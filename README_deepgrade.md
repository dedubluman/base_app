# deepgrade: SAM 3.1 region-aware deep-water cinematic grade

Colour grade for real underwater footage. The footage itself is never
regenerated or altered: only colour, tonal response and controlled local
contrast change. SAM 3.1 (repo weights, `checkpoints/sam3.1/sam3.1_multiplex.pt`)
decides *which* grade each pixel receives.

```
python -m deepgrade.run_sam      --video inputs/clip.mov --stages preview,diver,stuff,extra
python -m deepgrade.build_masks  --video inputs/clip.mov
python -m deepgrade.preview      --frames 0,60,150,204 --out output/preview   # tune on keyframes
python -m deepgrade.render       --video inputs/clip.mov --out output/clip_graded.mp4
python -m deepgrade.qc masks | verify | sheet
python -m deepgrade.palette_report
```

## Colour system (measured in OKLab)

| Role | Anchor | OKLab (L, h, C) | Where it lands |
|---|---|---|---|
| Deep water / shadows | Stratos `#070F54` | .228, 268°, .122 | water palette: deepest tones hue ~266°, mid-water ~252° (the reference photographs measure L .24-.30, h ~250°, C ~.067) |
| Seabed / rock | Mocha `#4F5733` | .440, 119°, .056 | sand and rock palette (hue ~110°, restrained chroma .040); far / hazy sand keeps the navy, near sand and rocks go earthy |
| Highlights / particles | Silver Sand `#BFC1C2` | .810, neutral | particles and bright detail neutralised to silver; global shoulder caps highlights at L .845 |
| Diver | natural skin | ref. lit skin L .57 | local white balance from the diver's own skin, hue pulled to ~72°, chroma .02-.07 |

The reference photographs keep the seabed teal (hue ~214°); the brief asks for Mocha olive-brown.
Hue follows the brief (Mocha), luminance / chroma level / local-contrast follow the references.

## How SAM 3.1 is used

Everything runs through the repo's SAM 3.1 multiplex video predictor (`shades/masking.py`,
`shades/sam31_cpu.py`: a mechanical CPU port of the same model when CUDA is missing, ~12 s per frame on 4 cores).

| Region | SAM 3.1 prompt | How |
|---|---|---|
| Diver | `person` | video tracking on every 2nd frame in overlapping 32-frame chunks (SAM keeps ~100 MB of features per frame), diver matched across chunks by IoU, overlaps cross-faded; in-between frames by optical-flow warping, 3-tap flow-compensated smoothing, guided-filter edge refinement bounded to a band around SAM's edge |
| Snorkel | `snorkel` | keyframes every 3rd frame, merged into the diver (SAM's person masklet leaves it out once it clears the head) |
| Rocks / coral | `rocks` | keyframes every 3rd frame, lowered detector thresholds |
| Open water | `water` | keyframes every 3rd frame, lowered detector thresholds |
| Sky | `sky` | keyframes every 3rd frame (above-water finale) |
| Hills | `mountain` | keyframes, treated as rock (Mocha) |
| Sand | not detectable by SAM 3.1 text prompts (stuff class; every sand/seabed phrasing scored below the detector floor, point prompts return nothing in single-frame sessions) | derived: the underwater scene SAM did not claim as diver / rock / water / sky, gated in time by SAM's own water coverage |

Region keyframe masks are linear in time, smoothed across keyframes and feathered; they fade in/out
over 6 frames at the edges of their range, so nothing pops.

## Grade (per frame, OKLab)

Regions blend in OKLab, so there is never a seam. Each region maps the source *luminance* through its own
curve and takes its *colour* from a palette ramp keyed on the graded luminance, keeping a controlled share of
the source's high-frequency chroma:

* water: anchored per frame (smoothed), slope 1.45, soft floor L .17, Stratos-to-reference hue ramp
* sand/rock: vertical distance gradient, polar (hue/chroma) blend between water haze and Mocha, cored band-pass micro-contrast (x1.9 sand, x2.5 rock)
* sky: compressed, cool silver with a share of the original horizon glow
* diver: per-frame local white balance from lit skin, filmic shoulder, region-normalised local contrast, gear (mask, snorkel, wristband, shorts) protected
* highlights: top-hat particle detection, chroma moves to Silver Sand, brightness kept
* global: highlight shoulder (no clipping), light vignette, gamut-safe chroma reduction, 8-bit TPDF dither before the encode

All numbers live in `deepgrade/params.py` and are dumped next to every render (`*.params.json`).

## Install

Python 3.12 venv, see `README_three_shades.md`; additionally `pip install einops "setuptools<81"`
(the sam3 package imports both).

## Measured result (clip `player_export`, 1232x1078, 230 frames, 29.97 fps)

| Check | Result |
|---|---|
| Output vs source | resolution, fps (30000/1001), 230 frames, duration, AAC audio stream, BT.709 tv-range tags, yuv420p: all identical (`qc verify`) |
| Water vs reference photo | OKLab dE 1.6 (L .250 / C .071 / h 253 vs .265 / .067 / 249) |
| Darkest 2 % vs Stratos | hue 262 vs 268, chroma .082 vs .122 (a,b dE 4.1) |
| Near seabed vs Mocha | hue 134 vs 119, a,b dE 1.8 (rocks: a,b dE 2.6) |
| Highlights vs Silver Sand | RGB (191,192,195) vs (191,193,194) on frames with real highlights; no clipping |
| Lit skin vs reference skin | a,b dE 1.1 |
| Diver mask stability | motion-compensated consecutive IoU mean .977 (p5 .916); only the hard surface-break cut (f193-198) and 3 splash frames fall below .90; tracked vs independent single-frame SAM detections IoU .979-.993 |
| Temporal luminance | graded mean-L follows the source smoothly (no exposure pops) |

Known limits: SAM 3.1 cannot text-detect sand (see table above), so sand is derived; "water" and "rocks"
are only marginally confident detections, handled by gap-filling between confident keyframes and a monotonic
sand-to-water gate; the far (hazy) half of the seabed stays navy by design, so the *median* sand pixel reads teal
(h ~199) while the near seabed reads Mocha (h ~134).

---

# v2: temporal colour consistency + 15 s optical-flow slow motion

## Diagnosis of the v1 colour fluctuation (`python -m deepgrade.diagnose`)

The source is steady: its background drifts only L .535 -> .521 and hue 197 -> 212 over the whole dive.
v1 graded the seabed ("sand") and the open water with **two different colour regimes** and cross-faded between
them while the camera tilted away from the sand (frames 48-64 = 1.5-2.1 s): background L .318 -> .261,
hue 215 -> 251, chroma .038 -> .069. That regime switch was the greenish-bright opening and the sudden dark-navy
step. A second, smaller defect: the diver's lit-skin hue drifted 122 -> 70 degrees over the first 1.5 s because the
skin white balance was clipped at 4x red gain and its likelihood window did not follow the source's skin hue.

## v2 grade (`grade2.py`, `params2.py`, `render2.py`)

* **One base look** for water, sand and the unclaimed remainder (one tone map, one palette). The seabed identity is
  restrained Mocha earth on lit sand texture and on SAM 3.1 rocks (constant share, hazing out with the gate) on top
  of that base - never a different overall colour.
* **Temporally smoothed anchors:** background brightness, skin white balance, skin hue window, skin hue/chroma
  residual, sky level, "air" fraction: measured per frame, Gaussian-smoothed (6-10 frames), then the grade is a pure
  function of (frame, weights, smoothed anchors). No per-frame auto-grading.
* **Natural above-water path:** sky, land (SAM `sky`, `mountain`) get a gentle tone curve only; the sea keeps half
  of the natural look. No navy spill onto sky or coast.
* **Diver:** skin white balance applied to skin only (adaptive hue window, closed-loop hue/chroma landing), the
  swimwear is segmented with SAM 3.1 (`shorts`) and keeps its own colour; mask, snorkel, wristband untouched.
* **Masks:** SAM 3.1 diver / rocks / water / sky / mountain / snorkel / shorts as before (no regeneration needed:
  tracked vs independent detections IoU .98-.99, motion-compensated frame-to-frame IoU .977).

## 15 s slow motion (`interp.py`, `rife_model.py`, `final_render.py`)

* Engine: **RIFE v4.26** (Practical-RIFE, official `hzwer/RIFE` release), CPU, arbitrary-timestep interpolation.
  It was not in the repo; it is the upstream model, not a duplicate. Weights in `checkpoints/rife/` (gitignored).
* Timing: 230 frames -> 450 frames at 30 fps = exactly 15.000 s; output frame k shows source position
  min(k*230/450, 229); speed 51.1 % (7.674 s / 15 s). Order preserved, nothing reversed, frozen or dropped.
* Audio: original AAC, rubberband pitch-preserving stretch to exactly 15.000 s.
* The surface break (frames ~192-198) is a fast *continuous* camera move, not a cut; RIFE handled it without
  ghosting (checked on every pair), and an automatic artifact detector with conservative fallbacks (coarser flow, then
  dissolve) covers any frame it would flag.

## Measured result (v2 final, `underwater_cinematic_final_15s.mp4`)

| Check | v1 (previous) | v2 (final) |
|---|---|---|
| Underwater background colour range over the dive (dE-ok x100) | 10.3 | 2.7 (3.4 in the delivered 15 s video; source itself 6.5) |
| Worst 12-frame background change | 7.8 | 1.9 |
| Background hue / L over frames 0-119 | 206-254 deg / .251-.318 | 251-255 deg / .235-.247 |
| High-pass flicker of the background | .48 | .21-.24 (source noise floor .19) |
| Lit-skin hue over the dive | 70-122 deg | 67-82 deg (chroma .029-.038, L .56-.64) |
| Shorts colour | drifted to lime (hue 138-145) | own pale-cyan white kept (hue 205-207) |
| Duration / frames / fps | 7.674 s / 230 / 30000:1001 | **15.000000 s / 450 / 30:1**, audio stream 15.000000 s |
| Interpolation accuracy (leave-one-out, 21 frames) | n/a | RIFE 28.1 dB vs plain average 25.2 dB vs frame repeat 22.6 dB |
| Artifact detector | n/a | 0 of 450 frames flagged (max local spike 1.72 vs threshold 1.9) |
| Audio | copied | rubberband stretch, best-match frequency scale 0.98 (pitch kept), loudness restored to the source's -27 dB mean |

Limits: the seabed's Mocha identity lives on the rocks and lit sand texture; the sand itself shares the navy base on
purpose (that sharing is what removes the fluctuation). The waterline is a fast continuous camera move; RIFE
interpolates it cleanly, but every frame there is synthesized from two motion-blurred neighbours, so it is softer than
the rest. The `interp_loo` numbers at the waterline (~21 dB) are low for the same reason (that test skips a whole frame).
