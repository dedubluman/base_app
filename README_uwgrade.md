# uwgrade — reference-driven cinematic underwater grade with SAM 3.1 masks

Grades an underwater clip so its colour, tone and atmosphere match two reference photographs,
using SAM 3.1 masks to give the water, the rocks, the diver and the surface their own controlled
adjustments. Every output pixel is the original pixel re-coloured; nothing is regenerated.

```
# SAM 3.1 (CPU port, existing weights) -- ~2 h on 4 cores for this 15 s clip
python -m uwgrade.sam_keyframes --input inputs/player_export.mov --frames 0:450:30 --prompts "rock|stone wall|water surface|shorts"
python -m uwgrade.sam_diver     --input inputs/player_export.mov --stride 2 --offset 0     # even frames
python -m uwgrade.sam_diver     --input inputs/player_export.mov --stride 2 --offset 1     # odd frames
python -m uwgrade.sam_keyframes --input inputs/player_export.mov --frames 87,88,89,90 --prompts "swim fin|fin" --thresh 0.05 --max-objects 8 --tag fin_fix
python -m uwgrade.patches       --video inputs/player_export.mov --frames 88:90
# masks, calibration, render, QC
python -m uwgrade.build_masks   --video inputs/player_export.mov --stage env      # rocks / water
python -m uwgrade.build_masks   --video inputs/player_export.mov --stage diver
python -m uwgrade.build_masks   --video inputs/player_export.mov --stage parts    # skin / shorts / gear
python -m uwgrade.calibrate     --video inputs/player_export.mov --out uwgrade/params_player_export.json
python -m uwgrade.render        --video inputs/player_export.mov --params uwgrade/params_player_export.json --out-dir output/underwater_grade --h264
python -m uwgrade.qc            sheet|masks|metrics|palette|stills ...
python -m uwgrade.deliver       --master <prores.mov> --out <name>.mp4 [--canvas 1920x1080]   # 1080p H.264 copies from the master
```

Install: Python 3.12 venv, CPU torch 2.10, `pip install -e sam3/ einops scikit-learn -r requirements_three_shades.txt`
(see `README_three_shades.md`). Weights: `checkpoints/sam3.1/sam3.1_multiplex.pt` (the existing install; verified, 0 missing keys).

## Source footage
HEVC Main 8-bit 4:2:0, 1232x1078, 29.97 fps CFR, 453 frames (15.113 s), BT.709 / limited range (tagged), AAC stereo.
Plain Rec.709 SDR, **not Log**, so no input transform; the decode converts limited -> full range with the BT.709 matrix in 16 bit.

## What was measured on the references (CIELAB, D65 — relationships, not pixel colours)
| | reference | source clip |
|---|---|---|
| water | L p5/p50/p95 3 / 19 / 38, a\* -4..-5, b\* -16..-18, hue 253-259 deg | L ~42, a\* -20, b\* -1, hue ~184 deg (green-teal) |
| water chroma vs L | 8 (L 4) -> 14 (L 11) -> 21 (L 24) -> 24 (L 33) -> 20 (L 44); hue drifts to ~232 deg (cyan) in the lights | flat |
| rocks | hue 75 deg (shadows) -> 50 deg (lit), chroma 3 -> 14; shadows are blue-black water | green-grey |
| skin / shorts | L ~60, a\* 5-10, b\* 9-12 / cool white, b\* -5..-12 | cyan veil |
| black point | L ~1 | L ~25 (veil) |

## SAM 3.1 on this footage — what it can and cannot segment
* `person`, `shorts`: p >= 0.9, boundary-precise. **Used directly** (diver on all 453 frames; shorts as seeds).
* `rock` -> individual boulders only (p ~0.3); `stone wall` -> sometimes the whole rock mass (p ~0.3);
  `water surface` -> the upper surface / water column (p ~0.1). Needs a lowered threshold (0.05); partial cover.
* `water` returns the whole frame minus the diver (everything is underwater); `seabed`, `sea floor`: nothing.

**So the environment masks are SAM-seeded and classifier-completed, not raw SAM output.** SAM masks on 16 keyframes are seeds;
a gradient-boosted classifier on colour / texture / haze / height features completes them for every frame (leave-one-keyframe-out
AUC 0.96-1.00 on the keyframes that have both classes), then a 5-tap temporal filter, small-blob removal and a 12 px haze-wide feather.
Skin and dark gear are derived inside the SAM diver mask (SAM has no usable "skin" concept) by a classifier trained on SAM "shorts"
seeds + a diver-relative lightness rule (leave-one-keyframe-out accuracy 0.93-1.00; 0.80 on frame 450, the tiny surface-distorted diver).

## Decisions worth knowing
* **SAM on every frame for the diver (even + odd streams).** Flow-interpolating the odd frames would have scored mean IoU 0.967 /
  0.9 px boundary error (p90 1.4 px) against the real SAM odd frames — good on average — but with a worst case of 11 px (frame 87).
  I first judged it much worse from a harder proxy test; the measured numbers above (`output/interp_vs_sam.json`) are the real ones.
  The extra pass removes the outliers that would flicker at the subject's edge; it was a more expensive precaution than the average case needed.
* **Accessory repair, frames 88-90.** SAM's `person` track lost the black fin and snorkel (still visible) for ~2 frames — the only temporal
  discontinuity in 453 masks (area x0.70). SAM was prompted for `swim fin` / `fin`; candidates merged only if dark, big and touching the diver
  or continuing a region the previous frame had (`uwgrade/patches.py`).
* **Node 1 black point from the environment, white point from all pixels**, with the black-point subtraction 55% relaxed on the diver
  (it is nearer the lens, carries less veil): otherwise hair / fins clipped to L 0.3 and the white shorts clipped at the stretch.
* **Haze-driven atmospheric perspective** (underwater dark channel, G and B only; diver excluded and filled): distant rock fades into the water grade,
  near rock stays warm. Without it every rock was brown; the references' rocks are blue where far and in shadow.
* **Shape prior for the shorts was tried and removed**: it fixed shaded forearms labelled as shorts at depth, but at close range it stripped the
  shorts' own cuffs (peach fabric). Restoring shaded skin to the skin training seeds fixed the forearms instead.

## Result (rendered master, 453 frames; `output/underwater_grade/qc/`)
| check | result |
|---|---|
| ProRes 422 HQ master | 1232x1078, 10-bit 4:2:2, BT.709 tagged, 453 frames, last pts 15.081733 s (identical to source), 711 MB |
| H.264 copy | High, yuv420p, same frames/timestamps, 51 MB |
| audio | AAC stream-copied: decoded **and** compressed-packet MD5 identical to the source |
| original | untouched (SHA-256 identical to the upload) |
| water vs reference | L p5/p50/p95 9.1 / 17.5 / 37.0 vs 3.1 / 18.6 / 38.0; ab (-5.3, -15.3) hue 251 vs (-4.4..-5.2, -15.7..-17.7) hue 253-259 |
| lit rock | hue 53 deg, chroma 8.1 (reference 47-81 deg, 8-16) |
| shorts | L p50 79.5, ab (-2.2, -6.1) (reference lit 79.4, (-2..-6, -5..-12)) |
| frame-to-frame dE (mean / p99 / max) | water 0.25 / 1.36 / 2.43 · rock 0.14 / 0.95 / 1.44 · diver 0.71 / 2.71 / 6.18 (fin leaving the frame) |
| mask flicker (mean abs alpha diff per frame) | diver 0.0039 · rock 0.0048 |

## Known limitations (stated, not hidden)
* **Water tail is lifted**: p5 L 9.1 vs the reference's 3.1. Ambiguous dark pixels (hazy dark rock the classifier called water) are blended toward the
  rock tone curve; without it they rendered as black voids (seen at frame 445). Median and highlights still match the references.
* **Frame 5 (rocks fill the frame) is browner than REFERENCE 1**, which is mostly blue; the near rock dominates the picture. Rocks are deliberately
  muted bronze, less saturated than REFERENCE 2's copper (asked: no orange / glowing rocks).
* **Skin is less saturated than the references** (chroma 8.9 vs ~14; L p50 56 vs 59-64): hazy / shaded skin and the haze veil on the distant diver.
* **A thin peach seam** can appear where skin meets the shorts' waistband (visible on a few frames, e.g. 87-90, 225): the classifier gives that sliver to skin.
* The rock/water boundary is classifier-completed, not pixel-exact SAM; it is feathered by design (the real boundary is haze).
* Contrast expansion of an 8-bit HEVC source amplifies codec noise in smooth water; the grade is dithered at 10-bit but not denoised.
* One clip, one dive: the calibration (`uwgrade/params_player_export.json`) is measured on it and is not meant to transfer unchanged.
