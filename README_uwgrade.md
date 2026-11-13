# uwgrade — reference-driven cinematic underwater grade with SAM 3.1 masks

Grades an underwater clip so its colour, tone and atmosphere match two reference
photographs, using SAM 3.1 masks to give the water, the rocks, the diver and the surface
their own controlled adjustments. The footage itself is never regenerated: every pixel is
the original pixel, re-coloured.

```
python -m uwgrade.sam_keyframes --input inputs/player_export.mov --frames 0:450:30 --prompts "rock|stone wall|water surface|shorts"
python -m uwgrade.sam_keyframes --input inputs/player_export.mov --frames 75,225,375 --prompts person --thresh 0.3 --max-objects 4 --tag person_check
python -m uwgrade.sam_diver     --input inputs/player_export.mov --stride 2
python -m uwgrade.build_masks   --video inputs/player_export.mov --stage all
python -m uwgrade.calibrate     --video inputs/player_export.mov --out uwgrade/params_player_export.json
python -m uwgrade.render        --video inputs/player_export.mov --params uwgrade/params_player_export.json --out-dir output/underwater_grade --h264
python -m uwgrade.qc            sheet|masks|metrics ...
```

Install: see `README_three_shades.md` (Python 3.12 venv, CPU or CUDA torch, `pip install -e sam3/`,
`pip install einops scikit-learn`). Weights: `checkpoints/sam3.1/sam3.1_multiplex.pt` (existing).

## Source footage

HEVC Main 8-bit 4:2:0, 1232x1078, 29.97 fps, 453 frames (15.113 s), BT.709 / limited range
(tagged), AAC stereo. It is plain Rec.709 SDR, **not** Log, so no input transform is applied;
the decode converts limited -> full range with the BT.709 matrix, in 16 bit.

## What was measured on the references

Two photographs, region-wise in CIELAB (D65). The *relationships* are the target, not pixels:

| | reference | source clip |
|---|---|---|
| water, mid | L 15-22, a* -4..-5, b* -15..-16 (hue ~257 deg) | L ~42, a* -20, b* -1 (hue ~184 deg) |
| water chroma vs L | 8 (L 4) -> 14 (L 11) -> 21 (L 24) -> 24 (L 33) -> 20 (L 44) | ~20, flat |
| water hue vs L | ~260 deg (L < 30) drifting to ~232 deg (cyan) at L 45-60 | ~184 deg |
| rocks | L p50 25, hue 75 deg (shadows) -> 50 deg (lit), C 3 -> 14 | green-grey |
| skin | L ~60, a* 5-10, b* 9-12 | |
| shorts | L 60-80, a* -3..-6, b* -12 (cool white in shade) | cyan-tinted |
| tone | water p5/p50/p95 L = 3 / 19 / 38; black point L ~1 | black point L ~25 (veil) |

## SAM 3.1 on this footage — what it can and cannot segment

Verified on this machine (CPU port, existing weights, 3.5 GB checkpoint, ~13 s per frame).

* `person`, `shorts`: p >= 0.9, boundary-precise. Used directly.
* `rock`: individual boulders only (p ~0.3); `stone wall`: sometimes the whole rock mass (p ~0.3);
  `water surface`: the upper surface / water column (p ~0.1) — all need a lowered detection
  threshold (0.05) and cover the class only partially.
* `water`: returns the whole frame minus the diver (everything is underwater). `seabed`,
  `sea floor`: nothing.

So the environment masks are **SAM-seeded, classifier-completed**, not raw SAM output: the SAM
masks above are seeds, a gradient-boosted classifier on colour/texture/haze/height features trained
on the pooled seeds of 16 keyframes completes them to a soft per-pixel probability for every frame,
then a short temporal filter. Skin and dark gear are likewise derived inside the SAM diver mask
(SAM has no usable "skin" concept). This is stated explicitly because it is not pure SAM.

SAM runs on every 2nd frame for the diver (CPU budget) and the odd frames are optical-flow
interpolated + edge-snapped; three odd frames were segmented directly by SAM to measure the error.
