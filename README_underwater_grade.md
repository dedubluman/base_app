# Reference-matched deep-ocean grade

Final: `output/UNDERWATER_CINEMATIC_FINAL_1080P.mp4` (git-ignored; regenerate with the commands below).

1. `python diver_masks.py --input inputs/player_export.mov --direction fwd` – SAM 3.1 (CPU port, ~100 min) -> `masks/`
2. `python refine_masks.py --raw masks/player_export__fwd/subject --out masks_refined/player_export` – temporal alpha mattes (kept in git)
3. `python render_grade.py --input inputs/player_export.mov --masks masks_refined/player_export --output output/UNDERWATER_CINEMATIC_FINAL_1080P.mp4`

Revise the look: edit `grade_data/params.json` (env: hi_gain, sat, warm_kill, contrast; diver: gains, exposure, gamma, sat,
neutral_below, white_desat) or swap `grade_data/reference/*` and rerun `fit_reference_grade.py`.
Grade = cubic RGB polynomial fitted to the reference/original pairs + luma curve, baked to a static 33^3 LUT (no flicker);
the diver gets a separate gentle LUT through the SAM mask. Export: 1920x1080 pillarboxed (source 1232x1078, never stretched),
H.264 High yuv420p BT.709 CRF 17 slow, faststart, source AAC copied bit-identically.
