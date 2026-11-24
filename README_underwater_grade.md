# Region-aware cinematic underwater grade (SAM 3.1)

Grades a snorkel clip toward the reference look: warm golden-amber highlights
and bubbles at the top of the water column, deep teal/aqua mid-water and
shadows, a protected subject silhouette, soft highlight roll-off, subtle
halation and very light grain. Output is H.264 / yuv420p / 1080x1920 with the
original audio copied untouched.

## Run order

```bash
# 0. environment (Python 3.12; CPU is fine, ~12 s/frame for SAM 3.1)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e sam3/ -r requirements_three_shades.txt einops colour-science

# 1. SAM 3.1 masks on every 4th frame (540 px wide), text prompts
python uw_masks.py --video clip.mov --out masks_uw/clip --concept subject=person --stride 4 --chunk 20
python uw_masks.py --video clip.mov --out masks_uw/water_part --concept +water=water --stride 4 --limit 57   # frames 0-224
# 2. underwater silhouette: SAM re-prompt (text, then box) + flow tracking to fill the gaps
python uw_masks.py --video clip.mov --out masks_uw/under_box --concept subject=person --range 228:320 \
       --prompt-frame 260 --box 0.470:0.122:0.385:0.445
python uw_track_gap.py --video clip.mov --first masks_uw/clip/subject.npz \
       --box masks_uw/under_box/subject.npz --out masks_uw/under_final
# 3. optical-flow interpolation of the masks to every frame
python uw_interp.py --video clip.mov --masks masks_uw/clip \
       --override subject=masks_uw/under_final/subject.npz --extra water=masks_uw/water_part/water.npz
# 4. calibrate the look on the reference pairs (scene:frame:graded_reference:original_still)
python uw_fit.py --masks masks_uw/clip --out look.json \
       --pair stand:0:ref_stand.jpg:orig_stand.jpg --pair entry:141:ref_entry.jpg:orig_entry.jpg \
       --pair under:255:ref_under.jpg:orig_under.jpg
# 5. render (H.264 libx264 crf 17 slow, 1080x1920 yuv420p, audio copied) and QC
python uw_render.py --video clip.mov --masks masks_uw/clip --look look.json --out output/clip_graded.mp4
python uw_qc.py --src clip.mov --out output/clip_graded.mp4 --masks masks_uw/clip --ref ... --report output/qc_report.json --sheet output/qc_contact_sheet.jpg
```

`python uw_render.py ... --stills 0,141,255 --still-dir out/` renders single
frames (orig + graded PNG) for quick look checks.

## How the grade works (`uw_look.py`)

1. **Regions** (soft weights summing to 1, SAM masks snapped to edges with a
   guided filter, then feathered):
   - surface scenes: *subject* = SAM "person"; *body* = SAM "water" (plus
     lower-frame seabed rocks); *top* = everything else in the upper frame
     (sky, mountains, town).
   - underwater: *top* = sunlit zone (blurred luminance x vertical prior),
     *body* = the rest; the subject mask drives a protection step (amber/teal
     saturation pulled back 30% inside the silhouette, luma kept).
2. **Regional colour**: per scene and region a cubic RGB polynomial (shared +
   per-region delta, degree-aware ridge) fitted to the pixel-aligned reference
   pairs, with a safe affine fallback blended in where the reference does not
   cover a colour. Scenes (`stand` -> `entry` -> `under`) are cross-faded by
   frame index (`scene_weights`), the plunge through the foam has no hard cut.
3. Soft tanh highlight roll-off, warm halation (screen blend), light grain.

## Measured results (this clip, 319 frames)

| Check | Result |
|---|---|
| Output | h264 High, 1080x1920, yuv420p, 30000/1001 fps, 319 frames, AAC copied |
| dE2000 vs the three reference stills | 2.68 (stand) / 2.67 (entry) / 3.23 (under); un-graded 7.8 / 7.2 / 10.6 |
| Added luma flicker (RMS 2nd difference) | 1.04x the source |
| Added chroma flicker | absolute 0.002 (source 0.0005); spikes only on the fast foam-to-underwater plunge (frames 229-231) which the source itself has |
| Edges | no halo or seam at 100% crops (subject/water boundary, silhouette) |

## Limits (stated plainly)

- **The look is calibrated on three stills.** Frames far from them rely on
  the model's smoothness; the affine fallback keeps unseen colours bounded.
- **SAM 3.1 ran on the CPU on every 4th frame** and masks were interpolated
  with optical flow. Fast motion during the entry is the weakest spot.
- **Underwater silhouette:** SAM 3.1 (text and box prompts) only detected the
  bubble silhouette on frames 256-288. Frames 236-252 and 292-316 are
  *optical-flow tracked* from those detections (`masks_uw/under_final/report.json`),
  frames 228-232 have no subject (foam). The tracked mask over-covers the
  silhouette a little at the end (~15% of the frame); it only drives the mild
  protection step.
- **Foam frames 184-232:** the swimmer is hidden, so no subject mask exists.
- SAM "sky" was not used; the top zone is derived from the water mask.
- Output is ~64 MB (about 48 Mbps) at CRF 17 because grain and bubbles are
  expensive to encode; use `--crf 18` or `--preset medium` for a smaller file.
