# 3 Shades underwater reel

Automates a palette-intro edit for vertical (9:16) underwater footage:

1. **Reference color grade from the first frame.** The water and sand are matched to the colors of a reference video (`--reference`), measured on its graded section. No graphics are added by default.
2. **Optional palette intro** (`--intro`): a title and three palette cards, then the grade wipes in.
3. **SAM 3.1 subject isolation.** The diver is segmented in every frame. The mask drives depth separation: the diver gets brighter, warmer skin and more detail, while the water gets slightly darker, less saturated and defocused.
4. **Optical-flow speed ramp.** The ascent toward the surface is detected from the mask and eased into slow motion. The in-between frames are synthesized with bidirectional optical flow, using an occlusion-aware blend.

The three swatches describe the **target grade**, not the raw footage. The same colors drive the split-toning in `grade.py`: shadows are lifted into *Abyssal Teal*, sand and skin are pushed toward *Golden Sand*, and highlights lean toward *Crystal Highlight*. The cards always match the final look.

```
three_shades/
  config.py         every tunable (timings, layout, palette, grade, flow, ramp, export)
  segmentation.py   SAM 3.1 video predictor + mask cache (+ CPU preview fallback)
  grade.py          palette-driven grade, mask-aware depth separation, .cube export
  palette_cards.py  title + card rendering, easing, grade wipe
  flow.py           DIS / DeepFlow / SF / Farneback flow, interpolation, ascent ramp
  video_io.py       FFmpeg decode (16-bit) / encode (BT.709, H.264/HEVC)
  pipeline.py       CLI that runs the stages in order
```

## Install

Python 3.10–3.12 is recommended (SAM 3 targets 3.12). FFmpeg 6+ must be on `PATH`.

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch with CUDA (pick the line for your CUDA version from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# SAM 3.1 code + this pipeline's dependencies
git clone https://github.com/facebookresearch/sam3.git
pip install -e sam3
pip install -r three_shades/requirements.txt

# SAM 3.1 weights (gated: accept the license on huggingface.co/facebook/sam3.1 first)
export HF_TOKEN=hf_...            # your token
./scripts/download_sam3.sh        # -> checkpoints/sam3.1/sam3.1_multiplex.pt
```

## Run

```bash
# Everything on a CUDA machine: SAM 3.1 masks, then the render
python -m three_shades all video_2.mov -o reel.mp4 --reference video_1.mov

# Split across machines: masks on a GPU box/Colab, render anywhere
python -m three_shades segment video_2.mov -o reel.mp4        # writes reel.sam31.masks.npz
python -m three_shades render  video_2.mov -o reel.mp4        # reuses the cache

# Iterate on timings/grade quickly: half resolution
python -m three_shades render video_2.mov -o reel.mp4 --preview

# No GPU at all: CPU stand-in segmenter for previews (edges are rougher; use SAM 3.1 for finals)
python -m three_shades render video_2.mov -o preview.mp4 --segmenter maskrcnn-preview --preview

# Add music (faded out at the end)
python -m three_shades render video_2.mov -o reel.mp4 --music track.m4a
```

Each render writes:
- `reel.mp4`: 1080×1920, 30 fps, H.264, tagged BT.709.
- `reel.cube`: the global grade as a 33³ LUT for Resolve, Premiere or Final Cut.
- `reel.json`: a report with the detected ascent window, frame counts and palette.

**Choosing the diver.** SAM 3.1 is prompted with the concept `person`, so it tracks every swimmer in the clip. Only one person gets the mask: by default (`seg.subject_select = "ascending"`) it's the person who rises toward the surface the most. Background swimmers and people at the frame edge are never masked, so they get the water's grade. If SAM 3.1 loses the diver near the surface and re-detects them under a new ID, the two tracks are joined, so the mask lasts through the exit. Set `subject_select` to `"largest"` to pick the biggest person instead. To force a specific person, click on them in frame 0 (pixel coordinates in the 1080×1920 output frame):

```bash
python -m three_shades segment video_2.mov -o reel.mp4 --subject-point 540,900
```

## Tuning

The palette-card intro is off by default. `--intro` (or `intro.enabled: true`) turns it back on, and the `intro` timings below apply only then.

Copy `config.example.json`, change what you need, and pass it with `--config my.json`. Any field from `config.py` can be overridden. `python -m three_shades dump-config` prints every field with its default. All times are in seconds.

### Animation timings (`intro`, output timeline)

| Field | Default | Effect |
|---|---|---|
| `title_in` / `title_out` / `title_fade` | 0.2 / 1.1 / 0.35 | Title fade window. Set `layout.title` to `""` to remove the title. |
| `cards_in_start` | 1.5 | When card 1 starts to drop. Keep it after `title_out + title_fade` so the title is gone first. |
| `card_stagger` | 0.2 | Delay between cards 1, 2 and 3. |
| `card_in_dur` | 0.8 | Length of one drop. The ease is "out-back", so the card overshoots slightly and settles. |
| `label_delay` / `label_dur` | 0.3 / 0.45 | The name and hex fade in shortly after each card starts. |
| `cards_out_start`, `card_out_stagger`, `card_out_dur` | 4.4, 0.1, 0.6 | Exit: the cards slide up and fade. |
| `grade_wipe_start` / `grade_wipe_dur` / `grade_wipe_feather` | 4.5 / 1.2 / 0.35 | Flat-to-graded wipe. A bigger feather gives a softer, longer edge. |
| `outro_fade` | 0 | Fade to black at the end, in seconds. 0 turns it off. |

Card geometry is set in `layout`: `cards_top_frac` (keep it at 0.08 or more to stay clear of the Reels/TikTok top UI), `card_aspect`, `corner_radius_frac`, `drop_distance_frac`, `start_scale` and font sizes. Fonts are found automatically (Inter, then Segoe UI, then DejaVu). Set `font_bold` / `font_medium` to use your own.

### Color (`palette`, `grade`)

- **Changing a hex in `palette`** changes both the card and the grade.
- `wb_strength`: how much of the green cast is removed (0 = none, 1 = full gray-world).
- `levels_strength`: how far the black/white points are stretched to the clip's percentiles.
- `contrast`, `pivot`, `highlight_knee`: the filmic S-curve and the highlight roll-off.
- `shadow_tint`, `shadow_lift`: the deep-teal shadows. Raise `shadow_lift` for a more lifted, matte look.
- `sand_boost`: how strongly sand and skin are pulled toward the Golden Sand hue.
- `sand_chroma`: how saturated that target is. Raise it for more golden sand; lower it if skin looks too warm. Because it's a bounded pull rather than an additive tint, skin can't go orange.
- `midtone_tint`: a gentle Golden Sand tint across all midtones.
- Reference match (needs `--reference`): `ref_match_luma` (brightness and contrast), `ref_match` (blue-yellow) and `ref_match_a` (green-magenta; keep it low or sand turns mauve) set how far the background moves toward the reference's Lab color statistics. `reference_start`/`reference_end` select the reference's graded section and `reference_crop` the region measured. The default crop skips a phone screen recording's status bar, side icons and caption. The diver inside the mask is not changed by the match.
- `skin_tone`, `skin_restore`, `skin_saturation`: skin color of the diver, applied only inside the SAM 3.1 mask. `skin_tone` defaults to `#A88065`, a natural tan (hue 24°, saturation 0.40). `skin_restore` sets how strongly the diver's midtones are pulled to it.
- `water_teal`: pushes the water midtones toward teal.
- `highlight_tint`, `highlight_mix`, `bloom`: the crystal highlights and the glow on sun rays.
- Depth separation (SAM 3.1 mask):
  - `subject_exposure`, `subject_warmth`, `subject_detail`: the diver.
  - `bg_exposure`, `bg_desaturate`, `bg_blur_frac`: the water behind them.
  - `mask_feather_frac`: softness of the mask edge.

The per-pixel chain in `grade_global()` runs in this order: white balance (a diagonal color matrix, with gains measured from the clip), then levels, the filmic curve, palette split-toning, and the warm/cool separation. To add a full 3×3 color matrix (for example a channel mixer), apply `x = x @ M.T` at the top of `grade_global()`. Clip-wide statistics are measured once, so the look doesn't flicker.

### Optical flow + ramp (`flow`, `ramp`)

- `flow.algorithm`:
  - `dis` (default): about 1.7 s per frame pair at 1080×1920 on 4 CPU cores. It gave the most accurate in-between frames on this footage.
  - `deepflow`: needs opencv-contrib, about 6 s per pair.
  - `sf` (`calcOpticalFlowSF`): needs opencv-contrib, about 32 s per pair.
  - `farneback`: in plain OpenCV, about 0.8 s per pair, slightly softer.
- `flow.work_scale`: flow is computed at this scale (0.5 by default). 1.0 gives finer detail and is about 4× slower.
- `dis_preset`, `dis_patch_size`, `dis_var_refine_iters`: the usual DIS accuracy/speed controls.
- `flow.occlusion_threshold`: where the two warped frames disagree by more than this, the nearer frame is used. Lower it if you see ghosting around limbs or bubbles.
- `ramp.slow_speed`: playback speed during the ascent. 0.35 is about 2.9× slow motion; 0.25 is 4×.
- `ramp.ramp_in` / `ramp.ramp_out` / `ramp.pre_roll`: how gently the speed eases (smoothstep, so there are no jerks).
- `ramp.min_up_velocity`, `ramp.min_duration`: sensitivity of the automatic ascent detector.
- `ramp.start` / `ramp.end`: set both (source seconds) to bypass auto-detection.

### Export (`export`)

- `width`/`height` (1080×1920) and `fps`. Any input is scaled to cover the frame and center-cropped to 9:16.
- `codec`: `libx264` or `libx265`. `crf` and `preset` control quality and encoding speed.
- `dither`: on by default. It prevents banding in the deep-water gradients.

## Notes

- **The SAM 3.1 multiplex video predictor needs a CUDA GPU.** It hard-codes `.cuda()` and FlashAttention. Running the `segment` step on a GPU or Colab and rendering locally from the `.npz` cache works well.
- **Upstream bug workaround.** The current `sam3` code fails in `start_session` because it passes `offload_state_to_cpu` to the SAM 3.1 model, which doesn't accept it. `segmentation._patch_init_state` works around this.
- **Audio.** The original audio is not kept, because the ramp changes timing. Use `--music`.
