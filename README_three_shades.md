# three_shades — raw → cinematic underwater reveal with SAM 3.1

The clip opens on the untouched source footage. It then eases into a
cinematic underwater grade: deep teal/blue shadows, warm sand and rock, and
clean highlights. SAM 3.1 masks keep the diver gently separated from the
seabed. Nothing is drawn on the picture: no cards, no text.

```
python three_shades.py --input inputs/video_1.mov --output output/video_1_graded.mp4
```

## Install

Use Python 3.12 (SAM 3 pins `numpy<2`) and a system `ffmpeg` with libx264.

```bash
python3.12 -m venv .venv && source .venv/bin/activate

# NVIDIA GPU (recommended for SAM 3.1):
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# ...or CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e sam3/
pip install -r requirements_three_shades.txt

# optional, GPU only: FlashAttention 3 (auto-detected)
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
```

The SAM 3.1 checkpoint is expected at `checkpoints/sam3.1/sam3.1_multiplex.pt`
(`scripts/download_sam3.sh` fetches it).

## How it runs

| Step | Module | Notes |
|---|---|---|
| Probe / decode / encode | `shades/io.py` | 10-bit decode to float RGB via an ffmpeg `rgb48le` pipe; TPDF-dithered H.264 High yuv420p BT.709 encode; source AAC copied untouched |
| SAM 3.1 masks | `shades/masking.py` | Text prompt `"person"`, overlapping chunks of 48 frames (SAM keeps ~100 MB of features per frame in RAM), diver matched across chunks by IoU, cached in `masks/<stem>/` |
| CPU port | `shades/sam31_cpu.py` | Upstream SAM 3.1 video code is CUDA-only. Without CUDA, a patched **copy** of the package is generated in `.cache/` (device strings, pinned memory, a Triton EDT kernel, bf16 autocast moved to the CPU). The `sam3/` clone is never modified |
| Grade | `shades/grade.py` | Per-pixel look baked into a 33³ 3D LUT, also exported as `.cube` |
| Separation | `shades/separation.py` | Feathered mask; diver gets slightly more exposure, warmth and local contrast; background gets slightly less clarity plus faint teal haze; fades out around frames where SAM lost the diver |
| Transition | `shades/transition.py` | Untouched source until `raw_hold_end`, then an eased crossfade of `xfade_dur` |
| Verification | `shades/verify.py` | ffprobe comparison, raw-section pixel diff, contact sheet, palette ΔE2000 |

Flags:

- `--masks-only`: run SAM 3.1 only, e.g. on a GPU machine, then copy `masks/` back.
- `--resegment`: ignore the mask cache.

SAM 3.1 is mandatory: every diver operation (separation, skin tone) runs only inside its masks. Without a GPU and without cached masks, SAM 3.1 runs through the CPU port (about 10–15 s per frame on 4 cores).

### Outputs (in `output/`)

- `<stem>_graded.mp4`: the video.
- `<stem>_grade.cube`: the same grade for Resolve, Premiere or `ffmpeg lut3d`.
- `palette.json` / `palette.png`: the three anchors next to the colours actually measured in the graded frames.
- `<stem>_contact_sheet.jpg`: frames at 0.5 / 1.5 / 2.3 / 2.7 / 3.5 / 8 / 14 s.
- `<stem>_report.json`: checks, mask coverage per frame, lost spans, the full config.

## Target palette (grade anchors, never shown on screen)

| Role | Name | Hex |
|---|---|---|
| shadow | Abyssal Teal | `#0B3A44` |
| mid | Golden Sand | `#B19A69` (requested `#C8A36B`; see below) |
| highlight | Crystal Highlight | `#DFF3F0` |

The anchors define the look. They are **not** sampled from the raw footage.
After rendering, each anchor's region is measured in the graded frames:
- shadow: luminance percentiles 2–12
- mid: saturated warm mid-tones
- highlight: the brightest 0.7%

The pipeline reports ΔE2000 for each (target ≤ 8). On `video_1` the warm content is
dark, algae-covered rock, and the original `#C8A36B` was only reachable at ΔE ≈ 11.7.
Pushing harder turned the rock rust-orange. The hex was therefore moved the smallest
distance that passes, instead of overcooking the grade.

## Tuning guide

Put overrides in a YAML/JSON file and pass `--config`; keys you leave out keep their defaults.

```yaml
transition: {raw_hold_end: 1.5, xfade_dur: 1.2, easing: cubic_in_out}
grade: {grade_strength: 0.85, contrast: 1.3}
separation: {separation_strength: 0.5}
```

| Key | Effect | Default | Safe range |
|---|---|---|---|
| `transition.raw_hold_end` | Seconds of untouched source at the start | 2.0 | 0 – 5 |
| `transition.xfade_dur` | Length of the raw → graded blend | 1.0 | 0.3 – 2.5 |
| `transition.easing` | `smoothstep`, `cubic_in_out` (snappier middle), `linear` | smoothstep | – |
| `grade.grade_strength` | Master mix of the whole grade | 1.0 | 0.5 – 1.0 |
| `grade.contrast` | S-curve slope around `pivot` | 1.22 | 1.0 – 1.45 |
| `grade.pivot` | Grey level the S-curve turns around | 0.42 | 0.35 – 0.5 |
| `grade.mid_gamma` | > 1 sinks mid-tones (deeper water) | 1.10 | 0.95 – 1.25 |
| `grade.black_point` / `white_point` | Output levels | 0.035 / 0.975 | 0 – 0.06 / 0.94 – 1 |
| `grade.channel_restore` | Per-channel (red-recovering) stretch vs common stretch | 0.72 | 0.5 – 0.9 (higher turns ripples copper) |
| `grade.green_cast_removal` / `wb_gains` | Pulls the green/cyan cast back to neutral | 0.55 / (1.25, 0.95, 0.97) | 0.3 – 0.8 |
| `grade.highlight_rolloff` | Fades the WB gains toward white (clean ripples) | 0.85 | 0.6 – 1.0 |
| `grade.green_to_teal_shift` | Rotates green-cyan water toward blue (deg) | 6 | 0 – 15 |
| `grade.water_saturation` | Calms cyan-blue saturation | 0.82 | 0.7 – 1.0 |
| `grade.saturation` | Global saturation | 1.0 | 0.9 – 1.15 |
| `grade.shadow_tint_amount` | Shadows → Abyssal Teal | 0.42 | 0.2 – 0.6 |
| `grade.mid_tint_amount` | Warm mid-tones → Golden Sand | 0.55 | 0.2 – 0.7 |
| `grade.highlight_tint_amount` | Highlights → Crystal Highlight | 0.30 | 0.1 – 0.5 |
| `grade.warm_saturation` / `warm_hue_pull` / `warm_lift` | Sand / rock / skin richness, hue pull toward the sand hue, brightness | 1.30 / 0.8 / 0.14 | 1.0 – 1.4 / 0.3 – 0.9 / 0 – 0.2 |
| `grade.warm_center` / `warm_width` | Hue band treated as warm (deg) | 42 / 30 | 30 – 50 / 20 – 40 |
| `grade.highlight_desat` | Desaturates warm highlights (copper ripples) | 0.55 | 0.3 – 0.8 |
| `grade.warm_protect` | Violet/magenta guard for water and skin | 0.7 | 0.4 – 1.0 |
| `palette[*].hex` | Anchor colours; they also drive the split toning | see above | – |
| `separation.separation_strength` | Overall diver pop | 0.35 | 0 – 0.6 (above ~0.6 starts to look cut out) |
| `separation.mask_feather_px` / `mask_dilate_px` | Edge softness / growth (px at 1080 wide) | 9 / 2 | 5 – 15 / 0 – 4 |
| `separation.subject_exposure` / `subject_warmth` / `subject_local_contrast` | Diver lift | 0.06 / 0.035 / 0.35 | 0 – 0.15 / 0 – 0.06 / 0 – 0.6 |
| `separation.bg_clarity_reduction` / `bg_haze` | Background softening / teal haze | 0.25 / 0.10 | 0 – 0.4 / 0 – 0.2 |
| `separation.lost_ramp_frames` | Fade length around frames where SAM lost the diver | 6 | 4 – 12 |
| `masking.text_prompt` | SAM 3.1 concept prompt | "person" | e.g. "diver", "swimmer" |
| `masking.point_prompts` | `[[frame, x, y, 1/0, obj_id], ...]` corrective clicks | [] | – |
| `masking.chunk_frames` / `chunk_overlap` | RAM vs. speed for SAM | 48 / 6 | 24 – 96 / 4 – 10 |
| `encode.crf` / `preset` | H.264 quality / speed | 16 / slow | 14 – 20 |

### 9:16 export notes

- **1080x1920 (default):** output = source resolution and fps; nothing to set.
- **2160x3840:** masks, feather and blur radii scale with frame width (reference: 1080 px), and the LUT is resolution-independent. Halve `masking.chunk_frames` (≈ 24) because SAM's per-frame memory grows with input size, and consider `encode.crf: 18` to keep files reasonable.
- **Other aspect ratios:** the pipeline never crops or scales; reframe in your editor afterwards, or before running so the masks match.

## Limitations

- SAM 3.1 on CPU is slow (~10 s/frame on 4 cores) and runs bf16 autocast; it needs a CPU with AVX512-BF16/AMX to be practical. On a CUDA GPU the upstream code is used unchanged.
- The upstream SAM 3.1 base predictor passes `offload_state_to_cpu` to a multiplex `init_state()` that does not accept it; `masking.py` filters unknown kwargs.
- The grade is a global LUT. Frames with very different light (surface vs. depth) share one look by design.
