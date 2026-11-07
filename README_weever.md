# Weever fish — SAM 3.1 isolated neon grade over a dark deep-sea floor

Source: `player_export.mov`, HEVC 1920x1080 with a −90° display matrix, so
it plays as **1080x1920 portrait**, 29.97 fps, 610 frames, 20.354 s, AAC audio.

Output: `output/weever_graded.mp4`, H.264 High, 1080x1920 upright, 29.97 fps,
610 frames, 20.354 s, CRF 12 (~20.6 Mb/s), 16-bit internal processing with
dither, original AAC stream copied.

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `weever_masks.py --only coarse` | SAM 3.1 multiplex, text prompt `fish`, every 4th frame of the full portrait frame, one session → fish trajectory |
| 2 | `weever_masks.py --only fine` | SAM 3.1 again on a 1080x1080 window that follows the fish (smoothed path), so SAM sees it at ~0.93x native scale instead of a 0.52x vertical squash; 36-frame chunks, 6-frame overlap, fish linked across seams by IoU, stray detections rejected |
| 3 | `weever_clicks.py --rich` + `weever_masks.py --points` | SAM 3.1 point refinement where the mask had absorbed the **cast shadow of the dorsal fin** (frames 11–91): full text propagation, then positive clicks over head/body/tail/pectoral fin (pale pixels only) and negative clicks on the shadow streak on the fish's own object id, then re-propagation |
| 4 | `weever_matte.py` | 2-of-3 temporal vote of SAM masks (DIS-flow aligned neighbours), hole fill / fragment cleanup, colour guided-filter matte constrained to a −3/+4 px band around the SAM edge (fin edges, no sand), flow-warped neighbour blend inside that band only (edge shimmer) |
| 5 | `weever_grade.py` | two independent layers composited by the matte (below) |
| 6 | `weever_qc.py` | mask/alpha overlays at 100%, before/after sheets, per-frame flicker statistics |

Run order:

```bash
python weever_masks.py --input inputs/weever.mov --only coarse
python weever_masks.py --input inputs/weever.mov --only fine
python weever_clicks.py --src fine --rich --every 10 --out output/clicks_rich.json
python weever_masks.py --input inputs/weever.mov --only fine --start 0 --end 96 --points output/clicks_rich_0_96.json
python weever_matte.py --input inputs/weever.mov
python weever_grade.py --input inputs/weever.mov --out output/weever_graded.mp4
```

The clicks file is filtered to frames below 96 before the refinement run.

### Fish layer (inside the matte only)
* Skin luminance (edge-preserving denoise) is split into an illumination base
  (caustics sliding over the body) and texture detail. The base is compressed
  with clip-wide constants (no per-frame auto exposure), and the detail is
  boosted 1.3x.
* Warm gradient map: maroon → fiery red → neon orange → amber → golden yellow.
* Cool gradient map: cobalt → electric blue → cyan → turquoise. Where it is
  used comes from the fish itself: the tail fin (body axis from the matte,
  head/tail tracked frame to frame), plus the fish's own pale reticulation
  (band-pass of real skin texture). That is why the cyan rings follow the
  real spot pattern. No patterns or anatomy are invented.
* LCh blend: hue travels orange → cyan → blue with chroma muted mid-way, so
  there is no magenta or green band. Gentle rounded-body shading. Low bloom
  (0.05) from the brightest fish highlights only.

### Environment layer (outside the matte)
Sand chroma removed and re-toned to the reference's dark teal/blue-green
(reference background median RGB ≈ 15/26/24; graded median luma 0.044–0.056 vs
reference 0.048). Fixed tone curve with restrained caustic highlights,
large-radius local contrast so sand texture survives, a soft light pool that
follows the fish (path smoothed over ~0.4 s), heavy vignette, faint haze. No
objects or particles added; seabed geometry untouched.

## Verification (measured, not assumed)
* SAM 3.1 actually executed: `sam3.1_multiplex.pt`, torch 2.15.0+cpu,
  `torch.cuda.is_available() == False`, CPU port of the upstream package
  (`shades/sam31_cpu.py`). Logged `loaded on device=cpu`; ~5.8 s/frame.
  `masks/weever__fish/meta.json`: device `cpu`, 610/610 frames, 30 SAM sessions,
  396 refinement clicks.
* Tracking: one fish id through all 21 chunks; 2 stray detections rejected.
  No frame without SAM output (0 flow-filled). 0 area outliers after the vote.
  Median consecutive-frame matte IoU 0.92 (lowest values = fast pectoral-fin
  flaps in the last frames).
* Temporal stability of the graded video: background luma max frame-to-frame
  change 0.42/255; fish saturation 208–228; fish luma changes correlate 0.98
  with the source (caustics moving over the fish) at 0.82x their size, so
  the grade adds no flicker.
* Rendered file: ffprobe 610 frames, 20.354 s, 1080x1920, 29.97 fps, audio present.

## Revision: tail slip at ~15.4–16.0 s (frames 463–480)

The fish whips its tail; the frames are motion-blurred and SAM had absorbed the
blurred ghost of the previous tail position (left of the caudal peduncle).
Measured as the body's extension left of its own dark-spot row: clean frames
27–29 px, affected frames 35–42 px.

* SAM 3.1 refinement on frames 452–491 (one 40-frame session) with clicks on
  every frame 463–480 (288 clicks): negative clicks at 40 px left of the spot
  row where the mask reached ≥ 44 px, and positive clicks on the spot row,
  head-to-tail interior and pale pectoral fins. Result: 23–34 px (467: 38).
* `weever_matte.py --smooth-range 462:482`: open/close and blur-threshold of
  the mask edge for this fast-motion section only, because SAM's edge follows
  the ragged blur there.
* `weever_grade.py --from-frame 310` renders the last 10 s (frames 310–609,
  300 frames, 10.01 s, H.264 High, audio re-encoded to AAC from the same point).
