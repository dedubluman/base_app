# Underwater grade: SAM 3.1 masks, reference match, 12 s optical-flow retime

Pipeline used for `output/underwater_graded_12s_1080p_h264.mp4` (source 1232x1078, 29.97 fps, 5.37 s).

```bash
source .venv/bin/activate                      # python 3.12, CPU torch, `pip install -e sam3/`

# 1. SAM 3.1 subject masks (text prompt "person"; ~45 min on 4 CPU cores)
python uw_masks.py --input clip.mov --masks masks/main
# targeted re-runs where the body leaves/enters frame (reverse/forward clips cut with ffmpeg)
python uw_masks.py --input auxA_rev0_20.mkv --masks masks/auxA        # frames 20 -> 0 (tracked backward)
python uw_masks.py --input auxC_rev66_78.mkv --masks masks/auxC       # frames 78 -> 66 (blurred hand)
python uw_masks.py --input clip_every3rd.mkv --masks masks/shorts_s3 --prompt shorts
python uw_merge_masks.py --main masks/main --out masks/merged \
    --aux masks/auxA:20:-1 --aux masks/auxB:64:1 --aux masks/auxC:78:-1

# 2. masked, reference-matched grade -> lossless 10-bit FFV1 intermediate
python uw_grade.py --input clip.mov --masks masks/merged \
    --shorts-masks masks/shorts_s3 --shorts-stride 3 --out graded.mkv
#    quick look at single frames: add --preview 12,40,100 and drop --out's file (it is a dir then)

# 3. optical-flow retime to exactly 12 s + 1080p H.264
./uw_retime.sh graded.mkv out.mp4 12 30        # FIT=crop for a 16:9 fill instead of side bars
```

Grade parameters live in `P` at the top of `uw_grade.py` (water chroma, bloom, skin target a*/b*,
subject lift/gain, mask feather/erode).  Retime: `minterpolate` (MCI, bidirectional ME, OBMC, VSBMC),
360 frames @ 30 fps, libx264 High CRF 17 slow, yuv420p BT.709, no audio, 1234x1080 centred on 1920x1080.
