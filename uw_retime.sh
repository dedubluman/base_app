#!/usr/bin/env bash
# Optical-flow slow motion to an exact duration + 1080p H.264 export.
#
#   ./uw_retime.sh graded.mkv out_1080p_h264.mp4 [seconds=12] [fps=30]
#
# * minterpolate (motion-compensated, bidirectional ME, overlapped block MC,
#   variable-size block MC) synthesises the in-between frames; the source is
#   stretched with setpts first so the interpolator invents ~2.2 frames per
#   source frame instead of repeating them.
# * Exactly SECONDS*FPS frames are written (default 12 s * 30 fps = 360).
# * Output is 1920x1080: the frame is scaled to full 1080 height and centred
#   with black side bars (the source is 1232x1078, nearly square, so cropping
#   to 16:9 would cut the subject).  Set FIT=crop to fill 16:9 instead.
# * libx264 High, yuv420p, BT.709, CRF 17, preset slow, no audio.
set -euo pipefail
IN=${1:?input}; OUT=${2:?output}; SEC=${3:-12}; FPS=${4:-30}
FIT=${FIT:-pad}
N=$(( SEC * FPS ))
DUR=$(ffprobe -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 "$IN")
K=$(python3 - <<E
print(f"{$SEC/$DUR:.6f}")
E
)
echo "[retime] source ${DUR}s -> ${SEC}s (setpts x${K}), ${N} frames @ ${FPS} fps"

if [ "$FIT" = crop ]; then
  FIT_VF="scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos+accurate_rnd,crop=1920:1080"
else
  FIT_VF="scale=-2:1080:flags=lanczos+accurate_rnd,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"
fi

ffmpeg -hide_banner -v error -nostats -y -i "$IN" -an \
  -vf "setpts=${K}*PTS,minterpolate=fps=${FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:mb_size=16:search_param=32:vsbmc=1:scd=none,tpad=stop_mode=clone:stop=${FPS},${FIT_VF},format=yuv420p" \
  -frames:v "$N" -r "$FPS" \
  -c:v libx264 -preset slow -crf 17 -profile:v high -level 4.2 -pix_fmt yuv420p \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -movflags +faststart "$OUT"

ffprobe -v error -show_entries stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames:format=duration \
  -of default=nw=1 "$OUT"
