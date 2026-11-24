#!/usr/bin/env bash
# End-to-end: raw underwater clip -> SAM 3.1 mattes -> reference-locked grade -> H.264 1080p.
#
#   scripts/run_fish_grade.sh inputs/fish.mov REF_STILL_1.jpg [REF_STILL_2.jpg ...]
#
# Needs the .venv from README_three_shades.md (Python 3.12, torch, `pip install -e sam3/`, einops) and
# checkpoints/sam3.1/sam3.1_multiplex.pt. On CPU the SAM 3.1 passes take ~15 s per segmented frame
# (4 cores); they are resumable (--resume) and run ONE model at a time (a second copy runs out of RAM).
set -euo pipefail
cd "$(dirname "$0")/.."
IN=${1:?input video}; shift
REFS=("$@"); [ ${#REFS[@]} -gt 0 ] || { echo "give at least one reference still"; exit 1; }
STEM=$(basename "${IN%.*}")
source .venv/bin/activate

# 1) SAM 3.1 keyframe masks (text prompts). Fish every 3rd frame, seabed every 6th; the rest is
#    optical-flow interpolation in fish_refine.py.  Prompts that SAM 3.1 answers on this clip:
#    "fish" (p~0.94), "sandy bottom" (p 0.85-0.93), "fin" (p 0.65-0.88).  "seagrass", "rock", "weed",
#    "plant", ... return no detections, so those two regions are derived (see fish_refine.py).
python fish_masks.py --input "$IN" --chunk 32 --overlap 4 --resume \
    --regions "fish:fish:3" "sand:sandy bottom:6"

# 2) Optional: fins as a separate SAM 3.1 part where the fish mask drops the pectoral fin
#    (frames 222-333 on this clip). Skip or change the range for other clips.
if [ -n "${FINS_RANGE:-}" ]; then
  python fish_masks.py --input "$IN" --chunk 32 --overlap 4 \
      --start "${FINS_RANGE%-*}" --end "${FINS_RANGE#*-}" --regions "fins:fin:3"
fi

# 3) mattes: fish / seabed / midground / background (+ overlay.mp4 for a visual check)
PARTS=""; [ -d "masks/${STEM}__fins" ] && PARTS="--parts fins"
python fish_refine.py --input "$IN" --fish-gids 1 $PARTS

# 4) measure the reference stills -> grade_params.json (tone curves, tints, fish chroma window)
python fish_grade.py calibrate --input "$IN" --refs "${REFS[@]}"

# 5) render H.264 High, yuv420p, BT.709, source audio copied (CRF 15, preset slow)
mkdir -p output
python fish_grade.py render --input "$IN" --output "output/${STEM}_graded.mp4"

# 6) point-wise looks as .cube LUTs for Resolve / OCIO, measured grade as text, numeric QC
python fish_grade.py luts --out "output/${STEM}_luts"
python fish_grade.py spec > "output/${STEM}_grade_numbers.txt"
python fish_qc.py video --video "output/${STEM}_graded.mp4" --source "$IN" | tee "output/${STEM}_qc.txt"
