#!/usr/bin/env bash
# End-to-end deep-abyss grade: SAM 3.1 masks -> propagation -> (optional look fit) -> 1080p H.264 render -> checks.
#
#   scripts/run_abyss.sh inputs/dive.mov                       # masks (cached) + render
#   scripts/run_abyss.sh inputs/dive.mov --reference ref.jpg --ref-frame 241    # also re-fit the look to a reference
#   STRIDE=1 scripts/run_abyss.sh inputs/dive.mov              # segment every frame (GPU)
#   NO_MASKS=1 scripts/run_abyss.sh inputs/dive.mov            # global grade only, no SAM
#
# Output: output/<stem>_abyss_1080p.mp4 (+ output/<stem>_abyss_sheet.jpg)
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${1:?usage: run_abyss.sh VIDEO [--reference STILL --ref-frame N]}"; shift || true
REF=""; REF_FRAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --reference) REF="$2"; shift 2;;
    --ref-frame) REF_FRAME="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
STEM="$(basename "${VIDEO%.*}")"
OUT="output/${STEM}_abyss_1080p.mp4"
mkdir -p output

if [ -z "${NO_MASKS:-}" ]; then
  python3 abyss_masks.py --input "$VIDEO" --stride "${STRIDE:-6}"
  python3 abyss_propagate.py --input "$VIDEO"
  MASK_ARGS=()
else
  MASK_ARGS=(--no-masks)
fi

if [ -n "$REF" ]; then
  : "${REF_FRAME:?--ref-frame is required with --reference}"
  python3 abyss_fit.py --video "$VIDEO" --frame "$REF_FRAME" --reference "$REF" --out abyss_look.json
fi

python3 abyss_grade.py --input "$VIDEO" --output "$OUT" "${MASK_ARGS[@]}"
python3 abyss_qc.py check --video "$VIDEO" --graded "$OUT"
python3 abyss_qc.py sheet --video "$VIDEO" --graded "$OUT" --out "output/${STEM}_abyss_sheet.jpg" --overlay
echo "done: $OUT"
