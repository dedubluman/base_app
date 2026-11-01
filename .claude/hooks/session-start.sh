#!/bin/bash
# Cloud sessions start from an empty container, so the SAM 3 source code and
# the SAM 3.1 weights (gitignored, ~3.3 GB) are downloaded again at startup.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Download only if the code or weights are missing, or if this hook or the
# download script changed since the last successful download.
hashes=checkpoints/sam3.1/.session-start.sha256
weights=checkpoints/sam3.1/sam3.1_multiplex.pt
if [ ! -d sam3 ] || [ ! -f "$weights" ] || ! sha256sum --check --status "$hashes" 2>/dev/null; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set: cannot download the gated facebook/sam3.1 weights." >&2
    exit 1
  fi

  # Delete the old hashes first, so a failed download runs again next time
  rm -f "$hashes"

  ./scripts/download_sam3.sh

  sha256sum .claude/hooks/session-start.sh scripts/download_sam3.sh > "$hashes" || rm -f "$hashes"
fi
