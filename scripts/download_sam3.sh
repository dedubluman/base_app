#!/usr/bin/env bash
# Downloads Meta's SAM 3 source code and model weights.
# Requires HF_TOKEN for an account that has accepted the facebook/sam3 license.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d sam3 ]; then
  git clone --depth 1 https://github.com/facebookresearch/sam3.git sam3
fi

pip install -q -U huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('facebook/sam3', local_dir='checkpoints/sam3'))
"
