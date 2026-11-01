#!/usr/bin/env bash
# Downloads Meta's SAM 3 source code and the SAM 3 / SAM 3.1 model weights.
# Requires HF_TOKEN for an account that has accepted the facebook/sam3 and
# facebook/sam3.1 licenses.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d sam3 ]; then
  git clone --depth 1 https://github.com/facebookresearch/sam3.git sam3
fi

pip install -q -U huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
for repo in ('facebook/sam3', 'facebook/sam3.1'):
    print(snapshot_download(repo, local_dir='checkpoints/' + repo.split('/')[1]))
"
