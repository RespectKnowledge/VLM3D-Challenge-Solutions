#!/usr/bin/env bash
set -euo pipefail

W="${FORITHMUS_WEIGHTS_DIR:-/weights}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$W/hf-cache"
export HUGGINGFACE_HUB_CACHE="$W/hf-cache/hub"

exec python /opt/app/run_inference.py
