#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-build-atx-metal}"
MODEL="${MODEL:-/Users/jkooker/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
POLICY_SOURCE="${POLICY_SOURCE:-$(dirname "$REPO")/runs/hf_combined_saliency_upload/staging/policies}"
OUT_ROOT="${OUT_ROOT:-$(dirname "$REPO")/runs/atx_moe_metal}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"

if [[ ! -f "$MODEL" ]]; then
  echo "MODEL not found: $MODEL" >&2
  exit 1
fi

cd "$REPO"
export MODEL POLICY_SOURCE

python3 scripts/atx_moe_autonomous_loop.py \
  --repo "$REPO" \
  --build-dir "$BUILD_DIR" \
  --model "$MODEL" \
  --policy-source "$POLICY_SOURCE" \
  --runs-root "$OUT_ROOT/autonomous" \
  --max-iterations "$MAX_ITERATIONS"
