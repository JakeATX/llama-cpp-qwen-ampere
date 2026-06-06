#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-build-atx-metal}"
MODEL="${MODEL:-}"
OUT_ROOT="${OUT_ROOT:-$(dirname "$REPO")/runs/atx_moe_metal}"

if [[ -z "$MODEL" ]]; then
  echo "Set MODEL to a local Qwen3.6 GGUF path" >&2
  exit 1
fi

cd "$REPO"
git fetch jakeatx atx-expert-residency
git merge --ff-only jakeatx/atx-expert-residency

cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$OUT_ROOT/session_$STAMP"
mkdir -p "$OUT"

python3 scripts/atx_moe_metal_acceptance.py \
  --repo "$REPO" \
  --build-dir "$BUILD_DIR" \
  --model "$MODEL" \
  --policy "${POLICY:-}" \
  --out-dir "$OUT/acceptance" \
  --skip-build

echo "Acceptance artifacts: $OUT/acceptance"
