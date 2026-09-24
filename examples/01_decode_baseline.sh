#!/usr/bin/env bash
# Decode + score the base model on a test set (all local GPUs).
set -euo pipefail
TEST=${1:?usage: $0 test.tsv [out_dir]}
OUT=${2:-decode_base}
NGPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

cohere-ara prepare "$TEST" -o data/test.prepared.tsv          # durations + sanity report (optional, cached)
cohere-ara decode data/test.prepared.tsv -o "$OUT" \
  --gpus "$NGPU" --max-batch-duration 800 --num-workers 6 --details
cat "$OUT/wer.txt"
