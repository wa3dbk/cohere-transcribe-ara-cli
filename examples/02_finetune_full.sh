#!/usr/bin/env bash
# Full fine-tuning with DDP on all local GPUs, then decode the test set with the best checkpoint.
set -euo pipefail
TRAIN=${1:?usage: $0 train.tsv dev.tsv test.tsv}
DEV=$2
TEST=$3
EXP=${EXP:-exp/full_ft}
NGPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

cohere-ara train --gpus "$NGPU" \
  --train-manifest "$TRAIN" --valid-manifest "$DEV" --exp-dir "$EXP" \
  --lr 1e-5 --num-epochs 5 --warmup-ratio 0.05 --lr-scheduler cosine \
  --max-batch-duration 240 --grad-accum 1 --gradient-checkpointing \
  --spec-augment --valid-interval 1000 --valid-decode-utts 500 \
  --early-stopping-patience 5 --num-workers 6

cohere-ara decode "$TEST" -o "$EXP/decode_test" --model "$EXP/best" --gpus "$NGPU" --details
cat "$EXP/decode_test/wer.txt"
