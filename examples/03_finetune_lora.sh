#!/usr/bin/env bash
# LoRA fine-tuning (fits smaller GPUs). best/ holds the adapter; final/ is a merged full model.
set -euo pipefail
TRAIN=${1:?usage: $0 train.tsv dev.tsv test.tsv}
DEV=$2
TEST=$3
EXP=${EXP:-exp/lora}
NGPU=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

cohere-ara train --gpus "$NGPU" \
  --train-manifest "$TRAIN" --valid-manifest "$DEV" --exp-dir "$EXP" \
  --lora --lora-r 32 --lora-alpha 64 --lora-targets decoder,encoder --lr 1e-4 \
  --num-epochs 5 --max-batch-duration 240 --gradient-checkpointing --spec-augment

# The adapter directory is auto-detected (base model + adapter merged at load time):
cohere-ara decode "$TEST" -o "$EXP/decode_test" --model "$EXP/best" --gpus "$NGPU"
