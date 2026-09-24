# Changelog

## 0.1.0
- `prepare`: validate TSV manifests, read durations from audio headers, filter, write normalized manifests.
- `decode`: duration-bucketed batched greedy/beam decoding, DataLoader feature extraction, CUDA-OOM back-off,
  resumable output, multi-GPU sharding (`--gpus N`), LoRA adapters, WER/CER with Arabic normalizers.
- `score`: WER/CER, per-utterance alignments and top confusions for any hypothesis file.
- `train`: full / partial / LoRA fine-tuning with Accelerate (single GPU or DDP), dynamic batching,
  bf16 mixed precision, gradient accumulation & checkpointing, SpecAugment, WER-based model selection,
  early stopping, resumable checkpoints.
- `merge-lora`: merge an adapter into its base model.
