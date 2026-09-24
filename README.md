# cohere-transcribe-ara-cli

A command-line tool to **decode, evaluate and fine-tune**
[Cohere Transcribe Arabic](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026)
(2B-parameter Conformer encoder / Transformer decoder ASR model) on Common-Voice-like TSV manifests,
on one GPU or several GPUs of the same machine.

```bash
cohere-ara decode test.tsv -o decode_base --gpus 4                  # transcribe + WER/CER
cohere-ara train --train-manifest train.tsv --valid-manifest dev.tsv --exp-dir exp/ft --gpus 4
cohere-ara decode test.tsv -o exp/ft/decode_test --model exp/ft/best --gpus 4
```

## Features

- **Fast batched inference**: utterances sorted by duration and packed into batches by *padded* audio
  seconds, audio loading + log-mel extraction in DataLoader workers, bf16 + SDPA (or FlashAttention-2),
  a per-batch token budget that stops hallucination loops early, automatic batch splitting on CUDA OOM.
- **Multi-GPU decoding** (`--gpus N`): the manifest is sharded across processes and results merged.
- **Resumable decoding** (`--resume`): hypotheses are flushed per batch; restarts skip finished segments.
- **Scoring**: WER / CER / SER with Arabic-aware normalizers, sclite-style per-utterance alignments,
  most frequent substitutions / deletions / insertions.
- **Fine-tuning** with 🤗 Accelerate: full, partially frozen, decoder-only or **LoRA**; single GPU or DDP;
  dynamic duration batching with bucketing, bf16 mixed precision, gradient accumulation and checkpointing,
  SpecAugment, WER-based best-model selection, early stopping, resumable checkpoints.
- Output directories load directly with `transformers` (`CohereAsrForConditionalGeneration.from_pretrained`).

## Installation

```bash
git clone https://github.com/wa3dbk/cohere-transcribe-ara-cli && cd cohere-transcribe-ara-cli
pip install -e .              # core
pip install -e ".[all]"       # + LoRA (peft), 8-bit Adam (bitsandbytes), tensorboard
```

Requires Python ≥ 3.10, PyTorch ≥ 2.4 and `transformers >= 5.4` (native `CohereAsr` support).

The model is **gated** on the Hugging Face Hub: accept the conditions on the model page, then
`huggingface-cli login` (or `export HF_TOKEN=...`, or pass `--hf-token`).

## Data format

One TSV per split, one segment per line — a segment id, a path to a 16 kHz wav, and the transcript:

```
seg_0001	clips/seg_0001.wav	شنوة الحكاية اليوم
seg_0002	clips/seg_0002.wav	I'm going to the office ثم نرجع
```

- **Headerless** files use columns `0=id, 1=path, 2=text`.
- **With a header**, common names are detected automatically: id (`id`, `segment_id`, `utt_id`, …),
  audio (`path`, `audio`, `audio_filepath`, `wav`, …), text (`sentence`, `text`, `transcript`, …),
  duration (`duration`, …). A real Common Voice TSV works as is (the id defaults to the clip file name).
- Override with `--id-col/--audio-col/--text-col/--duration-col` (name or 0-based index) and `--header yes|no`.
- Relative paths are resolved against `--audio-root`, or the TSV's directory by default.
- The TSV is read without quote processing, so `"` in transcripts is safe. Non-16 kHz or stereo files are
  resampled / down-mixed on the fly (with a warning from `prepare`).
- A decode manifest may omit the transcript column: you just get hypotheses, without scores.

`prepare` validates a manifest, reads durations from the audio headers (in parallel) and writes a clean
manifest with a duration column, so later runs don't need to re-read headers:

```bash
cohere-ara prepare train.tsv -o data/train.tsv --max-duration 30
# prints: hours, unreadable files, non-16 kHz files, empty transcripts, duration stats, segments > 30 s
```

## Decoding and evaluation

```bash
cohere-ara decode data/test.tsv -o decode_base \
    --gpus 4 --max-batch-duration 800 --batch-size 128 --num-workers 6 --details
```

| Output | Content |
|---|---|
| `hyp.tsv` | `id  ref  hyp` in manifest order |
| `hyp.txt` | Kaldi-style `id hyp` (for sclite or other tools) |
| `wer.txt` / `wer.json` | WER, CER, SER per normalizer, S/D/I counts, top confusions |
| `details.<normalizer>.txt` | aligned REF/HYP per utterance, worst first (`--details`) |
| `summary.json` | settings, scores, audio hours, wall time and RTFx |
| `parts/` | per-process partial results (used by `--resume`) |

Useful options:

- `--language ar|en` sets the prompt language token; `--no-punctuation` uses the `<|nopnc|>` prompt.
- `--num-beams 4` for beam search (greedy is the default and much faster);
  `--repetition-penalty`, `--no-repeat-ngram-size` are also available.
- `--max-new-tokens` (default 448) and `--tokens-per-second` (default 20): each batch may generate at most
  `16 + rate × longest duration` tokens. `summary.json` reports how many hypotheses hit the budget.
- `--dtype auto|bf16|fp16|fp32`, `--attn-implementation sdpa|flash_attention_2|eager`.
- `--normalizers arabic,none` picks the scoring normalizations (see below).
- `--limit 200` decodes only the first N segments (quick checks).

Audio longer than ~30 s is split by the model's feature extractor at low-energy points and the chunk
transcripts are re-joined, so long segments decode correctly too.

**Tuning speed.** `--max-batch-duration` is the main knob (padded seconds of audio per batch). Increase it
until GPU memory is well used; if a batch runs out of memory it is split in two and retried automatically.
Use enough `--num-workers` to keep the GPU busy (check with `nvidia-smi`).

### Scoring an existing hypothesis file

```bash
cohere-ara score --ref data/test.tsv --hyp other_system.txt -o score_other --details
```

`--hyp` accepts `hyp.tsv` from `decode`, `id<TAB>text`, or Kaldi `id text`.

### Normalizers

| Name | Applies |
|---|---|
| `none` | Unicode NFKC, whitespace collapsing only |
| `basic` | `none` + case folding + punctuation/symbol removal |
| `arabic` (default) | `basic` + diacritics & tatweel removal, أ/إ/آ/ٱ → ا, ى/ی → ي, ک → ك, Arabic-Indic digits → 0-9 |
| `arabic-strict` | `arabic` + ة → ه, ؤ → و, ئ → ي |

Scores are reported for every normalizer given (default: `arabic` and `none`).

## Fine-tuning

```bash
cohere-ara train \
    --train-manifest data/train.tsv --valid-manifest data/dev.tsv --exp-dir exp/ft \
    --gpus 4 --lr 1e-5 --num-epochs 5 --max-batch-duration 240 \
    --gradient-checkpointing --spec-augment --valid-interval 1000 --valid-decode-utts 500
```

What happens:

1. Manifests are validated once and cached in `EXP/data/` (durations included; unreadable files dropped).
2. Transcripts are tokenized; segments that are too short/long (`--min-duration`, `--max-duration` ≤ 30 s),
   empty, or implausibly dense (`--max-tokens-per-second`) are filtered and counts are logged.
3. A baseline validation runs first (`--no-eval-at-start` to skip): teacher-forced loss plus greedy WER on
   `--valid-decode-utts` random dev utterances (`-1` = all, `0` = loss only).
4. Training uses per-GPU micro-batches of ≤ `--max-batch-duration` padded seconds (bucketed and reshuffled
   every epoch), bf16 autocast, AdamW, warmup + cosine decay and gradient clipping.
5. Validation every `--valid-interval` updates and at each epoch end. The best model (by WER, or by loss
   when not decoding) is saved to `EXP/best/`; `EXP/final/` holds the last model.

Experiment directory:

```
exp/ft/
├── train.log, train_args.json, train_summary.json
├── data/{train,valid}.tsv, data/stats.json
├── best/            # inference model (bf16 safetensors + processor)
├── final/           # last model
└── checkpoints/checkpoint-<step>/   # full training state (model, optimizer, scheduler, RNG) for resuming
```

**Resume** an interrupted run with the same command plus `--resume` (or `--resume-from DIR`): training
continues mid-epoch at the exact batch it stopped at.

### Multiple GPUs

`--gpus N` re-launches the command with `torchrun` (DDP on one machine). You can also launch it yourself:

```bash
torchrun --standalone --nproc_per_node 4 -m cohere_transcribe_ara_cli train ...
accelerate launch --num_processes 4 -m cohere_transcribe_ara_cli train ...
```

`--max-batch-duration` is **per GPU**; the effective batch is `N_GPUs × grad_accum × micro-batch`. Every
process receives the same number of similarly sized batches, so GPUs stay in lock-step.

### What to train, and memory

The model has ~2B parameters (a 48-layer Conformer encoder dominates). Full fine-tuning keeps fp32 master
weights, gradients and AdamW states, i.e. roughly 16 bytes per trainable parameter **before activations**:

| Setup | Flags | Weights + optimizer (approx.) |
|---|---|---|
| Full fine-tuning | *(default)* | ~32 GB → 80 GB GPUs; use `--gradient-checkpointing` |
| Full, 8-bit Adam | `--optim adamw8bit` | ~20 GB |
| Freeze lower encoder | `--freeze-encoder-layers 24` | proportionally less |
| Decoder only | `--freeze-encoder` | encoder weights only + small decoder |
| LoRA | `--lora --lora-r 32` | ~4 GB bf16 base + small adapters → 24 GB GPUs |

These are rough figures; activation memory scales with `--max-batch-duration`. If you hit OOM, reduce
`--max-batch-duration` and compensate with `--grad-accum`, and/or enable `--gradient-checkpointing`.

With `--lora`, `best/` contains the **adapter only** (plus processor). `decode --model EXP/best` detects it
and loads base model + adapter automatically; `final/` is a merged full model
(`--no-merge-lora-at-end` to keep an adapter). `cohere-ara merge-lora --adapter DIR -o OUT` merges any
adapter. LoRA targets decoder self/cross-attention + MLP and encoder attention + feed-forward layers
(`--lora-targets decoder` for the decoder only). The default learning rate becomes 1e-4 with LoRA.

### Other training options

- `--no-punctuation`: train with the `<|nopnc|>` prompt, e.g. when transcripts have no punctuation.
  Use the same flag when decoding.
- `--train-text-normalizer none|basic|arabic|arabic-strict`: normalize training targets (default: as is).
- `--language`: prompt language (keep `ar` for Arabic and code-switched data).
- `--encoder-lr-scale 0.5`: smaller LR for the encoder than the decoder.
- `--label-smoothing`, `--dropout`, `--attention-dropout`, `--layerdrop`, `--weight-decay`.
- SpecAugment: `--spec-augment --freq-masks 2 --freq-mask-width 27 --time-masks 10 --time-mask-ratio 0.05`.
- `--early-stopping-patience N`, `--max-steps N`, `--keep-checkpoints K`, `--save-dtype bf16|fp16|fp32`.
- `--report-to tensorboard|wandb` for experiment tracking.

Starting points (to tune on your data): full fine-tuning `--lr 1e-5` to `3e-5` with 5 % warmup;
LoRA `--lr 1e-4`. Watch the baseline validation: the pre-trained model is already strong, and too high a
learning rate on a small dataset degrades it quickly.

### Before a long run

```bash
# 1. The pipeline works on your data (a few minutes)
cohere-ara decode data/dev.tsv -o /tmp/check --limit 200
cohere-ara train --train-manifest data/train.tsv --valid-manifest data/dev.tsv --exp-dir /tmp/smoke \
    --max-steps 50 --valid-interval 25 --valid-decode-utts 50 --save-interval 0
# 2. Check in /tmp/smoke/train.log that the step-0 validation WER is in line with the `decode` WER,
#    that the loss decreases, and that the reported peak memory leaves headroom.
```

## How training targets are built

At inference, `generate()` starts the decoder with the processor's prompt
`▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|ar|> <|ar|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>`
and prepends `decoder_start_token_id` when the prompt doesn't begin with it. Training reproduces exactly
that prefix (`inference_prefix`):

```
decoder input : prefix + text
labels        : [-100] * (len(prefix) - 1) + text + <eos>
```

The loss is token-level cross-entropy over transcript tokens and EOS only. The test-suite checks that the
prefix equals what `generate()` uses, and that a model overfit on one phrase reproduces it exactly with
greedy decoding.

## Development

```bash
pip install -e ".[dev]"
pytest -q            # ~1 min on CPU, no download: uses a tiny randomly initialised Cohere-ASR model
ruff check src tests
```

Tests cover manifest parsing, normalization, WER alignment, the batch sampler (DDP-equal, disjoint,
deterministic, resumable), collators, decoding + scoring + resume, overfit-and-decode, training
resume, LoRA, and 2-process decoding / DDP training (torchrun on CPU with gloo).
`python -m cohere_transcribe_ara_cli.testing DIR` builds the tiny model and a synthetic dataset for manual checks.

## Limitations

- Training segments must be ≤ 30 s (the model's single-chunk limit). Split longer segments first.
- Multi-GPU training is DDP on a single machine; FSDP / DeepSpeed / multi-node are not supported.
- No timestamps or diarization (not supported by the model).
- Like other attention encoder-decoder ASR models, it may transcribe noise or silence; remove
  non-speech segments with a VAD beforehand when possible.

## License

Apache-2.0. The model weights have their own license and terms on the Hugging Face Hub.
