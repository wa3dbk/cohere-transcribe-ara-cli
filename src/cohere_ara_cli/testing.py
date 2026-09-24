"""Utilities to build a *tiny, randomly initialised* Cohere-ASR model and a synthetic dataset.

Used by the test-suite and handy to smoke-test an installation without downloading the 2B checkpoint:

    python -m cohere_transcribe_ara_cli.testing /tmp/tiny   # writes /tmp/tiny/model, /tmp/tiny/{train,valid}.tsv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

SPECIAL_TOKENS = [
    "<unk>",
    "<|nospeech|>",
    "<pad>",
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|startofcontext|>",
    "<|emo:undefined|>",
    "<|pnc|>",
    "<|nopnc|>",
    "<|itn|>",
    "<|noitn|>",
    "<|timestamp|>",
    "<|notimestamp|>",
    "<|diarize|>",
    "<|nodiarize|>",
] + [f"<|{lang}|>" for lang in ("ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh")]

PHRASES = ["مرحبا بكم في تونس", "شكرا جزيلا يا صديقي", "hello world", "صباح الخير"]


def build_tiny_model(out_dir: str, seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import (
        CohereAsrConfig,
        CohereAsrFeatureExtractor,
        CohereAsrForConditionalGeneration,
        CohereAsrProcessor,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(seed)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always")
    trainer = trainers.BpeTrainer(vocab_size=160, special_tokens=SPECIAL_TOKENS, initial_alphabet=["\u2581"])
    tok.train_from_iterator(PHRASES * 20, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|endoftext|>",
        bos_token="<|startoftranscript|>",
    )
    processor = CohereAsrProcessor(feature_extractor=CohereAsrFeatureExtractor(), tokenizer=fast)
    ids = {t: fast.convert_tokens_to_ids(t) for t in ("<pad>", "<|endoftext|>", "<|startoftranscript|>")}
    enc = dict(hidden_size=48, num_hidden_layers=2, num_attention_heads=2, intermediate_size=96,
               subsampling_conv_channels=8, num_mel_bins=128)
    cfg = CohereAsrConfig(
        encoder_config=enc, vocab_size=len(fast), hidden_size=48, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=96, max_position_embeddings=128, pad_token_id=ids["<pad>"],
        eos_token_id=ids["<|endoftext|>"], bos_token_id=ids["<|startoftranscript|>"],
    )
    model = CohereAsrForConditionalGeneration(cfg)
    model_dir = os.path.join(out_dir, "model")
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    return model_dir


def _tone(freqs: list[float], dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    return (0.3 * y + 0.01 * rng.standard_normal(t.shape)).astype(np.float32)


def build_synthetic_dataset(out_dir: str, n_train: int = 48, n_valid: int = 8, seed: int = 0) -> tuple[str, str]:
    """Each phrase is tied to a distinct tone pattern, so a tiny model can learn the mapping."""
    rng = np.random.default_rng(seed)
    wav_dir = os.path.join(out_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    patterns = [[300.0], [700.0, 1100.0], [1500.0], [2200.0, 400.0]]

    def write_split(name: str, n: int) -> str:
        path = os.path.join(out_dir, f"{name}.tsv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("id\tpath\tsentence\n")
            for i in range(n):
                k = i % len(PHRASES)
                dur = float(rng.uniform(0.6, 2.0))
                uid = f"{name}_{i:04d}"
                wav = os.path.join("wavs", f"{uid}.wav")
                sf.write(os.path.join(out_dir, wav), _tone(patterns[k], dur, 16000, rng), 16000)
                f.write(f"{uid}\t{wav}\t{PHRASES[k]}\n")
        return path

    return write_split("train", n_train), write_split("valid", n_valid)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    os.makedirs(out, exist_ok=True)
    print(build_tiny_model(out))
    print(build_synthetic_dataset(out))
