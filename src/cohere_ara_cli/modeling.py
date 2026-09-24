"""Model / processor loading and decoder-prompt handling for Cohere Transcribe."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# Parameter-name fragments used for LoRA targeting (decoder self/cross attn + MLP, encoder attn + FFN).
LORA_TARGETS = {
    "decoder": r".*decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|o_proj)|.*decoder\.layers\.\d+\.mlp\.(fc1|fc2)",
    "encoder": r".*encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)|.*encoder\.layers\.\d+\.feed_forward\d\.(linear1|linear2)",
}


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return DTYPES[name]


def hf_kwargs(revision: str | None = None, token: str | None = None, cache_dir: str | None = None) -> dict:
    kw: dict = {}
    if revision:
        kw["revision"] = revision
    if token:
        kw["token"] = token
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return kw


def load_processor(model: str, **kw):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model, **kw)


def load_model(
    model: str,
    dtype: torch.dtype,
    attn_implementation: str | None = None,
    config=None,
    **kw,
):
    from transformers import CohereAsrForConditionalGeneration

    extra: dict = dict(kw)
    if attn_implementation:
        extra["attn_implementation"] = attn_implementation
    if config is not None:
        extra["config"] = config
    return CohereAsrForConditionalGeneration.from_pretrained(model, dtype=dtype, **extra)


def attach_adapter(model, adapter_path: str, merge: bool = True):
    """Load a PEFT LoRA adapter on top of a model (merged for inference speed by default)."""
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path)
    if merge:
        model = model.merge_and_unload()
    return model


def get_prompt_ids(processor, language: str, punctuation: bool) -> list[int]:
    ids = processor.get_decoder_prompt_ids(language=language, punctuation=punctuation)
    unk = getattr(processor.tokenizer, "unk_token_id", None)
    if unk is not None and unk in ids:
        raise ValueError(f"Decoder prompt for language={language!r} contains unknown tokens: {ids}")
    return list(ids)


def decoder_start_token_id(model) -> int | None:
    """Mirror of GenerationMixin._get_decoder_start_token_id."""
    gc = getattr(model, "generation_config", None)
    for src in (gc, model.config):
        if src is None:
            continue
        v = getattr(src, "decoder_start_token_id", None)
        if v is not None:
            return int(v)
    for src in (gc, model.config):
        if src is None:
            continue
        v = getattr(src, "bos_token_id", None)
        if v is not None:
            return int(v)
    return None


def inference_prefix(model, prompt_ids: Sequence[int]) -> list[int]:
    """The exact decoder prefix `generate()` will use given ``prompt_ids``.

    `generate()` prepends ``decoder_start_token_id`` when the user-supplied prompt doesn't start with it; training
    must use the same prefix so that teacher forcing matches inference.
    """
    start = decoder_start_token_id(model)
    prompt = list(prompt_ids)
    if start is not None and (not prompt or prompt[0] != start):
        return [start] + prompt
    return prompt


def eos_token_id(model) -> int:
    for src in (getattr(model, "generation_config", None), model.config):
        v = getattr(src, "eos_token_id", None) if src is not None else None
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
        if v is not None:
            return int(v)
    raise ValueError("Model has no eos_token_id")


def pad_token_id(model, processor) -> int:
    v = getattr(model.config, "pad_token_id", None)
    if v is None:
        v = processor.tokenizer.pad_token_id
    if v is None:
        raise ValueError("No pad_token_id available")
    return int(v)


def strip_prefix(sequences: torch.Tensor, prompt_ids: Sequence[int], prefix: Sequence[int]) -> torch.Tensor:
    """Remove the decoder prompt from generated sequences (handles an optional prepended start token)."""
    for cand in (list(prefix), list(prompt_ids)):
        n = len(cand)
        if sequences.shape[1] >= n:
            head = sequences[:, :n]
            ref = torch.tensor(cand, device=sequences.device, dtype=sequences.dtype).unsqueeze(0)
            if bool((head == ref).all()):
                return sequences[:, n:]
    return sequences  # fall back to skip_special_tokens during decoding


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_model_for_inference(model, processor, out_dir: str, dtype: torch.dtype | None, state_dict=None, is_main=True, save_function=None):
    """save_pretrained (+ processor) in a directory that `cohere-ara decode --model` can load directly."""
    if not is_main:
        return
    os.makedirs(out_dir, exist_ok=True)
    if state_dict is None:
        state_dict = model.state_dict()
    if dtype is not None:
        state_dict = {k: (v.to(dtype) if torch.is_floating_point(v) else v) for k, v in state_dict.items()}
        model.config.dtype = dtype
    kw = {"state_dict": state_dict}
    if save_function is not None:
        kw["save_function"] = save_function
    model.save_pretrained(out_dir, **kw)
    processor.save_pretrained(out_dir)
