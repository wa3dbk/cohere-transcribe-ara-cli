import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from cohere_transcribe_ara_cli.modeling import get_prompt_ids, inference_prefix, strip_prefix


def test_inference_prefix_matches_generate(tiny):
    """The teacher-forcing prefix must be exactly what generate() feeds the decoder."""
    proc = AutoProcessor.from_pretrained(tiny["model"])
    model = CohereAsrForConditionalGeneration.from_pretrained(tiny["model"]).eval()
    prompt = get_prompt_ids(proc, "ar", True)
    prefix = inference_prefix(model, prompt)
    feats = proc.feature_extractor([0.01 * torch.randn(16000).numpy()], sampling_rate=16000,
                                   return_tensors="pt", return_attention_mask=True)
    out = model.generate(input_features=feats["input_features"], attention_mask=feats["attention_mask"],
                         decoder_input_ids=torch.tensor([prompt]), max_new_tokens=3, do_sample=False)
    assert out[0, : len(prefix)].tolist() == prefix
    stripped = strip_prefix(out, prompt, prefix)
    assert stripped.shape[1] == out.shape[1] - len(prefix)


def test_nopnc_prompt_differs(tiny):
    proc = AutoProcessor.from_pretrained(tiny["model"])
    assert get_prompt_ids(proc, "ar", True) != get_prompt_ids(proc, "ar", False)
