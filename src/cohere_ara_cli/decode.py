"""Batched decoding over a TSV manifest with duration bucketing, resumability, multi-GPU sharding and scoring."""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from .data import (
    AudioDataset,
    DecodeCollator,
    DurationBatchSampler,
    Segment,
    fill_durations,
    read_manifest,
    worker_init_fn,
)
from .metrics import score, write_details, write_report
from .modeling import (
    attach_adapter,
    eos_token_id,
    get_prompt_ids,
    hf_kwargs,
    inference_prefix,
    load_model,
    load_processor,
    resolve_dtype,
    strip_prefix,
)
from .text import get_normalizer
from .utils import accelerate_kwargs, check_world

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def reassemble(texts: Sequence[str], chunk_index: Sequence[tuple[int, int | None]], n: int, sep: str = " ") -> list[str]:
    """Join per-chunk transcripts back into one string per input (mirrors CohereAsrProcessor)."""
    out = [""] * n
    chunks: dict[int, list[tuple[int, str]]] = {}
    for (sample_idx, chunk_idx), t in zip(chunk_index, texts):
        if chunk_idx is None:
            out[sample_idx] = t
        else:
            chunks.setdefault(sample_idx, []).append((chunk_idx, t))
    for sample_idx, items in chunks.items():
        items.sort(key=lambda x: x[0])
        out[sample_idx] = sep.join(t.strip() for _, t in items if t and t.strip())
    return out


class Transcriber:
    """Thin wrapper around model.generate with OOM back-off (splits a batch in halves on CUDA OOM)."""

    def __init__(self, model, processor, language: str, punctuation: bool, gen_kwargs: dict, device, dtype,
                 max_new_tokens: int, tokens_per_second: float | None):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompt_ids = get_prompt_ids(processor, language, punctuation)
        self.prefix = inference_prefix(model, self.prompt_ids)
        self.gen_kwargs = gen_kwargs
        self.device = device
        self.dtype = dtype
        max_pos = getattr(model.config, "max_position_embeddings", None)
        self.max_new_tokens = min(max_new_tokens, max_pos - len(self.prefix)) if max_pos else max_new_tokens
        self.tokens_per_second = tokens_per_second
        self.sep = "" if language in ("ja", "zh") else " "
        self.eos_id = eos_token_id(model)
        self.num_truncated = 0

    def _budget(self, attention_mask: torch.Tensor) -> int:
        if not self.tokens_per_second:
            return self.max_new_tokens
        frames = int(attention_mask.sum(dim=1).max().item())
        secs = frames * 0.01  # 10 ms hop
        return int(min(self.max_new_tokens, 16 + self.tokens_per_second * secs))

    @torch.inference_mode()
    def _generate_rows(self, feats: torch.Tensor, mask: torch.Tensor) -> list[str]:
        t = int(mask.sum(dim=1).max().item())
        feats, mask = feats[:, :t], mask[:, :t]
        prompt = torch.tensor([self.prompt_ids] * feats.shape[0], dtype=torch.long, device=self.device)
        budget = self._budget(mask)
        try:
            out = self.model.generate(
                input_features=feats.to(self.device, dtype=self.dtype, non_blocking=True),
                attention_mask=mask.to(self.device, non_blocking=True),
                decoder_input_ids=prompt,
                max_new_tokens=budget,
                **self.gen_kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            if feats.shape[0] == 1:
                raise
            torch.cuda.empty_cache()
            half = feats.shape[0] // 2
            logger.warning("CUDA OOM on a batch of %d rows; retrying as %d + %d", feats.shape[0], half, feats.shape[0] - half)
            return self._generate_rows(feats[:half], mask[:half]) + self._generate_rows(feats[half:], mask[half:])
        seqs = out.sequences if hasattr(out, "sequences") else out
        seqs = strip_prefix(seqs, self.prompt_ids, self.prefix)
        if seqs.shape[1] >= budget:  # rows that used the whole budget without emitting EOS were cut off
            self.num_truncated += int((~(seqs == self.eos_id).any(dim=1)).sum().item())
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        return [_clean(t) for t in texts]

    def transcribe_batch(self, batch: dict) -> list[str]:
        n = len(batch["indices"])
        texts = self._generate_rows(batch["input_features"], batch["attention_mask"])
        return reassemble(texts, batch["audio_chunk_index"], n, self.sep)


def _load_done(part_dir: str) -> dict[str, str]:
    done: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(part_dir, "hyp.rank*.tsv"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                uid, _, hyp = line.partition("\t")
                done[uid] = hyp
    return done


def run_decode(args) -> dict | None:
    from accelerate import PartialState

    state = PartialState(**accelerate_kwargs())
    check_world(state)
    out_dir = args.output_dir
    part_dir = os.path.join(out_dir, "parts")
    if state.is_main_process:
        os.makedirs(part_dir, exist_ok=True)
        if not args.resume:
            for p in glob.glob(os.path.join(part_dir, "hyp.rank*.tsv")):
                os.remove(p)
    state.wait_for_everyone()

    segments = read_manifest(
        args.manifest, id_col=args.id_col, audio_col=args.audio_col, text_col=args.text_col,
        duration_col=args.duration_col, audio_root=args.audio_root, header=args.header,
    )
    if args.limit:
        segments = segments[: args.limit]

    done = _load_done(part_dir) if args.resume else {}
    todo_all = [s for s in segments if s.id not in done]
    if state.is_main_process and done:
        logger.info("Resuming: %d already decoded, %d remaining", len(done), len(todo_all))

    # Shard by index first so each rank only reads headers of its own files.
    mine: list[Segment] = todo_all[state.process_index :: state.num_processes]
    bad = fill_durations(mine, num_threads=args.io_threads, show_progress=state.is_main_process)
    bad_set = set(bad)
    if bad:
        logger.warning("[rank %d] %d unreadable audio files will get empty hypotheses", state.process_index, len(bad))
    readable = [s for s in mine if s.id not in bad_set]

    device = state.device
    dtype = resolve_dtype(args.dtype, device)
    kw = hf_kwargs(args.revision, args.hf_token, args.cache_dir)
    processor = load_processor(args.processor or args.model, **kw)
    model = load_model(args.model, dtype=dtype, attn_implementation=args.attn_implementation, **kw)
    if args.adapter:
        model = attach_adapter(model, args.adapter, merge=True)
    model.to(device).eval()

    gen_kwargs = {"do_sample": False, "num_beams": args.num_beams}
    if args.num_beams > 1:
        gen_kwargs["length_penalty"] = args.length_penalty
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    tr = Transcriber(model, processor, args.language, not args.no_punctuation, gen_kwargs, device, dtype,
                     args.max_new_tokens, args.tokens_per_second)
    if state.is_main_process:
        logger.info("Decoder prompt ids: %s (inference prefix %s)", tr.prompt_ids, tr.prefix)

    part_path = os.path.join(part_dir, f"hyp.rank{state.process_index}.tsv")
    audio_secs = 0.0
    t0 = time.time()
    with open(part_path, "a", encoding="utf-8") as fout:
        for s in mine:
            if s.id in bad_set:
                fout.write(f"{s.id}\t\n")
        if readable:
            sampler = DurationBatchSampler(
                [s.duration for s in readable], args.max_batch_duration, args.batch_size, shuffle=False,
            )
            loader = DataLoader(
                AudioDataset(readable),
                batch_sampler=sampler,
                collate_fn=DecodeCollator(processor.feature_extractor),
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
                prefetch_factor=4 if args.num_workers > 0 else None,
                persistent_workers=False,
            )
            from tqdm import tqdm

            pbar = tqdm(total=len(readable), desc=f"Decoding[rank{state.process_index}]", unit="utt",
                        disable=not state.is_main_process and not args.progress_all_ranks, mininterval=2.0)
            for batch in loader:
                for idx, err in batch["failed"]:
                    logger.warning("Failed to load %s: %s", readable[idx].audio, err)
                    fout.write(f"{readable[idx].id}\t\n")
                if batch["indices"]:
                    hyps = tr.transcribe_batch(batch)
                    for idx, hyp in zip(batch["indices"], hyps):
                        fout.write(f"{readable[idx].id}\t{hyp}\n")
                    audio_secs += batch["audio_seconds"]
                fout.flush()
                n = len(batch["indices"]) + len(batch["failed"])
                pbar.update(n)
                el = time.time() - t0
                pbar.set_postfix(RTFx=f"{audio_secs / max(el, 1e-6):.1f}")
            pbar.close()
    elapsed = time.time() - t0
    if tr.num_truncated:
        logger.warning("[rank %d] %d hypotheses hit the token budget (no EOS): consider raising --max-new-tokens / "
                       "--tokens-per-second (or they are hallucination loops)", state.process_index, tr.num_truncated)
    with open(os.path.join(part_dir, f"stats.rank{state.process_index}.json"), "w") as f:
        json.dump({"audio_seconds": audio_secs, "wall_seconds": elapsed, "truncated": tr.num_truncated}, f)

    state.wait_for_everyone()
    if not state.is_main_process:
        return None
    return finalize(args, segments, part_dir)


def finalize(args, segments: list[Segment], part_dir: str) -> dict:
    hyps = _load_done(part_dir)
    missing = [s.id for s in segments if s.id not in hyps]
    if missing:
        logger.warning("%d segments have no hypothesis (e.g. %s)", len(missing), missing[:5])

    audio_secs = wall = 0.0
    truncated = 0
    for p in glob.glob(os.path.join(part_dir, "stats.rank*.json")):
        with open(p) as f:
            st = json.load(f)
        audio_secs += st["audio_seconds"]
        wall = max(wall, st["wall_seconds"])
        truncated += st.get("truncated", 0)

    out_dir = args.output_dir
    hyp_path = os.path.join(out_dir, "hyp.tsv")
    has_refs = any(s.text is not None for s in segments)
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.write("id\tref\thyp\n" if has_refs else "id\thyp\n")
        for s in segments:
            h = hyps.get(s.id, "")
            if has_refs:
                f.write(f"{s.id}\t{_clean(s.text or '')}\t{h}\n")
            else:
                f.write(f"{s.id}\t{h}\n")
    # Kaldi-style text file (id<space>hyp) for sclite / other tools.
    with open(os.path.join(out_dir, "hyp.txt"), "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"{s.id} {hyps.get(s.id, '')}\n")

    result: dict = {
        "manifest": os.path.abspath(args.manifest),
        "model": args.model,
        "adapter": args.adapter,
        "language": args.language,
        "punctuation": not args.no_punctuation,
        "num_segments": len(segments),
        "decode_audio_hours_this_run": round(audio_secs / 3600, 3),
        "decode_wall_seconds_this_run": round(wall, 1),
        "rtfx_this_run": round(audio_secs / wall, 2) if wall > 0 else None,
        "hit_token_budget_this_run": truncated,
    }
    logger.info("Hypotheses written to %s", hyp_path)
    if has_refs:
        triples = [(s.id, s.text or "", hyps.get(s.id, "")) for s in segments if s.text is not None]
        result.update(score_and_write(triples, out_dir, args.normalizers, args.details))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def score_and_write(triples, out_dir: str, normalizers: Sequence[str], details: bool) -> dict:
    reports = []
    lines = []
    for name in normalizers:
        rep, utts = score(triples, get_normalizer(name), name, keep_utts=details)
        reports.append(rep)
        lines.append(rep.summary_line())
        logger.info(rep.summary_line())
        if details:
            write_details(os.path.join(out_dir, f"details.{name}.txt"), utts)
    write_report(os.path.join(out_dir, "wer.json"), reports)
    with open(os.path.join(out_dir, "wer.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {f"wer_{r.normalizer}": round(100 * r.wer, 3) for r in reports} | {
        f"cer_{r.normalizer}": round(100 * r.cer, 3) for r in reports
    }
