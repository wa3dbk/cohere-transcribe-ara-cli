"""Command-line interface: `cohere-ara {prepare,decode,score,train,merge-lora}`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import DEFAULT_MODEL, __version__
from .text import PRESETS

logger = logging.getLogger("cohere_transcribe_ara_cli")


def _setup_logging(verbose: bool = False) -> None:
    rank = int(os.environ.get("RANK", "0"))
    level = logging.DEBUG if verbose else (logging.INFO if rank == 0 else logging.WARNING)
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


# --------------------------------------------------------------------------------------
# Argument groups
# --------------------------------------------------------------------------------------
def add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("manifest format")
    g.add_argument("--id-col", default=None, help="Segment-id column (name or 0-based index). Default: auto / 0")
    g.add_argument("--audio-col", default=None, help="Audio-path column (name or index). Default: auto / 1")
    g.add_argument("--text-col", default=None, help="Transcript column (name or index). Default: auto / 2")
    g.add_argument("--duration-col", default=None, help="Optional duration (seconds) column; read from audio headers otherwise")
    g.add_argument("--audio-root", default=None, help="Base dir for relative audio paths (default: the manifest's dir)")
    g.add_argument("--header", choices=["auto", "yes", "no"], default="auto", help="Whether the TSV has a header row")
    g.add_argument("--io-threads", type=int, default=16, help="Threads for reading audio headers")


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", default=DEFAULT_MODEL, help=f"HF id or local dir (default: {DEFAULT_MODEL})")
    g.add_argument("--processor", default=None, help="Processor id/dir if different from --model")
    g.add_argument("--revision", default=None)
    g.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token (gated model); default $HF_TOKEN")
    g.add_argument("--cache-dir", default=None)
    g.add_argument("--attn-implementation", default=None, choices=["sdpa", "eager", "flash_attention_2"],
                   help="Attention kernel (default: transformers' choice, usually sdpa)")
    g.add_argument("--language", default="ar", help="Prompt language token (ar/en/...)")
    g.add_argument("--no-punctuation", action="store_true",
                   help="Use the <|nopnc|> prompt (lower-case, punctuation-free output / training targets)")


def add_launch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpus", type=int, default=1,
                   help="Number of local GPUs; >1 re-launches itself with torchrun (DDP / sharded decoding)")


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    from .data import audio_info, read_manifest, write_manifest

    segs = read_manifest(args.manifest, id_col=args.id_col, audio_col=args.audio_col, text_col=args.text_col,
                         duration_col=args.duration_col, audio_root=args.audio_root, header=args.header)

    def info(s):
        try:
            return audio_info(s.audio)
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(args.io_threads) as ex:
        infos = list(tqdm(ex.map(info, segs, chunksize=64), total=len(segs), desc="Checking audio", mininterval=2))
    bad, sr_mismatch, multi_ch, empty_text, kept = [], 0, 0, 0, []
    for s, inf in zip(segs, infos):
        if inf is None:
            bad.append(s.id)
            continue
        s.duration, sr, ch = inf
        sr_mismatch += sr != 16000
        multi_ch += ch > 1
        if not (s.text or "").strip():
            empty_text += 1
        if args.min_duration <= s.duration <= args.max_duration:
            kept.append(s)
    durs = sorted(s.duration for s in kept) or [0.0]
    stats = {
        "segments_in": len(segs),
        "segments_out": len(kept),
        "unreadable": len(bad),
        "out_of_duration_range": len(segs) - len(bad) - len(kept),
        "not_16khz": sr_mismatch,
        "multichannel": multi_ch,
        "empty_transcripts": empty_text,
        "hours": round(sum(durs) / 3600, 3),
        "duration_min": round(durs[0], 3),
        "duration_median": round(durs[len(durs) // 2], 3),
        "duration_max": round(durs[-1], 3),
        "over_30s": sum(d > 30 for d in durs),
    }
    write_manifest(args.output, kept)
    print(json.dumps(stats, indent=2))
    if bad:
        print(f"Unreadable files (first 10): {bad[:10]}", file=sys.stderr)
    if sr_mismatch:
        print(f"Note: {sr_mismatch} files are not 16 kHz; they will be resampled on the fly.", file=sys.stderr)
    return 0


def _adapter_base_model(adapter_dir: str) -> str | None:
    marker = os.path.join(adapter_dir, "BASE_MODEL.txt")
    if os.path.exists(marker):
        with open(marker) as f:
            base = f.read().strip()
        if base:
            return base
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return json.load(f).get("base_model_name_or_path")
    return None


def _maybe_resolve_adapter(args) -> None:
    """`--model exp/best` where exp/best is a LoRA adapter dir -> base model + adapter."""
    if args.adapter or not os.path.isdir(args.model):
        return
    if not os.path.exists(os.path.join(args.model, "adapter_config.json")):
        return
    base = _adapter_base_model(args.model)
    if not base:
        raise SystemExit(f"{args.model} is a LoRA adapter but its base model is unknown; pass --model BASE --adapter DIR")
    args.adapter = args.model
    args.processor = args.processor or args.model
    args.model = base
    logger.info("Detected LoRA adapter; base model=%s adapter=%s", base, args.adapter)


def cmd_decode(args) -> int:
    from .decode import run_decode

    _maybe_resolve_adapter(args)
    res = run_decode(args)
    if res is not None:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_score(args) -> int:
    from .data import read_manifest
    from .decode import score_and_write

    refs = read_manifest(args.ref, id_col=args.id_col, audio_col=args.audio_col, text_col=args.text_col,
                         audio_root=args.audio_root, header=args.header)
    hyps = read_hyp_file(args.hyp)
    triples, missing = [], 0
    for s in refs:
        if s.text is None:
            continue
        if s.id not in hyps:
            missing += 1
        triples.append((s.id, s.text, hyps.get(s.id, "")))
    if missing:
        logger.warning("%d reference segments have no hypothesis (scored as empty)", missing)
    os.makedirs(args.output_dir, exist_ok=True)
    res = score_and_write(triples, args.output_dir, args.normalizers, args.details)
    print(json.dumps(res, indent=2))
    return 0


def read_hyp_file(path: str) -> dict[str, str]:
    """Accepts our hyp.tsv (header with a 'hyp' column), `id<TAB>text`, or Kaldi `id text`."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return out
    first = lines[0].split("\t")
    if "hyp" in [c.strip().lower() for c in first]:
        cols = [c.strip().lower() for c in first]
        hi, ii = cols.index("hyp"), (cols.index("id") if "id" in cols else 0)
        for ln in lines[1:]:
            r = ln.split("\t")
            out[r[ii]] = r[hi] if hi < len(r) else ""
        return out
    for ln in lines:
        if "\t" in ln:
            uid, _, text = ln.partition("\t")
        else:
            uid, _, text = ln.partition(" ")
        out[uid.strip()] = text.strip()
    return out


def cmd_train(args) -> int:
    from .train import run_train

    if args.lora and args.lr == TRAIN_DEFAULT_LR:
        args.lr = 1e-4
        logger.info("LoRA: using lr=%g (pass --lr to override)", args.lr)
    run_train(args)
    return 0


def cmd_merge_lora(args) -> int:
    import torch

    from .modeling import attach_adapter, hf_kwargs, load_model, load_processor

    base = args.model or _adapter_base_model(args.adapter)
    if not base:
        raise SystemExit("Cannot determine the base model of the adapter; pass --model")
    kw = hf_kwargs(args.revision, args.hf_token, args.cache_dir)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.save_dtype]
    model = attach_adapter(load_model(base, dtype=torch.float32, **kw), args.adapter, merge=True)
    model.to(dtype).save_pretrained(args.output_dir)
    load_processor(args.adapter if os.path.exists(os.path.join(args.adapter, "processor_config.json")) else base,
                   **kw).save_pretrained(args.output_dir)
    print(f"Merged model written to {args.output_dir}")
    return 0


TRAIN_DEFAULT_LR = 1e-5


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cohere-ara",
        description="Decode, evaluate and fine-tune Cohere Transcribe (Arabic) on TSV manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    fmt = argparse.ArgumentDefaultsHelpFormatter
    norms = ",".join(PRESETS)

    # ---- prepare
    sp = sub.add_parser("prepare", help="Validate a manifest, add durations, write a normalized TSV", formatter_class=fmt)
    sp.add_argument("manifest")
    sp.add_argument("-o", "--output", required=True, help="Output TSV (id, path, sentence, duration)")
    sp.add_argument("--min-duration", type=float, default=0.0)
    sp.add_argument("--max-duration", type=float, default=float("inf"))
    add_data_args(sp)
    sp.set_defaults(func=cmd_prepare)

    # ---- decode
    sp = sub.add_parser("decode", help="Transcribe a manifest (and score it if it has transcripts)", formatter_class=fmt)
    sp.add_argument("manifest")
    sp.add_argument("-o", "--output-dir", required=True)
    add_model_args(sp)
    sp.add_argument("--adapter", default=None, help="LoRA adapter dir (merged into the model at load time)")
    sp.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    g = sp.add_argument_group("batching / speed")
    g.add_argument("--max-batch-duration", type=float, default=800.0,
                   help="Max *padded* seconds of audio per batch (n_utts x longest)")
    g.add_argument("--batch-size", type=int, default=128, help="Max utterances per batch")
    g.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (audio loading + features)")
    g = sp.add_argument_group("search")
    g.add_argument("--num-beams", type=int, default=1)
    g.add_argument("--length-penalty", type=float, default=1.0)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--no-repeat-ngram-size", type=int, default=0)
    g.add_argument("--max-new-tokens", type=int, default=448)
    g.add_argument("--tokens-per-second", type=float, default=20.0,
                   help="Cap new tokens at 16 + rate x longest-duration in the batch (limits runaway loops); 0 disables")
    g = sp.add_argument_group("output / scoring")
    g.add_argument("--normalizers", type=_csv_list, default=["arabic", "none"], help=f"Scoring normalizers among {norms}")
    g.add_argument("--details", action="store_true", help="Write per-utterance alignments (details.<norm>.txt)")
    g.add_argument("--resume", action="store_true", help="Skip segments already decoded in OUTPUT_DIR")
    g.add_argument("--limit", type=int, default=0, help="Only decode the first N segments (debugging)")
    g.add_argument("--progress-all-ranks", action="store_true")
    add_data_args(sp)
    add_launch_args(sp)
    sp.set_defaults(func=cmd_decode)

    # ---- score
    sp = sub.add_parser("score", help="Compute WER/CER of a hypothesis file against a reference manifest", formatter_class=fmt)
    sp.add_argument("--ref", required=True, help="Reference manifest (TSV)")
    sp.add_argument("--hyp", required=True, help="hyp.tsv from decode, `id<TAB>text`, or Kaldi `id text`")
    sp.add_argument("-o", "--output-dir", required=True)
    sp.add_argument("--normalizers", type=_csv_list, default=["arabic", "none"], help=f"Among {norms}")
    sp.add_argument("--details", action="store_true")
    add_data_args(sp)
    sp.set_defaults(func=cmd_score)

    # ---- train
    sp = sub.add_parser("train", help="Fine-tune the model", formatter_class=fmt)
    sp.add_argument("--train-manifest", required=True)
    sp.add_argument("--valid-manifest", default=None)
    sp.add_argument("--exp-dir", required=True, help="Experiment dir (logs, checkpoints, best/, final/)")
    add_model_args(sp)
    add_data_args(sp)
    add_launch_args(sp)
    g = sp.add_argument_group("data filtering / batching")
    g.add_argument("--min-duration", type=float, default=0.3)
    g.add_argument("--max-duration", type=float, default=30.0, help="Must be <= 30 s (single-chunk limit of the model)")
    g.add_argument("--max-tokens", type=int, default=0, help="Max label tokens (0 = model limit)")
    g.add_argument("--max-tokens-per-second", type=float, default=30.0, help="Drop likely misaligned transcripts")
    g.add_argument("--train-text-normalizer", choices=list(PRESETS), default="none",
                   help="Normalization applied to training/validation targets")
    g.add_argument("--max-batch-duration", type=float, default=240.0, help="Max padded seconds per micro-batch per GPU")
    g.add_argument("--batch-size", type=int, default=64, help="Max utterances per micro-batch")
    g.add_argument("--num-workers", type=int, default=6)
    g.add_argument("--prefetch-factor", type=int, default=4)
    g.add_argument("--refresh-data-cache", action="store_true", help="Re-read manifests/durations into EXP/data")
    g = sp.add_argument_group("optimization")
    g.add_argument("--num-epochs", type=int, default=5)
    g.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer updates (0 = use epochs)")
    g.add_argument("--lr", type=float, default=TRAIN_DEFAULT_LR, help="Peak LR (LoRA default becomes 1e-4)")
    g.add_argument("--encoder-lr-scale", type=float, default=1.0, help="LR multiplier for encoder params")
    g.add_argument("--lr-scheduler", default="cosine",
                   choices=["linear", "cosine", "constant_with_warmup", "inverse_sqrt", "polynomial"])
    g.add_argument("--warmup-steps", type=int, default=None)
    g.add_argument("--warmup-ratio", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--adam-beta1", type=float, default=0.9)
    g.add_argument("--adam-beta2", type=float, default=0.98)
    g.add_argument("--adam-eps", type=float, default=1e-8)
    g.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw", help="adamw8bit needs bitsandbytes")
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--mixed-precision", choices=["auto", "bf16", "fp16", "no"], default="auto")
    g.add_argument("--gradient-checkpointing", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g = sp.add_argument_group("regularization / augmentation")
    g.add_argument("--spec-augment", action="store_true", help="Enable SpecAugment")
    g.add_argument("--freq-masks", type=int, default=2)
    g.add_argument("--freq-mask-width", type=int, default=27)
    g.add_argument("--time-masks", type=int, default=10)
    g.add_argument("--time-mask-ratio", type=float, default=0.05)
    g.add_argument("--max-time-mask-width", type=int, default=40)
    g.add_argument("--dropout", type=float, default=None, help="Encoder dropout override")
    g.add_argument("--attention-dropout", type=float, default=None, help="Encoder+decoder attention dropout override")
    g.add_argument("--layerdrop", type=float, default=0.0, help="Encoder layerdrop (enables find_unused_parameters)")
    g = sp.add_argument_group("what to train")
    g.add_argument("--freeze-encoder", action="store_true", help="Train the decoder only")
    g.add_argument("--freeze-encoder-layers", type=int, default=0, help="Freeze subsampling + first N encoder layers")
    g.add_argument("--lora", action="store_true", help="LoRA fine-tuning (peft)")
    g.add_argument("--lora-r", type=int, default=32)
    g.add_argument("--lora-alpha", type=int, default=64)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--lora-targets", default="decoder,encoder", help="Comma list among: decoder,encoder")
    g.add_argument("--merge-lora-at-end", action=argparse.BooleanOptionalAction, default=True,
                   help="Save final/ as a merged full model when using LoRA")
    g.add_argument("--find-unused-parameters", action="store_true")
    g = sp.add_argument_group("validation / checkpointing / logging")
    g.add_argument("--valid-interval", type=int, default=1000, help="Validate every N updates (0 = only per epoch)")
    g.add_argument("--valid-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--eval-at-start", action=argparse.BooleanOptionalAction, default=True, help="Baseline validation")
    g.add_argument("--valid-decode-utts", type=int, default=500,
                   help="Greedy-decode this many validation utts for WER (-1 = all, 0 = loss only)")
    g.add_argument("--valid-max-batch-duration", type=float, default=None)
    g.add_argument("--valid-max-new-tokens", type=int, default=448)
    g.add_argument("--valid-normalizer", choices=list(PRESETS), default="arabic")
    g.add_argument("--valid-log-examples", type=int, default=3)
    g.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after N evals without improvement")
    g.add_argument("--save-interval", type=int, default=1000, help="Save a resumable checkpoint every N updates (0=off)")
    g.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--keep-checkpoints", type=int, default=2)
    g.add_argument("--save-dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="dtype of best/ and final/")
    g.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in EXP_DIR")
    g.add_argument("--resume-from", default=None, help="Resume from a specific checkpoint dir")
    g.add_argument("--log-interval", type=int, default=50)
    g.add_argument("--report-to", choices=["none", "tensorboard", "wandb"], default="none")
    sp.set_defaults(func=cmd_train)

    # ---- merge-lora
    sp = sub.add_parser("merge-lora", help="Merge a LoRA adapter into its base model", formatter_class=fmt)
    sp.add_argument("--adapter", required=True)
    sp.add_argument("--model", default=None, help="Base model (default: read from the adapter dir)")
    sp.add_argument("-o", "--output-dir", required=True)
    sp.add_argument("--save-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    sp.add_argument("--revision", default=None)
    sp.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    sp.add_argument("--cache-dir", default=None)
    sp.set_defaults(func=cmd_merge_lora)
    return p


def _relaunch_with_torchrun(nproc: int) -> int:
    import subprocess

    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           "-m", "cohere_transcribe_ara_cli", *sys.argv[1:]]
    logger.info("Launching %d processes: %s", nproc, " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    if getattr(args, "gpus", 1) > 1 and "LOCAL_RANK" not in os.environ:
        return _relaunch_with_torchrun(args.gpus)
    if getattr(args, "max_duration", None) is not None and args.command == "train" and args.max_duration > 30.0:
        parser.error("--max-duration must be <= 30 s for training (longer audio is chunked by the model)")
    for n in getattr(args, "normalizers", []) or []:
        if n not in PRESETS:
            parser.error(f"Unknown normalizer {n!r}; choose from {list(PRESETS)}")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
